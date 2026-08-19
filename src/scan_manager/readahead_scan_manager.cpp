/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "scan_manager/readahead_scan_manager.hpp"

#include "io/prefetch_census.hpp"
#include "io/sirius_datasource.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/sirius_physical_operator.hpp"
#include "planner/query_index.hpp"

#include <algorithm>
#include <atomic>
#include <utility>

namespace sirius::scan_manager {

readahead_scan_manager::~readahead_scan_manager() { stop(); }

void readahead_scan_manager::on_task_deployed(query_id_t,
                                              std::size_t,
                                              op::SiriusPhysicalOperatorType operator_type,
                                              int) noexcept
{
  if (_strategy != prefetch_strategy::opportunistic) { return; }
  // A scan deployment means a pipeline thread is about to read for itself.
  // Prefetching alongside it only reorders the device queue; the whole point of
  // this strategy is to use the gaps when the executor is NOT reading.
  if (operator_type == op::SiriusPhysicalOperatorType::GPU_SCAN) { return; }

  {
    // Taken rather than using an atomic so the credit cannot land between the
    // worker's predicate check and its wait, which would lose the wake-up.  The
    // section is two stores; the worker holds this lock for longer, so this can
    // stall the scheduler thread briefly -- see count_ongoing_locked, which is
    // the long pole under it.
    std::lock_guard lock{_mutex};
    // Deliberately uncapped.  A credit is permission to read one more scan
    // ahead, not permission to occupy a slot right now -- the budget still caps
    // how many can be in flight.  Dropping credits earned while the slots
    // happened to be full would silently throw away exactly the compute-heavy
    // stretches this strategy exists to exploit.
    ++_credits;
    _wake = true;
  }
  _cv.notify_one();
}

void readahead_scan_manager::on_memory_downgrade_for_task(query_id_t,
                                                          std::size_t,
                                                          int,
                                                          std::size_t) noexcept
{
  if (_strategy != prefetch_strategy::opportunistic) { return; }
  {
    std::lock_guard lock{_mutex};
    _credits       = std::max(_credits, _budget);
    _may_run_ahead = true;
    _wake          = true;
  }
  _cv.notify_one();
}

void readahead_scan_manager::on_task_queue_empty() noexcept
{
  if (_strategy != prefetch_strategy::opportunistic) { return; }
  {
    std::lock_guard lock{_mutex};
    // An empty queue is not one opportunity, it is the absence of competition:
    // top up to at least a full budget so the worker runs as far ahead as it is
    // allowed to while nothing else wants the device.
    _credits = std::max(_credits, _budget);
    _wake    = true;
  }
  _cv.notify_one();
}

void readahead_scan_manager::start(std::size_t budget, prefetch_strategy strategy)
{
  // A backend that publishes a zero budget has opted out; running a worker that
  // can never issue anything would just be a thread parked on a condvar.
  if (budget == 0) { return; }

  std::lock_guard lock{_mutex};
  if (_worker.joinable()) { return; }  // already running

  _budget   = budget;
  _strategy = strategy;
  // Eager is always invited to fill the budget; opportunistic starts with
  // nothing and is invited one prefetch at a time by on_task_deployed.
  _credits     = 0;
  _wake        = false;
  _stop_source = std::stop_source{};
  _worker =
    std::jthread([this](const std::stop_token& st) { worker_loop(st); }, _stop_source.get_token());
}

void readahead_scan_manager::stop() noexcept
{
  {
    std::lock_guard lock{_mutex};
    if (!_worker.joinable()) { return; }
    _stop_source.request_stop();
  }
  // condition_variable_any's stop-aware wait already wakes on the request; this
  // also covers a worker parked between the predicate check and the wait.
  _cv.notify_all();
  _worker.join();
}

bool readahead_scan_manager::is_running() const noexcept
{
  std::lock_guard lock{_mutex};
  return _worker.joinable() && !_stop_source.stop_requested();
}

bool readahead_scan_manager::is_ongoing(task_entry const& t) noexcept
{
  if (t.stage == io::cache::scan_stage::disposed) { return false; }
  if (t.task.expired()) { return false; }
  // Our prefetch holds a slot only while its IO is in flight.
  if (t.prefetched) { return !t.prefetch_done; }
  // Nobody prefetched this one, so the executor's own read is doing the IO.
  return t.stage >= io::cache::scan_stage::reading;
}

std::size_t readahead_scan_manager::count_ongoing_locked() const
{
  std::size_t n = 0;
  for (auto const& [_, state] : _by_operator) {
    for (auto const& t : state.tasks) {
      if (is_ongoing(t)) { ++n; }
    }
  }
  return n;
}

void readahead_scan_manager::note_active_locked()
{
  auto const now = std::chrono::steady_clock::now();
  auto& census   = io::prefetch_census::instance();
  if (_active_mark.time_since_epoch().count() != 0) {
    auto const dt = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - _active_mark).count());
    census.active_total_ns.fetch_add(dt, std::memory_order_relaxed);
    census.active_weighted_ns.fetch_add(dt * _active_count, std::memory_order_relaxed);
    if (_budget > 0 && _active_count >= _budget) {
      census.active_at_max_ns.fetch_add(dt, std::memory_order_relaxed);
    }
  }
  _active_mark  = now;
  _active_count = count_ongoing_locked();
  census.active_budget.store(_budget, std::memory_order_relaxed);
}

std::size_t readahead_scan_manager::ongoing_scans() const
{
  std::lock_guard lock{_mutex};
  return count_ongoing_locked();
}

bool readahead_scan_manager::has_unprefetched_work() const
{
  std::lock_guard lock{_mutex};
  return std::ranges::any_of(_by_operator, [](auto const& kv) {
    return std::ranges::any_of(kv.second.tasks, [](task_entry const& t) {
      return !t.prefetched && !t.task.expired() && t.stage < io::cache::scan_stage::reading;
    });
  });
}

void readahead_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  auto index = planner::query_index::build_index(query, planner::build_index_options{});
  if (!index) {
    reset();
    return;
  }

  prepare_for_order(index->prefetching_orders());
}

void readahead_scan_manager::prepare_for_order(std::span<const planner::prefetch_step> order)
{
  reset();

  std::lock_guard lock{_mutex};
  {
    // Overwrite rather than append: this is the order for the query being
    // prepared, and a re-prepare must not stack onto the previous one.
    auto& census = io::prefetch_census::instance();
    std::lock_guard g{census.order_mtx};
    census.prefetch_order.clear();
    for (auto const& step : order) {
      if (step.scan != nullptr && step.scan->has_operator_id()) {
        census.prefetch_order.push_back(step.scan->get_operator_id());
      }
    }
  }
  for (auto const& step : order) {
    if (step.scan == nullptr || !step.scan->has_operator_id()) { continue; }
    auto& state     = _by_operator[step.scan->get_operator_id()];
    state.mode      = step.mode;
    state.branch_id = step.branch_id;
    state.count     = step.count;
  }
  _scheduler.reset(order);
}

void readahead_scan_manager::register_scan_task(std::shared_ptr<op::scan::scan_info> const& task,
                                                std::size_t operator_id)
{
  if (!task) { return; }

  {
    std::lock_guard lock{_mutex};
    auto& state     = _by_operator[operator_id];
    auto const* key = task.get();
    if (state.index.contains(key)) { return; }  // a split registers once
    state.index.emplace(key, state.tasks.size());
    state.tasks.push_back(task_entry{.task = task});
    note_active_locked();
    io::prefetch_census::instance().scans_registered.fetch_add(1, std::memory_order_relaxed);
    io::prefetch_census::instance().note_registration();

    // A newly emitted split is exactly what the worker may have been waiting for.
    _wake = true;
  }
  _cv.notify_one();
}

bool readahead_scan_manager::group_is_closed_locked(std::span<const std::size_t> operator_ids) const
{
  // An operator with no entry has registered nothing and been told nothing, so
  // it is still expected to emit -- the conservative answer is "not closed".
  return std::ranges::all_of(operator_ids, [this](std::size_t id) {
    auto it = _by_operator.find(id);
    return it != _by_operator.end() && it->second.closed;
  });
}

bool readahead_scan_manager::is_operator_depleted(std::size_t operator_id) const
{
  auto it = _by_operator.find(operator_id);
  if (it == _by_operator.end()) { return false; }
  // The producer may still emit more splits, so an all-disposed split list says
  // nothing yet.  Retirement is one-way in the scheduler -- see advance() -- so
  // retiring here on a between-waves lull would drop the operator out of the
  // prefetch order for the rest of the query.
  if (!it->second.closed) { return false; }
  return std::ranges::all_of(it->second.tasks, [](task_entry const& t) {
    return t.task.expired() || t.stage == io::cache::scan_stage::disposed;
  });
}

void readahead_scan_manager::publish_stage_locked(std::size_t operator_id,
                                                  io::cache::scan_stage reported)
{
  if (is_operator_depleted(operator_id)) {
    _scheduler.update(operator_id, io::cache::scan_stage::disposed);
  } else if (reported != io::cache::scan_stage::disposed) {
    // `disposed` is the scheduler's retirement edge and retirement is one-way,
    // so a single split reporting it must not reach the scheduler at all --
    // not even as the operator's "latest stage".  The operator has many splits
    // and only the depletion check above can speak for all of them.
    _scheduler.update(operator_id, reported);
  }
  _wake = true;
}

void readahead_scan_manager::mark_operator_closed(std::size_t operator_id)
{
  {
    std::lock_guard lock{_mutex};
    auto it = _by_operator.find(operator_id);
    if (it == _by_operator.end()) { return; }
    if (std::exchange(it->second.closed, true)) { return; }
    // Close is itself a retirement edge: a short operator typically disposes its
    // last split before the producer closes, so nothing else would re-evaluate
    // depletion and the scheduler would keep handing out an operator that can
    // never yield another split.
    publish_stage_locked(operator_id, it->second.stage);
  }
  _cv.notify_one();
}

void readahead_scan_manager::update_scan_state(std::size_t operator_id,
                                               const op::scan::scan_info* task,
                                               io::cache::scan_stage stage)
{
  {
    std::lock_guard lock{_mutex};
    auto& state = _by_operator[operator_id];
    state.stage = stage;

    if (task != nullptr) {
      if (auto it = state.index.find(task); it != state.index.end()) {
        state.tasks[it->second].stage = stage;
      }
    }
    note_active_locked();
    // First time this operator's data is actually being pulled: compare against
    // the issue order to see whether the readahead picked the right scans.
    if (stage == io::cache::scan_stage::reading) {
      io::prefetch_census::instance().note_read(operator_id);
      // Rank this read against the order its prefetch was issued in.
      if (task != nullptr) {
        auto const it = state.index.find(task);
        if (it != state.index.end()) {
          auto& self           = state.tasks[it->second];
          auto const read_rank = ++_next_read_rank;
          auto& census         = io::prefetch_census::instance();
          if (self.prefetch_rank == 0) {
            census.read_before_prefetch.fetch_add(1, std::memory_order_relaxed);
          } else {
            census.order_reads.fetch_add(1, std::memory_order_relaxed);
            census.order_lead.fetch_add(_next_prefetch_rank - self.prefetch_rank,
                                        std::memory_order_relaxed);
            census.order_displacement.fetch_add(self.prefetch_rank > read_rank
                                                  ? self.prefetch_rank - read_rank
                                                  : read_rank - self.prefetch_rank,
                                                std::memory_order_relaxed);
            if (self.prefetch_rank < _last_read_prefetch_rank) {
              census.order_inversions.fetch_add(1, std::memory_order_relaxed);
            }
            _last_read_prefetch_rank = self.prefetch_rank;
          }
        }
      }
      // Reading a split whose prefetch is still running while a sibling of the
      // same operator has already landed means the two were taken in the wrong
      // order: the ready one could have run without waiting on anything.
      if (task != nullptr) {
        auto const it = state.index.find(task);
        if (it != state.index.end()) {
          auto const& self = state.tasks[it->second];
          if (self.prefetched && !self.prefetch_done) {
            bool const ready_sibling = std::ranges::any_of(state.tasks, [&](task_entry const& t) {
              return &t != &self && t.prefetched && t.prefetch_done && !t.task.expired() &&
                     t.stage < io::cache::scan_stage::reading;
            });
            if (ready_sibling) {
              io::prefetch_census::instance().read_loading_while_ready_available.fetch_add(
                1, std::memory_order_relaxed);
            }
          }
        }
      }
    }

    // Splits report independently, so one of them reaching `disposed` does not
    // retire the operator.  Until the producer closes and every split has
    // finished, the scheduler sees the reported stage and keeps the operator in
    // its rotation.
    publish_stage_locked(operator_id, stage);
  }
  // A scan moved, so a slot may have opened -- let the worker refill it.
  // Notified outside the lock so the woken worker does not immediately block on
  // a mutex this thread still holds.
  _cv.notify_one();
}

op::sirius_physical_operator* readahead_scan_manager::get_next_prefetching_operator()
{
  std::lock_guard lock{_mutex};
  return _scheduler.get_next_prefetching_operator();
}

std::vector<readahead_scan_manager::pending_prefetch>
readahead_scan_manager::collect_prefetch_batch_locked()
{
  std::vector<pending_prefetch> batch;
  std::size_t ongoing = count_ongoing_locked();

  // Exactly one of these fires per pass; which one says whether the readahead is
  // throttled by its own budget or has exhausted every currently emitted split.
  auto& census = io::prefetch_census::instance();
  census.collect_passes.fetch_add(1, std::memory_order_relaxed);
  bool stopped = false;

  // Eager fills every free slot; opportunistic issues only what it has been
  // invited to, so an idle executor does not turn into a background read storm
  // competing with the reads the executor is about to do itself.
  std::size_t allowance =
    _strategy == prefetch_strategy::opportunistic ? std::min(_credits, _budget) : _budget;

  while (ongoing < _budget && allowance > 0) {
    // Prefer the scheduler's current group, but never leave budget idle solely
    // because that group has no emitted split ready. Check its peers first, then
    // later groups in order. A later-group pick is lookahead: it fills capacity
    // without consuming the current group's cursor or quantum.
    auto const candidates = _scheduler.peek_group_operator_ids();
    if (candidates.empty()) {  // order exhausted
      census.stop_order_exhausted.fetch_add(1, std::memory_order_relaxed);
      stopped = true;
      break;
    }

    std::size_t chosen_op = 0;
    task_entry* chosen    = nullptr;
    bool any_known        = false;
    for (auto const id : candidates) {
      auto op_it = _by_operator.find(id);
      if (op_it == _by_operator.end()) { continue; }
      any_known = true;
      // First split of this operator that nobody has prefetched yet.
      for (auto& t : op_it->second.tasks) {
        if (t.prefetched || t.task.expired() || t.stage >= io::cache::scan_stage::reading) {
          continue;
        }
        chosen    = &t;
        chosen_op = id;
        break;
      }
      if (chosen != nullptr) { break; }
    }

    // Nothing in the current group can use this slot.  Read ahead into later
    // groups ONLY once this one can emit nothing further.
    //
    // Every pipeline's metadata is parsed in parallel, so a later pipeline can
    // register its splits before the head pipeline registers its first.  A group
    // that is merely empty *right now* is still going to emit, and prefetching
    // past it would issue IO in the opposite order to the one the executor reads
    // in -- spending the budget on splits nobody wants yet while the pipeline
    // actually running waits for its own.  Idling here is the cheaper mistake.
    bool lookahead = false;
    if (chosen == nullptr && (_may_run_ahead || group_is_closed_locked(candidates))) {
      for (auto const id : _scheduler.peek_lookahead_operator_ids()) {
        auto op_it = _by_operator.find(id);
        if (op_it == _by_operator.end()) { continue; }
        any_known = true;
        for (auto& t : op_it->second.tasks) {
          if (t.prefetched || t.task.expired() || t.stage >= io::cache::scan_stage::reading) {
            continue;
          }
          chosen    = &t;
          chosen_op = id;
          lookahead = true;
          break;
        }
        if (chosen != nullptr) { break; }
      }
    }

    if (!any_known) {
      census.stop_operator_unknown.fetch_add(1, std::memory_order_relaxed);
      stopped = true;
      break;
    }
    // No operator in the remaining order has an emitted split ready. Stop
    // until registration or a stage transition changes the available work.
    if (chosen == nullptr) {
      census.stop_no_split_ready.fetch_add(1, std::memory_order_relaxed);
      stopped = true;
      break;
    }

    auto task = chosen->task.lock();
    if (!task) {
      chosen->prefetched    = true;  // expired between the scan above and here
      chosen->prefetch_done = true;
      continue;
    }

    // Charge the quantum only for an in-group pick; a lookahead pick must not
    // move the cursor past the order it is reading ahead of.
    if (!lookahead) {
      std::ignore = _scheduler.focus_member(chosen_op);
      std::ignore = _scheduler.get_next_prefetching_operator();
    }
    chosen->prefetched    = true;
    chosen->prefetch_rank = ++_next_prefetch_rank;

    // A split with no file ranges (host-backed, or fully cached) has no IO to
    // issue, so it never occupies a slot. Hints were computed with the split.
    if (task->fadvise_hints().empty()) {
      chosen->prefetch_done = true;
      continue;
    }

    batch.push_back(pending_prefetch{.task = std::move(task), .operator_id = chosen_op});
    ++ongoing;
    --allowance;
    if (_strategy == prefetch_strategy::opportunistic && _credits > 0) { --_credits; }
  }

  // One-shot: the stall that granted it is over by the time the next pass runs,
  // and leaving it set would quietly make read-ahead unordered for good.
  _may_run_ahead = false;

  // Falling out of the while condition (rather than through a break) means the
  // budget -- or, for opportunistic, the invitation -- ran out rather than the
  // order.  Only a genuinely full budget is reported as such.
  if (!stopped && ongoing >= _budget) {
    census.stop_budget_full.fetch_add(1, std::memory_order_relaxed);
  }
  return batch;
}

void readahead_scan_manager::issue_prefetches(std::vector<pending_prefetch> batch)
{
  for (auto& p : batch) {
    io::prefetch_census::instance().note_issue(p.operator_id);
    auto const* key = p.task.get();

    // The split fans the request out over its own datasources and reports once
    // they have all landed.  Nothing is waited on here: that single report is
    // what frees the slot and wakes the worker to collect the next batch, so it
    // has to come back as a callback rather than a return value.
    //
    // weak_from_this, not shared: a completion firing after the query tore the
    // manager down must not resurrect it.
    p.task->prefetch([weak = weak_from_this(), op_id = p.operator_id, key](
                       op::scan::scan_info::prefetch_outcome out) noexcept {
      auto self = weak.lock();
      if (!self) { return; }
      // Refused while the manager still holds splits nobody has prefetched:
      // capacity went unused with work sitting right there.
      if (out.declined > 0 && self->has_unprefetched_work()) {
        io::prefetch_census::instance().prefetch_declined_with_work.fetch_add(
          out.declined, std::memory_order_relaxed);
      }
      self->on_prefetch_complete(op_id, key);
    });
  }
}

void readahead_scan_manager::on_prefetch_complete(std::size_t operator_id,
                                                  const op::scan::scan_info* task)
{
  {
    std::lock_guard lock{_mutex};
    auto op_it = _by_operator.find(operator_id);
    if (op_it == _by_operator.end()) { return; }
    auto it = op_it->second.index.find(task);
    if (it == op_it->second.index.end()) { return; }
    op_it->second.tasks[it->second].prefetch_done = true;
    note_active_locked();
    _wake = true;
  }
  _cv.notify_one();
}

void readahead_scan_manager::worker_loop(const std::stop_token& st)
{
  std::unique_lock lock{_mutex};
  while (!st.stop_requested()) {
    // Only an update can change what should be prefetched next, so wait for one
    // rather than polling.  Returns false when the stop token fires.
    if (!_cv.wait(lock, st, [this] { return _wake; })) { break; }
    _wake = false;

    auto batch = collect_prefetch_batch_locked();
    if (batch.empty()) { continue; }

    // prefetch_async lands IO and can invoke its completion inline, which takes
    // this same mutex -- so the lock cannot be held across it.
    lock.unlock();
    issue_prefetches(std::move(batch));
    lock.lock();
    note_active_locked();
  }
}

void readahead_scan_manager::reset()
{
  std::lock_guard lock{_mutex};
  _by_operator.clear();
  _scheduler.clear();
}

}  // namespace sirius::scan_manager
