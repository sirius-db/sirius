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

void readahead_scan_manager::start(std::size_t budget)
{
  // A backend that publishes a zero budget has opted out; running a worker that
  // can never issue anything would just be a thread parked on a condvar.
  if (budget == 0) { return; }

  std::lock_guard lock{_mutex};
  if (_worker.joinable()) { return; }  // already running

  _budget      = budget;
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

  std::lock_guard lock{_mutex};
  auto& state     = _by_operator[operator_id];
  auto const* key = task.get();
  if (state.index.contains(key)) { return; }  // a split registers once
  state.index.emplace(key, state.tasks.size());
  state.tasks.push_back(task_entry{.task = task});
  note_active_locked();
  io::prefetch_census::instance().scans_registered.fetch_add(1, std::memory_order_relaxed);
  io::prefetch_census::instance().note_registration();

  // A newly emitted split is exactly what the worker may have been waiting for:
  // collect_prefetch_batch stops when the head operator has nothing ready.
  _wake = true;
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

void readahead_scan_manager::update(std::size_t operator_id,
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
          auto& self             = state.tasks[it->second];
          auto const read_rank   = ++_next_read_rank;
          auto& census           = io::prefetch_census::instance();
          if (self.prefetch_rank == 0) {
            census.read_before_prefetch.fetch_add(1, std::memory_order_relaxed);
          } else {
            census.order_reads.fetch_add(1, std::memory_order_relaxed);
            census.order_lead.fetch_add(_next_prefetch_rank - self.prefetch_rank,
                                        std::memory_order_relaxed);
            census.order_displacement.fetch_add(
              self.prefetch_rank > read_rank ? self.prefetch_rank - read_rank
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
            bool const ready_sibling =
              std::ranges::any_of(state.tasks, [&](task_entry const& t) {
                return &t != &self && t.prefetched && t.prefetch_done && !t.task.expired() &&
                       t.stage < io::cache::scan_stage::reading;
              });
            if (ready_sibling) {
              io::prefetch_census::instance()
                .read_loading_while_ready_available.fetch_add(1, std::memory_order_relaxed);
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
  // throttled by its own budget or head-of-line blocked behind an operator that
  // has not emitted a split yet.
  auto& census = io::prefetch_census::instance();
  census.collect_passes.fetch_add(1, std::memory_order_relaxed);
  bool stopped = false;

  while (ongoing < _budget) {
    // Peek before consuming.  Burning an operator's quantum on a split that is
    // not there yet would let a LATER operator jump the prefetch order, which is
    // the one thing the order exists to prevent.
    //
    // The head member may simply not have emitted its next split yet, and its
    // producer runs on another thread.  Stopping there leaves the link idle for
    // as long as that takes, so fall through to the head's PEERS — the other
    // members of the same rotation group, which the order already says may run
    // concurrently with it.  Peers only; the next group is gated behind this one
    // and is never considered.
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

    // Nothing in the current group can use this slot: look ahead rather than idle.
    bool lookahead = false;
    if (chosen == nullptr) {
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
    // No member of this group has a split ready.  Stop: their splits are simply
    // not emitted yet, and register_scan_task wakes us when they are.
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
    // issue, so it never occupies a slot.  Memoized, so this does not rebuild
    // the range list under the lock.
    if (task->fadvise_hints().empty()) {
      chosen->prefetch_done = true;
      continue;
    }

    batch.push_back(pending_prefetch{.task = std::move(task), .operator_id = chosen_op});
    ++ongoing;
  }

  // Falling out of the while condition (rather than through a break) means the
  // budget, not the order, is what stopped us.
  if (!stopped) { census.stop_budget_full.fetch_add(1, std::memory_order_relaxed); }
  return batch;
}

void readahead_scan_manager::issue_prefetches(std::vector<pending_prefetch> batch)
{
  for (auto& p : batch) {
    io::prefetch_census::instance().note_issue(p.operator_id);
    auto entries = p.task->fadvise_hints();
    // One completion per datasource, but the split only frees its slot once all
    // of them have landed -- so count them down and report once.
    auto remaining  = std::make_shared<std::atomic<std::size_t>>(entries.size());
    auto const* key = p.task.get();

    for (auto& hint : entries) {
      if (!hint.datasource) {
        if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
          on_prefetch_complete(p.operator_id, key);
        }
        continue;
      }
      // weak_from_this, not shared: a completion firing after the query tore
      // the manager down must not resurrect it.
      bool const issued = hint.datasource->prefetch_async(
        [weak = weak_from_this(), op_id = p.operator_id, key, remaining](bool) noexcept {
          if (remaining->fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
          if (auto self = weak.lock()) { self->on_prefetch_complete(op_id, key); }
        });
      // Refused while the manager still holds splits nobody has prefetched:
      // capacity went unused with work sitting right there.
      if (!issued && has_unprefetched_work()) {
        io::prefetch_census::instance().prefetch_declined_with_work.fetch_add(
          1, std::memory_order_relaxed);
      }
    }
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
