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

std::size_t readahead_scan_manager::ongoing_scans() const
{
  std::lock_guard lock{_mutex};
  return count_ongoing_locked();
}

void readahead_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  reset();

  auto index = planner::query_index::build_index(query, planner::build_index_options{});
  if (!index) { return; }

  auto const order = index->prefetching_orders();

  std::lock_guard lock{_mutex};
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

  // A newly emitted split is exactly what the worker may have been waiting for:
  // collect_prefetch_batch stops when the head operator has nothing ready.
  _wake = true;
}

bool readahead_scan_manager::is_operator_depleted(std::size_t operator_id) const
{
  auto it = _by_operator.find(operator_id);
  if (it == _by_operator.end()) { return false; }
  if (it->second.stage != io::cache::scan_stage::disposed) { return false; }
  return std::ranges::all_of(it->second.tasks, [](task_entry const& t) {
    return t.task.expired() || t.stage == io::cache::scan_stage::disposed;
  });
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

    // Splits report independently, so one of them reaching `disposed` does not
    // retire the operator.  Only forward `disposed` once every split has
    // finished as well; until then the scheduler sees the reported stage and
    // keeps the operator in its rotation.
    _scheduler.update(operator_id,
                      is_operator_depleted(operator_id) ? io::cache::scan_stage::disposed : stage);

    _wake = true;
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

  while (ongoing < _budget) {
    // Peek before consuming.  Burning the head operator's quantum on a split
    // that is not there yet would let a LATER operator jump the prefetch order,
    // which is the one thing the order exists to prevent.
    auto const next_id = _scheduler.peek_next_operator_id();
    if (!next_id) { break; }  // order exhausted

    auto op_it = _by_operator.find(*next_id);
    if (op_it == _by_operator.end()) { break; }

    // First split of this operator that nobody has prefetched yet.
    task_entry* chosen = nullptr;
    for (auto& t : op_it->second.tasks) {
      if (t.prefetched || t.task.expired() || t.stage == io::cache::scan_stage::disposed) {
        continue;
      }
      chosen = &t;
      break;
    }
    // Nothing to prefetch for the operator at the head of the order.  Stop
    // rather than skip ahead: its splits are simply not emitted yet, and
    // register_scan_task wakes us when they are.
    if (chosen == nullptr) { break; }

    auto task = chosen->task.lock();
    if (!task) {
      chosen->prefetched    = true;  // expired between the scan above and here
      chosen->prefetch_done = true;
      continue;
    }

    std::ignore        = _scheduler.get_next_prefetching_operator();  // consume the quantum
    chosen->prefetched = true;

    // A split with no file ranges (host-backed, or fully cached) has no IO to
    // issue, so it never occupies a slot.
    if (task->fadvise_entries().empty()) {
      chosen->prefetch_done = true;
      continue;
    }

    batch.push_back(pending_prefetch{.task = std::move(task), .operator_id = *next_id});
    ++ongoing;
  }

  return batch;
}

void readahead_scan_manager::issue_prefetches(std::vector<pending_prefetch> batch)
{
  for (auto& p : batch) {
    auto entries = p.task->fadvise_entries();
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
      hint.datasource->prefetch_async(
        [weak = weak_from_this(), op_id = p.operator_id, key, remaining](bool) noexcept {
          if (remaining->fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
          if (auto self = weak.lock()) { self->on_prefetch_complete(op_id, key); }
        });
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
    _wake                                         = true;
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
  }
}

void readahead_scan_manager::reset()
{
  std::lock_guard lock{_mutex};
  _by_operator.clear();
  _scheduler.clear();
}

}  // namespace sirius::scan_manager
