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
#include "log/logging.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/sirius_physical_operator.hpp"
#include "planner/query.hpp"
#include "planner/query_index.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace sirius::scan_manager {

readahead_scan_manager::~readahead_scan_manager() { stop(); }

void readahead_scan_manager::prepare_for_query(const sirius::planner::query& query)
{
  auto scan_orders = query.get_scan_operators();

  size_t index = 0;
  for (auto* scan_op : scan_orders) {
    if (scan_op->type != op::SiriusPhysicalOperatorType::GPU_SCAN) { continue; }
    _operator_id_to_queue_index[scan_op->get_operator_id()] = index++;
    _ordered_work_queues.emplace_back(std::make_unique<prefetch_work_queue>());
    // Without this every queue reports operator 0, so every completion is
    // attributed to the wrong operator.
    _ordered_work_queues.back()->operator_id = scan_op->get_operator_id();
  }
}

void readahead_scan_manager::on_task_deployed(query_id_t,
                                              std::size_t,
                                              op::SiriusPhysicalOperatorType operator_type,
                                              int) noexcept
{
  // A scan deployment means a pipeline thread is about to read for itself, and
  // prefetching alongside it only reorders the device queue.  Anything else is a
  // thread that went to compute instead, which is capacity the executor is not
  // about to use.
  if (operator_type == op::SiriusPhysicalOperatorType::GPU_SCAN) { return; }
  arm_prefetching();
}

void readahead_scan_manager::on_memory_downgrade_for_task(query_id_t,
                                                          std::size_t,
                                                          int,
                                                          std::size_t) noexcept
{
  // The executor is spilling to make room, so the GPU does no work for the
  // duration and the device's IO path is unambiguously free.
  arm_prefetching();
}

void readahead_scan_manager::on_wait_for_memory_for_task(query_id_t,
                                                         std::size_t,
                                                         int,
                                                         std::size_t) noexcept
{
  // Parked waiting on memory somebody else holds: same idle GPU as a downgrade,
  // arrived at differently.
  arm_prefetching();
}

void readahead_scan_manager::on_task_queue_empty() noexcept
{
  // Nothing is waiting to be dispatched, so whatever the executor is doing it is
  // not about to read.  The strongest idle signal there is.
  arm_prefetching();
}

void readahead_scan_manager::start(prefetch_strategy strategy)
{
  // A backend that publishes a zero budget has opted out; running a worker that
  // can never issue anything would just be a thread parked on a condvar.
  if (_budget == 0) { return; }
  _strategy    = strategy;
  _stop_source = std::stop_source{};
  _prefetch_worker =
    std::jthread([this](const std::stop_token& st) { worker_loop(st); }, _stop_source.get_token());
  // Only now start draining the stage-manager mailbox: the hooks below arm
  // prefetching, and there must be a worker for them to arm.
  exec::query_stage_listener::start();

  // Eager does not wait to be invited: it reads ahead as far as the budget
  // allows from the moment there is anything to read.  Opportunistic stays
  // parked until one of the executor-idle signals below arms it.
  if (_strategy == prefetch_strategy::eager) { arm_prefetching(); }
}

void readahead_scan_manager::stop() noexcept
{
  // First, so no hook can be dispatched into this object -- or re-arm the
  // prefetching we are about to tear down -- while the rest of teardown runs.
  exec::query_stage_listener::stop();
  _stop_source.request_stop();
  // Cuts short the worker's wait for a ticket, which is otherwise the one place
  // teardown can sit for a full timeout -- and a ticket handed out now would
  // only buy a prefetch that is about to be abandoned.
  _gatekeeper.stop();
  if (_prefetch_worker.joinable()) { _prefetch_worker.join(); }
  // Give in-flight prefetches a moment to return their tickets, so the summary
  // below describes a settled query.  Draining is about tickets coming back
  // rather than going out, so the stop above does not cut it short -- but it is
  // bounded anyway: an IO that never completes must not hold up teardown.
  //
  // Only worth waiting on if the gate was ever armed.  Unarmed it holds no
  // tickets to hand out and so has none outstanding, but it still reads as
  // undrained (nothing was ever handed out to come back), and waiting on that
  // would spend the full timeout for nothing.
  if (_prefetching_started.load(std::memory_order_relaxed)) {
    std::ignore = _gatekeeper.wait_for_all(std::chrono::milliseconds{200});
  }
  // One line per query, on the way down: the manager is per-query, so this is
  // the last moment its counters describe a whole query and nothing else.
  SIRIUS_LOG_INFO("[readahead] {}", summary());
}

std::string readahead_scan_manager::summary() const
{
  using kind         = prefetch_outcome_kind;
  auto const load    = [](auto const& c) { return c.load(std::memory_order_relaxed); };
  auto const outcome = [this](kind k) { return _counters.outcome(k); };

  auto const prefetched = outcome(kind::prefetched);
  auto const waited     = outcome(kind::wait_for_prefetch);
  auto const mem        = outcome(kind::skipped_memory_pressure);
  auto const behind     = outcome(kind::skipped_fell_behind);
  auto const nothing    = outcome(kind::nothing_to_issue);

  return std::format(
    "issued={}[prefetched={} wait_for_prefetch={}] "
    "skipped={}[memory_pressure={} fell_behind={} nothing_to_issue={}] "
    "candidates={}[dropped_expired={} dropped_fell_behind={}] "
    "operators_drained={} pacing[memory_retries={} idle_polls={} gate_timeouts={}] "
    "executor_reads={}[borrowed={}]",
    prefetched + waited,
    prefetched,
    waited,
    mem + behind + nothing,
    mem,
    behind,
    nothing,
    load(_counters.candidates_taken),
    load(_counters.dropped_expired),
    load(_counters.dropped_fell_behind),
    load(_counters.operators_drained),
    load(_counters.memory_retries),
    load(_counters.idle_polls),
    load(_counters.gate_timeouts),
    load(_counters.cold_read_tickets),
    load(_counters.borrowed));
}

bool readahead_scan_manager::is_running() const noexcept
{
  // The worker itself, not the budget: the budget is settled at construction
  // now, so it says only that a worker COULD run, not that one does.
  return _prefetch_worker.joinable() && !_stop_source.stop_requested();
}

void readahead_scan_manager::register_scan_task(std::shared_ptr<op::scan::scan_info> const& task,
                                                std::size_t operator_id)
{
  if (!task) { return; }
  auto op_index        = _operator_id_to_queue_index.at(operator_id);
  auto& prefetch_queue = _ordered_work_queues.at(op_index);
  prefetch_queue->push(task);
}

void readahead_scan_manager::mark_operator_closed(std::size_t operator_id)
{
  auto op_index        = _operator_id_to_queue_index.at(operator_id);
  auto& prefetch_queue = _ordered_work_queues.at(op_index);
  prefetch_queue->close();
}

void readahead_scan_manager::update_scan_state(std::size_t,
                                               const op::scan::scan_info* task,
                                               io::cache::scan_stage stage)
{
  // A zero budget means the backend opted out, so there is no readahead for this
  // read to compete with and no budget to charge it against.
  if (task == nullptr || _budget == 0) { return; }
  // const_cast: the ticket flag is the split's own bookkeeping, and every other
  // caller of update() hands us a const view of the split it is reporting on.
  auto* split = const_cast<op::scan::scan_info*>(task);

  if (stage == io::cache::scan_stage::reading) {
    // A split whose prefetch DID start is already paying for its IO out of the
    // ticket that prefetch holds; charging it twice would throttle the readahead
    // for work it is doing itself.
    if (split->get_prefetch_state() == op::scan::scan_info::prefetch_state::prefetched) { return; }
    // Nothing was prefetched, so this read is about to do the IO itself. It
    // spends from the same budget the readahead does -- and never waits for it.
    if (!split->take_readahead_ticket()) { return; }
    _counters.cold_read_tickets.fetch_add(1, std::memory_order_relaxed);
    if (_gatekeeper.acquire_or_borrow()) {
      _counters.borrowed.fetch_add(1, std::memory_order_relaxed);
    }
    return;
  }

  if (stage == io::cache::scan_stage::disposed) {
    // The read is over, so give the ticket back -- paying down any debt first.
    if (split->give_back_readahead_ticket()) { _gatekeeper.release(); }
  }
}

prefetch_outcome_kind readahead_scan_manager::classify_prefetch(
  bool allocation_failed, bool split_alive, bool issued_io, io::cache::scan_stage stage) noexcept
{
  // A hard failure outranks anything the consumer was doing: with no buffers
  // there was never an attempt to be early or late for.
  if (allocation_failed) { return prefetch_outcome_kind::skipped_memory_pressure; }
  // The split is gone, so whatever it was going to be read for already happened.
  if (!split_alive) { return prefetch_outcome_kind::skipped_fell_behind; }
  if (!issued_io) {
    return stage >= io::cache::scan_stage::preparing ? prefetch_outcome_kind::skipped_fell_behind
                                                     : prefetch_outcome_kind::nothing_to_issue;
  }
  // IO landed.  `reading` is carved out of "preparing or higher" because it is
  // the more specific case: the consumer is on this split right now and is
  // waiting on the very prefetch that just settled.
  if (stage == io::cache::scan_stage::reading) { return prefetch_outcome_kind::wait_for_prefetch; }
  if (stage >= io::cache::scan_stage::preparing) {
    return prefetch_outcome_kind::skipped_fell_behind;
  }
  return prefetch_outcome_kind::prefetched;
}

readahead_scan_manager::prefetch_candidate readahead_scan_manager::get_next_prefetching_candidate()
{
  // Never blocks.  There is nothing to wait on here that the worker cannot wait
  // on better: it holds a ticket while it is in here, and an empty queue is a
  // reason to give that ticket back, not to sit on it.
  while (!_stop_source.stop_requested()) {
    auto const index = _cursor.load(std::memory_order_relaxed);
    // Every operator has drained.  The queues are in execution order, so there
    // is nothing behind the cursor to go back for.
    if (index >= _ordered_work_queues.size()) { return {}; }

    auto& prefetch_queue = _ordered_work_queues.at(index);
    auto next            = prefetch_queue->get_next_candidate(_counters);
    if (!next.has_value()) { return {}; }  // empty, but this operator is still open
    if (*next) { return {std::move(*next), prefetch_queue->operator_id}; }

    // Finished: drained AND closed, so move past it for good and try the next
    // operator straight away -- it may have splits queued already.
    _counters.operators_drained.fetch_add(1, std::memory_order_relaxed);
    _cursor.fetch_add(1, std::memory_order_relaxed);
  }
  return {};
}

void readahead_scan_manager::on_prefetch_complete(std::size_t,
                                                  std::weak_ptr<op::scan::scan_info> const& split,
                                                  bool issued_io,
                                                  bool allocation_failed)
{
  // Judged at completion, not at issue: issue time says only what the readahead
  // intended, completion time says whether it actually got there first.  The
  // split carries its own stage, so no per-split bookkeeping is needed here.
  auto const task = split.lock();
  _counters.record(
    classify_prefetch(allocation_failed,
                      task != nullptr,
                      issued_io,
                      task ? task->get_scan_stage() : io::cache::scan_stage::disposed));
  // Pairs with the acquire in worker_loop.  The slot is free once the IO
  // settles, not once the split is finally read: the budget caps concurrent IO,
  // and a landed prefetch is doing none.
  _gatekeeper.release();
}

void readahead_scan_manager::worker_loop(const std::stop_token& st)
{
  // Bounds how long a stop can go unnoticed if it races the wait; a stop that
  // lands while parked here wakes the gate directly.
  constexpr auto k_slot_wait = std::chrono::milliseconds{100};
  // Back-off when the order has nothing to prefetch right now.
  constexpr auto k_idle_wait = std::chrono::milliseconds{10};
  // How long to let the evictor work before asking the pool again.
  constexpr auto k_memory_retry = std::chrono::milliseconds{20};

  // Prepare a candidate and, if that succeeds, issue its IO.  Returns whether
  // the prefetch was issued -- which is also whether the completion has taken
  // ownership of the slot.
  auto try_issue = [&](prefetch_candidate const& candidate) {
    while (!st.stop_requested()) {
      auto const prep = candidate.task->prepare_for_prefetching(/*wait_for_eviction=*/true);
      if (prep.ready()) {
        candidate.task->prefetch([weak  = weak_from_this(),
                                  op_id = candidate.operator_id,
                                  split = std::weak_ptr{candidate.task}](
                                   op::scan::scan_info::prefetch_outcome out) noexcept {
          // weak, not shared: a completion firing after the query tore the
          // manager down must not resurrect it.
          if (auto self = weak.lock()) {
            self->on_prefetch_complete(
              op_id, split, out.issued > 0, out.declined_memory_pressure > 0);
          }
        });
        return true;
      }
      // Nothing was refused for want of memory, so there is nothing to wait for:
      // this split has no request to prepare at all.
      if (prep.failed == 0) {
        _counters.record(prefetch_outcome_kind::nothing_to_issue);
        return false;
      }
      // The pool had nothing to give.  Let the evictor work, then ask again --
      // unless the consumer reached the split meanwhile, in which case a
      // prefetch would only duplicate the read it is already doing.
      _counters.memory_retries.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(k_memory_retry);
      if (candidate.task->has_fallen_behind()) {
        _counters.record(prefetch_outcome_kind::skipped_fell_behind);
        return false;
      }
    }
    // Stopped mid-preparation: the pool never satisfied it.
    _counters.record(prefetch_outcome_kind::skipped_memory_pressure);
    return false;
  };

  while (!st.stop_requested()) {
    // Every operator drained: nothing will ever be worth prefetching again.
    if (_cursor.load(std::memory_order_relaxed) >= _ordered_work_queues.size()) { break; }

    // One slot per in-flight prefetch.  Past this point the slot is held and has
    // to be released exactly once: by the completion when the IO was issued, or
    // by one of the give-up paths below.
    if (!_gatekeeper.acquire_for(k_slot_wait)) {
      _counters.gate_timeouts.fetch_add(1, std::memory_order_relaxed);
      continue;
    }
    if (st.stop_requested()) {
      _gatekeeper.release();
      break;
    }

    auto candidate = get_next_prefetching_candidate();
    if (!candidate) {
      // Nothing to read ahead of yet.  Hand the slot straight back rather than
      // sit on it -- budget held by an idle worker is budget the rest of the
      // order cannot use.
      _counters.idle_polls.fetch_add(1, std::memory_order_relaxed);
      _gatekeeper.release();
      std::this_thread::sleep_for(k_idle_wait);
      continue;
    }

    _counters.candidates_taken.fetch_add(1, std::memory_order_relaxed);
    if (!try_issue(candidate)) { _gatekeeper.release(); }
  }
}

void readahead_scan_manager::reset() { stop(); }

void readahead_scan_manager::arm_prefetching()
{
  // Once: reloading a live gatekeeper would forget how much the executor is
  // currently competing and hand the readahead a budget it has already spent.
  //
  // This is also the only arming there is -- the gatekeeper starts with no
  // tickets, so the worker's acquire simply times out until this runs.
  if (_prefetching_started.exchange(true)) { return; }
  _gatekeeper.reload();
}

}  // namespace sirius::scan_manager
