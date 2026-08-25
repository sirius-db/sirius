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

#pragma once

#include "exec/query_stage_manager.hpp"
#include "io/cache/types.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "planner/query_index.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/gatekeeper.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {
class scan_info;
}  // namespace sirius::op::scan

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/// How one prefetch attempt turned out, judged when it COMPLETES against what
/// the consumer was doing at that instant -- not when it was issued.  Issue time
/// says only what the readahead intended; completion time says whether it
/// actually got there first, which is the whole question.
///
/// Exactly one applies per attempt, in this priority order.  The first two rules
/// overlap on @c reading -- a consumer that is reading is also "preparing or
/// higher" -- and @ref wait_for_prefetch wins, because it is the more specific
/// statement: the prefetch did land, the consumer was simply already on it.
enum class prefetch_outcome_kind : std::uint8_t {
  /// The pool could not attach staging buffers, so nothing was ever claimable.
  skipped_memory_pressure,
  /// The split is gone (its weak_ptr expired) or the consumer had reached
  /// @c preparing or beyond by the time the prefetch settled: the readahead was
  /// working behind the executor rather than in front of it.
  skipped_fell_behind,
  /// IO went out and had not finished by the time the consumer started reading,
  /// so the read is waiting on a prefetch that is on the right split but late.
  wait_for_prefetch,
  /// IO went out and completed while the consumer had still not reached this
  /// split -- the readahead got there first, which is the point of it.
  prefetched,
  /// The attempt had nothing to issue: a split with no ranges, or a scan with no
  /// prefetching cache to issue into (@c cache.mode: os warms the page cache and
  /// nothing else).  Neither a win nor a miss.
  nothing_to_issue,

  /// Not an outcome -- the enumerator count, so the counters can be an array
  /// indexed by kind.  Keep it last, and add new kinds above it: a new kind then
  /// gets its counter for free instead of needing a matching case somewhere else.
  klast,
};

/// What one query's readahead did: every outcome, plus where the worker's time
/// went and how much the executor competed with it.
///
/// Deliberately a per-manager member rather than another slice of the global
/// @c prefetch_census: a readahead manager lives for exactly one query, so the
/// numbers are already scoped to one and need no epoch or delta bookkeeping to
/// be read.
struct readahead_counters {
  // ---- outcomes: exactly one per attempt that reached a decision ------------
  /// One counter per @ref prefetch_outcome_kind, indexed by the enum itself.
  std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(prefetch_outcome_kind::klast)>
    outcomes{};

  void record(prefetch_outcome_kind kind) noexcept
  {
    outcomes[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t outcome(prefetch_outcome_kind kind) const noexcept
  {
    return outcomes[static_cast<std::size_t>(kind)].load(std::memory_order_relaxed);
  }

  // ---- candidate selection -------------------------------------------------
  /// Splits handed to the worker.  The denominator the outcomes above divide.
  std::atomic<std::uint64_t> candidates_taken{0};
  /// Dropped at the head of a queue because the split was already gone.
  std::atomic<std::uint64_t> dropped_expired{0};
  /// Dropped at the head of a queue because the consumer had reached it first --
  /// the readahead never even got to attempt these, which is why they are
  /// counted apart from @ref skipped_fell_behind.
  std::atomic<std::uint64_t> dropped_fell_behind{0};
  /// Operators retired: drained and closed, so the cursor moved past them.
  std::atomic<std::uint64_t> operators_drained{0};

  // ---- pacing: where the worker's time went --------------------------------
  /// Preparations retried after the pool had nothing to give.
  std::atomic<std::uint64_t> memory_retries{0};
  /// Took a slot and found no candidate to spend it on.
  std::atomic<std::uint64_t> idle_polls{0};
  /// Could not even take a slot: every one was already in flight.
  std::atomic<std::uint64_t> gate_timeouts{0};

  // ---- competition from the executor ---------------------------------------
  /// Reads the readahead never covered, which therefore spent a ticket of its
  /// budget on their own IO.
  std::atomic<std::uint64_t> cold_read_tickets{0};
  /// Of those, the ones the budget could not cover, so the readahead was left
  /// over-subscribed until the ticket came back.  A quiet readahead with a high
  /// count here was crowded out; one with a low count simply had nothing to do.
  std::atomic<std::uint64_t> borrowed{0};
};

/// Per-query readahead for GPU scans.
///
/// One work queue per scan operator, held in execution order and fed by the
/// coalescer as it emits splits.  A single worker walks them with a cursor that
/// only moves forward -- an operator is left behind once its producer has closed
/// and its queue is drained -- spending a @ref gatekeeper ticket on each prefetch
/// it issues and getting it back when that IO settles.
///
/// Held by @c shared_ptr (splits report into it from their own threads) and
/// inherits @c enable_shared_from_this so an in-flight prefetch completion can
/// find the manager again without keeping it alive.
class readahead_scan_manager : public std::enable_shared_from_this<readahead_scan_manager>,
                               public exec::query_stage_listener {
 public:
  /// Registers for @p stage_manager's events; the mailbox is only drained once
  /// @ref start runs.
  ///
  /// @p budget is the serving backend's @c ioctx::n_max_concurrent_scans -- the
  /// number of scan IOs worth keeping in flight, which the gatekeeper rations
  /// between the readahead and the executor.  Zero means the backend opts out,
  /// and @ref start is then a no-op: there is no point running a worker that may
  /// never issue anything.
  readahead_scan_manager(exec::query_stage_manager& stage_manager, std::size_t budget)
    : exec::query_stage_listener(stage_manager,
                                 // The four executor-idle signals this readahead acts on, and
                                 // nothing else. Must list every hook overridden below: one left
                                 // out here is never delivered, however correctly it is written.
                                 {exec::query_stage_event_type::task_deployed,
                                  exec::query_stage_event_type::task_queue_empty,
                                  exec::query_stage_event_type::memory_downgrade_for_task,
                                  exec::query_stage_event_type::wait_for_memory_for_task}),
      _budget(budget),
      _gatekeeper(static_cast<int>(budget))
  {
  }
  /// Stops and joins both workers -- the prefetch worker and the event listener.
  ~readahead_scan_manager() override;
  readahead_scan_manager(readahead_scan_manager const&)            = delete;
  readahead_scan_manager& operator=(readahead_scan_manager const&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override { return "readahead"; }

  /// Seed the per-operator readahead order for @p query.
  void prepare_for_query(const sirius::planner::query& query);

  /// Start the worker that drives the readahead.  Idempotent: a second call
  /// while the worker is running does nothing.  A no-op when the budget this was
  /// built with is zero.
  void start(prefetch_strategy strategy = prefetch_strategy::eager);

  /// A task reached an executor.  Under @c prefetch_strategy::opportunistic a
  /// deployment that is NOT a scan is the signal to issue one prefetch: a
  /// pipeline thread just went to compute rather than to read, so the device
  /// has capacity the executor is not about to use.  Ignored under
  /// @c eager, which does not wait to be told.
  void on_task_deployed(query_id_t query_id,
                        std::size_t operator_id,
                        op::SiriusPhysicalOperatorType operator_type,
                        int gpu_id) noexcept override;

  /// The scheduler's queue ran dry.  Nothing is waiting to be dispatched, so
  /// whatever the executor is doing it is not about to read -- the strongest
  /// idle signal available, and under @c opportunistic the moment to fill the
  /// budget rather than trickle.  Ignored under @c eager.
  void on_task_queue_empty() noexcept override;

  /// An executor is spilling to make room for a task.  The GPU does no work for
  /// the duration, so under @c opportunistic this both refills the credits and
  /// lifts the restriction that read-ahead stay within the current group: with
  /// the executor stalled there is no order left to stay in step with, and the
  /// far more useful thing is to be further ahead when it resumes.  Ignored
  /// under @c eager, which is already reading as far ahead as its budget allows.
  void on_memory_downgrade_for_task(query_id_t query_id,
                                    std::size_t operator_id,
                                    int gpu_id,
                                    std::size_t shortfall_bytes) noexcept override;

  /// An executor is about to block waiting for memory to free up.  Same shape
  /// of opportunity as @ref on_memory_downgrade_for_task and handled the same
  /// way: the task is parked and the GPU is about to be idle, so the device's
  /// IO path is free and read-ahead costs the executor nothing it wanted.
  /// Ignored under @c eager, which is already reading as far ahead as it can.
  void on_wait_for_memory_for_task(query_id_t query_id,
                                   std::size_t operator_id,
                                   int gpu_id,
                                   std::size_t bytes_needed) noexcept override;

  /// Report that @p task, emitted under @p operator_id, reached @p stage.
  /// Wakes the worker so a freed slot is refilled.
  void update_scan_state(std::size_t operator_id,
                         const op::scan::scan_info* task,
                         io::cache::scan_stage stage);

  /// Report that @p operator_id's producer has finished: every split it will
  /// ever emit has been registered.  Until this arrives the operator cannot be
  /// retired, because "all splits registered so far are done" is indistinguishable
  /// from "the next split has not been emitted yet" — and retirement is one-way,
  /// so guessing wrong permanently drops the operator out of the prefetch order.
  ///
  /// Called from every close path of the producer, including failures: a slot
  /// that closed with an exception emits nothing further either.
  void mark_operator_closed(std::size_t operator_id);

  [[nodiscard]] bool is_running() const noexcept;

  /// Record a split under the operator that produced it.  Splits are kept in
  /// emission order, which is the order the worker prefetches them in.
  void register_scan_task(std::shared_ptr<op::scan::scan_info> const& task,
                          std::size_t operator_id);

  /// Request the worker to stop and join it.  Safe to call when not started.
  void stop() noexcept;

  void reset();

  /// The rule behind @ref prefetch_outcome_kind, a pure function so it can be
  /// read -- and tested -- without a live manager.  @p stage is the consumer's
  /// stage at completion; @p split_alive is false once the split's weak_ptr has
  /// expired.
  [[nodiscard]] static prefetch_outcome_kind classify_prefetch(
    bool allocation_failed, bool split_alive, bool issued_io, io::cache::scan_stage stage) noexcept;

  /// This query's readahead outcomes.  Exposed for tests and diagnostics; the
  /// log line built from them is @ref summary.
  [[nodiscard]] readahead_counters const& counters() const noexcept { return _counters; }

  /// One-line account of this query's readahead, in the shape
  /// @c prefetching_cache::summary uses.  Safe to call at any time; the
  /// counters are relaxed atomics and a concurrent update only means the line
  /// is a moment stale.
  [[nodiscard]] std::string summary() const;

 private:
  /// One split picked for prefetching, with the operator it was emitted under.
  /// The completion needs both, and only the queue it came from knows the
  /// second, so they travel together rather than being looked up again later.
  struct prefetch_candidate {
    std::shared_ptr<op::scan::scan_info> task;
    std::size_t operator_id{0};

    explicit operator bool() const noexcept { return task != nullptr; }
  };

  prefetch_candidate get_next_prefetching_candidate();

  /// Classify and record one settled prefetch, then hand its slot back.
  /// @p split is weak: a completion firing after the split was consumed still
  /// has to release the slot, and "gone" is itself an answer.
  void on_prefetch_complete(std::size_t operator_id,
                            std::weak_ptr<op::scan::scan_info> const& split,
                            bool issued_io,
                            bool allocation_failed);

  void arm_prefetching();

  /// This query's readahead outcomes; see @ref readahead_counters.
  readahead_counters _counters;

  std::stop_source _stop_source;
  std::jthread _prefetch_worker;

  struct prefetch_work_queue {
    std::mutex _mutex;
    std::size_t operator_id{0};
    std::deque<std::weak_ptr<op::scan::scan_info>> _queue;
    bool is_finished{false};

    void close()
    {
      std::lock_guard lock{_mutex};
      is_finished = true;
    }

    void push(const std::shared_ptr<op::scan::scan_info>& task)
    {
      std::lock_guard lock{_mutex};
      if (!is_finished) { _queue.push_back(task); }
    }

    /// The next split worth prefetching from this operator.  Never blocks: the
    /// three answers are what the caller needs to tell apart, and only it knows
    /// what to do about the third.
    ///
    ///   - a task              : prefetch this one
    ///   - engaged, but null   : this operator is finished -- drained AND its
    ///                           producer closed, so it can never emit again and
    ///                           the caller should move past it for good
    ///   - @c std::nullopt     : nothing right now, but the producer is still
    ///                           open; ask again after a push or a close
    std::optional<std::shared_ptr<op::scan::scan_info>> get_next_candidate(
      readahead_counters& counters)
    {
      std::lock_guard lock{_mutex};
      while (!_queue.empty()) {
        // Two reasons to drop an entry without handing it out: its split is gone
        // (nothing left to prefetch), or the consumer already reached it, so a
        // prefetch would only duplicate the read the executor is doing.  Neither
        // can recover -- a split does not come back and scan stages only advance
        // -- so both are discarded rather than left to block the head.
        auto task = _queue.front().lock();
        if (!task) {
          counters.dropped_expired.fetch_add(1, std::memory_order_relaxed);
          _queue.pop_front();
          continue;
        }
        if (task->has_fallen_behind()) {
          counters.dropped_fell_behind.fetch_add(1, std::memory_order_relaxed);
          _queue.pop_front();
          continue;
        }
        // Viable: this is the candidate, and popping it is the hand-off.
        _queue.pop_front();
        return task;
      }
      // Empty.  Which kind of empty is the whole question: closed means never
      // again, open means not yet.
      if (is_finished) { return std::shared_ptr<op::scan::scan_info>{}; }
      return std::nullopt;
    }
  };

  std::unordered_map<size_t, size_t> _operator_id_to_queue_index;
  std::vector<std::unique_ptr<prefetch_work_queue>> _ordered_work_queues;
  /// Scan IOs worth keeping in flight; 0 means the backend opted out.  Declared
  /// ahead of the gatekeeper because it initialises it.
  std::size_t _budget{0};
  /// Built with the manager, so it is reachable the moment this is subscribed
  /// for stage events -- which happens before @ref start runs.  It hands out no
  /// tickets until armed, so existing this early costs nothing.
  gatekeeper _gatekeeper;
  std::atomic<bool> _prefetching_started{false};
  std::atomic<size_t> _cursor{0};

  /// Waits for a scan to report progress, then tops the in-flight scan set back
  /// up from the scheduler.  Sleeps rather than polls: the only thing that can
  /// change what should be prefetched next is an @ref update_scan_state, so the loop is
  /// driven by those instead of a timer.
  void worker_loop(const std::stop_token& st);

  prefetch_strategy _strategy{prefetch_strategy::eager};
};

}  // namespace sirius::scan_manager
