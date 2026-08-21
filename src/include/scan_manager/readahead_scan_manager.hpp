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
#include "planner/query_index.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/prefetching_scheduler.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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

/// Per-query readahead bookkeeping for GPU scans.  Seeded from the query's
/// prefetching order, then driven by the scans as they advance.
///
/// The worker keeps the backend's scan budget occupied: every time available
/// work or split progress changes, it fills open slots from the scheduler's
/// current group, then walks later groups when the preferred group has no
/// emitted split ready. Later-group lookahead does not consume the preferred
/// group's cursor or quantum.
///
/// Held by @c shared_ptr (splits report into it from their own threads) and
/// inherits @c enable_shared_from_this so an in-flight prefetch completion can
/// find the manager again without keeping it alive.
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
};

/// What one query's readahead did, in the four outcomes that matter.
///
/// Deliberately a per-manager member rather than another slice of the global
/// @c prefetch_census: a readahead manager lives for exactly one query, so the
/// numbers are already scoped to one and need no epoch or delta bookkeeping to
/// be read.
struct readahead_counters {
  /// @ref prefetch_outcome_kind::prefetched -- the readahead got there first.
  std::atomic<std::uint64_t> prefetched{0};
  /// @ref prefetch_outcome_kind::wait_for_prefetch.
  std::atomic<std::uint64_t> wait_for_prefetch{0};
  /// Every attempt that produced no usable prefetch: the two reasons below plus
  /// @ref prefetch_outcome_kind::nothing_to_issue, which is why it can exceed
  /// their sum.
  std::atomic<std::uint64_t> skipped{0};
  /// @ref prefetch_outcome_kind::skipped_memory_pressure.
  std::atomic<std::uint64_t> skipped_memory_pressure{0};
  /// @ref prefetch_outcome_kind::skipped_fell_behind.
  std::atomic<std::uint64_t> skipped_fell_behind{0};
};

class readahead_scan_manager : public std::enable_shared_from_this<readahead_scan_manager>,
                               public exec::query_stage_listener {
 public:
  readahead_scan_manager() = default;
  /// Stops and joins the worker.
  ~readahead_scan_manager();
  readahead_scan_manager(readahead_scan_manager const&)            = delete;
  readahead_scan_manager& operator=(readahead_scan_manager const&) = delete;

  /// Start the worker that drives the prefetching scheduler.  Idempotent: a
  /// second call while the worker is running does nothing.
  ///
  /// @p budget is the backend's @c ioctx::n_max_concurrent_scans — the number
  /// of scan tasks the worker will try to keep in flight.  A budget of zero
  /// means the backend opts out, and start() is then a no-op: there is no point
  /// running a scheduler that may never issue anything.
  void start(std::size_t budget, prefetch_strategy strategy = prefetch_strategy::eager);

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

  /// Request the worker to stop and join it.  Safe to call when not started.
  void stop() noexcept;

  [[nodiscard]] bool is_running() const noexcept;

  /// Seed the per-operator readahead order for @p query.
  void prepare_for_query(const sirius::planner::query& query);

  /// Seed directly from a prefetching order, bypassing plan analysis.  The
  /// query path is this plus @c query_index::prefetching_orders.
  void prepare_for_order(std::span<const planner::prefetch_step> order);

  /// Record a split under the operator that produced it.  Splits are kept in
  /// emission order, which is the order the worker prefetches them in.
  void register_scan_task(std::shared_ptr<op::scan::scan_info> const& task,
                          std::size_t operator_id);

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

  /// The scan that should be prefetched next, or nullptr once the query's
  /// prefetching order is exhausted.  Consumes one unit of the current step's
  /// quantum — see @ref prefetching_scheduler for the rotation rules.
  [[nodiscard]] op::sirius_physical_operator* get_next_prefetching_operator();

  /// Scans currently occupying a slot against the backend's budget.  Exposed
  /// for tests and diagnostics; see @ref is_ongoing for what counts.
  [[nodiscard]] std::size_t ongoing_scans() const;

  /// Whether any registered split is still waiting to be prefetched.  Used to
  /// tell "declined because there was nothing to do" from "declined while work
  /// was sitting there".
  [[nodiscard]] bool has_unprefetched_work() const;

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
  /// This query's readahead outcomes; see @ref readahead_counters.
  readahead_counters _counters;

  /// One emitted split.
  struct task_entry {
    std::weak_ptr<op::scan::scan_info> task;
    io::cache::scan_stage stage{io::cache::scan_stage::none};
    /// Position in the order the readahead issued prefetches, and in the order
    /// the executor first read them.  0 means unset.  Comparing the two is how
    /// prefetch order is checked against execution order.
    std::uint64_t prefetch_rank{0};
    /// The worker issued prefetch IO for this split.
    bool prefetched{false};
    /// That IO has completed, so the split is cache-resident and its read costs
    /// no further IO.
    bool prefetch_done{false};
  };

  struct operator_state {
    planner::scheduling_mode mode{planner::scheduling_mode::pipeline};
    std::size_t branch_id{0};
    std::size_t count{0};
    /// Most recent stage reported by any of this operator's splits.
    io::cache::scan_stage stage{io::cache::scan_stage::none};
    /// The producer has emitted its last split; @ref tasks will not grow again.
    bool closed{false};
    /// Splits in emission order.  Never erased — indices below point into it.
    std::vector<task_entry> tasks;
    std::unordered_map<const op::scan::scan_info*, std::size_t> index;
  };

  /// One split's worth of prefetch work, collected under the lock and issued
  /// outside it.  Holds the split alive so its datasources outlive the call.
  struct pending_prefetch {
    std::shared_ptr<op::scan::scan_info> task;
    std::size_t operator_id{0};
  };

  /// Does this split occupy a slot against the budget?
  ///
  /// A split we prefetched holds a slot only while that IO is in flight — once
  /// it lands, the split is cache-resident and reading it costs no IO. A split
  /// we did NOT prefetch holds a slot once the executor starts reading it,
  /// because that read is doing the IO itself.  Disposed splits never count.
  [[nodiscard]] static bool is_ongoing(task_entry const& t) noexcept;

  [[nodiscard]] std::size_t count_ongoing_locked() const;

  /// True once @p operator_id's producer has closed and none of its splits are
  /// still live and unfinished.  A single split reaching @c disposed says
  /// nothing about the operator, and neither does "every split registered so
  /// far is disposed" — splits are emitted progressively, so that state is also
  /// what an operator looks like between two waves.  Only @ref
  /// mark_operator_closed distinguishes the two.
  [[nodiscard]] bool is_operator_depleted(std::size_t operator_id) const;

  /// True when every operator in @p operator_ids has had its producer close, so
  /// the group will emit no further splits.  Gates read-ahead into later groups:
  /// a group that is only momentarily empty must be waited for, not skipped, or
  /// the prefetch order stops matching the execution order.  Caller must hold
  /// @ref _mutex.
  [[nodiscard]] bool group_is_closed_locked(std::span<const std::size_t> operator_ids) const;

  /// Publish the operator's progress to the scheduler: @c disposed once the
  /// operator is depleted, otherwise @p reported — except that a bare
  /// @c disposed from one split is swallowed, since that is the scheduler's
  /// retirement edge and only @ref is_operator_depleted may trip it.  Caller
  /// must hold @ref _mutex; wakes the worker.
  void publish_stage_locked(std::size_t operator_id, io::cache::scan_stage reported);

  /// Pull work off the scheduler while the budget has room.  Marks each chosen
  /// split prefetched so it is only issued once.  Caller must hold @ref _mutex.
  [[nodiscard]] std::vector<pending_prefetch> collect_prefetch_batch_locked();

  /// Issue the collected prefetches.  Must run WITHOUT @ref _mutex held:
  /// prefetch_async lands IO and may invoke its completion inline.
  void issue_prefetches(std::vector<pending_prefetch> batch);

  /// A split's prefetch IO finished; free its slot and wake the worker.
  /// Classify and record one settled prefetch attempt, then free its slot.
  ///
  /// Takes the split's report as the two facts it actually carries rather than
  /// the struct itself: everything else the verdict needs is this manager's own
  /// per-split state, read under @ref _mutex at this instant.
  void on_prefetch_complete(std::size_t operator_id,
                            const op::scan::scan_info* task,
                            bool allocation_failed,
                            bool issued_io);

  /// Fold one verdict into @ref _counters.  Caller holds @ref _mutex; the
  /// counters are atomics so the lock is incidental, not load-bearing.
  void record_prefetch_outcome(prefetch_outcome_kind kind) noexcept;

  /// Waits for a scan to report progress, then tops the in-flight scan set back
  /// up from the scheduler.  Sleeps rather than polls: the only thing that can
  /// change what should be prefetched next is an @ref update_scan_state, so the loop is
  /// driven by those instead of a timer.
  void worker_loop(const std::stop_token& st);

  mutable std::mutex _mutex;
  std::unordered_map<std::size_t, operator_state> _by_operator;
  /// Cursor over the query's prefetching order.  Guarded by @ref _mutex — the
  /// scheduler does no locking of its own.
  prefetching_scheduler _scheduler;

  /// Scan tasks the worker tries to keep in flight; 0 means "backend opted out".
  std::size_t _budget{0};
  prefetch_strategy _strategy{prefetch_strategy::eager};
  /// Prefetches @c opportunistic has been invited to issue but has not yet.
  /// Unused by @c eager, which is always invited.
  std::size_t _credits{0};
  /// One-shot permission to read past the current group even though it can
  /// still emit.  Granted by a memory downgrade and consumed by the next
  /// collect pass -- see @ref group_is_closed_locked for why it is otherwise
  /// refused.
  bool _may_run_ahead{false};
  /// Set by @ref update_scan_state, cleared by the worker.  A flag rather than a counter:
  /// several updates arriving while the worker is busy collapse into one pass,
  /// which is what we want — the pass reads the current cursor, not a backlog.
  bool _wake{false};
  std::condition_variable_any _cv;
  std::stop_source _stop_source;
  std::jthread _worker;

  /// Fold the interval since the last call into the occupancy account, using the
  /// scan count as it was over that interval.  Called under @c _mutex at every
  /// point the count can change, so the account is exact rather than sampled.
  void note_active_locked();

  std::chrono::steady_clock::time_point _active_mark{};
  std::size_t _active_count{0};

  /// Monotonic rank sources for the two orders; see task_entry::prefetch_rank.
  std::uint64_t _next_prefetch_rank{0};
  std::uint64_t _next_read_rank{0};
  /// Prefetch rank of the split read most recently -- an inversion is a read
  /// whose prefetch came earlier than that.
  std::uint64_t _last_read_prefetch_rank{0};
};

}  // namespace sirius::scan_manager
