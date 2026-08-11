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

#include "io/cache/types.hpp"
#include "planner/query_index.hpp"
#include "scan_manager/prefetching_scheduler.hpp"

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stop_token>
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
/// The worker keeps the backend's scan budget occupied: every time a split
/// reports progress it re-checks how many scans are in flight and, while there
/// is room, pulls the next operator off @ref prefetching_scheduler and issues
/// prefetch IO for one of its splits.
///
/// Held by @c shared_ptr (splits report into it from their own threads) and
/// inherits @c enable_shared_from_this so an in-flight prefetch completion can
/// find the manager again without keeping it alive.
class readahead_scan_manager : public std::enable_shared_from_this<readahead_scan_manager> {
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
  void start(std::size_t budget);

  /// Request the worker to stop and join it.  Safe to call when not started.
  void stop() noexcept;

  [[nodiscard]] bool is_running() const noexcept;

  /// Seed the per-operator readahead order for @p query.
  void prepare_for_query(const sirius::planner::query& query);

  /// Record a split under the operator that produced it.  Splits are kept in
  /// emission order, which is the order the worker prefetches them in.
  void register_scan_task(std::shared_ptr<op::scan::scan_info> const& task,
                          std::size_t operator_id);

  /// Report that @p task, emitted under @p operator_id, reached @p stage.
  /// Wakes the worker so a freed slot is refilled.
  void update(std::size_t operator_id,
              const op::scan::scan_info* task,
              io::cache::scan_stage stage);

  /// The scan that should be prefetched next, or nullptr once the query's
  /// prefetching order is exhausted.  Consumes one unit of the current step's
  /// quantum — see @ref prefetching_scheduler for the rotation rules.
  [[nodiscard]] op::sirius_physical_operator* get_next_prefetching_operator();

  /// Scans currently occupying a slot against the backend's budget.  Exposed
  /// for tests and diagnostics; see @ref is_ongoing for what counts.
  [[nodiscard]] std::size_t ongoing_scans() const;

  void reset();

 private:
  /// One emitted split.
  struct task_entry {
    std::weak_ptr<op::scan::scan_info> task;
    io::cache::scan_stage stage{io::cache::scan_stage::none};
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

  /// True once @p operator_id has reported @c disposed and none of its splits
  /// are still live and unfinished.  A single split reaching @c disposed says
  /// nothing about the operator, which is why the scheduler is only told the
  /// operator is finished once its split list has drained too.
  [[nodiscard]] bool is_operator_depleted(std::size_t operator_id) const;

  /// Pull work off the scheduler while the budget has room.  Marks each chosen
  /// split prefetched so it is only issued once.  Caller must hold @ref _mutex.
  [[nodiscard]] std::vector<pending_prefetch> collect_prefetch_batch_locked();

  /// Issue the collected prefetches.  Must run WITHOUT @ref _mutex held:
  /// prefetch_async lands IO and may invoke its completion inline.
  void issue_prefetches(std::vector<pending_prefetch> batch);

  /// A split's prefetch IO finished; free its slot and wake the worker.
  void on_prefetch_complete(std::size_t operator_id, const op::scan::scan_info* task);

  /// Waits for a scan to report progress, then tops the in-flight scan set back
  /// up from the scheduler.  Sleeps rather than polls: the only thing that can
  /// change what should be prefetched next is an @ref update, so the loop is
  /// driven by those instead of a timer.
  void worker_loop(const std::stop_token& st);

  mutable std::mutex _mutex;
  std::unordered_map<std::size_t, operator_state> _by_operator;
  /// Cursor over the query's prefetching order.  Guarded by @ref _mutex — the
  /// scheduler does no locking of its own.
  prefetching_scheduler _scheduler;

  /// Scan tasks the worker tries to keep in flight; 0 means "backend opted out".
  std::size_t _budget{0};
  /// Set by @ref update, cleared by the worker.  A flag rather than a counter:
  /// several updates arriving while the worker is busy collapse into one pass,
  /// which is what we want — the pass reads the current cursor, not a backlog.
  bool _wake{false};
  std::condition_variable_any _cv;
  std::stop_source _stop_source;
  std::jthread _worker;
};

}  // namespace sirius::scan_manager
