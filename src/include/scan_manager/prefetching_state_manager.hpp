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

#include "creator/config.hpp"
#include "io/cache/types.hpp"
#include "query_id.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::planner {
class query;
}  // namespace sirius::planner

namespace sirius::scan_manager {

/**
 * @brief Per-query bookkeeping for the scan prefetch ladder.
 *
 * One instance per query, created fresh by @c sirius_scan_manager::prepare_for_query and shared
 * (by @c shared_ptr) with every @c op::scan::scan_operator_input the query produces. Each split
 * reports its ladder progress here; the counters drive the prefetch look-ahead policy and the
 * end-of-query summary log.
 *
 * **A fresh instance per query is mandatory, not stylistic.** A split can outlive its query:
 * @c ~split_connector runs when the query's pipelines are destroyed, which is *before* the scan
 * manager is reset, and a task can be mid-flight on a GPU executor thread. A straggler from query
 * N must decrement query N's instance, not query N+1's. Resetting a long-lived instance would
 * silently skew the next query's counts.
 *
 * All counters are relaxed atomics, following @c io::cache::prefetching_cache::counters. Every
 * mutating method is @c noexcept and non-blocking: they are called from destructors, from GPU
 * executor threads mid-pipeline, and during stack unwinding. A @c noexcept method here therefore
 * has to be *genuinely* non-throwing — see the per-method notes for the two places (@ref clean_up
 * and the two hook targets) where the body must contain its own exceptions rather than let them
 * reach @c std::terminate.
 */
class prefetching_state_manager {
 public:
  /// @brief Tunables, mirrored from @c scan_manager_config.
  struct config {
    /// Host-cache bytes above which look-ahead prefetching backs off. 0 disables the check.
    /// @note Reserved — not yet read; the look-ahead walk the hooks will drive is a follow-up.
    std::size_t memory_threshold{0};
    /// Prefetch look-ahead window: the most queued splits a single hook invocation will hint.
    /// Not a concurrency limit.
    /// @note Reserved — not yet read; the look-ahead walk the hooks will drive is a follow-up.
    std::size_t prefetch_lookahead_window{4};
  };

  /// @brief Immutable view of the counters. Plain integers — safe to copy and log.
  struct counters_snapshot {
    std::uint64_t n_inputs_created{0};    ///< scan_operator_input constructed
    std::uint64_t n_inputs_disposed{0};   ///< scan_operator_input destroyed
    std::uint64_t n_metadata_created{0};  ///< prefetch(metadata_created)
    std::uint64_t n_task_queued{0};       ///< prefetch(task_queued)
    std::uint64_t n_task_prepared{0};     ///< prefetch(task_preprocessing)
    std::uint64_t n_task_completed{0};    ///< prefetch(disposable)
    std::int64_t n_live{0};               ///< created - disposed; a gauge, may be read mid-flight
  };

  explicit prefetching_state_manager(config cfg) noexcept;

  prefetching_state_manager(const prefetching_state_manager&)            = delete;
  prefetching_state_manager& operator=(const prefetching_state_manager&) = delete;

  /**
   * @brief Bind this instance to @p query, zero the counters, and re-attach the hook targets.
   *
   * Captures @c query.query_id() only — it does not retain a reference. @c sirius::planner::query
   * is destroyed before the scan manager is reset, so retaining one would dangle.
   */
  void prepare_for_query(const sirius::planner::query& query) noexcept;

  /**
   * @brief Detach from the current query and log the final summary.
   *
   * Takes no argument **by design**: there is no point in the teardown order at which a
   * @c const query& is still valid. @c sirius::planner::query is destroyed before
   * @c sirius_scan_manager::reset, which is this method's only caller. Straggler splits destroyed
   * after this call still decrement the counters harmlessly — nobody reads them again.
   *
   * Also **latches the detached flag**, after which @ref on_task_queue_depleted and
   * @ref on_task_not_created return immediately. That is worth having because the @c weak_ptr the
   * hooks are installed with cannot expire while any split still holds a @c shared_ptr to this
   * manager, so it still locks successfully in exactly the window where the query's connectors
   * have already been destroyed. @ref prepare_for_query clears the flag again.
   *
   * @warning The flag narrows that window from one side only; it does not close it and it is
   *          **not** a liveness guarantee for anything a hook might walk. This method is reached
   *          from @c sirius_scan_manager::reset, which @c SiriusContext::run_mandatory_cleanup
   *          calls long *after* it has destroyed the query's pipelines and every
   *          @c split_connector with them — so the flag reads @c false for that whole interval,
   *          with the connectors already gone. A hook already past the check when this runs is
   *          also still inside. A hook body must own what it touches in its own right.
   *
   * @note The summary this logs comes from @ref summary, which builds a @c std::string and can
   *       therefore throw (@c std::bad_alloc, or a formatting error). Because this method is
   *       @c noexcept, an escaping exception would call @c std::terminate during query teardown.
   *       The implementation **must** wrap the summary/log call in @c try/catch(...) and swallow
   *       the failure: losing a diagnostic line is always preferable to aborting the process on
   *       the cleanup path.
   */
  void clean_up() noexcept;

  /// @brief The query this instance is bound to. Zero before @ref prepare_for_query.
  [[nodiscard]] sirius::query_id_t query_id() const noexcept;

  /**
   * @brief Record that a split reached ladder rung @p site.
   *
   * Called by @c op::scan::scan_operator_input::prefetch **before** its metadata check, so
   * resident (pinned-cache) splits are counted too — they climb the same ladder.
   * @c io::cache::prefetching_stage::none is ignored.
   *
   * One relaxed @c fetch_add and nothing else: this runs on four different thread families,
   * including a task_creator worker that holds @c sirius_pipeline::_status_mutex. Taking any
   * lock or allocating here is forbidden.
   */
  void update(io::cache::prefetching_stage site) noexcept;

  /// @brief Record a @c op::scan::scan_operator_input construction. Increments created and live.
  ///        Two relaxed atomics, no locks, no allocation.
  void on_input_created() noexcept;

  /// @brief Record a @c op::scan::scan_operator_input destruction. Increments disposed, decrements
  ///        live. Called from @c ~scan_operator_input, which runs on GPU executor threads and
  ///        during stack unwinding — must never throw, block, or allocate. Two relaxed atomics.
  void on_input_disposed() noexcept;

  /**
   * @brief Hook target for @c creator::task_creator::task_queue_depleted_hook. Non-blocking.
   *
   * Runs on the task_scheduler management thread with no lock held. Returns immediately once
   * @ref clean_up has detached this instance, which shortens — but does not close — the window in
   * which a straggler hook can fire against a query whose connectors are already gone; see the
   * @c \@warning on @ref clean_up.
   *
   * @note **Reserved — not yet implemented** beyond the detach gate. The bounded
   *       @c split_connector::prefetch_if walk this hook exists to trigger needs the query's
   *       connectors, which reach this object with the scan-manager wiring; see the TODO on the
   *       implementation for the constraints that walk has to satisfy — including that it must
   *       establish connector liveness itself rather than infer it from the detach gate.
   */
  void on_task_queue_depleted() noexcept;

  /**
   * @brief Hook target for @c creator::task_creator::task_not_created_hook. Non-blocking.
   *
   * Runs on the task_creator's single manager thread with no lock held — blocking here stalls
   * task creation engine-wide. Same detach gate and same constraints as
   * @ref on_task_queue_depleted.
   *
   * @note **Reserved — not yet implemented** beyond the detach gate.
   *
   * @param requested The operator the failed request started from. Borrowed for the call only.
   * @param kind      Whether the request was active or speculative look-ahead.
   */
  void on_task_not_created(const op::sirius_physical_operator* requested,
                           creator::request_type kind) noexcept;

  /// @brief Whether @ref clean_up has detached this instance from its query. Relaxed load.
  ///        Exposed so the detach gate is observable without driving a hook.
  [[nodiscard]] bool is_detached() const noexcept;

  /**
   * @brief A torn but cheap read of every counter.
   *
   * The fields are loaded independently, so the snapshot need not correspond to any single instant
   * (e.g. @c n_inputs_created may lag a split that has already been disposed). Best-effort, exactly
   * like @c io::cache::prefetching_cache::prepare_for_query's. Fine for logging and for a
   * look-ahead heuristic; not a synchronization point. Genuinely @c noexcept: seven relaxed loads
   * into a by-value aggregate of integers, no allocation.
   */
  [[nodiscard]] counters_snapshot snapshot() const noexcept;

  /// @brief One-line human-readable summary, in the style of
  ///        @c io::cache::prefetching_cache::summary. Builds a string, so it allocates and is
  ///        deliberately **not** @c noexcept; every @c noexcept caller must contain its throw.
  [[nodiscard]] std::string summary() const;

  /// @brief The tunables this instance was constructed with.
  [[nodiscard]] const config& get_config() const noexcept;

 private:
  /// Mirrors io::cache::prefetching_cache::counters: relaxed atomics, one per snapshot field.
  struct counters {
    std::atomic<std::uint64_t> n_inputs_created{0};
    std::atomic<std::uint64_t> n_inputs_disposed{0};
    std::atomic<std::uint64_t> n_metadata_created{0};
    std::atomic<std::uint64_t> n_task_queued{0};
    std::atomic<std::uint64_t> n_task_prepared{0};
    std::atomic<std::uint64_t> n_task_completed{0};
    std::atomic<std::int64_t> n_live{0};
  };

  const config _cfg;
  std::atomic<sirius::query_id_t> _query_id{};
  /// Latched by @ref clean_up, cleared by @ref prepare_for_query. Gates both hook targets.
  /// Relaxed: it publishes nothing, and a hook that reads it one instant stale is exactly as
  /// harmless as one that fired one instant earlier.
  std::atomic<bool> _detached{false};
  counters _counters;
};

}  // namespace sirius::scan_manager
