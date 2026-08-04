/*
 * Copyright 2025, Sirius Contributors.
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
#include "op/sirius_physical_operator.hpp"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

namespace sirius::scan_manager {

/**
 * @brief Which resource a batch prefetch should warm.
 *
 * @c io   — file bytes into the host prefetching cache (metadata splits).
 * @c memory — a resident cache batch onto the GPU tier (pinned-entry splits).
 * The two are disjoint: a split is at most one of them.
 */
enum class prefetch_kind : std::uint8_t { io, memory };

/**
 * @brief Bridge between a scan-side producer (the scan manager) and a scan source operator.
 *
 * The connector is a lock-protected queue of pre-built splits. The producer side enqueues
 * splits as they become ready and calls close() when no more will arrive. The consumer
 * pulls splits via get_next_split(), which BLOCKS until either a split is available or
 * the connector has been closed and drained:
 *
 *   - returns std::nullopt           → connector is closed and drained, no more will arrive.
 *   - returns a non-null unique_ptr  → next split.
 *   - throws                         → producer surfaced an error via close(exception_ptr)
 *                                       and the queue is drained.
 *
 * Pushes are gated: only @ref load_balancing_scan_batch_coalescer may enqueue splits, via the
 * @c friend relationship and the private @ref push_split. Its sequencer task is the single
 * producer for every connector in a query. close() and the consumer-side methods remain public
 * so the scan manager (driver loop) and the scan operator can drive the lifecycle.
 */
class split_connector : public std::enable_shared_from_this<split_connector> {
 public:
  split_connector();
  ~split_connector();

  split_connector(const split_connector&)            = delete;
  split_connector& operator=(const split_connector&) = delete;
  split_connector(split_connector&&)                 = delete;
  split_connector& operator=(split_connector&&)      = delete;

  /// \brief Mark the connector as closed: no more splits will be pushed. Idempotent.
  ///        Wakes all waiting consumers.
  ///
  /// \param exception Optional exception captured by the producer. The first
  ///                  non-null exception passed across all close() calls is
  ///                  stored and rethrown by get_next_split() once the queue
  ///                  has been drained. Subsequent close() calls do not
  ///                  overwrite an already-stored exception.
  void close(std::exception_ptr const& exception = nullptr);

  /// \brief Pull the next split, blocking until one is available or the connector
  ///        is closed and drained.
  ///
  /// **Strictly FIFO unless the connector is armed.** The selection policy below runs only when
  /// some pushed split reported @c op::scan::scan_operator_input::can_land_while_queued — i.e.
  /// only when a backend on this connector activates its prefetch at a rung that fires *before*
  /// the dequeue. No shipped backend does, so on every shipped configuration this method is the
  /// same @c move(front) + @c pop_front it was before the selection policy existed: no state
  /// reads, no virtual calls, no datasource walks.
  ///
  /// When armed it is not strictly FIFO: among the leading queued splits it prefers one whose
  /// prefetch has already landed (see @ref select_split_index). Splits are independent units of
  /// work — each becomes its own concurrently-executed task on its own device, in
  /// @c creator::task_creator's dispatch loop — so the engine has no cross-split ordering
  /// requirement. See @ref kSelectionFoldBudget for the second axis of the bound.
  ///
  /// Complexity: **O(1) in the queue length**, on both axes. The armed walk inspects at most
  /// @ref kSelectionWindow splits *and* spends at most @ref kSelectionFoldBudget datasource
  /// folds across them (plus the front split, which is always inspected); it stops early at the
  /// first landed split. The deque erase is within the window, so it moves at most
  /// @c kSelectionWindow-1 pointers, and index 0 is a plain @c pop_front. The unarmed path does
  /// none of the above.
  ///
  /// \warning The wait predicate is and must remain exactly `!_splits.empty() || _closed`, and the
  ///          `_exception` rethrow must remain the first statement after the wait. Preferring a
  ///          non-loading split is safe; **refusing** one is not. A split leaving
  ///          @c prefetch_progress::loading notifies @c io::cache::entry_state's atomic, not this
  ///          connector's condition variable — nothing in @c io/cache knows the condition variable
  ///          exists. On a produced-then-idle connector there is no further @ref push_split, so
  ///          "wait until a non-loading split shows up" is a **permanent** hang, not a delayed one.
  ///          @ref select_split_index therefore always returns a valid index (rule 3), and this
  ///          method always hands out whatever it selected.
  ///
  /// \return std::nullopt when closed and drained without error; the next split
  ///         otherwise.
  /// \throws The exception passed to close() (if any), **before** any queued split is drained.
  std::optional<std::unique_ptr<op::operator_data>> get_next_split();

  /// \brief True iff close() has been called and the queue is drained.
  [[nodiscard]] bool is_closed() const;

  [[nodiscard]] bool has_more_splits() const;

  /**
   * @brief Fire prefetch hints for the queued splits at the head of the connector.
   *
   * Inspects **at most** @p upto_n splits from the front — a bounded look-ahead window, not
   * a full-queue scan. The deque is unbounded (a 100 GB scan at the default 512 MB
   * @c scan_task_batch_size routinely holds ~200 splits) and this walk runs under the
   * connector's mutex, serialising against both the producer and the consumer, so an
   * unbounded walk would be a scalability hazard.
   *
   * A split is hinted when it is a @c scan_operator_input, @p predicate accepts it, and it is
   * prefetchable for @p kind (@c is_io_prefetchable / @c is_memory_prefetchable, the latter's
   * @c nullopt treated as "yes"). Splits stay in the queue — nothing is extracted, despite what
   * a name like @c extract_if would suggest.
   *
   * @param upto_n    Look-ahead window size. Zero inspects nothing and returns 0.
   * @param kind      Which resource to warm.
   * @param predicate Caller filter, evaluated on each inspected split before the prefetchability
   *                  test.
   *
   * @warning @p predicate runs at **lock rank L2** (@c split_connector::_mutex, held for the
   *          whole walk) and must not acquire any lock of rank L1 or lower: no
   *          @c pipeline::sirius_pipeline status lock, no @c creator::task_creator lock, no
   *          scan-manager lock, and no re-entry into this connector. The engine's established
   *          order is L0 @c task_creator::_lookahead_mutex → L1 @c sirius_pipeline::_status_mutex
   *          → L2 this mutex, and a task-creator worker routinely holds L1 while blocking on L2
   *          inside @ref get_next_split. A natural-looking filter such as "only splits whose
   *          pipeline is not finished" would take L2 → L1 and deadlock ABBA against it. The
   *          predicate must also be non-blocking, for the same reason.
   *
   * @return How many splits this call **advanced the ladder for**. A split already hinted at this
   *         rung by an earlier invocation is inspected but not counted again, so repeated calls
   *         over a static queue converge to 0. For @c prefetch_kind::memory no hint is issued yet
   *         (see @ref prefetch_kind) and the value reports the splits that *would* be hinted —
   *         advisory, not a count of work done.
   */
  std::size_t prefetch_if(std::size_t upto_n,
                          prefetch_kind kind,
                          const std::function<bool(const op::operator_data&)>& predicate);

  /**
   * @brief How many queued splits are structural candidates for @p kind prefetching.
   *
   * O(1): the counts are maintained incrementally by @c push_split and the consumer path,
   * classifying each split once at push time. Deliberately a *structural* count —
   * "how many queued splits have IO to fetch" / "how many are resident batches" — not a
   * dynamic one. It does not drop when a split's data lands, because that would require an
   * O(n) re-scan under the mutex on every call. Use @ref prefetch_if to apply the dynamic test.
   */
  [[nodiscard]] std::size_t n_prefetchable(prefetch_kind kind) const;

  /**
   * @brief The index @ref get_next_split will hand out, given the queue's prefetch states.
   *
   * The selection policy, extracted as a pure total function so it is unit-testable without
   * building real splits or a GPU. Preference order over @p states, front to back:
   *   1. the first @c prefetch_progress::cached split (its IO has landed — run it now);
   *   2. otherwise the first split that is **not** @c prefetch_progress::loading;
   *   3. otherwise index 0.
   *
   * Step 3 is what preserves liveness and is **not** optional. Refusing a loading split has no
   * wakeup path: leaving @c loading notifies @c entry_state's atomic, not this connector's
   * condition variable, and the producer may already have pushed its last split — so "wait for
   * a non-loading split" is a *permanent* hang, not a delayed one. Preferring is safe; refusing
   * is not.
   *
   * Bounding is the **caller's** job, not this function's: it stays total over an arbitrary span
   * so it is exhaustively testable, and @ref get_next_split passes it a window of at most
   * @ref kSelectionWindow entries. Rule 3 therefore means "index 0 of the window", which is the
   * front of the queue — the FIFO answer.
   *
   * @param states One entry per inspected split, in queue order, starting at the queue front.
   *               Must be non-empty.
   * @return An index into @p states. Always valid.
   */
  [[nodiscard]] static std::size_t select_split_index(
    std::span<const io::cache::prefetch_progress> states) noexcept;

  /**
   * @brief How many queued splits @ref get_next_split inspects before choosing one.
   *
   * @ref get_next_split runs on the critical path **while the consumer holds
   * @c sirius_pipeline::_status_mutex** (@c creator::task_creator's task-creation lock), and
   * every task completing on that pipeline blocks behind that mutex on its way through
   * @c mark_task_completed -> @c update_pipeline_status. The dequeue is O(1) today; a full-queue
   * scan would make it O(n) in a deque that routinely holds ~200 splits. This constant is what
   * keeps the critical section O(1) in the queue length.
   *
   * Small on purpose. The window only has to be wide enough to step over a short run of splits
   * whose IO has not landed; if the first few are all still in flight, the whole queue almost
   * certainly is, and rule 3 is the right answer anyway. Compile-time rather than a YAML knob:
   * making it runtime-configurable would add a load inside a lock-held critical section to tune
   * something with no workload-dependent optimum.
   */
  static constexpr std::size_t kSelectionWindow = 8;

  /**
   * @brief Hard ceiling on the datasource folds one @ref get_next_split may perform.
   *
   * @ref kSelectionWindow bounds how many *splits* are inspected; this bounds the work **inside**
   * them. A @c op::scan::parquet_split_info carries one datasource per row-group slice, so at the
   * 512 MB default @c scan_task_batch_size over small parquet files a single split can hold
   * several hundred — a per-split bound alone leaves the walk unbounded. Each fold is a
   * @c std::function indirect call, a @c io::cache::prefetching_handle::get_context()
   * @c shared_ptr copy (two atomic RMWs) and a @c combine_prefetch_progress step, all under
   * @c _mutex (L2) while the consumer holds @c sirius_pipeline::_status_mutex (L1) and every task
   * completion on the pipeline serialises behind it.
   *
   * The budget is spent against a per-split fold cost captured once at push time, on the producer
   * thread, so the consumer never walks a split's datasources to find out what it would cost.
   * The window always inspects at least the front split, whatever its fold cost, so selection is
   * never skipped entirely and rule 3 of @ref select_split_index always has a candidate.
   */
  static constexpr std::size_t kSelectionFoldBudget = 32;

 private:
  friend class load_balancing_scan_batch_coalescer;
  /// Test-only seam (test/cpp/scan_manager/split_connector_test_access.hpp). Never defined in
  /// production; grants tests the producer side without widening it for production callers.
  /// @note An unqualified friend declaration names a member of the innermost enclosing namespace,
  ///       so the test seam must be declared in @c namespace @c sirius::scan_manager. A struct of
  ///       the same name at global scope is a *different* class and is not a friend.
  friend struct split_connector_test_access;

  /// \brief Enqueue a ready split. Producer side. Wakes a waiting consumer.
  ///        Private: reachable only from @ref load_balancing_scan_batch_coalescer, whose
  ///        single sequencer task is the sole production producer, and from the
  ///        @c split_connector_test_access test seam.
  void push_split(std::unique_ptr<op::operator_data> split);

  /// Fill @p out with the prefetch progress of the queued splits, front first, and return how
  /// many entries were written (always >= 1 when the queue is non-empty). Bounded on both axes:
  /// at most @ref kSelectionWindow entries, and past the always-inspected front split it stops
  /// before its accumulated @c _split_fold_costs would exceed @ref kSelectionFoldBudget. It also
  /// **stops at the first landed split**: rule 1 of @ref select_split_index takes the first
  /// @c cached entry, so nothing past it can change the answer. Caller must hold @c _mutex.
  std::size_t fill_progress_window(
    std::array<io::cache::prefetch_progress, kSelectionWindow>& out) const noexcept;

  /// Remove the split at @p index (< @ref kSelectionWindow) from @c _splits, @c _split_kinds and
  /// @c _split_fold_costs, and decrement the affected count. O(index) pointer moves:
  /// std::deque::erase relocates the shorter side, and the index is always within the leading
  /// window. Caller must hold @c _mutex.
  void drop_at(std::size_t index) noexcept;

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::unique_ptr<op::operator_data>> _splits;
  /// Parallel to @c _splits: the @ref prefetch_kind candidacy bitmask of each queued split, so
  /// the counts below can be maintained without re-inspecting (and re-@c dynamic_cast ing) a
  /// split on the removal path, which runs under @c _mutex on the critical path.
  std::deque<std::uint8_t> _split_kinds;
  /// Parallel to @c _splits: what one @c prefetch_state() fold of that split would cost, in
  /// datasources. Captured on the producer thread by the same walk that computes
  /// @c _split_kinds, so @ref fill_progress_window can spend @ref kSelectionFoldBudget against it
  /// without the consumer ever calling @c datasource_count() itself.
  std::deque<std::uint32_t> _split_fold_costs;
  std::size_t _n_io_prefetchable{0};
  std::size_t _n_memory_prefetchable{0};
  /// Whether any split pushed so far could land while queued
  /// (@c op::scan::scan_operator_input::can_land_while_queued). Until this is true,
  /// @ref get_next_split skips the selection walk entirely and behaves exactly as the pre-policy
  /// FIFO dequeue did. Latched, never cleared: a connector serves one backend, and re-checking
  /// per split would reintroduce the per-split datasource walk this exists to avoid.
  ///
  /// Written under @c _mutex by @ref push_split and read under it by @ref get_next_split. The
  /// datasource walk that *computes* the value runs off the lock — that walk is exactly what the
  /// fold budget keeps out of the critical section — but the store itself is not hoisted with it.
  bool _selection_armed{false};
  /// Whether the one-shot arming decision above has been taken yet. Separate from
  /// @c _selection_armed so a connector whose first io-candidate says "no" does not re-walk on
  /// every subsequent push. Producer-private: only @ref push_split touches it, which is why it can
  /// be tested outside @c _mutex to skip the walk.
  bool _arming_checked{false};
  bool _closed{false};
  std::exception_ptr _exception;
};

}  // namespace sirius::scan_manager
