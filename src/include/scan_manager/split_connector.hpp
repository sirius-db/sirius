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
  /// Not strictly FIFO: among the first @ref kSelectionWindow queued splits it prefers one whose
  /// prefetch has already landed (see @ref select_split_index). Splits are independent units of
  /// work — each becomes its own concurrently-executed task on its own device
  /// (task_creator.cpp:518-524) — so the engine has no cross-split ordering requirement. Resident
  /// splits carry no prefetch handle and all report the same state, so the cached-serving path
  /// stays FIFO in practice.
  ///
  /// Complexity: **O(kSelectionWindow)** in the queue length — independent of how many splits are
  /// queued. At most @ref kSelectionWindow prefetch-state reads (each folding over that split's
  /// datasources), stopping early at the first landed split; plus a deque erase within the window,
  /// which moves at most kSelectionWindow-1 pointers. The common path is one state read and a
  /// @c pop_front, i.e. the same O(1) dequeue as before this policy existed.
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
   *                  test. Must be non-blocking and must not re-enter the connector — it runs
   *                  with the connector's mutex held.
   * @return How many splits were actually hinted.
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
   * @c sirius_pipeline::_status_mutex** (task_creator.cpp:376-524), and every task completing on
   * that pipeline blocks behind that mutex (gpu_pipeline_task.cpp:325 -> mark_task_completed ->
   * update_pipeline_status). The dequeue is O(1) today; a full-queue scan would make it O(n) in a
   * deque that routinely holds ~200 splits, times the datasources each split folds over. This
   * constant is what keeps the critical section O(1) in the queue length.
   *
   * Small on purpose. The window only has to be wide enough to step over a short run of splits
   * whose IO has not landed; if the first few are all still in flight, the whole queue almost
   * certainly is, and rule 3 is the right answer anyway. Compile-time rather than a YAML knob:
   * making it runtime-configurable would add a load inside a lock-held critical section to tune
   * something with no workload-dependent optimum.
   */
  static constexpr std::size_t kSelectionWindow = 8;

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
  /// many entries were written (always >= 1 when the queue is non-empty). Writes at most
  /// @ref kSelectionWindow entries and **stops at the first landed split**: rule 1 of
  /// @ref select_split_index takes the first @c cached entry, so nothing past it can change the
  /// answer. Caller must hold @c _mutex.
  std::size_t fill_progress_window(
    std::array<io::cache::prefetch_progress, kSelectionWindow>& out) const noexcept;

  /// Remove the split at @p index (< @ref kSelectionWindow) from @c _splits and @c _split_kinds
  /// and decrement the affected count. O(index) pointer moves: std::deque::erase relocates the
  /// shorter side, and the index is always within the leading window. Caller must hold @c _mutex.
  void drop_at(std::size_t index) noexcept;

  mutable std::mutex _mutex;
  std::condition_variable _cv;
  std::deque<std::unique_ptr<op::operator_data>> _splits;
  /// Parallel to @c _splits: the @ref prefetch_kind candidacy bitmask of each queued split, so
  /// the counts below can be maintained without re-inspecting (and re-@c dynamic_cast ing) a
  /// split on the removal path, which runs under @c _mutex on the critical path.
  std::deque<std::uint8_t> _split_kinds;
  std::size_t _n_io_prefetchable{0};
  std::size_t _n_memory_prefetchable{0};
  bool _closed{false};
  std::exception_ptr _exception;
};

}  // namespace sirius::scan_manager
