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

#include "exec/completion_controller.hpp"
#include "exec/try.hpp"
#include "io/io_context.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "scan_manager/split_connector.hpp"

#include <cstddef>
#include <functional>
#include <memory>

namespace sirius::op {
class operator_data;
}  // namespace sirius::op

namespace sirius::scan_manager {

/**
 * @brief Driver of splits for a scan operator.
 *
 * Composes an @c io::gpu_ingestible and delegates the per-format work to it.
 * Exposes two thread-safe primitives:
 *   - @ref has_more_splits — snapshot check for remaining work.
 *   - @ref next_split_provider — atomically claim the next batch and return a
 *     callable that processes it.
 *
 * Splitting the claim from the work lets @ref run() enqueue one task per
 * batch on the driver thread without serialising metadata processing
 * behind a chain of "claim-then-run" tasks. The connector is closed in the
 * shared @c worker_state's destructor, which fires when the last enqueued
 * task releases its @c shared_ptr; any captured exception
 * (first-writer-wins via atomic CAS) is forwarded so consumers see the
 * failure through @ref split_connector::get_next_split.
 *
 * Historical note: this class was abstract pre-gpu_ingestible refactor.
 * Legacy subclasses (parquet_split_provider, duckdb_native_split_provider,
 * cached_split_provider) override @ref has_more_splits and
 * @ref next_split_provider; they are scheduled for deletion in step 10 of
 * the refactor. The default (concrete) construction path takes a
 * @c shared_ptr<io::gpu_ingestible> and uses the base's default
 * implementations to delegate to it.
 */
class split_provider {
 public:
  using value_type      = exec::try_t<std::unique_ptr<op::scan::scan_info>>;
  using push_callback_t = std::function<void(value_type&&)>;

  /// Concrete construction path — borrows the given ingestible non-owningly.
  /// The ingestible must outlive the provider; in practice the operator owns
  /// the ingestible (single shared_ptr) and the provider is created after
  /// install and torn down by scan_manager reset() before the operator goes
  /// away. Callers that need a shared_ptr can promote via
  /// `provider.get_ingestible().shared_from_this()` (enabled by
  /// @c gpu_ingestible inheriting @c std::enable_shared_from_this).
  explicit split_provider(op::scan::gpu_ingestible& ingestible, io::sirius_ioctx& io_ctx);

  virtual ~split_provider() = default;

  /**
   * @brief Drive the provider to completion, dispatching one task per claimed
   *        batch onto @p scheduler and pushing each task's splits into
   *        @p connector.
   *
   * The calling thread iterates @ref has_more_splits / @ref next_split_provider
   * and hands the resulting callable to @p scheduler.enqueue. Enqueue is
   * expected to be non-blocking, so this loop returns quickly. The connector
   * is closed (with any worker-captured exception) when the last enqueued
   * task drops its reference to the shared coordination state.
   *
   * @tparam Scheduler Anything with a @c enqueue(callable) method that runs
   *                   the callable asynchronously. @c static_thread_pool and
   *                   @c scoped_dispatcher both satisfy this shape.
   */
  template <typename Scheduler>
  void run(Scheduler& scheduler, const push_callback_t& on_split);

  /**
   * @brief Snapshot check for remaining work. Thread-safe.
   *
   * Default impl delegates to the composed @c io::gpu_ingestible. Legacy
   * subclasses override and ignore @c _ingestible (which is null on the
   * default-ctor path they use).
   */
  [[nodiscard]] virtual bool has_more_splits() const;

  /**
   * @brief Atomically claim the next batch and return a callable that
   *        produces its splits. Thread-safe.
   *
   * Default impl delegates to the composed @c io::gpu_ingestible. Returns
   * a null @c std::function when no batch is available — callers that
   * already checked @ref has_more_splits() will normally not hit this
   * path, but the null fallback keeps the contract simple under concurrent
   * observers.
   */
  virtual std::function<std::unique_ptr<op::scan::scan_info>()> next_split_provider();

  /// Accessor for the composed ingestible. Undefined behavior on the legacy
  /// default-ctor path (which never calls this). Callers that need a
  /// @c shared_ptr<gpu_ingestible> can promote via
  /// `provider.get_ingestible().shared_from_this()`.
  [[nodiscard]] op::scan::gpu_ingestible& get_ingestible() const noexcept { return *_ingestible; }

 private:
  /// Non-owning pointer to the composed ingestible (null on the legacy
  /// default-ctor path). The operator owns the lifetime; the provider is
  /// always destroyed first via @c sirius_scan_manager::reset.
  op::scan::gpu_ingestible* _ingestible{nullptr};

  std::shared_ptr<io::sirius_ioctx> _io_ctx;

  /// Pipeline this provider's scan belongs to, forwarded to the strategy.
  std::size_t _pipeline_id{0};

  exec::completion_token _completion_token;
};

}  // namespace sirius::scan_manager
