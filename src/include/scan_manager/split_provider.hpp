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

#include "io/gpu_ingestible.hpp"
#include "scan_manager/split_connector.hpp"

#include <atomic>
#include <exception>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

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
  /// Default ctor — used by legacy subclasses that override both virtuals
  /// (their @c _ingestible stays null). Removed when those subclasses are
  /// deleted.
  split_provider() = default;

  /// Concrete construction path — composes the given ingestible.
  explicit split_provider(std::shared_ptr<io::gpu_ingestible> ingestible)
    : _ingestible(std::move(ingestible))
  {
  }

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
  void run(Scheduler& scheduler, split_connector& connector);

  /**
   * @brief Snapshot check for remaining work. Thread-safe.
   *
   * Default impl delegates to the composed @c io::gpu_ingestible. Legacy
   * subclasses override and ignore @c _ingestible (which is null on the
   * default-ctor path they use).
   */
  [[nodiscard]] virtual bool has_more_splits() const
  {
    return _ingestible && _ingestible->has_more_splits();
  }

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
  virtual std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider()
  {
    if (!_ingestible) { return nullptr; }
    return _ingestible->next_split_provider();
  }

  /// Accessor for the composed ingestible (null for the legacy ctor path).
  [[nodiscard]] io::gpu_ingestible* get_ingestible() const noexcept { return _ingestible.get(); }
  [[nodiscard]] std::shared_ptr<io::gpu_ingestible> get_ingestible_shared() const noexcept
  {
    return _ingestible;
  }

 protected:
  /**
   * @brief Push a split into a connector.
   *
   * The only entry point for enqueueing splits: @ref split_connector::push_split
   * is private and reaches in only via the @c friend relationship between
   * @ref split_connector and this class. Because the helper is a protected
   * static, only @ref split_provider and its subclasses can call it, so
   * unrelated code cannot bypass the provider abstraction.
   *
   * Defined out-of-line because the unique_ptr's deleter needs the complete
   * @c op::operator_data type, which we keep forward-declared in this header.
   */
  static void push_to_connector(split_connector& connector,
                                std::unique_ptr<op::operator_data> split);

 private:
  /// Composed ingestible (null on the legacy default-ctor path).
  std::shared_ptr<io::gpu_ingestible> _ingestible;

  /// RAII coordination shared across the enqueued tasks. Destructor closes
  /// the connector with the captured exception (if any) once the last task
  /// drops its ref. Workers race to record the first error via CAS, so no
  /// mutex is needed and the destructor reads `error_ptr` after all writers
  /// have gone.
  struct worker_state {
    split_connector& connector;
    std::atomic<bool> error_set{false};
    std::exception_ptr error_ptr;

    static std::shared_ptr<worker_state> create(split_connector& connector)
    {
      return std::shared_ptr<worker_state>(new worker_state(connector));
    }

    void set_error(std::exception_ptr eptr)
    {
      bool expected = false;
      if (error_set.compare_exchange_strong(expected, true)) { error_ptr = std::move(eptr); }
    }

    ~worker_state()
    {
      // close()-side exceptions are swallowed because destructors must not
      // propagate; close() is best-effort wakeup.
      try {
        connector.close(error_ptr);
      } catch (...) {
      }
    }

   private:
    explicit worker_state(split_connector& c) : connector(c) {}
  };
};

template <typename Scheduler>
void split_provider::run(Scheduler& scheduler, split_connector& connector)
{
  auto state = worker_state::create(connector);
  while (has_more_splits()) {
    auto work = next_split_provider();
    // Concurrent observer could have drained the last batch between the
    // has_more_splits() check and our claim; skip the empty handoff.
    if (!work) { continue; }
    scheduler.enqueue([state, work = std::move(work)]() mutable {
      try {
        auto splits = work();
        for (auto& split : splits) {
          push_to_connector(state->connector, std::move(split));
        }
      } catch (...) {
        state->set_error(std::current_exception());
      }
    });
  }
}

}  // namespace sirius::scan_manager
