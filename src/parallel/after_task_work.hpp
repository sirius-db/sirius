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

#include "exec/invocable.hpp"

#include <cuda/stream>

#include <concepts>
#include <type_traits>
#include <utility>

namespace sirius::parallel {

/**
 * @brief True iff `after_task_work` stores a callable of type `Fn` without allocating.
 *
 * Conservative with respect to `absl::AnyInvocable`, which backs `exec::invocable` and stores
 * inline every nothrow move constructible callable of at most two pointers whose alignment divides
 * `alignof(std::max_align_t)`.
 *
 * @tparam Fn The callable type
 */
template <class Fn>
inline constexpr bool stored_inline_v =
  sizeof(Fn) <= 2 * sizeof(void*) && alignof(Fn) <= alignof(void*) &&
  std::is_nothrow_move_constructible_v<Fn>;

/**
 * @brief Move-only work that an operator hands to the executor running its task.
 *
 * An operator returns this from `sirius_physical_operator::observe_task_input`, and
 * `gpu_pipeline_task` keeps it in its local state. `gpu_pipeline_executor` takes it only after the
 * task's `execute()` returned, so the task's allocation tracker is detached and its reservation is
 * released. The executor then runs it on the same worker thread and on the task's stream: after the
 * success epilogue (task destroyed, consumers scheduled), or after the retry of an out-of-memory or
 * launch reschedule was scheduled. It destroys the work uninvoked on fatal exits and when the query
 * has already completed.
 *
 * Contract for the wrapped callable:
 * - Its running time is bounded, and it never blocks on the progress of another task or pipeline.
 *   It occupies a worker slot, so waiting for other tasks could stall the executor.
 * - It schedules no tasks.
 * - It touches only state it owns or state owned by `SiriusContext`. The query may complete while
 *   it runs; the query thread joins it through `wait_all` before any operator is destroyed.
 * - Destroying it uninvoked is its cancellation path and must be safe on any thread.
 * - It reports its own failures. An exception that escapes is logged by the executor and dropped.
 */
class after_task_work final {
 public:
  after_task_work() noexcept = default;

  /**
   * @brief Wraps @p fn; never allocates when `stored_inline_v<Fn>` holds.
   *
   * @throw std::bad_alloc if @p fn is not stored inline and allocation fails
   * @tparam Fn A nothrow move constructible callable, invoked once as an rvalue with the task's
   * stream
   * @param fn The work to run
   */
  template <class Fn>
    requires(!std::same_as<std::remove_cvref_t<Fn>, after_task_work>) &&
            std::is_nothrow_move_constructible_v<Fn> && std::invocable<Fn&&, ::cuda::stream_ref>
  explicit after_task_work(Fn fn) noexcept(stored_inline_v<Fn>) : _fn{std::move(fn)}
  {
  }

  after_task_work(after_task_work&&) noexcept            = default;
  after_task_work& operator=(after_task_work&&) noexcept = default;

  [[nodiscard]] explicit operator bool() const noexcept { return static_cast<bool>(_fn); }

  /**
   * @brief Runs the work once and leaves `*this` empty.
   *
   * @pre `*this` is non-empty
   * @param stream The stream of the task that produced the work
   */
  void operator()(::cuda::stream_ref stream) && { std::exchange(_fn, nullptr)(stream); }

 private:
  exec::invocable<void(::cuda::stream_ref) &&> _fn;
};

}  // namespace sirius::parallel
