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

/**
 * @file read_retirement.hpp
 * @brief Keeps scan storage alive until submitted GPU reads have completed.
 *
 * Scan filtering, projection and copying can enqueue GPU reads before a later allocation throws.
 * Host exception unwinding alone does not stop those reads; destroying a borrowed source lease or
 * scratch buffer could free memory still in use by the GPU.
 *
 * Callers retain the relevant owners in one state and use retire_reads to wait on the submitting
 * stream before releasing them. Successful completion permits normal cleanup or source restoration
 * for retry. If completion cannot be established, production deliberately leaks the state and
 * throws sirius::unjoined_gpu_work rather than freeing potentially referenced storage.
 * prepare_unjoined_failure creates that exception before submission so this failure path needs no
 * allocation.
 *
 * Test hooks inject allocation and completion failures; retained test storage is exposed only after
 * a real successful stream wait.
 */

#pragma once

#include <rmm/error.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <sirius/exception.hpp>

#include <atomic>
#include <concepts>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <typeinfo>
#include <utility>

namespace sirius::op::scan::detail {

enum class read_checkpoint {
  RESIDUAL,
  MASK_FOLD,
  COUNTS_COPY,
  PAYLOAD_GATHER,
  READER_EVENT,
  COPY,
  SUBSTITUTION,
  NORMALIZATION,
  BATCH,
  PAYLOAD
};

/**
 * @brief Owns one retained state of any type and hands it back only as that same type
 *
 * `std::any` cannot hold the move-only owners that retirement retains.
 */
class retained_state {
 public:
  retained_state() = default;
  ~retained_state() { reset(); }
  retained_state(retained_state const&)            = delete;
  retained_state& operator=(retained_state const&) = delete;

  /// Takes ownership of @p state, destroying any state retained before it
  template <typename State>
  void reset(std::unique_ptr<State> state) noexcept
  {
    reset();
    _value   = state.release();
    _type    = &typeid(State);
    _destroy = [](void* value) noexcept { delete static_cast<State*>(value); };
  }

  /// The retained state, or null when nothing or a state of another type is retained
  template <typename State>
  [[nodiscard]] State const* get() const noexcept
  {
    return _value != nullptr && *_type == typeid(State) ? static_cast<State const*>(_value)
                                                        : nullptr;
  }

  [[nodiscard]] bool has_value() const noexcept { return _value != nullptr; }

 private:
  void reset() noexcept
  {
    if (_value != nullptr) { _destroy(_value); }
    _value = nullptr;
  }

  void* _value                     = nullptr;
  std::type_info const* _type      = nullptr;
  void (*_destroy)(void*) noexcept = nullptr;
};

struct read_retirement_test_hooks;
inline thread_local read_retirement_test_hooks* read_test_hooks = nullptr;

/// How many `read_retirement_test_hooks` are alive on any thread. Production checks read only this
/// counter, so they never touch the thread-local `read_test_hooks`.
inline std::atomic<std::size_t> live_read_test_hooks{0};

/**
 * @brief Injects local scan failures; retained storage is exposed only after a real successful
 * join.
 *
 * Only the first state a failed join retains is kept.
 */
struct read_retirement_test_hooks {
  std::optional<read_checkpoint> fail_at;
  bool fail_join          = false;
  bool checkpoint_reached = false;
  retained_state retained;
  read_retirement_test_hooks* previous = read_test_hooks;

  explicit read_retirement_test_hooks(std::optional<read_checkpoint> point,
                                      bool completion_fails = false)
    : fail_at(point), fail_join(completion_fails)
  {
    read_test_hooks = this;
    live_read_test_hooks.fetch_add(1, std::memory_order_relaxed);
  }
  ~read_retirement_test_hooks()
  {
    live_read_test_hooks.fetch_sub(1, std::memory_order_relaxed);
    read_test_hooks = previous;
  }
  read_retirement_test_hooks(read_retirement_test_hooks const&)            = delete;
  read_retirement_test_hooks& operator=(read_retirement_test_hooks const&) = delete;
};

[[nodiscard]] inline bool read_test_hooks_may_be_installed() noexcept
{
  return live_read_test_hooks.load(std::memory_order_relaxed) != 0;
}

[[gnu::cold, gnu::noinline]] inline void inject_read_checkpoint(read_checkpoint point)
{
  if (read_test_hooks && read_test_hooks->fail_at == point) {
    read_test_hooks->checkpoint_reached = true;
    throw rmm::out_of_memory{"injected scan read failure"};
  }
}

inline void check_read_checkpoint(read_checkpoint point)
{
  if (read_test_hooks_may_be_installed()) [[unlikely]] { inject_read_checkpoint(point); }
}

inline std::exception_ptr prepare_unjoined_failure()
{
  return std::make_exception_ptr(sirius::unjoined_gpu_work{});
}

struct no_retirement_faults {
  static bool fails_join(bool) noexcept
  {
    return read_test_hooks_may_be_installed() && injected_join_failure();
  }
  template <typename State>
  static void retain(std::unique_ptr<State>& state) noexcept
  {
    if (read_test_hooks_may_be_installed()) [[unlikely]] { retain_injected(state); }
  }

 private:
  [[gnu::cold, gnu::noinline]] static bool injected_join_failure() noexcept
  {
    return read_test_hooks && read_test_hooks->fail_join &&
           (!read_test_hooks->fail_at || read_test_hooks->checkpoint_reached);
  }
  template <typename State>
  [[gnu::cold, gnu::noinline]] static void retain_injected(std::unique_ptr<State>& state) noexcept
  {
    if (read_test_hooks && !read_test_hooks->retained.has_value()) {
      read_test_hooks->retained.reset(std::move(state));
    }
  }
};

/**
 * @brief Joins submitted reads or retains their preallocated owner without allocating.
 *
 * The caller creates `unjoined` before submission and keeps every source and scratch owner in
 * `state`. A failed join replaces even an allocation failure with the prepared nonretryable
 * exception.
 */
template <typename State, typename Faults>
  requires requires(Faults& faults, std::unique_ptr<State>& state) {
    { faults.fails_join(true) } noexcept -> std::same_as<bool>;
    { faults.retain(state) } noexcept -> std::same_as<void>;
  }
void retire_reads(std::unique_ptr<State>& state,
                  ::cuda::stream_ref stream,
                  std::exception_ptr const& unjoined,
                  Faults& faults,
                  bool cleanup)
{
  auto const status = cudaStreamSynchronize(stream.get());
  if (status == cudaSuccess && !faults.fails_join(cleanup)) { return; }
  if (status == cudaSuccess) { faults.retain(state); }
  (void)state.release();
  std::rethrow_exception(unjoined);
}

template <typename State>
void retire_reads(std::unique_ptr<State>& state,
                  ::cuda::stream_ref stream,
                  std::exception_ptr const& unjoined)
{
  no_retirement_faults faults;
  retire_reads(state, stream, unjoined, faults, true);
}

}  // namespace sirius::op::scan::detail
