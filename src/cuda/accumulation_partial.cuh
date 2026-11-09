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
 * @file accumulation_partial.cuh
 * @brief The per-GPU storage shared by the build-filter accumulators: `accumulated_bloom_builder`,
 * `exact_key_stager`, and `bounds_accumulator`.
 *
 * Each accumulator keeps one `accumulation_partial` per replica GPU: arrays leased on the untracked
 * creator thread, and an exclusive stream that orders every later use and every free of them.
 * Contributions wait for the partial's `init` event and fold their own completion back into the
 * partial's stream.
 */

#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cucascade/cuda/event.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <op/dynamic_filter/detail/accumulated_bloom_builder.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_reservation.hpp>
#include <op/dynamic_filter/dynamic_filter_replica_space.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace sirius::op::detail {

using cucascade::cuda::cuda_event;

/**
 * @brief Throws `accumulation_cuda_error` for a failed CUDA call after clearing the error.
 *
 * @param status The result of the CUDA call
 * @param call The name of the CUDA call, used in the error message
 * @param kernel_launch Whether @p status comes from a kernel launch
 */
inline void check_cuda(cudaError_t status, char const* call, bool kernel_launch = false)
{
  if (status == cudaSuccess) { return; }
  (void)cudaGetLastError();
  throw accumulation_cuda_error{status, kernel_launch, call};
}

/**
 * @brief Creates a timing-free event on @p device, restoring the current device afterwards.
 */
[[nodiscard]] inline cuda_event make_event_on(int device)
{
  rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
  return cuda_event{};
}

/**
 * @brief One GPU's private arrays and the exclusive stream that orders their use and release.
 */
struct accumulation_partial {
  cucascade::memory::memory_space* space;
  cucascade::memory::memory_space const* host_space;
  int device_id;
  std::shared_ptr<rmm::cuda_stream> stream;
  std::vector<std::unique_ptr<rmm::device_buffer>> arrays;  // Null for an array of no bytes
  std::optional<cuda_event> init;  // Recorded after allocation; every contribution waits on it.
  std::atomic<bool> contributed{false};
  std::atomic<bool> const* leak_storage;

  accumulation_partial(cucascade::memory::memory_space& owner,
                       cucascade::memory::memory_space const& host,
                       std::atomic<bool> const& leak)
    : space{&owner},
      host_space{&host},
      device_id{owner.get_device_id()},
      stream{std::make_shared<rmm::cuda_stream>(rmm::cuda_stream::flags::non_blocking)},
      leak_storage{&leak}
  {
  }

  ~accumulation_partial()
  {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
    if (leak_storage->load()) {
      // A failed host join left work that may still touch these arrays: never free them.
      for (auto& array : arrays) {
        (void)array.release();
      }
    }
    init.reset();
    arrays.clear();
    stream.reset();
  }

  accumulation_partial(accumulation_partial const&)            = delete;
  accumulation_partial& operator=(accumulation_partial const&) = delete;
  accumulation_partial(accumulation_partial&&)                 = delete;
  accumulation_partial& operator=(accumulation_partial&&)      = delete;

  [[nodiscard]] ::cuda::stream_ref stream_ref() const noexcept
  {
    return ::cuda::stream_ref{stream->value()};
  }

  /**
   * @brief Makes @p contribution wait for this partial's allocation, runs @p enqueue on it, and
   * folds its completion into this partial's stream.
   *
   * The caller wraps the call in `join_on_failure`.
   */
  template <class Enqueue>
  void enqueue_contribution(::cuda::stream_ref contribution, Enqueue&& enqueue)
  {
    check_cuda(cudaStreamWaitEvent(contribution.get(), init->get(), 0),
               "cudaStreamWaitEvent(init)");
    std::forward<Enqueue>(enqueue)();
    cuda_event done;
    check_cuda(cudaEventRecord(done.get(), contribution.get()), "cudaEventRecord(contribution)");
    // Every later read or free of the arrays follows the contribution.
    check_cuda(cudaStreamWaitEvent(stream->value(), done.get(), 0),
               "cudaStreamWaitEvent(fold contribution)");
    contributed.store(true, std::memory_order_relaxed);
  }

  /**
   * @brief Whether releasing this partial's arrays now would credit no allocation tracker.
   */
  [[nodiscard]] bool releasable_here() const
  {
    auto const* allocator = space->template get_memory_resource_of<cucascade::memory::Tier::GPU>();
    return allocator == nullptr || !allocator->is_stream_tracked(stream_ref());
  }
};

/**
 * @brief Creates one partial per distinct GPU of @p targets, each holding one array of
 * `array_bytes[i]` bytes per entry, under a single exact lease per GPU.
 *
 * Arrays of zero bytes stay null and take no lease. With @p zero_fill, every array is zeroed on its
 * partial's stream. Nothing waits on the host. The calling thread must hold no allocation tracker
 * for those GPUs.
 *
 * @return The partials, or nullopt iff a lease was refused, after releasing everything built so far
 */
[[nodiscard]] inline std::optional<std::vector<std::unique_ptr<accumulation_partial>>>
lease_partials(std::span<dynamic_filter_replica_space const> targets,
               std::span<std::size_t const> array_bytes,
               bool zero_fill,
               std::atomic<bool> const& leak_storage)
{
  std::size_t lease_bytes = 0;
  for (auto const bytes : array_bytes) {
    lease_bytes += tracked_replica_allocation_bytes(bytes);
  }
  std::vector<std::unique_ptr<accumulation_partial>> partials;
  for (auto const& target : targets) {
    auto& space = target.get_gpu_space();
    if (std::ranges::any_of(
          partials, [&](auto const& entry) { return entry->device_id == space.get_device_id(); })) {
      continue;
    }
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{space.get_device_id()}};
    auto owned =
      std::make_unique<accumulation_partial>(space, target.get_host_staging_space(), leak_storage);
    owned->arrays.resize(array_bytes.size());
    if (lease_bytes != 0) {
      // Detach the lease before `owned` can be destroyed, so its frees cannot credit that tracker.
      auto lease = scoped_replica_reservation::try_acquire(space, lease_bytes, owned->stream_ref());
      if (!lease) { return std::nullopt; }
      for (std::size_t index = 0; index < array_bytes.size(); ++index) {
        if (array_bytes[index] == 0) { continue; }
        owned->arrays[index] = std::make_unique<rmm::device_buffer>(
          array_bytes[index], owned->stream_ref(), lease->allocator());
        if (zero_fill) {
          check_cuda(cudaMemsetAsync(
                       owned->arrays[index]->data(), 0, array_bytes[index], owned->stream->value()),
                     "cudaMemsetAsync(accumulation array)");
        }
      }
    }
    owned->init.emplace();
    check_cuda(cudaEventRecord(owned->init->get(), owned->stream->value()),
               "cudaEventRecord(accumulation init)");
    partials.push_back(std::move(owned));
  }
  return partials;
}

/**
 * @brief The partial of @p partials on @p device_id, or null.
 */
[[nodiscard]] inline accumulation_partial* find_partial(
  std::span<std::unique_ptr<accumulation_partial> const> partials, int device_id) noexcept
{
  auto const found =
    std::ranges::find(partials, device_id, [](auto const& entry) { return entry->device_id; });
  return found == partials.end() ? nullptr : found->get();
}

}  // namespace sirius::op::detail
