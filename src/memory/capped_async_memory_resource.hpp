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

#include <rmm/aligned.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <memory>

namespace cucascade::memory {
class memory_space;
}

namespace sirius {
namespace memory {

/**
 * @brief A stream-ordered CUDA memory pool that can never grow past a hard maximum.
 *
 * The pool is created with `cudaMemPoolProps::maxSize`, so the driver refuses an allocation that
 * would take the pool past it, unlike `rmm::mr::cuda_async_memory_resource`, whose size argument
 * is only how much it primes. It is primed to the maximum and keeps what it holds across
 * synchronizations (release threshold `UINT64_MAX`), like the async pool; an OOM handler may
 * still trim it (`cudaMemPoolTrimTo`) and it grows back up to the maximum.
 *
 * Copies share the pool; the last one destroys it.
 */
class capped_async_memory_resource {
 public:
  /// Creates the pool on @p device with a maximum of @p max_bytes rounded up to 2 MiB, and primes
  /// it to that size. Priming is best effort: a pool the device cannot fill at once still works.
  /// @throws rmm::logic_error when @p max_bytes is below 2 MiB.
  /// @throws rmm::cuda_error when the pool cannot be created.
  capped_async_memory_resource(int device, std::size_t max_bytes);

  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;
  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT);
  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept;

  bool operator==(capped_async_memory_resource const& other) const noexcept
  {
    return _state == other._state;
  }

  friend void get_property(capped_async_memory_resource const&,
                           ::cuda::mr::device_accessible) noexcept
  {
  }

  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept;
  [[nodiscard]] int device() const noexcept;
  /// The pool's maximum, the requested size rounded up to 2 MiB.
  [[nodiscard]] std::size_t max_bytes() const noexcept;

 private:
  struct state;
  std::shared_ptr<state> _state;
};

/// A factory for a capped_async_memory_resource whose maximum is the space's capacity.
[[nodiscard]] cucascade::memory::DeviceMemoryResourceFactoryFn make_capped_async_pool_factory();

/// The capped pool behind @p gpu's allocator, or nullptr unless make_capped_async_pool_factory
/// built it.
[[nodiscard]] cudaMemPool_t find_capped_pool(cucascade::memory::memory_space const& gpu);

/// The most recently built capped pool still alive on @p device, or nullptr. cuCascade recovers
/// the pool handle of an OOM only from RMM's own async resources, so the OOM handler looks a
/// capped pool up here instead.
[[nodiscard]] cudaMemPool_t capped_pool_on_device(int device) noexcept;

}  // namespace memory
}  // namespace sirius
