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

#include "memory/capped_async_memory_resource.hpp"

#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/cuda_async_view_memory_resource.hpp>

#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>

namespace sirius {
namespace memory {

namespace {

constexpr std::size_t pool_granularity = std::size_t{2} << 20;

/// Live capped pools by device, newest last; see capped_pool_on_device.
struct pool_registry {
  std::mutex mutex;
  std::multimap<int, cudaMemPool_t> pools;
};

pool_registry& registry()
{
  // Never destroyed: a pool may die during static destruction.
  static auto* r = new pool_registry;
  return *r;
}

}  // namespace

struct capped_async_memory_resource::state {
  int device;
  std::size_t max_bytes;
  cudaMemPool_t pool{};

  state(int dev, std::size_t requested)
    : device{dev}, max_bytes{rmm::align_up(requested, pool_granularity)}
  {
    RMM_EXPECTS(requested >= pool_granularity, "capped async pool maximum is below 2 MiB");
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    cudaMemPoolProps props{};
    props.allocType     = cudaMemAllocationTypePinned;
    props.handleTypes   = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id   = device;
    props.maxSize       = max_bytes;
    RMM_CUDA_TRY(cudaMemPoolCreate(&pool, &props));
    // Keep what the pool holds across synchronizations, as the async pool does; trimming on OOM
    // is the only way it gives memory back.
    auto threshold = std::numeric_limits<std::uint64_t>::max();
    RMM_CUDA_TRY(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    prime();
    std::lock_guard lock{registry().mutex};
    registry().pools.emplace(device, pool);
  }

  ~state()
  {
    {
      std::lock_guard lock{registry().mutex};
      auto [first, last] = registry().pools.equal_range(device);
      for (auto it = first; it != last; ++it) {
        if (it->second == pool) {
          registry().pools.erase(it);
          break;
        }
      }
    }
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    // Allocations still outstanding keep their memory until freed; the driver releases the pool
    // then.
    static_cast<void>(cudaMemPoolDestroy(pool));
  }

  state(state const&)            = delete;
  state& operator=(state const&) = delete;

  /// Takes the whole maximum once and frees it, so the pool holds it from the start. The driver
  /// may keep a page of the maximum for itself, so one page less is tried next.
  void prime()
  {
    void* p = nullptr;
    for (auto const bytes : {max_bytes, max_bytes - pool_granularity}) {
      if (bytes == 0) { break; }
      if (cudaMallocFromPoolAsync(&p, bytes, pool, cudaStreamPerThread) == cudaSuccess) { break; }
      static_cast<void>(cudaGetLastError());
      p = nullptr;
    }
    if (p == nullptr) {
      SIRIUS_LOG_WARN(
        "capped async pool: could not prime device {} with {} bytes; it grows on demand instead",
        device,
        max_bytes);
      return;
    }
    RMM_CUDA_TRY(cudaFreeAsync(p, cudaStreamPerThread));
    RMM_CUDA_TRY(cudaStreamSynchronize(cudaStreamPerThread));
  }

  [[nodiscard]] rmm::mr::cuda_async_view_memory_resource view() const
  {
    return rmm::mr::cuda_async_view_memory_resource{pool};
  }
};

capped_async_memory_resource::capped_async_memory_resource(int device, std::size_t max_bytes)
  : _state{std::make_shared<state>(device, max_bytes)}
{
  SIRIUS_LOG_INFO("capped async pool on device {}: maximum {} bytes", device, _state->max_bytes);
}

void* capped_async_memory_resource::allocate(::cuda::stream_ref stream,
                                             std::size_t bytes,
                                             std::size_t alignment)
{
  return _state->view().allocate(stream, bytes, alignment);
}

void capped_async_memory_resource::deallocate(::cuda::stream_ref stream,
                                              void* ptr,
                                              std::size_t bytes,
                                              std::size_t alignment) noexcept
{
  _state->view().deallocate(stream, ptr, bytes, alignment);
}

void* capped_async_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment)
{
  return _state->view().allocate_sync(bytes, alignment);
}

void capped_async_memory_resource::deallocate_sync(void* ptr,
                                                   std::size_t bytes,
                                                   std::size_t alignment) noexcept
{
  _state->view().deallocate_sync(ptr, bytes, alignment);
}

cudaMemPool_t capped_async_memory_resource::pool_handle() const noexcept { return _state->pool; }

std::size_t capped_async_memory_resource::max_bytes() const noexcept { return _state->max_bytes; }

cucascade::memory::DeviceMemoryResourceFactoryFn make_capped_async_pool_factory()
{
  return [](int device,
            std::size_t capacity) -> ::cuda::mr::any_resource<::cuda::mr::device_accessible> {
    return capped_async_memory_resource(device, capacity);
  };
}

cudaMemPool_t capped_pool_on_device(int device) noexcept
{
  std::lock_guard lock{registry().mutex};
  auto [first, last] = registry().pools.equal_range(device);
  if (first == last) { return nullptr; }
  return std::prev(last)->second;
}

}  // namespace memory
}  // namespace sirius
