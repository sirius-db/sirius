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

#include "memory/slab_memory_resource.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

namespace sirius {
namespace memory {

namespace {

constexpr std::size_t slab_granularity = std::size_t{2} << 20;

/// Pool upstream that hands out its slab once; frees are no-ops because the slab is released
/// only when the last copy of this resource or of a slab_region taken from it dies.
class slab_upstream {
 public:
  slab_upstream(int device, std::size_t len) : _s{std::make_shared<state>(device, len)} {}

  void* allocate(::cuda::stream_ref, std::size_t bytes, std::size_t alignment)
  {
    return allocate_sync(bytes, alignment);
  }
  void deallocate(::cuda::stream_ref, void*, std::size_t, std::size_t) noexcept {}

  void* allocate_sync(std::size_t bytes, std::size_t)
  {
    RMM_EXPECTS(_s->base == nullptr && bytes <= _s->len, "slab exhausted", rmm::out_of_memory);
    RMM_CUDA_TRY_ALLOC(cudaMalloc(&_s->base, _s->len), _s->len);
    return _s->base;
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}

  bool operator==(slab_upstream const& other) const noexcept { return _s == other._s; }

  friend void get_property(slab_upstream const&, ::cuda::mr::device_accessible) noexcept {}

  [[nodiscard]] slab_region region() const
  {
    return {_s->device, reinterpret_cast<std::uintptr_t>(_s->base), _s->len, _s};
  }

 private:
  struct state {
    int device;
    std::size_t len;
    void* base{};

    ~state()
    {
      if (base == nullptr) { return; }
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
      RMM_ASSERT_CUDA_SUCCESS(cudaFree(base));
    }
  };
  std::shared_ptr<state> _s;
};

}  // namespace

cucascade::memory::DeviceMemoryResourceFactoryFn make_slab_pool_factory()
{
  return [](int device,
            std::size_t capacity) -> ::cuda::mr::any_resource<::cuda::mr::device_accessible> {
    // memory_space calls the factory without setting the device.
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    auto const len = rmm::align_down(capacity, slab_granularity);
    RMM_EXPECTS(len > 0, "slab pool capacity is below 2 MiB");
    return rmm::mr::pool_memory_resource(slab_upstream{device, len}, len, len);
  };
}

std::optional<slab_region> find_slab(cucascade::memory::memory_space const& gpu)
{
  // A resource_ref taken from an any_resource points at the stored resource, so resource_cast
  // can see through both type-erased layers.
  auto upstream =
    gpu.get_memory_resource_of<cucascade::memory::Tier::GPU>()->get_upstream_resource();
  auto* pool = ::cuda::mr::resource_cast<rmm::mr::pool_memory_resource>(&upstream);
  if (pool == nullptr) { return std::nullopt; }
  auto pool_upstream = pool->get_upstream_resource();
  auto* slab         = ::cuda::mr::resource_cast<slab_upstream>(&pool_upstream);
  if (slab == nullptr) { return std::nullopt; }
  return slab->region();
}

}  // namespace memory
}  // namespace sirius
