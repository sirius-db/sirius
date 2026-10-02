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

#include "catch.hpp"
#include "memory/defragmenter_oom_policy.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "memory/slab_memory_resource.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <cstdint>
#include <exception>
#include <optional>
#include <utility>

namespace {

using cucascade::memory::memory_space;

// Not a multiple of 2 MiB, so the slab is 1 MiB smaller than the space's capacity.
constexpr std::size_t capacity = std::size_t{65} << 20;

cucascade::memory::gpu_memory_space_config gpu_config(
  cucascade::memory::DeviceMemoryResourceFactoryFn factory)
{
  cucascade::memory::gpu_memory_space_config config;
  config.device_id              = 0;
  config.memory_capacity        = capacity;
  config.per_stream_reservation = false;
  config.mr_factory_fn          = std::move(factory);
  return config;
}

std::uintptr_t address(void const* p) { return reinterpret_cast<std::uintptr_t>(p); }

}  // namespace

TEST_CASE("the slab backs cuDF allocations and outlives its pool", "[slab_pool]")
{
  std::optional<sirius::memory::slab_region> slab;
  {
    sirius::memory::sirius_memory_reservation_manager manager(
      {gpu_config(sirius::memory::make_slab_pool_factory())});
    slab = sirius::memory::find_slab(*manager.get_memory_space(cucascade::memory::Tier::GPU, 0));
    REQUIRE(slab.has_value());
    CHECK(slab->device == 0);
    CHECK(slab->len == rmm::align_down(capacity, std::size_t{2} << 20));

    rmm::cuda_stream stream;
    auto const column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                  1000,
                                                  cudf::mask_state::ALL_VALID,
                                                  stream,
                                                  cudf::get_current_device_resource_ref());
    auto const view   = column->view();
    CHECK(slab->covers(address(view.head()), 1000 * sizeof(std::int64_t)));
    CHECK(slab->covers(address(view.null_mask()), cudf::bitmask_allocation_size_bytes(1000)));
  }

  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(slab->base)) ==
          cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeDevice);
}

TEST_CASE("find_slab finds nothing behind the default async pool", "[slab_pool]")
{
  memory_space gpu(gpu_config(nullptr));
  CHECK_FALSE(sirius::memory::find_slab(gpu).has_value());
}

TEST_CASE("the slab pool keeps the async pool's reservation accounting", "[slab_pool]")
{
  auto const measure = [](cucascade::memory::DeviceMemoryResourceFactoryFn factory) {
    memory_space gpu(gpu_config(std::move(factory)));
    auto const stream = gpu.acquire_stream();
    auto* adaptor     = gpu.get_memory_resource_of<cucascade::memory::Tier::GPU>();
    REQUIRE(adaptor->attach_reservation_to_tracker(stream, gpu.make_reservation(16 << 20)));
    std::pair<std::size_t, std::size_t> seen;
    {
      rmm::device_buffer buffer(4 << 20, stream, gpu.get_default_allocator());
      seen = {gpu.get_available_memory(), gpu.get_total_reserved_memory()};
    }
    stream.sync();
    adaptor->reset_stream_reservation(stream);
    return seen;
  };
  CHECK(measure(nullptr) == measure(sirius::memory::make_slab_pool_factory()));
}

TEST_CASE("the defragmenter rethrows a slab pool OOM without retrying", "[slab_pool]")
{
  memory_space gpu(gpu_config(sirius::memory::make_slab_pool_factory()));
  auto const stream = gpu.acquire_stream();

  // Within the space's capacity but larger than the slab, so the pool itself fails.
  std::exception_ptr oom;
  try {
    rmm::device_buffer buffer(capacity, stream, gpu.get_default_allocator());
  } catch (cucascade::memory::cucascade_out_of_memory const& e) {
    // Not CHECK(a == b): Catch would print MemoryError through std::error_code, and cuCascade
    // does not export the make_error_code that needs.
    bool const allocation_failed =
      e.error_kind == cucascade::memory::MemoryError::ALLOCATION_FAILED;
    CHECK(allocation_failed);
    CHECK(e.pool_handle == nullptr);
    oom = std::current_exception();
  }
  REQUIRE(oom);

  bool retried = false;
  sirius::memory::defragmenter_oom_policy policy;
  CHECK_THROWS_AS(policy.handle_oom(capacity,
                                    stream,
                                    oom,
                                    [&](std::size_t, ::cuda::stream_ref) -> void* {
                                      retried = true;
                                      return nullptr;
                                    }),
                  cucascade::memory::cucascade_out_of_memory);
  CHECK_FALSE(retried);
}
