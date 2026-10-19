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
#include "memory/capped_async_memory_resource.hpp"
#include "memory/defragmenter_oom_policy.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

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
using sirius::memory::capped_async_memory_resource;

constexpr std::size_t MiB = std::size_t{1} << 20;

cucascade::memory::gpu_memory_space_config gpu_config(
  cucascade::memory::DeviceMemoryResourceFactoryFn factory, std::size_t bytes)
{
  cucascade::memory::gpu_memory_space_config config;
  config.device_id              = 0;
  config.memory_capacity        = bytes;
  config.per_stream_reservation = false;
  config.mr_factory_fn          = std::move(factory);
  return config;
}

std::uint64_t pool_attribute(cudaMemPool_t pool, cudaMemPoolAttr attr)
{
  std::uint64_t value = 0;
  REQUIRE(cudaMemPoolGetAttribute(pool, attr, &value) == cudaSuccess);
  return value;
}

}  // namespace

TEST_CASE("the capped async pool refuses to grow past its maximum", "[capped_async_pool]")
{
  // 63 MiB rounds up to 64.
  capped_async_memory_resource mr(0, 63 * MiB);
  CHECK(mr.max_bytes() == 64 * MiB);
  rmm::cuda_stream stream;
  void* first = mr.allocate(stream, 40 * MiB);
  CHECK_THROWS_AS(mr.allocate(stream, 40 * MiB), rmm::out_of_memory);
  void* second = mr.allocate(stream, 20 * MiB);
  CHECK(pool_attribute(mr.pool_handle(), cudaMemPoolAttrReservedMemHigh) <= 64 * MiB);
  mr.deallocate(stream, second, 20 * MiB);
  mr.deallocate(stream, first, 40 * MiB);
  stream.synchronize();
}

TEST_CASE("the capped async pool is primed to its maximum", "[capped_async_pool]")
{
  capped_async_memory_resource mr(0, 64 * MiB);
  // Priming leaves the maximum (or one page less) held but unused.
  CHECK(pool_attribute(mr.pool_handle(), cudaMemPoolAttrReservedMemCurrent) >= 62 * MiB);
  CHECK(pool_attribute(mr.pool_handle(), cudaMemPoolAttrUsedMemCurrent) == 0);
}

TEST_CASE("the capped async pool is found behind its space and by device", "[capped_async_pool]")
{
  cudaMemPool_t pool = nullptr;
  {
    memory_space gpu(gpu_config(sirius::memory::make_capped_async_pool_factory(), 64 * MiB));
    pool = sirius::memory::find_capped_pool(gpu);
    REQUIRE(pool != nullptr);
    CHECK(sirius::memory::capped_pool_on_device(0) == pool);
  }
  // Gone with its space.
  CHECK(sirius::memory::capped_pool_on_device(0) != pool);

  memory_space async(gpu_config(nullptr, 64 * MiB));
  CHECK(sirius::memory::find_capped_pool(async) == nullptr);
}

TEST_CASE("the capped async pool keeps the async pool's reservation accounting",
          "[capped_async_pool]")
{
  auto const measure = [](cucascade::memory::DeviceMemoryResourceFactoryFn factory) {
    memory_space gpu(gpu_config(std::move(factory), 64 * MiB));
    auto const stream = gpu.acquire_stream();
    auto* adaptor     = gpu.get_memory_resource_of<cucascade::memory::Tier::GPU>();
    REQUIRE(adaptor->attach_reservation_to_tracker(stream, gpu.make_reservation(16 * MiB)));
    std::pair<std::size_t, std::size_t> seen;
    {
      rmm::device_buffer buffer(4 * MiB, stream, gpu.get_default_allocator());
      seen = {gpu.get_available_memory(), gpu.get_total_reserved_memory()};
    }
    stream.sync();
    adaptor->reset_stream_reservation(stream);
    return seen;
  };
  CHECK(measure(nullptr) == measure(sirius::memory::make_capped_async_pool_factory()));
}

TEST_CASE("the defragmenter trims a capped pool cuCascade cannot see", "[capped_async_pool]")
{
  memory_space gpu(gpu_config(sirius::memory::make_capped_async_pool_factory(), 64 * MiB));
  auto const pool = sirius::memory::find_capped_pool(gpu);
  REQUIRE(pool != nullptr);
  REQUIRE(cudaSetDevice(0) == cudaSuccess);
  // The pool is primed and holds 64 MiB with nothing in use: at least 10x a 1 MiB request free.
  auto const oom = std::make_exception_ptr(cucascade::memory::cucascade_out_of_memory(
    "test", cucascade::memory::MemoryError::ALLOCATION_FAILED, MiB, 0, nullptr));
  bool retried   = false;
  int marker     = 0;
  sirius::memory::defragmenter_oom_policy policy;
  void* got = policy.handle_oom(
    MiB, gpu.acquire_stream(), oom, [&](std::size_t, ::cuda::stream_ref) -> void* {
      retried = true;
      return &marker;
    });
  CHECK(retried);
  CHECK(got == &marker);
  // Trimmed down to what is in use.
  CHECK(pool_attribute(pool, cudaMemPoolAttrReservedMemCurrent) == 0);
}

TEST_CASE("the capped async pool refuses a maximum below 2 MiB", "[capped_async_pool]")
{
  CHECK_THROWS_WITH(capped_async_memory_resource(0, MiB),
                    Catch::Matchers::ContainsSubstring("below 2 MiB"));
}
