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

// test
#include <catch.hpp>

// sirius
#include <memory/registered_host_memory_resource.hpp>

// cuda
#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

// standard library
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

using sirius::memory::registered_host_memory_resource;

namespace {

constexpr std::size_t pool_bytes = std::size_t{64} << 20;

/// The pool is page-locked and device-mapped, and the device reads what the host wrote.
void check_pool(void* pool)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pool) == cudaSuccess);
  CHECK(attributes.type == cudaMemoryTypeHost);
  unsigned int flags = 0;
  REQUIRE(cudaHostGetFlags(&flags, pool) == cudaSuccess);
  CHECK(flags == (cudaHostAllocPortable | cudaHostAllocMapped));
  void* device_view = nullptr;
  REQUIRE(cudaHostGetDevicePointer(&device_view, pool, 0) == cudaSuccess);
  CHECK(device_view != nullptr);
  CHECK(reinterpret_cast<std::uintptr_t>(pool) % registered_host_memory_resource::huge_page_bytes ==
        0);

  constexpr std::size_t n = std::size_t{8} << 20;
  auto* host              = static_cast<unsigned char*>(pool);
  for (std::size_t i = 0; i < n; ++i) {
    host[i] = static_cast<unsigned char>(i * 31 + 7);
  }
  void* device = nullptr;
  REQUIRE(cudaMalloc(&device, n) == cudaSuccess);
  REQUIRE(cudaMemcpy(device, device_view, n, cudaMemcpyDefault) == cudaSuccess);
  std::vector<unsigned char> back(n);
  REQUIRE(cudaMemcpy(back.data(), device, n, cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaFree(device) == cudaSuccess);
  CHECK(std::memcmp(back.data(), host, n) == 0);
}

}  // namespace

TEST_CASE("registered_host_memory_resource hands out pinned, device-mapped pools",
          "[registered_host_memory_resource]")
{
  auto const [numa_node, huge_pages, threads] =
    GENERATE(std::tuple{-1, true, 8U}, std::tuple{0, true, 4U}, std::tuple{-1, false, 2U},
             std::tuple{-1, true, 0U});
  registered_host_memory_resource mr{numa_node, huge_pages, threads};

  std::vector<void*> pools;
  for (int i = 0; i < 4; ++i) {
    pools.push_back(mr.allocate_sync(pool_bytes));
    REQUIRE(pools.back() != nullptr);
  }
  for (void* pool : pools) {
    check_pool(pool);
  }
  for (void* pool : pools) {
    mr.deallocate_sync(pool, pool_bytes);
  }
}

TEST_CASE("registered_host_memory_resource rounds sizes up and frees them with the caller's size",
          "[registered_host_memory_resource]")
{
  registered_host_memory_resource mr{-1, true, 4};
  CHECK(mr.allocate_sync(0) == nullptr);

  constexpr std::size_t odd = 3 * 1000 * 1000 + 17;
  void* pool                = mr.allocate_sync(odd);
  REQUIRE(pool != nullptr);
  static_cast<unsigned char*>(pool)[odd - 1] = 1;
  mr.deallocate_sync(pool, odd);
  mr.deallocate_sync(nullptr, odd);
}

TEST_CASE("registered_host_memory_resource fits the memory library's resource factory",
          "[registered_host_memory_resource]")
{
  STATIC_REQUIRE(
    ::cuda::mr::resource_with<registered_host_memory_resource, ::cuda::mr::device_accessible>);
  STATIC_REQUIRE(
    ::cuda::mr::resource_with<registered_host_memory_resource, ::cuda::mr::host_accessible>);

  ::cuda::mr::any_resource<::cuda::mr::device_accessible> erased{
    registered_host_memory_resource{-1, true, 4}};
  void* pool = erased.allocate_sync(pool_bytes, alignof(std::max_align_t));
  REQUIRE(pool != nullptr);
  check_pool(pool);
  erased.deallocate_sync(pool, pool_bytes, alignof(std::max_align_t));

  CHECK(registered_host_memory_resource{-1, true, 4} == registered_host_memory_resource{-1, true, 4});
  CHECK_FALSE(registered_host_memory_resource{-1, true, 4} ==
              registered_host_memory_resource{0, true, 4});
}
