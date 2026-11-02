/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "memory/numa_small_pinned_mr.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>
using namespace cucascade::memory;
namespace {
// Helper to create a fixed_size_host_memory_resource backed by pinned host memory.
// Uses a small block size (64 KB) and pool to keep test memory footprint low.
struct test_fixture {
  static constexpr std::size_t block_size    = 64 * 1024;  // 64 KB
  static constexpr std::size_t pool_size     = 16;
  static constexpr std::size_t initial_pools = 1;
  static constexpr std::size_t mem_limit     = 16 * 1024 * 1024;  // 16 MB
  static constexpr std::size_t capacity      = 16 * 1024 * 1024;  // 16 MB

  rmm::mr::pinned_host_memory_resource pinned_mr;
  fixed_size_host_memory_resource upstream{
    0, pinned_mr, mem_limit, capacity, block_size, pool_size, initial_pools};
  sirius::memory::numa_small_pinned_mr slab_mr{make_pools(), nullptr, 0};

  auto make_pools() -> std::unordered_map<int, std::unique_ptr<small_pinned_host_memory_resource>>
  {
    std::unordered_map<int, std::unique_ptr<small_pinned_host_memory_resource>> pools;
    pools.emplace(0, std::make_unique<small_pinned_host_memory_resource>(upstream));
    return pools;
  }
};

// Keeps queued work pending until the test releases it. Cleanup also releases the callback when
// an assertion aborts the test, so stream destruction cannot leave it blocked.
class pending_stream_work {
 public:
  explicit pending_stream_work(rmm::cuda_stream const& stream) : stream_(stream)
  {
    CUCASCADE_CUDA_TRY(::cudaLaunchHostFunc(
      stream_.value(),
      [](void* context) {
        auto& gate = *static_cast<pending_stream_work*>(context);
        gate.released_.wait(false);
      },
      this));
  }

  pending_stream_work(pending_stream_work const&)            = delete;
  pending_stream_work& operator=(pending_stream_work const&) = delete;
  pending_stream_work(pending_stream_work&&)                 = delete;
  pending_stream_work& operator=(pending_stream_work&&)      = delete;

  ~pending_stream_work()
  {
    release();
    stream_.synchronize_no_throw();
  }

  void release() noexcept
  {
    released_.store(true);
    released_.notify_all();
  }

 private:
  rmm::cuda_stream const& stream_;
  std::atomic<bool> released_{false};
};

}  // namespace

TEST_CASE("NUMA facade synchronous allocation waits for pending DMA before host reuse",
          "[numa_small_pinned][stream_ordering]")
{
  // Slab reuse already existed before the large-cache change; keep this regression independent.
  constexpr std::size_t bytes = 2048;
  test_fixture f;
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  rmm::device_buffer device{bytes, ::cuda::stream_ref{producer.value()}};
  auto* original = f.slab_mr.allocate_sync(bytes);
  std::memset(original, 0x43, bytes);
  pending_stream_work gate{producer};
  CUCASCADE_CUDA_TRY(
    ::cudaMemcpyAsync(device.data(), original, bytes, cudaMemcpyHostToDevice, producer.value()));
  f.slab_mr.deallocate(producer, original, bytes);
  std::jthread release_thread{[&gate] {
    std::this_thread::sleep_for(std::chrono::milliseconds{250});
    gate.release();
  }};
  auto* reused = f.slab_mr.allocate_sync(bytes);
  // Query the producer, not PTDS on a different thread (which would be another stream).
  auto const ready_at_return = ::cudaStreamQuery(producer.value());
  release_thread.join();
  REQUIRE(ready_at_return == cudaSuccess);
  REQUIRE(reused == original);
  std::memset(reused, 0xA7, bytes);
  std::vector<std::byte> copied(bytes);
  CUCASCADE_CUDA_TRY(::cudaMemcpy(copied.data(), device.data(), bytes, cudaMemcpyDeviceToHost));
  REQUIRE(std::all_of(
    copied.begin(), copied.end(), [](std::byte value) { return value == std::byte{0x43}; }));
  f.slab_mr.deallocate_sync(reused, bytes);
}
