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

// get_target_ctas concurrency (register E6): the per-device CTA-count cache
// used to be one {device, value} pair of mutable function-local static
// variables with a non-atomic check-then-use — two queries on different GPUs
// could interleave the two word writes and read the OTHER device's CTA count
// (observable only when the devices differ in SM count; on identical GPUs the
// torn pair still yields the right number). The cache is now one atomic slot
// per device, so a read is a single word and can never pair a stale device id
// with the other device's value.
//
// This test hammers get_target_ctas from many threads and asserts every call
// returns exactly the value derived from the CALLING thread's current device.
// With one GPU it proves stability under contention; with two or more it
// alternates devices per iteration, which is the exact pre-fix tear shape.

#include <cuda/scan/strings/common.cuh>
#include <cuda_runtime.h>

#include <catch.hpp>

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace {

/// The value get_target_ctas must return for @p device, derived independently.
uint32_t expected_target_ctas(int device)
{
  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, device) == cudaSuccess);
  int const occupancy_blocks =
    prop.maxThreadsPerMultiProcessor / static_cast<int>(sirius::cuda::scan::STRINGS_BLOCK_DIM);
  return static_cast<uint32_t>(prop.multiProcessorCount * occupancy_blocks * 2);
}

}  // namespace

TEST_CASE("get_target_ctas returns the calling device's value under concurrent callers",
          "[scan][target_ctas][concurrency]")
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count < 1) {
    WARN("no CUDA device visible; skipping get_target_ctas concurrency test");
    return;
  }

  std::vector<uint32_t> expected(static_cast<std::size_t>(device_count));
  for (int d = 0; d < device_count; ++d) {
    expected[static_cast<std::size_t>(d)] = expected_target_ctas(d);
    REQUIRE(expected[static_cast<std::size_t>(d)] > 0);
  }

  constexpr int n_threads = 8;
  const int iterations    = 5000;

  // Catch2 assertions are not thread-safe: workers only count mismatches and
  // record one sample; the main thread asserts.
  std::atomic<int> mismatches{0};
  std::atomic<int> cuda_errors{0};
  std::atomic<uint32_t> sample_got{0};
  std::atomic<uint32_t> sample_want{0};

  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < iterations; ++i) {
        // Single GPU: everyone hammers device 0. Multi GPU: alternate devices
        // per thread AND per iteration — the pre-fix shape where the shared
        // {device, value} pair thrashes and can tear across devices.
        int const device = (t + i) % device_count;
        if (cudaSetDevice(device) != cudaSuccess) {
          cuda_errors.fetch_add(1);
          return;
        }
        uint32_t const got  = sirius::cuda::scan::get_target_ctas();
        uint32_t const want = expected[static_cast<std::size_t>(device)];
        if (got != want) {
          mismatches.fetch_add(1);
          sample_got.store(got);
          sample_want.store(want);
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  INFO("devices=" << device_count << " sample got=" << sample_got.load()
                  << " want=" << sample_want.load());
  CHECK(cuda_errors.load() == 0);
  REQUIRE(mismatches.load() == 0);

  if (device_count < 2) {
    // The cross-device tear needs >= 2 GPUs to exercise; on this box only the
    // same-device stability half ran. Documented rather than skipped so the
    // single-GPU CI signal stays green and meaningful.
    WARN("single CUDA device: cross-device alternation not exercised");
  }
}
