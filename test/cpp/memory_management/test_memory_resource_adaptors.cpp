/*
 * Copyright 2025, Sirius Contributors.
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
 * Test Tags:
 * [per_stream_tracking] - Basic per-stream tracking functionality tests
 * [threading] - Multi-threaded tests
 * [gpu] - GPU-specific tests requiring CUDA
 * [race_conditions] - Tests for race condition handling
 *
 * Running tests:
 * - Default: ./test_executable "[per_stream_tracking]"
 * - Threading tests: ./test_executable "[per_stream_tracking][threading]"
 * - Race condition tests: ./test_executable "[per_stream_tracking][race_conditions]"
 */

#include "catch.hpp"
#include "cuda_device.hpp"
#include "cuda_stream.hpp"
#include "memory/common.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "memory/memory_reservation_manager.hpp"
#include "memory/reservation_aware_resource_adaptor.hpp"
#include "memory/reservation_manager_configurator.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <vector>

// RMM includes
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime.h>

using namespace sirius::memory;

// Test fixture for per-stream tracking tests
class ReservationAwareTrackingTest {
 public:
  ReservationAwareTrackingTest() : guard(rmm::cuda_device_id{0}), streams{3}
  {
    initialize_engine();
    auto& manager = memory_reservation_manager::get_instance();
    for (auto& stream : streams) {
      auto res    = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, 0);
      tracking_mr = res->get_memory_resource_of<Tier::GPU>();
      tracking_mr->attach_reservation_to_tracker(stream, std::move(res));
    }
  }

  void initialize_engine()
  {
    memory_reservation_manager::reset_for_testing();
    reservation_manager_configurator builder;
    builder.set_gpu_usage_limit(100UL << 20);
    builder.set_reservation_limit_ratio_per_gpu(0.75);
    auto space_configs = builder.build_with_topology();
    memory_reservation_manager::initialize(std::move(space_configs));
  }

  ~ReservationAwareTrackingTest() { memory_reservation_manager::reset_for_testing(); }

  rmm::cuda_stream_view get_stream(std::size_t index) { return streams.at(index); }

  reservation_aware_resource_adaptor* get_memory_resource() noexcept { return tracking_mr; }

 private:
  reservation_aware_resource_adaptor* tracking_mr;
  rmm::cuda_set_device_raii guard;
  std::vector<rmm::cuda_stream> streams;
};

TEST_CASE_METHOD(ReservationAwareTrackingTest,
                 "Basic allocation and deallocation tracking",
                 "[per_stream_tracking]")
{
  const std::size_t alloc_size = 1024;

  SECTION("Single stream allocation")
  {
    rmm::cuda_stream_view stream_view = get_stream(0);
    auto* tracking_mr                 = get_memory_resource();

    // Initial state
    REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == 0);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) == 0);
    REQUIRE(tracking_mr->get_total_allocated_bytes() == 0);

    // Allocate memory
    void* ptr = tracking_mr->allocate(alloc_size, stream_view);
    REQUIRE(ptr != nullptr);

    // Check tracking
    REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == alloc_size);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) == alloc_size);
    REQUIRE(tracking_mr->get_total_allocated_bytes() == alloc_size);
    REQUIRE(tracking_mr->is_stream_tracked(stream_view));

    // Deallocate memory
    tracking_mr->deallocate(ptr, alloc_size, stream_view);

    // Check tracking after deallocation
    REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == 0);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) ==
            alloc_size);  // Peak should remain
    REQUIRE(tracking_mr->get_total_allocated_bytes() == 0);
  }

  SECTION("Multiple allocations on same stream")
  {
    rmm::cuda_stream_view stream_view = get_stream(0);
    auto* tracking_mr                 = get_memory_resource();

    // First allocation
    void* ptr1 = tracking_mr->allocate(alloc_size, stream_view);
    REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == alloc_size);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) == alloc_size);

    // Second allocation (should increase both current and peak)
    void* ptr2 = tracking_mr->allocate(alloc_size * 2, stream_view);
    REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == alloc_size * 3);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) == alloc_size * 3);

    // Deallocate first allocation
    tracking_mr->deallocate(ptr1, alloc_size, stream_view);
    REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == alloc_size * 2);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) ==
            alloc_size * 3);  // Peak unchanged

    // Deallocate second allocation
    tracking_mr->deallocate(ptr2, alloc_size * 2, stream_view);
    REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == 0);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) ==
            alloc_size * 3);  // Peak unchanged
  }
}

TEST_CASE_METHOD(ReservationAwareTrackingTest, "Multiple stream tracking", "[per_stream_tracking]")
{
  const std::size_t alloc_size = 1024;

  rmm::cuda_stream_view stream_view1 = get_stream(0);
  rmm::cuda_stream_view stream_view2 = get_stream(1);
  rmm::cuda_stream_view stream_view3 = get_stream(2);

  auto* tracking_mr = get_memory_resource();

  // Allocate on different streams
  void* ptr1 = tracking_mr->allocate(alloc_size, stream_view1);
  void* ptr2 = tracking_mr->allocate(alloc_size * 2, stream_view2);
  void* ptr3 = tracking_mr->allocate(alloc_size * 3, stream_view3);

  // Check individual stream tracking
  REQUIRE(tracking_mr->get_allocated_bytes(stream_view1) == alloc_size);
  REQUIRE(tracking_mr->get_allocated_bytes(stream_view2) == alloc_size * 2);
  REQUIRE(tracking_mr->get_allocated_bytes(stream_view3) == alloc_size * 3);

  REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view1) == alloc_size);
  REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view2) == alloc_size * 2);
  REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view3) == alloc_size * 3);

  // Check total tracking
  REQUIRE(tracking_mr->get_total_allocated_bytes() == alloc_size * 6);
  REQUIRE(tracking_mr->get_peak_total_allocated_bytes() == alloc_size * 6);

  // Deallocate from stream2
  tracking_mr->deallocate(ptr2, alloc_size * 2, stream_view2);
  REQUIRE(tracking_mr->get_allocated_bytes(stream_view2) == 0);
  REQUIRE(tracking_mr->get_total_allocated_bytes() == alloc_size * 4);
  REQUIRE(tracking_mr->get_peak_total_allocated_bytes() == alloc_size * 6);  // Peak unchanged

  // Clean up
  tracking_mr->deallocate(ptr1, alloc_size, stream_view1);
  tracking_mr->deallocate(ptr3, alloc_size * 3, stream_view3);
}

TEST_CASE_METHOD(ReservationAwareTrackingTest, "Reset functionality", "[per_stream_tracking]")
{
  const std::size_t alloc_size = 1024;

  rmm::cuda_stream_view stream_view1 = get_stream(0);
  rmm::cuda_stream_view stream_view2 = get_stream(1);

  auto* tracking_mr = get_memory_resource();

  // Allocate on both streams
  void* ptr1 = tracking_mr->allocate(alloc_size, stream_view1);
  void* ptr2 = tracking_mr->allocate(alloc_size * 2, stream_view2);

  REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view1) == alloc_size);
  REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view2) == alloc_size * 2);
  REQUIRE(tracking_mr->get_peak_total_allocated_bytes() == alloc_size * 3);

  SECTION("Reset single stream")
  {
    tracking_mr->reset_peak_allocated_bytes(stream_view1);

    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view1) == 0);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view2) == alloc_size * 2);  // Unchanged
    REQUIRE(tracking_mr->get_peak_total_allocated_bytes() ==
            alloc_size * 3);  // Global peak unchanged
  }

  SECTION("Reset all streams")
  {
    for (auto& stream_view : {stream_view1, stream_view2}) {
      tracking_mr->reset_peak_allocated_bytes(stream_view);
    }
    tracking_mr->reset_peak_total_allocated_bytes();

    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view1) == 0);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view2) == 0);
    REQUIRE(tracking_mr->get_peak_total_allocated_bytes() == 0);  // Global peak also reset
  }

  // Clean up
  tracking_mr->deallocate(ptr1, alloc_size, stream_view1);
  tracking_mr->deallocate(ptr2, alloc_size * 2, stream_view2);
}

TEST_CASE_METHOD(ReservationAwareTrackingTest,
                 "Concurrent allocation and deallocation",
                 "[per_stream_tracking][threading]")
{
  const std::size_t num_threads                = 4;
  const std::size_t num_allocations_per_thread = 100;
  const std::size_t alloc_size                 = 256;

  auto* tracking_mr = get_memory_resource();
  auto stream1      = get_stream(0);

  std::vector<std::thread> threads;
  std::vector<std::vector<void*>> thread_ptrs(num_threads);
  std::atomic<bool> start_flag{false};

  // Create worker threads
  for (std::size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&, i]() {
      // Wait for start signal
      while (!start_flag.load()) {
        std::this_thread::yield();
      }

      rmm::cuda_stream_view stream_view{stream1};  // All use same stream for this test
      thread_ptrs[i].reserve(num_allocations_per_thread);

      // Allocate phase
      for (std::size_t j = 0; j < num_allocations_per_thread; ++j) {
        void* ptr = tracking_mr->allocate(alloc_size, stream_view);
        thread_ptrs[i].push_back(ptr);

        // Small delay to increase chance of race conditions
        if (j % 10 == 0) { std::this_thread::sleep_for(std::chrono::microseconds(1)); }
      }

      // Deallocate phase
      for (void* ptr : thread_ptrs[i]) {
        tracking_mr->deallocate(ptr, alloc_size, stream_view);
      }
    });
  }

  // Start all threads
  start_flag.store(true);

  // Wait for completion
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify final state
  rmm::cuda_stream_view stream_view{stream1};
  REQUIRE(tracking_mr->get_allocated_bytes(stream_view) == 0);
  REQUIRE(tracking_mr->get_total_allocated_bytes() == 0);

  // Peak should reflect the peak usage during concurrent allocations
  std::size_t expected_peak = num_threads * num_allocations_per_thread * alloc_size;
  REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) <= expected_peak);
  REQUIRE(tracking_mr->get_peak_allocated_bytes(stream_view) > 0);
}

TEST_CASE_METHOD(ReservationAwareTrackingTest, "Edge cases", "[per_stream_tracking]")
{
  auto* tracking_mr = get_memory_resource();
  auto stream_view  = get_stream(0);

  SECTION("Zero-size allocation")
  {
    void* ptr = tracking_mr->allocate(0, stream_view);
    // Behavior depends on upstream resource, but tracking should handle it
    if (ptr != nullptr) { tracking_mr->deallocate(ptr, 0, stream_view); }
    // Should not crash or cause issues with tracking
  }

  SECTION("Query non-existent stream")
  {
    cudaStream_t non_existent_stream;
    cudaStreamCreate(&non_existent_stream);
    rmm::cuda_stream_view non_existent_view{non_existent_stream};

    REQUIRE(tracking_mr->get_allocated_bytes(non_existent_view) == 0);
    REQUIRE(tracking_mr->get_peak_allocated_bytes(non_existent_view) == 0);
    REQUIRE_FALSE(tracking_mr->is_stream_tracked(non_existent_view));

    cudaStreamDestroy(non_existent_stream);
  }

  SECTION("Reset non-existent stream")
  {
    cudaStream_t non_existent_stream;
    cudaStreamCreate(&non_existent_stream);
    rmm::cuda_stream_view non_existent_view{non_existent_stream};

    // Should not crash
    tracking_mr->reset_peak_allocated_bytes(non_existent_view);

    cudaStreamDestroy(non_existent_stream);
  }
}
