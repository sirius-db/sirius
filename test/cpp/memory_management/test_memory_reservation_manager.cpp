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
 * [memory_space] - Basic memory space functionality tests
 * [threading] - Multi-threaded tests
 * [gpu] - GPU-specific tests requiring CUDA
 * [.multi-device] - Tests requiring multiple GPU devices (hidden by default)
 *
 * Running tests:
 * - Default (includes single GPU): ./test_executable
 * - Include multi-device tests: ./test_executable "[.multi-device]"
 * - Exclude multi-device tests: ./test_executable "~[.multi-device]"
 * - Run all tests: ./test_executable "[memory_space]"
 */

#include "catch.hpp"
#include "device_buffer.hpp"
#include "memory/common.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/memory_reservation_manager.hpp"
#include "memory/reservation_aware_resource_adaptor.hpp"
#include "memory/reservation_manager_configurator.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

using namespace cucascade::memory;

// Expected memory capacities
const size_t expected_gpu_capacity  = 2ull << 30;  // 2GB
const size_t expected_host_capacity = 4ull << 30;  // 4GB
const double limit_ratio            = 0.75;

void initializeSingleDeviceMemoryManager()
{
  memory_reservation_manager::reset_for_testing();
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(expected_gpu_capacity);  // 2 GB
  builder.set_reservation_limit_ratio_per_gpu(limit_ratio);
  builder.set_capacity_per_numa_node(expected_host_capacity);  //  4 GB
  builder.set_host_id_to_numa_maps({{0, -1}});
  builder.set_reservation_limit_ratio_per_numa_node(limit_ratio);

  auto space_configs = builder.build_with_topology();
  memory_reservation_manager::initialize(std::move(space_configs));
}

void initializeDualGpuMemoryManager()
{
  memory_reservation_manager::reset_for_testing();
  reservation_manager_configurator builder;
  builder.set_gpu_usage_limit(expected_gpu_capacity);  // 2 GB
  builder.set_reservation_limit_ratio_per_gpu(limit_ratio);
  builder.set_capacity_per_numa_node(expected_host_capacity);  //  4 GB
  builder.set_host_id_to_numa_maps({{0, -1}});
  builder.set_device_tier_id_to_gpu_id_map({{0, 0}, {1, 0}});
  builder.set_reservation_limit_ratio_per_numa_node(limit_ratio);

  auto space_configs = builder.build_with_topology();
  memory_reservation_manager::initialize(std::move(space_configs));
}

TEST_CASE("Single-Device Memory Space Access", "[memory_space]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager = memory_reservation_manager::get_instance();

  // Test single GPU memory space
  auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);

  REQUIRE(gpu_device_0 != nullptr);
  REQUIRE(gpu_device_0->get_tier() == Tier::GPU);
  REQUIRE(gpu_device_0->get_device_id() == 0);
  REQUIRE(gpu_device_0->get_max_memory() == expected_gpu_capacity * limit_ratio);
  REQUIRE(gpu_device_0->get_available_memory() == expected_gpu_capacity);

  // Test single HOST memory space (NUMA node)
  auto host_numa_0 = manager.get_memory_space(Tier::HOST, 0);

  REQUIRE(host_numa_0 != nullptr);
  REQUIRE(host_numa_0->get_tier() == Tier::HOST);
  REQUIRE(host_numa_0->get_device_id() == 0);
  REQUIRE(host_numa_0->get_max_memory() == expected_host_capacity * limit_ratio);
  REQUIRE(host_numa_0->get_available_memory() == expected_host_capacity);

  // Test non-existent devices (only device 0 exists for each tier)
  REQUIRE(manager.get_memory_space(Tier::GPU, 1) == nullptr);
  REQUIRE(manager.get_memory_space(Tier::HOST, 1) == nullptr);

  // Verify all memory spaces are different objects
  REQUIRE(gpu_device_0 != host_numa_0);
}

TEST_CASE("Device-Specific Memory Reservations", "[memory_space]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager = memory_reservation_manager::get_instance();

  // Memory size constants
  const size_t gpu_allocation_size  = 200ull * 1024 * 1024;   // 200MB
  const size_t host_allocation_size = 500ull * 1024 * 1024;   // 500MB
  const size_t disk_allocation_size = 1000ull * 1024 * 1024;  // 1GB

  auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);
  auto host_numa_0  = manager.get_memory_space(Tier::HOST, 0);

  {
    // Test reservation on GPU device
    auto gpu_reservation =
      manager.request_reservation(specific_memory_space(Tier::GPU, 0), gpu_allocation_size);
    REQUIRE(gpu_reservation != nullptr);
    REQUIRE(gpu_reservation->tier() == Tier::GPU);
    REQUIRE(gpu_reservation->device_id() == 0);
    REQUIRE(gpu_reservation->size() == gpu_allocation_size);

    // Check memory accounting on GPU device
    REQUIRE(gpu_device_0->get_total_reserved_memory() == gpu_allocation_size);
    REQUIRE(gpu_device_0->get_active_reservation_count() == 1);
    REQUIRE(gpu_device_0->get_available_memory() == expected_gpu_capacity - gpu_allocation_size);

    // // Check that other devices are unaffected
    REQUIRE(host_numa_0->get_total_reserved_memory() == 0);
    REQUIRE(host_numa_0->get_active_reservation_count() == 0);

    // // Test reservation on HOST NUMA node
    auto host_reservation =
      manager.request_reservation(any_memory_space_in_tier(Tier::HOST), host_allocation_size);
    REQUIRE(host_reservation != nullptr);
    REQUIRE(host_reservation->tier() == Tier::HOST);
    REQUIRE(host_reservation->device_id() == 0);
    REQUIRE(host_reservation->size() == host_allocation_size);

    // // Check HOST memory accounting
    REQUIRE(host_numa_0->get_total_reserved_memory() == host_allocation_size);
    REQUIRE(host_numa_0->get_active_reservation_count() == 1);
    REQUIRE(host_numa_0->get_available_memory() == expected_host_capacity - host_allocation_size);
  }

  // Verify cleanup
  REQUIRE(gpu_device_0->get_total_reserved_memory() == 0);
  REQUIRE(gpu_device_0->get_active_reservation_count() == 0);
  REQUIRE(gpu_device_0->get_available_memory() == expected_gpu_capacity);

  REQUIRE(host_numa_0->get_total_reserved_memory() == 0);
  REQUIRE(host_numa_0->get_active_reservation_count() == 0);
  REQUIRE(host_numa_0->get_available_memory() == expected_host_capacity);
}

TEST_CASE("Reservation Strategies with Single Device", "[memory_space]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager = memory_reservation_manager::get_instance();

  // Test allocation sizes
  const size_t small_allocation  = 25ull * 1024 * 1024;   // 25MB
  const size_t medium_allocation = 50ull * 1024 * 1024;   // 50MB
  const size_t large_allocation  = 100ull * 1024 * 1024;  // 100MB

  // Test requesting reservation in any GPU
  auto gpu_any_reservation =
    manager.request_reservation(any_memory_space_in_tier(Tier::GPU), medium_allocation);
  REQUIRE(gpu_any_reservation != nullptr);
  REQUIRE(gpu_any_reservation->tier() == Tier::GPU);
  REQUIRE(gpu_any_reservation->size() == medium_allocation);

  // Should pick the single GPU device (device 0)
  REQUIRE(gpu_any_reservation->device_id() == 0);

  // Test requesting reservation across multiple tiers (simulates "anywhere")
  std::vector<Tier> any_tier_preferences = {Tier::GPU, Tier::HOST, Tier::DISK};
  auto anywhere_reservation =
    manager.request_reservation(any_memory_space_in_tiers(any_tier_preferences), small_allocation);
  REQUIRE(anywhere_reservation != nullptr);
  REQUIRE(anywhere_reservation->size() == small_allocation);

  // Should pick any available memory space
  Tier selected_tier = anywhere_reservation->tier();
  REQUIRE(
    (selected_tier == Tier::GPU || selected_tier == Tier::HOST || selected_tier == Tier::DISK));

  // Test specific memory space in tiers list with HOST preference
  std::vector<Tier> tier_preferences = {Tier::HOST, Tier::GPU, Tier::DISK};
  auto preference_reservation =
    manager.request_reservation(any_memory_space_in_tiers(tier_preferences), large_allocation);
  REQUIRE(preference_reservation != nullptr);
  REQUIRE(preference_reservation->size() == large_allocation);

  // Should prefer HOST first
  REQUIRE(preference_reservation->tier() == Tier::HOST);
}

SCENARIO("multi-reservation memory_resource mismatch", "[memory_space]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager = memory_reservation_manager::get_instance();

  const size_t res_size         = 1ull * 1024 * 1024;  // 1MB
  const size_t small_alloc_size = res_size / 2;        // 512KB
  const size_t large_alloc_size = res_size * 2;        // 2MB

  GIVEN("Two reservations of 1MB each on different on different streams")
  {
    auto res1 = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);
    auto res2 = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, res_size);

    auto* mr = res1->get_memory_resource_of<Tier::GPU>();
    rmm::cuda_stream stream1, stream2;

    mr->attach_reservation_to_tracker(stream1, std::move(res1));
    mr->attach_reservation_to_tracker(stream2, std::move(res2));

    WHEN("releasing allocation from a different memory resource")
    {
      auto upstrem_leftover = mr->get_available_memory();
      REQUIRE(mr->get_available_memory(stream1) == upstrem_leftover + res_size);
      REQUIRE(mr->get_available_memory(stream2) == upstrem_leftover + res_size);
      auto* buff1 = mr->allocate(small_alloc_size, stream1);
      REQUIRE(mr->get_allocated_bytes(stream1) == small_alloc_size);
      REQUIRE(mr->get_available_memory(stream1) ==
              mr->get_available_memory() + res_size - small_alloc_size);

      auto* buff2 = mr->allocate(large_alloc_size, stream2);
      REQUIRE(mr->get_allocated_bytes(stream2) == large_alloc_size);
      REQUIRE(mr->get_available_memory(stream2) == mr->get_available_memory());
      THEN(
        "allocations from another memory resource is absorbed by other stream as extra reservation")
      {
        mr->deallocate(buff2, large_alloc_size, stream1);
        CHECK(mr->get_available_memory_print(stream1) ==
              mr->get_available_memory() + large_alloc_size + res_size - small_alloc_size);

        mr->deallocate(buff1, small_alloc_size, stream2);
        CHECK(mr->get_available_memory_print(stream2) == mr->get_available_memory());
      }
    }
  }
}

SCENARIO("Peak Tracking On Streams with Reservation", "[memory_space][tracking]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager                 = memory_reservation_manager::get_instance();
  const size_t reservation_size = 2048;
  const size_t chunk_size       = 1024;

  GIVEN("A reservation of specific size[= 2048] on GPU")
  {
    auto res = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
    REQUIRE(res->size() == reservation_size);
    REQUIRE(res->tier() == Tier::GPU);
    auto* mr = res->get_memory_resource_of<Tier::GPU>();
    REQUIRE(mr != nullptr);

    rmm::cuda_stream other_streams;
    rmm::cuda_stream reserved_stream;

    std::vector<rmm::device_buffer> ptrs;
    mr->attach_reservation_to_tracker(reserved_stream, std::move(res));

    THEN("reservation is reflected in peak allocated bytes in upstream")
    {
      REQUIRE(mr->get_peak_total_allocated_bytes() == reservation_size);
      REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) == 0);
      REQUIRE(mr->get_peak_allocated_bytes(other_streams) == 0);
    }

    WHEN("allocation within reservation on reserved stream, upstream peak doesn't change")
    {
      ptrs.emplace_back(chunk_size, reserved_stream, mr);  // within reservation

      THEN("upstream peak allocated bytes remain the same, only stream peak changes")
      {
        REQUIRE(mr->get_peak_total_allocated_bytes() == reservation_size);
        REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) == chunk_size);
        REQUIRE(mr->get_peak_allocated_bytes(other_streams) == 0);
      }

      WHEN("allocation is larger than reservation on reserved stream, upstream tracks it")
      {
        ptrs.emplace_back(chunk_size, reserved_stream, mr);  // within reservation
        ptrs.emplace_back(chunk_size, reserved_stream, mr);  // exceeds reservation

        THEN("peak allocated bytes are tracked correctly")
        {
          auto total_allocated_bytes = mr->get_total_allocated_bytes();
          REQUIRE(total_allocated_bytes == 3 * chunk_size);
          REQUIRE(mr->get_peak_total_allocated_bytes() == total_allocated_bytes);
          REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) == total_allocated_bytes);
          REQUIRE(mr->get_peak_allocated_bytes(other_streams) == 0);
        }

        WHEN("allocations are freed")
        {
          std::size_t peak_stream_bytes = mr->get_peak_allocated_bytes(reserved_stream);

          REQUIRE(mr->get_total_allocated_bytes() == 3 * chunk_size);
          REQUIRE(mr->get_allocated_bytes(reserved_stream) == 3 * chunk_size);
          mr->reset_stream_reservation(reserved_stream);
          REQUIRE(mr->get_total_allocated_bytes() == 3 * chunk_size);
          ptrs.clear();
          REQUIRE(mr->get_total_allocated_bytes() == 0);

          THEN("peak allocated bytes are tracked correctly")
          {
            REQUIRE(mr->get_peak_total_allocated_bytes() == peak_stream_bytes);
            REQUIRE(mr->get_peak_allocated_bytes(reserved_stream) ==
                    0);                                     // doesn't have a tracker attached
            REQUIRE(mr->get_total_allocated_bytes() == 0);  // doesn't have a tracker attached
          }
        }
      }
    }
  }
}

SCENARIO("Reservation Concepts on Single Gpu Manager", "[memory_space]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager                 = memory_reservation_manager::get_instance();
  const size_t reservation_size = 1024;

  GIVEN("A single gpu manager")
  {
    WHEN("a reservation is made with overflow policy to ignore")
    {
      auto res = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->size() == reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      auto* mr = res->get_memory_resource_as<reservation_aware_resource_adaptor>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream reserved_stream;
      rmm::cuda_stream other_streams;
      mr->attach_reservation_to_tracker(reserved_stream, std::move(res));

      THEN("upstream and others see it as allocated/unavailable")
      {
        REQUIRE(mr->get_total_allocated_bytes() == 1024);
        REQUIRE(mr->get_available_memory(other_streams) ==
                expected_gpu_capacity - reservation_size);
        REQUIRE(mr->get_allocated_bytes(other_streams) == 0);
      }

      THEN("only reserved stream has access to it")
      {
        REQUIRE(mr->get_available_memory(reserved_stream) == expected_gpu_capacity);
        REQUIRE(mr->get_allocated_bytes(reserved_stream) == 0);
      }

      THEN("allocation within the reservations are seen by upstream/other stream")
      {
        std::size_t allocation_size = 512;
        void* ptr                   = mr->allocate(allocation_size, reserved_stream);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
        REQUIRE(mr->get_available_memory(other_streams) ==
                expected_gpu_capacity - reservation_size);
        REQUIRE(mr->get_allocated_bytes(other_streams) == 0);
        REQUIRE(mr->get_available_memory(reserved_stream) ==
                expected_gpu_capacity - allocation_size);
        REQUIRE(mr->get_allocated_bytes(reserved_stream) == allocation_size);
        mr->deallocate(ptr, allocation_size, reserved_stream);
      }

      THEN("allocation beyond the reservations are made from the upstream")
      {
        std::size_t allocation_size = reservation_size * 2;
        void* ptr                   = mr->allocate(allocation_size, reserved_stream);
        REQUIRE(mr->get_total_allocated_bytes() == allocation_size);
        REQUIRE(mr->get_available_memory(other_streams) == expected_gpu_capacity - allocation_size);
        REQUIRE(mr->get_allocated_bytes(other_streams) == 0);
        REQUIRE(mr->get_available_memory(reserved_stream) ==
                expected_gpu_capacity - allocation_size);
        REQUIRE(mr->get_allocated_bytes(reserved_stream) == allocation_size);
        mr->deallocate(ptr, allocation_size, reserved_stream);
      }
    }
  }
}

SCENARIO("Reservation Overflow Policy", "[memory_space][.overflow_policy]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager                 = memory_reservation_manager::get_instance();
  const size_t reservation_size = 1024;

  GIVEN("A single gpu manager")
  {
    WHEN("allocation beyond reservation with ignore policy")
    {
      auto res = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->size() == reservation_size);
      auto* mr = res->get_memory_resource_of<Tier::GPU>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream stream;
      mr->attach_reservation_to_tracker(
        stream, std::move(res), std::make_unique<ignore_reservation_limit_policy>());

      THEN("total reservation doesn't change")
      {
        auto* buffer = mr->allocate(reservation_size * 2, stream);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size * 2);
        mr->deallocate(buffer, reservation_size * 2, stream);
      }
    }

    WHEN("allocation beyond reservation with fail policy")
    {
      auto res = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->size() == reservation_size);

      auto* mr = res->get_memory_resource_as<reservation_aware_resource_adaptor>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream stream;
      mr->attach_reservation_to_tracker(
        stream, std::move(res), std::make_unique<fail_reservation_limit_policy>());

      THEN("oom on allocation")
      {
        REQUIRE_THROWS_AS(mr->allocate(reservation_size * 2, stream), rmm::bad_alloc);
      }
    }

    WHEN("allocation beyond reservation with increase policy")
    {
      auto res = manager.request_reservation(specific_memory_space{Tier::GPU, 0}, reservation_size);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->size() == reservation_size);

      auto* mr = res->get_memory_resource_of<Tier::GPU>();
      REQUIRE(mr != nullptr);

      rmm::cuda_stream stream;
      mr->attach_reservation_to_tracker(
        stream, std::move(res), std::make_unique<increase_reservation_limit_policy>(2.0));

      THEN("increased reservation on allocation")
      {
        auto* buffer = mr->allocate(reservation_size * 2, stream);
        REQUIRE(mr->get_total_reserved_bytes() >= reservation_size * 2);
        REQUIRE(mr->get_total_allocated_bytes() >= reservation_size * 2);
        mr->deallocate(buffer, reservation_size * 2, stream);
      }
    }
  }
}

SCENARIO("Reservation On Multi Gpu System", "[memory_space][.multi-device]")
{
  initializeDualGpuMemoryManager();
  auto& manager = memory_reservation_manager::get_instance();

  auto gpu_device_0 = manager.get_memory_space(Tier::GPU, 0);
  auto gpu_device_1 = manager.get_memory_space(Tier::GPU, 1);
  auto host_numa_0  = manager.get_memory_space(Tier::HOST, 0);

  // Test that we can get default allocators from each device
  auto gpu_0_allocator  = gpu_device_0->get_default_allocator();
  auto gpu_1_allocator  = gpu_device_1->get_default_allocator();
  auto host_0_allocator = host_numa_0->get_default_allocator();

  // Test that allocators are valid (basic smoke test)
  REQUIRE(gpu_0_allocator != nullptr);
  REQUIRE(gpu_1_allocator != nullptr);
  REQUIRE(host_0_allocator != nullptr);

  GIVEN("Dual gpu manager")
  {
    auto* gpu_space = manager.get_memory_space(Tier::GPU, 0);
    auto* mr =
      dynamic_cast<reservation_aware_resource_adaptor*>(gpu_space->get_default_allocator());
    REQUIRE(mr != nullptr);

    WHEN("a reservation doesn't fit on gpu 0 but fits on gpu 1")
    {
      size_t large_reservation = expected_gpu_capacity * limit_ratio - 1024;
      auto res =
        manager.request_reservation(specific_memory_space{Tier::GPU, 0}, large_reservation);
      REQUIRE(res->size() == large_reservation);
      REQUIRE(res->tier() == Tier::GPU);
      REQUIRE(res->device_id() == 0);

      THEN("reservation made on gpu 1")
      {
        auto other_res =
          manager.request_reservation(any_memory_space_in_tier{Tier::GPU}, large_reservation);
        REQUIRE(other_res->size() == large_reservation);
        REQUIRE(other_res->tier() == Tier::GPU);
        REQUIRE(other_res->device_id() == 1);
      }
    }
  }
}

SCENARIO("Host Reservation", "[memory_space][host_reservation]")
{
  initializeSingleDeviceMemoryManager();
  auto& manager                = memory_reservation_manager::get_instance();
  std::size_t reservation_size = 2UL << 20;
  std::size_t small_allocation = 1UL << 20;
  std::size_t large_allocation = 4UL << 20;

  GIVEN("making a host reservation")
  {
    auto reservation =
      manager.request_reservation(any_memory_space_in_tier{Tier::HOST}, reservation_size);
    REQUIRE(reservation->size() == reservation_size);
    REQUIRE(reservation->tier() == Tier::HOST);
    auto* mr = reservation->get_memory_resource_of<Tier::HOST>();
    REQUIRE(mr != nullptr);
    REQUIRE(mr->get_total_allocated_bytes() == reservation_size);

    WHEN("allocation made larger than reservation")
    {
      auto free_memory_before = mr->get_available_memory();
      auto blocks             = mr->allocate_multiple_blocks(large_allocation, reservation.get());

      THEN("upstream and others see it as allocated/unavailable")
      {
        REQUIRE(mr->get_available_memory() < free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == large_allocation);
      }

      blocks.reset(nullptr);

      THEN("after deallocation, reservation is still held, extra is freed")
      {
        REQUIRE(mr->get_available_memory() == free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
      }
    }

    WHEN("allocation made fits inside reservation")
    {
      rmm::cuda_stream stream;
      auto free_memory_before = mr->get_available_memory();
      auto blocks             = mr->allocate_multiple_blocks(small_allocation, reservation.get());

      THEN("upstream and others see doesn't change")
      {
        REQUIRE(mr->get_available_memory() == free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
      }

      blocks.reset(nullptr);

      THEN("after deallocation, reservation is still held")
      {
        REQUIRE(mr->get_available_memory() == free_memory_before);
        REQUIRE(mr->get_total_reserved_bytes() == reservation_size);
        REQUIRE(mr->get_total_allocated_bytes() == reservation_size);
      }
    }

    WHEN("reservation is freed before allocation")
    {
      rmm::cuda_stream stream;
      auto free_memory_before = mr->get_available_memory();
      auto blocks             = mr->allocate_multiple_blocks(small_allocation, reservation.get());
      reservation.reset();

      THEN("upstream shrink the reservation to fit")
      {
        REQUIRE(mr->get_total_reserved_bytes() == 0);
        REQUIRE(mr->get_total_allocated_bytes() == small_allocation);
      }

      blocks.reset(nullptr);

      THEN("after deallocation, reservation is still held")
      {
        REQUIRE(mr->get_total_reserved_bytes() == 0);
        REQUIRE(mr->get_total_allocated_bytes() == 0);
      }
    }
  }
}
