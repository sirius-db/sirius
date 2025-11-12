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

#pragma once

// sirius
#include <helper/helper.hpp>
#include <memory/fixed_size_host_memory_resource.hpp>
#include <memory/memory_reservation.hpp>
#include <memory/null_device_memory_resource.hpp>

// standard library
#include <memory>
#include <vector>

// rmm
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

using namespace sirius::memory;

/**
 * @brief Create test allocators for a specific memory tier.
 *
 * @param tier The memory tier (GPU, HOST, or DISK)
 * @param size The size of the allocator (only used for HOST tier)
 * @return A vector containing the appropriate allocator for the tier
 */
inline std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> create_test_allocators(
  Tier tier, size_t size = 0)
{
  std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;

  switch (tier) {
    case Tier::GPU: {
      auto cuda_async_allocator = sirius::make_unique<rmm::mr::cuda_async_memory_resource>();
      allocators.push_back(std::move(cuda_async_allocator));
      break;
    }
    case Tier::HOST: {
      // Use the specified size for the host memory resource
      if (size == 0) {
        size = 100ull * 1024 * 1024;  // Default to 100MB
      }
      auto host_allocator = sirius::make_unique<fixed_size_host_memory_resource>(size);
      allocators.push_back(std::move(host_allocator));
      break;
    }
    case Tier::DISK: {
      auto disk_allocator = sirius::make_unique<null_device_memory_resource>();
      allocators.push_back(std::move(disk_allocator));
      break;
    }
    default: throw std::invalid_argument("Unknown tier type");
  }

  return allocators;
}

/**
 * @brief Initialize the memory reservation manager for tests.
 *
 * Sets up GPU, HOST, and DISK memory tiers with test-appropriate sizes.
 * Uses static initialization to avoid reinitializing for every test (which is slow).
 * Only initializes once per test run.
 *
 */
inline void initialize_memory_manager()
{
  static bool initialized = false;
  if (!initialized) {
    memory_reservation_manager::reset_for_testing();
    std::vector<memory_reservation_manager::memory_space_config> configs;
    // Use appropriate memory sizes - allocator size must match the memory space limit
    // Need enough HOST memory for multiple columns with data + masks + offsets
    size_t gpu_size  = 1ULL * 1024 * 1024;    // 1MB
    size_t host_size = 100ULL * 1024 * 1024;  // 100MB - enough for test data
    size_t disk_size = 10ULL * 1024 * 1024;   // 10MB

    configs.emplace_back(Tier::GPU, 0, gpu_size, create_test_allocators(Tier::GPU, gpu_size));
    configs.emplace_back(Tier::HOST, 0, host_size, create_test_allocators(Tier::HOST, host_size));
    configs.emplace_back(Tier::DISK, 0, disk_size, create_test_allocators(Tier::DISK, disk_size));
    memory_reservation_manager::initialize(std::move(configs));
    initialized = true;
  }
}
