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

#include <rmm/mr/device/cuda_async_memory_resource.hpp>

#include <helper/helper.hpp>
#include <memory/fixed_size_host_memory_resource.hpp>
#include <memory/memory_reservation.hpp>
#include <memory/null_device_memory_resource.hpp>
#include <scan/duckdb_scan_task.hpp>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace sirius::memory;

/**
 * @brief Create test allocators for a specific memory tier.
 *
 * @param tier The memory tier (GPU, HOST, or DISK)
 * @return A vector containing the appropriate allocator for the tier
 */
inline std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> create_test_allocators(
  Tier tier)
{
  std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;

  switch (tier) {
    case Tier::GPU: {
      auto cuda_async_allocator = sirius::make_unique<rmm::mr::cuda_async_memory_resource>();
      allocators.push_back(std::move(cuda_async_allocator));
      break;
    }
    case Tier::HOST: {
      // Use a larger host memory resource for integration tests (100MB)
      auto host_allocator =
        sirius::make_unique<fixed_size_host_memory_resource>(100ull * 1024 * 1024);
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
 */
inline void initialize_memory_manager()
{
  static bool initialized = false;
  if (!initialized) {
    memory_reservation_manager::reset_for_testing();
    std::vector<memory_reservation_manager::memory_space_config> configs;
    // Use much smaller memory sizes for faster initialization (1MB GPU, 10MB HOST, 10MB DISK)
    configs.emplace_back(Tier::GPU, 0, 1ULL * 1024 * 1024, create_test_allocators(Tier::GPU));
    configs.emplace_back(Tier::HOST, 0, 10ULL * 1024 * 1024, create_test_allocators(Tier::HOST));
    configs.emplace_back(Tier::DISK, 0, 10ULL * 1024 * 1024, create_test_allocators(Tier::DISK));
    memory_reservation_manager::initialize(std::move(configs));
    initialized = true;
  }
}

//===----------------------------------------------------------------------===//
// Helper: Extract Data from Column Builders
//===----------------------------------------------------------------------===//

/**
 * @brief Extract fixed-width data from a column builder for verification.
 */
template <typename T>
inline std::vector<T> extract_fixed_width_data(
  const sirius::parallel::duckdb_scan_task_local_state::column_builder& builder)
{
  std::vector<T> result;
  size_t num_values = builder.total_data_bytes / sizeof(T);

  auto& accessor = const_cast<sirius::parallel::multiple_blocks_allocation_accessor<uint8_t>&>(
    builder.data_blocks_accessor);
  accessor.set_cursor(0);

  for (size_t i = 0; i < num_values; ++i) {
    T value;
    std::memcpy(&value,
                static_cast<uint8_t*>(accessor.allocation->blocks[accessor.block_index]) +
                  accessor.offset_in_block,
                sizeof(T));
    result.push_back(value);
    accessor.offset_in_block += sizeof(T);
    if (accessor.offset_in_block >= accessor.allocation->block_size) {
      ++accessor.block_index;
      accessor.offset_in_block = 0;
    }
  }

  return result;
}

/**
 * @brief Extract VARCHAR data from a column builder for verification.
 */
inline std::vector<std::string> extract_varchar_data(
  const sirius::parallel::duckdb_scan_task_local_state::column_builder& builder, size_t num_rows)
{
  std::vector<std::string> result;

  // Read offsets
  std::vector<int64_t> offsets;
  auto& offset_accessor =
    const_cast<sirius::parallel::multiple_blocks_allocation_accessor<int64_t>&>(
      builder.offset_blocks_accessor);
  offset_accessor.set_cursor(0);
  for (size_t i = 0; i <= num_rows; ++i) {
    offsets.push_back(offset_accessor.get_current());
    offset_accessor.advance();
  }

  // Read string data
  auto& data_accessor = const_cast<sirius::parallel::multiple_blocks_allocation_accessor<uint8_t>&>(
    builder.data_blocks_accessor);
  data_accessor.set_cursor(0);

  for (size_t i = 0; i < num_rows; ++i) {
    int64_t length = offsets[i + 1] - offsets[i];
    if (length > 0) {
      std::vector<char> buffer(length);
      for (int64_t j = 0; j < length; ++j) {
        buffer[j] = static_cast<char>(data_accessor.get_current());
        data_accessor.advance();
      }
      result.emplace_back(buffer.begin(), buffer.end());
    } else {
      result.emplace_back("");
    }
  }

  return result;
}

/**
 * @brief Extract validity mask from a column builder for verification.
 */
inline std::vector<bool> extract_validity_mask(
  const sirius::parallel::duckdb_scan_task_local_state::column_builder& builder, size_t num_rows)
{
  std::vector<bool> result;

  auto& mask_accessor = const_cast<sirius::parallel::multiple_blocks_allocation_accessor<uint8_t>&>(
    builder.mask_blocks_accessor);

  for (size_t i = 0; i < num_rows; ++i) {
    size_t byte_idx = i / 8;
    size_t bit_idx  = i % 8;

    mask_accessor.set_cursor(byte_idx);
    uint8_t byte_value = mask_accessor.get_current();

    bool is_valid = (byte_value & (1 << bit_idx)) != 0;
    result.push_back(is_valid);
  }

  return result;
}
