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

// cucascade
#include "duckdb/common/assert.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace sirius::memory {

/**
 * @brief Accessor for multiple blocks allocation from fixed_size_host_memory_resource.
 *
 * This accessor facilitates reading and writing data across multiple blocks
 * allocated by the fixed-size host memory resource. It manages the current
 * position within the allocation and provides methods to set/get values,
 * advance the cursor, and perform memcpy operations.
 * NOTE: the caller is responsible for ensuring the cursor does not go out of bounds. Otherwise,
 * behavior is undefined.
 *
 * @tparam T The underlying data type to be accessed. It is assumed that T is aligned with the block
 * size of the allocation.
 */
template <typename T>
struct multiple_blocks_allocation_accessor {
  using underlying_type = T;
  using multiple_blocks_allocation =
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation;

  //===----------Fields----------===//
  size_t block_size      = 0;  ///< The size of each block in bytes
  size_t num_blocks      = 0;  ///< The number of blocks in the allocation
  size_t block_index     = 0;  ///< The current block index
  size_t offset_in_block = 0;  ///< The current byte offset in the block
  size_t initial_byte_offset =
    0;  ///< The initial byte offset into the allocation set during initialize

  /**
   * @brief Initialize the accessor with a byte offset within the allocation.
   *
   * @param[in] byte_offset The byte offset within the allocation at which this accessor starts.
   * @param[in] allocation The multiple blocks allocation (unique_ptr or shared_ptr).
   * @throws std::runtime_error if the block size is not a multiple of the size of T.
   */
  /// Templated on the smart-pointer type so callers can pass either
  /// std::unique_ptr<...> (the duckdb/parquet scan task path that owns the
  /// allocation) or std::shared_ptr<...> (the host_table_chunk_reader path,
  /// where cucascade now hands out shared host_table_allocation buffers).
  template <typename Ptr>
  void initialize(size_t byte_offset, Ptr const& allocation)
  {
    assert(allocation != nullptr);

    block_size = allocation->block_size();
    if (block_size % sizeof(T) != 0) {
      throw std::runtime_error(
        "[multiple_blocks_allocation_accessor] The underlying type size must be aligned with the "
        "block size.");
    }
    num_blocks          = allocation->get_blocks().size();
    initial_byte_offset = byte_offset;
    set_cursor(byte_offset);
  }

  /**
   * @brief Get the current global byte offset within the allocation.
   *
   * @return The current global byte offset.
   */
  [[nodiscard]] size_t get_current_global_byte_offset() const
  {
    return initial_byte_offset + block_index * block_size + offset_in_block;
  }

  /**
   * @brief Set the cursor to a specific byte offset within the allocation.
   *
   * @param[in] byte_offset The global byte offset within the allocation.
   */
  void set_cursor(size_t byte_offset)
  {
    assert(block_size != 0);  // Ensure initialized

    block_index     = byte_offset / block_size;
    offset_in_block = byte_offset % block_size;
  };

  /**
   * @brief Reset the cursor to the initial byte offset set during initialization.
   */
  void reset_cursor() { set_cursor(initial_byte_offset); }

  /**
   * @brief Set value at the current position in the allocation.
   *
   * @param[in] value The value to set.
   * @param[in, out] allocation The allocation.
   */
  template <typename Ptr>
  void set_current(T value, Ptr& allocation)
  {
    assert(block_index < num_blocks);
    assert(allocation != nullptr);
    assert(offset_in_block + sizeof(T) <= block_size);
    static_assert(
      std::is_trivially_copyable_v<T>,
      "[multiple_blocks_allocation_accessor] The underlying type must be trivially copyable "
      "for memcpy operations.");

    std::memcpy(&allocation->at(block_index)[offset_in_block], &value, sizeof(T));
  }

  /**
   * @brief Get the value at the current position in the allocation as a different type.
   *
   * @tparam S The type to which to cast the value.
   * @param[in] allocation The allocation.
   */
  template <typename S, typename Ptr>
  [[nodiscard]] S get_current_as(Ptr const& allocation) const
  {
    D_ASSERT(block_index < num_blocks);
    D_ASSERT(allocation != nullptr);
    D_ASSERT(offset_in_block + sizeof(S) <= block_size);
    static_assert(
      std::is_trivially_copyable_v<S>,
      "[multiple_blocks_allocation_accessor] The underlying type must be trivially copyable "
      "for memcpy operations.");
    S s;
    std::memcpy(&s, &allocation->at(block_index)[offset_in_block], sizeof(S));
    return s;
  }

  /**
   * @brief Get the value at the current position in the allocation using the underlying type.
   *
   * @param[in] allocation The allocation.
   */
  template <typename Ptr>
  [[nodiscard]] T get_current(Ptr const& allocation) const
  {
    return get_current_as<underlying_type>(allocation);
  }

  /**
   * @brief Get the value at a specific offset position from the initial offset.
   */
  template <typename Ptr>
  [[nodiscard]] T get(size_t offset, Ptr const& allocation) const
  {
    size_t global_offset        = initial_byte_offset + sizeof(T) * offset;
    size_t temp_block_index     = global_offset / allocation->block_size();
    size_t temp_offset_in_block = global_offset % allocation->block_size();

    assert(temp_block_index < num_blocks);
    assert(allocation != nullptr);
    assert(temp_offset_in_block + sizeof(T) <= block_size);
    static_assert(
      std::is_trivially_copyable_v<T>,
      "[multiple_blocks_allocation_accessor] The underlying type must be trivially copyable "
      "for memcpy operations.");
    T value;
    std::memcpy(&value, &allocation->at(temp_block_index)[temp_offset_in_block], sizeof(T));
    return value;
  }

  /**
   * @brief Advance the cursor into the allocation to the next position as type S.
   *
   * @tparam S The type size to use for advancing the cursor.
   */
  template <typename S>
  void advance_as()
  {
    offset_in_block += sizeof(S);
    block_index += offset_in_block / block_size;
    offset_in_block %= block_size;
  }

  /**
   * @brief Advance the cursor into the allocation to the next position using the underlying type.
   */
  void advance() { advance_as<underlying_type>(); }

  /**
   * @brief Set a number of consecutive values starting at the current position in the allocation.
   *
   * @param[in] val The value to set.
   * @param[in] bytes The number of consecutive values to set.
   * @param[in, out] allocation The allocation.
   */
  template <typename Ptr>
  void memset(uint8_t val, size_t bytes, Ptr& allocation)
  {
    size_t bytes_set = 0;
    while (bytes_set < bytes) {
      assert(block_index < allocation->get_blocks().size());
      // Do as much of a bulk set as possible in the current block
      auto const bytes_to_set =
        std::min(bytes - bytes_set, allocation->block_size() - offset_in_block);
      std::memset(&allocation->at(block_index)[offset_in_block], val, bytes_to_set);
      bytes_set += bytes_to_set;
      offset_in_block += bytes_to_set;
      // Check if we need to advance to the next block
      if (offset_in_block == allocation->block_size()) {
        ++block_index;
        offset_in_block = 0;
      }
    }
  }

  /**
   * @brief Copy from a given source buffer into the allocation starting at the current position.
   *
   * @param[in] src Pointer to the source buffer.
   * @param[in] bytes Number of bytes to copy from the source buffer.
   * @param[in, out] allocation The allocation.
   */
  template <typename Ptr>
  void memcpy_from(void const* src, size_t bytes, Ptr& allocation)
  {
    size_t bytes_copied = 0;
    // Loop over blocks into which to copy the src
    while (bytes_copied < bytes) {
      assert(block_index < allocation->get_blocks().size());
      // Do as much of a bulk copy as possible in the current block
      auto const bytes_to_copy =
        std::min(bytes - bytes_copied, allocation->block_size() - offset_in_block);
      std::memcpy(&allocation->at(block_index)[offset_in_block],
                  static_cast<uint8_t const*>(src) + bytes_copied,
                  bytes_to_copy);
      bytes_copied += bytes_to_copy;
      offset_in_block += bytes_to_copy;
      // Check if we need to advance to the next block
      if (offset_in_block == allocation->block_size()) {
        ++block_index;
        offset_in_block = 0;
      }
    }
  }

  /**
   * @brief Copy the data from the allocation to a destination buffer.
   *
   * @param[in] allocation The allocation.
   * @param[in] dest Pointer to the destination buffer.
   * @param[in] bytes Number of bytes to copy to the destination buffer.
   */
  template <typename Ptr>
  void memcpy_to(Ptr const& allocation, void* dest, size_t bytes)
  {
    size_t bytes_copied = 0;
    // Loop over blocks from which to copy the data
    while (bytes_copied < bytes) {
      assert(block_index < allocation->get_blocks().size());
      // Do as much of a bulk copy as possible in the current block
      auto const bytes_to_copy =
        std::min(bytes - bytes_copied, allocation->block_size() - offset_in_block);
      std::memcpy(static_cast<uint8_t*>(dest) + bytes_copied,
                  &allocation->at(block_index)[offset_in_block],
                  bytes_to_copy);
      bytes_copied += bytes_to_copy;
      offset_in_block += bytes_to_copy;
      // Check if we need to advance to the next block
      if (offset_in_block == allocation->block_size()) {
        ++block_index;
        offset_in_block = 0;
      }
    }
  }
};

}  // namespace sirius::memory
