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

#include "memory/common.hpp"
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <vector>
#include <string>

// RMM includes for memory resource management
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/cuda_stream.hpp>

namespace sirius {
namespace memory {

// Forward declaration
struct reservation;

struct memory_space_id {
  Tier tier;
  int device_id;
  memory_space_id(Tier tier, int device_id) : tier(tier), device_id(device_id) {}
  bool operator==(const memory_space_id& other) const
  {
    return tier == other.tier && device_id == other.device_id;
  }
  bool operator!=(const memory_space_id& other) const { return !(*this == other); }
  bool operator<(const memory_space_id& other) const
  {
    return tier < other.tier || (tier == other.tier && device_id < other.device_id);
  }
  bool operator>(const memory_space_id& other) const
  {
    return tier > other.tier || (tier == other.tier && device_id > other.device_id);
  }
  bool operator<=(const memory_space_id& other) const
  {
    return tier < other.tier || (tier == other.tier && device_id <= other.device_id);
  }
  bool operator>=(const memory_space_id& other) const
  {
    return tier > other.tier || (tier == other.tier && device_id >= other.device_id);
  }
};

/**
 * memory_space represents a specific memory location identified by a tier and device ID.
 * It manages memory reservations within that space and owns allocator resources.
 *
 * Each memory_space:
 * - Has a fixed memory limit
 * - Tracks active reservations
 * - Provides thread-safe reservation management
 * - Owns one or more RMM memory allocators
 */
class memory_space {
 public:
  /**
   * Construct a memory_space with the given parameters.
   *
   * @param id The memory_space identifier (tier + device)
   * @param memory_limit Maximum memory capacity in bytes
   * @param allocators Vector of RMM memory allocators (must be non-empty)
   */
  memory_space(memory_space_id id,
               size_t memory_limit,
               size_t start_downgrading_memory_threshold,
               size_t stop_downgrading_memory_threshold,
               std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators);

  // Disable copy/move to ensure stable addresses for reservations
  memory_space(const memory_space&)            = delete;
  memory_space& operator=(const memory_space&) = delete;
  memory_space(memory_space&&)                 = delete;
  memory_space& operator=(memory_space&&)      = delete;

  ~memory_space() = default;

  // Comparison operators
  bool operator==(const memory_space& other) const;
  bool operator!=(const memory_space& other) const;

  // Basic properties
  memory_space_id get_id() const;
  Tier get_tier() const;
  int get_device_id() const;

  // Reservation management - these are the core methods that do the actual work
  std::unique_ptr<reservation> request_reservation(size_t size);
  void release_reservation(std::unique_ptr<reservation> res);

  bool shrink_reservation(reservation* res, size_t new_size);
  bool grow_reservation(reservation* res, size_t new_size);

  // State queries
  size_t get_available_memory() const;
  size_t get_total_reserved_memory() const;
  size_t get_max_memory() const;
  size_t get_active_reservation_count() const;
  bool should_downgrade_memory() const;
  bool should_stop_downgrading_memory() const;
  size_t get_amount_to_downgrade() const;

  // Allocator management
  rmm::device_async_resource_ref get_default_allocator() const;
  rmm::device_async_resource_ref get_allocator(size_t index) const;
  size_t get_allocator_count() const;

  /**
   * @brief Attempt to retrieve the default allocator as a concrete type.
   *
   * Returns nullptr if the default allocator is not of the requested type.
   */
  template <typename T>
  T* get_default_allocator_as() const
  {
    if (_allocators.empty()) { return nullptr; }
    return dynamic_cast<T*>(_allocators[0].get());
  }

  // Stream pool management
  /**
   * @brief Acquire a CUDA stream associated with this memory_space's device.
   *        If all streams are in use, the pool grows.
   */
  rmm::cuda_stream_view acquire_stream() const;
  /**
   * @brief Release a CUDA stream back to the pool.
   */
  void release_stream(rmm::cuda_stream_view stream) const;

  // Utility methods
  bool can_reserve(size_t size) const;
  std::string to_string() const;

 private:
  const memory_space_id _id;
  const size_t _memory_limit;
  const size_t _start_downgrading_memory_threshold;
  const size_t _stop_downgrading_memory_threshold;

  // Memory resources owned by this memory_space
  std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> _allocators;

  mutable std::mutex _mutex;
  std::condition_variable _cv;

  std::atomic<size_t> _total_reserved{0};
  std::atomic<size_t> _active_count{0};

  void wait_for_memory(size_t size, std::unique_lock<std::mutex>& lock);
  bool validate_reservation(const reservation* res) const;

  // Stream pool (GPU-only usage). Mutable to allow acquisition in const context.
  mutable std::mutex _streams_mutex;
  mutable std::vector<std::unique_ptr<rmm::cuda_stream>> _streams;
  mutable std::vector<bool> _stream_in_use;
  void initialize_stream_pool_if_needed() const;
  void grow_stream_pool_unlocked(size_t additional_streams) const;
};

/**
 * Hash function for memory_space to enable use in unordered containers.
 * Hash is based on tier and device_id combination.
 */
struct memory_space_hash {
  size_t operator()(const memory_space& ms) const;
};

}  // namespace memory
}  // namespace sirius
