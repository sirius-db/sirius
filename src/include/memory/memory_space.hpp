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
#include "memory/disk_access_limiter.hpp"
#include "memory/notification_channel.hpp"

#include <concepts>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <variant>

// RMM includes for memory resource management
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace cucascade {
namespace memory {

// Forward declaration
struct reservation;
struct reservation_aware_resource_adaptor;
struct fixed_size_host_memory_resource;

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
   * @param tier The memory tier (GPU, HOST, DISK)
   * @param device_id The device identifier within the tier
   * @param memory_limit Maximum memory capacity in bytes
   * @param capacity Total memory capacity in bytes (optional [default: device capacity])
   * @param allocator RMM memory allocator (must be non-empty)
   */
  memory_space(Tier tier,
               int device_id,
               size_t memory_limit,
               size_t start_downgrading_memory_threshold,
               size_t stop_downgrading_memory_threshold,
               size_t capacity,
               std::unique_ptr<rmm::mr::device_memory_resource> allocator);

  // Disable copy/move to ensure stable addresses for reservations
  memory_space(const memory_space&)            = delete;
  memory_space& operator=(const memory_space&) = delete;
  memory_space(memory_space&&)                 = delete;
  memory_space& operator=(memory_space&&)      = delete;

  ~memory_space();

  // Comparison operators
  bool operator==(const memory_space& other) const;
  bool operator!=(const memory_space& other) const;

  // Basic properties
  memory_space_id get_id() const noexcept;
  Tier get_tier() const noexcept;
  int get_device_id() const noexcept;

  // Reservation management - these are the core methods that do the actual work
  std::unique_ptr<reservation> make_reservation_or_null(size_t size);
  std::unique_ptr<reservation> make_reservation_upto(size_t size);
  std::unique_ptr<reservation> make_reservation(size_t size);
  rmm::cuda_stream_view acquire_stream() const;

  std::size_t get_active_reservation_count() const;
  bool should_downgrade_memory() const;
  bool should_stop_downgrading_memory() const;
  size_t get_amount_to_downgrade() const;

  // State queries
  size_t get_available_memory(rmm::cuda_stream_view stream) const;
  size_t get_available_memory() const;
  size_t get_total_reserved_memory() const;
  size_t get_max_memory() const noexcept;

  // Allocator management
  rmm::mr::device_memory_resource* get_default_allocator() const noexcept;

  template <typename T>
    requires std::derived_from<T, rmm::mr::device_memory_resource>
  T* get_memory_resource_as() const noexcept
  {
    return dynamic_cast<T*>(get_default_allocator());
  }

  template <Tier TIER>
  auto* get_memory_resource_of() const noexcept
  {
    return get_memory_resource_as<typename tier_memory_resource_trait<TIER>::type>();
  }

  // Utility methods
  std::string to_string() const;

  void shutdown();

 protected:
  friend struct reservation;

  const memory_space_id _id;
  const size_t _memory_limit;
  const size_t _start_downgrading_memory_threshold;
  const size_t _stop_downgrading_memory_threshold;
  const size_t _capacity;
  using reserving_adaptor_type = std::variant<std::unique_ptr<reservation_aware_resource_adaptor>,
                                              std::unique_ptr<fixed_size_host_memory_resource>,
                                              std::unique_ptr<disk_access_limiter>>;

  std::shared_ptr<notification_channel> notification_channel_;

  // Memory resources owned by this memory_space
  std::unique_ptr<rmm::mr::device_memory_resource> _allocator;
  reserving_adaptor_type _reservation_allocator;
  std::unique_ptr<rmm::cuda_stream_pool> stream_pool_;
};

/**
 * Hash function for memory_space to enable use in unordered containers.
 * Hash is based on tier and device_id combination.
 */
struct memory_space_hash {
  size_t operator()(const memory_space& ms) const;
};

}  // namespace memory
}  // namespace cucascade
