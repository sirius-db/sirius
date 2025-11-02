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

namespace sirius {
namespace memory {

// Forward declaration
struct reservation;

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
     * @param allocators Vector of RMM memory allocators (must be non-empty)
     */
    memory_space(Tier tier, size_t device_id, size_t memory_limit, 
                std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators);
    
    // Disable copy/move to ensure stable addresses for reservations
    memory_space(const memory_space&) = delete;
    memory_space& operator=(const memory_space&) = delete;
    memory_space(memory_space&&) = delete;
    memory_space& operator=(memory_space&&) = delete;
    
    ~memory_space() = default;

    // Comparison operators
    bool operator==(const memory_space& other) const;
    bool operator!=(const memory_space& other) const;

    // Basic properties
    Tier get_tier() const;
    size_t get_device_id() const;
    
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
    
    // Allocator management
    rmm::device_async_resource_ref get_default_allocator() const;
    rmm::device_async_resource_ref get_allocator(size_t index) const;
    size_t get_allocator_count() const;
    
    // Utility methods
    bool can_reserve(size_t size) const;
    std::string to_string() const;

private:
    const Tier _tier;
    const size_t _device_id;
    const size_t _memory_limit;
    
    // Memory resources owned by this memory_space
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> _allocators;
    
    mutable std::mutex _mutex;
    std::condition_variable _cv;
    
    std::atomic<size_t> _total_reserved{0};
    std::atomic<size_t> _active_count{0};
    
    void wait_for_memory(size_t size, std::unique_lock<std::mutex>& lock);
    bool validate_reservation(const reservation* res) const;
};

/**
 * Hash function for memory_space to enable use in unordered containers.
 * Hash is based on tier and device_id combination.
 */
struct memory_space_hash {
    size_t operator()(const memory_space& ms) const;
};

} // namespace memory
} // namespace sirius
