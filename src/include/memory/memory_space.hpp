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
#include <rmm/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

namespace sirius {
namespace memory {

// Forward declaration
struct Reservation;

/**
 * MemorySpace represents a specific memory location identified by a tier and device ID.
 * It manages memory reservations within that space and owns allocator resources.
 * 
 * Each MemorySpace:
 * - Has a fixed memory limit
 * - Tracks active reservations
 * - Provides thread-safe reservation management
 * - Owns one or more RMM memory allocators
 */
class MemorySpace {
public:
    /**
     * Construct a MemorySpace with the given parameters.
     * 
     * @param tier The memory tier (GPU, HOST, DISK)
     * @param device_id The device identifier within the tier
     * @param memory_limit Maximum memory capacity in bytes
     * @param allocators Vector of RMM memory allocators (must be non-empty)
     */
    MemorySpace(Tier tier, size_t device_id, size_t memory_limit, 
                std::vector<std::unique_ptr<rmm::device_memory_resource>> allocators);
    
    // Disable copy/move to ensure stable addresses for reservations
    MemorySpace(const MemorySpace&) = delete;
    MemorySpace& operator=(const MemorySpace&) = delete;
    MemorySpace(MemorySpace&&) = delete;
    MemorySpace& operator=(MemorySpace&&) = delete;
    
    ~MemorySpace() = default;

    // Comparison operators
    bool operator==(const MemorySpace& other) const;
    bool operator!=(const MemorySpace& other) const;

    // Basic properties
    Tier getTier() const;
    size_t getDeviceId() const;
    
    // Reservation management - these are the core methods that do the actual work
    std::unique_ptr<Reservation> requestReservation(size_t size);
    void releaseReservation(std::unique_ptr<Reservation> reservation);
    
    bool shrinkReservation(Reservation* reservation, size_t new_size);
    bool growReservation(Reservation* reservation, size_t new_size);
    
    // State queries
    size_t getAvailableMemory() const;
    size_t getTotalReservedMemory() const;
    size_t getMaxMemory() const;
    size_t getActiveReservationCount() const;
    
    // Allocator management
    rmm::device_async_resource_ref getDefaultAllocator() const;
    rmm::device_async_resource_ref getAllocator(size_t index) const;
    size_t getAllocatorCount() const;
    
    // Utility methods
    bool canReserve(size_t size) const;
    std::string toString() const;

private:
    const Tier tier_;
    const size_t device_id_;
    const size_t memory_limit_;
    
    // Memory resources owned by this MemorySpace
    std::vector<std::unique_ptr<rmm::device_memory_resource>> allocators_;
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    std::atomic<size_t> total_reserved_{0};
    std::atomic<size_t> active_count_{0};
    
    void waitForMemory(size_t size, std::unique_lock<std::mutex>& lock);
    bool validateReservation(const Reservation* reservation) const;
};

/**
 * Hash function for MemorySpace to enable use in unordered containers.
 * Hash is based on tier and device_id combination.
 */
struct MemorySpaceHash {
    size_t operator()(const MemorySpace& ms) const;
};

} // namespace memory
} // namespace sirius
