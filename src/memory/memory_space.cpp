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

#include "memory/memory_space.hpp"
#include "memory/memory_reservation.hpp"  // For Reservation struct
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace sirius {
namespace memory {

//===----------------------------------------------------------------------===//
// MemorySpace Implementation
//===----------------------------------------------------------------------===//

MemorySpace::MemorySpace(Tier tier, size_t device_id, size_t memory_limit, 
                         std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators)
    : tier_(tier), device_id_(device_id), memory_limit_(memory_limit), allocators_(std::move(allocators)) {
    if (memory_limit == 0) {
        throw std::invalid_argument("Memory limit must be greater than 0");
    }
    if (allocators_.empty()) {
        throw std::invalid_argument("At least one allocator must be provided");
    }
}

bool MemorySpace::operator==(const MemorySpace& other) const {
    return tier_ == other.tier_ && device_id_ == other.device_id_;
}

bool MemorySpace::operator!=(const MemorySpace& other) const {
    return !(*this == other);
}

Tier MemorySpace::getTier() const {
    return tier_;
}

size_t MemorySpace::getDeviceId() const {
    return device_id_;
}

std::unique_ptr<Reservation> MemorySpace::requestReservation(size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    //TODO: This is kind of wrong. Given that we are trying to handle the blocking 
    //on the memory reservation manager. For now  I am going to leave it but
    //we should probably and some locking mechanism for seeing if there is space AND returning the
    //reservation if there is space in one operation.
    // Wait until we can allocate the requested size
    waitForMemory(size, lock);
    
    // Create the reservation
    auto reservation = std::make_unique<Reservation>(tier_, device_id_, size);
    
    // Update tracking
    total_reserved_.fetch_add(size);
    active_count_.fetch_add(1);
    
    return reservation;
}

void MemorySpace::releaseReservation(std::unique_ptr<Reservation> reservation) {
    if (!reservation) {
        return;
    }
    
    if (!validateReservation(reservation.get())) {
        throw std::invalid_argument("Reservation does not belong to this MemorySpace");
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update tracking
    total_reserved_.fetch_sub(reservation->size);
    active_count_.fetch_sub(1);
    
    // Notify waiting threads
    cv_.notify_all();
}

bool MemorySpace::shrinkReservation(Reservation* reservation, size_t new_size) {
    if (!reservation || new_size >= reservation->size) {
        return false; // Invalid operation
    }
    
    if (!validateReservation(reservation)) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t size_diff = reservation->size - new_size;
    
    // Update reservation size
    reservation->size = new_size;
    
    // Update tracking
    total_reserved_.fetch_sub(size_diff);
    
    // Notify waiting threads
    cv_.notify_all();
    
    return true;
}

bool MemorySpace::growReservation(Reservation* reservation, size_t new_size) {
    if (!reservation || new_size <= reservation->size) {
        return false; // Invalid operation
    }
    
    if (!validateReservation(reservation)) {
        return false;
    }
    
    size_t size_diff = new_size - reservation->size;
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Check if we can grow
    if (!canReserve(size_diff)) {
        return false; // Not enough memory available
    }
    
    // Update reservation size
    reservation->size = new_size;
    
    // Update tracking
    total_reserved_.fetch_add(size_diff);
    
    return true;
}

size_t MemorySpace::getAvailableMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t reserved = total_reserved_.load();
    return (reserved >= memory_limit_) ? 0 : (memory_limit_ - reserved);
}

size_t MemorySpace::getTotalReservedMemory() const {
    return total_reserved_.load();
}

size_t MemorySpace::getMaxMemory() const {
    return memory_limit_;
}

size_t MemorySpace::getActiveReservationCount() const {
    return active_count_.load();
}

rmm::device_async_resource_ref MemorySpace::getDefaultAllocator() const {
    if (allocators_.empty()) {
        throw std::runtime_error("No allocators available in MemorySpace");
    }
    return *allocators_[0];
}

rmm::device_async_resource_ref MemorySpace::getAllocator(size_t index) const {
    if (index >= allocators_.size()) {
        throw std::out_of_range("Allocator index out of range");
    }
    return *allocators_[index];
}

size_t MemorySpace::getAllocatorCount() const {
    return allocators_.size();
}

bool MemorySpace::canReserve(size_t size) const {
    size_t current_reserved = total_reserved_.load();
    size_t current_active = active_count_.load();
    // Allow a single initial reservation to exceed the memory limit if there are
    // currently zero outstanding reservations. Subsequent reservations must obey the limit.
    if (current_active == 0) {
        return true;
    }
    return (current_reserved + size) <= memory_limit_;
}

std::string MemorySpace::toString() const {
    std::ostringstream oss;
    oss << "MemorySpace(tier=";
    switch (tier_) {
        case Tier::GPU: oss << "GPU"; break;
        case Tier::HOST: oss << "HOST"; break;
        case Tier::DISK: oss << "DISK"; break;
        default: oss << "UNKNOWN"; break;
    }
    oss << ", device_id=" << device_id_ << ", limit=" << memory_limit_ << ")";
    return oss.str();
}

void MemorySpace::waitForMemory(size_t size, std::unique_lock<std::mutex>& lock) {
    while (!canReserve(size)) {
        cv_.wait(lock);
    }
}

bool MemorySpace::validateReservation(const Reservation* reservation) const {
    return reservation && reservation->tier == tier_ && reservation->device_id == device_id_;
}

//===----------------------------------------------------------------------===//
// MemorySpaceHash Implementation
//===----------------------------------------------------------------------===//

size_t MemorySpaceHash::operator()(const MemorySpace& ms) const {
    return std::hash<int>{}(static_cast<int>(ms.getTier())) ^ 
           (std::hash<size_t>{}(ms.getDeviceId()) << 1);
}

} // namespace memory
} // namespace sirius
