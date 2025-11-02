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
// memory_space Implementation
//===----------------------------------------------------------------------===//

memory_space::memory_space(Tier tier, size_t device_id, size_t memory_limit, 
                         std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators)
    : _tier(tier), _device_id(device_id), _memory_limit(memory_limit), _allocators(std::move(allocators)) {
    if (memory_limit == 0) {
        throw std::invalid_argument("Memory limit must be greater than 0");
    }
    if (_allocators.empty()) {
        throw std::invalid_argument("At least one allocator must be provided");
    }
}

bool memory_space::operator==(const memory_space& other) const {
    return _tier == other._tier && _device_id == other._device_id;
}

bool memory_space::operator!=(const memory_space& other) const {
    return !(*this == other);
}

Tier memory_space::get_tier() const {
    return _tier;
}

size_t memory_space::get_device_id() const {
    return _device_id;
}

std::unique_ptr<reservation> memory_space::request_reservation(size_t size) {
    std::unique_lock<std::mutex> lock(_mutex);
    
    //TODO: This is kind of wrong. Given that we are trying to handle the blocking 
    //on the memory reservation manager. For now  I am going to leave it but
    //we should probably and some locking mechanism for seeing if there is space AND returning the
    //reservation if there is space in one operation.
    // Wait until we can allocate the requested size
    wait_for_memory(size, lock);
    
    // Create the reservation
    auto res = std::make_unique<reservation>(_tier, _device_id, size);
    
    // Update tracking
    _total_reserved.fetch_add(size);
    _active_count.fetch_add(1);
    
    return res;
}

void memory_space::release_reservation(std::unique_ptr<reservation> res) {
    if (!res) {
        return;
    }

    if (!validate_reservation(res.get())) {
        throw std::invalid_argument("Reservation does not belong to this memory_space");
    }
    
    std::lock_guard<std::mutex> lock(_mutex);
    
    // Update tracking
    _total_reserved.fetch_sub(res->size);
    _active_count.fetch_sub(1);
    
    // Notify waiting threads
    _cv.notify_all();
}

bool memory_space::shrink_reservation(reservation* res, size_t new_size) {
    if (!res || new_size >= res->size) {
        return false; // Invalid operation
    }
    
    if (!validate_reservation(res)) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(_mutex);
    
    size_t size_diff = res->size - new_size;
    
    // Update reservation size
    res->size = new_size;
    
    // Update tracking
    _total_reserved.fetch_sub(size_diff);
    
    // Notify waiting threads
    _cv.notify_all();
    
    return true;
}

bool memory_space::grow_reservation(reservation* res, size_t new_size) {
    if (!res || new_size <= res->size) {
        return false; // Invalid operation
    }

    if (!validate_reservation(res)) {
        return false;
    }

    size_t size_diff = new_size - res->size;

    std::unique_lock<std::mutex> lock(_mutex);
    
    // Check if we can grow
    if (!can_reserve(size_diff)) {
        return false; // Not enough memory available
    }
    
    // Update reservation size
    res->size = new_size;

    // Update tracking
    _total_reserved.fetch_add(size_diff);
    
    return true;
}

size_t memory_space::get_available_memory() const {
    std::lock_guard<std::mutex> lock(_mutex);
    size_t reserved = _total_reserved.load();
    return (reserved >= _memory_limit) ? 0 : (_memory_limit - reserved);
}

size_t memory_space::get_total_reserved_memory() const {
    return _total_reserved.load();
}

size_t memory_space::get_max_memory() const {
    return _memory_limit;
}

size_t memory_space::get_active_reservation_count() const {
    return _active_count.load();
}

rmm::device_async_resource_ref memory_space::get_default_allocator() const {
    if (_allocators.empty()) {
        throw std::runtime_error("No allocators available in memory_space");
    }
    return *_allocators[0];
}

rmm::device_async_resource_ref memory_space::get_allocator(size_t index) const {
    if (index >= _allocators.size()) {
        throw std::out_of_range("Allocator index out of range");
    }
    return *_allocators[index];
}

size_t memory_space::get_allocator_count() const {
    return _allocators.size();
}

bool memory_space::can_reserve(size_t size) const {
    size_t current_reserved = _total_reserved.load();
    size_t current_active = _active_count.load();
    // Allow a single initial reservation to exceed the memory limit if there are
    // currently zero outstanding reservations. Subsequent reservations must obey the limit.
    if (current_active == 0) {
        return true;
    }
    return (current_reserved + size) <= _memory_limit;
}

std::string memory_space::to_string() const {
    std::ostringstream oss;
    oss << "memory_space(tier=";
    switch (_tier) {
        case Tier::GPU: oss << "GPU"; break;
        case Tier::HOST: oss << "HOST"; break;
        case Tier::DISK: oss << "DISK"; break;
        default: oss << "UNKNOWN"; break;
    }
    oss << ", device_id=" << _device_id << ", limit=" << _memory_limit << ")";
    return oss.str();
}

void memory_space::wait_for_memory(size_t size, std::unique_lock<std::mutex>& lock) {
    while (!can_reserve(size)) {
        _cv.wait(lock);
    }
}

bool memory_space::validate_reservation(const reservation* res) const {
    return res && res->tier == _tier && res->device_id == _device_id;
}

//===----------------------------------------------------------------------===//
// memory_space_hash Implementation
//===----------------------------------------------------------------------===//

size_t memory_space_hash::operator()(const memory_space& ms) const {
    return std::hash<int>{}(static_cast<int>(ms.get_tier())) ^ 
           (std::hash<size_t>{}(ms.get_device_id()) << 1);
}

} // namespace memory
} // namespace sirius
