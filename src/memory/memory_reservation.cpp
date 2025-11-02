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

#include "memory/memory_reservation.hpp"
#include <algorithm>
#include <stdexcept>
#include <mutex>
#include <cstdint>

namespace sirius {
namespace memory {

//===----------------------------------------------------------------------===//
// reservation Implementation  
//===----------------------------------------------------------------------===//

reservation::reservation(Tier t, size_t dev_id, size_t s) : tier(t), device_id(dev_id), size(s) {}


bool reservation::grow_to(size_t new_size) {
    if (new_size <= size) {
        return false; // Invalid operation - must grow to larger size
    }
    
    auto& manager = memory_reservation_manager::get_instance();
    return manager.grow_reservation(this, new_size);
}

bool reservation::grow_by(size_t additional_bytes) {
    if (additional_bytes == 0) {
        return true; // No change needed
    }
    
    // Check for overflow
    if (size > SIZE_MAX - additional_bytes) {
        return false;
    }
    
    return grow_to(size + additional_bytes);
}

bool reservation::shrink_to(size_t new_size) {
    if (new_size >= size) {
        return false; // Invalid operation - must shrink to smaller size
    }
    
    auto& manager = memory_reservation_manager::get_instance();
    return manager.shrink_reservation(this, new_size);
}

bool reservation::shrink_by(size_t bytes_to_remove) {
    if (bytes_to_remove == 0) {
        return true; // No change needed
    }
    
    if (bytes_to_remove >= size) {
        return false; // Cannot shrink by more than current size
    }
    
    return shrink_to(size - bytes_to_remove);
}

//===----------------------------------------------------------------------===//
// Reservation Limit Policy Implementations
//===----------------------------------------------------------------------===//

void fail_reservation_limit_policy::handle_over_reservation(
    rmm::cuda_stream_view stream,
    std::size_t current_allocated,
    std::size_t requested_bytes,
    reservation* reservation) {
    
    std::size_t reservation_size = reservation ? reservation->size : 0;
    RMM_FAIL("Allocation of " + std::to_string(requested_bytes) + 
             " bytes would exceed stream reservation of " + std::to_string(reservation_size) +
             " bytes (current: " + std::to_string(current_allocated) + " bytes)",
             rmm::out_of_memory);
}

void increase_reservation_limit_policy::handle_over_reservation(
    rmm::cuda_stream_view stream,
    std::size_t current_allocated,
    std::size_t requested_bytes,
    reservation* reservation) {
    
    if (!reservation) {
        RMM_FAIL("No reservation set for stream", rmm::out_of_memory);
    }
    
    // Calculate how much we need
    std::size_t needed_size = current_allocated + requested_bytes;
    
    // Add padding to avoid frequent increases
    std::size_t new_reservation_size = static_cast<std::size_t>(needed_size * _padding_factor);
    
    // Try to grow the reservation
    if (!reservation->grow_to(new_reservation_size)) {
        // If we can't grow to the padded size, try to grow to just what we need
        if (!reservation->grow_to(needed_size)) {
            // If we can't even grow to what we need, throw an error
            RMM_FAIL("Failed to increase stream reservation from " + std::to_string(reservation->size) +
                     " to " + std::to_string(needed_size) + " bytes", rmm::out_of_memory);
        }
    }
}

//===----------------------------------------------------------------------===//
// memory_reservation_manager Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<memory_reservation_manager> memory_reservation_manager::_instance = nullptr;
std::once_flag memory_reservation_manager::_initialized;
bool memory_reservation_manager::_allow_reinitialize_for_tests{false};

memory_reservation_manager::memory_reservation_manager(std::vector<memory_space_config> configs) {
    if (configs.empty()) {
        throw std::invalid_argument("At least one memory_space configuration must be provided");
    }
    
    // Create memory_space instances
    for (auto& config : configs) {
        // Move the allocators from config to the memory_space
        auto mem_space = std::make_unique<memory_space>(
            config.tier, 
            config.device_id, 
            config.memory_limit,
            std::move(config.allocators)
        );
        _memory_spaces.push_back(std::move(mem_space));
    }
    
    // Build lookup tables
    build_lookup_tables();
}

memory_reservation_manager::memory_space_config::memory_space_config(
    Tier t, size_t dev_id, size_t mem_limit, 
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocs)
    : tier(t), device_id(dev_id), memory_limit(mem_limit), allocators(std::move(allocs)) {
    if (allocators.empty()) {
        throw std::invalid_argument("At least one allocator must be provided");
    }
}

void memory_reservation_manager::initialize(std::vector<memory_space_config> configs) {
    // Test hook: if a test called reset_for_testing(), allow reinitialization bypassing call_once
    if (_allow_reinitialize_for_tests) {
        _allow_reinitialize_for_tests = false;
        _instance = std::unique_ptr<memory_reservation_manager>(new memory_reservation_manager(std::move(configs)));
        return;
    }
    std::call_once(_initialized, [configs = std::move(configs)]() mutable {
        _instance = std::unique_ptr<memory_reservation_manager>(new memory_reservation_manager(std::move(configs)));
    });
}

void memory_reservation_manager::reset_for_testing() {
    // Not thread-safe; intended for unit tests only
    _instance.reset();
    _allow_reinitialize_for_tests = true;
}

memory_reservation_manager& memory_reservation_manager::get_instance() {
    if (!_instance) {
        throw std::runtime_error("memory_reservation_manager not initialized. Call initialize() first.");
    }
    return *_instance;
}

std::unique_ptr<reservation> memory_reservation_manager::request_reservation(const reservation_request& request, size_t size) {
    // Fast path: try to make a reservation immediately
    if (auto res = select_memory_space_and_make_reservation(request, size); res.has_value()) {
        return std::move(res.value());
    }

    // If none available, block until any memory_space can satisfy the request
    std::unique_lock<std::mutex> lock(_wait_mutex);
    for (;;) {
        if (auto res = select_memory_space_and_make_reservation(request, size); res.has_value()) {
            // Release the wait lock before returning the reservation
            lock.unlock();
            return std::move(res.value());
        }
        // Wait until notified that memory may be available again
        _wait_cv.wait(lock);
    }
}


void memory_reservation_manager::release_reservation(std::unique_ptr<reservation> reservation) {
    if (!reservation) {
        return;
    }
    
    // Look up the appropriate memory_space
    const memory_space* mem_space = get_memory_space(reservation->tier, reservation->device_id);
    if (!mem_space) {
        throw std::invalid_argument("Invalid tier/device_id in reservation");
    }
    
    // Delegate to the appropriate memory_space
    const_cast<memory_space*>(mem_space)->release_reservation(std::move(reservation));

    // Notify all waiters that memory availability may have changed
    _wait_cv.notify_all();
}

bool memory_reservation_manager::shrink_reservation(reservation* reservation, size_t new_size) {
    if (!reservation) {
        return false;
    }
    
    // Look up the appropriate memory_space
    const memory_space* mem_space = get_memory_space(reservation->tier, reservation->device_id);
    if (!mem_space) {
        return false;
    }
    
    // Delegate to the appropriate memory_space
    return const_cast<memory_space*>(mem_space)->shrink_reservation(reservation, new_size);
}

bool memory_reservation_manager::grow_reservation(reservation* reservation, size_t new_size) {
    if (!reservation) {
        return false;
    }
    
    // Look up the appropriate memory_space
    const memory_space* mem_space = get_memory_space(reservation->tier, reservation->device_id);
    if (!mem_space) {
        return false;
    }
    
    // Delegate to the appropriate memory_space
    return const_cast<memory_space*>(mem_space)->grow_reservation(reservation, new_size);
}

const memory_space* memory_reservation_manager::get_memory_space(Tier tier, size_t device_id) const {
    auto key = std::make_pair(tier, device_id);
    auto it = _memory_space_lookup.find(key);
    return (it != _memory_space_lookup.end()) ? it->second : nullptr;
}

std::vector<const memory_space*> memory_reservation_manager::get_memory_spaces_for_tier(Tier tier) const {
    auto it = _tier_to_memory_spaces.find(tier);
    return (it != _tier_to_memory_spaces.end()) ? it->second : std::vector<const memory_space*>{};
}

std::vector<const memory_space*> memory_reservation_manager::get_all_memory_spaces() const {
    std::vector<const memory_space*> result;
    result.reserve(_memory_spaces.size());
    
    for (const auto& ms : _memory_spaces) {
        result.push_back(ms.get());
    }
    
    return result;
}

size_t memory_reservation_manager::get_available_memory_for_tier(Tier tier) const {
    size_t total_available = 0;
    auto spaces = get_memory_spaces_for_tier(tier);
    
    for (const auto* space : spaces) {
        total_available += space->get_available_memory();
    }
    
    return total_available;
}

size_t memory_reservation_manager::get_total_reserved_memory_for_tier(Tier tier) const {
    size_t total_reserved = 0;
    auto spaces = get_memory_spaces_for_tier(tier);
    
    for (const auto* space : spaces) {
        total_reserved += space->get_total_reserved_memory();
    }
    
    return total_reserved;
}

size_t memory_reservation_manager::get_active_reservation_count_for_tier(Tier tier) const {
    size_t total_count = 0;
    auto spaces = get_memory_spaces_for_tier(tier);
    
    for (const auto* space : spaces) {
        total_count += space->get_active_reservation_count();
    }
    
    return total_count;
}

size_t memory_reservation_manager::get_total_available_memory() const {
    size_t total = 0;
    for (const auto& space : _memory_spaces) {
        total += space->get_available_memory();
    }
    return total;
}

size_t memory_reservation_manager::get_total_reserved_memory() const {
    size_t total = 0;
    for (const auto& space : _memory_spaces) {
        total += space->get_total_reserved_memory();
    }
    return total;
}

size_t memory_reservation_manager::get_active_reservation_count() const {
    size_t total = 0;
    for (const auto& space : _memory_spaces) {
        total += space->get_active_reservation_count();
    }
    return total;
}

std::optional<std::unique_ptr<reservation>> memory_reservation_manager::select_memory_space_and_make_reservation(const reservation_request& request, size_t size) const {
    auto try_candidates = [this, size](const std::vector<const memory_space*>& candidates) -> std::optional<std::unique_ptr<reservation>> {
        for (const memory_space* space : candidates) {
            if (space && space->can_reserve(size)) {
                // Delegate to memory_space to create the reservation
                return const_cast<memory_space*>(space)->request_reservation(size);
            }
        }
        return std::nullopt;
    };

    return std::visit([this, size, &try_candidates](const auto& req) -> std::optional<std::unique_ptr<reservation>> {
        using T = std::decay_t<decltype(req)>;

        if constexpr (std::is_same_v<T, any_memory_space_in_tier_with_preference>) {
            auto candidates = get_memory_spaces_for_tier(req.tier);

            // If a preferred device is specified, try it first
            if (req.preferred_device_id.has_value()) {
                for (const memory_space* space : candidates) {
                    //TODO: we need can_reserve and request_reservation to happen in one operation so we can lock and prevent
                    //race conditions. because we can wait on the memory_space itself it will work for now but 
                    //we should change that behavior at some point
                    if (space && space->get_device_id() == req.preferred_device_id.value() && space->can_reserve(size)) {
                        return const_cast<memory_space*>(space)->request_reservation(size);
                    }
                }
            }

            // Fall back to any space in the tier
            return try_candidates(candidates);
        } else if constexpr (std::is_same_v<T, any_memory_space_in_tier>) {
            auto candidates = get_memory_spaces_for_tier(req.tier);
            return try_candidates(candidates);
        } else if constexpr (std::is_same_v<T, any_memory_space_in_tiers>) {
            for (Tier tier : req.tiers) {
                auto candidates = get_memory_spaces_for_tier(tier);
                if (auto res = try_candidates(candidates); res.has_value()) {
                    return res;
                }
            }
            return std::nullopt;
        } else {
            static_assert(!sizeof(T*), "Unhandled reservation_request type");
        }
    }, request);
}

void memory_reservation_manager::build_lookup_tables() {
    _memory_space_lookup.clear();
    _tier_to_memory_spaces.clear();
    
    for (const auto& space : _memory_spaces) {
        const memory_space* space_ptr = space.get();
        
        // Build direct lookup table
        auto key = std::make_pair(space_ptr->get_tier(), space_ptr->get_device_id());
        _memory_space_lookup[key] = space_ptr;
        
        // Build tier-to-spaces mapping
        _tier_to_memory_spaces[space_ptr->get_tier()].push_back(space_ptr);
    }
}

} // namespace memory
} // namespace sirius

