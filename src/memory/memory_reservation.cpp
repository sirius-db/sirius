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
// Reservation Implementation  
//===----------------------------------------------------------------------===//

Reservation::Reservation(Tier t, size_t dev_id, size_t s) : tier(t), device_id(dev_id), size(s) {}

const MemorySpace* Reservation::getMemorySpace(const MemoryReservationManager& manager) const {
    return manager.getMemorySpace(tier, device_id);
}

bool Reservation::grow_to(size_t new_size) {
    if (new_size <= size) {
        return false; // Invalid operation - must grow to larger size
    }
    
    auto& manager = MemoryReservationManager::getInstance();
    return manager.growReservation(this, new_size);
}

bool Reservation::grow_by(size_t additional_bytes) {
    if (additional_bytes == 0) {
        return true; // No change needed
    }
    
    // Check for overflow
    if (size > SIZE_MAX - additional_bytes) {
        return false;
    }
    
    return grow_to(size + additional_bytes);
}

bool Reservation::shrink_to(size_t new_size) {
    if (new_size >= size) {
        return false; // Invalid operation - must shrink to smaller size
    }
    
    auto& manager = MemoryReservationManager::getInstance();
    return manager.shrinkReservation(this, new_size);
}

bool Reservation::shrink_by(size_t bytes_to_remove) {
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

void FailReservationLimitPolicy::handle_over_reservation(
    per_stream_tracking_resource_adaptor& adaptor,
    rmm::cuda_stream_view stream,
    std::size_t current_allocated,
    std::size_t requested_bytes,
    Reservation* reservation) {
    
    std::size_t reservation_size = reservation ? reservation->size : 0;
    RMM_FAIL("Allocation of " + std::to_string(requested_bytes) + 
             " bytes would exceed stream reservation of " + std::to_string(reservation_size) +
             " bytes (current: " + std::to_string(current_allocated) + " bytes)",
             rmm::out_of_memory);
}

void IncreaseReservationLimitPolicy::handle_over_reservation(
    per_stream_tracking_resource_adaptor& adaptor,
    rmm::cuda_stream_view stream,
    std::size_t current_allocated,
    std::size_t requested_bytes,
    Reservation* reservation) {
    
    if (!reservation) {
        RMM_FAIL("No reservation set for stream", rmm::out_of_memory);
    }
    
    // Calculate how much we need
    std::size_t needed_size = current_allocated + requested_bytes;
    
    // Add padding to avoid frequent increases
    std::size_t new_reservation_size = static_cast<std::size_t>(needed_size * padding_factor_);
    
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
// MemoryReservationManager Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<MemoryReservationManager> MemoryReservationManager::instance_ = nullptr;
std::once_flag MemoryReservationManager::initialized_;

MemoryReservationManager::MemoryReservationManager(std::vector<MemorySpaceConfig> configs) {
    if (configs.empty()) {
        throw std::invalid_argument("At least one MemorySpace configuration must be provided");
    }
    
    // Create MemorySpace instances
    for (auto& config : configs) {
        // Move the allocators from config to the MemorySpace
        auto memory_space = std::make_unique<MemorySpace>(
            config.tier, 
            config.device_id, 
            config.memory_limit,
            std::move(config.allocators)
        );
        memory_spaces_.push_back(std::move(memory_space));
    }
    
    // Build lookup tables
    buildLookupTables();
}

MemoryReservationManager::MemorySpaceConfig::MemorySpaceConfig(
    Tier t, size_t dev_id, size_t mem_limit, 
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocs)
    : tier(t), device_id(dev_id), memory_limit(mem_limit), allocators(std::move(allocs)) {
    if (allocators.empty()) {
        throw std::invalid_argument("At least one allocator must be provided");
    }
}

void MemoryReservationManager::initialize(std::vector<MemorySpaceConfig> configs) {
    std::call_once(initialized_, [configs = std::move(configs)]() mutable {
        instance_ = std::unique_ptr<MemoryReservationManager>(new MemoryReservationManager(std::move(configs)));
    });
}

MemoryReservationManager& MemoryReservationManager::getInstance() {
    if (!instance_) {
        throw std::runtime_error("MemoryReservationManager not initialized. Call initialize() first.");
    }
    return *instance_;
}

std::unique_ptr<Reservation> MemoryReservationManager::requestReservation(const ReservationRequest& request, size_t size) {
    // Fast path: try to find a MemorySpace immediately
    const MemorySpace* selected_space = selectMemorySpace(request, size);
    if (selected_space) {
        return const_cast<MemorySpace*>(selected_space)->requestReservation(size);
    }

    // If none available, block until any MemorySpace can satisfy the request
    std::unique_lock<std::mutex> lock(wait_mutex_);
    for (;;) {
        selected_space = selectMemorySpace(request, size);
        if (selected_space) {
            // Release the wait lock before delegating to the MemorySpace
            lock.unlock();
            return const_cast<MemorySpace*>(selected_space)->requestReservation(size);
        }
        // Wait until notified that memory may be available again
        wait_cv_.wait(lock);
    }
}


void MemoryReservationManager::releaseReservation(std::unique_ptr<Reservation> reservation) {
    if (!reservation) {
        return;
    }
    
    // Look up the appropriate MemorySpace
    const MemorySpace* memory_space = getMemorySpace(reservation->tier, reservation->device_id);
    if (!memory_space) {
        throw std::invalid_argument("Invalid tier/device_id in reservation");
    }
    
    // Delegate to the appropriate MemorySpace
    const_cast<MemorySpace*>(memory_space)->releaseReservation(std::move(reservation));

    // Notify all waiters that memory availability may have changed
    wait_cv_.notify_all();
}

bool MemoryReservationManager::shrinkReservation(Reservation* reservation, size_t new_size) {
    if (!reservation) {
        return false;
    }
    
    // Look up the appropriate MemorySpace
    const MemorySpace* memory_space = getMemorySpace(reservation->tier, reservation->device_id);
    if (!memory_space) {
        return false;
    }
    
    // Delegate to the appropriate MemorySpace
    return const_cast<MemorySpace*>(memory_space)->shrinkReservation(reservation, new_size);
}

bool MemoryReservationManager::growReservation(Reservation* reservation, size_t new_size) {
    if (!reservation) {
        return false;
    }
    
    // Look up the appropriate MemorySpace
    const MemorySpace* memory_space = getMemorySpace(reservation->tier, reservation->device_id);
    if (!memory_space) {
        return false;
    }
    
    // Delegate to the appropriate MemorySpace
    return const_cast<MemorySpace*>(memory_space)->growReservation(reservation, new_size);
}

const MemorySpace* MemoryReservationManager::getMemorySpace(Tier tier, size_t device_id) const {
    auto key = std::make_pair(tier, device_id);
    auto it = memory_space_lookup_.find(key);
    return (it != memory_space_lookup_.end()) ? it->second : nullptr;
}

std::vector<const MemorySpace*> MemoryReservationManager::getMemorySpacesForTier(Tier tier) const {
    auto it = tier_to_memory_spaces_.find(tier);
    return (it != tier_to_memory_spaces_.end()) ? it->second : std::vector<const MemorySpace*>{};
}

std::vector<const MemorySpace*> MemoryReservationManager::getAllMemorySpaces() const {
    std::vector<const MemorySpace*> result;
    result.reserve(memory_spaces_.size());
    
    for (const auto& ms : memory_spaces_) {
        result.push_back(ms.get());
    }
    
    return result;
}

size_t MemoryReservationManager::getAvailableMemoryForTier(Tier tier) const {
    size_t total_available = 0;
    auto spaces = getMemorySpacesForTier(tier);
    
    for (const auto* space : spaces) {
        total_available += space->getAvailableMemory();
    }
    
    return total_available;
}

size_t MemoryReservationManager::getTotalReservedMemoryForTier(Tier tier) const {
    size_t total_reserved = 0;
    auto spaces = getMemorySpacesForTier(tier);
    
    for (const auto* space : spaces) {
        total_reserved += space->getTotalReservedMemory();
    }
    
    return total_reserved;
}

size_t MemoryReservationManager::getActiveReservationCountForTier(Tier tier) const {
    size_t total_count = 0;
    auto spaces = getMemorySpacesForTier(tier);
    
    for (const auto* space : spaces) {
        total_count += space->getActiveReservationCount();
    }
    
    return total_count;
}

size_t MemoryReservationManager::getTotalAvailableMemory() const {
    size_t total = 0;
    for (const auto& space : memory_spaces_) {
        total += space->getAvailableMemory();
    }
    return total;
}

size_t MemoryReservationManager::getTotalReservedMemory() const {
    size_t total = 0;
    for (const auto& space : memory_spaces_) {
        total += space->getTotalReservedMemory();
    }
    return total;
}

size_t MemoryReservationManager::getActiveReservationCount() const {
    size_t total = 0;
    for (const auto& space : memory_spaces_) {
        total += space->getActiveReservationCount();
    }
    return total;
}

const MemorySpace* MemoryReservationManager::selectMemorySpace(const ReservationRequest& request, size_t size) const {
    return std::visit([this, size](const auto& req) -> const MemorySpace* {
        using T = std::decay_t<decltype(req)>;
        
        if constexpr (std::is_same_v<T, AnyMemorySpaceInTierWithPreference>) {
            // Find space in the specified tier, preferring the specified device if available
            auto candidates = getMemorySpacesForTier(req.tier);
            
            // If a preferred device is specified, try it first
            if (req.preferred_device_id.has_value()) {
                for (const MemorySpace* space : candidates) {
                    if (space && space->getDeviceId() == req.preferred_device_id.value() && space->canReserve(size)) {
                        return space;
                    }
                }
            }
            
            // Fall back to any space in the tier
            return selectFromList(candidates, size);
        }
        else if constexpr (std::is_same_v<T, AnyMemorySpaceInTier>) {
            // Find any space in the specified tier that can handle the request
            auto candidates = getMemorySpacesForTier(req.tier);
            return selectFromList(candidates, size);
        }
        else if constexpr (std::is_same_v<T, AnyMemorySpaceInTiers>) {
            // Try tiers in order of preference
            for (Tier tier : req.tiers) {
                auto candidates = getMemorySpacesForTier(tier);
                const MemorySpace* selected = selectFromList(candidates, size);
                if (selected) {
                    return selected;
                }
            }
            return nullptr;
        }
        else {
            static_assert(false, "Unhandled ReservationRequest type");
        }
    }, request);
}

const MemorySpace* MemoryReservationManager::selectFromList(const std::vector<const MemorySpace*>& candidates, size_t size) const {
    for (const MemorySpace* space : candidates) {
        if (space && space->canReserve(size)) {
            return space;
        }
    }
    return nullptr;
}

void MemoryReservationManager::buildLookupTables() {
    memory_space_lookup_.clear();
    tier_to_memory_spaces_.clear();
    
    for (const auto& space : memory_spaces_) {
        const MemorySpace* space_ptr = space.get();
        
        // Build direct lookup table
        auto key = std::make_pair(space_ptr->getTier(), space_ptr->getDeviceId());
        memory_space_lookup_[key] = space_ptr;
        
        // Build tier-to-spaces mapping
        tier_to_memory_spaces_[space_ptr->getTier()].push_back(space_ptr);
    }
}

} // namespace memory
} // namespace sirius
