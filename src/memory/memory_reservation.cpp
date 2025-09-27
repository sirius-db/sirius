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
#include <array>

namespace sirius {
namespace memory {


std::unique_ptr<MemoryReservationManager> MemoryReservationManager::instance_ = nullptr;
std::once_flag MemoryReservationManager::initialized_;

MemoryReservationManager::MemoryReservationManager(const std::array<size_t, static_cast<size_t>(Tier::SIZE)>& tier_limits)
    : tier_info_{TierInfo(tier_limits[0]), TierInfo(tier_limits[1]), TierInfo(tier_limits[2])} {
    for (size_t i = 0; i < static_cast<size_t>(Tier::SIZE); ++i) {
        if (tier_limits[i] == 0) {
            throw std::invalid_argument("Tier limit must be greater than 0");
        }
    }
}

std::unique_ptr<Reservation> MemoryReservationManager::requestReservation(Tier tier, size_t size) {
    if (size == 0) {
        throw std::invalid_argument("Reservation size must be greater than 0");
    }
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Wait until we can allocate the requested size
    waitForMemory(tier, size, lock);
    
    // Create the reservation
    auto reservation = std::make_unique<Reservation>(tier, size);
    
    // Update tracking
    size_t tier_idx = getTierIndex(tier);
    tier_info_[tier_idx].total_reserved.fetch_add(size);
    tier_info_[tier_idx].active_count.fetch_add(1);
    
    return reservation;
}

void MemoryReservationManager::releaseReservation(std::unique_ptr<Reservation> reservation) {
    if (!reservation) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update tracking
    size_t tier_idx = getTierIndex(reservation->tier);
    tier_info_[tier_idx].total_reserved.fetch_sub(reservation->size);
    tier_info_[tier_idx].active_count.fetch_sub(1);
    
    // Notify waiting threads
    cv_.notify_all();
}

bool MemoryReservationManager::shrinkReservation(Reservation* reservation, size_t new_size) {
    if (!reservation || new_size >= reservation->size) {
        return false; // Invalid operation
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t size_diff = reservation->size - new_size;
    size_t tier_idx = getTierIndex(reservation->tier);
    
    // Update reservation size
    reservation->size = new_size;
    
    // Update tracking
    tier_info_[tier_idx].total_reserved.fetch_sub(size_diff);
    
    // Notify waiting threads
    cv_.notify_all();
    
    return true;
}

bool MemoryReservationManager::growReservation(Reservation* reservation, size_t new_size) {
    if (!reservation || new_size <= reservation->size) {
        return false; // Invalid operation
    }
    
    size_t size_diff = new_size - reservation->size;
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // Check if we can grow
    if (!canReserve(reservation->tier, size_diff)) {
        return false; // Not enough memory available
    }
    
    // Update reservation size
    reservation->size = new_size;
    
    // Update tracking
    size_t tier_idx = getTierIndex(reservation->tier);
    tier_info_[tier_idx].total_reserved.fetch_add(size_diff);
    
    return true;
}

size_t MemoryReservationManager::getAvailableMemory(Tier tier) const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t tier_idx = getTierIndex(tier);
    size_t reserved = tier_info_[tier_idx].total_reserved.load();
    size_t limit = tier_info_[tier_idx].limit;
    return (reserved >= limit) ? 0 : (limit - reserved);
}

size_t MemoryReservationManager::getTotalReservedMemory(Tier tier) const {
    size_t tier_idx = getTierIndex(tier);
    return tier_info_[tier_idx].total_reserved.load();
}

size_t MemoryReservationManager::getMaxReservation(Tier tier) const {
    size_t tier_idx = getTierIndex(tier);
    return tier_info_[tier_idx].limit;
}

size_t MemoryReservationManager::getActiveReservationCount(Tier tier) const {
    size_t tier_idx = getTierIndex(tier);
    return tier_info_[tier_idx].active_count.load();
}

size_t MemoryReservationManager::getTierIndex(Tier tier) const {
    switch (tier) {
        case Tier::GPU: return 0;
        case Tier::HOST: return 1;
        case Tier::DISK: return 2;
        default: throw std::invalid_argument("Invalid tier");
    }
}

bool MemoryReservationManager::canReserve(Tier tier, size_t size) const {
    size_t tier_idx = getTierIndex(tier);
    size_t current_reserved = tier_info_[tier_idx].total_reserved.load();
    size_t limit = tier_info_[tier_idx].limit;
    return (current_reserved + size) <= limit;
}

void MemoryReservationManager::waitForMemory(Tier tier, size_t size, std::unique_lock<std::mutex>& lock) {
    while (!canReserve(tier, size)) {
        cv_.wait(lock);
    }
}

void MemoryReservationManager::initialize(const std::array<size_t, static_cast<size_t>(Tier::SIZE)>& tier_limits) {
    std::call_once(initialized_, [&tier_limits]() {
        instance_ = std::make_unique<MemoryReservationManager>(tier_limits);
    });
}

MemoryReservationManager& MemoryReservationManager::getInstance() {
    if (!instance_) {
        throw std::runtime_error("MemoryReservationManager not initialized. Call initialize() first.");
    }
    return *instance_;
}

} // namespace memory
} // namespace sirius
