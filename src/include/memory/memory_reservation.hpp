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

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>
#include <memory>
#include <thread>
#include <array>

namespace sirius {
namespace memory {

enum class Tier {
    GPU,
    HOST,
    DISK,
    SIZE //value = size of the enum, allows code to be more dynamic
};

struct Reservation {
    Tier tier;
    size_t size;
    
    Reservation(Tier t, size_t s) : tier(t), size(s) {}
};

class MemoryReservationManager {
public:

    static void initialize(const std::array<size_t, static_cast<size_t>(Tier::SIZE)>& tier_limits);
    static MemoryReservationManager& getInstance();
    
    MemoryReservationManager(const MemoryReservationManager&) = delete;
    MemoryReservationManager& operator=(const MemoryReservationManager&) = delete;
    MemoryReservationManager(MemoryReservationManager&&) = delete;
    MemoryReservationManager& operator=(MemoryReservationManager&&) = delete;
    
    std::unique_ptr<Reservation> requestReservation(Tier tier, size_t size);
    void releaseReservation(std::unique_ptr<Reservation> reservation);
    
    bool shrinkReservation(Reservation* reservation, size_t new_size);
    bool growReservation(Reservation* reservation, size_t new_size);
    
    size_t getAvailableMemory(Tier tier) const;
    size_t getTotalReservedMemory(Tier tier) const;
    size_t getMaxReservation(Tier tier) const;
    
    size_t getActiveReservationCount(Tier tier) const;
    
public:
    explicit MemoryReservationManager(const std::array<size_t, static_cast<size_t>(Tier::SIZE)>& tier_limits);
    ~MemoryReservationManager() = default;

private:
    
    static std::unique_ptr<MemoryReservationManager> instance_;
    static std::once_flag initialized_;
    
    struct TierInfo {
        const size_t limit;
        std::atomic<size_t> total_reserved{0};
        std::atomic<size_t> active_count{0};
        
        TierInfo(size_t l) : limit(l) {}
    };
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    TierInfo tier_info_[static_cast<size_t>(Tier::SIZE)]; //dynamic size of the enum since SIZE is the last value
    
    size_t getTierIndex(Tier tier) const;
    bool canReserve(Tier tier, size_t size) const;
    void waitForMemory(Tier tier, size_t size, std::unique_lock<std::mutex>& lock);
};

} // namespace memory
} // namespace sirius
