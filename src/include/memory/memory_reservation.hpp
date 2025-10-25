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
#include "memory/memory_space.hpp"
#include <memory>
#include <vector>
#include <variant>
#include <optional>
#include <unordered_map>

// RMM includes for memory resource management
#include <rmm/device_memory_resource.hpp>

namespace sirius {
namespace memory {

// Forward declarations
class MemoryReservationManager;

//===----------------------------------------------------------------------===//
// Reservation Request Strategies
//===----------------------------------------------------------------------===//

/**
 * Request reservation in any memory space within a tier, with optional device preference.
 * If preferred_device_id is specified, that device will be tried first.
 */
struct AnyMemorySpaceInTierWithPreference {
    Tier tier;
    std::optional<size_t> preferred_device_id; // Optional preferred device within tier
    
    explicit AnyMemorySpaceInTierWithPreference(Tier t, std::optional<size_t> device_id = std::nullopt) 
        : tier(t), preferred_device_id(device_id) {}
};

/**
 * Request reservation in any memory space within a specific tier.
 */
struct AnyMemorySpaceInTier {
    Tier tier;
    explicit AnyMemorySpaceInTier(Tier t) : tier(t) {}
};

/**
 * Request reservation in memory spaces across multiple tiers, ordered by preference.
 * The first available tier in the list will be selected.
 */
struct AnyMemorySpaceInTiers {
    std::vector<Tier> tiers; // Ordered by preference
    explicit AnyMemorySpaceInTiers(std::vector<Tier> t) : tiers(std::move(t)) {}
};

/**
 * Variant type for different reservation request strategies.
 * Supports three main approaches:
 * 1. Specific tier with optional device preference
 * 2. Any device in a specific tier 
 * 3. Any device across multiple tiers (ordered by preference)
 */
using ReservationRequest = std::variant<
    AnyMemorySpaceInTierWithPreference,
    AnyMemorySpaceInTier, 
    AnyMemorySpaceInTiers
>;

//===----------------------------------------------------------------------===//
// Reservation
//===----------------------------------------------------------------------===//

/**
 * Represents a memory reservation in a specific memory space.
 * Contains only the essential identifying information (tier, device_id, size).
 * The actual MemorySpace can be obtained through the MemoryReservationManager.
 */
struct Reservation {
    Tier tier;
    size_t device_id;
    size_t size;
    
    Reservation(Tier t, size_t dev_id, size_t s);
    
    // Helper method to get the memory space from the manager
    const MemorySpace* getMemorySpace(const MemoryReservationManager& manager) const;
    
    // Disable copy/move to prevent issues with MemorySpace tracking
    Reservation(const Reservation&) = delete;
    Reservation& operator=(const Reservation&) = delete;
    Reservation(Reservation&&) = delete;
    Reservation& operator=(Reservation&&) = delete;
    
    ~Reservation() = default;
};

//===----------------------------------------------------------------------===//
// MemoryReservationManager
//===----------------------------------------------------------------------===//

/**
 * Central manager for memory reservations across multiple memory spaces.
 * Implements singleton pattern and coordinates reservation requests using
 * different strategies (specific space, tier-based, multi-tier fallback).
 */
class MemoryReservationManager {
public:
    //===----------------------------------------------------------------------===//
    // Configuration and Initialization
    //===----------------------------------------------------------------------===//
    
    /**
     * Configuration for a single MemorySpace.
     * Contains all parameters needed to create a MemorySpace instance.
     */
    struct MemorySpaceConfig {
        Tier tier;
        size_t device_id;
        size_t memory_limit;
        std::vector<std::unique_ptr<rmm::device_memory_resource>> allocators;
        
        // Constructor - allocators must be explicitly provided
        MemorySpaceConfig(Tier t, size_t dev_id, size_t mem_limit, 
                         std::vector<std::unique_ptr<rmm::device_memory_resource>> allocs);
        
        // Move constructor
        MemorySpaceConfig(MemorySpaceConfig&&) = default;
        MemorySpaceConfig& operator=(MemorySpaceConfig&&) = default;
        
        // Delete copy constructor/assignment since allocators contain unique_ptr
        MemorySpaceConfig(const MemorySpaceConfig&) = delete;
        MemorySpaceConfig& operator=(const MemorySpaceConfig&) = delete;
    };
    
    /**
     * Initialize the singleton instance with the given memory space configurations.
     * Must be called before getInstance() can be used.
     */
    static void initialize(std::vector<MemorySpaceConfig> configs);
    
    /**
     * Get the singleton instance. 
     * Throws if initialize() has not been called first.
     */
    static MemoryReservationManager& getInstance();
    
    // Disable copy/move for singleton
    MemoryReservationManager(const MemoryReservationManager&) = delete;
    MemoryReservationManager& operator=(const MemoryReservationManager&) = delete;
    MemoryReservationManager(MemoryReservationManager&&) = delete;
    MemoryReservationManager& operator=(MemoryReservationManager&&) = delete;
    
    //===----------------------------------------------------------------------===//
    // Reservation Interface
    //===----------------------------------------------------------------------===//
    
    /**
     * Main reservation interface using strategy pattern.
     * Supports different reservation strategies through the ReservationRequest variant.
     */
    std::unique_ptr<Reservation> requestReservation(const ReservationRequest& request, size_t size);
    
    
    //===----------------------------------------------------------------------===//
    // Reservation Management 
    //===----------------------------------------------------------------------===//
    
    /**
     * Release a reservation, making its memory available for other requests.
     * Looks up the appropriate MemorySpace using the reservation's tier and device_id.
     */
    void releaseReservation(std::unique_ptr<Reservation> reservation);
    
    /**
     * Attempt to shrink an existing reservation to a smaller size.
     * Returns true if successful, false otherwise.
     */
    bool shrinkReservation(Reservation* reservation, size_t new_size);
    
    /**
     * Attempt to grow an existing reservation to a larger size.
     * Returns true if successful, false if insufficient memory available.
     */
    bool growReservation(Reservation* reservation, size_t new_size);
    
    //===----------------------------------------------------------------------===//
    // MemorySpace Access and Queries
    //===----------------------------------------------------------------------===//
    
    /**
     * Get a specific MemorySpace by tier and device ID.
     * Returns nullptr if no such space exists.
     */
    const MemorySpace* getMemorySpace(Tier tier, size_t device_id) const;
    
    /**
     * Get all MemorySpaces for a specific tier.
     * Returns empty vector if no spaces exist for that tier.
     */
    std::vector<const MemorySpace*> getMemorySpacesForTier(Tier tier) const;
    
    /**
     * Get all MemorySpaces managed by this instance.
     */
    std::vector<const MemorySpace*> getAllMemorySpaces() const;
    
    //===----------------------------------------------------------------------===//
    // Aggregated Queries
    //===----------------------------------------------------------------------===//
    
    // Tier-level aggregations
    size_t getAvailableMemoryForTier(Tier tier) const;
    size_t getTotalReservedMemoryForTier(Tier tier) const;
    size_t getActiveReservationCountForTier(Tier tier) const;
    
    // System-wide aggregations
    size_t getTotalAvailableMemory() const;
    size_t getTotalReservedMemory() const;
    size_t getActiveReservationCount() const;

private:
    /**
     * Private constructor - use initialize() and getInstance() instead.
     */
    explicit MemoryReservationManager(std::vector<MemorySpaceConfig> configs);
    ~MemoryReservationManager() = default;

    // Singleton state
    static std::unique_ptr<MemoryReservationManager> instance_;
    static std::once_flag initialized_;
    
    // Storage for MemorySpace instances (owned by the manager)
    std::vector<std::unique_ptr<MemorySpace>> memory_spaces_;
    
    // Fast lookups
    std::unordered_map<std::pair<Tier, size_t>, const MemorySpace*, 
                      std::hash<std::pair<Tier, size_t>>> memory_space_lookup_;
    std::unordered_map<Tier, std::vector<const MemorySpace*>> tier_to_memory_spaces_;
    
    // Helper methods for request processing
    const MemorySpace* selectMemorySpace(const ReservationRequest& request, size_t size) const;
    const MemorySpace* selectFromList(const std::vector<const MemorySpace*>& candidates, size_t size) const;
    
    void buildLookupTables();
};

} // namespace memory
} // namespace sirius
