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
#include <string>
#include <stdexcept>
#include <mutex>
#include <condition_variable>

// RMM includes for memory resource management
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

namespace sirius {
namespace memory {

// Forward declarations
class MemoryReservationManager;
struct Reservation;

//===----------------------------------------------------------------------===//
// Reservation Limit Policy Interface
//===----------------------------------------------------------------------===//

/**
 * @brief Base class for reservation limit policies that control behavior when stream reservations are exceeded.
 * 
 * Reservation limit policies are pluggable strategies that determine what happens when an allocation
 * would cause a stream's memory usage to exceed its reservation limit.
 */
class ReservationLimitPolicy {
public:
    virtual ~ReservationLimitPolicy() = default;

    /**
     * @brief Handle an allocation that would exceed the stream's reservation.
     * 
     * This method is called when an allocation would cause the current allocated bytes
     * plus the new allocation to exceed the stream's reservation. The policy can:
     * 1. Allow the allocation to proceed (ignore policy)
     * 2. Increase the reservation and allow the allocation (increase policy)
     * 3. Throw an exception to prevent the allocation (fail policy)
     * 
     * @param adaptor Reference to the tracking resource adaptor
     * @param stream The stream that would exceed its reservation
     * @param current_allocated Current allocated bytes on the stream
     * @param requested_bytes Number of bytes being requested
     * @param reservation Pointer to the stream's reservation (may be null if no reservation set)
     * @throws rmm::out_of_memory if the policy decides to reject the allocation
     */
    virtual void handle_over_reservation(
        rmm::cuda_stream_view stream,
        std::size_t current_allocated,
        std::size_t requested_bytes,
        Reservation* reservation) = 0;

    /**
     * @brief Get a human-readable name for this policy.
     * @return Policy name string
     */
    virtual std::string get_policy_name() const = 0;
};

/**
 * @brief Ignore policy - allows allocations to proceed even if they exceed reservations.
 * 
 * This policy simply ignores reservation limits and allows all allocations to proceed.
 * It's useful for soft reservations where you want to track usage but not enforce limits.
 */
class IgnoreReservationLimitPolicy : public ReservationLimitPolicy {
public:
    void handle_over_reservation(
        rmm::cuda_stream_view stream,
        std::size_t current_allocated,
        std::size_t requested_bytes,
        Reservation* reservation) override {
        // Do nothing - allow the allocation to proceed
    }

    std::string get_policy_name() const override {
        return "ignore";
    }
};

/**
 * @brief Fail policy - throws an exception when reservations are exceeded.
 * 
 * This policy enforces strict reservation limits by throwing rmm::out_of_memory
 * when an allocation would exceed the stream's reservation.
 */
class FailReservationLimitPolicy : public ReservationLimitPolicy {
public:
    void handle_over_reservation(
        rmm::cuda_stream_view stream,
        std::size_t current_allocated,
        std::size_t requested_bytes,
        Reservation* reservation) override;

    std::string get_policy_name() const override {
        return "fail";
    }
};

/**
 * @brief Increase policy - automatically increases reservations when limits are exceeded.
 * 
 * This policy automatically increases the stream's reservation when an allocation would
 * exceed the current limit. It uses a padding factor to avoid frequent reservation increases.
 */
class IncreaseReservationLimitPolicy : public ReservationLimitPolicy {
public:
    /**
     * @brief Constructs an increase policy with the specified padding factor.
     * 
     * @param padding_factor Factor by which to pad reservation increases (default 1.5)
     *                      For example, 1.5 means increase by 50% more than needed
     */
     explicit IncreaseReservationLimitPolicy(double padding_factor = 2.0d)
        : padding_factor_(padding_factor) {
        if (padding_factor < 1.0) {
            throw std::invalid_argument("Padding factor must be >= 1.0");
        }
    }

    void handle_over_reservation(
        rmm::cuda_stream_view stream,
        std::size_t current_allocated,
        std::size_t requested_bytes,
        Reservation* reservation) override;

    std::string get_policy_name() const override {
        return "increase(padding=" + std::to_string(padding_factor_) + ")";
    }

    /**
     * @brief Get the padding factor used by this policy.
     * @return The padding factor
     */
    double get_padding_factor() const { return padding_factor_; }

private:
    double padding_factor_;
};

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
    
    
    //===----------------------------------------------------------------------===//
    // Reservation Size Management
    //===----------------------------------------------------------------------===//
    
    /**
     * @brief Attempts to grow this reservation to a new larger size.
     * @param new_size The new size for the reservation (must be larger than current size)
     * @return true if the reservation was successfully grown, false otherwise
     */
    bool grow_to(size_t new_size);
    
    /**
     * @brief Attempts to grow this reservation by additional bytes.
     * @param additional_bytes Number of bytes to add to the current reservation
     * @return true if the reservation was successfully grown, false otherwise
     */
    bool grow_by(size_t additional_bytes);
    
    /**
     * @brief Attempts to shrink this reservation to a new smaller size.
     * @param new_size The new size for the reservation (must be smaller than current size)
     * @return true if the reservation was successfully shrunk, false otherwise
     */
    bool shrink_to(size_t new_size);
    
    /**
     * @brief Attempts to shrink this reservation by removing bytes.
     * @param bytes_to_remove Number of bytes to remove from the current reservation
     * @return true if the reservation was successfully shrunk, false otherwise
     */
    bool shrink_by(size_t bytes_to_remove);
    
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
        std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;
        
        // Constructor - allocators must be explicitly provided
        MemorySpaceConfig(Tier t, size_t dev_id, size_t mem_limit, 
                         std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocs);
        
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
     * Test-only: Reset the singleton so tests can reinitialize with different configs.
     * Not thread-safe; intended only for unit tests.
     */
    static void reset_for_testing();
    
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
    
    // Public destructor (required for std::unique_ptr)
    ~MemoryReservationManager() = default;
    
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

    // Singleton state
    static std::unique_ptr<MemoryReservationManager> instance_;
    static std::once_flag initialized_;
    static bool allow_reinitialize_for_tests_;
    
    // Storage for MemorySpace instances (owned by the manager)
    std::vector<std::unique_ptr<MemorySpace>> memory_spaces_;
    
    // Fast lookups
    std::unordered_map<std::pair<Tier, size_t>, const MemorySpace*, 
                      std::hash<std::pair<Tier, size_t>>> memory_space_lookup_;
    std::unordered_map<Tier, std::vector<const MemorySpace*>> tier_to_memory_spaces_;
    
    // Helper method: attempts to select a space and immediately make a reservation
    // Returns a reservation when successful, or std::nullopt if none can satisfy the request
    std::optional<std::unique_ptr<Reservation>> selectMemorySpaceAndMakeReservation(const ReservationRequest& request, size_t size) const;
    
    void buildLookupTables();

    // Synchronization for cross-space waiting when no MemorySpace can currently satisfy a request
    mutable std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
};

} // namespace memory
} // namespace sirius
