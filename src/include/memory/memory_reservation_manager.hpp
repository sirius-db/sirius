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
#include "memory/memory_reservation_manager.hpp"
#include "memory/memory_space.hpp"

#include <rmm/cuda_device.hpp>

#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// RMM includes for memory resource management
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace cucascade {
namespace memory {

// Forward declarations
class memory_reservation_manager;
class reservation_aware_resource_adaptor;
class fixed_size_host_memory_resource;
struct reservation;

//===----------------------------------------------------------------------===//
// Reservation Request Strategies
//===----------------------------------------------------------------------===//

struct reservation_request_strategy {
  explicit reservation_request_strategy(bool strong_ordering) : strong_ordering_(strong_ordering) {}

  virtual std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const = 0;

  [[nodiscard]] bool has_strong_ordering() const noexcept { return strong_ordering_; }

 protected:
  static std::span<memory_space*> get_all_memory_resource(memory_reservation_manager& manager);

  static std::span<memory_space*> get_all_memory_resource(memory_reservation_manager& manager,
                                                          Tier tier);

  static memory_space* get_memory_resource(memory_space_id source_id);

  static std::vector<memory_space*> get_memory_resource(std::span<memory_space_id> source_ids);

  bool strong_ordering_{false};
};

/**
 * Request reservation in any memory space within a tier, with optional device preference.
 * If preferred_device_id is specified, that device will be tried first.
 */
struct any_memory_space_in_tier_with_preference : public reservation_request_strategy {
  Tier tier;
  std::optional<size_t> preferred_device_id;  // Optional preferred device within tier

  explicit any_memory_space_in_tier_with_preference(Tier t,
                                                    std::optional<size_t> device_id = std::nullopt)
    : reservation_request_strategy(true), tier(t), preferred_device_id(device_id)
  {
  }

  std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const override;
};

/**
 * Request reservation in any memory space within a specific tier.
 */
struct any_memory_space_in_tier : public reservation_request_strategy {
  Tier tier;
  explicit any_memory_space_in_tier(Tier t) : reservation_request_strategy(false), tier(t) {}

  std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const override;
};

/**
 * Request reservation in memory spaces across multiple tiers, ordered by preference.
 * The first available tier in the list will be selected.
 */
struct any_memory_space_in_tiers : public reservation_request_strategy {
  std::vector<Tier> tiers;  // Ordered by preference

  explicit any_memory_space_in_tiers(std::vector<Tier> t)
    : reservation_request_strategy(false), tiers(std::move(t))
  {
  }

  std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const override;
};

/**
 * Request reservation in memory spaces across multiple tiers, ordered by preference.
 * The first available tier in the list will be selected.
 */
struct specific_memory_space : public reservation_request_strategy {
  memory_space_id target_id;

  specific_memory_space(Tier t, int32_t dev_id)
    : reservation_request_strategy(true), target_id(t, dev_id)
  {
  }

  std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const override;
};

/**
 * Request reservation by downgrading from a source memory space to target tiers.
 */
struct any_memory_space_to_downgrade : public reservation_request_strategy {
  memory_space_id src_id;
  std::vector<Tier> target_tiers;

  explicit any_memory_space_to_downgrade(memory_space_id src, Tier target)
    : reservation_request_strategy(true), src_id(src), target_tiers({target})
  {
  }

  explicit any_memory_space_to_downgrade(memory_space_id src, std::vector<Tier> targets)
    : reservation_request_strategy(true), src_id(src), target_tiers(std::move(targets))
  {
  }

  std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const override;
};

/**
 * Request reservation by upgrading from a source memory space to a target tier.
 */
struct any_memory_space_to_upgrade : public reservation_request_strategy {
  memory_space_id src_id;
  Tier target_tier;

  explicit any_memory_space_to_upgrade(memory_space_id src, Tier target)
    : reservation_request_strategy(true), src_id(src), target_tier(target)
  {
  }

  std::vector<memory_space*> get_candidates(memory_reservation_manager& manager) const override;
};

//===----------------------------------------------------------------------===//
// memory_reservation_manager
//===----------------------------------------------------------------------===//

/**
 * Central manager for memory reservations across multiple memory spaces.
 * Implements singleton pattern and coordinates reservation requests using
 * different strategies (specific space, tier-based, multi-tier fallback).
 */
class memory_reservation_manager {
  friend struct reservation;
  friend struct reservation_request_strategy;

 public:
  //===----------------------------------------------------------------------===//
  // Configuration and Initialization
  //===----------------------------------------------------------------------===//

  /**
   * Configuration for a single memory_space.
   * Contains all parameters needed to create a memory_space instance.
   */
  struct memory_space_config {
    Tier tier;
    int device_id;
    size_t memory_limit;
    float downgrade_tigger_threshold{0.75};
    float downgrade_stop_threshold{0.65};
    std::size_t memory_capacity;  // Optional total capacity, defaults to device capacity
    DeviceMemoryResourceFactoryFn mr_factory_fn;

    // Constructor - allocators must be explicitly provided
    memory_space_config(Tier t,
                        int dev_id,
                        size_t mem_limit,
                        DeviceMemoryResourceFactoryFn mr_fn = nullptr);

    // Constructor - allocators must be explicitly provided
    memory_space_config(Tier t,
                        int dev_id,
                        size_t mem_limit,
                        size_t mem_capacity,
                        DeviceMemoryResourceFactoryFn mr_fn = nullptr);
  };

  /**
   * Initialize the singleton instance with the given memory space configurations.
   * Must be called before get_instance() can be used.
   */
  static void initialize(std::vector<memory_space_config> configs);

  /**
   * Test-only: Reset the singleton so tests can reinitialize with different configs.
   * Not thread-safe; intended only for unit tests.
   */
  static void reset_for_testing();

  /**
   * Get the singleton instance.
   * Throws if initialize() has not been called first.
   */
  static memory_reservation_manager& get_instance();

  // Disable copy/move for singleton
  memory_reservation_manager(const memory_reservation_manager&)            = delete;
  memory_reservation_manager& operator=(const memory_reservation_manager&) = delete;
  memory_reservation_manager(memory_reservation_manager&&)                 = delete;
  memory_reservation_manager& operator=(memory_reservation_manager&&)      = delete;

  ~memory_reservation_manager();

  //===----------------------------------------------------------------------===//
  // Reservation Interface
  //===----------------------------------------------------------------------===//

  /**
   * Main reservation interface using strategy pattern.
   * Supports different reservation strategies through the reservation_request variant.
   */
  std::unique_ptr<reservation> request_reservation(const reservation_request_strategy& request,
                                                   size_t size);

  //===----------------------------------------------------------------------===//
  // memory_space Access and Queries
  //===----------------------------------------------------------------------===//

  /**
   * Get a specific memory_space by tier and device ID.
   * Returns nullptr if no such space exists.
   */
  const memory_space* get_memory_space(Tier tier, int32_t device_id) const;

  /**
   * Get all memory_spaces for a specific tier.
   * Returns empty vector if no spaces exist for that tier.
   */
  std::span<const memory_space*> get_memory_spaces_for_tier(Tier tier) const;

  /**
   * Get all memory_spaces managed by this instance.
   */
  std::span<const memory_space*> get_all_memory_spaces() const noexcept;

  //===----------------------------------------------------------------------===//
  // Aggregated Queries
  //===----------------------------------------------------------------------===//

  // Tier-level aggregations
  size_t get_available_memory_for_tier(Tier tier) const;
  size_t get_total_reserved_memory_for_tier(Tier tier) const;
  size_t get_active_reservation_count_for_tier(Tier tier) const;

  // System-wide aggregations
  size_t get_total_available_memory() const;
  size_t get_total_reserved_memory() const;
  size_t get_active_reservation_count() const;

  void shutdown();

 private:
  /**
   * Private constructor - use initialize() and get_instance() instead.
   */
  explicit memory_reservation_manager(std::vector<memory_space_config> configs);

  memory_space* get_mutable_memory_space(Tier tier, int32_t device_id);
  std::vector<memory_space*> get_mutable_memory_space(std::span<memory_space_id> ids);
  std::span<memory_space*> get_mutable_memory_spaces_for_tier(Tier tier);
  std::span<memory_space*> get_mutable_all_memory_spaces() noexcept;

  // Singleton state
  static std::unique_ptr<memory_reservation_manager> _instance;
  static std::once_flag _initialized;
  static bool _allow_reinitialize_for_tests;

  // Storage for memory_space instances (owned by the manager)
  std::vector<std::unique_ptr<memory_space>> _memory_spaces;

  // Fast lookups
  std::vector<memory_space*> _memory_space_views;
  std::unordered_map<memory_space_id, memory_space*> _memory_space_lookup;
  std::unordered_map<Tier, std::vector<memory_space*>> _tier_to_memory_spaces;

  // Helper method: attempts to select a space and immediately make a reservation
  // Returns a reservation when successful, or std::nullopt if none can satisfy the request
  std::optional<std::unique_ptr<reservation>> select_memory_space_and_make_reservation(
    const reservation_request_strategy& request, size_t size);

  void build_lookup_tables();

  // Synchronization for cross-space waiting when no memory_space can currently satisfy a request
  mutable std::mutex _wait_mutex;
  std::condition_variable _wait_cv;
};

}  // namespace memory
}  // namespace cucascade
