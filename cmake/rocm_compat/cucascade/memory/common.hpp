/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuCascade memory common types — ROCm stub.
//! Tier enum, memory_space_id, config structs, traits.

#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

namespace cucascade::memory {

/// Memory tier: GPU (device VRAM), HOST (pinned host RAM), DISK.
enum class Tier { GPU, HOST, DISK, SIZE };

/// Unique identifier for a memory space (tier + device).
struct memory_space_id {
  Tier tier{Tier::GPU};
  int32_t device_id{0};

  bool operator==(memory_space_id const& other) const = default;
};

/// Hash for memory_space_id (used in unordered_map).
struct memory_space_hash {
  std::size_t operator()(memory_space_id const& id) const noexcept {
    return std::hash<int64_t>{}(static_cast<int64_t>(id.tier) << 32 | id.device_id);
  }
};

// Forward declarations
class memory_space;
class memory_reservation_manager;
class reservation;
class reserved_arena;

/// GPU memory space configuration.
struct gpu_memory_space_config {
  Tier tier() const { return Tier::GPU; }
  int32_t device_id{0};
  std::size_t usage_limit{0};
  std::size_t reservation_limit{0};
};

/// Host memory space configuration.
struct host_memory_space_config {
  Tier tier() const { return Tier::HOST; }
  int32_t device_id{0};
  std::size_t capacity{0};
  bool portable{false};
};

/// Disk memory space configuration.
struct disk_memory_space_config {
  Tier tier() const { return Tier::DISK; }
  std::string mounting_point;
  std::size_t capacity{0};
};

/// Variant of all memory space configs. Sirius uses std::holds_alternative /
/// std::get_if on this.
using memory_space_config =
  std::variant<std::monostate, gpu_memory_space_config,
               host_memory_space_config, disk_memory_space_config>;

/// Device memory resource factory function type.
using DeviceMemoryResourceFactoryFn =
  std::function<rmm::device_async_resource_ref(int32_t device_id)>;

/// Trait mapping Tier → resource type. Specialized below.
template <Tier T>
struct tier_memory_resource_trait {
  using type = void;
};

// Forward-declare the resource types (defined in their own headers).
class fixed_size_host_memory_resource;
class reservation_aware_resource_adaptor;

template <>
struct tier_memory_resource_trait<Tier::HOST> {
  using type = fixed_size_host_memory_resource;
};
template <>
struct tier_memory_resource_trait<Tier::GPU> {
  using type = reservation_aware_resource_adaptor;
};

/// Default constants (used by Sirius as compile-time values).
inline constexpr std::size_t default_block_size = 1 << 20;        // 1 MiB
inline constexpr std::size_t default_pool_size = 1 << 24;          // 16 MiB
inline constexpr std::size_t default_initial_number_pools = 4;

/// Check if peer DMA works between two devices (stub: always false).
inline bool probe_peer_dma_works(int32_t /*src*/, int32_t /*dst*/) { return false; }

}  // namespace cucascade::memory
