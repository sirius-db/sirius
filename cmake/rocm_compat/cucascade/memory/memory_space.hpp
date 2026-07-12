/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuCascade memory_space — ROCm stub.
//! The central type: ties together a tier, device, stream pool, and allocator.

#pragma once

#include "cucascade/memory/common.hpp"
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/stream_pool.hpp"
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>

namespace cucascade::memory {

/// A memory space represents a tier (GPU/HOST/DISK) on a specific device,
/// with an associated allocator and stream pool. This stub provides the
/// interface Sirius calls; methods throw at runtime (graceful degradation
/// to DuckDB CPU execution).
class memory_space {
 public:
  memory_space(Tier tier, int32_t device_id,
               rmm::device_async_resource_ref allocator,
               std::shared_ptr<exclusive_stream_pool> streams)
    : tier_(tier), device_id_(device_id), allocator_(allocator), streams_(std::move(streams)) {}

  memory_space_id get_id() const { return {tier_, device_id_}; }
  Tier get_tier() const { return tier_; }
  int32_t get_device_id() const { return device_id_; }
  rmm::device_async_resource_ref get_default_allocator() const { return allocator_; }

  rmm::cuda_stream_view acquire_stream() {
    if (!streams_) throw std::runtime_error("cuCascade stub: no stream pool");
    return streams_->acquire_stream(stream_acquire_policy::GROW);
  }
  // Const overload — Sirius calls acquire_stream() on memory_space const&
  rmm::cuda_stream_view acquire_stream() const {
    return rmm::cuda_stream_per_thread;
  }

  bool should_downgrade_memory() const { return false; }

  std::string to_string() const {
    return "cuCascade stub memory_space(tier=" + std::to_string(static_cast<int>(tier_)) +
           ",device=" + std::to_string(device_id_) + ")";
  }

  bool operator==(memory_space const& other) const { return get_id() == other.get_id(); }
  bool operator!=(memory_space const& other) const { return !(*this == other); }

  /// Template: get memory resource as a specific type.
  /// Throws — cuCascade stub has no real reservation system. The exception
  /// propagates up; Sirius's error handling catches it and falls back to
  /// DuckDB CPU execution.
  template <typename T>
  T* get_memory_resource_as() {
    throw std::runtime_error("cuCascade stub: get_memory_resource_as<T>() — no reservation system");
  }
  // Const overload — Sirius calls get_memory_resource_as<T>() on const memory_space&
  template <typename T>
  T const* get_memory_resource_as() const {
    throw std::runtime_error("cuCascade stub: get_memory_resource_as<T>() const — no reservation system");
  }

  /// Template: get memory resource for a specific tier.
  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type* get_memory_resource_of() {
    throw std::runtime_error("cuCascade stub: get_memory_resource_of<TIER>() — no reservation system");
  }
  // Const overload
  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type const* get_memory_resource_of() const {
    throw std::runtime_error("cuCascade stub: get_memory_resource_of<TIER>() const — no reservation system");
  }

  // --- Memory tracking (real: tracks allocated/reserved bytes) ---
  std::size_t get_max_memory() const { return max_memory_; }
  std::size_t get_available_memory() const { return max_memory_ - reserved_memory_; }
  std::size_t get_total_reserved_memory() const { return reserved_memory_; }

  std::unique_ptr<reservation> make_reservation(std::size_t bytes) {
    if (bytes > get_available_memory()) {
      throw std::runtime_error("cuCascade: insufficient memory for reservation");
    }
    reserved_memory_ += bytes;
    auto arena = std::make_unique<simple_arena>(bytes);
    return reservation::create(*this, std::move(arena));
  }
  std::unique_ptr<reservation> make_reservation_or_null(std::size_t bytes) {
    if (bytes > get_available_memory()) return nullptr;
    reserved_memory_ += bytes;
    auto arena = std::make_unique<simple_arena>(bytes);
    return reservation::create(*this, std::move(arena));
  }
  std::unique_ptr<reservation> make_reservation_upto(std::size_t bytes) {
    std::size_t actual = std::min(bytes, get_available_memory());
    if (actual == 0) return nullptr;
    reserved_memory_ += actual;
    auto arena = std::make_unique<simple_arena>(actual);
    return reservation::create(*this, std::move(arena));
  }

  void release_reservation(std::size_t bytes) { if (reserved_memory_ >= bytes) reserved_memory_ -= bytes; }

 private:
  Tier tier_;
  int32_t device_id_;
  rmm::device_async_resource_ref allocator_;
  std::shared_ptr<exclusive_stream_pool> streams_;
  std::size_t max_memory_{0};
  std::size_t reserved_memory_{0};

  /// Simple arena: tracks a byte count (no actual allocation — the
  /// reservation is bookkeeping; actual device memory is allocated
  /// separately via the allocator or hipMalloc).
  class simple_arena : public reserved_arena {
   public:
    explicit simple_arena(std::size_t n) : bytes_(n) {}
    std::size_t size() const override { return bytes_; }
    void grow_by(std::size_t n) override { bytes_ += n; }
    void shrink_to_fit() override {}
   private:
    std::size_t bytes_{0};
  };
};

}  // namespace cucascade::memory
