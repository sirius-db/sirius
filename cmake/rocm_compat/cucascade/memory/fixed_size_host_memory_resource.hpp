/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade fixed_size_host_memory_resource — ROCm stub.
//! multiple_blocks_allocation is NESTED inside the class (Sirius references
//! it as fixed_size_host_memory_resource::multiple_blocks_allocation).

#pragma once
#include "cucascade/memory/common.hpp"
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/notification_channel.hpp"
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cucascade::memory {

/// Fixed-size-block host memory resource (pinned host RAM pool).
/// Stub: allocate/deallocate throw.
class fixed_size_host_memory_resource {
 public:
  static constexpr std::size_t default_pool_size_val = default_pool_size;
  static constexpr std::size_t default_block_size_val = default_block_size;
  static constexpr std::size_t default_initial_number_pools_val = default_initial_number_pools;

  /// Multiple-block allocation from this resource. NESTED type — Sirius
  /// references it as fixed_size_host_memory_resource::multiple_blocks_allocation.
  struct multiple_blocks_allocation {
    /// Factory: creates from a buffer, memory resource, and reservation.
    /// Sirius calls multiple_blocks_allocation::create(buf, mr, reservation*).
    static std::shared_ptr<multiple_blocks_allocation> create(
      std::shared_ptr<void> /*buffer*/,
      rmm::device_async_resource_ref /*mr*/,
      reservation* /*res*/) {
      return std::make_shared<multiple_blocks_allocation>();
    }

    static std::unique_ptr<multiple_blocks_allocation> create_unique() {
      return std::make_unique<multiple_blocks_allocation>();
    }

    bool empty() const { return blocks_.empty(); }
    std::size_t size_bytes() const { return total_bytes_; }
    std::size_t size() const { return blocks_.size(); }
    void release_blocks() { blocks_.clear(); }

    void* operator[](std::size_t i) const {
      return i < blocks_.size() ? blocks_[i] : throw std::out_of_range("block index");
    }
    void* at(std::size_t i) const { return operator[](i); }

    /// Accessor for the blocks vector. Sirius calls .get_blocks().size().
    std::vector<void*> const& get_blocks() const { return blocks_; }
    std::vector<void*>& get_blocks() { return blocks_; }

    std::size_t block_size() const { return block_size_; }
    void grow_by(std::size_t n) { total_bytes_ += n; }
    void trim_to(std::size_t n) { total_bytes_ = n; }

    std::vector<void*> blocks_;
    std::size_t block_size_{default_block_size};
    std::size_t total_bytes_{0};
  };

  using fixed_multiple_blocks_allocation = std::unique_ptr<multiple_blocks_allocation>;

  fixed_size_host_memory_resource(int32_t /*device_id*/,
                                  rmm::device_async_resource_ref /*upstream*/,
                                  std::size_t /*mem_limit*/ = 0,
                                  std::size_t /*capacity*/ = 0,
                                  std::size_t /*block_size*/ = default_block_size,
                                  std::size_t /*pool_size*/ = default_pool_size,
                                  std::size_t /*initial_pools*/ = default_initial_number_pools) {}

  std::size_t get_block_size() const { return default_block_size; }
  std::size_t get_available_memory() const { return 0; }
  std::size_t get_free_blocks() const { return 0; }
  std::size_t get_total_blocks() const { return 0; }
  std::size_t get_total_allocated_bytes() const { return 0; }
  std::size_t get_total_reserved_bytes() const { return 0; }
  std::size_t get_active_reservation_count() const { return 0; }
  std::size_t get_peak_total_allocated_bytes() const { return 0; }

  std::unique_ptr<reservation> reserve(std::size_t /*bytes*/,
                                       notification_channel* /*notifier*/ = nullptr) {
    throw std::runtime_error("cuCascade stub: fixed_size_host_memory_resource::reserve");
  }

  fixed_multiple_blocks_allocation allocate_multiple_blocks(std::size_t /*bytes*/,
                                                            reservation* /*res*/ = nullptr) {
    throw std::runtime_error("cuCascade stub: allocate_multiple_blocks");
  }

  void* allocate(rmm::cuda_stream_view, std::size_t /*bytes*/,
                 std::size_t /*alignment*/ = 0) {
    throw std::runtime_error("cuCascade stub: host allocate");
  }
  void deallocate(rmm::cuda_stream_view, void* /*ptr*/, std::size_t /*bytes*/,
                  std::size_t /*alignment*/ = 0) {}

  operator rmm::device_async_resource_ref() const {
    throw std::runtime_error("cuCascade stub: no upstream host resource");
  }
};

}  // namespace cucascade::memory
