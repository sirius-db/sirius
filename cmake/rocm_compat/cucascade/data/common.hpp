/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade data common — idata_representation abstract base.
//! REAL virtual base — 2 Sirius classes + 1 concept derive from it.

#pragma once
#include "cucascade/memory/common.hpp"
#include <rmm/cuda_stream.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

/// Batch state lifecycle.
enum class batch_state { idle, read_only, mutable_locked };

/// Abstract base for a data representation (GPU table, host table, disk table).
/// Sirius subclasses this (cached_shared_representation, host_parquet_representation)
/// and uses a concept constraint std::derived_from<T, idata_representation>.
class idata_representation {
 public:
  explicit idata_representation(memory::memory_space& space) : space_(&space) {}
  virtual ~idata_representation() = default;

  virtual memory::Tier get_current_tier() const { return memory::Tier::GPU; }
  virtual int32_t get_device_id() const { return 0; }
  virtual memory::memory_space& get_memory_space() const { return *space_; }
  virtual memory::memory_space& get_memory_space() { return *space_; }

  virtual std::size_t get_size_in_bytes() const = 0;
  virtual std::size_t get_uncompressed_data_size_in_bytes() const = 0;
  virtual std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) = 0;

  virtual void record_writer_event(rmm::cuda_stream_view) {}
  virtual cudaEvent_t get_writer_event() const { return nullptr; }
  virtual void rebind_stream(rmm::cuda_stream_view) {}

  /// Template cast using dynamic_cast. Heavily used as ->cast<gpu_table_representation>().
  template <typename TargetType>
  TargetType& cast() {
    auto* ptr = dynamic_cast<TargetType*>(this);
    if (!ptr) throw std::runtime_error("cuCascade stub: cast failed");
    return *ptr;
  }

  /// Const overload — Sirius calls cast() on const idata_representation*.
  template <typename TargetType>
  TargetType const& cast() const {
    auto const* ptr = dynamic_cast<TargetType const*>(this);
    if (!ptr) throw std::runtime_error("cuCascade stub: cast failed");
    return *ptr;
  }

 protected:
  memory::memory_space* space_;
};

/// No-op batch probe. Sirius subclasses this (quent_data_batch_probe) and
/// also std::make_unique<cucascade::idata_batch_probe>() — must be constructible.
class idata_batch_probe {
 public:
  virtual ~idata_batch_probe() = default;
  virtual void created(uint64_t /*batch_id*/, idata_representation const* /*data*/) {}
  virtual void conversion_started(idata_representation const* /*data*/,
                                  memory::memory_space const* /*target*/) {}
  virtual void conversion_completed(idata_representation const* /*data*/, bool /*success*/) {}
  virtual void data_replaced(idata_representation const* /*new_data*/) {}
};

}  // namespace cucascade
