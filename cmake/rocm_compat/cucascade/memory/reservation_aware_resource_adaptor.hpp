/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file cuCascade reservation_aware_resource_adaptor — ROCm stub.

#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/oom_handling_policy.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace cucascade::memory {

class memory_space;

/// Wraps an RMM memory resource with reservation tracking.
/// Stub: all methods throw or return defaults.
class reservation_aware_resource_adaptor {
 public:
  reservation_aware_resource_adaptor(rmm::device_async_resource_ref /*upstream*/,
                                     std::size_t /*reservation_limit*/ = 0) {}

  void* allocate(rmm::cuda_stream_view, std::size_t /*bytes*/,
                 std::size_t /*alignment*/ = 0) {
    throw std::runtime_error("cuCascade stub: reservation_aware_resource_adaptor::allocate");
  }
  void deallocate(rmm::cuda_stream_view, void* /*ptr*/, std::size_t /*bytes*/,
                  std::size_t /*alignment*/ = 0) {}

  std::size_t get_peak_allocated_bytes(rmm::cuda_stream_view) const { return 0; }
  void reset_stream_reservation(rmm::cuda_stream_view) {}
  // Sirius calls with 4 args: (stream, reservation, limit_policy, oom_policy)
  // and with 3 args: (stream, reservation, limit_policy)
  // and with 2 args: (stream, reservation) [test code]
  // Both return bool (Sirius checks: if (!allocator->attach_reservation_to_tracker(...)))
  bool attach_reservation_to_tracker(rmm::cuda_stream_view /*stream*/,
                                     std::unique_ptr<reservation> /*reservation*/,
                                     std::unique_ptr<reservation_limit_policy> /*limit_policy*/ = nullptr,
                                     std::unique_ptr<oom_handling_policy> /*oom_policy*/ = nullptr) {
    return false; // stub: no tracking
  }

  operator rmm::device_async_resource_ref() const {
    throw std::runtime_error("cuCascade stub: no upstream resource");
  }
};

}  // namespace cucascade::memory
