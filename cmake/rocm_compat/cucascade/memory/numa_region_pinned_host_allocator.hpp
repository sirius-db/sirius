/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade numa_region_pinned_host_memory_resource — ROCm stub.

#pragma once
#include "cucascade/memory/small_pinned_host_memory_resource.hpp"
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <cstddef>
#include <stdexcept>

namespace cucascade::memory {

class numa_region_pinned_host_memory_resource {
 public:
  // Standard ctor
  numa_region_pinned_host_memory_resource(int32_t /*device_id*/,
                                          rmm::device_async_resource_ref /*upstream*/,
                                          std::size_t /*mem_limit*/ = 0,
                                          std::size_t /*slab_size*/ = 0) {}

  // Sirius test/benchmark ctor: (device_id, make_portable)
  numa_region_pinned_host_memory_resource(int32_t /*device_id*/, bool /*make_portable*/) {}

  void* allocate(rmm::cuda_stream_view, std::size_t, std::size_t = 0) {
    throw std::runtime_error("cuCascade stub: numa allocate");
  }
  void deallocate(rmm::cuda_stream_view, void*, std::size_t, std::size_t = 0) {}

  operator rmm::device_async_resource_ref() const {
    throw std::runtime_error("cuCascade stub: no upstream");
  }
};

}  // namespace cucascade::memory
