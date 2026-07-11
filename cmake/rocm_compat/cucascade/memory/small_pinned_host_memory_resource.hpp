/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade small_pinned_host_memory_resource — ROCm stub.

#pragma once
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#include <cstddef>
#include <stdexcept>

namespace cucascade::memory {

class small_pinned_host_memory_resource {
 public:
  static constexpr std::size_t MAX_SLAB_SIZE = 1 << 20; // 1 MiB

  small_pinned_host_memory_resource(int32_t /*device_id*/,
                                    rmm::device_async_resource_ref /*upstream*/,
                                    std::size_t /*mem_limit*/ = 0,
                                    std::size_t /*slab_size*/ = MAX_SLAB_SIZE) {}

  void* allocate(rmm::cuda_stream_view, std::size_t, std::size_t = 0) {
    throw std::runtime_error("cuCascade stub: small_pinned allocate");
  }
  void deallocate(rmm::cuda_stream_view, void*, std::size_t, std::size_t = 0) {}

  operator rmm::device_async_resource_ref() const {
    throw std::runtime_error("cuCascade stub: no upstream");
  }
};

}  // namespace cucascade::memory
