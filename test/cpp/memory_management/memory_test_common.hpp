/*
 * Common test utilities for Sirius memory tests
 */

#pragma once

#include <memory>
#include <vector>

// RMM includes for creating test allocators
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

// Sirius memory components
#include "memory/null_device_memory_resource.hpp"
#include "memory/numa_region_pinned_host_allocator.hpp"

namespace sirius {
namespace memory {

// Helper function to create test allocators for a given tier
inline std::unique_ptr<rmm::mr::device_memory_resource> create_test_allocators(Tier tier)
{
  switch (tier) {
    case Tier::GPU: {
      // Use cuda_async_memory_resource for GPU tier
      return std::make_unique<rmm::mr::cuda_async_memory_resource>();
    }
    case Tier::HOST: {
      // Use a predictable fixed-size host memory resource for tests (e.g., 10MB)
      return std::make_unique<sirius::memory::numa_region_pinned_host_memory_resource>(-1);
    }
    case Tier::DISK: {
      // DISK tier uses a null allocator to satisfy API without real allocations
      return std::make_unique<null_device_memory_resource>();
    }
    default: throw std::invalid_argument("Unknown tier type");
  }

  return nullptr;
}

}  // namespace memory
}  // namespace sirius
