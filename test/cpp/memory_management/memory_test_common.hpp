/*
 * Common test utilities for Sirius memory tests
 */

#pragma once

#include <vector>
#include <memory>

// RMM includes for creating test allocators
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>

// Sirius memory components
#include "memory/memory_reservation.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "memory/null_device_memory_resource.hpp"

namespace sirius {
namespace memory {

// Helper function to create test allocators for a given tier
inline std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> create_test_allocators(Tier tier) {
    std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators;

    switch (tier) {
        case Tier::GPU: {
            // Use cuda_async_memory_resource for GPU tier
            auto cuda_async_allocator = std::make_unique<rmm::mr::cuda_async_memory_resource>();
            allocators.push_back(std::move(cuda_async_allocator));
            break;
        }
        case Tier::HOST: {
            // Use a predictable fixed-size host memory resource for tests (e.g., 10MB)
            auto host_allocator = std::make_unique<fixed_size_host_memory_resource>(10ull * 1024 * 1024);
            allocators.push_back(std::move(host_allocator));
            break;
        }
        case Tier::DISK: {
            // DISK tier uses a null allocator to satisfy API without real allocations
            auto disk_allocator = std::make_unique<null_device_memory_resource>();
            allocators.push_back(std::move(disk_allocator));
            break;
        }
        default:
            throw std::invalid_argument("Unknown tier type");
    }

    return allocators;
}

} // namespace memory
} // namespace sirius


