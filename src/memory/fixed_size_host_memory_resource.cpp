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

#include "memory/fixed_size_host_memory_resource.hpp"
#include <rmm/bad_alloc.hpp>
#include <algorithm>
#include <cstdlib>

namespace sirius {
namespace memory {

fixed_size_host_memory_resource::fixed_size_host_memory_resource(std::size_t total_size)
    : total_size_(total_size), allocated_size_(0) {
    if (total_size == 0) {
        throw rmm::bad_alloc("Total size must be greater than 0");
    }
}

void* fixed_size_host_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) {
        return nullptr;
    }

    // Ensure minimum alignment of sizeof(void*) for portability
    alignment = std::max(alignment, sizeof(void*));
    
    // Align the size to the requested alignment
    std::size_t aligned_size = (bytes + alignment - 1) & ~(alignment - 1);
    
    // Thread-safe allocation size tracking with compare-and-swap loop
    std::size_t current_allocated = allocated_size_.load();
    while (true) {
        if (current_allocated + aligned_size > total_size_) {
            throw rmm::bad_alloc("Insufficient memory in fixed_size_host_memory_resource: "
                               "requested " + std::to_string(aligned_size) + 
                               " bytes, available " + std::to_string(total_size_ - current_allocated) + 
                               " bytes");
        }
        
        if (allocated_size_.compare_exchange_weak(current_allocated, 
                                                 current_allocated + aligned_size)) {
            break;
        }
        // current_allocated is automatically updated by compare_exchange_weak on failure
    }

    // Allocate aligned host memory
    void* ptr = std::aligned_alloc(alignment, aligned_size);
    if (!ptr) {
        // Rollback the allocation accounting on system allocation failure
        allocated_size_.fetch_sub(aligned_size);
        throw rmm::bad_alloc("Failed to allocate " + std::to_string(aligned_size) + 
                           " bytes of aligned host memory with alignment " + 
                           std::to_string(alignment));
    }
    
    return ptr;
}

void fixed_size_host_memory_resource::do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) {
    if (ptr == nullptr || bytes == 0) {
        return;
    }

    // Ensure minimum alignment matches what was used in allocation
    alignment = std::max(alignment, sizeof(void*));
    
    // Align the size to match what was allocated
    std::size_t aligned_size = (bytes + alignment - 1) & ~(alignment - 1);
    
    // Free the memory
    std::free(ptr);
    
    // Update allocation tracking (thread-safe)
    allocated_size_.fetch_sub(aligned_size);
}

bool fixed_size_host_memory_resource::do_is_equal(const rmm::device_memory_resource& other) const noexcept {
    // Two memory resources are equal if they are the same instance
    return this == &other;
}

} // namespace memory
} // namespace sirius
