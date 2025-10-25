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

#pragma once

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/aligned.hpp>
#include <atomic>
#include <cstdlib>

namespace sirius {
namespace memory {

/**
 * @brief A fixed-size host memory resource that implements rmm::mr::device_memory_resource interface.
 * 
 * This memory resource provides a fixed pool of host memory with thread-safe allocation tracking.
 * It's designed to work within the RMM ecosystem while managing host memory allocations.
 * 
 * Based on the implementation from:
 * https://github.com/felipeblazing/memory_spilling/blob/main/include/fixed_size_host_memory_resource.hpp
 * Modified to derive from device_memory_resource instead of host_memory_resource.
 * 
 * Features:
 * - Thread-safe allocation/deallocation using atomic operations
 * - Fixed total memory pool size with overflow protection
 * - Proper alignment handling for memory allocations
 * - Exception-safe with proper rollback on allocation failures
 * 
 * @note This resource allocates host memory using std::aligned_alloc, which provides
 *       aligned memory suitable for efficient CPU access patterns.
 */
class fixed_size_host_memory_resource : public rmm::mr::device_memory_resource {
public:
    /**
     * @brief Constructs a fixed-size host memory resource.
     * 
     * @param total_size The total size of the memory pool in bytes. Must be greater than 0.
     * @throws rmm::out_of_memory if total_size is 0
     */
    explicit fixed_size_host_memory_resource(std::size_t total_size);

    /**
     * @brief Destructor.
     */
    ~fixed_size_host_memory_resource() override = default;

    // Non-copyable and non-movable to ensure resource stability
    fixed_size_host_memory_resource(const fixed_size_host_memory_resource&) = delete;
    fixed_size_host_memory_resource& operator=(const fixed_size_host_memory_resource&) = delete;
    fixed_size_host_memory_resource(fixed_size_host_memory_resource&&) = delete;
    fixed_size_host_memory_resource& operator=(fixed_size_host_memory_resource&&) = delete;

    /**
     * @brief Gets the total size of the memory pool.
     * @return The total size in bytes
     */
    std::size_t get_total_size() const noexcept { return total_size_; }

    /**
     * @brief Gets the currently allocated size.
     * @return The allocated size in bytes
     */
    std::size_t get_allocated_size() const noexcept { return allocated_size_.load(); }

    /**
     * @brief Gets the available size in the memory pool.
     * @return The available size in bytes
     */
    std::size_t get_available_size() const noexcept { 
        return total_size_ - allocated_size_.load(); 
    }

    /**
     * @brief Checks if the resource can allocate the requested size.
     * @param bytes The number of bytes to check
     * @return true if allocation would succeed, false otherwise
     */
    bool can_allocate(std::size_t bytes) const noexcept {
        return allocated_size_.load() + bytes <= total_size_;
    }

private:
    /**
     * @brief Allocates memory from the fixed-size pool.
     * 
     * @param bytes The number of bytes to allocate
     * @param stream The CUDA stream to use for the allocation
     * @return Pointer to allocated memory
     * @throws rmm::out_of_memory if allocation fails (insufficient memory or system allocation failure)
     */
    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;

    /**
     * @brief Deallocates previously allocated memory.
     * 
     * @param ptr Pointer to memory to deallocate
     * @param bytes The number of bytes that were allocated
     * @param stream The CUDA stream to use for the deallocation
     */
    void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) override;

    /**
     * @brief Checks equality with another memory resource.
     * 
     * @param other The other memory resource to compare with
     * @return true if this resource is the same as other
     */
    bool do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept override;

private:
    const std::size_t total_size_;           ///< Total size of the memory pool
    std::atomic<std::size_t> allocated_size_; ///< Currently allocated size (thread-safe)
};

} // namespace memory
} // namespace sirius
