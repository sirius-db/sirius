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
#include <string>

namespace sirius {
namespace memory {

fixed_size_host_memory_resource::fixed_size_host_memory_resource(
    std::size_t block_size,
    std::size_t pool_size,
    std::size_t initial_pools)
    : block_size_(rmm::align_up(block_size, alignof(std::max_align_t))),
      pool_size_(pool_size)
{
    for (std::size_t i = 0; i < initial_pools; ++i) {
        expand_pool();
    }
}

fixed_size_host_memory_resource::fixed_size_host_memory_resource(
    std::unique_ptr<rmm::mr::pinned_host_memory_resource> upstream_mr,
    std::size_t block_size,
    std::size_t pool_size,
    std::size_t initial_pools)
    : block_size_(rmm::align_up(block_size, alignof(std::max_align_t))),
      pool_size_(pool_size),
      upstream_mr_(std::move(upstream_mr))
{
    for (std::size_t i = 0; i < initial_pools; ++i) {
        expand_pool();
    }
}

fixed_size_host_memory_resource::~fixed_size_host_memory_resource() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : allocated_blocks_) {
        const std::size_t dealloc_size = block_size_ * pool_size_;
        
        if (upstream_mr_) {
            upstream_mr_->deallocate(block, dealloc_size);
        } else {
            rmm::mr::pinned_host_memory_resource::deallocate(block, dealloc_size);
        }
    }
    allocated_blocks_.clear();
    free_blocks_.clear();
}

std::size_t fixed_size_host_memory_resource::get_block_size() const noexcept {
    return block_size_;
}

std::size_t fixed_size_host_memory_resource::get_free_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
}

std::size_t fixed_size_host_memory_resource::get_total_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_blocks_.size() * pool_size_;
}

rmm::mr::pinned_host_memory_resource* fixed_size_host_memory_resource::get_upstream_resource() const noexcept {
    return upstream_mr_.get();
}

fixed_size_host_memory_resource::multiple_blocks_allocation fixed_size_host_memory_resource::allocate_multiple_blocks(std::size_t total_bytes) {
    RMM_FUNC_RANGE();
    
    if (total_bytes == 0) {
        return multiple_blocks_allocation({}, this, block_size_);
    }

    const std::size_t num_blocks = (total_bytes + block_size_ - 1) / block_size_;
    
    std::vector<void*> allocated_blocks;
    allocated_blocks.reserve(num_blocks);

    std::lock_guard<std::mutex> lock(mutex_);

    for (std::size_t i = 0; i < num_blocks; ++i) {
        if (free_blocks_.empty()) {
            expand_pool();
        }

        if (free_blocks_.empty()) {
            // Cleanup on failure
            for (void* ptr : allocated_blocks) {
                free_blocks_.push_back(ptr);
            }
            throw rmm::out_of_memory("Not enough free blocks available in fixed_size_host_memory_resource and pool expansion failed.");
        }

        void* ptr = free_blocks_.back();
        free_blocks_.pop_back();
        allocated_blocks.push_back(ptr);
    }

    return multiple_blocks_allocation(std::move(allocated_blocks), this, block_size_);
}

void* fixed_size_host_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) {
    RMM_FUNC_RANGE();
    
    if (bytes == 0) {
        return nullptr;
    }
    RMM_EXPECTS(
        bytes <= block_size_,
        (std::string("Allocation size exceeds block size: requested=") + std::to_string(bytes) +
         " bytes, block_size=" + std::to_string(block_size_) + " bytes")
            .c_str());

    std::lock_guard<std::mutex> lock(mutex_);
    
    if (free_blocks_.empty()) {
        expand_pool();
    }

    if (free_blocks_.empty()) {
        throw rmm::out_of_memory("No free blocks available in fixed_size_host_memory_resource.");
    }

    void* ptr = free_blocks_.back();
    free_blocks_.pop_back();
    
    return ptr;
}

void fixed_size_host_memory_resource::do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) {
    RMM_FUNC_RANGE();
    
    if (ptr == nullptr) {
        return;
    }

    if (bytes > block_size_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.push_back(ptr);
}

bool fixed_size_host_memory_resource::do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept {
    return this == &other;
}

void fixed_size_host_memory_resource::expand_pool() {
    const std::size_t total_size = block_size_ * pool_size_;
    
    void* large_allocation;
    if (upstream_mr_) {
        large_allocation = upstream_mr_->allocate(total_size);
    } else {
        large_allocation = rmm::mr::pinned_host_memory_resource::allocate(total_size);
    }
    
    allocated_blocks_.push_back(large_allocation);
    
    for (std::size_t i = 0; i < pool_size_; ++i) {
        void* block = static_cast<char*>(large_allocation) + (i * block_size_);
        free_blocks_.push_back(block);
    }
}

} // namespace memory
} // namespace sirius