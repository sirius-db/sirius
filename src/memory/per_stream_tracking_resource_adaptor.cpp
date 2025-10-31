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

#include "memory/per_stream_tracking_resource_adaptor.hpp"
#include <rmm/detail/error.hpp>
#include <algorithm>

namespace sirius {
namespace memory {

per_stream_tracking_resource_adaptor::per_stream_tracking_resource_adaptor(
    std::unique_ptr<rmm::mr::device_memory_resource> upstream)
    : upstream_(std::move(upstream)) {
    
    RMM_EXPECTS(upstream_ != nullptr, "Upstream resource cannot be null");
}

rmm::mr::device_memory_resource& per_stream_tracking_resource_adaptor::get_upstream_resource() const noexcept {
    return *upstream_;
}

std::size_t per_stream_tracking_resource_adaptor::get_allocated_bytes(rmm::cuda_stream_view stream) const {
    const auto* stats = get_stream_stats_const(stream);
    return stats ? stats->allocated_bytes.load() : 0;
}

std::size_t per_stream_tracking_resource_adaptor::get_peak_allocated_bytes(rmm::cuda_stream_view stream) const {
    const auto* stats = get_stream_stats_const(stream);
    return stats ? stats->peak_allocated_bytes.load() : 0;
}

std::size_t per_stream_tracking_resource_adaptor::get_total_allocated_bytes() const {
    return total_allocated_bytes_.load();
}

std::size_t per_stream_tracking_resource_adaptor::get_peak_total_allocated_bytes() const {
    return peak_total_allocated_bytes_.load();
}

void per_stream_tracking_resource_adaptor::reset_peak_allocated_bytes(rmm::cuda_stream_view stream) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = stream_stats_.find(stream.value());
    if (it != stream_stats_.end()) {
        it->second->peak_allocated_bytes.store(0);
    }
}

void per_stream_tracking_resource_adaptor::reset_all_peak_allocated_bytes() {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    for (auto& [stream_id, stats] : stream_stats_) {
        stats->peak_allocated_bytes.store(0);
    }
    
    // Also reset the global peak
    peak_total_allocated_bytes_.store(0);
}

std::vector<rmm::cuda_stream_view> per_stream_tracking_resource_adaptor::get_tracked_streams() const {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    std::vector<rmm::cuda_stream_view> streams;
    streams.reserve(stream_stats_.size());
    
    for (const auto& [stream_id, stats] : stream_stats_) {
        streams.emplace_back(rmm::cuda_stream_view{stream_id});
    }
    
    return streams;
}

bool per_stream_tracking_resource_adaptor::is_stream_tracked(rmm::cuda_stream_view stream) const {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    return stream_stats_.find(stream.value()) != stream_stats_.end();
}

void* per_stream_tracking_resource_adaptor::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) {
    // Perform the upstream allocation first
    void* ptr = upstream_->allocate(bytes, stream);
    
    try {
        // Update stream-specific tracking
        auto& stats = get_stream_stats(stream);
        std::size_t new_allocated = stats.allocated_bytes.fetch_add(bytes) + bytes;
        
        // Update peak if this is a new peak for the stream
        std::size_t current_peak = stats.peak_allocated_bytes.load();
        while (new_allocated > current_peak) {
            if (stats.peak_allocated_bytes.compare_exchange_weak(current_peak, new_allocated)) {
                break;
            }
        }
        
        // Update global totals
        std::size_t new_total = total_allocated_bytes_.fetch_add(bytes) + bytes;
        
        // Update global peak if this is a new peak
        std::size_t current_peak = peak_total_allocated_bytes_.load();
        while (new_total > current_peak) {
            if (peak_total_allocated_bytes_.compare_exchange_weak(current_peak, new_total)) {
                break;
            }
        }
        
    } catch (...) {
        // If tracking fails, deallocate the upstream allocation to maintain consistency
        upstream_->deallocate(ptr, bytes, stream);
        throw;
    }
    
    return ptr;
}

void per_stream_tracking_resource_adaptor::do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept {
    // Deallocate from upstream first
    upstream_->deallocate(ptr, bytes, stream);
    
    // Update stream-specific tracking
    auto& stats = get_stream_stats(stream);
    std::size_t current_allocated = stats.allocated_bytes.load();
    std::size_t new_allocated;
    
    do {
        // Handle potential race condition where deallocation might go negative
        // (e.g., if reset was called and then a deallocation came in)
        if (current_allocated < bytes) {
            new_allocated = 0;
        } else {
            new_allocated = current_allocated - bytes;
        }
    } while (!stats.allocated_bytes.compare_exchange_weak(current_allocated, new_allocated));
    
    // Update global total (with similar protection against negative values)
    std::size_t current_total = total_allocated_bytes_.load();
    std::size_t new_total;
    
    do {
        if (current_total < bytes) {
            new_total = 0;
        } else {
            new_total = current_total - bytes;
        }
    } while (!total_allocated_bytes_.compare_exchange_weak(current_total, new_total));
}

bool per_stream_tracking_resource_adaptor::do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept {
    // Check if it's the same type
    const auto* other_adaptor = dynamic_cast<const per_stream_tracking_resource_adaptor*>(&other);
    if (other_adaptor == nullptr) {
        return false;
    }
    
    // Check if the upstream resources are equal
    return upstream_->is_equal(other_adaptor->get_upstream_resource());
}

per_stream_tracking_resource_adaptor::StreamStats& 
per_stream_tracking_resource_adaptor::get_stream_stats(rmm::cuda_stream_view stream) {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = stream_stats_.find(stream.value());
    if (it == stream_stats_.end()) {
        // Create new stats for this stream
        auto stats = std::make_unique<StreamStats>();
        auto* stats_ptr = stats.get();
        stream_stats_[stream.value()] = std::move(stats);
        return *stats_ptr;
    }
    
    return *it->second;
}

const per_stream_tracking_resource_adaptor::StreamStats* 
per_stream_tracking_resource_adaptor::get_stream_stats_const(rmm::cuda_stream_view stream) const {
    std::lock_guard<std::mutex> lock(streams_mutex_);
    
    auto it = stream_stats_.find(stream.value());
    if (it == stream_stats_.end()) {
        return nullptr;
    }
    
    return it->second.get();
}

} // namespace memory
} // namespace sirius
