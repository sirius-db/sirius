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
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cuda_runtime_api.h>

// Include existing reservation system
#include "memory/memory_reservation.hpp"

namespace sirius {
namespace memory {

/**
 * @brief A memory resource adaptor that tracks allocations on a per-stream basis.
 * 
 * This adaptor wraps another device memory resource and provides detailed tracking
 * of allocations per CUDA stream. It maintains both current allocated bytes and 
 * the maximum allocated bytes observed for each stream.
 * 
 * Features:
 * - Per-stream allocation tracking
 * - Maximum allocated bytes tracking per stream
 * - Thread-safe operations using atomic operations and mutexes
 * - Reset capability for maximum allocated bytes
 * - Race condition handling for deallocations after reset
 * 
 * Based on RMM's tracking_resource_adaptor but extended for per-stream tracking.
 */
class per_stream_tracking_resource_adaptor : public rmm::mr::device_memory_resource {
public:
    /**
     * @brief Constructs a per-stream tracking resource adaptor.
     * 
     * @param upstream The upstream memory resource to wrap
     */
    explicit per_stream_tracking_resource_adaptor(
        std::unique_ptr<rmm::mr::device_memory_resource> upstream);

    /**
     * @brief Destructor.
     */
    ~per_stream_tracking_resource_adaptor() override = default;

    // Non-copyable and non-movable to ensure resource stability
    per_stream_tracking_resource_adaptor(const per_stream_tracking_resource_adaptor&) = delete;
    per_stream_tracking_resource_adaptor& operator=(const per_stream_tracking_resource_adaptor&) = delete;
    per_stream_tracking_resource_adaptor(per_stream_tracking_resource_adaptor&&) = delete;
    per_stream_tracking_resource_adaptor& operator=(per_stream_tracking_resource_adaptor&&) = delete;

    /**
     * @brief Gets the upstream memory resource.
     * @return Reference to the upstream resource
     */
    rmm::mr::device_memory_resource& get_upstream_resource() const noexcept;

    /**
     * @brief Gets the currently allocated bytes for a specific stream.
     * @param stream The CUDA stream to query
     * @return The allocated bytes for the stream
     */
    std::size_t get_allocated_bytes(rmm::cuda_stream_view stream) const;

    /**
     * @brief Gets the peak allocated bytes observed for a specific stream.
     * @param stream The CUDA stream to query
     * @return The peak allocated bytes for the stream
     */
    std::size_t get_peak_allocated_bytes(rmm::cuda_stream_view stream) const;

    /**
     * @brief Gets the total currently allocated bytes across all streams.
     * @return The total allocated bytes
     */
    std::size_t get_total_allocated_bytes() const;

    /**
     * @brief Gets the peak total allocated bytes across all streams.
     * @return The peak total allocated bytes
     */
    std::size_t get_peak_total_allocated_bytes() const;

    /**
     * @brief Resets the peak allocated bytes for a specific stream to 0.
     * @param stream The CUDA stream to reset
     */
    void reset_peak_allocated_bytes(rmm::cuda_stream_view stream);

    /**
     * @brief Resets the peak allocated bytes for all streams to 0.
     */
    void reset_all_peak_allocated_bytes();

    /**
     * @brief Gets all streams that have been tracked.
     * @return Vector of stream views that have allocations
     */
    std::vector<rmm::cuda_stream_view> get_tracked_streams() const;

    /**
     * @brief Checks if a stream is currently being tracked.
     * @param stream The CUDA stream to check
     * @return true if the stream is tracked, false otherwise
     */
    bool is_stream_tracked(rmm::cuda_stream_view stream) const;

    //===----------------------------------------------------------------------===//
    // Reservation Management
    //===----------------------------------------------------------------------===//

    /**
     * @brief Sets the memory reservation for a specific stream by requesting from the memory manager.
     * @param stream The CUDA stream to set reservation for
     * @param reservation_request The reservation request specifying which memory space to use
     * @param reservation_bytes The reservation size in bytes (0 = remove reservation)
     * @return true if reservation was successfully set, false otherwise
     */
    bool set_stream_reservation(rmm::cuda_stream_view stream, 
                               const ReservationRequest& reservation_request, 
                               std::size_t reservation_bytes);

    /**
     * @brief Gets the memory reservation size for a specific stream.
     * @param stream The CUDA stream to query
     * @return The reservation size in bytes (0 = no reservation)
     */
    std::size_t get_stream_reservation_size(rmm::cuda_stream_view stream) const;

    /**
     * @brief Gets the reservation object for a specific stream.
     * @param stream The CUDA stream to query
     * @return Pointer to the reservation (may be null if no reservation set)
     */
    const Reservation* get_stream_reservation(rmm::cuda_stream_view stream) const;

    /**
     * @brief Sets the reservation policy for a specific stream.
     * @param stream The CUDA stream to set policy for
     * @param policy The reservation policy to use (takes ownership)
     */
    void set_stream_policy(rmm::cuda_stream_view stream, std::unique_ptr<ReservationLimitPolicy> policy);

    /**
     * @brief Gets the reservation policy for a specific stream.
     * @param stream The CUDA stream to query
     * @return Reference to the policy (never null)
     */
    const ReservationLimitPolicy& get_stream_policy(rmm::cuda_stream_view stream) const;

    /**
     * @brief Gets the name of the reservation policy for a specific stream.
     * @param stream The CUDA stream to query
     * @return Policy name string
     */
    std::string get_stream_policy_name(rmm::cuda_stream_view stream) const;

    /**
     * @brief Sets the default reservation policy for new streams.
     * @param policy The default policy to use (takes ownership)
     */
    void set_default_policy(std::unique_ptr<ReservationLimitPolicy> policy);

    /**
     * @brief Gets the default reservation policy.
     * @return Reference to the default policy
     */
    const ReservationLimitPolicy& get_default_policy() const;

private:
    /**
     * @brief Stream tracking information.
     */
    struct StreamStats {
        std::atomic<std::size_t> allocated_bytes{0};      ///< Current allocated bytes
        std::atomic<std::size_t> peak_allocated_bytes{0}; ///< Peak allocated bytes observed
        std::unique_ptr<Reservation> reservation;         ///< Stream memory reservation (may be null)
        std::unique_ptr<ReservationLimitPolicy> policy;        ///< Reservation policy for this stream

        StreamStats() = default;
        StreamStats(const StreamStats& other) 
            : allocated_bytes(other.allocated_bytes.load())
            , peak_allocated_bytes(other.peak_allocated_bytes.load()) {
            // Note: reservation and policy are not copied, will be set separately if needed
        }
        
        // Move constructor
        StreamStats(StreamStats&& other) noexcept
            : allocated_bytes(other.allocated_bytes.load())
            , peak_allocated_bytes(other.peak_allocated_bytes.load())
            , reservation(std::move(other.reservation))
            , policy(std::move(other.policy)) {}
        
        // Move assignment
        StreamStats& operator=(StreamStats&& other) noexcept {
            if (this != &other) {
                allocated_bytes.store(other.allocated_bytes.load());
                peak_allocated_bytes.store(other.peak_allocated_bytes.load());
                reservation = std::move(other.reservation);
                policy = std::move(other.policy);
            }
            return *this;
        }
        
        // Delete copy assignment since we have unique_ptr
        StreamStats& operator=(const StreamStats&) = delete;
    };

    /**
     * @brief Allocates memory from the upstream resource and tracks it.
     * 
     * @param bytes The number of bytes to allocate
     * @param stream The CUDA stream to use for the allocation
     * @return Pointer to allocated memory
     */
    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override;

    /**
     * @brief Deallocates previously allocated memory and updates tracking.
     * 
     * @param ptr Pointer to memory to deallocate
     * @param bytes The number of bytes that were allocated
     * @param stream The CUDA stream to use for the deallocation
     */
    void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override;

    /**
     * @brief Checks equality with another memory resource.
     * 
     * @param other The other memory resource to compare with
     * @return true if this resource is the same as other
     */
    bool do_is_equal(const rmm::mr::device_memory_resource& other) const noexcept override;

    /**
     * @brief Gets or creates stream statistics for the given stream.
     * @param stream The CUDA stream
     * @return Reference to the stream statistics
     */
    StreamStats& get_stream_stats(rmm::cuda_stream_view stream);

    /**
     * @brief Gets stream statistics for the given stream (const version).
     * @param stream The CUDA stream
     * @return Pointer to stream statistics, or nullptr if not found
     */
    const StreamStats* get_stream_stats_const(rmm::cuda_stream_view stream) const;

    /// The upstream memory resource
    std::unique_ptr<rmm::mr::device_memory_resource> upstream_;
    
    /// Per-stream allocation tracking
    mutable std::mutex streams_mutex_;
    std::unordered_map<cudaStream_t, std::unique_ptr<StreamStats>> stream_stats_;
    
    /// Global totals for efficiency
    std::atomic<std::size_t> total_allocated_bytes_{0};
    std::atomic<std::size_t> peak_total_allocated_bytes_{0};
    
    /// Default policy for new streams
    std::unique_ptr<ReservationLimitPolicy> default_policy_;
    
    /// Helper method to check if allocation would exceed reservation and handle policy
    void check_reservation_and_handle_policy(rmm::cuda_stream_view stream, 
                                           std::size_t requested_bytes,
                                           StreamStats& stats);
};

} // namespace memory
} // namespace sirius
