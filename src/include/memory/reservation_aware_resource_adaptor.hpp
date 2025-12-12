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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <memory>
#include <set>

// Include existing reservation system
#include "memory/common.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/notification_channel.hpp"
#include "memory/oom_handling_policy.hpp"

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
class reservation_aware_resource_adaptor : public rmm::mr::device_memory_resource {
 public:
  /**
   * @brief Reservation state
   */
  struct stream_ordered_tracker_state {
    std::atomic<std::int64_t> allocated_bytes{0};
    std::atomic<std::int64_t> peak_allocated_bytes{0};   /// Peak allocated bytes observed
    std::unique_ptr<reserved_arena> memory_reservation;  /// Stream memory reservation (may be null)
    std::unique_ptr<reservation_limit_policy>
      reservation_policy;                             /// Reservation policy for this stream
    std::unique_ptr<oom_handling_policy> oom_policy;  /// out-of-memory handling policy

    friend class reservation_aware_resource_adaptor;

    stream_ordered_tracker_state() = default;

    [[nodiscard]] std::size_t get_available_memory() const noexcept
    {
      auto current = allocated_bytes.load();
      return current < memory_reservation->size() ? memory_reservation->size() - current : 0UL;
    }

    /**
     * @brief Checks reservation and handles overflow for an allocation request.
     *
     * @param adaptor The reservation aware resource adaptor
     * @param allocation_size The size of the allocation request
     * @param stream The CUDA stream for the allocation
     * @return The tracking size for the allocation
     */
    std::size_t check_reservation_and_handle_overflow(reservation_aware_resource_adaptor& adaptor,
                                                      std::size_t allocation_size,
                                                      rmm::cuda_stream_view stream);

   private:
    mutable std::mutex _arbitration_mutex;
  };

  /**
   * @brief Container for reservation state management. [Per-stream or per-thread]
   */
  struct allocation_tracker_iface {
    virtual ~allocation_tracker_iface() = default;

    virtual void reset_tracker_state(rmm::cuda_stream_view stream) = 0;

    virtual void assign_reservation_to_tracker(rmm::cuda_stream_view stream,
                                               std::unique_ptr<reserved_arena> reservation,
                                               std::unique_ptr<reservation_limit_policy> policy,
                                               std::unique_ptr<oom_handling_policy> oom_policy) = 0;

    virtual stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) = 0;

    virtual const stream_ordered_tracker_state* get_tracker_state(
      rmm::cuda_stream_view stream) const = 0;
  };

  struct device_reserved_arena : public reserved_arena {
    friend class reservation_aware_resource_adaptor;

    explicit device_reserved_arena(reservation_aware_resource_adaptor& mr,
                                   std::size_t bytes,
                                   std::unique_ptr<event_notifier> notifer)
      : reserved_arena(bytes, std::move(notifer)), mr_(&mr)
    {
    }

    ~device_reserved_arena() noexcept { mr_->do_release_reservation(this); }

    const stream_ordered_tracker_state* get_tracker_or_null() const noexcept { return tracker_; }

    bool grow_by(std::size_t additional_bytes) final
    {
      return mr_->grow_reservation_by(*this, additional_bytes);
    }

    void shrink_to_fit() final { mr_->shrink_reservation_to_fit(*this); }

   private:
    void assign_tracker(const stream_ordered_tracker_state& tracker) { tracker_ = &tracker; }

    reservation_aware_resource_adaptor* mr_;
    const stream_ordered_tracker_state* tracker_{nullptr};
  };

  enum class AllocationTrackingScope {
    PER_STREAM,  // Track allocations separately for each stream
    PER_THREAD   // Track allocations separately for each host thread
  };

  /**
   * @brief Constructs a per-stream tracking resource adaptor.
   *
   * @param upstream The upstream memory resource to wrap
   * @param capacity The total capacity for allocations
   * @param tracking_scope [default: PER_STREAM] The scope of allocation tracking (per-stream,
   * per-thread)
   */
  explicit reservation_aware_resource_adaptor(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM);

  /**
   * @brief Constructs a per-stream tracking resource adaptor.
   *
   * @param upstream The upstream memory resource to wrap
   * @param capacity The total capacity for allocations
   * @param memory_limit The memory limit for reservations
   * @param tracking_scope [default: PER_STREAM] The scope of allocation tracking (per-stream,
   * per-thread)
   */
  explicit reservation_aware_resource_adaptor(
    memory_space_id space_id,
    rmm::device_async_resource_ref upstream,
    std::size_t memory_limit,
    std::size_t capacity,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> default_oom_policy             = nullptr,
    AllocationTrackingScope tracking_scope = AllocationTrackingScope::PER_STREAM);

  /**
   * @brief Destructor.
   */
  ~reservation_aware_resource_adaptor() override = default;

  // Non-copyable and non-movable to ensure resource stability
  reservation_aware_resource_adaptor(const reservation_aware_resource_adaptor&)            = delete;
  reservation_aware_resource_adaptor& operator=(const reservation_aware_resource_adaptor&) = delete;
  reservation_aware_resource_adaptor(reservation_aware_resource_adaptor&&)                 = delete;
  reservation_aware_resource_adaptor& operator=(reservation_aware_resource_adaptor&&)      = delete;

  /**
   * @brief Gets the upstream memory resource.
   * @return Reference to the upstream resource
   */
  rmm::device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Returns the available memory left in the resource
   */
  std::size_t get_available_memory() const noexcept;

  /**
   * @brief Returns the available memory left in the resource
   */
  std::size_t get_available_memory(rmm::cuda_stream_view stream) const noexcept;

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
   * @brief Gets the total reserved bytes across all streams.
   * @return The total reserved bytes
   */
  std::size_t get_total_reserved_bytes() const;

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
   * @brief makes reservations
   * @param bytes the size of reservation
   * @param on_release_notifer used to hook callbacks for when the reservation is released
   */
  std::unique_ptr<reserved_arena> reserve(
    std::size_t bytes, std::unique_ptr<event_notifier> release_notifer = nullptr);

  /**
   * @brief makes reservations
   * @param bytes the size of reservation
   * @param on_release_notifer used to hook callbacks for when the reservation is released
   */
  std::unique_ptr<reserved_arena> reserve_upto(
    std::size_t bytes, std::unique_ptr<event_notifier> release_notifer = nullptr);

  /**
   * @brief Return number of activate reservation count
   */
  std::size_t get_active_reservation_count() const noexcept;

  /**
   * @brief Sets the memory reservation for a specific stream by requesting from the memory manager.
   * @param stream The CUDA stream to set reservation for
   * @param reserved_bytes The reservation object (0 = remove reservation)
   * @param stream_oom_policy The OOM policy for the stream
   * @return true if reservation was successfully set, false otherwise
   */
  bool attach_reservation_to_tracker(
    rmm::cuda_stream_view stream,
    std::unique_ptr<reservation> reserved_bytes,
    std::unique_ptr<reservation_limit_policy> stream_reservation_policy = nullptr,
    std::unique_ptr<oom_handling_policy> stream_oom_policy              = nullptr);

  /**
   * @brief Rests the reservation object for a specific stream.
   * @param stream The CUDA stream to query
   */
  void reset_stream_reservation(rmm::cuda_stream_view stream);

  /**
   * @brief Sets the default reservation policy for new streams.
   * @param policy The default policy to use (takes ownership)
   */
  void set_default_policy(std::unique_ptr<reservation_limit_policy> policy);

  /**
   * @brief Gets the default reservation policy.
   * @return Reference to the default policy
   */
  const reservation_limit_policy& get_default_reservation_policy() const;

  /**
   * @brief Gets the default reservation policy.
   * @return Reference to the default policy
   */
  const oom_handling_policy& get_default_oom_handling_policy() const;

 private:
  /**
   * @brief grows reservation by a `bytes` size
   * @param res current_reservation
   * @param bytes the size of reservation
   */
  bool grow_reservation_by(device_reserved_arena& arena, std::size_t bytes);

  /**
   * @brief grows reservation by a `bytes` size
   * @param res current_reservation
   */
  void shrink_reservation_to_fit(device_reserved_arena& arena);

  /**
   * @brief Allocates memory from the upstream resource and tracks it, the oom in this method is not
   * handled by the oom handler.
   *
   * @param bytes The number of bytes to allocate
   * @param stream The CUDA stream to use for the allocation
   * @return Pointer to allocated memory
   */
  void* do_allocate_managed(std::size_t bytes, rmm::cuda_stream_view stream);

  /**
   * @brief Allocates memory from the upstream resource and tracks it, the oom in this method is not
   * handled by the oom handler.
   *
   * @param bytes The number of bytes to allocate
   * @param stream The CUDA stream to use for the allocation
   * @return Pointer to allocated memory
   */
  void* do_allocate_managed(std::size_t bytes,
                            stream_ordered_tracker_state* state,
                            rmm::cuda_stream_view stream);

  /**
   * @brief Allocates memory from the upstream resource and tracks it, the oom in this method is not
   * handled by the oom handler.
   *
   * @param bytes The number of bytes to allocate
   * @param tracking_bytes The number of bytes to track
   * @param stream The CUDA stream to use for the allocation
   * @return Pointer to allocated memory
   */
  void* do_allocate_unmanaged(std::size_t bytes,
                              std::size_t tracking_bytes,
                              rmm::cuda_stream_view stream);

  /**
   * @brief releases reservations and returns the unsed reservation back to allocator
   * @param reservation pointer to the reservation being released
   */
  bool do_reserve(std::size_t size_bytes, std::size_t limit_bytes);

  /**
   * @brief releases reservations and returns the unsed reservation back to allocator
   * @param reservation pointer to the reservation being released
   */
  std::size_t do_reserve_upto(std::size_t size_bytes, std::size_t limit_bytes);

  /**
   * @brief releases reservations and returns the unsed reservation back to allocator
   * @param reservation pointer to the reservation being released
   */
  void do_release_reservation(device_reserved_arena* reservation) noexcept;

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

  memory_space_id _space_id;

  /// The upstream memory resource
  rmm::device_async_resource_ref _upstream;
  const std::size_t _memory_limit;
  const std::size_t _capacity;

  /// Per-stream allocation tracking
  mutable std::mutex reservation_mutex;
  std::set<const device_reserved_arena*> reservation_views;
  std::unique_ptr<allocation_tracker_iface> _allocation_tracker;

  /// Global totals for efficiency
  std::atomic<std::size_t> _total_allocated_bytes{0};
  std::atomic<std::size_t> _peak_total_allocated_bytes{0};

  /// Default policy for new streams
  std::unique_ptr<reservation_limit_policy> _default_reservation_policy;
  std::unique_ptr<oom_handling_policy> _default_oom_policy;
};

}  // namespace memory
}  // namespace sirius
