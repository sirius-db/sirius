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

#include <atomic>
#include <memory>

// Include existing reservation system
#include "memory/common.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/notification_channel.hpp"

namespace sirius {
namespace memory {

class disk_access_limiter {
 public:
  struct disk_reserved_arena : public reserved_arena {
    friend class disk_access_limiter;

    explicit disk_reserved_arena(disk_access_limiter& mr,
                                 std::size_t bytes,
                                 std::string_view base_fname,
                                 std::unique_ptr<event_notifier> notifer)
      : reserved_arena(bytes, std::move(notifer)),
        base_name_(base_fname.data(), base_fname.size()),
        mr_(&mr)
    {
    }

    ~disk_reserved_arena() noexcept { mr_->do_release_reservation(this); }

    std::string_view base_name() const noexcept { return base_name_; }

    bool grow_by(std::size_t) final { return false; }

    void shrink_to_fit() final {}

   private:
    std::string base_name_;
    disk_access_limiter* mr_;
  };

  explicit disk_access_limiter(memory_space_id space_id, std::size_t capacity);

  /**
   * @brief Constructs a per-stream tracking resource adaptor.
   *
   * @param upstream The upstream memory resource to wrap
   * @param capacity The total capacity for allocations
   * @param memory_limit The memory limit for reservations
   * @param tracking_scope [default: PER_STREAM] The scope of allocation tracking (per-stream,
   * per-thread)
   */
  explicit disk_access_limiter(memory_space_id space_id,
                               std::size_t memory_limit,
                               std::size_t capacity);

  /**
   * @brief Destructor.
   */
  ~disk_access_limiter() = default;

  /**
   * @brief Returns the available memory left in the resource
   */
  std::size_t get_available_memory() const noexcept;

  /**
   * @brief Gets the total reserved bytes across all streams.
   * @return The total reserved bytes
   */
  std::size_t get_total_reserved_bytes() const;

  /**
   * @brief Gets the total reserved bytes across all streams.
   * @return The total reserved bytes
   */
  std::size_t get_peak_reserved_bytes() const;

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

 private:
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
  void do_release_reservation(disk_reserved_arena* reservation) noexcept;

  void update_peak_allocated_bytes() noexcept;

  /**
   * @brief grows reservation by a `bytes` size
   * @param res current_reservation
   * @param bytes the size of reservation
   */
  bool grow_reservation_by(reserved_arena& res, std::size_t bytes);

  /**
   * @brief grows reservation by a `bytes` size
   * @param res current_reservation
   */
  void shrink_reservation_to_fit(reserved_arena& res);

  memory_space_id _space_id;

  /// The upstream memory resource
  const std::size_t _memory_limit;
  const std::size_t _capacity;

  /// Global totals for efficiency
  std::atomic<std::size_t> _total_allocated_bytes{0};
  std::atomic<std::size_t> _peak_total_allocated_bytes{0};
  std::atomic<std::size_t> _total_reservation_count{0};
};

}  // namespace memory
}  // namespace sirius
