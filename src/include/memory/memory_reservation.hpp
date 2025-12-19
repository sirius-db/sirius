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

#include "memory/common.hpp"
#include "memory/notification_channel.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <concepts>
#include <memory>
#include <string>
#include <utility>

namespace cucascade {
namespace memory {

// Forward declarations
class reservation_aware_resource_adaptor;
class fixed_size_host_memory_resource;
class disk_access_limiter;
struct reservation;
struct reserved_arena;
class memory_space;

template <Tier TIER>
struct tier_memory_resource_trait {
  using upstream_type = rmm::mr::device_memory_resource;
  using type          = rmm::mr::device_memory_resource;
  Tier tier           = TIER;
};

template <>
struct tier_memory_resource_trait<Tier::HOST> {
  using upstream_type = rmm::mr::device_memory_resource;
  using type          = fixed_size_host_memory_resource;
  Tier tier           = Tier::HOST;
};

template <>
struct tier_memory_resource_trait<Tier::GPU> {
  using upstream_type = rmm::mr::device_memory_resource;
  using type          = reservation_aware_resource_adaptor;
  Tier tier           = Tier::GPU;
};

//===----------------------------------------------------------------------===//
// Reservation Limit Policy Interface
//===----------------------------------------------------------------------===//

/**
 * @brief Base class for reservation limit policies that control behavior when stream reservations
 * are exceeded.
 *
 * Reservation limit policies are pluggable strategies that determine what happens when an
 * allocation would cause a stream's memory usage to exceed its reservation limit.
 */
class reservation_limit_policy {
 public:
  virtual ~reservation_limit_policy();

  /**
   * @brief Handle an allocation that would exceed the stream's reservation.
   *
   * This method is called when an allocation would cause the current allocated bytes
   * plus the new allocation to exceed the stream's reservation. The policy can:
   * 1. Allow the allocation to proceed (ignore policy)
   * 2. Increase the reservation and allow the allocation (increase policy)
   * 3. Throw an exception to prevent the allocation (fail policy)
   *
   * @param adaptor Reference to the tracking resource adaptor
   * @param stream The stream that would exceed its reservation
   * @param requested_bytes Number of bytes being requested
   * @param current_allocated Number of bytes currently allocated
   * @param reserved_bytes Pointer to the reservation object
   * @throws rmm::out_of_memory if the policy decides to reject the allocation
   */
  virtual void handle_over_reservation(rmm::cuda_stream_view stream,
                                       std::size_t requested_bytes,
                                       std::size_t current_allocated,
                                       reserved_arena* reserved_bytes) = 0;

  /**
   * @brief Get a human-readable name for this policy.
   * @return Policy name string
   */
  virtual std::string get_policy_name() const = 0;
};

/**
 * @brief Ignore policy - allows allocations to proceed even if they exceed reservations.
 *
 * This policy simply ignores reservation limits and allows all allocations to proceed.
 * It's useful for soft reservations where you want to track usage but not enforce limits.
 */
class ignore_reservation_limit_policy : public reservation_limit_policy {
 public:
  ignore_reservation_limit_policy();

  void handle_over_reservation(rmm::cuda_stream_view stream,
                               std::size_t requested_bytes,
                               std::size_t current_allocated,
                               reserved_arena* reserved_bytes) final;

  std::string get_policy_name() const override;
};

/**
 * @brief Fail policy - throws an exception when reservations are exceeded.
 *
 * This policy enforces strict reservation limits by throwing rmm::out_of_memory
 * when an allocation would exceed the stream's reservation.
 */
class fail_reservation_limit_policy : public reservation_limit_policy {
 public:
  fail_reservation_limit_policy();

  void handle_over_reservation(rmm::cuda_stream_view stream,
                               std::size_t requested_bytes,
                               std::size_t current_allocated,
                               reserved_arena* reserved_bytes) final;

  std::string get_policy_name() const override;
};

/**
 * @brief remaining policy - automatically reserves remaining memory up to a limit.
 *
 * This policy implements a best effort policy that reserves the stream's reservation upto the
 * specified limit.
 */
class increase_reservation_limit_policy : public reservation_limit_policy {
 public:
  increase_reservation_limit_policy();

  /**
   * @brief Constructs an increase policy with the specified padding factor.
   */
  explicit increase_reservation_limit_policy(double padding_factor,
                                             bool allow_beyond_limit = false);

  void handle_over_reservation(rmm::cuda_stream_view stream,
                               std::size_t requested_bytes,
                               std::size_t current_allocated,
                               reserved_arena* reserved_bytes) override;

  std::string get_policy_name() const override;

 private:
  double _padding_factor{1.25f};               ///< Padding factor when increasing reservations
  bool allow_reservation_beyond_limit{false};  ///< Allow reservation beyond limit
};

std::unique_ptr<reservation_limit_policy> make_default_reservation_limit_policy();

//===----------------------------------------------------------------------===//
// Reservation
//===----------------------------------------------------------------------===//

struct reserved_arena {
  friend class reservation_aware_resource_adaptor;
  friend class fixed_size_host_memory_resource;
  friend class disk_access_limiter;

  explicit reserved_arena(int64_t len, std::unique_ptr<event_notifier> release_notifer = nullptr)
    : size_(len), on_exit_(std::move(release_notifer))
  {
  }

  virtual ~reserved_arena() = default;

  virtual bool grow_by(std::size_t additional_bytes) = 0;

  virtual void shrink_to_fit() = 0;

  [[gnu::always_inline]] int64_t size() const noexcept { return size_; }

 private:
  int64_t size_;
  const notify_on_exit on_exit_;
};

/**
 * Represents a memory reservation in a specific memory space.
 * Contains only the essential identifying information (tier, device_id, size).
 * The actual memory_space can be obtained through the memory_reservation_manager.
 */
struct reservation {
  friend class reservation_aware_resource_adaptor;
  friend class fixed_size_host_memory_resource;
  friend class disk_access_limiter;

  static std::unique_ptr<reservation> create(memory_space& space,
                                             std::unique_ptr<reserved_arena> arena);

  size_t size() const noexcept;

  [[nodiscard]] Tier tier() const noexcept;

  [[nodiscard]] int device_id() const noexcept;

  [[nodiscard]] rmm::mr::device_memory_resource* get_memory_resource() const noexcept;

  [[nodiscard]] const memory_space& get_memory_space() const noexcept;

  template <typename T>
    requires std::derived_from<T, rmm::mr::device_memory_resource>
  T* get_memory_resource_as() const noexcept
  {
    return dynamic_cast<T*>(get_memory_resource());
  }

  template <Tier TIER>
  auto* get_memory_resource_of() const noexcept
  {
    return get_memory_resource_as<typename tier_memory_resource_trait<TIER>::type>();
  }

  //===----------------------------------------------------------------------===//
  // Reservation Size Management
  //===----------------------------------------------------------------------===//

  /**
   * @brief Attempts to grow this reservation by additional bytes.
   * @param additional_bytes Number of bytes to add to the current reservation
   * @return true if the reservation was successfully grown, false otherwise
   */
  bool grow_by(size_t additional_bytes);

  /**
   * @brief Attempts to shrink this reservation to a new smaller size.
   * @param new_size The new size for the reservation (must be smaller than current size)
   * @return true if the reservation was successfully shrunk, false otherwise
   */
  void shrink_to_fit();

  // Disable copy/move to prevent issues with memory_space tracking
  reservation(const reservation&)            = delete;
  reservation& operator=(const reservation&) = delete;
  reservation(reservation&&)                 = delete;
  reservation& operator=(reservation&&)      = delete;

  ~reservation();

 private:
  explicit reservation(const memory_space* space, std::unique_ptr<reserved_arena> arena);

  const memory_space* space_;
  std::unique_ptr<reserved_arena> arena_;
};

}  // namespace memory
}  // namespace cucascade
