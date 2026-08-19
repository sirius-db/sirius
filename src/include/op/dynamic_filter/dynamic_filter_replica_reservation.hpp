/*
 * Copyright 2026, Sirius Contributors.
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

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::op::detail {

/**
 * @brief Returns CuCascade's aligned charge for one allocation
 *
 * Reservations spanning multiple allocations must sum per-allocation aligned charges:
 * `sum(align_up(b_i)) != align_up(sum(b_i))`.
 */
[[nodiscard]] inline std::size_t tracked_replica_allocation_bytes(
  std::size_t allocation_bytes) noexcept
{
  return allocation_bytes == 0 ? 0
                               : rmm::align_up(allocation_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

/**
 * @brief RAII reservation attached to an allocator tracker
 *
 * Destruction detaches the tracker, releasing unused reserved capacity while live allocations
 * retain their accounting. Destroy on the host thread that acquired the reservation.
 */
class scoped_replica_reservation final {
 public:
  /**
   * @brief Attempts to reserve and attach destination capacity
   *
   * @throw std::invalid_argument if @p bytes is zero
   * @throw std::logic_error if @p target has no GPU reservation-aware allocator
   * @return An attached scope, or `std::nullopt` when capacity or tracker state rejects it
   */
  [[nodiscard]] static std::optional<scoped_replica_reservation> try_acquire(
    dynamic_filter_replica_space const& target, std::size_t bytes, rmm::cuda_stream_view stream)
  {
    if (bytes == 0) {
      throw std::invalid_argument(
        "[scoped_replica_reservation] a replica reservation must be non-empty");
    }

    auto reservation = target.get_gpu_space().make_reservation_or_null(bytes);
    if (!reservation) { return std::nullopt; }

    auto* allocator = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
    if (allocator == nullptr) {
      throw std::logic_error(
        "[scoped_replica_reservation] destination has no GPU reservation-aware allocator");
    }
    if (!allocator->attach_reservation_to_tracker(
          stream,
          std::move(reservation),
          std::make_unique<cucascade::memory::fail_reservation_limit_policy>())) {
      return std::nullopt;
    }

    scoped_replica_reservation scope{allocator, stream};
    return std::optional<scoped_replica_reservation>{std::move(scope)};
  }

  scoped_replica_reservation(scoped_replica_reservation const&)            = delete;
  scoped_replica_reservation& operator=(scoped_replica_reservation const&) = delete;
  scoped_replica_reservation& operator=(scoped_replica_reservation&&)      = delete;

  scoped_replica_reservation(scoped_replica_reservation&& other) noexcept
    : _allocator{std::exchange(other._allocator, nullptr)}, _stream{other._stream}
  {
  }

  ~scoped_replica_reservation()
  {
    if (_allocator != nullptr) { _allocator->reset_stream_reservation(_stream); }
  }

  [[nodiscard]] rmm::device_async_resource_ref allocator() const noexcept
  {
    return rmm::device_async_resource_ref{*_allocator};
  }

 private:
  scoped_replica_reservation(cucascade::memory::reservation_aware_resource_adaptor* allocator,
                             rmm::cuda_stream_view stream) noexcept
    : _allocator{allocator}, _stream{stream}
  {
  }

  cucascade::memory::reservation_aware_resource_adaptor* _allocator;
  rmm::cuda_stream_view _stream;
};

}  // namespace sirius::op::detail
