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
 * @brief Return the number of bytes charged by CuCascade for one RMM allocation
 *
 * The reservation-aware allocator tracks each allocation at CUDA's allocation alignment. Callers
 * with multiple allocations must align each allocation separately before summing them.
 */
[[nodiscard]] inline std::size_t tracked_replica_allocation_bytes(
  std::size_t allocation_bytes) noexcept
{
  return allocation_bytes == 0 ? 0
                               : rmm::align_up(allocation_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

/**
 * @brief Scoped destination-space reservation for constructing one dynamic-filter replica
 *
 * @ref try_acquire first reserves capacity in the destination GPU memory space, then attaches that
 * reservation to the destination allocator's tracker. Allocations made through @ref allocator are
 * charged to the reservation instead of being counted a second time. Destruction resets the
 * tracker; CuCascade releases unused reserved capacity while retaining accounting for allocations
 * that remain live in the completed replica.
 *
 * @warning The scope must be destroyed on the same host thread that acquired it when CuCascade uses
 * per-thread reservation tracking.
 */
class scoped_replica_reservation final {
 public:
  /**
   * @brief Try to reserve and attach @p bytes in @p target's GPU memory space
   *
   * @throw std::invalid_argument if @p bytes is zero
   * @throw std::logic_error if @p target has no GPU reservation-aware allocator
   * @param[in] stream Stream the reservation is attached to
   * @return An attached scope, or `std::nullopt` when the destination reservation limit cannot
   * admit the request or the selected execution context already tracks another reservation
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

  /**
   * @brief The destination allocator backed by the attached reservation
   */
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
