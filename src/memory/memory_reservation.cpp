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

#include "memory/memory_reservation.hpp"

#include "memory/common.hpp"
#include "memory/memory_reservation_manager.hpp"
#include "memory/memory_space.hpp"

#include <rmm/cuda_device.hpp>

namespace sirius {
namespace memory {

// Provide out-of-line definition for the virtual destructor to satisfy linker
reservation_limit_policy::~reservation_limit_policy() = default;

//===----------------------------------------------------------------------===//
// reservation Implementation
//===----------------------------------------------------------------------===//

std::unique_ptr<reservation> reservation::create(memory_space& space,
                                                 std::unique_ptr<reserved_arena> arena)
{
  return std::unique_ptr<reservation>(new reservation(&space, std::move(arena)));
}

reservation::reservation(const memory_space* space, std::unique_ptr<reserved_arena> arena)
  : space_(space), arena_(std::move(arena))
{
  assert(arena_ != nullptr && "Release callback must be provided");
}

reservation::~reservation() = default;

rmm::mr::device_memory_resource* reservation::get_memory_resource() const noexcept
{
  return space_->get_default_allocator();
}

const memory_space& reservation::get_memory_space() const noexcept { return *space_; }

size_t reservation::size() const noexcept { return arena_->size(); }

[[nodiscard]] Tier reservation::tier() const noexcept { return space_->get_tier(); }

[[nodiscard]] int reservation::device_id() const noexcept { return space_->get_device_id(); }

bool reservation::grow_by(size_t additional_bytes) { return arena_->grow_by(additional_bytes); }

void reservation::shrink_to_fit() { return arena_->shrink_to_fit(); }

//===----------------------------------------------------------------------===//
// Reservation Limit Policy Implementations
//===----------------------------------------------------------------------===//

ignore_reservation_limit_policy::ignore_reservation_limit_policy() = default;

void ignore_reservation_limit_policy::handle_over_reservation(
  [[maybe_unused]] rmm::cuda_stream_view stream,
  [[maybe_unused]] std::size_t requested_bytes,
  [[maybe_unused]] std::size_t current_allocated,
  [[maybe_unused]] reserved_arena* reserved_bytes)
{
  // do nothing
}

std::string ignore_reservation_limit_policy::get_policy_name() const { return "ignore"; }

fail_reservation_limit_policy::fail_reservation_limit_policy() = default;

void fail_reservation_limit_policy::handle_over_reservation(
  [[maybe_unused]] rmm::cuda_stream_view stream,
  std::size_t requested_bytes,
  std::size_t current_allocated,
  reserved_arena* reserved_bytes)
{
  std::size_t reservation_size = reserved_bytes ? reserved_bytes->size() : 0;
  RMM_FAIL("Allocation of " + std::to_string(requested_bytes) +
             " bytes would exceed stream reservation of " + std::to_string(reservation_size) +
             " bytes (current: " + std::to_string(current_allocated) + " bytes)",
           rmm::out_of_memory);
}

std::string fail_reservation_limit_policy::get_policy_name() const { return "fail"; }

increase_reservation_limit_policy::increase_reservation_limit_policy() = default;

increase_reservation_limit_policy::increase_reservation_limit_policy(double padding_factor,
                                                                     bool allow_beyond_limit)
  : _padding_factor(padding_factor), allow_reservation_beyond_limit(allow_beyond_limit)
{
}

void increase_reservation_limit_policy::handle_over_reservation(rmm::cuda_stream_view stream,
                                                                std::size_t requested_bytes,
                                                                std::size_t current_allocated,
                                                                reserved_arena* reserved_bytes)
{
  if (!reserved_bytes) { RMM_FAIL("No reservation set for stream", rmm::out_of_memory); }

  std::size_t post_allocation_size = current_allocated + requested_bytes;
  std::size_t extra_space_needed   = post_allocation_size - reserved_bytes->size();

  // Add padding to avoid frequent increases
  std::size_t padded_reservation = static_cast<std::size_t>(extra_space_needed * _padding_factor);

  // Try to grow the reservation
  if (!reserved_bytes->grow_by(padded_reservation)) {
    // If we can't grow to the padded size, try to grow to just what we need
    if (!reserved_bytes->grow_by(extra_space_needed)) {
      // If we can't even grow to what we need, throw an error
      RMM_FAIL("Failed to increase stream reservation from " +
                 std::to_string(reserved_bytes->size()) + " to " +
                 std::to_string(extra_space_needed) + " bytes",
               rmm::out_of_memory);
    }
  }
}

std::string increase_reservation_limit_policy::get_policy_name() const { return "increase"; }

std::unique_ptr<reservation_limit_policy> make_default_reservation_limit_policy()
{
  return std::make_unique<ignore_reservation_limit_policy>();
}

//===----------------------------------------------------------------------===//
// memory_reservation_manager Implementation
//===----------------------------------------------------------------------===//

}  // namespace memory
}  // namespace sirius
