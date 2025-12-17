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

#include "memory/reservation_aware_resource_adaptor.hpp"

#include "memory/common.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/notification_channel.hpp"

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace cucascade {
namespace memory {

using stream_ordered_tracker_state =
  reservation_aware_resource_adaptor::stream_ordered_tracker_state;

using device_reserved_arena = reservation_aware_resource_adaptor::device_reserved_arena;

namespace {

struct stream_ordered_allocation_tracker
  : public reservation_aware_resource_adaptor::allocation_tracker_iface {
  mutable std::mutex mutex;
  std::unordered_map<cudaStream_t, std::unique_ptr<stream_ordered_tracker_state>> stream_stats_map;

  stream_ordered_allocation_tracker() = default;

  void reset_tracker_state(rmm::cuda_stream_view stream) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return; }
    auto* res_pnt = it->second->memory_reservation.get();
    stream_stats_map.erase(stream.value());
  }

  void assign_reservation_to_tracker(rmm::cuda_stream_view stream,
                                     std::unique_ptr<device_reserved_arena> arena,
                                     std::unique_ptr<reservation_limit_policy> policy,
                                     std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it != stream_stats_map.end()) {
      throw rmm::logic_error("Stream already has reservation state set");
    }

    stream_stats_map[stream.value()] = std::make_unique<stream_ordered_tracker_state>(
      std::move(arena), std::move(policy), std::move(oom_policy));
  }

  stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return nullptr; }
    return it->second.get();
  }

  const stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) const override
  {
    std::lock_guard lock(mutex);
    auto it = stream_stats_map.find(stream.value());
    if (it == stream_stats_map.end()) { return nullptr; }
    return it->second.get();
  }
};

struct ptds_allocation_tracker
  : public reservation_aware_resource_adaptor::allocation_tracker_iface {
  static inline thread_local std::unique_ptr<stream_ordered_tracker_state> thread_reservation_state;

  ptds_allocation_tracker() = default;

  void reset_tracker_state(rmm::cuda_stream_view stream) override
  {
    if (thread_reservation_state) { thread_reservation_state.reset(); }
  }

  void assign_reservation_to_tracker(rmm::cuda_stream_view stream,
                                     std::unique_ptr<device_reserved_arena> arena,
                                     std::unique_ptr<reservation_limit_policy> policy,
                                     std::unique_ptr<oom_handling_policy> oom_policy) override
  {
    if (thread_reservation_state) {
      throw rmm::logic_error("Thread already has reservation state set");
    }

    thread_reservation_state = std::make_unique<stream_ordered_tracker_state>(
      std::move(arena), std::move(policy), std::move(oom_policy));
  }

  stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) override
  {
    return thread_reservation_state.get();
  }

  const stream_ordered_tracker_state* get_tracker_state(rmm::cuda_stream_view stream) const override
  {
    return thread_reservation_state.get();
  }
};

}  // namespace

stream_ordered_tracker_state::stream_ordered_tracker_state(
  std::unique_ptr<device_reserved_arena> arena,
  std::unique_ptr<reservation_limit_policy> res_policy,
  std::unique_ptr<oom_handling_policy> oom_policy)
  : memory_reservation(std::move(arena)),
    reservation_policy(std::move(res_policy)),
    oom_policy(std::move(oom_policy))
{
}

std::size_t reservation_aware_resource_adaptor::stream_ordered_tracker_state::
  check_reservation_and_handle_overflow(reservation_aware_resource_adaptor& adaptor,
                                        std::size_t allocation_size,
                                        rmm::cuda_stream_view stream)
{
  int64_t stream_tracking_size = static_cast<int64_t>(allocation_size);
  auto upstream_tracking_size  = stream_tracking_size;

  auto [success, post_allocation_inc] =
    memory_reservation->allocated_bytes.try_add(stream_tracking_size, memory_reservation->size());
  if (!success) {
    if (reservation_policy) {
      std::lock_guard lock(_arbitration_mutex);
      reservation_policy->handle_over_reservation(
        stream, allocation_size, post_allocation_inc, memory_reservation.get());
    }
    post_allocation_inc = memory_reservation->allocated_bytes.add(stream_tracking_size);
  }
  memory_reservation->peak_allocated_bytes.update_peak(post_allocation_inc);

  int64_t pre_allocation_inc = post_allocation_inc - stream_tracking_size;
  if (post_allocation_inc < memory_reservation->size()) {
    upstream_tracking_size = 0UL;
  } else if (pre_allocation_inc < memory_reservation->size()) {
    upstream_tracking_size = post_allocation_inc - memory_reservation->size();
  }
  return upstream_tracking_size;
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope)
  : _space_id(space_id),
    _upstream(std::move(upstream)),
    _memory_limit(capacity),
    _capacity(capacity),
    _allocation_tracker([&]() -> std::unique_ptr<allocation_tracker_iface> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<stream_ordered_allocation_tracker>();
      } else {
        return std::make_unique<ptds_allocation_tracker>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

reservation_aware_resource_adaptor::reservation_aware_resource_adaptor(
  memory_space_id space_id,
  rmm::device_async_resource_ref upstream,
  std::size_t memory_limit,
  std::size_t capacity,
  std::unique_ptr<reservation_limit_policy> default_reservation_policy,
  std::unique_ptr<oom_handling_policy> default_oom_policy,
  AllocationTrackingScope tracking_scope)
  : _space_id(space_id),
    _upstream(std::move(upstream)),
    _memory_limit(memory_limit),
    _capacity(capacity),
    _allocation_tracker([&]() -> std::unique_ptr<allocation_tracker_iface> {
      if (tracking_scope == AllocationTrackingScope::PER_STREAM) {
        return std::make_unique<stream_ordered_allocation_tracker>();
      } else {
        return std::make_unique<ptds_allocation_tracker>();
      }
    }()),
    _default_reservation_policy(default_reservation_policy
                                  ? std::move(default_reservation_policy)
                                  : make_default_reservation_limit_policy()),
    _default_oom_policy(default_oom_policy ? std::move(default_oom_policy)
                                           : make_default_oom_policy())
{
}

rmm::device_async_resource_ref reservation_aware_resource_adaptor::get_upstream_resource()
  const noexcept
{
  return _upstream;
}

std::size_t reservation_aware_resource_adaptor::get_available_memory() const noexcept
{
  auto current_bytes = _total_allocated_bytes.load();
  return _capacity > current_bytes ? _capacity - current_bytes : 0;
}

std::size_t reservation_aware_resource_adaptor::get_available_memory(
  rmm::cuda_stream_view stream) const noexcept
{
  auto upstream_available_memory = get_available_memory();
  if (auto* state = _allocation_tracker->get_tracker_state(stream); state) {
    upstream_available_memory += state->memory_reservation->get_available_memory();
  }
  return upstream_available_memory;
}

std::size_t reservation_aware_resource_adaptor::get_available_memory_print(
  rmm::cuda_stream_view stream) const noexcept
{
  auto upstream_available_memory = get_available_memory();
  if (auto* state = _allocation_tracker->get_tracker_state(stream); state) {
    upstream_available_memory += state->memory_reservation->get_available_memory();
  }
  return upstream_available_memory;
}

std::size_t reservation_aware_resource_adaptor::get_allocated_bytes(
  rmm::cuda_stream_view stream) const
{
  const auto* stats = _allocation_tracker->get_tracker_state(stream);
  return stats ? stats->memory_reservation->allocated_bytes.load() : 0;
}

std::size_t reservation_aware_resource_adaptor::get_peak_allocated_bytes(
  rmm::cuda_stream_view stream) const
{
  const auto* stats = _allocation_tracker->get_tracker_state(stream);
  return stats ? stats->memory_reservation->peak_allocated_bytes.peak() : 0;
}

std::size_t reservation_aware_resource_adaptor::get_total_allocated_bytes() const
{
  return _total_allocated_bytes.load();
}

std::size_t reservation_aware_resource_adaptor::get_peak_total_allocated_bytes() const
{
  return _peak_total_allocated_bytes.peak();
}

void reservation_aware_resource_adaptor::reset_peak_allocated_bytes(rmm::cuda_stream_view stream)
{
  auto* stats = _allocation_tracker->get_tracker_state(stream);
  if (stats) { stats->memory_reservation->peak_allocated_bytes.reset(0); }
}

std::size_t reservation_aware_resource_adaptor::get_total_reserved_bytes() const
{
  return _total_reserved_bytes.load();
}

bool reservation_aware_resource_adaptor::is_stream_tracked(rmm::cuda_stream_view stream) const
{
  return _allocation_tracker->get_tracker_state(stream) != nullptr;
}

bool reservation_aware_resource_adaptor::attach_reservation_to_tracker(
  rmm::cuda_stream_view stream,
  std::unique_ptr<reservation> reserved_bytes,
  std::unique_ptr<reservation_limit_policy> stream_reservation_policy,
  std::unique_ptr<oom_handling_policy> stream_oom_policy)
{
  auto* stats = _allocation_tracker->get_tracker_state(stream);
  if (stats) { return false; }

  if (!stream_reservation_policy) {
    stream_reservation_policy = make_default_reservation_limit_policy();
  }

  if (!stream_oom_policy) { stream_oom_policy = make_default_oom_policy(); }

  _allocation_tracker->assign_reservation_to_tracker(
    stream,
    std::unique_ptr<device_reserved_arena>(
      dynamic_cast<device_reserved_arena*>(reserved_bytes->arena_.release())),
    std::move(stream_reservation_policy),
    std::move(stream_oom_policy));

  return true;
}
void reservation_aware_resource_adaptor::reset_stream_reservation(rmm::cuda_stream_view stream)
{
  _allocation_tracker->reset_tracker_state(stream);
}

std::unique_ptr<reserved_arena> reservation_aware_resource_adaptor::reserve(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  if (do_reserve(bytes, _memory_limit)) {
    _number_of_allocations.fetch_add(1);
    return std::make_unique<device_reserved_arena>(*this, bytes, std::move(release_notifer));
  }
  return nullptr;
}

std::unique_ptr<reserved_arena> reservation_aware_resource_adaptor::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  auto reserved_size = do_reserve_upto(bytes, _memory_limit);
  _number_of_allocations.fetch_add(1);
  return std::make_unique<device_reserved_arena>(*this, reserved_size, std::move(release_notifer));
}

bool reservation_aware_resource_adaptor::grow_reservation_by(device_reserved_arena& arena,
                                                             std::size_t bytes)
{
  if (do_reserve(bytes, _memory_limit)) {
    arena.size_ += bytes;
    return true;
  }
  return false;
}

void reservation_aware_resource_adaptor::shrink_reservation_to_fit(device_reserved_arena& arena)
{
  int64_t current_bytes = arena.allocated_bytes.load();
  if (current_bytes < arena.size()) {
    auto reclaimed_bytes = std::exchange(arena.size_, current_bytes) - current_bytes;
    _total_allocated_bytes.sub(reclaimed_bytes);
  }
}

std::size_t reservation_aware_resource_adaptor::get_active_reservation_count() const noexcept
{
  return _number_of_allocations.load();
}

void* reservation_aware_resource_adaptor::do_allocate(std::size_t bytes,
                                                      rmm::cuda_stream_view stream)
{
  auto* reservation_state = _allocation_tracker->get_tracker_state(stream);
  if (reservation_state != nullptr) {
    return do_allocate_managed(bytes, reservation_state, stream);
  } else {
    return do_allocate_managed(bytes, stream);
  }
}

void* reservation_aware_resource_adaptor::do_allocate_managed(std::size_t bytes,
                                                              rmm::cuda_stream_view stream)
{
  auto tracking_size = rmm::align_up(bytes, 256);
  try {
    return do_allocate_unmanaged(bytes, tracking_size, stream);
  } catch (...) {
    return _default_oom_policy->handle_oom(
      bytes,
      stream,
      std::current_exception(),
      std::bind(&reservation_aware_resource_adaptor::do_allocate_unmanaged,
                this,
                std::placeholders::_1,
                tracking_size,
                std::placeholders::_2));
  }
}

void* reservation_aware_resource_adaptor::do_allocate_managed(std::size_t bytes,
                                                              stream_ordered_tracker_state* state,
                                                              rmm::cuda_stream_view stream)
{
  auto padded_bytes  = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto tracking_size = state->check_reservation_and_handle_overflow(*this, padded_bytes, stream);
  try {
    return do_allocate_unmanaged(bytes, tracking_size, stream);
  } catch (...) {
    try {
      return state->oom_policy->handle_oom(
        bytes,
        stream,
        std::current_exception(),
        std::bind(&reservation_aware_resource_adaptor::do_allocate_unmanaged,
                  this,
                  std::placeholders::_1,
                  tracking_size,
                  std::placeholders::_2));
    } catch (...) {
      state->memory_reservation->allocated_bytes.sub(padded_bytes);
      throw;
    }
  }
}

void* reservation_aware_resource_adaptor::do_allocate_unmanaged(std::size_t allocation_bytes,
                                                                std::size_t tracking_bytes,
                                                                rmm::cuda_stream_view stream)
{
  auto [success, post_allocation_size] = _total_allocated_bytes.try_add(tracking_bytes, _capacity);
  if (success) {
    _peak_total_allocated_bytes.update_peak(post_allocation_size);
    try {
      return _upstream.allocate_async(allocation_bytes, stream);
    } catch (std::exception& e) {
      _total_allocated_bytes.sub(tracking_bytes);
      throw cucascade_out_of_memory(e.what(), allocation_bytes, post_allocation_size);
    }
  } else {
    throw cucascade_out_of_memory(
      "not enough capacity to allocate memory", allocation_bytes, post_allocation_size);
  }
}

void reservation_aware_resource_adaptor::do_deallocate(void* ptr,
                                                       std::size_t bytes,
                                                       rmm::cuda_stream_view stream) noexcept
{
  auto tracking_bytes           = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
  auto upstream_reclaimed_bytes = tracking_bytes;
  auto* reservation_state       = _allocation_tracker->get_tracker_state(stream);
  if (reservation_state != nullptr) {
    auto* reservation              = reservation_state->memory_reservation.get();
    int64_t post_deallocation_size = reservation->allocated_bytes.sub(tracking_bytes);
    int64_t pre_deallocation_size  = post_deallocation_size + tracking_bytes;
    if (pre_deallocation_size <= reservation->size()) {
      // if it was made using the reserved space
      upstream_reclaimed_bytes = 0;
    } else if (post_deallocation_size < reservation->size()) {
      // if it was partially made using the reserved space
      upstream_reclaimed_bytes = reservation->size() - post_deallocation_size;
    }
  }
  _upstream.deallocate_async(ptr, bytes, stream);
  _total_allocated_bytes.sub(upstream_reclaimed_bytes);
}

bool reservation_aware_resource_adaptor::do_is_equal(
  const rmm::mr::device_memory_resource& other) const noexcept
{
  // Check if it's the same type
  const auto* other_adaptor = dynamic_cast<const reservation_aware_resource_adaptor*>(&other);
  if (other_adaptor == nullptr) { return false; }

  return _upstream == other_adaptor->get_upstream_resource();
}

bool reservation_aware_resource_adaptor::do_reserve(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto [success, post_increase_bytes] = _total_allocated_bytes.try_add(size_bytes, limit_bytes);
  if (success) {
    _peak_total_allocated_bytes.update_peak(post_increase_bytes);
    _total_reserved_bytes.fetch_add(size_bytes);
  }
  return success;
}

std::size_t reservation_aware_resource_adaptor::do_reserve_upto(std::size_t size_bytes,
                                                                std::size_t limit_bytes)
{
  auto post_increase_bytes = _total_allocated_bytes.add_bounded(size_bytes, limit_bytes);
  if (post_increase_bytes > 0) {
    _peak_total_allocated_bytes.update_peak(post_increase_bytes);
    _total_reserved_bytes.fetch_add(size_bytes);
  }
  return size_bytes;
}

void reservation_aware_resource_adaptor::do_release_reservation(
  device_reserved_arena* arena) noexcept
{
  if (!arena) return;

  int64_t allocation_size    = arena->allocated_bytes.load();
  std::size_t released_bytes = arena->size();
  if (arena->size() > allocation_size) {
    released_bytes = arena->size() - allocation_size;
  } else {
    released_bytes = 0;
  }

  _number_of_allocations.fetch_sub(1);
  _total_reserved_bytes.fetch_sub(arena->size());
  _total_allocated_bytes.sub(released_bytes);
}

}  // namespace memory
}  // namespace cucascade
