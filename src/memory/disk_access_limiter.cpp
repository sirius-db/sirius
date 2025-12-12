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

#include "memory/disk_access_limiter.hpp"

namespace sirius {
namespace memory {

disk_access_limiter::disk_access_limiter(memory_space_id space_id, std::size_t capacity)
  : _space_id(space_id), _memory_limit(capacity), _capacity(capacity)
{
}

disk_access_limiter::disk_access_limiter(memory_space_id space_id,
                                         std::size_t memory_limit,
                                         std::size_t capacity)
  : _space_id(space_id), _memory_limit(memory_limit), _capacity(capacity)
{
}

std::size_t disk_access_limiter::get_available_memory() const noexcept
{
  auto current = _total_allocated_bytes.load();
  if (current <= _capacity) {
    return _capacity - current;
  } else {
    return 0;
  }
}

std::size_t disk_access_limiter::get_peak_reserved_bytes() const
{
  return _peak_total_allocated_bytes.load();
}

std::size_t disk_access_limiter::get_total_reserved_bytes() const
{
  return _total_allocated_bytes.load();
}

std::unique_ptr<reserved_arena> disk_access_limiter::reserve(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  if (do_reserve(bytes, _memory_limit)) {
    auto slot = std::make_unique<disk_reserved_arena>(
      *this, bytes, "disk_reservation", std::move(release_notifer));
    _total_reservation_count.fetch_add(1);
    update_peak_allocated_bytes();
    return slot;
  }
  return nullptr;
}

std::unique_ptr<reserved_arena> disk_access_limiter::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> release_notifer)
{
  auto reserved_bytes = do_reserve_upto(bytes, _memory_limit);
  auto slot           = std::make_unique<disk_reserved_arena>(
    *this, reserved_bytes, "disk_reservation", std::move(release_notifer));
  _total_reservation_count.fetch_add(1);
  update_peak_allocated_bytes();
  return slot;
}

bool disk_access_limiter::grow_reservation_by(reserved_arena& arena, std::size_t bytes)
{
  auto* disk_reservation = dynamic_cast<disk_reserved_arena*>(&arena);
  if (do_reserve(bytes, _memory_limit)) {
    disk_reservation->size_ += bytes;
    update_peak_allocated_bytes();
    return true;
  }
  return false;
}

void disk_access_limiter::shrink_reservation_to_fit(reserved_arena& arena)
{
  _total_allocated_bytes.fetch_sub(arena.size());
  arena.size_ = 0UL;
}

std::size_t disk_access_limiter::get_active_reservation_count() const noexcept
{
  return _total_reservation_count.load();
}

bool disk_access_limiter::do_reserve(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto pre_reservation_bytes = _total_allocated_bytes.load();
  while (pre_reservation_bytes + size_bytes < limit_bytes) {
    if (_total_allocated_bytes.compare_exchange_weak(
          pre_reservation_bytes, pre_reservation_bytes + size_bytes, std::memory_order_seq_cst)) {
      return true;
    }
  }
  return false;
}

std::size_t disk_access_limiter::do_reserve_upto(std::size_t size_bytes, std::size_t limit_bytes)
{
  auto pre_reservation_bytes = _total_allocated_bytes.load();
  while (pre_reservation_bytes < limit_bytes) {
    auto target = std::min(limit_bytes, pre_reservation_bytes + size_bytes);
    if (_total_allocated_bytes.compare_exchange_weak(
          pre_reservation_bytes, target, std::memory_order_seq_cst)) {
      return target - pre_reservation_bytes;
    }
  }
  return 0;
}

void disk_access_limiter::do_release_reservation(disk_reserved_arena* arena) noexcept
{
  if (!arena) return;
  _total_allocated_bytes.fetch_sub(arena->size());
  _total_reservation_count.fetch_sub(1);
  arena->size_ = 0UL;
}

void disk_access_limiter::update_peak_allocated_bytes() noexcept
{
  auto current = _total_allocated_bytes.load();
  auto peak    = _peak_total_allocated_bytes.load();
  while (current > peak) {
    if (_peak_total_allocated_bytes.compare_exchange_weak(
          peak, current, std::memory_order_seq_cst)) {
      break;
    }
  }
}

}  // namespace memory
}  // namespace sirius
