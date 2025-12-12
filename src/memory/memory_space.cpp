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

#include "memory/memory_space.hpp"
#include "memory/memory_reservation.hpp"  // For Reservation struct
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace sirius {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_space Implementation
//===----------------------------------------------------------------------===//

memory_space::memory_space(memory_space_id id,
                           size_t memory_limit,
                           size_t start_downgrading_memory_threshold,
                           size_t stop_downgrading_memory_threshold,
                           std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> allocators)
  : _id(id),
    _memory_limit(memory_limit),
    _start_downgrading_memory_threshold(start_downgrading_memory_threshold),
    _stop_downgrading_memory_threshold(stop_downgrading_memory_threshold),
    _allocators(std::move(allocators))
{
  if (memory_limit == 0) { throw std::invalid_argument("Memory limit must be greater than 0"); }
  if (_allocators.empty()) {
    throw std::invalid_argument("At least one allocator must be provided");
  }
}

bool memory_space::operator==(const memory_space& other) const { return _id == other._id; }

bool memory_space::operator!=(const memory_space& other) const { return !(*this == other); }

memory_space_id memory_space::get_id() const { return _id; }

Tier memory_space::get_tier() const { return _id.tier; }

int memory_space::get_device_id() const { return _id.device_id; }

std::unique_ptr<reservation> memory_space::request_reservation(size_t size)
{
  std::unique_lock<std::mutex> lock(_mutex);

  // TODO: This is kind of wrong. Given that we are trying to handle the blocking
  // on the memory reservation manager. For now  I am going to leave it but
  // we should probably and some locking mechanism for seeing if there is space AND returning the
  // reservation if there is space in one operation.
  //  Wait until we can allocate the requested size
  wait_for_memory(size, lock);

  // Create the reservation
  auto res = std::make_unique<reservation>(_id.tier, static_cast<int>(_id.device_id), size);

  // Update tracking
  _total_reserved.fetch_add(size);
  _active_count.fetch_add(1);

  return res;
}

void memory_space::release_reservation(std::unique_ptr<reservation> res)
{
  if (!res) { return; }

  if (!validate_reservation(res.get())) {
    throw std::invalid_argument("Reservation does not belong to this memory_space");
  }

  std::lock_guard<std::mutex> lock(_mutex);

  // Update tracking
  _total_reserved.fetch_sub(res->size);
  _active_count.fetch_sub(1);

  // Notify waiting threads
  _cv.notify_all();
}

bool memory_space::should_downgrade_memory() const
{
  return _memory_limit - get_available_memory() >= _start_downgrading_memory_threshold;
}

bool memory_space::should_stop_downgrading_memory() const
{
  return _memory_limit - get_available_memory() <= _stop_downgrading_memory_threshold;
}

size_t memory_space::get_amount_to_downgrade() const
{
  // Reverse engineer from should_stop_downgrading_memory():
  // should_stop_downgrading_memory() returns true when:
  //     (_memory_limit - get_available_memory()) <= _stop_downgrading_memory_threshold
  // i.e., consumed_bytes <= stop_threshold.
  // Therefore, the amount to downgrade is:
  //     max(0, consumed_bytes - stop_threshold)
  size_t consumed = _memory_limit - get_available_memory();
  if (consumed <= _stop_downgrading_memory_threshold) { return 0; }
  return consumed - _stop_downgrading_memory_threshold;
}

bool memory_space::shrink_reservation(reservation* res, size_t new_size)
{
  if (!res || new_size >= res->size) {
    return false;  // Invalid operation
  }

  if (!validate_reservation(res)) { return false; }

  std::lock_guard<std::mutex> lock(_mutex);

  size_t size_diff = res->size - new_size;

  // Update reservation size
  res->size = new_size;

  // Update tracking
  _total_reserved.fetch_sub(size_diff);

  // Notify waiting threads
  _cv.notify_all();

  return true;
}

bool memory_space::grow_reservation(reservation* res, size_t new_size)
{
  if (!res || new_size <= res->size) {
    return false;  // Invalid operation
  }

  if (!validate_reservation(res)) { return false; }

  size_t size_diff = new_size - res->size;

  std::unique_lock<std::mutex> lock(_mutex);

  // Check if we can grow
  if (!can_reserve(size_diff)) {
    return false;  // Not enough memory available
  }

  // Update reservation size
  res->size = new_size;

  // Update tracking
  _total_reserved.fetch_add(size_diff);

  return true;
}

size_t memory_space::get_available_memory() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  size_t reserved = _total_reserved.load();
  return (reserved >= _memory_limit) ? 0 : (_memory_limit - reserved);
}

size_t memory_space::get_total_reserved_memory() const { return _total_reserved.load(); }

size_t memory_space::get_max_memory() const { return _memory_limit; }

size_t memory_space::get_active_reservation_count() const { return _active_count.load(); }

rmm::device_async_resource_ref memory_space::get_default_allocator() const
{
  if (_allocators.empty()) { throw std::runtime_error("No allocators available in memory_space"); }
  return *_allocators[0];
}

rmm::device_async_resource_ref memory_space::get_allocator(size_t index) const
{
  if (index >= _allocators.size()) { throw std::out_of_range("Allocator index out of range"); }
  return *_allocators[index];
}

size_t memory_space::get_allocator_count() const { return _allocators.size(); }

bool memory_space::can_reserve(size_t size) const
{
  size_t current_reserved = _total_reserved.load();
  size_t current_active   = _active_count.load();
  // Allow a single initial reservation to exceed the memory limit if there are
  // currently zero outstanding reservations. Subsequent reservations must obey the limit.
  if (current_active == 0) { return true; }
  return (current_reserved + size) <= _memory_limit;
}

std::string memory_space::to_string() const
{
  std::ostringstream oss;
  oss << "memory_space(tier=";
  switch (_id.tier) {
    case Tier::GPU: oss << "GPU"; break;
    case Tier::HOST: oss << "HOST"; break;
    case Tier::DISK: oss << "DISK"; break;
    default: oss << "UNKNOWN"; break;
  }
  oss << ", device_id=" << _id.device_id << ", limit=" << _memory_limit << ")";
  return oss.str();
}

void memory_space::wait_for_memory(size_t size, std::unique_lock<std::mutex>& lock)
{
  while (!can_reserve(size)) {
    _cv.wait(lock);
  }
}

bool memory_space::validate_reservation(const reservation* res) const
{
  return res && res->tier == _id.tier && res->device_id == _id.device_id;
}

void memory_space::initialize_stream_pool_if_needed() const
{
  if (_id.tier != Tier::GPU) { return; }
  std::lock_guard<std::mutex> lg(_streams_mutex);
  if (!_streams.empty()) { return; }
  // Initialize with a small pool size (e.g., 4)
  int prev_device = -1;
  cudaGetDevice(&prev_device);
  cudaSetDevice(_id.device_id);
  constexpr size_t initial_streams = 4;
  _streams.reserve(initial_streams);
  _stream_in_use.reserve(initial_streams);
  for (size_t i = 0; i < initial_streams; ++i) {
    _streams.emplace_back(std::make_unique<rmm::cuda_stream>());
    _stream_in_use.emplace_back(false);
  }
  cudaSetDevice(prev_device);
}

void memory_space::grow_stream_pool_unlocked(size_t additional_streams) const
{
  int prev_device = -1;
  cudaGetDevice(&prev_device);
  cudaSetDevice(_id.device_id);
  for (size_t i = 0; i < additional_streams; ++i) {
    _streams.emplace_back(std::make_unique<rmm::cuda_stream>());
    _stream_in_use.emplace_back(false);
  }
  cudaSetDevice(prev_device);
}

rmm::cuda_stream_view memory_space::acquire_stream() const
{
  initialize_stream_pool_if_needed();
  std::lock_guard<std::mutex> lg(_streams_mutex);
  // Find a free stream
  for (size_t i = 0; i < _streams.size(); ++i) {
    if (!_stream_in_use[i]) {
      _stream_in_use[i] = true;
      return _streams[i]->view();
    }
  }
  // Grow pool and return first new stream
  size_t old_size = _streams.size();
  size_t grow_by  = std::max<size_t>(1, old_size);  // double size (or at least +1)
  grow_stream_pool_unlocked(grow_by);
  _stream_in_use[old_size] = true;
  return _streams[old_size]->view();
}

void memory_space::release_stream(rmm::cuda_stream_view stream) const
{
  std::lock_guard<std::mutex> lg(_streams_mutex);
  for (size_t i = 0; i < _streams.size(); ++i) {
    if (_streams[i] && _streams[i]->value() == stream.value()) {
      _stream_in_use[i] = false;
      return;
    }
  }
  // If not found, ignore silently (defensive)
}

//===----------------------------------------------------------------------===//
// memory_space_hash Implementation
//===----------------------------------------------------------------------===//

size_t memory_space_hash::operator()(const memory_space& ms) const
{
  return std::hash<int>{}(static_cast<int>(ms.get_tier())) ^
         (std::hash<int>{}(ms.get_device_id()) << 1);
}

}  // namespace memory
}  // namespace sirius
