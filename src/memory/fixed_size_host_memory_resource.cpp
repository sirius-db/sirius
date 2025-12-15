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

#include "memory/fixed_size_host_memory_resource.hpp"

#include "memory/common.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/notification_channel.hpp"

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include <memory>
#include <mutex>
#include <stdexcept>

namespace sirius {
namespace memory {

namespace {
// Returns available system memory in bytes if detectable; 0 if unknown
std::size_t get_available_system_memory_bytes()
{
  // todo (fix this)
  // std::ifstream meminfo("/proc/meminfo");
  // if (!meminfo) { return 0; }
  // std::string line;
  // while (std::getline(meminfo, line)) {
  //   if (line.rfind("MemAvailable:", 0) == 0) {
  //     std::istringstream iss(line);
  //     std::string key, unit;
  //     std::size_t value_kb = 0;
  //     iss >> key >> value_kb >> unit;  // e.g., "MemAvailable:" 123456 "kB"
  //     if (value_kb > 0) {
  //       // kB in /proc/meminfo are kibibytes
  //       return static_cast<std::size_t>(value_kb) * 1024ULL;
  //     }
  //     break;
  //   }
  // }
  return 0;
}
}  // namespace

fixed_size_host_memory_resource::fixed_size_host_memory_resource(
  int device_id,
  rmm::mr::device_memory_resource& upstream_mr,
  std::size_t memory_limit,
  std::size_t memory_capacity,
  std::size_t block_size,
  std::size_t pool_size,
  std::size_t initial_pools)
  : space_id_(Tier::HOST, device_id),
    memory_limit_(memory_limit),
    memory_capacity_(memory_capacity),
    block_size_(rmm::align_up(block_size, alignof(std::max_align_t))),
    pool_size_(pool_size),
    upstream_mr_(&upstream_mr)
{
  assert(upstream_mr_);
  for (std::size_t i = 0; i < initial_pools; ++i) {
    expand_pool();
  }
}

fixed_size_host_memory_resource::~fixed_size_host_memory_resource()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& block : allocated_blocks_) {
    const std::size_t dealloc_size = block_size_ * pool_size_;
    upstream_mr_->deallocate(block, dealloc_size);
  }
  allocated_blocks_.clear();
  free_blocks_.clear();
}

std::size_t fixed_size_host_memory_resource::get_available_memory() const noexcept
{
  auto current_bytes = allocated_bytes_.load();
  return memory_capacity_ > current_bytes ? memory_capacity_ - current_bytes : 0;
}

std::size_t fixed_size_host_memory_resource::get_total_reserved_bytes() const noexcept
{
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t total = 0;
  for (const auto& [res, tracker] : active_reservations_) {
    total += res->size();
  }
  return total;
}

std::size_t fixed_size_host_memory_resource::get_block_size() const noexcept { return block_size_; }

std::size_t fixed_size_host_memory_resource::get_free_blocks() const noexcept
{
  std::lock_guard<std::mutex> lock(mutex_);
  return free_blocks_.size();
}

std::size_t fixed_size_host_memory_resource::get_total_blocks() const noexcept
{
  std::lock_guard<std::mutex> lock(mutex_);
  return allocated_blocks_.size() * pool_size_;
}

rmm::mr::device_memory_resource* fixed_size_host_memory_resource::get_upstream_resource()
  const noexcept
{
  return upstream_mr_;
}

fixed_size_host_memory_resource::fixed_multiple_blocks_allocation
fixed_size_host_memory_resource::allocate_multiple_blocks(std::size_t total_bytes, reservation* res)
{
  RMM_FUNC_RANGE();

  if (total_bytes == 0) { return multiple_blocks_allocation::empty(); }

  total_bytes                        = rmm::align_up(total_bytes, block_size_);
  size_t num_blocks                  = total_bytes / block_size_;
  std::size_t tracked_bytes          = total_bytes;
  std::size_t upstream_tracked_bytes = total_bytes;
  allocation_tracker* tracker        = nullptr;
  std::lock_guard<std::mutex> lock(mutex_);
  if (res) {
    auto* h_reservation_slot = dynamic_cast<chunked_reserved_area*>(res->arena_.get());
    if (h_reservation_slot == nullptr) {
      throw std::runtime_error("cannot make allocation with other reservation type");
    }
    auto iter = active_reservations_.find(h_reservation_slot);
    if (iter == active_reservations_.end()) {
      throw std::runtime_error("reservation has been freed already");
    }
    tracker                      = std::addressof(iter->second);
    int64_t pre_allocation_size  = tracker->allocated_bytes.fetch_add(total_bytes);
    int64_t post_allocation_size = pre_allocation_size + total_bytes;
    if (post_allocation_size <= h_reservation_slot->size()) {
      upstream_tracked_bytes = 0;
    } else if (pre_allocation_size < h_reservation_slot->size()) {
      upstream_tracked_bytes = post_allocation_size - h_reservation_slot->size();
    }
  }

  auto [success, post_allocation_size] =
    allocated_bytes_.try_add(upstream_tracked_bytes, memory_capacity_);
  if (success) {
    std::vector<std::byte*> allocated_blocks;
    allocated_blocks.reserve(num_blocks);

    for (std::size_t i = 0; i < num_blocks; ++i) {
      if (free_blocks_.empty()) { expand_pool(); }

      if (free_blocks_.empty()) {
        // Cleanup on failure
        for (std::byte* ptr : allocated_blocks) {
          free_blocks_.push_back(ptr);
        }
        allocated_bytes_.sub(upstream_tracked_bytes);
        if (tracker) tracker->allocated_bytes.fetch_sub(total_bytes);
        throw rmm::out_of_memory(
          "Not enough free blocks available in fixed_size_host_memory_resource and pool expansion "
          "failed.");
      }

      std::byte* ptr = static_cast<std::byte*>(free_blocks_.back());
      free_blocks_.pop_back();
      allocated_blocks.push_back(ptr);
    }
    peak_allocated_bytes_.update_peak(post_allocation_size);
    return multiple_blocks_allocation::create(std::move(allocated_blocks), *this, res);
  } else {
    if (tracker) tracker->allocated_bytes.fetch_sub(total_bytes);
  }
  return std::unique_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation>(nullptr);
}

void* fixed_size_host_memory_resource::do_allocate(std::size_t bytes, rmm::cuda_stream_view stream)
{
  throw rmm::logic_error(
    "fixed_size_host_memory_resource doesn't support allocate, use allocate_multiple_blocks");
}

void fixed_size_host_memory_resource::do_deallocate(void* ptr,
                                                    std::size_t bytes,
                                                    rmm::cuda_stream_view stream) noexcept
{
}

bool fixed_size_host_memory_resource::do_is_equal(
  const rmm::mr::device_memory_resource& other) const noexcept
{
  return this == &other;
}

std::unique_ptr<reserved_arena> fixed_size_host_memory_resource::reserve(
  std::size_t bytes, std::unique_ptr<event_notifier> on_release)
{
  bytes = rmm::align_up(bytes, block_size_);
  if (do_reserve(bytes, memory_limit_)) {
    auto host_slot = std::make_unique<chunked_reserved_area>(*this, bytes, std::move(on_release));
    this->register_reservation(host_slot.get());
    return host_slot;
  }
  return nullptr;
}

std::unique_ptr<reserved_arena> fixed_size_host_memory_resource::reserve_upto(
  std::size_t bytes, std::unique_ptr<event_notifier> on_release)
{
  bytes               = rmm::align_up(bytes, block_size_);
  auto reserved_bytes = do_reserve_upto(bytes, memory_limit_);
  auto host_slot =
    std::make_unique<chunked_reserved_area>(*this, reserved_bytes, std::move(on_release));
  this->register_reservation(host_slot.get());
  return host_slot;
}

bool fixed_size_host_memory_resource::grow_reservation_by(reserved_arena& arena, std::size_t bytes)
{
  std::lock_guard lock(mutex_);
  bytes = rmm::align_up(bytes, block_size_);
  if (do_reserve(bytes, memory_limit_)) {
    arena.size_ += bytes;
    return true;
  }
  return false;
}

void fixed_size_host_memory_resource::shrink_reservation_to_fit(reserved_arena& arena)
{
  auto* h_reservation_slot = dynamic_cast<chunked_reserved_area*>(&arena);
  assert(h_reservation_slot);
  std::lock_guard lock(mutex_);
  auto iter = active_reservations_.find(h_reservation_slot);
  assert(iter != active_reservations_.end());
  auto& tracker = iter->second;
  auto current  = tracker.allocated_bytes.load();
  if (current < h_reservation_slot->size()) {
    auto old_res = std::exchange(h_reservation_slot->size_, current);
    allocated_bytes_.sub(old_res - current);
  }
}

std::size_t fixed_size_host_memory_resource::get_active_reservation_count() const noexcept
{
  std::lock_guard lock(mutex_);
  return active_reservations_.size();
}

std::size_t fixed_size_host_memory_resource::get_peak_total_allocated_bytes() const
{
  return peak_allocated_bytes_.peak();
}

void fixed_size_host_memory_resource::expand_pool()
{
  const std::size_t total_size = block_size_ * pool_size_;

  void* large_allocation = upstream_mr_->allocate(total_size);

  allocated_blocks_.push_back(large_allocation);

  for (std::size_t i = 0; i < pool_size_; ++i) {
    void* block = static_cast<char*>(large_allocation) + (i * block_size_);
    free_blocks_.push_back(block);
  }
}

void fixed_size_host_memory_resource::register_reservation(chunked_reserved_area* res)
{
  std::lock_guard lock(mutex_);
  auto r = active_reservations_.insert(std::make_pair(res, res->uuid()));
  assert(r.second && "insertion failed");
}

void fixed_size_host_memory_resource::release_reservation(chunked_reserved_area* arena)
{
  if (!arena) return;

  std::lock_guard guard(mutex_);
  auto iter = active_reservations_.find(arena);
  if (iter == active_reservations_.end()) {
    throw std::runtime_error("reservation was not registered or already freed");
  }

  auto current                = iter->second.allocated_bytes.load();
  std::size_t reclaimed_bytes = arena->size() > current ? arena->size() - current : 0;
  allocated_bytes_.sub(reclaimed_bytes);
  active_reservations_.erase(iter);
}

void fixed_size_host_memory_resource::return_allocated_chunks(std::vector<std::byte*> chunks,
                                                              chunked_reserved_area* arena)
{
  size_t reclaimed_bytes = chunks.size() * block_size_;
  std::lock_guard lock(mutex_);
  if (arena != nullptr && active_reservations_.contains(arena)) {
    auto& tracker                  = active_reservations_.at(arena);
    int64_t pre_reclaimation_size  = tracker.allocated_bytes.fetch_sub(reclaimed_bytes);
    int64_t post_reclaimation_size = pre_reclaimation_size - reclaimed_bytes;
    if (pre_reclaimation_size <= arena->size()) {
      // allocation fit in reservation
      reclaimed_bytes = 0;
    } else if (post_reclaimation_size < arena->size()) {
      // part of allocation fit in the reservation
      reclaimed_bytes = pre_reclaimation_size - arena->size();
    }
  }
  allocated_bytes_.sub(reclaimed_bytes);
}

bool fixed_size_host_memory_resource::do_reserve(std::size_t bytes, std::size_t mem_limit)
{
  auto [success, post_allocation_size] = allocated_bytes_.try_add(bytes, mem_limit);
  if (success) { peak_allocated_bytes_.update_peak(post_allocation_size); }
  return success;
}

std::size_t fixed_size_host_memory_resource::do_reserve_upto(std::size_t bytes,
                                                             std::size_t mem_limit)
{
  auto post_allocation_size = allocated_bytes_.add_bounded(bytes, mem_limit);
  if (bytes > 0) { peak_allocated_bytes_.update_peak(post_allocation_size); }
  return bytes;
}

}  // namespace memory
}  // namespace sirius