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

#include "cuda_stream_pool.hpp"
#include "cuda_stream_view.hpp"
#include "memory/common.hpp"
#include "memory/disk_access_limiter.hpp"
#include "memory/fixed_size_host_memory_resource.hpp"
#include "memory/memory_reservation.hpp"
#include "memory/reservation_aware_resource_adaptor.hpp"

#include <rmm/cuda_device.hpp>

#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace cucascade {
namespace memory {

//===----------------------------------------------------------------------===//
// memory_space Implementation
//===----------------------------------------------------------------------===//

memory_space::memory_space(Tier tier,
                           int device_id,
                           size_t memory_limit,
                           size_t start_downgrading_memory_threshold,
                           size_t stop_downgrading_memory_threshold,
                           size_t capacity,
                           std::unique_ptr<rmm::mr::device_memory_resource> allocator)
  : _id(tier, device_id),
    _memory_limit(memory_limit),
    _start_downgrading_memory_threshold(start_downgrading_memory_threshold),
    _stop_downgrading_memory_threshold(stop_downgrading_memory_threshold),
    _capacity(capacity),
    _allocator(std::move(allocator)),
    stream_pool_{[&]() -> std::unique_ptr<rmm::cuda_stream_pool> {
      rmm::cuda_set_device_raii guard{rmm::cuda_device_id(device_id)};
      if (tier == Tier::GPU) { return std::make_unique<rmm::cuda_stream_pool>(16); }
      return nullptr;
    }()}
{
  if (memory_limit == 0) { throw std::invalid_argument("Memory limit must be greater than 0"); }
  if (!_allocator) { throw std::invalid_argument("At least one allocator must be provided"); }
  if (tier == Tier::GPU) {
    _reservation_allocator = std::make_unique<reservation_aware_resource_adaptor>(
      _id, *_allocator, _memory_limit, _capacity);
  } else if (tier == Tier::HOST) {
    _reservation_allocator = std::make_unique<fixed_size_host_memory_resource>(
      _id.device_id, *_allocator, _memory_limit, _capacity);
  } else if (tier == Tier::DISK) {
    _reservation_allocator = std::make_unique<disk_access_limiter>(_id, _memory_limit, _capacity);
  }
  notification_channel_ = std::make_shared<notification_channel>();
}

memory_space::~memory_space() = default;

bool memory_space::operator==(const memory_space& other) const { return _id == other.get_id(); }

bool memory_space::operator!=(const memory_space& other) const { return !(*this == other); }

memory_space_id memory_space::get_id() const noexcept { return _id; }

Tier memory_space::get_tier() const noexcept { return _id.tier; }

int memory_space::get_device_id() const noexcept { return _id.device_id; }

std::unique_ptr<reservation> memory_space::make_reservation_or_null(size_t size)
{
  std::unique_ptr<reserved_arena> arena = std::visit(
    cucascade::overloaded{[&](std::unique_ptr<disk_access_limiter>& mr) {
                            return mr->reserve(size, notification_channel_->get_notifier());
                          },
                          [&](std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                            return mr->reserve(size, notification_channel_->get_notifier());
                          },
                          [&](std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                            return mr->reserve(size, notification_channel_->get_notifier());
                          }},
    _reservation_allocator);
  return reservation::create(*this, std::move(arena));
}

std::unique_ptr<reservation> memory_space::make_reservation_upto(size_t size)
{
  std::unique_ptr<reserved_arena> arena = std::visit(
    cucascade::overloaded{[&](std::unique_ptr<disk_access_limiter>& mr) {
                            return mr->reserve_upto(size, notification_channel_->get_notifier());
                          },
                          [&](std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                            return mr->reserve_upto(size, notification_channel_->get_notifier());
                          },
                          [&](std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                            return mr->reserve_upto(size, notification_channel_->get_notifier());
                          }},
    _reservation_allocator);
  return reservation::create(*this, std::move(arena));
}

std::unique_ptr<reservation> memory_space::make_reservation(size_t size)
{
  std::unique_ptr<reservation> res = make_reservation_or_null(size);
  while (!res) {
    auto status = notification_channel_->wait();
    if (status == notification_channel::wait_status::SHUTDOWN) { return nullptr; }
    if (status == notification_channel::wait_status::IDLE) { return make_reservation_upto(size); }
    res = make_reservation_or_null(size);
  }
  return res;
}

rmm::cuda_stream_view memory_space::acquire_stream() const
{
  if (!stream_pool_) {
    throw std::runtime_error("Stream pool is not available for non-GPU memory spaces");
  }
  return stream_pool_->get_stream();
}

std::size_t memory_space::get_active_reservation_count() const
{
  return std::visit(
    cucascade::overloaded{[&](const std::unique_ptr<disk_access_limiter>& mr) {
                            return mr->get_active_reservation_count();
                          },
                          [&](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                            return mr->get_active_reservation_count();
                          },
                          [&](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                            return mr->get_active_reservation_count();
                          }},
    _reservation_allocator);
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
  size_t consumed = _memory_limit - get_available_memory();
  if (consumed <= _stop_downgrading_memory_threshold) { return 0; }
  return consumed - _stop_downgrading_memory_threshold;
}

size_t memory_space::get_available_memory(rmm::cuda_stream_view stream) const
{
  return std::visit(
    cucascade::overloaded{
      [&](const std::unique_ptr<disk_access_limiter>& mr) { return mr->get_available_memory(); },
      [&](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
        return mr->get_available_memory(stream);
      },
      [&](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
        return mr->get_available_memory();
      }},
    _reservation_allocator);
}

size_t memory_space::get_available_memory() const
{
  return std::visit(
    cucascade::overloaded{
      [&](const std::unique_ptr<disk_access_limiter>& mr) { return mr->get_available_memory(); },
      [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
        return mr->get_available_memory();
      },
      [](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
        return mr->get_available_memory();
      }},
    _reservation_allocator);
}

size_t memory_space::get_total_reserved_memory() const
{
  return std::visit(
    cucascade::overloaded{[&](const std::unique_ptr<disk_access_limiter>& mr) {
                            return mr->get_total_reserved_bytes();
                          },
                          [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr) {
                            return mr->get_total_reserved_bytes();
                          },
                          [](const std::unique_ptr<fixed_size_host_memory_resource>& mr) {
                            return mr->get_total_reserved_bytes();
                          }},
    _reservation_allocator);
}

size_t memory_space::get_max_memory() const noexcept { return _memory_limit; }

rmm::mr::device_memory_resource* memory_space::get_default_allocator() const noexcept
{
  return std::visit(
    cucascade::overloaded{[this](const std::unique_ptr<disk_access_limiter>& other)
                            -> rmm::mr::device_memory_resource* { return _allocator.get(); },
                          [](const std::unique_ptr<reservation_aware_resource_adaptor>& mr)
                            -> rmm::mr::device_memory_resource* { return mr.get(); },
                          [](const std::unique_ptr<fixed_size_host_memory_resource>& mr)
                            -> rmm::mr::device_memory_resource* { return mr.get(); }},
    _reservation_allocator);
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

void memory_space::shutdown()
{
  if (notification_channel_) { notification_channel_->shutdown(); }
}

//===----------------------------------------------------------------------===//
// memory_space_hash Implementation
//===----------------------------------------------------------------------===//

size_t memory_space_hash::operator()(const memory_space& ms) const
{
  return std::hash<int>{}(static_cast<int>(ms.get_tier())) ^
         (std::hash<size_t>{}(ms.get_device_id()) << 1);
}

}  // namespace memory
}  // namespace cucascade
