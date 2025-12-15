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

#include "memory/stream_pool.hpp"

#include "cuda_device.hpp"

#include <functional>
#include <mutex>
#include <utility>

namespace sirius {
namespace memory {

borrowed_stream::borrowed_stream(rmm::cuda_stream s,
                                 std::function<void(rmm::cuda_stream&&)> release_fn) noexcept
  : stream_(std::move(s)), release_fn_(std::move(release_fn))
{
}

borrowed_stream::~borrowed_stream() = default;

borrowed_stream::borrowed_stream(borrowed_stream&& other) noexcept
  : stream_(std::move(other.stream_)), release_fn_(std::exchange(other.release_fn_, nullptr))
{
}

borrowed_stream& borrowed_stream::operator=(borrowed_stream&& other) noexcept
{
  if (this != &other) {
    reset();
    stream_     = std::move(other.stream_);
    release_fn_ = std::exchange(other.release_fn_, nullptr);
  }
  return *this;
}

borrowed_stream::operator rmm::cuda_stream_view() const { return stream_; }

void borrowed_stream::reset() noexcept
{
  if (release_fn_) { std::exchange(release_fn_, nullptr)(std::move(stream_)); }
}

rmm::cuda_stream_view borrowed_stream::get() const noexcept { return stream_; }
const rmm::cuda_stream* borrowed_stream::operator->() const noexcept { return &stream_; }
const rmm::cuda_stream* borrowed_stream::operator->() noexcept { return &stream_; }

exclusive_stream_pool::exclusive_stream_pool(rmm::cuda_device_id device_id,
                                             std::size_t pool_size,
                                             rmm::cuda_stream::flags flags)
  : device_id_(device_id), flags_(flags)
{
  rmm::cuda_set_device_raii set_device{device_id_};
  if (pool_size == 0) { throw std::logic_error("Stream pool size must be greater than zero"); }

  streams_.reserve(pool_size);
  for (std::size_t i = 0; i < pool_size; ++i) {
    streams_.emplace_back(rmm::cuda_stream(flags_));
  }
}

borrowed_stream exclusive_stream_pool::acquire_stream(stream_acquire_policy policy) noexcept
{
  std::unique_lock lock(mutex_);
  if (streams_.empty()) {
    if (policy == stream_acquire_policy::GROW) {
      rmm::cuda_set_device_raii set_device{device_id_};
      return borrowed_stream(rmm::cuda_stream(flags_),
                             std::bind_front(&exclusive_stream_pool::release_stream, this));
    } else {
      cv_.wait(lock, [this]() { return !streams_.empty(); });
    }
  }
  auto stream = std::move(streams_.back());
  streams_.pop_back();
  return borrowed_stream(std::move(stream),
                         std::bind_front(&exclusive_stream_pool::release_stream, this));
}

std::size_t exclusive_stream_pool::size() const noexcept
{
  std::lock_guard lock(mutex_);
  return streams_.size();
}

void exclusive_stream_pool::release_stream(rmm::cuda_stream&& s) noexcept
{
  std::lock_guard lock(mutex_);
  streams_.emplace_back(std::move(s));
  cv_.notify_one();
}

}  // namespace memory
}  // namespace sirius
