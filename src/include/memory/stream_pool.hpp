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

#include "cuda_device.hpp"

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <condition_variable>
#include <mutex>

namespace cucascade {
namespace memory {

class exclusive_stream_pool;

class borrowed_stream {
 public:
  friend class exclusive_stream_pool;

  [[nodiscard]] rmm::cuda_stream_view get() const noexcept;
  [[nodiscard]] const rmm::cuda_stream* operator->() const noexcept;
  [[nodiscard]] const rmm::cuda_stream* operator->() noexcept;
  operator rmm::cuda_stream_view() const;

  ~borrowed_stream() noexcept;

  borrowed_stream(borrowed_stream&& other) noexcept;
  borrowed_stream& operator=(borrowed_stream&& other) noexcept;
  borrowed_stream(borrowed_stream const&)            = delete;
  borrowed_stream& operator=(borrowed_stream const&) = delete;

  void reset() noexcept;

 private:
  borrowed_stream(rmm::cuda_stream s, std::function<void(rmm::cuda_stream&&)> release_fn) noexcept;

  rmm::cuda_stream stream_;
  std::function<void(rmm::cuda_stream&&)> release_fn_;
};

class exclusive_stream_pool {
 public:
  static constexpr std::size_t default_size{16};

  enum class stream_acquire_policy { GROW, BLOCK };

  /**
   * @brief Construct a new cuda stream pool object of the given non-zero size
   *
   * @throws logic_error if `pool_size` is zero
   * @param pool_size The number of streams in the pool
   * @param flags Flags used when creating streams in the pool.
   */
  explicit exclusive_stream_pool(
    rmm::cuda_device_id device_id = {},
    std::size_t pool_size         = default_size,
    rmm::cuda_stream::flags flags = rmm::cuda_stream::flags::sync_default);

  ~exclusive_stream_pool() = default;

  exclusive_stream_pool(exclusive_stream_pool&&)                 = delete;
  exclusive_stream_pool(exclusive_stream_pool const&)            = delete;
  exclusive_stream_pool& operator=(exclusive_stream_pool&&)      = delete;
  exclusive_stream_pool& operator=(exclusive_stream_pool const&) = delete;

  /**
   * @brief Get a `cuda_stream_view` of a stream in the pool.
   *
   * This function is thread safe with respect to other calls to the same function.
   *
   * @return rmm::cuda_stream_view
   */
  borrowed_stream acquire_stream(
    stream_acquire_policy policy = stream_acquire_policy::BLOCK) noexcept;

  std::size_t size() const noexcept;

 private:
  void release_stream(rmm::cuda_stream&& s) noexcept;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  rmm::cuda_device_id device_id_;
  rmm::cuda_stream::flags flags_;
  std::vector<rmm::cuda_stream> streams_;
};

}  // namespace memory
}  // namespace cucascade
