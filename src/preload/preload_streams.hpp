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

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/stream_pool.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace sirius::preload {

/**
 * @brief The preload executor's own CUDA streams: one growable pool per GPU.
 *
 * Each pool starts with one stream on its device and grows on demand (promotions acquire with
 * the GROW policy), so a lane never waits for a stream.
 *
 * GPU buffers that a DISK->GPU promotion allocates are freed on the promotion's stream until
 * their consumer rebinds them. The pools must therefore outlive every repository that may hold
 * such a batch, which is why they are owned separately from the executor. Every stream borrowed
 * from a pool must be returned (its @c borrowed_stream destroyed) before the pool is destroyed.
 *
 * A stream-creation failure while growing a pool terminates the process: cuCascade's
 * @c acquire_stream is @c noexcept.
 */
class preload_streams {
 public:
  /**
   * @brief One pool for each distinct device in @p gpu_device_ids.
   *
   * @throws Whatever stream creation throws; pools created before the failure are destroyed
   *         under their device.
   */
  explicit preload_streams(std::span<int const> gpu_device_ids)
  {
    // So that a created pool is always owned by _pools, and destroyed under its device.
    _pools.reserve(gpu_device_ids.size());
    try {
      for (int const device : gpu_device_ids) {
        if (device < 0 || for_device(device) != nullptr) { continue; }
        _pools.push_back({device,
                          std::make_unique<cucascade::memory::exclusive_stream_pool>(
                            rmm::cuda_device_id{device}, std::size_t{1})});
      }
    } catch (...) {
      destroy_all();
      throw;
    }
  }

  /// Destroys each pool with its own device current.
  ~preload_streams() { destroy_all(); }

  preload_streams(preload_streams const&)            = delete;
  preload_streams& operator=(preload_streams const&) = delete;
  preload_streams(preload_streams&&)                 = delete;
  preload_streams& operator=(preload_streams&&)      = delete;

  /// The pool of GPU @p device, or nullptr when there is none.
  [[nodiscard]] cucascade::memory::exclusive_stream_pool* for_device(int device) const noexcept
  {
    auto const it = std::find_if(
      _pools.begin(), _pools.end(), [device](entry const& e) { return e.device == device; });
    return it == _pools.end() ? nullptr : it->pool.get();
  }

  /// Number of pools (distinct GPUs).
  [[nodiscard]] std::size_t size() const noexcept { return _pools.size(); }

 private:
  struct entry {
    int device{-1};
    std::unique_ptr<cucascade::memory::exclusive_stream_pool> pool;
  };

  void destroy_all() noexcept
  {
    for (auto& e : _pools) {
      try {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{e.device}};
        e.pool.reset();
      } catch (...) {
        // Best effort: destroy the streams below on whatever device is current.
      }
      e.pool.reset();
    }
    _pools.clear();
  }

  std::vector<entry> _pools;
};

}  // namespace sirius::preload
