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

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::nvrtc {

/**
 * @brief Device copies of the string-constant bytes that `NVRTC_JIT` kernels read through their
 * `c_N_ptr` and `c_N_len` parameters, each uploaded once per device for the lifetime of the cache.
 *
 * Before each launch, `kernel_arguments` concatenates one kernel's string constants into a pool and
 * acquires the pool's device copy here. Copies are keyed by the pool's bytes and the device, so a
 * query that evaluates the same expression over many batches, in many task-local
 * `expression_evaluator` instances, uploads it once per device. Each query's
 * `pipeline::pipeline_build_context` owns one cache, inside the `expression_query_caches` it shares
 * with every evaluator of the query, so the copies live exactly as long as the query.
 *
 * Member functions are thread-safe and never wait on the host for GPU work. Each works on the
 * calling thread's current device, which must be the device of the stream it is given. Every device
 * has its own copies, its own lock, and its own stream, which the cache creates at the device's
 * first acquisition. The first acquisition of a pool on a device uploads it on that stream, so the
 * upload never waits for work queued on the caller's stream. Concurrent acquisitions of the same
 * pool wait only until that upload is queued, acquisitions of other pools do not wait for it, and
 * every acquisition makes the caller's stream wait for the upload on the device.
 *
 * Destruction is stream-ordered too: each device's stream, which already orders the uploads, waits
 * for the work queued before every `lease` was released, and the copies are freed on it. Every
 * lease must be released before the cache is destroyed.
 */
class string_constant_cache {
 public:
  class lease;

  /**
   * @brief A cache that allocates the copies of each device from the device's current memory
   * resource, for code that runs without Sirius memory spaces, such as unit tests.
   *
   * Destruction waits on the host only when that resource frees synchronously, as
   * `rmm::mr::cuda_memory_resource` does.
   */
  string_constant_cache();

  /**
   * @brief A cache that charges its copies on the device of each of @p gpu_spaces to that GPU
   * memory space, for exactly as long as the cache holds them.
   *
   * Each copy is allocated from the upstream resource of the space's reservation-aware allocator
   * under a reservation of the copy's aligned size that the copy holds. The charge therefore never
   * lands on the reservation of the task whose acquisition uploaded the copy, nor on whichever
   * thread or stream frees it. A device without a space in @p gpu_spaces uses its current memory
   * resource. The spaces must outlive the cache.
   */
  explicit string_constant_cache(std::vector<cucascade::memory::memory_space const*> gpu_spaces);

  ~string_constant_cache();
  string_constant_cache(string_constant_cache const&)            = delete;
  string_constant_cache& operator=(string_constant_cache const&) = delete;
  string_constant_cache(string_constant_cache&&)                 = delete;
  string_constant_cache& operator=(string_constant_cache&&)      = delete;

  /**
   * @brief Returns a lease on the current device's copy of @p pool, ordered before any work queued
   * on @p stream after this call.
   *
   * @p pool is read before the call returns.
   *
   * @throws internal_exception if @p pool is empty
   * @throws rmm::out_of_memory if the device's GPU memory space cannot hold the copy
   * @throws std::runtime_error carrying the CUDA error string if the stream, the allocation, the
   * copy, or an event fails
   */
  [[nodiscard]] lease acquire(std::string_view pool, ::cuda::stream_ref stream) const;

  /** @brief Returns the number of uploads, one per distinct (pool, device) acquired. */
  [[nodiscard]] std::size_t upload_count_for_testing() const;

 private:
  struct device_copies;

  /** @brief Returns the copies of @p device, creating them on the device's first acquisition. */
  [[nodiscard]] device_copies& copies_of(int device) const;

  std::vector<cucascade::memory::memory_space const*> _gpu_spaces;
  mutable std::mutex _mutex;  ///< Guards `_devices`; never held across a CUDA call.
  mutable std::map<int, std::unique_ptr<device_copies>> _devices;
};

/**
 * @brief Read access to one device copy for the work its holder queues on the stream it was
 * acquired with.
 *
 * Releasing the lease, by destroying it or assigning over it, records that the work queued on that
 * stream so far may read the copy, so the cache frees the copy only after that work completes. A
 * lease must be released on the device it was acquired on, while its stream still exists.
 */
class string_constant_cache::lease {
 public:
  lease(lease&& other) noexcept;
  lease& operator=(lease&& other) noexcept;
  lease(lease const&)            = delete;
  lease& operator=(lease const&) = delete;
  ~lease();

  /** @brief Returns the device address of the first byte of the copy. */
  [[nodiscard]] char const* data() const noexcept { return _data; }

 private:
  friend class string_constant_cache;

  lease(device_copies& copies, char const* data, cudaEvent_t last_use, ::cuda::stream_ref stream);

  void release() noexcept;

  device_copies* _copies;
  char const* _data;
  cudaEvent_t _last_use;  ///< Owned by `_copies`: the event of `_stream`'s latest release.
  ::cuda::stream_ref _stream;
};

}  // namespace sirius::nvrtc
