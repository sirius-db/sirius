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

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <string_view>

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
 * with every evaluator of the query, so the copies are freed with the query.
 *
 * Member functions are thread-safe. Each works on the calling thread's current device, which must
 * be the device of the stream it is given. The first acquisition of a pool on a device allocates
 * the copy from the given memory resource and uploads it on a stream the cache owns, so the upload
 * never waits for work already queued on the caller's stream; every acquisition then makes the
 * caller's stream wait for the upload on the device, so the host never waits for it.
 *
 * Destroying the cache waits on the host for its uploads and for the work queued before each
 * `lease` was released, then frees the copies. Every lease must be released before the cache is
 * destroyed.
 */
class string_constant_cache {
 public:
  class lease;

  string_constant_cache();
  ~string_constant_cache();
  string_constant_cache(string_constant_cache const&)            = delete;
  string_constant_cache& operator=(string_constant_cache const&) = delete;
  string_constant_cache(string_constant_cache&&)                 = delete;
  string_constant_cache& operator=(string_constant_cache&&)      = delete;

  /**
   * @brief Returns a lease on the current device's copy of @p pool, ordered before any work queued
   * on @p stream after this call.
   *
   * @p pool is read before the call returns. @p mr allocates the copy when this is the first
   * acquisition of @p pool on the current device.
   *
   * @throws internal_exception if @p pool is empty
   * @throws std::runtime_error carrying the CUDA error string if the stream, the allocation, the
   * copy, or an event fails
   */
  [[nodiscard]] lease acquire(std::string_view pool,
                              ::cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr) const;

  /** @brief Returns the number of uploads, one per distinct (pool, device) acquired. */
  [[nodiscard]] std::size_t upload_count_for_testing() const;

 private:
  struct device_copies;

  /** @brief Records that the work queued on @p stream so far may read the copies of @p device. */
  void record_use(int device, ::cuda::stream_ref stream) const noexcept;

  mutable std::mutex _mutex;
  mutable std::map<int, std::unique_ptr<device_copies>> _devices;
};

/**
 * @brief Read access to one device copy for the work its holder queues on the stream it was
 * acquired with.
 *
 * Releasing the lease, by destroying it or assigning over it, records that the work queued on that
 * stream so far may read the copy, so the cache frees the copy only after that work completes. A
 * lease must be released on the thread and device it was acquired on, while its stream still
 * exists.
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

  lease(string_constant_cache const& cache,
        char const* data,
        int device,
        ::cuda::stream_ref stream);

  void release() noexcept;

  string_constant_cache const* _cache;
  char const* _data;
  int _device;
  ::cuda::stream_ref _stream;
};

}  // namespace sirius::nvrtc
