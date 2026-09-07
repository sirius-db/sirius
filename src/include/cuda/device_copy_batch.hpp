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

#include <cudf/detail/utilities/cuda_memcpy.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace sirius::cuda {

// ---------------------------------------------------------------------------
// device_copy_batch — accumulate many copies, issue them as one
// ---------------------------------------------------------------------------
//
// A scatter of N small copies issued as N @c cudaMemcpyAsync calls pays N
// driver round-trips (each taking the driver's internal lock) to move data the
// DMA engines could be told about once.  Accumulate them here instead and
// @ref enqueue once: the whole batch reaches the driver as a single
// @c cudaMemcpyBatchAsync.
//
// The batch is direction-agnostic (@c cudaMemcpyDefault is inferred per copy
// from the pointers), so it serves host-to-device, device-to-device and
// device-to-host equally.  It owns no stream and no device state, so one
// instance can be filled, enqueued, cleared and refilled.
//
// ---- Source lifetime -------------------------------------------------------
//
// The underlying batch API is issued with @c cudaMemcpySrcAccessOrderStream:
// each source is read when the STREAM reaches that copy, not when @ref enqueue
// returns.  Every source buffer must therefore stay alive and unmodified until
// the stream has passed the copy — the same requirement a plain
// @c cudaMemcpyAsync from pinned memory already carries, so this is not a new
// constraint, but it is a load-bearing one: releasing a cache pin (or freeing a
// staging buffer) on enqueue rather than on stream completion is a use-after-
// free that only shows up as corrupt data.
//
// ---- Portability -----------------------------------------------------------
//
// @c cudf::detail::memcpy_batch_async is the version and stream gate:
//   * CUDA < 13.0 has no @c cudaMemcpyBatchAsync, so it loops
//     @c cudaMemcpyAsync — this class stays correct, it just stops batching.
//   * @c cudaMemcpyBatchAsync does not accept the default stream, so it loops
//     there too.  Passing a non-default stream is what actually buys the
//     batching; the cache's streams come from a pool and qualify.
//   * Entries with a null pointer or a zero size are rejected by the batch API,
//     so they are filtered out before submission.
// Callers get correct behaviour on every configuration and the fast path where
// it is available.

class device_copy_batch {
 public:
  device_copy_batch() = default;

  /// Pre-size the backing arrays for @p n copies.
  void reserve(std::size_t n)
  {
    _dsts.reserve(n);
    _srcs.reserve(n);
    _sizes.reserve(n);
  }

  /// Append a copy of @p bytes from @p src to @p dst.
  ///
  /// Entries are never fused, even when two consecutive copies look contiguous
  /// in both source and destination.  Adjacent addresses do not imply a single
  /// allocation: neighbouring staging chunks can come from different pool slabs
  /// and neighbouring device destinations from different rmm blocks, and a copy
  /// that straddles two separately-registered regions is rejected outright
  /// (cudaErrorInvalidValue) — or, worse, would read across a boundary the
  /// driver happens to tolerate.  Establishing that two buffers share one
  /// allocation costs more than the copies it would save.
  void add(void* dst, void const* src, std::size_t bytes)
  {
    if (bytes == 0 || dst == nullptr || src == nullptr) { return; }
    _dsts.push_back(dst);
    _srcs.push_back(src);
    _sizes.push_back(bytes);
    _bytes += bytes;
  }

  /// Number of copies that will be submitted (after fusing).
  [[nodiscard]] std::size_t count() const noexcept { return _dsts.size(); }

  /// Total bytes accumulated.
  [[nodiscard]] std::size_t bytes() const noexcept { return _bytes; }

  [[nodiscard]] bool empty() const noexcept { return _dsts.empty(); }

  /// Enqueue every accumulated copy on @p stream as a single batch.
  ///
  /// Does NOT clear — call @ref clear to reuse the instance.  Returns
  /// @c cudaSuccess for an empty batch.
  [[nodiscard]] cudaError_t enqueue(rmm::cuda_stream_view stream) const
  {
    if (_dsts.empty()) { return cudaSuccess; }
    return cudf::detail::memcpy_batch_async(
      _dsts.data(), _srcs.data(), _sizes.data(), _dsts.size(), stream);
  }

  void clear() noexcept
  {
    _dsts.clear();
    _srcs.clear();
    _sizes.clear();
    _bytes = 0;
  }

 private:
  std::vector<void*> _dsts;
  std::vector<void const*> _srcs;
  std::vector<std::size_t> _sizes;
  std::size_t _bytes{0};
};

}  // namespace sirius::cuda
