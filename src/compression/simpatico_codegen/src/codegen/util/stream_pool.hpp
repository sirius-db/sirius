// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace simpatico {

/// A fixed set of CUDA streams for cross-column parallelism, submitted from
/// the calling thread. Sirius builds lease these from cuCascade; standalone
/// builds own CUDA streams directly. Per-column operations are single-stream.
struct stream_pool {
  std::vector<cudaStream_t> streams;

  stream_pool();
  ~stream_pool();

  stream_pool(const stream_pool&)            = delete;
  stream_pool& operator=(const stream_pool&) = delete;
  stream_pool(stream_pool&& other) noexcept;
  stream_pool& operator=(stream_pool&& other) noexcept;

  /// Initialize with n streams. Returns false on error.
  bool init(size_t n);
  /// Synchronize and release streams (return leases in Sirius builds).
  void shutdown();
  /// Synchronize all streams. Returns the first failure while still
  /// synchronizing the remaining streams, so an async kernel error is
  /// reported instead of silently dropped. Callable from destructors.
  cudaError_t sync_all();

 private:
  // Hide the optional cuCascade backend from standalone consumers and keep
  // the public layout identical regardless of how the library was built.
  struct lease_state;
  std::unique_ptr<lease_state> leases_;
};

/// A pool of @p n streams belonging to the calling thread and the CURRENT
/// device, created on first use and reused for the thread's lifetime.
///
/// Keyed by device because CUDA streams are device-bound: launching on a stream
/// that belongs to another device fails. A single thread_local pool would hand
/// device-0 streams to a thread that has since switched to device 1, so callers
/// that may see more than one device must come through here rather than holding
/// their own.
///
/// Never destroyed before the thread ends, which is deliberate: buffers record
/// the stream they were built on for their eventual async free, so a handle has
/// to stay valid for as long as anything allocated on it might be freed.
///
/// @throws std::runtime_error if the streams cannot be created.
stream_pool& thread_device_stream_pool(size_t n);

}  // namespace simpatico
