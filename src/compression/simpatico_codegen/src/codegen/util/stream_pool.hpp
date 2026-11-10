// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/stream>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <span>
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
  /// The streams as borrowed references, for the `std::span<const ::cuda::stream_ref>` overloads.
  [[nodiscard]] std::vector<::cuda::stream_ref> refs() const;

 private:
  // Hide the optional cuCascade backend from standalone consumers and keep
  // the public layout identical regardless of how the library was built.
  struct lease_state;
  std::unique_ptr<lease_state> leases_;
};

/// Wait for each distinct stream in @p streams, synchronizing only one that a query reports busy.
/// Tries every stream and returns the first failure. Allocation-free, so cleanup paths may call it.
cudaError_t synchronize_distinct(std::span<const ::cuda::stream_ref> streams) noexcept;

/// synchronize_distinct() for cleanup paths that must not throw: a failure is reported on stderr
/// after @p context.
void synchronize_distinct_or_log(std::span<const ::cuda::stream_ref> streams,
                                 char const* context) noexcept;

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
