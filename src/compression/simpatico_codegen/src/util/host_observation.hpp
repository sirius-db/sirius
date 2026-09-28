// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/stream>

#include <cstddef>

namespace simpatico {

/**
 * @brief Pinned host storage that one thread reuses as the destination of its device-to-host
 * observation copies.
 *
 * Allocations range from 64 KiB to `cap_bytes`; a request above the cap is refused with nullptr
 * instead of served. Thread-exit destruction may run after the CUDA context is gone, so the
 * destructor ignores the `cudaFreeHost` result, the convention `stream_pool::shutdown` follows.
 */
class pinned_staging_slab {
 public:
  static constexpr std::size_t cap_bytes = std::size_t{8} << 20;

  pinned_staging_slab() = default;
  ~pinned_staging_slab();
  pinned_staging_slab(pinned_staging_slab const&)            = delete;
  pinned_staging_slab& operator=(pinned_staging_slab const&) = delete;
  pinned_staging_slab(pinned_staging_slab&&)                 = delete;
  pinned_staging_slab& operator=(pinned_staging_slab&&)      = delete;

  /**
   * @brief Pinned storage of at least @p bytes, or nullptr when @p bytes exceeds `cap_bytes`.
   *
   * The returned storage is borrowed until the slab grows or is destroyed. Finish any GPU copy
   * using it before reserving again; reserve() does not wait for pending copies.
   *
   * @throw std::runtime_error if the pinned allocation fails
   */
  void* reserve(std::size_t bytes);

 private:
  static constexpr std::size_t initial_bytes = std::size_t{64} << 10;
  void* data_                                = nullptr;
  std::size_t capacity_                      = 0;
};

/**
 * @brief The calling thread's staging slab, created on first use and kept until the thread ends.
 */
pinned_staging_slab& thread_pinned_staging();

/**
 * @brief Copy @p bytes device bytes at @p source into @p destination and return once @p stream has
 * completed.
 *
 * Copies through thread_pinned_staging(), waits for @p stream, then copies the bytes to @p
 * destination. Reads above `pinned_staging_slab::cap_bytes` copy directly to @p destination
 * instead. The wait also covers other work queued on @p stream.
 *
 * On failure, stream completion is not guaranteed. The caller must complete any pending copy before
 * releasing its buffers or reusing the staging storage. decode_frame::read_bytes() attempts to
 * complete the stream before rethrowing.
 *
 * @throw std::runtime_error carrying the CUDA error string if the copy, the wait, or the pinned
 * allocation fails
 */
void read_device_bytes_completed(void* destination,
                                 void const* source,
                                 std::size_t bytes,
                                 ::cuda::stream_ref stream);

}  // namespace simpatico
