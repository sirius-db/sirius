// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace simpatico {

/**
 * @brief Pinned host storage that one thread reuses as the destination of its device-to-host
 * observation copies.
 *
 * Capacity doubles from 64 KiB up to `cap_bytes`; a request above the cap is refused with nullptr
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
 * Decode-time host observations that need one copy completed before the caller continues go through
 * this call; the row-id selection helpers (`chunk_row_set_build`, `row_id_space`) pair two copies
 * with one wait and still copy into pageable memory. The copy lands in the calling thread's pinned
 * staging slab (`thread_pinned_staging`) and completes with an explicit `cudaStreamSynchronize` on
 * @p stream, after which the bytes are moved into @p destination; a copy into pageable memory would
 * instead wait inside the driver and serialize other threads' CUDA calls behind it. Only a read
 * above `pinned_staging_slab::cap_bytes` copies straight into @p destination. Because every read
 * completes before returning, the slab is idle again at the next reservation; do not hand it to
 * work that outlives the call. The wait also covers whatever other work is queued on @p stream.
 *
 * @throw std::runtime_error carrying the CUDA error string if the copy, the wait, or the pinned
 * allocation fails
 */
void read_device_bytes_completed(void* destination,
                                 void const* source,
                                 std::size_t bytes,
                                 rmm::cuda_stream_view stream);

}  // namespace simpatico
