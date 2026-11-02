// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/stream>

#include <cstddef>
#include <span>

namespace simpatico {

/// The largest read that read_device_bytes_completed() stages through pinned storage.
inline constexpr std::size_t pinned_staging_cap_bytes = std::size_t{8} << 20;

/**
 * @brief Copy @p bytes device bytes at @p source into @p destination and return once @p stream has
 * completed.
 *
 * Copies through pinned host storage that the calling thread keeps for its lifetime (growing from
 * 64 KiB up to `pinned_staging_cap_bytes`), waits for @p stream, then copies the bytes to
 * @p destination. Larger reads copy directly to @p destination. The wait also covers other work
 * queued on @p stream.
 *
 * On failure, waits for @p stream before rethrowing, so @p destination may be local storage.
 *
 * @throw std::runtime_error carrying the CUDA error string if the copy, the wait, or the pinned
 * allocation fails
 */
void read_device_bytes_completed(void* destination,
                                 void const* source,
                                 std::size_t bytes,
                                 ::cuda::stream_ref stream);

/**
 * @brief Copy @p bytes host bytes at @p source into device memory at @p destination on @p stream.
 *
 * Stages through pinned host storage that the calling thread keeps for its lifetime (growing from
 * 64 KiB up to `pinned_staging_cap_bytes`), waiting for any earlier copy from that storage before
 * rewriting it, so @p source may be reclaimed as soon as the call returns without blocking the
 * host on a pageable staging copy. Larger writes copy directly from @p source and wait for
 * @p stream before returning.
 *
 * @throw std::runtime_error carrying the CUDA error string if the copy, the event synchronization,
 * or the pinned allocation fails
 */
void write_device_bytes_async(void* destination,
                              void const* source,
                              std::size_t bytes,
                              ::cuda::stream_ref stream);

/// One device-to-host copy for the batched read_device_bytes_completed().
struct device_read {
  void* destination;
  void const* source;
  std::size_t bytes;
};

/**
 * @brief Perform every read of @p reads and return once @p stream has completed, waiting once.
 *
 * Behaves like the single read, staging the reads together through the same pinned storage while
 * they fit. A caller that has device values to observe besides one it must wait for anyway reads
 * them all with one wait.
 */
void read_device_bytes_completed(std::span<device_read const> reads, ::cuda::stream_ref stream);

}  // namespace simpatico
