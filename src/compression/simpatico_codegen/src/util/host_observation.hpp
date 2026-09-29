// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/stream>

#include <cstddef>

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

}  // namespace simpatico
