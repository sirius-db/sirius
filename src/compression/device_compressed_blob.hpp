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

// Internal header: only include from .cpp files that have the simpatico include path.

#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda/memory_resource>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace sirius {

/// Non-owning slab allocator that sub-allocates from a pre-allocated contiguous device buffer,
/// handing out `base + offsets[cursor++]` on each allocation and no-op'ing deallocation (the
/// payload buffer owns the backing memory). Intended solely for read_compressed_table_from_memory
/// so the reconstructed leaf channels reference slices of the payload without copying.
///
/// The advance cursor is held BY POINTER, not by value: rmm's device_async_resource_ref copies
/// the resource by value on each allocation, so an inline cursor would reset to 0 every call
/// (every leaf would alias offset 0). The cursor lives in the owning compressed_device_blob and
/// is shared across all copies.
struct slab_memory_resource {
  std::byte* base{nullptr};
  std::vector<std::uint64_t> const* offsets{nullptr};
  std::size_t* cursor{nullptr};

  [[nodiscard]] void* bump()
  {
    if (cursor == nullptr || offsets == nullptr) { return base; }
    // Hand out the next leaf slice. offsets is sized to exactly the number of leaf
    // allocations the re-read makes (zero-footprint leaves, which cudf allocates
    // nothing for, are given no slot), so the cursor stays in step with the read.
    std::size_t const k = (*cursor)++;
    if (k >= offsets->size()) { return base; }
    return base + (*offsets)[k];
  }

  [[nodiscard]] void* allocate(
    [[maybe_unused]] ::cuda::stream_ref stream,
    [[maybe_unused]] std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return bump();
  }

  void deallocate([[maybe_unused]] ::cuda::stream_ref stream,
                  void*,
                  [[maybe_unused]] std::size_t bytes,
                  [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
  }

  [[nodiscard]] void* allocate_sync(
    [[maybe_unused]] std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return bump();
  }

  void deallocate_sync(
    void*,
    [[maybe_unused]] std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
  }

  [[nodiscard]] bool operator==(slab_memory_resource const& other) const noexcept
  {
    return base == other.base && offsets == other.offsets;
  }

  constexpr friend void get_property(slab_memory_resource const&,
                                     ::cuda::mr::device_accessible) noexcept
  {
  }
};

/// Holds the single contiguous device payload and the compressed_table whose leaf channels_
/// are non-owning slices of that payload. Shared among all projections of the same pin chunk
/// via shared_ptr. payload must be declared before table so it is destroyed after table:
/// table's channels' device_buffers call slab_mr's (no-op) deallocate during their destruction,
/// then payload is freed by cudaFree.
struct compressed_device_blob {
  rmm::device_buffer payload;
  std::vector<std::uint64_t> offsets;
  std::size_t slab_cursor{0};  // owns the slab's advance cursor (see slab_memory_resource)
  slab_memory_resource slab_mr;

  /// The structural header the payload was staged from, retained so `table` can be
  /// rebuilt on first use. Small (a compact binary node array) and host-side, so
  /// holding it costs no device memory.
  std::vector<std::uint8_t> header;
  std::once_flag table_built;

  simpatico::compressed_table table;

  /// Reconstruct `table` on first call and return it. Thread-safe: a blob is shared
  /// between every projection and clone of the same chunk, and the decode path may
  /// reach several of them concurrently.
  ///
  /// Reconstruction is not free — for a non-fused codec (ans, lz4, snappy, dictionary,
  /// str_split, ALP) it takes decode scratch from @p scratch_mr — which is why the
  /// tiers that decompress a batch at most once defer it to here rather than paying
  /// it at staging time, when the device is at its most constrained.
  ///
  /// A throw leaves the flag unset, so a later call retries rather than handing back
  /// a half-built table.
  const simpatico::compressed_table& ensure_table(rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref scratch_mr);
};

/**
 * @brief Stage compressed leaf buffers into one contiguous device payload and
 *        reconstruct the compressed_table as views into it.
 *
 * Shared by the pin path and the task-output compression converter — the layout
 * rules below are subtle enough that a second copy would drift:
 *
 * - Leaves are placed at ALIGNED offsets, not the header's dense ones: nvcomp's
 *   batched decode requires aligned input pointers.
 * - Each slice is sized to the leaf's *reconstructed* footprint (`alloc_bytes`),
 *   not its compressed `size_bytes`: the re-read allocates each leaf at its
 *   decoded element count and a decode kernel touches the whole column, so a
 *   slice sized to size_bytes would run off its end into the next leaf.
 * - A zero-footprint buffer (e.g. the empty "output" leaf of an all-null chunk)
 *   gets no offset slot at all: cudf allocates nothing for it, so rmm never calls
 *   the slab and the cursor does not advance. Giving it a slot would hand every
 *   later leaf the wrong slice.
 *
 * @p reconstruct_now decides whether the compressed_table is built here or on
 * first use. Build it here only when the chunk will be decompressed many times
 * and the parse amortises — that is the pin path. The output and downgrade tiers
 * decompress a batch at most once, so for them an eager reconstruct is pure cost
 * paid at the worst possible moment; they pass false and let
 * compressed_device_blob::ensure_table do it later. Either way the header is
 * retained on the blob.
 *
 * @p scratch_mr supplies codec decode scratch during an eager reconstruct, kept
 * separate from the slab so it neither disturbs the slab's positional placement
 * nor lands inside the payload. It is unused when @p reconstruct_now is false.
 *
 * Synchronizes @p stream before returning, so the caller may release the source
 * buffers enumerated in @p buffers.
 */
/// Called with each buffer index once its bytes are safely in the payload, so a
/// caller can drop the source that owns it and keep peak residency down. The
/// copies are enqueued on @p stream and synchronized before the *next* callback,
/// so releasing inside it is ordered.
using buffer_copied_fn = std::function<void(std::size_t buffer_index)>;

[[nodiscard]] std::shared_ptr<compressed_device_blob> build_device_compressed_blob(
  std::span<const std::uint8_t> header,
  std::span<const simpatico::payload_buffer_ref> buffers,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref payload_mr,
  rmm::device_async_resource_ref scratch_mr,
  bool reconstruct_now,
  const buffer_copied_fn& on_buffer_copied = {});

}  // namespace sirius
