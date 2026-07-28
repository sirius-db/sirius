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

#include <api/simpatico_codegen.hpp>

#include <cstddef>
#include <cstdint>
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
  simpatico::compressed_table table;
};

}  // namespace sirius
