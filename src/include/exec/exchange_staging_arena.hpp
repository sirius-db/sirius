/*
 * Copyright 2025, Sirius Contributors.
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

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

namespace sirius::exec {

/// One device staging region for the cross-node exchange: packed batches are gathered into a
/// lease on the send side and land in a lease on the receive side, so the transport only ever
/// registers this one region.
///
/// The region is deliberately plain `cudaMalloc`, never pool / stream-ordered memory: UCX's
/// `cuda_ipc` path cannot export `cudaMallocAsync` allocations and silently degrades ~220x to
/// staged host copies — correct bytes, no error. Keeping the arena outside every pool is the
/// reason this class exists.
///
/// Leases are allocated from an address-ordered free list under a mutex and freed explicitly
/// (`lease` / `release` cross an FFI). Each release returns the block and coalesces it with its
/// free neighbours, so ANY released space is immediately reusable — the allocator has no bump
/// head and cannot drift: capacity bounds concurrent live bytes, not lifetime totals. A lease
/// remains one contiguous, `kAlignment`-aligned range, so it is still a valid RDMA target and a
/// valid `cudf::unpack` source.
class exchange_staging_arena {
 public:
  /// Every lease offset is a multiple of this, so any lease is a valid aligned transfer target.
  static constexpr std::uint64_t kAlignment = 256;

  /// Environment variable naming the arena capacity. Byte suffixes per
  /// `sirius::yaml::parse_bytes` ("512MiB", "1Gi", ...). Unset means no arena: every staging
  /// call fails loudly instead of silently taking a slow path.
  static constexpr const char* kCapacityEnvVar = "SIRIUS_EXCHANGE_STAGING_BYTES";

  /// Selects how the arena's device memory is allocated. Unset or `cudamalloc` (the default)
  /// keeps the plain `cudaMalloc` contract described above, which is correct for every
  /// single-host deployment and is the only path the unit tests exercise. `fabric` is opt-in and
  /// only needed when peers live on DIFFERENT hosts: it allocates through the CUDA driver's VMM
  /// API with `CU_MEM_HANDLE_TYPE_FABRIC`, the only allocation a peer on another host can map.
  /// `cudaMalloc`'s IPC handle is node-local by construction, so a multi-host exchange over it
  /// degrades to a host bounce (measured 0.32-0.43 GB/s between two GB200 hosts, far below what
  /// a cross-node exchange can accept); a fabric handle rides the MNNVL fabric instead (a
  /// standalone two-host harness measured 765 GB/s on the same pair).
  ///
  /// Requires a live IMEX domain and access to /dev/nvidia-caps-imex-channels; `cuMemCreate`
  /// fails loudly here rather than degrading silently later.
  static constexpr const char* kArenaKindEnvVar = "SIRIUS_EXCHANGE_STAGING_ARENA";

  /// Allocates the region with `cudaMalloc`.
  /// @throws sirius::invalid_input_exception on a zero capacity.
  /// @throws sirius::internal_exception when the allocation fails, naming the size and the
  ///         CUDA error.
  explicit exchange_staging_arena(std::uint64_t capacity_bytes);
  ~exchange_staging_arena();

  // The base pointer is handed out for transport registration, so the arena can never move.
  exchange_staging_arena(const exchange_staging_arena&)            = delete;
  exchange_staging_arena& operator=(const exchange_staging_arena&) = delete;
  exchange_staging_arena(exchange_staging_arena&&)                 = delete;
  exchange_staging_arena& operator=(exchange_staging_arena&&)      = delete;

  /// The arena sized from `kCapacityEnvVar`, or nullptr when the variable is unset.
  /// @throws sirius::invalid_input_exception on an unparsable value.
  static std::unique_ptr<exchange_staging_arena> from_env();

  /// `*arena`, or the loud not-configured error. Every call site that may run without an
  /// arena funnels through here so the operator-facing message has exactly one definition.
  /// @throws sirius::invalid_input_exception when `arena` is null.
  static exchange_staging_arena& require(exchange_staging_arena* arena);

  /// Lease `len` bytes; returns the lease's byte offset from `base()`.
  /// @throws sirius::invalid_input_exception on a zero-length request, or on exhaustion —
  ///         naming the requested, free, and capacity byte counts.
  std::uint64_t lease(std::uint64_t len);

  /// Return the lease at `offset`. The block goes back to the free list and coalesces with its
  /// free neighbours, so the space is immediately reusable regardless of release order.
  /// @throws sirius::invalid_input_exception when `offset` is not an outstanding lease
  ///         (double release, or a corrupted offset).
  void release(std::uint64_t offset);

  /// Device base address, for transport memory registration and for addressing leases.
  [[nodiscard]] std::uintptr_t base() const noexcept
  {
    return reinterpret_cast<std::uintptr_t>(base_);
  }

  [[nodiscard]] std::uint64_t capacity() const noexcept { return capacity_; }

  /// Leases currently held. Diagnostics: nonzero at quiesce means a leaked lease.
  [[nodiscard]] std::size_t outstanding() const;

  /// Peak sum of outstanding lease lengths — the working set the workload actually needed.
  /// With a coalescing free list this is also the peak arena occupancy, because released space
  /// is immediately reusable; a bump allocator would let the two diverge (that gap is drift).
  [[nodiscard]] std::uint64_t peak_live_bytes() const;

  /// Current sum of outstanding lease lengths.
  [[nodiscard]] std::uint64_t live_bytes() const;

  /// Largest single contiguous free block. `total_free() - largest_free()` is the external
  /// fragmentation; a lease no larger than this is guaranteed to succeed.
  [[nodiscard]] std::uint64_t largest_free() const;

  /// Sum of all free blocks.
  [[nodiscard]] std::uint64_t total_free() const;

 private:
  void* base_             = nullptr;
  std::uint64_t capacity_ = 0;

  //! VMM bookkeeping, set only on the `fabric` path; `fabric_handle_ == 0` means this arena was
  //! allocated with cudaMalloc and must be released with cudaFree. The VMM path has to unmap,
  //! free the reserved address range and release the physical handle -- cudaFree cannot do any
  //! of that and would silently leak the whole arena.
  unsigned long long fabric_handle_ = 0;
  std::uint64_t mapped_bytes_       = 0;

  mutable std::mutex mutex_;
  //! Free blocks, address-ordered and always coalesced: no two entries are adjacent.
  //! Seeded with the whole region at construction.
  std::map<std::uint64_t, std::uint64_t> free_;
  std::map<std::uint64_t, std::uint64_t> leases_;  // offset -> aligned length
  std::uint64_t live_bytes_      = 0;
  std::uint64_t peak_live_bytes_ = 0;

  //! Total free bytes and the largest single contiguous free block. Both are reported on
  //! exhaustion: the gap between them IS the external fragmentation, and it is the number that
  //! tells an operator whether to raise capacity or to fix retention.
  [[nodiscard]] std::uint64_t total_free_locked() const;
  [[nodiscard]] std::uint64_t largest_free_locked() const;
};

}  // namespace sirius::exec
