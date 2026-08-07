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
/// Leases are bump-allocated under a mutex and freed explicitly (`lease` / `release` cross an
/// FFI, so RAII cannot carry them). They are short-lived by design — a send lease is released
/// after the transmit, a receive lease after the copy-out-on-arrival — so there is no free
/// list: each release drops the bump head back to the end of the highest lease still
/// outstanding (to the base when none remain), so trailing space is reclaimed immediately and
/// a long-lived lease pins at most the region up to its own end.
class exchange_staging_arena {
 public:
  /// Every lease offset is a multiple of this, so any lease is a valid aligned transfer target.
  static constexpr std::uint64_t kAlignment = 256;

  /// Environment variable naming the arena capacity. Byte suffixes per
  /// `sirius::yaml::parse_bytes` ("512MiB", "1Gi", ...). Unset means no arena: every staging
  /// call fails loudly instead of silently taking a slow path.
  static constexpr const char* kCapacityEnvVar = "SIRIUS_EXCHANGE_STAGING_BYTES";

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

  /// Return the lease at `offset`. The bump head drops back to the end of the highest lease
  /// still outstanding (to the base when none remain) — trailing reclamation, relying on
  /// leases being short-lived.
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

  /// Highest bump-head watermark ever reached — how much arena a workload actually needed.
  [[nodiscard]] std::uint64_t high_water() const;

 private:
  void* base_             = nullptr;
  std::uint64_t capacity_ = 0;

  mutable std::mutex mutex_;
  std::uint64_t head_       = 0;  // next free offset; always kAlignment-aligned
  std::uint64_t high_water_ = 0;
  std::map<std::uint64_t, std::uint64_t> leases_;  // offset -> aligned length
};

}  // namespace sirius::exec
