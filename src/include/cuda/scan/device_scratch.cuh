/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// Thread-local device scratch arena for decode.
//
// Problem: decode per batch fires thousands of cudaMallocFromPoolAsync +
// cudaFreeAsync for small/medium scratch (bitpacking descriptors, RLE cumsum,
// per-column validity counts). RMM amortizes the underlying cudaMalloc but
// each pool alloc still costs ~40 µs of CPU time and implicitly forces a
// stream sync on the mempool lock.
//
// Solution: bump-allocated device arena per thread. Reset at end of each
// decode batch (after stream.synchronize() drains all GPU work that held
// pointers into the arena). Hot path = zero API calls — just offset math.
//
// Growth policy: on reset, if peak usage from the completed batch exceeded
// capacity, free+realloc to max(peak × 1.25, cap × 2), rounded up to 1 MB.
// After the first batch the arena is sized to the query's decode shape;
// steady-state batches never touch the allocator.
//
// Overflow: if an individual request can't fit the current arena (first
// batch, peak spike >25% above prior peak, or a fresh thread), fall back
// to cudaMallocAsync for that request and record the peak so the next
// reset grows. Correctness is preserved.
//
// Memory cost: capacity grows to the per-thread high-water mark (a few MB
// for TPC-H shapes), held for the lifetime of the pipeline thread. Bounded
// and tiny vs cuCascade's GB-scale pool.
//===----------------------------------------------------------------------===//

#include "log/logging.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace sirius::cuda::scan {

struct arena_alloc {
  void* ptr;          ///< Device pointer (arena slice or malloc'd).
  bool needs_free;    ///< True if ptr was allocated via cudaMallocAsync fallback.
};

namespace detail {

struct device_scratch_arena {
  void*  buf      = nullptr;
  size_t capacity = 0;
  size_t offset   = 0;
  size_t peak     = 0;

  ~device_scratch_arena()
  {
    // Thread exit: blocking free is fine here; cudaFreeAsync would race
    // with thread teardown. Any error is best-effort.
    if (buf) ::cudaFree(buf);
  }
};

inline device_scratch_arena& get_tls_arena()
{
  thread_local device_scratch_arena arena;
  return arena;
}

/// Round up `v` to the next multiple of `align` (power-of-two alignment).
inline size_t align_up(size_t v, size_t align)
{
  return (v + align - 1) & ~(align - 1);
}

}  // namespace detail

/// Allocate `bytes` from the per-thread scratch arena, 256 B aligned.
/// Returns `{ptr, needs_free}`: when `needs_free` is false the arena reclaims
/// the storage on the next `arena_reset`; when true the caller must
/// `cudaFreeAsync(ptr, stream)` at teardown (overflow fallback). Must be
/// paired with `arena_reset` after the stream sync that drains readers.
inline arena_alloc arena_allocate(size_t bytes, cudaStream_t stream, size_t align = 256)
{
  if (bytes == 0) return {nullptr, false};

  auto& a = detail::get_tls_arena();
  size_t aligned_off = detail::align_up(a.offset, align);
  if (aligned_off + bytes <= a.capacity) {
    void* p  = static_cast<uint8_t*>(a.buf) + aligned_off;
    a.offset = aligned_off + bytes;
    a.peak   = std::max(a.peak, a.offset);
    return {p, false};
  }

  // Arena too small for this request — record for next-reset grow and
  // fall back to stream-ordered malloc. The caller will free on teardown.
  a.peak = std::max(a.peak, aligned_off + bytes);
  void* p = nullptr;
  ::cudaMallocAsync(&p, bytes, stream);
  return {p, true};
}

/// Reset the per-thread arena and, if the last batch overflowed the current
/// capacity, grow to fit the next batch's peak. MUST be called only after
/// the stream has been synchronized (all pointers from `arena_allocate` are
/// no longer in use by any device work).
///
/// Growth policy: `new_cap = max(peak × 1.25, cap × 2)`, floored at 1 MB
/// and rounded up to 1 MB. 1.25× slack absorbs modest batch-to-batch
/// variance; 2× minimum growth prevents thrash when peak creeps upward.
inline void arena_reset(cudaStream_t stream)
{
  auto& a = detail::get_tls_arena();
  if (a.peak > a.capacity) {
    size_t target = std::max<size_t>(a.peak + a.peak / 4, a.capacity * 2);
    target        = std::max<size_t>(target, 1ULL << 20);  // floor at 1 MB
    target        = detail::align_up(target, 1ULL << 20);  // round to 1 MB
    if (a.buf) ::cudaFreeAsync(a.buf, stream);
    a.buf = nullptr;
    cudaError_t rc = ::cudaMallocAsync(&a.buf, target, stream);
    if (rc != cudaSuccess) {
      ::cudaGetLastError();
      SIRIUS_LOG_WARN("[decode_arena] grow to {} bytes failed: {}", target,
                      ::cudaGetErrorString(rc));
      a.buf      = nullptr;
      a.capacity = 0;
    } else {
      a.capacity = target;
    }
  }
  a.offset = 0;
  a.peak   = 0;
}

}  // namespace sirius::cuda::scan
