/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// Thread-local pinned bounce ring for pageable→device H2D.
//
// Problem: every H2D in the scan/decode path reads from mmap'd DuckDB pages.
// Those pages are pageable, so cudaMemcpyAsync silently becomes synchronous
// via the driver's internal pinned-bounce buffer (~450 µs/call, A100).
//
// Solution: user-space bounce through per-thread pinned slots. memcpy into
// a slot at ~20 GB/s, then cudaMemcpyAsync from the slot hits the true
// async-DMA fast path.
//
// Why a 2-slot ring (not 1): with one slot, every call has to wait for the
// previous DMA to drain before overwriting (602 ms of cudaEventSynchronize
// observed on Q01 SF=20). With 2 slots, the CPU can memcpy into slot B
// while DMA drains slot A — we only pay a wait when both slots are still
// in flight (GPU slower than CPU memcpy). Cost: 2× memory (32 MB/thread
// default vs 16 MB/thread), still bounded and tiny vs anything else.
//
// Sizing: 16 MB per slot × 2 slots × 4 pipeline threads = 128 MB total
// pinned. SIRIUS_BOUNCE_MB env var tunes per-slot size; 0 disables.
//
// Fallback: transfers larger than a slot or a failed cudaHostAlloc drop
// to plain cudaMemcpyAsync — slower but correct.
//===----------------------------------------------------------------------===//

#include "log/logging.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace sirius::cuda::scan {

namespace detail {

struct bounce_slot {
  uint8_t*    buf    = nullptr;
  cudaEvent_t ev     = nullptr;
  bool        in_use = false;
};

struct pinned_bounce_ring {
  static constexpr int NUM_SLOTS = 2;
  bounce_slot slots[NUM_SLOTS];
  size_t      capacity = 0;   // per-slot capacity
  int         next     = 0;

  ~pinned_bounce_ring()
  {
    for (auto& s : slots) {
      if (s.ev) ::cudaEventDestroy(s.ev);
      if (s.buf) ::cudaFreeHost(s.buf);
    }
  }
};

inline size_t bounce_slab_capacity_bytes()
{
  static const size_t cap = []() {
    size_t mb = 16;  // per-slot default: covers observed max 16 MB coalesced run
    if (const char* env = std::getenv("SIRIUS_BOUNCE_MB")) {
      long v = std::atol(env);
      if (v >= 0) mb = static_cast<size_t>(v);
    }
    return mb * 1024ULL * 1024ULL;
  }();
  return cap;
}

inline pinned_bounce_ring& get_tls_bounce_ring()
{
  thread_local pinned_bounce_ring ring;
  if (ring.capacity == 0) {
    const size_t cap = bounce_slab_capacity_bytes();
    if (cap == 0) return ring;  // disabled → slots stay null; caller falls back
    bool all_ok = true;
    for (auto& s : ring.slots) {
      cudaError_t rc = ::cudaHostAlloc(reinterpret_cast<void**>(&s.buf), cap,
                                       cudaHostAllocPortable);
      if (rc != cudaSuccess) {
        ::cudaGetLastError();
        s.buf = nullptr;
        SIRIUS_LOG_WARN("[pinned_bounce] slot alloc failed ({} MB): {}",
                        cap / (1 << 20), ::cudaGetErrorString(rc));
        all_ok = false;
        break;
      }
      ::cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming);
    }
    if (all_ok) { ring.capacity = cap; }
  }
  return ring;
}

}  // namespace detail

/// Copy [src, src+bytes) → d_dst over `stream`, routing through the TLS
/// pinned ring when possible. Ping-pongs between ring slots so the next
/// memcpy can start while the prior DMA is still draining — only waits when
/// the chosen slot is still in flight. Falls back to plain cudaMemcpyAsync
/// when the transfer is larger than a slot or the ring is disabled.
inline void bounce_h2d_async(void* d_dst,
                             const void* src,
                             size_t bytes,
                             cudaStream_t stream)
{
  auto& ring = detail::get_tls_bounce_ring();
  if (ring.capacity == 0 || bytes > ring.capacity) {
    ::cudaMemcpyAsync(d_dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return;
  }
  auto& slot = ring.slots[ring.next];
  ring.next  = (ring.next + 1) % detail::pinned_bounce_ring::NUM_SLOTS;
  if (slot.in_use) { ::cudaEventSynchronize(slot.ev); }
  std::memcpy(slot.buf, src, bytes);
  ::cudaMemcpyAsync(d_dst, slot.buf, bytes, cudaMemcpyHostToDevice, stream);
  ::cudaEventRecord(slot.ev, stream);
  slot.in_use = true;
}

}  // namespace sirius::cuda::scan
