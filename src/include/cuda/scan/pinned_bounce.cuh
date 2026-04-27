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
// in flight (GPU slower than CPU memcpy). Cost: 2× memory (32 MB/thread),
// still bounded and tiny vs anything else.
//
// Sizing: 16 MB per slot × 2 slots × 4 pipeline threads = 128 MB total
// pinned. The per-slot capacity is a hardcoded default chosen to cover the
// observed worst-case coalesced H2D run in TPC-H/TPC-DS; tune here if a
// new workload regresses.
//
// Fallback: transfers larger than a slot or a failed cudaHostAlloc drop
// to plain cudaMemcpyAsync — slower but correct.
//===----------------------------------------------------------------------===//

#include "log/logging.hpp"

#include <cuda_runtime.h>

#include <cstring>

namespace sirius::cuda::scan {

namespace detail {

/// Per-slot pinned-bounce buffer size. 16 MB covers the largest coalesced
/// H2D run seen on TPC-H SF=20/100; larger workloads can bump this, at a
/// 2× memory cost per pipeline thread.
inline constexpr size_t BOUNCE_SLOT_BYTES = 16ULL * 1024 * 1024;

/// Transfers at or below this size bypass the ring entirely: the
/// cudaEventRecord + potential cudaEventSynchronize pair (~5–10 µs) exceeds
/// the pageable-bounce savings on sub-4 KB copies. Fixes tiny-query tail
/// regressions seen on Q11/Q16/Q22 where startup-dominated transfers pay
/// ring overhead without benefiting from DMA overlap.
inline constexpr size_t BOUNCE_SMALL_BYPASS = 4ULL * 1024;

struct bounce_slot {
  uint8_t* buf   = nullptr;
  cudaEvent_t ev = nullptr;
  bool in_use    = false;
};

struct pinned_bounce_ring {
  static constexpr int NUM_SLOTS = 2;
  bounce_slot slots[NUM_SLOTS];
  size_t capacity = 0;  // per-slot capacity (0 = not yet initialized)
  int next        = 0;

  ~pinned_bounce_ring()
  {
    for (auto& s : slots) {
      if (s.ev) ::cudaEventDestroy(s.ev);
      if (s.buf) ::cudaFreeHost(s.buf);
    }
  }
};

inline pinned_bounce_ring& get_tls_bounce_ring()
{
  thread_local pinned_bounce_ring ring;
  if (ring.capacity == 0) {
    bool all_ok = true;
    for (auto& s : ring.slots) {
      cudaError_t rc =
        ::cudaHostAlloc(reinterpret_cast<void**>(&s.buf), BOUNCE_SLOT_BYTES, cudaHostAllocPortable);
      if (rc != cudaSuccess) {
        ::cudaGetLastError();
        s.buf = nullptr;
        SIRIUS_LOG_WARN("[pinned_bounce] slot alloc failed ({} MB): {}",
                        BOUNCE_SLOT_BYTES / (1 << 20),
                        ::cudaGetErrorString(rc));
        all_ok = false;
        break;
      }
      // cudaEventCreate can fail on driver/device exhaustion; ignoring the
      // status leaves s.ev null and a later cudaEventRecord/Query corrupts the
      // stream silently. Treat it the same as a host-alloc failure.
      cudaError_t ev_rc = ::cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming);
      if (ev_rc != cudaSuccess) {
        ::cudaGetLastError();
        s.ev = nullptr;
        SIRIUS_LOG_WARN("[pinned_bounce] event create failed: {}", ::cudaGetErrorString(ev_rc));
        all_ok = false;
        break;
      }
    }
    if (!all_ok) {
      // Release any slots we did manage to allocate before the failure so a
      // transient cudaHostAlloc error doesn't permanently pin tens of MB
      // for the lifetime of this thread.
      for (auto& s : ring.slots) {
        if (s.ev) {
          ::cudaEventDestroy(s.ev);
          s.ev = nullptr;
        }
        if (s.buf) {
          ::cudaFreeHost(s.buf);
          s.buf = nullptr;
        }
      }
    } else {
      ring.capacity = BOUNCE_SLOT_BYTES;
    }
  }
  return ring;
}

}  // namespace detail

/// Copy [src, src+bytes) → d_dst over `stream`, routing through the TLS
/// pinned ring when possible. Ping-pongs between ring slots so the next
/// memcpy can start while the prior DMA is still draining — only waits when
/// the chosen slot is still in flight. Falls back to plain cudaMemcpyAsync
/// when the transfer exceeds slot capacity or the ring failed to initialize.
inline void bounce_h2d_async(void* d_dst, const void* src, size_t bytes, cudaStream_t stream)
{
  if (bytes <= detail::BOUNCE_SMALL_BYPASS) {
    ::cudaMemcpyAsync(d_dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return;
  }
  auto& ring = detail::get_tls_bounce_ring();
  if (ring.capacity == 0 || bytes > ring.capacity) {
    ::cudaMemcpyAsync(d_dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return;
  }
  auto& slot = ring.slots[ring.next];
  ring.next  = (ring.next + 1) % detail::pinned_bounce_ring::NUM_SLOTS;
  if (slot.in_use) {
    // Query before sync: with the 2-slot ring on healthy hardware the
    // prior DMA has typically already drained by the time we cycle back,
    // so cudaEventQuery returns cudaSuccess and we skip the cudaEvent-
    // Synchronize driver round-trip entirely.  Any non-success status
    // (including cudaErrorNotReady) falls through to sync, which is the
    // correct behaviour for both pending events and surfacing errors.
    if (::cudaEventQuery(slot.ev) != cudaSuccess) { ::cudaEventSynchronize(slot.ev); }
  }
  std::memcpy(slot.buf, src, bytes);
  ::cudaMemcpyAsync(d_dst, slot.buf, bytes, cudaMemcpyHostToDevice, stream);
  ::cudaEventRecord(slot.ev, stream);
  slot.in_use = true;
}

}  // namespace sirius::cuda::scan
