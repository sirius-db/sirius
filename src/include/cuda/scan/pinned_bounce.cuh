/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// Thread-local pinned bounce slab for pageable→device H2D.
//
// Problem: every H2D in the scan/decode path reads from mmap'd DuckDB pages.
// Those pages are pageable, so cudaMemcpyAsync silently becomes synchronous
// via the driver's internal pinned-bounce buffer (~450 µs/call, A100).
//
// Solution: user-space bounce through a per-thread pinned slab. memcpy into
// the slab at ~20 GB/s, then cudaMemcpyAsync from the slab hits the true
// async-DMA fast path. A per-slab cudaEvent serialises recycling so the next
// memcpy cannot overwrite bytes still being DMA'd.
//
// Sizing: default 16 MB per thread (SIRIUS_BOUNCE_MB env var). 0 disables
// and reverts to the old pageable path. Outside cucascade's host-memory
// accounting — direct cudaHostAlloc.
//
// Fallback: transfers larger than the slab or a failed cudaHostAlloc drop
// to plain cudaMemcpyAsync — slower but correct.
//===----------------------------------------------------------------------===//

#include "log/logging.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace sirius::cuda::scan {

namespace detail {

struct pinned_bounce_slab {
  uint8_t*    buf      = nullptr;
  size_t      capacity = 0;
  cudaEvent_t ev       = nullptr;
  bool        in_use   = false;

  ~pinned_bounce_slab()
  {
    if (ev) ::cudaEventDestroy(ev);
    if (buf) ::cudaFreeHost(buf);
  }
};

inline size_t bounce_slab_capacity_bytes()
{
  static const size_t cap = []() {
    size_t mb = 16;  // per-thread default: covers observed max 16 MB coalesced run
    if (const char* env = std::getenv("SIRIUS_BOUNCE_MB")) {
      long v = std::atol(env);
      if (v >= 0) mb = static_cast<size_t>(v);
    }
    return mb * 1024ULL * 1024ULL;
  }();
  return cap;
}

inline pinned_bounce_slab& get_tls_bounce_slab()
{
  thread_local pinned_bounce_slab slab;
  if (!slab.buf) {
    const size_t cap = bounce_slab_capacity_bytes();
    if (cap == 0) return slab;  // disabled → buf stays null; caller falls back
    cudaError_t rc = ::cudaHostAlloc(reinterpret_cast<void**>(&slab.buf), cap,
                                     cudaHostAllocPortable);
    if (rc != cudaSuccess) {
      ::cudaGetLastError();
      slab.buf = nullptr;
      SIRIUS_LOG_WARN("[pinned_bounce] slab alloc failed ({} MB): {}",
                      cap / (1 << 20), ::cudaGetErrorString(rc));
      return slab;
    }
    slab.capacity = cap;
    ::cudaEventCreateWithFlags(&slab.ev, cudaEventDisableTiming);
  }
  return slab;
}

}  // namespace detail

/// Copy [src, src+bytes) → d_dst over `stream`, routing through the TLS
/// pinned slab when possible. Falls back to plain cudaMemcpyAsync when the
/// transfer is larger than the slab or the slab is disabled/unavailable.
inline void bounce_h2d_async(void* d_dst,
                             const void* src,
                             size_t bytes,
                             cudaStream_t stream)
{
  auto& slab = detail::get_tls_bounce_slab();
  if (!slab.buf || bytes > slab.capacity) {
    ::cudaMemcpyAsync(d_dst, src, bytes, cudaMemcpyHostToDevice, stream);
    return;
  }
  if (slab.in_use) { ::cudaEventSynchronize(slab.ev); }
  std::memcpy(slab.buf, src, bytes);
  ::cudaMemcpyAsync(d_dst, slab.buf, bytes, cudaMemcpyHostToDevice, stream);
  ::cudaEventRecord(slab.ev, stream);
  slab.in_use = true;
}

}  // namespace sirius::cuda::scan
