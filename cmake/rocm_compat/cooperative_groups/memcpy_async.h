/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! @c <cooperative_groups/memcpy_async.h> shim for ROCm.
//!
//! HIP's cooperative_groups does NOT provide @c memcpy_async or @c wait.
//! This shim provides serial fallback implementations that work in device
//! code. The aligned_size_t alignment hint is ignored (generic copy).
//! Sirius's shared_staging.cuh has a manual-copy fallback for sub-word-aligned
//! data; this shim covers the aligned path.

#pragma once

#include <hip/hip_cooperative_groups.h>
#include "cuda/__memory/aligned_size.h"
#include <cstring>

namespace cooperative_groups {

/// Serial memcpy_async — copies n bytes from src to dst, one thread per block.
/// This is a fallback; the real CUDA version uses cp.async hardware on Ampere+.
/// On AMD, the generic copy is functionally correct (just slower).
template <typename Group, typename T>
__device__ void memcpy_async(Group const& /*block*/, void* dst, void const* src,
                             ::cuda::aligned_size_t<0> n) {
  // Only thread 0 copies; others wait at the barrier.
  if (threadIdx.x == 0) {
    __builtin_memcpy(dst, src, static_cast<std::size_t>(n));
  }
  __syncthreads();
}

/// Overload for explicit alignment (4/8/16) — same serial fallback.
template <std::size_t Alignment, typename Group>
__device__ void memcpy_async(Group const& /*block*/, void* dst, void const* src,
                             ::cuda::aligned_size_t<Alignment> n) {
  if (threadIdx.x == 0) {
    __builtin_memcpy(dst, src, static_cast<std::size_t>(n));
  }
  __syncthreads();
}

/// Wait for all async copies to complete. Serial impl: already complete
/// after __syncthreads in memcpy_async above.
template <typename Group>
__device__ void wait(Group const& /*block*/) {
  __syncthreads();
}

}  // namespace cooperative_groups
