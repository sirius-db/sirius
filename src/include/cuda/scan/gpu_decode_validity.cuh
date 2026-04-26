/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// GPU validity-mask helpers shared across decode codecs.
//
// Two kernels are exposed:
//   * `kernel_fill_valid` — initialises a cuDF bitmask to all-1s before any
//     per-segment validity bits are overlaid.
//   * `kernel_count_valid_bits` — single-block popcount + reduction that
//     produces the column's null count without round-tripping the mask
//     back to host.
//
// Definitions live in `gpu_native_decode.cu`; this header is imported by
// every other decode TU that produces a cuDF column with a null mask
// (e.g. `gpu_decode_batched_string.cu`).  Kept at file scope (not in an
// anonymous namespace) precisely so cross-TU launches resolve.
//===----------------------------------------------------------------------===//

#include <cuda_runtime.h>

#include <cstdint>

namespace sirius::cuda::scan {

__global__ void kernel_fill_valid(uint64_t* mask, uint32_t num_words);

__global__ void kernel_count_valid_bits(const uint64_t* __restrict__ mask,
                                        uint32_t num_words,
                                        uint32_t total_rows,
                                        uint32_t* __restrict__ d_valid_count);

}  // namespace sirius::cuda::scan
