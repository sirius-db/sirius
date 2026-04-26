/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

//===----------------------------------------------------------------------===//
// GPU validity-mask helpers shared by every decode codec.
//
// `kernel_fill_valid` initialises a cuDF bitmask to all-1s before any
// per-segment validity bits are overlaid. `kernel_count_valid_bits` runs
// a single-block popcount + reduction to produce the column's null count
// without round-tripping the mask back to host.
//
// Definitions live in gpu_native_decode.cu — both kernels are tiny and
// share the validity-decode call site there. This header declares them
// so other decode .cu files can launch them directly when they want to
// produce a cuDF column with the same null-mask shape.
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
