/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace sirius::cuda::scan {

__global__ void kernel_fill_valid(uint64_t* mask, uint32_t num_words);

__global__ void kernel_count_valid_bits(const uint64_t* __restrict__ mask,
                                        uint32_t num_words,
                                        uint32_t total_rows,
                                        uint32_t* __restrict__ d_valid_count);

}  // namespace sirius::cuda::scan
