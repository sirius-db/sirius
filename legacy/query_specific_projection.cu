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

#include "../operator/cuda_helper.cuh"
#include "gpu_expression_executor.hpp"
#include "log/logging.hpp"

// l_extendedprice * (1 - l_discount)
// #4 * (1 + l_tax)
// ((L_EXTENDEDPRICE * (1.0 - L_DISCOUNT)) - (PS_SUPPLYCOST * L_QUANTITY))
// l_extendedprice * l_discount
// sum(case when nation = 1 then volume else 0 end) / sum(volume)
// (CAST(O_ORDERDATE AS DOUBLE) / 10000.0)
// sum(ps_supplycost * ps_availqty) * 0.0001
// sum(l_extendedprice) / 7.0

namespace duckdb {

template <typename T, int B, int I>
__global__ void common_arithmetic_expression(T *a, T *b, T* c, T* d, T *result, uint64_t N, int op_mode) {
    
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            if (op_mode == 0) {
                result[offset] = a[offset] * (1 - b[offset]);
            } else if (op_mode == 1) {
                result[offset] = a[offset] * (1 + b[offset]);
            } else if (op_mode == 2) {
                result[offset] = (a[offset] * (1 - b[offset])) - (c[offset] * d[offset]);
            } else if (op_mode == 3) {
                result[offset] = (a[offset] * 100 / b[offset]);
            } else {
                cudaAssert(0);
            }
        }
    }
}

template <int B, int I>
__global__ void extract_year(uint64_t *date, uint64_t *year, uint64_t N) {
    
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            year[offset] = date[offset] / 10000;
        }
    }
}

template <int B, int I>
__global__ void common_case_expression(uint64_t *a, uint64_t *b, double *result, uint64_t N, int op_mode) {
    
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            if (op_mode == 0) {
                if (a[offset] == 1) {
                    result[offset] = b[offset];
                } else {
                    result[offset] = 0;
                }
            } else if (op_mode == 1) {
                if ((a[offset] != 0) && a[offset] != 1) {
                    result[offset] = 1;
                } else {
                    result[offset] = 0;
                }                
            }  else if (op_mode == 2) {
                if ((a[offset] == 0) || a[offset] == 1) {
                    result[offset] = 1;
                } else {
                    result[offset] = 0;
                }  
            } else {
                cudaAssert(0);
            }
        }
    }
}

template <int B, int I>
__global__ void q14_case_expression(uint64_t *p_type, double *l_extendedprice, double *l_discount, uint64_t p_type_val1, uint64_t p_type_val2, double *result, uint64_t N) {
    
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            if ((p_type[offset] >= p_type_val1) && p_type[offset] < p_type_val2) {
                result[offset] = l_extendedprice[offset] * (1 - l_discount[offset]);
            } else {
                result[offset] = 0;
            }
        }
    }
}

template <int B, int I>
__global__ void q8_case_expression(uint64_t *nation, double *volume, uint64_t nation_val, double else_val, double *result, uint64_t N) {
    
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            if (nation[offset] == nation_val) {
                result[offset] = volume[offset];
            } else {
                result[offset] = else_val;
            }
        }
    }
}

template
__global__ void common_arithmetic_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(double *a, double *b, double* c, double* d, double *result, uint64_t N, int op_mode);
template
__global__ void common_case_expression<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *a, uint64_t *b, double *result, uint64_t N, int op_mode);
template
__global__ void q14_case_expression<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *p_type, double *l_extendedprice, double *l_discount, uint64_t p_type_val1, uint64_t p_type_val2, double *result, uint64_t N);
template
__global__ void q8_case_expression<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *nation, double *volume, uint64_t nation_val, double else_val, double *result, uint64_t N);
template
__global__ void extract_year<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *date, uint64_t *year, uint64_t N);


// Define the host function that launches the CUDA kernel
void commonArithmeticExpression(double *a, double *b, double* c, double* d, double *result, uint64_t N, int op_mode) {
    CHECK_ERROR();
    if (N == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Binary Expression Kernel");
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    common_arithmetic_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, c, d, result, N, op_mode);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

// Define the host function that launches the CUDA kernel
void extractYear(uint64_t* date, uint64_t *year, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Extract Year Kernel");
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    extract_year<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(date, year, N);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

// Define the host function that launches the CUDA kernel
void commonCaseExpression(uint64_t *a, uint64_t *b, double *result, uint64_t N, int op_mode) {
    CHECK_ERROR();
    if (N == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Common Case Expression Kernel");
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    common_case_expression<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, result, N, op_mode);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

// Define the host function that launches the CUDA kernel
void q14CaseExpression(uint64_t *p_type, double *l_extendedprice, double *l_discount, uint64_t p_type_val1, uint64_t p_type_val2, double *result, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Q14 Case Expression Kernel");
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    q14_case_expression<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(p_type, l_extendedprice, l_discount, p_type_val1, p_type_val2, result, N);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

void q8CaseExpression(uint64_t *nation, double *volume, uint64_t nation_val, double else_val, double *result, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Q14 Case Expression Kernel");
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    q8_case_expression<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(nation, volume, nation_val, else_val, result, N);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

} // namespace duckdb