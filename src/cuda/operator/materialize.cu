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

#include "cuda_helper.cuh"
#include "gpu_columns.hpp"
#include "log/logging.hpp"

#include <chrono>
#include <cmath>

namespace duckdb {

__device__ uint32_t warp_bitmask_set(uint32_t bitset) {
    uint32_t lane_id = threadIdx.x % warpSize;
    uint32_t set = (bitset << lane_id);
    for (int offset = 16; offset >= 1; offset /= 2) {
        set |= __shfl_down_sync(0xFFFFFFFF, set, offset);
    }
    return set;
}


template <typename T, int B, int I>
__global__ void materialize_expression_with_null(const T *a, T* result, uint32_t* mask, uint32_t* out_mask, uint64_t *row_ids, uint64_t N) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    uint32_t isvalid[I];
    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        isvalid[ITEM] = 0;
    }

    uint64_t mask_tile_size = (B * I) / 32;
    uint64_t mask_tile_offset = blockIdx.x * mask_tile_size;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t items_ids = row_ids[tile_offset + threadIdx.x + ITEM * B];
            uint64_t word_offset = items_ids / 32;
            uint64_t bit_offset = items_ids % 32;
            isvalid[ITEM] = mask[word_offset] & (1 << bit_offset) ? 1 : 0;
            result[tile_offset + threadIdx.x + ITEM * B] = a[items_ids];
        }
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        uint32_t set = warp_bitmask_set(isvalid[ITEM]);
        __syncwarp();
        if (threadIdx.x % 32 == 0 && threadIdx.x + ITEM * B < num_tile_items) {
            out_mask[mask_tile_offset + (threadIdx.x / 32) + ITEM * (B / 32)] = set;
        }
        __syncwarp();
    }
}


__global__ void materialize_offset_with_null(uint64_t* offset, uint64_t* result_length, uint32_t* mask, uint32_t* out_mask, uint64_t* row_ids, size_t N) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    bool isvalid = 0;
    if(tid < N) {
        uint64_t copy_row_id = row_ids[tid];
        uint64_t new_length = offset[copy_row_id + 1] - offset[copy_row_id];
        result_length[tid] = new_length;

        uint64_t word_offset = copy_row_id / 32;
        uint64_t bit_offset = copy_row_id % 32;
        isvalid = mask[word_offset] & (1 << bit_offset) ? 1 : 0;
    }

    uint32_t set = warp_bitmask_set(isvalid);
    __syncwarp();
    if (threadIdx.x % 32 == 0 && tid < N) {
        out_mask[tid / 32] = set;
    }
    __syncwarp();
}

template <typename T, int B, int I>
__global__ void materialize_without_null(const T *a, T* result, uint64_t *row_ids, uint64_t N) {

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
            uint64_t items_ids = row_ids[tile_offset + threadIdx.x + ITEM * B];
            result[tile_offset + threadIdx.x + ITEM * B] = a[items_ids];
        }
    }

}

__global__ void materialize_offset(uint64_t* offset, uint64_t* result_length, uint64_t* row_ids, size_t N) {
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N) {
        uint64_t copy_row_id = row_ids[tid];
        uint64_t new_length = offset[copy_row_id + 1] - offset[copy_row_id];
        result_length[tid] = new_length;
    }
}

__global__ void materialize_string_per_warp(
    const uint8_t* __restrict__ data,
    uint8_t* __restrict__ result,
    const uint64_t* __restrict__ input_offset,
    const uint64_t* __restrict__ materialized_offset,
    const uint64_t* __restrict__ row_ids,
    size_t num_rows)
{
    constexpr int WARP_SIZE = 32;
    size_t warp_id = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t lane = threadIdx.x % WARP_SIZE;
    if (warp_id >= num_rows) return;

    uint64_t copy_row_id = row_ids[warp_id];
    uint64_t input_start_idx = input_offset[copy_row_id];
    uint64_t input_length = input_offset[copy_row_id + 1] - input_start_idx;
    uint64_t output_start_idx = materialized_offset[warp_id];

    const uint8_t* __restrict__ src = data + input_start_idx;
    uint8_t* __restrict__ dst = result + output_start_idx;

    uintptr_t src_addr = reinterpret_cast<uintptr_t>(src);
    uintptr_t dst_addr = reinterpret_cast<uintptr_t>(dst);

    bool aligned16 = ((src_addr | dst_addr) & 15) == 0;
    bool aligned8  = ((src_addr | dst_addr) & 7) == 0;
    bool aligned4  = ((src_addr | dst_addr) & 3) == 0;
    bool aligned2  = ((src_addr | dst_addr) & 1) == 0;

    // 16-byte vectorized copy
    if (aligned16 && input_length >= 16) {
        for (uint64_t i = lane * 16; i + 16 <= input_length; i += WARP_SIZE * 16) {
            *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
        }
        uint64_t rem_start = (input_length / 16) * 16;
        for (uint64_t i = lane + rem_start; i < input_length; i += WARP_SIZE) {
            dst[i] = src[i];
        }
    }
    // 8-byte path
    else if (aligned8 && input_length >= 8) {
        for (uint64_t i = lane * 8; i + 8 <= input_length; i += WARP_SIZE * 8) {
            *reinterpret_cast<uint64_t*>(dst + i) = *reinterpret_cast<const uint64_t*>(src + i);
        }
        uint64_t rem_start = (input_length / 8) * 8;
        for (uint64_t i = lane + rem_start; i < input_length; i += WARP_SIZE) {
            dst[i] = src[i];
        }
    }
    // 4-byte path
    else if (aligned4 && input_length >= 4) {
        for (uint64_t i = lane * 4; i + 4 <= input_length; i += WARP_SIZE * 4) {
            *reinterpret_cast<uint32_t*>(dst + i) = *reinterpret_cast<const uint32_t*>(src + i);
        }
        uint64_t rem_start = (input_length / 4) * 4;
        for (uint64_t i = lane + rem_start; i < input_length; i += WARP_SIZE) {
            dst[i] = src[i];
        }
    }
    // 2-byte path
    else if (aligned2 && input_length >= 2) {
        for (uint64_t i = lane * 2; i + 2 <= input_length; i += WARP_SIZE * 2) {
            *reinterpret_cast<uint16_t*>(dst + i) = *reinterpret_cast<const uint16_t*>(src + i);
        }
        uint64_t rem_start = (input_length / 2) * 2;
        for (uint64_t i = lane + rem_start; i < input_length; i += WARP_SIZE) {
            dst[i] = src[i];
        }
    }
    // byte-by-byte fallback
    else {
        for (uint64_t i = lane; i < input_length; i += WARP_SIZE) {
            dst[i] = src[i];
        }
    }
}

__global__ void materialize_string_per_thread(uint8_t* data, uint8_t* result, uint64_t* input_offset, uint64_t* materialized_offset, uint64_t* row_ids, size_t num_rows) {
    size_t tid = threadIdx.x + size_t(blockIdx.x) * blockDim.x;
    if (tid < num_rows) {
        uint64_t copy_row_id = row_ids[tid];
        uint64_t input_start_idx = input_offset[copy_row_id];
        uint64_t input_length = input_offset[copy_row_id + 1] - input_offset[copy_row_id];
        uint64_t output_start_idx = materialized_offset[tid];
        memcpy(result + output_start_idx, data + input_start_idx, input_length * sizeof(uint8_t));
    }
}

template
__global__ void materialize_without_null<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, int* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_without_null<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_without_null<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, float* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_without_null<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_without_null<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, uint8_t* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_without_null<int64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const int64_t *a, int64_t* result, uint64_t *row_ids, uint64_t N);

template
__global__ void materialize_expression_with_null<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, int* result, uint32_t* mask, uint32_t* out_mask, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression_with_null<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, uint32_t* mask, uint32_t* out_mask, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression_with_null<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, float* result, uint32_t* mask, uint32_t* out_mask, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression_with_null<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, uint32_t* mask, uint32_t* out_mask, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression_with_null<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, uint8_t* result, uint32_t* mask, uint32_t* out_mask, uint64_t *row_ids, uint64_t N);

template <typename T>
void materializeWithoutNull(T *a, T*& result, uint64_t *row_ids, uint64_t result_len) {
    CHECK_ERROR();
    if (result_len == 0) {
        SIRIUS_LOG_DEBUG("result_len is 0");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    SIRIUS_LOG_DEBUG("Launching Materialize Kernel");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    result = gpuBufferManager->customCudaMalloc<T>(result_len, 0, 0);
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    materialize_without_null<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(result_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, result, row_ids, result_len);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(a), 0);
    CHECK_ERROR();
    STOP_TIMER();
}

template <typename T>
void materializeExpression(T *a, T*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type*& out_mask) {
    CHECK_ERROR();
    if (result_len == 0) {
        SIRIUS_LOG_DEBUG("result_len is 0");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    SIRIUS_LOG_DEBUG("Launching Materialize Kernel");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    result = gpuBufferManager->customCudaMalloc<T>(result_len, 0, 0);

    if (mask != nullptr) {
        out_mask = gpuBufferManager->customCudaMalloc<uint32_t>(getMaskBytesSize(result_len) / sizeof(uint32_t), 0, 0);
    } else {
        out_mask = nullptr;
    }

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    if (mask != nullptr) {
        materialize_expression_with_null<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(result_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, result, mask, out_mask, row_ids, result_len);
    } else {
        materialize_without_null<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(result_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, result, row_ids, result_len);
    }
    
    CHECK_ERROR();
    cudaDeviceSynchronize();
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(a), 0);
    if (mask != nullptr) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(mask), 0);
    }
    CHECK_ERROR();
    STOP_TIMER();
}

// void materializeString(uint8_t* data, uint64_t* offset, uint8_t* &result, uint64_t* &result_offset, uint64_t* row_ids, uint64_t* &result_bytes, uint64_t result_len) {
//     CHECK_ERROR();
//     if (result_len == 0) {
//         SIRIUS_LOG_DEBUG("result_len is 0");
//         return;
//     }
//     SETUP_TIMING();
//     START_TIMER();
//     SIRIUS_LOG_DEBUG("Launching Materialize String Kernel");
//     GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
//     //allocate temp memory and copying keys
//     uint64_t* temp_len = gpuBufferManager->customCudaMalloc<uint64_t>(result_len + 1, 0, 0);
//     result_offset = gpuBufferManager->customCudaMalloc<uint64_t>(result_len + 1, 0, 0);

//     cudaMemset(temp_len + result_len, 0, sizeof(uint64_t));
//     CHECK_ERROR();

//     // Copy over the offsets
//     uint64_t num_blocks = std::max((uint64_t) 1, (uint64_t) (result_len + BLOCK_THREADS - 1)/BLOCK_THREADS);
//     materialize_offset<<<num_blocks, BLOCK_THREADS>>>(offset, temp_len, row_ids, result_len);
//     cudaDeviceSynchronize();
//     CHECK_ERROR();

//     //cub scan
//     void* d_temp_storage = nullptr;
//     size_t temp_storage_bytes = 0;
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp_len, result_offset, result_len + 1);

//     // Allocate temporary storage for exclusive prefix sum
//     d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

//     // Run exclusive prefix sum
//     cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp_len, result_offset, result_len + 1);
//     CHECK_ERROR();

//     result_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
//     cudaMemcpy(result_bytes, result_offset + result_len, sizeof(uint64_t), cudaMemcpyDeviceToHost);

//     CHECK_ERROR();

//     result = gpuBufferManager->customCudaMalloc<uint8_t>(result_bytes[0], 0, 0);

//     materialize_string<<<num_blocks, BLOCK_THREADS>>>(data, result, offset, result_offset, row_ids, result_len);
//     cudaDeviceSynchronize();

//     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(temp_len), 0);
//     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
//     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(data), 0);
//     gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset), 0);
//     CHECK_ERROR();
//     STOP_TIMER();
// }

void materializeString(uint8_t* data, uint64_t* offset, uint8_t* &result, uint64_t* &result_offset, uint64_t* row_ids, uint64_t* &result_bytes, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask) {
    CHECK_ERROR();
    if (result_len == 0) {
        SIRIUS_LOG_DEBUG("result_len is 0");
        return;
    }
    SETUP_TIMING();
    START_TIMER();
    SIRIUS_LOG_DEBUG("Launching Materialize String Kernel");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    //allocate temp memory and copying keys
    uint64_t* temp_len = gpuBufferManager->customCudaMalloc<uint64_t>(result_len + 1, 0, 0);
    result_offset = gpuBufferManager->customCudaMalloc<uint64_t>(result_len + 1, 0, 0);

    if (mask != nullptr) {
        out_mask = gpuBufferManager->customCudaMalloc<uint32_t>(getMaskBytesSize(result_len) / sizeof(uint32_t), 0, 0);
    } else {
        out_mask = nullptr;
    }

    cudaMemset(temp_len + result_len, 0, sizeof(uint64_t));
    CHECK_ERROR();

    // Copy over the offsets
    uint64_t num_blocks = std::max((uint64_t) 1, (uint64_t) (result_len + BLOCK_THREADS - 1)/BLOCK_THREADS);

    if (mask != nullptr) {
        materialize_offset_with_null<<<num_blocks, BLOCK_THREADS>>>(offset, temp_len, mask, out_mask, row_ids, result_len);
    } else {
        materialize_offset<<<num_blocks, BLOCK_THREADS>>>(offset, temp_len, row_ids, result_len);
    }
    
    cudaDeviceSynchronize();
    CHECK_ERROR();

    //cub scan
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp_len, result_offset, result_len + 1);

    // Allocate temporary storage for exclusive prefix sum
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp_len, result_offset, result_len + 1);
    CHECK_ERROR();

    result_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    cudaMemcpy(result_bytes, result_offset + result_len, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    CHECK_ERROR();

    result = gpuBufferManager->customCudaMalloc<uint8_t>(result_bytes[0], 0, 0);

    // Adaptivly choose between thread-per-row and warp-per-row based on average string len,
    // so far use a empricial threshold which can be refined later
    uint64_t result_avg_len = *result_bytes / result_len;
    if (result_avg_len > 8) {
        constexpr int WARP_SIZE = 32;
        uint64_t warps_per_block = BLOCK_THREADS / WARP_SIZE;
        num_blocks = std::max<uint64_t>(1, (result_len + warps_per_block - 1) / warps_per_block);
        materialize_string_per_warp<<<num_blocks, BLOCK_THREADS>>>(data, result, offset, result_offset, row_ids, result_len);
    } else {
        materialize_string_per_thread<<<num_blocks, BLOCK_THREADS>>>(data, result, offset, result_offset, row_ids, result_len);
    }

    cudaDeviceSynchronize();

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(temp_len), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(data), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset), 0);
    if (mask != nullptr) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(mask), 0);
    }
    CHECK_ERROR();
    STOP_TIMER();
}

__global__ void create_cpu_strings(duckdb_string_type* gpu_strings, char* cpu_chars_buffer, char* gpu_chars, uint64_t* string_offsets, size_t num_strings, size_t inline_threshold) { 
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < num_strings) {
        uint64_t str_offset = string_offsets[idx];
        uint64_t str_length = string_offsets[idx + 1] - str_offset;
        
        // Use the threshold to determine if we should inline the string or not
        duckdb_string_type& curr_string = gpu_strings[idx];
        if(str_length <= inline_threshold) {
            curr_string.value.inlined.length = str_length;
            char* gpu_data_ptr = gpu_chars + str_offset;
            memcpy(curr_string.value.inlined.inlined, gpu_data_ptr, str_length);
        } else {
            curr_string.value.pointer.length = str_length;
            curr_string.value.pointer.ptr = cpu_chars_buffer + str_offset;
        }
    }
}

void materializeStringColumnToDuckdbFormat(shared_ptr<GPUColumn> column, char* column_char_write_buffer, string_t* column_string_write_buffer) {
    // First copy the characters from the GPU to the CPU
    SIRIUS_LOG_DEBUG("Materialize String Column to Duckdb format");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    DataWrapper column_data = column->data_wrapper;

    size_t column_num_chars = column_data.num_bytes;
    size_t column_chars_bytes = column_num_chars * sizeof(char);
    char* gpu_chars = reinterpret_cast<char*>(column_data.data);
    cudaMemcpy(column_char_write_buffer, gpu_chars, column_chars_bytes, cudaMemcpyDeviceToHost);
    
    // Now use the CPU buffer to create the strings on the GPU (using the CPU buffer to set the address)
    size_t num_strings = column_data.size;
    size_t num_blocks = (num_strings + BLOCK_THREADS - 1)/BLOCK_THREADS;
    duckdb_string_type* d_column_strings = gpuBufferManager->customCudaMalloc<duckdb_string_type>(num_strings, 0, 0);
    size_t cpu_str_bytes = num_strings * sizeof(string_t);
    create_cpu_strings<<<num_blocks, BLOCK_THREADS>>>(d_column_strings, column_char_write_buffer, gpu_chars, column_data.offset, num_strings, static_cast<size_t>(string_t::INLINE_LENGTH));

    // Copy over the strings to the CPU
    column->data_wrapper.data = reinterpret_cast<uint8_t*>(column_char_write_buffer);
    cudaMemcpy((uint8_t*) column_string_write_buffer, (uint8_t*) d_column_strings, cpu_str_bytes, cudaMemcpyDeviceToHost);
    CHECK_ERROR();
    STOP_TIMER();
}

template
void materializeWithoutNull<int>(int *a, int*& result, uint64_t *row_ids, uint64_t result_len);
template
void materializeWithoutNull<uint64_t>(uint64_t *a, uint64_t*& result, uint64_t *row_ids, uint64_t result_len);
template
void materializeWithoutNull<float>(float *a, float*& result, uint64_t *row_ids, uint64_t result_len);
template
void materializeWithoutNull<double>(double *a, double*& result, uint64_t *row_ids, uint64_t result_len);
template
void materializeWithoutNull<uint8_t>(uint8_t *a, uint8_t*& result, uint64_t *row_ids, uint64_t result_len);
template
void materializeWithoutNull<int64_t>(int64_t *a, int64_t*& result, uint64_t *row_ids, uint64_t result_len);
template
void materializeWithoutNull<__int128_t>(__int128_t *a, __int128_t*& result, uint64_t *row_ids, uint64_t result_len);

template
void materializeExpression<int16_t>(int16_t *a, int16_t*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);
template
void materializeExpression<int>(int *a, int*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);
template
void materializeExpression<uint64_t>(uint64_t *a, uint64_t*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);
template
void materializeExpression<float>(float *a, float*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);
template
void materializeExpression<double>(double *a, double*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);
template
void materializeExpression<uint8_t>(uint8_t *a, uint8_t*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);
template
void materializeExpression<int64_t>(int64_t *a, int64_t*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);
template
void materializeExpression<__int128_t>(__int128_t *a, __int128_t*& result, uint64_t *row_ids, uint64_t result_len, cudf::bitmask_type* mask, cudf::bitmask_type* &out_mask);

} // namespace duckdb