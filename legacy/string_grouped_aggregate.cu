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
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

struct sort_keys_type_string {
  uint64_t* keys;
  uint64_t num_key;

  __host__ __device__ sort_keys_type_string() {}
  __host__ __device__ sort_keys_type_string(uint64_t* _keys, uint64_t _num_key) : keys(_keys), num_key(_num_key) {}

  __host__ __device__ bool operator<(const sort_keys_type_string& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) {
            return keys[i] < other.keys[i];
        }
      }
      return true;
    }

    __host__ __device__ bool operator==(const sort_keys_type_string& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) return false;
      }
      return true;
    }

    __host__ __device__ bool operator!=(const sort_keys_type_string& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) return true;
      }
      return false;
    }
};

struct CustomLessString
{
  __device__ bool operator()(const sort_keys_type_string &lhs, const sort_keys_type_string &rhs) {
      for (uint64_t i = 0; i < lhs.num_key; i++) {
            if (lhs.keys[i] != rhs.keys[i]) {
                return lhs.keys[i] < rhs.keys[i];
            }
      }
      return true;
  }
};

struct CustomSumString
{
    template <typename T>
    __host__ __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

struct CustomMinString
{
    template <typename T>
    __host__ __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct CustomMaxString
{
    template <typename T>
    __host__ __device__ __forceinline__
     T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

template <typename T, int B, int I>
__global__ void fill_offset(uint64_t* offset, uint64_t N) {
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
            offset[tile_offset + threadIdx.x + ITEM * B] = sizeof(T) * (tile_offset + threadIdx.x + ITEM * B);
        }
    }
}

template <int B, int I>
__global__ void columns_to_rows_string(uint8_t **a, uint8_t* result, uint64_t **input_offset, uint64_t* key_length,
            sort_keys_type_string* temp, uint64_t N, uint64_t num_keys) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    uint64_t total_length = 0;
    for (uint64_t key = 0; key < (num_keys - 1); key ++) {
        total_length += key_length[key];
    }
    //add the row ids into the total length
    total_length += sizeof(uint64_t);

    uint64_t meta_num_keys = (total_length + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t total_length_bytes = meta_num_keys * sizeof(uint64_t);

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            uint64_t output_start_idx = offset * total_length_bytes;
            memset(result + output_start_idx, 0, total_length_bytes * sizeof(uint8_t));
            //copy the keys without the row ids
            for (uint64_t key = 0; key < (num_keys - 1); key ++) {
                uint64_t input_length = input_offset[key][offset + 1] - input_offset[key][offset];
                uint64_t input_start_idx = input_offset[key][offset];
                memcpy(result + output_start_idx, a[key] + input_start_idx, input_length * sizeof(uint8_t));
                output_start_idx += key_length[key];
            }
            //copy the row ids
            memcpy(result + (offset * total_length_bytes) + ((meta_num_keys - 1) * sizeof(uint64_t)), a[num_keys - 1] + (offset * sizeof(uint64_t)), sizeof(uint64_t));
            temp[offset] = sort_keys_type_string(reinterpret_cast<uint64_t*>(&result[offset * total_length_bytes]), meta_num_keys);
        }
    }
}

template <int B, int I>
__global__ void compact_string_offset(uint64_t* group_idx, uint64_t** group_byte_offset, uint64_t** result_offset, uint64_t N, uint64_t num_keys) {
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
            if (offset == N - 1) {
                uint64_t out_idx = group_idx[offset];
                for (uint64_t key = 0; key < num_keys; key ++) {
                    result_offset[key][out_idx] = group_byte_offset[key][offset];
                }
            } else if ((offset < (N - 1)) && (group_idx[offset] != group_idx[offset + 1])) {
                uint64_t out_idx = group_idx[offset];
                for (uint64_t key = 0; key < num_keys; key ++) {
                    cudaAssert(group_byte_offset[key][offset] != group_byte_offset[key][offset + 1]);
                    result_offset[key][out_idx] = group_byte_offset[key][offset];
                }
            }
        }
    }
}

template <int B, int I>
__global__ void rows_to_columns_string(uint64_t* group_idx, sort_keys_type_string *row_keys, uint8_t** col_keys, uint64_t **group_byte_offset, uint64_t* key_length,
    uint64_t N, uint64_t num_keys) {

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
            //we should write out the offset
            if (group_idx[offset] != group_idx[offset + 1]) {
                uint64_t out_idx = group_idx[offset];
                uint64_t key_length_bytes = 0;
                for (uint64_t key = 0; key < num_keys; key ++) {
                    cudaAssert(group_byte_offset[key][offset] != group_byte_offset[key][offset + 1]);
                    uint64_t out_offset = group_byte_offset[key][offset];
                    uint64_t actual_key_length = group_byte_offset[key][offset + 1] - group_byte_offset[key][offset];
                    uint8_t* ptr = reinterpret_cast<uint8_t*>(row_keys[out_idx].keys);
                    memcpy(col_keys[key] + out_offset, ptr + key_length_bytes, actual_key_length * sizeof(uint8_t));
                    key_length_bytes += key_length[key];
                }
            }
        }
    }
}

template <int B, int I>
__global__ void get_len(uint64_t* offset, uint64_t* len, uint64_t N) {
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
            uint64_t idx = tile_offset + threadIdx.x + ITEM * B;
            len[idx] = offset[idx + 1] - offset[idx];
        }
    }
}


template <int B, int I>
__global__ void distinct_string(uint64_t* distinct_mark, uint64_t* distinct_len, uint64_t* len, sort_keys_type_string *sort_keys, uint64_t N) {
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
            if (offset == 0 || (offset > 0  && (sort_keys[offset] != sort_keys[offset - 1]))) {
                distinct_mark[offset] = 1;
                distinct_len[offset] = len[offset];
            } else {
                distinct_mark[offset] = 0;
                distinct_len[offset] = 0;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void gather_and_modify(const T *a, T* result, sort_keys_type_string *sort_keys, uint64_t N, uint64_t meta_num_keys) {
    cudaAssert(meta_num_keys > 1);
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
            uint64_t items_ids = sort_keys[offset].keys[meta_num_keys - 1];
            result[offset] = a[items_ids];
            sort_keys[offset] = sort_keys_type_string(sort_keys[offset].keys, meta_num_keys - 1);
        }
    }
}

template <typename T, int B, int I>
__global__ void gather(const T *a, T* result, sort_keys_type_string *sort_keys, uint64_t N, uint64_t num_keys) {

    cudaAssert(num_keys > 1);
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
            uint64_t items_ids = sort_keys[offset].keys[num_keys - 1];
            result[offset] = a[items_ids];
        }
    }
}

template <int B, int I>
__global__ void modify(sort_keys_type_string *sort_keys, uint64_t N, uint64_t meta_num_keys) {

    cudaAssert(meta_num_keys > 1);
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
            sort_keys[offset] = sort_keys_type_string(sort_keys[offset].keys, meta_num_keys - 1);
        }
    }
}

template <int B, int I>
__global__ void sequence(uint64_t* result, uint64_t N) {

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
            result[tile_offset + threadIdx.x + ITEM * B] = tile_offset + threadIdx.x + ITEM * B;
        }
    }
}

template <typename T, int B, int I>
__global__ void divide(T* a, uint64_t* b, T* result, uint64_t N) {

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
            int offset = tile_offset + threadIdx.x + ITEM * B;
            result[offset] = a[offset] / b[offset];
        }
    }
}

template <typename T, int B, int I>
__global__ void fill_n(T* a, T b, uint64_t N) {
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
            a[tile_offset + threadIdx.x + ITEM * B] = b;
        }
    }
}

template
__global__ void gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, sort_keys_type_string* sort_keys, uint64_t N, uint64_t meta_num_keys);
template
__global__ void gather_and_modify<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, sort_keys_type_string* sort_keys, uint64_t N, uint64_t meta_num_keys);
template
__global__ void gather<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, sort_keys_type_string* sort_keys, uint64_t N, uint64_t num_keys);

template <typename V>
void groupedStringAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }

    SIRIUS_LOG_DEBUG("Launching String Grouped Aggregate Kernel");

    SETUP_TIMING();
    START_TIMER();
    
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;

    //cubmax
    // Get the maximum key length for each key
    uint64_t* key_length = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys, 0, 0); // store the maximum length of each key
    uint64_t** len = gpuBufferManager->customCudaHostAlloc<uint64_t*>(num_keys);
    uint64_t* original_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(num_keys);
    for (int key = 0; key < num_keys; key++) {
        len[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);

        if (offset[key] == nullptr) {
            offset[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
            fill_offset<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD), BLOCK_THREADS>>>(offset[key], N+1);
            CHECK_ERROR();
        }
        cudaMemcpy(original_bytes + key, offset[key] + N, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        get_len<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD), BLOCK_THREADS>>>(offset[key], len[key], N);
        CHECK_ERROR();
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;

        if (offset[key] == nullptr) {
            cudaMemcpy(key_length + key, len[key], sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        } else {
            cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, len[key], key_length + key, N);

            // Allocate temporary storage
            d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

            // Run min-reduction
            cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, len[key], key_length + key, N);
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
        }
    }

    uint64_t* h_key_length = gpuBufferManager->customCudaHostAlloc<uint64_t>(num_keys);
    cudaMemcpy(h_key_length, key_length, num_keys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();

    uint64_t row_id_size = sizeof(uint64_t);
    uint64_t total_length = 0;
    for (uint64_t key = 0; key < num_keys; key ++) {
        total_length += h_key_length[key];
    }
    //add the row ids into the total length
    total_length += row_id_size;
    uint64_t meta_num_keys = (total_length + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t total_length_bytes = meta_num_keys * sizeof(uint64_t);

    //allocate temp memory and copying keys
    uint8_t* row_keys = gpuBufferManager->customCudaMalloc<uint8_t>((total_length_bytes) * N, 0, 0);
    sort_keys_type_string* materialized_temp = reinterpret_cast<sort_keys_type_string*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));

    uint8_t** keys_row_id = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys + 1);
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_row_id[i] = keys[i];
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t* row_sequence = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    sequence<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(row_sequence, N);
    keys_row_id[num_keys] = reinterpret_cast<uint8_t*> (row_sequence);

    uint8_t** keys_dev = gpuBufferManager->customCudaMalloc<uint8_t*>(num_keys + 1, 0, 0);
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + 1) * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    uint64_t** offset_dev = gpuBufferManager->customCudaMalloc<uint64_t*>(num_keys, 0, 0);
    cudaMemcpy(offset_dev, offset, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    columns_to_rows_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, offset_dev, key_length,
            materialized_temp, N, num_keys + 1);
    CHECK_ERROR();

    //perform sort-based groupby
    // Determine temporary device storage requirements
    CustomLessString custom_less;
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_less);

    CHECK_ERROR();

    // Allocate temporary storage
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run sorting operation
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_less);

    CHECK_ERROR();

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

    SIRIUS_LOG_DEBUG("Gathering offset");
    uint64_t** group_byte_offset = gpuBufferManager->customCudaHostAlloc<uint64_t*>(num_keys);
    uint64_t* distinct_bound = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    uint64_t* group_idx = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
    uint64_t* d_num_bytes = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys, 0, 0);

    for (uint64_t key = 0; key < num_keys; key++) {
        uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
        group_byte_offset[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
        cudaMemset(group_byte_offset[key] + N, 0, sizeof(uint64_t));

        gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(len[key], temp, materialized_temp, N, meta_num_keys);
        CHECK_ERROR();
        distinct_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(distinct_bound, temp, temp, materialized_temp, N);
        CHECK_ERROR();
        //cub scan
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp, group_byte_offset[key], N + 1);

        // Allocate temporary storage for exclusive prefix sum
        d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

        // Run exclusive prefix sum
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp, group_byte_offset[key], N + 1);
        CHECK_ERROR();
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

        cudaMemcpy(d_num_bytes + key, group_byte_offset[key] + N, sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(temp), 0);
        CHECK_ERROR();
    }

    //copy num_bytes over
    cudaMemcpy(num_bytes, d_num_bytes, num_keys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint64_t** group_byte_offset_dev = gpuBufferManager->customCudaMalloc<uint64_t*>(num_keys, 0, 0);;
    cudaMemcpy(group_byte_offset_dev, group_byte_offset, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    //cub scan
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, distinct_bound, group_idx, N + 1);

    // Allocate temporary storage for exclusive prefix sum
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, distinct_bound, group_idx, N + 1);
    CHECK_ERROR();
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

    //gather the aggregates based on the row_sequence
    SIRIUS_LOG_DEBUG("Gathering Aggregates");
    V** aggregate_keys_temp = gpuBufferManager->customCudaHostAlloc<V*>(num_aggregates);
    uint64_t** aggregate_star_temp = gpuBufferManager->customCudaHostAlloc<uint64_t*>(num_aggregates);
    sort_keys_type_string* group_by_rows = reinterpret_cast<sort_keys_type_string*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    uint8_t** output_agg = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_aggregates);
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);

    for (int agg = 0; agg < num_aggregates; agg++) {
        SIRIUS_LOG_DEBUG("Aggregating {}", agg);
        cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
        if (agg_mode[agg] == 4 || agg_mode[agg] == 5) { //count_star or count(null) or sum(null)
            aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            if (agg_mode[agg] == 4) {
                fill_n<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], 1, N);
            } else if (agg_mode[agg] == 5) {
                cudaMemset(aggregate_star_temp[agg], 0, N * sizeof(double));
            }

            modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(materialized_temp, N, meta_num_keys);
            CHECK_ERROR();

            //perform reduce_by_key
            uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));

            SIRIUS_LOG_DEBUG("Reduce by key count_star");
            // Determine temporary device storage requirements
            d_temp_storage = nullptr;
            temp_storage_bytes = 0;
            CustomSumString custom_sum;
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, aggregate_star_temp[agg],
                agg_star_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            // Allocate temporary storage
            d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

            // Run reduce-by-key
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, aggregate_star_temp[agg],
                agg_star_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_star_temp[agg]), 0);
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
            count[0] = h_count[0];

            SIRIUS_LOG_DEBUG("Count: {}", count[0]);

            CHECK_ERROR();
            output_agg[agg] = reinterpret_cast<uint8_t*> (agg_star_out);
        } else {
            aggregate_keys_temp[agg] = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            V* temp = reinterpret_cast<V*> (aggregate_keys[agg]);
            gather_and_modify<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(temp, aggregate_keys_temp[agg], materialized_temp, N, meta_num_keys);
            CHECK_ERROR();

            V* agg_out = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            cudaMemset(agg_out, 0, N * sizeof(V));

            CHECK_ERROR();
            if (agg_mode[agg] == 0) {
                SIRIUS_LOG_DEBUG("Reduce by key sum");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSumString custom_sum;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (agg_out);
                SIRIUS_LOG_DEBUG("Count: {}", count[0]);
            } else if (agg_mode[agg] == 1) {
                //Currently typename V has to be a double
                SIRIUS_LOG_DEBUG("Reduce by key avg");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSumString custom_sum;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

                aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
                fill_n<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], 1, N);

                uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
                cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));
                cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));

                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_star_temp[agg],
                    agg_star_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_star_temp[agg],
                    agg_star_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                V* output = gpuBufferManager->customCudaMalloc<V>(count[0], 0, 0);
                divide<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(agg_out, agg_star_out, output, count[0]);

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_star_temp[agg]), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(agg_star_out), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(agg_out), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (output);
            } else if (agg_mode[agg] == 2) {
                SIRIUS_LOG_DEBUG("Reduce by key max");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMaxString custom_max;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_max, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_max, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (agg_out);
            } else if (agg_mode[agg] == 3) {
                SIRIUS_LOG_DEBUG("Reduce by key min");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMinString custom_min;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_min, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_min, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (agg_out);
            }
        }
    }

    uint64_t** offset_dev_result = gpuBufferManager->customCudaMalloc<uint64_t*>(num_keys, 0, 0);
    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset[i]), 0);
        offset[i] = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
    }
    cudaMemcpy(offset_dev_result, offset, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    compact_string_offset<BLOCK_THREADS, ITEMS_PER_THREAD><<<((N + 1) + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            group_idx, group_byte_offset_dev, offset_dev_result, N + 1, num_keys);

    CHECK_ERROR();

    uint8_t** keys_dev_result = gpuBufferManager->customCudaMalloc<uint8_t*>(num_keys, 0, 0);
    for (uint64_t i = 0; i < num_keys; i++) {
        uint64_t* temp_num_bytes = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        cudaMemcpy(temp_num_bytes, offset[i] + count[0], sizeof(uint64_t), cudaMemcpyDeviceToHost);
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys[i]), 0);
        keys[i] = gpuBufferManager->customCudaMalloc<uint8_t>(temp_num_bytes[0], 0, 0);
    }
    cudaMemcpy(keys_dev_result, keys, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    rows_to_columns_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            group_idx, group_by_rows, keys_dev_result, group_byte_offset_dev, key_length, N, num_keys);

    CHECK_ERROR();

    for (int agg = 0; agg < num_aggregates; agg++) {
        if (agg_mode[agg] >= 0 && agg_mode[agg] <= 3) {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys[agg]), 0);
            aggregate_keys[agg] = output_agg[agg];
        } else {
            aggregate_keys[agg] = output_agg[agg];
        }
    }

    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(len[i]), 0);
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_byte_offset[i]), 0);
    }

    //free row_keys, row_sequence, materialized_temp
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_byte_offset_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset_dev_result), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev_result), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_sequence), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(materialized_temp), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_by_rows), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_runs_out), 0); 
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(key_length), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(distinct_bound), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_idx), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_bytes), 0);
    cudaDeviceSynchronize();
    SIRIUS_LOG_DEBUG("String Grouped Aggregate Count: {}\n", count[0]);

    STOP_TIMER();
}

// __global__ void add_offset(uint64_t* a, uint64_t* b, uint64_t offset, uint64_t N) {
//     uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < N) {
//         a[idx] = b[idx] + offset;
//     }
// }

// void combineStrings(uint8_t* a, uint8_t* b, uint8_t*& c, 
//         uint64_t* offset_a, uint64_t* offset_b, uint64_t*& offset_c, 
//         uint64_t num_bytes_a, uint64_t num_bytes_b, uint64_t N_a, uint64_t N_b) {
//     CHECK_ERROR();
//     if (N_a == 0 || N_b == 0) {
//         SIRIUS_LOG_DEBUG("Input size is 0");
//         return;
//     }
//     GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
//     c = gpuBufferManager->customCudaMalloc<uint8_t>(num_bytes_a + num_bytes_b, 0, 0);
//     offset_c = gpuBufferManager->customCudaMalloc<uint64_t>(N_a + N_b + 1, 0, 0);
//     cudaMemcpy(c, a, num_bytes_a * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
//     cudaMemcpy(c + num_bytes_a, b, num_bytes_b * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

//     cudaMemcpy(offset_c, offset_a, N_a * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
//     add_offset<<<((N_b + 1) + BLOCK_THREADS - 1)/(BLOCK_THREADS), BLOCK_THREADS>>>(offset_c + N_a, offset_b, num_bytes_a, N_b + 1);
//     CHECK_ERROR();
//     cudaDeviceSynchronize();
// }

template
void groupedStringAggregate<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedStringAggregate<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

}