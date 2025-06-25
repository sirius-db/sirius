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

struct sort_keys_type {
  uint64_t* keys;
  uint64_t num_key;

  __host__ __device__ sort_keys_type() {}
  __host__ __device__ sort_keys_type(uint64_t* keys, uint64_t num_key) : keys(keys), num_key(num_key) {}

  __host__ __device__ bool operator<(const sort_keys_type& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) {
            return keys[i] < other.keys[i];
        }
      }
      return true;
    }

    __host__ __device__ bool operator==(const sort_keys_type& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) return false;
      }
      return true;
    }
};

struct CustomLess
{
  __device__ bool operator()(const sort_keys_type &lhs, const sort_keys_type &rhs) {
      for (uint64_t i = 0; i < lhs.num_key; i++) {
            if (lhs.keys[i] != rhs.keys[i]) {
                return lhs.keys[i] < rhs.keys[i];
            }
      }
      return true;
  }
};

struct CustomSum
{
    template <typename T>
    __device__ T operator()(const T &a, const T &b) {
        return a + b;
    }
};

struct CustomMin
{
    template <typename T>
    __device__ T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct CustomMax
{
    template <typename T>
    __device__ T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

template <typename T, int B, int I>
__global__ void columns_to_rows(T **a, T* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys) {

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
            for (uint64_t i = 0; i < num_keys; i++) {
                result[offset * num_keys + i] = a[i][offset];
            }
            temp[offset] = sort_keys_type(&result[offset * num_keys], num_keys);
        }
    }
}

template <typename T, int B, int I>
__global__ void rows_to_columns(sort_keys_type *row_keys, T** col_keys, uint64_t N, uint64_t num_keys) {

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
            for (int i = 0; i < num_keys; i++) {
                col_keys[i][offset] = row_keys[offset].keys[i];
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void gather_and_modify(const T *a, T* result, sort_keys_type *sort_keys, uint64_t N, uint64_t num_keys) {

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
            sort_keys[offset] = sort_keys_type(sort_keys[offset].keys, num_keys - 1);
        }
    }
}

template <int B, int I>
__global__ void modify(sort_keys_type *sort_keys, uint64_t N, uint64_t num_keys) {

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
            sort_keys[offset] = sort_keys_type(sort_keys[offset].keys, num_keys - 1);
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
__global__ void gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);
template
__global__ void gather_and_modify<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);

template
__global__ void columns_to_rows<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **a, uint64_t* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);

template
__global__ void rows_to_columns<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sort_keys_type *row_keys, uint64_t** col_keys, uint64_t N, uint64_t num_keys);

template <typename T, typename V>
void groupedAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }

    SIRIUS_LOG_DEBUG("Launching Grouped Aggregate Kernel");

    SETUP_TIMING();
    cudaEventRecord(start, 0);
    
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
    T* row_keys = gpuBufferManager->customCudaMalloc<T>((num_keys + 1) * N, 0, 0);
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    // T* keys_row_id[num_keys + 1];
    T** keys_row_id = gpuBufferManager->customCudaHostAlloc<T*>(num_keys + 1);
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_row_id[i] = reinterpret_cast<T*> (keys[i]);
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t* row_sequence = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    sequence<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(row_sequence, N);
    keys_row_id[num_keys] = row_sequence;

    T** keys_dev = gpuBufferManager->customCudaMalloc<T*>(num_keys + 1, 0, 0);
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + 1) * sizeof(T*), cudaMemcpyHostToDevice);

    columns_to_rows<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, materialized_temp, N, num_keys + 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    //perform sort-based groupby
    // Determine temporary device storage requirements
    CustomLess custom_less;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
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


    //gather the aggregates based on the row_sequence
    SIRIUS_LOG_DEBUG("Gathering Aggregates");
    V** aggregate_keys_temp = gpuBufferManager->customCudaHostAlloc<V*>(num_aggregates);
    uint64_t** aggregate_star_temp = gpuBufferManager->customCudaHostAlloc<uint64_t*>(num_aggregates);
    sort_keys_type* group_by_rows = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    uint8_t** output_agg = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_aggregates);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

    for (int agg = 0; agg < num_aggregates; agg++) {
        // SIRIUS_LOG_DEBUG("Aggregating {}", agg);
        if (agg_mode[agg] == 4 || agg_mode[agg] == 5) { //count_star or count(null) or sum(null)
            aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            if (agg_mode[agg] == 4) 
                fill_n<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], 1, N);
            else if (agg_mode[agg] == 5)
                cudaMemset(aggregate_star_temp[agg], 0, N * sizeof(uint64_t));

            modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(materialized_temp, N, num_keys + 1);

            //perform reduce_by_key
            uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));

            SIRIUS_LOG_DEBUG("Reduce by key count_star");
            // Determine temporary device storage requirements
            d_temp_storage = nullptr;
            temp_storage_bytes = 0;
            CustomSum custom_sum;
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

            SIRIUS_LOG_DEBUG("Count: {}", count[0]);

            CHECK_ERROR();
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_star_temp[agg]), 0);
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
            output_agg[agg] = reinterpret_cast<uint8_t*> (agg_star_out);
        } else {
            aggregate_keys_temp[agg] = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            V* temp = reinterpret_cast<V*> (aggregate_keys[agg]);
            gather_and_modify<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(temp, aggregate_keys_temp[agg], materialized_temp, N, num_keys + 1);

            V* agg_out = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            cudaMemset(agg_out, 0, N * sizeof(V));

            CHECK_ERROR();
            if (agg_mode[agg] == 0) {
                SIRIUS_LOG_DEBUG("Reduce by key sum");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSum custom_sum;
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
            } else if (agg_mode[agg] == 1) {
                //Currently typename V has to be a double
                SIRIUS_LOG_DEBUG("Reduce by key avg");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSum custom_sum;
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
                CustomMax custom_max;
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
                CustomMin custom_min;
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

    T** keys_dev_result = gpuBufferManager->customCudaMalloc<T*>(num_keys, 0, 0);
    T** keys_result = gpuBufferManager->customCudaHostAlloc<T*>(num_keys);
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_result[i] = gpuBufferManager->customCudaMalloc<T>(count[0], 0, 0);
    }
    cudaMemcpy(keys_dev_result, keys_result, num_keys * sizeof(T*), cudaMemcpyHostToDevice);

    rows_to_columns<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(group_by_rows, keys_dev_result, count[0], num_keys);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    SIRIUS_LOG_DEBUG("Grouped Aggregate Result Count: {}", count[0]);

    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys[i]), 0);
        keys[i] = reinterpret_cast<uint8_t*> (keys_result[i]);
    }

    for (int agg = 0; agg < num_aggregates; agg++) {
        if (agg_mode[agg] >= 0 && agg_mode[agg] <= 3) {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys[agg]), 0);
            aggregate_keys[agg] = output_agg[agg];
        } else {
            aggregate_keys[agg] = output_agg[agg];
        }
    }

    //call customCudaFree for all the allocated memory
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev_result), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(materialized_temp), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_sequence), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_by_rows), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_runs_out), 0); 
    STOP_TIMER();
}


template <typename T>
void groupedWithoutAggregate(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Grouped Without Aggregate Kernel");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
    T* row_keys = gpuBufferManager->customCudaMalloc<T>(num_keys * N, 0, 0);
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    T** keys_row_id = gpuBufferManager->customCudaHostAlloc<T*>(num_keys);
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_row_id[i] = reinterpret_cast<T*> (keys[i]);
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    T** keys_dev = gpuBufferManager->customCudaMalloc<T*>(num_keys, 0, 0);
    cudaMemcpy(keys_dev, keys_row_id, num_keys * sizeof(T*), cudaMemcpyHostToDevice);

    columns_to_rows<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, materialized_temp, N, num_keys);
    CHECK_ERROR();

    //perform sort-based groupby
    // Determine temporary device storage requirements
    CustomLess custom_less;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
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

    //gather the aggregates based on the row_sequence
    sort_keys_type* group_by_rows = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);

    uint64_t* aggregate_star_temp = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    cudaMemset(aggregate_star_temp, 0, N * sizeof(uint64_t));

    //perform reduce_by_key
    uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));

    // Determine temporary device storage requirements
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    CustomSum custom_sum;
    cub::DeviceReduce::ReduceByKey(
        d_temp_storage, temp_storage_bytes,
        materialized_temp, group_by_rows, aggregate_star_temp,
        agg_star_out, d_num_runs_out, custom_sum, N);

    CHECK_ERROR();

    // Allocate temporary storage
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run reduce-by-key
    cub::DeviceReduce::ReduceByKey(
        d_temp_storage, temp_storage_bytes,
        materialized_temp, group_by_rows, aggregate_star_temp,
        agg_star_out, d_num_runs_out, custom_sum, N);

    CHECK_ERROR();

    cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
    count[0] = h_count[0];

    T** keys_dev_result = gpuBufferManager->customCudaMalloc<T*>(num_keys, 0, 0);
    T** keys_result = gpuBufferManager->customCudaHostAlloc<T*>(num_keys);
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_result[i] = gpuBufferManager->customCudaMalloc<T>(count[0], 0, 0);
    }
    cudaMemcpy(keys_dev_result, keys_result, num_keys * sizeof(T*), cudaMemcpyHostToDevice);

    rows_to_columns<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(group_by_rows, keys_dev_result, count[0], num_keys);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    SIRIUS_LOG_DEBUG("Grouped Without Aggregate Result Count: {}", count[0]);

    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys[i]), 0);
        keys[i] = reinterpret_cast<uint8_t*> (keys_result[i]);
    }

    //call customCudaFree for all the allocated memory
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev_result), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(materialized_temp), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_by_rows), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_runs_out), 0); 
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_star_temp), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(agg_star_out), 0);
    STOP_TIMER();
}

template<typename T>
void combineColumns(T* a, T* b, T*& c, uint64_t N_a, uint64_t N_b) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Combine Columns Kernel");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    c = gpuBufferManager->customCudaMalloc<T>(N_a + N_b, 0, 0);
    cudaMemcpy(c, a, N_a * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(c + N_a, b, N_b * sizeof(T), cudaMemcpyDeviceToDevice);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(a), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(b), 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

__global__ void add_offset(uint64_t* a, uint64_t* b, uint64_t offset, uint64_t N) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = b[idx] + offset;
    }
}

void combineStrings(uint8_t* a, uint8_t* b, uint8_t*& c, 
        uint64_t* offset_a, uint64_t* offset_b, uint64_t*& offset_c, 
        uint64_t num_bytes_a, uint64_t num_bytes_b, uint64_t N_a, uint64_t N_b) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    c = gpuBufferManager->customCudaMalloc<uint8_t>(num_bytes_a + num_bytes_b, 0, 0);
    offset_c = gpuBufferManager->customCudaMalloc<uint64_t>(N_a + N_b + 1, 0, 0);
    cudaMemcpy(c, a, num_bytes_a * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(c + num_bytes_a, b, num_bytes_b * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    cudaMemcpy(offset_c, offset_a, N_a * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    add_offset<<<((N_b + 1) + BLOCK_THREADS - 1)/(BLOCK_THREADS), BLOCK_THREADS>>>(offset_c + N_a, offset_b, num_bytes_a, N_b + 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template
void groupedAggregate<uint64_t, uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedAggregate<uint64_t, double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedWithoutAggregate<uint64_t>(uint8_t **keys, uint64_t* count, uint64_t N, uint64_t num_keys);

template
void combineColumns<int32_t>(int32_t* a, int32_t* b, int32_t*& c, uint64_t N_a, uint64_t N_b);

template
void combineColumns<uint64_t>(uint64_t* a, uint64_t* b, uint64_t*& c, uint64_t N_a, uint64_t N_b);

template
void combineColumns<double>(double* a, double* b, double*& c, uint64_t N_a, uint64_t N_b);

}