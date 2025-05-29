#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

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

    __host__ __device__ bool operator!=(const sort_keys_type& other) const {
      for (uint64_t i = 0; i < num_key; i++) {
        if (keys[i] != other.keys[i]) return true;
      }
      return false;
    }
};

template <typename T, int B, int I>
__global__ void 
distinct_bound(T* result, sort_keys_type *sort_keys, uint64_t N) {
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
            if (offset == 0 || (offset > 0  && sort_keys[offset] != sort_keys[offset - 1])) {
                result[offset] = 1;
                // printf("Setting %lu to 1\n", offset);
            } else {
                result[offset] = 0;
            }
        }
    }
}

template <int B, int I>
__global__ void new_modify(sort_keys_type *sort_keys, uint64_t N, uint64_t new_num_keys) {
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
            sort_keys[offset] = sort_keys_type(sort_keys[offset].keys, new_num_keys);
        }
    }
}

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
                // printf("Offset: %lu, Key[%d]: %lu\n", offset, i, row_keys[offset].keys[i]);
                col_keys[i][offset] = row_keys[offset].keys[i];
            }
        }
    }
}

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
  __device__ uint64_t operator()(const uint64_t &lhs, const uint64_t &rhs) {
      return lhs + rhs;
  }
};

template
__global__ void distinct_bound<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t* result, sort_keys_type *sort_keys, uint64_t N);

template
__global__ void columns_to_rows<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **a, uint64_t* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);

template
__global__ void rows_to_columns<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sort_keys_type *row_keys, uint64_t** col_keys, uint64_t N, uint64_t num_keys);

template 
__global__ void new_modify<BLOCK_THREADS, ITEMS_PER_THREAD>(sort_keys_type *sort_keys, uint64_t N, uint64_t new_num_keys);

template <typename T, typename V>
void groupedDistinctAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* distinct_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }
    printf("Launching Distinct Grouped Aggregate Kernel\n");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
    T* row_keys = gpuBufferManager->customCudaMalloc<T>((num_keys + num_aggregates) * N, 0, 0);
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    T** keys_row_id = new T*[num_keys + num_aggregates];
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_row_id[i] = reinterpret_cast<T*> (keys[i]);
    }

    for (uint64_t i = 0; i < num_aggregates; i++) {
        keys_row_id[num_keys + i] = reinterpret_cast<T*> (aggregate_keys[i]);
    }

    T** keys_dev;
    cudaMalloc((void**) &keys_dev, (num_keys + num_aggregates) * sizeof(T*));
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + num_aggregates) * sizeof(T*), cudaMemcpyHostToDevice);

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    columns_to_rows<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, materialized_temp, N, num_keys + num_aggregates);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    // Determine temporary device storage requirements
    CustomLess custom_op;
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_op);

    CHECK_ERROR();

    // Allocate temporary storage
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run sorting operation
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_op);

    //perform sort-based groupby
    CHECK_ERROR();
    sort_keys_type* group_by_rows = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    uint8_t** output_agg = new uint8_t*[num_aggregates];

    for (int agg = 0; agg < num_aggregates; agg++) {
        // printf("Aggregating %d\n", agg);
        if (distinct_mode[agg] == 0) { // count distinct

            uint64_t* d_aggregates_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            uint64_t* distinct_boundary = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            distinct_bound<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(distinct_boundary, materialized_temp, N);

            CHECK_ERROR();

            new_modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(materialized_temp, N, num_keys);

            CHECK_ERROR();

            // Determine temporary device storage requirements
            d_temp_storage = nullptr;
            temp_storage_bytes = 0;
            CustomSum custom_sum;
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, distinct_boundary,
                d_aggregates_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            // Allocate temporary storage
            d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

            // Run reduce-by-key
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, distinct_boundary,
                d_aggregates_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);

            CHECK_ERROR();

            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(distinct_boundary), 0);
            output_agg[agg] = reinterpret_cast<uint8_t*> (d_aggregates_out);
        }
    }

    cudaDeviceSynchronize();
    printf("Count: %lu\n", h_count[0]);
    count[0] = h_count[0];

    T** keys_dev_result;
    T** keys_result = new T*[num_keys];
    cudaMalloc((void**) &keys_dev_result, num_keys * sizeof(T*));
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_result[i] = gpuBufferManager->customCudaMalloc<T>(count[0], 0, 0);
    }
    cudaMemcpy(keys_dev_result, keys_result, num_keys * sizeof(T*), cudaMemcpyHostToDevice);

    rows_to_columns<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(h_count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(group_by_rows, keys_dev_result, h_count[0], num_keys);
    CHECK_ERROR();

    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys[i]), 0);
        keys[i] = reinterpret_cast<uint8_t*> (keys_result[i]);
    }

    for (int agg = 0; agg < num_aggregates; agg++) {
        if (distinct_mode[agg] == 0) {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys[agg]), 0);
            aggregate_keys[agg] = output_agg[agg];
        }
    }

    cudaFree(keys_dev_result);
    cudaFree(keys_dev);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(materialized_temp), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_by_rows), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_runs_out), 0);

    STOP_TIMER();
}

template
void groupedDistinctAggregate<uint64_t, uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* distinct_mode);

} // namespace duckdb