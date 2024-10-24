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
                // col_keys[i][offset] = row_keys[offset * num_keys + i];
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

__global__ void test(sort_keys_type* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 15; i++) {
            for (uint64_t j = 0; j < a[i].num_key + 1; j++) {
                printf("%ld ", a[i].keys[j]);
            }
        }
        printf("\n");
    }
}

__global__ void test4(double* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 15; i++) {
            printf("%.2f ", a[i]);
        }
        printf("\n");
    }
}

__global__ void test3(uint64_t* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 15; i++) {
            printf("%ld ", a[i]);
        }
        printf("\n");
    }
}

__global__ void test2(double* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 15; i++) {
            printf("%.2f ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void gather_and_modify<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, int* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);
template
__global__ void gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);
template
__global__ void gather_and_modify<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, float* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);
template
__global__ void gather_and_modify<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);
template
__global__ void gather_and_modify<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, uint8_t* result, sort_keys_type* sort_keys, uint64_t N, uint64_t num_keys);

// template
// __global__ void columns_to_rows<int, BLOCK_THREADS, ITEMS_PER_THREAD>(int **a, int* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);
template
__global__ void columns_to_rows<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **a, uint64_t* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);
// template
// __global__ void columns_to_rows<float, BLOCK_THREADS, ITEMS_PER_THREAD>(float **a, float* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);
// template
// __global__ void columns_to_rows<double, BLOCK_THREADS, ITEMS_PER_THREAD>(double **a, double* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);
// template
// __global__ void columns_to_rows<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint8_t **a, uint8_t* result, sort_keys_type* temp, uint64_t N, uint64_t num_keys);

// template
// __global__ void rows_to_columns<int, BLOCK_THREADS, ITEMS_PER_THREAD>(int *row_keys, int** col_keys, uint64_t N, uint64_t num_keys);
template
__global__ void rows_to_columns<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(sort_keys_type *row_keys, uint64_t** col_keys, uint64_t N, uint64_t num_keys);
// template
// __global__ void rows_to_columns<float, BLOCK_THREADS, ITEMS_PER_THREAD>(float *row_keys, float** col_keys, uint64_t N, uint64_t num_keys);
// template
// __global__ void rows_to_columns<double, BLOCK_THREADS, ITEMS_PER_THREAD>(double *row_keys, double** col_keys, uint64_t N, uint64_t num_keys);
// template
// __global__ void rows_to_columns<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint8_t *row_keys, uint8_t** col_keys, uint64_t N, uint64_t num_keys);

template <typename T, typename V>
void groupedAggregate(T **keys, V *&aggregate, uint64_t* count, uint64_t N, uint64_t num_keys) {

    printf("Launching Grouped Aggregate Kernel\n");
    // test3<<<1, 1>>>(keys[0], N);
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //allocate temp memory and copying keys
    T* row_keys = gpuBufferManager->customCudaMalloc<T>((num_keys + 1) * N, 0, 0);
    T* keys_temp = gpuBufferManager->customCudaMalloc<T>((num_keys + 1) * N, 0, 0);
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    T* keys_row_id[num_keys + 1];
    for (uint64_t i = 0; i < num_keys; i++) {
        cudaMemcpy(keys_temp + i * N, keys[i], N * sizeof(T), cudaMemcpyDeviceToDevice);
        keys_row_id[i] = keys_temp + i * N;
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t* row_sequence = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    sequence<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(row_sequence, N);
    keys_row_id[num_keys] = row_sequence;

    T** keys_dev;
    cudaMalloc((void**) &keys_dev, (num_keys + 1) * sizeof(T*));
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + 1) * sizeof(T*), cudaMemcpyHostToDevice);

    columns_to_rows<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, materialized_temp, N, num_keys + 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    //perform sort-based groupby
    thrust::device_vector<sort_keys_type> sorted_keys(materialized_temp, materialized_temp + N);
    // thrust::device_vector<uint64_t> value_columns(row_sequence, row_sequence + N);
    thrust::sort(thrust::device, sorted_keys.begin(), sorted_keys.end());
    CHECK_ERROR();

    //gather the aggregates based on the row_sequence
    printf("Gathering Aggregates\n");
    sort_keys_type* gather_idx = thrust::raw_pointer_cast(sorted_keys.data());
    V* aggregate_temp = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
    gather_and_modify<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate, aggregate_temp, gather_idx, N, num_keys + 1);
    thrust::device_vector<V> aggregate_columns(aggregate_temp, aggregate_temp + N);
    // thrust::device_vector<V> aggregate_columns(aggregate, aggregate + N);
    // thrust::device_vector<double> aggregate_columns(N);
    // thrust::fill_n(aggregate_columns.begin(), aggregate_columns.size(), 1.0);

    // sort_keys_type* raw_sorted_keys_temp = thrust::raw_pointer_cast(sorted_keys.data());
    // test4<<<1, 1>>>(aggregate_temp, N);
    // test3<<<1, 1>>>(row_sequence, N);
    // test<<<1, 1>>>(raw_sorted_keys_temp, N);

    //perform reduce_by_key
    thrust::device_vector<sort_keys_type> group_by_rows(N);
    thrust::device_vector<V> agg_out(N);
    // thrust::equal_to<sort_keys_type> binary_pred; thrust::plus<V> binary_op;
    V* raw_agg_out = thrust::raw_pointer_cast(agg_out.data());
    cudaMemset(raw_agg_out, 0, N * sizeof(V));
    // sort_keys_type* raw_sorted_keys2 = thrust::raw_pointer_cast(sorted_keys.data());
    // test<<<1, 1>>>(raw_agg_out, count[0]);

    CHECK_ERROR();
    auto reduce_result = thrust::reduce_by_key(thrust::device, 
                                sorted_keys.begin(), 
                                sorted_keys.end(),
                                aggregate_columns.begin(), 
                                group_by_rows.begin(), 
                                agg_out.begin());

    count[0] = reduce_result.second - agg_out.begin();

    CHECK_ERROR();
    sort_keys_type* raw_sorted_keys = thrust::raw_pointer_cast(group_by_rows.data());
    rows_to_columns<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(raw_sorted_keys, keys_dev, count[0], num_keys);
    cudaMemcpy(aggregate_temp, raw_agg_out, count[0] * sizeof(V), cudaMemcpyDeviceToDevice);
    aggregate = aggregate_temp;

    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template
void groupedAggregate<uint64_t, double>(uint64_t** keys, double *&aggregate, uint64_t* count, uint64_t N, uint64_t num_keys);

// template
// void groupedAggregate<uint64_t, uint64_t>(uint64_t** keys, uint64_t *&aggregate, uint64_t* count, uint64_t N, uint64_t num_keys);

}