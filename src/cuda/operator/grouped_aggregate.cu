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

    T items_a[I];

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

__global__ void test(sort_keys_type* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 1000; i++) {
            for (uint64_t j = 0; j < a[i].num_key; j++) {
                printf("%ld ", a[i].keys[j]);
            }
            printf("\n");
        }
    }
}

__global__ void test2(uint64_t* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 1000; i++) {
            printf("%ld ", a[i]);
        }
        printf("\n");
    }
}


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
void groupedAggregate(T **keys, V *aggregate, uint64_t* count, uint64_t N, uint64_t num_keys) {

    printf("Launching Grouped Aggregate Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* row_keys = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys * N, 0, 0);
    sort_keys_type* materialized_temp = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    sort_keys_type* group_reduce = reinterpret_cast<sort_keys_type*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    V* aggregate_output = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
    T** keys_dev;
    cudaMalloc((void**) &keys_dev, num_keys * sizeof(T*));
    cudaMemcpy(keys_dev, keys, num_keys * sizeof(T*), cudaMemcpyHostToDevice);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    columns_to_rows<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, materialized_temp, N, num_keys);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    //perform group by
    thrust::device_vector<sort_keys_type> sorted_keys(materialized_temp, materialized_temp + N);
    thrust::device_vector<V> aggregate_columns(aggregate, aggregate + N);
    thrust::sort(sorted_keys.begin(), sorted_keys.end());

    // // test2<<<1, 1>>>(keys[0], N);
    // // test<<<1, 1>>>(raw_ptr, N);

    // uint64_t* temp = new uint64_t[N * sizeof(sort_keys_)];
    printf("Launching Grouped Aggregate Kernel\n");
    thrust::device_vector<sort_keys_type> group_by_rows(group_reduce, group_reduce + N);
    thrust::device_vector<V> agg_out(aggregate_output, aggregate_output + N);

    CHECK_ERROR();
    auto reduce_result = thrust::reduce_by_key( 
                                sorted_keys.begin(), 
                                sorted_keys.end(),
                                aggregate_columns.begin(), 
                                group_by_rows.begin(), 
                                agg_out.begin());

    CHECK_ERROR();

    printf("Launching Grouped Aggregate Kernel\n");

    printf("%ld\n", reduce_result.second - agg_out.begin());

    count[0] = reduce_result.second - agg_out.begin();

    aggregate = aggregate_output;

    CHECK_ERROR();
    cudaDeviceSynchronize();
    auto raw_sorted_keys = thrust::raw_pointer_cast(sorted_keys.data());
    rows_to_columns<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(raw_sorted_keys, keys_dev, count[0], num_keys);

    CHECK_ERROR();
    cudaDeviceSynchronize();

    printf("Launching Grouped Aggregate Kernel\n");
    
}

template
void groupedAggregate<uint64_t, double>(uint64_t** keys, double* aggregate, uint64_t* count, uint64_t N, uint64_t num_keys);

}