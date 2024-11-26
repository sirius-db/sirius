#include "cuda_helper.cuh"
#include "gpu_columns.hpp"

namespace duckdb {

template <typename T, int B, int I>
__global__ void materialize_expression(const T *a, T* result, uint64_t *row_ids, uint64_t N) {

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
            int items_ids = row_ids[tile_offset + threadIdx.x + ITEM * B];
            // if (N == 3793296 && (items_ids < 0 || items_ids >= 3793296)) printf("items_ids: %d\n", items_ids);
            result[tile_offset + threadIdx.x + ITEM * B] = a[items_ids];
            // cudaAssert(a[items_ids] == 19940101);
            // printf("Result: %ld\n", result[tile_offset + threadIdx.x + ITEM * B]);
        }
    }

}

template <typename T>
__global__ void testprintmat(T* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = N - 1000; i < N; i++) {
            printf("%lu ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void materialize_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, int* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, uint64_t* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, float* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, double* result, uint64_t *row_ids, uint64_t N);
template
__global__ void materialize_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, uint8_t* result, uint64_t *row_ids, uint64_t N);

template
__global__ void testprintmat<uint64_t>(uint64_t* a, uint64_t N);
template
__global__ void testprintmat<double>(double* a, uint64_t N);
template
__global__ void testprintmat<int>(int* a, uint64_t N);
template
__global__ void testprintmat<float>(float* a, uint64_t N);
template
__global__ void testprintmat<uint8_t>(uint8_t* a, uint64_t N);

template <typename T>
void materializeExpression(T *a, T* result, uint64_t *row_ids, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Materialize Kernel\n");
    printf("N: %lu\n", N);
    // testprintmat<T><<<1, 1>>>(a, N);
    // CHECK_ERROR();
    // testprintmat<uint64_t><<<1, 1>>>(row_ids, N);
    // CHECK_ERROR();
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    materialize_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, result, row_ids, N);
    CHECK_ERROR();
    // thrust::device_vector<T> sorted(result, result + N);
    // thrust::sort(thrust::device, sorted.begin(), sorted.end());
    // T* raw_sorted = thrust::raw_pointer_cast(sorted.data());
    // cudaMemcpy(result, raw_sorted, N * sizeof(T), cudaMemcpyDeviceToDevice);
    // test<T><<<1, 1>>>(a, N);
    cudaDeviceSynchronize();
}

template
void materializeExpression<int>(int *a, int* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<uint64_t>(uint64_t *a, uint64_t* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<float>(float *a, float* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<double>(double *a, double* result, uint64_t *row_ids, uint64_t N);
template
void materializeExpression<uint8_t>(uint8_t *a, uint8_t* result, uint64_t *row_ids, uint64_t N);

} // namespace duckdb