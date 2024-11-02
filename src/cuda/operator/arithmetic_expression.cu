#include "cuda_helper.cuh"
#include "gpu_expression_executor.hpp"

namespace duckdb {

template <typename T, int B, int I>
__global__ void binary_expression(T *a, T *b, T *result, uint64_t N, int op_mode) {
    
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
            } else {
                cudaAssert(0);
            }
        }
    }
}

template
__global__ void binary_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(int *a, int *b, int *result, uint64_t N, int op_mode);
template
__global__ void binary_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *a, uint64_t *b, uint64_t *result, uint64_t N, int op_mode);
template
__global__ void binary_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(float *a, float *b, float *result, uint64_t N, int op_mode);
template
__global__ void binary_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(double *a, double *b, double *result, uint64_t N, int op_mode);
template
__global__ void binary_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint8_t *a, uint8_t *b, uint8_t *result, uint64_t N, int op_mode);

// Define the host function that launches the CUDA kernel
template <typename T>
void binaryExpression(T *a, T *b, T *result, uint64_t N, int op_mode) {
    printf("Launching Binary Expression Kernel\n");
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    binary_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, result, N, op_mode);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template
void binaryExpression<int>(int *a, int *b, int *result, uint64_t N, int op_mode);
template
void binaryExpression<uint64_t>(uint64_t *a, uint64_t *b, uint64_t *result, uint64_t N, int op_mode);
template
void binaryExpression<float>(float *a, float *b, float *result, uint64_t N, int op_mode);
template
void binaryExpression<double>(double *a, double *b, double *result, uint64_t N, int op_mode);
template
void binaryExpression<uint8_t>(uint8_t *a, uint8_t *b, uint8_t *result, uint64_t N, int op_mode);

} // namespace duckdb