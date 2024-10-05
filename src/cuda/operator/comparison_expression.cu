#include "cuda_helper.cuh"
#include "gpu_expression_executor.hpp"

namespace duckdb {

template <typename T, int B, int I>
__global__ void comparison_expression(const T *a, const T *b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T items_a[I];
    T items_b[I];
    int selection_flags[I];

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0; // Number of items selected per thread
    int c_t_count = 0; //Prefix sum of t_count
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            items_a[ITEM] = a[tile_offset + threadIdx.x + ITEM * B];
            items_b[ITEM] = b[tile_offset + threadIdx.x + ITEM * B];
            if (compare_mode == 0) {
                selection_flags[ITEM] = (items_a[ITEM] == items_b[ITEM]);
            } else if (compare_mode == 1) {
                selection_flags[ITEM] = (items_a[ITEM] > items_b[ITEM]);
            } else if (compare_mode == 2) {
                selection_flags[ITEM] = (items_a[ITEM] < items_b[ITEM]);
            } else {
                cudaAssert(0);
            }
            selection_flags[ITEM] = (items_a[ITEM] == items_b[ITEM]);
            if(selection_flags[ITEM]) t_count++;
        }
    }

    //Barrier
    __syncthreads();

    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count); //doing a prefix sum of all the previous threads in the block and store it to c_t_count
    if(threadIdx.x == blockDim.x - 1) { //if the last thread in the block, add the prefix sum of all the prev threads + sum of my threads to global variable total
        block_off = atomicAdd(count, (unsigned long long) t_count+c_t_count); //the previous value of total is gonna be assigned to block_off
    } //block_off does not need to be global (it's just need to be shared), because it will get the previous value from total which is global

    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                result[offset] = blockIdx.x * tile_size + threadIdx.x + ITEM * B;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void comparison_constant_expression(const T *a, const T b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T items_a[I];
    int selection_flags[I];

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0; // Number of items selected per thread
    int c_t_count = 0; //Prefix sum of t_count
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            items_a[ITEM] = a[tile_offset + threadIdx.x + ITEM * B];
            if (compare_mode == 0) {
                selection_flags[ITEM] = (items_a[ITEM] == b);
            } else if (compare_mode == 1) {
                selection_flags[ITEM] = (items_a[ITEM] > b);
            } else if (compare_mode == 2) {
                selection_flags[ITEM] = (items_a[ITEM] < b);
            } else {
                cudaAssert(0);
            }
            if(selection_flags[ITEM]) t_count++;
        }
    }

    //Barrier
    __syncthreads();

    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count); //doing a prefix sum of all the previous threads in the block and store it to c_t_count
    if(threadIdx.x == blockDim.x - 1) { //if the last thread in the block, add the prefix sum of all the prev threads + sum of my threads to global variable total
        block_off = atomicAdd(count, (unsigned long long) t_count+c_t_count); //the previous value of total is gonna be assigned to block_off
    } //block_off does not need to be global (it's just need to be shared), because it will get the previous value from total which is global

    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                result[offset] = blockIdx.x * tile_size + threadIdx.x + ITEM * B;
            }
        }
    }
}

template
__global__ void comparison_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, const int *b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, const uint64_t *b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, const float *b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, const double *b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, const uint8_t *b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);

template
__global__ void comparison_constant_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, const int b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_constant_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, const uint64_t b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_constant_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, const float b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_constant_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, const double b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);
template
__global__ void comparison_constant_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, const uint8_t b, uint64_t *result, unsigned long long* count, uint64_t N, int compare_mode);

} // namespace duckdb