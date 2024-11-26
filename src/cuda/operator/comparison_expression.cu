#include "cuda_helper.cuh"
#include "gpu_expression_executor.hpp"

namespace duckdb {

// string t = "((P_BRAND != 45) AND ((P_TYPE < 65) OR (P_TYPE >= 70)) AND (P_SIZE IN (49, 14, 23, 45, 19, 3, 36, 9)))";
// string t = "((L_COMMITDATE < L_RECEIPTDATE) AND (L_SHIPDATE < L_COMMITDATE) AND (L_SHIPMODE IN (4, 6)))";
// string t = "(((P_BRAND = 12) AND (L_QUANTITY <= 11) AND (P_SIZE <= 5) AND (P_CONTAINER IN (0, 1, 4, 5))) OR ((P_BRAND = 23) AND (L_QUANTITY >= 10) AND (L_QUANTITY <= 20) AND (P_SIZE <= 10) AND (P_CONTAINER IN (17, 18, 20, 21))) OR ((P_BRAND = 34) AND (L_QUANTITY >= 20) AND (L_QUANTITY <= 30) AND (P_SIZE <= 15) AND (P_CONTAINER IN (8, 9, 12, 13))))";
// string t = "(((N_NATIONKEY = 6) AND (N_NATIONKEY = 7)) OR ((N_NATIONKEY = 7) AND (N_NATIONKEY = 6)))";
// string t = "(((P_TYPE + 3) % 5) = 0)";

template <typename T, int B, int I>
__global__ void comparison_expression(const T *a, const T *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count) {

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
                selection_flags[ITEM] = (items_a[ITEM] != items_b[ITEM]);
            } else if (compare_mode == 2) {
                selection_flags[ITEM] = (items_a[ITEM] > items_b[ITEM]);
            } else if (compare_mode == 3) {
                selection_flags[ITEM] = (items_a[ITEM] >= items_b[ITEM]);
            } else if (compare_mode == 4) {
                selection_flags[ITEM] = (items_a[ITEM] < items_b[ITEM]);
            } else if (compare_mode == 5) {
                selection_flags[ITEM] = (items_a[ITEM] <= items_b[ITEM]);
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

    if (is_count) return;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                row_ids[offset] = tile_offset + threadIdx.x + ITEM * B;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void comparison_constant_expression(const T *a, const T b, const T c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count) {

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
                selection_flags[ITEM] = (items_a[ITEM] != b);
            } else if (compare_mode == 2) {
                selection_flags[ITEM] = (items_a[ITEM] > b);
            } else if (compare_mode == 3) {
                selection_flags[ITEM] = (items_a[ITEM] >= b);
            } else if (compare_mode == 4) {
                selection_flags[ITEM] = (items_a[ITEM] < b);
            } else if (compare_mode == 5) {
                selection_flags[ITEM] = (items_a[ITEM] <= b);
            } else if (compare_mode == 6) {
                selection_flags[ITEM] = ((items_a[ITEM] >= b) && (items_a[ITEM] <= c));
            } else if (compare_mode == 7) {
                selection_flags[ITEM] = ((items_a[ITEM] < b) || (items_a[ITEM] > c));
            } else {
                cudaAssert(0);
            }
            if(selection_flags[ITEM]) {
                t_count++;
            }
        }
    }

    //Barrier
    __syncthreads();

    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count); //doing a prefix sum of all the previous threads in the block and store it to c_t_count
    if(threadIdx.x == blockDim.x - 1) { //if the last thread in the block, add the prefix sum of all the prev threads + sum of my threads to global variable total
        block_off = atomicAdd(count, (unsigned long long) t_count+c_t_count); //the previous value of total is gonna be assigned to block_off
    } //block_off does not need to be global (it's just need to be shared), because it will get the previous value from total which is global

    __syncthreads();

    if (is_count) return;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                row_ids[offset] = tile_offset + threadIdx.x + ITEM * B;
            }
        }
    }
}

template<typename T>
__global__ void test(T* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 100; i++) {
            printf("%.2f ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void comparison_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, const int *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, const uint64_t *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, const float *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, const double *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, const uint8_t *b, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);

template
__global__ void comparison_constant_expression<int, BLOCK_THREADS, ITEMS_PER_THREAD>(const int *a, const int b, const int c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *a, const uint64_t b, const uint64_t c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<float, BLOCK_THREADS, ITEMS_PER_THREAD>(const float *a, const float b, const float c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<double, BLOCK_THREADS, ITEMS_PER_THREAD>(const double *a, const double b, const double c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);
template
__global__ void comparison_constant_expression<uint8_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint8_t *a, const uint8_t b, const uint8_t c, uint64_t *row_ids, unsigned long long* count, uint64_t N, int compare_mode, int is_count);

template <typename T>
void comparisonConstantExpression(T *a, T b, T c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Comparison Expression Kernel\n");
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    comparison_constant_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, c, row_ids, (unsigned long long*) count, N, op_mode, 1);
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    comparison_constant_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, c, row_ids, (unsigned long long*) count, N, op_mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

template <typename T>
void comparisonExpression(T *a, T *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Comparison Expression Kernel\n");
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    comparison_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, row_ids, (unsigned long long*) count, N, op_mode, 1);
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    comparison_expression<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(a, b, row_ids, (unsigned long long*) count, N, op_mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

template
void comparisonConstantExpression<int>(int *a, int b, int c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<uint64_t>(uint64_t *a, uint64_t b, uint64_t c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<float>(float *a, float b, float c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<double>(double *a, double b, double c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonConstantExpression<uint8_t>(uint8_t *a, uint8_t b, uint8_t c, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);


template
void comparisonExpression<int>(int *a, int *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<uint64_t>(uint64_t *a, uint64_t *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<float>(float *a, float *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<double>(double *a, double *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);
template
void comparisonExpression<uint8_t>(uint8_t *a, uint8_t *b, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int op_mode);

} // namespace duckdb