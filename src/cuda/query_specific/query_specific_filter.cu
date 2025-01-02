#include "../operator/cuda_helper.cuh"
#include "gpu_expression_executor.hpp"

namespace duckdb {

// string t = "((P_BRAND != 45) AND ((P_TYPE < 65) OR (P_TYPE >= 70)) AND (P_SIZE IN (49, 14, 23, 45, 19, 3, 36, 9)))";
// string t = "((L_COMMITDATE < L_RECEIPTDATE) AND (L_SHIPDATE < L_COMMITDATE) AND (L_SHIPMODE IN (4, 6)))";
// string t = "(((P_BRAND = 12) AND (L_QUANTITY <= 11) AND (P_SIZE <= 5) AND (P_CONTAINER IN (0, 1, 4, 5))) OR ((P_BRAND = 23) AND (L_QUANTITY >= 10) AND (L_QUANTITY <= 20) AND (P_SIZE <= 10) AND (P_CONTAINER IN (17, 18, 20, 21))) OR ((P_BRAND = 34) AND (L_QUANTITY >= 20) AND (L_QUANTITY <= 30) AND (P_SIZE <= 15) AND (P_CONTAINER IN (8, 9, 12, 13))))";
// string t = "(((N_NATIONKEY = 6) AND (N_NATIONKEY = 7)) OR ((N_NATIONKEY = 7) AND (N_NATIONKEY = 6)))";
// string t = "(((P_TYPE + 3) % 5) = 0)";

// int first = (item_p_brand[ITEM] == 12) && (item_l_quantity[ITEM] <= 11) && (item_p_size[ITEM] <= 5) && ((item_p_container[ITEM] == 0 || item_p_container[ITEM] == 1 || item_p_container[ITEM] == 4 || item_p_container[ITEM] == 5));
// int second = (item_p_brand[ITEM] == 23) && (item_l_quantity[ITEM] >= 10) && (item_l_quantity[ITEM] <= 20) && (item_p_size[ITEM] <= 10) && ((item_p_container[ITEM] == 17 || item_p_container[ITEM] == 18 || item_p_container[ITEM] == 20 || item_p_container[ITEM] == 21));
// int third = (item_p_brand[ITEM] == 34) && (item_l_quantity[ITEM] >= 20) && (item_l_quantity[ITEM] <= 30) && (item_p_size[ITEM] <= 15) && ((item_p_container[ITEM] == 8 || item_p_container[ITEM] == 9 || item_p_container[ITEM] == 12 || item_p_container[ITEM] == 13));

template <typename T, typename V, int B, int I>
__global__ void q19_filter(T *p_brand, V *l_quantity, T *p_size, T* p_container, 
    T *p_brand_val, V *l_quantity_val, T *p_size_val, T* p_container_val,
    uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T item_p_brand[I];
    T item_l_quantity[I];
    T item_p_size[I];
    T item_p_container[I];
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
            item_p_brand[ITEM] = p_brand[tile_offset + threadIdx.x + ITEM * B];
            item_l_quantity[ITEM] = l_quantity[tile_offset + threadIdx.x + ITEM * B];
            item_p_size[ITEM] = p_size[tile_offset + threadIdx.x + ITEM * B];
            item_p_container[ITEM] = p_container[tile_offset + threadIdx.x + ITEM * B];
            int first = (item_p_brand[ITEM] == p_brand_val[0]) && (item_l_quantity[ITEM] >= l_quantity_val[0]) && (item_l_quantity[ITEM] <= l_quantity_val[1]) && (item_p_size[ITEM] <= p_size_val[0]) && \
                            ((item_p_container[ITEM] == p_container_val[0] || item_p_container[ITEM] == p_container_val[1] || item_p_container[ITEM] == p_container_val[2] || item_p_container[ITEM] == p_container_val[3]));
            int second = (item_p_brand[ITEM] == p_brand_val[1]) && (item_l_quantity[ITEM] >= l_quantity_val[2]) && (item_l_quantity[ITEM] <= l_quantity_val[3]) && (item_p_size[ITEM] <= p_size_val[1]) && \
                            ((item_p_container[ITEM] == p_container_val[4] || item_p_container[ITEM] == p_container_val[5] || item_p_container[ITEM] == p_container_val[6] || item_p_container[ITEM] == p_container_val[7]));
            int third = (item_p_brand[ITEM] == p_brand_val[2]) && (item_l_quantity[ITEM] >= l_quantity_val[4]) && (item_l_quantity[ITEM] <= l_quantity_val[5]) && (item_p_size[ITEM] <= p_size_val[2]) && \
                            ((item_p_container[ITEM] == p_container_val[8] || item_p_container[ITEM] == p_container_val[9] || item_p_container[ITEM] == p_container_val[10] || item_p_container[ITEM] == p_container_val[11]));
            selection_flags[ITEM] = first || second || third;
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
__global__ void q16_filter(T *p_brand, T *p_type, T *p_size,
    T p_brand_val, T p_type_val1, T p_type_val2, T *p_size_val,
    uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T item_p_brand[I];
    T item_p_type[I];
    T item_p_size[I];
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
            item_p_brand[ITEM] = p_brand[tile_offset + threadIdx.x + ITEM * B];
            item_p_type[ITEM] = p_type[tile_offset + threadIdx.x + ITEM * B];
            item_p_size[ITEM] = p_size[tile_offset + threadIdx.x + ITEM * B];
            selection_flags[ITEM] = (item_p_brand[ITEM] != p_brand_val) && ((item_p_type[ITEM] < p_type_val1) || (item_p_type[ITEM] >= p_type_val2)) && \
                            ((item_p_size[ITEM] == p_size_val[0] || item_p_size[ITEM] == p_size_val[1] || item_p_size[ITEM] == p_size_val[2] || item_p_size[ITEM] == p_size_val[3]) || \
                            (item_p_size[ITEM] == p_size_val[4] || item_p_size[ITEM] == p_size_val[5] || item_p_size[ITEM] == p_size_val[6] || item_p_size[ITEM] == p_size_val[7]));
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
__global__ void q12_filter(T *l_commitdate, T *l_receiptdate, T *l_shipdate, T *l_shipmode,
    T l_shipmode_val1, T l_shipmode_val2,
    uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T item_l_commitdate[I];
    T item_l_receiptdate[I];
    T item_l_shipdate[I];
    T item_l_shipmode[I];
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
            item_l_commitdate[ITEM] = l_commitdate[tile_offset + threadIdx.x + ITEM * B];
            item_l_receiptdate[ITEM] = l_receiptdate[tile_offset + threadIdx.x + ITEM * B];
            item_l_shipdate[ITEM] = l_shipdate[tile_offset + threadIdx.x + ITEM * B];
            item_l_shipmode[ITEM] = l_shipmode[tile_offset + threadIdx.x + ITEM * B];
            selection_flags[ITEM] = (item_l_commitdate[ITEM] < item_l_receiptdate[ITEM]) && (item_l_shipdate[ITEM] < item_l_commitdate[ITEM]) && \
                            ((item_l_shipmode[ITEM] == l_shipmode_val1 || item_l_shipmode[ITEM] == l_shipmode_val2));
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
__global__ void q7_filter(T *n1_nationkey, T *n2_nationkey, T val1, T val2, T val3, T val4,
    uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T item_n1_nationkey[I];
    T item_n2_nationkey[I];
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
            item_n1_nationkey[ITEM] = n1_nationkey[tile_offset + threadIdx.x + ITEM * B];
            item_n2_nationkey[ITEM] = n2_nationkey[tile_offset + threadIdx.x + ITEM * B];
            selection_flags[ITEM] = ((item_n1_nationkey[ITEM] == val1) && (item_n2_nationkey[ITEM] == val2)) || ((item_n1_nationkey[ITEM] == val3) && (item_n2_nationkey[ITEM] == val4));
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
__global__ void q2_filter(T *p_type, T p_type_val,
    uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    T item_p_type[I];
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
            item_p_type[ITEM] = p_type[tile_offset + threadIdx.x + ITEM * B];
            selection_flags[ITEM] = (((item_p_type[ITEM] + 3) % 5) == p_type_val);
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

template
__global__ void q19_filter<uint64_t, double, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *p_brand, double *l_quantity, uint64_t *p_size, uint64_t* p_container, uint64_t *p_brand_val, double *l_quantity_val, uint64_t *p_size_val, uint64_t* p_container_val, 
                                uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count);
template
__global__ void q16_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *p_brand, uint64_t *p_type, uint64_t *p_size, uint64_t p_brand_val, uint64_t p_type_val1, uint64_t p_type_val2, uint64_t *p_size_val, 
                                uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count);
template
__global__ void q12_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *l_commitdate, uint64_t *l_receiptdate, uint64_t *l_shipdate, uint64_t *l_shipmode, uint64_t l_shipmode_val1, uint64_t l_shipmode_val2,
                                uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count);
template
__global__ void q7_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *n1_nationkey, uint64_t *n2_nationkey, uint64_t val1, uint64_t val2, uint64_t val3, uint64_t val4, 
                                uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count);
template
__global__ void q2_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *p_type, uint64_t p_type_val, uint64_t *row_ids, unsigned long long* count, uint64_t N, int is_count);


void q19FilterExpression(uint64_t *p_brand, double *l_quantity, uint64_t *p_size, uint64_t* p_container, uint64_t *p_brand_val, double *l_quantity_val, uint64_t *p_size_val, uint64_t* p_container_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Q19 Filter Kernel\n");

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* d_p_brand_val = gpuBufferManager->customCudaMalloc<uint64_t>(3, 0, 0).data_;
    double* d_l_quantity_val = gpuBufferManager->customCudaMalloc<double>(6, 0, 0).data_;
    uint64_t* d_p_size_val = gpuBufferManager->customCudaMalloc<uint64_t>(3, 0, 0).data_;
    uint64_t* d_p_container_val = gpuBufferManager->customCudaMalloc<uint64_t>(12, 0, 0).data_;
    callCudaMemcpyHostToDevice<uint64_t>(d_p_brand_val, p_brand_val, 3, 0);
    callCudaMemcpyHostToDevice<double>(d_l_quantity_val, l_quantity_val, 6, 0);
    callCudaMemcpyHostToDevice<uint64_t>(d_p_size_val, p_size_val, 3, 0);
    callCudaMemcpyHostToDevice<uint64_t>(d_p_container_val, p_container_val, 12, 0);

    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    q19_filter<uint64_t, double, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(p_brand, l_quantity, p_size, p_container, d_p_brand_val, d_l_quantity_val, d_p_size_val, d_p_container_val, row_ids, (unsigned long long*) count, N, 1);
    CHECK_ERROR();
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0).data_;
    cudaMemset(count, 0, sizeof(uint64_t));
    q19_filter<uint64_t, double, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(p_brand, l_quantity, p_size, p_container, d_p_brand_val, d_l_quantity_val, d_p_size_val, d_p_container_val, row_ids, (unsigned long long*) count, N, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

void q16FilterExpression(uint64_t *p_brand, uint64_t *p_type, uint64_t *p_size, uint64_t p_brand_val, uint64_t p_type_val1, uint64_t p_type_val2, uint64_t *p_size_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Q16 Filter Kernel\n");

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* d_p_size_val = gpuBufferManager->customCudaMalloc<uint64_t>(8, 0, 0).data_;
    callCudaMemcpyHostToDevice<uint64_t>(d_p_size_val, p_size_val, 8, 0);

    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    q16_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(p_brand, p_type, p_size, p_brand_val, p_type_val1, p_type_val2, d_p_size_val, row_ids, (unsigned long long*) count, N, 1);
    CHECK_ERROR();
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0).data_;
    cudaMemset(count, 0, sizeof(uint64_t));
    q16_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(p_brand, p_type, p_size, p_brand_val, p_type_val1, p_type_val2, d_p_size_val, row_ids, (unsigned long long*) count, N, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

void q12FilterExpression(uint64_t *l_commitdate, uint64_t *l_receiptdate, uint64_t *l_shipdate, uint64_t *l_shipmode, uint64_t l_shipmode_val1, uint64_t l_shipmode_val2, uint64_t* &row_ids, uint64_t* &count, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Q12 Filter Kernel\n");
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    q12_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(l_commitdate, l_receiptdate, l_shipdate, l_shipmode, l_shipmode_val1, l_shipmode_val2, row_ids, (unsigned long long*) count, N, 1);
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0).data_;
    cudaMemset(count, 0, sizeof(uint64_t));
    q12_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(l_commitdate, l_receiptdate, l_shipdate, l_shipmode, l_shipmode_val1, l_shipmode_val2, row_ids, (unsigned long long*) count, N, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

void q2FilterExpression(uint64_t *p_type, uint64_t p_type_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Q2 Filter Kernel\n");
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    q2_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(p_type, p_type_val, row_ids, (unsigned long long*) count, N, 1);
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0).data_;
    cudaMemset(count, 0, sizeof(uint64_t));
    q2_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(p_type, p_type_val, row_ids, (unsigned long long*) count, N, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

void q7FilterExpression(uint64_t *n1_nationkey, uint64_t *n2_nationkey, uint64_t val1, uint64_t val2, uint64_t val3, uint64_t val4, 
                                uint64_t* &row_ids, uint64_t* &count, uint64_t N) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Q7 Filter Kernel\n");
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    q7_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(n1_nationkey, n2_nationkey, val1, val2, val3, val4, row_ids, (unsigned long long*) count, N, 1);
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0).data_;
    cudaMemset(count, 0, sizeof(uint64_t));
    q7_filter<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(n1_nationkey, n2_nationkey, val1, val2, val3, val4, row_ids, (unsigned long long*) count, N, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

__global__ void q22_filter(char* a, uint64_t* offset, uint64_t start_idx, uint64_t length, uint8_t** d_c_phone_val, 
            int num_predicates, uint64_t* row_ids, unsigned long long* count, uint64_t N, bool is_count) {

    // Get which string this thread workers on
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N) return; 

    // Determine the range for this substring
    uint64_t curr_str_start_idx = offset[tid]; uint64_t curr_str_end_idx = offset[tid + 1];
    uint64_t substring_start_idx = min(curr_str_start_idx + start_idx, curr_str_end_idx);

    int t_count = 0; // Number of items selected per thread
    // int c_t_count = 0; //Prefix sum of t_count
    bool match = false;

    __syncthreads();

    for (int i = 0; i < num_predicates; i++) {
        bool local_match = true;
        for (int j = 0; j < length; j++) {
            if (a[substring_start_idx + j] != d_c_phone_val[i][j]) {
                local_match = false;
                break;
            }
        }
        if (local_match) {
            // printf("%c %c %c %c\n", a[substring_start_idx], a[substring_start_idx + 1], d_c_phone_val[i][0], d_c_phone_val[i][1]);
            match = true;
            t_count = 1;
            break;
        }
    }

    __shared__ unsigned long long buffer[1];
    __shared__ uint64_t block_off;

    if (threadIdx.x == 0) buffer[0] = 0;
    __syncthreads();

    unsigned long long local_idx = atomicAdd(buffer, (unsigned long long) t_count); //add the sum of t_count to global variable total
    __syncthreads();

    if (threadIdx.x == 0) {
        block_off = atomicAdd(count, buffer[0]);
    }
    __syncthreads();

    if (is_count) return;

    if (match) {
        // if (block_off >= 41996) printf("Warning here %ld %d %ld %ld\n", block_off, c_t_count, tid, N);
        // uint64_t idx = block_off + c_t_count;
        row_ids[block_off + local_idx] = tid;
    }
}

__global__ void printstring(uint8_t* a, uint64_t* offset, uint64_t N) {
    for (int i = 0; i < 100; i++) {
        printf("%c %c %c %c %c\n", a[offset[i]], a[offset[i] + 1], a[offset[i] + 2], a[offset[i] + 3], a[offset[i] + 4]);
    }
    printf("\n");
}

void q22FilterExpression(uint8_t *a, uint64_t* offset, uint64_t start_idx, uint64_t length, string c_phone_val, uint64_t* &row_ids, uint64_t* &count, uint64_t N, int num_predicates) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Q22 Filter Kernel\n");

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    uint8_t* temp = gpuBufferManager->customCudaMalloc<uint8_t>(num_predicates * length, 0, 0);
    cudaMemcpy(temp, c_phone_val.c_str(), num_predicates * length * sizeof(uint8_t), cudaMemcpyHostToDevice);
    uint8_t** h_c_phone_val = new uint8_t*[num_predicates];
    for (int i = 0; i < num_predicates; i++) {
        h_c_phone_val[i] = temp + i * length;
    }
    uint8_t** d_c_phone_val;
    cudaMalloc((void**) &d_c_phone_val, num_predicates * sizeof(uint8_t*));
    cudaMemcpy(d_c_phone_val, h_c_phone_val, num_predicates * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemset(count, 0, sizeof(uint64_t));

    printf("N is %ld\n", N);
    uint64_t num_blocks = (N + 128 - 1)/128;

    // printstring<<<1, 1>>>(a, offset, N);

    q22_filter<<<num_blocks, 128>>>(reinterpret_cast<char*>(a), offset, start_idx, length, d_c_phone_val, 
            num_predicates, row_ids, (unsigned long long*) count, N, 1);
    CHECK_ERROR();
    
    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);

    cudaMemset(count, 0, sizeof(uint64_t));
    q22_filter<<<num_blocks, 128>>>(reinterpret_cast<char*>(a), offset, start_idx, length, d_c_phone_val, 
            num_predicates, row_ids, (unsigned long long*) count, N, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    count = h_count;
    printf("Count: %lu\n", h_count[0]);
}

} // namespace duckdb