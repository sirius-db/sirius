#include "cuda_helper.cuh"
#include "gpu_physical_nested_loop_join.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

//TODO: Currently only support a single key
template <typename T, int B, int I>
__global__ void nested_loop_join_count(T *left_keys, T* right_keys, uint64_t *offset_each_thread, 
            unsigned long long* total_count, uint64_t left_size, uint64_t right_size, int condition_mode) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    int items_count[I];

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (left_size + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0; // Number of items selected per thread
    int c_t_count = 0; //Prefix sum of t_count
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = left_size - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        items_count[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            for (int i = 0; i < right_size; i++) {
                bool local_found = 1;
                if (condition_mode == 0 && left_keys[tile_offset + threadIdx.x + ITEM * B] != right_keys[i]) {
                    local_found = 0;
                } else if (condition_mode == 1 && left_keys[tile_offset + threadIdx.x + ITEM * B] == right_keys[i]) {
                    local_found = 0;
                } else if (condition_mode == 2 && left_keys[tile_offset + threadIdx.x + ITEM * B] >= right_keys[i]) {
                    local_found = 0;
                } else if (condition_mode == 3 && left_keys[tile_offset + threadIdx.x + ITEM * B] <= right_keys[i]) {
                    // printf("left_keys: %.2f right_keys: %.2f\n", left_keys[tile_offset + threadIdx.x + ITEM * B], right_keys[i]);
                    local_found = 0;
                }

                if (local_found) {
                    items_count[ITEM]++;
                }
            }
            t_count += items_count[ITEM];
        }
    }

    //Barrier
    __syncthreads();

    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count); //doing a prefix sum of all the previous threads in the block and store it to c_t_count
    if(threadIdx.x == blockDim.x - 1) { //if the last thread in the block, add the prefix sum of all the prev threads + sum of my threads to global variable total
        block_off = atomicAdd(total_count, (unsigned long long) t_count+c_t_count); //the previous value of total is gonna be assigned to block_off
    } //block_off does not need to be global (it's just need to be shared), because it will get the previous value from total which is global

    __syncthreads();

     if (blockIdx.x * tile_size + threadIdx.x < left_size) {
        offset_each_thread[blockIdx.x * B + threadIdx.x] = block_off + c_t_count;
    }

}

template <typename T, int B, int I>
__global__ void nested_loop_join(T *left_keys, T* right_keys, uint64_t *offset_each_thread, uint64_t *row_ids_left, uint64_t *row_ids_right,
            uint64_t left_size, uint64_t right_size, int condition_mode) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (left_size + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = left_size - tile_offset;
    }

    uint64_t output_offset = 0;
    if (blockIdx.x * tile_size + threadIdx.x < left_size) {
        output_offset = offset_each_thread[blockIdx.x * B + threadIdx.x];
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            for (int i = 0; i < right_size; i++) {
                bool local_found = 1;
                if (condition_mode == 0 && left_keys[tile_offset + threadIdx.x + ITEM * B] != right_keys[i]) {
                    local_found = 0;
                } else if (condition_mode == 1 && left_keys[tile_offset + threadIdx.x + ITEM * B] == right_keys[i]) {
                    local_found = 0;
                } else if (condition_mode == 2 && left_keys[tile_offset + threadIdx.x + ITEM * B] >= right_keys[i]) {
                    local_found = 0;
                } else if (condition_mode == 3 && left_keys[tile_offset + threadIdx.x + ITEM * B] <= right_keys[i]) {
                    local_found = 0;
                }

                if (local_found) {
                    row_ids_right[output_offset] = i;
                    row_ids_left[output_offset] = tile_offset + threadIdx.x + ITEM * B;
                    output_offset++;
                }
            }
        }
    }
}

template
__global__ void nested_loop_join_count<double, BLOCK_THREADS, 1>(double *left_keys, double* right_keys, uint64_t *offset_each_thread, 
            unsigned long long* total_count, uint64_t left_size, uint64_t right_size, int condition_mode);

template
__global__ void nested_loop_join_count<uint64_t, BLOCK_THREADS, 1>(uint64_t *left_keys, uint64_t* right_keys, uint64_t *offset_each_thread, 
            unsigned long long* total_count, uint64_t left_size, uint64_t right_size, int condition_mode);

template
__global__ void nested_loop_join<double, BLOCK_THREADS, 1>(double *left_keys, double* right_keys, uint64_t *offset_each_thread, uint64_t *row_ids_left, uint64_t *row_ids_right,
            uint64_t left_size, uint64_t right_size, int condition_mode);

template
__global__ void nested_loop_join<uint64_t, BLOCK_THREADS, 1>(uint64_t *left_keys, uint64_t* right_keys, uint64_t *offset_each_thread, uint64_t *row_ids_left, uint64_t *row_ids_right,
            uint64_t left_size, uint64_t right_size, int condition_mode);

template <typename T>
void nestedLoopJoin(T** left_data, T** right_data, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t left_size, uint64_t right_size, int* condition_mode, int num_keys) {
    CHECK_ERROR();
    if (left_size == 0 || right_size == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Nested Loop Join Kernel\n");
    // printf("left size: %lu right size : %lu\n", left_size, right_size);
    int tile_items = BLOCK_THREADS * 1;
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    uint64_t* offset_each_thread = gpuBufferManager->customCudaMalloc<uint64_t>(((left_size + tile_items - 1)/tile_items) * BLOCK_THREADS, 0, 0);
    
    //TODO: Currently only support a single key
    CHECK_ERROR();
    nested_loop_join_count<T, BLOCK_THREADS, 1><<<(left_size + tile_items - 1)/tile_items, BLOCK_THREADS>>>(left_data[0], right_data[0], 
            offset_each_thread, (unsigned long long*) count, left_size, right_size, condition_mode[0]);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    uint64_t* h_count = new uint64_t[1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(h_count[0] > 0);
    printf("Count: %lu\n", h_count[0]);

    row_ids_left = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    nested_loop_join<T, BLOCK_THREADS, 1><<<(left_size + tile_items - 1)/tile_items, BLOCK_THREADS>>>(left_data[0], right_data[0], 
            offset_each_thread, row_ids_left, row_ids_right, left_size, right_size, condition_mode[0]);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    gpuBufferManager->customCudaFree<uint64_t>(offset_each_thread, ((left_size + tile_items - 1)/tile_items) * BLOCK_THREADS, 0);
    gpuBufferManager->customCudaFree<uint64_t>(count, 1, 0);
    count = h_count;
}

template
void nestedLoopJoin<double>(double** left_data, double** right_data, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t left_size, uint64_t right_size, int* condition_mode, int num_keys);

template
void nestedLoopJoin<uint64_t>(uint64_t** left_data, uint64_t** right_data, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t left_size, uint64_t right_size, int* condition_mode, int num_keys);

} // namespace duckdb