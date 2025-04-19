#include "cuda_helper.cuh"
#include "gpu_physical_ungrouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

template <typename T, int B, int I>
__global__ void ungrouped_aggregate(T *a, T *result, uint64_t N, int op_mode) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    T local;
    if (op_mode == 0) local = 0;
    else if (op_mode == 1) local = 0;
    else if (op_mode == 2) local = a[0];
    else if (op_mode == 3) local = a[0];

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
            if (op_mode == 0) local += a[tile_offset + threadIdx.x + ITEM * B];
            else if (op_mode == 1) local += a[tile_offset + threadIdx.x + ITEM * B];
            else if (op_mode == 2) local = max(local, a[tile_offset + threadIdx.x + ITEM * B]);
            else if (op_mode == 3) local = min(local, a[tile_offset + threadIdx.x + ITEM * B]);
            else cudaAssert(0);
        }
    }

    __syncthreads();
    static __shared__ T buffer[32];
    cuda::atomic_ref<T, cuda::thread_scope_device> res_atomic(*result);

    if (op_mode == 0 || op_mode == 1) { //sum or avg
        
        T block_reduce = BlockReduce<T, BLOCK_THREADS, ITEMS_PER_THREAD>(local, (T*)buffer, op_mode);
        __syncthreads();

        if (threadIdx.x == 0) {
            // atomicAdd(reinterpret_cast<T*>(result), block_reduce); 
            res_atomic.fetch_add(block_reduce);  
        }

        __syncthreads();

    } else if (op_mode == 2) { //max
        T block_reduce = BlockReduce<T, BLOCK_THREADS, ITEMS_PER_THREAD>(local, (T*)buffer, op_mode);
        __syncthreads();

        if (threadIdx.x == 0) {
            // atomicMax(reinterpret_cast<T*>(result), block_reduce);  
            res_atomic.fetch_max(block_reduce); 
        }

        __syncthreads();
    } else if (op_mode == 3) { //min
        T block_reduce = BlockReduce<T, BLOCK_THREADS, ITEMS_PER_THREAD>(local, (T*)buffer, op_mode);
        __syncthreads();

        if (threadIdx.x == 0) {
            // atomicMin(reinterpret_cast<T*>(result), block_reduce); 
            res_atomic.fetch_min(block_reduce);  
        }

        __syncthreads();
    } else {
        cudaAssert(0);
    }
}

template
__global__ void ungrouped_aggregate<int, BLOCK_THREADS, ITEMS_PER_THREAD>(int *a, int *result, uint64_t N, int op_mode);
template
__global__ void ungrouped_aggregate<unsigned long long, BLOCK_THREADS, ITEMS_PER_THREAD>(unsigned long long *a, unsigned long long *result, uint64_t N, int op_mode);
template
__global__ void ungrouped_aggregate<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t *a, uint64_t *result, uint64_t N, int op_mode);
template
__global__ void ungrouped_aggregate<double, BLOCK_THREADS, ITEMS_PER_THREAD>(double *a, double *result, uint64_t N, int op_mode);


// Define the host function that launches the CUDA kernel
template <typename T>
void ungroupedAggregate(uint8_t **a, uint8_t **result, uint64_t N, int* agg_mode, int num_aggregates) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Aggregation Kernel\n");
    printf("N: %ld\n", N);
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    for (int agg = 0; agg < num_aggregates; agg++) {
        if (agg_mode[agg] == 4) {
            uint64_t* res = new uint64_t[1];
            res[0] = N;
            uint64_t* result_temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
            cudaMemcpy(result_temp, res, sizeof(uint64_t), cudaMemcpyHostToDevice);
            CHECK_ERROR();
            cudaDeviceSynchronize();
            result[agg] = reinterpret_cast<uint8_t*> (result_temp);
        } else if (agg_mode[agg] == 5) {
            uint64_t* result_temp = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
            cudaMemset(result_temp, 0, sizeof(uint64_t));
            CHECK_ERROR();
            cudaDeviceSynchronize();
            result[agg] = reinterpret_cast<uint8_t*> (result_temp);
        } else if (agg_mode[agg] == 6) {
            T* result_temp = gpuBufferManager->customCudaMalloc<T>(1, 0, 0);
            cudaMemcpy(result_temp, reinterpret_cast<T*>(a[agg]), sizeof(T), cudaMemcpyDeviceToDevice);
            CHECK_ERROR();
            cudaDeviceSynchronize();
            result[agg] = reinterpret_cast<uint8_t*> (result_temp);
        } else if (agg_mode[agg] >= 0 && agg_mode[agg] <= 3) {
            T* a_temp = reinterpret_cast<T*> (a[agg]);
            T* result_temp = gpuBufferManager->customCudaMalloc<T>(1, 0, 0);

            if (agg_mode[agg] == 0) cudaMemset(result_temp, 0, sizeof(T));
            else if (agg_mode[agg] == 1) cudaMemset(result_temp, 0, sizeof(T));
            else if (agg_mode[agg] == 2) cudaMemcpy(result_temp, a_temp, sizeof(T), cudaMemcpyDeviceToDevice);
            else if (agg_mode[agg] == 3) cudaMemcpy(result_temp, a_temp, sizeof(T), cudaMemcpyDeviceToDevice);

            ungrouped_aggregate<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
                a_temp, result_temp, N, agg_mode[agg]);
            CHECK_ERROR();
            cudaDeviceSynchronize();

            if (agg_mode[agg] == 1) {
                //Currently typename T has to be a double to be here
                T* result_host_temp = new T[1];
                cudaMemcpy(result_host_temp, result_temp, sizeof(T), cudaMemcpyDeviceToHost);
                T avg = result_host_temp[0] / N;
                // printf("Result: %.2f and N: %d\n", result_host_temp[0], N);
                cudaMemcpy(result_temp, &avg, sizeof(T), cudaMemcpyHostToDevice);
                CHECK_ERROR();
                cudaDeviceSynchronize();
            } 
            // else {
                // T* result_host_temp = new T[1];
                // cudaMemcpy(result_host_temp, result_temp, sizeof(T), cudaMemcpyDeviceToHost);
                // CHECK_ERROR();
                // cudaDeviceSynchronize();
                // printf("Result: %.2f and N: %d\n", result_host_temp[0], N);
                // printf("Result: %ld and N: %d\n", reinterpret_cast<uint64_t*>(result_host_temp)[0], N);
            // }
            result[agg] = reinterpret_cast<uint8_t*> (result_temp);
        } else {
            printf("Unsupported aggregation mode\n");
            return;
        }
    }

    for (int agg = 0; agg < num_aggregates; agg++) {
        if (agg_mode[agg] >= 0 && agg_mode[agg] <= 3) {
            gpuBufferManager->customCudaFree<T>(reinterpret_cast<T*>(a[agg]), N, 0);
        }
    }
}

template
void ungroupedAggregate<int>(uint8_t **a, uint8_t **result, uint64_t N, int* agg_mode, int num_aggregates);
template
void ungroupedAggregate<uint64_t>(uint8_t **a, uint8_t **result, uint64_t N, int* agg_mode, int num_aggregates);
template
void ungroupedAggregate<double>(uint8_t **a, uint8_t **result, uint64_t N, int* agg_mode, int num_aggregates);

} // namespace duckdb