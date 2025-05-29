#include "cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

__device__ uint64_t hash64_right(uint64_t key1, uint64_t key2) {
    uint64_t h = key1 * 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    h ^= key2 * 0xc6a4a7935bd1e995ull;
    h *= 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    return h;
}

template <int B, int I>
__global__ void probe_right_semi_anti(uint64_t **keys, unsigned long long* ht, uint64_t ht_len,
            uint64_t N, int* condition_mode, int num_keys, int equal_keys) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            
            uint64_t slot;
            if (equal_keys == 1) slot = keys[0][tile_offset + threadIdx.x + ITEM * B] % ht_len;
            else if (equal_keys == 2) slot = hash64_right(keys[0][tile_offset + threadIdx.x + ITEM * B], keys[1][tile_offset + threadIdx.x + ITEM * B]) % ht_len;
            else cudaAssert(0);
            
            while (ht[slot * (num_keys + 2)] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    uint64_t item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0 && ht[slot * (num_keys + 2) + n] != item) local_found = 0;
                    else if (condition_mode[n] == 1 && ht[slot * (num_keys + 2) + n] == item) local_found = 0;
                }
                if (local_found) {
                    ht[slot * (num_keys + 2) + num_keys + 1] = tile_offset + threadIdx.x + ITEM * B;
                }
                slot = (slot + 100007) % ht_len;
            }
        }
    }
}

template <int B, int I>
__global__ void scan_right(unsigned long long* ht, unsigned long long* count, uint64_t ht_len, 
                uint64_t *row_ids, uint64_t num_keys, int join_mode, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    uint64_t items_off[I];
    int selection_flags[I];
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (ht_len + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0; // Number of items selected per thread
    int c_t_count = 0; //Prefix sum of t_count
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = ht_len - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            uint64_t slot = tile_offset + threadIdx.x + ITEM * B;  
            if (join_mode == 0) { // semi join
                if (ht[slot * (num_keys + 2) + num_keys + 1] != 0xFFFFFFFFFFFFFFFF) {
                    items_off[ITEM] = ht[slot * (num_keys + 2) + num_keys];
                    selection_flags[ITEM] = 1;
                    t_count++;
                }
            } else if (join_mode == 1) { // anti join
                if (ht[slot * (num_keys + 2) + num_keys + 1] == 0xFFFFFFFFFFFFFFFF && ht[slot * (num_keys + 2) + num_keys] != 0xFFFFFFFFFFFFFFFF) {
                    items_off[ITEM] = ht[slot * (num_keys + 2) + num_keys];
                    selection_flags[ITEM] = 1;
                    t_count++;
                }
            } else {
                cudaAssert(0);
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
                row_ids[offset] = items_off[ITEM];
            }
        }
    }

}

template
__global__ void scan_right<BLOCK_THREADS, ITEMS_PER_THREAD>(unsigned long long* ht, unsigned long long* count, uint64_t ht_len, 
                uint64_t *row_ids, uint64_t num_keys, int join_mode, int is_count);

template
__global__ void probe_right_semi_anti<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys, int equal_keys);

void scanHashTableRight(unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids, uint64_t* &count, int join_mode, int num_keys) {
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    if (ht_len == 0) {
        uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        h_count[0] = 0;
        count = h_count;
        SIRIUS_LOG_DEBUG("N is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Scan Kernel");
    SETUP_TIMING();
    START_TIMER();
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_ERROR();
    scan_right<BLOCK_THREADS, ITEMS_PER_THREAD><<<(ht_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(ht, (unsigned long long*) count, ht_len, row_ids, num_keys, join_mode, 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(h_count[0] > 0);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    scan_right<BLOCK_THREADS, ITEMS_PER_THREAD><<<(ht_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(ht, (unsigned long long*) count, ht_len, row_ids, num_keys, join_mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    SIRIUS_LOG_DEBUG("Scan Count: {}", h_count[0]);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;

    // thrust::device_vector<uint64_t> sorted_keys(row_ids, row_ids + h_count[0]);
    // thrust::sort(thrust::device, sorted_keys.begin(), sorted_keys.end());
    // uint64_t* raw_row_ids = thrust::raw_pointer_cast(sorted_keys.data());
    // testprint<<<1, 1>>>(raw_row_ids, h_count[0]);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(ht), 0);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    STOP_TIMER();
}

void probeHashTableRightSemiAnti(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys) {
    CHECK_ERROR();
    if (N == 0) {
        SIRIUS_LOG_DEBUG("N is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Probe Kernel");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //reinterpret cast the keys to uint64_t
    uint64_t** keys_data = new uint64_t*[num_keys];
    for (int idx = 0; idx < num_keys; idx++) {
        keys_data[idx] = reinterpret_cast<uint64_t*>(keys[idx]);
    }

    uint64_t** keys_dev;
    cudaMalloc((void**) &keys_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(keys_dev, keys_data, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    int equal_keys = 0;
    for (int idx = 0; idx < num_keys; idx++) {
        if (condition_mode[idx] == 0) equal_keys++;
    }

    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0);
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    probe_right_semi_anti<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, ht, ht_len, N, condition_mode_dev, num_keys, equal_keys);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    SIRIUS_LOG_DEBUG("Finished probe right");
    STOP_TIMER();
    cudaFree(keys_dev);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(condition_mode_dev), 0);
}

} // namespace duckdb