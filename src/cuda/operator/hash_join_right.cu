#include "cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

__device__ uint64_t hash64_right(uint64_t key1, uint64_t key2) {
    uint64_t h = key1 * 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    h ^= key2 * 0xc6a4a7935bd1e995ull;
    h *= 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    return h;
}

__device__ uint64_t hash32_right(int32_t key1, int32_t key2) {
    uint64_t k1 = (uint64_t)key1;
    uint64_t k2 = (uint64_t)key2;
    uint64_t h = k1 * 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    h ^= k2 * 0xc6a4a7935bd1e995ull;
    h *= 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    return h;
}

template <typename T, int B, int I>
__global__ void probe_right_semi_anti_t(T **keys, unsigned long long* ht, uint64_t ht_len,
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
            if (equal_keys == 1) {
                T key0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = (int32_t)key0 % ht_len; 
                } else {
                    slot = ((uint64_t)key0) % ht_len;
                }
            } else if (equal_keys == 2) {
                T k0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                T k1 = keys[1][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = hash32_right((int)k0, (int)k1) % ht_len;
                } else {
                    slot = hash64_right((uint64_t)k0, (uint64_t)k1) % ht_len;
                }
            } else {
                cudaAssert(0);
            }
            
            while (ht[slot * (num_keys + 2)] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    T item = keys[n][tile_offset + threadIdx.x + ITEM * B];
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
            if (join_mode == 0) { //semi join
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

template __global__ void probe_right_semi_anti_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N,
    int* condition_mode, int num_keys, int equal_keys
);

template __global__ void probe_right_semi_anti_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    int32_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N,
    int* condition_mode, int num_keys, int equal_keys
);

void scanHashTableRight(unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids, uint64_t* &count, int join_mode, int num_keys) {
    CHECK_ERROR();
    if (ht_len == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Scan Kernel\n");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudaMemset(count, 0, sizeof(uint64_t));

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_ERROR();
    scan_right<BLOCK_THREADS, ITEMS_PER_THREAD><<<(ht_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(ht, (unsigned long long*) count, ht_len, row_ids, num_keys, join_mode, 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    uint64_t* h_count = new uint64_t [1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(h_count[0] > 0);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    scan_right<BLOCK_THREADS, ITEMS_PER_THREAD><<<(ht_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(ht, (unsigned long long*) count, ht_len, row_ids, num_keys, join_mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Scan Count: %lu\n", h_count[0]);
    count = h_count;

    // thrust::device_vector<uint64_t> sorted_keys(row_ids, row_ids + h_count[0]);
    // thrust::sort(thrust::device, sorted_keys.begin(), sorted_keys.end());
    // uint64_t* raw_row_ids = thrust::raw_pointer_cast(sorted_keys.data());
    // testprint<<<1, 1>>>(raw_row_ids, h_count[0]);

    CHECK_ERROR();
    cudaDeviceSynchronize();
    STOP_TIMER();
}

void probeHashTableRightSemiAnti(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys, bool is_32_bit) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel\n");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    int equal_keys = 0;
    for (int idx = 0; idx < num_keys; idx++) {
        if (condition_mode[idx] == 0) equal_keys++;
    }

    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0);
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

    if (is_32_bit) {
        int32_t** keys_data = new int32_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<int32_t*>(keys[i]);
        }
        int32_t** dev_keys;
        cudaMalloc(&dev_keys, num_keys * sizeof(int32_t*));
        cudaMemcpy(dev_keys, keys_data, num_keys * sizeof(int32_t*), cudaMemcpyHostToDevice);

        probe_right_semi_anti_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len, N, condition_mode_dev, num_keys, equal_keys
            );

        cudaDeviceSynchronize();
    } else {
        uint64_t** keys_data = new uint64_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<uint64_t*>(keys[i]);
        }
        uint64_t** dev_keys;
        cudaMalloc(&dev_keys, num_keys * sizeof(uint64_t*));
        cudaMemcpy(dev_keys, keys_data, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

        probe_right_semi_anti_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len, N, condition_mode_dev, num_keys, equal_keys
            );

        cudaDeviceSynchronize();
    }

    printf("Finished probe right\n");
    STOP_TIMER();
}

} // namespace duckdb