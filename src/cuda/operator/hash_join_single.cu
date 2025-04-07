#include "cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

__device__ uint64_t hash64(uint64_t key1, uint64_t key2) {
    uint64_t h = key1 * 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    h ^= key2 * 0xc6a4a7935bd1e995ull;
    h *= 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    return h;
}

__device__ uint64_t hash32(int32_t k1, int32_t k2) {
    uint64_t x1 = (uint64_t)k1;
    uint64_t x2 = (uint64_t)k2;
    uint64_t h = x1 * 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    h ^= x2 * 0xc6a4a7935bd1e995ull;
    h *= 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    return h;
}


template <typename T, int B, int I>
__global__ void probe_right_semi_anti_single_t(T **keys, unsigned long long* ht, uint64_t ht_len,
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
                T k0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = (int32_t)k0 % ht_len;
                } else {
                    slot = ((uint64_t)k0) % ht_len;
                }
            } else if (equal_keys == 2) {
                T k0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                T k1 = keys[1][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = hash32((int32_t)k0, (int32_t)k1) % ht_len;
                } else {
                    slot = hash64((uint64_t)k0, (uint64_t)k1) % ht_len;
                }
            } else {
                cudaAssert(0);
            }
            
            while (ht[slot * (num_keys + 2)] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    T item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    //(uint64_t)
                    if (condition_mode[n] == 0 && ht[slot * (num_keys + 2) + n] != item) local_found = 0;
                    else if (condition_mode[n] == 1 && ht[slot * (num_keys + 2) + n] == item) local_found = 0;
                }
                if (local_found) {
                    ht[slot * (num_keys + 2) + num_keys + 1] = tile_offset + threadIdx.x + ITEM * B;
                    break;
                }
                slot = (slot + 100007) % ht_len;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void probe_single_match_t(T **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *row_ids_left, uint64_t *row_ids_right, unsigned long long* count, 
            uint64_t N, int* condition_mode, int num_keys, int equal_keys, int join_mode, bool is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    int items_off[I];
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

    // int n_ht_column = num_keys + 1;
    int n_ht_column;
    if (join_mode == 3) n_ht_column = num_keys + 2;
    else n_ht_column = num_keys + 1;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            
            uint64_t slot;
            if (equal_keys == 1) {
                T k0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = (int32_t)k0 % ht_len;
                } else {
                    slot = ((uint64_t)k0) % ht_len;
                }
            } else if (equal_keys == 2) {
                T k0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                T k1 = keys[1][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = hash32((int32_t)k0, (int32_t)k1) % ht_len;
                } else {
                    slot = hash64((uint64_t)k0, (uint64_t)k1) % ht_len;
                }
            } else {
                cudaAssert(0);
            }
            
            bool found = 0;
            while (ht[slot * n_ht_column] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    T item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0 && ht[slot * n_ht_column  + n] != item) local_found = 0;
                    else if (condition_mode[n] == 1 && ht[slot * n_ht_column + n] == item) local_found = 0;
                }
                if (local_found) {
                    items_off[ITEM] = ht[slot * n_ht_column + num_keys];
                    found = 1;
                    break;
                }
                slot = (slot + 100007) % ht_len;
            }

            if (join_mode == 2) { // anti join
                if (!found) {
                    t_count++;
                    selection_flags[ITEM] = 1;
                }
            } else {
                if (found) {
                    if (join_mode == 3) ht[slot * (num_keys + 2) + num_keys + 1] = tile_offset + threadIdx.x + ITEM * B;
                    t_count++;
                    selection_flags[ITEM] = 1;
                }
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
                if (join_mode == 0 || join_mode == 3) { // inner join and right join
                    row_ids_right[offset] = items_off[ITEM];
                    row_ids_left[offset] = tile_offset + threadIdx.x + ITEM * B;
                } else if (join_mode == 1 || join_mode == 2) { // semi join and anti join
                    row_ids_left[offset] = tile_offset + threadIdx.x + ITEM * B;
                } else {
                    cudaAssert(0);
                }
            }
        }
    }

}

template <typename T, int B, int I>
__global__ void probe_mark_t(T **keys, unsigned long long* ht, uint64_t ht_len, uint8_t* output,
            uint64_t N, int* condition_mode, int num_keys, int equal_keys) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    int n_ht_column = num_keys + 1;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            
            uint64_t slot;
            if (equal_keys == 1) {
                T k0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = (int32_t)k0 % ht_len;
                } else {
                    slot = ((uint64_t)k0) % ht_len;
                }
            } else if (equal_keys == 2) {
                T k0 = keys[0][tile_offset + threadIdx.x + (ITEM * B)];
                T k1 = keys[1][tile_offset + threadIdx.x + (ITEM * B)];
                if (sizeof(T) == 4) {
                    slot = hash32((int32_t)k0, (int32_t)k1) % ht_len;
                } else {
                    slot = hash64((uint64_t)k0, (uint64_t)k1) % ht_len;
                }
            } else {
                cudaAssert(0);
            }
            
            bool found = 0;
            while (ht[slot * n_ht_column] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    T item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0 && ht[slot * n_ht_column  + n] != item) local_found = 0;
                    else if (condition_mode[n] == 1 && ht[slot * n_ht_column  + n] == item) local_found = 0;
                }
                if (local_found) {
                    found = 1;
                    break;
                }
                slot = (slot + 100007) % ht_len;
            }

            output[tile_offset + threadIdx.x + ITEM * B] = found;
        }
    }
}

template __global__ void probe_right_semi_anti_single_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint64_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint64_t N, int* condition_mode, int num_keys, int equal_keys
);
template __global__ void probe_right_semi_anti_single_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    int32_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint64_t N, int* condition_mode, int num_keys, int equal_keys
);

template __global__ void probe_single_match_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint64_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint64_t *row_ids_left, uint64_t *row_ids_right,
    unsigned long long* count, uint64_t N, int* condition_mode,
    int num_keys, int equal_keys, int join_mode, bool is_count
);
template __global__ void probe_single_match_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    int32_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint64_t *row_ids_left, uint64_t *row_ids_right,
    unsigned long long* count, uint64_t N, int* condition_mode,
    int num_keys, int equal_keys, int join_mode, bool is_count
);

template __global__ void probe_mark_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint64_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint8_t* output, uint64_t N, int* condition_mode,
    int num_keys, int equal_keys
);
template __global__ void probe_mark_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    int32_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint8_t* output, uint64_t N, int* condition_mode,
    int num_keys, int equal_keys
);


void probeHashTableSingleMatch(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
            uint64_t* &count, uint64_t N, int* condition_mode, int num_keys, int join_mode, bool is_32_bit) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel Unique Join\n");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudaMemset(count, 0, sizeof(uint64_t));

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

        probe_single_match_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len,
                row_ids_left, row_ids_right, (unsigned long long*) count,
                N, condition_mode_dev, num_keys, equal_keys, join_mode,
                true
            );
        cudaDeviceSynchronize();

        uint64_t* h_count = new uint64_t[1];
        cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        assert(h_count[0] > 0);
        row_ids_left = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
        if (join_mode == 0 || join_mode == 3) {
            row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
        }

        cudaMemset(count, 0, sizeof(uint64_t));
        probe_single_match_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len,
                row_ids_left, row_ids_right, (unsigned long long*) count,
                N, condition_mode_dev, num_keys, equal_keys, join_mode,
                false
            );
        cudaDeviceSynchronize();

        printf("Count: %lu\n", h_count[0]);
        count = h_count;
    } else { //64 bit
        uint64_t** keys_data = new uint64_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<uint64_t*>(keys[i]);
        }
        uint64_t** dev_keys;
        cudaMalloc(&dev_keys, num_keys * sizeof(uint64_t*));
        cudaMemcpy(dev_keys, keys_data, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

        probe_single_match_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len,
                row_ids_left, row_ids_right, (unsigned long long*) count,
                N, condition_mode_dev, num_keys, equal_keys, join_mode,
                true
            );
        cudaDeviceSynchronize();

        uint64_t* h_count = new uint64_t[1];
        cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        assert(h_count[0] > 0);
        row_ids_left = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
        if (join_mode == 0 || join_mode == 3) {
            row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
        }

        cudaMemset(count, 0, sizeof(uint64_t));
        probe_single_match_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len,
                row_ids_left, row_ids_right, (unsigned long long*) count,
                N, condition_mode_dev, num_keys, equal_keys, join_mode,
                false
            );
        cudaDeviceSynchronize();

        printf("Count: %lu\n", h_count[0]);
        count = h_count;
    }
    STOP_TIMER();
}

void probeHashTableRightSemiAntiSingleMatch(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int* condition_mode, int num_keys, bool is_32_bit) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel Unique Join\n");
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

        probe_right_semi_anti_single_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len, N, condition_mode_dev,
                num_keys, equal_keys
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

        probe_right_semi_anti_single_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len, N, condition_mode_dev,
                num_keys, equal_keys
            );
        cudaDeviceSynchronize();
    }

    printf("Finished probe right\n");
    STOP_TIMER();
}

void probeHashTableMark(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint8_t* &output, uint64_t N, int* condition_mode, int num_keys, bool is_32_bit) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel Mark\n");
    SETUP_TIMING();
    START_TIMER();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    int equal_keys = 0;
    for (int idx = 0; idx < num_keys; idx++) {
        if (condition_mode[idx] == 0) equal_keys++;
    }

    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0);
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);
    output = gpuBufferManager->customCudaMalloc<uint8_t>(N, 0, 0);

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    if (is_32_bit) {
        int32_t** keys_data = new int32_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<int*>(keys[i]);
        }
        int32_t** dev_keys;
        cudaMalloc(&dev_keys, num_keys * sizeof(int32_t*));
        cudaMemcpy(dev_keys, keys_data, num_keys * sizeof(int32_t*), cudaMemcpyHostToDevice);

        probe_mark_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len, output, N,
                condition_mode_dev, num_keys, equal_keys
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

        probe_mark_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                dev_keys, ht, ht_len, output, N,
                condition_mode_dev, num_keys, equal_keys
            );
        cudaDeviceSynchronize();

    }
    STOP_TIMER();
}

}