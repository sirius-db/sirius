#include "cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

__device__ uint64_t hash64_multikey(uint64_t key1, uint64_t key2) {
    uint64_t h = key1 * 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    h ^= key2 * 0xc6a4a7935bd1e995ull;
    h *= 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    return h;
}

__device__ uint64_t hash32_multikey(int32_t key1, int32_t key2) {
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
__global__ void probe_multikey_count_t(T **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *offset_each_thread,
        unsigned long long* total_count, uint64_t N, int* condition_mode, int num_keys, int equal_keys, bool is_right) {
    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    int items_count[I];

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
        items_count[ITEM] = 0;
    }

    int n_ht_column;
    if (is_right) n_ht_column = num_keys + 2;
    else n_ht_column = num_keys + 1;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {

            uint64_t slot;
            if (equal_keys == 1) {
                T k0 = keys[0][tile_offset + threadIdx.x + ITEM * B];
                if (sizeof(T) == 4) {
                    slot = (int32_t)(k0) % ht_len;
                } else {
                    slot = ((uint64_t)k0) % ht_len;
                }
            } else if (equal_keys == 2) {
                T k0 = keys[0][tile_offset + threadIdx.x + ITEM * B];
                T k1 = keys[1][tile_offset + threadIdx.x + ITEM * B];
                if (sizeof(T) == 4) {
                    slot = hash32_multikey((int32_t)k0, (int32_t)k1) % ht_len;
                } else {
                    slot = hash64_multikey((uint64_t)k0, (uint64_t)k1) % ht_len;
                }
            } else {
                cudaAssert(0);
            }

            // if (keys[0][tile_offset + threadIdx.x + ITEM * B] == 4701966275692616012 || keys[0][tile_offset + threadIdx.x + ITEM * B] == 4701966275692616011) {
            //     printf("key found %ld\n", tile_offset + threadIdx.x + ITEM * B);
            // }

            // if (tile_offset + threadIdx.x + ITEM * B == 69997) {
            //     printf("key %ld\n", keys[0][tile_offset + threadIdx.x + ITEM * B]);
            // }

            while (ht[slot * n_ht_column] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                // printf("key1: %lu key2: %lu ht1: %lu ht2: %lu\n", keys[0][tile_offset + threadIdx.x + ITEM * B], keys[1][tile_offset + threadIdx.x + ITEM * B], ht[slot * n_ht_column], ht[slot * n_ht_column + 1]);
                for (int n = 0; n < num_keys; n++) {
                    T item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0 && item != ht[slot * n_ht_column + n]) {
                            local_found = 0;  
                    } else if (condition_mode[n] == 1 && item == ht[slot * n_ht_column + n]) {
                            local_found = 0;
                            // printf("key1: %lu key2: %lu ht1: %lu ht2: %lu\n", keys[0][tile_offset + threadIdx.x + ITEM * B], keys[1][tile_offset + threadIdx.x + ITEM * B], ht[slot * n_ht_column], ht[slot * n_ht_column + 1]);
                            // break;
                    }
                }
                if (local_found) {
                    items_count[ITEM]++;
                }
                slot = (slot + 100007) % ht_len;
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

    if (blockIdx.x * tile_size + threadIdx.x < N) {
        offset_each_thread[blockIdx.x * B + threadIdx.x] = block_off + c_t_count;
    }
}

template <typename T, int B, int I>
__global__ void probe_multikey_t(
    T **keys,
    unsigned long long* ht,
    uint64_t ht_len,
    uint64_t *offset_each_thread,
    uint64_t *row_ids_left,
    uint64_t *row_ids_right,
    uint64_t N,
    int* condition_mode,
    int num_keys,
    int equal_keys,
    bool is_right
) {
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;
    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    int n_ht_column;
    if (is_right) n_ht_column = num_keys + 2;
    else n_ht_column = num_keys + 1;

    uint64_t output_offset = 0;
    if (blockIdx.x * tile_size + threadIdx.x < N) {
        output_offset = offset_each_thread[blockIdx.x * B + threadIdx.x];
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t slot;
            if (equal_keys == 1) {
                T k0 = keys[0][tile_offset + threadIdx.x + ITEM * B];
                if (sizeof(T) == 4) {
                    slot = (int32_t)(k0) % ht_len;
                } else {
                    slot = ((uint64_t)k0) % ht_len;
                }
            } else if (equal_keys == 2) {
                T k0 = keys[0][tile_offset + threadIdx.x + ITEM * B];
                T k1 = keys[1][tile_offset + threadIdx.x + ITEM * B];
                if (sizeof(T) == 4) {
                    slot = hash32_multikey((int32_t)k0, (int32_t)k1) % ht_len;
                } else {
                    slot = hash64_multikey((uint64_t)k0, (uint64_t)k1) % ht_len;
                }
            } else {
                cudaAssert(0);
            }
            while (ht[slot * n_ht_column] != 0xFFFFFFFFFFFFFFFFULL) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    T item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0) {
                        if ((uint64_t)item != ht[slot * n_ht_column + n]) {
                            local_found = 0;
                            // break;
                        }
                    } else if (condition_mode[n] == 1) {
                        if ((uint64_t)item == ht[slot * n_ht_column + n]) {
                            local_found = 0;
                            // break;
                        }
                    }
                }
                if (local_found) {
                    row_ids_right[output_offset] = ht[slot * n_ht_column + num_keys];
                    row_ids_left[output_offset]  = tile_offset + threadIdx.x + ITEM * B;
                    if (is_right) ht[slot * n_ht_column + num_keys + 1] = tile_offset + threadIdx.x + ITEM * B;
                    
                    output_offset++;
                }
                slot = (slot + 100007) % ht_len;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void build_multikey_t(
    T **keys,
    unsigned long long* ht,
    uint64_t ht_len,
    uint64_t N,
    int num_keys,
    int equal_keys,
    bool is_right
) {
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;
    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    int n_ht_column;
    if (is_right) n_ht_column = num_keys + 2;
    else  n_ht_column = num_keys + 1;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            uint64_t slot;
            if (equal_keys == 1) {
                T k0 = keys[0][tile_offset + threadIdx.x + ITEM * B];
                if (sizeof(T) == 4) {
                    slot = (int32_t)(k0) % ht_len;
                } else {
                    slot = ((uint64_t)k0) % ht_len;
                }
            } else if (equal_keys == 2) {
                T k0 = keys[0][tile_offset + threadIdx.x + ITEM * B];
                T k1 = keys[1][tile_offset + threadIdx.x + ITEM * B];
                if (sizeof(T) == 4) {
                    slot = hash32_multikey((int32_t)k0, (int32_t)k1) % ht_len;
                } else {
                    slot = hash64_multikey((uint64_t)k0, (uint64_t)k1) % ht_len;
                }
            } else {
                cudaAssert(0);
            }

            T item = keys[0][tile_offset + threadIdx.x + ITEM * B];
            while (atomicCAS(&ht[slot * n_ht_column], 0xFFFFFFFFFFFFFFFFULL,
                            (unsigned long long)( (uint64_t)item ))
                != 0xFFFFFFFFFFFFFFFFULL)
            {
            slot = (slot + 100007ULL) % ht_len;
            }
            for (int n = 1; n < num_keys; n++) {
                ht[slot * n_ht_column + n] = keys[n][tile_offset + threadIdx.x + ITEM * B];
            }
            ht[slot * n_ht_column + num_keys] = tile_offset + threadIdx.x + ITEM * B;
        }
    }
}

template __global__ void build_multikey_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N,
    int num_keys, int equal_keys, bool is_right
);

template __global__ void build_multikey_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    int32_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N,
    int num_keys, int equal_keys, bool is_right
);

template __global__ void probe_multikey_count_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *offset_each_thread,
    unsigned long long* total_count, uint64_t N, int* condition_mode,
    int num_keys, int equal_keys, bool is_right
);

template __global__ void probe_multikey_count_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    int32_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *offset_each_thread,
    unsigned long long* total_count, uint64_t N, int* condition_mode,
    int num_keys, int equal_keys, bool is_right
);

template __global__ void probe_multikey_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint64_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint64_t *offset_each_thread, uint64_t *row_ids_left, uint64_t *row_ids_right,
    uint64_t N, int* condition_mode, int num_keys, int equal_keys, bool is_right
);

template __global__ void probe_multikey_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>(
    int32_t **keys, unsigned long long* ht, uint64_t ht_len,
    uint64_t *offset_each_thread, uint64_t *row_ids_left, uint64_t *row_ids_right,
    uint64_t N, int* condition_mode, int num_keys, int equal_keys, bool is_right
);

void buildHashTableTempl(
    uint8_t **keys,
    unsigned long long* ht,
    uint64_t ht_len,
    uint64_t N,
    int* condition_mode,
    int num_keys,
    bool is_right,
    bool is_32_bit  // added int 32 parameter
) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Build Kernel\n");
    SETUP_TIMING();
    START_TIMER();
    printf("N: %lu ht_len: %ld\n", N, ht_len);

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0);
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);

    int equal_keys = 0;
    for (int idx = 0; idx < num_keys; idx++) {
        if (condition_mode[idx] == 0) equal_keys++;
    }

    if (is_right) cudaMemset(ht, 0xFF, ht_len * (num_keys + 2) * sizeof(unsigned long long));
    else cudaMemset(ht, 0xFF, ht_len * (num_keys + 1) * sizeof(unsigned long long));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

    if (is_32_bit) {
        int32_t** keys_data = new int32_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<int32_t*>(keys[i]);
        }
        int32_t** keys_dev;
        cudaMalloc(&keys_dev, num_keys * sizeof(int32_t*));
        cudaMemcpy(keys_dev, keys_data, num_keys * sizeof(int32_t*), cudaMemcpyHostToDevice);

        build_multikey_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                keys_dev, ht, ht_len, N, num_keys, equal_keys, is_right
            );
        cudaDeviceSynchronize();
    } else {
        uint64_t** keys_data = new uint64_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<uint64_t*>(keys[i]);
        }
        uint64_t** keys_dev;
        cudaMalloc(&keys_dev, num_keys * sizeof(uint64_t*));
        cudaMemcpy(keys_dev, keys_data, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

        build_multikey_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                keys_dev, ht, ht_len, N, num_keys, equal_keys, is_right
            );
        cudaDeviceSynchronize();
    }

    STOP_TIMER();
}

void probeHashTableTempl(
    uint8_t **keys,
    unsigned long long* ht,
    uint64_t ht_len,
    uint64_t*& row_ids_left,
    uint64_t*& row_ids_right,
    uint64_t*& count,
    uint64_t N,
    int* condition_mode,
    int num_keys,
    bool is_right,
    bool is_32_bit  
) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel\n");
    SETUP_TIMING();
    START_TIMER();
    printf("N: %lu\n", N);

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudaMemset(count, 0, sizeof(uint64_t));

    uint64_t* offset_each_thread = gpuBufferManager->customCudaMalloc<uint64_t>(((N + tile_items - 1)/tile_items) * BLOCK_THREADS, 0, 0);

    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0);
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);

    int equal_keys = 0;
    for (int i = 0; i < num_keys; i++) {
        if (condition_mode[i] == 0) equal_keys++;
    }

    if (is_32_bit) {
        int32_t** keys_data = new int32_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<int32_t*>(keys[i]);
        }
        int32_t** keys_dev;
        cudaMalloc(&keys_dev, num_keys * sizeof(int32_t*));
        cudaMemcpy(keys_dev, keys_data, num_keys * sizeof(int32_t*), cudaMemcpyHostToDevice);

        probe_multikey_count_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                keys_dev, ht, ht_len,
                offset_each_thread, (unsigned long long*)count,
                N, condition_mode_dev, num_keys, equal_keys, is_right
            );
        CHECK_ERROR();
        cudaDeviceSynchronize();

        uint64_t* h_count = new uint64_t[1];
        cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        assert(h_count[0] > 0);
        printf("Count: %lu\n", h_count[0]);
        row_ids_left  = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
        row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
        probe_multikey_t<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                keys_dev, ht, ht_len,
                offset_each_thread,
                row_ids_left, row_ids_right,
                N, condition_mode_dev, num_keys, equal_keys, is_right
            );
        CHECK_ERROR();
        cudaDeviceSynchronize();
        count = h_count;
    } else {
        uint64_t** keys_data = new uint64_t*[num_keys];
        for (int i = 0; i < num_keys; i++) {
            keys_data[i] = reinterpret_cast<uint64_t*>(keys[i]);
        }
        uint64_t** keys_dev;
        cudaMalloc(&keys_dev, num_keys * sizeof(uint64_t*));
        cudaMemcpy(keys_dev, keys_data, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

        probe_multikey_count_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                keys_dev, ht, ht_len,
                offset_each_thread, (unsigned long long*)count,
                N, condition_mode_dev, num_keys, equal_keys, is_right
            );
        CHECK_ERROR();
        cudaDeviceSynchronize();

        uint64_t* h_count = new uint64_t[1];
        cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        assert(h_count[0] > 0);
        printf("Count: %lu\n", h_count[0]);

        row_ids_left  = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
        row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);

        probe_multikey_t<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>
            <<< (N + tile_items - 1)/tile_items, BLOCK_THREADS >>>(
                keys_dev, ht, ht_len,
                offset_each_thread,
                row_ids_left, row_ids_right,
                N, condition_mode_dev, num_keys, equal_keys, is_right
            );
        CHECK_ERROR();
        cudaDeviceSynchronize();
        count = h_count;
    }

    STOP_TIMER();
}


void buildHashTable(uint8_t **keys,
                    unsigned long long* ht,
                    uint64_t ht_len,
                    uint64_t N,
                    int* condition_mode,
                    int num_keys,
                    bool is_right,
                    bool is_32_bit)
{
    buildHashTableTempl(keys, ht, ht_len, N, condition_mode, num_keys, is_right, is_32_bit);
}

void probeHashTable(uint8_t **keys,
                    unsigned long long* ht,
                    uint64_t ht_len,
                    uint64_t* &row_ids_left,
                    uint64_t* &row_ids_right,
                    uint64_t* &count,
                    uint64_t N,
                    int* condition_mode,
                    int num_keys,
                    bool is_right,
                    bool is_32_bit)
{
    probeHashTableTempl(keys, ht, ht_len,
                        row_ids_left, row_ids_right, count,
                        N, condition_mode,
                        num_keys, is_right,
                        is_32_bit);
}

} // namespace duckdb
