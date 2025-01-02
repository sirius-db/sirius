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

template <int B, int I>
__global__ void probe_single_match(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *row_ids_left, uint64_t *row_ids_right, unsigned long long* count, 
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

    int n_ht_column = num_keys + 1;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            
            uint64_t slot;
            if (equal_keys == 1) slot = keys[0][tile_offset + threadIdx.x + ITEM * B] % ht_len;
            else if (equal_keys == 2) slot = hash64(keys[0][tile_offset + threadIdx.x + ITEM * B], keys[1][tile_offset + threadIdx.x + ITEM * B]) % ht_len;
            else cudaAssert(0);
            
            bool found = 0;
            while (ht[slot * n_ht_column] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    uint64_t item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0 && ht[slot * n_ht_column + n] != item) local_found = 0;
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
                if (join_mode == 0) { // inner join
                    row_ids_right[offset] = items_off[ITEM];
                    row_ids_left[offset] = tile_offset + threadIdx.x + ITEM * B;
                } else if (join_mode == 1 || join_mode == 2) { // semi join and anti join
                    row_ids_left[offset] = tile_offset + threadIdx.x + ITEM * B;
                    // printf("row_ids_left[%lu] = %lu\n", offset, row_ids_left[offset]);
                } else {
                    cudaAssert(0);
                }
            }
        }
    }

}

template <int B, int I>
__global__ void probe_mark(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint8_t* output,
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
            if (equal_keys == 1) slot = keys[0][tile_offset + threadIdx.x + ITEM * B] % ht_len;
            else if (equal_keys == 2) slot = hash64(keys[0][tile_offset + threadIdx.x + ITEM * B], keys[1][tile_offset + threadIdx.x + ITEM * B]) % ht_len;
            else cudaAssert(0);
            
            bool found = 0;
            while (ht[slot * n_ht_column] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    uint64_t item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0 && ht[slot * n_ht_column + n] != item) local_found = 0;
                    else if (condition_mode[n] == 1 && ht[slot * n_ht_column + n] == item) local_found = 0;
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

__global__ void print_key2(uint64_t* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int x = N;
        if (N > 1000) x = 1000;
        for (uint64_t i = 0; i < x; i++) {
            printf("%ld ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void probe_single_match<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *row_ids_left, uint64_t *row_ids_right, unsigned long long* count, 
            uint64_t N, int* condition_mode, int num_keys, int equal_keys, int join_mode, bool is_count);

template
__global__ void probe_mark<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint8_t* output,
            uint64_t N, int* condition_mode, int num_keys, int equal_keys);


void probeHashTableSingleMatch(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, 
            uint64_t* &count, uint64_t N, int* condition_mode, int num_keys, int join_mode) {
    CHECK_ERROR();
    if (N == 0) {
        uint64_t* h_count = new uint64_t[1];
        h_count[0] = 0;
        count = h_count;
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudaMemset(count, 0, sizeof(uint64_t));

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

    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0).data_;
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_ERROR();
    probe_single_match<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, ht, ht_len, row_ids_left, row_ids_right, (unsigned long long*) count, 
            N, condition_mode_dev, num_keys, equal_keys, join_mode, 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    uint64_t* h_count = new uint64_t [1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(h_count[0] > 0);
    row_ids_left = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0).data_;
    if (join_mode == 0) row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0).data_;
    cudaMemset(count, 0, sizeof(uint64_t));
    probe_single_match<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, ht, ht_len, row_ids_left, row_ids_right, (unsigned long long*) count, 
            N, condition_mode_dev, num_keys, equal_keys, join_mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    // print_key2<<<1, 1>>>(row_ids_left, h_count[0]);
    printf("Count: %lu\n", h_count[0]);
    count = h_count;
}

void probeHashTableMark(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, uint8_t* &output, uint64_t N, int* condition_mode, int num_keys) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    //reinterpret cast the keys to uint64_t
    uint64_t** keys_data = new uint64_t*[num_keys];
    for (int idx = 0; idx < num_keys; idx++) {
        keys_data[idx] = reinterpret_cast<uint64_t*>(keys[idx]);
    }
    printf("Launching Probe Kernel\n");

    uint64_t** keys_dev;
    cudaMalloc((void**) &keys_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(keys_dev, keys_data, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    CHECK_ERROR();
    printf("Launching Probe Kernel\n");

    int equal_keys = 0;
    for (int idx = 0; idx < num_keys; idx++) {
        if (condition_mode[idx] == 0) equal_keys++;
    }
    printf("Launching Probe Kernel\n");

    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0).data_;
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);
    output = gpuBufferManager->customCudaMalloc<uint8_t>(N, 0, 0).data_;
    printf("Launching Probe Kernel\n");

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_ERROR();
    probe_mark<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, ht, ht_len, output, 
            N, condition_mode_dev, num_keys, equal_keys);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Launching Probe Kernel\n");
}

}