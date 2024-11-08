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

//TODO: currently this probe does not support many to many join, for many to many join, we use cuCollections
template <int B, int I>
__global__ void probe_multikey(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *row_ids_left, uint64_t *row_ids_right, 
            unsigned long long* count, uint64_t N, int* condition_mode, int num_keys, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    // uint64_t items_key[I];
    uint64_t items_off[I];
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
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            
            uint64_t slot;
            if (num_keys == 1) slot = keys[0][tile_offset + threadIdx.x + ITEM * B] % ht_len;
            else if (num_keys == 2) slot = hash64_multikey(keys[0][tile_offset + threadIdx.x + ITEM * B], keys[1][tile_offset + threadIdx.x + ITEM * B]) % ht_len;
            else cudaAssert(0);
            
            bool found = 0;
            while (ht[slot * (num_keys + 1)] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                for (int n = 0; n < num_keys; n++) {
                    uint64_t item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    if (condition_mode[n] == 0 && ht[slot * (num_keys + 1) + n] != item) local_found = 0;
                    else if (condition_mode[n] == 1 && ht[slot * (num_keys + 1) + n] == item) local_found = 0;
                    else cudaAssert(0);
                }
                if (local_found) {
                    items_off[ITEM] = ht[slot * (num_keys + 1) + num_keys];
                    found = 1;
                    break;
                }
                slot = (slot + 100007) % ht_len;
            }
            if (found) {
                selection_flags[ITEM] = 1;
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
                row_ids_right[offset] = items_off[ITEM];
                row_ids_left[offset] = tile_offset + threadIdx.x + ITEM * B;
            }
        }
    }
}

template <int B, int I>
__global__ void build_multikey(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int num_keys) {

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
            if (num_keys == 1) slot = keys[0][tile_offset + threadIdx.x + ITEM * B] % ht_len;
            else if (num_keys == 2) slot = hash64_multikey(keys[0][tile_offset + threadIdx.x + ITEM * B], keys[1][tile_offset + threadIdx.x + ITEM * B]) % ht_len;
            else cudaAssert(0);
            
            uint64_t item = keys[0][tile_offset + threadIdx.x + ITEM * B];
            while(atomicCAS(&ht[slot * (num_keys + 1)], 0xFFFFFFFFFFFFFFFF, (unsigned long long) item) != 0xFFFFFFFFFFFFFFFF) {                
                slot = (slot + 100007) % ht_len;
            }

            for (int n = 1; n < num_keys; n++) {
                ht[slot * (num_keys + 1) + n] = keys[n][tile_offset + threadIdx.x + ITEM * B];
            }
            ht[slot * (num_keys + 1) + num_keys] = tile_offset + threadIdx.x + (ITEM * B);
        }
    }
}

template
__global__ void probe_multikey<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t *row_ids_left, uint64_t *row_ids_right, 
            unsigned long long* count, uint64_t N, int* condition_mode, int num_keys, int is_count);

template
__global__ void build_multikey<BLOCK_THREADS, ITEMS_PER_THREAD>(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int num_keys);

void probeHashTableMultiKey(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t N, int* condition_mode, int num_keys) {
    printf("Launching Probe Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudaMemset(count, 0, sizeof(uint64_t));

    uint64_t** keys_dev;
    cudaMalloc((void**) &keys_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(keys_dev, keys, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_ERROR();
    probe_multikey<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, ht, ht_len, row_ids_left, row_ids_right, (unsigned long long*) count, N, condition_mode, num_keys, 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    uint64_t* h_count = new uint64_t [1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(h_count[0] > 0);
    row_ids_left = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    probe_multikey<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, ht, ht_len, row_ids_left, row_ids_right, (unsigned long long*) count, N, condition_mode, num_keys, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Count: %lu\n", h_count[0]);
    count = h_count;
}

void buildHashTableMultiKey(uint64_t **keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int num_keys) {
    printf("Launching Build Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t** keys_dev;
    cudaMalloc((void**) &keys_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(keys_dev, keys, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    
    cudaMemset(ht, 0xFF, ht_len * 2 * sizeof(unsigned long long));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    build_multikey<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, ht, ht_len, N, num_keys);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}


} // namespace duckdb