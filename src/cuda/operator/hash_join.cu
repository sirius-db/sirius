#include "cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"

namespace duckdb {

//TODO: currently this probe does not support many to many join
template <typename T, int B, int I>
__global__ void probe(const uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t *row_ids_left, uint64_t *row_ids_right, unsigned long long* count, uint64_t N, int mode, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    uint64_t items_key[I];
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
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            items_key[ITEM] = keys[tile_offset + threadIdx.x + ITEM * B];
        }
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            bool found = 0;
            uint64_t slot = items_key[ITEM] % ht_len;
            // printf("items key : %ld %ld %ld\n", items_key[ITEM], slot, ht[slot << 1]);
            while (ht[slot << 1] != 0xFFFFFFFFFFFFFFFF) {
                if (ht[slot << 1] == items_key[ITEM]) {
                    items_off[ITEM] = ht[(slot << 1) + 1];
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

template <typename T, int B, int I>
__global__ void build(const uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int mode) {

    uint64_t items_key[I];

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            items_key[ITEM] = keys[tile_offset + threadIdx.x + ITEM * B];
        }
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            uint64_t slot = items_key[ITEM] % ht_len;
            while(atomicCAS(&ht[slot << 1], 0xFFFFFFFFFFFFFFFF, (unsigned long long) items_key[ITEM]) != 0xFFFFFFFFFFFFFFFF) {
                slot = (slot + 100007) % ht_len;
            }
            ht[(slot << 1) + 1] = tile_offset + threadIdx.x + (ITEM * B);
        }
    }
}

__global__ void test(uint64_t* a, uint64_t N) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (uint64_t i = 0; i < 1000; i++) {
            printf("%ld ", a[i]);
        }
        printf("\n");
    }
}

template
__global__ void probe<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *keys, unsigned long long* ht, uint64_t ht_len, 
            uint64_t *row_ids_left, uint64_t *row_ids_right, unsigned long long* count, uint64_t N, int mode, int is_count);

template
__global__ void build<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>(const uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int mode);

template <typename T>
void probeHashTableOri(uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t N, int mode) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Probe Kernel\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudaMemset(count, 0, sizeof(uint64_t));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    CHECK_ERROR();
    probe<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys, ht, ht_len, row_ids_left, row_ids_right, (unsigned long long*) count, N, mode, 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    uint64_t* h_count = new uint64_t [1];
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(h_count[0] > 0);
    row_ids_left = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    row_ids_right = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));
    probe<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys, ht, ht_len, row_ids_left, row_ids_right, (unsigned long long*) count, N, mode, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    printf("Count: %lu\n", h_count[0]);
    count = h_count;
}

template <typename T>
void buildHashTableOri(uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int mode) {
    CHECK_ERROR();
    if (N == 0) {
        printf("N is 0\n");
        return;
    }
    printf("Launching Build Kernel\n");
    cudaMemset(ht, 0xFF, ht_len * 2 * sizeof(unsigned long long));
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    build<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys, ht, ht_len, N, mode);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

template
void probeHashTableOri<uint64_t>(uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t* &row_ids_left, uint64_t* &row_ids_right, uint64_t* &count, uint64_t N, int mode);

template
void buildHashTableOri<uint64_t>(uint64_t *keys, unsigned long long* ht, uint64_t ht_len, uint64_t N, int mode);

}