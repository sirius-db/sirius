@@ -0,0 +1,307 @@
/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_helper.cuh"
#include "gpu_physical_hash_join.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

__device__ uint64_t hash64_left(uint64_t key1, uint64_t key2) {
    uint64_t h = key1 * 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    h ^= key2 * 0xc6a4a7935bd1e995ull;
    h *= 0xc6a4a7935bd1e995ull;
    h ^= (h >> 33);
    return h;
}

// Probe kernel for LEFT SEMI/ANTI joins
// Marks which LHS (build) rows have matches from RHS (probe)
template <int B, int I, typename T>
__global__ void probe_left_semi_anti(T **keys, unsigned long long* ht, uint64_t ht_len,
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
            // Hash the probe key (RHS row)
            if (equal_keys == 1) 
                slot = hash64_single(keys[0][tile_offset + threadIdx.x + ITEM * B]) % ht_len;
            else if (equal_keys == 2) 
                slot = hash64_multikey(keys[0][tile_offset + threadIdx.x + ITEM * B], 
                                       keys[1][tile_offset + threadIdx.x + ITEM * B]) % ht_len;
            else 
                cudaAssert(0);
            
            // Linear probe through hash table
            while (ht[slot * (num_keys + 2)] != 0xFFFFFFFFFFFFFFFF) {
                bool local_found = 1;
                
                // Check all join conditions
                for (int n = 0; n < num_keys; n++) {
                    uint64_t item = keys[n][tile_offset + threadIdx.x + ITEM * B];
                    
                    if (condition_mode[n] == 0 && ht[slot * (num_keys + 2) + n] != item) 
                        local_found = 0;
                    else if (condition_mode[n] == 1 && ht[slot * (num_keys + 2) + n] == item) 
                        local_found = 0;
                }
                
                if (local_found) {
                    // Mark this LHS (build) row as having a match from RHS (probe)
                    ht[slot * (num_keys + 2) + num_keys + 1] = tile_offset + threadIdx.x + ITEM * B;
                    break;  // Only mark once per LHS row
                }
                
                // Move to next slot
                slot = (slot + 65599) % ht_len;
            }
        }
    }
}

// Scan probe data to find unmatched LHS rows (for LEFT and LEFT ANTI joins)
// tracked by the matched array
template <int B, int I>
__global__ void scan_left_probe_data(uint8_t *matched, uint64_t N, unsigned long long* count, 
                uint64_t *row_ids, int is_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

    uint64_t items_off[I];
    int selection_flags[I];
    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    int t_count = 0;           // Number of unmatched rows selected per thread
    int c_t_count = 0;         // Prefix sum of t_count
    __shared__ uint64_t block_off;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    // Initialize selection flags
    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    // First pass: identify unmatched rows
    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            uint64_t probe_idx = tile_offset + threadIdx.x + ITEM * B;
            
            // If probe row (LHS) was not matched, include it in output
            if (matched[probe_idx] == 0) {
                items_off[ITEM] = probe_idx;
                selection_flags[ITEM] = 1;
                t_count++;
            }
        }
    }

    // Barrier to ensure all threads have completed first pass
    __syncthreads();

    // Prefix sum to compute offsets
    BlockScanInt(temp_storage.scan).ExclusiveSum(t_count, c_t_count);
    if(threadIdx.x == blockDim.x - 1) {
        // Last thread atomically adds total count from this block to global count
        block_off = atomicAdd(count, (unsigned long long) t_count + c_t_count);
    }

    __syncthreads();

    // If just counting, exit early
    if (is_count) return;

    // Second pass: write row IDs for unmatched rows
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

// Explicit template instantiation for the scan kernel
template
__global__ void scan_left_probe_data<BLOCK_THREADS, ITEMS_PER_THREAD>(
    uint8_t *matched, uint64_t N, unsigned long long* count, 
    uint64_t *row_ids, int is_count);

// Host function to scan probe data and extract unmatched LHS rows
void scanProbeDataLeft(uint8_t *matched, uint64_t N, uint64_t* &row_ids, uint64_t* &count) {
    CHECK_ERROR();
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    
    if (N == 0) {
        uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
        h_count[0] = 0;
        count = h_count;
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    
    SIRIUS_LOG_DEBUG("Launching Scan Left Probe Data Kernel for {} rows", N);
    SETUP_TIMING();
    START_TIMER();
    
    // Allocate device count variable
    count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(count, 0, sizeof(uint64_t));

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    
    // First kernel invocation: count unmatched rows
    CHECK_ERROR();
    scan_left_probe_data<BLOCK_THREADS, ITEMS_PER_THREAD><<<
        (N + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
        matched, N, (unsigned long long*) count, nullptr, 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    // Copy count from device to host
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // If no unmatched rows, return early
    if (h_count[0] == 0) {
        row_ids = nullptr;
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
        count = h_count;
        SIRIUS_LOG_DEBUG("No unmatched rows in LEFT JOIN");
        return;
    }
    
    // Allocate row_ids array
    SIRIUS_LOG_DEBUG("Found {} unmatched LHS rows", h_count[0]);
    row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
    
    // Reset count for second pass
    cudaMemset(count, 0, sizeof(uint64_t));
    
    // Second kernel invocation: extract row IDs
    scan_left_probe_data<BLOCK_THREADS, ITEMS_PER_THREAD><<<
        (N + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
        matched, N, (unsigned long long*) count, row_ids, 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
    
    SIRIUS_LOG_DEBUG("Scan Left Unmatched Count: {}", h_count[0]);
    
    // Cleanup device count and set host count
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
    count = h_count;

    CHECK_ERROR();
    cudaDeviceSynchronize();
    STOP_TIMER();
}

// Probe hash table for LEFT SEMI/ANTI joins
// Marks LHS rows that have matches from RHS probe data
template <typename T>
void probeHashTableLeftSemiAnti(uint8_t **keys, unsigned long long* ht, uint64_t ht_len, 
                                 uint64_t N, int* condition_mode, int num_keys) {
    CHECK_ERROR();
    if (N == 0 || ht_len == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0 or hash table is empty");
        return;
    }
    
    SIRIUS_LOG_DEBUG("Launching Left Semi/Anti Probe Kernel");
    SETUP_TIMING();
    START_TIMER();
    
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    // Cast keys to correct type
    T** keys_data = gpuBufferManager->customCudaHostAlloc<T*>(num_keys);
    for (int idx = 0; idx < num_keys; idx++) {
        keys_data[idx] = reinterpret_cast<T*>(keys[idx]);
    }

    // Copy to device
    T** keys_dev = gpuBufferManager->customCudaMalloc<T*>(num_keys, 0, 0);
    cudaMemcpy(keys_dev, keys_data, num_keys * sizeof(T*), cudaMemcpyHostToDevice);

    // Count equality conditions
    int equal_keys = 0;
    for (int idx = 0; idx < num_keys; idx++) {
        if (condition_mode[idx] == 0) equal_keys++;
    }

    // Copy condition modes to device
    int* condition_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_keys, 0, 0);
    cudaMemcpy(condition_mode_dev, condition_mode, num_keys * sizeof(int), cudaMemcpyHostToDevice);

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    
    // Launch probe kernel
    probe_left_semi_anti<BLOCK_THREADS, ITEMS_PER_THREAD, T><<<
        (N + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
        keys_dev, ht, ht_len, N, condition_mode_dev, num_keys, equal_keys);
    CHECK_ERROR();
    cudaDeviceSynchronize();

    SIRIUS_LOG_DEBUG("Finished probe left semi/anti");
    STOP_TIMER();
    
    // Cleanup
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(condition_mode_dev), 0);
}

// Explicit template instantiations
template
void probeHashTableLeftSemiAnti<int32_t>(uint8_t **keys, unsigned long long* ht, 
                                          uint64_t ht_len, uint64_t N, 
                                          int* condition_mode, int num_keys);

template
void probeHashTableLeftSemiAnti<int64_t>(uint8_t **keys, unsigned long long* ht, 
                                          uint64_t ht_len, uint64_t N, 
                                          int* condition_mode, int num_keys);

} // namespace duckdb
