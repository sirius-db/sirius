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
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

template <typename T, typename V, int B, int I>
__global__ void hash_groupby_gmem(T **group_key, V** aggregate, unsigned long long* ht, uint64_t ht_len, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, bool need_count) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;
    uint64_t final_slot[I];

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    int n_ht_column;
    if (need_count) n_ht_column = num_keys + num_aggregates + 1;
    else n_ht_column = num_keys + num_aggregates;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            uint64_t hash_key = 0xFFFFFFFFFFFFFFFF;
            for (int n = 0; n < num_keys; n++) {
                hash_key = hash_combine(hash_key, group_key[n][offset]);
            }

            uint64_t slot = hash_key % ht_len;

            int n = 0;
            bool can_break = false;
            bool atomic = true; // initially all thread do the atomic
            while(!can_break) {
                uint64_t item = group_key[n][offset];
                unsigned long long old;
                if (atomic) {
                    old = atomicCAS(&ht[slot * n_ht_column + n], 0xFFFFFFFFFFFFFFFF, (unsigned long long) item); //compete for atomic
                    if (old != 0xFFFFFFFFFFFFFFFF) atomic = false; // lose the atomic, no longer do the atomic for the next keys
                    else atomic = true;
                } else {
                    old = ht[slot * n_ht_column + n];
                }

                if ((atomic && old == 0xFFFFFFFFFFFFFFFF)) { // atomic succeeds, try atomic on the next keys
                    if (n == num_keys - 1) {
                        can_break = true;
                        final_slot[ITEM] = slot;
                    } else n++;
                } else if (!atomic && old == 0xFFFFFFFFFFFFFFFF) { // not the thread with the atomic and need to wait until the atomic is done
                    continue;
                } else if (old == item) { // already exist
                    cudaAssert(atomic == false);
                    if (n == num_keys - 1) {
                        can_break = true;
                        final_slot[ITEM] = slot;
                    } else n++;
                } else {
                    n = 0;
                    atomic = true;
                    slot = (slot + 100007) % ht_len;
                }
            }
        }
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            for (int n = 0 ; n < num_aggregates; n++) {
                V* ptr = reinterpret_cast<V*>(ht + (final_slot[ITEM] * n_ht_column + num_keys + n));
                cuda::atomic_ref<V, cuda::thread_scope_device> res_atomic(*ptr);
                if (agg_mode[n] == 0) {
                    res_atomic.fetch_add(aggregate[n][offset]);
                } else if (agg_mode[n] == 1) { //avg (sum and count)
                    res_atomic.fetch_add(aggregate[n][offset]);
                } else if (agg_mode[n] == 2) {
                    res_atomic.fetch_max(aggregate[n][offset]);
                } else if (agg_mode[n] == 3) {
                    res_atomic.fetch_min(aggregate[n][offset]);
                } else if (agg_mode[n] == 4) {
                    atomicAdd(&ht[final_slot[ITEM] * n_ht_column + num_keys + n], 1LLU); // count
                } else if (agg_mode[n] == 5) {
                    // should be zero, so do nothing
                } else cudaAssert(false);
            }
            if (need_count) { // need count to count average
                atomicAdd(&ht[final_slot[ITEM] * n_ht_column + num_keys + num_aggregates], 1LLU); // count
            }
        }
    }
}

template <typename T, typename V, int B, int I>
__global__ void hash_groupby_smem(T **group_key, V** aggregate, T* max, T* min, unsigned long long* ht, 
        uint64_t ht_len, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, 
        V init_min, V init_max, int* agg_mode, bool need_count) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;
    uint64_t final_slot[I];

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    int n_ht_column;
    if (need_count) n_ht_column = num_keys + num_aggregates + 1;
    else n_ht_column = num_keys + num_aggregates;

    extern __shared__ unsigned long long local_ht[];

    if (threadIdx.x < ht_len) {
        for (int n = 0; n < num_keys; n++) {
            local_ht[threadIdx.x * n_ht_column + n] = 0xFFFFFFFFFFFFFFFF;
        }
        for (int n = 0; n < num_aggregates; n++) {
            if (agg_mode[n] == 2) {
                local_ht[threadIdx.x * n_ht_column + num_keys + n] = init_max;
            } else if (agg_mode[n] == 3) {
                local_ht[threadIdx.x * n_ht_column + num_keys + n] = init_min;
            } else if (agg_mode[n] == 1) {
                local_ht[threadIdx.x * n_ht_column + num_keys + n] = 0;
                local_ht[threadIdx.x * n_ht_column + num_keys + num_aggregates] = 0;
            } else {
                local_ht[threadIdx.x * n_ht_column + num_keys + n] = 0;
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            uint64_t hash_key = 0;
            uint64_t range = 1;
            
            for (int n = 0; n < num_keys; n++) {
                hash_key += (group_key[n][offset] - min[n]) * range;
                range *= (max[n] - min[n] + 1);
            }

            // final_slot[ITEM] = hash_key % ht_len;
            cudaAssert(hash_key < ht_len);
            final_slot[ITEM] = hash_key;

            for (int n = 0; n < num_keys; n++) {
                uint64_t item = group_key[n][offset];
                atomicCAS(&local_ht[final_slot[ITEM] * n_ht_column + n], 0xFFFFFFFFFFFFFFFF, (unsigned long long) item);
            }
        }
    }

    __syncthreads();

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            uint64_t offset = tile_offset + threadIdx.x + ITEM * B;
            for (int n = 0 ; n < num_aggregates; n++) {
                V* ptr = reinterpret_cast<V*>(local_ht + (final_slot[ITEM] * n_ht_column + num_keys + n));
                cuda::atomic_ref<V, cuda::thread_scope_block> res_atomic(*ptr);
                if (agg_mode[n] == 0) {
                    res_atomic.fetch_add(aggregate[n][offset]);
                } else if (agg_mode[n] == 1) {
                    res_atomic.fetch_add(aggregate[n][offset]);
                } else if (agg_mode[n] == 2) {
                    res_atomic.fetch_max(aggregate[n][offset]);
                } else if (agg_mode[n] == 3) {
                    res_atomic.fetch_min(aggregate[n][offset]);
                } else if (agg_mode[n] == 4) {
                    atomicAdd(&local_ht[final_slot[ITEM] * n_ht_column + num_keys + n], 1LLU); // count
                } else if (agg_mode[n] == 5) {
                    // should be zero, so do nothing
                } else cudaAssert(false);
            }
            if (need_count) { // need count to count average
                atomicAdd(&local_ht[final_slot[ITEM] * n_ht_column + num_keys + num_aggregates], 1LLU); // count
            }
        }
    }

    __syncthreads();

    if (threadIdx.x < ht_len) {
        for (int n = 0; n < num_keys; n++) {
            if (local_ht[threadIdx.x * n_ht_column + n] != 0xFFFFFFFFFFFFFFFF) {
                ht[threadIdx.x * n_ht_column + n] = local_ht[threadIdx.x * n_ht_column + n];
            }
            // ht[threadIdx.x * n_ht_column + n] = local_ht[threadIdx.x * n_ht_column + n];
        }
        for (int n = 0; n < num_aggregates; n++) {
            V* ptr = reinterpret_cast<V*>(ht + (threadIdx.x * n_ht_column + num_keys + n));
            V* ptr2 = reinterpret_cast<V*>(local_ht + (threadIdx.x * n_ht_column + num_keys + n));
            cuda::atomic_ref<V, cuda::thread_scope_device> res_atomic(*ptr);
            if (agg_mode[n] == 0) {
                if (*ptr2 != 0) res_atomic.fetch_add(*ptr2);
            } else if (agg_mode[n] == 1) {
                if (*ptr2 != 0) res_atomic.fetch_add(*ptr2);
            } else if (agg_mode[n] == 2) {
                res_atomic.fetch_max(*ptr2);
            } else if (agg_mode[n] == 3) {
                res_atomic.fetch_min(*ptr2);
            } else if (agg_mode[n] == 4) {
                if (local_ht[threadIdx.x * n_ht_column + num_keys + n] != 0)
                    atomicAdd(&ht[threadIdx.x * n_ht_column + num_keys + n], local_ht[threadIdx.x * n_ht_column + num_keys + n]); // count
            } else if (agg_mode[n] == 5) {
                // should be zero, so do nothing
            } else cudaAssert(false);
        }
        if (need_count) { // need count to count average
            atomicAdd(&ht[threadIdx.x * n_ht_column + num_keys + num_aggregates], local_ht[threadIdx.x * n_ht_column + num_keys + num_aggregates]); // count
        }
    }
}



template <typename T, typename V, int B, int I>
__global__ void scan_hash_group(unsigned long long* ht, uint8_t** group, uint8_t** aggregate, unsigned long long* count,
        uint64_t ht_len, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode, bool is_count, bool need_count) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

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

    int n_ht_column;
    if (need_count) n_ht_column = num_keys + num_aggregates + 1;
    else n_ht_column = num_keys + num_aggregates;

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        selection_flags[ITEM] = 0;
    }

    #pragma unroll
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            uint64_t slot = tile_offset + threadIdx.x + ITEM * B; 
            if (ht[slot * n_ht_column] != 0xFFFFFFFFFFFFFFFF) {
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
    for (int ITEM = 0; ITEM < I; ITEM++) {
        if (threadIdx.x + (ITEM * B) < num_tile_items) {
            if(selection_flags[ITEM]) {
                uint64_t offset = block_off + c_t_count++;
                uint64_t slot = tile_offset + threadIdx.x + ITEM * B; 
                for (int n = 0; n < num_keys; n++) {
                    T* group_ptr = reinterpret_cast<T*>(group[n]);
                    T* ptr = reinterpret_cast<T*>(ht + (slot * n_ht_column + n));
                    group_ptr[offset] = ptr[0];
                }
                for (int n = 0; n < num_aggregates; n++) {
                    V* aggregate_ptr = reinterpret_cast<V*>(aggregate[n]);
                    V* ptr = reinterpret_cast<V*>(ht + (slot * n_ht_column + num_keys + n));
                    if (agg_mode[n] == 0 || agg_mode[n] == 2 || agg_mode[n] == 3) {
                        aggregate_ptr[offset] = ptr[0];
                    } else if (agg_mode[n] == 1) {
                        uint64_t* ptr_cnt = reinterpret_cast<uint64_t*>(ht + (slot * n_ht_column + num_keys + num_aggregates));
                        aggregate_ptr[offset] = ptr[0] / ptr_cnt[0];
                    } else if (agg_mode[n] == 4) {
                        uint64_t* aggregate_cnt_ptr = reinterpret_cast<uint64_t*>(aggregate[n]);
                        uint64_t* ptr_cnt = reinterpret_cast<uint64_t*>(ht + (slot * n_ht_column + num_keys + n));
                        aggregate_cnt_ptr[offset] = ptr_cnt[0];
                    } else if (agg_mode[n] == 5) {
                        aggregate_ptr[offset] = 0;
                    } else cudaAssert(false);
                }
            }
        }
    }
}

template <typename V>
__global__ void set_group_ht(unsigned long long* ht, V** aggregate, uint64_t num_keys, uint64_t num_aggregates, uint64_t ht_len, 
    V init_min, V init_max, int* agg_mode, bool need_count) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ht_len) {

        int n_ht_column;
        if (need_count) n_ht_column = num_keys + num_aggregates + 1;
        else n_ht_column = num_keys + num_aggregates;

        for (int n = 0; n < num_keys; n++) {
            ht[tid * n_ht_column + n] = 0xFFFFFFFFFFFFFFFF;
        }
        for (int n = 0; n < num_aggregates; n++) {
            if (agg_mode[n] == 2) {
                ht[tid * n_ht_column + num_keys + n] = init_max;
            } else if (agg_mode[n] == 3) {
                ht[tid * n_ht_column + num_keys + n] = init_min;
            } else if (agg_mode[n] == 1) {
                ht[tid * n_ht_column+ num_keys + n] = 0;
                ht[tid * n_ht_column + num_keys + num_aggregates] = 0;
            } else {
                ht[tid * n_ht_column + num_keys + n] = 0;
            }
        }
    }
}

template <typename T, int B, int I>
__global__ void get_min_max(T *keys, T* res_max, T* res_min, uint64_t N) {

    uint64_t tile_size = B * I;
    uint64_t tile_offset = blockIdx.x * tile_size;

    uint64_t num_tiles = (N + tile_size - 1) / tile_size;
    uint64_t num_tile_items = tile_size;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = N - tile_offset;
    }

    T local_max = keys[0];
    T local_min = keys[0];

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if (threadIdx.x + ITEM * BLOCK_THREADS < num_tile_items) {
            local_max = max(local_max, keys[tile_offset + threadIdx.x + ITEM * B]);
            local_min = min(local_min, keys[tile_offset + threadIdx.x + ITEM * B]);
        }
    }

    __syncthreads();
    static __shared__ T buffer[32];
    cuda::atomic_ref<T, cuda::thread_scope_device> res_atomic_max(*res_max);
    cuda::atomic_ref<T, cuda::thread_scope_device> res_atomic_min(*res_min);

    T block_reduce_max = BlockReduce<T, BLOCK_THREADS, ITEMS_PER_THREAD>(local_max, (T*)buffer, 2);
    __syncthreads();

    if (threadIdx.x == 0) {
        res_atomic_max.fetch_max(block_reduce_max); 
    }

    __syncthreads();
    T block_reduce_min = BlockReduce<T, BLOCK_THREADS, ITEMS_PER_THREAD>(local_min, (T*)buffer, 3);
    __syncthreads();

    if (threadIdx.x == 0) {
        res_atomic_min.fetch_min(block_reduce_min);  
    }
}

template <typename T, typename V>
void hashGroupedAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SETUP_TIMING();
    START_TIMER();

    SIRIUS_LOG_DEBUG("Launching Hash Grouped Aggregate Kernel");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    T** keys_dev = gpuBufferManager->customCudaMalloc<T*>(num_keys, 0, 0);
    cudaMemcpy(keys_dev, reinterpret_cast<T*>(keys), num_keys * sizeof(T*), cudaMemcpyHostToDevice);

    V** aggregate_keys_dev = gpuBufferManager->customCudaMalloc<V*>(num_aggregates, 0, 0);
    cudaMemcpy(aggregate_keys_dev, reinterpret_cast<V*>(aggregate_keys), num_aggregates * sizeof(V*), cudaMemcpyHostToDevice);

    int* agg_mode_dev = gpuBufferManager->customCudaMalloc<int>(num_aggregates, 0, 0);
    cudaMemcpy(agg_mode_dev, agg_mode, num_aggregates * sizeof(int), cudaMemcpyHostToDevice);

    bool need_count = 0;
    for (int mode = 0; mode < num_aggregates; mode++) {
        if (agg_mode[mode] == 1) {
            need_count = 1;
            break;
        }
    }

    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    T* max_key = gpuBufferManager->customCudaMalloc<T>(num_keys, 0, 0);
    T* min_key = gpuBufferManager->customCudaMalloc<T>(num_keys, 0, 0);
    for (int i = 0; i < num_keys; i++) {
        T* temp = reinterpret_cast<T*> (keys[i]);
        T* max_key_ptr = max_key + i;
        T* min_key_ptr = min_key + i;
        cudaMemcpy(max_key_ptr, temp, sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(min_key_ptr, temp, sizeof(T), cudaMemcpyDeviceToDevice);
        get_min_max<T, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(reinterpret_cast<T*>(keys[i]), max_key_ptr, min_key_ptr, N);
        CHECK_ERROR();
    }

    T* max_key_host = gpuBufferManager->customCudaHostAlloc<T>(num_keys);
    T* min_key_host = gpuBufferManager->customCudaHostAlloc<T>(num_keys);
    cudaMemcpy(max_key_host, max_key, num_keys * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(min_key_host, min_key, num_keys * sizeof(T), cudaMemcpyDeviceToHost);

    T C = 1;
    for (int i = 0; i < num_keys; i++) {
        C *= (max_key_host[i] - min_key_host[i] + 1);
    }

    SIRIUS_LOG_DEBUG("Input size: {}", N);
    SIRIUS_LOG_DEBUG("max {} min {}", max_key_host[0], min_key_host[0]);

    V init_max; V init_min;
    if constexpr (std::is_same<V, double>::value) {
        // Do something if T is int
        init_max = -DBL_MAX; init_min = DBL_MAX;
    } else if constexpr (std::is_same<V, uint64_t>::value) {
        // Do something else if T is not int
        init_max = INT_MIN; init_min = INT64_MAX;
    } else {
        assert(0);
    }
    
    unsigned long long* ht;
    uint64_t ht_len;
    if (C < BLOCK_THREADS) {
        SIRIUS_LOG_DEBUG("Hash grouped aggregate in shared memory");
        ht_len = C;
        size_t smem_size;
        if (need_count) {
            ht = (unsigned long long*) gpuBufferManager->customCudaMalloc<uint64_t>(ht_len * (num_keys + num_aggregates + 1), 0, 0);
            smem_size = ht_len * (num_keys + num_aggregates + 1) * sizeof(unsigned long long);
        } else {
            ht = (unsigned long long*) gpuBufferManager->customCudaMalloc<uint64_t>(ht_len * (num_keys + num_aggregates), 0, 0);
            smem_size = ht_len * (num_keys + num_aggregates) * sizeof(unsigned long long);
        }
        set_group_ht<<<(ht_len + BLOCK_THREADS - 1)/BLOCK_THREADS, BLOCK_THREADS>>>(ht, aggregate_keys_dev, num_keys, num_aggregates, ht_len, 
                init_min, init_max, agg_mode_dev, need_count);
        CHECK_ERROR();
        hash_groupby_smem<T, V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS, smem_size>>>(keys_dev, aggregate_keys_dev, 
                max_key, min_key, ht, ht_len, N, num_keys, num_aggregates, 
                init_min, init_max, agg_mode_dev, need_count);
        CHECK_ERROR();

    } else{
        SIRIUS_LOG_DEBUG("Hash grouped aggregate in global memory");
        ht_len = N * 2;
        if (need_count) {
            ht = (unsigned long long*) gpuBufferManager->customCudaMalloc<uint64_t>(ht_len * (num_keys + num_aggregates + 1), 0, 0);         
        } else {
            ht = (unsigned long long*) gpuBufferManager->customCudaMalloc<uint64_t>(ht_len * (num_keys + num_aggregates), 0, 0);
        }
        set_group_ht<<<(ht_len + BLOCK_THREADS - 1)/BLOCK_THREADS, BLOCK_THREADS>>>(ht, aggregate_keys_dev, num_keys, num_aggregates, ht_len, 
            init_min, init_max, agg_mode_dev, need_count);
        CHECK_ERROR();
        hash_groupby_gmem<T, V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, aggregate_keys_dev, 
            ht, ht_len, N, num_keys, num_aggregates, agg_mode_dev, need_count);
        // CHECK_ERROR();
        CHECK_ERROR();
    }
    
    CHECK_ERROR();

    uint8_t** keys_dev_result = gpuBufferManager->customCudaMalloc<uint8_t*>(num_keys, 0, 0);
    uint8_t** aggregate_keys_dev_result = gpuBufferManager->customCudaMalloc<uint8_t*>(num_aggregates, 0, 0);
    uint64_t* d_count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_count, 0, sizeof(uint64_t));
    scan_hash_group<T, V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(ht_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(ht, nullptr, nullptr, 
            (unsigned long long*) d_count, ht_len, num_keys, num_aggregates, agg_mode_dev, true, need_count); 

    CHECK_ERROR();
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    cudaMemcpy(h_count, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    assert(h_count[0] > 0);

    uint8_t** keys_result = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_keys);
    uint8_t** aggregate_keys_result = gpuBufferManager->customCudaHostAlloc<uint8_t*>(num_aggregates);
    for (int i = 0; i < num_keys; i++) {
        keys_result[i] = gpuBufferManager->customCudaMalloc<uint8_t>(h_count[0] * sizeof(T), 0, 0);
    }
    for (int i = 0; i < num_aggregates; i++) {
        if (agg_mode[i] == 0 || agg_mode[i] == 1 || agg_mode[i] == 2 || agg_mode[i] == 3) {
            aggregate_keys_result[i] = gpuBufferManager->customCudaMalloc<uint8_t>(h_count[0] * sizeof(V), 0, 0);
        } else if (agg_mode[i] == 4 || agg_mode[i] == 5) {
            aggregate_keys_result[i] = gpuBufferManager->customCudaMalloc<uint8_t>(h_count[0] * sizeof(uint64_t), 0, 0);
        } else {
            assert(false);
        }
    }
    
    cudaMemcpy(keys_dev_result, keys_result, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemcpy(aggregate_keys_dev_result, aggregate_keys_result, num_aggregates * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(uint64_t));
    CHECK_ERROR();

    scan_hash_group<T, V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(ht_len + tile_items - 1)/tile_items, BLOCK_THREADS>>>(ht, keys_dev_result, aggregate_keys_dev_result, 
            (unsigned long long*) d_count, ht_len, num_keys, num_aggregates, agg_mode_dev, false, need_count);

    CHECK_ERROR();

    // CHECK_ERROR();

    SIRIUS_LOG_DEBUG("Hash Grouped Aggregate Count: {}", h_count[0]);

    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys[i]), 0);
        keys[i] = keys_result[i];
    }

    for (uint64_t i = 0; i < num_aggregates; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys[i]), 0);
        aggregate_keys[i] = aggregate_keys_result[i];
    }

    count[0] = h_count[0];

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys_dev_result), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_dev_result), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_count), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(agg_mode_dev), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(max_key), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(min_key), 0);

    if (C < BLOCK_THREADS) {
        ht_len = C;
        if (need_count) {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(ht), 0);
        } else {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(ht), 0);    
        }

    } else{
        ht_len = N * 2;
        if (need_count) {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(ht), 0);
        } else {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(ht), 0);
        }
    }
    STOP_TIMER();

}

template
void hashGroupedAggregate<uint64_t, uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void hashGroupedAggregate<uint64_t, double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

}