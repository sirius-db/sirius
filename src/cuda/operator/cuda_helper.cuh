#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <float.h>
#include <thrust/device_vector.h>
#include "gpu_buffer_manager.hpp"

#include <iostream> 
#include <string> 
#include <sstream>

#define CUB_STDERR

#define cudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!\n", blockIdx.x, threadIdx.x, __FILE__, __LINE__ ); return; }

#define CHECK_ERROR() { \
  cudaDeviceSynchronize(); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    gpuErrchk(error); \
    exit(-1); \
  } \
}

#define CHECK_ERROR_STREAM(stream) { \
  cudaStreamSynchronize(stream); \
  cudaError_t error = cudaGetLastError(); \
  if(error != cudaSuccess) \
  { \
    gpuErrchk(error); \
    exit(-1); \
  } \
}

#define CHECK_CU_ERROR(err, cufunc)                                     \
    if (err != CUDA_SUCCESS) { printf ("Error %d for CUDA Driver API function '%s'\n", err, cufunc); return -1; }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define SETUP_TIMING() cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);

#define TIME_FUNC(f,t) { \
    cudaEventRecord(start, 0); \
    f; \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&t, start,stop); \
}

#define CLEANUP(vec) if(vec)CubDebugExit(g_allocator.DeviceFree(vec))

#define ALLOCATE(vec,size) CubDebugExit(g_allocator.DeviceAllocate((void**)&vec, size))

#define START_TIMER() { \
    cudaEventRecord(start, 0); \
    cudaEventSynchronize(start); \
}

#define STOP_TIMER() { \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    float elapsedTime = 0; \
    cudaEventElapsedTime(&elapsedTime, start, stop); \
    printf("Elapsed time op: %f\n", elapsedTime); \
}

#define BLOCK_THREADS 128
#define ITEMS_PER_THREAD 4 
// #define TILE_SIZE (BLOCK_THREADS * ITEMS_PER_THREAD)

template <int B, int I>
__global__ void compact_valid_rows(bool* selection_flags, uint64_t* row_ids, unsigned long long* count, uint64_t N, int is_count, int not_equal) {

    typedef cub::BlockScan<int, B> BlockScanInt;

    __shared__ union TempStorage
    {
        typename BlockScanInt::TempStorage scan;
    } temp_storage;

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
    for (int ITEM = 0; ITEM < I; ++ITEM) {
        if (threadIdx.x + ITEM * B < num_tile_items) {
            bool is_selected = selection_flags[tile_offset + threadIdx.x + ITEM * B];
            if (not_equal) {
              if(!is_selected) t_count++;
            } else {
              if(is_selected) t_count++;
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
            bool is_selected = selection_flags[tile_offset + threadIdx.x + ITEM * B];
            if (not_equal) {
              if(!is_selected) {
                uint64_t offset = block_off + c_t_count++;
                row_ids[offset] = tile_offset + threadIdx.x + ITEM * B;
              }
            } else {
              if(is_selected) {
                uint64_t offset = block_off + c_t_count++;
                row_ids[offset] = tile_offset + threadIdx.x + ITEM * B;
              }
            }
        }
    }
}

template<typename T, int B, int I>
__device__ __forceinline__ T BlockReduce(
    T  item,
    T* shared,
    int op_mode
    ) {
    __syncthreads();

    T val = item;
    const int warp_size = 32;
    int lane = threadIdx.x % warp_size;
    int wid = threadIdx.x / warp_size;

    // Calculate sum across warp
    for (int offset = 16; offset > 0; offset /= 2) {
        if (op_mode == 0 || op_mode == 1) { //sum or avg
            val += __shfl_down_sync(0xffffffff, val, offset);
        } else if (op_mode == 2) { //max
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        } else if (op_mode == 3) { //min
            val = min(val, __shfl_down_sync(0xffffffff, val, offset));
        }
    }

    // Store sum in buffer
    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    // Load the sums into the first warp
    if (op_mode == 0 || op_mode == 1) { //sum or avg
        val = (threadIdx.x < blockDim.x / warp_size) ? shared[lane] : 0;
    } else if (op_mode == 2) { //max
        val = (threadIdx.x < blockDim.x / warp_size) ? shared[lane] : shared[0];
    } else if (op_mode == 3) { //min
        val = (threadIdx.x < blockDim.x / warp_size) ? shared[lane] : shared[0];
    }

    // Calculate sum of sums
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            if (op_mode == 0 || op_mode == 1) { //sum or avg
                val += __shfl_down_sync(0xffffffff, val, offset);
            } else if (op_mode == 2) { //max
                val = max(val, __shfl_down_sync(0xffffffff, val, offset));
            } else if (op_mode == 3) { //min
                val = min(val, __shfl_down_sync(0xffffffff, val, offset));
            }
        }
    }

  return val;
}

// Simple hash function for integers
inline __device__ __host__ uint64_t custom_hash_int(uint64_t key) {
    key ^= (key >> 21);
    key ^= (key << 37);
    key ^= (key >> 4);
    key *= 2685821657736338717ULL;
    return key;
}

// Combine hashes for an array of keys
inline __device__ uint64_t hash_combine(uint64_t old_key, uint64_t new_key) {
    if (old_key == 0xFFFFFFFFFFFFFFFF) {
        return custom_hash_int(new_key);
    }
    return old_key ^ (custom_hash_int(new_key) + 0x9e3779b9 + (old_key << 6) + (old_key >> 2));
}

#define STRING_HASH_POWER 31
#define STRING_HASH_MOD_VALUE 1000000009
#define BITS_IN_BYTE 8
#define BYTES_IN_INTEGER 8