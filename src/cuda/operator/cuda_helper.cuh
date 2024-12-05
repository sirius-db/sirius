#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <thrust/sort.h>
#include <thrust/reduce.h>
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

#define BLOCK_THREADS 128
#define ITEMS_PER_THREAD 4 
// #define TILE_SIZE (BLOCK_THREADS * ITEMS_PER_THREAD)