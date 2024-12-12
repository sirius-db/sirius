#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "utils.hpp"

namespace duckdb {

#define WARMUP_BLOCKS 1000000
#define WARMUP_THREADS_PER_BLOCK 512

//Define the CUDA kernel
__global__ void warmup_kernel(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

void warmup_gpu() {
    // Perform the warmup
    cudaFree(0);
    warmup_kernel<<<WARMUP_BLOCKS, WARMUP_THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
}

} // namespace duckdb