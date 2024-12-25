#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <inttypes.h>
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

constexpr uint32_t BOTTOM_BYTES_PERM_MASK = (0 << 12) | (1 << 8) | (2 << 4) | (3 << 0);
__device__ __forceinline__ uint64_t convert_endianess(uint64_t value) {
  uint32_t bottom_bytes = value;
  uint32_t top_bytes = value >> 32;
  uint64_t bottom_swapped = __byte_perm(bottom_bytes, top_bytes, BOTTOM_BYTES_PERM_MASK);
  uint64_t top_swapped = __byte_perm(top_bytes, bottom_bytes, BOTTOM_BYTES_PERM_MASK);
  return (bottom_swapped << 32) | top_swapped;
}

void warmup_gpu() {
  // Perform the warmup
  cudaFree(0);
  warmup_kernel<<<WARMUP_BLOCKS, WARMUP_THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
}

} // namespace duckdb