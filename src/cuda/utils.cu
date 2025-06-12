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

__global__ void convert_int64_to_int128(uint8_t *input, uint8_t *output, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Store the converted value as 16 bytes in output
    uint64_t* input_ptr = reinterpret_cast<uint64_t*>(input + idx * 8);
    uint64_t* output_ptr = reinterpret_cast<uint64_t*>(output + idx * 16);
    // for (int i = 0; i < 8; ++i) {
      // output[idx * 16 + i] = input[idx * 8 + i];
      // output[idx * 16 + i + 8] = 0;
      output_ptr[0] = input_ptr[0];
      output_ptr[1] = 0; // Set the upper 64 bits to zero
    // }
  }
}

__global__ void convert_int32_to_int128(uint8_t *input, uint8_t *output, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Store the converted value as 16 bytes in output
    int32_t* input_ptr = reinterpret_cast<int32_t*>(input + idx * 4);
    int32_t* output_ptr = reinterpret_cast<int32_t*>(output + idx * 16);
    // for (int i = 0; i < 8; ++i) {
      output_ptr[0] = input_ptr[0];
      output_ptr[1] = 0; // Set the upper 64 bits to zero
      output_ptr[2] = 0; // Set the upper 64 bits to zero
      output_ptr[3] = 0; // Set the upper 64 bits to zero
    // }
  }
}

__global__ void convert_int32_to_int64(uint8_t *input, uint8_t *output, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // Store the converted value as 16 bytes in output
    int32_t* input_ptr = reinterpret_cast<int32_t*>(input + idx * 4);
    int64_t* output_ptr = reinterpret_cast<int64_t*>(output + idx * 8);
    output_ptr[0] = input_ptr[0];
  }
}

void warmup_gpu() {
  // Perform the warmup
  cudaFree(0);
  warmup_kernel<<<WARMUP_BLOCKS, WARMUP_THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
}

void convertInt64ToInt128(uint8_t *input, uint8_t *output, size_t count) {
  if (count == 0) return;

  // Launch the kernel to convert the data
  size_t threads_per_block = 256;
  size_t blocks = (count + threads_per_block - 1) / threads_per_block;

  convert_int64_to_int128<<<blocks, threads_per_block>>>(input, output, count);
  cudaDeviceSynchronize();
}

void convertInt32ToInt128(uint8_t *input, uint8_t *output, size_t count) {
  if (count == 0) return;

  // Launch the kernel to convert the data
  size_t threads_per_block = 256;
  size_t blocks = (count + threads_per_block - 1) / threads_per_block;

  convert_int32_to_int128<<<blocks, threads_per_block>>>(input, output, count);
  cudaDeviceSynchronize();
}

void convertInt32ToInt64(uint8_t *input, uint8_t *output, size_t count) {
  if (count == 0) return;

  // Launch the kernel to convert the data
  size_t threads_per_block = 256;
  size_t blocks = (count + threads_per_block - 1) / threads_per_block;

  convert_int32_to_int64<<<blocks, threads_per_block>>>(input, output, count);
  cudaDeviceSynchronize();
}

} // namespace duckdb