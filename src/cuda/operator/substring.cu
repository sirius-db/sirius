#include "cuda_helper.cuh"
#include "gpu_physical_substring.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"

#define THREADS_PER_BLOCK 512

namespace duckdb {

__global__ void perform_substring(uint64_t* prev_offsets, uint64_t* new_offsets, uint64_t num_strings, uint64_t start_idx, uint64_t length) {
  // Get which string this thread workers on
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_strings) return; 

  // Determine the length of this substring
  uint64_t curr_str_start_idx = prev_offsets[tid]; uint64_t curr_str_end_idx = prev_offsets[tid + 1];
  uint64_t substring_start_idx = min(curr_str_start_idx + start_idx, curr_str_end_idx);
  uint64_t substring_end_idx = min(substring_start_idx + length, curr_str_end_idx);
  uint64_t substring_length = substring_end_idx - substring_start_idx;
  new_offsets[tid] = substring_length;
}

__global__ void substring_copy_chars(char* prev_chars, char* new_chars, uint64_t* prev_offsets, uint64_t* new_offsets, uint64_t num_strings, 
    uint64_t start_idx, uint64_t length) {

  // Get which string this thread workers on
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_strings) return; 

  // Determine the range for this substring
  uint64_t curr_str_start_idx = prev_offsets[tid]; uint64_t curr_str_end_idx = prev_offsets[tid + 1];
  uint64_t substring_start_idx = min(curr_str_start_idx + start_idx, curr_str_end_idx);
  uint64_t substring_end_idx = min(substring_start_idx + length, curr_str_end_idx);
  uint64_t substring_length = substring_end_idx - substring_start_idx;

  // Copy over the chars
  uint64_t string_new_offset = new_offsets[tid];
  memcpy(new_chars + string_new_offset, prev_chars + substring_start_idx, substring_length * sizeof(char));
}

std::tuple<char*, uint64_t*, uint64_t> PerformSubstring(char* char_data, uint64_t* str_indices, uint64_t num_chars, uint64_t num_strings, 
  uint64_t start_idx, uint64_t length) {

    // Get the write offsets
    int blocks_needed = (num_strings + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    uint64_t* updated_offsets = reinterpret_cast<uint64_t*>(callCudaMalloc<uint64_t>(num_strings + 1, 0));
    perform_substring<<<blocks_needed, THREADS_PER_BLOCK>>>(str_indices, updated_offsets, num_strings, start_idx, length);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Perform the prefix sum to get the updated offsets
    thrust::device_ptr<uint64_t> offsets_device_ptr(updated_offsets);
    uint64_t total_chars = thrust::reduce(offsets_device_ptr, offsets_device_ptr + num_strings, (uint64_t) 0, thrust::plus<uint64_t>());
    thrust::exclusive_scan(offsets_device_ptr, offsets_device_ptr + num_strings, offsets_device_ptr);
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Get the updated count
    cudaMemcpy(updated_offsets + num_strings, &total_chars, sizeof(uint64_t), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    // Create the chars buffer
    char* updated_chars = reinterpret_cast<char*>(callCudaMalloc<uint8_t>(total_chars, 0));
    substring_copy_chars<<<blocks_needed, THREADS_PER_BLOCK>>>(char_data, updated_chars, str_indices, updated_offsets, num_strings, start_idx, length);

    // Return the result
    return std::make_tuple(updated_chars, updated_offsets, total_chars);
}

}