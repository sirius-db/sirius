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
#include "gpu_physical_substring.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_columns.hpp"
#include "log/logging.hpp"

#define THREADS_PER_BLOCK 512

namespace duckdb {

//----------Kernels----------//
template<typename OffsetT, typename SizeT>
__global__ void get_new_length(const OffsetT* prev_offsets, OffsetT* new_len, SizeT num_strings, OffsetT start_idx, OffsetT length) {
  // Get which string this thread workers on
  OffsetT tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_strings) return; 

  // Determine the length of this substring
  OffsetT curr_str_start_idx = prev_offsets[tid]; OffsetT curr_str_end_idx = prev_offsets[tid + 1];
  OffsetT substring_start_idx = min(curr_str_start_idx + start_idx, curr_str_end_idx);
  OffsetT substring_end_idx = min(substring_start_idx + length, curr_str_end_idx);
  OffsetT substring_length = substring_end_idx - substring_start_idx;
  new_len[tid] = substring_length;
}

template<typename OffsetT, typename SizeT>
__global__ void substring_copy_chars(const char* prev_chars, char* new_chars, const OffsetT* prev_offsets, OffsetT* new_offsets, SizeT num_strings, 
    OffsetT start_idx, OffsetT length) {

  // Get which string this thread workers on
  OffsetT tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_strings) return; 

  // Determine the range for this substring
  OffsetT curr_str_start_idx = prev_offsets[tid]; OffsetT curr_str_end_idx = prev_offsets[tid + 1];
  OffsetT substring_start_idx = min(curr_str_start_idx + start_idx, curr_str_end_idx);
  OffsetT substring_end_idx = min(substring_start_idx + length, curr_str_end_idx);
  OffsetT substring_length = substring_end_idx - substring_start_idx;

  // Copy over the chars
  OffsetT string_new_offset = new_offsets[tid];
  memcpy(new_chars + string_new_offset, prev_chars + substring_start_idx, substring_length * sizeof(char));
}

// Instantiations
template __global__ void get_new_length<uint64_t, uint64_t>(const uint64_t* prev_offsets,
                                                            uint64_t* new_len,
                                                            uint64_t num_strings,
                                                            uint64_t start_idx,
                                                            uint64_t length);
template __global__ void get_new_length<int64_t, cudf::size_type>(const int64_t* prev_offsets,
                                                                  int64_t* new_len,
                                                                  cudf::size_type num_strings,
                                                                  int64_t start_idx,
                                                                  int64_t length);
template __global__ void substring_copy_chars<uint64_t, uint64_t>(const char* prev_chars,
                                                                  char* new_chars,
                                                                  const uint64_t* prev_offsets,
                                                                  uint64_t* new_offsets,
                                                                  uint64_t num_strings,
                                                                  uint64_t start_idx,
                                                                  uint64_t length);
template __global__ void substring_copy_chars<int64_t, cudf::size_type>(const char* prev_chars,
                                                                        char* new_chars,
                                                                        const int64_t* prev_offsets,
                                                                        int64_t* new_offsets,
                                                                        cudf::size_type num_strings,
                                                                        int64_t start_idx,
                                                                        int64_t length);

//----------Kernel Wrappers----------//
std::tuple<char*, uint64_t*, uint64_t> PerformSubstring(char* char_data, uint64_t* str_indices, uint64_t num_chars, uint64_t num_strings, 
  uint64_t start_idx, uint64_t length) {
    CHECK_ERROR();
    if (num_strings == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        char* empty = nullptr;
        uint64_t* empty_offset = nullptr;
        return std::make_tuple(empty, empty_offset, 0);
    }
    SIRIUS_LOG_DEBUG("Launching substring kernel");
    SETUP_TIMING();
    START_TIMER();

    // Get the write offsets
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t blocks_needed = (num_strings + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
    uint64_t* new_len = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings + 1, 0, 0);
    uint64_t* result_offset = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings + 1, 0, 0);
    cudaMemset(new_len + num_strings, 0, sizeof(uint64_t));
    get_new_length<<<blocks_needed, THREADS_PER_BLOCK>>>(str_indices, new_len, num_strings, start_idx, length);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    //cub scan
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, new_len, result_offset, num_strings + 1);

    // Allocate temporary storage for exclusive prefix sum
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, new_len, result_offset, num_strings + 1);
    CHECK_ERROR();

    cudaDeviceSynchronize();
    CHECK_ERROR();

    uint64_t* total_chars = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    // Get the updated count
    cudaMemcpy(total_chars, result_offset + num_strings, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();

    // Create the chars buffer
    char* updated_chars = gpuBufferManager->customCudaMalloc<char>(total_chars[0], 0, 0);
    substring_copy_chars<<<blocks_needed, THREADS_PER_BLOCK>>>(char_data, updated_chars, str_indices, result_offset, num_strings, start_idx, length);

    // Return the result
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(new_len), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
    STOP_TIMER();
    return std::make_tuple(updated_chars, result_offset, total_chars[0]);
}

//----------Substring for CuDF compatibility----------//
// Macro to simplify kernel launch syntax
#define LAUNCH_KERNEL(K, T1, T2, N, S)                                                             \
  K<T1, T2><<<cuda::ceil_div((N), THREADS_PER_BLOCK), THREADS_PER_BLOCK, 0, S>>>

// This is a replication of PerformSubstring() above for compatibility with CuDF
// The key points are 1) avoid conversion back and forth from cudf::size_type and uint64_t
//                    2) manage memory with rmm::device_uvector, so ownership can be transferred
//                       to cudf::columns
std::unique_ptr<cudf::column> DoSubstring(const char* input_data,
                                          cudf::size_type input_count,
                                          const int64_t* input_offsets,
                                          int64_t start_idx,
                                          int64_t length,
                                          rmm::device_async_resource_ref mr,
                                          rmm::cuda_stream_view stream)
{
  static_assert(std::is_same_v<int32_t, cudf::size_type>); // Sanity check

  auto offset_count = input_count + 1;

  // Allocate temporary string length and output offsets buffer
  rmm::device_uvector<int64_t> temp_string_lengths(offset_count, stream, mr);
  rmm::device_uvector<int64_t> output_offsets(offset_count, stream, mr);
  // Set the last string length to 0, so that the exclusive scan places the total sum at the end
  CUDF_CUDA_TRY(cudaMemset(temp_string_lengths.data() + input_count, 0, sizeof(int64_t)));

  // Compute the new lengths
  LAUNCH_KERNEL(get_new_length, int64_t, cudf::size_type, input_count, stream)
  (input_offsets, temp_string_lengths.data(), input_count, start_idx, length);

  // Compute the new offsets
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr,
                                temp_storage_bytes,
                                temp_string_lengths.data(),
                                output_offsets.data(),
                                offset_count,
                                stream);
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream, mr);
  cub::DeviceScan::ExclusiveSum(d_temp_storage.data(),
                                temp_storage_bytes,
                                temp_string_lengths.data(),
                                output_offsets.data(),
                                offset_count,
                                stream);
  const auto output_bytes = output_offsets.back_element(stream);

  // Allocate output data buffer
  rmm::device_uvector<char> output_data(output_bytes, stream, mr);

  // Copy the substring data
  LAUNCH_KERNEL(substring_copy_chars, int64_t, cudf::size_type, input_count, stream)
  (input_data,
   output_data.data(),
   input_offsets,
   output_offsets.data(),
   input_count,
   start_idx,
   length);

  // Return a cudf::column
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                                    offset_count,
                                                    output_offsets.release(),
                                                    rmm::device_buffer{0, stream, mr},
                                                    0);
  return cudf::make_strings_column(input_count,
                                   std::move(offsets_col),
                                   output_data.release(),
                                   0,
                                   rmm::device_buffer{0, stream, mr});
}

#undef LAUNCH_KERNEL

}