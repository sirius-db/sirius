#include "operator/gpu_physical_string_matching.hpp"

#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <iostream> 
#include <string> 
#include <sstream>

namespace duckdb {

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32
#define CHARS_IN_BYTE 256
#define CHAR_INCREMENT 128
#define INITIAL_MEMORY_FACTOR 2.0
#define CHUNK_SIZE 8192
#define TILE_ITEMS_PER_TILE 10

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void determine_start_kernel(int* indicies, int num_strings, int* worker_start_term, int num_workers, int chunk_size, int last_char) {  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_workers) return; 

  int curr_chunk_start = min(tid * chunk_size, last_char);
  int search_start_term = 0; int search_end_term = num_strings;
  int curr_worker_start = -1; int curr_search_term;
  while(search_start_term <= search_end_term) {
    curr_search_term = (search_start_term + search_end_term)/2;

    // Determine if this workers chunk is in this terms range
    if(curr_chunk_start >= indicies[curr_search_term] && curr_chunk_start < indicies[curr_search_term + 1]) {
        curr_worker_start = curr_search_term;
        break;
    } else if(curr_chunk_start < indicies[curr_search_term]) {
        // The chunk starts before this term so search lower range
        search_end_term = curr_search_term - 1;
    } else {
        // The chunk starts after this term so serach an upper range
        search_start_term = curr_search_term + 1;
    }
  }

  worker_start_term[tid] = curr_worker_start;
}

__global__ void single_term_kmp_kernel(char* char_data, int* indicies, int* kmp_automato, int* worker_start_term, bool* results, 
int pattern_size, int num_workers, int chunk_size, int sub_chunk_size, int last_char, int num_strings) {
    
    // See if have any work to do
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_workers) return;

    const int curr_chunk_start = min(chunk_id * chunk_size, last_char);
    const int curr_chunk_end = min(curr_chunk_start + chunk_size + pattern_size, last_char);
    const int curr_sub_chunk_start = min(curr_chunk_start + threadIdx.x * sub_chunk_size, curr_chunk_end);
    const int curr_sub_chunk_end = min(curr_sub_chunk_start + sub_chunk_size, curr_chunk_end);

    // Determine the subchunk that the current string is going to be working on
    int curr_term = worker_start_term[chunk_id];
    while (curr_term < num_strings && (curr_sub_chunk_start < indicies[curr_term] || curr_sub_chunk_start >= indicies[curr_term + 1])) {
      curr_term++;
    }
    int curr_term_end = indicies[curr_term + 1];

    // Perform the actual string matching
    int j = 0; int curr_idx = 0; 
    #pragma unroll
    for(int i = curr_sub_chunk_start; i <= curr_sub_chunk_end; i++) {
        // See if we need to switch to a new term
        if(i >= curr_term_end) {
            curr_term = curr_term + 1;
            curr_term_end = indicies[curr_term + 1];
            j = 0; // Reset because we are at the start of the string
        }

        curr_idx = (int) char_data[i] + CHAR_INCREMENT;
        j = kmp_automato[j * CHARS_IN_BYTE + curr_idx];

        // Record that we have a hit
        if(j >= pattern_size) {
          results[curr_term] = true;
          j = 0;
        }
    }
}

__global__ void write_matching_rows(bool* results, int num_strings, uint64_t* matching_rows, uint64_t* num_match_rows) {
  int tile_size = gridDim.x * blockDim.x;
  int start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = start_idx; i < num_strings; i += tile_size) {
    if(results[i]) {
      uint64_t write_offset = atomicAdd(reinterpret_cast<unsigned long long int*>(num_match_rows), 
        static_cast<unsigned long long int>(1));
      matching_rows[write_offset] = static_cast<uint64_t>(i);
    }
  }
}

uint64_t* StringMatching(GPUColumn* string_column, std::string match_string, uint64_t* num_match_rows) {
  // Get the data from the metadata
  DataWrapper str_data_wrapper = string_column->data_wrapper;
  int num_chars = str_data_wrapper.size;
  char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
  int num_strings = str_data_wrapper.num_strings;
  int* d_str_indicies = reinterpret_cast<int*>(str_data_wrapper.offsets);
  int workers_needed = (num_chars + CHUNK_SIZE - 1)/CHUNK_SIZE;
  std::cout << "Running single term string matching for " << num_strings << " strings and " << num_chars << " chars using " << workers_needed << " workers" << std::endl;

  // Compute the automato for this string
  const int match_length = match_string.size();
  const char* match_char = match_string.c_str();
  int kmp_automato_size = match_length * CHARS_IN_BYTE;
  int* kmp_automato = new int[kmp_automato_size];
  std::memset(kmp_automato, 0, kmp_automato_size * sizeof(int));
  int first_idx = (int) match_char[0] + CHAR_INCREMENT;
  kmp_automato[first_idx] = 1;
  for(int X = 0, j = 1; j < match_length; j++) {
    int curr_idx = (int) match_char[j] + CHAR_INCREMENT;

    // Copy over the chars from the previous automato
    for(int c = 0; c < CHARS_IN_BYTE; c++) {
        kmp_automato[j * CHARS_IN_BYTE + c] = kmp_automato[X * CHARS_IN_BYTE + c];
    }
    kmp_automato[j * CHARS_IN_BYTE + curr_idx] = j + 1;
    X = kmp_automato[X * CHARS_IN_BYTE + curr_idx];
  }

  // Allocate the buffers we need
  char* d_match_str = reinterpret_cast<char*>(callCudaMalloc<uint8_t>(match_string.length() * sizeof(uint8_t), 0));
  int* d_kmp_automato = reinterpret_cast<int*>(callCudaMalloc<int>(kmp_automato_size * sizeof(int), 0));
  int* d_worker_start_term = reinterpret_cast<int*>(callCudaMalloc<int>(workers_needed * sizeof(int), 0));
  bool* d_answers = reinterpret_cast<bool*>(callCudaMalloc<bool>(num_strings * sizeof(bool), 0));
  uint64_t* d_matching_rows = reinterpret_cast<uint64_t*>(callCudaMalloc<uint64_t>(num_strings * sizeof(uint64_t), 0));
  uint64_t* d_num_match = reinterpret_cast<uint64_t*>(callCudaMalloc<uint64_t>(sizeof(uint64_t), 0));

  // Copy over the data to the buffers
  callCudaMemcpyHostToDevice<int>(d_kmp_automato, kmp_automato, kmp_automato_size * sizeof(int), 0);

  // Also set the initial values
  cudaMemset(d_matching_rows, 0, num_strings * sizeof(uint64_t));
  cudaMemset(d_num_match, 0, sizeof(uint64_t));
  
  // Set the start terms
  int last_char = num_chars - 1;
  int kernel_block_needed = (workers_needed + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  int block_sub_chunk_size = (CHUNK_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  std::cout << "Determined kernel blocks of " << kernel_block_needed << " and sub chunk size of " << block_sub_chunk_size << std::endl;
  determine_start_kernel<<<kernel_block_needed, THREADS_PER_BLOCK>>>(d_str_indicies, num_strings, d_worker_start_term, 
            workers_needed, CHUNK_SIZE, last_char);
  
  single_term_kmp_kernel<<<workers_needed, THREADS_PER_BLOCK>>>(d_char_data, d_str_indicies, d_kmp_automato, d_worker_start_term, 
    d_answers, match_length, workers_needed, CHUNK_SIZE, block_sub_chunk_size, last_char, num_strings);
  cudaDeviceSynchronize();
  cudaCheckErrors("Single Term String Matching Kernel");

  // Write the matching rows
  int num_match_blocks = std::max(1, (num_strings + THREADS_PER_BLOCK - 1)/(THREADS_PER_BLOCK * TILE_ITEMS_PER_TILE));
  write_matching_rows<<<num_match_blocks, THREADS_PER_BLOCK>>>(d_answers, num_strings, d_matching_rows, d_num_match);
  callCudaMemcpyDeviceToHost<uint64_t>(num_match_rows, d_num_match, sizeof(uint64_t), 0);

  // Check there are no errors
  cudaDeviceSynchronize();
  cudaCheckErrors("Single Term Result Writer");

  // Free the memory
  callCudaFree<uint8_t>(reinterpret_cast<uint8_t*>(d_match_str), 0);
  callCudaFree<int>(d_kmp_automato, 0);
  callCudaFree<int>(d_worker_start_term, 0);
  callCudaFree<bool>(d_answers, 0);
  callCudaFree<uint64_t>(d_num_match, 0);

  // Set the result values
  uint64_t first_id;
  cudaMemcpy(&first_id, d_matching_rows, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  return d_matching_rows;
}

__global__ void multi_term_kmp_kernel(char* char_data, int* indicies, int* kmp_automato, int* worker_start_term, 
  int* curr_term_answer, int* prev_term_answer, bool* found_term, int pattern_size, int num_workers, int chunk_size, int sub_chunk_size, 
  int last_char, int num_strings) {
    
    // See if have any work to do
    int chunk_id = blockIdx.x;
    if (chunk_id >= num_workers) return;

    const int curr_chunk_start = min(chunk_id * chunk_size, last_char); 
    const int curr_chunk_end = min(curr_chunk_start + chunk_size + pattern_size, last_char);
    const int curr_sub_chunk_start = min(curr_chunk_start + threadIdx.x * sub_chunk_size, curr_chunk_end);
    const int curr_sub_chunk_end = min(curr_sub_chunk_start + sub_chunk_size + pattern_size, curr_chunk_end);

    // Determine the subchunk that the current string is going to be working on
    int curr_term = worker_start_term[chunk_id];
    while (curr_term < num_strings && (curr_sub_chunk_start < indicies[curr_term] || curr_sub_chunk_start >= indicies[curr_term + 1])) {
      curr_term++;
    }
    int curr_term_end = indicies[curr_term + 1];

    // Perform the actual string matching
    int j = 0; int curr_idx = 0; 
    #pragma unroll
    for(int i = curr_sub_chunk_start; i <= curr_sub_chunk_end; i++) {
      // See if we need to switch to a new term
      if(i >= curr_term_end) {
          curr_term = curr_term + 1;
          curr_term_end = indicies[curr_term + 1];
          j = 0; // Reset because we are at the start of the string
      }

      curr_idx = (int) char_data[i] + CHAR_INCREMENT;
      j = kmp_automato[j * CHARS_IN_BYTE + curr_idx];

      // Record that we have a hit
      if(j >= pattern_size) {
        // Only write the result if we current match index is > than the lowest match index for the previous term
        if(i > prev_term_answer[curr_term]) {
          found_term[curr_term] = true;
          atomicMin(curr_term_answer + curr_term, i);
        }
        j = 0;
      }
    }
}

__global__ void update_term_answers(int* curr_term_answer, bool* found_term, int num_chars, int num_strings) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < num_strings && !found_term[tid]) {
      curr_term_answer[tid] = num_chars;
    }
}

__global__ void multi_write_matching_rows(int* curr_term_answer, int num_strings, int num_chars, uint64_t* matching_rows, uint64_t* num_match_rows) {
  int tile_size = gridDim.x * blockDim.x;
  int start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for(int i = start_idx; i < num_strings; i += tile_size) {
    if(curr_term_answer[i] < num_chars) {
      uint64_t write_offset = atomicAdd(reinterpret_cast<unsigned long long int*>(num_match_rows), 
        static_cast<unsigned long long int>(1));
      matching_rows[write_offset] = static_cast<uint64_t>(i);
    }
  }
}

uint64_t* MultiStringMatching(GPUColumn* string_column, std::vector<std::string> all_terms, uint64_t* num_match_rows) {
  // Get the data from the metadata
  DataWrapper str_data_wrapper = string_column->data_wrapper;
  int num_chars = str_data_wrapper.size;
  char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
  int num_strings = str_data_wrapper.num_strings;
  int* d_str_indicies = reinterpret_cast<int*>(str_data_wrapper.offsets);
  int workers_needed = (num_chars + CHUNK_SIZE - 1)/CHUNK_SIZE;

  // Create the automato for each term
  int num_terms = all_terms.size();
  int** all_terms_automato = new int*[num_terms];
  for(int i = 0; i < num_terms; i++) {
    std::string curr_term = all_terms[i];
    const int match_length = curr_term.size();
    const char* match_char = curr_term.c_str();
    int kmp_automato_size = match_length * CHARS_IN_BYTE;
    int* kmp_automato = new int[kmp_automato_size];
    std::memset(kmp_automato, 0, kmp_automato_size * sizeof(int));

    // Create the automato for this term
    int first_idx = (int) match_char[0] + CHAR_INCREMENT;
    kmp_automato[first_idx] = 1;
    for(int X = 0, j = 1; j < match_length; j++) {
      int curr_idx = (int) match_char[j] + CHAR_INCREMENT;
      for(int c = 0; c < CHARS_IN_BYTE; c++) {
        kmp_automato[j * CHARS_IN_BYTE + c] = kmp_automato[X * CHARS_IN_BYTE + c];
      }
      kmp_automato[j * CHARS_IN_BYTE + curr_idx] = j + 1;
      X = kmp_automato[X * CHARS_IN_BYTE + curr_idx];
    }

    // Save the automato for this term
    all_terms_automato[i] = kmp_automato;
  }

  // Allocate the buffers on the GPU 
  int* d_worker_start_term = reinterpret_cast<int*>(callCudaMalloc<int>(workers_needed * sizeof(int), 0));
  int* d_prev_term_answers = reinterpret_cast<int*>(callCudaMalloc<int>(num_strings * sizeof(int), 0));
  int* d_answer_idxs = reinterpret_cast<int*>(callCudaMalloc<int>(num_strings * sizeof(int), 0));
  bool* d_found_answer = reinterpret_cast<bool*>(callCudaMalloc<bool>(num_strings * sizeof(bool), 0));
  uint64_t* d_matching_rows = reinterpret_cast<uint64_t*>(callCudaMalloc<uint64_t>(num_strings * sizeof(uint64_t), 0));
  uint64_t* d_num_match = reinterpret_cast<uint64_t*>(callCudaMalloc<uint64_t>(sizeof(uint64_t), 0));

  // Create buffer for each automato
  int** d_all_automatos = new int*[num_terms];
  for(int i = 0; i < num_terms; i++) {
    int kmp_automato_size = all_terms[i].size() * CHARS_IN_BYTE;
    d_all_automatos[i] = reinterpret_cast<int*>(callCudaMalloc<int>(kmp_automato_size * sizeof(int), 0));
  }

  // Copy over the necessary data 
  cudaMemcpy(d_prev_term_answers, d_str_indicies, num_strings * sizeof(int), cudaMemcpyDeviceToDevice);
  for(int i = 0; i < num_terms; i++) {
    int kmp_automato_size = all_terms[i].size() * CHARS_IN_BYTE;
    callCudaMemcpyHostToDevice<int>(d_all_automatos[i], all_terms_automato[i], kmp_automato_size * sizeof(int), 0);
  }

  // Initialize the other buffers
  cudaMemset(d_matching_rows, 0, num_strings * sizeof(uint64_t));
  cudaMemset(d_num_match, 0, sizeof(uint64_t));

  // Determine the start offset for each kernel
  int last_char = num_chars - 1;
  int kernel_block_needed = (workers_needed + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  int block_sub_chunk_size = (CHUNK_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  determine_start_kernel<<<kernel_block_needed, THREADS_PER_BLOCK>>>(d_str_indicies, num_strings, d_worker_start_term, 
            workers_needed, CHUNK_SIZE, last_char);
  
  // Perform the string matching term by term
  int post_process_num_blocks = (num_strings + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  for(int i = 0; i < num_terms; i++) {
    // Determine the current terms variables
    int curr_term_length = all_terms[i].size();
    int* curr_term_automato = d_all_automatos[i]; 

    // Reset the hit buffer
    cudaMemset(d_found_answer, 0, num_strings * sizeof(bool));

    // Run the search
    multi_term_kmp_kernel<<<workers_needed, THREADS_PER_BLOCK>>>(d_char_data, d_str_indicies, curr_term_automato, d_worker_start_term, 
      d_answer_idxs, d_prev_term_answers, d_found_answer, curr_term_length, workers_needed, CHUNK_SIZE, block_sub_chunk_size, 
      last_char, num_strings);

    // Perform post processing
    update_term_answers<<<post_process_num_blocks, THREADS_PER_BLOCK>>>(d_answer_idxs, d_found_answer, num_chars, num_strings);

    // If there are future terms, the make the current answer the prev term answers
    if(i < (num_terms - 1)) {
        int* temp_ptr = d_answer_idxs;
        d_answer_idxs = d_prev_term_answers;
        d_prev_term_answers = temp_ptr;
    }
  }

  // Write the matching rows
  int num_match_blocks = std::max(1, (num_strings + THREADS_PER_BLOCK - 1)/(THREADS_PER_BLOCK * TILE_ITEMS_PER_TILE));
  multi_write_matching_rows<<<num_match_blocks, THREADS_PER_BLOCK>>>(d_answer_idxs, num_strings, num_chars, d_matching_rows, d_num_match);
  callCudaMemcpyDeviceToHost<uint64_t>(num_match_rows, d_num_match, sizeof(uint64_t), 0);
  std::cout << "MULTI RESULT: Got num matching rows of " << num_match_rows[0] << std::endl;

  // Check for errors
  cudaDeviceSynchronize();
  cudaCheckErrors("Multi Term String Matching Kernel");

  // Cleanup the allocated GPU memory
  callCudaFree<int>(d_worker_start_term, 0);
  callCudaFree<int>(d_prev_term_answers, 0);
  callCudaFree<int>(d_answer_idxs, 0);
  callCudaFree<bool>(d_found_answer, 0);
  for(int i = 0; i < num_terms; i++) {
    callCudaFree<int>(d_all_automatos[i], 0);
  }

  // Clean up allocated cpu memory
  delete[] d_all_automatos;
  for(int i = 0; i < num_terms; i++) {
    delete[] all_terms_automato[i];
  }
  delete[] all_terms_automato;

  // Set the return values
  std::cout << "Finished multi term string matching" << std::endl;
  return d_matching_rows;
}

__global__ void perform_substring(int* prev_offsets, int* new_offsets, int num_strings, int start_idx, int length) {
  // Get which string this thread workers on
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_strings) return; 

  // Determine the length of this substring
  int curr_str_start_idx = prev_offsets[tid]; int curr_str_end_idx = prev_offsets[tid + 1];
  int substring_start_idx = min(curr_str_start_idx + start_idx, curr_str_end_idx);
  int substring_end_idx = min(substring_start_idx + length, curr_str_end_idx);
  int substring_length = substring_end_idx - substring_start_idx;
  new_offsets[tid] = substring_length;
}

__global__ void substring_copy_chars(char* prev_chars, char* new_chars, int* prev_offsets, int* new_offsets, int num_strings, int start_idx, int length) {
  // Get which string this thread workers on
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_strings) return; 

  // Determine the range for this substring
  int curr_str_start_idx = prev_offsets[tid]; int curr_str_end_idx = prev_offsets[tid + 1];
  int substring_start_idx = min(curr_str_start_idx + start_idx, curr_str_end_idx);
  int substring_end_idx = min(substring_start_idx + length, curr_str_end_idx);
  int substring_length = substring_end_idx - substring_start_idx;

  // Copy over the chars
  int string_new_offset = new_offsets[tid];
  memcpy(new_chars + string_new_offset, prev_chars + substring_start_idx, substring_length * sizeof(char));
}

void PerformSubstring(GPUColumn* string_column, int start_idx, int length) {
  // Get the data from the metadata
  DataWrapper str_data_wrapper = string_column->data_wrapper;
  int num_chars = str_data_wrapper.size;
  char* d_char_data = reinterpret_cast<char*>(str_data_wrapper.data);
  int num_strings = str_data_wrapper.num_strings;
  int* d_str_indicies = reinterpret_cast<int*>(str_data_wrapper.offsets);
  int blocks_needed = (num_strings + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK;
  std::cout << "PerformSubstring got values: " << start_idx << "," << length << "," << num_chars << "," << num_strings << "," << blocks_needed << std::endl;

  // Get the write offsets
  int* d_updated_offsets = reinterpret_cast<int*>(callCudaMalloc<int>((num_strings + 1) * sizeof(int), 0));
  perform_substring<<<blocks_needed, THREADS_PER_BLOCK>>>(d_str_indicies, d_updated_offsets, num_strings, start_idx, length);
  cudaDeviceSynchronize();
  cudaCheckErrors("Initial substring calculation");
  std::cout << "perform_substring finished execution" << std::endl;

  // Perform the prefix sum and get the total number of chars
  thrust::device_ptr<int> offsets_device_ptr(d_updated_offsets);
  int total_chars = thrust::reduce(offsets_device_ptr, offsets_device_ptr + num_strings, 0, thrust::plus<int>());
  thrust::exclusive_scan(offsets_device_ptr, offsets_device_ptr + num_strings, offsets_device_ptr);
  
  // Create the chears buffer
  char* d_updated_chars = reinterpret_cast<char*>(callCudaMalloc<uint8_t>(total_chars * sizeof(char), 0));
  substring_copy_chars<<<blocks_needed, THREADS_PER_BLOCK>>>(d_char_data, d_updated_chars, d_str_indicies, d_updated_offsets, num_strings, start_idx, length);

  // Write the updated metadata to the column
  string_column->data_wrapper.data = reinterpret_cast<uint8_t*>(d_updated_chars);
  string_column->data_wrapper.size = total_chars;
  string_column->data_wrapper.offsets = d_updated_offsets;
  std::cout << "After substring got a total chars of " << total_chars << std::endl;
}

}