#include "cuda_helper.cuh"
#include "gpu_physical_strings_matching.hpp"
#include "gpu_buffer_manager.hpp"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#define THREADS_PER_BLOCK_STRINGS 512
#define WARP_SIZE 32
#define CHARS_IN_BYTE 256
#define CHAR_INCREMENT 128
#define INITIAL_MEMORY_FACTOR 2.0
#define CHUNK_SIZE 8192
#define TILE_ITEMS_PER_TILE 10

namespace duckdb {

__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

__global__ void determine_start_kernel(uint64_t* indices, uint64_t num_strings, uint64_t* worker_start_term, uint64_t num_workers, uint64_t chunk_size, uint64_t last_char) {  
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_workers) return; 

  uint64_t curr_chunk_start = min(tid * chunk_size, last_char);
  uint64_t search_start_term = 0; uint64_t search_end_term = num_strings;
  uint64_t curr_worker_start = 0; // TODO: CHECK IT'S OKAY TO INITIALIZE IT TO 0
  // int curr_worker_start = -1;
  uint64_t curr_search_term;
  while(search_start_term <= search_end_term) {
    curr_search_term = (search_start_term + search_end_term)/2;

    // Determine if this workers chunk is in this terms range
    if(curr_chunk_start >= indices[curr_search_term] && curr_chunk_start < indices[curr_search_term + 1]) {
        curr_worker_start = curr_search_term;
        break;
    } else if(curr_chunk_start < indices[curr_search_term]) {
        // The chunk starts before this term so search lower range
        search_end_term = curr_search_term - 1;
    } else {
        // The chunk starts after this term so serach an upper range
        search_start_term = curr_search_term + 1;
    }
  }

  worker_start_term[tid] = curr_worker_start;
}

__global__ void single_term_kmp_kernel(char* char_data, uint64_t* indices, int* kmp_automato, uint64_t* worker_start_term, bool* results, 
uint64_t pattern_size, uint64_t num_workers, uint64_t chunk_size, uint64_t sub_chunk_size, uint64_t last_char, uint64_t num_strings) {
    
    // See if have any work to do
    uint64_t chunk_id = blockIdx.x;
    int worker_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (chunk_id >= num_workers) return;

    const uint64_t curr_chunk_start = min(chunk_id * chunk_size, last_char);
    const uint64_t curr_chunk_end = min(curr_chunk_start + chunk_size + pattern_size, last_char);
    const uint64_t curr_sub_chunk_start = min(curr_chunk_start + threadIdx.x * sub_chunk_size, curr_chunk_end);
    const uint64_t curr_sub_chunk_end = min(curr_sub_chunk_start + sub_chunk_size + pattern_size, curr_chunk_end);

    // Determine the subchunk that the current string is going to be working on
    uint64_t curr_term = worker_start_term[chunk_id];
    while (curr_term < num_strings && (curr_sub_chunk_start < indices[curr_term] || curr_sub_chunk_start >= indices[curr_term + 1])) {
      curr_term++;
    }
    uint64_t curr_term_end = indices[curr_term + 1];

    // Perform the actual string matching
    int j = 0; int curr_idx = 0; 
    #pragma unroll
    for(uint64_t i = curr_sub_chunk_start; i <= curr_sub_chunk_end; i++) {
        // See if we need to switch to a new term
        if(i >= curr_term_end) {
          curr_term = curr_term + 1;
          curr_term_end = indices[curr_term + 1];
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

__global__ void write_matching_rows(bool* results, uint64_t num_strings, uint64_t* matching_rows, uint64_t* count) {
  uint64_t tile_size = gridDim.x * blockDim.x;
  uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t i = start_idx; i < num_strings; i += tile_size) {
    if(results[i]) {
      uint64_t write_offset = atomicAdd(reinterpret_cast<unsigned long long int*>(count), 1);
      matching_rows[write_offset] = i;
    }
  }
}

__global__ void print_matching_rows(uint64_t* indices, uint64_t total_strings, uint64_t strings_to_print, bool is_ascending) {
  // Get the ptr and the length of the current string
  printf("STRING MATCHING ROWS: ");
  for(int i = 0; i < strings_to_print; i++) {
    uint64_t curr_idx = is_ascending ? i : (total_strings - 1 - i);
    printf("%d ", (int) (indices[curr_idx] + 1));
  }
  printf("\n");
}

void StringMatching(char* char_data, uint64_t* str_indices, std::string match_string, uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings) {
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  if (num_strings == 0) {
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    h_count[0] = 0;
    count = h_count;
    return;
  }

  // Get the data from the metadata
  uint64_t workers_needed = (num_chars + CHUNK_SIZE - 1)/CHUNK_SIZE;

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
  char* d_match_str = gpuBufferManager->customCudaMalloc<char>(match_string.length(), 0, 0);
  int* d_kmp_automato = gpuBufferManager->customCudaMalloc<int>(kmp_automato_size, 0, 0);
  uint64_t* d_worker_start_term = gpuBufferManager->customCudaMalloc<uint64_t>(workers_needed, 0, 0);
  bool* d_answers = reinterpret_cast<bool*> (gpuBufferManager->customCudaMalloc<uint8_t>(num_strings, 0, 0));
  // TODO: Do it twice for more accurate allocation
  uint64_t* d_matching_rows = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);

  // Copy over the data to the buffers
  cudaMemcpy(d_kmp_automato, kmp_automato, kmp_automato_size * sizeof(int), cudaMemcpyHostToDevice);

  // Also set the initial values
  cudaMemset(d_matching_rows, 0, num_strings * sizeof(uint64_t));
  cudaMemset(count, 0, sizeof(uint64_t));
  CHECK_ERROR();
  
  // Set the start terms
  uint64_t last_char = num_chars - 1;
  uint64_t preprocess_blocks_needed = (workers_needed + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  std::cout << "Sirius running preprocessing for " << workers_needed << " workers with " << num_strings << " strings and " << num_chars << " chars" << std::endl;

  auto preprocessing_start = std::chrono::high_resolution_clock::now();
  determine_start_kernel<<<preprocess_blocks_needed, THREADS_PER_BLOCK_STRINGS>>>(str_indices, num_strings, d_worker_start_term, 
            workers_needed, CHUNK_SIZE, last_char);
  cudaDeviceSynchronize();
  auto preprocessing_end = std::chrono::high_resolution_clock::now();
  int preprocessing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(preprocessing_end - preprocessing_start).count();
  std::cout << "String matching preprocessing took " << preprocessing_time_us/1000.0 << " ms" << std::endl;

  auto str_match_start = std::chrono::high_resolution_clock::now();
  uint64_t block_sub_chunk_size = (CHUNK_SIZE + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  single_term_kmp_kernel<<<workers_needed, THREADS_PER_BLOCK_STRINGS>>>(char_data, str_indices, d_kmp_automato, d_worker_start_term, 
    d_answers, match_length, workers_needed, CHUNK_SIZE, block_sub_chunk_size, last_char, num_strings);
  cudaDeviceSynchronize();
  auto str_match_end = std::chrono::high_resolution_clock::now();
  int str_match_time_us = std::chrono::duration_cast<std::chrono::microseconds>(str_match_end - str_match_start).count();
  std::cout << "Actual String matching took " << str_match_time_us/1000.0 << " ms" << std::endl;

  // Create a buffer of the valid idxs
  auto valid_rows_start_time = std::chrono::high_resolution_clock::now();
  uint64_t* d_valid_idxs = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);

  // First get the indices that have true
  thrust::device_ptr<bool> d_answers_ptr(d_answers);
  thrust::device_ptr<uint64_t> d_valid_idxs_ptr(d_valid_idxs);
  auto end = thrust::copy_if(
    thrust::counting_iterator<uint64_t>(0),
    thrust::counting_iterator<uint64_t>(num_strings),
    d_answers_ptr,
    d_valid_idxs_ptr,
    thrust::identity<bool>()
  );

  // Record the number of valid strings
  uint64_t num_valid_strings = end - d_valid_idxs_ptr;
  uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
  cudaMemcpy(h_count, &num_valid_strings, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // Check there are no errors
  cudaDeviceSynchronize();
  CHECK_ERROR();

  row_id = d_valid_idxs;
  count = h_count;
  printf("Finished single term string matching\n");
}

__global__ void multi_term_kmp_kernel(char* char_data, uint64_t* indices, int* kmp_automato, uint64_t* worker_start_term, 
  uint64_t* curr_term_answer, uint64_t* prev_term_answer, bool* found_term, int pattern_size, uint64_t num_workers, uint64_t chunk_size, uint64_t sub_chunk_size, 
  uint64_t last_char, uint64_t num_strings) {
    
    // See if have any work to do
    uint64_t chunk_id = blockIdx.x;
    if (chunk_id >= num_workers) return;

    const uint64_t curr_chunk_start = min(chunk_id * chunk_size, last_char); 
    const uint64_t curr_chunk_end = min(curr_chunk_start + chunk_size + pattern_size, last_char);
    const uint64_t curr_sub_chunk_start = min(curr_chunk_start + threadIdx.x * sub_chunk_size, curr_chunk_end);
    const uint64_t curr_sub_chunk_end = min(curr_sub_chunk_start + sub_chunk_size + pattern_size, curr_chunk_end);

    // Determine the subchunk that the current string is going to be working on
    uint64_t curr_term = worker_start_term[chunk_id];
    while (curr_term < num_strings && (curr_sub_chunk_start < indices[curr_term] || curr_sub_chunk_start >= indices[curr_term + 1])) {
      curr_term++;
    }
    uint64_t curr_term_end = indices[curr_term + 1];

    // Perform the actual string matching
    int j = 0; int curr_idx = 0; 
    #pragma unroll
    for(uint64_t i = curr_sub_chunk_start; i <= curr_sub_chunk_end; i++) {
      // See if we need to switch to a new term
      if(i >= curr_term_end) {
          curr_term = curr_term + 1;
          curr_term_end = indices[curr_term + 1];
          j = 0; // Reset because we are at the start of the string
      }

      curr_idx = (int) char_data[i] + CHAR_INCREMENT;
      j = kmp_automato[j * CHARS_IN_BYTE + curr_idx];

      // Record that we have a hit
      if(j >= pattern_size) {
        // Only write the result if we current match index is > than the lowest match index for the previous term
        if(i > prev_term_answer[curr_term]) {
          found_term[curr_term] = true;
          atomicMin(reinterpret_cast<unsigned long long int*> (curr_term_answer + curr_term), static_cast<unsigned long long int> (i));
        }
        j = 0;
      }
    }
}

__global__ void update_term_answers(uint64_t* curr_term_answer, bool* found_term, uint64_t num_chars, uint64_t num_strings) {
    uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < num_strings && !found_term[tid]) {
      curr_term_answer[tid] = num_chars;
    }
}

__global__ void multi_write_matching_rows(uint64_t* curr_term_answer, uint64_t num_strings, uint64_t num_chars, uint64_t* matching_rows, uint64_t* count) {
  uint64_t tile_size = gridDim.x * blockDim.x;
  uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  for(uint64_t i = start_idx; i < num_strings; i += tile_size) {
    if(curr_term_answer[i] < num_chars) {
      uint64_t write_offset = atomicAdd(reinterpret_cast<unsigned long long int*>(count), 
        static_cast<unsigned long long int>(1));
      matching_rows[write_offset] = static_cast<uint64_t>(i);
    }
  }
}

void MultiStringMatching(char* char_data, uint64_t* str_indices, std::vector<std::string> all_terms,
       uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings) {
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  if (num_strings == 0) {
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    h_count[0] = 0;
    count = h_count;
    return;
  }
  // Get the data from the metadata
  uint64_t workers_needed = (num_chars + CHUNK_SIZE - 1)/CHUNK_SIZE;

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
  uint64_t* d_worker_start_term = gpuBufferManager->customCudaMalloc<uint64_t>(workers_needed, 0, 0);
  uint64_t* d_prev_term_answers = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);
  uint64_t* d_answer_idxs = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);
  bool* d_found_answer = reinterpret_cast<bool*> (gpuBufferManager->customCudaMalloc<uint8_t>(num_strings, 0, 0));
  uint64_t* d_matching_rows = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);

  // Create buffer for each automato
  int** d_all_automatos = new int*[num_terms];
  for(int i = 0; i < num_terms; i++) {
    int kmp_automato_size = all_terms[i].size() * CHARS_IN_BYTE;
    d_all_automatos[i] = gpuBufferManager->customCudaMalloc<int>(kmp_automato_size * sizeof(int), 0, 0);
  }

  // Copy over the necessary data 
  cudaMemcpy(d_prev_term_answers, str_indices, num_strings * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
  for(int i = 0; i < num_terms; i++) {
    int kmp_automato_size = all_terms[i].size() * CHARS_IN_BYTE;
    cudaMemcpy(d_all_automatos[i], all_terms_automato[i], kmp_automato_size * sizeof(int), cudaMemcpyHostToDevice);
  }

  // Initialize the other buffers
  cudaMemset(d_matching_rows, 0, num_strings * sizeof(uint64_t));
  cudaMemset(count, 0, sizeof(uint64_t));
  CHECK_ERROR();

  // Determine the start offset for each kernel
  uint64_t last_char = num_chars - 1;
  uint64_t kernel_block_needed = (workers_needed + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  uint64_t block_sub_chunk_size = (CHUNK_SIZE + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  determine_start_kernel<<<kernel_block_needed, THREADS_PER_BLOCK_STRINGS>>>(str_indices, num_strings, d_worker_start_term, 
            workers_needed, CHUNK_SIZE, last_char);
  CHECK_ERROR();
  
  // Perform the string matching term by term
  uint64_t post_process_num_blocks = (num_strings + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  for(int i = 0; i < num_terms; i++) {
    // Determine the current terms variables
    int curr_term_length = all_terms[i].size();
    int* curr_term_automato = d_all_automatos[i]; 

    // Reset the hit buffer
    cudaMemset(d_found_answer, 0, num_strings * sizeof(bool));

    // Run the search
    multi_term_kmp_kernel<<<workers_needed, THREADS_PER_BLOCK_STRINGS>>>(char_data, str_indices, curr_term_automato, d_worker_start_term, 
      d_answer_idxs, d_prev_term_answers, d_found_answer, curr_term_length, workers_needed, CHUNK_SIZE, block_sub_chunk_size, 
      last_char, num_strings);
    CHECK_ERROR();

    // Perform post processing
    update_term_answers<<<post_process_num_blocks, THREADS_PER_BLOCK_STRINGS>>>(d_answer_idxs, d_found_answer, num_chars, num_strings);
    CHECK_ERROR();

    // If there are future terms, the make the current answer the prev term answers
    if(i < (num_terms - 1)) {
        uint64_t* temp_ptr = d_answer_idxs;
        d_answer_idxs = d_prev_term_answers;
        d_prev_term_answers = temp_ptr;
    }
  }

  // Write the matching rows
  uint64_t num_match_blocks = std::max((uint64_t) 1, (num_strings + THREADS_PER_BLOCK_STRINGS - 1)/(THREADS_PER_BLOCK_STRINGS * TILE_ITEMS_PER_TILE));
  multi_write_matching_rows<<<num_match_blocks, THREADS_PER_BLOCK_STRINGS>>>(d_answer_idxs, num_strings, num_chars, d_matching_rows, count);
  std::cout << "MULTI RESULT: Got num matching rows of " << count[0] << std::endl;
  CHECK_ERROR();

  uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
  cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // Check for errors
  cudaDeviceSynchronize();
  CHECK_ERROR();

  // Set the return values
  std::cout << "Finished multi term string matching" << std::endl;
  row_id = d_matching_rows;
  count = h_count;
  // return d_matching_rows;
}

} // namespace duckdb
