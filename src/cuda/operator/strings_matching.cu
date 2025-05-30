#include "cuda_helper.cuh"
#include "gpu_physical_strings_matching.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

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

//--------------------------------------------------//
// String Matching
//--------------------------------------------------//
template<typename IdxT>
__global__ void determine_start_kernel(const IdxT* indices, IdxT num_strings, IdxT* worker_start_term, IdxT num_workers, IdxT chunk_size, IdxT last_char) {  
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= num_workers) return; 

  IdxT curr_chunk_start = min(tid * chunk_size, last_char);
  IdxT search_start_term = 0; IdxT search_end_term = num_strings;
  IdxT curr_worker_start = 0; // TODO: CHECK IT'S OKAY TO INITIALIZE IT TO 0
  // int curr_worker_start = -1;
  IdxT curr_search_term;
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
        // The chunk starts after this term so search an upper range
        search_start_term = curr_search_term + 1;
    }
  }

  worker_start_term[tid] = curr_worker_start;
}

// Instantiations
template __global__ void determine_start_kernel<uint64_t>(const uint64_t* indices,
                                                          uint64_t num_strings,
                                                          uint64_t* worker_start_term,
                                                          uint64_t num_workers,
                                                          uint64_t chunk_size,
                                                          uint64_t last_char);
template __global__ void determine_start_kernel<cudf::size_type>(const cudf::size_type* indices,
                                                                 cudf::size_type num_strings,
                                                                 cudf::size_type* worker_start_term,
                                                                 cudf::size_type num_workers,
                                                                 cudf::size_type chunk_size,
                                                                 cudf::size_type last_char);

template<typename IdxT>
__global__ void single_term_kmp_kernel(const char* char_data, const IdxT* indices, const int* kmp_automato, const IdxT* worker_start_term, bool* results, 
IdxT pattern_size, IdxT num_workers, IdxT chunk_size, IdxT sub_chunk_size, IdxT last_char, IdxT num_strings) {
    
    // See if have any work to do
    auto chunk_id = blockIdx.x;
    auto worker_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (chunk_id >= num_workers) return;

    const auto curr_chunk_start = min(chunk_id * chunk_size, last_char);
    const auto curr_chunk_end = min(curr_chunk_start + chunk_size + pattern_size, last_char);
    const auto curr_sub_chunk_start = min(curr_chunk_start + threadIdx.x * sub_chunk_size, curr_chunk_end);
    const auto curr_sub_chunk_end = min(curr_sub_chunk_start + sub_chunk_size + pattern_size, curr_chunk_end);

    // Determine the subchunk that the current string is going to be working on
    auto curr_term = worker_start_term[chunk_id];
    while (curr_term < num_strings && (curr_sub_chunk_start < indices[curr_term] || curr_sub_chunk_start >= indices[curr_term + 1])) {
      curr_term++;
    }
    auto curr_term_end = indices[curr_term + 1];

    // Perform the actual string matching
    int j = 0; int curr_idx = 0; 
    #pragma unroll
    for(int i = curr_sub_chunk_start; i <= curr_sub_chunk_end; i++) {
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

// Instantiations
template __global__ void single_term_kmp_kernel<uint64_t>(const char* char_data,
                                                          const uint64_t* indices,
                                                          const int* kmp_automato,
                                                          const uint64_t* worker_start_term,
                                                          bool* results,
                                                          uint64_t pattern_size,
                                                          uint64_t num_workers,
                                                          uint64_t chunk_size,
                                                          uint64_t sub_chunk_size,
                                                          uint64_t last_char,
                                                          uint64_t num_strings);
template __global__ void single_term_kmp_kernel<cudf::size_type>(const char* char_data,
                                                                 const cudf::size_type* indices,
                                                                 const int* kmp_automato,
                                                                 const cudf::size_type* worker_start_term,
                                                                 bool* results,
                                                                 cudf::size_type pattern_size,
                                                                 cudf::size_type num_workers,
                                                                 cudf::size_type chunk_size,
                                                                 cudf::size_type sub_chunk_size,
                                                                 cudf::size_type last_char,
                                                                 cudf::size_type num_strings);

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

void StringMatching(char* char_data, uint64_t* str_indices, std::string match_string, uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings, int not_equal) {
  CHECK_ERROR();
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  if (num_strings == 0) {
    SIRIUS_LOG_DEBUG("Input size is 0");
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    h_count[0] = 0;
    count = h_count;
    return;
  }

    SETUP_TIMING();
    START_TIMER();
  SIRIUS_LOG_DEBUG("Launching single term string matching kernel");
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
  count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
  char* d_match_str = gpuBufferManager->customCudaMalloc<char>(match_string.length(), 0, 0);
  int* d_kmp_automato = gpuBufferManager->customCudaMalloc<int>(kmp_automato_size, 0, 0);
  uint64_t* d_worker_start_term = gpuBufferManager->customCudaMalloc<uint64_t>(workers_needed, 0, 0);
  bool* d_answers = reinterpret_cast<bool*> (gpuBufferManager->customCudaMalloc<uint8_t>(num_strings, 0, 0));
  cudaMemset(d_answers, 0, num_strings * sizeof(bool));
  // TODO: Do it twice for more accurate allocation

  // Copy over the data to the buffers
  cudaMemcpy(d_kmp_automato, kmp_automato, kmp_automato_size * sizeof(int), cudaMemcpyHostToDevice);

  // Also set the initial values
  // cudaMemset(d_matching_rows, 0, num_strings * sizeof(uint64_t));
  CHECK_ERROR();
  
  // Set the start terms
  uint64_t last_char = num_chars - 1;
  uint64_t preprocess_blocks_needed = (workers_needed + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  SIRIUS_LOG_DEBUG("Sirius running preprocessing for {} workers with {} strings and {} chars", workers_needed, num_strings, num_chars);

  auto preprocessing_start = std::chrono::high_resolution_clock::now();
  determine_start_kernel<uint64_t><<<preprocess_blocks_needed, THREADS_PER_BLOCK_STRINGS>>>(str_indices, num_strings, d_worker_start_term, 
            workers_needed, CHUNK_SIZE, last_char);
  cudaDeviceSynchronize();
  auto preprocessing_end = std::chrono::high_resolution_clock::now();
  int preprocessing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(preprocessing_end - preprocessing_start).count();

  auto str_match_start = std::chrono::high_resolution_clock::now();
  uint64_t block_sub_chunk_size = (CHUNK_SIZE + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  single_term_kmp_kernel<uint64_t><<<workers_needed, THREADS_PER_BLOCK_STRINGS>>>(char_data, str_indices, d_kmp_automato, d_worker_start_term, 
    d_answers, match_length, workers_needed, CHUNK_SIZE, block_sub_chunk_size, last_char, num_strings);
  cudaDeviceSynchronize();
  auto str_match_end = std::chrono::high_resolution_clock::now();
  int str_match_time_us = std::chrono::duration_cast<std::chrono::microseconds>(str_match_end - str_match_start).count();
  CHECK_ERROR();

  cudaMemset(count, 0, sizeof(uint64_t));
  compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_answers, row_id, (unsigned long long*) count, num_strings, 1, not_equal);

  // Record the number of valid strings
  uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
  cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CHECK_ERROR();
  row_id = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);

  cudaMemset(count, 0, sizeof(uint64_t));
  compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_answers, row_id, (unsigned long long*) count, num_strings, 0, not_equal);

  // Check there are no errors
  CHECK_ERROR();

  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_match_str), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_kmp_automato), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_worker_start_term), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_answers), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);
  count = h_count;
  SIRIUS_LOG_DEBUG("String Matching Result Count = {}", h_count[0]);

  STOP_TIMER();
}

//--------------------------------------------------//
// Multi-Term String Matching
//--------------------------------------------------//
template<typename IdxT>
__global__ void multi_term_kmp_kernel(const char* char_data, const IdxT* indices, const int* kmp_automato, IdxT* worker_start_term, 
  IdxT* curr_term_answer, IdxT* prev_term_answer, bool* found_term, int pattern_size, IdxT num_workers, IdxT chunk_size, IdxT sub_chunk_size, 
  IdxT last_char, IdxT num_strings) {
    
    // See if have any work to do
    const auto chunk_id = blockIdx.x;
    const auto worker_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (chunk_id >= num_workers) return;

    const auto curr_chunk_start = min(chunk_id * chunk_size, last_char);
    const auto curr_chunk_end = min(curr_chunk_start + chunk_size + pattern_size, last_char);
    const auto curr_sub_chunk_start = min(curr_chunk_start + threadIdx.x * sub_chunk_size, curr_chunk_end);
    const auto curr_sub_chunk_end = min(curr_sub_chunk_start + sub_chunk_size + pattern_size, curr_chunk_end);

    // Determine the subchunk that the current string is going to be working on
    auto curr_term = worker_start_term[chunk_id];
    while (curr_term < num_strings && (curr_sub_chunk_start < indices[curr_term] || curr_sub_chunk_start >= indices[curr_term + 1])) {
      curr_term++;
    }
    auto curr_term_end = indices[curr_term + 1];

    // Perform the actual string matching
    int j = 0; int curr_idx = 0; 
    #pragma unroll
    for(int i = curr_sub_chunk_start; i <= curr_sub_chunk_end; i++) {
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
        if(i >= prev_term_answer[curr_term]) {
          found_term[curr_term] = true;
          cuda::atomic_ref<IdxT, cuda::thread_scope_device> curr_term_answer_ref(curr_term_answer[curr_term]);
          curr_term_answer_ref.fetch_min(i + pattern_size, cuda::std::memory_order_relaxed);
        }

        j = 0;
      }
    }
}

// Instantiations
template __global__ void multi_term_kmp_kernel<uint64_t>(const char* char_data,
                                                         const uint64_t* indices,
                                                         const int* kmp_automato,
                                                         uint64_t* worker_start_term,
                                                         uint64_t* curr_term_answer,
                                                         uint64_t* prev_term_answer,
                                                         bool* found_term,
                                                         int pattern_size,
                                                         uint64_t num_workers,
                                                         uint64_t chunk_size,
                                                         uint64_t sub_chunk_size,
                                                         uint64_t last_char,
                                                         uint64_t num_strings);
template __global__ void multi_term_kmp_kernel<cudf::size_type>(const char* char_data,
                                                                const cudf::size_type* indices,
                                                                const int* kmp_automato,
                                                                cudf::size_type* worker_start_term,
                                                                cudf::size_type* curr_term_answer,
                                                                cudf::size_type* prev_term_answer,
                                                                bool* found_term,
                                                                int pattern_size,
                                                                cudf::size_type num_workers,
                                                                cudf::size_type chunk_size,
                                                                cudf::size_type sub_chunk_size,
                                                                cudf::size_type last_char,
                                                                cudf::size_type num_strings);

template<typename IdxT>
__global__ void initialize_term_answers(IdxT* curr_term_answer, IdxT num_chars, IdxT num_strings) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < num_strings) {
      curr_term_answer[tid] = num_chars;
    } 
}

// Instantiations
template __global__ void initialize_term_answers<uint64_t>(uint64_t* curr_term_answer,
                                                           uint64_t num_chars,
                                                           uint64_t num_strings);
template __global__ void initialize_term_answers<cudf::size_type>(cudf::size_type* curr_term_answer,
                                                                  cudf::size_type num_chars,
                                                                  cudf::size_type num_strings);

void MultiStringMatching(char* char_data, uint64_t* str_indices, std::vector<std::string> all_terms,
       uint64_t* &row_id, uint64_t* &count, uint64_t num_chars, uint64_t num_strings, int not_equal) {
  CHECK_ERROR();
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  if (num_strings == 0) {
    SIRIUS_LOG_DEBUG("Input size is 0");
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    h_count[0] = 0;
    count = h_count;
    return;
  }
  
    SETUP_TIMING();
    START_TIMER();
  SIRIUS_LOG_DEBUG("Launching multi term string matching kernel");
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
  count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
  uint64_t* d_worker_start_term = gpuBufferManager->customCudaMalloc<uint64_t>(workers_needed, 0, 0);
  uint64_t* d_prev_term_answers = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);
  uint64_t* d_answer_idxs = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);
  cudaMemset(d_answer_idxs, 0, num_strings * sizeof(uint64_t));
  bool* d_found_answer = reinterpret_cast<bool*> (gpuBufferManager->customCudaMalloc<uint8_t>(num_strings, 0, 0));
  cudaMemset(d_found_answer, 0, num_strings * sizeof(bool));
  // uint64_t* d_matching_rows = gpuBufferManager->customCudaMalloc<uint64_t>(num_strings, 0, 0);

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
  // cudaMemset(d_matching_rows, 0, num_strings * sizeof(uint64_t));
  CHECK_ERROR();

  // Determine the start offset for each kernel
  uint64_t last_char = num_chars - 1;
  uint64_t kernel_block_needed = (workers_needed + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  uint64_t block_sub_chunk_size = (CHUNK_SIZE + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  determine_start_kernel<uint64_t><<<kernel_block_needed, THREADS_PER_BLOCK_STRINGS>>>(str_indices, num_strings, d_worker_start_term, 
            workers_needed, CHUNK_SIZE, last_char);
  CHECK_ERROR();
  
  // Perform the string matching term by term
  uint64_t preprocess_num_blocks = (num_strings + THREADS_PER_BLOCK_STRINGS - 1)/THREADS_PER_BLOCK_STRINGS;
  for(int i = 0; i < num_terms; i++) {
    // Determine the current terms variables
    int curr_term_length = all_terms[i].size();
    int* curr_term_automato = d_all_automatos[i];

    // Perform pre processing
    cudaMemset(d_found_answer, 0, num_strings * sizeof(bool));
    initialize_term_answers<uint64_t><<<preprocess_num_blocks, THREADS_PER_BLOCK_STRINGS>>>(d_answer_idxs, num_chars, num_strings);
    CHECK_ERROR();

    // Run the search
    multi_term_kmp_kernel<uint64_t><<<workers_needed, THREADS_PER_BLOCK_STRINGS>>>(char_data, str_indices, curr_term_automato, d_worker_start_term, 
      d_answer_idxs, d_prev_term_answers, d_found_answer, curr_term_length, workers_needed, CHUNK_SIZE, block_sub_chunk_size, 
      last_char, num_strings);
    CHECK_ERROR();

    // If there are future terms, the make the current answer the prev term answers
    if(i < (num_terms - 1)) {
      uint64_t* temp_ptr = d_answer_idxs;
      d_answer_idxs = d_prev_term_answers;
      d_prev_term_answers = temp_ptr;
    }
  }

  cudaMemset(count, 0, sizeof(uint64_t));
  compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_found_answer, row_id, (unsigned long long*) count, num_strings, 1, not_equal);

  // Record the number of valid strings
  uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
  cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CHECK_ERROR();
  row_id = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
  cudaMemset(count, 0, sizeof(uint64_t));

  compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_found_answer, row_id, (unsigned long long*) count, num_strings, 0, not_equal);

  // Check there are no errors
  CHECK_ERROR();

  //free the memory
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_worker_start_term), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_prev_term_answers), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>( d_answer_idxs), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_found_answer), 0);
  for(int i = 0; i < num_terms; i++) {
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_all_automatos[i]), 0);
  }
  SIRIUS_LOG_DEBUG("Multi String Matching Result Count = {}", h_count[0]);
  STOP_TIMER();

  count = h_count;
}

//--------------------------------------------------//
// Prefix Matching
//--------------------------------------------------//
template<typename IdxT>
__global__ void prefix_kernel(const char* char_data, IdxT num_chars, const IdxT* str_indices, IdxT num_strings, const char* prefix_chars, 
  IdxT num_prefix_chars, bool* results) {
  const IdxT start_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const IdxT tile_size = gridDim.x * blockDim.x;
  for(IdxT i = start_idx; i < num_strings; i += tile_size) {
    // First get the current strings details and check its length
    IdxT start_offset = str_indices[i]; 
    IdxT end_offset = str_indices[i + 1];
    IdxT curr_str_length = end_offset - start_offset;
    if(curr_str_length < num_prefix_chars) {
      results[i] = false;
      continue;
    }
    const char* curr_str_chars = char_data + start_offset;
    // Now actually compare the initial chars
    bool is_valid = true;
    for(IdxT j = 0; j < num_prefix_chars; j++) {
      if(curr_str_chars[j] != prefix_chars[j]) {
        is_valid = false;
        break;
      }
    }
    results[i] = is_valid;
  }
}

// Instantiations
template __global__ void prefix_kernel<uint64_t>(const char* char_data,
                                                 uint64_t num_chars,
                                                 const uint64_t* str_indices,
                                                 uint64_t num_strings,
                                                 const char* prefix_chars,
                                                 uint64_t num_prefix_chars,
                                                 bool* results);
template __global__ void prefix_kernel<cudf::size_type>(const char* char_data,
                                                        cudf::size_type num_chars,
                                                        const cudf::size_type* str_indices,
                                                        cudf::size_type num_strings,
                                                        const char* prefix_chars,
                                                        cudf::size_type num_prefix_chars,
                                                        bool* results);

void PrefixMatching(char* char_data, uint64_t* str_indices, std::string match_prefix, uint64_t* &row_id, uint64_t* &count, 
  uint64_t num_chars, uint64_t num_strings, int not_equal) {

  // Allocate the necesary buffers on the GPU
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
  if (num_strings == 0) {
    SIRIUS_LOG_DEBUG("Input size is 0");
    uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
    h_count[0] = 0;
    count = h_count;
    return;
  }

    SETUP_TIMING();
    START_TIMER();
    SIRIUS_LOG_DEBUG("Launching Prefix Matching kernel");

  count = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
  uint64_t num_prefix_chars = match_prefix.length();
  char* d_prefix_chars = gpuBufferManager->customCudaMalloc<char>(num_prefix_chars, 0, 0);
  cudaMemcpy(d_prefix_chars, match_prefix.c_str(), num_prefix_chars * sizeof(char), cudaMemcpyHostToDevice);
  bool* d_results = gpuBufferManager->customCudaMalloc<bool>(num_strings, 0, 0);
  cudaMemset(d_results, 0, num_strings * sizeof(bool));

  // Run the kernel
  uint64_t items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  uint64_t num_blocks = (num_strings + items_per_block - 1)/items_per_block;
  prefix_kernel<<<num_blocks, BLOCK_THREADS>>>(char_data, num_chars, str_indices, num_strings, d_prefix_chars, num_prefix_chars, d_results);
  cudaDeviceSynchronize();
  CHECK_ERROR();

  cudaMemset(count, 0, sizeof(uint64_t));
  compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_results, row_id, (unsigned long long*) count, num_strings, 1, not_equal);

  // Record the number of valid strings
  uint64_t* h_count = gpuBufferManager->customCudaHostAlloc<uint64_t>(1);
  cudaMemcpy(h_count, count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
  CHECK_ERROR();
  row_id = gpuBufferManager->customCudaMalloc<uint64_t>(h_count[0], 0, 0);
  cudaMemset(count, 0, sizeof(uint64_t));

  compact_valid_rows<BLOCK_THREADS, ITEMS_PER_THREAD><<<((num_strings + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD)), BLOCK_THREADS>>>(d_results, row_id, (unsigned long long*) count, num_strings, 0, not_equal);

  // Check there are no errors
  cudaDeviceSynchronize();
  CHECK_ERROR();

  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_prefix_chars), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_results), 0);
  gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(count), 0);

  count = h_count;
  SIRIUS_LOG_DEBUG("PrefixMatching Result Count {}", h_count[0]);
  STOP_TIMER();
}

//--------------------------------------------------//
// String Matching for CuDF Compatibility
//--------------------------------------------------//
// This is a replication of the above functions for compatibility with CuDF
// The key points are 1) use cudf::size_type instead of uint64_t
//                    2) manage memory with rmm::device_uvector, so ownership can be transferred
//                       to cudf::columns
//                    3) emit a boolean column instead of row ids

// Macros to simplify kernel launch syntax
#define LAUNCH_KERNEL_DIV(K, T, N, B, S) K<T><<<cuda::ceil_div((N), (B)), (B), 0, (S)>>>
#define LAUNCH_KERNEL_DIRECT(K, T, N, B, S) K<T><<<(N), (B), 0, (S)>>>

//----------String Matching----------//
std::unique_ptr<cudf::column> DoStringMatching(const char* input_data,
                                               cudf::size_type input_count,
                                               const cudf::size_type* input_offsets,
                                               cudf::size_type byte_count,
                                               const std::string& match_string,
                                               rmm::device_async_resource_ref mr)
{
  static_assert(std::is_same_v<int32_t, cudf::size_type>); // Sanity check

  auto stream = cudf::get_default_stream();

  // Compute the automato for this string
  const auto match_length      = static_cast<cudf::size_type>(match_string.size());
  const auto* match_char       = match_string.c_str();
  const auto kmp_automato_size = match_length * CHARS_IN_BYTE;
  std::vector<int32_t> kmp_automato(kmp_automato_size, 0);
  const auto first_idx    = static_cast<int32_t>(match_char[0]) + CHAR_INCREMENT;
  kmp_automato[first_idx] = 1;
  for (int32_t X = 0, j = 1; j < match_length; j++)
  {
    const auto curr_idx = static_cast<int32_t>(match_char[j]) + CHAR_INCREMENT;

    // Copy over the chars from the previous automato
    for (int32_t c = 0; c < CHARS_IN_BYTE; c++)
    {
      kmp_automato[j * CHARS_IN_BYTE + c] = kmp_automato[X * CHARS_IN_BYTE + c];
    }
    kmp_automato[j * CHARS_IN_BYTE + curr_idx] = j + 1;
    X                                          = kmp_automato[X * CHARS_IN_BYTE + curr_idx];
  }

  // Copy match string to device memory
  const auto match_byte_count = static_cast<cudf::size_type>(match_string.size());
  rmm::device_uvector<char> d_match_string(match_byte_count, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_match_string.data(),
                                match_string.data(),
                                match_byte_count,
                                cudaMemcpyHostToDevice,
                                stream));

  // Copy automato to device memory
  rmm::device_uvector<int32_t> d_kmp_automato(kmp_automato_size, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_kmp_automato.data(),
                                kmp_automato.data(),
                                kmp_automato_size * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream));

  // Allocate start terms memory and the boolean output buffer
  const auto workers_needed = cuda::ceil_div(byte_count, CHUNK_SIZE);
  rmm::device_uvector<int32_t> d_worker_start_term(workers_needed, stream, mr);
  rmm::device_uvector<bool> output(input_count, stream, mr);

  // Initialize the output buffer to false
  CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), 0, input_count * sizeof(bool), stream));

  // Launch kernel to determine the start offset for each worker
  LAUNCH_KERNEL_DIV(determine_start_kernel,
                    cudf::size_type,
                    workers_needed,
                    THREADS_PER_BLOCK_STRINGS,
                    stream)
  (input_offsets,
   input_count,
   d_worker_start_term.data(),
   workers_needed,
   CHUNK_SIZE,
   byte_count - 1);

  // Launch KMP kernel
  LAUNCH_KERNEL_DIRECT(single_term_kmp_kernel,
                       cudf::size_type,
                       workers_needed,
                       THREADS_PER_BLOCK_STRINGS,
                       stream)
  (input_data,
   input_offsets,
   d_kmp_automato.data(),
   d_worker_start_term.data(),
   output.data(),
   match_length,
   workers_needed,
   CHUNK_SIZE,
   cuda::ceil_div(CHUNK_SIZE, THREADS_PER_BLOCK_STRINGS),
   byte_count - 1,
   input_count);

  // Return a boolean cudf::column
  return std::make_unique<cudf::column>(std::move(output), rmm::device_buffer(0, stream, mr), 0);
}

//----------Multi-Term String Matching----------//
std::unique_ptr<cudf::column> DoMultiStringMatching(const char* input_data,
                                                    cudf::size_type input_count,
                                                    const cudf::size_type* input_offsets,
                                                    cudf::size_type byte_count,
                                                    const std::vector<std::string>& match_strings,
                                                    rmm::device_async_resource_ref mr)
{
  static_assert(std::is_same_v<int32_t, cudf::size_type>); // Sanity check

  auto stream = cudf::get_default_stream();

  // Compute the automato for each term
  std::vector<rmm::device_uvector<int32_t>> d_kmp_automatos;
  for (const auto& match_string : match_strings)
  {
    const auto match_length      = static_cast<cudf::size_type>(match_string.size());
    const auto* match_char       = match_string.c_str();
    const auto kmp_automato_size = match_length * CHARS_IN_BYTE;
    std::vector<int32_t> kmp_automato(kmp_automato_size, 0);
    const auto first_idx    = static_cast<int32_t>(match_char[0]) + CHAR_INCREMENT;
    kmp_automato[first_idx] = 1;
    for (int32_t X = 0, j = 1; j < match_length; j++)
    {
      const auto curr_idx = static_cast<int32_t>(match_char[j]) + CHAR_INCREMENT;

      // Copy over the chars from the previous automato
      for (int32_t c = 0; c < CHARS_IN_BYTE; c++)
      {
        kmp_automato[j * CHARS_IN_BYTE + c] = kmp_automato[X * CHARS_IN_BYTE + c];
      }
      kmp_automato[j * CHARS_IN_BYTE + curr_idx] = j + 1;
      X                                          = kmp_automato[X * CHARS_IN_BYTE + curr_idx];
    }

    // Copy automato to device memory
    d_kmp_automatos.emplace_back(kmp_automato_size, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_kmp_automatos.back().data(),
                                  kmp_automato.data(),
                                  kmp_automato_size * sizeof(int32_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
  }

  // Allocate start terms memory, rotating answer indices, and the boolean output buffer
  const auto workers_needed = cuda::ceil_div(byte_count, CHUNK_SIZE);
  rmm::device_uvector<int32_t> d_worker_start_term(workers_needed, stream, mr);
  rmm::device_uvector<cudf::size_type> d_answer_idxs(input_count, stream, mr);
  rmm::device_uvector<cudf::size_type> d_prev_answer_idxs(input_count, stream, mr);
  rmm::device_uvector<bool> output(input_count, stream, mr);

  // Initialize answer indices to zero, and copy offsets to previous answer indices
  CUDF_CUDA_TRY(
    cudaMemsetAsync(d_answer_idxs.data(), 0, input_count * sizeof(cudf::size_type), stream));
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_prev_answer_idxs.data(),
                                input_offsets,
                                input_count * sizeof(cudf::size_type),
                                cudaMemcpyDeviceToDevice,
                                stream));

  // Launch kernel to determine the start offset for each worker
  LAUNCH_KERNEL_DIV(determine_start_kernel,
                    cudf::size_type,
                    workers_needed,
                    THREADS_PER_BLOCK_STRINGS,
                    stream)
  (input_offsets,
   input_count,
   d_worker_start_term.data(),
   workers_needed,
   CHUNK_SIZE,
   byte_count - 1);

  // Perform the string matching for each term
  auto* answer_idxs_ptr      = d_answer_idxs.data();
  auto* prev_answer_idxs_ptr = d_prev_answer_idxs.data();
  for (int32_t i = 0; i < match_strings.size(); i++)
  {
    const auto curr_term_length    = static_cast<cudf::size_type>(match_strings[i].size());
    const auto* curr_term_automato = d_kmp_automatos[i].data();

    // Preprocessing
    CUDF_CUDA_TRY(cudaMemsetAsync(output.data(), 0, input_count * sizeof(bool), stream));
    LAUNCH_KERNEL_DIV(initialize_term_answers,
                      cudf::size_type,
                      input_count,
                      THREADS_PER_BLOCK_STRINGS,
                      stream)
    (answer_idxs_ptr, byte_count, input_count);

    // Launch the KMP kernel for the current term
    LAUNCH_KERNEL_DIRECT(multi_term_kmp_kernel,
                         cudf::size_type,
                         workers_needed,
                         THREADS_PER_BLOCK_STRINGS,
                         stream)
    (input_data,
     input_offsets,
     curr_term_automato,
     d_worker_start_term.data(),
     answer_idxs_ptr,
     prev_answer_idxs_ptr,
     output.data(),
     curr_term_length,
     workers_needed,
     CHUNK_SIZE,
     cuda::ceil_div(CHUNK_SIZE, THREADS_PER_BLOCK_STRINGS),
     byte_count - 1,
     input_count);

    // If there are future terms, swap the answer indices
    if (i < (match_strings.size() - 1))
    {
      // Swap the answer indices for the next term
      std::swap(answer_idxs_ptr, prev_answer_idxs_ptr);
    }
  }

  // Return a boolean cudf::column
  return std::make_unique<cudf::column>(std::move(output), rmm::device_buffer(0, stream, mr), 0);
}

//----------Prefix Matching----------//
std::unique_ptr<cudf::column> DoPrefixMatching(const char* input_data,
                                               cudf::size_type input_count,
                                               const cudf::size_type* input_offsets,
                                               cudf::size_type byte_count,
                                               const std::string& match_prefix,
                                               rmm::device_async_resource_ref mr)
{
  static_assert(std::is_same_v<int32_t, cudf::size_type>); // Sanity check

  auto stream = cudf::get_default_stream();

  // Copy prefix string to device memory
  const auto prefix_byte_count = static_cast<cudf::size_type>(match_prefix.size());
  rmm::device_uvector<char> d_match_prefix(prefix_byte_count, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_match_prefix.data(),
                                match_prefix.data(),
                                prefix_byte_count,
                                cudaMemcpyHostToDevice,
                                stream));

  // Allocate boolean output buffer
  rmm::device_uvector<bool> output(input_count, stream, mr);

  // Launch kernel to perform prefix matching
  LAUNCH_KERNEL_DIV(prefix_kernel, cudf::size_type, input_count, BLOCK_THREADS, stream)
  (input_data,
   byte_count,
   input_offsets,
   input_count,
   d_match_prefix.data(),
   prefix_byte_count,
   output.data());

  // Return a cudf::column
  return std::make_unique<cudf::column>(std::move(output), rmm::device_buffer(0, stream, mr), 0);
}

#undef LAUNCH_KERNEL_DIV
#undef LAUNCH_KERNEL_DIRECT

} // namespace duckdb
