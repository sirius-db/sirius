#include "cuda_helper.cuh"
#include "gpu_physical_order.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"

#include <chrono>
#include <stdexcept>
#include <cub/cub.cuh>
#include <assert.h>
#include <limits>

namespace duckdb { 

using std::chrono::duration;
using std::chrono::high_resolution_clock;

struct OrderByCustomComparator {
    uint8_t** d_col_keys;
    uint64_t** d_col_offsets;
    int* d_sort_orders;
    uint64_t num_cols;

    __host__ __device__ OrderByCustomComparator(uint8_t** _d_col_keys, uint64_t** _d_col_offsets, int* _d_sort_orders, 
        uint64_t _num_cols) : d_col_keys(_d_col_keys), d_col_offsets(_d_col_offsets), d_sort_orders(_d_sort_orders), num_cols(_num_cols) {}
    
    __device__ __forceinline__ bool operator()(const uint64_t &left_idx, const uint64_t &right_idx) {
        for (uint64_t i = 0; i < num_cols; i++) {
            // Determine the details of the left and right row records
            uint64_t* curr_column_offsets = d_col_offsets[i];
            uint64_t left_read_offset = curr_column_offsets[left_idx];
            uint64_t right_read_offset = curr_column_offsets[right_idx];
            const uint64_t left_length = curr_column_offsets[left_idx + 1] - left_read_offset;
            const uint64_t right_length = curr_column_offsets[right_idx + 1] - right_read_offset;

            // Initialize state
            bool sort_in_ascending_order = d_sort_orders[i] == 0;
            uint8_t* curr_column_chars = d_col_keys[i];
            uint64_t* curr_column_keys = reinterpret_cast<uint64_t*>(curr_column_chars);
            uint64_t bytes_remaining = min(left_length, right_length);

            while (bytes_remaining > 0) {
                // Read in the left and right value
                uint64_t left_int_idx = left_read_offset / BYTES_IN_INTEGER;
                uint64_t left_read_idx = left_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_left_int = curr_column_keys[left_int_idx];

                uint64_t right_int_idx = right_read_offset / BYTES_IN_INTEGER;
                uint64_t right_read_idx = right_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_right_int = curr_column_keys[right_int_idx];

                // Compare current batch of bytes
                uint64_t batch_size = min(BYTES_IN_INTEGER - max(left_read_idx, right_read_idx), bytes_remaining);
                uint64_t keep_mask = (1ULL << (BITS_IN_BYTE * batch_size)) - 1;
                uint64_t left_shifted_val = curr_left_int >> (BYTES_IN_INTEGER * left_read_idx);
                uint64_t left_val = left_shifted_val & keep_mask;
                uint64_t right_shifted_val = curr_right_int >> (BYTES_IN_INTEGER * right_read_idx);
                uint64_t right_val = right_shifted_val & keep_mask;

                // If they are not equal then actually perform the comparsion character by character
                if(left_val != right_val) {
                    #pragma unroll
                    for(uint64_t j = 0; j < batch_size; j++) {
                        uint8_t left_char = curr_column_chars[left_read_offset + j];
                        uint8_t right_char = curr_column_chars[right_read_offset + j];

                        // If the characters are different then return the result
                        if(left_char != right_char) {
                            return (sort_in_ascending_order ? left_char < right_char : left_char > right_char);
                        }
                    }
                }

                // Update trackers
                bytes_remaining -= batch_size;
                left_read_offset += batch_size;
                right_read_offset += batch_size;
            }

            // If the characters were equal but the lengths were not then compare based on the chars
            if(left_length != right_length) {
                return (sort_in_ascending_order ? left_length < right_length : left_length > right_length);
            }
        }

        // Return the result
        return left_idx < right_idx;
    }
};

__global__ void fill_row_ids_buffer(uint64_t* d_row_ids, uint64_t num_rows) {
    const uint64_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(thread_idx < num_rows) {
        d_row_ids[thread_idx] = thread_idx;
    }
}

void orderByString(uint8_t** col_keys, uint64_t** col_offsets, int* sort_orders, uint64_t* col_num_bytes, uint64_t num_rows, uint64_t num_cols) {
    CHECK_ERROR();
    if(num_rows == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }

    SIRIUS_LOG_DEBUG("Launching Order By String kernel");
    SETUP_TIMING();
    START_TIMER();

    // Copy over the ptrs onto the GPU
    auto preprocess_start_time = high_resolution_clock::now();
    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint8_t** d_col_keys = reinterpret_cast<uint8_t**>(gpuBufferManager->customCudaMalloc<void*>(num_cols, 0, 0));
    cudaMemcpy(d_col_keys, col_keys, num_cols * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    uint64_t** d_col_offsets = reinterpret_cast<uint64_t**>(gpuBufferManager->customCudaMalloc<void*>(num_cols, 0, 0));
    cudaMemcpy(d_col_offsets, col_offsets, num_cols * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    // Now create a vector of row ids
    uint64_t* d_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(num_rows, 0, 0);
    uint64_t num_populate_blocks = (num_rows + BLOCK_THREADS - 1) / BLOCK_THREADS;
    fill_row_ids_buffer<<<num_populate_blocks, BLOCK_THREADS>>>(d_row_ids, num_rows);

    int* d_sort_orders = reinterpret_cast<int*>(gpuBufferManager->customCudaMalloc<int>(num_cols, 0, 0));
    cudaMemcpy(d_sort_orders, sort_orders, num_cols * sizeof(int), cudaMemcpyHostToDevice);
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    SIRIUS_LOG_DEBUG("STRING ORDER BY: Preprocessing took {} ms", preprocess_time_ms);

    // Now sort those row ids using the custom comparator
    auto sort_start_time = high_resolution_clock::now();

    void* sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
    OrderByCustomComparator order_by_comparator(d_col_keys, d_col_offsets, d_sort_orders, num_cols);
    cub::DeviceMergeSort::SortKeys(
        sort_temp_storage,
        sort_temp_storage_bytes,
        d_row_ids,
        num_rows,
        order_by_comparator
    );

    sort_temp_storage = reinterpret_cast<void*>(gpuBufferManager->customCudaMalloc<uint8_t>(sort_temp_storage_bytes, 0, 0));

    cub::DeviceMergeSort::SortKeys(
        sort_temp_storage,
        sort_temp_storage_bytes,
        d_row_ids,
        num_rows,
        order_by_comparator
    );

    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto sort_end_time = high_resolution_clock::now();
    auto sort_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(sort_end_time - sort_start_time).count();
    SIRIUS_LOG_DEBUG("STRING ORDER BY: Sorting required {} bytes and took {} ms", sort_temp_storage_bytes, sort_time_ms);

    for(uint64_t i = 0; i < num_cols; i++) {
        // Get the original column
        uint8_t* unsorted_col_chars = col_keys[i];
        uint64_t* unsorted_col_offsets = col_offsets[i];
        
        // Materialize the column in the new order
        uint8_t* sorted_chars; uint64_t* sorted_offsets; uint64_t* new_num_bytes;
        materializeString(unsorted_col_chars, unsorted_col_offsets, sorted_chars, sorted_offsets, d_row_ids, new_num_bytes, num_rows, num_rows, col_num_bytes[i]);

        // Write back the result
        col_keys[i] = sorted_chars;
        col_offsets[i] = sorted_offsets;
        col_num_bytes[i] = new_num_bytes[0];
    }

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_col_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_col_offsets), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_row_ids), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_sort_orders), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(sort_temp_storage), 0);

    STOP_TIMER();
}

}