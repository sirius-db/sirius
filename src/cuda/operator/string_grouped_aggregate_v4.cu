#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

#include <chrono>
#include <stdexcept>

namespace duckdb {

using std::chrono::high_resolution_clock;
using std::chrono::duration;

constexpr bool V4_LOG_MODE = false;
__device__ int d_comparator_keys_compared_v4 = 0;
__device__ int d_comparator_num_comparsions_v4 = 0;

struct sort_keys_type_string_v4 {
  uint64_t row_id;
  uint64_t row_signature;

  __host__ __device__ sort_keys_type_string_v4() {}
  __host__ __device__ sort_keys_type_string_v4(uint64_t _row_id, uint64_t _row_signature) : row_id(_row_id), row_signature(_row_signature) {}
};

struct CustomLessStringV4 {
    uint64_t** aligned_strings;
    uint64_t** aligned_offsets;
    uint64_t num_keys;

    __host__ __device__ CustomLessStringV4() {}
    __host__ __device__ CustomLessStringV4(uint64_t** _aligned_strings, uint64_t** _aligned_offsets, uint64_t _num_keys) : aligned_strings(_aligned_strings), aligned_offsets(_aligned_offsets),  num_keys(_num_keys) {}

    __device__ bool operator()(const sort_keys_type_string_v4& lhs, const sort_keys_type_string_v4& rhs) {
        if constexpr(V4_LOG_MODE) {
            atomicAdd(&d_comparator_num_comparsions_v4, (int) 1);
        }

        // First compare the signatures
        if (lhs.row_signature != rhs.row_signature) {
            if constexpr(V4_LOG_MODE) {
                atomicAdd(&d_comparator_keys_compared_v4, (int) 1);
            }
            return lhs.row_signature < rhs.row_signature;
        }

        // If the signature is the same then compare the invidiual lengths
        uint64_t values_compared = 1; 
        uint64_t left_val; uint64_t right_val;
        for(uint64_t i = 0; i < num_keys; i++) {
            uint64_t* curr_column_offsets = aligned_offsets[i];
            left_val = curr_column_offsets[lhs.row_id + 1] - curr_column_offsets[lhs.row_id];
            right_val = curr_column_offsets[rhs.row_id + 1] - curr_column_offsets[rhs.row_id];
            if(left_val != right_val) {
                if constexpr(V4_LOG_MODE) {
                    atomicAdd(&d_comparator_keys_compared_v4, (int) (values_compared + i));
                }
                return left_val < right_val;
            }
        }
        values_compared += num_keys;

        // If the lengths are the same then compare the individiual chars
        for(uint64_t i = 0; i < num_keys; i++) {
            // Get the offset details for this key
            uint64_t* curr_column_offsets = aligned_offsets[i];
            uint64_t left_read_offset = curr_column_offsets[lhs.row_id];
            uint64_t right_read_offset = curr_column_offsets[rhs.row_id];
            const uint64_t curr_length = curr_column_offsets[lhs.row_id + 1] - left_read_offset;

            // Determine the chars to compare
            uint64_t* curr_column_strings = aligned_strings[i];
            uint64_t* left_read_values = curr_column_strings + left_read_offset;
            uint64_t* right_read_values = curr_column_strings + right_read_offset;

            #pragma unroll
            for(uint64_t j = 0; j < curr_length; j++) {
                left_val = left_read_values[j]; right_val = right_read_values[j];
                if(left_val != right_val) {
                    if constexpr(V4_LOG_MODE) {
                        atomicAdd(&d_comparator_keys_compared_v4, (int) (values_compared + j));
                    }
                    return left_val < right_val;
                }
            }

            if constexpr(V4_LOG_MODE) {
                values_compared += curr_length;
            }
        }

        if constexpr(V4_LOG_MODE) {
            atomicAdd(&d_comparator_keys_compared_v4, (int) values_compared);
        }
        return true;
    }
}; 

__global__ void determine_column_lengths(uint64_t* column_offsets, uint64_t* write_offsets, const uint64_t num_rows) {
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i = start_idx; i < num_rows; i += tile_size) {
        uint64_t curr_record_length = column_offsets[i + 1] - column_offsets[i];
        write_offsets[i] = (curr_record_length + BYTES_IN_INTEGER - 1)/BYTES_IN_INTEGER;
    }
}

__global__ void fill_column_aligned_buffer(uint8_t* column_key, uint64_t* column_offset, uint64_t* aligned_buffer, uint64_t* aligned_offset, uint64_t* write_offset, uint64_t num_rows) {
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i = start_idx; i < num_rows; i += tile_size) {
        // Determine where to read write the current record
        uint64_t row_write_offset = i > 0 ? write_offset[i - 1] : 0;
        uint64_t row_read_offset = column_offset[i];
        uint64_t curr_record_length = column_offset[i + 1] - row_read_offset;

        // Copy the chars and update the offset for the aligned column
        memcpy(reinterpret_cast<uint8_t*>(aligned_buffer + row_write_offset), column_key + row_read_offset, curr_record_length * sizeof(uint8_t));
        aligned_offset[i] = row_write_offset;
    }
}

__global__ void create_row_records(uint64_t** aligned_strings,  uint64_t** aligned_offsets, sort_keys_type_string_v4* row_records, const uint64_t num_rows, const uint64_t num_keys) {
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Create the record for the current row in a tile based manner
    uint64_t curr_value;
    for(uint64_t i = start_idx; i < num_rows; i += tile_size) {
        // Get the signature for this row
        uint64_t signature = 0;
        uint64_t curr_power = 1;

        #pragma unroll
        for(uint64_t j = 0; j < num_keys; j++) {
            // Get the current record's details
            uint64_t* curr_column_offsets = aligned_offsets[j];
            uint64_t curr_row_start = curr_column_offsets[i];
            uint64_t curr_record_length = curr_column_offsets[i + 1] - curr_row_start;
            uint64_t* column_hash_values = aligned_strings[j] + curr_row_start;

            // Add it to the hash
            #pragma unroll
            for(uint64_t k = 0; k < curr_record_length; k++) {
                curr_value = column_hash_values[k];
                signature = (signature + curr_value * curr_power) % HASH_MOD_VALUE;
                curr_power = (curr_power * HASH_POWER) % HASH_MOD_VALUE;
            }
        }

        // Create the records struct
        row_records[i] = sort_keys_type_string_v4(i, signature);
    }
}

__global__ void print_sort_metadata_v4() {
    float average_compare_values = (1.0 * d_comparator_keys_compared_v4)/d_comparator_num_comparsions_v4;
    printf("STRING GROUP BY V4: Performed %d row comparsions checking an average of %f values\n", d_comparator_num_comparsions_v4, average_compare_values);
}

template <typename V>
void groupedStringAggregateV4(uint8_t** keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }

    printf("Launching String Grouped Aggregate Kernel V4\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    // Create the temporary buffer
    uint64_t total_preprocessing_bytes = N * sizeof(uint64_t);
    auto preprocess_start_time = high_resolution_clock::now();

    // Create an aligned buffer for each column
    uint64_t items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t num_blocks = (N + items_per_block - 1)/items_per_block;
    uint64_t** d_aligned_strs = gpuBufferManager->customCudaMalloc<uint64_t*>(num_keys, 0, 0);
    uint64_t** d_aligned_offsets = gpuBufferManager->customCudaMalloc<uint64_t*>(num_keys, 0, 0);
    
    // Create the temporary needed to create the aligned columns
    uint64_t* d_preprocess_write_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0); 
    void* preprocess_prefix_temp_storage = nullptr;
    size_t preprocess_prefix_temp_storage_bytes = 0;
    for(uint64_t i = 0; i < num_keys; i++) {
        // Determine the number of ints required for each string
        uint64_t* curr_col_offsets = offset[i];
        uint8_t* curr_col_keys = keys[i];
        determine_column_lengths<<<num_blocks, BLOCK_THREADS>>>(curr_col_offsets, d_preprocess_write_offsets, N);

        // Now run a prefix sum to determine the write offsets
        cub::DeviceScan::InclusiveSum(preprocess_prefix_temp_storage, preprocess_prefix_temp_storage_bytes, d_preprocess_write_offsets, N);
        preprocess_prefix_temp_storage = reinterpret_cast<void*>(gpuBufferManager->customCudaMalloc<uint8_t>(preprocess_prefix_temp_storage_bytes, 0, 0));
        cub::DeviceScan::InclusiveSum(preprocess_prefix_temp_storage, preprocess_prefix_temp_storage_bytes, d_preprocess_write_offsets, N);

        // Create the aligned strs buffer based on the prefix sum results
        uint64_t column_ints_needed;
        cudaMemcpy(&column_ints_needed, d_preprocess_write_offsets + (N - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
        total_preprocessing_bytes += column_ints_needed * sizeof(uint64_t);

        uint64_t* column_aligned_strs_buffer = gpuBufferManager->customCudaMalloc<uint64_t>(column_ints_needed, 0, 0);
        cudaMemset(column_aligned_strs_buffer, 0, column_ints_needed * sizeof(uint64_t));
        uint64_t* column_aligned_offsets = gpuBufferManager->customCudaMalloc<uint64_t>((N + 1), 0, 0);

        // Fill in the column aligned strs buffer
        fill_column_aligned_buffer<<<num_blocks, BLOCK_THREADS>>>(curr_col_keys, curr_col_offsets, column_aligned_strs_buffer, column_aligned_offsets, d_preprocess_write_offsets, N);

        // Record the buffers for this columns
        cudaMemcpy(column_aligned_offsets + N, &column_ints_needed, sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_aligned_strs + i, &column_aligned_strs_buffer, sizeof(uint64_t*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_aligned_offsets + i, &column_aligned_offsets, sizeof(uint64_t*), cudaMemcpyHostToDevice);
    }

    // Create the buffer of sort records
    sort_keys_type_string_v4* d_row_records = reinterpret_cast<sort_keys_type_string_v4*>(gpuBufferManager->customCudaMalloc<key_and_signature>(N, 0, 0));
    create_row_records<<<num_blocks, BLOCK_THREADS>>>(d_aligned_strs, d_aligned_offsets, d_row_records, N, num_keys);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    std::cout << "STRING GROUP BY V4: Preprocessing requires " << total_preprocessing_bytes << " bytes" << std::endl;
    std::cout << "STRING GROUP BY V4: Preprocessing took " << preprocess_time_ms << " ms" << std::endl;

    // Perform the sort
    auto sort_start_time = high_resolution_clock::now();

    CustomLessStringV4 custom_less_comparator(d_aligned_strs, d_aligned_offsets, num_keys);
    void* sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        sort_temp_storage,
        sort_temp_storage_bytes,
        d_row_records,
        N,
        custom_less_comparator);

    CHECK_ERROR();

    // Allocate temporary storage
    sort_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(sort_temp_storage_bytes, 0, 0));

    // Run sorting operation
    cub::DeviceMergeSort::SortKeys(
        sort_temp_storage,
        sort_temp_storage_bytes,
        d_row_records,
        N,
        custom_less_comparator);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto sort_end_time = high_resolution_clock::now();
    auto sort_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(sort_end_time - sort_start_time).count();
    std::cout << "STRING GROUP BY V4: Sorting took " << sort_time_ms << " ms" << std::endl;
    print_sort_metadata_v4<<<1, 1>>>();

    auto group_by_start_time = high_resolution_clock::now();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto group_by_end_time = high_resolution_clock::now();
    auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
    std::cout << "STRING GROUP BY V4: Group By took " << group_by_time_ms << " ms" << std::endl;

    auto post_processing_start_time = high_resolution_clock::now();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto post_processing_end_time = high_resolution_clock::now();
    auto post_processing_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(post_processing_end_time - post_processing_start_time).count();
    std::cout << "STRING GROUP BY V4: Post Processing took " << post_processing_time_ms << " ms" << std::endl;

    throw std::runtime_error("Grouped String Aggregate V4 implementation incomplete");
}

template
void groupedStringAggregateV4<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedStringAggregateV4<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

}