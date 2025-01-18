#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

#include <chrono>
#include <stdexcept>

namespace duckdb {

using std::chrono::high_resolution_clock;
using std::chrono::duration;

constexpr bool V3_LOG_MODE = false;
__device__ int d_comparator_keys_compared_v3 = 0;
__device__ int d_comparator_num_comparsions_v3 = 0;

struct sort_keys_type_string_v3 {
  uint64_t row_id;
  uint64_t row_signature;

  __host__ __device__ sort_keys_type_string_v3() {}
  __host__ __device__ sort_keys_type_string_v3(uint64_t _row_id, uint64_t _row_signature) : row_id(_row_id), row_signature(_row_signature) {}
};

struct CustomLessStringV3 {
    uint8_t** all_keys;
    uint64_t** offsets;
    uint64_t num_keys;

    __host__ __device__ CustomLessStringV3() {}
    __host__ __device__ CustomLessStringV3(uint8_t** _all_keys, uint64_t** _offsets, uint64_t _num_keys) : all_keys(_all_keys), offsets(_offsets),  num_keys(_num_keys) {}

    __device__ bool operator()(const sort_keys_type_string_v3& lhs, const sort_keys_type_string_v3& rhs) {
        if constexpr(V3_LOG_MODE) {
            atomicAdd(&d_comparator_num_comparsions_v3, (int) 1);
        }

        // First compare the signatures
        if (lhs.row_signature != rhs.row_signature) {
            if constexpr(V3_LOG_MODE) {
                atomicAdd(&d_comparator_keys_compared_v3, (int) 1);
            }
            return lhs.row_signature < rhs.row_signature;
        }

        // If the signature is the same then compare the invidiual lengths
        uint64_t values_compared = 1; 
        uint64_t left_val; uint64_t right_val;
        for(uint64_t i = 0; i < num_keys; i++) {
            uint64_t* curr_column_offsets = offsets[i];
            left_val = curr_column_offsets[lhs.row_id + 1] - curr_column_offsets[lhs.row_id];
            right_val = curr_column_offsets[rhs.row_id + 1] - curr_column_offsets[rhs.row_id];
            if(left_val != right_val) {
                if constexpr(V3_LOG_MODE) {
                    atomicAdd(&d_comparator_keys_compared_v3, (int) (values_compared + i));
                }
                return left_val < right_val;
            }
        }
        values_compared += num_keys;

        // If the lengths are the same then compare the individiual chars
        uint8_t curr_left_val; uint8_t curr_right_val;
        for(uint64_t i = 0; i < num_keys; i++) {
            // Get the offset details for this key
            uint64_t* curr_column_offsets = offsets[i];
            uint64_t left_read_offset = curr_column_offsets[lhs.row_id];
            uint64_t right_read_offset = curr_column_offsets[rhs.row_id];
            const uint64_t curr_length = curr_column_offsets[lhs.row_id + 1] - left_read_offset;

            // Determine the chars to compare
            uint8_t* curr_column_keys = all_keys[i];
            uint8_t* left_read_chars = curr_column_keys + left_read_offset;
            uint8_t* right_read_chars = curr_column_keys + right_read_offset;

            #pragma unroll
            for(uint64_t j = 0; j < curr_length; j++) {
                curr_left_val = left_read_chars[j]; curr_right_val = right_read_chars[j];
                if(curr_left_val != curr_right_val) {
                    if constexpr(V3_LOG_MODE) {
                        atomicAdd(&d_comparator_keys_compared_v3, (int) (values_compared + j));
                    }
                    return curr_left_val < curr_right_val;
                }
            }

            if constexpr(V3_LOG_MODE) {
                values_compared += curr_length;
            }
        }

        if constexpr(V3_LOG_MODE) {
            atomicAdd(&d_comparator_keys_compared_v3, (int) values_compared);
        }
        return true;
    }
}; 


__global__ void fill_preprocess_buffer(uint8_t** keys, uint64_t** column_length_offsets, sort_keys_type_string_v3* row_records, const uint64_t num_rows, const uint64_t num_keys) {
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
            // Get the chars for this row in this column
            uint64_t* curr_column_offsets = column_length_offsets[j];
            uint64_t curr_row_start = curr_column_offsets[i];
            uint64_t curr_record_length = curr_column_offsets[i + 1] - curr_row_start;
            uint8_t* column_hash_chars = keys[j] + curr_row_start;

            // Update the signature using this record
            #pragma unroll
            for(uint64_t k = 0; k < curr_record_length; k++) {
                curr_value = static_cast<uint64_t>(column_hash_chars[k]);
                signature = (signature + curr_value * curr_power) % HASH_MOD_VALUE;
                curr_power = (curr_power * HASH_POWER) % HASH_MOD_VALUE;
            }
        }

        row_records[i] = sort_keys_type_string_v3(i, signature);
    }
}

__global__ void print_sort_metadata_v3() {
    float average_compare_values = (1.0 * d_comparator_keys_compared_v3)/d_comparator_num_comparsions_v3;
    printf("STRING GROUP BY V3: Performed %d row comparsions checking an average of %f values\n", d_comparator_num_comparsions_v3, average_compare_values);
}

template <typename V>
void groupedStringAggregateV3(uint8_t** keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }

    printf("Launching String Grouped Aggregate Kernel V3\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    // Create the temporary buffer
    uint64_t total_preprocessing_bytes = 2 * N * sizeof(uint64_t);
    auto preprocess_start_time = high_resolution_clock::now();

    sort_keys_type_string_v3* d_row_records = reinterpret_cast<sort_keys_type_string_v3*>(gpuBufferManager->customCudaMalloc<key_and_signature>(N, 0, 0));
    uint64_t items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t num_blocks = (N + items_per_block - 1)/items_per_block;
    fill_preprocess_buffer<<<num_blocks, BLOCK_THREADS>>>(keys, offset, d_row_records, N, num_keys);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    std::cout << "STRING GROUP BY V3: Preprocessing requires " << total_preprocessing_bytes << " bytes" << std::endl;
    std::cout << "STRING GROUP BY V3: Preprocessing took " << preprocess_time_ms << " ms" << std::endl;

    // Perform the sort
    auto sort_start_time = high_resolution_clock::now();

    CustomLessStringV3 custom_less_comparator(keys, offset, num_keys);
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
    std::cout << "STRING GROUP BY V3: Sorting took " << sort_time_ms << " ms" << std::endl;
    print_sort_metadata_v3<<<1, 1>>>();

    auto group_by_start_time = high_resolution_clock::now();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto group_by_end_time = high_resolution_clock::now();
    auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
    std::cout << "STRING GROUP BY V3: Group By took " << group_by_time_ms << " ms" << std::endl;

    auto post_processing_start_time = high_resolution_clock::now();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto post_processing_end_time = high_resolution_clock::now();
    auto post_processing_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(post_processing_end_time - post_processing_start_time).count();
    std::cout << "STRING GROUP BY V3: Post Processing took " << post_processing_time_ms << " ms" << std::endl;

    throw std::runtime_error("Grouped String Aggregate V3 implementation incomplete");
}

template
void groupedStringAggregateV3<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedStringAggregateV3<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

}