#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

#include <chrono>
#include <stdexcept>

namespace duckdb {

#define HASH_POWER 31
#define HASH_MOD_VALUE 1000000009

using std::chrono::high_resolution_clock;
using std::chrono::duration;

struct sort_keys_type_string_v2 {
  uint64_t* row_record;
  uint64_t num_records;

  __host__ __device__ sort_keys_type_string_v2() {}
  __host__ __device__ sort_keys_type_string_v2(uint64_t* _keys, uint64_t _num_records) : row_record(_keys), num_records(_num_records) {}
};

constexpr bool V2_LOG_MODE = false;
__device__ int d_comparator_keys_compared_v2 = 0;
__device__ int d_comparator_num_comparsions_v2 = 0;

struct CustomLessStringV2
{
  __device__ bool operator()(const sort_keys_type_string_v2 &lhs, const sort_keys_type_string_v2 &rhs) {
    // First check if the sum of the lengths are the same
    if constexpr(V2_LOG_MODE) {
        atomicAdd(&d_comparator_num_comparsions_v2, (int) 1);
    }

    if(lhs.num_records != rhs.num_records) {
        if constexpr(V2_LOG_MODE) {
            atomicAdd(&d_comparator_keys_compared_v2, (int) 1);
        }
        return lhs.num_records < rhs.num_records;
    }

    // Then if that is the same then compare all of the records in sequential order
    // This effectivelly first compares things in this order: 1) signature, 2) string lengths, 3) strings chars
    #pragma unroll
    for (uint64_t i = 0; i < lhs.num_records; i++) {
        if (lhs.row_record[i] != rhs.row_record[i]) {
            if constexpr(V2_LOG_MODE) {
                atomicAdd(&d_comparator_keys_compared_v2, (int) i);
            }
            return lhs.row_record[i] < rhs.row_record[i];
        }
    }

    if constexpr(V2_LOG_MODE) {
        atomicAdd(&d_comparator_keys_compared_v2, (int) lhs.num_records);
    }
    return true;

  }
};

__global__ void outer_preprocess_determine_row_memory(uint64_t** column_length_offsets, uint64_t* row_offsets, const uint64_t num_keys, const uint64_t num_rows) {
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i = start_idx; i < num_rows; i += tile_size) {
        // Get the ints necessary for this row
        uint64_t curr_row_ints_required = num_keys + 1;
        #pragma unroll
        for(uint64_t j = 0; j < num_keys; j++) {
            uint64_t* curr_column_offset = column_length_offsets[j];
            uint64_t curr_record_length = curr_column_offset[i + 1] - curr_column_offset[i];
            curr_row_ints_required += (curr_record_length + BYTES_IN_INTEGER - 1)/BYTES_IN_INTEGER;
        }
        row_offsets[i] = curr_row_ints_required;
    }
}

__global__ void fill_preprocess_buffer(uint8_t** keys, uint64_t** column_length_offsets, uint64_t* write_offsets, uint64_t* preprocess_buffer, sort_keys_type_string_v2* row_records, const uint64_t num_keys, const uint64_t num_rows) {
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i = start_idx; i < num_rows; i += tile_size) {
        // Get the ptr to write this row
        uint64_t row_write_offset = i > 0 ? write_offsets[i - 1] : 0;
        uint64_t* curr_row_write_ptr = preprocess_buffer + row_write_offset;
        uint64_t chars_write_offset = num_keys + 1;
        
        // Now write each column for this row
        uint64_t signature = 0;
        uint64_t curr_power = 1;

        #pragma unroll
        for(uint64_t j = 0; j < num_keys; j++) {
            // Determine the offset in the src to start copying from
            uint8_t* curr_column_chars = keys[j];
            uint64_t* curr_column_offsets = column_length_offsets[j];
            uint64_t curr_row_start = curr_column_offsets[i];
            uint64_t curr_record_length = curr_column_offsets[i + 1] - curr_row_start;
            curr_row_write_ptr[j + 1] = curr_record_length;
            
            // Copy over the chars
            uint8_t* column_hash_chars = curr_column_chars + curr_row_start;
            memcpy(reinterpret_cast<uint8_t*>(curr_row_write_ptr + chars_write_offset), column_hash_chars, curr_record_length * sizeof(uint8_t));
            
            uint64_t curr_value;
            #pragma unroll
            for(uint64_t k = 0; k < curr_record_length; k++) {
                curr_value = static_cast<uint64_t>(column_hash_chars[k]);
                signature = (signature + curr_value * curr_power) % HASH_MOD_VALUE;
                curr_power = (curr_power * HASH_POWER) % HASH_MOD_VALUE;
            }

            // Update the write ptr
            uint64_t curr_record_int_required = (curr_record_length + BYTES_IN_INTEGER - 1)/BYTES_IN_INTEGER;
            chars_write_offset += curr_record_int_required;
        }

        curr_row_write_ptr[0] = signature;
        row_records[i] = sort_keys_type_string_v2(curr_row_write_ptr, chars_write_offset);
    }
}

__global__ void print_sort_metadata_v2() {
    float average_compare_values = (1.0 * d_comparator_keys_compared_v2)/d_comparator_num_comparsions_v2;
    printf("STRING GROUP BY V2: Performed %d row comparsions checking an average of %f values\n", d_comparator_num_comparsions_v2, average_compare_values);
}

template <typename V>
void groupedStringAggregateV2(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }

    printf("Launching String Grouped Aggregate Kernel V2\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    // Create the temporary buffer
    uint64_t total_preprocessing_bytes = 0;
    auto preprocess_start_time = high_resolution_clock::now();

    // First use the lengths to determine the offsets
    uint64_t* d_preprocess_write_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    uint64_t items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t num_blocks = (N + items_per_block - 1)/items_per_block;
    outer_preprocess_determine_row_memory<<<num_blocks, BLOCK_THREADS>>>(offset, d_preprocess_write_offsets, num_keys, N);

    // Now run a prefix sum to determine the write offsets
    void* preprocess_prefix_temp_storage = nullptr;
    size_t preprocess_prefix_temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(preprocess_prefix_temp_storage, preprocess_prefix_temp_storage_bytes, d_preprocess_write_offsets, N);
    preprocess_prefix_temp_storage = reinterpret_cast<void*>(gpuBufferManager->customCudaMalloc<uint8_t>(preprocess_prefix_temp_storage_bytes, 0, 0));
    cub::DeviceScan::InclusiveSum(preprocess_prefix_temp_storage, preprocess_prefix_temp_storage_bytes, d_preprocess_write_offsets, N);

    // Create and fill the temporary buffer
    uint64_t preprocessing_ints_needed;
    cudaMemcpy(&preprocessing_ints_needed, d_preprocess_write_offsets + (N - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
    total_preprocessing_bytes = preprocessing_ints_needed * sizeof(uint64_t);

    uint64_t* d_preprocess_buffer = gpuBufferManager->customCudaMalloc<uint64_t>(preprocessing_ints_needed, 0, 0);
    cudaMemset(d_preprocess_buffer, 0, total_preprocessing_bytes);
    sort_keys_type_string_v2* d_row_records = reinterpret_cast<sort_keys_type_string_v2*>(gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    fill_preprocess_buffer<<<num_blocks, BLOCK_THREADS>>>(keys, offset, d_preprocess_write_offsets, d_preprocess_buffer, d_row_records, num_keys, N);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    std::cout << "STRING GROUP BY V2: Preprocessing requires " << total_preprocessing_bytes << " bytes" << std::endl;
    std::cout << "STRING GROUP BY V2: Preprocessing took " << preprocess_time_ms << " ms" << std::endl;

    // Perform the sort
    auto sort_start_time = high_resolution_clock::now();
    CustomLessStringV2 custom_less_comparator;
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
    print_sort_metadata_v2<<<1, 1>>>();
    std::cout << "STRING GROUP BY V2: Sorting took " << sort_time_ms << " ms" << std::endl;

    auto group_by_start_time = high_resolution_clock::now();
    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto group_by_end_time = high_resolution_clock::now();
    auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
    std::cout << "STRING GROUP BY V2: Group By took " << group_by_time_ms << " ms" << std::endl;

    auto post_processing_start_time = high_resolution_clock::now();
    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto post_processing_end_time = high_resolution_clock::now();
    auto post_processing_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(post_processing_end_time - post_processing_start_time).count();
    std::cout << "STRING GROUP BY V2: Post Processing took " << post_processing_time_ms << " ms" << std::endl;

    throw std::runtime_error("Grouped String Aggregate V2 implementation incomplete");
}

template
void groupedStringAggregateV2<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedStringAggregateV2<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

}