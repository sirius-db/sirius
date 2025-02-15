#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

#include <chrono>
#include <stdexcept>

namespace duckdb {

using std::chrono::high_resolution_clock;
using std::chrono::duration;

/*
Each row record is going to have the following format:
Row ID | Column 1 Record Length |  Column 2 Record Length | ... | Column 1 Value | Column 2 Value | ...
*/
struct group_by_cols_row_record {
    uint64_t* row_record_values;
    uint64_t signature;
    uint64_t num_records;

    __host__ __device__ group_by_cols_row_record() {}
    __host__ __device__ group_by_cols_row_record(uint64_t* _record_values, uint64_t _signature, uint64_t _num_records) : 
    row_record_values(_record_values), signature(_signature), num_records(_num_records) {}

    __host__ __device__ bool operator<(const group_by_cols_row_record& other) const {
        // First compare the signatures
        if (signature != other.signature) {
            return signature < other.signature;
        }

        // Then compare the actual record values
        #pragma unroll
        for(uint64_t j = 1; j < num_records; j++) {
            if(row_record_values[j] != other.row_record_values[j]) {
                return row_record_values[j] < other.row_record_values[j];
            }
        }

        
        return true;
    }    
};

struct CustomLessRow
{
    __host__ __device__ CustomLessRow() {}

    __device__ bool operator()(const group_by_cols_row_record& lhs, const group_by_cols_row_record& rhs) {
        return lhs < rhs;
    }
};

__global__ void determine_row_memories(uint64_t** column_length_offsets, uint64_t* row_offsets, const uint64_t num_keys, const uint64_t num_rows) {
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

__global__ void fill_preprocess_buffer(uint8_t** keys, uint64_t** column_length_offsets, uint64_t* write_offsets, uint64_t* preprocess_buffer, group_by_cols_row_record* row_records, const uint64_t num_keys, const uint64_t num_rows) {
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
            
            // Update the signature
            uint64_t curr_value;
            #pragma unroll
            for(uint64_t k = 0; k < curr_record_length; k++) {
                curr_value = static_cast<uint64_t>(column_hash_chars[k]);
                signature = (signature + curr_value * curr_power) % HASH_MOD_VALUE;
                curr_power = (curr_power * HASH_POWER) % HASH_MOD_VALUE;
            }

            // Update the write offset
            chars_write_offset += (curr_record_length + BYTES_IN_INTEGER - 1)/BYTES_IN_INTEGER;
        }

        // Create the row record
        curr_row_write_ptr[0] = i;
        row_records[i] = group_by_cols_row_record(curr_row_write_ptr, signature, chars_write_offset);
    }
}

template <typename V>
void groupedStringAggregateV4(uint8_t** keys, uint8_t** aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }

    printf("Launching String Grouped Aggregate Kernel V4\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    // Create the temporary buffer
    uint64_t total_preprocessing_bytes = 0;
    auto preprocess_start_time = high_resolution_clock::now();

    // First use the lengths to determine the memory need for this row
    uint64_t* d_preprocess_write_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    uint64_t items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t num_blocks = (N + items_per_block - 1)/items_per_block;
    determine_row_memories<<<num_blocks, BLOCK_THREADS>>>(offset, d_preprocess_write_offsets, num_keys, N);

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
    group_by_cols_row_record* d_row_records = reinterpret_cast<group_by_cols_row_record*>(gpuBufferManager->customCudaMalloc<pointer_and_two_values>(N, 0, 0));
    fill_preprocess_buffer<<<num_blocks, BLOCK_THREADS>>>(keys, offset, d_preprocess_write_offsets, d_preprocess_buffer, d_row_records, num_keys, N);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    std::cout << "STRING GROUP BY V4: Preprocessing required " << total_preprocessing_bytes << " bytes and took " << preprocess_time_ms << " ms" << std::endl;

    // Perform the sort
    auto sort_start_time = high_resolution_clock::now();
    CustomLessRow custom_less_comparator;
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
    std::cout << "STRING GROUP BY V4: Sorting required " << sort_temp_storage_bytes << " bytes and took " << sort_time_ms << " ms" << std::endl;

    // First create the buffer of aggregate values
    uint64_t total_aggregation_bytes = 0;
    auto group_by_start_time = high_resolution_clock::now();
    /*
    uint64_t total_aggregates_values = N * num_aggregates;
    V* d_aggregate_buffer = gpuBufferManager->customCudaMalloc<V>(total_aggregates_values, 0, 0);
    aggregate_cols_row_record<V>* d_aggregate_records = reinterpret_cast<aggregate_cols_row_record<V>*>(gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));

    int* d_agg_mode = gpuBufferManager->customCudaMalloc<int>(num_aggregates, 0, 0);
    cudaMemcpy(d_agg_mode, agg_mode, num_aggregates * sizeof(int), cudaMemcpyHostToDevice);
    fill_aggregate_buffer<V><<<num_blocks, BLOCK_THREADS>>>(aggregate_keys, d_aggregate_buffer, d_aggregate_records, d_row_records, d_agg_mode, N, num_aggregates);
    total_aggregation_bytes += total_aggregates_values * sizeof(V) + N * sizeof(aggregate_cols_row_record<V>) + num_aggregates * sizeof(int); 
    */

    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto group_by_end_time = high_resolution_clock::now();
    auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
    std::cout << "STRING GROUP BY V4: Group By required " << total_aggregation_bytes << " bytes taking " << group_by_time_ms << " ms" << std::endl;

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