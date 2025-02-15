#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

#include <chrono>
#include <stdexcept>
#include <cub/cub.cuh>
#include <assert.h>

namespace duckdb {

using std::chrono::high_resolution_clock;
using std::chrono::duration;

struct string_groupby_metadata_v3 {
    uint8_t** all_keys;
    uint64_t** offsets;
    uint64_t num_keys;

    __host__ __device__ string_groupby_metadata_v3() {}
    __host__ __device__ string_groupby_metadata_v3(uint8_t** _all_keys, uint64_t** _offsets, uint64_t _num_keys) : 
        all_keys(_all_keys), offsets(_offsets), num_keys(_num_keys) {}
}; 

struct sort_keys_type_string_v3 {
  string_groupby_metadata_v3* group_by_metadata;  
  uint64_t row_id;
  uint64_t row_signature;

  __host__ __device__ sort_keys_type_string_v3() {}
  __host__ __device__ sort_keys_type_string_v3(string_groupby_metadata_v3* _metadata, uint64_t _row_id, uint64_t _row_signature) : 
    group_by_metadata(_metadata), row_id(_row_id), row_signature(_row_signature) {}

    __host__ __device__ bool operator==(const sort_keys_type_string_v3& other) const {
        return row_signature == other.row_signature;
    }

    __host__ __device__ bool operator<(const sort_keys_type_string_v3& other) const {
        return row_signature < other.row_signature;
        /*
        // First compare the signature
        if (row_signature != other.row_signature) {
            return row_signature < other.row_signature;
        }

        // Load the metadata into the local variables
        uint8_t** all_keys = this->group_by_metadata->all_keys;
        uint64_t** offsets = this->group_by_metadata->offsets;
        uint64_t num_keys = this->group_by_metadata->num_keys;

        // Then compare the lengths
        for(uint64_t i = 0; i < num_keys; i++) {
            uint64_t* curr_column_offsets = offsets[i];
            uint64_t left_length = curr_column_offsets[this->row_id + 1] - curr_column_offsets[this->row_id];
            uint64_t right_length = curr_column_offsets[other.row_id + 1] - curr_column_offsets[other.row_id];
            if(left_length != right_length) {
                return left_length < right_length;
            }
        }

        // If the signature and lengths match then compare the actual values
        for(uint64_t i = 0; i < num_keys; i++) {
            // Read in the left and right offsets
            uint64_t* curr_column_offsets = offsets[i];
            const uint64_t left_read_offset = curr_column_offsets[this->row_id];
            const uint64_t right_read_offset = curr_column_offsets[other.row_id];
            const uint64_t curr_length = curr_column_offsets[this->row_id + 1] - left_read_offset;
            
            // Initialize left state
            uint64_t* curr_column_keys = reinterpret_cast<uint64_t*>(all_keys[i]);
            uint64_t left_int_idx = left_read_offset / BYTES_IN_INTEGER;
            uint64_t left_read_idx = left_read_offset % BYTES_IN_INTEGER;
            uint64_t curr_left_int = curr_column_keys[left_int_idx];
            
            // Initialize right state
            uint64_t right_int_idx = right_read_offset / BYTES_IN_INTEGER;
            uint64_t right_read_idx = right_read_offset % BYTES_IN_INTEGER;
            uint64_t curr_right_int = curr_column_keys[right_int_idx];

            uint64_t bytes_remaining = curr_length;
            while(bytes_remaining > 0) {
                // Compare current batch of bytes
                uint8_t batch_size = static_cast<uint8_t>(min(BYTES_IN_INTEGER - max(left_read_idx, right_read_idx), bytes_remaining));
                uint8_t base_subtract_val = BYTES_IN_INTEGER - batch_size;
                uint64_t keep_mask = (1ULL << (BITS_IN_BYTE * batch_size)) - 1;
                uint8_t left_shift_val = base_subtract_val - left_read_idx;
                uint64_t left_val = (curr_left_int >> (left_shift_val * BITS_IN_BYTE)) & keep_mask;
                uint8_t right_shift_val = base_subtract_val - right_read_idx;
                uint64_t right_val = (curr_right_int >> (right_shift_val * BITS_IN_BYTE)) & keep_mask;

                // Now actually compare the values
                if(left_val != right_val) {  
                    return left_val < right_val;
                }

                // Update trackers
                bytes_remaining -= batch_size; 

                // Reload left integer if needed
                left_read_idx += batch_size; 
                if(left_read_idx == BYTES_IN_INTEGER) {
                    left_int_idx++;
                    curr_left_int = curr_column_keys[left_int_idx];
                    left_read_idx = 0;
                }

                // Reload right integer if needed
                right_read_idx += batch_size;
                if(right_read_idx == BYTES_IN_INTEGER) {
                    right_int_idx++;
                    curr_right_int = curr_column_keys[right_int_idx];
                    right_read_idx = 0;
                }
            }
        }

        return false;
        */
    }
};

struct CustomLessStringV3 {
    __host__ __device__ CustomLessStringV3() {}

    __device__ bool operator()(const sort_keys_type_string_v3& lhs, const sort_keys_type_string_v3& rhs) {
       return lhs < rhs; 
    }

}; 

template <typename V>
struct CustomCombineOperatorV3
{

    V* aggregate_buffer_start;
    V* aggregate_buffer_end;
    int* agg_mode;
    uint64_t num_aggregates;
    uint64_t num_rows;

    __host__ CustomCombineOperatorV3(V* _buffer_start, V* _buffer_end, int* _agg_mode, uint64_t _num_aggregates, uint64_t _num_rows) : 
    aggregate_buffer_start(_buffer_start), aggregate_buffer_end(_buffer_end), agg_mode(_agg_mode), num_aggregates(_num_aggregates), num_rows(_num_rows) {
        
    }

    __device__ __forceinline__ 
    uint64_t operator()(const uint64_t& left, const uint64_t& right) const
    {
        // Get the lower and right from the left and the right
        uint64_t lower_idx = min(left, right);
        uint64_t upper_idx = max(left, right);
        if(lower_idx >= num_rows || upper_idx >= num_rows) {
            printf("COMBINE ERROR - Num Rows: %lu, Left: %lu, Right: %lu, Lower: %lu, Upper: %lu\n", 
                num_rows, left, right, lower_idx, upper_idx);
        }
        
        /*
        V* lower_ptr = reinterpret_cast<V*>(reinterpret_cast<char*>(aggregate_buffer_start) + lower_idx * sizeof(V));
        V* upper_ptr = reinterpret_cast<V*>(reinterpret_cast<char*>(aggregate_buffer_start) + upper_idx * sizeof(V));
        if(!(lower_ptr >= aggregate_buffer_start && lower_ptr <= aggregate_buffer_end) || 
            !(upper_ptr >= aggregate_buffer_start && upper_ptr <= aggregate_buffer_end)) {

            printf("COMBINE ERROR - Buffer Range: [%p, %p], Lower Idx %lu Ptr: %p, Upper Idx %lu Ptr: %p\n", 
                (void*) aggregate_buffer_start, (void*) aggregate_buffer_end, lower_idx, (void*) lower_ptr, 
                upper_idx, (void*) upper_ptr);
        }

        // Merge the upper records into the lower record
        lower_ptr[0] += upper_ptr[0];
        
        for(uint64_t i = 0; i < num_aggregates; i++) {
            if(agg_mode[i] == 2) {
                lower_ptr[i + 1] = max(lower_ptr[i + 1], upper_ptr[i + 1]);
            } else if(agg_mode[i] == 3) {
                lower_ptr[i + 1] = min(lower_ptr[i + 1], upper_ptr[i + 1]);
            } else {
                lower_ptr[i + 1] += upper_ptr[i + 1];
            }
        }
        */

        // Return the lower record
        return lower_idx;
    }
};

__global__ void create_metadata_record(string_groupby_metadata_v3* group_by_metadata, uint8_t** keys, uint64_t** column_length_offsets, const uint64_t num_keys) {
    group_by_metadata->all_keys = keys;
    group_by_metadata->offsets = column_length_offsets;
    group_by_metadata->num_keys = num_keys;
}

__global__ void fill_preprocess_buffer(string_groupby_metadata_v3* group_by_metadata, sort_keys_type_string_v3* row_records, const uint64_t num_rows) {
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Create the record for the current row in a tile based manner
    uint64_t curr_value;
    uint64_t num_keys = group_by_metadata->num_keys;
    for(uint64_t i = start_idx; i < num_rows; i += tile_size) {
        // Get the signature for this row
        uint64_t signature = 0;
        uint64_t curr_power = 1;
        #pragma unroll
        for(uint64_t j = 0; j < num_keys; j++) {
            // Get the chars for this row in this column
            uint64_t* curr_column_offsets = group_by_metadata->offsets[j];
            uint64_t curr_row_start = curr_column_offsets[i];
            uint64_t curr_record_length = curr_column_offsets[i + 1] - curr_row_start;
            uint8_t* column_hash_chars = group_by_metadata->all_keys[j] + curr_row_start;

            // Update the signature using this record
            #pragma unroll
            for(uint64_t k = 0; k < curr_record_length; k++) {
                curr_value = static_cast<uint64_t>(column_hash_chars[k]);
                signature = (signature + curr_value * curr_power) % HASH_MOD_VALUE;
                curr_power = (curr_power * HASH_POWER) % HASH_MOD_VALUE;
            }
        }

        row_records[i] = sort_keys_type_string_v3(group_by_metadata, i, signature);
    }
}

template<typename V>
__global__ void fill_aggregate_buffer(uint8_t** aggregate_input_keys, V* aggregate_write_buffer, uint64_t* aggregate_row_records, 
    sort_keys_type_string_v3* group_by_row_records, int* agg_mode, const uint64_t num_rows, const uint64_t num_aggregates) {
    
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    for(uint64_t i = start_idx; i < num_rows; i += tile_size) {
        uint64_t idx_row_id = group_by_row_records[i].row_id;

        // Copy over the aggregates into the buffers
        V* buffer_write_ptr = aggregate_write_buffer + i * (num_aggregates + 1);
        buffer_write_ptr[0] = static_cast<V>(1);
        #pragma unroll
        for(uint64_t j = 0; j < num_aggregates; j++) {
            if(agg_mode[j] == 4) {
                buffer_write_ptr[j + 1] = static_cast<V>(1);
            } else if(agg_mode[j] == 5) {
                buffer_write_ptr[j + 1] = static_cast<V>(0);
            } else {
                V* curr_aggregate_column = reinterpret_cast<V*>(aggregate_input_keys[j]);
                buffer_write_ptr[j + 1] = curr_aggregate_column[idx_row_id];
            }
        }

        // Update the row record to contain the index to use to read these fields
        aggregate_row_records[i] = i;
    }
}

template <typename V>
void groupedStringAggregateV3(uint8_t** keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("N is 0\n");
        return;
    }

    int max_value = std::numeric_limits<int>::max();
    if(N > max_value) {
        printf("String Group By currently only supported for at most %lu rows but got %lu rows\n", (uint64_t) max_value, (uint64_t) N);
        throw std::runtime_error("");
    }

    printf("Launching String Grouped Aggregate Kernel V3\n");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

    // First create the group by metadata
    uint64_t total_preprocessing_bytes = 2 * N * sizeof(uint64_t);
    auto preprocess_start_time = high_resolution_clock::now();
    string_groupby_metadata_v3* d_group_by_metadata = reinterpret_cast<string_groupby_metadata_v3*>(gpuBufferManager->customCudaMalloc<string_group_by_metadata_type>(1, 0, 0));
    create_metadata_record<<<1, 1>>>(d_group_by_metadata, keys, offset, num_keys);

    // Then create the row records using the metadata
    sort_keys_type_string_v3* d_row_records = reinterpret_cast<sort_keys_type_string_v3*>(gpuBufferManager->customCudaMalloc<pointer_and_two_values>(N, 0, 0));
    uint64_t items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t num_blocks = (N + items_per_block - 1)/items_per_block;
    fill_preprocess_buffer<<<num_blocks, BLOCK_THREADS>>>(d_group_by_metadata, d_row_records, N);

    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    std::cout << "STRING GROUP BY V3: Preprocessing required " << total_preprocessing_bytes << " bytes and took " << preprocess_time_ms << " ms" << std::endl;

    // Perform the sort
    auto sort_start_time = high_resolution_clock::now();

    CustomLessStringV3 custom_less_comparator;
    void* sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        sort_temp_storage,
        sort_temp_storage_bytes,
        d_row_records,
        N,
        custom_less_comparator);

    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Allocate temporary storage
    sort_temp_storage = reinterpret_cast<void*>(gpuBufferManager->customCudaMalloc<uint8_t>(sort_temp_storage_bytes, 0, 0));

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
    std::cout << "STRING GROUP BY V3: Sorting required " << sort_temp_storage_bytes << " bytes and took " << sort_time_ms << " ms" << std::endl;

    // Create a buffer of the aggregate values as well as an array of row ids
    uint64_t total_aggregation_bytes = 0;
    auto group_by_start_time = high_resolution_clock::now();
    uint64_t total_aggregates_values = N * (num_aggregates + 1);
    V* d_aggregate_buffer = reinterpret_cast<V*>(gpuBufferManager->customCudaMalloc<uint8_t>(total_aggregates_values * sizeof(V), 0, 0));
    uint64_t* d_aggregate_records = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    total_aggregation_bytes += total_aggregates_values * sizeof(V) + N * sizeof(uint64_t);
    
    int* d_agg_mode = gpuBufferManager->customCudaMalloc<int>(num_aggregates, 0, 0);
    cudaMemcpy(d_agg_mode, agg_mode, num_aggregates * sizeof(int), cudaMemcpyHostToDevice);
    fill_aggregate_buffer<V><<<num_blocks, BLOCK_THREADS>>>(aggregate_keys, d_aggregate_buffer, d_aggregate_records, d_row_records, 
        d_agg_mode, N, num_aggregates);
    total_aggregation_bytes += num_aggregates * sizeof(int); 

    // Get the range of values in the aggregate records
    void* d_max_temp_storage = nullptr;
    size_t max_temp_storage_bytes = 0;
    uint64_t* d_max_val = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cub::DeviceReduce::Max(d_max_temp_storage, max_temp_storage_bytes, d_aggregate_records, d_max_val, N);
    d_max_temp_storage = reinterpret_cast<void*>(gpuBufferManager->customCudaMalloc<uint8_t>(max_temp_storage_bytes, 0, 0));
    cub::DeviceReduce::Max(d_max_temp_storage, max_temp_storage_bytes, d_aggregate_records, d_max_val, N);
    uint64_t max_val = 0;
    cudaMemcpy(&max_val, d_max_val, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    std::cout << "STRING GROUP BY V3: Aggregate Records has maximum value of " << max_val << std::endl;

    // Create the additional fields we need to perform the group by
    sort_keys_type_string_v3* d_result_row_records = reinterpret_cast<sort_keys_type_string_v3*>(gpuBufferManager->customCudaMalloc<pointer_and_two_values>(N, 0, 0));
    uint64_t* d_result_aggregate_records = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
    V* d_aggregate_end = reinterpret_cast<V*>(reinterpret_cast<char*>(d_aggregate_buffer) + total_aggregates_values * sizeof(V));
    CustomCombineOperatorV3<V> reduction_operation(d_aggregate_buffer, d_aggregate_end, d_agg_mode, num_aggregates, N);
    
    // Set both to zero for testing purposes
    cudaMemset(d_aggregate_records, 0, N * sizeof(uint64_t));
    cudaMemset(d_result_aggregate_records, 0, N * sizeof(uint64_t));
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Now actually perform the group by
    void*  d_group_by_temp_storage = nullptr;
    size_t group_by_temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey<
        sort_keys_type_string_v3*,
        sort_keys_type_string_v3*,
        uint64_t*,
        uint64_t*,
        uint64_t*,
        CustomCombineOperatorV3<V>,
        uint64_t    
    >(
        d_group_by_temp_storage, 
        group_by_temp_storage_bytes, 
        d_row_records, 
        d_result_row_records, 
        d_aggregate_records, 
        d_result_aggregate_records, 
        d_num_runs_out, 
        reduction_operation, 
        N
    );
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Allocate temporary storage
    cudaMalloc(&d_group_by_temp_storage, group_by_temp_storage_bytes);
    total_aggregation_bytes += group_by_temp_storage_bytes;
    cudaDeviceSynchronize();
    CHECK_ERROR();
    cudaMemset(d_group_by_temp_storage, 0, group_by_temp_storage_bytes * sizeof(char));

    cub::DeviceReduce::ReduceByKey<
        sort_keys_type_string_v3*,
        sort_keys_type_string_v3*,
        uint64_t*,
        uint64_t*,
        uint64_t*,
        CustomCombineOperatorV3<V>,
        uint64_t    
    >(
        d_group_by_temp_storage, 
        group_by_temp_storage_bytes, 
        d_row_records, 
        d_result_row_records, 
        d_aggregate_records,
        d_result_aggregate_records, 
        d_num_runs_out, 
        reduction_operation, 
        N
    );
    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Get the number of groups
    cudaMemcpy(count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    auto group_by_end_time = high_resolution_clock::now();
    auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
    std::cout << "STRING GROUP BY V3: Group By found " << count[0] << " distinct groups" << std::endl;
    std::cout << "STRING GROUP BY V3: Group By required " << total_aggregation_bytes << " bytes and took " << group_by_time_ms << " ms" << std::endl;

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