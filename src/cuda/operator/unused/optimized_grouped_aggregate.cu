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
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_materialize.hpp"
#include "log/logging.hpp"

namespace duckdb {

using std::chrono::duration;
using std::chrono::high_resolution_clock;

struct string_groupby_metadata {
    uint8_t** all_keys;
    uint64_t** offsets;
    uint64_t num_keys;

    __host__ __device__ string_groupby_metadata() {}
    __host__ __device__ string_groupby_metadata(uint8_t** _all_keys, uint64_t** _offsets, uint64_t _num_keys) : all_keys(_all_keys), offsets(_offsets), num_keys(_num_keys) {}
};

struct string_group_by_record {
    string_groupby_metadata* group_by_metadata;
    uint64_t row_id;
    uint64_t row_signature;

    __host__ __device__ string_group_by_record() {}
    __host__ __device__ string_group_by_record(string_groupby_metadata* _metadata, uint64_t _row_id, uint64_t _row_signature) : group_by_metadata(_metadata), row_id(_row_id), row_signature(_row_signature) {}

    __device__ __forceinline__ bool operator==(const string_group_by_record &other) const {
        // First compare the signature
        if (row_signature != other.row_signature) {
            return false;
        }

        // If the signatures match then compare the lengths
        for (uint64_t i = 0; i < group_by_metadata->num_keys; i++) {
            uint64_t* curr_column_offsets = group_by_metadata->offsets[i];
            uint64_t left_length = curr_column_offsets[row_id + 1] - curr_column_offsets[row_id];
            uint64_t right_length = curr_column_offsets[other.row_id + 1] - curr_column_offsets[other.row_id];
            if (left_length != right_length) {
                return false;
            }
        }

        // If the signature and lengths match then compare the actual chars
        for (uint64_t i = 0; i < group_by_metadata->num_keys; i++) {
            // Read in the left and right offsets
            uint64_t *curr_column_offsets = group_by_metadata->offsets[i];
            uint64_t left_read_offset = curr_column_offsets[row_id];
            uint64_t right_read_offset = curr_column_offsets[other.row_id];
            const uint64_t curr_length = curr_column_offsets[row_id + 1] - left_read_offset;

            // Initialize state for comparing the current key
            uint8_t* curr_column_chars = group_by_metadata->all_keys[i];
            uint64_t* curr_column_keys = reinterpret_cast<uint64_t*>(curr_column_chars);
            uint64_t bytes_remaining = curr_length;
            while (bytes_remaining > 0) {
                // Read in the left and right value
                uint64_t left_int_idx = left_read_offset / BYTES_IN_INTEGER;
                uint64_t left_read_idx = left_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_left_int = curr_column_keys[left_int_idx];

                uint64_t right_int_idx = right_read_offset / BYTES_IN_INTEGER;
                uint64_t right_read_idx = right_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_right_int = curr_column_keys[right_int_idx];

                // Extract the bytes we care about from these integers
                uint64_t batch_size = min(BYTES_IN_INTEGER - max(left_read_idx, right_read_idx), bytes_remaining);
                uint64_t keep_mask = (1ULL << (BITS_IN_BYTE * batch_size)) - 1;
                uint64_t left_shifted_val = curr_left_int >> (BYTES_IN_INTEGER * left_read_idx);
                uint64_t left_val = left_shifted_val & keep_mask;
                uint64_t right_shifted_val = curr_right_int >> (BYTES_IN_INTEGER * right_read_idx);
                uint64_t right_val = right_shifted_val & keep_mask;

                // Now actually compare the values
                if (left_val != right_val) {
                    return false;
                }

                // Update the trackers
                bytes_remaining -= batch_size;
                left_read_offset += batch_size;
                right_read_offset += batch_size;
            }
        }

        return true;
    }

    __device__ __forceinline__ bool operator<(const string_group_by_record &other) const {
        // First compare the signature
        if (row_signature != other.row_signature) {
            return row_signature < other.row_signature;
        }

        // If the signatures match then compare the lengths
        for (uint64_t i = 0; i < group_by_metadata->num_keys; i++) {
            uint64_t* curr_column_offsets = group_by_metadata->offsets[i];
            uint64_t left_length = curr_column_offsets[row_id + 1] - curr_column_offsets[row_id];
            uint64_t right_length = curr_column_offsets[other.row_id + 1] - curr_column_offsets[other.row_id];
            if (left_length != right_length) {
                return left_length < right_length;
            }
        }

        // If the signature and lengths match then compare the actual chars
        for (uint64_t i = 0; i < group_by_metadata->num_keys; i++) {
            // Read in the left and right offsets
            uint64_t *curr_column_offsets = group_by_metadata->offsets[i];
            uint64_t left_read_offset = curr_column_offsets[row_id];
            uint64_t right_read_offset = curr_column_offsets[other.row_id];
            const uint64_t curr_length = curr_column_offsets[row_id + 1] - left_read_offset;

            // Initialize state for comparing the current key
            uint8_t* curr_column_chars = group_by_metadata->all_keys[i];
            uint64_t* curr_column_keys = reinterpret_cast<uint64_t*>(curr_column_chars);
            uint64_t bytes_remaining = curr_length;
            while (bytes_remaining > 0) {
                // Read in the left and right value
                uint64_t left_int_idx = left_read_offset / BYTES_IN_INTEGER;
                uint64_t left_read_idx = left_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_left_int = curr_column_keys[left_int_idx];

                uint64_t right_int_idx = right_read_offset / BYTES_IN_INTEGER;
                uint64_t right_read_idx = right_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_right_int = curr_column_keys[right_int_idx];

                // Extract the bytes we care about from these integers
                uint64_t batch_size = min(BYTES_IN_INTEGER - max(left_read_idx, right_read_idx), bytes_remaining);
                uint64_t keep_mask = (1ULL << (BITS_IN_BYTE * batch_size)) - 1;
                uint64_t left_shifted_val = curr_left_int >> (BYTES_IN_INTEGER * left_read_idx);
                uint64_t left_val = left_shifted_val & keep_mask;
                uint64_t right_shifted_val = curr_right_int >> (BYTES_IN_INTEGER * right_read_idx);
                uint64_t right_val = right_shifted_val & keep_mask;

                // Now actually compare the values
                if (left_val != right_val) {
                    return left_val < right_val;
                }

                // Update the trackers
                bytes_remaining -= batch_size;
                left_read_offset += batch_size;
                right_read_offset += batch_size;
            }
        }

        return row_id < other.row_id;
    }
};

struct CustomLessGroupByRecord { 
    __host__ __device__ CustomLessGroupByRecord() {}

    __device__ __forceinline__ bool operator()(const string_group_by_record &lhs, const string_group_by_record &rhs) {
        return lhs < rhs;
    }
};

// Create the custom operators to combine the aggregate values
struct CustomMinOperator {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (b < a) ? b : a;
    }
};

struct CustomMaxOperator {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return (b > a) ? b : a;
    }
};

struct CustomSumOperator {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        return a + b;
    }
};

__global__ void create_metadata_record(string_groupby_metadata* group_by_metadata, uint8_t** keys, uint64_t** column_length_offsets, const uint64_t num_keys) {   
    group_by_metadata->all_keys = keys;
    group_by_metadata->offsets = column_length_offsets;
    group_by_metadata->num_keys = num_keys;
}

__global__ void create_row_records(string_groupby_metadata* group_by_metadata, string_group_by_record* row_records, const uint64_t num_rows) {
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Iterate through the rows in a tile based manner
    uint64_t curr_value;
    uint64_t num_keys = group_by_metadata->num_keys;
    for (uint64_t i = start_idx; i < num_rows; i += tile_size) {
        // Get the signature for this row
        uint64_t signature = 0;
        uint64_t curr_power = 1;

        for (uint64_t j = 0; j < num_keys; j++) {
            // Get the chars for this row in this column
            uint64_t* curr_column_offsets = group_by_metadata->offsets[j];
            uint64_t curr_row_start = curr_column_offsets[i];
            uint64_t curr_record_length = curr_column_offsets[i + 1] - curr_row_start;
            uint8_t* column_hash_chars = group_by_metadata->all_keys[j] + curr_row_start;

            // Update the row's signature based on this record
            for (uint64_t k = 0; k < curr_record_length; k++) {
                curr_value = static_cast<uint64_t>(column_hash_chars[k]);
                signature = (signature + curr_value * curr_power) % STRING_HASH_MOD_VALUE;
                curr_power = (curr_power * STRING_HASH_POWER) % STRING_HASH_MOD_VALUE;
            }
        }

        // Create the record for this row
        row_records[i] = string_group_by_record(group_by_metadata, i, signature);
    }
}

template <typename V>
__global__ void populate_aggregate_buffer(uint8_t** aggregate_input_keys, V* aggregate_write_buffer, uint64_t* aggregate_idxs,
    string_group_by_record* group_by_row_records, int *agg_mode, const uint64_t num_rows, const uint64_t num_aggregates) {
    
    // Use a tile based approach to 
    const uint64_t tile_size = gridDim.x * blockDim.x;
    const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (uint64_t i = start_idx; i < num_rows; i += tile_size) {
        // Copy over the aggregates into the buffer into columnar format:
        // First have column 0 records, then column 1 records, column 2 records, etc ...
        uint64_t idx_row_id = group_by_row_records[i].row_id;

        #pragma unroll
        for (uint64_t j = 0; j < num_aggregates; j++) {
            V value_to_write;
            if(agg_mode[j] == 4) {
                value_to_write = static_cast<V>(1);
            } else if(agg_mode[j] == 5) {
                value_to_write = static_cast<V>(0);
            } else {
                V* curr_aggregate_column = reinterpret_cast<V*>(aggregate_input_keys[j]);
                value_to_write = curr_aggregate_column[idx_row_id];
            }
            aggregate_write_buffer[j * num_rows + i] = value_to_write;
        }

        // Update the aggregate row record to contain the index associated with these values
        aggregate_idxs[i] = i;
    }
}

template <typename V>
__global__ void perform_average_reduction(V* aggreate_col_keys, uint64_t* group_row_ids, uint64_t num_groups) {
    const uint64_t curr_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(curr_idx < num_groups) {
        uint64_t rows_in_group = group_row_ids[curr_idx + 1] - group_row_ids[curr_idx];
        aggreate_col_keys[curr_idx] /= rows_in_group;
    }
}

__global__ void populate_original_row_ids(string_group_by_record* grouped_row_records, uint64_t* original_row_ids, uint64_t num_groups) { 
    const uint64_t curr_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(curr_idx < num_groups) {
        original_row_ids[curr_idx] = grouped_row_records[curr_idx].row_id;
    }
}

template <typename V>
void optimizedGroupedStringAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    // First check that we actually have some rows 
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        SIRIUS_LOG_DEBUG("groupedStringAggregate called with 0 rows");
        return;
    }

    // First perform preprocessing to convert the input group by columns into row level records
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    SIRIUS_LOG_DEBUG("Launching String Grouped Aggregate Kernel");

    uint64_t total_preprocessing_bytes = 2 * N * sizeof(uint64_t);
    auto preprocess_start_time = high_resolution_clock::now();
    uint8_t** d_keys = reinterpret_cast<uint8_t**>(gpuBufferManager->customCudaMalloc<void*>(num_keys, 0, 0));
    cudaMemcpy(d_keys, keys, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    uint64_t** d_offset = reinterpret_cast<uint64_t**>(gpuBufferManager->customCudaMalloc<void*>(num_keys, 0, 0));
    cudaMemcpy(d_offset, offset, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    // Create the metadata record 
    string_groupby_metadata* d_group_by_metadata = reinterpret_cast<string_groupby_metadata*>(gpuBufferManager->customCudaMalloc<string_group_by_metadata_type>(1, 0, 0));
    create_metadata_record<<<1, 1>>>(d_group_by_metadata, d_keys, d_offset, num_keys);

    // Now create a record for each row using this metadata
    string_group_by_record* d_row_records = reinterpret_cast<string_group_by_record*>(gpuBufferManager->customCudaMalloc<string_group_by_record_type>(N, 0, 0));
    uint64_t preprocess_items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t num_preprocess_blocks = (N + preprocess_items_per_block - 1) / preprocess_items_per_block;
    create_row_records<<<num_preprocess_blocks, BLOCK_THREADS>>>(d_group_by_metadata, d_row_records, N);

    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    SIRIUS_LOG_DEBUG("STRING GROUP BY: Preprocessing required {} bytes and took {} ms", total_preprocessing_bytes, preprocess_time_ms);

    // Now sort the row records
    auto sort_start_time = high_resolution_clock::now();
    CustomLessGroupByRecord custom_less_operator;
    void* sort_temp_storage = nullptr;
    size_t sort_temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        sort_temp_storage,
        sort_temp_storage_bytes,
        d_row_records,
        N,
        custom_less_operator
    );

    sort_temp_storage = reinterpret_cast<void *>(gpuBufferManager->customCudaMalloc<uint8_t>(sort_temp_storage_bytes, 0, 0));

    cub::DeviceMergeSort::SortKeys(
        sort_temp_storage,
        sort_temp_storage_bytes,
        d_row_records,
        N,
        custom_less_operator
    );

    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto sort_end_time = high_resolution_clock::now();
    auto sort_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(sort_end_time - sort_start_time).count();
    SIRIUS_LOG_DEBUG("STRING GROUP BY: Sorting required {} bytes and took {} ms", sort_temp_storage_bytes, sort_time_ms);

    // Create a seperate buffer storing the aggregate values for each sorted row record
    uint64_t total_aggregation_bytes = 0;
    auto group_by_start_time = high_resolution_clock::now();
    uint64_t num_aggregate_values = N * num_aggregates;
    V* d_aggregate_buffer = gpuBufferManager->customCudaMalloc<V>(num_aggregate_values, 0, 0);
    uint64_t* d_aggregate_idxs = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    total_aggregation_bytes += num_aggregate_values * sizeof(V) + N * sizeof(uint64_t);
    uint8_t** d_aggregate_keys = reinterpret_cast<uint8_t**>(gpuBufferManager->customCudaMalloc<void*>(num_aggregates, 0, 0));
    cudaMemcpy(d_aggregate_keys, aggregate_keys, num_aggregates * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    int *d_agg_mode = gpuBufferManager->customCudaMalloc<int>(num_aggregates, 0, 0);
    cudaMemcpy(d_agg_mode, agg_mode, num_aggregates * sizeof(int), cudaMemcpyHostToDevice);
    uint64_t aggregate_items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t num_aggregate_blocks = (N + aggregate_items_per_block - 1) / aggregate_items_per_block;
    populate_aggregate_buffer<<<num_aggregate_blocks, BLOCK_THREADS>>>(d_aggregate_keys, d_aggregate_buffer, d_aggregate_idxs, d_row_records, d_agg_mode, N, num_aggregates);

    // Now performing a reduction to determine the start and end idx associated with each group
    string_group_by_record* d_result_row_records = reinterpret_cast<string_group_by_record*>(gpuBufferManager->customCudaMalloc<string_group_by_record_type>(N, 0, 0));
    uint64_t* d_result_aggregate_idxs = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
    CustomMinOperator min_reduction_operator;

    void *d_group_by_temp_storage = nullptr;
    size_t group_by_temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(
        d_group_by_temp_storage,
        group_by_temp_storage_bytes,
        d_row_records,
        d_result_row_records,
        d_aggregate_idxs,
        d_result_aggregate_idxs,
        d_num_runs_out,
        min_reduction_operator,
        N
    );

    d_group_by_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(group_by_temp_storage_bytes, 0, 0);
    total_aggregation_bytes += group_by_temp_storage_bytes;

    cub::DeviceReduce::ReduceByKey(
        d_group_by_temp_storage,
        group_by_temp_storage_bytes,
        d_row_records,
        d_result_row_records,
        d_aggregate_idxs,
        d_result_aggregate_idxs,
        d_num_runs_out,
        min_reduction_operator,
        N
    );

    cudaDeviceSynchronize();
    CHECK_ERROR();

    // Get the number of groups
    cudaMemcpy(count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint64_t num_groups = count[0];
    cudaMemcpy(d_result_aggregate_idxs + num_groups, &N, sizeof(uint64_t), cudaMemcpyHostToDevice);

    // The resulting values of the previous ReduceByKey is an array like 0 100 300 325 450 representing the starting index for each group
    // Now we can run DeviceSegmentedReduce for each aggregate column without having to redo the string comparsion
    // This reduction will also write the result to the aggregate column directly and thus we don't need to any sort of materialization later
    for(int i = 0; i < num_aggregates; i++) {
        V* buffer_aggregate_col_values = d_aggregate_buffer + i * N;
        V* aggreate_col_keys = reinterpret_cast<V*>(aggregate_keys[i]);

        void* d_aggregate_temp_storage = nullptr;
        size_t aggregate_temp_storage_bytes = 0;
        if(agg_mode[i] == 2) {
            // Perform a max reduction
            CustomMaxOperator aggregate_max_operator;
            V aggregate_max_initial_value = static_cast<V>(std::numeric_limits<V>::min());

            cub::DeviceSegmentedReduce::Reduce(
                d_aggregate_temp_storage,
                aggregate_temp_storage_bytes,
                buffer_aggregate_col_values,
                aggreate_col_keys,
                num_groups,
                d_result_aggregate_idxs,
                d_result_aggregate_idxs + 1,
                aggregate_max_operator,
                aggregate_max_initial_value
            );

            d_aggregate_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(aggregate_temp_storage_bytes, 0, 0);
            total_aggregation_bytes += aggregate_temp_storage_bytes;

            cub::DeviceSegmentedReduce::Reduce(
                d_aggregate_temp_storage,
                aggregate_temp_storage_bytes,
                buffer_aggregate_col_values,
                aggreate_col_keys,
                num_groups,
                d_result_aggregate_idxs,
                d_result_aggregate_idxs + 1,
                aggregate_max_operator,
                aggregate_max_initial_value
            );

            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_aggregate_temp_storage), 0);
        } else if(agg_mode[i] == 3) {
            // Perform a min reduction
            CustomMinOperator aggregate_min_operator;
            V aggregate_min_initial_value = static_cast<V>(std::numeric_limits<V>::max());

            cub::DeviceSegmentedReduce::Reduce(
                d_aggregate_temp_storage,
                aggregate_temp_storage_bytes,
                buffer_aggregate_col_values,
                aggreate_col_keys,
                num_groups,
                d_result_aggregate_idxs,
                d_result_aggregate_idxs + 1,
                aggregate_min_operator,
                aggregate_min_initial_value
            );

            d_aggregate_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(aggregate_temp_storage_bytes, 0, 0);
            total_aggregation_bytes += aggregate_temp_storage_bytes;

            cub::DeviceSegmentedReduce::Reduce(
                d_aggregate_temp_storage,
                aggregate_temp_storage_bytes,
                buffer_aggregate_col_values,
                aggreate_col_keys,
                num_groups,
                d_result_aggregate_idxs,
                d_result_aggregate_idxs + 1,
                aggregate_min_operator,
                aggregate_min_initial_value
            );

            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_aggregate_temp_storage), 0);
        } else {
            // Perform a sum reduction
            CustomSumOperator aggregate_sum_operator;
            V aggregate_sum_initial_value = static_cast<V>(0);
            
            cub::DeviceSegmentedReduce::Reduce(
                d_aggregate_temp_storage,
                aggregate_temp_storage_bytes,
                buffer_aggregate_col_values,
                aggreate_col_keys,
                num_groups,
                d_result_aggregate_idxs,
                d_result_aggregate_idxs + 1,
                aggregate_sum_operator,
                aggregate_sum_initial_value
            );

            d_aggregate_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(aggregate_temp_storage_bytes, 0, 0);
            total_aggregation_bytes += aggregate_temp_storage_bytes;

            cub::DeviceSegmentedReduce::Reduce(
                d_aggregate_temp_storage,
                aggregate_temp_storage_bytes,
                buffer_aggregate_col_values,
                aggreate_col_keys,
                num_groups,
                d_result_aggregate_idxs,
                d_result_aggregate_idxs + 1,
                aggregate_sum_operator,
                aggregate_sum_initial_value
            );

            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_aggregate_temp_storage), 0);

            if(agg_mode[i] == 1) {
                // Since this is an average reduction then divide the result by the number of records in the group
                uint64_t num_group_blocks = (num_groups + BLOCK_THREADS - 1)/BLOCK_THREADS;
                perform_average_reduction<<<num_group_blocks , BLOCK_THREADS>>>(aggreate_col_keys, d_result_aggregate_idxs, num_groups);
            }
        }
        
    }

    cudaDeviceSynchronize();
    CHECK_ERROR();

    auto group_by_end_time = high_resolution_clock::now();
    auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
    SIRIUS_LOG_DEBUG("STRING GROUP BY: Group By got {} unique groups", num_groups);
    SIRIUS_LOG_DEBUG("STRING GROUP BY: Group By required {} bytes and took {} ms", total_aggregation_bytes, group_by_time_ms);

    auto post_processing_start_time = high_resolution_clock::now();
    uint64_t* d_original_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(num_groups, 0, 0);
    uint64_t populate_num_blocks = (num_groups + BLOCK_THREADS - 1)/BLOCK_THREADS;
    populate_original_row_ids<<<populate_num_blocks, BLOCK_THREADS>>>(d_result_row_records, d_original_row_ids, num_groups);

    // Materialize the string columns based on the original row ids
    for(uint64_t i = 0; i < num_keys; i++) {
        // Get the original column
        uint8_t* group_key_chars = keys[i];
        uint64_t* group_key_offsets = offset[i];
        
        // Materialize the string column
        uint8_t* result; uint64_t* result_offset; uint64_t* new_num_bytes;
        // void materializeString(uint8_t* data, uint64_t* offset, uint8_t* &result, uint64_t* &result_offset, uint64_t* row_ids, uint64_t* &result_bytes, uint64_t result_len, uint64_t input_size, uint64_t input_bytes)
        materializeString(group_key_chars, group_key_offsets, result, result_offset, d_original_row_ids, new_num_bytes, num_groups);

        // Write back the result
        keys[i] = result;
        offset[i] = result_offset;
        num_bytes[i] = new_num_bytes[0];
    }

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_group_by_metadata), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_row_records), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_offset), 0);

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_aggregate_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_aggregate_buffer), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_aggregate_idxs), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_agg_mode), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_result_row_records), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_result_aggregate_idxs), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_runs_out), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_group_by_temp_storage), 0);

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_original_row_ids), 0);

    auto post_processing_end_time = high_resolution_clock::now();
    auto post_processing_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(post_processing_end_time - post_processing_start_time).count();
    SIRIUS_LOG_DEBUG("STRING GROUP BY V3: Post Processing took {} ms", post_processing_time_ms);

    cudaDeviceSynchronize();
    CHECK_ERROR(); 
}

template
void optimizedGroupedStringAggregate<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void optimizedGroupedStringAggregate<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

__global__ void add_offset(uint64_t* a, uint64_t* b, uint64_t offset, uint64_t N) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = b[idx] + offset;
    }
}

void combineStrings(uint8_t* a, uint8_t* b, uint8_t*& c, 
        uint64_t* offset_a, uint64_t* offset_b, uint64_t*& offset_c, 
        uint64_t num_bytes_a, uint64_t num_bytes_b, uint64_t N_a, uint64_t N_b) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    c = gpuBufferManager->customCudaMalloc<uint8_t>(num_bytes_a + num_bytes_b, 0, 0);
    offset_c = gpuBufferManager->customCudaMalloc<uint64_t>(N_a + N_b + 1, 0, 0);
    cudaMemcpy(c, a, num_bytes_a * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(c + num_bytes_a, b, num_bytes_b * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    cudaMemcpy(offset_c, offset_a, N_a * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    add_offset<<<((N_b + 1) + BLOCK_THREADS - 1)/(BLOCK_THREADS), BLOCK_THREADS>>>(offset_c + N_a, offset_b, num_bytes_a, N_b + 1);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

__global__ void populate_fixed_size_offsets(uint64_t* offsets, uint64_t record_size, uint64_t num_records) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_records) {
        offsets[idx] = idx * record_size;
    }
}

uint64_t* createFixedSizeOffsets(size_t record_size, uint64_t num_rows) {
    // Create and populate offsets array
    uint64_t records_to_populate = num_rows + 1;
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint64_t* d_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(records_to_populate, 0, 0);

    uint64_t num_blocks = (records_to_populate + BLOCK_THREADS - 1)/BLOCK_THREADS;
    populate_fixed_size_offsets<<<num_blocks, BLOCK_THREADS>>>(d_offsets, static_cast<uint64_t>(record_size), records_to_populate);
    return d_offsets;
}

}