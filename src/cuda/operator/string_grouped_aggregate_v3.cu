#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"

#include <chrono>
#include <stdexcept>
#include <cub/cub.cuh>
#include <assert.h>

namespace duckdb
{

    using std::chrono::duration;
    using std::chrono::high_resolution_clock;

    struct string_groupby_metadata_v3
    {
        uint8_t **all_keys;
        uint64_t **offsets;
        uint64_t num_keys;

        __host__ __device__ string_groupby_metadata_v3() {}
        __host__ __device__ string_groupby_metadata_v3(uint8_t **_all_keys, uint64_t **_offsets, uint64_t _num_keys) : all_keys(_all_keys), offsets(_offsets), num_keys(_num_keys) {}
    };

    struct sort_keys_type_string_v3
    {
        string_groupby_metadata_v3 *group_by_metadata;
        uint64_t row_id;
        uint64_t row_signature;

        __host__ __device__ sort_keys_type_string_v3() {}
        __host__ __device__ sort_keys_type_string_v3(string_groupby_metadata_v3 *_metadata, uint64_t _row_id, uint64_t _row_signature) : group_by_metadata(_metadata), row_id(_row_id), row_signature(_row_signature) {}

        __device__ __forceinline__ bool operator==(const sort_keys_type_string_v3 &other) const
        {
            // First compare the signature
            if (row_signature != other.row_signature)
            {
                return false;
            }

            // Load the metadata into the local variables
            uint8_t **all_keys = this->group_by_metadata->all_keys;
            uint64_t **offsets = this->group_by_metadata->offsets;
            uint64_t num_keys = this->group_by_metadata->num_keys;

            // Then compare the lengths
            for (uint64_t i = 0; i < num_keys; i++)
            {
                uint64_t *curr_column_offsets = offsets[i];
                uint64_t left_length = curr_column_offsets[this->row_id + 1] - curr_column_offsets[this->row_id];
                uint64_t right_length = curr_column_offsets[other.row_id + 1] - curr_column_offsets[other.row_id];
                if (left_length != right_length)
                {
                    return false;
                }
            }

            // If the signature and lengths match then compare the actual values
            for (uint64_t i = 0; i < num_keys; i++)
            {
                // Read in the left and right offsets
                uint64_t *curr_column_offsets = offsets[i];
                const uint64_t left_read_offset = curr_column_offsets[this->row_id];
                const uint64_t right_read_offset = curr_column_offsets[other.row_id];
                const uint64_t curr_length = curr_column_offsets[this->row_id + 1] - left_read_offset;

                // Initialize left state
                uint64_t *curr_column_keys = reinterpret_cast<uint64_t *>(all_keys[i]);
                uint64_t left_int_idx = left_read_offset / BYTES_IN_INTEGER;
                uint64_t left_read_idx = left_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_left_int = curr_column_keys[left_int_idx];

                // Initialize right state
                uint64_t right_int_idx = right_read_offset / BYTES_IN_INTEGER;
                uint64_t right_read_idx = right_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_right_int = curr_column_keys[right_int_idx];

                uint64_t bytes_remaining = curr_length;
                while (bytes_remaining > 0)
                {
                    // Compare current batch of bytes
                    uint8_t batch_size = static_cast<uint8_t>(min(BYTES_IN_INTEGER - max(left_read_idx, right_read_idx), bytes_remaining));
                    uint8_t base_subtract_val = BYTES_IN_INTEGER - batch_size;
                    uint64_t keep_mask = (1ULL << (BITS_IN_BYTE * batch_size)) - 1;
                    uint8_t left_shift_val = base_subtract_val - left_read_idx;
                    uint64_t left_val = (curr_left_int >> (left_shift_val * BITS_IN_BYTE)) & keep_mask;
                    uint8_t right_shift_val = base_subtract_val - right_read_idx;
                    uint64_t right_val = (curr_right_int >> (right_shift_val * BITS_IN_BYTE)) & keep_mask;

                    // Now actually compare the values
                    if (left_val != right_val)
                    {
                        return false;
                    }

                    // Update trackers
                    bytes_remaining -= batch_size;

                    // Reload left integer if needed
                    left_read_idx += batch_size;
                    if (left_read_idx == BYTES_IN_INTEGER)
                    {
                        left_int_idx++;
                        curr_left_int = curr_column_keys[left_int_idx];
                        left_read_idx = 0;
                    }

                    // Reload right integer if needed
                    right_read_idx += batch_size;
                    if (right_read_idx == BYTES_IN_INTEGER)
                    {
                        right_int_idx++;
                        curr_right_int = curr_column_keys[right_int_idx];
                        right_read_idx = 0;
                    }
                }
            }

            return true;
        }

        __device__ __forceinline__ bool operator<(const sort_keys_type_string_v3 &other) const
        {
            // First compare the signature
            if (row_signature != other.row_signature)
            {
                return row_signature < other.row_signature;
            }

            // Load the metadata into the local variables
            uint8_t **all_keys = this->group_by_metadata->all_keys;
            uint64_t **offsets = this->group_by_metadata->offsets;
            uint64_t num_keys = this->group_by_metadata->num_keys;

            // Then compare the lengths
            for (uint64_t i = 0; i < num_keys; i++)
            {
                uint64_t *curr_column_offsets = offsets[i];
                uint64_t left_length = curr_column_offsets[this->row_id + 1] - curr_column_offsets[this->row_id];
                uint64_t right_length = curr_column_offsets[other.row_id + 1] - curr_column_offsets[other.row_id];
                if (left_length != right_length)
                {
                    return left_length < right_length;
                }
            }

            // If the signature and lengths match then compare the actual values
            for (uint64_t i = 0; i < num_keys; i++)
            {
                // Read in the left and right offsets
                uint64_t *curr_column_offsets = offsets[i];
                const uint64_t left_read_offset = curr_column_offsets[this->row_id];
                const uint64_t right_read_offset = curr_column_offsets[other.row_id];
                const uint64_t curr_length = curr_column_offsets[this->row_id + 1] - left_read_offset;

                // Initialize left state
                uint64_t *curr_column_keys = reinterpret_cast<uint64_t *>(all_keys[i]);
                uint64_t left_int_idx = left_read_offset / BYTES_IN_INTEGER;
                uint64_t left_read_idx = left_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_left_int = curr_column_keys[left_int_idx];

                // Initialize right state
                uint64_t right_int_idx = right_read_offset / BYTES_IN_INTEGER;
                uint64_t right_read_idx = right_read_offset % BYTES_IN_INTEGER;
                uint64_t curr_right_int = curr_column_keys[right_int_idx];

                uint64_t bytes_remaining = curr_length;
                while (bytes_remaining > 0)
                {
                    // Compare current batch of bytes
                    uint8_t batch_size = static_cast<uint8_t>(min(BYTES_IN_INTEGER - max(left_read_idx, right_read_idx), bytes_remaining));
                    uint8_t base_subtract_val = BYTES_IN_INTEGER - batch_size;
                    uint64_t keep_mask = (1ULL << (BITS_IN_BYTE * batch_size)) - 1;
                    uint8_t left_shift_val = base_subtract_val - left_read_idx;
                    uint64_t left_val = (curr_left_int >> (left_shift_val * BITS_IN_BYTE)) & keep_mask;
                    uint8_t right_shift_val = base_subtract_val - right_read_idx;
                    uint64_t right_val = (curr_right_int >> (right_shift_val * BITS_IN_BYTE)) & keep_mask;

                    // Now actually compare the values
                    if (left_val != right_val)
                    {
                        return left_val < right_val;
                    }

                    // Update trackers
                    bytes_remaining -= batch_size;

                    // Reload left integer if needed
                    left_read_idx += batch_size;
                    if (left_read_idx == BYTES_IN_INTEGER)
                    {
                        left_int_idx++;
                        curr_left_int = curr_column_keys[left_int_idx];
                        left_read_idx = 0;
                    }

                    // Reload right integer if needed
                    right_read_idx += batch_size;
                    if (right_read_idx == BYTES_IN_INTEGER)
                    {
                        right_int_idx++;
                        curr_right_int = curr_column_keys[right_int_idx];
                        right_read_idx = 0;
                    }
                }
            }

            return true;
        }
    };

    struct CustomLessStringV3
    {
        __host__ __device__ CustomLessStringV3() {}

        __device__ __forceinline__ bool operator()(const sort_keys_type_string_v3 &lhs, const sort_keys_type_string_v3 &rhs)
        {
            return lhs < rhs;
        }
    };

    template <typename V>
    struct CustomCombineOperatorV3
    {

        V *aggregate_buffer_start;
        int *agg_mode;
        int num_aggregates;
        uint64_t N;

        __host__ CustomCombineOperatorV3(V *_buffer_start, int *_agg_mode, int _num_aggregates, uint64_t _N) : aggregate_buffer_start(_buffer_start), agg_mode(_agg_mode), num_aggregates(_num_aggregates), N(_N)
        {
        }

        __device__ __forceinline__ uint64_t operator()(const uint64_t &left, const uint64_t &right) const
        {
            // Get the lower and right from the left and the right
            uint64_t lower_idx = min(left, right);
            uint64_t upper_idx = max(left, right);
            if(lower_idx >= N || upper_idx >= N) {
                // We can't just skip because this currently occurs due to inactive threads calling this reduction operator
                // which means that the result of this is anyways going to be ignored.
                return upper_idx;
            }

            // Merge the upper records into the lower record
            V *lower_ptr = aggregate_buffer_start + lower_idx * (num_aggregates + 1);
            V *upper_ptr = aggregate_buffer_start + upper_idx * (num_aggregates + 1);
            lower_ptr[0] += upper_ptr[0];
            for (uint64_t i = 0; i < num_aggregates; i++)
            {
                if (agg_mode[i] == 2)
                {
                    lower_ptr[i + 1] = std::max(lower_ptr[i + 1], upper_ptr[i + 1]);
                }
                else if (agg_mode[i] == 3)
                {
                    lower_ptr[i + 1] = std::min(lower_ptr[i + 1], upper_ptr[i + 1]);
                }
                else
                {
                    lower_ptr[i + 1] += upper_ptr[i + 1];
                }
            }

            // Return the lower record
            return lower_idx;
        }
    };

    __global__ void create_metadata_record(string_groupby_metadata_v3 *group_by_metadata, uint8_t **keys, uint64_t **column_length_offsets, const uint64_t num_keys)
    {
        group_by_metadata->all_keys = keys;
        group_by_metadata->offsets = column_length_offsets;
        group_by_metadata->num_keys = num_keys;
    }

    __global__ void fill_preprocess_buffer(string_groupby_metadata_v3 *group_by_metadata, sort_keys_type_string_v3 *row_records, const uint64_t num_rows)
    {
        const uint64_t tile_size = gridDim.x * blockDim.x;
        const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Create the record for the current row in a tile based manner
        uint64_t curr_value;
        uint64_t num_keys = group_by_metadata->num_keys;
        for (uint64_t i = start_idx; i < num_rows; i += tile_size)
        {
            // Get the signature for this row
            uint64_t signature = 0;
            uint64_t curr_power = 1;
#pragma unroll
            for (uint64_t j = 0; j < num_keys; j++)
            {
                // Get the chars for this row in this column
                uint64_t *curr_column_offsets = group_by_metadata->offsets[j];
                uint64_t curr_row_start = curr_column_offsets[i];
                uint64_t curr_record_length = curr_column_offsets[i + 1] - curr_row_start;
                uint8_t *column_hash_chars = group_by_metadata->all_keys[j] + curr_row_start;

// Update the signature using this record
#pragma unroll
                for (uint64_t k = 0; k < curr_record_length; k++)
                {
                    curr_value = static_cast<uint64_t>(column_hash_chars[k]);
                    signature = (signature + curr_value * curr_power) % HASH_MOD_VALUE;
                    curr_power = (curr_power * HASH_POWER) % HASH_MOD_VALUE;
                }
            }

            row_records[i] = sort_keys_type_string_v3(group_by_metadata, i, signature);
        }
    }

    template <typename V>
    __global__ void fill_aggregate_buffer(uint8_t** aggregate_input_keys, V *aggregate_write_buffer, uint64_t *aggregate_row_records,
                                          sort_keys_type_string_v3 *group_by_row_records, int *agg_mode, const uint64_t num_rows, const uint64_t num_aggregates)
    {

        const uint64_t tile_size = gridDim.x * blockDim.x;
        const uint64_t start_idx = threadIdx.x + blockIdx.x * blockDim.x;
        for (uint64_t i = start_idx; i < num_rows; i += tile_size)
        {
            uint64_t idx_row_id = group_by_row_records[i].row_id;

            // Copy over the aggregates into the buffers
            V *buffer_write_ptr = aggregate_write_buffer + i * (num_aggregates + 1);
            buffer_write_ptr[0] = static_cast<V>(1);
#pragma unroll
            for (uint64_t j = 0; j < num_aggregates; j++)
            {
                if (agg_mode[j] == 4)
                {
                    buffer_write_ptr[j + 1] = static_cast<V>(1);
                }
                else if (agg_mode[j] == 5)
                {
                    buffer_write_ptr[j + 1] = static_cast<V>(0);
                }
                else
                {
                    V* curr_aggregate_column = reinterpret_cast<V*>(aggregate_input_keys[j]);
                    buffer_write_ptr[j + 1] = curr_aggregate_column[idx_row_id];
                }
            }

            // Update the row record to contain the index to use to read these fields
            aggregate_row_records[i] = i;
        }
    }

    template <typename V>
    __global__ void perform_post_processing(uint64_t* result_aggregate_row_ids, uint64_t* group_row_ids, uint8_t** aggregate_input_keys, 
        V* aggregate_write_buffer, int* agg_mode, uint64_t num_groups, uint64_t num_aggregates) {
        
        const uint64_t curr_group_idx = threadIdx.x + blockIdx.x * blockDim.x;
        if(curr_group_idx < num_groups) {
            // Get the row id that this thread should copy over
            const uint64_t curr_row_id = result_aggregate_row_ids[curr_group_idx];
            group_row_ids[curr_group_idx] = curr_row_id;

            // Copy over the aggregates from the buffer back into the aggregate columns
            V* buffer_read_ptr = aggregate_write_buffer + curr_row_id * (num_aggregates + 1);
            V num_rows_in_group = buffer_read_ptr[0];
            
            #pragma unroll
            for (uint64_t i = 0; i < num_aggregates; i++) {
                V* curr_aggregate_column = reinterpret_cast<V*>(aggregate_input_keys[i]);
                curr_aggregate_column[curr_group_idx] = buffer_read_ptr[i + 1];
                if(agg_mode[i] == 1) {
                    curr_aggregate_column[curr_group_idx] /= num_rows_in_group;
                }
            }
        }
    }

    template <typename V>
    void groupedStringAggregateV3(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode)
    {
        CHECK_ERROR();
        if (N == 0)
        {
            count[0] = 0;
            printf("N is 0\n");
            return;
        }

        uint64_t max_value = std::numeric_limits<int>::max();
        if (N > max_value)
        {
            printf("String Group By currently only supported for at most %lu rows but got %lu rows\n", (uint64_t)max_value, (uint64_t)N);
            throw std::runtime_error("");
        }

        printf("Launching String Grouped Aggregate Kernel V3\n");
        GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());

        // First create the group by metadata
        uint64_t total_preprocessing_bytes = 2 * N * sizeof(uint64_t);
        auto preprocess_start_time = high_resolution_clock::now();
        string_groupby_metadata_v3 *d_group_by_metadata = reinterpret_cast<string_groupby_metadata_v3 *>(gpuBufferManager->customCudaMalloc<string_group_by_metadata_type>(1, 0, 0));
        create_metadata_record<<<1, 1>>>(d_group_by_metadata, keys, offset, num_keys);

        // Then create the row records using the metadata
        sort_keys_type_string_v3 *d_row_records = reinterpret_cast<sort_keys_type_string_v3 *>(gpuBufferManager->customCudaMalloc<pointer_and_two_values>(N, 0, 0));
        uint64_t items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
        uint64_t num_blocks = (N + items_per_block - 1) / items_per_block;
        fill_preprocess_buffer<<<num_blocks, BLOCK_THREADS>>>(d_group_by_metadata, d_row_records, N);

        cudaDeviceSynchronize();
        CHECK_ERROR();
        auto preprocess_end_time = high_resolution_clock::now();
        auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
        std::cout << "STRING GROUP BY V3: Preprocessing required " << total_preprocessing_bytes << " bytes and took " << preprocess_time_ms << " ms" << std::endl;

        // Perform the sort
        auto sort_start_time = high_resolution_clock::now();

        CustomLessStringV3 custom_less_comparator;
        void *sort_temp_storage = nullptr;
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
        sort_temp_storage = reinterpret_cast<void *>(gpuBufferManager->customCudaMalloc<uint8_t>(sort_temp_storage_bytes, 0, 0));

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
        uint64_t num_aggregate_values = N * (num_aggregates + 1);
        V* d_aggregate_buffer = gpuBufferManager->customCudaMalloc<V>(num_aggregate_values, 0, 0);
        uint64_t* d_aggregate_records = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
        total_aggregation_bytes += num_aggregate_values * sizeof(V) + N * sizeof(uint64_t);

        int *d_agg_mode = gpuBufferManager->customCudaMalloc<int>(num_aggregates, 0, 0);
        cudaMemcpy(d_agg_mode, agg_mode, num_aggregates * sizeof(int), cudaMemcpyHostToDevice);
        fill_aggregate_buffer<V><<<num_blocks, BLOCK_THREADS>>>(aggregate_keys, d_aggregate_buffer, d_aggregate_records, d_row_records,
                                                                d_agg_mode, N, num_aggregates);
        total_aggregation_bytes += num_aggregates * sizeof(int);

        // Create the additional fields we need to perform the group by
        sort_keys_type_string_v3* d_result_row_records = reinterpret_cast<sort_keys_type_string_v3 *>(gpuBufferManager->customCudaMalloc<pointer_and_two_values>(N, 0, 0));
        uint64_t* d_result_aggregate_records = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
        uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
        cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
        CustomCombineOperatorV3<V> reduction_operation(d_aggregate_buffer, d_agg_mode, static_cast<int>(num_aggregates), N);

        // Now actually perform the group by
        void *d_group_by_temp_storage = nullptr;
        size_t group_by_temp_storage_bytes = 0;
        cub::DeviceReduce::ReduceByKey(
            d_group_by_temp_storage,
            group_by_temp_storage_bytes,
            d_row_records,
            d_result_row_records,
            d_aggregate_records,
            d_result_aggregate_records,
            d_num_runs_out,
            reduction_operation,
            N);

        // Allocate temporary storage
        d_group_by_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(group_by_temp_storage_bytes, 0, 0);
        total_aggregation_bytes += group_by_temp_storage_bytes;

        cub::DeviceReduce::ReduceByKey(
            d_group_by_temp_storage,
            group_by_temp_storage_bytes,
            d_row_records,
            d_result_row_records,
            d_aggregate_records,
            d_result_aggregate_records,
            d_num_runs_out,
            reduction_operation,
            N);
        cudaDeviceSynchronize();
        CHECK_ERROR();

        // Get the number of groups
        cudaMemcpy(count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        auto group_by_end_time = high_resolution_clock::now();
        auto group_by_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(group_by_end_time - group_by_start_time).count();
        std::cout << "STRING GROUP BY V3: Group By required " << total_aggregation_bytes << " bytes and took " << group_by_time_ms << " ms" << std::endl;

        auto post_processing_start_time = high_resolution_clock::now();
        
        // Create the vector of row_ids and update the aggregates for those rows to the aggregates for those groups
        uint64_t num_groups = count[0];
        uint64_t* d_group_row_ids = gpuBufferManager->customCudaMalloc<uint64_t>(num_groups, 0, 0);
        uint64_t num_group_blocks = (num_groups + BLOCK_THREADS - 1) / BLOCK_THREADS;
        
        perform_post_processing<<<num_group_blocks, BLOCK_THREADS>>>(d_result_aggregate_records, d_group_row_ids, aggregate_keys, 
            d_aggregate_buffer, d_agg_mode, num_groups, num_aggregates);
        cudaDeviceSynchronize();
        CHECK_ERROR();

        // Materialize the string columns based on the row ids
        for(uint64_t i = 0; i < num_keys; i++) {
            // Get the original column
            uint8_t* group_key_chars = keys[i];
            uint64_t* group_key_offsets = offset[i];
            
            // Materailize the string column
            uint8_t* result; uint64_t* result_offset; uint64_t* new_num_bytes;
            materializeString(group_key_chars, group_key_offsets, result, result_offset, d_group_row_ids, new_num_bytes, num_groups);

            // Write back the result
            keys[i] = result;
            offset[i] = result_offset;
            num_bytes[i] = new_num_bytes[0];
        }
        
        auto post_processing_end_time = high_resolution_clock::now();
        auto post_processing_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(post_processing_end_time - post_processing_start_time).count();
        std::cout << "STRING GROUP BY V3: Post Processing took " << post_processing_time_ms << " ms" << std::endl;
        
        std::cout << "STRING GROUP BY V3: Returning NDV of " << num_groups << std::endl;
    }

    template void groupedStringAggregateV3<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

    template void groupedStringAggregateV3<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

}