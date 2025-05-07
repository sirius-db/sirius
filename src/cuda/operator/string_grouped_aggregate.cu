#include "cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
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
void groupedStringAggregate(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode) {
    // First check that we actually have some rows 
    CHECK_ERROR();
    if (N == 0) {
        count[0] = 0;
        printf("groupedStringAggregate called with 0 rows\n");
        return;
    }

    // First perform preprocessing to convert the input group by columns into row level records
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    printf("Launching String Grouped Aggregate Kernel\n");
    
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

    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;

    //cubmax
    // Get the maximum key length for each key
    uint64_t* key_length = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys, 0, 0); // store the maximum length of each key
    uint64_t** len = new uint64_t*[num_keys];
    uint64_t* original_bytes = new uint64_t[num_keys];
    for (int key = 0; key < num_keys; key++) {
        len[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);

        if (offset[key] == nullptr) {
            offset[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
            fill_offset<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD), BLOCK_THREADS>>>(offset[key], N+1);
            CHECK_ERROR();
        }
        cudaMemcpy(original_bytes + key, offset[key] + N, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        get_len<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + BLOCK_THREADS * ITEMS_PER_THREAD - 1)/(BLOCK_THREADS * ITEMS_PER_THREAD), BLOCK_THREADS>>>(offset[key], len[key], N);
        CHECK_ERROR();
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;

        if (offset[key] == nullptr) {
            cudaMemcpy(key_length + key, len[key], sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        } else {
            cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, len[key], key_length + key, N);

            // Allocate temporary storage
            d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

            // Run min-reduction
            cub::DeviceReduce::Max(
            d_temp_storage, temp_storage_bytes, len[key], key_length + key, N);
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
        }
    }

    uint64_t* h_key_length = new uint64_t[num_keys];
    cudaMemcpy(h_key_length, key_length, num_keys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    CHECK_ERROR();

    uint64_t row_id_size = sizeof(uint64_t);
    uint64_t total_length = 0;
    for (uint64_t key = 0; key < num_keys; key ++) {
        total_length += h_key_length[key];
    }
    //add the row ids into the total length
    total_length += row_id_size;
    uint64_t meta_num_keys = (total_length + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    uint64_t total_length_bytes = meta_num_keys * sizeof(uint64_t);
    // printf("Total Length: %lu\n", total_length);
    // printf("Total Length Bytes: %lu\n", total_length_bytes);

    //allocate temp memory and copying keys
    uint8_t* row_keys = gpuBufferManager->customCudaMalloc<uint8_t>((total_length_bytes) * N, 0, 0);
    sort_keys_type_string* materialized_temp = reinterpret_cast<sort_keys_type_string*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));

    uint8_t** keys_row_id = new uint8_t*[num_keys + 1];
    for (uint64_t i = 0; i < num_keys; i++) {
        keys_row_id[i] = keys[i];
    }

    //generate sequence
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    uint64_t* row_sequence = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    sequence<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(row_sequence, N);
    keys_row_id[num_keys] = reinterpret_cast<uint8_t*> (row_sequence);

    uint8_t** keys_dev;
    cudaMalloc((void**) &keys_dev, (num_keys + 1) * sizeof(uint8_t*));
    cudaMemcpy(keys_dev, keys_row_id, (num_keys + 1) * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    uint64_t** offset_dev;
    cudaMalloc((void**) &offset_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(offset_dev, offset, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    columns_to_rows_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(keys_dev, row_keys, offset_dev, key_length,
            materialized_temp, N, num_keys + 1);
    CHECK_ERROR();

    //perform sort-based groupby
    // Determine temporary device storage requirements
    CustomLessString custom_less;
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_less);

    CHECK_ERROR();

    // Allocate temporary storage
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run sorting operation
    cub::DeviceMergeSort::SortKeys(
        d_temp_storage,
        temp_storage_bytes,
        materialized_temp,
        N,
        custom_less);

    CHECK_ERROR();

    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

    printf("Gathering offset\n");
    uint64_t** group_byte_offset = new uint64_t*[num_keys];
    uint64_t* distinct_bound = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
    uint64_t* group_idx = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
    uint64_t* d_num_bytes = gpuBufferManager->customCudaMalloc<uint64_t>(num_keys, 0, 0);

    for (uint64_t key = 0; key < num_keys; key++) {
        uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
        group_byte_offset[key] = gpuBufferManager->customCudaMalloc<uint64_t>(N + 1, 0, 0);
        cudaMemset(group_byte_offset[key] + N, 0, sizeof(uint64_t));

        gather_and_modify<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(len[key], temp, materialized_temp, N, meta_num_keys);
        CHECK_ERROR();
        distinct_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(distinct_bound, temp, temp, materialized_temp, N);
        CHECK_ERROR();
        //cub scan
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp, group_byte_offset[key], N + 1);

        // Allocate temporary storage for exclusive prefix sum
        d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

        // Run exclusive prefix sum
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, temp, group_byte_offset[key], N + 1);
        CHECK_ERROR();
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

        cudaMemcpy(d_num_bytes + key, group_byte_offset[key] + N, sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(temp), 0);
        CHECK_ERROR();
    }

    //copy num_bytes over
    cudaMemcpy(num_bytes, d_num_bytes, num_keys * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint64_t** group_byte_offset_dev;
    cudaMalloc((void**) &group_byte_offset_dev, num_keys * sizeof(uint64_t*));
    cudaMemcpy(group_byte_offset_dev, group_byte_offset, num_keys * sizeof(uint64_t*), cudaMemcpyHostToDevice);

    //cub scan
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, distinct_bound, group_idx, N + 1);

    // Allocate temporary storage for exclusive prefix sum
    d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, distinct_bound, group_idx, N + 1);
    CHECK_ERROR();
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

    //gather the aggregates based on the row_sequence
    printf("Gathering Aggregates\n");
    V** aggregate_keys_temp = new V*[num_aggregates];
    uint64_t** aggregate_star_temp = new uint64_t*[num_aggregates];
    sort_keys_type_string* group_by_rows = reinterpret_cast<sort_keys_type_string*> (gpuBufferManager->customCudaMalloc<pointer_and_key>(N, 0, 0));
    uint64_t* d_num_runs_out = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);
    uint8_t** output_agg = new uint8_t*[num_aggregates];
    uint64_t* h_count = new uint64_t[1];

    for (int agg = 0; agg < num_aggregates; agg++) {
        printf("Aggregating %d\n", agg);
        cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));
        if (agg_mode[agg] == 4 || agg_mode[agg] == 5) { //count_star or count(null) or sum(null)
            aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            if (agg_mode[agg] == 4) {
                fill_n<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], 1, N);
            } else if (agg_mode[agg] == 5) {
                cudaMemset(aggregate_star_temp[agg], 0, N * sizeof(double));
            }

            modify<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(materialized_temp, N, meta_num_keys);
            CHECK_ERROR();

            //perform reduce_by_key
            uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
            cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));

            printf("Reduce by key count_star\n");
            // Determine temporary device storage requirements
            d_temp_storage = nullptr;
            temp_storage_bytes = 0;
            CustomSumString custom_sum;
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, aggregate_star_temp[agg],
                agg_star_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            // Allocate temporary storage
            d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

            // Run reduce-by-key
            cub::DeviceReduce::ReduceByKey(
                d_temp_storage, temp_storage_bytes,
                materialized_temp, group_by_rows, aggregate_star_temp[agg],
                agg_star_out, d_num_runs_out, custom_sum, N);

            CHECK_ERROR();

            cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_star_temp[agg]), 0);
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
            count[0] = h_count[0];

            printf("Count: %lu\n", count[0]);

            CHECK_ERROR();
            output_agg[agg] = reinterpret_cast<uint8_t*> (agg_star_out);
        } else {
            aggregate_keys_temp[agg] = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            V* temp = reinterpret_cast<V*> (aggregate_keys[agg]);
            gather_and_modify<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(temp, aggregate_keys_temp[agg], materialized_temp, N, meta_num_keys);
            CHECK_ERROR();

            V* agg_out = gpuBufferManager->customCudaMalloc<V>(N, 0, 0);
            cudaMemset(agg_out, 0, N * sizeof(V));

            CHECK_ERROR();
            if (agg_mode[agg] == 0) {
                printf("Reduce by key sum\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSumString custom_sum;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (agg_out);
                printf("Count: %lu\n", count[0]);
            } else if (agg_mode[agg] == 1) {
                //Currently typename V has to be a double
                printf("Reduce by key avg\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomSumString custom_sum;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);

                aggregate_star_temp[agg] = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
                fill_n<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(aggregate_star_temp[agg], 1, N);

                uint64_t* agg_star_out = gpuBufferManager->customCudaMalloc<uint64_t>(N, 0, 0);
                cudaMemset(agg_star_out, 0, N * sizeof(uint64_t));
                cudaMemset(d_num_runs_out, 0, sizeof(uint64_t));

                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_star_temp[agg],
                    agg_star_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_star_temp[agg],
                    agg_star_out, d_num_runs_out, custom_sum, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                V* output = gpuBufferManager->customCudaMalloc<V>(count[0], 0, 0);
                divide<V, BLOCK_THREADS, ITEMS_PER_THREAD><<<(count[0] + tile_items - 1)/tile_items, BLOCK_THREADS>>>(agg_out, agg_star_out, output, count[0]);

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_star_temp[agg]), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(agg_star_out), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(agg_out), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (output);
            } else if (agg_mode[agg] == 2) {
                printf("Reduce by key max\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMaxString custom_max;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_max, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_max, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (agg_out);
            } else if (agg_mode[agg] == 3) {
                printf("Reduce by key min\n");
                // Determine temporary device storage requirements
                d_temp_storage = nullptr;
                temp_storage_bytes = 0;
                CustomMinString custom_min;
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_min, N);

                CHECK_ERROR();

                // Allocate temporary storage
                d_temp_storage = reinterpret_cast<void*> (gpuBufferManager->customCudaMalloc<uint8_t>(temp_storage_bytes, 0, 0));

                // Run reduce-by-key
                cub::DeviceReduce::ReduceByKey(
                    d_temp_storage, temp_storage_bytes,
                    materialized_temp, group_by_rows, aggregate_keys_temp[agg],
                    agg_out, d_num_runs_out, custom_min, N);

                CHECK_ERROR();

                cudaMemcpy(h_count, d_num_runs_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
                count[0] = h_count[0];

                CHECK_ERROR();
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_temp_storage), 0);
                gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys_temp[agg]), 0);
                output_agg[agg] = reinterpret_cast<uint8_t*> (agg_out);
            }
        }
    }

    uint64_t** offset_dev_result;
    cudaMalloc((void**) &offset_dev_result, num_keys * sizeof(uint64_t*));
    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(offset[i]), 0);
        offset[i] = gpuBufferManager->customCudaMalloc<uint64_t>(count[0], 0, 0);
    }
    cudaMemcpy(offset_dev_result, offset, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    compact_string_offset<BLOCK_THREADS, ITEMS_PER_THREAD><<<((N + 1) + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            group_idx, group_byte_offset_dev, offset_dev_result, N + 1, num_keys);

    CHECK_ERROR();

    uint8_t** keys_dev_result;
    cudaMalloc((void**) &keys_dev_result, num_keys * sizeof(uint8_t*));
    for (uint64_t i = 0; i < num_keys; i++) {
        uint64_t* temp_num_bytes = new uint64_t[1];
        cudaMemcpy(temp_num_bytes, offset[i] + count[0], sizeof(uint64_t), cudaMemcpyDeviceToHost);
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(keys[i]), 0);
        keys[i] = gpuBufferManager->customCudaMalloc<uint8_t>(temp_num_bytes[0], 0, 0);
    }
    cudaMemcpy(keys_dev_result, keys, num_keys * sizeof(uint8_t*), cudaMemcpyHostToDevice);
    CHECK_ERROR();

    rows_to_columns_string<BLOCK_THREADS, ITEMS_PER_THREAD><<<(N + tile_items - 1)/tile_items, BLOCK_THREADS>>>(
            group_idx, group_by_rows, keys_dev_result, group_byte_offset_dev, key_length, N, num_keys);

    CHECK_ERROR();

    // testprint<uint64_t><<<1, 1>>>(group_idx, N);
    // testprint<double><<<1, 1>>>(reinterpret_cast<double*> (aggregate_keys[0]), N);
    // testprint<uint64_t><<<1, 1>>>(offset[1], N);
    // CHECK_ERROR();

    for (int agg = 0; agg < num_aggregates; agg++) {
        if (agg_mode[agg] >= 0 && agg_mode[agg] <= 3) {
            gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(aggregate_keys[agg]), 0);
            aggregate_keys[agg] = output_agg[agg];
        } else {
            aggregate_keys[agg] = output_agg[agg];
        }
    }

    for (uint64_t i = 0; i < num_keys; i++) {
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(len[i]), 0);
        gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_byte_offset[i]), 0);
    }

    //free row_keys, row_sequence, materialized_temp
    cudaFree(keys_dev);
    cudaFree(offset_dev);
    cudaFree(keys_dev_result);
    cudaFree(offset_dev_result);
    cudaFree(group_byte_offset_dev);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_keys), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(row_sequence), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(materialized_temp), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_by_rows), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_runs_out), 0); 
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(key_length), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(distinct_bound), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(group_idx), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(d_num_bytes), 0);
    cudaDeviceSynchronize();
    CHECK_ERROR();
    auto preprocess_end_time = high_resolution_clock::now();
    auto preprocess_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(preprocess_end_time - preprocess_start_time).count();
    std::cout << "STRING GROUP BY: Preprocessing required " << total_preprocessing_bytes << " bytes and took " << preprocess_time_ms << " ms" << std::endl;

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
    std::cout << "STRING GROUP BY: Sorting required " << sort_temp_storage_bytes << " bytes and took " << sort_time_ms << " ms" << std::endl;

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
    std::cout << "STRING GROUP BY: Group By got " << num_groups << " unique groups" << std::endl;
    std::cout << "STRING GROUP BY: Group By required " << total_aggregation_bytes << " bytes and took " << group_by_time_ms << " ms" << std::endl;

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
        materializeString(group_key_chars, group_key_offsets, result, result_offset, d_original_row_ids, new_num_bytes, num_groups);

        // Write back the result
        keys[i] = result;
        offset[i] = result_offset;
        num_bytes[i] = new_num_bytes[0];
    }

    auto post_processing_end_time = high_resolution_clock::now();
    auto post_processing_time_ms = std::chrono::duration_cast<duration<double, std::milli>>(post_processing_end_time - post_processing_start_time).count();
    std::cout << "STRING GROUP BY V3: Post Processing took " << post_processing_time_ms << " ms" << std::endl;
}

template
void groupedStringAggregate<double>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

template
void groupedStringAggregate<uint64_t>(uint8_t **keys, uint8_t **aggregate_keys, uint64_t** offset, uint64_t* num_bytes, uint64_t* count, uint64_t N, uint64_t num_keys, uint64_t num_aggregates, int* agg_mode);

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
        printf("N is 0\n");
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