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

#include "config.hpp"
#include "cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_physical_order.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

namespace duckdb {

struct str_top_n_record_type { 
    uint32_t row_id;
	uint32_t key_prefix;

    __host__ __device__ str_top_n_record_type() {}
    __host__ __device__ str_top_n_record_type(uint32_t _row_id, uint32_t _key_prefix) : row_id(_row_id), key_prefix(_key_prefix) {}

    __device__ __forceinline__ bool operator==(const str_top_n_record_type& other) const { 
        return key_prefix == other.key_prefix;
    }

    __device__ __forceinline__ bool operator<(const str_top_n_record_type& other) const { 
        return key_prefix < other.key_prefix;
    }
};

struct CustomTopNStringPrefixLessThan { 
  __forceinline__ __device__ bool operator()(const str_top_n_record_type &lhs, const str_top_n_record_type &rhs) { 
    return lhs.key_prefix < rhs.key_prefix;
  }
};

struct CustomTopNStringLessThan { 
  uint8_t* col_chars;
  uint64_t* col_offsets;

  __host__ __device__ CustomTopNStringLessThan(uint8_t* _col_chars_, uint64_t* _col_offsets_) : col_chars(_col_chars_), col_offsets(_col_offsets_) {}

  __forceinline__ __device__ bool operator()(const str_top_n_record_type &lhs, const str_top_n_record_type &rhs) { 
    // First compare the prefixes
    if(lhs.key_prefix != rhs.key_prefix) { 
        return lhs.key_prefix < rhs.key_prefix;
    }

    // Load the chars and lengths for the input records
    uint32_t left_row_id = lhs.row_id; uint64_t left_start_offset = col_offsets[left_row_id]; 
    uint64_t left_len = col_offsets[left_row_id + 1] - left_start_offset; uint8_t* left_chars = col_chars + left_start_offset;

    uint32_t right_row_id = rhs.row_id; uint64_t right_start_offset = col_offsets[right_row_id]; 
    uint64_t right_len = col_offsets[right_row_id + 1] - right_start_offset; uint8_t* right_chars = col_chars + right_start_offset;

    // Now compare the chars
    const uint64_t num_cmp_values = min(left_len, right_len);
    #pragma unroll
    for(uint64_t i = 0; i < num_cmp_values; i++) { 
        if(left_chars[i] != right_chars[i]) return left_chars[i] < right_chars[i];
    }

    return left_len < right_len;
  }
};

struct CustomTopNStringPrefixGreaterThan { 
  __forceinline__ __device__ bool operator()(const str_top_n_record_type &lhs, const str_top_n_record_type &rhs) { 
    return lhs.key_prefix > rhs.key_prefix;
  }
};

struct CustomTopNStringGreaterThan { 
  uint8_t* col_chars;
  uint64_t* col_offsets;

  __host__ __device__ CustomTopNStringGreaterThan(uint8_t* _col_chars_, uint64_t* _col_offsets_) : col_chars(_col_chars_), col_offsets(_col_offsets_) {}

  __forceinline__ __device__ bool operator()(const str_top_n_record_type &lhs, const str_top_n_record_type &rhs) { 
    // First compare the prefixes
    if(lhs.key_prefix != rhs.key_prefix) { 
        return lhs.key_prefix > rhs.key_prefix;
    }

    // Load the chars and lengths for the input records
    uint32_t left_row_id = lhs.row_id; uint64_t left_start_offset = col_offsets[left_row_id]; 
    uint64_t left_len = col_offsets[left_row_id + 1] - left_start_offset; uint8_t* left_chars = col_chars + left_start_offset;

    uint32_t right_row_id = rhs.row_id; uint64_t right_start_offset = col_offsets[right_row_id]; 
    uint64_t right_len = col_offsets[right_row_id + 1] - right_start_offset; uint8_t* right_chars = col_chars + right_start_offset;

    // Now compare the chars
    const uint64_t num_cmp_values = min(left_len, right_len);
    #pragma unroll
    for(uint64_t i = 0; i < num_cmp_values; i++) { 
        if(left_chars[i] != right_chars[i]) return left_chars[i] > right_chars[i];
    }

    return left_len > right_len;
  }
};

struct CustomTopNMin {
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
      return a < b;
    }
};

__global__ void create_top_n_records(str_top_n_record_type* records, uint32_t* indexes, uint8_t* col_chars, uint64_t* col_offsets, uint32_t num_records) {
    uint32_t row_id = threadIdx.x + blockIdx.x * blockDim.x;
    if(row_id < num_records) { 
        uint64_t curr_record_start = col_offsets[row_id];
        uint64_t curr_record_len = col_offsets[row_id + 1] - curr_record_start;
        const uint64_t prefix_bytes = min(curr_record_len, static_cast<uint64_t>(4));

        uint8_t* curr_record_chars = col_chars + curr_record_start;
        uint32_t curr_record_prefix = 0; uint32_t curr_shift = 3 * BITS_IN_BYTE;

        #pragma unroll
        for(uint64_t i = 0; i < prefix_bytes; i++) { 
            curr_record_prefix |= static_cast<uint32_t>(curr_record_chars[i]) << curr_shift;
            curr_shift -= BITS_IN_BYTE;
        }

        // Save the record values
        records[row_id] = str_top_n_record_type(row_id, curr_record_prefix);
        indexes[row_id] = row_id;
    }
}

__global__ void top_groups_calculator(uint32_t* group_starting_offsets, uint32_t num_results_needed, uint32_t* num_records_to_consider) {
    uint32_t group_id = 0;
    while(group_starting_offsets[group_id] < num_results_needed) { 
        group_id++;
    }

    num_records_to_consider[0] = group_starting_offsets[group_id];
}

__global__ void materialize_determine_lengths(str_top_n_record_type* ordered_records, uint64_t* src_col_offsets, uint64_t* result_lengths, uint64_t num_records) { 
  uint32_t curr_record = threadIdx.x + blockIdx.x * blockDim.x;
  if(curr_record < num_records) { // Determine the record length from the offsets buffer
    uint32_t row_id = ordered_records[curr_record].row_id;
    result_lengths[curr_record] = src_col_offsets[row_id + 1] - src_col_offsets[row_id];
  } else if(curr_record == num_records) { 
    // Set the value of the last string to zero so therefore it will populate the last offset properly
    result_lengths[curr_record] = 0;
  }
}

__global__ void materialize_copy_string(str_top_n_record_type* ordered_records, uint8_t* src_chars, uint64_t* src_offsets, uint8_t* dst_chars, uint64_t* dst_offsets, uint64_t num_records) { 
    uint32_t curr_record = threadIdx.x + blockIdx.x * blockDim.x;
    if(curr_record < num_records) {
        // Determine the src and dst pointers
        uint32_t row_id = ordered_records[curr_record].row_id;
        uint64_t src_start_offset = src_offsets[row_id];
        const uint64_t record_length = src_offsets[row_id + 1] - src_start_offset;
        uint8_t* read_ptr = src_chars + src_start_offset;
        uint8_t* write_ptr = dst_chars + dst_offsets[curr_record];

        // Now use to copy over the bytes
        #pragma unroll
        for(uint64_t i = 0; i < record_length; i++) { 
            write_ptr[i] = read_ptr[i];
        }
    }
}

void CustomStringTopN(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& projection, uint64_t num_keys, uint64_t num_projections, OrderByType* order_by_type, idx_t num_results) {
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    SETUP_TIMING();
    START_TIMER();

    // First create the records from the input data
    shared_ptr<GPUColumn> src_col = keys[0];
    DataWrapper col_data = src_col->data_wrapper;
    uint8_t* col_chars = col_data.data; uint64_t* col_offsets = col_data.offset;
    const uint32_t num_records = keys[0]->column_length;

    uint32_t num_create_workers = (num_records + BLOCK_THREADS - 1)/BLOCK_THREADS;
    uint32_t* d_indexes = gpuBufferManager->customCudaMalloc<uint32_t>(num_records, 0, 0);
    str_top_n_record_type* d_records = reinterpret_cast<str_top_n_record_type*>(gpuBufferManager->customCudaMalloc<string_top_n_record_type>(num_records, 0, 0));
    create_top_n_records<<<num_create_workers, BLOCK_THREADS>>>(d_records, d_indexes, col_chars, col_offsets, num_records);
    RECORD_TIMER("STRING TOP N Key Create Time");

    // Now sort the records on just the prefix
    void* d_prefix_sort_temp_storage = nullptr;
    size_t prefix_sort_temp_storage_bytes = 0;

    START_TIMER();
    if(order_by_type[0] == OrderByType::ASCENDING) { 
        CustomTopNStringPrefixLessThan prefix_less_than_compartor;
        cub::DeviceMergeSort::SortKeys(d_prefix_sort_temp_storage, prefix_sort_temp_storage_bytes, d_records, num_records, prefix_less_than_compartor);
        d_prefix_sort_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(prefix_sort_temp_storage_bytes, 0, 0);
        cub::DeviceMergeSort::SortKeys(d_prefix_sort_temp_storage, prefix_sort_temp_storage_bytes, d_records, num_records, prefix_less_than_compartor);
    } else { 
        CustomTopNStringPrefixLessThan prefix_greater_than_compartor;
        cub::DeviceMergeSort::SortKeys(d_prefix_sort_temp_storage, prefix_sort_temp_storage_bytes, d_records, num_records, prefix_greater_than_compartor);
        d_prefix_sort_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(prefix_sort_temp_storage_bytes, 0, 0);
        cub::DeviceMergeSort::SortKeys(d_prefix_sort_temp_storage, prefix_sort_temp_storage_bytes, d_records, num_records, prefix_greater_than_compartor);
    }
    RECORD_TIMER("STRING TOP N Key Prefix Sort Time");

    // We only need to sort the groups whose starting row id < N that we need as the resulting records might be in the last row of that group.
    // In order to do that we first need to perform a reduction to get group boundaries and then use that to determine how many of the initial
    // groups do we actually need to sort
    START_TIMER();
    CustomTopNMin reduce_min_operator;
    str_top_n_record_type* d_result_groups = reinterpret_cast<str_top_n_record_type*>(gpuBufferManager->customCudaMalloc<string_top_n_record_type>(num_records, 0, 0));
    uint32_t* d_group_boundaries = gpuBufferManager->customCudaMalloc<uint32_t>(num_records, 0, 0);
    uint32_t* d_num_groups = gpuBufferManager->customCudaMalloc<uint32_t>(1, 0, 0);

    void* d_key_reduction_temp_storage = nullptr;
    size_t key_reduction_temp_storage_bytes = 0;
    cub::DeviceReduce::ReduceByKey(
        d_key_reduction_temp_storage, key_reduction_temp_storage_bytes, d_records, d_result_groups, d_indexes,
        d_group_boundaries, d_num_groups, reduce_min_operator, num_records
    );
    d_key_reduction_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(key_reduction_temp_storage_bytes, 0, 0);
    cub::DeviceReduce::ReduceByKey(
        d_key_reduction_temp_storage, key_reduction_temp_storage_bytes, d_records, d_result_groups, d_indexes,
        d_group_boundaries, d_num_groups, reduce_min_operator, num_records
    );

    top_groups_calculator<<<1, 1>>>(d_group_boundaries, num_results, d_num_groups);
    uint32_t num_complete_sort_records = 0;
    cudaMemcpy(&num_complete_sort_records, d_num_groups, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    RECORD_TIMER("STRING TOP N Key Prefix Num Groups Calculate Time");

    // Now actually perform the complete sort only on the top groups
    void* d_record_sort_temp_storage = nullptr;
    size_t record_sort_temp_storage_bytes = 0;

    START_TIMER();
    if(order_by_type[0] == OrderByType::ASCENDING) { 
        CustomTopNStringLessThan string_less_than_cmp(col_chars, col_offsets);
        cub::DeviceMergeSort::SortKeys(d_record_sort_temp_storage, record_sort_temp_storage_bytes, d_records, num_complete_sort_records, string_less_than_cmp);
        d_record_sort_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(record_sort_temp_storage_bytes, 0, 0);
        cub::DeviceMergeSort::SortKeys(d_record_sort_temp_storage, record_sort_temp_storage_bytes, d_records, num_complete_sort_records, string_less_than_cmp);
    } else { 
        CustomTopNStringGreaterThan string_greater_than_cmp(col_chars, col_offsets);
        cub::DeviceMergeSort::SortKeys(d_record_sort_temp_storage, record_sort_temp_storage_bytes, d_records, num_complete_sort_records, string_greater_than_cmp);
        d_record_sort_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(record_sort_temp_storage_bytes, 0, 0);
        cub::DeviceMergeSort::SortKeys(d_record_sort_temp_storage, record_sort_temp_storage_bytes, d_records, num_complete_sort_records, string_greater_than_cmp);
    }
    RECORD_TIMER("STRING TOP N Record Sort Time");

    // Finally copy over the resulting records to the output columns
    START_TIMER();

    DataWrapper project_col_data = projection[0]->data_wrapper;
    uint8_t* project_col_chars = project_col_data.data; uint64_t* project_col_offsets = project_col_data.offset;

    // First determine the new offsets using a prefix sum
    uint32_t num_materialize_worker = (num_results + BLOCK_THREADS)/BLOCK_THREADS;
    uint64_t* d_new_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(num_results + 1, 0, 0);
    materialize_determine_lengths<<<num_materialize_worker, BLOCK_THREADS>>>(d_records, project_col_offsets, d_new_offsets, num_results);

    void* d_prefix_sum_temp_storage = nullptr;
    size_t prefix_sum_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_prefix_sum_temp_storage, prefix_sum_temp_storage_bytes, d_new_offsets, d_new_offsets, num_results + 1);
    d_prefix_sum_temp_storage = reinterpret_cast<void*>(gpuBufferManager->customCudaMalloc<uint8_t>(prefix_sum_temp_storage_bytes, 0, 0));
    cub::DeviceScan::ExclusiveSum(d_prefix_sum_temp_storage, prefix_sum_temp_storage_bytes, d_new_offsets, d_new_offsets, num_results + 1);

    // Now copy over the actual characters
    uint64_t num_total_bytes;
    cudaMemcpy(&num_total_bytes, d_new_offsets + num_results, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint8_t* d_result_chars = gpuBufferManager->customCudaMalloc<uint8_t>(num_total_bytes, 0, 0);
    materialize_copy_string<<<num_materialize_worker, BLOCK_THREADS>>>(d_records, project_col_chars, project_col_offsets, d_result_chars, d_new_offsets, num_results);
    
    // Set the new column
    GPUColumnType project_col_type =  project_col_data.type;
    bool is_str_col = project_col_data.type.id() == GPUColumnTypeId::VARCHAR;
    //TODO: FOR DEVESH, WE NEED TO MAKE SURE THAT THE VALIDITY MASK IS ADDED HERE, RIGHT NOW WE ASSUME IT'S ALL VALID
    auto validity_mask = createNullMask(num_results, cudf::mask_state::ALL_VALID);
    projection[0] = make_shared_ptr<GPUColumn>(num_results, project_col_type, d_result_chars, d_new_offsets, num_total_bytes, is_str_col, validity_mask);

    RECORD_TIMER("STRING TOP N Result Write Time");
}

void cudf_orderby(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& projection, uint64_t num_keys, uint64_t num_projections, OrderByType* order_by_type, idx_t num_results) 
{
    // See if we can use the custom kernel to perform the top N by performing the following checks:
    // - num_results > 0
    // - We have a singular varchar column as the key and projection column
    if (Config::USE_CUSTOM_TOP_N) { 
        bool use_customed_implementation = num_results > 0 && num_keys == num_projections && num_keys == 1;
        for(size_t col = 0; col < num_keys; col++) { 
            use_customed_implementation = use_customed_implementation && keys[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR;
        }
        for(size_t col = 0; col < num_projections; col++) { 
            use_customed_implementation = use_customed_implementation && projection[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR;
        }
        SIRIUS_LOG_DEBUG("Cudf order using custom top n of {} has val {}", num_results, use_customed_implementation);
        
        if(use_customed_implementation) {
            CustomStringTopN(keys, projection, num_keys, num_projections, order_by_type, num_results);
            return;
        }
    }

    if (keys[0]->column_length == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        for (idx_t col = 0; col < num_projections; col++) {
            bool old_unique = projection[col]->is_unique;
            if (projection[col]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
                projection[col] = make_shared_ptr<GPUColumn>(0, projection[col]->data_wrapper.type, projection[col]->data_wrapper.data, projection[col]->data_wrapper.offset, 0, true, nullptr);
            } else {
                projection[col] = make_shared_ptr<GPUColumn>(0, projection[col]->data_wrapper.type, projection[col]->data_wrapper.data, nullptr);
            }
            projection[col]->is_unique = old_unique;
        }
        return;
    }

    SIRIUS_LOG_DEBUG("CUDF Order By");
    SIRIUS_LOG_DEBUG("Input size: {}", keys[0]->column_length);
    SETUP_TIMING();
    START_TIMER();

    GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
    cudf::set_current_device_resource(gpuBufferManager->mr);


    std::vector<cudf::column_view> columns_cudf;
    for (int key = 0; key < num_keys; key++) {
        auto cudf_column_view = keys[key]->convertToCudfColumn();
        columns_cudf.push_back(cudf_column_view);
    }

    std::vector<cudf::order> orders;
    std::vector<cudf::null_order> null_orders;
    for (int i = 0; i < num_keys; i++) {
        if (order_by_type[i] == OrderByType::ASCENDING) {
            orders.push_back(cudf::order::ASCENDING);
            null_orders.push_back(cudf::null_order::AFTER);
        } else {
            orders.push_back(cudf::order::DESCENDING);
            null_orders.push_back(cudf::null_order::BEFORE);
        }
    }

    auto keys_table = cudf::table_view(columns_cudf);
    auto sorted_order = cudf::sorted_order(keys_table, orders, null_orders);
    auto sorted_order_view = sorted_order->view();

    std::vector<cudf::column_view> projection_cudf;
    for (int col = 0; col < num_projections; col++) {
        auto cudf_column = projection[col]->convertToCudfColumn();
        projection_cudf.push_back(cudf_column);
    }
    auto projection_table = cudf::table_view(projection_cudf);

    auto gathered_table = cudf::gather(projection_table, sorted_order_view);

    for (int col = 0; col < num_projections; col++) {
        auto sorted_column = gathered_table->get_column(col);
        projection[col]->setFromCudfColumn(sorted_column, projection[col]->is_unique, nullptr, 0, gpuBufferManager);
    }

    SIRIUS_LOG_DEBUG("Order by done");

    STOP_TIMER();
}

} //namespace duckdb