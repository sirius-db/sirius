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

#include "cudf/cudf_utils.hpp"
#include "../operator/cuda_helper.cuh"
#include "gpu_physical_grouped_aggregate.hpp"
#include "gpu_buffer_manager.hpp"
#include "log/logging.hpp"

#include <limits>

namespace duckdb {

template<typename T>
void combineColumns(T* a, T* b, T*& c, uint64_t N_a, uint64_t N_b) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Combine Columns Kernel");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    c = gpuBufferManager->customCudaMalloc<T>(N_a + N_b, 0, 0);
    cudaMemcpy(c, a, N_a * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(c + N_a, b, N_b * sizeof(T), cudaMemcpyDeviceToDevice);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(a), 0);
    gpuBufferManager->customCudaFree(reinterpret_cast<uint8_t*>(b), 0);
    CHECK_ERROR();
    cudaDeviceSynchronize();
}

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

struct CustomLess
{
  template <typename T>
  __forceinline__ __device__ bool operator()(const T &lhs, const T &rhs)
  {
    return lhs < rhs;
  }
};

struct CustomSum
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
      return a + b;
    }
};

__global__ void initialize_counts(int32_t* sorted_keys, int32_t* sorted_values, int32_t* record_count, uint64_t num_records) { 
  const uint64_t curr_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(curr_idx > 0 && curr_idx < num_records) { 
    // Consider it unique if: 1) It is the first pair for this key, 2) It has a different value for the same key
    record_count[curr_idx] = (sorted_keys[curr_idx] != sorted_keys[curr_idx - 1]) || (sorted_values[curr_idx] != sorted_values[curr_idx - 1]);
  } else if(curr_idx == 0) { 
    record_count[curr_idx] = 1;
  }
}

struct __align__(16) str_count_distinct_type { 
  string_group_by_metadata_type* group_by_metadata;
	uint32_t row_id;
	uint32_t key_signature;

  __host__ __device__ str_count_distinct_type() : group_by_metadata(nullptr), row_id(0), key_signature(0) {}
  __host__ __device__ str_count_distinct_type(string_group_by_metadata_type* _metadata, uint32_t _row_id, uint32_t _signature) : 
    group_by_metadata(_metadata), row_id(_row_id), key_signature(_signature) {}

  __device__ __forceinline__ bool operator==(const str_count_distinct_type &other) const { 
    // First check if the signatures match
    if(key_signature != other.key_signature) return false;

    // If the signatures are the same then compare the keys (need to do this since we may have hash collisions)
    const uint32_t num_keys = group_by_metadata->num_keys;
    uint32_t left_row_id = row_id; uint32_t right_row_id = other.row_id;
    for(uint32_t col_id = 0; col_id < num_keys; col_id++) { 
      // Determine the details for the left and right row
      uint64_t* col_offsets = group_by_metadata->all_offsets[col_id];
      uint64_t left_start = col_offsets[left_row_id];
      const uint64_t left_length = col_offsets[left_row_id + 1] - left_start;

      uint64_t right_start = col_offsets[right_row_id];
      const uint64_t right_length = col_offsets[right_row_id + 1] - right_start;

      // First ensure that the lengths match
      if(left_length != right_length) return false;

      // If the lengths match then compare the chars
      uint8_t* col_chars = group_by_metadata->all_keys[col_id];
      uint8_t* left_chars = col_chars + left_start; uint8_t* right_chars = col_chars + right_start;

      #pragma unroll
      for(uint32_t i = 0; i < left_length; i++) { 
        if(left_chars[i] != right_chars[i]) return false;
      }
    }

    return true;
  }

  // This is used during the sorting so in that case we also want to compare the aggregates so that not only the records are sorted
  // in key order but values within a key are also sorted in order
  __device__ __forceinline__ bool operator<(const str_count_distinct_type &other) const { 
    // If the signatures are different then compare the keys
    if(key_signature != other.key_signature) return key_signature < other.key_signature;

    // If the signatures are the same then compare the keys (need to do this since we may have hash collisions)
    const uint32_t num_keys = group_by_metadata->num_keys;
    uint32_t left_row_id = row_id; uint32_t right_row_id = other.row_id;
    for(uint32_t col_id = 0; col_id < num_keys; col_id++) { 
      // Determine the details for the left and right row
      uint64_t* col_offsets = group_by_metadata->all_offsets[col_id];
      uint64_t left_start = col_offsets[left_row_id];
      const uint64_t left_length = col_offsets[left_row_id + 1] - left_start;

      uint64_t right_start = col_offsets[right_row_id];
      const uint64_t right_length = col_offsets[right_row_id + 1] - right_start;

      // First ensure that the lengths match
      if(left_length != right_length) return left_length < right_length;

      // If the lengths match then compare the chars
      uint8_t* col_chars = group_by_metadata->all_keys[col_id];
      uint8_t* left_chars = col_chars + left_start; uint8_t* right_chars = col_chars + right_start;

      #pragma unroll
      for(uint32_t i = 0; i < left_length; i++) { 
        if(left_chars[i] != right_chars[i]) return left_chars[i] < right_chars[i];
      }
    }

    // If the keys match finally compare the aggregates
    uint64_t* agg_values = reinterpret_cast<uint64_t*>(group_by_metadata->all_keys[num_keys]);
    return agg_values[left_row_id] < agg_values[right_row_id];
  }
};

struct CustomGroupByLess { 
  __forceinline__ __device__ bool operator()(const str_count_distinct_type &lhs, const str_count_distinct_type &rhs) { 
    return lhs < rhs;
  }
};

struct CustomGroupBySum
{
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
      return a + b;
    }
};

__global__ void create_metadata_record(string_group_by_metadata_type* group_by_metadata, uint8_t** keys, uint64_t** offsets, uint32_t num_keys) { 
  group_by_metadata->all_keys = keys;
  group_by_metadata->all_offsets = offsets;
  group_by_metadata->num_keys = num_keys;
}

__global__ void create_col_offsets(uint64_t* offsets_buffer, uint64_t num_records, uint64_t col_size) { 
  uint64_t worker_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(worker_idx <= num_records) { 
    offsets_buffer[worker_idx] = worker_idx * col_size;
  }
}

__global__ void create_group_by_records(str_count_distinct_type* records, string_group_by_metadata_type* group_by_metadata, uint32_t num_records) { 
  uint32_t row_id = threadIdx.x + blockIdx.x * blockDim.x;
  if(row_id < num_records) { 
    // Calculate the hash for this row by iterating through its group by keys
    uint32_t signature = 0;
    uint32_t curr_power = 1;
    const uint64_t num_keys = group_by_metadata->num_keys;

    for (uint64_t col_id = 0; col_id < num_keys; col_id++) { 
      uint64_t* curr_col_offsets = group_by_metadata->all_offsets[col_id];
      uint64_t curr_row_start = curr_col_offsets[row_id];
      const uint64_t curr_row_length = curr_col_offsets[row_id + 1] - curr_row_start;
      uint8_t* curr_record_chars = group_by_metadata->all_keys[col_id] + curr_row_start;

      // Update the signature based on the record value. Note that we don't need to worry about
      // overflow as we are using unsigned data types which is well defined in the C++ standard:
      // https://stackoverflow.com/questions/18195715/why-is-unsigned-integer-overflow-defined-behavior-but-signed-integer-overflow-is
      for(uint32_t i = 0; i < curr_row_length; i++) { 
        signature = signature + static_cast<uint32_t>(curr_record_chars[i]) * curr_power;
        curr_power = curr_power * STRING_HASH_POWER;
      }
    }

    // Use this to create the record
    records[row_id] = str_count_distinct_type(group_by_metadata, row_id, signature);
  }
}

__global__ void determine_is_unique(string_group_by_metadata_type* group_by_metadata, str_count_distinct_type* records, uint64_t* is_unique, uint32_t num_records) { 
  uint32_t curr_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(curr_idx > 0 && curr_idx < num_records) { 
    const bool is_key_different = !(records[curr_idx] == records[curr_idx - 1]); // First check if the records are equal
    
    // Also check if the aggregates are
    uint32_t left_row_id = records[curr_idx].row_id; uint32_t right_row_id = records[curr_idx - 1].row_id;
    uint64_t* agg_values = reinterpret_cast<uint64_t*>(group_by_metadata->all_keys[group_by_metadata->num_keys]);
    bool is_record_different = is_key_different || (agg_values[left_row_id] != agg_values[right_row_id]); 
    is_unique[curr_idx] = static_cast<uint64_t>(is_record_different);
  } else if(curr_idx == 0) { 
    is_unique[curr_idx] = 1;
  }
}

__global__ void materialize_determine_lengths(str_count_distinct_type* result_groups, uint64_t* col_lengths, uint64_t col_id, uint64_t num_groups) { 
  uint32_t curr_group = threadIdx.x + blockIdx.x * blockDim.x;
  if(curr_group < num_groups) { 
    // Get the details for this (group, col)
    uint32_t row_id = result_groups[curr_group].row_id;
    uint64_t* col_offsets = result_groups[curr_group].group_by_metadata->all_offsets[col_id];
    col_lengths[curr_group] = col_offsets[row_id + 1] - col_offsets[row_id];
  } else if(curr_group == num_groups) { 
    // Set the value of the last string to zero so therefore it will populate the last offset properly
    col_lengths[curr_group] = 0;
  }
}

__global__ void materialize_copy_string(str_count_distinct_type* result_groups, uint8_t* result_chars, uint64_t* result_offsets, uint64_t col_id, uint64_t num_groups) { 
  uint32_t curr_group = threadIdx.x + blockIdx.x * blockDim.x;
  if(curr_group < num_groups) { 
    string_group_by_metadata_type* group_by_metadata = result_groups[curr_group].group_by_metadata;
    uint32_t group_row_id = result_groups[curr_group].row_id;

    uint64_t* col_offsets = group_by_metadata->all_offsets[col_id];
    uint64_t start_offset = col_offsets[group_row_id];
    const uint64_t record_length = col_offsets[group_row_id + 1] - start_offset;

    uint8_t* read_ptr = group_by_metadata->all_keys[col_id] + start_offset;
    uint8_t* write_ptr = result_chars + result_offsets[curr_group];

    #pragma unroll
    for(uint64_t i = 0; i < record_length; i++) { 
      write_ptr[i] = read_ptr[i];
    }
  }
}

void ValSortCountDistinct(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& aggregate_keys, uint64_t num_keys, uint64_t num_aggregates, AggregationType* agg_mode, uint32_t num_records) { 
  SETUP_TIMING();

  uint32_t num_offsets_worker = (num_records + BLOCK_THREADS - 1)/BLOCK_THREADS;
  GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());

  // First convert any non string keys to string type
  for(uint64_t i = 0; i < num_keys; i++) { 
    DataWrapper& col_data = keys[i]->data_wrapper;
    if(col_data.type.id() != GPUColumnTypeId::VARCHAR) { 
      // Create the offsets buffer based on the col size
      uint64_t col_size = static_cast<uint64_t>(col_data.getColumnTypeSize());
      col_data.offset = gpuBufferManager->customCudaMalloc<uint64_t>(num_records + 1, 0, 0);
      create_col_offsets<<<num_offsets_worker, BLOCK_THREADS>>>(col_data.offset, num_records, col_size);
    }
  }

  // Also create the metadata
  uint32_t num_hash_cols = static_cast<uint32_t>(num_keys) + 1;
  uint8_t** d_keys = reinterpret_cast<uint8_t**>(gpuBufferManager->customCudaMalloc<void*>(num_hash_cols, 0, 0));
  uint64_t** d_offsets = reinterpret_cast<uint64_t**>(gpuBufferManager->customCudaMalloc<void*>(num_hash_cols, 0, 0));
  uint8_t* h_keys[num_hash_cols]; uint64_t* h_offsets[num_hash_cols];
  for(uint64_t i = 0; i < num_keys; i++) { 
    h_keys[i] = keys[i]->data_wrapper.data; h_offsets[i] = keys[i]->data_wrapper.offset;
  }

  // We also store the aggregate column (just data) as we use it for the sort
  h_keys[num_keys] = aggregate_keys[0]->data_wrapper.data; 
  h_offsets[num_keys] = nullptr;

  cudaMemcpy(d_keys, h_keys, num_hash_cols * sizeof(uint8_t*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_offsets, h_offsets, num_hash_cols * sizeof(uint64_t*), cudaMemcpyHostToDevice);

  string_group_by_metadata_type* d_group_by_metadata = gpuBufferManager->customCudaMalloc<string_group_by_metadata_type>(1, 0, 0);
  create_metadata_record<<<1, 1>>>(d_group_by_metadata, d_keys, d_offsets, num_keys);

  // Determine the record for each row by hashing it
  str_count_distinct_type* d_records = reinterpret_cast<str_count_distinct_type*>(gpuBufferManager->customCudaMalloc<string_group_by_record_type>(num_records, 0, 0));
  uint32_t num_hash_workers = num_offsets_worker;
  create_group_by_records<<<num_hash_workers, BLOCK_THREADS>>>(d_records, d_group_by_metadata, num_records);

  // Now sort the records considering both the key and the aggregate (check the str_count_distinct_type < operator for details)
  CustomGroupByLess record_compare_operator;
  void* d_sort_temp_storage = nullptr;
  size_t sort_temp_storage_bytes = 0;

  cub::DeviceMergeSort::SortKeys(d_sort_temp_storage, sort_temp_storage_bytes, d_records, num_records, record_compare_operator);
  d_sort_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(sort_temp_storage_bytes, 0, 0);
  cub::DeviceMergeSort::SortKeys(d_sort_temp_storage, sort_temp_storage_bytes, d_records, num_records, record_compare_operator);

  // Now determine if this is a unique record or not by comparing the previous record
  uint64_t* d_is_unique_record = gpuBufferManager->customCudaMalloc<uint64_t>(num_records, 0, 0);
  uint32_t num_unique_workers = num_offsets_worker;
  determine_is_unique<<<num_unique_workers, BLOCK_THREADS>>>(d_group_by_metadata, d_records, d_is_unique_record, num_records);

  // Now perform the key reduction
  CustomGroupBySum reduce_sum_operator;
  str_count_distinct_type* d_result_groups = reinterpret_cast<str_count_distinct_type*>(gpuBufferManager->customCudaMalloc<string_group_by_record_type>(num_records, 0, 0));
  uint64_t* d_result_aggs = gpuBufferManager->customCudaMalloc<uint64_t>(num_records, 0, 0);
  uint64_t* d_num_groups = gpuBufferManager->customCudaMalloc<uint64_t>(1, 0, 0);

  void* d_key_reduction_temp_storage = nullptr;
  size_t key_reduction_temp_storage_bytes = 0;
  cub::DeviceReduce::ReduceByKey(
    d_key_reduction_temp_storage, key_reduction_temp_storage_bytes, d_records, d_result_groups, d_is_unique_record,
    d_result_aggs, d_num_groups, reduce_sum_operator, num_records
  );
  d_key_reduction_temp_storage = gpuBufferManager->customCudaMalloc<uint8_t>(key_reduction_temp_storage_bytes, 0, 0);
  cub::DeviceReduce::ReduceByKey(
    d_key_reduction_temp_storage, key_reduction_temp_storage_bytes, d_records, d_result_groups, d_is_unique_record,
    d_result_aggs, d_num_groups, reduce_sum_operator, num_records
  );

  // Finally materialize the results
  uint64_t num_groups = 0;
  cudaMemcpy(&num_groups, d_num_groups, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  // First copy over each of the group by columns
  uint32_t num_materialize_worker = (num_groups + BLOCK_THREADS)/BLOCK_THREADS;
  for(uint32_t col_id = 0; col_id < num_keys; col_id++) {
    shared_ptr<GPUColumn> src_col = keys[col_id];

    // First determine the new offsets using a prefix sum
    uint64_t* d_new_offsets = gpuBufferManager->customCudaMalloc<uint64_t>(num_groups + 1, 0, 0);
    materialize_determine_lengths<<<num_materialize_worker, BLOCK_THREADS>>>(d_result_groups, d_new_offsets, col_id, num_groups);

    void* d_prefix_sum_temp_storage = nullptr;
    size_t prefix_sum_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_prefix_sum_temp_storage, prefix_sum_temp_storage_bytes, d_new_offsets, d_new_offsets, num_groups + 1);
    d_prefix_sum_temp_storage = reinterpret_cast<void*>(gpuBufferManager->customCudaMalloc<uint8_t>(prefix_sum_temp_storage_bytes, 0, 0));
    cub::DeviceScan::ExclusiveSum(d_prefix_sum_temp_storage, prefix_sum_temp_storage_bytes, d_new_offsets, d_new_offsets, num_groups + 1);

    // Now copy over the actual characters
    uint64_t num_total_bytes;
    cudaMemcpy(&num_total_bytes, d_new_offsets + num_groups, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    uint8_t* d_result_chars = gpuBufferManager->customCudaMalloc<uint8_t>(num_total_bytes, 0, 0);
    materialize_copy_string<<<num_materialize_worker, BLOCK_THREADS>>>(d_result_groups, d_result_chars, d_new_offsets, col_id, num_groups);

    // Set the new column
    GPUColumnType src_col_type = src_col->data_wrapper.type;
    bool is_str_col = src_col_type.id() == GPUColumnTypeId::VARCHAR;
    keys[col_id] = make_shared_ptr<GPUColumn>(num_groups, src_col_type, d_result_chars, d_new_offsets, num_total_bytes, is_str_col);
  }

  // Also set the aggregate column
  aggregate_keys[0] = make_shared_ptr<GPUColumn>(num_groups, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(d_result_aggs));
}

constexpr bool USE_CUSTOM_COUNT_DISTINCT = false; // Set this to false if you want to use the default CUDF implementation

void cudf_groupby(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& aggregate_keys, uint64_t num_keys, uint64_t num_aggregates, AggregationType* agg_mode) 
{

  if constexpr(USE_CUSTOM_COUNT_DISTINCT) { 
    // See if we can use the custom count distinct implementation
    uint32_t num_records = static_cast<uint32_t>(keys[0]->column_length);
    bool valid_agg_type = (num_aggregates == 1) && agg_mode[0] == AggregationType::COUNT_DISTINCT && aggregate_keys[0]->data_wrapper.type.id() == GPUColumnTypeId::INT64;
    bool has_string_col = false;

    for(uint64_t i = 0; i < num_keys; i++) {
      if (keys[i]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
        has_string_col = true;
        break;
      }
    }

    if (valid_agg_type && has_string_col) {
      ValSortCountDistinct(keys, aggregate_keys, num_keys, num_aggregates, agg_mode, num_records);
      return;
    }
  }

  if (keys[0]->column_length == 0) {
    SIRIUS_LOG_DEBUG("Input size is 0");
    for (idx_t group = 0; group < num_keys; group++) {
      bool old_unique = keys[group]->is_unique;
      if (keys[group]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
        keys[group] = make_shared_ptr<GPUColumn>(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data, keys[group]->data_wrapper.offset, 0, true);
      } else {
        keys[group] = make_shared_ptr<GPUColumn>(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data);
      }
      keys[group]->is_unique = old_unique;
    }

    for (int agg_idx = 0; agg_idx < num_aggregates; agg_idx++) {
      if (agg_mode[agg_idx] == AggregationType::COUNT_STAR || agg_mode[agg_idx] == AggregationType::COUNT) {
        aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), aggregate_keys[agg_idx]->data_wrapper.data);
      } else {
        aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(0, aggregate_keys[agg_idx]->data_wrapper.type, aggregate_keys[agg_idx]->data_wrapper.data);
      }
    }
    return;
  }

  SIRIUS_LOG_DEBUG("CUDF Group By");
  SIRIUS_LOG_DEBUG("Input size: {}", keys[0]->column_length);

  SETUP_TIMING();

  GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
  cudf::set_current_device_resource(gpuBufferManager->mr);

  std::vector<cudf::column_view> keys_cudf;

  //TODO: This is a hack to get the size of the keys
  size_t size = 0;

  for (int key = 0; key < num_keys; key++) {
    if (keys[key]->data_wrapper.data != nullptr) {
      auto cudf_column = keys[key]->convertToCudfColumn();
      keys_cudf.push_back(cudf_column);
      size = keys[key]->column_length;
    } else {
      throw NotImplementedException("Group by on non-nullable column not supported");
    }
  }

  auto keys_table = cudf::table_view(keys_cudf);
  cudf::groupby::groupby grpby_obj(keys_table);
  std::vector<cudf::groupby::aggregation_request> requests;
  for (int agg = 0; agg < num_aggregates; agg++) {
    requests.emplace_back(cudf::groupby::aggregation_request());
    if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT && aggregate_keys[agg]->column_length == 0) {
      auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
      cudaMemset(temp, 0, size * sizeof(uint64_t));
      shared_ptr<GPUColumn> temp_column = make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp));
      requests[agg].values = temp_column->convertToCudfColumn();
    } else if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::SUM && aggregate_keys[agg]->column_length == 0) {
      auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
      cudaMemset(temp, 0, size * sizeof(uint64_t));
      shared_ptr<GPUColumn> temp_column = make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp));
      requests[agg].values = temp_column->convertToCudfColumn();
    } else if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT_STAR && aggregate_keys[agg]->column_length != 0) {
      auto aggregate = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
      requests[agg].aggregations.push_back(std::move(aggregate));
      uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
      cudaMemset(temp, 0, size * sizeof(uint64_t));
      shared_ptr<GPUColumn> temp_column = make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp));
      requests[agg].values = temp_column->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::SUM) {
      auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::AVERAGE) {
      auto aggregate = cudf::make_mean_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      // If aggregate input column is decimal, need to convert to double following duckdb
      if (aggregate_keys[agg]->data_wrapper.type.id() == GPUColumnTypeId::DECIMAL) {
        if (aggregate_keys[agg]->data_wrapper.getColumnTypeSize() != sizeof(int64_t)) {
          throw NotImplementedException("Only support decimal64 for decimal AVG group-by");
        }
        auto from_cudf_column_view = aggregate_keys[agg]->convertToCudfColumn();
        auto to_cudf_type = cudf::data_type(cudf::type_id::FLOAT64);
        auto to_cudf_column = cudf::cast(
          from_cudf_column_view, to_cudf_type, rmm::cuda_stream_default, GPUBufferManager::GetInstance().mr);
        aggregate_keys[agg]->setFromCudfColumn(*to_cudf_column, false, nullptr, 0, gpuBufferManager);
      }
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::MIN) {
      auto aggregate = cudf::make_min_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::MAX) {
      auto aggregate = cudf::make_max_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::COUNT) {
      auto aggregate = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else if (agg_mode[agg] == AggregationType::COUNT_DISTINCT) {
      auto aggregate = cudf::make_nunique_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
      requests[agg].aggregations.push_back(std::move(aggregate));
      requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
    } else {
      throw NotImplementedException("Aggregate function not supported in `cudf_groupby`: %d",
                                    static_cast<int>(agg_mode[agg]));
    }
  }
  auto result = grpby_obj.aggregate(requests);

  auto result_key = std::move(result.first);
  for (int key = 0; key < num_keys; key++) {
      cudf::column group_key = result_key->get_column(key);
      keys[key]->setFromCudfColumn(group_key, keys[key]->is_unique, nullptr, 0, gpuBufferManager);
  }

  for (int agg = 0; agg < num_aggregates; agg++) {
      auto agg_val = std::move(result.second[agg].results[0]);
      if (agg_mode[agg] == AggregationType::COUNT || agg_mode[agg] == AggregationType::COUNT_STAR || agg_mode[agg] == AggregationType::COUNT_DISTINCT) {
        auto agg_val_view = agg_val->view();
        auto temp_data = convertInt32ToUInt64(const_cast<int32_t*>(agg_val_view.data<int32_t>()), agg_val_view.size());
        aggregate_keys[agg] = make_shared_ptr<GPUColumn>(agg_val_view.size(), GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp_data));
      } else {
        aggregate_keys[agg]->setFromCudfColumn(*agg_val, false, nullptr, 0, gpuBufferManager);
      }
  }

  SIRIUS_LOG_DEBUG("CUDF Groupby result count: {}", keys[0]->column_length);
}

template
void combineColumns<int32_t>(int32_t* a, int32_t* b, int32_t*& c, uint64_t N_a, uint64_t N_b);

template
void combineColumns<uint64_t>(uint64_t* a, uint64_t* b, uint64_t*& c, uint64_t N_a, uint64_t N_b);

template
void combineColumns<double>(double* a, double* b, double*& c, uint64_t N_a, uint64_t N_b);

} //namespace duckdb