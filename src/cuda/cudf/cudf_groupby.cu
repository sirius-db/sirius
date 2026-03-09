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
#include <cudf/stream_compaction.hpp>
#include <cstdlib>

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


__global__ void copy_mask(uint32_t* b, uint32_t* c, uint64_t offset, uint64_t N) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t int_idx = (idx + offset) / 32;
    uint64_t bit_idx = (idx + offset) % 32;
    uint32_t mask = 0;
    if (idx < N) {
        mask = (b[int_idx] >> bit_idx) & 1;
    }

    uint32_t lane_id = threadIdx.x % 32;
    uint32_t set = (mask << lane_id);
    for (int lane = 16; lane >= 1; lane /= 2) {
        set |= __shfl_down_sync(0xFFFFFFFF, set, lane);
    }
    __syncwarp();
    if (lane_id == 0) {
        c[idx / 32] = set;
    }
}

void combineMasks(cudf::bitmask_type* a, cudf::bitmask_type* b, cudf::bitmask_type*& c, uint64_t N_a, uint64_t N_b) {
    CHECK_ERROR();
    if (N_a == 0 || N_b == 0) {
        SIRIUS_LOG_DEBUG("Input size is 0");
        return;
    }
    SIRIUS_LOG_DEBUG("Launching Combine Columns Kernel");
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    auto size_a = getMaskBytesSize(N_a) / sizeof(cudf::bitmask_type);
    auto size_c = getMaskBytesSize(N_a + N_b) / sizeof(cudf::bitmask_type);
    c = gpuBufferManager->customCudaMalloc<uint32_t>(size_c, 0, 0);
    cudaMemcpy(c, a, size_a * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    if (N_a % 32 == 0) {
      auto offset_after_a = (N_a + 31) / 32;
      auto offset_remain_after_a = 0;
      auto N = N_b - offset_remain_after_a;
      copy_mask<<<(N + BLOCK_THREADS - 1) / BLOCK_THREADS, BLOCK_THREADS>>>(b, c + offset_after_a, offset_remain_after_a, N);
      CHECK_ERROR();
    } else {
      auto offset_after_a = (N_a + 31) / 32;
      auto offset_remain_after_a = 32 - (N_a % 32);
      auto N = N_b - offset_remain_after_a;
      copy_mask<<<(N + BLOCK_THREADS - 1) / BLOCK_THREADS, BLOCK_THREADS>>>(b, c + offset_after_a, offset_remain_after_a, N);
      CHECK_ERROR();

      uint32_t temp = 0;
      uint32_t temp2 = 0;
      cudaMemcpy(&temp, c + offset_after_a - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(&temp2, b, sizeof(uint32_t), cudaMemcpyDeviceToHost);
      temp = (temp << offset_remain_after_a) >> offset_remain_after_a;
      temp2 = temp2 << (N_a % 32);
      temp = temp | temp2;
      cudaMemcpy(c + offset_after_a - 1, &temp, sizeof(uint32_t), cudaMemcpyHostToDevice);
      CHECK_ERROR();
    }

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

void cudf_groupby(vector<shared_ptr<GPUColumn>>& keys, vector<shared_ptr<GPUColumn>>& aggregate_keys, uint64_t num_keys, uint64_t num_aggregates, AggregationType* agg_mode, idx_t estimated_output_groups)
{
  if (keys[0]->column_length == 0) {
    SIRIUS_LOG_DEBUG("Input size is 0");
    for (idx_t group = 0; group < num_keys; group++) {
      bool old_unique = keys[group]->is_unique;
      if (keys[group]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
        keys[group] = make_shared_ptr<GPUColumn>(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data, keys[group]->data_wrapper.offset, 0, true, nullptr);
      } else {
        keys[group] = make_shared_ptr<GPUColumn>(0, keys[group]->data_wrapper.type, keys[group]->data_wrapper.data, nullptr);
      }
      keys[group]->is_unique = old_unique;
    }

    for (int agg_idx = 0; agg_idx < num_aggregates; agg_idx++) {
      if (agg_mode[agg_idx] == AggregationType::COUNT_STAR || agg_mode[agg_idx] == AggregationType::COUNT) {
        aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(0, GPUColumnType(GPUColumnTypeId::INT64), aggregate_keys[agg_idx]->data_wrapper.data, nullptr);
      } else {
        aggregate_keys[agg_idx] = make_shared_ptr<GPUColumn>(0, aggregate_keys[agg_idx]->data_wrapper.type, aggregate_keys[agg_idx]->data_wrapper.data, nullptr);
      }
    }
    return;
  }

  SIRIUS_LOG_DEBUG("CUDF Group By");
  SIRIUS_LOG_DEBUG("Input size: {}", keys[0]->column_length);
  SETUP_TIMING();
  START_TIMER();

  GPUBufferManager *gpuBufferManager = &(GPUBufferManager::GetInstance());
  cudf::set_current_device_resource(gpuBufferManager->mr);

  std::vector<cudf::column_view> keys_cudf;
  bool has_nullable_key = false;
  for (int key = 0; key < num_keys; key++) {
    if (keys[key]->data_wrapper.validity_mask != nullptr) {
      has_nullable_key = true;
      break;
    }
  }

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

  // --- Two-phase COUNT DISTINCT optimization ---
  // When all aggregates are COUNT_DISTINCT, use: distinct(group_keys + value) → groupby COUNT_STAR
  // This is faster than cudf's internal nunique which does sort-per-group + unique.
  {
    bool all_count_distinct = (num_aggregates > 0);
    for (int agg = 0; agg < num_aggregates; agg++) {
      if (agg_mode[agg] != AggregationType::COUNT_DISTINCT) {
        all_count_distinct = false;
        break;
      }
    }

    if (all_count_distinct) {
      SIRIUS_LOG_DEBUG("Two-phase COUNT DISTINCT: {} aggregates", num_aggregates);

      for (int agg = 0; agg < num_aggregates; agg++) {
        // Phase 1: Build table (group_keys..., value) and deduplicate
        std::vector<cudf::column_view> dedup_columns;
        for (int key = 0; key < num_keys; key++) {
          dedup_columns.push_back(keys_cudf[key]);
        }
        auto value_view = aggregate_keys[agg]->convertToCudfColumn();
        dedup_columns.push_back(value_view);

        auto dedup_table = cudf::table_view(dedup_columns);

        // COUNT DISTINCT must not count NULLs — drop rows where value is NULL
        std::unique_ptr<cudf::table> null_filtered_owner;
        cudf::table_view effective_dedup_table = dedup_table;
        if (value_view.nullable() && value_view.null_count() > 0) {
          null_filtered_owner = cudf::drop_nulls(dedup_table, {static_cast<cudf::size_type>(num_keys)},
                                                 rmm::cuda_stream_default, gpuBufferManager->mr);
          effective_dedup_table = null_filtered_owner->view();
        }

        // All columns are keys for deduplication
        std::vector<cudf::size_type> all_key_indices;
        for (int k = 0; k < static_cast<int>(effective_dedup_table.num_columns()); k++) {
          all_key_indices.push_back(k);
        }

        auto distinct_result = cudf::distinct(
          effective_dedup_table, all_key_indices,
          cudf::duplicate_keep_option::KEEP_ANY,
          cudf::null_equality::EQUAL,
          cudf::nan_equality::ALL_EQUAL,
          rmm::cuda_stream_default,
          gpuBufferManager->mr);

        SIRIUS_LOG_DEBUG("Two-phase COUNT DISTINCT: {} -> {} rows after distinct",
                         size, distinct_result->num_rows());

        // Phase 2: GroupBy COUNT_STAR on the deduplicated table
        std::vector<cudf::column_view> dedup_keys_views;
        for (int key = 0; key < num_keys; key++) {
          dedup_keys_views.push_back(distinct_result->get_column(key));
        }
        auto dedup_keys_table = cudf::table_view(dedup_keys_views);

        cudf::groupby::groupby grpby_phase2(
          dedup_keys_table, has_nullable_key ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);

        std::vector<cudf::groupby::aggregation_request> phase2_requests;
        phase2_requests.emplace_back(cudf::groupby::aggregation_request());
        auto count_agg = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
        phase2_requests[0].aggregations.push_back(std::move(count_agg));
        // Use the value column from distinct result as the values input (just need something to count)
        phase2_requests[0].values = distinct_result->get_column(num_keys);

        auto phase2_result = grpby_phase2.aggregate(phase2_requests);

        // Extract results: group keys from first aggregate, count from each
        if (agg == 0) {
          auto result_key = std::move(phase2_result.first);
          for (int key = 0; key < num_keys; key++) {
            cudf::column group_key = result_key->get_column(key);
            keys[key]->setFromCudfColumn(group_key, keys[key]->is_unique, nullptr, 0, gpuBufferManager);
          }
        }

        auto agg_val      = std::move(phase2_result.second[0].results[0]);
        auto agg_val_view = agg_val->view();
        auto temp_data    = convertInt32ToUInt64(const_cast<int32_t*>(agg_val_view.data<int32_t>()), agg_val_view.size());
        auto validity_mask = createNullMask(agg_val_view.size());
        aggregate_keys[agg] = make_shared_ptr<GPUColumn>(agg_val_view.size(),
                                                         GPUColumnType(GPUColumnTypeId::INT64),
                                                         reinterpret_cast<uint8_t*>(temp_data),
                                                         validity_mask);
      }

      STOP_TIMER();
      SIRIUS_LOG_DEBUG("CUDF Groupby (two-phase COUNT DISTINCT) result count: {}", keys[0]->column_length);
      return;
    }
  }

  // --- Mixed aggregate: COUNT_DISTINCT + other aggregates ---
  // When some (but not all) aggregates are COUNT_DISTINCT, split into:
  //   1. Normal groupby for non-CD aggregates (with cheap placeholder for CD slots)
  //   2. Two-phase COUNT DISTINCT for each CD aggregate
  // Both groupbys sort output by the same keys → results are positionally aligned.
  //
  // Guard: skip P1b when input is large relative to expected output groups.
  // At high rows/group the distinct step doesn't shrink data enough to justify
  // the two-pass overhead (e.g. Q09 regression at 100M rows).
  // Tune via SIRIUS_P1B_K env var (default 1000); disable with SIRIUS_P1B_NO_GUARD=1.
  {
    static idx_t p1b_K = []() -> idx_t {
      const char* env = std::getenv("SIRIUS_P1B_K");
      return env ? std::stoull(env) : 1000ULL;
    }();
    static bool p1b_no_guard = []() -> bool {
      const char* env = std::getenv("SIRIUS_P1B_NO_GUARD");
      return env != nullptr && env[0] == '1';
    }();

    int num_cd = 0;
    for (int agg = 0; agg < num_aggregates; agg++) {
      if (agg_mode[agg] == AggregationType::COUNT_DISTINCT) num_cd++;
    }

    bool skip_p1b = false;
    if (!p1b_no_guard && num_cd > 0 && num_cd < num_aggregates && estimated_output_groups > 0) {
      idx_t size_threshold = p1b_K * estimated_output_groups;
      if (size > size_threshold) {
        skip_p1b = true;
        SIRIUS_LOG_DEBUG("P1b guard: skip (size={} > K={} * groups={} = {})",
                         size, p1b_K, estimated_output_groups, size_threshold);
      }
    }

    if (num_cd > 0 && num_cd < num_aggregates && !skip_p1b) {
      SIRIUS_LOG_DEBUG("Mixed aggregate: {} COUNT_DISTINCT + {} other", num_cd, num_aggregates - num_cd);

      // Step 1: Normal groupby with COUNT placeholder for CD aggregates
      cudf::groupby::groupby grpby_normal(
        keys_table, has_nullable_key ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);

      std::vector<cudf::groupby::aggregation_request> normal_requests;
      for (int agg = 0; agg < num_aggregates; agg++) {
        normal_requests.emplace_back(cudf::groupby::aggregation_request());

        if (agg_mode[agg] == AggregationType::COUNT_DISTINCT) {
          // Cheap placeholder — will be overwritten by two-phase result
          normal_requests[agg].aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
          normal_requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
        } else if (aggregate_keys[agg]->data_wrapper.data == nullptr &&
                   agg_mode[agg] == AggregationType::COUNT &&
                   aggregate_keys[agg]->column_length == 0) {
          auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
          normal_requests[agg].aggregations.push_back(std::move(aggregate));
          uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
          cudaMemset(temp, 0, size * sizeof(uint64_t));
          auto validity_mask = createNullMask(size);
          shared_ptr<GPUColumn> temp_column =
            make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64),
                                       reinterpret_cast<uint8_t*>(temp), validity_mask);
          normal_requests[agg].values = temp_column->convertToCudfColumn();
        } else if (aggregate_keys[agg]->data_wrapper.data == nullptr &&
                   agg_mode[agg] == AggregationType::SUM &&
                   aggregate_keys[agg]->column_length == 0) {
          auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
          normal_requests[agg].aggregations.push_back(std::move(aggregate));
          uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
          cudaMemset(temp, 0, size * sizeof(uint64_t));
          auto validity_mask = createNullMask(size, cudf::mask_state::ALL_NULL);
          shared_ptr<GPUColumn> temp_column =
            make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64),
                                       reinterpret_cast<uint8_t*>(temp), validity_mask);
          normal_requests[agg].values = temp_column->convertToCudfColumn();
        } else if (aggregate_keys[agg]->data_wrapper.data == nullptr &&
                   agg_mode[agg] == AggregationType::COUNT_STAR &&
                   aggregate_keys[agg]->column_length != 0) {
          auto aggregate = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
          normal_requests[agg].aggregations.push_back(std::move(aggregate));
          uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
          cudaMemset(temp, 0, size * sizeof(uint64_t));
          auto validity_mask = createNullMask(size);
          shared_ptr<GPUColumn> temp_column =
            make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64),
                                       reinterpret_cast<uint8_t*>(temp), validity_mask);
          normal_requests[agg].values = temp_column->convertToCudfColumn();
        } else if (agg_mode[agg] == AggregationType::SUM) {
          normal_requests[agg].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
          normal_requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
        } else if (agg_mode[agg] == AggregationType::AVERAGE) {
          normal_requests[agg].aggregations.push_back(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
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
          normal_requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
        } else if (agg_mode[agg] == AggregationType::MIN) {
          normal_requests[agg].aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
          normal_requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
        } else if (agg_mode[agg] == AggregationType::MAX) {
          normal_requests[agg].aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
          normal_requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
        } else if (agg_mode[agg] == AggregationType::COUNT) {
          normal_requests[agg].aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE));
          normal_requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
        } else if (agg_mode[agg] == AggregationType::COUNT_STAR) {
          normal_requests[agg].aggregations.push_back(
            cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
          normal_requests[agg].values = aggregate_keys[agg]->convertToCudfColumn();
        } else {
          throw NotImplementedException("Aggregate function not supported in mixed path: %d",
                                        static_cast<int>(agg_mode[agg]));
        }
      }

      auto normal_result = grpby_normal.aggregate(normal_requests);

      // Extract group keys
      auto result_key = std::move(normal_result.first);
      for (int key = 0; key < num_keys; key++) {
        cudf::column group_key = result_key->get_column(key);
        keys[key]->setFromCudfColumn(group_key, keys[key]->is_unique, nullptr, 0, gpuBufferManager);
      }

      // Extract non-CD aggregate results
      for (int agg = 0; agg < num_aggregates; agg++) {
        if (agg_mode[agg] == AggregationType::COUNT_DISTINCT) continue;
        auto agg_val = std::move(normal_result.second[agg].results[0]);
        if (agg_mode[agg] == AggregationType::COUNT || agg_mode[agg] == AggregationType::COUNT_STAR) {
          auto agg_val_view  = agg_val->view();
          auto temp_data     = convertInt32ToUInt64(const_cast<int32_t*>(agg_val_view.data<int32_t>()), agg_val_view.size());
          auto validity_mask = createNullMask(agg_val_view.size());
          aggregate_keys[agg] = make_shared_ptr<GPUColumn>(agg_val_view.size(),
                                                           GPUColumnType(GPUColumnTypeId::INT64),
                                                           reinterpret_cast<uint8_t*>(temp_data),
                                                           validity_mask);
        } else {
          aggregate_keys[agg]->setFromCudfColumn(*agg_val, false, nullptr, 0, gpuBufferManager);
        }
      }

      // Step 2: Two-phase COUNT DISTINCT for each CD aggregate
      for (int agg = 0; agg < num_aggregates; agg++) {
        if (agg_mode[agg] != AggregationType::COUNT_DISTINCT) continue;

        // Phase 1: distinct(group_keys + value), excluding NULL values
        std::vector<cudf::column_view> dedup_columns;
        for (int key = 0; key < num_keys; key++) {
          dedup_columns.push_back(keys_cudf[key]);
        }
        auto value_view = aggregate_keys[agg]->convertToCudfColumn();
        dedup_columns.push_back(value_view);

        auto dedup_table = cudf::table_view(dedup_columns);

        // COUNT DISTINCT must not count NULLs — drop rows where value is NULL
        std::unique_ptr<cudf::table> null_filtered_owner;
        cudf::table_view effective_dedup_table = dedup_table;
        if (value_view.nullable() && value_view.null_count() > 0) {
          null_filtered_owner = cudf::drop_nulls(dedup_table, {static_cast<cudf::size_type>(num_keys)},
                                                 rmm::cuda_stream_default, gpuBufferManager->mr);
          effective_dedup_table = null_filtered_owner->view();
        }

        std::vector<cudf::size_type> all_key_indices;
        for (int k = 0; k < static_cast<int>(effective_dedup_table.num_columns()); k++) {
          all_key_indices.push_back(k);
        }

        auto distinct_result = cudf::distinct(
          effective_dedup_table, all_key_indices,
          cudf::duplicate_keep_option::KEEP_ANY,
          cudf::null_equality::EQUAL,
          cudf::nan_equality::ALL_EQUAL,
          rmm::cuda_stream_default,
          gpuBufferManager->mr);

        SIRIUS_LOG_DEBUG("Mixed two-phase: {} -> {} after distinct", size, distinct_result->num_rows());

        // Phase 2: groupby COUNT_STAR on deduplicated table
        std::vector<cudf::column_view> dedup_keys_views;
        for (int key = 0; key < num_keys; key++) {
          dedup_keys_views.push_back(distinct_result->get_column(key));
        }
        auto dedup_keys_table = cudf::table_view(dedup_keys_views);

        cudf::groupby::groupby grpby_phase2(
          dedup_keys_table, has_nullable_key ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);

        std::vector<cudf::groupby::aggregation_request> phase2_requests;
        phase2_requests.emplace_back();
        phase2_requests[0].aggregations.push_back(
          cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE));
        phase2_requests[0].values = distinct_result->get_column(num_keys);

        auto phase2_result = grpby_phase2.aggregate(phase2_requests);

        // Both the normal groupby (step 1) and this phase2 groupby sort their output by the
        // same key columns and operate on the same key domain, so results are positionally
        // aligned — no join needed.
        auto cd_count_col  = std::move(phase2_result.second[0].results[0]);
        auto cd_view       = cd_count_col->view();
        auto temp_data     = convertInt32ToUInt64(const_cast<int32_t*>(cd_view.data<int32_t>()), cd_view.size());
        auto validity_mask = createNullMask(cd_view.size());
        aggregate_keys[agg] = make_shared_ptr<GPUColumn>(
          cd_view.size(), GPUColumnType(GPUColumnTypeId::INT64),
          reinterpret_cast<uint8_t*>(temp_data), validity_mask);
      }

      STOP_TIMER();
      SIRIUS_LOG_DEBUG("CUDF Groupby (mixed COUNT DISTINCT) result count: {}", keys[0]->column_length);
      return;
    }
  }

  cudf::groupby::groupby grpby_obj(
    keys_table, has_nullable_key ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE);

  std::vector<cudf::groupby::aggregation_request> requests;
  for (int agg = 0; agg < num_aggregates; agg++) {
    requests.emplace_back(cudf::groupby::aggregation_request());
    if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT && aggregate_keys[agg]->column_length == 0) {
      auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
      cudaMemset(temp, 0, size * sizeof(uint64_t));
      auto validity_mask = createNullMask(size);
      shared_ptr<GPUColumn> temp_column = make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp), validity_mask);
      requests[agg].values = temp_column->convertToCudfColumn();
    } else if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::SUM && aggregate_keys[agg]->column_length == 0) {
      auto aggregate = cudf::make_sum_aggregation<cudf::groupby_aggregation>();
      requests[agg].aggregations.push_back(std::move(aggregate));
      uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
      cudaMemset(temp, 0, size * sizeof(uint64_t));
      auto validity_mask = createNullMask(size, cudf::mask_state::ALL_NULL);
      shared_ptr<GPUColumn> temp_column = make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp), validity_mask);
      requests[agg].values = temp_column->convertToCudfColumn();
    } else if (aggregate_keys[agg]->data_wrapper.data == nullptr && agg_mode[agg] == AggregationType::COUNT_STAR && aggregate_keys[agg]->column_length != 0) {
      auto aggregate = cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
      requests[agg].aggregations.push_back(std::move(aggregate));
      uint64_t* temp = gpuBufferManager->customCudaMalloc<uint64_t>(size, 0, 0);
      cudaMemset(temp, 0, size * sizeof(uint64_t));
      auto validity_mask = createNullMask(size);
      shared_ptr<GPUColumn> temp_column = make_shared_ptr<GPUColumn>(size, GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp), validity_mask);
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

      // cudf::bitmask_type* host_mask = gpuBufferManager->customCudaHostAlloc<cudf::bitmask_type>(1);
      // callCudaMemcpyDeviceToHost<uint8_t>(reinterpret_cast<uint8_t*>(host_mask), 
      //     reinterpret_cast<uint8_t*>(agg_val->view()->), sizeof(cudf::bitmask_type), 0);
      // printf("host_mask: %x\n", host_mask);
      // printf("host_mask: %b\n", host_mask);
      if (agg_mode[agg] == AggregationType::COUNT || agg_mode[agg] == AggregationType::COUNT_STAR || agg_mode[agg] == AggregationType::COUNT_DISTINCT) {
        auto agg_val_view = agg_val->view();
        auto temp_data = convertInt32ToUInt64(const_cast<int32_t*>(agg_val_view.data<int32_t>()), agg_val_view.size());
        auto validity_mask = createNullMask(agg_val_view.size());
        aggregate_keys[agg] = make_shared_ptr<GPUColumn>(agg_val_view.size(), GPUColumnType(GPUColumnTypeId::INT64), reinterpret_cast<uint8_t*>(temp_data), validity_mask);
      } else {
        aggregate_keys[agg]->setFromCudfColumn(*agg_val, false, nullptr, 0, gpuBufferManager);
      }
  }

  STOP_TIMER();
  SIRIUS_LOG_DEBUG("CUDF Groupby result count: {}", keys[0]->column_length);
}

template
void combineColumns<int32_t>(int32_t* a, int32_t* b, int32_t*& c, uint64_t N_a, uint64_t N_b);

template
void combineColumns<uint64_t>(uint64_t* a, uint64_t* b, uint64_t*& c, uint64_t N_a, uint64_t N_b);

template
void combineColumns<double>(double* a, double* b, double*& c, uint64_t N_a, uint64_t N_b);

} //namespace duckdb