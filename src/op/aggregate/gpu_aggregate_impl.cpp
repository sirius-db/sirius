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

#include "op/aggregate/gpu_aggregate_impl.hpp"

#include "data/data_batch_utils.hpp"

#include <cudf/column/column_factories.hpp>

namespace sirius {
namespace op {

template <typename Base = cudf::aggregation>
std::unique_ptr<Base> get_local_aggregation(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::MIN: return cudf::make_min_aggregation<Base>();
    case cudf::aggregation::Kind::MAX: return cudf::make_max_aggregation<Base>();
    case cudf::aggregation::Kind::COUNT_ALL:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::INCLUDE);
    case cudf::aggregation::Kind::COUNT_VALID:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::EXCLUDE);
    case cudf::aggregation::Kind::SUM: return cudf::make_sum_aggregation<Base>();
    default:
      throw std::runtime_error("Unsupported cudf aggregate kind in `get_local_aggregation()`: " +
                               std::to_string(static_cast<int>(kind)));
  }
}

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_ungrouped_aggregate(
  std::shared_ptr<cucascade::data_batch> input,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_ungrouped_aggregate()`");
  }
  std::vector<std::unique_ptr<cudf::column>> output_cols;
  auto input_table = get_cudf_table_view(*input);
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& input_col       = input_table.column(aggregate_idx[i]);
    auto reduce_aggregation     = get_local_aggregation<cudf::reduce_aggregation>(aggregates[i]);
    cudf::data_type output_type = input_col.type();
    switch (aggregates[i]) {
      case cudf::aggregation::Kind::SUM: {
        switch (output_type.id()) {
          case cudf::type_id::INT8:
          case cudf::type_id::INT16:
          case cudf::type_id::INT32: {
            output_type = cudf::data_type(cudf::type_id::INT64);
            break;
          }
          case cudf::type_id::UINT8:
          case cudf::type_id::UINT16:
          case cudf::type_id::UINT32: {
            output_type = cudf::data_type(cudf::type_id::UINT64);
            break;
          }
          case cudf::type_id::DECIMAL64:
            if (input_col.type().id() == cudf::type_id::DECIMAL64) {
              output_type = cudf::data_type(cudf::type_id::DECIMAL128, output_type.scale());
            }
            break;
          case cudf::type_id::DECIMAL32:
            if (input_col.type().id() == cudf::type_id::DECIMAL32) {
              output_type = cudf::data_type(cudf::type_id::DECIMAL64, output_type.scale());
            }
            break;
          default: break;
        }
        break;
      }
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        output_type = cudf::data_type(cudf::type_id::INT64);
        break;
      }
      default: break;
    }
    auto output_scalar = cudf::reduce(
      input_col, *reduce_aggregation, output_type, stream, memory_space.get_default_allocator());
    output_cols.push_back(cudf::make_column_from_scalar(
      *output_scalar, 1, stream, memory_space.get_default_allocator()));
  }
  auto output_table = std::make_unique<cudf::table>(
    std::move(output_cols), stream, memory_space.get_default_allocator());

  return make_data_batch(std::move(output_table), memory_space);
}

std::shared_ptr<cucascade::data_batch> gpu_aggregate_impl::local_grouped_aggregate(
  std::shared_ptr<cucascade::data_batch> input,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  const std::vector<std::vector<int>>& aggregate_struct_col_indices,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_grouped_aggregate()`");
  }

  const bool has_struct_col_indices = !aggregate_struct_col_indices.empty();

  // Create cudf groupby
  auto input_table = get_cudf_table_view(*input);
  std::vector<cudf::column_view> group_cols;
  for (int idx : group_idx) {
    group_cols.push_back(input_table.column(idx));
  }
  cudf::groupby::groupby grpby_obj(cudf::table_view(group_cols), cudf::null_policy::INCLUDE);

  // Make aggregation requests, group aggregations on the same column in the single request.
  // For multi-column COLLECT_SET, a synthetic negative key -(i+1) is used so that each such
  // aggregate gets its own request with a freshly synthesized struct column.
  std::unordered_map<int, std::vector<std::unique_ptr<cudf::groupby_aggregation>>> input_col_to_agg;
  std::unordered_map<int, std::vector<size_t>> input_col_to_output_idx;
  std::vector<int> input_col_order;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    const auto& aggregate_kind = aggregates[i];
    int aggregate_col_id;
    if (has_struct_col_indices && !aggregate_struct_col_indices[i].empty()) {
      // Multi-column COLLECT_SET: use a unique synthetic negative key for this slot.
      aggregate_col_id = -(static_cast<int>(i) + 1);
    } else {
      aggregate_col_id = aggregate_idx[i];
    }
    if (!input_col_to_agg.contains(aggregate_col_id)) {
      input_col_order.push_back(aggregate_col_id);
    }
    std::unique_ptr<cudf::groupby_aggregation> groupby_aggregation;
    if (aggregate_kind == cudf::aggregation::Kind::COLLECT_SET) {
      groupby_aggregation =
        cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(cudf::null_policy::EXCLUDE);
    } else {
      groupby_aggregation = get_local_aggregation<cudf::groupby_aggregation>(aggregate_kind);
    }
    input_col_to_agg[aggregate_col_id].push_back(std::move(groupby_aggregation));
    input_col_to_output_idx[aggregate_col_id].push_back(i);
  }

  // Temp struct columns for multi-col COLLECT_SET; must outlive the groupby call.
  std::vector<std::unique_ptr<cudf::column>> temp_struct_cols;

  std::vector<cudf::groupby::aggregation_request> requests;
  for (int aggregate_col_id : input_col_order) {
    cudf::groupby::aggregation_request request;
    if (aggregate_col_id < 0) {
      // Multi-col COLLECT_SET: synthesize a struct column from the component columns.
      // The synthetic key is -(slot_index + 1), so slot_index = -aggregate_col_id - 1.
      size_t slot_idx            = static_cast<size_t>(-aggregate_col_id - 1);
      const auto& struct_indices = aggregate_struct_col_indices[slot_idx];
      std::vector<std::unique_ptr<cudf::column>> struct_children;
      for (int col_idx : struct_indices) {
        struct_children.push_back(std::make_unique<cudf::column>(
          input_table.column(col_idx), stream, memory_space.get_default_allocator()));
      }
      auto struct_col = cudf::make_structs_column(input_table.num_rows(),
                                                  std::move(struct_children),
                                                  0,
                                                  rmm::device_buffer{},
                                                  stream,
                                                  memory_space.get_default_allocator());
      request.values  = struct_col->view();
      temp_struct_cols.push_back(std::move(struct_col));
    } else {
      request.values = input_table.column(aggregate_col_id);
    }
    request.aggregations = std::move(input_col_to_agg[aggregate_col_id]);
    requests.push_back(std::move(request));
  }

  // Call cudf groupby and populate output columns
  auto groupby_result = grpby_obj.aggregate(requests, stream, memory_space.get_default_allocator());
  auto output_cols    = groupby_result.first->release();
  output_cols.resize(group_idx.size() + aggregate_idx.size());
  for (size_t i = 0; i < input_col_order.size(); ++i) {
    int aggregate_col_id     = input_col_order[i];
    auto& aggregation_result = groupby_result.second[i];

    // need to cast count aggregation result to int64 (not applicable for COLLECT_SET)
    if (requests[i].aggregations.size() == 1 &&
        requests[i].aggregations[0]->kind != cudf::aggregation::Kind::COLLECT_SET &&
        (requests[i].aggregations[0]->kind == cudf::aggregation::Kind::COUNT_VALID ||
         requests[i].aggregations[0]->kind == cudf::aggregation::Kind::COUNT_ALL)) {
      if (aggregation_result.results.size() != 1) {
        throw std::runtime_error("Expected 1 result for count aggregation, got " +
                                 std::to_string(aggregation_result.results.size()));
      }
      auto result_view = aggregation_result.results[0]->view();
      if (result_view.type().id() != cudf::type_id::INT64) {
        aggregation_result.results[0] = cudf::cast(result_view,
                                                   cudf::data_type(cudf::type_id::INT64),
                                                   stream,
                                                   memory_space.get_default_allocator());
      }
    }

    const auto& output_idx = input_col_to_output_idx[aggregate_col_id];
    for (size_t j = 0; j < output_idx.size(); ++j) {
      auto result_view = aggregation_result.results[j]->view();
      // Widen decimal result for SUM (expected by duckdb)
      if (requests[i].aggregations[j]->kind == cudf::aggregation::Kind::SUM) {
        if (requests[i].values.type().id() == cudf::type_id::DECIMAL64) {
          aggregation_result.results[j] =
            cudf::cast(result_view,
                       cudf::data_type(cudf::type_id::DECIMAL128, result_view.type().scale()),
                       stream,
                       memory_space.get_default_allocator());
        } else if (requests[i].values.type().id() == cudf::type_id::DECIMAL32) {
          aggregation_result.results[j] =
            cudf::cast(result_view,
                       cudf::data_type(cudf::type_id::DECIMAL64, result_view.type().scale()),
                       stream,
                       memory_space.get_default_allocator());
        }
      }
      size_t output_col_id       = group_idx.size() + output_idx[j];
      output_cols[output_col_id] = std::move(aggregation_result.results[j]);
    }
  }

  // Create the output data batch
  auto output_table = std::make_unique<cudf::table>(
    std::move(output_cols), stream, memory_space.get_default_allocator());
  return make_data_batch(std::move(output_table), memory_space);
}

}  // namespace op
}  // namespace sirius
