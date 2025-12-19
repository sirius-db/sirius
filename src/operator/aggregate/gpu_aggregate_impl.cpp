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

#include "operator/aggregate/gpu_aggregate_impl.hpp"

#include "data/gpu_data_representation.hpp"

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

std::unique_ptr<cucascade::data_batch> gpu_aggregate_impl::local_ungrouped_aggregate(
  const cucascade::data_batch_view& input,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  cucascade::data_repository_manager& data_repository_mgr)
{
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_ungrouped_aggregate()`");
  }
  std::vector<std::unique_ptr<cudf::column>> output_cols;
  auto input_table = input.get_cudf_table_view();
  for (int i = 0; i < aggregates.size(); ++i) {
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
        }
        break;
      }
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        output_type = cudf::data_type(cudf::type_id::INT64);
        break;
      }
    }
    auto output_scalar = cudf::reduce(
      input_col, *reduce_aggregation, output_type, stream, memory_space.get_default_allocator());
    output_cols.push_back(cudf::make_column_from_scalar(
      *output_scalar, 1, cudf::get_default_stream(), memory_space.get_default_allocator()));
  }
  auto output_table = std::make_unique<cudf::table>(std::move(output_cols));

  auto gpu_table_representation =
    std::make_unique<cucascade::gpu_table_representation>(*output_table, memory_space);
  return std::make_unique<cucascade::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                 data_repository_mgr,
                                                 std::move(gpu_table_representation));
}

std::unique_ptr<cucascade::data_batch> gpu_aggregate_impl::local_grouped_aggregate(
  const cucascade::data_batch_view& input,
  const std::vector<int>& group_idx,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<int>& aggregate_idx,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space,
  cucascade::data_repository_manager& data_repository_mgr)
{
  // Sanity check
  if (aggregates.size() != aggregate_idx.size()) {
    throw std::runtime_error(
      "mismatch between the size of `aggregates` and `aggregate_idx` in "
      "`local_grouped_aggregate()`");
  }

  // Create cudf groupby
  auto input_table = input.get_cudf_table_view();
  std::vector<cudf::column_view> group_cols;
  for (int idx : group_idx) {
    group_cols.push_back(input_table.column(idx));
  }
  cudf::groupby::groupby grpby_obj(cudf::table_view(group_cols), cudf::null_policy::INCLUDE);

  // Make aggregation requests, group aggregations on the same column in the single request.
  // Here we don't need to explicitly cast input/output for count or sum of integers,
  // because cudf groupby produces INT32 for count, and promotes to INT64 for sum of integers
  // (both signed and unsigned), so types of local aggregation results are consistent.
  std::unordered_map<int, std::vector<std::unique_ptr<cudf::groupby_aggregation>>> input_col_to_agg;
  std::unordered_map<int, std::vector<int>> input_col_to_output_idx;
  std::vector<int> input_col_order;
  for (int i = 0; i < aggregates.size(); ++i) {
    const auto& aggregate_kind = aggregates[i];
    int aggregate_col_id       = aggregate_idx[i];
    if (!input_col_to_agg.contains(aggregate_col_id)) {
      input_col_order.push_back(aggregate_col_id);
    }
    auto groupby_aggregation = get_local_aggregation<cudf::groupby_aggregation>(aggregate_kind);
    input_col_to_agg[aggregate_col_id].push_back(std::move(groupby_aggregation));
    input_col_to_output_idx[aggregate_col_id].push_back(i);
  }

  std::vector<cudf::groupby::aggregation_request> requests;
  for (int aggregate_col_id : input_col_order) {
    cudf::groupby::aggregation_request request;
    request.values       = input_table.column(aggregate_col_id);
    request.aggregations = std::move(input_col_to_agg[aggregate_col_id]);
    requests.push_back(std::move(request));
  }

  // Call cudf groupby and populate output columns
  auto groupby_result = grpby_obj.aggregate(requests, stream, memory_space.get_default_allocator());
  auto output_cols    = groupby_result.first->release();
  output_cols.resize(group_idx.size() + aggregate_idx.size());
  for (int i = 0; i < input_col_order.size(); ++i) {
    int aggregate_col_id     = input_col_order[i];
    auto& aggregation_result = groupby_result.second[i];
    const auto& output_idx   = input_col_to_output_idx[aggregate_col_id];
    for (int j = 0; j < output_idx.size(); ++j) {
      int output_col_id          = group_idx.size() + output_idx[j];
      output_cols[output_col_id] = std::move(aggregation_result.results[j]);
    }
  }

  // Create the output data batch
  auto output_table = std::make_unique<cudf::table>(std::move(output_cols));
  auto gpu_table_representation =
    std::make_unique<cucascade::gpu_table_representation>(*output_table, memory_space);
  return std::make_unique<cucascade::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                 data_repository_mgr,
                                                 std::move(gpu_table_representation));
}

}  // namespace op
}  // namespace sirius
