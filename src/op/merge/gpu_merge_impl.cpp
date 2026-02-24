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

#include "op/merge/gpu_merge_impl.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/merge.hpp>

namespace sirius {
namespace op {

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::concat(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error("`input` in `concat()` should at least contain two data batches");
  }

  // Pull input cudf tables and merge.
  std::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.reserve(input.size());
  for (const auto& batch : input) {
    input_cudf_table_views.push_back(get_cudf_table_view(*batch));
  }
  auto output_cudf_table =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Create output data batch.
  return make_data_batch(std::move(output_cudf_table), memory_space);
}

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::merge_ungrouped_aggregate(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  const std::vector<std::optional<cudf::size_type>>& merge_nth_index,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_ungrouped_aggregate()` should at least contain two data batches");
  }
  if (merge_nth_index.size() != aggregates.size()) {
    throw std::runtime_error(
      "`merge_nth_index` must have the same size as `aggregates` in `merge_ungrouped_aggregate()`");
  }

  // Pull input cudf tables and concatenate.
  std::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.reserve(input.size());
  for (const auto& batch : input) {
    input_cudf_table_views.push_back(get_cudf_table_view(*batch));
  }
  if (input_cudf_table_views[0].num_columns() != static_cast<cudf::size_type>(aggregates.size())) {
    throw std::runtime_error(
      "mismatch between num columns and num aggregates in `merge_ungrouped_aggregate()`");
  }
  auto concatenated =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Aggregate on the concatenated table
  std::vector<std::unique_ptr<cudf::column>> output_cudf_cols;
  for (size_t c = 0; c < aggregates.size(); ++c) {
    std::unique_ptr<cudf::reduce_aggregation> reduce_aggregation = nullptr;
    cudf::data_type output_type = concatenated->get_column(c).type();
    switch (aggregates[c]) {
      case cudf::aggregation::Kind::MIN: {
        reduce_aggregation = cudf::make_min_aggregation<cudf::reduce_aggregation>();
        break;
      }
      case cudf::aggregation::Kind::MAX: {
        reduce_aggregation = cudf::make_max_aggregation<cudf::reduce_aggregation>();
        break;
      }
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
          default: break;
        }
        reduce_aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        break;
      }
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        output_type        = cudf::data_type(cudf::type_id::INT64);
        reduce_aggregation = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        break;
      }
      case cudf::aggregation::Kind::NTH_ELEMENT: {
        if (!merge_nth_index[c].has_value()) {
          throw std::runtime_error(
            "NTH_ELEMENT aggregate requires a value in `merge_nth_index` in "
            "`merge_ungrouped_aggregate()`");
        }
        reduce_aggregation = cudf::make_nth_element_aggregation<cudf::reduce_aggregation>(
          *merge_nth_index[c], cudf::null_policy::INCLUDE);
        break;
      }
      default:
        throw std::runtime_error(
          "Unsupported cudf aggregate kind in `merge_ungrouped_aggregate()`: " +
          std::to_string(static_cast<int>(aggregates[c])));
    }
    auto output_scalar = cudf::reduce(concatenated->get_column(c),
                                      *reduce_aggregation,
                                      output_type,
                                      stream,
                                      memory_space.get_default_allocator());
    output_cudf_cols.push_back(cudf::make_column_from_scalar(
      *output_scalar, 1, stream, memory_space.get_default_allocator()));
  }
  auto output_cudf_table = std::make_unique<cudf::table>(
    std::move(output_cudf_cols), stream, memory_space.get_default_allocator());

  // Create output data batch.
  return make_data_batch(std::move(output_cudf_table), memory_space);
}

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::merge_grouped_aggregate(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
  int num_group_cols,
  const std::vector<cudf::aggregation::Kind>& aggregates,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_grouped_aggregate()` should at least contain two data batches");
  }

  // Pull input cudf tables and concatenate.
  std::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.reserve(input.size());
  for (const auto& batch : input) {
    input_cudf_table_views.push_back(get_cudf_table_view(*batch));
  }
  if (input_cudf_table_views[0].num_columns() !=
      num_group_cols + static_cast<int>(aggregates.size())) {
    throw std::runtime_error(
      "`num columns = num_group_cols + num aggregates` not true in `merge_grouped_aggregate()`");
  }
  auto concatenated =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Create cudf groupby and make aggregation requests.
  std::vector<cudf::column_view> group_cols;
  for (int c = 0; c < num_group_cols; ++c) {
    group_cols.push_back(concatenated->get_column(c).view());
  }
  cudf::groupby::groupby grpby_obj(cudf::table_view(group_cols), cudf::null_policy::INCLUDE);
  std::vector<cudf::groupby::aggregation_request> requests;
  for (size_t i = 0; i < aggregates.size(); ++i) {
    int aggregate_col_id = num_group_cols + static_cast<int>(i);
    cudf::groupby::aggregation_request request;
    request.values = concatenated->get_column(aggregate_col_id).view();
    switch (aggregates[i]) {
      case cudf::aggregation::Kind::MIN: {
        request.aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case cudf::aggregation::Kind::MAX: {
        request.aggregations.push_back(cudf::make_max_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case cudf::aggregation::Kind::SUM:
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: {
        request.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        break;
      }
      case cudf::aggregation::Kind::COLLECT_SET: {
        // Intermediate column is a LIST produced by local COLLECT_SET. MERGE_SETS unions the
        // per-partition lists and drops duplicates, producing a deduplicated LIST per group.
        request.aggregations.push_back(
          cudf::make_merge_sets_aggregation<cudf::groupby_aggregation>());
        break;
      }
      default:
        throw std::runtime_error(
          "Unsupported cudf aggregate kind in `merge_grouped_aggregate()`: " +
          std::to_string(static_cast<int>(aggregates[i])));
    }
    requests.push_back(std::move(request));
  }

  // Call cudf groupby and populate output columns
  auto groupby_result = grpby_obj.aggregate(requests, stream, memory_space.get_default_allocator());
  auto output_cols    = groupby_result.first->release();
  for (auto& aggregation_result : groupby_result.second) {
    output_cols.push_back(std::move(aggregation_result.results[0]));
  }

  // Create the output data batch
  auto output_table = std::make_unique<cudf::table>(
    std::move(output_cols), stream, memory_space.get_default_allocator());
  return make_data_batch(std::move(output_table), memory_space);
}

std::shared_ptr<cucascade::data_batch> gpu_merge_impl::merge_order_by(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input,
  const std::vector<int>& order_key_idx,
  const std::vector<cudf::order>& column_order,
  const std::vector<cudf::null_order>& null_precedence,
  rmm::cuda_stream_view stream,
  cucascade::memory::memory_space& memory_space)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_order_by()` should at least contain two data batches");
  }
  if (order_key_idx.size() != column_order.size() ||
      order_key_idx.size() != null_precedence.size()) {
    throw std::runtime_error(
      "mismatch between the sizes of `order_key_idx`, `column_order`, and "
      "`null_precedence` in `merge_order_by()`");
  }

  // Pull input cudf tables and merge.
  std::vector<cudf::table_view> input_tables;
  input_tables.reserve(input.size());
  for (const auto& batch : input) {
    input_tables.push_back(get_cudf_table_view(*batch));
  }
  auto output_table = cudf::merge(input_tables,
                                  order_key_idx,
                                  column_order,
                                  null_precedence,
                                  stream,
                                  memory_space.get_default_allocator());

  // Create the output data batch
  return make_data_batch(std::move(output_table), memory_space);
}

}  // namespace op
}  // namespace sirius
