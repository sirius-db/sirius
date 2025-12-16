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

#include "operator/merge/gpu_merge_impl.hpp"

#include "data/gpu_data_representation.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/merge.hpp>

namespace sirius {
namespace op {

sirius::unique_ptr<data_batch> gpu_merge_impl::concat(
  const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
  rmm::cuda_stream_view stream,
  memory::memory_space& memory_space,
  data_repository_manager& data_repository_mgr)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error("`input` in `concat()` should at least contain two data batches");
  }

  // Pull input cudf tables and merge.
  sirius::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.resize(input.size());
  for (int i = 0; i < input.size(); ++i) {
    input_cudf_table_views[i] = input[i]->get_cudf_table_view();
  }
  auto output_cudf_table =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Create output data batch.
  auto gpu_table_representation =
    sirius::make_unique<sirius::gpu_table_representation>(*output_cudf_table, memory_space);
  return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                 data_repository_mgr,
                                                 std::move(gpu_table_representation));
}

sirius::unique_ptr<data_batch> gpu_merge_impl::merge_ungrouped_aggregate(
  const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
  const sirius::vector<cudf::aggregation::Kind>& aggregates,
  rmm::cuda_stream_view stream,
  memory::memory_space& memory_space,
  data_repository_manager& data_repository_mgr)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_ungrouped_aggregate()` should at least contain two data batches");
  }

  // Pull input cudf tables and concatenate.
  sirius::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.resize(input.size());
  for (int i = 0; i < input.size(); ++i) {
    input_cudf_table_views[i] = input[i]->get_cudf_table_view();
  }
  if (input_cudf_table_views[0].num_columns() != aggregates.size()) {
    throw std::runtime_error(
      "mismatch between num columns and num aggregates in `merge_ungrouped_aggregate()`");
  }
  auto concatenated =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Aggregate on the concatenated table
  sirius::vector<sirius::unique_ptr<cudf::column>> output_cudf_cols;
  for (int c = 0; c < aggregates.size(); ++c) {
    sirius::unique_ptr<cudf::reduce_aggregation> reduce_aggregation = nullptr;
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
  auto output_cudf_table = sirius::make_unique<cudf::table>(std::move(output_cudf_cols));

  // Create output data batch.
  auto gpu_table_representation =
    sirius::make_unique<sirius::gpu_table_representation>(*output_cudf_table, memory_space);
  return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                 data_repository_mgr,
                                                 std::move(gpu_table_representation));
}

sirius::unique_ptr<data_batch> gpu_merge_impl::merge_grouped_aggregate(
  const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
  int num_group_cols,
  const sirius::vector<cudf::aggregation::Kind>& aggregates,
  rmm::cuda_stream_view stream,
  memory::memory_space& memory_space,
  data_repository_manager& data_repository_mgr)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error(
      "`input` in `merge_grouped_aggregate()` should at least contain two data batches");
  }

  // Pull input cudf tables and concatenate.
  sirius::vector<cudf::table_view> input_cudf_table_views;
  input_cudf_table_views.resize(input.size());
  for (int i = 0; i < input.size(); ++i) {
    input_cudf_table_views[i] = input[i]->get_cudf_table_view();
  }
  if (input_cudf_table_views[0].num_columns() != num_group_cols + aggregates.size()) {
    throw std::runtime_error(
      "`num columns = num_group_cols + num aggregates` not true in `merge_grouped_aggregate()`");
  }
  auto concatenated =
    cudf::concatenate(input_cudf_table_views, stream, memory_space.get_default_allocator());

  // Create cudf groupby and make aggregation requests.
  // Here we don't need to explicitly cast input/output for count or sum of integers,
  // because cudf groupby produces INT64 for sum of integers (both signed and unsigned).
  sirius::vector<cudf::column_view> group_cols;
  for (int c = 0; c < num_group_cols; ++c) {
    group_cols.push_back(concatenated->get_column(c).view());
  }
  cudf::groupby::groupby grpby_obj(cudf::table_view(group_cols), cudf::null_policy::INCLUDE);
  sirius::vector<cudf::groupby::aggregation_request> requests;
  sirius::vector<sirius::unique_ptr<cudf::column>> cast_columns;
  for (int i = 0; i < aggregates.size(); ++i) {
    int aggregate_col_id = num_group_cols + i;
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
  auto output_table = sirius::make_unique<cudf::table>(std::move(output_cols));
  auto gpu_table_representation =
    sirius::make_unique<sirius::gpu_table_representation>(*output_table, memory_space);
  return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                 data_repository_mgr,
                                                 std::move(gpu_table_representation));
}

sirius::unique_ptr<data_batch> gpu_merge_impl::merge_order_by(
  const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
  const sirius::vector<int>& order_key_idx,
  const sirius::vector<cudf::order>& column_order,
  const sirius::vector<cudf::null_order>& null_precedence,
  rmm::cuda_stream_view stream,
  memory::memory_space& memory_space,
  data_repository_manager& data_repository_mgr)
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
  sirius::vector<cudf::table_view> input_tables;
  input_tables.resize(input.size());
  for (int i = 0; i < input.size(); ++i) {
    input_tables[i] = input[i]->get_cudf_table_view();
  }
  auto output_table = cudf::merge(input_tables,
                                  order_key_idx,
                                  column_order,
                                  null_precedence,
                                  stream,
                                  memory_space.get_default_allocator());

  // Create the output data batch
  auto gpu_table_representation =
    sirius::make_unique<sirius::gpu_table_representation>(*output_table, memory_space);
  return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                 data_repository_mgr,
                                                 std::move(gpu_table_representation));
}

sirius::unique_ptr<data_batch> gpu_merge_impl::merge_top_n(
  const sirius::vector<sirius::unique_ptr<data_batch_view>>& input,
  const int limit,
  const int offset,
  const sirius::vector<int>& order_key_idx,
  const sirius::vector<cudf::order>& column_order,
  const sirius::vector<cudf::null_order>& null_precedence,
  rmm::cuda_stream_view stream,
  memory::memory_space& memory_space,
  data_repository_manager& data_repository_mgr)
{
  // Sanity check.
  if (input.size() < 2) {
    throw std::runtime_error("`input` in `merge_top_n()` should at least contain two data batches");
  }
  if (order_key_idx.size() != column_order.size() ||
      order_key_idx.size() != null_precedence.size()) {
    throw std::runtime_error(
      "mismatch between the sizes of `order_key_idx`, `column_order`, and "
      "`null_precedence` in `merge_top_n()`");
  }

  // Pull input cudf tables and merge.
  sirius::vector<cudf::table_view> input_tables;
  input_tables.resize(input.size());
  for (int i = 0; i < input.size(); ++i) {
    input_tables[i] = input[i]->get_cudf_table_view();
  }
  auto merged_table = cudf::merge(input_tables,
                                  order_key_idx,
                                  column_order,
                                  null_precedence,
                                  stream,
                                  memory_space.get_default_allocator());

  // Process limit and offset
  sirius::unique_ptr<cudf::table> output_table = nullptr;
  if (offset >= merged_table->num_rows()) {
    sirius::vector<sirius::unique_ptr<cudf::column>> empty_cols;
    for (int c = 0; c < merged_table->num_columns(); ++c) {
      empty_cols.push_back(cudf::make_empty_column(merged_table->get_column(c).type()));
    }
    output_table = sirius::make_unique<cudf::table>(std::move(empty_cols));
  } else if (offset == 0 && limit >= merged_table->num_rows()) {
    output_table = std::move(merged_table);
  } else {
    output_table = sirius::make_unique<cudf::table>(
      cudf::slice(merged_table->view(),
                  {offset, std::min(merged_table->num_rows(), offset + limit)},
                  stream)[0],
      stream,
      memory_space.get_default_allocator());
  }

  // Create the output data batch
  auto gpu_table_representation =
    sirius::make_unique<sirius::gpu_table_representation>(*output_table, memory_space);
  return sirius::make_unique<sirius::data_batch>(data_repository_mgr.get_next_data_batch_id(),
                                                 data_repository_mgr,
                                                 std::move(gpu_table_representation));
}

}  // namespace op
}  // namespace sirius
