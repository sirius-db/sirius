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

#include "op/sirius_physical_table_scan.hpp"

#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"

#include <cudf/table/table.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <format>

namespace sirius {
namespace op {

uint64_t get_chunk_data_byte_size(duckdb::LogicalType type, duckdb::idx_t cardinality)
{
  auto physical_size = duckdb::GetTypeIdSize(type.InternalType());
  return cardinality * physical_size;
}

sirius_physical_table_scan::sirius_physical_table_scan(
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::TableFunction function_p,
  duckdb::unique_ptr<duckdb::FunctionData> bind_data_p,
  duckdb::vector<duckdb::LogicalType> returned_types_p,
  duckdb::vector<duckdb::ColumnIndex> column_ids_p,
  duckdb::vector<duckdb::idx_t> projection_ids_p,
  duckdb::vector<std::string> names_p,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters_p,
  duckdb::idx_t estimated_cardinality,
  duckdb::ExtraOperatorInfo extra_info,
  duckdb::vector<duckdb::Value> parameters_p,
  duckdb::virtual_column_map_t virtual_columns_p)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::TABLE_SCAN, std::move(types), estimated_cardinality),
    function(std::move(function_p)),
    bind_data(std::move(bind_data_p)),
    returned_types(std::move(returned_types_p)),
    column_ids(std::move(column_ids_p)),
    projection_ids(std::move(projection_ids_p)),
    names(std::move(names_p)),
    table_filters(std::move(table_filters_p)),
    extra_info(std::move(extra_info)),
    parameters(std::move(parameters_p)),
    virtual_columns(std::move(virtual_columns_p))
{
}

/// Build a mapping from column_ids index to batch column position.
///
/// The parquet scan (make_selected_column_indices) produces batch columns in
/// column_ids order, but only for indices present in projection_ids.
/// For example, if column_ids has 5 entries and projection_ids = {1, 3}:
///   batch position 0 → column_ids[1]
///   batch position 1 → column_ids[3]
///
/// Returns a vector of size column_ids_count where:
///   result[i] = batch position of column_ids[i], or idx_t(-1) if not projected.
///
/// When projection_ids is empty, every column_ids entry maps to its own index.
static std::vector<duckdb::idx_t> build_batch_column_map(
  const duckdb::vector<duckdb::idx_t>& projection_ids,
  duckdb::idx_t column_ids_count)
{
  constexpr auto NOT_PROJECTED = static_cast<duckdb::idx_t>(-1);
  std::vector<duckdb::idx_t> map(column_ids_count, NOT_PROJECTED);

  if (projection_ids.empty()) {
    for (duckdb::idx_t i = 0; i < column_ids_count; i++) {
      map[i] = i;
    }
    return map;
  }

  // Sort projected indices — this matches the iteration order in
  // make_selected_column_indices which walks column_ids[0..N) and
  // includes only indices present in the projected set.
  std::vector<duckdb::idx_t> sorted(projection_ids.begin(), projection_ids.end());
  std::sort(sorted.begin(), sorted.end());

  for (duckdb::idx_t batch_pos = 0; batch_pos < sorted.size(); batch_pos++) {
    if (sorted[batch_pos] < column_ids_count) {
      map[sorted[batch_pos]] = batch_pos;
    }
  }
  return map;
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<duckdb::LogicalType>& returned_types,
  const std::vector<duckdb::idx_t>& batch_column_map)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_expressions;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    if (column_index >= column_ids.size()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: column_index ({}) >= column_ids.size() ({})",
                    column_index,
                    column_ids.size()));
    }
    auto primary_idx = column_ids[column_index].GetPrimaryIndex();
    if (primary_idx >= returned_types.size()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: primary_idx ({}) >= returned_types.size() ({})",
                    primary_idx,
                    returned_types.size()));
    }
    auto col_type = returned_types[primary_idx];

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: column_index={}, primary_idx={}, type={}, filter_type={}",
                     column_index,
                     primary_idx,
                     col_type.ToString(),
                     static_cast<int>(filter->filter_type));

    auto batch_column_index = batch_column_map[column_index];
    if (batch_column_index == static_cast<duckdb::idx_t>(-1)) {
      throw std::runtime_error(
        std::format("TABLE_SCAN filter: column_index ({}) not in projected batch", column_index));
    }

    SIRIUS_LOG_DEBUG("TABLE_SCAN filter: batch_column_index={}", batch_column_index);

    auto column_ref =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(col_type, batch_column_index);
    auto expr = filter->ToExpression(*column_ref);
    filter_expressions.push_back(std::move(expr));
  }

  if (filter_expressions.empty()) { return nullptr; }
  if (filter_expressions.size() == 1) { return std::move(filter_expressions[0]); }

  auto conjunction =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& expr : filter_expressions) {
    conjunction->children.push_back(std::move(expr));
  }
  return conjunction;
}

std::unique_ptr<operator_data> sirius_physical_table_scan::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_table_scan::execute"};
  const auto& input_batches = input_data.get_data_batches();

  // Build the column_ids index → batch position mapping once.
  // Both filter expression construction and post-filter projection use this.
  auto batch_column_map = build_batch_column_map(projection_ids, column_ids.size());

  duckdb::unique_ptr<duckdb::Expression> filter_expr;
  if (table_filters) {
    filter_expr = convert_table_filters_to_expression(
      *table_filters, column_ids, returned_types, batch_column_map);
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  if (filter_expr != nullptr) {
    duckdb::sirius::GpuExpressionExecutor gpu_expression_executor(*filter_expr);
    for (size_t batch_idx = 0; batch_idx < input_batches.size(); batch_idx++) {
      auto const& batch = input_batches[batch_idx];
      if (!batch) { continue; }
      auto filtered_batch = gpu_expression_executor.select(batch, stream);
      if (filtered_batch) { output_batches.push_back(std::move(filtered_batch)); }
    }
  } else {
    for (auto const& batch : input_batches) {
      if (batch) { output_batches.push_back(batch); }
    }
  }

  // After filtering, project away filter-only columns if the batch has more
  // columns than the operator's output type list expects.
  duckdb::idx_t expected_output_columns = types.size();
  bool needs_projection                 = false;

  if (!output_batches.empty() && output_batches[0]) {
    auto& first_batch_rep =
      output_batches[0]->get_data()->cast<cucascade::gpu_table_representation>();
    auto& first_table = first_batch_rep.get_table();
    if (first_table.num_columns() > expected_output_columns) { needs_projection = true; }
  }

  if (needs_projection) {
    SIRIUS_LOG_DEBUG("TABLE_SCAN projection: expected_output_columns={}, projection_ids.size()={}, "
                     "column_ids.size()={}",
                     expected_output_columns,
                     projection_ids.size(),
                     column_ids.size());

    if (expected_output_columns > projection_ids.size()) {
      throw std::runtime_error(
        std::format("TABLE_SCAN projection error: expected_output_columns ({}) > "
                    "projection_ids.size() ({})",
                    expected_output_columns,
                    projection_ids.size()));
    }

    std::vector<std::shared_ptr<cucascade::data_batch>> projected_batches;
    projected_batches.reserve(output_batches.size());

    for (auto& batch : output_batches) {
      if (!batch) { continue; }

      auto& gpu_rep = batch->get_data()->cast<cucascade::gpu_table_representation>();
      auto table    = gpu_rep.release_table();
      auto columns  = table->release();

      // Select output columns using the batch column map.
      // projection_ids[0..expected_output_columns) are the output columns
      // in the order the downstream operator expects.
      std::vector<std::unique_ptr<cudf::column>> selected;
      selected.reserve(expected_output_columns);
      for (duckdb::idx_t i = 0; i < expected_output_columns; i++) {
        auto batch_idx = batch_column_map[projection_ids[i]];
        if (batch_idx == static_cast<duckdb::idx_t>(-1) || batch_idx >= columns.size()) {
          throw std::runtime_error(
            std::format("TABLE_SCAN projection OOB: projection_ids[{}]={} → batch_idx={} >= "
                        "columns.size()={}",
                        i,
                        projection_ids[i],
                        batch_idx,
                        columns.size()));
        }
        selected.push_back(std::move(columns[batch_idx]));
      }

      auto projected_table = std::make_unique<cudf::table>(std::move(selected));
      auto* space          = batch->get_memory_space();
      auto projected_rep =
        std::make_unique<cucascade::gpu_table_representation>(std::move(projected_table), *space);
      auto projected_batch =
        std::make_shared<cucascade::data_batch>(batch->get_batch_id(), std::move(projected_rep));

      projected_batches.push_back(std::move(projected_batch));
    }

    output_batches = std::move(projected_batches);
  }

  return std::make_unique<operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
