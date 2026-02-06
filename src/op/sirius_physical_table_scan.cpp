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

#include "config.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "utils.hpp"

#include <cudf/table/table.hpp>

#include <cucascade/data/gpu_data_representation.hpp>

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
    virtual_columns(std::move(virtual_columns_p)),
    gen_row_id_column(column_ids.back().GetPrimaryIndex() == duckdb::DConstants::INVALID_INDEX)
{
  auto num_cols = column_ids.size() - gen_row_id_column;
  for (int col = 0; col < num_cols; col++) {
    scanned_types.push_back(returned_types[column_ids[col].GetPrimaryIndex()]);
    scanned_ids.push_back(col);
  }

  if (num_cols == 0) {  // Ensure that scanned_types and ids are properly initialized
    scanned_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::UBIGINT));
  }

  fake_table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  SIRIUS_LOG_DEBUG("Table scan column ids: {}", column_ids.size());
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<duckdb::LogicalType>& returned_types,
  const duckdb::vector<duckdb::idx_t>& projection_ids)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_expressions;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    auto primary_idx = column_ids[column_index].GetPrimaryIndex();
    auto col_type    = returned_types[primary_idx];

    // The batch columns are produced by DuckDB scan in the same order as column_ids.
    // So the batch column index is just the column_index itself.
    duckdb::idx_t batch_column_index = column_index;

    // Create column reference for this filter - uses the batch column index
    auto column_ref =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(col_type, batch_column_index);

    // Convert filter to expression
    auto expr = filter->ToExpression(*column_ref);
    filter_expressions.push_back(std::move(expr));
  }

  // No filters to apply
  if (filter_expressions.empty()) { return nullptr; }

  // Single filter - return directly without conjunction wrapper
  if (filter_expressions.size() == 1) { return std::move(filter_expressions[0]); }

  // Multiple filters - wrap in CONJUNCTION_AND
  auto conjunction =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& expr : filter_expressions) {
    conjunction->children.push_back(std::move(expr));
  }
  return conjunction;
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_table_scan::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches)
{
  auto start = std::chrono::high_resolution_clock::now();

  duckdb::unique_ptr<duckdb::Expression> filter_expr;
  if (table_filters) {
    filter_expr = convert_table_filters_to_expression(
      *table_filters, column_ids, returned_types, projection_ids);
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  if (filter_expr != nullptr) {
    // The executor uses the data_batch API to filter rows according to `expression`.
    duckdb::sirius::GpuExpressionExecutor gpu_expression_executor(*filter_expr);
    for (size_t batch_idx = 0; batch_idx < input_batches.size(); batch_idx++) {
      auto const& batch = input_batches[batch_idx];
      if (!batch) { continue; }

      auto filtered_batch = gpu_expression_executor.select(batch);
      if (filtered_batch) { output_batches.push_back(std::move(filtered_batch)); }
    }
  } else {
    for (auto const& batch : input_batches) {
      if (batch) { output_batches.push_back(batch); }
    }
  }

  // After filtering, we may need to project away filter-only columns.
  // The 'types' member indicates the expected output columns (not including filter-only columns).
  // If we have more columns than expected output types, we need to project.
  duckdb::idx_t expected_output_columns = types.size();
  bool needs_projection                 = false;

  if (!output_batches.empty() && output_batches[0]) {
    auto& first_batch_rep =
      output_batches[0]->get_data()->cast<cucascade::gpu_table_representation>();
    auto& first_table = first_batch_rep.get_table();
    if (first_table.num_columns() > expected_output_columns) { needs_projection = true; }
  }

  if (needs_projection) {
    // The batch columns are in the same order as column_ids.
    // projection_ids tells us which column_ids indices to select for output.
    // We want the first expected_output_columns elements from projection_ids.
    std::vector<std::shared_ptr<cucascade::data_batch>> projected_batches;
    projected_batches.reserve(output_batches.size());

    for (auto& batch : output_batches) {
      if (!batch) { continue; }

      // Release the table from the batch (zero-copy: we're the sole consumer)
      auto& gpu_rep = batch->get_data()->cast<cucascade::gpu_table_representation>();
      auto table    = gpu_rep.release_table();
      auto columns  = table->release();

      // Select only the output columns by moving ownership
      std::vector<std::unique_ptr<cudf::column>> selected;
      selected.reserve(expected_output_columns);
      for (duckdb::idx_t i = 0; i < expected_output_columns; i++) {
        selected.push_back(std::move(columns[projection_ids[i]]));
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

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Filter time: {:.2f} ms", duration.count() / 1000.0);
  return output_batches;
}

}  // namespace op
}  // namespace sirius
