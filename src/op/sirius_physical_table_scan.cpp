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
  // duckdb::GPUBufferManager* gpuBufferManager = &(duckdb::GPUBufferManager::GetInstance());
  // column_size = gpuBufferManager->customCudaHostAlloc<uint64_t>(column_ids.size());
  // mask_size   = gpuBufferManager->customCudaHostAlloc<uint64_t>(column_ids.size());
  for (int col = 0; col < num_cols; col++) {
    // column_size[col] = 0;
    // mask_size[col]   = 0;
    scanned_types.push_back(returned_types[column_ids[col].GetPrimaryIndex()]);
    scanned_ids.push_back(col);
  }

  if (num_cols == 0) {  // Ensure that scanned_types and ids are properly initialized
    scanned_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::UBIGINT));
  }

  fake_table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  // already_cached     = gpuBufferManager->customCudaHostAlloc<bool>(column_ids.size());
  // if (Config::USE_OPT_TABLE_SCAN) {
  //   num_rows = 0;
  //   cuda_streams.resize(Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS);
  //   for (int i = 0; i < Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS; i++) {
  //     cudaStreamCreate(&cuda_streams[i]);
  //   }
  // }
  SIRIUS_LOG_DEBUG("Table scan column ids: {}", column_ids.size());
}

duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<duckdb::LogicalType>& returned_types)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filter_expressions;

  for (auto& [column_index, filter] : filters.filters) {
    // Skip optional and IS_NOT_NULL filters
    if (filter->filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
        filter->filter_type == duckdb::TableFilterType::IS_NOT_NULL) {
      continue;
    }

    // Create column reference for this filter
    auto col_type   = returned_types[column_ids[column_index].GetPrimaryIndex()];
    auto column_ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(col_type, column_index);

    // Convert filter to expression
    filter_expressions.push_back(filter->ToExpression(*column_ref));
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
  SIRIUS_LOG_DEBUG("Executing table scan");
  auto filter_expr =
    convert_table_filters_to_expression(*table_filters, column_ids, returned_types);

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  if (filter_expr) {
    SIRIUS_LOG_DEBUG("Converted table filters to expression: {}", filter_expr->ToString());

    // The executor uses the data_batch API to filter rows according to `expression`.

    duckdb::sirius::GpuExpressionExecutor gpu_expression_executor(*filter_expr);
    for (auto const& batch : input_batches) {
      if (!batch) { continue; }
      auto filtered_batch = gpu_expression_executor.select(batch);
      if (filtered_batch) { output_batches.push_back(std::move(filtered_batch)); }
    }
  } else {
    for (auto const& batch : input_batches) {
      if (batch) { output_batches.push_back(batch); }
    }
  }

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  SIRIUS_LOG_DEBUG("Filter time: {:.2f} ms", duration.count() / 1000.0);
  return output_batches;
}

}  // namespace op
}  // namespace sirius
