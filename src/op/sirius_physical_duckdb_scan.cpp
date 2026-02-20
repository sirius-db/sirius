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

#include "op/sirius_physical_duckdb_scan.hpp"

#include "config.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "log/logging.hpp"
#include "utils.hpp"

namespace sirius {
namespace op {

// Helper function to deep copy ExtraOperatorInfo
duckdb::ExtraOperatorInfo copy_extra_info_duckdb_scan(const duckdb::ExtraOperatorInfo& src)
{
  duckdb::ExtraOperatorInfo copy;
  copy.file_filters = src.file_filters;
  if (src.total_files.IsValid()) { copy.total_files = src.total_files.GetIndex(); }
  if (src.filtered_files.IsValid()) { copy.filtered_files = src.filtered_files.GetIndex(); }
  if (src.sample_options) { copy.sample_options = src.sample_options->Copy(); }
  return copy;
}

sirius_physical_duckdb_scan::sirius_physical_duckdb_scan(sirius_physical_table_scan* table_scan)
  : sirius_physical_duckdb_scan(
      table_scan->types,
      table_scan->function,
      table_scan->bind_data ? table_scan->bind_data->Copy() : nullptr,
      table_scan->returned_types,
      table_scan->column_ids,
      table_scan->projection_ids,
      table_scan->names,
      table_scan->table_filters ? table_scan->table_filters->Copy() : nullptr,
      table_scan->estimated_cardinality,
      copy_extra_info_duckdb_scan(table_scan->extra_info),
      table_scan->parameters,
      table_scan->virtual_columns)
{
}

sirius_physical_duckdb_scan::sirius_physical_duckdb_scan(
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
      SiriusPhysicalOperatorType::DUCKDB_SCAN, std::move(types), estimated_cardinality),
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
  // Build scanned_types: the types of ALL columns DuckDB will output, in column_ids order.
  // DuckDB's table function fills the DataChunk with columns in column_ids order, regardless
  // of projection_ids. projection_ids only control which columns the PhysicalTableScan keeps
  // after the scan function returns. Since Sirius handles projection at the TABLE_SCAN level,
  // we must initialize the DataChunk with ALL column_ids types in their original order.
  auto num_cols = column_ids.size();
  for (duckdb::idx_t i = 0; i < num_cols; i++) {
    auto col_idx = column_ids[i].GetPrimaryIndex();
    if (col_idx == duckdb::DConstants::INVALID_INDEX) {
      // ROW_ID virtual column
      scanned_types.push_back(duckdb::LogicalType::BIGINT);
    } else {
      scanned_types.push_back(returned_types[col_idx]);
    }
  }

  if (scanned_types.empty()) {
    scanned_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  }

  fake_table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  SIRIUS_LOG_DEBUG("Table scan column ids: {}", column_ids.size());
}

}  // namespace op
}  // namespace sirius
