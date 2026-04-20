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

#include "op/sirius_physical_parquet_scan.hpp"

#include "expression_executor/gpu_expression_translator.hpp"
#include "log/logging.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/sirius_physical_table_scan.hpp"

namespace sirius {
namespace op {

// Helper function to deep copy ExtraOperatorInfo
duckdb::ExtraOperatorInfo copy_extra_info_parquet_scan(const duckdb::ExtraOperatorInfo& src)
{
  duckdb::ExtraOperatorInfo copy;
  copy.file_filters = src.file_filters;
  if (src.total_files.IsValid()) { copy.total_files = src.total_files.GetIndex(); }
  if (src.filtered_files.IsValid()) { copy.filtered_files = src.filtered_files.GetIndex(); }
  if (src.sample_options) { copy.sample_options = src.sample_options->Copy(); }
  return copy;
}

sirius_physical_parquet_scan::sirius_physical_parquet_scan(sirius_physical_table_scan* table_scan)
  : sirius_physical_parquet_scan(
      table_scan->types,
      table_scan->function,
      table_scan->bind_data ? table_scan->bind_data->Copy() : nullptr,
      table_scan->returned_types,
      table_scan->column_ids,
      table_scan->projection_ids,
      table_scan->names,
      table_scan->table_filters ? table_scan->table_filters->Copy() : nullptr,
      table_scan->estimated_cardinality,
      copy_extra_info_parquet_scan(table_scan->extra_info),
      table_scan->parameters,
      table_scan->virtual_columns,
      table_scan)  // Pass pointer to the table scan for filter pushdown
{
}

sirius_physical_parquet_scan::sirius_physical_parquet_scan(
  duckdb::vector<sirius::logical_type> types,
  duckdb::TableFunction function_p,
  duckdb::unique_ptr<duckdb::FunctionData> bind_data_p,
  duckdb::vector<sirius::logical_type> returned_types_p,
  duckdb::vector<duckdb::ColumnIndex> column_ids_p,
  duckdb::vector<std::size_t> projection_ids_p,
  duckdb::vector<std::string> names_p,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters_p,
  std::size_t estimated_cardinality,
  duckdb::ExtraOperatorInfo extra_info,
  duckdb::vector<duckdb::Value> parameters_p,
  duckdb::virtual_column_map_t virtual_columns_p,
  sirius_physical_table_scan* table_scan)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARQUET_SCAN, std::move(types), estimated_cardinality),
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
  if (table_filters && !table_filters->filters.empty()) {
    auto name_resolver = [&](duckdb::idx_t ref_index) -> std::string {
      auto const primary_idx = column_ids[ref_index].GetPrimaryIndex();
      return names[primary_idx];
    };
    auto batch_column_map  = build_batch_column_map(projection_ids, column_ids.size());
    auto duckdb_expression = convert_table_filters_to_expression(
      *table_filters, column_ids, returned_types, batch_column_map);
    if (duckdb_expression) {
      gpu_expression_translator translator(rmm::cuda_stream_default,
                                           cudf::get_current_device_resource_ref());
      translated_filter =
        translator.translate_expression_with_names(*duckdb_expression, name_resolver);
      if (!translated_filter) {
        SIRIUS_LOG_INFO(
          "[sirius_physical_parquet_scan] Failed to translate filter expression for pushdown. "
          "Filter will be applied in the table scan operator.");
      } else {
        // Instruct the sirius_physical_table_scan operator to bypass execute()
        if (table_scan) { table_scan->passthrough = true; }
      }
      // Move the duckdb_expression into the table scan
      if (table_scan) { table_scan->filter_expr = std::move(duckdb_expression); }
    }
  } else {
    translated_filter = std::nullopt;
  }
}

}  // namespace op
}  // namespace sirius
