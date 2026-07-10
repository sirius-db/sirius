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
#include "log/logging.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"

#include <algorithm>

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
    gen_row_id_column(column_ids.back().IsRowIdColumn())
{
  // Sort projection_ids so that DuckDB's ReferenceColumns() outputs columns in
  // ascending column_ids order. This matches the parquet scan convention (where
  // make_selected_column_indices iterates column_ids in order) and the expectation
  // of build_batch_column_map in sirius_physical_table_scan::execute.
  std::sort(projection_ids.begin(), projection_ids.end());

  // Build scanned_types: the types of columns DuckDB will actually output.
  //
  // When projection_ids is non-empty, DuckDB's seq_scan uses CanRemoveFilterColumns()
  // and ReferenceColumns() internally to output only the projected columns.
  // The DataChunk must have matching types, otherwise ReferenceColumns triggers
  // a type mismatch assertion.
  //
  // When projection_ids is empty, DuckDB outputs all column_ids columns.
  //
  // Both scan paths (parquet + duckdb) produce only projected columns in sorted
  // order. build_batch_column_map in sirius_physical_table_scan::execute remaps
  // filter/projection indices to actual batch column positions.
  if (!projection_ids.empty()) {
    for (auto pid : projection_ids) {
      auto col_idx = column_ids[pid].GetPrimaryIndex();
      if (column_ids[pid].IsRowIdColumn()) {
        scanned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
      } else {
        scanned_types.push_back(returned_types[col_idx]);
      }
    }
  } else {
    auto num_cols = column_ids.size();
    for (std::size_t i = 0; i < num_cols; i++) {
      auto col_idx = column_ids[i].GetPrimaryIndex();
      if (column_ids[i].IsRowIdColumn()) {
        scanned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
      } else {
        scanned_types.push_back(returned_types[col_idx]);
      }
    }
  }

  if (scanned_types.empty()) {
    scanned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  }

  fake_table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  SIRIUS_LOG_DEBUG("Table scan column ids: {}", column_ids.size());
}

void sirius_physical_duckdb_scan::build_pipelines(pipeline::sirius_pipeline& current,
                                                  pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Leaf sink: create_child_meta_pipeline pre-populates [*this], yielding a one-operator
  // pipeline with source=sink=*this. No children, so no recursion.
  D_ASSERT(children.empty());
  meta_pipeline.create_child_meta_pipeline(current, *this);
}

}  // namespace op
}  // namespace sirius
