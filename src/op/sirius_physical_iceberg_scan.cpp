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

#include "op/sirius_physical_iceberg_scan.hpp"

#include "config.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"

namespace sirius {
namespace op {

sirius_physical_iceberg_scan::sirius_physical_iceberg_scan(sirius_physical_table_scan* table_scan)
  : sirius_physical_iceberg_scan(
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
      table_scan->virtual_columns)
{
}

sirius_physical_iceberg_scan::sirius_physical_iceberg_scan(
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
  : sirius_physical_parquet_scan(std::move(types),
                                 std::move(function_p),
                                 std::move(bind_data_p),
                                 std::move(returned_types_p),
                                 std::move(column_ids_p),
                                 std::move(projection_ids_p),
                                 std::move(names_p),
                                 std::move(table_filters_p),
                                 estimated_cardinality,
                                 std::move(extra_info),
                                 std::move(parameters_p),
                                 std::move(virtual_columns_p),
                                 nullptr)
{
  // Override the type set by the parquet scan base constructor
  type = SiriusPhysicalOperatorType::ICEBERG_SCAN;
}

void sirius_physical_iceberg_scan::build_pipelines(pipeline::sirius_pipeline& current,
                                                   pipeline::sirius_meta_pipeline& meta_pipeline)
{
  if (!duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    sirius_physical_operator::build_pipelines(current, meta_pipeline);
    return;
  }
  // Leaf sink: create_child_meta_pipeline pre-populates [*this], yielding a one-operator
  // pipeline with source=sink=*this. No children, so no recursion.
  D_ASSERT(children.empty());
  meta_pipeline.create_child_meta_pipeline(current, *this);
}

}  // namespace op
}  // namespace sirius
