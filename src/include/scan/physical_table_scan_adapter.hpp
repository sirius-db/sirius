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

#pragma once

// sirius
#include <gpu_physical_operator.hpp>

// duckdb
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/function/table_function.hpp>

namespace duckdb {
//===--------------------------------------------------===//
// DuckDBPhysicalTableScan
//===--------------------------------------------------===//

/**
 * @brief Adapter class that wraps a DuckDB PhysicalTableScan operator for use in Sirius pipelines.
 */
class physical_table_scan_adapter : GPUPhysicalOperator {
 public:
  static constexpr PhysicalOperatorType TYPE = PhysicalOperatorType::TABLE_SCAN;

  //===----------Constructor----------===//
  physical_table_scan_adapter(vector<duckdb::LogicalType> types,
                              TableFunction function,
                              unique_ptr<FunctionData> bind_data,
                              vector<LogicalType> returned_types,
                              vector<ColumnIndex> column_ids,
                              vector<idx_t> projection_ids,
                              vector<string> names,
                              unique_ptr<TableFilterSet> table_filters,
                              idx_t estimated_cardinality,
                              ExtraOperatorInfo extra_info,
                              vector<Value> parameters)
    : GPUPhysicalOperator(PhysicalOperatorType::TABLE_SCAN, types, estimated_cardinality),
      physical_table_scan(std::move(types),
                          std::move(function),
                          std::move(bind_data),
                          std::move(returned_types),
                          std::move(column_ids),
                          std::move(projection_ids),
                          std::move(names),
                          std::move(table_filters),
                          estimated_cardinality,
                          extra_info,
                          std::move(parameters)) {};

  //===----------Fields----------===//
  duckdb::PhysicalTableScan physical_table_scan;
};

}  // namespace duckdb