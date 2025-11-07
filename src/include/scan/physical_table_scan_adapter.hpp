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
//===----------------------------------------------------------------------===//
// Physical Table Scan Adapter
//===----------------------------------------------------------------------===//

/**
 * @brief Adapter class that wraps a DuckDB PhysicalTableScan operator and inherits from
 * GPUPhysicalOperator.
 */
class physical_table_scan_adapter : GPUPhysicalOperator {
 public:
  static constexpr PhysicalOperatorType TYPE = PhysicalOperatorType::TABLE_SCAN;

  //===----------Constructor----------===//
  /**
   * @brief Construct a new physical_table_scan_adapter object (has the same signature as DuckDB's
   * PhysicalTableScan).
   *
   * @param[in] type
   * @param[in] function the table function
   * @param[in] bind_data bind data for the table function
   * @param[in] returned_types the types of ALL columns that can be returned by the table function
   * @param[in] column_ids the column ids used within the table function
   * @param[in] projection_ids the projected-out column ids
   * @param[in] names the names of the columns
   * @param[in] table_filters the table filters
   * @param[in] estimated_cardinality the estimated cardinality of the scan
   * @param[in] extra_info extra operator info (currently stores info related to filters pushed down
   * into MultiFileLists and sample rate pushed down into the table scan)
   * @param[in] parameters contains a reference to dynamically generated table filters (through e.g.
   * a join up in the tree)
   */
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