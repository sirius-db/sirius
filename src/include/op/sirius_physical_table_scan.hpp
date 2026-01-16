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

#include "duckdb/common/extra_operator_info.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "op/sirius_physical_operator.hpp"

namespace sirius {
namespace op {

enum scan_data_type {
  INT16,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
  BOOLEAN,
  DATE,
  VARCHAR,
  DECIMAL32,
  DECIMAL64,
  SQLNULL
};

enum compare_type {
  EQUAL,
  NOTEQUAL,
  GREATERTHAN,
  GREATERTHANOREQUALTO,
  LESSTHAN,
  LESSTHANOREQUALTO,
  IS_NULL,
  IS_NOT_NULL
};

class sirius_physical_table_scan : public sirius_physical_operator {
 public:
  static constexpr const duckdb::PhysicalOperatorType TYPE =
    duckdb::PhysicalOperatorType::TABLE_SCAN;

 public:
  //! Table scan that immediately projects out filter columns that are unused in the remainder of
  //! the query plan
  sirius_physical_table_scan(duckdb::vector<duckdb::LogicalType> types,
                             duckdb::TableFunction function,
                             duckdb::unique_ptr<duckdb::FunctionData> bind_data,
                             duckdb::vector<duckdb::LogicalType> returned_types,
                             duckdb::vector<duckdb::ColumnIndex> column_ids,
                             duckdb::vector<duckdb::idx_t> projection_ids,
                             duckdb::vector<std::string> names,
                             duckdb::unique_ptr<duckdb::TableFilterSet> table_filters,
                             duckdb::idx_t estimated_cardinality,
                             duckdb::ExtraOperatorInfo extra_info,
                             duckdb::vector<duckdb::Value> parameters,
                             duckdb::virtual_column_map_t virtual_columns);

  //! The table function
  duckdb::TableFunction function;
  //! Bind data of the function
  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  //! The types of ALL columns that can be returned by the table function
  duckdb::vector<duckdb::LogicalType> returned_types;
  //! The column ids used within the table function
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  //! The projected-out column ids
  duckdb::vector<duckdb::idx_t> projection_ids;
  //! The names of the columns
  duckdb::vector<std::string> names;
  //! The table filters
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  //! Currently stores info related to filters pushed down into MultiFileLists and sample rate
  //! pushed down into the table scan
  duckdb::ExtraOperatorInfo extra_info;
  //! Parameters
  duckdb::vector<duckdb::Value> parameters;
  //! Contains a reference to dynamically generated table filters (through e.g. a join up in the
  //! tree)
  duckdb::shared_ptr<duckdb::DynamicTableFilterSet> dynamic_filters;
  //! Virtual columns
  duckdb::virtual_column_map_t virtual_columns;

  duckdb::PhysicalTableScan* physical_table_scan;

  duckdb::unique_ptr<duckdb::ColumnDataCollection> collection;

  uint64_t* column_size;

  uint64_t* mask_size;

  bool* already_cached;

  duckdb::vector<duckdb::LogicalType> scanned_types;

  duckdb::vector<duckdb::idx_t> scanned_ids;

  duckdb::unique_ptr<duckdb::TableFilterSet> fake_table_filters;

  //! Whether it's required to generate a separate row id column (e.g., in some select *)
  bool gen_row_id_column;

  //! Only used in optimized table scan
  bool exhausted = false;

 public:
  bool is_source() const override { return true; }
  bool parallel_source() const override { return true; }

  duckdb::unique_ptr<duckdb::LocalSourceState> get_local_source_state(
    duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const override;
  duckdb::unique_ptr<duckdb::GlobalSourceState> get_global_source_state(
    duckdb::ClientContext& context) const override;
};

}  // namespace op
}  // namespace sirius
