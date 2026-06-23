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
#include "expression_executor/gpu_expression_translator_internal.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_table_scan.hpp"

namespace sirius {
namespace op {

/// Deep-copies an ExtraOperatorInfo. Defined in sirius_physical_parquet_scan.cpp.
duckdb::ExtraOperatorInfo copy_extra_info_parquet_scan(const duckdb::ExtraOperatorInfo& src);

class sirius_physical_parquet_scan : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::PARQUET_SCAN;

 public:
  sirius_physical_parquet_scan(sirius_physical_table_scan* table_scan,
                               std::vector<int> gpu_device_ids = {});

  //! Table scan that immediately projects out filter columns that are unused in the remainder of
  //! the query plan
  sirius_physical_parquet_scan(duckdb::vector<sirius::logical_type> types,
                               duckdb::TableFunction function,
                               duckdb::unique_ptr<duckdb::FunctionData> bind_data,
                               duckdb::vector<sirius::logical_type> returned_types,
                               duckdb::vector<duckdb::ColumnIndex> column_ids,
                               duckdb::vector<std::size_t> projection_ids,
                               duckdb::vector<std::string> names,
                               duckdb::unique_ptr<duckdb::TableFilterSet> table_filters,
                               std::size_t estimated_cardinality,
                               duckdb::ExtraOperatorInfo extra_info,
                               duckdb::vector<duckdb::Value> parameters,
                               duckdb::virtual_column_map_t virtual_columns,
                               sirius_physical_table_scan* physical_table_scan,
                               std::vector<int> gpu_device_ids = {});

  std::optional<task_creation_hint> get_next_task_hint(
    std::optional<std::size_t> downstream_request = std::nullopt) override
  {
    if (exhausted.load()) { return std::nullopt; }
    // Sources have no upstream; honor an explicit downstream request when given so a downstream
    // PIPELINE/PARTIAL barrier doesn't force the scan to drain its entire split queue.
    std::size_t cap = downstream_request.value_or(task_creation_hint::ALL_TASKS);
    return task_creation_hint{TaskCreationHint::READY, this, cap};
  }

  //! The table function
  duckdb::TableFunction function;
  //! Bind data of the function
  duckdb::unique_ptr<duckdb::FunctionData> bind_data;
  //! The types of ALL columns that can be returned by the table function
  duckdb::vector<sirius::logical_type> returned_types;
  //! The column ids used within the table function
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  //! The projected-out column ids
  duckdb::vector<std::size_t> projection_ids;
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

  duckdb::vector<sirius::logical_type> scanned_types;

  duckdb::vector<std::size_t> scanned_ids;

  duckdb::unique_ptr<duckdb::TableFilterSet> fake_table_filters;

  //! Whether it's required to generate a separate row id column (e.g., in some select *)
  bool gen_row_id_column;

  std::atomic<bool> exhausted{false};

  std::atomic<bool> has_more_partitions{true};

  //! Per-GPU translated filter expressions. Each entry binds the filter's cudf::scalar
  //! device buffers to a specific device so tasks dispatched to that device can
  //! evaluate the AST without hitting cudaErrorInvalidValue / cudaErrorIllegalAddress
  //! under num_gpus>1. Empty = no filter / translation failed.
  std::unordered_map<int, gpu_expression_translator::translated_expression>
    translated_filter_by_device;

 public:
  bool is_source() const override { return true; }

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;
};

}  // namespace op
}  // namespace sirius
