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
#include "expression_executor/gpu_expression_translator_internal.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_table_scan.hpp"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {
struct IcebergDeleteData;
}

namespace sirius {
namespace op {

/**
 * @brief Physical operator for scanning Apache Iceberg tables on the GPU.
 *
 * Owns the scan machinery and the iceberg-specific additions are the delete file lists for V2
 * row-level delete support.
 *
 * For V1 Iceberg tables (append-only), positional_delete_files and
 * equality_delete_files are both empty and execution is identical to a plain
 * parquet scan. For V2 tables the iceberg_scan_task_global_state reads these
 * files and installs a post-convert hook that applies the deletes on GPU after
 * each row-group batch is decompressed.
 */
class sirius_physical_iceberg_scan : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::ICEBERG_SCAN;

 public:
  /// Construct from a generic table scan (used in sirius_engine.cpp routing).
  explicit sirius_physical_iceberg_scan(sirius_physical_table_scan* table_scan,
                                        std::vector<int> gpu_device_ids = {});

  sirius_physical_iceberg_scan(duckdb::vector<sirius::logical_type> types,
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
                               sirius_physical_table_scan* table_scan = nullptr,
                               std::vector<int> gpu_device_ids        = {});

  std::optional<task_creation_hint> get_next_task_hint() override
  {
    if (exhausted.load()) { return std::nullopt; }
    return task_creation_hint{TaskCreationHint::READY, this};
  }

  bool is_source() const override { return true; }

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

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

  std::atomic<bool> exhausted{false};
  std::atomic<bool> has_more_partitions{true};

  //! Per-GPU translated filter expressions. Each entry binds the filter's cudf::scalar
  //! device buffers to a specific device so tasks dispatched to that device can
  //! evaluate the AST without hitting cudaErrorInvalidValue / cudaErrorIllegalAddress
  //! under num_gpus>1. Empty = no filter / translation failed.
  std::unordered_map<int, gpu_expression_translator::translated_expression>
    translated_filter_by_device;

  // -------------------------------------------------------------------------
  // Fully materialized delete data (populated by sirius_engine.cpp).
  // nullptr or empty() for V1 tables.
  // -------------------------------------------------------------------------

  /// Pre-read, immutable delete data shared across task instances.
  std::shared_ptr<const scan::IcebergDeleteData> delete_data;
};

}  // namespace op
}  // namespace sirius
