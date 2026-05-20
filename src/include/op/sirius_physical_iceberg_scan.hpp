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

#include "op/sirius_physical_parquet_scan.hpp"

#include <memory>
#include <string>
#include <vector>

namespace sirius::op::scan {
struct IcebergDeleteData;
}

namespace sirius {
namespace op {

/**
 * @brief Physical operator for scanning Apache Iceberg tables on the GPU.
 *
 * Inherits the parquet scan machinery (bind_data, column_ids, projection_ids, etc.)
 * from sirius_physical_parquet_scan. The iceberg-specific additions are the delete
 * file lists for V2 row-level delete support.
 *
 * For V1 Iceberg tables (append-only), positional_delete_files and
 * equality_delete_files are both empty and execution is identical to a plain
 * parquet scan. For V2 tables the iceberg_scan_task_global_state reads these
 * files and installs a post-convert hook that applies the deletes on GPU after
 * each row-group batch is decompressed.
 */
class sirius_physical_iceberg_scan : public sirius_physical_parquet_scan {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::ICEBERG_SCAN;

 public:
  /// Construct from a generic table scan (used in sirius_engine.cpp routing).
  explicit sirius_physical_iceberg_scan(sirius_physical_table_scan* table_scan);

  /// Full-parameter constructor (mirrors sirius_physical_parquet_scan).
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
                               duckdb::virtual_column_map_t virtual_columns);

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  // -------------------------------------------------------------------------
  // Fully materialized delete data (populated by sirius_engine.cpp).
  // nullptr or empty() for V1 tables.
  // -------------------------------------------------------------------------

  /// Pre-read, immutable delete data shared across task instances.
  std::shared_ptr<const scan::IcebergDeleteData> delete_data;
};

}  // namespace op
}  // namespace sirius
