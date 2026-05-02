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

#include "helper/logical_type.hpp"
#include "sirius_config.hpp"

#include <cudf/io/datasource.hpp>

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Carries the parquet bind-data extracted by the pipeline converter.
 *
 * Populated once during plan generation and parked on the gpu parquet scan
 * operator. The scan_manager reads it during prepare_for_query to construct
 * a parquet_split_provider for the operator.
 */
struct parquet_scan_info {
  duckdb::vector<sirius::logical_type> returned_types;
  std::vector<std::string> file_paths;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  duckdb::vector<std::string> names;
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filters;
  duckdb::vector<duckdb::HivePartitioningIndex> partition_indices;
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;

  /// Per-URI datasource builder, captured by the pipeline converter with a
  /// reference to the live @c sirius_engine. Routes scheme-qualified URIs
  /// (@c s3://, future object stores) through the engine's
  /// @c datasource_factory so the right @c sirius_ioctx is selected; for bare
  /// paths the dispatch is governed by
  /// @c datasource_factory::create_for_parquet_scan — relative bare paths go
  /// straight to cudf's default datasource (DuckDB's iceberg / hive fixtures
  /// hand out relative paths), absolute and @c file:/// paths still flow
  /// through the strict factory's file branch.
  ///
  /// Lifetime invariant: must outlive any @c parquet_split_provider built
  /// from this @c parquet_scan_info. The captured @c sirius_engine reference
  /// is valid as long as the operator that holds this scan_info is owned by
  /// the engine's plan (which is always the case for converter-built plans).
  ///
  /// Empty closure → caller falls back to @c cudf::io::datasource::create(uri)
  /// directly. Used by tests and any non-engine call path that constructs a
  /// @c parquet_split_provider standalone.
  std::function<std::unique_ptr<cudf::io::datasource>(std::string_view)> open_datasource;
};

}  // namespace sirius::op::scan
