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
#include <helper/logical_type.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>

// cudf
#include <cudf/table/table.hpp>

// standard library
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>

namespace sirius::op::scan {

/**
 * @brief Hive partition column metadata (not in parquet file)
 */
struct hive_partition_column {
  std::string column_name;          ///< Partition column name (e.g. "year")
  std::size_t duckdb_column_index;  ///< Index into scan_op->names / column_ids
};

}  // namespace sirius::op::scan

namespace sirius {

/**
 * @brief Legacy-path closure: inject constant columns into a GPU table.
 *
 * Used by @c host_parquet_representation and @c parquet_scan_task.
 * Carries @p file_path because the schema-reconciliation closure
 * (@c parquet_scan_task_global_state::build_schema_reconciliation) keys per-file column-set
 * lookups by path. Simple closures (the standard hive-injection one built by
 * @c build_partition_inject_fn) accept the @p file_path argument but ignore it.
 *
 * @p partition_values is in @c hive_partition_columns order; empty when the table has no
 *    hive partitions.
 */
using partition_inject_fn_t =
  std::function<std::unique_ptr<cudf::table>(std::unique_ptr<cudf::table>,
                                             std::string const& file_path,
                                             std::vector<std::string> const& partition_values,
                                             rmm::cuda_stream_view)>;

/**
 * @brief Build the legacy hive-partition injection closure.
 *
 * Returns a closure conforming to the legacy @c sirius::partition_inject_fn_t signature so it can
 * coexist with the schema-reconciliation closure on the same storage slot. The closure ignores
 * its @c file_path argument; partition values come from the supplied vector.
 */
partition_inject_fn_t build_partition_inject_fn(
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<std::string> const& names,
  duckdb::vector<sirius::logical_type> const& returned_types,
  std::vector<size_t> const& selected_column_indices,
  std::vector<sirius::op::scan::hive_partition_column> const& hive_partition_columns,
  std::unordered_set<size_t> const& hive_partition_index_set);

}  // namespace sirius
