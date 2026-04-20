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

/**
 * @brief Function that injects hive partition columns into a GPU table.
 *
 * Add constant partition columns (whose values come from the file path, not the parquet data) at
 * the correct positions in the output table.
 */
using partition_inject_fn_t = std::function<std::unique_ptr<cudf::table>(
  std::unique_ptr<cudf::table>, std::string const& data_file_path, rmm::cuda_stream_view)>;

partition_inject_fn_t build_partition_inject_fn(
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<std::string> const& names,
  duckdb::vector<sirius::logical_type> const& returned_types,
  std::vector<size_t> const& selected_column_indices,
  std::vector<hive_partition_column> const& hive_partition_columns,
  std::unordered_set<size_t> const& hive_partition_index_set);

}  // namespace sirius::op::scan