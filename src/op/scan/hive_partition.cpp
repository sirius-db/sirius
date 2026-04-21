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

// sirius
#include <helper/type_conversions.hpp>
#include <log/logging.hpp>
#include <op/scan/hive_partition.hpp>

// duckdb
#include <duckdb/common/hive_partitioning.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>

// standard library
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sirius::op::scan {

partition_inject_fn_t build_partition_inject_fn(
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<std::string> const& names,
  duckdb::vector<sirius::logical_type> const& returned_types,
  std::vector<size_t> const& selected_column_indices,
  std::vector<hive_partition_column> const& hive_partition_columns,
  std::unordered_set<size_t> const& hive_partition_index_set)
{
  // Build the output column map in the order the pipeline expects.
  //
  // cuDF returns data columns in selected_column_indices order (which follows column_ids order).
  // Build a DuckDB-index → cuDF-position map, then iterate column_ids to produce the output in
  // the order DuckDB's pipeline operators expect.
  struct col_source {
    bool is_partition;
    size_t data_col_idx;
    std::string partition_name;
    sirius::logical_type type;
  };

  // Map DuckDB primary index → cuDF column position.
  std::unordered_map<size_t, size_t> duckdb_to_cudf;
  for (size_t i = 0; i < selected_column_indices.size(); ++i) {
    duckdb_to_cudf[selected_column_indices[i]] = i;
  }

  // Build output_map in column_ids order (the order the pipeline expects).
  std::vector<col_source> output_map;
  std::unordered_set<size_t> seen;
  for (auto const& col_id : column_ids) {
    auto primary_idx = col_id.GetPrimaryIndex();
    if (duckdb::IsVirtualColumn(primary_idx)) continue;
    if (!seen.insert(primary_idx).second) continue;

    if (hive_partition_index_set.count(primary_idx)) {
      output_map.push_back(col_source{/* is_partition */ true,
                                      /* data_col_idx */ 0,
                                      names[primary_idx],
                                      returned_types[primary_idx]});
    } else {
      auto it = duckdb_to_cudf.find(primary_idx);
      if (it != duckdb_to_cudf.end()) {
        output_map.push_back(col_source{/* is_partition */ false,
                                        /* data_col_idx */ it->second,
                                        /* partition_name */ {},
                                        /* type */ {}});
      }
    }
  }

  SIRIUS_LOG_INFO(
    "[hive_partition] Hive partitions detected: {} partition col(s), {} data col(s), "
    "{} output col(s).",
    hive_partition_columns.size(),
    duckdb_to_cudf.size(),
    output_map.size());

  return [output_map = std::move(output_map)](
           std::unique_ptr<cudf::table> tbl,
           std::string const& file_path,
           rmm::cuda_stream_view stream) -> std::unique_ptr<cudf::table> {
    if (!tbl || tbl->num_rows() == 0) return tbl;

    auto partitions     = duckdb::HivePartitioning::Parse(file_path);
    auto const num_rows = tbl->num_rows();
    auto data_columns   = tbl->release();  // move columns out, no GPU copy

    std::vector<std::unique_ptr<cudf::column>> output_columns;
    output_columns.reserve(output_map.size());

    for (auto const& src : output_map) {
      if (!src.is_partition) {
        output_columns.push_back(std::move(data_columns[src.data_col_idx]));
      } else {
        auto it = partitions.find(src.partition_name);
        if (it == partitions.end()) {
          throw std::runtime_error("[hive_partition] Missing hive partition key '" +
                                   src.partition_name + "' in file path: " + file_path);
        }
        // DefaultCastAs requires a DuckDB type; the scalar factory takes the sirius type.
        auto duckdb_val = duckdb::Value(it->second).DefaultCastAs(sirius::to_duckdb(src.type));
        auto scalar     = sirius::value_to_cudf_scalar(duckdb_val, src.type, stream);
        output_columns.push_back(cudf::make_column_from_scalar(*scalar, num_rows, stream));
      }
    }

    return std::make_unique<cudf::table>(std::move(output_columns));
  };
}

}  // namespace sirius::op::scan
