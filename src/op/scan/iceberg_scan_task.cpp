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

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <io/sirius_datasource.hpp>
#include <io/types.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_delete_filter.hpp>
#include <op/scan/iceberg_metadata_reader.hpp>
#include <op/scan/iceberg_scan_task.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state — prepare
//===----------------------------------------------------------------------===//

iceberg_scan_task_global_state::init_data iceberg_scan_task_global_state::prepare(
  sirius_physical_iceberg_scan* scan_op)
{
  auto& bind_data = scan_op->bind_data->Cast<duckdb::MultiFileBindData>();
  if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
    throw std::runtime_error("[iceberg] No input data files to scan");
  }

  auto files = bind_data.file_list->GetAllFiles();
  std::vector<std::string> file_paths;
  file_paths.reserve(files.size());
  for (auto const& f : files) {
    file_paths.push_back(f.path);
  }

  // Compute selected columns. Any hive partition columns present here will be
  // detected and stripped by the base class's initialize_from_files() via
  // schema comparison, then re-injected via init_hive_partitions() below.
  auto selected =
    detail::make_selected_column_indices(scan_op->column_ids, scan_op->projection_ids);

  // Force equality-delete key columns into the scan projection.
  // DuckDB's optimizer may prune columns not referenced in the query
  // (e.g. SELECT count(tx_id) drops user_id, tx_date), but we need
  // those columns to probe the equality-delete hash joins.
  // Extra columns are stripped after delete filtering by the pipeline.
  size_t extra_cols = 0;
  if (scan_op->delete_data && !scan_op->delete_data->equality_delete_groups.empty()) {
    // Build name→index map for O(1) lookups instead of linear scan.
    std::unordered_map<std::string, size_t> name_to_idx;
    for (size_t i = 0; i < scan_op->names.size(); ++i) {
      name_to_idx.emplace(scan_op->names[i], i);
    }
    std::unordered_set<size_t> already_selected(selected.begin(), selected.end());
    for (auto const& group : scan_op->delete_data->equality_delete_groups) {
      for (auto const& key_name : group.key_names) {
        auto it = name_to_idx.find(key_name);
        if (it != name_to_idx.end() &&
            already_selected.find(it->second) == already_selected.end()) {
          selected.push_back(it->second);
          already_selected.insert(it->second);
          ++extra_cols;
          SIRIUS_LOG_DEBUG(
            "[iceberg] Forced equality-delete key column '{}' (idx={}) "
            "into scan projection.",
            key_name,
            it->second);
        }
      }
    }
  }

  return {std::move(file_paths), std::move(selected), extra_cols};
}

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state — constructors
//===----------------------------------------------------------------------===//

iceberg_scan_task_global_state::iceberg_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_iceberg_scan* scan_op,
  size_t approximate_batch_size,
  std::shared_ptr<sirius::io::sirius_ioctx> ioctx)
  : iceberg_scan_task_global_state(
      std::move(pipeline), scan_op, prepare(scan_op), approximate_batch_size, std::move(ioctx))
{
  // Propagate hive partition info to the base class so it can build
  // the partition injection function (same as the public constructor does).
  auto& bind_data = scan_op->bind_data->Cast<duckdb::MultiFileBindData>();
  init_hive_partitions(bind_data, scan_op);
  build_schema_reconciliation(scan_op);
}

iceberg_scan_task_global_state::iceberg_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_iceberg_scan* scan_op,
  init_data init,
  size_t approximate_batch_size,
  std::shared_ptr<sirius::io::sirius_ioctx> ioctx)
  : parquet_scan_task_global_state(std::move(pipeline),
                                   static_cast<sirius_physical_parquet_scan*>(scan_op),
                                   std::move(init.file_paths),
                                   std::move(init.selected_column_indices),
                                   approximate_batch_size,
                                   std::move(ioctx))
{
  build_delete_pipeline(scan_op, init.extra_eq_delete_columns);
}

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state — delete pipeline construction (no I/O)
//===----------------------------------------------------------------------===//

void iceberg_scan_task_global_state::build_delete_pipeline(sirius_physical_iceberg_scan* scan_op,
                                                           size_t extra_eq_delete_columns)
{
  auto const& dd = scan_op->delete_data;
  if (!dd || dd->empty()) {
    SIRIUS_LOG_DEBUG("[iceberg] No delete data; running as plain parquet scan.");
    return;
  }

  // Iceberg delete-file helpers DO NOT construct sirius_datasource internally
  // — read_positional_delete_file uses DuckDB read_parquet (CPU), and
  // read_equality_delete_file uses cudf::io::datasource::create directly. The
  // ioctx map is therefore not needed here; we still require at least one
  // ioctx be configured so the base parquet_scan_task_global_state's
  // planning-time footer reads can resolve a datasource.
  if (!this->get_ioctx()) {
    throw std::runtime_error(
      "[iceberg] No sirius_ioctx available — "
      "SiriusContext must have configured an IO backend.");
  }

  // -----------------------------------------------------------------------
  // Positional deletes (V2) + Deletion vectors (V3)
  // Data is pre-materialized in IcebergDeleteData::positional_deletes (CPU).
  // -----------------------------------------------------------------------
  if (!dd->positional_deletes.empty()) {
    _delete_pipeline.add_filter(std::make_shared<positional_delete_filter>(dd));
  }

  // -----------------------------------------------------------------------
  // Equality deletes (V2)
  // Hash joins are pre-built per group in IcebergDeleteData (GPU).
  // We resolve column mapping here, preferring field ID matching for schema
  // evolution safety (same field renamed ⇒ still matches by ID).
  // Falls back to name matching when field IDs are unavailable.
  // One filter per group supports heterogeneous delete schemas.
  // -----------------------------------------------------------------------
  if (!dd->equality_delete_groups.empty()) {
    auto const& selected = get_selected_column_indices();

    // Build a field-ID map for the first data file for equality-delete
    // key column matching. Data file schema evolution (missing columns)
    // is handled separately by build_schema_reconciliation().
    std::unordered_map<int32_t, cudf::size_type> data_field_id_to_idx;
    if (num_files() > 0) {
      auto data_id_map = extract_field_id_map(get_file_metadata(0));
      for (cudf::size_type j = 0; j < static_cast<cudf::size_type>(selected.size()); ++j) {
        auto const& col_name = scan_op->names[selected[j]];
        auto it              = data_id_map.find(col_name);
        if (it != data_id_map.end()) { data_field_id_to_idx[it->second] = j; }
      }
    }

    for (size_t gi = 0; gi < dd->equality_delete_groups.size(); ++gi) {
      auto const& group = dd->equality_delete_groups[gi];

      std::vector<cudf::size_type> data_key_indices;
      bool all_found = true;

      for (size_t k = 0; k < group.key_names.size(); ++k) {
        auto const& key_name = group.key_names[k];
        bool found           = false;

        // Prefer field ID matching when both sides have IDs.
        if (k < group.key_field_ids.size() && group.key_field_ids[k].has_value()) {
          auto it = data_field_id_to_idx.find(group.key_field_ids[k].value());
          if (it != data_field_id_to_idx.end()) {
            data_key_indices.push_back(it->second);
            found = true;
            SIRIUS_LOG_DEBUG("[iceberg] Matched equality key '{}' by field ID {}.",
                             key_name,
                             group.key_field_ids[k].value());
          }
        }

        // Fall back to name matching.
        if (!found) {
          for (cudf::size_type j = 0; j < static_cast<cudf::size_type>(selected.size()); ++j) {
            if (scan_op->names[selected[j]] == key_name) {
              data_key_indices.push_back(j);
              found = true;
              break;
            }
          }
        }

        if (!found) {
          SIRIUS_LOG_WARN(
            "[iceberg] Equality-delete key column '{}' not found in scan output — "
            "skipping equality-delete filter group.",
            key_name);
          all_found = false;
          break;
        }
      }

      if (all_found) {
        _delete_pipeline.add_filter(
          std::make_shared<equality_delete_filter>(dd, gi, std::move(data_key_indices)));
      }
    }
  }

  // -----------------------------------------------------------------------
  // Install the composed hook.
  // -----------------------------------------------------------------------
  if (!_delete_pipeline.empty()) {
    _delete_pipeline.set_extra_column_count(extra_eq_delete_columns);
    set_post_convert_fn(_delete_pipeline.build_hook());
  }
}

}  // namespace sirius::op::scan
