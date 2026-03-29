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
#include <op/scan/iceberg_scan_task.hpp>
#include <log/logging.hpp>

// duckdb
#include <duckdb/common/multi_file/multi_file_states.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

// cuda
#include <cuda_runtime_api.h>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state helpers
//===----------------------------------------------------------------------===//

namespace {

/**
 * @brief Read a positional-delete parquet file and append its records to @p out_map.
 *
 * The file must have schema: { file_path STRING, pos BIGINT }.
 * For each (file_path, pos) pair found, appends @p pos to out_map[file_path].
 * Callers are responsible for sorting after all files have been processed.
 */
void read_positional_delete_file(
  std::string const& delete_file_path,
  std::unordered_map<std::string, std::vector<int64_t>>& out_map)
{
  auto stream = cudf::get_default_stream();

  auto opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{delete_file_path}).build();
  auto result = cudf::io::read_parquet(opts, stream);

  if (!result.tbl || result.tbl->num_rows() == 0) { return; }

  auto const num_rows = result.tbl->num_rows();

  // -----------------------------------------------------------------------
  // Extract the pos column (INT64) — assumed to be column 1
  // -----------------------------------------------------------------------
  if (result.tbl->num_columns() < 2) {
    throw std::runtime_error(
      "[iceberg_scan_task] positional-delete file must have at least 2 columns (file_path, pos): " +
      delete_file_path);
  }

  auto const& pos_col = result.tbl->get_column(1);
  if (pos_col.type().id() != cudf::type_id::INT64) {
    throw std::runtime_error(
      "[iceberg_scan_task] positional-delete file 'pos' column is not INT64: " + delete_file_path);
  }

  std::vector<int64_t> host_pos(num_rows);
  cudaMemcpy(host_pos.data(),
             pos_col.view().data<int64_t>(),
             num_rows * sizeof(int64_t),
             cudaMemcpyDeviceToHost);

  // -----------------------------------------------------------------------
  // Extract the file_path column (STRING) — assumed to be column 0
  // -----------------------------------------------------------------------
  auto const& fp_col_view = result.tbl->get_column(0).view();
  if (fp_col_view.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error(
      "[iceberg_scan_task] positional-delete file 'file_path' column is not STRING: " +
      delete_file_path);
  }

  cudf::strings_column_view sv(fp_col_view);

  // Copy chars buffer to host
  auto const chars_bytes = sv.chars_size(stream);
  std::vector<char> host_chars(chars_bytes);
  if (chars_bytes > 0) {
    cudaMemcpy(host_chars.data(), sv.chars_begin(stream), chars_bytes, cudaMemcpyDeviceToHost);
  }

  // Copy offsets to host (INT32 offsets)
  auto const& offsets_col = sv.offsets();
  std::vector<int32_t> host_offsets(num_rows + 1);
  cudaMemcpy(host_offsets.data(),
             offsets_col.data<int32_t>(),
             (num_rows + 1) * sizeof(int32_t),
             cudaMemcpyDeviceToHost);

  // -----------------------------------------------------------------------
  // Populate the output map
  // -----------------------------------------------------------------------
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    auto const start = host_offsets[i];
    auto const end   = host_offsets[i + 1];
    std::string file_path(host_chars.data() + start, end - start);

    out_map[file_path].push_back(host_pos[i]);
  }
}

/**
 * @brief Build the post-convert hook for iceberg V2 positional deletes.
 *
 * The hook captures the delete state by shared_ptr and is safe to call
 * concurrently from multiple task threads.
 */
post_convert_fn_t make_positional_delete_hook(
  std::shared_ptr<iceberg_delete_state> delete_state)
{
  return [state = std::move(delete_state)](std::unique_ptr<cudf::table> tbl,
                                           std::string const& data_file_path,
                                           int64_t first_row,
                                           rmm::cuda_stream_view stream)
           -> std::unique_ptr<cudf::table> {
    auto it = state->positional_deletes.find(data_file_path);
    if (it == state->positional_deletes.end() || it->second.empty()) {
      return tbl;  // No deletes recorded for this file
    }

    auto const& delete_positions = it->second;
    auto const num_rows          = static_cast<int64_t>(tbl->num_rows());
    auto const last_row          = first_row + num_rows;

    // Find the sub-range of delete positions that fall within this batch.
    auto lo = std::lower_bound(delete_positions.begin(), delete_positions.end(), first_row);
    auto hi = std::lower_bound(lo, delete_positions.end(), last_row);

    if (lo == hi) { return tbl; }  // Nothing to delete in this batch

    // Build a keep-mask on the host: true = keep row, false = deleted.
    std::vector<uint8_t> keep(num_rows, 1u);
    for (auto it2 = lo; it2 != hi; ++it2) {
      keep[static_cast<size_t>(*it2 - first_row)] = 0u;
    }

    // Copy to GPU as a BOOL8 column.
    auto bool_col =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                    static_cast<cudf::size_type>(num_rows),
                                    cudf::mask_state::UNALLOCATED,
                                    stream);
    cudaMemcpyAsync(bool_col->mutable_view().data<uint8_t>(),
                    keep.data(),
                    num_rows * sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    stream.value());

    return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream);
  };
}

// ---------------------------------------------------------------------------
// Equality delete helpers
// ---------------------------------------------------------------------------

/**
 * @brief Read an equality-delete parquet file and return (table, column_names).
 *
 * The returned table contains only the equality-key columns.
 * Column names are extracted from the parquet schema metadata.
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<std::string>>
read_equality_delete_file(std::string const& delete_file_path)
{
  auto stream = cudf::get_default_stream();
  auto opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{delete_file_path}).build();
  auto result = cudf::io::read_parquet(opts, stream);

  if (!result.tbl) {
    throw std::runtime_error(
      "[iceberg_scan_task] Failed to read equality-delete file: " + delete_file_path);
  }

  // Extract column names from schema info
  std::vector<std::string> col_names;
  col_names.reserve(result.metadata.schema_info.size());
  for (auto const& si : result.metadata.schema_info) {
    col_names.push_back(si.name);
  }

  stream.synchronize();
  SIRIUS_LOG_INFO("[read_equality_delete_file] path={} rows={} cols={} schema_info_size={}",
                  delete_file_path, result.tbl->num_rows(), result.tbl->num_columns(),
                  result.metadata.schema_info.size());
  for (auto const& si : result.metadata.schema_info) {
    SIRIUS_LOG_INFO("[read_equality_delete_file]   col={}", si.name);
  }
  return {std::move(result.tbl), std::move(col_names)};
}

/**
 * @brief Chain two post-convert hooks into one that applies them in sequence.
 */
post_convert_fn_t chain_hooks(post_convert_fn_t first, post_convert_fn_t second)
{
  return [f1 = std::move(first), f2 = std::move(second)](std::unique_ptr<cudf::table> tbl,
                                                          std::string const& path,
                                                          int64_t first_row,
                                                          rmm::cuda_stream_view stream)
           -> std::unique_ptr<cudf::table> {
    tbl = f1(std::move(tbl), path, first_row, stream);
    return f2(std::move(tbl), path, first_row, stream);
  };
}

/**
 * @brief Build a post-convert hook for Iceberg V2 equality deletes.
 *
 * The hook probes the pre-built distinct_hash_join with each data-chunk's
 * key columns, retrieves the per-row match result (build_indices) via a
 * device-to-host copy, builds a boolean keep-mask, and applies
 * cudf::apply_boolean_mask.
 */
post_convert_fn_t make_equality_delete_hook(
  std::shared_ptr<iceberg_equality_delete_state> eq_state)
{
  return [state = std::move(eq_state)](std::unique_ptr<cudf::table> tbl,
                                       std::string const& /*path*/,
                                       int64_t /*first_row*/,
                                       rmm::cuda_stream_view stream)
           -> std::unique_ptr<cudf::table> {
    auto const n_rows = tbl->num_rows();
    SIRIUS_LOG_INFO("[equality_delete_hook] chunk: {} rows, {} cols, key_indices={}", n_rows,
                    tbl->num_columns(), state->data_key_indices.size());
    if (n_rows == 0) return tbl;

    // Verify all key columns are present in this chunk.  DuckDB can push down
    // column projections (e.g. for count(*)) that exclude equality-key columns.
    // When that happens the delete cannot be applied — return the original chunk.
    // This gives incorrect results for count(*)-style queries on tables with
    // equality deletes, but it avoids a crash; a full fix requires projecting
    // the key columns unconditionally at scan time.
    for (auto idx : state->data_key_indices) {
      if (idx >= static_cast<cudf::size_type>(tbl->num_columns())) { return tbl; }
    }

    // Project data chunk to the equality key columns
    auto data_key_view = tbl->select(state->data_key_indices);

    // Probe the hash join: build_indices[i] = matching row in delete table,
    // or JoinNoMatch if no match.  Keep rows where build_indices[i] == JoinNoMatch.
    auto build_indices = state->hash_join->left_join(data_key_view, stream);

    // Sync so the device→host copy sees the completed result
    stream.synchronize();

    // Download build_indices and build a keep-mask on the host
    // build_indices->size() should equal n_rows for distinct left_join
    auto const bi_size = static_cast<cudf::size_type>(build_indices->size());
    auto const copy_n  = std::min(n_rows, bi_size);
    std::vector<cudf::size_type> host_indices(n_rows, cudf::JoinNoMatch);
    cudaMemcpy(host_indices.data(),
               build_indices->data(),
               copy_n * sizeof(cudf::size_type),
               cudaMemcpyDeviceToHost);

    std::vector<uint8_t> keep(n_rows);
    bool any_deleted = false;
    for (cudf::size_type i = 0; i < n_rows; ++i) {
      bool matched = (host_indices[i] != cudf::JoinNoMatch);
      keep[i]      = matched ? uint8_t{0} : uint8_t{1};
      if (matched) any_deleted = true;
    }

    if (!any_deleted) {
      SIRIUS_LOG_INFO("[equality_delete_hook] no rows deleted");
      return tbl;
    }
    SIRIUS_LOG_INFO("[equality_delete_hook] deleting rows from chunk");

    // Upload the boolean mask to the GPU and filter
    auto bool_col =
      cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                    n_rows,
                                    cudf::mask_state::UNALLOCATED,
                                    stream);
    cudaMemcpyAsync(bool_col->mutable_view().data<uint8_t>(),
                    keep.data(),
                    n_rows * sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    stream.value());

    return cudf::apply_boolean_mask(tbl->view(), bool_col->view(), stream);
  };
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state — public constructor
//===----------------------------------------------------------------------===//

iceberg_scan_task_global_state::init_data
iceberg_scan_task_global_state::prepare(sirius_physical_iceberg_scan* scan_op)
{
  // Extract data file paths from the multi-file bind data.
  auto& bind_data = scan_op->bind_data->Cast<duckdb::MultiFileBindData>();
  if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
    throw std::runtime_error("[iceberg_scan_task_global_state] No input data files to scan");
  }

  auto files = bind_data.file_list->GetAllFiles();
  std::vector<std::string> file_paths;
  file_paths.reserve(files.size());
  for (auto const& f : files) {
    file_paths.push_back(f.path);
  }

  // Compute column projection indices (same logic as plain parquet scan).
  auto selected = detail::make_selected_column_indices(*scan_op);

  return {std::move(file_paths), std::move(selected)};
}

// Public constructor delegates to the private constructor via a two-stage
// pattern that lets us compute init_data before the base-class initialiser.
iceberg_scan_task_global_state::iceberg_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_iceberg_scan* scan_op,
  size_t approximate_batch_size)
  : iceberg_scan_task_global_state(std::move(pipeline),
                                   scan_op,
                                   prepare(scan_op),
                                   approximate_batch_size)
{
}

// Private delegating constructor: receives pre-computed init_data so that the
// protected base constructor can be invoked in the member initialiser list.
iceberg_scan_task_global_state::iceberg_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_iceberg_scan* scan_op,
  init_data init,
  size_t approximate_batch_size)
  : parquet_scan_task_global_state(std::move(pipeline),
                                   static_cast<sirius_physical_parquet_scan*>(scan_op),
                                   std::move(init.file_paths),
                                   std::move(init.selected_column_indices),
                                   approximate_batch_size)
{
  build_delete_state(scan_op);
}

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state — delete state construction
//===----------------------------------------------------------------------===//

void iceberg_scan_task_global_state::build_delete_state(sirius_physical_iceberg_scan* scan_op)
{
  // V1 tables and V2 tables with no delete files: nothing to do.
  if (scan_op->positional_delete_files.empty() && scan_op->equality_delete_files.empty()) {
    SIRIUS_LOG_DEBUG(
      "[iceberg_scan_task_global_state] No delete files; running as plain parquet scan.");
    return;
  }

  // -------------------------------------------------------------------------
  // V2 positional deletes
  // -------------------------------------------------------------------------
  post_convert_fn_t hook;

  if (!scan_op->positional_delete_files.empty()) {
    SIRIUS_LOG_INFO("[iceberg_scan_task_global_state] Loading {} positional-delete file(s).",
                    scan_op->positional_delete_files.size());

    _delete_state = std::make_shared<iceberg_delete_state>();

    for (auto const& del_path : scan_op->positional_delete_files) {
      SIRIUS_LOG_DEBUG("[iceberg_scan_task_global_state] Reading positional-delete file: {}",
                       del_path);
      read_positional_delete_file(del_path, _delete_state->positional_deletes);
    }

    for (auto& [path, positions] : _delete_state->positional_deletes) {
      std::sort(positions.begin(), positions.end());
    }

    SIRIUS_LOG_INFO(
      "[iceberg_scan_task_global_state] Loaded positional deletes for {} data file(s).",
      _delete_state->positional_deletes.size());

    hook = make_positional_delete_hook(_delete_state);
  }

  // -------------------------------------------------------------------------
  // V2 equality deletes
  // -------------------------------------------------------------------------
  if (!scan_op->equality_delete_files.empty()) {
    SIRIUS_LOG_INFO("[iceberg_scan_task_global_state] Loading {} equality-delete file(s).",
                    scan_op->equality_delete_files.size());

    // Read all equality-delete files and collect their rows.
    std::vector<cudf::table_view> parts_views;
    std::vector<std::unique_ptr<cudf::table>> parts_owned;
    std::vector<std::string> key_column_names;  // from the first file

    for (auto const& eq_path : scan_op->equality_delete_files) {
      SIRIUS_LOG_DEBUG("[iceberg_scan_task_global_state] Reading equality-delete file: {}",
                       eq_path);
      auto [part, names] = read_equality_delete_file(eq_path);

      if (key_column_names.empty()) {
        key_column_names = names;
      } else if (key_column_names != names) {
        SIRIUS_LOG_WARN(
          "[iceberg_scan_task_global_state] Equality-delete column names mismatch "
          "across files — using first file's schema.");
      }

      parts_views.push_back(part->view());
      parts_owned.push_back(std::move(part));
    }

    if (parts_views.empty() || key_column_names.empty()) {
      SIRIUS_LOG_WARN(
        "[iceberg_scan_task_global_state] No equality-delete rows found; skipping hook.");
    } else {
      // Concatenate all delete rows and deduplicate.
      auto stream = cudf::get_default_stream();
      auto all_delete_rows =
        (parts_views.size() == 1)
          ? std::make_unique<cudf::table>(parts_views[0], stream)
          : cudf::concatenate(parts_views, stream);

      // Build explicit key indices for all columns (empty vector means "no keys" in cudf,
      // not "all keys", so we must enumerate them explicitly).
      std::vector<cudf::size_type> all_key_indices;
      all_key_indices.resize(static_cast<std::size_t>(all_delete_rows->num_columns()));
      std::iota(all_key_indices.begin(), all_key_indices.end(), cudf::size_type{0});

      SIRIUS_LOG_DEBUG(
        "[iceberg_scan_task_global_state] Pre-distinct equality-delete rows: {}",
        all_delete_rows->num_rows());

      auto deduped_delete_rows = cudf::distinct(all_delete_rows->view(),
                                                all_key_indices,
                                                cudf::duplicate_keep_option::KEEP_FIRST,
                                                cudf::null_equality::EQUAL,
                                                cudf::nan_equality::ALL_EQUAL,
                                                stream);

      SIRIUS_LOG_INFO(
        "[iceberg_scan_task_global_state] Equality-delete table: {} row(s), {} key column(s).",
        deduped_delete_rows->num_rows(),
        key_column_names.size());

      // Map equality-key column names to positions in the cudf table.
      //
      // The cudf table produced by the parquet reader has columns in the order of
      // get_selected_column_indices().  Each entry selected[j] is an index into
      // scan_op->names, so the j-th column of the cudf table is scan_op->names[selected[j]].
      // We therefore search for the key name in that "names-indexed-by-selected" view.
      auto const& selected = get_selected_column_indices();
      std::vector<cudf::size_type> data_key_indices;
      bool all_found = true;
      for (auto const& key_name : key_column_names) {
        bool found = false;
        for (cudf::size_type j = 0; j < static_cast<cudf::size_type>(selected.size()); ++j) {
          if (scan_op->names[selected[j]] == key_name) {
            data_key_indices.push_back(j);
            found = true;
            break;
          }
        }
        if (!found) {
          SIRIUS_LOG_WARN(
            "[iceberg_scan_task_global_state] Equality-delete key column '{}' not found in "
            "scan output schema — skipping equality-delete hook.",
            key_name);
          all_found = false;
          break;
        }
      }

      if (all_found) {
        // Build the distinct hash join (build side = equality-delete key table).
        auto eq_state           = std::make_shared<iceberg_equality_delete_state>();
        eq_state->data_key_indices = std::move(data_key_indices);
        eq_state->delete_key_table = std::move(deduped_delete_rows);
        eq_state->hash_join = std::make_unique<cudf::distinct_hash_join>(
          eq_state->delete_key_table->view(), cudf::null_equality::EQUAL, 0.5, stream);
        stream.synchronize();

        _equality_delete_state = std::move(eq_state);

        auto eq_hook = make_equality_delete_hook(_equality_delete_state);
        hook         = hook ? chain_hooks(std::move(hook), std::move(eq_hook)) : std::move(eq_hook);
      }
    }
  }

  // Install whatever hook(s) were built.
  if (hook) { set_post_convert_fn(std::move(hook)); }
}

}  // namespace sirius::op::scan
