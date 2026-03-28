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
#include <cudf/io/parquet.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

// cuda
#include <cuda_runtime_api.h>

// rmm
#include <rmm/cuda_stream_view.hpp>

// standard library
#include <algorithm>
#include <cstring>
#include <memory>
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
  // Equality deletes are not yet implemented.
  if (!scan_op->equality_delete_files.empty()) {
    throw std::runtime_error(
      "[iceberg_scan_task_global_state] Equality deletes are not yet supported. "
      "Found " +
      std::to_string(scan_op->equality_delete_files.size()) + " equality-delete file(s).");
  }

  // V1 tables and V2 tables with no delete files: nothing to do.
  if (scan_op->positional_delete_files.empty()) {
    SIRIUS_LOG_DEBUG(
      "[iceberg_scan_task_global_state] No delete files; running as plain parquet scan.");
    return;
  }

  // V2 positional deletes.
  SIRIUS_LOG_INFO("[iceberg_scan_task_global_state] Loading {} positional-delete file(s).",
                  scan_op->positional_delete_files.size());

  _delete_state = std::make_shared<iceberg_delete_state>();

  for (auto const& del_path : scan_op->positional_delete_files) {
    SIRIUS_LOG_DEBUG("[iceberg_scan_task_global_state] Reading positional-delete file: {}",
                     del_path);
    read_positional_delete_file(del_path, _delete_state->positional_deletes);
  }

  // Sort each file's delete positions so that binary-search in the hook is correct.
  for (auto& [path, positions] : _delete_state->positional_deletes) {
    std::sort(positions.begin(), positions.end());
  }

  SIRIUS_LOG_INFO("[iceberg_scan_task_global_state] Loaded positional deletes for {} data file(s).",
                  _delete_state->positional_deletes.size());

  // Install the post-convert hook.
  set_post_convert_fn(make_positional_delete_hook(_delete_state));
}

}  // namespace sirius::op::scan
