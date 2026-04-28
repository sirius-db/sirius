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

#include <cucascade/data/disk_io_backend.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <io/cucascade_datasource.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_delete_filter.hpp>
#include <op/scan/iceberg_scan_task.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::op::scan {

namespace {

/**
 * @brief Read a positional-delete parquet file and append its records to @p out_map.
 *
 * The file must have schema: { file_path STRING, pos BIGINT }.
 *
 * @param backend Per-GPU cucascade io backend used to construct the stack-local
 *                sirius::io::cucascade_datasource adapter. Approach A
 *                (Plan 05-05): backend resolved by the caller via
 *                SiriusContext::get_gpu_io_backends() (inherited map from
 *                parquet_scan_task_global_state) and passed in explicitly.
 */
void read_positional_delete_file(std::string const& delete_file_path,
                                 std::unordered_map<std::string, std::vector<int64_t>>& out_map,
                                 std::shared_ptr<cucascade::idisk_io_backend> backend)
{
  auto stream = cudf::get_default_stream();

  // IO-06: cucascade-backed datasource for iceberg positional-delete read.
  // Adapter is stack-allocated — must outlive the synchronous read_parquet
  // call (source_info takes a non-owning raw pointer; see research Pitfall 4).
  // Each callsite constructs its own adapter — do not share (Pitfall 5).
  auto const file_size = std::filesystem::file_size(delete_file_path);
  sirius::io::cucascade_datasource ds{
    std::move(backend), std::filesystem::path{delete_file_path}, file_size};

  auto opts   = cudf::io::parquet_reader_options::builder(cudf::io::source_info{&ds}).build();
  auto result = cudf::io::read_parquet(opts, stream);

  if (!result.tbl || result.tbl->num_rows() == 0) { return; }

  auto const num_rows = result.tbl->num_rows();

  if (result.tbl->num_columns() < 2) {
    throw std::runtime_error(
      "[iceberg] positional-delete file must have at least 2 columns (file_path, pos): " +
      delete_file_path);
  }

  auto const& pos_col = result.tbl->get_column(1);
  if (pos_col.type().id() != cudf::type_id::INT64) {
    throw std::runtime_error("[iceberg] positional-delete file 'pos' column is not INT64: " +
                             delete_file_path);
  }

  std::vector<int64_t> host_pos(static_cast<size_t>(num_rows));
  RMM_CUDA_TRY(cudaMemcpy(host_pos.data(),
                          pos_col.view().data<int64_t>(),
                          static_cast<size_t>(num_rows) * sizeof(int64_t),
                          cudaMemcpyDeviceToHost));

  auto const& fp_col_view = result.tbl->get_column(0).view();
  if (fp_col_view.type().id() != cudf::type_id::STRING) {
    throw std::runtime_error("[iceberg] positional-delete file 'file_path' column is not STRING: " +
                             delete_file_path);
  }

  cudf::strings_column_view sv(fp_col_view);

  auto const chars_bytes = sv.chars_size(stream);
  std::vector<char> host_chars(chars_bytes);
  if (chars_bytes > 0) {
    RMM_CUDA_TRY(
      cudaMemcpy(host_chars.data(), sv.chars_begin(stream), chars_bytes, cudaMemcpyDeviceToHost));
  }

  auto const& offsets_col = sv.offsets();
  std::vector<int32_t> host_offsets(static_cast<size_t>(num_rows) + 1);
  RMM_CUDA_TRY(cudaMemcpy(host_offsets.data(),
                          offsets_col.data<int32_t>(),
                          (static_cast<size_t>(num_rows) + 1) * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

  for (cudf::size_type i = 0; i < num_rows; ++i) {
    auto const start = host_offsets[i];
    auto const end   = host_offsets[i + 1];
    std::string file_path(host_chars.data() + start, end - start);
    out_map[file_path].push_back(host_pos[i]);
  }
}

/**
 * @brief Read an equality-delete parquet file and return (table, column_names).
 *
 * @param backend Per-GPU cucascade io backend used to construct the stack-local
 *                sirius::io::cucascade_datasource adapter. Approach A
 *                (Plan 05-05): backend resolved by the caller via
 *                SiriusContext::get_gpu_io_backends() (inherited map from
 *                parquet_scan_task_global_state) and passed in explicitly.
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<std::string>> read_equality_delete_file(
  std::string const& delete_file_path, std::shared_ptr<cucascade::idisk_io_backend> backend)
{
  auto stream = cudf::get_default_stream();

  // IO-06: cucascade-backed datasource for iceberg equality-delete read.
  // Adapter is stack-allocated — must outlive the synchronous read_parquet
  // call (source_info takes a non-owning raw pointer; see research Pitfall 4).
  // Each callsite constructs its own adapter — do not share (Pitfall 5).
  auto const file_size = std::filesystem::file_size(delete_file_path);
  sirius::io::cucascade_datasource ds{
    std::move(backend), std::filesystem::path{delete_file_path}, file_size};

  auto opts   = cudf::io::parquet_reader_options::builder(cudf::io::source_info{&ds}).build();
  auto result = cudf::io::read_parquet(opts, stream);

  if (!result.tbl) {
    throw std::runtime_error("[iceberg] Failed to read equality-delete file: " + delete_file_path);
  }

  std::vector<std::string> col_names;
  col_names.reserve(result.metadata.schema_info.size());
  for (auto const& si : result.metadata.schema_info) {
    col_names.push_back(si.name);
  }

  stream.synchronize();
  SIRIUS_LOG_INFO("[iceberg] read equality-delete: path={} rows={} cols={}",
                  delete_file_path,
                  result.tbl->num_rows(),
                  result.tbl->num_columns());
  return {std::move(result.tbl), std::move(col_names)};
}

}  // anonymous namespace

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

  return {std::move(file_paths), std::move(selected)};
}

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state — constructors
//===----------------------------------------------------------------------===//

iceberg_scan_task_global_state::iceberg_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_iceberg_scan* scan_op,
  size_t approximate_batch_size,
  std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends)
  : iceberg_scan_task_global_state(std::move(pipeline),
                                   scan_op,
                                   prepare(scan_op),
                                   approximate_batch_size,
                                   std::move(gpu_io_backends))
{
  // Propagate hive partition info to the base class so it can build
  // the partition injection function (same as the public constructor does).
  auto& bind_data = scan_op->bind_data->Cast<duckdb::MultiFileBindData>();
  init_hive_partitions(bind_data, scan_op);
}

iceberg_scan_task_global_state::iceberg_scan_task_global_state(
  duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline,
  sirius_physical_iceberg_scan* scan_op,
  init_data init,
  size_t approximate_batch_size,
  std::unordered_map<int, std::shared_ptr<cucascade::idisk_io_backend>> gpu_io_backends)
  : parquet_scan_task_global_state(std::move(pipeline),
                                   static_cast<sirius_physical_parquet_scan*>(scan_op),
                                   std::move(init.file_paths),
                                   std::move(init.selected_column_indices),
                                   approximate_batch_size,
                                   std::move(gpu_io_backends))
{
  build_delete_pipeline(scan_op);
}

//===----------------------------------------------------------------------===//
// iceberg_scan_task_global_state — delete pipeline construction
//===----------------------------------------------------------------------===//

void iceberg_scan_task_global_state::build_delete_pipeline(sirius_physical_iceberg_scan* scan_op)
{
  if (scan_op->positional_delete_files.empty() && scan_op->equality_delete_files.empty()) {
    SIRIUS_LOG_DEBUG("[iceberg] No delete files; running as plain parquet scan.");
    return;
  }

  // Resolve per-GPU cucascade io backend used for stack-local cucascade_datasource
  // adapters in the delete-file helpers (Approach A — iceberg helper signatures
  // extended with backend parameter; see Plan 05-05).
  //
  // The base parquet_scan_task_global_state owns the map (seeded by task_creator
  // from SiriusContext::get_gpu_io_backends() — Plan 05-04 Approach C). Delete
  // files are read here at global-state construction time, so we pick the first
  // available GPU's backend deterministically (same rationale as planning-time
  // metadata reads — research Pitfall 6; reads are metadata-only so context
  // mismatch is correctness-neutral).
  auto const& gpu_io_backends = this->get_gpu_io_backends();
  if (gpu_io_backends.empty()) {
    throw std::runtime_error(
      "[iceberg] No GPU io_backends available for delete-file reads — "
      "SiriusContext must have registered at least one backend.");
  }
  auto iceberg_io_backend = gpu_io_backends.begin()->second;

  // -----------------------------------------------------------------------
  // Positional deletes → positional_delete_filter
  // -----------------------------------------------------------------------
  if (!scan_op->positional_delete_files.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} positional-delete file(s).",
                    scan_op->positional_delete_files.size());

    std::unordered_map<std::string, std::vector<int64_t>> pos_deletes;

    for (auto const& del_path : scan_op->positional_delete_files) {
      SIRIUS_LOG_DEBUG("[iceberg] Reading positional-delete file: {}", del_path);
      read_positional_delete_file(del_path, pos_deletes, iceberg_io_backend);
    }

    for (auto& [path, positions] : pos_deletes) {
      std::sort(positions.begin(), positions.end());
    }

    SIRIUS_LOG_INFO("[iceberg] Loaded positional deletes for {} data file(s).", pos_deletes.size());

    _delete_pipeline.add_filter(std::make_shared<positional_delete_filter>(std::move(pos_deletes)));
  }

  // -----------------------------------------------------------------------
  // Equality deletes → equality_delete_filter
  // -----------------------------------------------------------------------
  if (!scan_op->equality_delete_files.empty()) {
    SIRIUS_LOG_INFO("[iceberg] Loading {} equality-delete file(s).",
                    scan_op->equality_delete_files.size());

    std::vector<cudf::table_view> parts_views;
    std::vector<std::unique_ptr<cudf::table>> parts_owned;
    std::vector<std::string> key_column_names;

    for (auto const& eq_path : scan_op->equality_delete_files) {
      SIRIUS_LOG_DEBUG("[iceberg] Reading equality-delete file: {}", eq_path);
      auto [part, names] = read_equality_delete_file(eq_path, iceberg_io_backend);

      if (key_column_names.empty()) {
        key_column_names = names;
      } else if (key_column_names != names) {
        SIRIUS_LOG_WARN(
          "[iceberg] Equality-delete column names mismatch across files — using first file's "
          "schema.");
      }

      parts_views.push_back(part->view());
      parts_owned.push_back(std::move(part));
    }

    if (parts_views.empty() || key_column_names.empty()) {
      SIRIUS_LOG_WARN("[iceberg] No equality-delete rows found; skipping filter.");
    } else {
      auto stream = cudf::get_default_stream();

      auto all_delete_rows = (parts_views.size() == 1)
                               ? std::make_unique<cudf::table>(parts_views[0], stream)
                               : cudf::concatenate(parts_views, stream);

      std::vector<cudf::size_type> all_key_indices(
        static_cast<size_t>(all_delete_rows->num_columns()));
      std::iota(all_key_indices.begin(), all_key_indices.end(), cudf::size_type{0});

      auto deduped = cudf::distinct(all_delete_rows->view(),
                                    all_key_indices,
                                    cudf::duplicate_keep_option::KEEP_FIRST,
                                    cudf::null_equality::EQUAL,
                                    cudf::nan_equality::ALL_EQUAL,
                                    stream);

      SIRIUS_LOG_INFO("[iceberg] Equality-delete table: {} row(s), {} key column(s).",
                      deduped->num_rows(),
                      key_column_names.size());

      // Map equality-key column names to positions in the cudf table.
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
            "[iceberg] Equality-delete key column '{}' not found in scan output — "
            "skipping equality-delete filter.",
            key_name);
          all_found = false;
          break;
        }
      }

      if (all_found) {
        auto hash_join = std::make_unique<cudf::distinct_hash_join>(
          deduped->view(), cudf::null_equality::EQUAL, 0.5, stream);
        stream.synchronize();

        _delete_pipeline.add_filter(std::make_shared<equality_delete_filter>(
          std::move(deduped), std::move(hash_join), std::move(data_key_indices)));
      }
    }
  }

  // -----------------------------------------------------------------------
  // Install the composed hook.
  // -----------------------------------------------------------------------
  if (!_delete_pipeline.empty()) { set_post_convert_fn(_delete_pipeline.build_hook()); }
}

}  // namespace sirius::op::scan
