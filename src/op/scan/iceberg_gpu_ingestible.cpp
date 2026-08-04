/*
 * Copyright 2026, Sirius Contributors.
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

#include "sirius/exception.hpp"

#include <cudf/table/table.hpp>

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/exception.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_gpu_ingestible.hpp>

#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

namespace sirius::op::scan {

namespace {

//===----------------------------------------------------------------------===//
// iceberg_batch_coalescer
//===----------------------------------------------------------------------===//
/**
 * @brief Wraps the parquet coalescer and suppresses reader-side filter pushdown.
 *
 * Batching is entirely the parquet coalescer's job; the only iceberg-specific decision is
 * that a split whose rows will be position-matched against a delete list must come back from
 * the reader with its rows intact. `disable_filter_pushdown` is the flag the parquet
 * materialize path already honours for that (it exists for FLBA-decimal files), and setting
 * it here also disables the dynamic-filter merge, which would drop rows for the same reason.
 */
class iceberg_batch_coalescer : public batch_coalescer {
 public:
  explicit iceberg_batch_coalescer(std::unique_ptr<batch_coalescer> inner)
    : _inner(std::move(inner))
  {
  }

  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    return suppress_pushdown(_inner->push(std::move(info)));
  }

  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    return suppress_pushdown(_inner->flush());
  }

 private:
  static std::vector<std::unique_ptr<scan_info>> suppress_pushdown(
    std::vector<std::unique_ptr<scan_info>> batches)
  {
    for (auto& batch : batches) {
      auto* split = dynamic_cast<parquet_split_info*>(batch.get());
      if (split == nullptr) {
        throw sirius::internal_exception(
          "[iceberg_gpu_ingestible] parquet coalescer emitted a split that is not a "
          "parquet_split_info; the iceberg path cannot guarantee delete positions for it");
      }
      split->disable_filter_pushdown = true;
    }
    return batches;
  }

  std::unique_ptr<batch_coalescer> _inner;
};

}  // namespace

//===----------------------------------------------------------------------===//
// build_batch_layout
//===----------------------------------------------------------------------===//
std::vector<batch_row_run> build_batch_layout(parquet_split_info const& split)
{
  std::vector<batch_row_run> runs;
  int64_t batch_row_offset = 0;

  for (auto const& slice : split.rg_slices) {
    if (slice.row_group_indices.empty()) { continue; }  // fully-pruned file: contributes no rows
    if (!slice.file_metadata) {
      throw sirius::internal_exception(
        "[iceberg_gpu_ingestible] row-group slice for '" + slice.file_path +
        "' has no footer metadata; row positions for iceberg deletes cannot be derived");
    }

    // File-level first row of each row group, as a prefix sum over the file's row groups —
    // including the ones pruning removed, since delete positions are relative to the file.
    auto const& row_groups = slice.file_metadata->row_groups;
    std::vector<int64_t> first_row_of(row_groups.size() + 1, 0);
    for (std::size_t i = 0; i < row_groups.size(); ++i) {
      first_row_of[i + 1] = first_row_of[i] + row_groups[i].num_rows;
    }

    for (auto const rg_index : slice.row_group_indices) {
      auto const idx = static_cast<std::size_t>(rg_index);
      if (rg_index < 0 || idx >= row_groups.size()) {
        throw sirius::internal_exception("[iceberg_gpu_ingestible] row group index " +
                                         std::to_string(rg_index) + " for '" + slice.file_path +
                                         "' is outside its footer's row-group list");
      }
      auto const num_rows = static_cast<int64_t>(row_groups[idx].num_rows);
      runs.push_back(batch_row_run{slice.file_path, first_row_of[idx], batch_row_offset, num_rows});
      batch_row_offset += num_rows;
    }
  }

  return runs;
}

//===----------------------------------------------------------------------===//
// iceberg_gpu_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<iceberg_gpu_ingestible> make_ingestible(
  std::unique_ptr<iceberg_ingestible_table_info> info)
{
  return std::make_shared<iceberg_gpu_ingestible>(std::move(info));
}

iceberg_gpu_ingestible::iceberg_gpu_ingestible(std::unique_ptr<iceberg_ingestible_table_info> info)
  : parquet_gpu_ingestible(std::move(info))
{
  // The base owns the bind data now; read it back typed, the same way it does.
  auto const& bind = static_cast<iceberg_ingestible_table_info const&>(table_info());
  _delete_data     = bind.delete_data;
  _table_path      = bind.table_path;

  if (!_delete_data) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] no delete data for '" + _table_path +
      "'; the planner must resolve it (or decline the scan) before building the ingestible");
  }

  // Hive-partition assembly happens inline in the parquet materialize step, and on that path
  // the reader's predicate is applied before this class sees the table — which would break the
  // row-position mapping deletes depend on. Iceberg tables do not travel with hive partition
  // indices, so this is a guard on an unreachable shape rather than a limitation in practice.
  if (!bind.partition_indices.empty() && !_delete_data->positional_deletes.empty()) {
    throw duckdb::NotImplementedException(
      "iceberg table '{}' combines hive partition columns with positional deletes, which the "
      "GPU scan path cannot order correctly",
      _table_path);
  }

  if (!_delete_data->positional_deletes.empty()) {
    build_delete_key_map(bind.resolved_file_paths);
    _pipeline.add_filter(std::make_shared<positional_delete_filter>(_delete_data));
    SIRIUS_LOG_DEBUG("[iceberg_gpu_ingestible] '{}': positional deletes for {} data file(s)",
                     _table_path,
                     _delete_data->positional_deletes.size());
  }

  // Equality deletes need their key columns force-projected into the scan; until that is
  // wired, a table carrying them must not reach this class.
  if (!_delete_data->equality_delete_groups.empty()) {
    throw duckdb::NotImplementedException(
      "iceberg table '{}' carries equality deletes, which the GPU scan path does not apply yet",
      _table_path);
  }
}

void iceberg_gpu_ingestible::build_delete_key_map(
  std::vector<std::string> const& resolved_file_paths)
{
  // The delete map is keyed on the data file path as the Iceberg manifest wrote it; the scan
  // reads files at the paths DuckDB's multi-file binder resolved. Those are usually the same
  // string, but "usually" is not good enough here: a key that fails to match simply finds no
  // deletes for that file, and the scan returns deleted rows while looking healthy. So the
  // correspondence is established once, explicitly, and a file whose deletes cannot be
  // attributed to exactly one scanned file declines the whole scan to CPU.
  auto strip_scheme = [](std::string_view path) -> std::string_view {
    constexpr std::string_view kFileScheme = "file://";
    return path.starts_with(kFileScheme) ? path.substr(kFileScheme.size()) : path;
  };

  // One is a suffix of the other on a path-component boundary — the relative/absolute case.
  auto same_file = [&](std::string_view a, std::string_view b) {
    a = strip_scheme(a);
    b = strip_scheme(b);
    if (a == b) { return true; }
    std::string_view longer  = a.size() >= b.size() ? a : b;
    std::string_view shorter = a.size() >= b.size() ? b : a;
    if (shorter.empty() || !longer.ends_with(shorter)) { return false; }
    return longer[longer.size() - shorter.size() - 1] == '/';
  };

  for (auto const& [delete_key, positions] : _delete_data->positional_deletes) {
    if (positions.empty()) { continue; }

    std::string const* match = nullptr;
    for (auto const& resolved : resolved_file_paths) {
      if (!same_file(delete_key, resolved)) { continue; }
      if (match != nullptr) {
        throw duckdb::NotImplementedException(
          "iceberg table '{}': delete file entry '{}' matches more than one scanned data file, so "
          "its deleted rows cannot be attributed",
          _table_path,
          delete_key);
      }
      match = &resolved;
    }

    if (match == nullptr) {
      // The file may simply not be in this scan's file list (a snapshot that no longer
      // references it). That is fine — deletes for unscanned files are irrelevant.
      SIRIUS_LOG_DEBUG(
        "[iceberg_gpu_ingestible] '{}': delete entry '{}' names a file this scan does not read",
        _table_path,
        delete_key);
      continue;
    }
    if (*match != delete_key) { _delete_key_by_scan_path.emplace(*match, delete_key); }
  }
}

std::string const& iceberg_gpu_ingestible::delete_key_for(std::string const& scan_path) const
{
  auto it = _delete_key_by_scan_path.find(scan_path);
  return it == _delete_key_by_scan_path.end() ? scan_path : it->second;
}

std::unique_ptr<batch_coalescer> iceberg_gpu_ingestible::create_batch_coalescer() const
{
  auto inner = parquet_gpu_ingestible::create_batch_coalescer();
  // Only a table with deletes pays for pushdown suppression. An append-only iceberg table is a
  // parquet scan in every respect, and taking reader-side row filtering away from it would be a
  // pure loss — there are no positions to preserve.
  if (_pipeline.empty()) { return inner; }
  return std::make_unique<iceberg_batch_coalescer>(std::move(inner));
}

filtered_table iceberg_gpu_ingestible::materialize_metadata_to_table(
  scan_info const& info,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  auto base = parquet_gpu_ingestible::materialize_metadata_to_table(info, mem_space, stream);

  if (_pipeline.empty()) { return base; }

  // Deletes are position-matched, so they are only valid against a table whose rows are
  // exactly the decoded rows in file order. Anything else means the reader filtered or the
  // partition path assembled, and the mapping below would silently delete the wrong rows.
  if (base.state != filter_state::UNFILTERED) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] '" + _table_path +
      "': decode returned a filtered table, so row positions no longer identify file rows; "
      "reader-side pushdown must be suppressed for iceberg splits carrying deletes");
  }

  auto const& split = static_cast<parquet_split_info const&>(info);
  auto layout       = build_batch_layout(split);
  // Runs are keyed on the path the scan reads; the delete map is keyed on the path the manifest
  // recorded. Translate once, here, using the correspondence established at construction.
  for (auto& run : layout) {
    run.data_file_path = delete_key_for(run.data_file_path);
  }

  auto const expected_rows = std::accumulate(
    layout.begin(), layout.end(), int64_t{0}, [](int64_t acc, batch_row_run const& run) {
      return acc + run.num_rows;
    });
  if (expected_rows != static_cast<int64_t>(base.table.num_rows())) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] '" + _table_path + "': decoded " +
      std::to_string(base.table.num_rows()) + " rows but the split's row groups describe " +
      std::to_string(expected_rows) + "; iceberg delete positions cannot be mapped");
  }

  // Nothing to delete from, and release() on an empty handle would hand back a null table.
  if (expected_rows == 0) { return base; }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto table = base.table.release(stream, mr_ref);
  if (!table) {
    throw sirius::internal_exception("[iceberg_gpu_ingestible] '" + _table_path +
                                     "': decoded table has " + std::to_string(expected_rows) +
                                     " rows but no owned state to filter");
  }

  auto filtered = _pipeline.apply(std::move(table), layout, stream, mr_ref);
  // The state is unchanged: deletes are not the scan's row filter, and
  // post_filter_and_project must still apply the query predicate — after the deletes, which
  // is the order Iceberg requires.
  return filtered_table{owning_table_view{std::move(filtered)}, filter_state::UNFILTERED};
}

}  // namespace sirius::op::scan
