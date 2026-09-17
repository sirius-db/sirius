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
#include <io/uri_parser.hpp>
#include <log/logging.hpp>
#include <op/scan/iceberg_gpu_ingestible.hpp>

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

/// Wraps the parquet coalescer and suppresses reader-side filter pushdown: a split whose rows
/// get position-matched against a delete list must come back with its rows intact. The flag also
/// disables the dynamic-filter merge, which would drop rows for the same reason.
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

std::shared_ptr<iceberg_gpu_ingestible> make_ingestible(
  std::unique_ptr<iceberg_ingestible_table_info> info)
{
  return std::make_shared<iceberg_gpu_ingestible>(std::move(info));
}

iceberg_gpu_ingestible::iceberg_gpu_ingestible(std::unique_ptr<iceberg_ingestible_table_info> info)
  : parquet_gpu_ingestible(std::move(info))
{
  auto const& bind = static_cast<iceberg_ingestible_table_info const&>(table_info());
  _delete_data     = bind.delete_data;
  _table_path      = bind.table_path;

  if (!_delete_data) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] no delete data for '" + _table_path +
      "'; the planner must resolve it (or decline the scan) before building the ingestible");
  }

  // On the hive-partition path the reader's predicate is applied before this class sees the
  // table, breaking the row-position mapping. Unreachable for iceberg; guarded anyway.
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

  if (!_delete_data->equality_delete_groups.empty()) {
    throw duckdb::NotImplementedException(
      "iceberg table '{}' carries equality deletes, which the GPU scan path does not apply yet",
      _table_path);
  }
}

void iceberg_gpu_ingestible::build_delete_key_map(
  std::vector<std::string> const& resolved_file_paths)
{
  // The delete map is keyed on the path the manifest wrote; the scan reads the path DuckDB's
  // binder resolved. A key that fails to match finds no deletes and returns deleted rows while
  // looking healthy, so the correspondence is established once here and ambiguity declines the
  // scan. Two paths name one file when one is a suffix of the other on a component boundary.
  auto same_file = [](std::string_view a, std::string_view b) {
    if (a == b) { return true; }
    std::string_view longer  = a.size() >= b.size() ? a : b;
    std::string_view shorter = a.size() >= b.size() ? b : a;
    if (shorter.empty() || !longer.ends_with(shorter)) { return false; }
    return longer[longer.size() - shorter.size() - 1] == '/';
  };

  auto basename = [](std::string_view path) {
    auto const slash = path.find_last_of('/');
    return slash == std::string_view::npos ? path : path.substr(slash + 1);
  };

  // Matching implies an identical final path component, so indexing the scanned paths by that
  // component finds every candidate without comparing against all of them. Comparing each key
  // against every file would be O(keys × files) — quadratic in the number of data files, on the
  // planning thread — and would re-strip both paths on every one of those comparisons.
  std::vector<std::string> stripped;  // owns the views held by the index below
  stripped.reserve(resolved_file_paths.size());
  for (auto const& resolved : resolved_file_paths) {
    stripped.push_back(sirius::io::strip_file_scheme(resolved));
  }
  std::unordered_map<std::string_view, std::vector<std::size_t>> scanned_by_basename;
  for (std::size_t i = 0; i < stripped.size(); ++i) {
    scanned_by_basename[basename(stripped[i])].push_back(i);
  }

  // Both manifest-keyed maps need translating: a table with only equality deletes has no
  // positional entries, so positional_deletes alone would leave this empty.
  std::vector<std::string const*> manifest_keys;
  manifest_keys.reserve(_delete_data->positional_deletes.size() +
                        _delete_data->data_file_manifest_sequence_numbers.size());
  for (auto const& [delete_key, positions] : _delete_data->positional_deletes) {
    if (!positions.empty()) { manifest_keys.push_back(&delete_key); }
  }
  for (auto const& entry : _delete_data->data_file_manifest_sequence_numbers) {
    manifest_keys.push_back(&entry.first);
  }

  // Collects EVERY matching key, including one that already spells the file exactly as the scan
  // does. Skipping that one early would leave a second, differently-spelled entry looking like the
  // only claimant (a bare v2 path vs a `file://` v3 one), so the guard below would not fire and
  // the other entry's deleted rows would come back.
  std::unordered_map<std::string, std::vector<std::string const*>> keys_by_scan_path;

  for (auto const* delete_key_ptr : manifest_keys) {
    auto const& delete_key = *delete_key_ptr;

    auto const key_stripped  = sirius::io::strip_file_scheme(delete_key);
    std::string const* match = nullptr;
    if (auto const bucket = scanned_by_basename.find(basename(key_stripped));
        bucket != scanned_by_basename.end()) {
      for (auto const index : bucket->second) {
        if (!same_file(key_stripped, stripped[index])) { continue; }
        if (match != nullptr) {
          throw duckdb::NotImplementedException(
            "iceberg table '{}': delete file entry '{}' matches more than one scanned data file, "
            "so its deleted rows cannot be attributed",
            _table_path,
            delete_key);
        }
        match = &resolved_file_paths[index];
      }
    }

    if (match == nullptr) {
      // Not in this scan's file list — deletes for unscanned files are irrelevant.
      SIRIUS_LOG_DEBUG(
        "[iceberg_gpu_ingestible] '{}': delete entry '{}' names a file this scan does not read",
        _table_path,
        delete_key);
      continue;
    }
    keys_by_scan_path[*match].push_back(&delete_key);
  }

  for (auto const& [scan_path, keys] : keys_by_scan_path) {
    if (keys.size() > 1) {
      throw duckdb::NotImplementedException(
        "iceberg table '{}': scanned data file '{}' is named by {} different manifest entries "
        "(including '{}' and '{}'), so its deletes cannot be attributed to one of them",
        _table_path,
        scan_path,
        keys.size(),
        *keys[0],
        *keys[1]);
    }
    if (*keys[0] == scan_path) { continue; }
    _delete_key_by_scan_path.emplace(scan_path, *keys[0]);
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
  // An append-only table has no positions to preserve, so it keeps reader-side filtering.
  if (_pipeline.empty()) { return inner; }
  return std::make_unique<iceberg_batch_coalescer>(std::move(inner));
}

filtered_table iceberg_gpu_ingestible::materialize_metadata_to_table(
  scan_info const& info,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream,
  bool like_swar_fastpath,
  std::shared_ptr<const sirius::like_multiliteral_cache> like_cache)
{
  // Forwarded untouched: these steer the base decode's LIKE evaluation and have nothing to say
  // about deletes. The check below is what guards the case that matters -- if they cause the
  // reader to filter rows, positions stop identifying file rows and we refuse rather than
  // delete the wrong ones.
  auto base = parquet_gpu_ingestible::materialize_metadata_to_table(
    info, mem_space, stream, like_swar_fastpath, std::move(like_cache));

  if (_pipeline.empty()) { return base; }

  // Position-matched deletes require the decoded rows in file order; anything else means the
  // reader filtered, and the mapping below would delete the wrong rows.
  if (base.state != filter_state::UNFILTERED) {
    throw sirius::internal_exception(
      "[iceberg_gpu_ingestible] '" + _table_path +
      "': decode returned a filtered table, so row positions no longer identify file rows; "
      "reader-side pushdown must be suppressed for iceberg splits carrying deletes");
  }

  auto const& split = static_cast<parquet_split_info const&>(info);
  auto layout       = build_batch_layout(split);
  // Runs carry the scan's path; the delete map is keyed on the manifest's.
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

  // release() on an empty handle would hand back a null table.
  if (expected_rows == 0) { return base; }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto table = base.table.release(stream, mr_ref);
  if (!table) {
    throw sirius::internal_exception("[iceberg_gpu_ingestible] '" + _table_path +
                                     "': decoded table has " + std::to_string(expected_rows) +
                                     " rows but no owned state to filter");
  }

  auto filtered = _pipeline.apply(std::move(table), layout, stream, mr_ref);
  // State is unchanged: deletes are not the query predicate, which post_filter_and_project must
  // still apply — after the deletes, as Iceberg requires.
  return filtered_table{owning_table_view{std::move(filtered)}, filter_state::UNFILTERED};
}

}  // namespace sirius::op::scan
