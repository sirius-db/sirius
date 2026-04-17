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
#include <op/scan/scan_source_resolver.hpp>
// detail::make_selected_column_indices lives in parquet_scan_task.{hpp,cpp}.
#include <op/scan/parquet_scan_task.hpp>

// standard library
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

parquet_scan_source_resolver::parquet_scan_source_resolver(
  std::vector<std::string> file_paths,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids,
  duckdb::vector<std::string> const& names)
  : _file_paths(std::move(file_paths)),
    _column_ids(column_ids),
    _projection_ids(projection_ids),
    _names(names)
{
}

scan_source_resolver::resolved parquet_scan_source_resolver::resolve()
{
  bool const is_projected = !_projection_ids.empty();
  if (is_projected && _names.empty()) {
    throw std::runtime_error(
      "[parquet_scan_source_resolver] Projection requires column names to be provided.");
  }

  resolved out;
  out.file_paths              = std::move(_file_paths);
  out.selected_column_indices = detail::make_selected_column_indices(_column_ids, _projection_ids);

  if (is_projected) {
    out.projected_column_names.reserve(out.selected_column_indices.size());
    for (auto idx : out.selected_column_indices) {
      out.projected_column_names.push_back(_names[idx]);
    }
  }

  // Plain parquet: no per-batch transform, no trailing columns to strip.
  out.per_batch_transform       = nullptr;
  out.trailing_columns_to_strip = 0;

  return out;
}

}  // namespace sirius::op::scan
