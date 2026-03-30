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

#include <op/scan/iceberg_delete_filter.hpp>

#include <cudf/table/table.hpp>

#include <numeric>

namespace sirius::op::scan {

void iceberg_delete_pipeline::add_filter(std::shared_ptr<iceberg_delete_filter> filter)
{
  _filters.push_back(std::move(filter));
}

post_convert_fn_t iceberg_delete_pipeline::build_hook() const
{
  auto filters = _filters;  // shared_ptr copies — safe to capture
  auto extra   = _extra_column_count;

  return [filters = std::move(filters), extra](std::unique_ptr<cudf::table> tbl,
                                               std::string const& path,
                                               int64_t first_row,
                                               rmm::cuda_stream_view stream)
           -> std::unique_ptr<cudf::table> {
    // Apply each filter in order.
    for (auto const& f : filters) {
      tbl = f->apply(std::move(tbl), path, first_row, stream);
    }

    // Strip force-projected extra columns (appended at the end for equality keys).
    if (extra > 0) {
      auto const total = tbl->num_columns();
      auto const keep  = static_cast<cudf::size_type>(total) - static_cast<cudf::size_type>(extra);
      if (keep > 0 && keep < total) {
        std::vector<cudf::size_type> indices(static_cast<size_t>(keep));
        std::iota(indices.begin(), indices.end(), cudf::size_type{0});
        // Release columns from the original table, take the subset, build a new table.
        // This avoids a deep copy — select() on a table_view just creates views,
        // and the new cudf::table deep-copies only the selected columns.
        tbl = std::make_unique<cudf::table>(tbl->select(indices), stream);
      }
    }

    return tbl;
  };
}

}  // namespace sirius::op::scan
