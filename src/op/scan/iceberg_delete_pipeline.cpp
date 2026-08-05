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

#include <cudf/table/table.hpp>

#include <op/scan/iceberg_delete_filter.hpp>

namespace sirius::op::scan {

void iceberg_delete_pipeline::add_filter(std::shared_ptr<iceberg_delete_filter> filter)
{
  _filters.push_back(std::move(filter));
}

std::unique_ptr<cudf::table> iceberg_delete_pipeline::apply(std::unique_ptr<cudf::table> tbl,
                                                            batch_layout layout,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  for (auto const& f : _filters) {
    tbl = f->apply(std::move(tbl), layout, stream, mr);
  }

  // Strip force-projected extra columns (equality keys appended at the end).
  if (_extra_column_count > 0) {
    auto const total = tbl->num_columns();
    auto const keep =
      static_cast<cudf::size_type>(total) - static_cast<cudf::size_type>(_extra_column_count);
    if (keep > 0 && keep < total) {
      // Move columns out and truncate — no GPU memory copy.
      auto cols = tbl->release();
      cols.resize(static_cast<size_t>(keep));
      tbl = std::make_unique<cudf::table>(std::move(cols));
    }
  }

  return tbl;
}

}  // namespace sirius::op::scan
