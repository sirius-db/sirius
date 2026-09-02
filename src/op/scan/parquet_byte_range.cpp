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

#include "op/scan/parquet_byte_range.hpp"

#include "sirius/exception.hpp"

#include <algorithm>
#include <limits>

namespace sirius::op::scan::detail {

std::int64_t row_group_start_offset(cudf::io::parquet::FileMetaData const& metadata,
                                    std::size_t index)
{
  auto const& row_group = metadata.row_groups.at(index);
  auto start            = std::numeric_limits<std::int64_t>::max();
  if (row_group.file_offset.has_value() && *row_group.file_offset > 0) {
    start = std::min(start, *row_group.file_offset);
  }
  if (!row_group.columns.empty()) {
    auto const& first_column = row_group.columns.front().meta_data;
    for (auto const offset : {first_column.data_page_offset,
                              first_column.index_page_offset,
                              first_column.dictionary_page_offset}) {
      if (offset > 0) { start = std::min(start, offset); }
    }
  }
  if (start == std::numeric_limits<std::int64_t>::max()) {
    throw sirius::invalid_input_exception(
      "parquet row group {} has no page or file offset; byte-range ownership would be "
      "undefined",
      index);
  }
  return start;
}

std::vector<cudf::size_type> row_groups_in_byte_range(
  cudf::io::parquet::FileMetaData const& metadata, std::uint64_t start, std::uint64_t length)
{
  std::vector<cudf::size_type> owned;
  if (length == 0) { return owned; }
  auto const end = start + length;
  for (std::size_t i = 0; i < metadata.row_groups.size(); ++i) {
    auto const rg_start = static_cast<std::uint64_t>(row_group_start_offset(metadata, i));
    if (start <= rg_start && rg_start < end) { owned.push_back(static_cast<cudf::size_type>(i)); }
  }
  return owned;
}

}  // namespace sirius::op::scan::detail
