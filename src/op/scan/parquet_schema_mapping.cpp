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
#include <op/scan/parquet_schema_mapping.hpp>

namespace sirius::op::scan::detail {

std::vector<std::size_t> leaf_indices_for_column(cudf::io::parquet::FileMetaData const& metadata,
                                                 std::string const& column_name)
{
  std::vector<std::size_t> result;
  if (metadata.row_groups.empty()) { return result; }

  auto const& columns = metadata.row_groups.front().columns;
  for (std::size_t i = 0; i < columns.size(); ++i) {
    auto const& path = columns[i].meta_data.path_in_schema;
    if (!path.empty() && path.front() == column_name) { result.push_back(i); }
  }
  return result;
}

}  // namespace sirius::op::scan::detail
