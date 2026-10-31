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

#pragma once

#include "helper/type_conversions.hpp"

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/exception.hpp>

namespace sirius::planner {

// Preserve primary column indices and supported row-count carriers. Unsupported
// types may remain in the bound schema only when the scan does not read them.
inline duckdb::vector<logical_type> scan_returned_types(
  duckdb::vector<duckdb::LogicalType> const& types,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids)
{
  duckdb::vector<bool> required(types.size(), false);
  for (auto const& column : column_ids) {
    if (column.IsVirtualColumn()) { continue; }
    auto const primary = column.GetPrimaryIndex();
    if (primary >= types.size()) {
      throw duckdb::InternalException("Scan column index exceeds the bound schema");
    }
    required[primary] = true;
  }
  duckdb::vector<logical_type> result;
  result.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    try {
      result.push_back(sirius::from_duckdb(types[i]));
    } catch (duckdb::InvalidInputException const&) {
      if (required[i]) { throw; }
      result.push_back(logical_type::make(type_id::INVALID));
    }
  }
  return result;
}

}  // namespace sirius::planner
