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
#include "ducklake/preparation_identity.hpp"
#include "sirius_config.hpp"

#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/table_filter_set.hpp>

#include <optional>
#include <stdexcept>
#include <vector>
namespace sirius::op::scan {
struct physical_profile_table;
}
namespace sirius::ducklake {
struct preparation_gate;
class preparation_slot;
// Filters and output positions index the selected required_columns, not the
// original table or physical file. Omitted output means all selected columns;
// an empty output retains only a row carrier after filtering.
struct unit_scan_options {
  size_t scan_task_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE;
  duckdb::unique_ptr<duckdb::TableFilterSet> filters;
  std::optional<std::vector<size_t>> output_columns;
  // A whole-scan binding supplies common immutable policy and unique credit keys.
  std::shared_ptr<op::scan::physical_profile_table> profiles;
  std::shared_ptr<preparation_identity> identities;
  std::shared_ptr<preparation_gate> inline_gate;
  std::shared_ptr<preparation_slot> parquet_slot;
  bool owns_admission = true;
};

inline void validate_scan_options(unit_scan_options const& options, size_t width)
{
  if (options.output_columns)
    for (auto column : *options.output_columns)
      if (column >= width)
        throw std::invalid_argument("DuckLake output column is outside the selected working set");
  if (!options.filters) return;
  for (auto const& entry : *options.filters)
    if (entry.GetIndex().GetIndex() >= width)
      throw std::invalid_argument("DuckLake filter column is outside the selected working set");
  for (auto const& entry : options.filters->GetMultiColumnFilters()) {
    auto const& filter =
      duckdb::ExpressionFilter::GetExpressionFilter(*entry, "DuckLake filter projection");
    for (auto column : filter.column_indexes)
      if (column.GetIndex() >= width)
        throw std::invalid_argument("DuckLake filter column is outside the selected working set");
  }
}
}  // namespace sirius::ducklake
