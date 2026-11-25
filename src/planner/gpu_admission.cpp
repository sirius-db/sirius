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

#include "planner/gpu_admission.hpp"

#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "transparent/parquet_footer_statistics.hpp"

#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/node_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>

#include <algorithm>
#include <exception>
#include <string>

namespace sirius::planner {

void for_each_gpu_scan(const op::sirius_physical_operator& node,
                       absl::FunctionRef<void(const op::sirius_physical_operator&)> visit)
{
  if (node.type == op::SiriusPhysicalOperatorType::GPU_SCAN) { visit(node); }

  for (auto& child : node.children) {
    if (child) { for_each_gpu_scan(*child, visit); }
  }

  if (node.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    auto& rc = node.Cast<op::sirius_physical_result_collector>();
    for_each_gpu_scan(rc.plan, visit);
  }
}

void collect_gpu_scans(const op::sirius_physical_operator& node,
                       std::vector<const op::sirius_physical_operator*>& out)
{
  for_each_gpu_scan(node,
                    [&out](const op::sirius_physical_operator& scan) { out.push_back(&scan); });
}

namespace {

/// The table function's own row count, before filters; nullopt when it has no cardinality
/// callback, the callback reports nothing, or it throws. A Parquet scan's callback reports nothing
/// here: it reads the count off the scan's first open file, and the plan Sirius converts is a
/// copy whose bind data has no open file (`copy_logical_plan`).
std::optional<uint64_t> table_function_row_count(duckdb::ClientContext& context,
                                                 const op::sirius_physical_table_scan& scan)
{
  if (scan.function.cardinality == nullptr || scan.bind_data == nullptr) { return std::nullopt; }
  try {
    auto stats = scan.function.cardinality(context, scan.bind_data.get());
    if (!stats) { return std::nullopt; }
    return detail::reported_row_count(*stats);
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG("gpu_admission: cardinality callback of {} threw: {}",
                     scan.function.GetName().GetIdentifierName(),
                     e.what());
  }
  return std::nullopt;
}

/// The table column behind output column @p output_idx of @p scan, or nullopt for virtual
/// columns (row id, file name) and out-of-range indices.
std::optional<duckdb::column_t> table_column_of_output(const op::sirius_physical_table_scan& scan,
                                                       std::size_t output_idx)
{
  if (!scan.projection_ids.empty() && output_idx >= scan.projection_ids.size()) {
    return std::nullopt;
  }
  auto const idx = scan.projection_ids.empty() ? output_idx : scan.projection_ids[output_idx];
  if (idx >= scan.column_ids.size()) { return std::nullopt; }
  auto const& column = scan.column_ids[idx];
  if (column.IsVirtualColumn()) { return std::nullopt; }
  return column.GetPrimaryIndex();
}

/// The table columns the scan reads only to evaluate pushed-down filters: the projection entries
/// the planner appended behind the scan's own output (`projected_output_count`). Empty when the
/// scan has no explicit projection, since it then outputs every column it reads.
std::vector<duckdb::column_t> filter_only_table_columns(const op::sirius_physical_table_scan& scan)
{
  std::vector<duckdb::column_t> columns;
  // A scan built without the planner's count of its output columns cannot tell the two apart.
  if (scan.projected_output_count != scan.types.size()) { return columns; }
  for (auto i = scan.projected_output_count; i < scan.projection_ids.size(); ++i) {
    auto const idx = scan.projection_ids[i];
    if (idx >= scan.column_ids.size() || scan.column_ids[idx].IsVirtualColumn()) { continue; }
    auto const column = scan.column_ids[idx].GetPrimaryIndex();
    if (column < scan.returned_types.size()) { columns.push_back(column); }
  }
  return columns;
}

}  // namespace

namespace detail {

std::optional<uint64_t> reported_row_count(const duckdb::NodeStatistics& stats)
{
  std::optional<uint64_t> rows;
  if (stats.has_max_cardinality) { rows = stats.max_cardinality; }
  if (stats.has_estimated_cardinality) {
    rows = std::max(rows.value_or(0), static_cast<uint64_t>(stats.estimated_cardinality));
  }
  return rows;
}

}  // namespace detail

op::scan::scan_admission_hint derive_scan_admission_hint(duckdb::ClientContext& context,
                                                         const op::sirius_physical_table_scan& scan)
{
  op::scan::scan_admission_hint hint;
  // A Parquet scan's files were counted from their footers before DuckDB optimized the plan,
  // where the footer statistics pass covered the scan: `parquet_footer_statistics` on and no more
  // uncached footers than its read limits allow. That count is exact; the callback's is the first
  // file's times the number of files, and is absent for a copied plan.
  auto const footer_rows = transparent::parquet_scan_row_count(
    scan.function.GetName().GetIdentifierName(), scan.bind_data.get());
  hint.unfiltered_rows = footer_rows.has_value() ? std::optional<uint64_t>{*footer_rows}
                                                 : table_function_row_count(context, scan);
  hint.column_widths.assign(scan.types.size(), std::nullopt);

  // String sizes come from the same footer pass, summed over every file of the scan.
  auto const leaf_widths = transparent::parquet_scan_string_widths(
    scan.function.GetName().GetIdentifierName(), scan.bind_data.get());
  // Bytes per row of table column `column`, read as `type`, where a source states one.
  auto const width_of = [&](duckdb::column_t column,
                            sirius::logical_type const& type) -> std::optional<uint64_t> {
    if (type.is_fixed_width()) { return std::nullopt; }
    if (column < leaf_widths.size() && leaf_widths[column].has_value()) {
      return leaf_widths[column];
    }
    if (scan.function.statistics == nullptr || scan.bind_data == nullptr || !type.is_varchar()) {
      return std::nullopt;
    }
    try {
      auto stats = scan.function.statistics(context, scan.bind_data.get(), column);
      if (stats && duckdb::StringStats::HasMaxStringLength(*stats)) {
        // Upper bound: every string fits, plus cuDF's 4-byte offset per row.
        return static_cast<uint64_t>(duckdb::StringStats::MaxStringLength(*stats)) + 4;
      }
    } catch (std::exception const& e) {
      SIRIUS_LOG_DEBUG("gpu_admission: statistics callback of {} threw for column {}: {}",
                       scan.function.GetName().GetIdentifierName(),
                       column,
                       e.what());
    }
    return std::nullopt;
  };
  for (std::size_t i = 0; i < scan.types.size(); ++i) {
    if (auto const column = table_column_of_output(scan, i)) {
      hint.column_widths[i] = width_of(*column, scan.types[i]);
    }
  }
  for (auto const column : filter_only_table_columns(scan)) {
    hint.filter_only_types.push_back(scan.returned_types[column]);
    hint.filter_only_widths.push_back(width_of(column, scan.returned_types[column]));
  }
  // One line per scan, so a run can show which source each estimate came from and, when the
  // unfiltered row count is missing, whether the callback or the bind data was absent.
  auto const known = [](auto const& width) { return width.has_value(); };
  SIRIUS_LOG_DEBUG(
    "gpu_admission: scan of {} (cardinality callback {}, bind data {}): unfiltered rows {} ({}), "
    "{} of {} output and {} of {} filter-only column width(s) known",
    scan.function.GetName().GetIdentifierName(),
    scan.function.cardinality != nullptr ? "present" : "absent",
    scan.bind_data != nullptr ? "present" : "absent",
    hint.unfiltered_rows.has_value() ? std::to_string(*hint.unfiltered_rows) : "unknown",
    footer_rows.has_value()            ? "from Parquet footers"
    : hint.unfiltered_rows.has_value() ? "from the cardinality callback"
                                       : "no source",
    std::count_if(hint.column_widths.begin(), hint.column_widths.end(), known),
    hint.column_widths.size(),
    std::count_if(hint.filter_only_widths.begin(), hint.filter_only_widths.end(), known),
    hint.filter_only_widths.size());
  return hint;
}

}  // namespace sirius::planner
