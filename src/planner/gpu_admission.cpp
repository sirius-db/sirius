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
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "transparent/parquet_footer_statistics.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/node_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

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
/// callback, the callback reports nothing or zero rows, or it throws.
///
/// A Parquet scan answers here too, although the plan Sirius converts is a copy whose bind data
/// has no open file (`copy_logical_plan`): `ParquetReadBindData::Copy` keeps the first file's row
/// count, and `ParquetMultiFileInfo::GetCardinality` returns it for one file and, for several,
/// the sum of the footers DuckDB has cached, or else an estimate from the first file (its rows
/// per byte times the files' sizes, or its row count times the number of files, at least 1000 per
/// file). That is an estimate of the unfiltered rows, so it is the fallback for a Parquet scan
/// the footer pass did not cover.
///
/// Zero is not a count to size from: it is what a callback returns for a provably empty scan and
/// what it can return when it has nothing to count, and admission treats a scan of zero rows as
/// unsized either way (total_scan_bytes).
std::optional<uint64_t> table_function_row_count(duckdb::ClientContext& context,
                                                 const op::sirius_physical_table_scan& scan)
{
  if (scan.function.cardinality == nullptr || scan.bind_data == nullptr) { return std::nullopt; }
  try {
    auto stats = scan.function.cardinality(context, scan.bind_data.get());
    if (!stats) { return std::nullopt; }
    auto const rows = detail::reported_row_count(*stats);
    if (rows.value_or(0) == 0) { return std::nullopt; }
    return rows;
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

std::optional<uint64_t> table_function_max_string_bytes(duckdb::ClientContext& context,
                                                        const duckdb::BoundTableFunction& function,
                                                        const duckdb::FunctionData* bind_data,
                                                        duckdb::column_t column)
{
  if (bind_data == nullptr) { return std::nullopt; }
  if (function.statistics_extended == nullptr && function.statistics == nullptr) {
    return std::nullopt;
  }
  try {
    duckdb::unique_ptr<duckdb::BaseStatistics> stats;
    if (function.statistics_extended != nullptr) {
      duckdb::TableFunctionGetStatisticsInput input(bind_data, duckdb::ColumnIndex(column));
      stats = function.statistics_extended(context, input);
    } else {
      stats = function.statistics(context, bind_data, column);
    }
    // The string accessors read the statistics' string payload unchecked, so the kind is checked
    // here: a column of another type has numeric or nested statistics.
    if (stats && stats->GetStatsType() == duckdb::StatisticsType::STRING_STATS &&
        duckdb::StringStats::HasMaxStringLength(*stats)) {
      return static_cast<uint64_t>(duckdb::StringStats::MaxStringLength(*stats));
    }
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG("gpu_admission: statistics callback of {} threw for column {}: {}",
                     function.GetName().GetIdentifierName(),
                     column,
                     e.what());
  }
  return std::nullopt;
}

}  // namespace detail

op::scan::scan_admission_hint derive_scan_admission_hint(duckdb::ClientContext& context,
                                                         const op::sirius_physical_table_scan& scan,
                                                         uint64_t avg_var_bytes)
{
  op::scan::scan_admission_hint hint;
  // A Parquet scan's files were counted from their footers before DuckDB optimized the plan,
  // where the footer statistics pass covered the scan: `parquet_footer_statistics` on and no more
  // uncached footers than its read limits allow. That count is exact. Any other Parquet scan has
  // the callback's estimate (see table_function_row_count).
  auto const function_name = scan.function.GetName().GetIdentifierName();
  auto const footer_rows = transparent::parquet_scan_row_count(function_name, scan.bind_data.get());
  hint.unfiltered_rows   = footer_rows.has_value() ? std::optional<uint64_t>{*footer_rows}
                                                   : table_function_row_count(context, scan);
  hint.column_widths.assign(scan.types.size(), std::nullopt);

  // String sizes come from the same footer pass, summed over every file of the scan; they exist
  // only where the row count does.
  auto const footer_sizes =
    transparent::parquet_scan_string_sizes(function_name, scan.bind_data.get());
  // Bytes per row of table column `column`, read as `type`, where a source states one.
  auto const width_of = [&](duckdb::column_t column,
                            sirius::logical_type const& type) -> std::optional<uint64_t> {
    if (type.is_fixed_width()) { return std::nullopt; }
    if (column < footer_sizes.size() && footer_rows.has_value()) {
      // The rows of the files and row groups that state no size keep the configured average.
      if (auto const width = blended_string_width(
            footer_sizes[column].bytes, footer_sizes[column].rows, *footer_rows, avg_var_bytes)) {
        return width;
      }
    }
    if (!type.is_varchar()) { return std::nullopt; }
    // Upper bound: every string fits, plus cuDF's 4-byte offset per row.
    auto const longest =
      detail::table_function_max_string_bytes(context, scan.function, scan.bind_data.get(), column);
    return longest.has_value() ? std::optional<uint64_t>{*longest + 4} : std::nullopt;
  };
  for (std::size_t i = 0; i < scan.types.size(); ++i) {
    if (auto const column = table_column_of_output(scan, i)) {
      hint.column_widths[i] = width_of(*column, scan.types[i]);
    }
  }
  for (auto const column : filter_only_table_columns(scan)) {
    auto const& type = scan.returned_types[column];
    hint.filter_only_columns.push_back({type, width_of(column, type)});
  }
  // One line per scan, so a run can show which source each estimate came from and, when the
  // unfiltered row count is missing, whether the callback or the bind data was absent.
  SIRIUS_LOG_DEBUG(
    "gpu_admission: scan of {} (cardinality callback {}, bind data {}): unfiltered rows {} ({}), "
    "{} of {} output and {} of {} filter-only column width(s) known",
    function_name,
    scan.function.cardinality != nullptr ? "present" : "absent",
    scan.bind_data != nullptr ? "present" : "absent",
    hint.unfiltered_rows.has_value() ? std::to_string(*hint.unfiltered_rows) : "unknown",
    footer_rows.has_value()            ? "from Parquet footers"
    : hint.unfiltered_rows.has_value() ? "from the cardinality callback"
                                       : "no source",
    std::ranges::count_if(hint.column_widths, [](auto const& width) { return width.has_value(); }),
    hint.column_widths.size(),
    std::ranges::count_if(hint.filter_only_columns,
                          [](auto const& column) { return column.width.has_value(); }),
    hint.filter_only_columns.size());
  return hint;
}

bool admission_estimate_needed(duckdb::ClientContext& context)
{
  if (!context.registered_state) { return false; }
  auto const sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  // No GPU memory spaces exist before the runtime is initialized, and no query runs on it.
  if (!sirius_ctx || !sirius_ctx->is_initialized()) { return false; }
  auto const bytes_per_gpu = duckdb::query_operator_options(context)->admission_bytes_per_gpu;
  // Off is the common case: answer it before looking at the GPUs.
  if (bytes_per_gpu == 0) { return false; }
  // The same candidates sirius_engine admits from: one per GPU memory space, capped by
  // `topology.gpus_per_query`.
  std::vector<int> gpu_ids;
  for (auto const* space :
       sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
    if (space != nullptr) { gpu_ids.push_back(space->get_device_id()); }
  }
  std::ranges::sort(gpu_ids);
  gpu_ids.erase(std::unique(gpu_ids.begin(), gpu_ids.end()), gpu_ids.end());
  auto const n_gpus =
    apply_gpu_cap(std::move(gpu_ids), sirius_ctx->get_config().gpus_per_query()).size();
  return admission_estimate_needed(
    bytes_per_gpu, n_gpus, sirius::log::get_sink()->should_log(sirius::log::level::debug));
}

}  // namespace sirius::planner
