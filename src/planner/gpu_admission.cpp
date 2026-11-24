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

#include <duckdb/common/multi_file/multi_file_list.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/node_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>

#include <algorithm>
#include <exception>

// <linux/fs.h> defines BLOCK_SIZE as a macro and reaches this file through the result collector
// (sirius_context.hpp -> scan manager -> liburing). DuckDB's parquet_crypto.hpp, which
// parquet_reader.hpp includes, declares a constant of that name; hide the macro around it.
#pragma push_macro("BLOCK_SIZE")
#undef BLOCK_SIZE
#include <parquet_reader.hpp>
#pragma pop_macro("BLOCK_SIZE")

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
/// callback, the callback reports nothing, or it throws.
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

/// Only use an already-open footer when it covers the entire scan. Do not open additional
/// files just to size admission; multi-file scans retain their configured width.
std::vector<std::optional<uint64_t>> parquet_leaf_column_widths(const duckdb::FunctionData* bind)
{
  auto const* multi = dynamic_cast<const duckdb::MultiFileBindData*>(bind);
  if (multi == nullptr || !multi->initial_reader || !multi->file_list) { return {}; }
  auto const* reader = dynamic_cast<const duckdb::ParquetReader*>(multi->initial_reader.get());
  if (reader == nullptr) { return {}; }
  auto const* metadata = reader->GetFileMetadata();
  if (metadata == nullptr) { return {}; }
  return detail::parquet_column_widths(
    *metadata,
    multi->names.size(),
    multi->file_list->GetExpandResult() == duckdb::FileExpandResult::SINGLE_FILE);
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

std::vector<std::optional<uint64_t>> parquet_column_widths(
  const duckdb_parquet::FileMetaData& metadata, std::size_t bound_columns, bool single_file)
{
  if (!single_file || metadata.row_groups.empty()) { return {}; }
  auto const num_leaves = metadata.row_groups.front().columns.size();
  // Only flat schemas have one leaf per bound column.
  if (num_leaves == 0 || num_leaves != bound_columns) { return {}; }
  std::vector<uint64_t> bytes(num_leaves, 0), values(num_leaves, 0);
  std::vector<bool> complete(num_leaves, true);
  for (auto const& row_group : metadata.row_groups) {
    if (row_group.columns.size() != num_leaves) { return {}; }
    for (std::size_t c = 0; c < num_leaves; ++c) {
      auto const& chunk = row_group.columns[c];
      auto const& meta  = chunk.meta_data;
      if (row_group.num_rows == 0) { continue; }
      // Encoded sizes (including DELTA and dictionary encodings) cannot describe decoded
      // strings. Require unencoded size statistics for EVERY nonempty row group.
      if (!chunk.__isset.meta_data || meta.type != duckdb_parquet::Type::BYTE_ARRAY ||
          meta.num_values <= 0 || meta.num_values != row_group.num_rows ||
          !meta.__isset.size_statistics ||
          !meta.size_statistics.__isset.unencoded_byte_array_data_bytes ||
          meta.size_statistics.unencoded_byte_array_data_bytes < 0) {
        complete[c] = false;
        continue;
      }
      auto const n    = static_cast<uint64_t>(meta.num_values);
      auto const size = static_cast<uint64_t>(meta.size_statistics.unencoded_byte_array_data_bytes);
      // Reject overflowing metadata rather than turning it into a small admission estimate.
      constexpr auto k_max = std::numeric_limits<uint64_t>::max();
      if (bytes[c] > k_max - size || values[c] > k_max - n) {
        complete[c] = false;
        continue;
      }
      bytes[c] += size;
      values[c] += n;
    }
  }
  std::vector<std::optional<uint64_t>> widths(num_leaves);
  for (std::size_t c = 0; c < num_leaves; ++c) {
    if (!complete[c] || values[c] == 0) { continue; }
    // Ceiling average of decoded chars plus cuDF's four-byte offset per row. Zero-byte
    // statistics are valid for empty/all-null strings and still need offsets.
    auto const chars = bytes[c] / values[c] + (bytes[c] % values[c] != 0);
    widths[c]        = accumulate_scan_bytes(chars, 1, 4);
  }
  return widths;
}

}  // namespace detail

op::scan::scan_admission_hint derive_scan_admission_hint(duckdb::ClientContext& context,
                                                         const op::sirius_physical_table_scan& scan)
{
  op::scan::scan_admission_hint hint;
  hint.unfiltered_rows = table_function_row_count(context, scan);
  hint.column_widths.assign(scan.types.size(), std::nullopt);

  auto const leaf_widths = parquet_leaf_column_widths(scan.bind_data.get());
  for (std::size_t i = 0; i < scan.types.size(); ++i) {
    if (scan.types[i].is_fixed_width()) { continue; }
    auto const column = table_column_of_output(scan, i);
    if (!column.has_value()) { continue; }

    if (*column < leaf_widths.size() && leaf_widths[*column].has_value()) {
      hint.column_widths[i] = leaf_widths[*column];
      continue;
    }
    if (scan.function.statistics == nullptr || scan.bind_data == nullptr ||
        !scan.types[i].is_varchar()) {
      continue;
    }
    try {
      auto stats = scan.function.statistics(context, scan.bind_data.get(), *column);
      if (stats && duckdb::StringStats::HasMaxStringLength(*stats)) {
        // Upper bound: every string fits, plus cuDF's 4-byte offset per row.
        hint.column_widths[i] =
          static_cast<uint64_t>(duckdb::StringStats::MaxStringLength(*stats)) + 4;
      }
    } catch (std::exception const& e) {
      SIRIUS_LOG_DEBUG("gpu_admission: statistics callback of {} threw for column {}: {}",
                       scan.function.GetName().GetIdentifierName(),
                       *column,
                       e.what());
    }
  }
  return hint;
}

}  // namespace sirius::planner
