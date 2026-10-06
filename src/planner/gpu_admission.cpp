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
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_table_scan.hpp"

#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/node_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <parquet_reader.hpp>

#include <algorithm>
#include <exception>

namespace sirius::planner {

void collect_gpu_scans(const op::sirius_physical_operator& node,
                       std::vector<const op::sirius_physical_operator*>& out)
{
  if (node.type == op::SiriusPhysicalOperatorType::GPU_SCAN) { out.push_back(&node); }

  for (auto& child : node.children) {
    if (child) { collect_gpu_scans(*child, out); }
  }

  if (node.type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      node.type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = node.Cast<op::sirius_physical_delim_join>();
    if (delim.join) { collect_gpu_scans(*delim.join, out); }
    if (delim.distinct_root) { collect_gpu_scans(*delim.distinct_root, out); }
  }
  if (node.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    auto& rc = node.Cast<op::sirius_physical_result_collector>();
    collect_gpu_scans(rc.plan, out);
  }
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
    // max_cardinality includes rows appended in this transaction; prefer it.
    if (stats->has_max_cardinality) { return static_cast<uint64_t>(stats->max_cardinality); }
    if (stats->has_estimated_cardinality) {
      return static_cast<uint64_t>(stats->estimated_cardinality);
    }
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG(
      "gpu_admission: cardinality callback of {} threw: {}", scan.function.name, e.what());
  }
  return std::nullopt;
}

/// Average uncompressed bytes per value of each leaf column of the first Parquet file behind a
/// read_parquet / parquet_scan bind, for columns stored verbatim. Empty when the bind is not a
/// multi-file Parquet bind, no file has been opened, the schema is nested (leaf columns no longer
/// line up with table columns), or a row group disagrees on the column count.
std::vector<std::optional<uint64_t>> parquet_leaf_column_widths(const duckdb::FunctionData* bind)
{
  auto const* multi = dynamic_cast<const duckdb::MultiFileBindData*>(bind);
  if (multi == nullptr || !multi->initial_reader) { return {}; }
  auto const* reader = dynamic_cast<const duckdb::ParquetReader*>(multi->initial_reader.get());
  if (reader == nullptr) { return {}; }
  auto const* metadata = reader->GetFileMetadata();
  if (metadata == nullptr || metadata->row_groups.empty()) { return {}; }

  auto const num_leaves = metadata->row_groups.front().columns.size();
  // Flat schema only: one leaf per bound column. Nested types add leaves the table column
  // indices do not account for.
  if (num_leaves == 0 || num_leaves != multi->names.size()) { return {}; }

  std::vector<uint64_t> bytes(num_leaves, 0);
  std::vector<uint64_t> values(num_leaves, 0);
  std::vector<bool> dictionary(num_leaves, false);
  for (auto const& row_group : metadata->row_groups) {
    if (row_group.columns.size() != num_leaves) { return {}; }
    for (std::size_t c = 0; c < num_leaves; ++c) {
      auto const& meta = row_group.columns[c].meta_data;
      if (meta.num_values > 0 && meta.total_uncompressed_size > 0) {
        bytes[c] += static_cast<uint64_t>(meta.total_uncompressed_size);
        values[c] += static_cast<uint64_t>(meta.num_values);
      }
      if (meta.__isset.dictionary_page_offset) { dictionary[c] = true; }
      for (auto const encoding : meta.encodings) {
        if (encoding == duckdb_parquet::Encoding::PLAIN_DICTIONARY ||
            encoding == duckdb_parquet::Encoding::RLE_DICTIONARY) {
          dictionary[c] = true;
        }
      }
    }
  }

  std::vector<std::optional<uint64_t>> widths(num_leaves);
  for (std::size_t c = 0; c < num_leaves; ++c) {
    if (dictionary[c] || values[c] == 0) { continue; }
    widths[c] = std::max<uint64_t>(1, (bytes[c] + values[c] - 1) / values[c]);
  }
  return widths;
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
                       scan.function.name,
                       *column,
                       e.what());
    }
  }
  return hint;
}

}  // namespace sirius::planner
