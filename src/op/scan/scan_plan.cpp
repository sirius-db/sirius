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
#include <helper/type_conversions.hpp>
#include <log/logging.hpp>
#include <op/scan/scan_plan.hpp>
#include <sirius/exception.hpp>

// duckdb
#include <duckdb/common/hive_partitioning.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>

// standard library
#include <unordered_map>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// scan_plan accessors
//===----------------------------------------------------------------------===//

bool scan_plan::is_projected() const
{
  // Driven by the @c needs_reader_projection flag set in build_scan_plan. We
  // cannot derive this from @c data_columns alone: the factory populates
  // data_columns even for the plain "read everything" case, so checking
  // !data_columns.empty() would treat SELECT * as projected and spuriously
  // trigger set_column_names / projected_columns_are_flat / per-file name
  // resolution — silently regressing support for nested-schema SELECT *.
  return needs_reader_projection;
}

std::vector<std::string> scan_plan::data_column_names() const
{
  std::vector<std::string> names;
  names.reserve(data_columns.size());
  for (auto const& c : data_columns) {
    names.push_back(c.name);
  }
  return names;
}

std::vector<duckdb::idx_t> scan_plan::make_batch_column_map() const
{
  // This is a shim for the existing @c convert_table_filters_to_expression API,
  // which uses @c idx_t(-1) as a "not in batch" sentinel. The canonical form
  // is the @c vector<optional<size_t>> already stored in
  // @c batch_position_by_column_id; once the filter-translator API is updated
  // to consume the optional form directly, this conversion (and the sentinel)
  // can be deleted.
  constexpr auto NOT_PROJECTED = static_cast<duckdb::idx_t>(-1);
  std::vector<duckdb::idx_t> map(batch_position_by_column_id.size(), NOT_PROJECTED);
  for (std::size_t c = 0; c < batch_position_by_column_id.size(); ++c) {
    if (batch_position_by_column_id[c]) {
      map[c] = static_cast<duckdb::idx_t>(*batch_position_by_column_id[c]);
    }
  }
  return map;
}

std::string scan_plan::batch_column_name(duckdb::idx_t batch_position) const
{
  return data_columns.at(batch_position).name;
}

std::unordered_set<std::size_t> scan_plan::pure_filter_batch_positions() const
{
  std::unordered_set<std::size_t> positions;
  for (std::size_t d = 0; d < data_columns.size(); ++d) {
    positions.insert(d);
  }
  for (auto const& entry : output_layout) {
    if (entry.source == output_entry::DATA) { positions.erase(entry.idx); }
  }
  return positions;
}

//===----------------------------------------------------------------------===//
// scan_plan::build_inject_fn
//===----------------------------------------------------------------------===//

partition_inject_fn_t scan_plan::build_inject_fn() const
{
  // If there are no output columns (e.g. SELECT count(*), with or without a
  // filter that pulls in pure-filter data columns), the reader's natural batch
  // is what downstream needs: count-style aggregations propagate row counts
  // from the batch they receive. Projecting it down to a 0-column table would
  // erase the row count. Output partitions are only recorded when they appear
  // in the output, so an empty output_layout also implies no partitions to
  // inject — nothing to do, return a null closure.
  if (output_layout.empty()) { return nullptr; }

  // If assembly is a trivial identity — no partitions and output_layout covers
  // data_columns 1:1 in order — return a null closure so the GPU scan operator
  // skips the assembly step and the reader's output flows through untouched.
  bool identity = !has_partitions() && output_layout.size() == data_columns.size();
  for (std::size_t i = 0; identity && i < output_layout.size(); ++i) {
    if (output_layout[i].source != output_entry::DATA || output_layout[i].idx != i) {
      identity = false;
    }
  }
  if (identity) { return nullptr; }

  // Capture everything the closure needs by value, so it survives moves of
  // the scan_plan.
  auto layout     = output_layout;
  auto partitions = partition_columns;

  return [layout = std::move(layout), partitions = std::move(partitions)](
           std::unique_ptr<cudf::table> tbl,
           std::string const& file_path,
           rmm::cuda_stream_view stream) -> std::unique_ptr<cudf::table> {
    if (!tbl) return tbl;

    // Parse once; the closure is called per scan task (per file).
    auto path_parts     = duckdb::HivePartitioning::Parse(file_path);
    auto const num_rows = tbl->num_rows();
    auto data_cols      = tbl->release();  // move columns out, no GPU copy

    std::vector<std::unique_ptr<cudf::column>> out_cols;
    out_cols.reserve(layout.size());

    for (auto const& entry : layout) {
      if (entry.source == scan_plan::output_entry::DATA) {
        out_cols.push_back(std::move(data_cols.at(entry.idx)));
      } else {
        auto const& pcol = partitions.at(entry.idx);
        auto it          = path_parts.find(pcol.name);
        if (it == path_parts.end()) {
          throw sirius::internal_exception(
            "[scan_plan::inject] missing hive partition key '{}' in file path: {}",
            pcol.name,
            file_path);
        }
        auto duckdb_val = duckdb::Value(it->second).DefaultCastAs(sirius::to_duckdb(pcol.type));
        auto scalar     = sirius::value_to_cudf_scalar(duckdb_val, pcol.type, stream);
        out_cols.push_back(cudf::make_column_from_scalar(*scalar, num_rows, stream));
      }
    }

    // Unused data columns (pure-filter columns not in output_layout) fall out
    // of scope with data_cols and are freed.
    return std::make_unique<cudf::table>(std::move(out_cols));
  };
}

//===----------------------------------------------------------------------===//
// build_scan_plan — factory
//===----------------------------------------------------------------------===//
namespace {

bool is_output_position(std::size_t i, std::size_t output_types_size)
{
  return i < output_types_size;
}

}  // namespace

scan_plan build_scan_plan(duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                          duckdb::vector<duckdb::idx_t> const& projection_ids,
                          duckdb::vector<std::string> const& names,
                          duckdb::vector<sirius::logical_type> const& returned_types,
                          std::size_t output_types_size,
                          duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices)
{
  scan_plan plan;

  // Register partition primary indices up-front so the filter-expression builder
  // can drop filters on partition columns regardless of whether they appear in
  // projection_ids.
  for (auto const& hpi : partition_indices) {
    plan.partition_primary_indices.insert(hpi.index);
  }

  // Reader projection is needed whenever the planner pruned / reordered columns
  // (non-empty projection_ids) or hive-partition columns must be stripped from
  // the physical read. When both are false, column_ids already describes the
  // natural read order and the reader's unset-projection output matches what
  // the pipeline expects — matching the old @c _is_projected semantics.
  plan.needs_reader_projection = !projection_ids.empty() || !partition_indices.empty();

  // Walk positions in output-first order. When projection_ids is non-empty the
  // first output_types_size entries are the output columns in output order;
  // the remaining entries are pure-filter columns that must be read but not
  // emitted. When projection_ids is empty, column_ids is both the read list
  // and the output (no pure-filter columns).
  //
  // In both cases we translate each walked position into either a data_column
  // (copy-from-batch), a partition_column (inject-from-path), or nothing
  // (virtual / duplicate / filter-only partition).
  std::unordered_set<std::size_t> seen_primary_indices;
  std::unordered_map<std::size_t, std::size_t> primary_to_batch;  // P → D

  auto handle_position = [&](std::size_t column_ids_pos, bool is_output) {
    auto const primary_idx = column_ids.at(column_ids_pos).GetPrimaryIndex();
    if (duckdb::IsVirtualColumn(primary_idx)) { return; }
    if (!seen_primary_indices.insert(primary_idx).second) { return; }

    bool const is_partition = plan.partition_primary_indices.count(primary_idx) > 0;

    if (is_partition) {
      // Filter-only partition columns are dropped: DuckDB prunes at the file
      // level and our filter builder will skip them. We only materialize
      // partition metadata for output columns.
      if (!is_output) { return; }
      // names.at / returned_types.at guard against planner inputs where the
      // primary index exceeds the schema (empty names, schema mismatch, etc.).
      plan.partition_columns.push_back(scan_plan::partition_column{
        primary_idx, names.at(primary_idx), returned_types.at(primary_idx)});
      plan.output_layout.push_back(scan_plan::output_entry{scan_plan::output_entry::PARTITION,
                                                           plan.partition_columns.size() - 1});
    } else {
      // Data column — always added to the batch (even if filter-only, we need
      // it for filter evaluation). Store an empty name when @c names is empty:
      // the caller's guard only forces non-empty names for name-dependent paths
      // (projection, filter, partitions), and the plain-read case populates
      // data_columns without ever consuming the name downstream.
      std::size_t const batch_pos = plan.data_columns.size();
      std::string col_name        = names.empty() ? std::string{} : names.at(primary_idx);
      plan.data_columns.push_back(scan_plan::data_column{primary_idx, std::move(col_name)});
      primary_to_batch[primary_idx] = batch_pos;
      if (is_output) {
        plan.output_layout.push_back(
          scan_plan::output_entry{scan_plan::output_entry::DATA, batch_pos});
      }
    }
  };

  if (projection_ids.empty()) {
    // No projection: iterate column_ids in natural order; every entry is output.
    for (std::size_t c = 0; c < column_ids.size(); ++c) {
      handle_position(c, /* is_output */ true);
    }
  } else {
    for (std::size_t i = 0; i < projection_ids.size(); ++i) {
      handle_position(projection_ids[i], is_output_position(i, output_types_size));
    }
  }

  // Build the C → D map. An entry is nullopt when the column is a hive
  // partition, virtual, or simply not referenced by projection_ids.
  plan.batch_position_by_column_id.assign(column_ids.size(), std::nullopt);
  for (std::size_t c = 0; c < column_ids.size(); ++c) {
    auto const primary_idx = column_ids[c].GetPrimaryIndex();
    if (duckdb::IsVirtualColumn(primary_idx)) { continue; }
    auto it = primary_to_batch.find(primary_idx);
    if (it == primary_to_batch.end()) { continue; }
    plan.batch_position_by_column_id[c] = it->second;
  }

  SIRIUS_LOG_DEBUG("[scan_plan] built plan: {} data col(s), {} partition col(s), {} output entries",
                   plan.data_columns.size(),
                   plan.partition_columns.size(),
                   plan.output_layout.size());

  return plan;
}

}  // namespace sirius::op::scan
