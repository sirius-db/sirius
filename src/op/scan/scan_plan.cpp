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
  // trigger set_column_names / per-file name resolution — silently regressing
  // support for nested-schema SELECT *.
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
// needs_output_assembly / assemble_scan_output
//===----------------------------------------------------------------------===//

bool needs_output_assembly(scan_plan const& plan)
{
  // SELECT count(*) shape (with or without a filter that pulls in pure-filter
  // data columns): the reader's natural batch is what downstream needs —
  // count-style aggregations propagate row counts from the batch they receive,
  // so projecting down to a 0-column table would erase the row count. Output
  // partitions are only recorded when they appear in the output, so an empty
  // output_layout also implies no partitions to inject.
  if (plan.output_layout.empty()) { return false; }

  // Trivial identity: no partitions and output_layout covers data_columns 1:1
  // in order. The reader's natural output already matches what the pipeline
  // expects.
  if (plan.has_partitions() || plan.output_layout.size() != plan.data_columns.size()) {
    return true;
  }
  for (std::size_t i = 0; i < plan.output_layout.size(); ++i) {
    if (plan.output_layout[i].source != scan_plan::output_entry::DATA ||
        plan.output_layout[i].idx != i) {
      return true;
    }
  }
  return false;
}

std::vector<cudf::size_type> output_data_positions(scan_plan const& plan)
{
  std::vector<cudf::size_type> positions;
  positions.reserve(plan.output_layout.size());
  for (auto const& entry : plan.output_layout) {
    if (entry.source == scan_plan::output_entry::DATA) {
      positions.push_back(static_cast<cudf::size_type>(entry.idx));
    }
  }
  return positions;
}

owning_table_view assemble_scan_output(scan_plan const& plan,
                                       owning_table_view&& table,
                                       std::vector<std::string> const& partition_values,
                                       rmm::cuda_stream_view stream)
{
  if (!table) { return std::move(table); }

  // Nothing to reshape (SELECT count(*) — empty output layout). Emitting a
  // 0-column table would erase the row count downstream aggregations consume.
  if (plan.output_layout.empty()) { return std::move(table); }

  // No partition columns: the output is a pure projection / reordering of the
  // reader's data columns. Express it as a non-owning view selection — no GPU
  // copy. Every output_entry is DATA here (PARTITION entries only exist when the
  // plan has partition columns), and entry.idx is the column's position in the
  // current (D-order) view. Pure-filter data columns are dropped by the
  // selection and freed when the view is later materialized.
  if (!plan.has_partitions()) {
    std::vector<std::size_t> positions;
    positions.reserve(plan.output_layout.size());
    for (auto const& entry : plan.output_layout) {
      positions.push_back(entry.idx);
    }
    table.select_columns(positions);
    return std::move(table);
  }

  // Partition columns present: materialize the reader batch and rebuild, moving
  // DATA columns out and synthesizing constant PARTITION columns from the path.
  auto reader_output  = table.release(stream);
  auto const num_rows = reader_output->num_rows();
  auto data_cols      = reader_output->release();  // move columns out, no GPU copy

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(plan.output_layout.size());

  for (auto const& entry : plan.output_layout) {
    if (entry.source == scan_plan::output_entry::DATA) {
      out_cols.push_back(std::move(data_cols.at(entry.idx)));
    } else {
      auto const& pcol = plan.partition_columns.at(entry.idx);
      auto const& pval = partition_values.at(entry.idx);
      // Hive path segments are URL-encoded (DuckDB's own partitioned writer emits
      // e.g. col=a%20b for the value "a b"); decode once before the cast to match
      // DuckDB's CPU value materialization (hive_partitioning.cpp Value(Unescape(...))).
      auto duckdb_val = duckdb::Value(duckdb::HivePartitioning::Unescape(pval))
                          .DefaultCastAs(sirius::to_duckdb(pcol.type));
      auto scalar = sirius::value_to_cudf_scalar(duckdb_val, pcol.type, stream);
      out_cols.push_back(cudf::make_column_from_scalar(*scalar, num_rows, stream));
    }
  }

  // Unused data columns (pure-filter columns not in output_layout) fall out
  // of scope with data_cols and are freed.
  return owning_table_view{std::make_unique<cudf::table>(std::move(out_cols))};
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

bool column_ids_need_reader_projection(duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                                       std::size_t full_schema_size)
{
  // count(*) / zero-column scans are NOT a projection: assemble_scan_output keeps
  // the reader's natural batch (its empty-output_layout path) so the row count the
  // downstream aggregation consumes is preserved. Projecting to 0 columns would
  // erase it.
  if (column_ids.empty()) { return false; }
  // Virtual columns (e.g. count(*)'s row-id marker) are not physical file columns
  // and must never drive a by-name reader projection — a scan that reads ONLY
  // virtual columns (count(*)) keeps the reader's natural batch.  Mirror the
  // IsVirtualColumn guard handle_position uses below.
  bool any_real = false;
  for (std::size_t i = 0; i < column_ids.size(); ++i) {
    auto const primary_idx = column_ids[i].GetPrimaryIndex();
    if (duckdb::IsVirtualColumn(primary_idx)) { continue; }
    any_real = true;
    // A real column read out of its identity position ⇒ pruned / reordered.
    if (primary_idx != i) { return true; }
  }
  if (!any_real) { return false; }  // only virtual columns (count(*)) — natural batch
  // All real columns sit at identity positions: a projection only if the read is a
  // proper prefix (fewer columns than the file's full schema).
  return column_ids.size() != full_schema_size;
}

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

  // First mark every shape that may need a by-name reader projection:
  // explicit projection, hive partitions, or a pruned/reordered column_ids
  // subset. After the walk below, we clear this again for scans with no real
  // parquet data columns.
  plan.needs_reader_projection =
    !projection_ids.empty() || !partition_indices.empty() ||
    (!names.empty() && column_ids_need_reader_projection(column_ids, names.size()));

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
      auto const partition_cols_idx = plan.partition_columns.size();
      plan.partition_columns.push_back(scan_plan::partition_column{
        primary_idx, names.at(primary_idx), returned_types.at(primary_idx)});
      plan.output_layout.push_back(
        scan_plan::output_entry{scan_plan::output_entry::PARTITION, partition_cols_idx});
    } else {
      // Data column — always added to the batch (even if filter-only, we need
      // it for filter evaluation). Store an empty name when @c names is empty:
      // the caller's guard only forces non-empty names for name-dependent paths
      // (projection, filter, partitions), and the plain-read case populates
      // data_columns without ever consuming the name downstream.
      auto const batch_idx = plan.data_columns.size();
      std::string col_name = names.empty() ? std::string{} : names.at(primary_idx);
      plan.data_columns.push_back(scan_plan::data_column{primary_idx, std::move(col_name)});
      primary_to_batch[primary_idx] = batch_idx;
      if (is_output) {
        plan.output_layout.push_back(
          scan_plan::output_entry{scan_plan::output_entry::DATA, batch_idx});
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

  // The gate: a column-less scan (count(*)/virtual-only or partition-only, only
  // known after the walk) must keep the natural batch — projecting it hands cuDF
  // set_column_names({}), a zero-column read over live row groups that hangs.
  plan.needs_reader_projection = plan.needs_reader_projection && !plan.data_columns.empty();

  SIRIUS_LOG_DEBUG("[scan_plan] built plan: {} data col(s), {} partition col(s), {} output entries",
                   plan.data_columns.size(),
                   plan.partition_columns.size(),
                   plan.output_layout.size());

  return plan;
}

}  // namespace sirius::op::scan
