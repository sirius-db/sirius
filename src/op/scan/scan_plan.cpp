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
#include <duckdb/common/multi_file/multi_file_reader.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

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

std::unique_ptr<cudf::table> append_parquet_virtual_columns(std::unique_ptr<cudf::table> table,
                                                            scan_plan const& plan,
                                                            std::string const& file_path,
                                                            std::size_t file_index,
                                                            std::int64_t file_row_offset,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  if (!table || plan.virtual_columns.empty()) { return table; }
  if (static_cast<std::size_t>(table->num_columns()) != plan.data_columns.size()) {
    throw sirius::internal_exception(
      "parquet virtual scan: decoded column count does not match planned layout");
  }
  auto const rows                                    = table->num_rows();
  std::vector<std::unique_ptr<cudf::column>> columns = table->release();
  columns.reserve(columns.size() + plan.virtual_columns.size());
  for (auto const& virtual_column : plan.virtual_columns) {
    switch (virtual_column.kind) {
      case scan_plan::parquet_virtual_column_kind::FILENAME: {
        cudf::string_scalar value(file_path, true, stream, mr);
        columns.push_back(cudf::make_column_from_scalar(value, rows, stream, mr));
        break;
      }
      case scan_plan::parquet_virtual_column_kind::FILE_INDEX: {
        cudf::numeric_scalar<std::uint64_t> value(
          static_cast<std::uint64_t>(file_index), true, stream, mr);
        columns.push_back(cudf::make_column_from_scalar(value, rows, stream, mr));
        break;
      }
      case scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER: {
        cudf::numeric_scalar<std::int64_t> initial(file_row_offset, true, stream, mr);
        cudf::numeric_scalar<std::int64_t> step(1, true, stream, mr);
        columns.push_back(cudf::sequence(rows, initial, step, stream, mr));
        break;
      }
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

owning_table_view assemble_scan_output(scan_plan const& plan,
                                       owning_table_view&& table,
                                       std::vector<std::string> const& partition_values,
                                       ::cuda::stream_ref stream)
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
    std::unordered_set<std::size_t> seen;
    bool has_duplicates = false;
    for (auto const& entry : plan.output_layout) {
      positions.push_back(entry.idx);
      has_duplicates = !seen.insert(entry.idx).second || has_duplicates;
    }
    if (!has_duplicates) {
      table.select_columns(positions);
      return std::move(table);
    }

    auto materialized = table.release(stream);
    auto source       = materialized->release();
    std::vector<std::unique_ptr<cudf::column>> output;
    output.reserve(positions.size());
    std::unordered_map<std::size_t, std::size_t> first_output;
    for (auto const position : positions) {
      auto [it, inserted] = first_output.emplace(position, output.size());
      if (inserted) {
        output.push_back(std::move(source.at(position)));
      } else {
        output.push_back(std::make_unique<cudf::column>(output.at(it->second)->view(), stream));
      }
    }
    return owning_table_view{std::make_unique<cudf::table>(std::move(output))};
  }

  // Partition columns present: materialize the reader batch and rebuild, moving
  // DATA columns out and synthesizing constant PARTITION columns from the path.
  auto reader_output  = table.release(stream);
  auto const num_rows = reader_output->num_rows();
  auto data_cols      = reader_output->release();  // move columns out, no GPU copy

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(plan.output_layout.size());
  std::unordered_map<std::size_t, std::size_t> first_data_output;

  for (auto const& entry : plan.output_layout) {
    if (entry.source == scan_plan::output_entry::DATA) {
      auto [it, inserted] = first_data_output.emplace(entry.idx, out_cols.size());
      if (inserted) {
        out_cols.push_back(std::move(data_cols.at(entry.idx)));
      } else {
        out_cols.push_back(std::make_unique<cudf::column>(out_cols.at(it->second)->view(), stream));
      }
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
  // count(*) / zero-column scans carry no real column to project by; build_scan_plan
  // gives them a row-count carrier instead.
  if (column_ids.empty()) { return false; }
  // Virtual columns (e.g. count(*)'s row-id marker) are not physical file columns
  // and must never drive a by-name reader projection. Mirror the IsVirtualColumn
  // guard handle_position uses below.
  std::size_t real_count = 0;
  for (std::size_t i = 0; i < column_ids.size(); ++i) {
    auto const primary_idx = column_ids[i].GetPrimaryIndex();
    if (duckdb::IsVirtualColumn(primary_idx)) { continue; }
    auto const physical_position = real_count++;
    // A real column read out of its identity position ⇒ pruned / reordered.
    if (primary_idx != physical_position) { return true; }
  }
  if (real_count == 0) { return false; }  // only virtual columns (count(*) / metadata)
  // All real columns sit at identity positions: a projection only if the read is a
  // proper prefix (fewer columns than the file's full schema).
  return real_count != full_schema_size;
}

scan_plan build_scan_plan(duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                          duckdb::vector<duckdb::idx_t> const& projection_ids,
                          duckdb::vector<std::string> const& names,
                          duckdb::vector<sirius::logical_type> const& returned_types,
                          std::size_t output_types_size,
                          duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
                          std::vector<bound_virtual_column> const& virtual_columns)
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

  std::unordered_map<duckdb::column_t, bound_virtual_column const*> virtual_by_id;
  for (auto const& column : virtual_columns) {
    virtual_by_id.emplace(column.column_id, &column);
  }

  // Walk positions in output-first order. When projection_ids is non-empty the
  // first output_types_size entries are the output columns in output order;
  // the remaining entries are pure-filter columns that must be read but not
  // emitted. When projection_ids is empty, column_ids is both the read list
  // and the output (no pure-filter columns).
  //
  // Read once; preserve duplicate outputs.
  std::unordered_map<std::size_t, std::size_t> primary_to_batch;  // P → D
  std::unordered_map<std::size_t, std::size_t> primary_to_partition;
  std::unordered_map<duckdb::column_t, std::size_t> virtual_to_ordinal;
  struct output_request {
    enum class kind : std::uint8_t { DATA, PARTITION, VIRTUAL } source;
    std::size_t key;
  };
  std::vector<output_request> output_requests;

  auto handle_position = [&](std::size_t column_ids_pos, bool is_output) {
    auto const primary_idx = column_ids.at(column_ids_pos).GetPrimaryIndex();
    auto const definition  = virtual_by_id.find(primary_idx);
    if (duckdb::IsVirtualColumn(primary_idx) || definition != virtual_by_id.end()) {
      // Count and empty markers are execution sentinels, not user columns.
      if (primary_idx == duckdb::COLUMN_IDENTIFIER_ROW_ID ||
          primary_idx == duckdb::COLUMN_IDENTIFIER_EMPTY) {
        return;
      }
      if (definition == virtual_by_id.end()) {
        throw duckdb::NotImplementedException("parquet scan: unsupported virtual column id %llu",
                                              static_cast<unsigned long long>(primary_idx));
      }

      auto [it, inserted] = virtual_to_ordinal.emplace(primary_idx, plan.virtual_columns.size());
      if (inserted) {
        scan_plan::parquet_virtual_column_kind kind;
        if (definition->second->kind) {
          kind = *definition->second->kind;
        } else if (primary_idx == duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME) {
          kind = scan_plan::parquet_virtual_column_kind::FILENAME;
        } else if (primary_idx == duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX) {
          kind = scan_plan::parquet_virtual_column_kind::FILE_INDEX;
        } else if (primary_idx == duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER) {
          kind = scan_plan::parquet_virtual_column_kind::FILE_ROW_NUMBER;
        } else {
          throw duckdb::NotImplementedException("parquet scan: unsupported virtual column id %llu",
                                                static_cast<unsigned long long>(primary_idx));
        }
        auto const* column       = definition->second;
        auto const expected_type = kind == scan_plan::parquet_virtual_column_kind::FILENAME
                                     ? sirius::type_id::VARCHAR
                                     : (kind == scan_plan::parquet_virtual_column_kind::FILE_INDEX
                                          ? sirius::type_id::UBIGINT
                                          : sirius::type_id::BIGINT);
        if (column->type.id() != expected_type) {
          throw duckdb::NotImplementedException(
            "parquet scan: virtual column '%s' has unsupported type %s",
            column->name,
            column->type.to_string());
        }
        plan.virtual_columns.push_back(
          scan_plan::virtual_column{primary_idx, column->name, column->type, kind, 0});
      }
      if (is_output) {
        output_requests.push_back(
          {output_request::kind::VIRTUAL, static_cast<std::size_t>(primary_idx)});
      }
      return;
    }

    bool const is_partition = plan.partition_primary_indices.count(primary_idx) > 0;

    if (is_partition) {
      // Filter-only partition columns are dropped: DuckDB prunes at the file
      // level and our filter builder will skip them. We only materialize
      // partition metadata for output columns.
      if (!is_output) { return; }
      auto [it, inserted] =
        primary_to_partition.emplace(primary_idx, plan.partition_columns.size());
      if (inserted) {
        plan.partition_columns.push_back(scan_plan::partition_column{
          primary_idx, names.at(primary_idx), returned_types.at(primary_idx)});
      }
      output_requests.push_back({output_request::kind::PARTITION, primary_idx});
    } else {
      // Data column — always added to the batch (even if filter-only, we need
      // it for filter evaluation). Store an empty name when @c names is empty:
      // the caller's guard only forces non-empty names for name-dependent paths
      // (projection, filter, partitions), and the plain-read case populates
      // data_columns without ever consuming the name downstream.
      auto [it, inserted] = primary_to_batch.emplace(primary_idx, plan.data_columns.size());
      if (inserted) {
        std::string col_name = names.empty() ? std::string{} : names.at(primary_idx);
        plan.data_columns.push_back(scan_plan::data_column{primary_idx, std::move(col_name)});
      }
      if (is_output) { output_requests.push_back({output_request::kind::DATA, primary_idx}); }
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

  // Legacy virtual columns use ordinary primary indices and still need projection.
  plan.needs_reader_projection = plan.needs_reader_projection || !plan.virtual_columns.empty();

  // Virtual-only scans need one physical column to establish row count.
  if (plan.data_columns.empty() && !names.empty() && returned_types.size() == names.size()) {
    std::optional<std::size_t> carrier;
    std::size_t carrier_width = 0;
    for (std::size_t p = 0; p < returned_types.size(); ++p) {
      if (plan.partition_primary_indices.count(p) > 0 || virtual_by_id.contains(p)) { continue; }
      if (!returned_types[p].is_fixed_width()) { continue; }
      auto const width = returned_types[p].fixed_width_byte_size();
      if (width > 0 && (!carrier || width < carrier_width)) {
        carrier       = p;
        carrier_width = width;
      }
    }
    if (!carrier && !plan.virtual_columns.empty()) {
      for (std::size_t p = 0; p < returned_types.size(); ++p) {
        if (plan.partition_primary_indices.count(p) == 0 && !virtual_by_id.contains(p) &&
            returned_types[p].id() == sirius::type_id::VARCHAR) {
          carrier = p;
          break;
        }
      }
    }
    if (carrier) {
      plan.carrier_batch_index   = plan.data_columns.size();
      primary_to_batch[*carrier] = plan.data_columns.size();
      plan.data_columns.push_back(scan_plan::data_column{*carrier, names.at(*carrier)});
      plan.needs_reader_projection = true;
    }
  }

  if (plan.has_user_virtual_columns() && plan.data_columns.empty()) {
    throw duckdb::NotImplementedException(
      "parquet virtual scan: no supported physical row-count carrier");
  }

  for (std::size_t v = 0; v < plan.virtual_columns.size(); ++v) {
    plan.virtual_columns[v].materialized_idx = plan.data_columns.size() + v;
  }

  for (auto const& request : output_requests) {
    if (request.source == output_request::kind::PARTITION) {
      plan.output_layout.push_back(
        {scan_plan::output_entry::PARTITION, primary_to_partition.at(request.key)});
    } else if (request.source == output_request::kind::DATA) {
      plan.output_layout.push_back(
        {scan_plan::output_entry::DATA, primary_to_batch.at(request.key)});
    } else {
      auto const ordinal = virtual_to_ordinal.at(static_cast<duckdb::column_t>(request.key));
      plan.output_layout.push_back(
        {scan_plan::output_entry::DATA, plan.virtual_columns.at(ordinal).materialized_idx});
    }
  }

  plan.batch_position_by_column_id.assign(column_ids.size(), std::nullopt);
  for (std::size_t c = 0; c < column_ids.size(); ++c) {
    auto const primary_idx = column_ids[c].GetPrimaryIndex();
    if (auto it = virtual_to_ordinal.find(primary_idx); it != virtual_to_ordinal.end()) {
      plan.batch_position_by_column_id[c] = plan.virtual_columns.at(it->second).materialized_idx;
      continue;
    }
    auto it = primary_to_batch.find(primary_idx);
    if (it == primary_to_batch.end()) { continue; }
    plan.batch_position_by_column_id[c] = it->second;
  }

  // The gate: a column-less scan with no usable carrier must keep the natural
  // batch — projecting it hands cuDF set_column_names({}), a zero-column read
  // over live row groups that hangs.
  plan.needs_reader_projection = plan.needs_reader_projection && !plan.data_columns.empty();

  SIRIUS_LOG_DEBUG("[scan_plan] built plan: {} data col(s), {} partition col(s), {} output entries",
                   plan.data_columns.size(),
                   plan.partition_columns.size(),
                   plan.output_layout.size());

  return plan;
}

}  // namespace sirius::op::scan
