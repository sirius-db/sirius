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

#pragma once

// sirius
#include <helper/logical_type.hpp>
#include <op/scan/hive_partition.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/table_filter.hpp>

// cudf
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Canonical plan for a parquet-style scan.
 *
 * Built once by the scan operator from DuckDB planner inputs (column_ids,
 * projection_ids, names, returned_types, partition_indices). All downstream
 * consumers — reader column projection, filter-expression building, hive
 * injection, post-filter output assembly — read from this single structure
 * instead of maintaining parallel vectors.
 *
 * Three index spaces appear throughout the scan pipeline:
 *   P = primary index       (position in DuckDB's full schema @c names / @c returned_types)
 *   C = column_ids position (position in the operator's @c column_ids list)
 *   D = batch position      (position in the reader's output table, after hive removal)
 *
 * Hive-partition columns live in P-space but are not in the parquet file, so
 * they never appear in D-space. They are injected post-read into the final
 * output in the order column_ids expects. Also, pure filter columns live in D-space
 * but are not in the final output layout, so they are dropped after filter evaluation.
 */
struct scan_plan {
  /// A column produced by the parquet reader, in batch order.
  struct data_column {
    std::size_t primary_idx;  ///< P — index into the DuckDB schema
    std::string name;         ///< parquet column name
  };

  /// A column synthesized from the file path, injected after read.
  struct partition_column {
    std::size_t primary_idx;    ///< P — index into the DuckDB schema
    std::string name;           ///< partition key name (e.g. "year")
    sirius::logical_type type;  ///< DuckDB type for the cudf scalar factory
  };

  /// One entry per final output column, in the order DuckDB expects.
  struct output_entry {
    enum source_t : std::uint8_t { DATA, PARTITION };
    source_t source;
    std::size_t
      idx;  ///< index into @c data_columns (if DATA) or @c partition_columns (if PARTITION)
  };

  /// Columns read from parquet, in D order.
  std::vector<data_column> data_columns;

  /// Partition columns injected post-read.
  std::vector<partition_column> partition_columns;

  /// Final output layout. Length equals the scan operator's @c types.size().
  std::vector<output_entry> output_layout;

  /// C → D map. @c nullopt when the column is not in the data batch
  /// (either a hive partition, a virtual column, or not projected).
  std::vector<std::optional<std::size_t>> batch_position_by_column_id;

  /// Primary indices of hive-partition columns. Supplied to
  /// @c convert_table_filters_to_expression so those filters are dropped
  /// from pushdown (they aren't in the parquet file).
  std::unordered_set<std::size_t> partition_primary_indices;

  /// True iff the reader needs explicit column projection — set when the planner
  /// pruned or reordered columns (non-empty @c projection_ids) or hive-partition
  /// columns must be dropped from the physical read. When false, the reader's
  /// natural "read everything in column_ids order" output already matches what
  /// the pipeline expects, and the scan can skip @c set_column_names and
  /// per-file name-based leaf resolution.
  bool needs_reader_projection = false;

  //===--------------------------------------------------------------------===//
  // Convenience views
  //===--------------------------------------------------------------------===//

  [[nodiscard]] bool has_partitions() const { return !partition_columns.empty(); }

  /// True iff the scan is producing a strict subset / reordering of the file's
  /// columns — i.e. the reader must be told which columns to read.
  [[nodiscard]] bool is_projected() const;

  /// Names of data columns in D order. Used for @c reader_options.set_column_names
  /// and as the D-indexed resolver for AST filter translation.
  [[nodiscard]] std::vector<std::string> data_column_names() const;

  /// Resolve a D-space index to a parquet column name, for use as the AST
  /// translator's name resolver.
  [[nodiscard]] std::string batch_column_name(duckdb::idx_t batch_position) const;

  /// Batch positions of data columns that are read for filter evaluation but
  /// not part of the output layout. The metadata scan uses this to exclude
  /// pure-filter columns from the uncompressed-byte accounting that drives
  /// row-group partitioning.
  [[nodiscard]] std::unordered_set<std::size_t> pure_filter_batch_positions() const;
};

//===--------------------------------------------------------------------===//
// Output assembly — free functions
//===--------------------------------------------------------------------===//

/// True when the reader's natural batch needs to be reshaped to match the plan's
/// final output layout — i.e. when @ref assemble_scan_output should be called.
///
/// Returns @c false in two short-circuit cases where the reader's output flows
/// through to the pipeline unchanged:
///   1. @c output_layout is empty (SELECT count(*) — emitting a 0-column table
///      would erase the row count downstream count-style aggregations consume).
///   2. The plan is a trivial identity: no partitions, and @c output_layout
///      covers @c data_columns 1:1 in order.
[[nodiscard]] bool needs_output_assembly(scan_plan const& plan);

/// Reshape the reader's D-order batch to the plan's output layout: walk
/// @c output_layout, moving DATA columns out of the batch and synthesizing
/// PARTITION columns from @p partition_values. Pure-filter data columns
/// (present in @c data_columns but not referenced by @c output_layout) are
/// implicitly freed when the released batch goes out of scope.
///
/// Callers should gate the call on @ref needs_output_assembly to avoid the
/// release/rebuild round-trip when assembly is a no-op. When called regardless,
/// the function still produces a correct output but performs unnecessary work
/// in the identity case.
///
/// @param plan              The scan plan describing the layout.
/// @param reader_output     The reader's D-order batch to reshape.
/// @param partition_values  Partition values for this split, in
///                          @c partition_columns order. Empty when the plan has
///                          no partition columns.
/// @param stream            CUDA stream for any GPU work (scalar-backed column
///                          construction).
[[nodiscard]] std::unique_ptr<cudf::table> assemble_scan_output(
  scan_plan const& plan,
  std::unique_ptr<cudf::table> reader_output,
  std::vector<std::string> const& partition_values,
  rmm::cuda_stream_view stream);

/// Batch (D-space) positions of the output DATA columns, in @c output_layout order.
/// Empty when @c output_layout has no DATA entries (SELECT count(*) or a partition-only output), in
/// which case the caller gathers all columns instead.
/// @param plan  The scan plan describing the layout.
[[nodiscard]] std::vector<cudf::size_type> output_data_positions(scan_plan const& plan);

/// Build a scan_plan from DuckDB planner inputs. See @c scan_plan for semantics.
///
/// @param column_ids          Column ids exposed by the table function.
/// @param projection_ids      Indices into column_ids that the planner projects
///                            (first @c output_types_size entries are outputs;
///                             remaining entries are pure-filter columns that
///                             must be read for filter evaluation but not emitted).
///                            Empty = read everything in column_ids order.
/// @param names               DuckDB schema column names.
/// @param returned_types      DuckDB schema column types.
/// @param output_types_size   The scan operator's types.size() — used to split
///                            projection_ids into output vs pure-filter.
/// @param partition_indices   Hive partition indices advertised by the bind data.
///                            TODO: other partition indices may need to be supported for other
///                            formats.
scan_plan build_scan_plan(duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                          duckdb::vector<duckdb::idx_t> const& projection_ids,
                          duckdb::vector<std::string> const& names,
                          duckdb::vector<sirius::logical_type> const& returned_types,
                          std::size_t output_types_size,
                          duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices);

}  // namespace sirius::op::scan
