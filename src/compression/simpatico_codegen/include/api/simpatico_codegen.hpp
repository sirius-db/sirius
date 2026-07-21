// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace simpatico {

/// A single column after compression: metadata plus the compressed plan compound.
struct compressed_column {
  /// Optional name copied from the source table (populated when column names are
  /// passed to compress_with_plan).
  std::optional<std::string> name;
  /// Original element type of the column.
  cudf::data_type dtype{};
  /// Number of logical rows in the original column.
  std::int64_t num_rows = 0;
  /// Plan tree describing the compression layout; owns every compressed rep.
  std::unique_ptr<PlanTree> compound;
};

/// Compressed representation of a cuDF table: one compressed_column per source
/// column, in the same order, with optional per-column names.
class compressed_table {
 public:
  std::vector<compressed_column> columns;

  /// Number of compressed columns.
  std::size_t num_columns() const { return columns.size(); }

  /// Common row count across all columns (0 if the table is empty).
  std::int64_t num_rows() const;

  /// Return a flat leaf descriptor for every stored representation in each
  /// column. The outer vector is indexed by column; the inner by leaf within
  /// that column. Leaf order follows PlanTree node order (node.rep before
  /// node.channels; channels in output_paths order where available).
  std::vector<std::vector<simpatico::leaf_desc>> describe(
    rmm::cuda_stream_view stream = cudf::get_default_stream()) const;

  /// Decompress on a single CUDA stream.
  ///
  /// Equivalent to calling the free function simpatico::decompress(*this, ...).
  std::unique_ptr<cudf::table> decompress(
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref()) const;
};

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Split a multi-column plan DSL string on `---` separator lines.
///
/// Lines that are blank or begin with `#` are skipped. Each resulting block is
/// trimmed of leading/trailing whitespace. The returned vector contains one
/// entry per column plan, in order.
///
/// @param plan_dsl  Multi-column plan string in Simpatico DSL format.
std::vector<std::string> split_plan_dsl(std::string_view plan_dsl);

// ── Compression ───────────────────────────────────────────────────────────────

/// Compress all columns of @p table sequentially on a single CUDA stream.
///
/// @param table         Source table. Column count must equal the number of
///                      `---`-separated blocks in @p plan_dsl.
/// @param plan_dsl      Multi-column plan DSL string.
/// @param stream        CUDA stream used for all GPU operations.
/// @param mr            Device memory resource; nullptr selects the RMM default.
/// @param column_names  Optional per-column names stored in the result. Must be
///                      empty or exactly @p table.num_columns() long.
/// @throws std::runtime_error  plan/table column count mismatch or GPU error.
compressed_table compress_with_plan(
  cudf::table_view table,
  std::string_view plan_dsl,
  rmm::cuda_stream_view stream          = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr     = rmm::mr::get_current_device_resource_ref(),
  std::vector<std::string> column_names = {});

/// Compress all columns in parallel using @p column_threads worker threads.
///
/// `max(1, column_threads)` streams are leased from a process-lifetime internal
/// cache (never destroyed), so the returned table's buffers are safe to free on
/// any stream — including the RMM default — after the call returns.
///
/// @param table          Source table.
/// @param plan_dsl       Multi-column plan DSL string.
/// @param column_threads Number of parallel CUDA streams / worker threads.
/// @param mr             Device memory resource; nullptr selects the RMM default.
/// @param column_names   Optional per-column names.
/// @throws std::runtime_error  plan/table column count mismatch or GPU error.
compressed_table compress_with_plan(
  cudf::table_view table,
  std::string_view plan_dsl,
  int column_threads,
  rmm::device_async_resource_ref mr     = rmm::mr::get_current_device_resource_ref(),
  std::vector<std::string> column_names = {});

/// Compress all columns in parallel using a caller-owned stream pool.
///
/// The pool must remain valid for the duration of the call. Reusing the same
/// pool across multiple calls is safe and avoids repeated stream allocation.
///
/// @param table          Source table.
/// @param plan_dsl       Multi-column plan DSL string.
/// @param pool           Caller-supplied stream pool.
/// @param mr             Device memory resource; nullptr selects the RMM default.
/// @param column_names   Optional per-column names.
/// @throws std::runtime_error  plan/table column count mismatch or GPU error.
compressed_table compress_with_plan(
  cudf::table_view table,
  std::string_view plan_dsl,
  simpatico::stream_pool& pool,
  rmm::device_async_resource_ref mr     = rmm::mr::get_current_device_resource_ref(),
  std::vector<std::string> column_names = {});

// ── Decompression ─────────────────────────────────────────────────────────────

/// Decompress all columns of @p table sequentially on a single CUDA stream.
///
/// @param table   Compressed table.
/// @param stream  CUDA stream for all GPU operations.
/// @param mr      Device memory resource; nullptr selects the RMM default.
/// @returns  Newly allocated decompressed table (column order matches input).
/// @throws std::runtime_error on GPU error.
std::unique_ptr<cudf::table> decompress(
  const compressed_table& table,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/// Decompress all columns in parallel using @p column_threads worker threads.
///
/// @param table          Compressed table.
/// @param column_threads Number of parallel CUDA streams / worker threads.
/// @param mr             Device memory resource; nullptr selects the RMM default.
/// @throws std::runtime_error on GPU error.
std::unique_ptr<cudf::table> decompress(
  const compressed_table& table,
  int column_threads,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/// Decompress all columns in parallel using a caller-owned stream pool.
///
/// @param table  Compressed table.
/// @param pool   Caller-supplied stream pool.
/// @param mr     Device memory resource; nullptr selects the RMM default.
/// @throws std::runtime_error on GPU error.
std::unique_ptr<cudf::table> decompress(
  const compressed_table& table,
  simpatico::stream_pool& pool,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

// ── Selective decompression ───────────────────────────────────────────────────
//
// Decompress only a subset of columns from a cached compressed_table, avoiding
// a full re-fetch when only a projection is needed. The output table contains
// exactly the requested columns in the order given by @p selected_columns.
// Out-of-range indices throw std::runtime_error.

/// Decompress a column subset sequentially on a single CUDA stream.
std::unique_ptr<cudf::table> decompress(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/// Decompress a column subset in parallel using @p column_threads worker threads.
std::unique_ptr<cudf::table> decompress(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  int column_threads,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/// Decompress a column subset in parallel using a caller-owned stream pool.
std::unique_ptr<cudf::table> decompress(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  simpatico::stream_pool& pool,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

}  // namespace simpatico
