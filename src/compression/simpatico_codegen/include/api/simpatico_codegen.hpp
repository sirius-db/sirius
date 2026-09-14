// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/selection/selection.hpp"
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

/// A single column after compression: metadata plus the compressed plan tree.
struct compressed_column {
  /// Optional name copied from the source table (populated when column names are
  /// passed to compress_with_plan).
  std::optional<std::string> name;
  /// Original element type of the column.
  cudf::data_type dtype{};
  /// Number of logical rows in the original column.
  std::int64_t num_rows = 0;
  /// Plan tree describing the compression layout; owns every compressed rep.
  std::unique_ptr<PlanTree> plan_tree;
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

// ── Predicate-pushdown decompression ─────────────────────────────────────────

/// Decompress a column subset, answering a set-membership predicate on selected
/// columns instead of reconstructing them.
///
/// @p predicates is parallel to @p selected_columns; an entry with an empty
/// @c equals_any reconstructs that column normally. A column with an active
/// directive comes back as BOOL8 of the same row count (`value ∈ equals_any`,
/// nulls propagated) — never its declared dtype — so the caller must be prepared
/// for the type change.
///
/// The point is to skip the decode entirely for dictionary-compressed columns
/// consumed only by an equality / IN filter: the predicate is resolved against
/// the key set and mapped over the indices, so the key chars are never gathered
/// into a full-width column. Use @c simpatico::probe_column's
/// @c can_answer_equality to check that a column's plan can actually do this
/// before pushing a predicate into it.
///
/// @throws std::runtime_error if @p predicates and @p selected_columns differ in
///         size, or on the usual decompression failures.
std::unique_ptr<cudf::table> decompress(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  std::span<const decode_predicate> predicates,
  simpatico::stream_pool& pool,
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());

/// Decompress a column subset with the caller's row filter applied DURING the
/// decode (experimental, env gate SIRIUS_EXP_FUSED_SCAN_FILTER=1). @p request
/// carries the conjuncts a decode can resolve plus the output shape tag per
/// selected column (see codegen/selection/selection.hpp).
///
/// When the gate is on and every precondition holds, columns are decoded with
/// the two-wave mask pipeline: wave 1 ballots each filter column's rows into
/// mask words, the masks are AND-combined and counted (one host sync for the
/// survivor count), then wave 2 decodes the compactable columns straight into
/// survivor_count-row columns and the rest full width. @p result comes back
/// with applied=true, the selection mask/offsets, and the gather map
/// (row_indices) it used. The returned table is uniformly survivor-sized —
/// the compacted routes came back that way and the full-width ones are
/// compacted here — so the caller only has to skip its own filter pass.
///
/// When the gate is off, @p request is empty, or any precondition fails
/// (non-bitpack filter column, nulls, ...), this is EXACTLY the unfiltered
/// decompress(table, selected_columns, pool, mr) — same kernels, same
/// allocations — returned as released columns, and result.applied is false.
/// result.status refines the applied=false cases: `refused` (no device work),
/// `declined_unselective` (too many rows survived for compaction to pay off —
/// the caller should remember this per scan and drop the row selection from its
/// remaining batches), or `failed` (mid-flight fallback, exceptional).
///
/// Equality conjuncts answerable off a dictionary ride INSIDE the request
/// (scan_filter_request::bool8_filters): wave 1 resolves them via the
/// decode_predicate path, packs the BOOL8 result to mask words and ANDs it into
/// the batch mask. On ANY non-applied outcome with bool8_filters present, the
/// rerun is the PREDICATED decompress — those columns come back as BOOL8
/// substitution columns exactly like the ordinary pushdown, never a plain
/// decode (the dictionary win survives every fallback). Callers must therefore
/// be ready for BOOL8 at those columns whenever result.applied is false.
/// Assembling the output can itself refuse (a null-masked column, an output
/// that is neither full width nor survivor-sized): the call then falls back to
/// the unfiltered decode, sets result.status = failed and writes @p error_out.
/// A caller never sees a half-filtered batch.
///
/// Synchronizes @p stream before returning when the filtering applied, so the
/// caller may free or rebind the inputs immediately.
std::unique_ptr<cudf::table> decompress_scan_filter(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  simpatico::stream_pool& pool,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref(),
  std::string* error_out            = nullptr);

}  // namespace simpatico
