/*
 * Copyright 2026, Sirius Contributors.
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

#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <op/scan/scan_plan.hpp>
#include <op/sirius_dynamic_filter.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::op::scan {

/// @brief AND-merge AST-capable filters from @p filters into @p tree, AND-ing with @p
/// existing_root when non-null. Returns the new root, or @p existing_root unchanged when no
/// fragment contributed (empty set, filters lack the AST capability, or referenced columns are
/// all hive partitions).
///
/// Column index resolution uses @p plan: hive-partition columns are skipped (their values come
/// from the file path, not the parquet file itself), DATA columns resolve through
/// @c plan.output_layout to @c plan.data_columns[i].name. Emitted column references use
/// @c cudf::ast::column_name_reference so the result is compatible with the parquet reader's
/// @c set_filter API.
///
/// The returned pointer (when non-null) points into @p tree and remains valid for the lifetime
/// of @p tree. Device scalars referenced by filter literals are owned by the filter objects in
/// @p filters and must outlive any installed AST built from this call.
[[nodiscard]] cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan);

/// @brief Runtime-apply: drop rows of an already-materialized output table that fail the dynamic
/// filters in @p filters.
///
/// This is the post-decode / cached-scan consumption path — used where the parquet reader's
/// @c set_filter pushdown is unavailable (cached GPU/host data) or arrived too late (filter
/// published after the reader options were built). @p table must already be in the scan's output
/// layout, i.e. column @c i corresponds to @c plan.output_layout[i]; this matches both the parquet
/// path (post @ref assemble_scan_output) and the cached path.
///
/// For each consumer column with AST-lowerable filter(s) that resolves to a DATA column present in
/// @p table, the filter fragments are evaluated against that column (@c cudf::compute_column) and
/// AND-conjoined across filters and columns; rows failing the combined predicate — including rows
/// with a null key, which can never match an equi-join — are dropped via
/// @c cudf::apply_boolean_mask. Hive-partition columns and out-of-range indices are skipped.
///
/// Applies two filter kinds, AND-combined into one keep-mask: AST-lowerable filters (zone-maps, via
/// @c cudf::compute_column) and apply-lowerable filters (IN-list / bloom membership, via each
/// filter's @c compute_mask → @c cudf::contains). When @p include_ast_lowerable is false, only the
/// membership filters are applied — used by the disk scan path, where zone-maps are already
/// consumed as row-group pruning at read and re-applying them per-row would be wasted work. The
/// cached/pinned path (no row groups to prune) leaves it true so both kinds apply.
///
/// Returns @p table moved through unchanged when no filter contributes (empty channel, no filter of
/// an enabled capability, zero rows, or all referenced columns are partitions / out of range). The
/// returned table's buffers are allocated on @p stream; @p table is consumed.
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_to_output_table(
  std::unique_ptr<cudf::table> table,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan,
  rmm::cuda_stream_view stream,
  bool include_ast_lowerable = true);

/// @brief Row-GROUP pruning by dynamic filters using cached parquet column statistics.
///
/// A dynamic join filter is redundant with the join that produced it: the join itself discards
/// every non-matching probe row, so the filter never needs to remove individual rows. Its only
/// useful effect is to SKIP whole parquet row groups that cannot contain a join match — saving the
/// I/O and decode of those groups. This evaluates the dynamic filters' zone-map predicate against
/// each source's row-group statistics (via @c hybrid_scan_reader::filter_row_groups_with_stats) and
/// returns the surviving row groups per source, in the same order and outer size as @p rg_per_src.
///
/// This is deliberately row-GROUP only: there is no row-level predicate evaluation, so when the
/// filter excludes nothing (e.g. scattered keys whose @c [min,max] spans the column domain) the
/// result equals @p rg_per_src and the only cost is a cheap stats check — net-zero versus not
/// pushing the filter at all. Correctness does not depend on it pruning anything, because the join
/// remains authoritative.
///
/// @p per_source_metadata[i] is the parsed footer backing @p rg_per_src[i]; a null entry (or a
/// source with no AST-lowerable filter) is passed through unpruned. @p base_options supplies reader
/// settings; a private copy carries the dynamic predicate for the stats evaluation, so
/// @p base_options is not mutated. @p pruned_out receives the total number of row groups skipped.
[[nodiscard]] std::vector<std::vector<cudf::size_type>> prune_row_groups_by_dynamic_filters(
  std::vector<cudf::io::parquet::FileMetaData const*> const& per_source_metadata,
  std::vector<std::vector<cudf::size_type>> const& rg_per_src,
  cudf::io::parquet_reader_options const& base_options,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan,
  rmm::cuda_stream_view stream,
  std::size_t& pruned_out);

}  // namespace sirius::op::scan
