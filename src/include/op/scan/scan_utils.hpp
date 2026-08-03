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
#include <codegen/selection/selection.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <helper/logical_type.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/table_filter.hpp>

// standard library
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op {

/**
 * @brief Build a mapping from column_ids index to batch column position.
 *
 * The parquet scan (make_selected_column_indices) produces batch columns in
 * column_ids order, but only for indices present in projection_ids.
 * For example, if column_ids has 5 entries and projection_ids = {1, 3}:
 *   batch position 0 → column_ids[1]
 *   batch position 1 → column_ids[3]
 *
 * When projection_ids is empty, every column_ids entry maps to its own index.
 *
 * Returns a vector of size column_ids_count where:
 *   result[i] = batch position of column_ids[i], or nullopt if the column is not in the batch.
 */
std::vector<std::optional<std::size_t>> build_batch_column_map(
  const duckdb::vector<duckdb::idx_t>& projection_ids, std::size_t column_ids_count);

/**
 * @brief Convert a DuckDB TableFilterSet into a single bound DuckDB expression (conjunction of
 * all filters), suitable for passing to gpu_expression_translator::translate_expression().
 *
 * @p batch_position_by_column_id is the canonical column_ids → batch-position map, with @c nullopt
 * for columns that are not present in the batch (e.g. unprojected columns or hive partitions).
 *
 * Filters whose column's primary index is in @p skip_primary_indices are omitted. Parquet scans
 * pass the hive-partition primary-index set here: partition columns don't exist in the parquet
 * file, so pushing a filter that references one crashes libcudf's reader. DuckDB already applies
 * partition filters at the file-list level when hive_partitioning is enabled, so dropping them
 * here is safe.
 *
 * @p boolean_substituted_primary_indices names columns that arrive already reduced to a BOOL8
 * predicate result (see @ref extract_string_equality_pushdown and
 * @c sirius::decode_equality_pushdown). Their filter is not re-expressed as a comparison; the
 * batch column *is* the answer, so it contributes a bare boolean reference instead. Passing a
 * column here whose batch column is not actually BOOL8 silently mis-types the expression, so
 * callers must confirm the substitution happened for the batch in hand.
 *
 * Returns nullptr if the filter set is empty or contains only unsupported/skipped filter types.
 */
duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices                = {},
  const std::unordered_set<std::size_t>& boolean_substituted_primary_indices = {});

/**
 * @brief Per-column string constants for filters that are pure equality / IN tests.
 *
 * Returns, keyed by column primary index, the value set a column is compared
 * against when its whole pushed-down filter is an equality, an @c IN, or an OR of
 * those over non-null VARCHAR constants (an ANDed @c IS NOT NULL is absorbed —
 * an equality already rejects nulls). Columns with any other filter shape, or a
 * non-string constant, are absent.
 *
 * This is the decision input for pushing a predicate into decompression: such a
 * column can be answered off a dictionary's key set instead of being decoded
 * (@c simpatico::decode_predicate). The caller must additionally confirm the
 * column is never projected — the pushdown replaces its values with a BOOL8
 * mask — and that its compression plan can actually exploit it.
 */
std::unordered_map<std::size_t, std::vector<std::string>> extract_string_equality_pushdown(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types);

/**
 * @brief Numeric range predicates extracted from a scan's pushed-down filters
 * (fused scan-filter pipeline, env gate @c SIRIUS_EXP_FUSED_SCAN_FILTER).
 *
 * Bounds live in the DECODED integer domain — the values a bitpack decoder
 * reconstructs: DATE → stored day count, DECIMAL → unscaled integer at the
 * *column's* scale, plain integers as-is. Inclusive both ends; @c lo > @c hi is
 * a provably empty range (e.g. an equality against a constant the column's
 * scale cannot represent) and legitimately selects nothing.
 */
struct numeric_range_extraction {
  /// Inclusive [lo,hi] per filtered column, keyed by column primary index.
  std::unordered_map<std::size_t, sirius::codegen::range_predicate> ranges;
  /// True iff EVERY row-restricting filter in the set was converted into
  /// @c ranges. This is the iteration-1 gate for decode-side compaction: rows
  /// may only be dropped during decompression when the extracted ranges are
  /// the *whole* filter. When false, @c ranges is left empty — one unsupported
  /// conjunct sends the entire scan down today's decode-then-filter path.
  bool all_conjuncts_convertible = false;
};

/**
 * @brief Extract per-column numeric range predicates from a TableFilterSet.
 *
 * Recognizes CONSTANT_COMPARISON (<, <=, >, >=, =) and CONJUNCTION_AND of
 * those, on DATE / DECIMAL(≤18) / signed- and small-unsigned-integer columns,
 * intersecting all conjuncts into one inclusive [lo,hi] per column. Strict
 * inequalities tighten the bound by one; decimal constants are rescaled to the
 * column's scale exactly (floor/ceil on the correct side when the constant has
 * more fractional digits than the column can store).
 *
 * Top-level OPTIONAL_FILTER and IS_NOT_NULL filters are non-restricting on the
 * scan's post-decompress path — @ref convert_table_filters_to_expression drops
 * them — so they are skipped here without blocking the gate; likewise for
 * filters on @p skip_primary_indices (hive partitions, enforced at file-list
 * level). Any other unconvertible filter clears the result and reports
 * @c all_conjuncts_convertible == false.
 *
 * The caller must still confirm, per batch, that every filtered column's
 * compression plan can evaluate its range during decode
 * (@c sirius::build_fused_scan_directives) — this function only speaks for the
 * filter shapes and constant types.
 */
numeric_range_extraction extract_numeric_range_pushdown(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::unordered_set<std::size_t>& skip_primary_indices = {});

/**
 * @brief Bridge a DuckDB filter expression through sirius::ast::from_duckdb into the
 * cuDF-AST translator's column-name pathway.
 *
 * Converts @p expr to a Sirius AST node (returning std::nullopt if the expression cannot be
 * lowered), then forwards it to @p translator.translate_expression_with_names. The helper owns
 * the intermediate Sirius AST node for the duration of the call.
 *
 * This is a free function (not a translator member) so the translator's public surface accepts
 * only Sirius AST; the DuckDB-to-AST conversion lives at the scan boundary.
 *
 * @param translator The translator to lower the converted Sirius AST into a cuDF AST.
 * @param expr The DuckDB filter expression to translate.
 * @param resolver Function mapping a reference column index to a column name string.
 * @return The translated expression, or std::nullopt if conversion or translation failed.
 */
std::optional<gpu_expression_translator::translated_expression>
translate_duckdb_expression_with_names(
  gpu_expression_translator& translator,
  duckdb::Expression const& expr,
  gpu_expression_translator::column_name_resolver_fxn resolver);

}  // namespace sirius::op
