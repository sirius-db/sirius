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
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <helper/logical_type.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/table_filter.hpp>

// standard library
#include <cstdint>
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

/// Where a filter's column lands in the decoded batch.
enum class filter_column_status : std::uint8_t {
  /// Resolved: the filter constrains @c batch_position of the decoded batch.
  usable,
  /// Not this scan's to apply — a hive partition, which DuckDB already enforced
  /// at the file-list level, or a filter naming no column of the scan.
  skipped,
  /// The column is the scan's but is not in the batch. Every caller has to
  /// decide this one for itself: a filter that must be evaluated cannot
  /// reference a column that was never materialized (a wiring bug worth
  /// failing on), while a filter used only to PRUNE can be dropped silently.
  not_in_batch,
};

struct resolved_filter_column {
  filter_column_status status = filter_column_status::skipped;
  std::size_t primary_index   = 0;
  std::size_t batch_position  = 0;
};

/**
 * @brief Resolve one entry of a TableFilterSet onto the decoded batch.
 *
 * The bookkeeping every walk over a filter set repeats: column_index → primary
 * index → is it ours → which batch column. Shared so the skip rules and the
 * bounds checks are stated once; what each walk does with a given filter TYPE
 * is its own business and deliberately not here.
 */
[[nodiscard]] resolved_filter_column resolve_filtered_column(
  duckdb::idx_t column_index,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices);

/**
 * @brief One top-level conjunct of a scan's pushed-down filter.
 *
 * @c expr is the comparison over batch positions, exactly as
 * @ref convert_table_filters_to_expression would have emitted it. @c primary_index
 * and @c batch_position name the column it constrains, so a caller that learns
 * the column arrived already reduced to a boolean answer can swap this conjunct
 * for a reference to it instead of re-expressing the comparison.
 */
struct table_filter_conjunct {
  std::size_t primary_index  = 0;
  std::size_t batch_position = 0;
  duckdb::unique_ptr<duckdb::Expression> expr;
};

/**
 * @brief Decompose @p filters into its top-level conjuncts.
 *
 * The conjunct set is exactly what @ref convert_table_filters_to_expression
 * ANDs together — same skips (optional / IS NOT NULL / @p skip_primary_indices),
 * same order — because that function is implemented on top of this one. A caller
 * that needs to vary the conjunction per batch decomposes once here rather than
 * rebuilding the whole expression each time.
 *
 * @throws std::runtime_error if a filtered column is not present in the batch.
 */
std::vector<table_filter_conjunct> decompose_table_filters(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
  const std::unordered_set<std::size_t>& skip_primary_indices = {});

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
 * Returns nullptr if the filter set is empty or contains only unsupported/skipped filter types.
 */
duckdb::unique_ptr<duckdb::Expression> convert_table_filters_to_expression(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::vector<std::optional<std::size_t>>& batch_position_by_column_id,
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
