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

// sirius
#include <compression/compressed_scan.hpp>
#include <expression/ast/node.hpp>
#include <helper/logical_type.hpp>

// sirius (table_filter_conjunct)
#include <op/scan/scan_utils.hpp>

// duckdb
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/table_filter.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::op {

/**
 * @brief One scan's pushed-down filter, digested once into what a decompressor
 * can evaluate for itself, keyed by column primary index.
 *
 * The scan hands over its whole filter and gets back the parts that survive as
 * decode-time work, keyed by column primary index. What any given chunk can
 * actually do with them is decided later and elsewhere (see
 * @c sirius::decompression_pushdown_scan) — this speaks only for filter shapes and constant
 * types.
 */
struct scan_filter_analysis {
  /// Columns whose ENTIRE filter is an equality / IN over non-null string
  /// constants (an ANDed IS NOT NULL is absorbed — an equality already rejects
  /// nulls), mapped to the values tested against. Feeds
  /// @c scan_decode_request::column_entry::equals_any, which states what the
  /// decoder does with them and why only unprojected columns qualify.
  std::unordered_map<std::size_t, std::vector<std::string>> equality_sets;

  /// Inclusive bounds per filtered column. Always a sound conjunctive
  /// over-approximation of that column's filter: every row the bounds reject is
  /// a row the full filter rejects.
  std::unordered_map<std::size_t, sirius::decode_range> ranges;

  /// True iff EVERY row-restricting conjunct became a range; see
  /// @c scan_decode_request::ranges_cover_whole_filter for what that buys. When
  /// false, @c ranges is still usable — the decode then filters only partially.
  bool ranges_cover_whole_filter = false;
};

/**
 * @brief Digest @p filters into the decode-time work it can support.
 *
 * Ranges recognize constant comparisons (<, <=, >, >=, =) and AND-trees of them
 * on DATE / DECIMAL(≤18) / signed- and small-unsigned-integer columns,
 * intersecting all conjuncts into one inclusive range per column. Strict
 * inequalities tighten by one; a decimal constant finer than the column's scale
 * is floored or ceiled on the side that cannot drop a surviving row.
 *
 * The EXPRESSION_FILTER shape `CAST(date_col AS TIMESTAMP[_S|_MS|_NS]) CMP
 * <timestamp constant>` — what DuckDB's constant folding makes of qgen-style
 * date arithmetic — is recognized too, in either operand order, by lowering
 * the constant to the stored-day domain. That range is exact, so it keeps
 * full coverage; TRY_CAST, TIMESTAMP_TZ, and ±infinity constants are refused.
 *
 * Filters that do not restrict the emitted rows are skipped WITHOUT clearing
 * coverage — OPTIONAL_FILTER and IS_NOT_NULL, which
 * @ref convert_table_filters_to_expression also drops, and @p
 * skip_primary_indices (hive partitions, enforced at file-list level). Any other
 * unconvertible conjunct clears only @c ranges_cover_whole_filter.
 *
 * Equality sets are collected only for @p filter_only_primary_indices.
 */
scan_filter_analysis analyze_scan_filters(
  const duckdb::TableFilterSet& filters,
  const duckdb::vector<duckdb::ColumnIndex>& column_ids,
  const duckdb::vector<sirius::logical_type>& returned_types,
  const std::unordered_set<std::size_t>& skip_primary_indices        = {},
  const std::unordered_set<std::size_t>& filter_only_primary_indices = {});

/**
 * @brief Turn @p analysis into the request one decoder can act on.
 *
 * @p primary_index_by_slot maps each column the decode will produce onto the
 * scan's column primary index, so the request comes out parallel to that
 * column list. Analysis entries that map to no slot are dropped — a partition
 * filter, say, which is enforced elsewhere.
 */
sirius::pushdown_request build_pushdown_request(scan_filter_analysis const& analysis,
                                                std::span<const std::size_t> primary_index_by_slot);

/**
 * @brief The part of a scan's filter that still has to be evaluated after the
 * decode, as a predicate.
 *
 * Built once at bind from the filter's top-level conjuncts, each pre-lowered to
 * Sirius AST, so per batch the residual is assembled by choosing a form per
 * conjunct — nothing is rebuilt or converted on the batch path.
 *
 * A column answered in place arrives as the BOOL8 answer rather than its
 * declared type, so its conjunct MUST become a bare reference to it; re-running
 * the comparison would compare a mask against a string constant. Which columns
 * that happened to is a per-batch fact the decoder reports
 * (@c sirius::pushdown_outcome::predicate_columns).
 */
class residual_filter {
 public:
  residual_filter() = default;

  /// @p answerable_batch_positions are the columns the scan is willing to
  /// receive as an answer instead of values; a conjunct on any other column
  /// always keeps its comparison form.
  ///
  /// @throws std::runtime_error if a conjunct cannot be lowered to Sirius AST.
  /// That is a bind-time failure by design: an unlowerable conjunct cannot be
  /// evaluated on any batch, and silently dropping it would return unfiltered
  /// rows.
  residual_filter(std::vector<table_filter_conjunct> conjuncts,
                  std::unordered_set<std::size_t> const& answerable_batch_positions);

  /// True when there is nothing to evaluate at all (no filter, or every
  /// conjunct was skipped as non-restricting).
  [[nodiscard]] bool empty() const noexcept { return _conjuncts.empty(); }

  /// The predicate to evaluate over a batch in which @p answered_positions
  /// arrived as BOOL8 answers. With @p answers_enforced the decode also APPLIED
  /// them, so those conjuncts leave the residual entirely.
  ///
  /// Null means nothing is left to evaluate — which the caller must read as
  /// "these rows are already filtered", NOT as "no filtering needed".
  [[nodiscard]] std::unique_ptr<sirius::ast::node> against(
    std::vector<std::size_t> const& answered_positions, bool answers_enforced = false) const;

 private:
  struct conjunct {
    std::unique_ptr<sirius::ast::node> comparison;
    /// Set when this conjunct's column can arrive as the answer; the batch
    /// position that answer occupies.
    std::optional<std::size_t> answered_at;
  };
  std::vector<conjunct> _conjuncts;
};

}  // namespace sirius::op
