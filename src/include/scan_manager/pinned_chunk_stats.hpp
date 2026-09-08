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

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>

#include <cstddef>
#include <vector>

namespace sirius::scan_manager {

/**
 * @brief Compute per-column min/max (zone-map) statistics for a pinned chunk.
 *
 * A null entry indicates the absence of statistics for that column chunk and never prunes.
 * Supported types:
 *  - integers <= 8B
 *  - cudf TIMESTAMP_DAYS (duckdb DATE)
 *  - cudf TIMESTAMP_MICROSECONDS (duckdb TIMESTAMP)
 */
[[nodiscard]] std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>> compute_pinned_chunk_stats(
  cudf::table_view const& chunk,
  duckdb::vector<duckdb::LogicalType> const& column_types,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Per-group zone-map statistics for ONE pinned chunk.
 *
 * A group is @c group_rows consecutive rows of the chunk — by construction a whole number of
 * simpatico 1024-row decode chunks, so group g covers decode chunks [g*G, (g+1)*G) and a
 * surviving group expands into a chunk-id list with a shift. The final group of a chunk may be
 * short; the parquet pin path does not pack whole 122,880-row units.
 *
 * @c groups is group-major: groups[g][i] = statistics of chunk column i over group g, with the
 * same "null cell never prunes" contract as @ref compute_pinned_chunk_stats. Empty @c groups
 * means capture did not run or produced nothing usable.
 */
struct chunk_group_stats {
  std::size_t group_rows{0};
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> groups;

  [[nodiscard]] bool empty() const noexcept { return groups.empty(); }
  [[nodiscard]] std::size_t group_count() const noexcept { return groups.size(); }
};

/**
 * @brief Compute per-group min/max (zone-map) statistics for a pinned chunk.
 *
 * Same type allowlist and same soundness contract as @ref compute_pinned_chunk_stats, but at
 * @p group_rows granularity instead of one cell per chunk. Uses a single segmented reduction per
 * column rather than one reduction per group: measured at 1.7x the cost of the whole-chunk
 * capture for a 189M-row chunk at group_rows=8192 (see CHUNK_SKIPPING_PLAN.md §4.3.0).
 *
 * Null handling matches the coarse capture's precision deliberately: a column with no nulls marks
 * every group CANNOT_HAVE_NULL_VALUES, otherwise every group is marked "may have nulls". Per-group
 * exact null counts would need a second segmented reduction and nothing consumes them yet.
 *
 * @p group_rows must be non-zero; a zero or out-of-range value yields an empty result (no stats,
 * never prunes) rather than throwing.
 */
[[nodiscard]] chunk_group_stats compute_pinned_group_stats(
  cudf::table_view const& chunk,
  duckdb::vector<duckdb::LogicalType> const& column_types,
  std::size_t group_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Zone-map sidecar of a pinned entry: pin-time DuckDB types plus per-chunk BaseStatistics
 * for each cached column, positional with the entry's cache_info.column_ids.
 *
 * The representation guarantees "all columns or nothing": from_capture normalizes any
 * malformed capture to absent, and merge growth either appends consistently or degrades to
 * absent.
 */
class pinned_zone_maps {
 public:
  pinned_zone_maps() = default;

  /**
   * @brief Normalize a pin-time capture into a sidecar.
   *
   * @p column_types must cover exactly @p n_columns positions and @p chunk_stats must be an
   * n_chunks × n_columns matrix (chunk-major, as compute_pinned_chunk_stats emits per materialized
   * chunk). The matrix is pivoted to column-major for the sidecar representation.
   */
  [[nodiscard]] static pinned_zone_maps from_capture(
    duckdb::vector<duckdb::LogicalType> column_types,
    std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> chunk_stats,
    std::size_t n_columns,
    std::size_t n_chunks);

  [[nodiscard]] bool has_stats() const noexcept { return !_column_types.empty(); }
  [[nodiscard]] std::size_t column_count() const noexcept { return _column_types.size(); }
  [[nodiscard]] duckdb::LogicalType const& column_type(std::size_t pos) const
  {
    return _column_types[pos];
  }
  [[nodiscard]] duckdb::BaseStatistics const* cell(std::size_t pos, std::size_t chunk) const
  {
    if (pos >= _column_stats.size() || chunk >= _column_stats[pos].size()) { return nullptr; }
    return _column_stats[pos][chunk].get();
  }

  /**
   * @brief Merge mirroring: move @p incoming's column @p incoming_pos (pin-time type + per-chunk
   * stats row) onto the end of this sidecar — call exactly when the data merge appends that column
   * to the entry's cache_info.column_ids.
   *
   * If either side is absent, @p incoming_pos is out of range, or the chunk counts disagree, this
   * sidecar degrades to absent and remains that way.
   */
  void append_column_from(pinned_zone_maps& incoming, std::size_t incoming_pos);

  /**
   * @brief Rebuild a sidecar positionally from @p incoming: result position p takes incoming's
   * column @p incoming_pos_by_pos[p] (pin-time type + per-chunk stats row, moved out).
   *
   * Returns an absent sidecar when @p incoming is absent, the mapping is empty, a position is out
   * of range, or a position repeats (its row was already moved out). Used by the GPU re-pin merge
   * to adopt a fresh capture onto a statless entry whose column order may differ from the
   * incoming pin's.
   */
  [[nodiscard]] static pinned_zone_maps remap(pinned_zone_maps incoming,
                                              std::vector<std::size_t> const& incoming_pos_by_pos);

 private:
  // Invariants: _column_types.size() == _column_stats.size();
  //             every inner vector shares one chunk count;
  //             empty == absent.
  duckdb::vector<duckdb::LogicalType> _column_types;
  // Column-major: _column_stats[i][j] = column i of chunk j
  std::vector<std::vector<duckdb::unique_ptr<duckdb::BaseStatistics>>> _column_stats;
};

/**
 * @brief One column's per-group min/max cells, packed as parallel typed arrays.
 *
 * The group index needs a representation that scales: SF1000 lineitem at G = 8 has ~732,000
 * groups, and the @ref chunk_provably_empty path costs ~83 ns per cell (a @c BaseStatistics copy
 * plus a virtual @c CheckStatistics), i.e. ~61 ms of plan time per filter column per query. This
 * holds the same information in three flat arrays and evaluates a filter that was lowered once.
 *
 * Every supported type (integers <= 8 B, DATE, TIMESTAMP) fits in 8 bytes, so bounds are stored
 * as the raw carrier: signed integers, DATE days and TIMESTAMP micros as themselves; unsigned
 * integers as their bit pattern, reinterpreted via @c is_unsigned at comparison time (a UBIGINT
 * above INT64_MAX stores as negative and compares correctly as uint64).
 *
 * @c valid is false for a cell with no statistics, which never prunes — the same contract as a
 * null @c BaseStatistics.
 */
struct packed_column_bounds {
  duckdb::LogicalType type;
  bool is_unsigned{false};
  /// True when the whole column had no nulls at capture time (see compute_pinned_group_stats).
  bool column_has_no_nulls{false};
  std::vector<std::int64_t> mins;
  std::vector<std::int64_t> maxs;
  std::vector<bool> valid;

  [[nodiscard]] std::size_t size() const noexcept { return mins.size(); }
};

/**
 * @brief A @c TableFilter lowered once into bounds arithmetic, to be evaluated against many
 * @ref packed_column_bounds cells.
 *
 * Lowering fails (returns nullopt) for exactly the shapes @ref filter_safe_for_stats rejects, so
 * an un-lowerable filter prunes nothing rather than pruning wrongly. The evaluation is verified
 * against @ref chunk_provably_empty by a randomized cross-check in the unit tests: the two must
 * agree on every input, since this is the release-mode safety line for dropping data.
 */
class lowered_bound_filter {
 public:
  /// @return nullopt when @p filter is not a shape this can evaluate against @p stats_type.
  [[nodiscard]] static std::optional<lowered_bound_filter> lower(
    duckdb::TableFilter const& filter, duckdb::LogicalType const& stats_type);

  /// True iff no row with bounds [@p min, @p max] can match. @p has_null / @p all_null describe
  /// the cell's null facts, mirroring BaseStatistics' CAN_HAVE_NULL_VALUES / all-null cases.
  [[nodiscard]] bool provably_empty(std::int64_t min,
                                    std::int64_t max,
                                    bool has_null,
                                    bool all_null) const noexcept;

  /// Evaluate every cell of @p bounds, appending surviving indices to @p survivors.
  void select_survivors(packed_column_bounds const& bounds,
                        std::vector<std::uint32_t>& survivors) const;

 private:
  enum class op : std::uint8_t {
    cmp_eq,
    cmp_ne,
    cmp_lt,
    cmp_le,
    cmp_gt,
    cmp_ge,
    in_list,
    is_null,
    is_not_null,
    conj,
    disj
  };
  struct node {
    op kind{op::conj};
    bool is_unsigned{false};
    std::int64_t constant{0};
    /// in_list values / conj+disj children, as [begin, end) into _values or _children.
    std::uint32_t begin{0}, end{0};
  };
  [[nodiscard]] bool eval(std::uint32_t node_index,
                          std::int64_t min,
                          std::int64_t max,
                          bool has_null,
                          bool all_null) const noexcept;

  std::vector<node> _nodes;           ///< _nodes[0] is the root
  std::vector<std::int64_t> _values;  ///< in_list constants
  std::vector<std::uint32_t> _children;
};

/**
 * @brief True iff @p filter is a shape whose CheckStatistics method can be called on a statistics
 * object of type @p stats_type.
 *
 * Positive allow-list, checked recursively:
 *  - constant comparisons (=, !=, <, <=, >, >=) whose constant's LogicalType exactly equals @p
 *    stats_type
 *  - IN whose values are not null and exactly equal to @p stats_type
 *  - IS NULL, IS NOT NULL
 *  - AND, OR, OPTIONAL whose descendants all pass (an OPTIONAL with no child rejects)
 */
[[nodiscard]] bool filter_safe_for_stats(duckdb::TableFilter const& filter,
                                         duckdb::LogicalType const& stats_type);

/**
 * @brief True iff @p filter proves that no row of a chunk with statistics @p stats can match
 * (FILTER_ALWAYS_FALSE).
 */
[[nodiscard]] bool chunk_provably_empty(duckdb::TableFilter const& filter,
                                        duckdb::BaseStatistics const& stats) noexcept;

}  // namespace sirius::scan_manager
