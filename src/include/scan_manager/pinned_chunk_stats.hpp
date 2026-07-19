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
