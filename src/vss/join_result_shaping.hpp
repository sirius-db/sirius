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

/**
 * @file join_result_shaping.hpp
 * @brief Turning a per-row top-k block into the join mode the query asked for.
 *
 * Every mode starts from the same thing: a row-major `[n_left, k]` block of neighbor ids
 * and distances, which is what both the fused fold and the split path's merge produce.
 * What differs is which of those `n_left * k` pairs survive, and that is the only thing
 * these helpers decide.
 *
 * The result of each is a `[left_row, neighbor_id, distance]` triple whose row count no
 * longer has to be a multiple of `n_left` -- carrying `left_row` explicitly is what lets
 * threshold and global top-k share one materialize stage with per-row top-k.
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>

namespace sirius::vss {

/// A shaped join result: parallel columns, one row per surviving pair.
struct shaped_join_result {
  std::unique_ptr<cudf::column> left_rows;  ///< INT32 index into the left batch
  std::unique_ptr<cudf::column> neighbors;  ///< INT64 global right-table row id
  std::unique_ptr<cudf::column> distances;  ///< FLOAT32 distance
};

/**
 * @brief The left-row index for each element of a row-major `[n_left, k]` block.
 *
 * Element `i` of the flattened block belongs to left row `i / k`. Materialize used to
 * rely on that identity implicitly by repeating each left row k times; making it a column
 * is what lets the row count stop being a multiple of `n_left`.
 */
std::unique_ptr<cudf::column> make_left_row_index(std::int64_t n_left,
                                                  std::int64_t k,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr);

/**
 * @brief Per-row top-k: keep every pair, tagged with its left row.
 */
shaped_join_result shape_per_row_top_k(std::unique_ptr<cudf::column> neighbors,
                                       std::unique_ptr<cudf::column> distances,
                                       std::int64_t n_left,
                                       std::int64_t k,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Global top-k: the @p k_global closest pairs over the whole block, ignoring which
 *        left row they came from.
 *
 * Correct to run over a per-row top-k block rather than over all `n_left * n_right` pairs:
 * a left row can contribute at most `k_global` pairs to the global top-k, so searching each
 * left row to depth `k_global` cannot miss one. The caller is responsible for using
 * `k = k_global` in the fold that produced @p distances.
 */
shaped_join_result shape_global_top_k(cudf::column_view const& neighbors,
                                      cudf::column_view const& distances,
                                      std::int64_t n_left,
                                      std::int64_t k,
                                      std::int64_t k_global,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Threshold (range) join: every pair at distance <= @p max_distance.
 *
 * Output is ragged -- a left row may contribute anywhere from zero to @p k pairs.
 *
 * The candidate block is only searched to depth `k`, so a left row with more than `k`
 * neighbours inside the threshold would be silently truncated. @p truncated reports that
 * instead: it is set when any left row's k-th neighbour is itself within the threshold,
 * which is exactly the condition under which the answer may be incomplete. Reading it
 * synchronizes @p stream.
 */
shaped_join_result shape_threshold(cudf::column_view const& neighbors,
                                   cudf::column_view const& distances,
                                   std::int64_t n_left,
                                   std::int64_t k,
                                   float max_distance,
                                   bool& truncated,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace sirius::vss
