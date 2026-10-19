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

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>

// Device-side glue for the clustered vector join's per-slice loop. Each slice of a clustered
// search runs once per (corpus chunk, cluster) pair, so anything issued per slice is paid about
// a thousand times per query. Built from cuDF calls, that glue synchronized the host on every
// scalar and copied the whole accumulator per slice; these do the same work as single
// asynchronous kernels on the caller's stream, with no allocation of their own.
namespace sirius::vss {

/**
 * @brief out[i, :] = src[rows[i], :] for i in [0, m); both row-major with @p dim columns.
 */
void gather_rows(float const* src,
                 int64_t dim,
                 int64_t const* rows,
                 int64_t m,
                 float* out,
                 rmm::cuda_stream_view stream);

/**
 * @brief Fold one slice's per-row top-k into a running [n x k] accumulator, in place.
 *
 * Part row i belongs to accumulator row rows[i]; its first @p k_eff of @p part_width columns
 * are its nearest-first answer, with ids local to the slice, and @p id_base shifts them into
 * the accumulator's id space. Accumulator rows are nearest-first and stay so. On equal
 * distances the accumulator's entry ranks first, which is the order a merge of
 * [accumulator, part] would give. @p rows must not repeat within a call. With @p id_map the
 * part's local id j becomes id_map[j] instead of j + @p id_base.
 */
void fold_topk_rows(float* acc_distances,
                    int64_t* acc_neighbors,
                    int64_t k,
                    float const* part_distances,
                    int64_t const* part_neighbors,
                    int64_t part_width,
                    int64_t k_eff,
                    int64_t const* rows,
                    int64_t m,
                    int64_t id_base,
                    rmm::cuda_stream_view stream,
                    int64_t const* id_map = nullptr);

/// Probe rows one task of nearest_cluster_topk serves, one warp each.
inline constexpr int32_t kNearestTaskProbes = 16;

/// Up to kNearestTaskProbes probe rows whose nearest cluster is one slice of a chunk's compact
/// rows.
struct nearest_task {
  int64_t row_offset;  ///< the run's first row within the compact rows passed
  int64_t n_rows;
  int64_t edge_begin;  ///< the task's first entry in the routed probe rows
  int32_t n_probes;
  int64_t id_base;  ///< layout id of the slice's first row
};

/**
 * @brief Every task's probe rows searched in full against their slice, in one launch: the
 * nearest-first top-@p k by the compact rows' distance, written as accumulator row
 * routed_rows[e] (whose first min(k, n_rows) entries it replaces), ids as layout ids.
 *
 * The rows are INT8 codes or, with @p f16, half precision, @p row_bytes each (a multiple of 4);
 * @p row_sq / @p probe_sq are their squared norms (INT32, or FLOAT for half rows), indexed by
 * layout id and probe row. Each block loads a slice's rows once into shared memory for all of a
 * task's probe rows, which is what one call per slice could not do without its setup costing far
 * more than the arithmetic. @p k is at most 64.
 */
void nearest_cluster_topk(nearest_task const* tasks,
                          int64_t n_tasks,
                          bool f16,
                          void const* compact,
                          int64_t row_bytes,
                          void const* row_sq,
                          int64_t const* routed_rows,
                          void const* probes,
                          void const* probe_sq,
                          int64_t k,
                          float* acc_distances,
                          int64_t* acc_neighbors,
                          rmm::cuda_stream_view stream);

/**
 * @brief Map a slice's radius edges back to probe rows and corpus rows, in place of a cast.
 *
 * left[e] = rows[query_rows[e]] narrowed to INT32; neighbors[e] += id_base, or with @p id_map
 * neighbors[e] = id_map[neighbors[e]].
 */
void remap_radius_edges(int64_t const* query_rows,
                        int64_t const* rows,
                        int32_t* left,
                        int64_t* neighbors,
                        int64_t n_edges,
                        int64_t id_base,
                        rmm::cuda_stream_view stream,
                        int64_t const* id_map = nullptr);

// A threshold join may be asked for less than every pair within the radius: only the pairs whose
// corpus row comes before the probe row in some order (each pair of a self-join once), and of
// those only each probe row's nearest (a best-match join). Both are decided here, on the pairs a
// search emits, so the rest are neither re-scored nor handed to the operators above.

/// Which pairs a threshold join keeps: those with corpus[id] < probe[row], or with @c differ
/// those with corpus[id] != probe[row] (a self-join's "any other row"). A null @c corpus stands
/// for corpus[id] = id, the order of the ids themselves. A null @c probe keeps every pair.
struct earlier_keys {
  int64_t const* probe{nullptr};   ///< one key per probe row
  int64_t const* corpus{nullptr};  ///< one key per candidate id
  bool differ{false};
};

/// Pairs taken out of a candidate buffer, or a join's best matches.
struct kept_pairs {
  rmm::device_uvector<int32_t> rows;
  rmm::device_uvector<int64_t> ids;
  rmm::device_uvector<float> distances;
};

/**
 * @brief The pairs among @p n that @p keys keep and whose distance is <= @p max_distance, in
 * buffer order, with ids mapped through @p id_map when it is given and distances square-rooted
 * when @p take_sqrt. Blocks until the count is known.
 */
kept_pairs take_pairs(int32_t const* rows,
                      int64_t const* ids,
                      float const* distances,
                      int64_t n,
                      earlier_keys keys,
                      float max_distance,
                      int64_t const* id_map,
                      bool take_sqrt,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr);

// A best-match join keeps, per probe row, only its nearest corpus row within the radius. The
// pairs are folded into one (distance, id) per row as they arrive; and where the search filters
// with inexact distances, only the pairs that can still be a row's nearest are re-scored.

/// No row has a match: best_order[r] sorts after every distance, best_ids[r] = INT64_MAX.
void best_match_reset(uint32_t* best_order,
                      int64_t* best_ids,
                      int64_t n_rows,
                      rmm::cuda_stream_view stream);

/**
 * @brief Fold @p n pairs into each probe row's best match.
 *
 * A pair counts when its distance is <= @p max_distance and @p keys keep it. best_order[r] is row
 * r's smallest distance so far, as bits that order like the float; best_ids[r] the smallest id at
 * that distance. With @p bound, bound[r] is lowered to the row's best distance, which is what a
 * bounded search filters its next pairs with. @p scratch is n_rows entries.
 */
void best_match_fold(int32_t const* rows,
                     int64_t const* ids,
                     float const* distances,
                     int64_t n,
                     float max_distance,
                     earlier_keys keys,
                     uint32_t* best_order,
                     int64_t* best_ids,
                     uint32_t* scratch,
                     int64_t n_rows,
                     float* bound,
                     rmm::cuda_stream_view stream);

/**
 * @brief winner[r] = the index of row r's pair with the smallest distance among the @p n pairs
 * that @p keys keep and limit[r] admits (INT64_MAX when it has none; the lowest index on a tie).
 *
 * The distances are a filter's inexact ones. Re-scoring a row's winner bounds its best exact
 * distance, and the filter limit of that bound then admits few of the row's other pairs.
 */
void best_match_winners(int32_t const* rows,
                        int64_t const* ids,
                        float const* distances,
                        int64_t n,
                        earlier_keys keys,
                        float const* limit,
                        uint32_t* scratch,
                        int64_t* winner,
                        int64_t n_rows,
                        rmm::cuda_stream_view stream);

/**
 * @brief The winners (@p winners) or, without it, every other pair that @p keys keep and
 * limit[row] admits, in buffer order. Blocks until the count is known.
 */
kept_pairs best_match_take(int32_t const* rows,
                           int64_t const* ids,
                           float const* distances,
                           int64_t n,
                           earlier_keys keys,
                           float const* limit,
                           int64_t const* winner,
                           bool winners,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);

/**
 * @brief The rows that have a match, as (row, id, distance) in row order, with ids mapped through
 * @p id_map when it is given and distances square-rooted when @p take_sqrt. Blocks until the
 * count is known.
 */
kept_pairs best_match_emit(uint32_t const* best_order,
                           int64_t const* best_ids,
                           int64_t n_rows,
                           int64_t const* id_map,
                           bool take_sqrt,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr);

}  // namespace sirius::vss
