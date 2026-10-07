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

}  // namespace sirius::vss
