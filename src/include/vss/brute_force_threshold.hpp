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

#include "vss/cudf_raft_interop.hpp"

#include <cudf/column/column.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/resource_ref.hpp>

#include <cuvs/distance/distance.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace sirius::vss {

/**
 * @brief Result of a threshold (radius) join between one query batch and one
 * dataset batch.
 *
 * Unlike @ref knn_result, the output is a ragged edge list: one row per
 * surviving (query, dataset) pair, in no particular order. All three columns
 * share the same length @p n_edges. There is no `k` — a query row contributes as
 * many edges as it has dataset rows within @p eps (possibly zero).
 */
struct threshold_join_result {
  std::unique_ptr<cudf::column> query_rows;  ///< INT64 local query-batch row index.
  std::unique_ptr<cudf::column> neighbors;   ///< INT64 local dataset-batch row index.
  std::unique_ptr<cudf::column> distances;   ///< FLOAT32 distance for that pair.
  int64_t n_edges;
};

/**
 * @brief Exact threshold (radius) join via tiled GEMM + stream compaction.
 *
 * For every query vector, emits every dataset vector whose distance under @p
 * metric is within @p eps. This is a fork of cuVS's `tiled_brute_force_knn`
 * loop: it reuses the same tiled `pairwise_distance` (GEMM) + one-time norm
 * correction, then replaces the per-tile `select_k` with a `copy_if` on the
 * threshold. Because "within eps" is independent across column tiles, no
 * cross-tile merge is needed, so the column-tile reconciliation that top-k
 * requires is simply gone.
 *
 * Memory stays tiled: only a [tile_rows x tile_cols] distance block plus the
 * ragged output (proportional to the number of surviving edges, not m*n) is
 * held. The dense [m, n] matrix is never materialized.
 *
 * @p eps is in @p metric's own distance units, matching what the tile produces:
 *   - L2SqrtExpanded / L2SqrtUnexpanded: a Euclidean radius.
 *   - CosineExpanded: a cosine-distance cutoff `1 - min_similarity` (handled
 *     natively via GEMM — no unit-normalization needed, unlike ball_cover eps_nn).
 * The keep direction follows `cuvs::distance::is_min_close(metric)`.
 *
 * Output columns are allocated through @p mr. Enqueued on @p res's stream and
 * not synchronized before returning (same contract as @ref brute_force_knn).
 *
 * @param res        Caller-owned RAFT resources; the join runs on its stream.
 * @param dataset    Row-major dataset (right) batch to search.
 * @param queries    Row-major query (left) batch.
 * @param eps        Threshold in @p metric's distance units.
 * @param metric     Distance metric.
 * @param mr         Device resource for the output columns.
 * @param tile_rows  Row tile size; 0 = pick automatically.
 * @param tile_cols  Column tile size; 0 = pick automatically.
 * @return Ragged edge list of (query_row, dataset_row, distance).
 */
threshold_join_result brute_force_threshold(
  raft::device_resources const& res,
  dataset_matrix_view dataset,
  dataset_matrix_view queries,
  float eps,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded,
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref(),
  std::size_t tile_rows               = 0,
  std::size_t tile_cols               = 0);

}  // namespace sirius::vss
