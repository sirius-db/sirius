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

#include "vss/brute_force_search.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/neighbors/brute_force.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sirius::vss {

knn_result brute_force_knn(raft::device_resources const& res,
                           dataset_matrix_view dataset,
                           dataset_matrix_view queries,
                           int64_t k,
                           cuvs::distance::DistanceType metric,
                           rmm::device_async_resource_ref mr)
{
  namespace bf = cuvs::neighbors::brute_force;

  auto const n_rows    = dataset.extent(0);
  auto const n_queries = queries.extent(0);

  CUDF_EXPECTS(dataset.extent(1) == queries.extent(1),
               "VSS dataset and query dimensionality must match");
  CUDF_EXPECTS(k >= 1 && k <= n_rows, "VSS k must satisfy 1 <= k <= n_rows");

  // Everything runs on res's stream so the search, the output allocations, and
  // the caller's downstream work all order on that single stream. res is
  // caller-owned and reused across chunks, so its handle setup is paid once.
  auto const stream = raft::resource::get_cuda_stream(res);

  // Build the brute-force index. With the non-owning dataset view this stores a
  // reference to Sirius-owned memory and precomputes norms.
  bf::index_params index_params;
  index_params.metric = metric;
  auto index          = bf::build(res, index_params, dataset);

  // Allocate flattened [n_queries * k] outputs through the caller's resource
  // (mr) so they are reserved against the owning memory space.
  auto const out_size = static_cast<cudf::size_type>(n_queries * k);
  auto neighbors_col  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);

  auto neighbors_view = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
    neighbors_col->mutable_view().data<int64_t>(), n_queries, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
    distances_col->mutable_view().data<float>(), n_queries, k);

  bf::search_params search_params;
  bf::search(res, search_params, index, queries, neighbors_view, distances_view);

  // The search runs async on res's stream.
  return knn_result{std::move(neighbors_col), std::move(distances_col), n_queries, k};
}

std::size_t brute_force_peak_scratch_bytes(
  int64_t n_queries, int64_t n_dataset, int64_t dim, int64_t k, cuvs::distance::DistanceType metric)
{
  using cuvs::distance::DistanceType;
  constexpr std::size_t kFloat  = sizeof(float);
  constexpr std::size_t kIndex  = sizeof(int64_t);
  constexpr std::size_t kFloor  = std::size_t{1} << 20;  // 1 MiB
  constexpr std::size_t k512MiB = std::size_t{512} << 20;
  constexpr std::size_t k1GiB   = std::size_t{1} << 30;

  auto const m = static_cast<std::size_t>(std::max<int64_t>(n_queries, 1));
  auto const n = static_cast<std::size_t>(std::max<int64_t>(n_dataset, 1));

  // Query + index norms; the expanded/cosine paths compute both, and it is a
  // negligible upper bound for the unexpanded ones.
  std::size_t const norms = (m + n) * kFloat;

  bool const l2_family =
    metric == DistanceType::L2Unexpanded || metric == DistanceType::L2SqrtUnexpanded ||
    metric == DistanceType::L2Expanded || metric == DistanceType::L2SqrtExpanded;

  // Fused path (fusedL2Knn): distances stay on-chip, so norms are the only scratch.
  if (k <= 64 && l2_family) { return norms + kFloor; }

  // Tiled path (tiled_brute_force_knn): mirror chooseTileSize, then size the
  // pairwise-distance tile + per-tile top-k buffers.
  std::size_t const tile_rows = std::min<std::size_t>(dim <= 32 ? 1024 : 512, m);
  std::size_t tile_cols;
  if (tile_rows * n * 2 * kFloat <= k512MiB) {
    tile_cols = n;  // whole width fits the initial budget: no column tiling
  } else {
    // Assume the largest (>8 GB GPU) budget; under pressure cuVS tiles smaller,
    // i.e. uses less, so this stays an upper bound.
    tile_cols = std::min<std::size_t>(k1GiB / (2 * kFloat * tile_rows), n);
  }

  std::size_t const num_col_tiles = (n + tile_cols - 1) / tile_cols;
  std::size_t const temp_out_cols =
    static_cast<std::size_t>(std::max<int64_t>(k, 1)) * num_col_tiles;
  std::size_t const temp_distances = tile_rows * tile_cols * kFloat;
  std::size_t const temp_out       = tile_rows * temp_out_cols * (kFloat + kIndex);
  // select_k workspace over [tile_rows x tile_cols]; empirical, ~1x the tile.
  std::size_t const select_k_ws = temp_distances;

  return temp_distances + temp_out + norms + select_k_ws + kFloor;
}

}  // namespace sirius::vss
