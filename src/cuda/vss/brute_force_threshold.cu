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

#include "vss/brute_force_threshold.hpp"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>

#include <cuvs/distance/distance.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace sirius::vss {

namespace {

// Mirror of cuVS faiss_select::chooseTileSize, kept here so this wrapper depends
// only on public headers. Same shape as brute_force_peak_scratch_bytes(): a row
// tile sized by dimensionality, and a column tile capped to a fixed byte budget.
void choose_tile_size(std::size_t m,
                      std::size_t n,
                      std::size_t d,
                      std::size_t& tile_rows,
                      std::size_t& tile_cols)
{
  constexpr std::size_t k512MiB = std::size_t{512} << 20;
  constexpr std::size_t k1GiB   = std::size_t{1} << 30;
  constexpr std::size_t kFloat  = sizeof(float);

  tile_rows = std::min<std::size_t>(d <= 32 ? 1024 : 512, std::max<std::size_t>(m, 1));
  if (tile_rows * n * 2 * kFloat <= k512MiB) {
    tile_cols = n;  // whole width fits the budget: no column tiling
  } else {
    tile_cols = std::min<std::size_t>(k1GiB / (2 * kFloat * tile_rows), n);
  }
  tile_cols = std::max<std::size_t>(tile_cols, 1);
}

// Wrap a filled device_uvector as an owning cudf column with no copy.
template <typename T>
std::unique_ptr<cudf::column> uvector_to_column(rmm::device_uvector<T>&& v, cudf::data_type dt)
{
  auto const size = static_cast<cudf::size_type>(v.size());
  return std::make_unique<cudf::column>(
    dt, size, v.release(), rmm::device_buffer{}, 0);
}

}  // namespace

threshold_join_result brute_force_threshold(raft::device_resources const& res,
                                           dataset_matrix_view dataset,
                                           dataset_matrix_view queries,
                                           float eps,
                                           cuvs::distance::DistanceType metric,
                                           rmm::device_async_resource_ref mr,
                                           std::size_t tile_rows,
                                           std::size_t tile_cols)
{
  auto const stream = raft::resource::get_cuda_stream(res);
  auto const policy = raft::resource::get_thrust_policy(res);

  auto const m = static_cast<std::size_t>(queries.extent(0));
  auto const n = static_cast<std::size_t>(dataset.extent(0));
  auto const d = static_cast<std::size_t>(dataset.extent(1));
  CUDF_EXPECTS(queries.extent(1) == dataset.extent(1),
               "VSS query and dataset dimensionality must match");

  bool const select_min = cuvs::distance::is_min_close(metric);

  if (tile_rows == 0 || tile_cols == 0) { choose_tile_size(m, n, d, tile_rows, tile_cols); }

  // --- one-time norms, exactly as tiled_brute_force_knn does ------------------
  // For the expanded L2/cosine metrics, pairwise_distance produces raw inner
  // products and we correct them with precomputed row/col norms per tile. For
  // the unexpanded metrics, pairwise_distance already yields the true distance.
  auto pairwise_metric = metric;
  rmm::device_uvector<float> q_norms(0, stream);
  rmm::device_uvector<float> d_norms(0, stream);
  bool const expanded = metric == cuvs::distance::DistanceType::L2Expanded ||
                        metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                        metric == cuvs::distance::DistanceType::CosineExpanded;
  if (expanded) {
    q_norms.resize(m, stream);
    d_norms.resize(n, stream);
    bool const cosine = metric == cuvs::distance::DistanceType::CosineExpanded;
    // cosine wants the L2 norm; L2-expanded wants the squared norm.
    auto q_mat = raft::make_device_matrix_view<const float, int64_t>(queries.data_handle(), m, d);
    auto d_mat = raft::make_device_matrix_view<const float, int64_t>(dataset.data_handle(), n, d);
    if (cosine) {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        res, q_mat, raft::make_device_vector_view<float, int64_t>(q_norms.data(), m), raft::sqrt_op{});
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        res, d_mat, raft::make_device_vector_view<float, int64_t>(d_norms.data(), n), raft::sqrt_op{});
    } else {
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        res, q_mat, raft::make_device_vector_view<float, int64_t>(q_norms.data(), m));
      raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
        res, d_mat, raft::make_device_vector_view<float, int64_t>(d_norms.data(), n));
    }
    pairwise_metric = cuvs::distance::DistanceType::InnerProduct;
  }

  // --- growing ragged output --------------------------------------------------
  // Edges accumulate across tiles. We track a logical size and grow capacity
  // geometrically so the total copy work stays O(n_edges), not O(n_edges * tiles).
  rmm::device_uvector<int64_t> out_q(0, stream, mr);
  rmm::device_uvector<int64_t> out_n(0, stream, mr);
  rmm::device_uvector<float> out_dist(0, stream, mr);
  std::size_t out_size = 0;
  auto const grow_to = [&](std::size_t need) {
    if (need <= out_q.capacity()) return;
    std::size_t cap = std::max<std::size_t>(need, out_q.capacity() * 2);
    out_q.reserve(cap, stream);
    out_n.reserve(cap, stream);
    out_dist.reserve(cap, stream);
  };

  // Per-tile scratch: the distance block and a staging buffer of kept flat idx.
  rmm::device_uvector<float> temp_dist(tile_rows * tile_cols, stream);
  rmm::device_uvector<int64_t> kept(tile_rows * tile_cols, stream);

  auto const* q_ptr = queries.data_handle();
  auto const* d_ptr = dataset.data_handle();

  for (std::size_t i = 0; i < m; i += tile_rows) {
    std::size_t const qs = std::min(tile_rows, m - i);
    for (std::size_t j = 0; j < n; j += tile_cols) {
      std::size_t const cs   = std::min(tile_cols, n - j);
      std::size_t const ntil = qs * cs;

      // (1) GEMM: raw inner products (expanded) or true distances (unexpanded).
      cuvs::distance::pairwise_distance(
        res,
        raft::make_device_matrix_view<const float, int64_t>(q_ptr + i * d, qs, d),
        raft::make_device_matrix_view<const float, int64_t>(d_ptr + j * d, cs, d),
        raft::make_device_matrix_view<float, int64_t>(temp_dist.data(), qs, cs),
        pairwise_metric,
        2.0f);

      // (2) Norm correction → true distance for the expanded metrics. Copied
      //     faithfully from tiled_brute_force_knn's map_offset epilogue.
      if (expanded) {
        auto* dist        = temp_dist.data();
        auto const* qn    = q_norms.data();
        auto const* dn    = d_norms.data();
        bool const sqrt_l2 = metric == cuvs::distance::DistanceType::L2SqrtExpanded;
        bool const cosine  = metric == cuvs::distance::DistanceType::CosineExpanded;
        raft::linalg::map_offset(
          res,
          raft::make_device_vector_view<float, int64_t>(dist, ntil),
          [=] __device__(int64_t idx) {
            int64_t row = i + (idx / cs);
            int64_t col = j + (idx % cs);
            if (cosine) { return 1.0f - dist[idx] / (qn[row] * dn[col]); }
            // L2 expanded, inlined from cuVS l2_exp_cutlass_op: qn/dn are squared
            // norms, dist holds the inner product. d^2 = |q|^2 + |d|^2 - 2<q,d>.
            float outv = qn[row] + dn[col] - 2.0f * dist[idx];
            // Self-neighbor round-off guard: when the two rows have equal norm and
            // the residual is within float precision, snap it to exactly zero.
            if (outv * outv < 1e-6f && qn[row] == dn[col]) { outv = 0.0f; }
            return sqrt_l2 ? sqrtf(outv > 0.0f ? outv : 0.0f) : outv;
          });
      }

      // (3) Compaction: keep flat indices whose distance passes the threshold.
      //     The `<= / >=` is the predicate; copy_if turns "which pass" into a
      //     compact list, so we never keep the dense tile beyond this pass.
      auto const first = thrust::make_counting_iterator<int64_t>(0);
      auto const* dist = temp_dist.data();
      auto* kept_end   = thrust::copy_if(
        policy, first, first + static_cast<int64_t>(ntil), kept.data(),
        [=] __device__(int64_t f) {
          float v = dist[f];
          return select_min ? (v <= eps) : (v >= eps);
        });
      auto const tile_nnz = static_cast<std::size_t>(kept_end - kept.data());
      if (tile_nnz == 0) continue;

      // (4) Append this tile's survivors as global (query_row, dataset_row, dist).
      std::size_t const base = out_size;
      grow_to(base + tile_nnz);
      out_q.resize(base + tile_nnz, stream);
      out_n.resize(base + tile_nnz, stream);
      out_dist.resize(base + tile_nnz, stream);
      out_size = base + tile_nnz;

      auto const* kept_idx = kept.data();
      auto* oq             = out_q.data() + base;
      auto* on             = out_n.data() + base;
      auto* od             = out_dist.data() + base;
      thrust::for_each(
        policy, thrust::make_counting_iterator<int64_t>(0),
        thrust::make_counting_iterator<int64_t>(static_cast<int64_t>(tile_nnz)),
        [=] __device__(int64_t t) {
          int64_t f = kept_idx[t];
          oq[t]     = i + (f / cs);  // local query-batch row
          on[t]     = j + (f % cs);  // local dataset-batch row
          od[t]     = dist[f];
        });
    }
  }

  return threshold_join_result{uvector_to_column(std::move(out_q), cudf::data_type{cudf::type_id::INT64}),
                               uvector_to_column(std::move(out_n), cudf::data_type{cudf::type_id::INT64}),
                               uvector_to_column(std::move(out_dist), cudf::data_type{cudf::type_id::FLOAT32}),
                               static_cast<int64_t>(out_size)};
}

}  // namespace sirius::vss
