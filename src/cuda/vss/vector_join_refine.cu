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

#include "vss/vector_join_refine.hpp"

#include <cudf/utilities/error.hpp>

#include <cstdint>

namespace sirius::vss {

namespace {

// One thread per (query, neighbor) pair. Recomputes the exact distance directly
// from the two vectors — no norm decomposition, so no near-zero cancellation.
template <bool IsCosine>
__global__ void refine_kernel(const float* __restrict__ queries,
                              const float* __restrict__ dataset,
                              const std::int64_t* __restrict__ neighbors,
                              float* __restrict__ distances,
                              std::int64_t n_pairs,
                              std::int64_t k,
                              int dim)
{
  auto const idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_pairs) { return; }

  auto const q_row = idx / k;
  auto const d_row = neighbors[idx];
  const float* q   = queries + q_row * dim;
  const float* d   = dataset + d_row * dim;

  if constexpr (IsCosine) {
    float dot = 0.0F;
    float nq  = 0.0F;
    float nd  = 0.0F;
    for (int i = 0; i < dim; ++i) {
      float const a = q[i];
      float const b = d[i];
      dot += a * b;
      nq += a * a;
      nd += b * b;
    }
    float const denom = sqrtf(nq) * sqrtf(nd);
    float const sim   = denom > 0.0F ? dot / denom : 0.0F;
    distances[idx]    = 1.0F - sim;
  } else {
    float sum = 0.0F;
    for (int i = 0; i < dim; ++i) {
      float const diff = q[i] - d[i];
      sum += diff * diff;
    }
    distances[idx] = sqrtf(sum);
  }
}

}  // namespace

void refine_topk_distances(dataset_matrix_view queries,
                           dataset_matrix_view dataset,
                           cudf::column_view const& neighbors,
                           cudf::mutable_column_view const& distances,
                           int64_t k,
                           cuvs::distance::DistanceType metric,
                           rmm::cuda_stream_view stream)
{
  auto const n_pairs = static_cast<std::int64_t>(neighbors.size());
  CUDF_EXPECTS(static_cast<std::int64_t>(distances.size()) == n_pairs,
               "VSS refine: neighbors and distances must have equal length");
  CUDF_EXPECTS(queries.extent(1) == dataset.extent(1), "VSS refine: dim mismatch");
  if (n_pairs == 0) { return; }

  auto const dim   = static_cast<int>(queries.extent(1));
  bool const cosine = metric == cuvs::distance::DistanceType::CosineExpanded;

  constexpr int block = 256;
  auto const grid     = static_cast<unsigned int>((n_pairs + block - 1) / block);

  const float* q       = queries.data_handle();
  const float* d       = dataset.data_handle();
  const std::int64_t* n = neighbors.data<std::int64_t>();
  float* out           = distances.data<float>();

  if (cosine) {
    refine_kernel<true><<<grid, block, 0, stream.value()>>>(q, d, n, out, n_pairs, k, dim);
  } else {
    refine_kernel<false><<<grid, block, 0, stream.value()>>>(q, d, n, out, n_pairs, k, dim);
  }
  CUDF_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace sirius::vss
