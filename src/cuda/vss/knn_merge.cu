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

#include "vss/knn_merge.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>

#include <rmm/device_uvector.hpp>

#include <cuvs/neighbors/knn_merge_parts.hpp>

namespace sirius::vss {

knn_result knn_merge_parts_topk(raft::device_resources const& res,
                                cudf::column_view const& stacked_distances,
                                cudf::column_view const& stacked_neighbors,
                                int64_t n_samples,
                                int64_t n_parts,
                                int64_t k,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(n_samples >= 1 && n_parts >= 1 && k >= 1,
               "VSS merge: n_samples/n_parts/k must be >= 1");
  CUDF_EXPECTS(stacked_distances.type().id() == cudf::type_id::FLOAT32,
               "VSS merge: distances must be FLOAT32");
  CUDF_EXPECTS(stacked_neighbors.type().id() == cudf::type_id::INT64,
               "VSS merge: neighbors must be INT64");
  CUDF_EXPECTS(static_cast<int64_t>(stacked_distances.size()) == n_parts * n_samples * k &&
                 static_cast<int64_t>(stacked_neighbors.size()) == n_parts * n_samples * k,
               "VSS merge: stacked inputs must be [n_parts * n_samples * k]");

  // Part-major inputs: part p's [n_samples, k] block at row p*n_samples.
  auto const in_dist = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    stacked_distances.data<float>(), n_parts * n_samples, k);
  auto const in_idx = raft::make_device_matrix_view<const int64_t, int64_t, raft::row_major>(
    stacked_neighbors.data<int64_t>(), n_parts * n_samples, k);

  auto const out_size = static_cast<cudf::size_type>(n_samples * k);
  auto out_distances  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto out_neighbors = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);

  auto const out_dist = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
    out_distances->mutable_view().data<float>(), n_samples, k);
  auto const out_idx = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
    out_neighbors->mutable_view().data<int64_t>(), n_samples, k);

  // Neighbor ids are already global (shifted in the selection stage), so every
  // part's translation is zero.
  rmm::device_uvector<int64_t> translations(static_cast<std::size_t>(n_parts), stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(translations.data(), 0, translations.size() * sizeof(int64_t), stream.value()));
  auto const trans_view =
    raft::make_device_vector_view<int64_t, int64_t>(translations.data(), n_parts);

  cuvs::neighbors::knn_merge_parts(res, in_dist, in_idx, out_dist, out_idx, trans_view);

  return knn_result{std::move(out_neighbors), std::move(out_distances), n_samples, k};
}

}  // namespace sirius::vss
