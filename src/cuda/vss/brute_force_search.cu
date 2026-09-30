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

}  // namespace sirius::vss
