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

#include <cstdint>
#include <memory>

namespace sirius::vss {

/**
 * @brief Result of a brute-force k-NN search.
 *
 * Both columns are flattened row-major with length `n_queries * k`: query `q`'s
 * results occupy the half-open range `[q * k, (q + 1) * k)`, ordered nearest
 * first.
 */
struct knn_result {
  std::unique_ptr<cudf::column> neighbors;  ///< INT64 row indices into the dataset.
  std::unique_ptr<cudf::column> distances;  ///< FLOAT32 distances to those rows.
  int64_t n_queries;
  int64_t k;
};

/**
 * @brief Exact (brute-force) k-nearest-neighbor search via cuVS.
 *
 * For every query vector, finds the @p k nearest dataset vectors under @p
 * metric. @p dataset and @p queries must share the same dimensionality and be
 * row-major `[n, dim]` FLOAT32 matrices (see @ref list_column_as_dataset_view).
 *
 * Output columns are allocated through @p mr (defaulting to cudf's current
 * device resource). In Sirius, pass the owning memory space's allocator so the
 * results are reserved against that exact space rather than the ambient default.
 * cuVS's internal scratch is separate: it draws from rmm's current device
 * resource (not @p mr), which Sirius installs as the cucascade allocator, so it
 * is still reserved, just against the ambient current space rather than @p mr.
 *
 * The search is enqueued on @p res's stream and is not synchronized before returning.
 * Results are only valid to read on the host after the caller syncs that stream. The
 * returned columns, the borrowed inputs, and the caller's downstream work therefore all
 * order on @p res's stream. Pass one @p res reused across chunks so the handle's
 * workspace setup is paid once rather than per call.
 *
 * @param res     Caller-owned RAFT resources; the search runs on its stream.
 * @param dataset Row-major dataset to search.
 * @param queries Row-major query vectors.
 * @param k       Number of neighbors per query.
 * @param metric  Distance metric.
 * @param mr      Device resource for the output columns (default: cudf's current).
 * @return Flattened neighbor-index and distance columns, plus `n_queries`/`k`.
 */
knn_result brute_force_knn(
  raft::device_resources const& res,
  dataset_matrix_view dataset,
  dataset_matrix_view queries,
  int64_t k,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtUnexpanded,
  rmm::device_async_resource_ref mr   = cudf::get_current_device_resource_ref());

}  // namespace sirius::vss
