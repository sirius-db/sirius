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
 * @brief Exact (brute-force) k-nearest-neighbour search via cuVS.
 *
 * For every query vector, finds the @p k nearest dataset vectors under @p
 * metric. @p dataset and @p queries must share the same dimensionality and be
 * row-major `[n, dim]` FLOAT32 matrices (see @ref list_column_as_dataset_view).
 *
 * Output columns are allocated through cudf's current device resource — i.e.
 * Sirius's cucascade-backed memory resource during execution — so the results
 * (and cuVS's internal workspace, which also draws from the current device
 * resource) are tracked by the reservation system.
 *
 * @warning NON-OWNING inputs: @p dataset and @p queries borrow device memory
 *          that must remain alive AND resident on-device for the duration of the
 *          call — a free or a tiering spill (D2H) of either would be a
 *          use-after-free the compiler cannot catch. In Sirius, hold a
 *          `cucascade::read_only_data_batch` (shared lock) on the batch backing
 *          @p dataset across this call: its shared lock blocks the exclusive
 *          lock a spill would need (see @ref list_column_as_dataset_view). The
 *          call being synchronous bounds exactly how long that lock must be held.
 *
 * The call is synchronous: the work stream is synchronized before returning.
 *
 * @param dataset Row-major `[n_rows, dim]` dataset to search.
 * @param queries Row-major `[n_queries, dim]` query vectors.
 * @param k       Number of neighbours per query (`1 <= k <= n_rows`).
 * @param metric  Distance metric (default: squared L2).
 * @return Flattened neighbour-index and distance columns, plus `n_queries`/`k`.
 */
knn_result brute_force_knn(
  dataset_matrix_view dataset,
  dataset_matrix_view queries,
  int64_t k,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded);

}  // namespace sirius::vss
