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

#include "vss/cuvs_index_cache.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuvs/distance/distance.hpp>

#include <cstdint>
#include <memory>
#include <string_view>

namespace sirius::vss {

/// Map a user metric string to the cuVS DistanceType.
///
/// The values are chosen to agree with what the VSS recognizer derives from the
/// distance function (`array_distance` to L2SqrtExpanded, `array_cosine_distance`
/// to CosineExpanded; see `vss_pattern.cpp::metric_for_function`), so a pinned
/// index's metric compares equal during auto-route matching. Throws
/// `std::invalid_argument` on an unknown metric.
cuvs::distance::DistanceType ann_distance_type_from_metric(std::string_view metric);

/// Map a user metric string to the cuVS DistanceType for an exact BRUTE-FORCE
/// search over raw vectors. Same as @ref ann_distance_type_from_metric except l2
/// uses the Unexpanded form (`||a-b||^2` directly) to avoid catastrophic
/// cancellation on large-magnitude vectors.
cuvs::distance::DistanceType enn_distance_type_from_metric(std::string_view metric);

/// Build an IVF-Flat index over @p vectors (a contiguous, unsliced, gap-free
/// FLOAT32 LIST column of fixed width @p dim), allocating the index through
/// @p index_mr, the GPU reservation's memory resource, so the whole index lives
/// in Sirius-reserved GPU memory. The current CUDA device must already be the
/// reservation's device. And it's synchronous so the index is fully built and
/// resident on return.
///
/// Returned type-erased so callers need not name the cuVS index type; recover the
/// concrete `ivf_flat::index<float, int64_t>` with
/// `pinned_index_entry::index_as<...>()`. @p vectors is only read during the
/// build (the index keeps its own copy via `add_data_on_build`).
///
/// \param vectors   Fixed-width FLOAT32 LIST column holding all dataset vectors.
/// \param dim       Vector dimensionality.
/// \param n_lists   IVF-Flat inverted-list count (1 <= n_lists <= n_rows).
/// \param metric    Distance metric (must be IVF-Flat-supported, e.g. L2SqrtExpanded).
/// \param index_mr  Reservation-backed device resource the index allocates through.
std::unique_ptr<any_cuvs_index> build_ivf_flat_index(cudf::column_view const& vectors,
                                                     std::int64_t dim,
                                                     std::uint32_t n_lists,
                                                     cuvs::distance::DistanceType metric,
                                                     rmm::device_async_resource_ref index_mr);

/// Flattened k-NN result from an ANN search: both columns have length @c k,
/// ordered nearest-first.
struct ann_search_result {
  std::unique_ptr<cudf::column> neighbors;  ///< INT64 row indices into the indexed dataset.
  std::unique_ptr<cudf::column> distances;  ///< FLOAT32 distances to those rows.
};

/// Search a pinned IVF-Flat index (held type-erased in @p index) for the @p k
/// nearest neighbors of a single query vector.
///
/// @p query_device points to a @p dim-length FLOAT32 vector ALREADY ON THE
/// DEVICE (caller uploads it). Outputs are allocated through @p mr. The call is
/// synchronous (the work stream is synchronized before returning).
///
/// \param index        Type-erased IVF-Flat index (throws if it is not one).
/// \param query_device Device pointer to the [dim] FLOAT32 query vector.
/// \param dim          Vector dimensionality (must equal the index's).
/// \param k            Neighbors to return (1 <= k <= n_rows).
/// \param n_probes     IVF lists to probe (accuracy/speed knob; <= n_lists).
ann_search_result search_ivf_flat_index(any_cuvs_index const& index,
                                        const float* query_device,
                                        std::int64_t dim,
                                        std::int64_t k,
                                        std::uint32_t n_probes,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr);

}  // namespace sirius::vss
