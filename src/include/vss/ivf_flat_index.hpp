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
#include <vector>

namespace sirius::vss {

/// Build an IVF-Flat index from the dataset held as many separate FLOAT32 LIST
/// @p batches (each unsliced, gap-free, fixed width @p dim), without concatenating
/// them into one column. This sidesteps cudf's 2^31-element per-column limit,
/// which a full coalesce of a large dataset overflows in the LIST child.
///
/// Centroids are trained on the first chunk (cuVS subsamples it internally per
/// `kmeans_trainset_fraction`); the index is then populated chunk by chunk via
/// `ivf_flat::extend`, assigning each vector its global row id so search returns
/// indices into the whole dataset (in @p batches order). Same ownership contract
/// as @ref build_ivf_flat_index: everything allocates through @p index_mr and the
/// index is fully resident on return. @p batches are only read during the build.
///
/// Ownership: the index and its build-time scratch allocates through @p index_mr,
/// i.e., the GPU reservation's memory resource, so the whole index lives in
/// Sirius-reserved GPU memory. The current CUDA device must already be the
/// reservation's device. The build is synchronous, so the index is fully resident
/// on return and @p batches are only read during it (the index keeps its own copy).
/// Returned type-erased so callers need not name the cuVS index type.
///
/// The build runs on @p stream, so the index's device buffers are bound to it and
/// are freed on it when the index is destroyed. The caller must keep @p stream
/// alive at least as long as the returned index. Binding the reservation to this
/// same stream (attach it before calling) is what routes the build's allocations
/// through the reservation.
///
/// \param batches    Fixed-width FLOAT32 LIST column, each holding part of the dataset vectors.
/// \param dim       Vector dimensionality.
/// \param n_lists   IVF-Flat inverted-list count (1 <= n_lists <= n_rows).
/// \param metric    Distance metric (must be IVF-Flat-supported, e.g. L2SqrtExpanded).
/// \param index_mr  Reservation-backed device resource the index allocates through.
/// \param stream    Stream the build runs on; the index's buffers are bound to it.
std::unique_ptr<any_cuvs_index> build_ivf_flat_index_from_batches(
  std::vector<cudf::column_view> const& batches,
  std::int64_t dim,
  std::uint32_t n_lists,
  cuvs::distance::DistanceType metric,
  rmm::device_async_resource_ref index_mr,
  rmm::cuda_stream_view stream);

/// Flattened k-NN result from an ANN search: both columns have length @c k.
struct ann_result {
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
ann_result search_ivf_flat_index(any_cuvs_index const& index,
                                 const float* query_device,
                                 std::int64_t dim,
                                 std::int64_t k,
                                 std::uint32_t n_probes,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

}  // namespace sirius::vss
