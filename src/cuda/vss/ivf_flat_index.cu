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

#include "vss/cudf_raft_interop.hpp"
#include "vss/ivf_flat_index.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/memory_resource>
#include <thrust/sequence.h>

#include <cuvs/neighbors/ivf_flat.hpp>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::vss {

namespace {

/// RAII: route allocations on the current device through @p mr for the guard's
/// lifetime, restoring the previous resource on exit. The evicted resource is
/// captured by value (an owning any_resource) per the rmm 26.06 behavior
/// documented in sirius_memory_reservation_manager, capturing a non-owning ref
/// would dangle once the per-device map entry is moved out.
struct scoped_current_device_resource {
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> prev;

  explicit scoped_current_device_resource(rmm::device_async_resource_ref mr)
    : prev(cudf::set_current_device_resource(mr))
  {
  }
  scoped_current_device_resource(const scoped_current_device_resource&)            = delete;
  scoped_current_device_resource& operator=(const scoped_current_device_resource&) = delete;
  scoped_current_device_resource(scoped_current_device_resource&&)                 = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource&&)      = delete;
  ~scoped_current_device_resource() { cudf::set_current_device_resource(std::move(prev)); }
};

}  // namespace

std::unique_ptr<any_cuvs_index> build_ivf_flat_index_from_batches(
  std::vector<cudf::column_view> const& batches,
  std::int64_t dim,
  std::uint32_t n_lists,
  cuvs::distance::DistanceType metric,
  rmm::device_async_resource_ref index_mr,
  rmm::cuda_stream_view stream)
{
  if (batches.empty()) {
    throw std::invalid_argument("build_ivf_flat_index_from_batches: no batches to index");
  }

  cuvs::neighbors::ivf_flat::index_params index_params;
  index_params.n_lists = n_lists;
  index_params.metric  = metric;
  // Train the kmeans centroids only; the dataset is added batch by batch.
  index_params.add_data_on_build = false;
  // Size each list to what it holds instead of the default padded growth.
  // We extend batch by batch, so the default policy over-allocates every list and
  // re-grows it on each batch, inflating the build-time peak well past the final
  // index size. Exact sizing keeps the peak close to the real footprint so the
  // build fits inside the reservation alongside the still-pinned source table.
  index_params.conservative_memory_allocation = true;

  // Allocate the index (and its build-time scratch) through the reservation's
  // resource so the whole thing lives in Sirius-reserved GPU memory. The handle
  // is constructed inside the guard so RAFT's workspace resource is the
  // reservation's too, not just the per-allocation default. The scope ends after
  // the build, restoring the prior current device resource.
  auto index = [&] {
    scoped_current_device_resource route{index_mr};
    // Build on the caller's stream so the index's device buffers are bound to it
    // and so allocations charge the reservation attached to it.
    raft::device_resources res{stream};

    // Train centroids on the first batch that has at least n_lists rows.
    // NOTE: could improve recall if we do a cross-batch training sample
    cudf::column_view const* train_batch = nullptr;
    bool any_non_empty                   = false;
    for (auto const& batch : batches) {
      auto const rows = static_cast<std::int64_t>(batch.size());
      if (rows == 0) { continue; }
      any_non_empty = true;
      if (rows >= static_cast<std::int64_t>(n_lists)) {
        train_batch = &batch;
        break;
      }
    }
    if (train_batch == nullptr) {
      if (!any_non_empty) {
        throw std::invalid_argument("build_ivf_flat_index_from_batches: all batches are empty");
      }
      throw std::invalid_argument(
        "build_ivf_flat_index_from_batches: no batch has at least n_lists=" +
        std::to_string(n_lists) + " rows to train IVF-Flat centroids; lower n_lists");
    }
    auto const train_view = list_column_as_dataset_view(*train_batch, dim);
    auto idx              = cuvs::neighbors::ivf_flat::build(res, index_params, train_view);

    // Populate the index batch by batch, tagging each vector with its global row id
    // (offset by the rows already added) so search returns dataset-global indices.
    std::int64_t base = 0;
    for (auto const& batch : batches) {
      auto const batch_view = list_column_as_dataset_view(batch, dim);
      auto const rows       = batch_view.extent(0);
      if (rows == 0) { continue; }
      if (base == 0) {
        // Index is still empty: nullopt implies the contiguous range [0, rows).
        cuvs::neighbors::ivf_flat::extend(res, batch_view, std::nullopt, &idx);
      } else {
        rmm::device_uvector<std::int64_t> ids(static_cast<std::size_t>(rows), stream, index_mr);
        thrust::sequence(rmm::exec_policy(stream), ids.begin(), ids.end(), base);
        auto const ids_view =
          raft::make_device_vector_view<const std::int64_t, std::int64_t>(ids.data(), rows);
        cuvs::neighbors::ivf_flat::extend(res, batch_view, std::optional{ids_view}, &idx);
        // ids is freed async on `stream` at scope exit, ordered after this extend.
      }
      base += rows;
    }
    // Synchronous contract: the index is fully resident before we restore the
    // resource and return (the dataset view borrows caller memory only for here).
    raft::resource::sync_stream(res);
    return idx;
  }();

  return make_cuvs_index(std::move(index));
}

namespace {
using ivf_flat_index_t = cuvs::neighbors::ivf_flat::index<float, int64_t>;
}  // namespace

ann_result search_ivf_flat_index(any_cuvs_index const& index,
                                 const float* query_device,
                                 std::int64_t dim,
                                 std::int64_t k,
                                 std::uint32_t n_probes,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto const* holder = dynamic_cast<cuvs_index_holder<ivf_flat_index_t> const*>(&index);
  if (holder == nullptr) {
    throw std::invalid_argument("search_ivf_flat_index: pinned index is not an IVF-Flat index");
  }
  auto const& idx = holder->index;

  auto const query_view = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    query_device, int64_t{1}, dim);

  // Allocate flattened [1 * k] outputs through `mr` so the reservation system tracks them
  auto const out_size = static_cast<cudf::size_type>(k);
  auto neighbors_col  = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);
  auto distances_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, out_size, cudf::mask_state::UNALLOCATED, stream, mr);

  auto neighbors_view = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
    neighbors_col->mutable_view().data<int64_t>(), int64_t{1}, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
    distances_col->mutable_view().data<float>(), int64_t{1}, k);

  raft::device_resources res{stream};
  cuvs::neighbors::ivf_flat::search_params search_params;
  search_params.n_probes = n_probes;
  cuvs::neighbors::ivf_flat::search(
    res, search_params, idx, query_view, neighbors_view, distances_view);
  raft::resource::sync_stream(res);

  return ann_result{std::move(neighbors_col), std::move(distances_col)};
}

}  // namespace sirius::vss
