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

#include "vss/vector_search.hpp"

#include "data/sirius_converter_registry.hpp"
#include "duckdb/common/exception.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/cuvs_index_cache.hpp"
#include "vss/ivf_flat_index.hpp"
#include "vss/pinned_column.hpp"
#include "vss/pinned_column_cache.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <limits>

namespace sirius::vss {

namespace {

// Build a 0-row [output_columns... , distance FLOAT32] table for the no-result
// cases (empty table or k == 0). Output column types come from the pinned table.
std::unique_ptr<cudf::table> make_empty_output(const scan_manager::pinned_entry& pin,
                                               const std::vector<std::string>& output_columns)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(output_columns.size() + 1);
  for (auto const& name : output_columns) {
    auto it = pin.data_batches_by_column.find(name);
    if (it == pin.data_batches_by_column.end() || it->second.empty()) {
      throw duckdb::InvalidInputException(
        "sirius_vector_search: pinned table missing output column '" + name + "'");
    }
    cols.push_back(cudf::empty_like(it->second.front()->view()));
  }
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32}));
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

std::unique_ptr<cucascade::host_data_representation> run_vector_search(
  duckdb::SiriusContext& ctx, const vector_search_request& req)
{
  auto& memory_manager = ctx.get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw duckdb::InvalidInputException("sirius_vector_search: no GPU memory space available");
  }
  auto* space          = const_cast<cucascade::memory::memory_space*>(gpu_spaces.front());
  int const target_gpu = space->get_device_id();
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_gpu}};
  auto mr = space->get_default_allocator();
  // The GPU->host converter's cudaMemcpyBatchAsync rejects the default stream.
  // Synchronized before we return, so the owned stream is safe to destroy on exit.
  rmm::cuda_stream stream_owner;
  auto stream = stream_owner.view();

  // The table must be GPU-pinned: the search reads its vectors and gathers the
  // output columns straight from GPU-resident chunks (same order the index built in).
  const auto* pin = ctx.get_scan_manager().find_pinned_entry(req.table_name);
  if (pin == nullptr || pin->tier != cucascade::memory::Tier::GPU) {
    throw duckdb::InvalidInputException("sirius_vector_search: table '" + req.table_name +
                                        "' must be pinned on the GPU tier");
  }

  auto host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (host_spaces.empty()) {
    throw duckdb::InvalidInputException("sirius_vector_search: no HOST memory space available");
  }
  const auto* host_space = host_spaces.front();

  // Move GPU result table to host_data_representation the table
  // function can stream out via host_table_chunk_reader.
  auto to_host = [&](std::unique_ptr<cudf::table> tbl) {
    cucascade::gpu_table_representation gpu_repr(std::move(tbl), *space, stream);
    auto host_repr = converter_registry::get().convert<cucascade::host_data_representation>(
      gpu_repr, host_space, stream);
    stream.synchronize();
    return host_repr;
  };

  // Coalesce a pinned column (many GPU chunks) into one contiguous column, reused
  // across searches via the session cache so we don't re-concatenate every query.
  auto& col_cache = ctx.get_pinned_column_cache();
  auto coalesced  = [&](const std::string& name) {
    return col_cache.get_or_build(req.table_name,
                                  name,
                                  pinned_column_alloc_size(*pin, name),
                                  target_gpu,
                                  [&](rmm::device_async_resource_ref build_mr,
                                      rmm::cuda_stream_view build_stream) {
                                    return concat_pinned_column(
                                      *pin, name, *space, build_stream, build_mr);
                                  });
  };

  auto const num_rows = static_cast<int64_t>(pin->num_rows);
  auto const k        = std::min<int64_t>(num_rows, req.k);
  if (k <= 0 || num_rows == 0) { return to_host(make_empty_output(*pin, req.output_columns)); }

  auto const metric = ann_distance_type_from_metric(req.metric);

  // Upload the (constant) query vector once
  rmm::device_buffer query_buf(req.query.size() * sizeof(float), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    query_buf.data(), req.query.data(), query_buf.size(), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();

  std::unique_ptr<cudf::column> neighbors;
  std::unique_ptr<cudf::column> distances;
  // ANN output order is not guaranteed, so it must be sorted after
  bool sort_by_distance = false;

  if (req.use_index) {
    const auto* index_entry =
      ctx.get_cuvs_index_cache().find_by_column(req.table_name, req.column_name, metric);
    if (index_entry == nullptr || !index_entry->index) {
      throw duckdb::InvalidInputException(
        "sirius_vector_search: no ANN index for '" + req.table_name + "." + req.column_name +
        "' under the requested metric; create one with sirius_create_ann_index or pass "
        "use_index => false");
    }
    auto const n_lists =
      index_entry->meta.n_lists > 0 ? static_cast<std::uint32_t>(index_entry->meta.n_lists) : 1u;
    auto const n_probes =
      req.n_probes > 0 ? std::min<std::uint32_t>(static_cast<std::uint32_t>(req.n_probes), n_lists)
                       : std::min<std::uint32_t>(n_lists, 32u);
    auto search = search_ivf_flat_index(*index_entry->index,
                                        static_cast<const float*>(query_buf.data()),
                                        req.dim,
                                        k,
                                        n_probes,
                                        stream,
                                        mr);

    // When the probed lists hold fewer than k vectors, IVF-Flat pads the result
    // with dummy slots whose distance is the sort-key sentinel. A low n_probes
    // yields fewer than k rows.
    auto const finite_max =
      cudf::numeric_scalar<float>(std::numeric_limits<float>::max(), true, stream);
    auto valid = cudf::binary_operation(search.distances->view(),
                                        finite_max,
                                        cudf::binary_operator::LESS,
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);
    auto kept  = cudf::apply_boolean_mask(
      cudf::table_view{{search.neighbors->view(), search.distances->view()}},
      valid->view(),
      stream,
      mr);
    auto kept_cols   = kept->release();
    neighbors        = std::move(kept_cols[0]);
    distances        = std::move(kept_cols[1]);
    sort_by_distance = true;
  } else {
    auto vectors      = coalesced(req.column_name);
    auto dataset_view = list_column_as_dataset_view(vectors->view(), req.dim);
    auto query_view   = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      static_cast<const float*>(query_buf.data()), int64_t{1}, req.dim);
    auto knn = brute_force_knn(
      dataset_view, query_view, k, enn_distance_type_from_metric(req.metric), stream, mr);
    neighbors = std::move(knn.neighbors);
    distances = std::move(knn.distances);
  }

  // Gather the requested output columns from the pinned table by the returned global
  // row indices. Both branches produce in-range neighbors, so DONT_CHECK is safe.
  std::vector<std::shared_ptr<cudf::column>> full_cols;
  full_cols.reserve(req.output_columns.size());
  for (auto const& name : req.output_columns) {
    full_cols.push_back(coalesced(name));
  }
  std::vector<cudf::column_view> full_views;
  full_views.reserve(full_cols.size());
  for (auto const& col : full_cols) {
    full_views.push_back(col->view());
  }
  auto gathered      = cudf::gather(cudf::table_view(full_views),
                               neighbors->view(),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               stream,
                               mr);
  auto gathered_cols = gathered->release();
  gathered_cols.push_back(std::move(distances));  // trailing distance column
  auto output_table = std::make_unique<cudf::table>(std::move(gathered_cols));

  if (sort_by_distance) {
    auto const dist_idx = static_cast<cudf::size_type>(req.output_columns.size());
    output_table        = cudf::sort_by_key(output_table->view(),
                                     cudf::table_view({output_table->view().column(dist_idx)}),
                                            {cudf::order::ASCENDING},
                                            {cudf::null_order::AFTER},
                                     stream,
                                     mr);
  }

  return to_host(std::move(output_table));
}

}  // namespace sirius::vss
