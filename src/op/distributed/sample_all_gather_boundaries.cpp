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

#include "op/distributed/sample_all_gather_boundaries.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "sirius/exception.hpp"

#include <cudf/copying.hpp>
#include <cudf/search.hpp>

#include <algorithm>
#include <numeric>
#include <utility>

namespace sirius::op {

sample_all_gather_boundaries::sample_all_gather_boundaries(
  std::shared_ptr<exchange::exchange_service> service,
  exchange::stream_id stream,
  cucascade::memory::memory_space& gpu,
  std::vector<cudf::order> column_order,
  std::vector<cudf::null_order> null_precedence,
  std::function<void(std::shared_ptr<cudf::table const>)> set_exchange_boundaries,
  std::chrono::milliseconds timeout)
  : _service(std::move(service)),
    _stream(stream),
    _gpu(&gpu),
    _column_order(std::move(column_order)),
    _null_precedence(std::move(null_precedence)),
    _set_exchange_boundaries(std::move(set_exchange_boundaries)),
    _timeout(timeout)
{
}

sort_boundaries sample_all_gather_boundaries::make(request const& request,
                                                   ::cuda::stream_ref stream)
{
  auto const& sample_keys = request.sample_keys;
  auto const sample_bytes = request.sample_bytes;
  auto const num_parts    = request.num_parts;
  auto& gpu               = *_gpu;
  auto const mr           = gpu.get_default_allocator();

  if (!_gathered) {
    // The peers are past this gather, whether it failed part way or finished and released its
    // samples: another would wait out the timeout.
    if (_gather_entered) {
      throw sirius::internal_exception("[sort_sample] the boundary all-gather cannot run again");
    }
    // 1. This rank's summary: evenly spaced rows of its sample, enough to give each of num_parts
    // partitions summary_rows_per_partition rows of the global sample in this rank's range. Fewer
    // when the sample fell short of sort_sample_bytes, so a rank with little input has little say
    // in where the ranges fall. Not the local sample alone: it describes the rows this rank sends,
    // not the ones it receives.
    std::shared_ptr<::cucascade::data_batch> summary;
    if (auto const rows = sample_keys.num_rows(); rows > 0) {
      auto const full  = std::max(distributed_summary_rows, summary_rows_per_partition * num_parts);
      auto const share = request.target_sample_bytes == 0
                           ? 1.0
                           : std::min(1.0,
                                      static_cast<double>(sample_bytes) /
                                        static_cast<double>(request.target_sample_bytes));
      auto const take =
        std::clamp<std::size_t>(static_cast<std::size_t>(share * static_cast<double>(full)),
                                1,
                                static_cast<std::size_t>(rows));
      std::vector<int32_t> indices(take);
      for (std::size_t i = 0; i < take; ++i) {
        indices[i] = static_cast<int32_t>(i * static_cast<std::size_t>(rows) / take);
      }
      auto const column = sirius_physical_sort_sample::index_column(indices, stream, mr);
      summary           = make_data_batch(
        cudf::gather(
          sample_keys, column->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr),
        gpu,
        stream,
        request.telemetry);
    }

    // 2. Every rank's summary, merged in rank order: the same global sample on every rank.
    // TODO(#2375): all_gather blocks this GPU worker thread until every rank's summary arrives.
    _gather_entered = true;
    _gathered = exchange::all_gather(*_service, _stream, summary, gpu, stream.get(), _timeout);
  }
  auto made = boundaries_from(*_gathered, num_parts, request.telemetry, stream);
  // The boundaries may be copied straight from a gathered batch: that read ends before the
  // batch is freed.
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  _gathered.reset();
  return made;
}

sort_boundaries sample_all_gather_boundaries::boundaries_from(
  std::vector<std::shared_ptr<::cucascade::data_batch>> const& gathered,
  std::size_t num_parts,
  telemetry::batch_telemetry_info const& telemetry,
  ::cuda::stream_ref stream)
{
  auto const& column_order    = _column_order;
  auto const& null_precedence = _null_precedence;
  auto& gpu                   = *_gpu;
  auto const mr               = gpu.get_default_allocator();
  auto const world            = _service->world_size();
  auto const rank             = _service->rank();

  std::vector<::cucascade::read_only_data_batch> parts;
  for (auto const& g : gathered) {
    if (g) { parts.push_back(g->to_read_only()); }
  }
  if (parts.empty()) { return {}; }
  std::vector<int> keys(column_order.size());
  std::iota(keys.begin(), keys.end(), 0);
  std::shared_ptr<::cucascade::data_batch> merged;
  if (parts.size() > 1) {
    merged = gpu_merge_impl::merge_order_by(
      parts, keys, column_order, null_precedence, stream, gpu, telemetry);
  }
  auto const global =
    merged ? get_cudf_table_view(merged->to_read_only()) : get_cudf_table_view(parts.front());
  auto const total = global.num_rows();

  // 3. The range exchange's world - 1 boundaries, evenly spaced over the global sample.
  cudf::size_type lo = 0;
  cudf::size_type hi = total;
  if (world > 1) {
    std::shared_ptr<cudf::table const> exchange_boundaries =
      sirius_physical_sort_sample::evenly_spaced(global, 0, total, world, stream, mr);
    // This rank's range of the global sample: from the first row not below its lower boundary
    // to the first not below its upper one, as the exchange routes rows.
    auto const starts = cudf::lower_bound(
      global, exchange_boundaries->view(), column_order, null_precedence, stream, mr);
    std::vector<cudf::size_type> at(world - 1);
    CUDF_CUDA_TRY(cudaMemcpyAsync(at.data(),
                                  starts->view().data<cudf::size_type>(),
                                  at.size() * sizeof(cudf::size_type),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    if (rank > 0) { lo = at[rank - 1]; }
    if (rank + 1 < world) { hi = at[rank]; }
    _set_exchange_boundaries(std::move(exchange_boundaries));
  }

  // 4. This rank's partition boundaries, from the part of the global sample inside its range.
  auto const parts_here = std::min<std::size_t>(num_parts, std::max<cudf::size_type>(1, hi - lo));
  if (parts_here <= 1) { return {}; }
  return {parts_here,
          sirius_physical_sort_sample::evenly_spaced(global, lo, hi - lo, parts_here, stream, mr)};
}

}  // namespace sirius::op
