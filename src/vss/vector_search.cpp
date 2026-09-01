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

#include "cudf/cudf_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "duckdb/common/exception.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/vector_search_internal.hpp"

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>

namespace sirius::vss {

std::unique_ptr<cudf::table> make_empty_vss_output(
  const std::vector<sirius::logical_type>& output_column_types)
{
  duckdb::vector<sirius::logical_type> types(output_column_types.begin(),
                                             output_column_types.end());
  types.push_back(sirius::logical_type::make(sirius::type_id::FLOAT));
  return sirius::make_empty_table(types);
}

std::unique_ptr<cucascade::host_data_representation> vss_result_to_host(
  const vector_search_context& c, std::unique_ptr<cudf::table> table)
{
  cucascade::gpu_table_representation gpu_repr(std::move(table), c.space, c.stream);
  auto host_repr = converter_registry::get().convert<cucascade::host_data_representation>(
    gpu_repr, &c.host_space, c.stream);
  c.stream.synchronize();
  return host_repr;
}

std::unique_ptr<cucascade::host_data_representation> run_vector_search(
  duckdb::SiriusContext& ctx, const vector_search_request& req)
{
  auto& memory_manager = ctx.get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw duckdb::InvalidInputException("sirius_knn_search: no GPU memory space available");
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
  const auto* pin = ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(
    req.catalog, req.schema, req.table_name);
  if (pin == nullptr || pin->tier != cucascade::memory::Tier::GPU) {
    throw duckdb::InvalidInputException("sirius_knn_search: table '" + req.table_name +
                                        "' must be pinned on the GPU tier");
  }

  auto host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  if (host_spaces.empty()) {
    throw duckdb::InvalidInputException("sirius_knn_search: no HOST memory space available");
  }
  const auto* host_space = host_spaces.front();

  auto const num_rows = static_cast<int64_t>(pin->num_rows);
  auto const k        = std::min<int64_t>(num_rows, req.k);
  if (k <= 0 || num_rows == 0) {
    vector_search_context empty_ctx{
      ctx, req, *space, *host_space, *pin, mr, stream, nullptr, target_gpu, k};
    return vss_result_to_host(empty_ctx, make_empty_vss_output(req.output_column_types));
  }

  // Upload the (constant) query vector once; both search impls read it on the device.
  rmm::device_buffer query_buf(req.query.size() * sizeof(float), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    query_buf.data(), req.query.data(), query_buf.size(), cudaMemcpyHostToDevice, stream.value()));
  stream.synchronize();

  vector_search_context c{ctx,
                          req,
                          *space,
                          *host_space,
                          *pin,
                          mr,
                          stream,
                          static_cast<const float*>(query_buf.data()),
                          target_gpu,
                          k};
  return req.use_index ? run_vector_search_ann(c) : run_vector_search_enn(c);
}

}  // namespace sirius::vss
