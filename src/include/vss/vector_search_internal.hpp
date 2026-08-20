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

#include "vss/vector_search.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudf {
class table;
}  // namespace cudf
namespace cucascade {
class host_data_representation;
namespace memory {
class memory_space;
}  // namespace memory
}  // namespace cucascade
namespace duckdb {
class SiriusContext;
}  // namespace duckdb
namespace sirius::scan_manager {
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace sirius::vss {

/// Resolved handles for vector search table function implementation. Built
/// once by @ref run_vector_search after it locates the GPU space, pinned table,
/// host space, and uploads the query, then passed by const-ref to whichever impl
/// runs. All references outlive the impl call.
struct vector_search_context {
  duckdb::SiriusContext& ctx;
  const vector_search_request& req;
  cucascade::memory::memory_space& space;
  const cucascade::memory::memory_space& host_space;
  const scan_manager::pinned_entry& pin;
  rmm::device_async_resource_ref mr;
  rmm::cuda_stream_view stream;
  const float* query_device;  ///< [dim] FLOAT32 query already uploaded to the device.
  int target_gpu;
  std::int64_t k;  ///< min(num_rows, req.k).
};

/// Return an empty table; column types come from the pinned table.
/// Used for the no-result cases (empty table / k == 0 / no chunks).
std::unique_ptr<cudf::table> make_empty_vss_output(const scan_manager::pinned_entry& pin,
                                                   const std::vector<std::string>& output_columns);

/// Move a GPU result table to a host_data_representation the table function can
/// stream out. Synchronizes @c c.stream before returning.
std::unique_ptr<cucascade::host_data_representation> vss_result_to_host(
  const vector_search_context& c, std::unique_ptr<cudf::table> table);

/// ENN search, tiled per pinned chunk and merged.
std::unique_ptr<cucascade::host_data_representation> run_vector_search_enn(
  const vector_search_context& c);

}  // namespace sirius::vss
