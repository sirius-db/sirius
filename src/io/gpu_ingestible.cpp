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

#include "io/gpu_ingestible.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace sirius::io {

std::unique_ptr<cudf::table> gpu_ingestible::post_filter_and_project(
  cudf::table_view const& input,
  post_filter_and_projection_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  // Default: materialize the view and delegate to the owning overload. Implementations
  // whose inputs are views over caller-owned memory (the pinned cache) override this
  // to avoid copying rows and columns the filter/projection would drop anyway.
  return post_filter_and_project(
    std::make_unique<cudf::table>(input, stream, cudf::get_current_device_resource_ref()),
    info,
    mem_space,
    stream);
}

std::shared_ptr<gpu_ingestible> make_gpu_ingestible(
  std::unique_ptr<ingestible_table_info> info,
  scan_manager::sirius_scan_manager const& mgr,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs)
{
  if (!info) {
    throw std::runtime_error("[sirius::io::make_gpu_ingestible] table_info must not be null.");
  }
  // Dispatch through the table_info's virtual factory. Pass self by move so
  // the constructed ingestible can co-own the bind data through
  // gpu_ingestible's _table_info member — overrides retain typed access via
  // static_cast on table_info().
  auto* raw = info.get();
  return raw->make_ingestible(std::move(info), mgr, gpu_ioctxs);
}

}  // namespace sirius::io
