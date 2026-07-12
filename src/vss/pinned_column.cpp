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

#include "vss/pinned_column.hpp"

#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <vector>

namespace sirius::vss {

std::unique_ptr<cudf::column> concat_pinned_column(const scan_manager::pinned_entry& pin,
                                                   const std::string& column_name,
                                                   cucascade::memory::memory_space& space,
                                                   rmm::cuda_stream_view stream)
{
  return concat_pinned_column(pin, column_name, space, stream, space.get_default_allocator());
}

std::unique_ptr<cudf::column> concat_pinned_column(const scan_manager::pinned_entry& pin,
                                                   const std::string& column_name,
                                                   cucascade::memory::memory_space& space,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto it = pin.data_batches_by_column.find(column_name);
  if (it == pin.data_batches_by_column.end() || it->second.empty()) {
    throw internal_exception("VSS: pinned table missing column '" + column_name + "'");
  }
  auto const& chunks = it->second;

  std::vector<cudf::column_view> views;
  views.reserve(chunks.size());
  for (std::size_t c = 0; c < chunks.size(); ++c) {
    if (c < pin.chunk_memory_spaces.size() && pin.chunk_memory_spaces[c] != nullptr &&
        pin.chunk_memory_spaces[c]->get_device_id() != space.get_device_id()) {
      throw internal_exception(
        "VSS: pinned table spans multiple GPUs (multi-GPU not supported yet)");
    }
    views.push_back(chunks[c]->view());
  }

  if (views.size() == 1) { return std::make_unique<cudf::column>(views.front(), stream, mr); }
  return cudf::concatenate(views, stream, mr);
}

std::size_t pinned_column_alloc_size(const scan_manager::pinned_entry& pin,
                                     const std::string& column_name)
{
  auto it = pin.data_batches_by_column.find(column_name);
  if (it == pin.data_batches_by_column.end() || it->second.empty()) {
    throw internal_exception("VSS: pinned table missing column '" + column_name + "'");
  }
  std::size_t bytes = 0;
  for (auto const& chunk : it->second) {
    bytes += chunk->alloc_size();
  }
  return bytes;
}

}  // namespace sirius::vss
