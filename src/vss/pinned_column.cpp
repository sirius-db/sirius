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

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius/exception.hpp"
#include "vss/staging_shortfall.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace sirius::vss {

std::vector<cudf::column_view> pinned_column_chunk_views(const scan_manager::pinned_entry& pin,
                                                         const std::string& column_name,
                                                         cucascade::memory::memory_space& space)
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
  return views;
}

std::size_t pinned_column_chunk_count(const scan_manager::pinned_entry& pin,
                                      const std::string& column_name)
{
  if (pin.tier == cucascade::memory::Tier::HOST) { return pin.host_chunks.size(); }
  auto it = pin.data_batches_by_column.find(column_name);
  if (it == pin.data_batches_by_column.end()) {
    throw internal_exception("VSS: pinned table missing column '" + column_name + "'");
  }
  return it->second.size();
}

staged_pinned_chunk stage_pinned_column_chunk(const scan_manager::pinned_entry& pin,
                                              const std::string& column_name,
                                              std::size_t chunk_index,
                                              cucascade::memory::memory_space& gpu_space,
                                              rmm::cuda_stream_view stream,
                                              const telemetry::batch_telemetry_info& telemetry_info)
{
  if (pin.tier != cucascade::memory::Tier::HOST) {
    auto const views = pinned_column_chunk_views(pin, column_name, gpu_space);
    if (chunk_index >= views.size()) {
      throw internal_exception("VSS: pinned column chunk index out of range");
    }
    return staged_pinned_chunk{views[chunk_index], nullptr, nullptr};
  }

  auto const& names = pin.cache_info.column_names();
  auto const it     = std::find(names.begin(), names.end(), column_name);
  if (it == names.end()) {
    throw internal_exception("VSS: host-tier pinned table missing column '" + column_name + "'");
  }
  std::array<std::size_t, 1> const cols{static_cast<std::size_t>(std::distance(names.begin(), it))};

  if (chunk_index >= pin.host_chunks.size()) {
    throw internal_exception("VSS: pinned column chunk index out of range");
  }
  auto const& chunk = pin.host_chunks[chunk_index];
  if (!chunk) { throw internal_exception("VSS: host-tier pinned chunk is null"); }
  auto const* host = dynamic_cast<cucascade::host_data_representation const*>(chunk.get());
  if (host == nullptr) {
    throw internal_exception("VSS: column '" + column_name +
                             "' is pinned with host compression, which staging cannot slice");
  }

  auto data_rep    = host->slice(cols);
  auto const bytes = data_rep->get_size_in_bytes();
  std::shared_ptr<cucascade::memory::reservation> reservation{
    gpu_space.make_reservation_or_null(bytes)};
  if (!reservation) {
    throw_staging_shortfall(
      gpu_space,
      bytes,
      "VSS: staging host-tier column '" + column_name + "' chunk " + std::to_string(chunk_index));
  }

  auto const batch_id = sirius::get_next_batch_id();
  auto batch          = cucascade::data_batch::make(
    batch_id,
    std::move(data_rep),
    telemetry::quent_data_batch_probe::create(telemetry_info, batch_id));
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(
      sirius::converter_registry::get(), *reservation, stream);
    // The converter copies on a stream of its own and binds the copy to it, so the copy's free
    // would be ordered behind nothing the caller does: dropping the chunk right after enqueueing a
    // gather on `stream` hands its memory back while the gather still reads it, and another task
    // can overwrite it first. Freed on the reading stream instead.
    mut.rebind_stream(stream);
  }
  auto view = sirius::get_cudf_table_view(*batch).column(0);
  return staged_pinned_chunk{view, std::move(batch), std::move(reservation)};
}

staged_pinned_column stage_pinned_column(const scan_manager::pinned_entry& pin,
                                         const std::string& column_name,
                                         cucascade::memory::memory_space& gpu_space,
                                         rmm::cuda_stream_view stream,
                                         const telemetry::batch_telemetry_info& telemetry_info)
{
  staged_pinned_column out;
  if (pin.tier != cucascade::memory::Tier::HOST) {
    out.views = pinned_column_chunk_views(pin, column_name, gpu_space);
    return out;
  }

  auto const n_chunks = pin.host_chunks.size();
  out.views.reserve(n_chunks);
  out.owners.reserve(n_chunks);
  out.reservations.reserve(n_chunks);
  for (std::size_t c = 0; c < n_chunks; ++c) {
    auto staged = stage_pinned_column_chunk(pin, column_name, c, gpu_space, stream, telemetry_info);
    out.views.push_back(staged.view);
    out.owners.push_back(std::move(staged.owner));
    out.reservations.push_back(std::move(staged.reservation));
  }
  return out;
}

cucascade::memory::memory_space& pinned_entry_gpu_space(const scan_manager::pinned_entry& pin)
{
  for (auto* space : pin.chunk_memory_spaces) {
    if (space != nullptr) { return *space; }
  }
  throw internal_exception("VSS: pinned table has no GPU-resident chunk (host-tier pin?)");
}

}  // namespace sirius::vss
