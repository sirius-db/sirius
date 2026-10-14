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

#include "duckdb/common/exception.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sirius::vss {

namespace {

/// A GPU-tier pin made with compression requested keeps its chunks in @c device_chunks instead,
/// each one compressed or, where compression did not pay or failed, the plain columns.
std::vector<cudf::column_view> device_chunk_views(const scan_manager::pinned_entry& pin,
                                                  const std::string& column_name,
                                                  cucascade::memory::memory_space& space)
{
  auto const& names = pin.cache_info.column_names();
  auto const it     = std::find(names.begin(), names.end(), column_name);
  if (it == names.end()) {
    throw internal_exception("VSS: pinned table missing column '" + column_name + "'");
  }
  auto const col = static_cast<std::size_t>(std::distance(names.begin(), it));

  std::vector<cudf::column_view> views;
  views.reserve(pin.device_chunks.size());
  for (auto const& chunk : pin.device_chunks) {
    if (chunk.compressed) {
      throw duckdb::InvalidInputException(
        "column '%s' is pinned with GPU compression, which the vector functions cannot read; pin "
        "it with compression => false",
        column_name);
    }
    if (chunk.memory_space != nullptr &&
        chunk.memory_space->get_device_id() != space.get_device_id()) {
      throw internal_exception(
        "VSS: pinned table spans multiple GPUs (multi-GPU not supported yet)");
    }
    if (col >= chunk.columns.size() || !chunk.columns[col]) {
      throw internal_exception("VSS: pinned chunk missing column '" + column_name + "'");
    }
    views.push_back(chunk.columns[col]->view());
  }
  return views;
}

}  // namespace

std::vector<cudf::column_view> pinned_column_chunk_views(const scan_manager::pinned_entry& pin,
                                                         const std::string& column_name,
                                                         cucascade::memory::memory_space& space)
{
  if (!pin.device_chunks.empty()) { return device_chunk_views(pin, column_name, space); }
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

namespace {

/// Column @p column_name of HOST-tier chunk @p chunk_index as stored: its metadata and the pinned
/// blocks its offsets point into.
struct host_column {
  cucascade::memory::column_metadata const* meta;
  cucascade::memory::host_table_allocation const* table;
};

std::optional<host_column> find_host_column(const scan_manager::pinned_entry& pin,
                                            const std::string& column_name,
                                            std::size_t chunk_index)
{
  if (pin.tier != cucascade::memory::Tier::HOST || chunk_index >= pin.host_chunks.size()) {
    return std::nullopt;
  }
  auto const& names = pin.cache_info.column_names();
  auto const it     = std::find(names.begin(), names.end(), column_name);
  auto const* host =
    dynamic_cast<cucascade::host_data_representation const*>(pin.host_chunks[chunk_index].get());
  if (it == names.end() || host == nullptr || !host->get_host_table() ||
      !host->get_host_table()->allocation) {
    return std::nullopt;
  }
  auto const& table = *host->get_host_table();
  auto const col    = static_cast<std::size_t>(std::distance(names.begin(), it));
  if (col >= table.columns.size()) { return std::nullopt; }
  return host_column{&table.columns[col], &table};
}

/// @p bytes from byte @p offset of @p table's blocks into @p out, across block boundaries.
void read_host_bytes(cucascade::memory::host_table_allocation const& table,
                     std::size_t offset,
                     std::size_t bytes,
                     std::byte* out)
{
  auto const& blocks = *table.allocation;
  auto const bsz     = blocks.block_size();
  while (bytes > 0) {
    auto const in_block = offset % bsz;
    auto const now      = std::min(bytes, bsz - in_block);
    std::memcpy(out, blocks.at(offset / bsz).data() + in_block, now);
    offset += now;
    out += now;
    bytes -= now;
  }
}

}  // namespace

std::int64_t pinned_host_chunk_rows(const scan_manager::pinned_entry& pin,
                                    const std::string& column_name,
                                    std::size_t chunk_index)
{
  auto const column = find_host_column(pin, column_name, chunk_index);
  return column ? static_cast<std::int64_t>(column->meta->num_rows) : -1;
}

std::unique_ptr<cudf::column> gather_host_pinned_rows(const scan_manager::pinned_entry& pin,
                                                      const std::string& column_name,
                                                      std::size_t chunk_index,
                                                      std::int64_t dim,
                                                      std::int64_t start,
                                                      std::int64_t stride,
                                                      std::int64_t count,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  auto const column = find_host_column(pin, column_name, chunk_index);
  if (!column || dim <= 0 || count <= 0) { return nullptr; }
  auto const& meta  = *column->meta;
  auto const& table = *column->table;
  auto const n      = static_cast<std::int64_t>(meta.num_rows);
  // null_count may also be the "unknown" sentinel, which is as good as a null here.
  if (meta.type_id != static_cast<std::int32_t>(cudf::type_id::LIST) || meta.null_count != 0 ||
      meta.children.size() != 2 || start < 0 || stride < 1 || start + (count - 1) * stride >= n) {
    return nullptr;
  }
  auto const& offsets = meta.children[0];
  auto const& values  = meta.children[1];
  if (offsets.type_id != static_cast<std::int32_t>(cudf::type_id::INT32) || !offsets.has_data ||
      offsets.num_rows != n + 1 ||
      values.type_id != static_cast<std::int32_t>(cudf::type_id::FLOAT32) || !values.has_data ||
      values.null_count != 0 || static_cast<std::int64_t>(values.num_rows) != n * dim) {
    return nullptr;
  }
  // Row r is values[r * dim, (r + 1) * dim) only if the offsets start at 0 and end at n * dim;
  // with no null rows and every row dim wide, nothing in between can differ.
  std::int32_t first = -1;
  std::int32_t last  = -1;
  read_host_bytes(table, offsets.data_offset, sizeof(first), reinterpret_cast<std::byte*>(&first));
  read_host_bytes(table,
                  offsets.data_offset + static_cast<std::size_t>(n) * sizeof(last),
                  sizeof(last),
                  reinterpret_cast<std::byte*>(&last));
  if (first != 0 || static_cast<std::int64_t>(last) != n * dim) { return nullptr; }

  auto const row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
  std::vector<std::byte> rows(static_cast<std::size_t>(count) * row_bytes);
  for (std::int64_t j = 0; j < count; ++j) {
    read_host_bytes(table,
                    values.data_offset + static_cast<std::size_t>(start + j * stride) * row_bytes,
                    row_bytes,
                    rows.data() + static_cast<std::size_t>(j) * row_bytes);
  }
  rmm::device_buffer data(rows.size(), stream, mr);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(data.data(), rows.data(), rows.size(), cudaMemcpyHostToDevice, stream.value()));
  // `rows` is pageable and dropped on return.
  stream.synchronize();
  auto child = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::FLOAT32},
                                              static_cast<cudf::size_type>(count * dim),
                                              std::move(data),
                                              rmm::device_buffer{},
                                              0);
  cudf::numeric_scalar<std::int32_t> const zero(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const width(static_cast<std::int32_t>(dim), true, stream);
  auto list_offsets =
    cudf::sequence(static_cast<cudf::size_type>(count + 1), zero, width, stream, mr);
  return cudf::make_lists_column(static_cast<cudf::size_type>(count),
                                 std::move(list_offsets),
                                 std::move(child),
                                 0,
                                 rmm::device_buffer{});
}

std::size_t pinned_column_chunk_count(const scan_manager::pinned_entry& pin,
                                      const std::string& column_name)
{
  if (pin.tier == cucascade::memory::Tier::HOST) { return pin.host_chunks.size(); }
  if (!pin.device_chunks.empty()) { return pin.device_chunks.size(); }
  auto it = pin.data_batches_by_column.find(column_name);
  if (it == pin.data_batches_by_column.end()) {
    throw internal_exception("VSS: pinned table missing column '" + column_name + "'");
  }
  return it->second.size();
}

namespace {

/// @ref stage_pinned_column_chunk, except that with @p may_decline a chunk whose own reservation
/// the space cannot grant is not staged (nullopt) instead of reported as a shortfall.
std::optional<staged_pinned_chunk> stage_chunk(const scan_manager::pinned_entry& pin,
                                               const std::string& column_name,
                                               std::size_t chunk_index,
                                               cucascade::memory::memory_space& gpu_space,
                                               rmm::cuda_stream_view stream,
                                               const telemetry::batch_telemetry_info& telemetry_info,
                                               staging_charge charge,
                                               bool may_decline)
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
  std::shared_ptr<cucascade::memory::reservation> reservation;
  if (charge == staging_charge::own_reservation) {
    reservation = gpu_space.make_reservation_or_null(bytes);
    if (!reservation && may_decline) { return std::nullopt; }
    if (!reservation) {
      throw_staging_shortfall(
        gpu_space,
        bytes,
        "VSS: staging host-tier column '" + column_name + "' chunk " + std::to_string(chunk_index));
    }
  }

  auto const batch_id = sirius::get_next_batch_id();
  auto batch          = cucascade::data_batch::make(
    batch_id,
    std::move(data_rep),
    telemetry::quent_data_batch_probe::create(telemetry_info, batch_id));
  {
    auto mut = batch->to_mutable();
    if (reservation) {
      mut.convert_to<cucascade::gpu_table_representation>(
        sirius::converter_registry::get(), *reservation, stream);
    } else {
      mut.convert_to<cucascade::gpu_table_representation>(
        sirius::converter_registry::get(), &gpu_space, stream);
    }
    // The converter copies on a stream of its own and binds the copy to it, so the copy's free
    // would be ordered behind nothing the caller does: dropping the chunk right after enqueueing a
    // gather on `stream` hands its memory back while the gather still reads it, and another task
    // can overwrite it first. Freed on the reading stream instead.
    mut.rebind_stream(stream);
  }
  auto view = sirius::get_cudf_table_view(*batch).column(0);
  return staged_pinned_chunk{view, std::move(batch), std::move(reservation)};
}

}  // namespace

staged_pinned_chunk stage_pinned_column_chunk(const scan_manager::pinned_entry& pin,
                                              const std::string& column_name,
                                              std::size_t chunk_index,
                                              cucascade::memory::memory_space& gpu_space,
                                              rmm::cuda_stream_view stream,
                                              const telemetry::batch_telemetry_info& telemetry_info,
                                              staging_charge charge)
{
  return *stage_chunk(
    pin, column_name, chunk_index, gpu_space, stream, telemetry_info, charge, false);
}

pinned_chunk_walk::pinned_chunk_walk(const scan_manager::pinned_entry& pin,
                                     std::string column_name,
                                     cucascade::memory::memory_space& gpu_space,
                                     rmm::cuda_stream_view stream,
                                     const telemetry::batch_telemetry_info& telemetry_info)
  : _pin(pin),
    _column_name(std::move(column_name)),
    _gpu_space(gpu_space),
    _stream(stream),
    _telemetry_info(telemetry_info),
    _n_chunks(pinned_column_chunk_count(pin, _column_name))
{
}

staged_pinned_chunk pinned_chunk_walk::take(std::size_t chunk_index)
{
  if (_ahead && _ahead_index == chunk_index) {
    auto staged = std::move(*_ahead);
    _ahead.reset();
    // Staged on the copy stream; freed behind the caller's work on it instead.
    if (staged.owner) { staged.owner->to_mutable().rebind_stream(_stream); }
    return staged;
  }
  _ahead.reset();
  return stage_pinned_column_chunk(
    _pin, _column_name, chunk_index, _gpu_space, _stream, _telemetry_info);
}

void pinned_chunk_walk::prefetch_after(std::size_t chunk_index)
{
  auto const next = chunk_index + 1;
  if (_pin.tier != cucascade::memory::Tier::HOST || next >= _n_chunks) { return; }
  // The staging copy syncs only the stream it is given, so on this one it does not wait for the
  // caller's work on the current chunk, and the transfer runs alongside it.
  _ahead       = stage_chunk(_pin,
                       _column_name,
                       next,
                       _gpu_space,
                       _copy_stream.view(),
                       _telemetry_info,
                       staging_charge::own_reservation,
                       true);
  _ahead_index = next;
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
  for (auto const& chunk : pin.device_chunks) {
    if (chunk.memory_space != nullptr) { return *chunk.memory_space; }
  }
  throw internal_exception("VSS: pinned table has no GPU-resident chunk (host-tier pin?)");
}

}  // namespace sirius::vss
