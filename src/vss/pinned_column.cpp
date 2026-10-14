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

#include "cuda/vss/cudf_raft_interop.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius/exception.hpp"
#include "vss/bound_gemm.hpp"
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
#include <bit>
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

/// The floats of a HOST-tier chunk laid out as plain rows `dim` floats apart: no nulls, and the
/// list offsets 0, dim, 2 dim, ..., which with no null rows the first and last decide.
struct plain_host_rows {
  cucascade::memory::host_table_allocation const* table;
  std::size_t values_offset;
  std::int64_t rows;
};

std::optional<plain_host_rows> find_plain_host_rows(const scan_manager::pinned_entry& pin,
                                                    const std::string& column_name,
                                                    std::size_t chunk_index,
                                                    std::int64_t dim)
{
  auto const column = find_host_column(pin, column_name, chunk_index);
  if (!column || dim <= 0) { return std::nullopt; }
  auto const& meta  = *column->meta;
  auto const& table = *column->table;
  auto const n      = static_cast<std::int64_t>(meta.num_rows);
  // null_count may also be the "unknown" sentinel, which is as good as a null here.
  if (meta.type_id != static_cast<std::int32_t>(cudf::type_id::LIST) || meta.null_count != 0 ||
      meta.children.size() != 2) {
    return std::nullopt;
  }
  auto const& offsets = meta.children[0];
  auto const& values  = meta.children[1];
  if (offsets.type_id != static_cast<std::int32_t>(cudf::type_id::INT32) || !offsets.has_data ||
      offsets.num_rows != n + 1 ||
      values.type_id != static_cast<std::int32_t>(cudf::type_id::FLOAT32) || !values.has_data ||
      values.null_count != 0 || static_cast<std::int64_t>(values.num_rows) != n * dim ||
      values.data_offset % sizeof(float) != 0) {
    return std::nullopt;
  }
  std::int32_t first = -1;
  std::int32_t last  = -1;
  read_host_bytes(table, offsets.data_offset, sizeof(first), reinterpret_cast<std::byte*>(&first));
  read_host_bytes(table,
                  offsets.data_offset + static_cast<std::size_t>(n) * sizeof(last),
                  sizeof(last),
                  reinterpret_cast<std::byte*>(&last));
  if (first != 0 || static_cast<std::int64_t>(last) != n * dim) { return std::nullopt; }
  return plain_host_rows{&table, values.data_offset, n};
}

template <typename T>
std::unique_ptr<rmm::device_buffer> to_device(std::vector<T> const& host,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto buf = std::make_unique<rmm::device_buffer>(host.size() * sizeof(T), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    buf->data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream.value()));
  return buf;
}

}  // namespace

exact_rows pinned_rows_map::view(std::int64_t const* row_map) const
{
  exact_rows out;
  out.row_map      = row_map;
  out.chunk_rows   = static_cast<std::int64_t const*>(chunk_rows->data());
  out.chunk_block  = static_cast<std::int64_t const*>(chunk_block->data());
  out.chunk_offset = static_cast<std::int64_t const*>(chunk_offset->data());
  out.pin_blocks   = static_cast<std::byte const* const*>(blocks->data());
  out.n_chunks     = n_chunks;
  out.n_rows       = n_rows();
  out.block_shift  = block_shift;
  return out;
}

std::optional<pinned_rows_map> map_pinned_rows(const scan_manager::pinned_entry& pin,
                                               const std::string& column_name,
                                               std::int64_t dim,
                                               cucascade::memory::memory_space& gpu_space,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  pinned_rows_map out;
  std::vector<std::int64_t> chunk_block, chunk_offset;
  std::vector<std::byte const*> blocks;
  out.chunk_first_row.push_back(0);
  if (pin.tier == cucascade::memory::Tier::HOST) {
    std::size_t block_bytes = 0;
    for (std::size_t c = 0; c < pin.host_chunks.size(); ++c) {
      auto const rows = find_plain_host_rows(pin, column_name, c, dim);
      if (!rows) { return std::nullopt; }
      auto const& allocation = *rows->table->allocation;
      if (block_bytes == 0) { block_bytes = allocation.block_size(); }
      if (allocation.block_size() != block_bytes) { return std::nullopt; }
      chunk_block.push_back(static_cast<std::int64_t>(blocks.size()));
      chunk_offset.push_back(static_cast<std::int64_t>(rows->values_offset));
      for (std::size_t b = 0; b < allocation.size(); ++b) {
        void* device = nullptr;
        if (cudaHostGetDevicePointer(&device, allocation.at(b).data(), 0) != cudaSuccess) {
          cudaGetLastError();
          return std::nullopt;
        }
        blocks.push_back(static_cast<std::byte const*>(device));
      }
      out.chunk_first_row.push_back(out.chunk_first_row.back() + rows->rows);
      out.owners.emplace_back(pin.host_chunks[c]);
    }
    if (block_bytes == 0 || (block_bytes & (block_bytes - 1)) != 0) { return std::nullopt; }
    out.block_shift = std::countr_zero(block_bytes);
  } else {
    // A device chunk is one contiguous run of rows: one "block" that no offset leaves.
    std::vector<cudf::column_view> views;
    try {
      views = pinned_column_chunk_views(pin, column_name, gpu_space);
      for (auto const& v : views) {
        blocks.push_back(
          reinterpret_cast<std::byte const*>(list_column_as_dataset_view(v, dim).data_handle()));
        chunk_block.push_back(static_cast<std::int64_t>(chunk_block.size()));
        chunk_offset.push_back(0);
        out.chunk_first_row.push_back(out.chunk_first_row.back() +
                                      static_cast<std::int64_t>(v.size()));
      }
    } catch (std::exception const&) {
      return std::nullopt;
    }
    if (!pin.device_chunks.empty()) {
      for (auto const& chunk : pin.device_chunks) {
        for (auto const& col : chunk.columns) {
          out.owners.emplace_back(col);
        }
      }
    } else if (auto it = pin.data_batches_by_column.find(column_name);
               it != pin.data_batches_by_column.end()) {
      for (auto const& col : it->second) {
        out.owners.emplace_back(col);
      }
    }
    out.block_shift = 62;
  }
  out.n_chunks     = static_cast<std::int64_t>(chunk_block.size());
  out.chunk_rows   = to_device(out.chunk_first_row, stream, mr);
  out.chunk_block  = to_device(chunk_block, stream, mr);
  out.chunk_offset = to_device(chunk_offset, stream, mr);
  out.blocks       = to_device(blocks, stream, mr);
  stream.synchronize();
  return out;
}

std::unique_ptr<cudf::column> gather_pinned_rows(exact_rows const& source,
                                                 std::int64_t start,
                                                 std::int64_t stride,
                                                 std::int64_t count,
                                                 std::int64_t dim,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  rmm::device_buffer data(static_cast<std::size_t>(count * dim) * sizeof(float), stream, mr);
  gather_exact_rows(source, start, stride, count, dim, static_cast<float*>(data.data()), stream);
  auto child = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::FLOAT32},
                                              static_cast<cudf::size_type>(count * dim),
                                              std::move(data),
                                              rmm::device_buffer{},
                                              0);
  cudf::numeric_scalar<std::int32_t> const zero(0, true, stream);
  cudf::numeric_scalar<std::int32_t> const width(static_cast<std::int32_t>(dim), true, stream);
  auto offsets = cudf::sequence(static_cast<cudf::size_type>(count + 1), zero, width, stream, mr);
  return cudf::make_lists_column(static_cast<cudf::size_type>(count),
                                 std::move(offsets),
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
