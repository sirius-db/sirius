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

#include "exchange/batch_io.hpp"

#include "cuda/device_copy_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "sirius/exception.hpp"
#include "telemetry/data_batch_probe.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>

#include <rmm/aligned.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <utility>

namespace sirius::exchange {

using cucascade::memory::column_metadata;
using cucascade::memory::Tier;

namespace {

/// Copies between the device range [@p device, +@p size) and host blocks @p blocks at logical
/// offset @p at, one piece per block the range touches.
void add_block_copies(
  sirius::cuda::device_copy_batch& copies,
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation const& blocks,
  std::size_t at,
  std::uint8_t const* device,
  std::size_t size)
{
  auto const block_size = blocks.block_size();
  std::size_t done      = 0;
  while (done < size) {
    auto const block  = (at + done) / block_size;
    auto const within = (at + done) % block_size;
    auto const n      = std::min(size - done, block_size - within);
    copies.add(reinterpret_cast<std::uint8_t*>(blocks.at(block).data()) + within, device + done, n);
    done += n;
  }
}

/// Rebuilds one column from its buffers, consumed in walk order from @p next.
std::unique_ptr<cudf::column> rebuild(column_metadata const& c,
                                      std::vector<rmm::device_buffer>::iterator& next)
{
  auto const id   = static_cast<cudf::type_id>(c.type_id);
  auto const rows = static_cast<cudf::size_type>(c.num_rows);
  rmm::device_buffer mask =
    c.has_null_mask && c.null_mask_size > 0 ? std::move(*next++) : rmm::device_buffer{};
  rmm::device_buffer data =
    c.has_data && c.data_size > 0 ? std::move(*next++) : rmm::device_buffer{};
  if (id == cudf::type_id::STRING) {
    auto const& o = c.children.front();
    rmm::device_buffer offsets_data =
      o.has_data && o.data_size > 0 ? std::move(*next++) : rmm::device_buffer{};
    auto offsets =
      std::make_unique<cudf::column>(cudf::data_type{static_cast<cudf::type_id>(o.type_id)},
                                     rows + 1,
                                     std::move(offsets_data),
                                     rmm::device_buffer{},
                                     0);
    return cudf::make_strings_column(
      rows, std::move(offsets), std::move(data), c.null_count, std::move(mask));
  }
  cudf::data_type const type =
    cudf::is_fixed_point(cudf::data_type{id}) ? cudf::data_type{id, c.scale} : cudf::data_type{id};
  return std::make_unique<cudf::column>(type, rows, std::move(data), std::move(mask), c.null_count);
}

}  // namespace

// ---- outgoing ------------------------------------------------------------------------------

outgoing_batch::outgoing_batch(std::shared_ptr<cucascade::data_batch> batch,
                               cucascade::read_only_data_batch read_only)
  : _batch(std::move(batch)), _read_only(std::move(read_only))
{
}

std::optional<outgoing_batch> outgoing_batch::make(std::shared_ptr<cucascade::data_batch> batch,
                                                   rmm::cuda_stream_view stream)
{
  auto read_only = batch->to_read_only();
  if (read_only.get_current_tier() != Tier::GPU) {
    throw sirius::invalid_input_exception("exchange: the batch to send is not on the GPU");
  }
  auto view = get_cudf_table_view(read_only);
  if (view.num_rows() == 0) { return std::nullopt; }
  if (view.num_columns() == 0) {
    throw sirius::invalid_input_exception("exchange cannot send a batch with no columns");
  }
  if (cudaEvent_t writer = read_only.get_writer_event()) {
    RMM_CUDA_TRY(cudaStreamWaitEvent(stream.value(), writer, 0));
  }
  auto* space = read_only.get_memory_space();
  outgoing_batch out{std::move(batch), std::move(read_only)};
  if (is_sliced(view)) {
    out._unsliced = std::make_unique<cudf::table>(view, stream, space->get_default_allocator());
    view          = out._unsliced->view();
  }
  std::size_t stream_bytes = 0;
  out._header.columns      = plan_stream(view, stream, stream_bytes);
  out._header.rows         = static_cast<std::uint64_t>(view.num_rows());
  out._header.stream_bytes = stream_bytes;
  out._buffers             = stream_map(out._header.columns);
  out._sources             = stream_sources(view, out._header.columns);
  if (out._sources.size() != out._buffers.size()) {
    throw sirius::internal_exception(
      "exchange: {} sources for {} stream buffers", out._sources.size(), out._buffers.size());
  }
  return out;
}

void outgoing_batch::copy_range(std::size_t begin,
                                std::size_t end,
                                void* dst,
                                rmm::cuda_stream_view stream) const
{
  if (begin > end || end > _header.stream_bytes) {
    throw sirius::invalid_input_exception(
      "exchange: range [{}, {}) outside a {}-byte stream", begin, end, _header.stream_bytes);
  }
  sirius::cuda::device_copy_batch copies;
  auto* out = static_cast<std::uint8_t*>(dst);
  for (auto const& piece : pieces_in(_buffers, begin, end)) {
    copies.add(out + piece.range_offset,
               static_cast<std::uint8_t const*>(_sources[piece.buffer]) + piece.buffer_offset,
               piece.size);
  }
  RMM_CUDA_TRY(copies.enqueue(stream));
}

// ---- incoming ------------------------------------------------------------------------------

incoming_batch::incoming_batch(batch_header header, cucascade::memory::memory_space& space)
  : _header(std::move(header)), _buffers(stream_map(_header.columns)), _space(&space)
{
}

std::unique_ptr<incoming_batch> incoming_batch::land(batch_header header,
                                                     cucascade::memory::memory_space& gpu,
                                                     cucascade::memory::memory_space* host,
                                                     rmm::cuda_stream_view stream)
{
  std::unique_ptr<incoming_batch> out{new incoming_batch(std::move(header), gpu)};

  // 1. GPU: a reservation only if one is free now, and an allocation that falls over to host
  //    instead of waiting.
  std::size_t gpu_need = 0;
  for (auto const& b : out->_buffers) {
    gpu_need += rmm::align_up(b.size, rmm::CUDA_ALLOCATION_ALIGNMENT);
  }
  if (auto reservation = gpu.make_reservation_or_null(gpu_need)) {
    auto* tracker = gpu.get_memory_resource_of<Tier::GPU>();
    if (tracker->attach_reservation_to_tracker(
          stream,
          std::move(reservation),
          std::make_unique<cucascade::memory::fail_reservation_limit_policy>())) {
      absl::Cleanup detach = [&] { tracker->reset_stream_reservation(stream); };
      try {
        out->_gpu_buffers.reserve(out->_buffers.size());
        for (auto const& b : out->_buffers) {
          out->_gpu_buffers.emplace_back(b.size, stream, gpu.get_default_allocator());
        }
        return out;
      } catch (rmm::out_of_memory const&) {
        out->_gpu_buffers.clear();
      }
    }
  }

  // 2. Host blocks, laid out at the stream's own offsets.
  if (host == nullptr) { return nullptr; }
  auto const bytes = out->_header.stream_bytes;
  if (auto reservation = host->make_reservation_or_null(bytes)) {
    auto* mr = host->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
    if (mr == nullptr) { return nullptr; }
    try {
      out->_host_blocks = mr->allocate_multiple_blocks(bytes, reservation.get());
      out->_tier        = Tier::HOST;
      out->_space       = host;
      return out;
    } catch (rmm::out_of_memory const&) {
      return nullptr;
    }
  }
  return nullptr;
}

void incoming_batch::copy_in(std::size_t begin,
                             void const* src,
                             std::size_t size,
                             rmm::cuda_stream_view stream)
{
  if (begin > _header.stream_bytes || size > _header.stream_bytes - begin) {
    throw sirius::invalid_input_exception(
      "exchange: {} bytes at {} outside a {}-byte stream", size, begin, _header.stream_bytes);
  }
  auto const* in = static_cast<std::uint8_t const*>(src);
  sirius::cuda::device_copy_batch copies;
  if (_tier == Tier::GPU) {
    for (auto const& piece : pieces_in(_buffers, begin, begin + size)) {
      copies.add(
        static_cast<std::uint8_t*>(_gpu_buffers[piece.buffer].data()) + piece.buffer_offset,
        in + piece.range_offset,
        piece.size);
    }
  } else {
    add_block_copies(copies, *_host_blocks, begin, in, size);
  }
  RMM_CUDA_TRY(copies.enqueue(stream));
}

std::shared_ptr<cucascade::data_batch> incoming_batch::seal(rmm::cuda_stream_view stream)
{
  auto const batch_id = sirius::get_next_batch_id();
  auto probe          = telemetry::quent_data_batch_probe::create({}, batch_id);
  if (_tier == Tier::HOST) {
    auto table = cucascade::memory::host_table_allocation::create(
      std::move(_host_blocks), _header.columns, _header.stream_bytes);
    return cucascade::data_batch::make(
      batch_id,
      std::make_unique<cucascade::host_data_representation>(std::move(table), _space),
      std::move(probe));
  }
  // Freed later on one of the space's streams; the copies are ordered before the batch's writer
  // event, recorded on @p stream.
  for (auto& buffer : _gpu_buffers) {
    buffer.set_stream(_space->acquire_stream());
  }
  auto next = _gpu_buffers.begin();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(_header.columns.size());
  for (auto const& c : _header.columns) {
    columns.push_back(rebuild(c, next));
  }
  _gpu_buffers.clear();
  auto repr = std::make_unique<cucascade::gpu_table_representation>(
    std::make_unique<cudf::table>(std::move(columns)), *_space, stream);
  if (repr->get_writer_event() == nullptr) { repr->record_writer_event(stream); }
  return cucascade::data_batch::make(batch_id, std::move(repr), std::move(probe));
}

}  // namespace sirius::exchange
