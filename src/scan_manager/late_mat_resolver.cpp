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

#include "scan_manager/late_mat_resolver.hpp"

#include "compression/compressed_representation.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

#include <cudf/column/column.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda_runtime.h>

#include <api/simpatico_codegen.hpp>

#include <cstdint>
#include <memory>
#include <utility>

namespace sirius::scan_manager {

namespace {

/// A GPU-tier entry stores its chunks one of two ways, and only one is
/// populated: device_pin_chunks (compression-enabled, possibly interleaved) or
/// a per-column-name map (the plain pin). Reading the wrong one silently gives
/// an entry with no chunks, so the choice is made once, here.
bool uses_device_chunks(pinned_entry const& entry) { return !entry.device_chunks.empty(); }

/// Rows in one device_pin_chunk, whichever form it took.
std::int64_t chunk_rows(sirius::device_pin_chunk const& chunk)
{
  if (chunk.compressed) { return chunk.compressed->num_rows(); }
  if (chunk.columns.empty() || !chunk.columns.front()) { return 0; }
  return chunk.columns.front()->size();
}

/// Whether this device can dereference a registered pinned host pointer as it
/// stands. The two attributes together are the documented condition under which
/// a pinned host allocation's own address is a valid device address; without
/// them the blocks would have to be translated or staged, and a host-tier
/// deferral is refused instead.
///
/// Asked once per process. Host-tier deferral is admitted only for a single-GPU
/// pin, so there is no second device whose answer could differ.
bool host_pins_are_device_addressable()
{
  static bool const addressable = [] {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return false; }
    int unified      = 0;
    int host_pointer = 0;
    if (cudaDeviceGetAttribute(&unified, cudaDevAttrUnifiedAddressing, device) != cudaSuccess) {
      return false;
    }
    if (cudaDeviceGetAttribute(
          &host_pointer, cudaDevAttrCanUseHostPointerForRegisteredMem, device) != cudaSuccess) {
      return false;
    }
    return unified != 0 && host_pointer != 0;
  }();
  return addressable;
}

/// The uncompressed host representation of chunk @p index, or nullptr when the
/// chunk is absent or Simpatico-compressed. compressed_host_representation is a
/// sibling of host_data_representation rather than a subclass, so the cast
/// separates the two.
cucascade::host_data_representation const* plain_host_chunk(pinned_entry const& entry,
                                                            std::size_t index)
{
  if (index >= entry.host_chunks.size() || !entry.host_chunks[index]) { return nullptr; }
  return dynamic_cast<cucascade::host_data_representation const*>(entry.host_chunks[index].get());
}

/// One host chunk's column, as the blocked gather addresses it, or nullopt when
/// it cannot be addressed that way. @p dtype carries the carrier agreed so far
/// and is filled in from the first chunk.
std::optional<late_mat::batch_source> host_batch_source(pinned_entry const& entry,
                                                        std::size_t index,
                                                        std::size_t column_position,
                                                        cudf::data_type& dtype)
{
  auto const* chunk = plain_host_chunk(entry, index);
  if (chunk == nullptr) { return std::nullopt; }
  auto const& allocation = chunk->get_host_table();
  if (!allocation || !allocation->allocation) { return std::nullopt; }
  if (column_position >= allocation->columns.size()) { return std::nullopt; }
  auto const& meta = allocation->columns[column_position];

  // Children mean a nested or variable-width column: there is no element width
  // to multiply a row by, and its offsets would have to be rebuilt against
  // buffers that are not contiguous.
  if (!meta.children.empty()) { return std::nullopt; }
  auto const carrier = host_column_carrier(meta);
  if (!cudf::is_fixed_width(carrier)) { return std::nullopt; }
  if (dtype.id() == cudf::type_id::EMPTY) {
    dtype = carrier;
  } else if (dtype != carrier) {
    return std::nullopt;
  }

  auto const elem_size  = cudf::size_of(carrier);
  auto const block_size = allocation->allocation->block_size();
  if (elem_size == 0 || block_size == 0) { return std::nullopt; }

  auto buffers        = std::make_shared<late_mat::host_blocked_buffers>();
  buffers->block_size = block_size;
  buffers->blocks.reserve(allocation->allocation->size());
  for (std::size_t block = 0; block < allocation->allocation->size(); ++block) {
    buffers->blocks.push_back(allocation->allocation->at(block).data());
  }
  auto const addressable_bytes = buffers->blocks.size() * block_size;

  auto const rows = static_cast<std::int64_t>(meta.num_rows);
  if (rows > 0) {
    // A zero-row chunk carries no data buffer at all; anything else must.
    if (!meta.has_data) { return std::nullopt; }
    // The gather turns a row into a byte offset by multiplying, then divides
    // that by the block size. An element straddling a block boundary — or a
    // buffer not starting on an element — has no address it can compute.
    if (block_size % elem_size != 0 || meta.data_offset % elem_size != 0) { return std::nullopt; }
    if (meta.data_offset + (static_cast<std::size_t>(rows) * elem_size) > addressable_bytes) {
      return std::nullopt;
    }
    buffers->data_offset = meta.data_offset;
  }

  if (meta.has_null_mask) {
    // Validity is read one mask word at a time, in the same coordinates.
    constexpr std::size_t kMaskWord = sizeof(cudf::bitmask_type);
    if (block_size % kMaskWord != 0 || meta.null_mask_offset % kMaskWord != 0) {
      return std::nullopt;
    }
    if (meta.null_mask_offset + meta.null_mask_size > addressable_bytes) { return std::nullopt; }
    buffers->has_null_mask    = true;
    buffers->null_mask_offset = meta.null_mask_offset;
  }

  late_mat::batch_source source;
  source.num_rows = rows;
  source.host     = std::move(buffers);
  return source;
}

/// Rows in host chunk @p index, read off whichever column the chunk holds first
/// — every column of a chunk shares its row count.
std::optional<std::int64_t> host_chunk_rows(pinned_entry const& entry, std::size_t index)
{
  auto const* chunk = plain_host_chunk(entry, index);
  if (chunk == nullptr) { return std::nullopt; }
  auto const& allocation = chunk->get_host_table();
  if (!allocation || allocation->columns.empty()) { return std::nullopt; }
  return static_cast<std::int64_t>(allocation->columns.front().num_rows);
}

}  // namespace

bool host_pinned_column_is_addressable(pinned_entry const& entry, std::size_t column_position)
{
  if (entry.tier != cucascade::memory::Tier::HOST) { return false; }
  if (entry.host_chunks.empty()) { return false; }
  if (!host_pins_are_device_addressable()) { return false; }
  cudf::data_type dtype{cudf::type_id::EMPTY};
  for (std::size_t index = 0; index < entry.host_chunks.size(); ++index) {
    if (!host_batch_source(entry, index, column_position, dtype)) { return false; }
  }
  return true;
}

std::optional<late_mat::pinned_table_layout> resolve_pinned_layout(
  late_mat::column_origin const& origin)
{
  auto const* entry = origin.resolve();
  if (entry == nullptr) { return std::nullopt; }
  if (entry->tier == cucascade::memory::Tier::HOST) {
    std::vector<std::int64_t> host_rows;
    host_rows.reserve(entry->host_chunks.size());
    for (std::size_t index = 0; index < entry->host_chunks.size(); ++index) {
      auto const rows_in_chunk = host_chunk_rows(*entry, index);
      if (!rows_in_chunk) { return std::nullopt; }
      host_rows.push_back(*rows_in_chunk);
    }
    if (host_rows.empty()) { return std::nullopt; }
    return late_mat::pinned_table_layout::from_batch_rows(std::move(host_rows), origin.generation);
  }
  if (entry->tier != cucascade::memory::Tier::GPU) { return std::nullopt; }

  std::vector<std::int64_t> rows;
  if (uses_device_chunks(*entry)) {
    rows.reserve(entry->device_chunks.size());
    for (auto const& chunk : entry->device_chunks) {
      rows.push_back(chunk_rows(chunk));
    }
  } else {
    // The plain pin holds one chunk vector per column NAME; every column has
    // the same chunking, so any present column gives the layout.
    auto const& names = entry->cache_info.column_names();
    if (names.empty()) { return std::nullopt; }
    auto const it = entry->data_batches_by_column.find(names.front());
    if (it == entry->data_batches_by_column.end()) { return std::nullopt; }
    rows.reserve(it->second.size());
    for (auto const& col : it->second) {
      if (!col) { return std::nullopt; }
      rows.push_back(col->size());
    }
  }
  if (rows.empty()) { return std::nullopt; }
  return late_mat::pinned_table_layout::from_batch_rows(std::move(rows), origin.generation);
}

std::optional<late_mat::pinned_column_view> resolve_pinned_column(
  late_mat::column_origin const& origin)
{
  auto const* entry = origin.resolve();
  if (entry == nullptr) { return std::nullopt; }

  auto const& names = entry->cache_info.column_names();
  if (origin.column_pos >= names.size()) { return std::nullopt; }

  late_mat::pinned_column_view view;
  view.pin_generation = origin.generation;

  if (entry->tier == cucascade::memory::Tier::HOST) {
    if (entry->host_chunks.empty() || !host_pins_are_device_addressable()) { return std::nullopt; }
    for (std::size_t index = 0; index < entry->host_chunks.size(); ++index) {
      auto source = host_batch_source(*entry, index, origin.column_pos, view.dtype);
      if (!source) { return std::nullopt; }
      view.batches.push_back(std::move(*source));
    }
    if (view.batches.empty()) { return std::nullopt; }
    return view;
  }
  if (entry->tier != cucascade::memory::Tier::GPU) { return std::nullopt; }

  if (uses_device_chunks(*entry)) {
    // Every uncompressed gather shape (single-batch, multi-batch fixed-width, multi-batch
    // variable-width) propagates validity; only a compressed origin cannot (materialize.cpp's
    // require_non_null, at decode time).
    for (auto const& chunk : entry->device_chunks) {
      late_mat::batch_source source;
      source.num_rows = chunk_rows(chunk);
      if (chunk.compressed) {
        if (!chunk.compressed->has_table()) { return std::nullopt; }
        source.compressed   = chunk.compressed.get();
        source.column_index = origin.column_pos;
        // The compressed table carries its own carrier dtype per column, and it
        // is the one the decode re-tags to. That carrier is chosen PER CHUNK —
        // narrow_pin_chunk() computes each chunk's exact range independently, so
        // one logical BIGINT can be stored INT8 in one chunk and INT16 in the
        // next. The view carries a single dtype, which the gather reads every
        // batch at, so a pin whose chunks disagree has no single answer here and
        // is refused rather than read at the first chunk's width.
        auto const& table = chunk.compressed->table();
        if (origin.column_pos >= table.columns.size()) { return std::nullopt; }
        auto const carrier = table.columns[origin.column_pos].dtype;
        if (view.dtype.id() == cudf::type_id::EMPTY) {
          view.dtype = carrier;
        } else if (view.dtype != carrier) {
          return std::nullopt;
        }
      } else {
        if (origin.column_pos >= chunk.columns.size() || !chunk.columns[origin.column_pos]) {
          return std::nullopt;
        }
        // Same per-chunk carrier caveat as the compressed branch above: with
        // compressed materialization on, narrow_pin_chunk() narrows every chunk
        // it materializes, including ones the sink then stores uncompressed.
        auto const& col     = *chunk.columns[origin.column_pos];
        source.uncompressed = col.view();
        if (view.dtype.id() == cudf::type_id::EMPTY) {
          view.dtype = col.type();
        } else if (view.dtype != col.type()) {
          return std::nullopt;
        }
      }
      view.batches.push_back(source);
    }
  } else {
    auto const it = entry->data_batches_by_column.find(names[origin.column_pos]);
    if (it == entry->data_batches_by_column.end()) { return std::nullopt; }
    for (auto const& col : it->second) {
      if (!col) { return std::nullopt; }
      late_mat::batch_source source;
      source.uncompressed = col->view();
      source.num_rows     = col->size();
      view.batches.push_back(source);
      if (view.dtype.id() == cudf::type_id::EMPTY) {
        view.dtype = col->type();
      } else if (view.dtype != col->type()) {
        return std::nullopt;
      }
    }
  }

  if (view.batches.empty()) { return std::nullopt; }
  return view;
}

}  // namespace sirius::scan_manager
