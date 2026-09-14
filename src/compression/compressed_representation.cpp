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

#include "compressed_representation.hpp"

#include "device_compressed_blob.hpp"

#include <cuda_runtime.h>

#include <cucascade/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace sirius {

// ── Block-aware pinned copies ────────────────────────────────────────────────

void copy_device_to_pinned_blocks(
  const void* src_device,
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& dst,
  std::uint64_t dst_offset,
  std::size_t size,
  rmm::cuda_stream_view stream)
{
  if (size == 0) return;
  const std::size_t bs = dst.block_size();
  std::size_t d_idx    = dst_offset / bs;
  std::size_t d_off    = dst_offset % bs;
  const auto* src      = static_cast<const std::byte*>(src_device);
  std::size_t copied   = 0;
  while (copied < size) {
    const std::size_t chunk = std::min(size - copied, bs - d_off);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      dst.at(d_idx).data() + d_off, src + copied, chunk, cudaMemcpyDeviceToHost, stream.value()));
    copied += chunk;
    d_off += chunk;
    if (d_off == bs) {
      ++d_idx;
      d_off = 0;
    }
  }
}

void copy_pinned_blocks_to_device(
  const cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& src,
  std::uint64_t src_offset,
  void* dst_device,
  std::size_t size,
  rmm::cuda_stream_view stream)
{
  if (size == 0) return;
  const std::size_t bs = src.block_size();
  std::size_t s_idx    = src_offset / bs;
  std::size_t s_off    = src_offset % bs;
  auto* dst            = static_cast<std::byte*>(dst_device);
  std::size_t copied   = 0;
  while (copied < size) {
    const std::size_t chunk = std::min(size - copied, bs - s_off);
    CUCASCADE_CUDA_TRY(cudaMemcpyAsync(
      dst + copied, src.at(s_idx).data() + s_off, chunk, cudaMemcpyHostToDevice, stream.value()));
    copied += chunk;
    s_off += chunk;
    if (s_off == bs) {
      ++s_idx;
      s_off = 0;
    }
  }
}

// ── Owning constructor ───────────────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<pinned_compressed_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _column_sizes(std::move(column_sizes))
{
}

// ── Sharing constructor (private) ────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<pinned_compressed_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::optional<std::vector<std::size_t>> selected_indices,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _selected_indices(std::move(selected_indices)),
    _column_sizes(std::move(column_sizes))
{
}

// ── idata_representation interface ───────────────────────────────────────────

std::unique_ptr<cucascade::idata_representation> compressed_host_representation::clone(
  rmm::cuda_stream_view /*stream*/)
{
  // Share the same backing blob — no byte copy needed.
  auto copy = std::unique_ptr<compressed_host_representation>(
    new compressed_host_representation(get_memory_space(),
                                       _blob,
                                       _column_names,
                                       _compressed_bytes,
                                       _uncompressed_bytes,
                                       _num_rows,
                                       _selected_indices,
                                       _column_sizes));
  // The request is indexed by the selected column list, which the clone shares.
  copy->set_pushdown_scan(_pushdown_scan);
  return copy;
}

// ── Projection ───────────────────────────────────────────────────────────────

namespace {

/// Bytes of the selected columns: exact sum when the pin recorded per-column
/// sizes, otherwise the totals scaled by the selected fraction of columns.
std::pair<std::size_t, std::size_t> projected_bytes(
  std::span<const std::size_t> absolute,
  const std::shared_ptr<const per_column_byte_sizes>& sizes,
  std::size_t total_columns,
  std::size_t total_compressed,
  std::size_t total_uncompressed)
{
  if (sizes && sizes->compressed.size() == total_columns &&
      sizes->uncompressed.size() == total_columns) {
    std::size_t compressed   = 0;
    std::size_t uncompressed = 0;
    for (auto idx : absolute) {
      compressed += sizes->compressed[idx];
      uncompressed += sizes->uncompressed[idx];
    }
    return {compressed, uncompressed};
  }
  if (total_columns == 0) { return {0, 0}; }
  const auto selected = absolute.size();
  return {total_compressed * selected / total_columns,
          total_uncompressed * selected / total_columns};
}

}  // namespace

std::unique_ptr<compressed_host_representation> compressed_host_representation::select_columns(
  std::span<const std::size_t> indices) const
{
  // Build absolute indices into _column_names, respecting any existing projection.
  std::vector<std::size_t> absolute;
  absolute.reserve(indices.size());
  for (auto idx : indices) {
    if (_selected_indices.has_value()) {
      if (idx >= _selected_indices->size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back((*_selected_indices)[idx]);
    } else {
      if (idx >= _column_names.size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back(idx);
    }
  }

  auto const [compressed, uncompressed] = projected_bytes(
    absolute, _column_sizes, _column_names.size(), _compressed_bytes, _uncompressed_bytes);

  // const_cast is safe: select_columns is logically const (it creates a
  // projection sharing the same blob) but the base-class constructor requires
  // a non-const memory_space& — the underlying object is non-const.
  return std::unique_ptr<compressed_host_representation>(new compressed_host_representation(
    const_cast<cucascade::memory::memory_space&>(get_memory_space()),
    _blob,
    _column_names,
    compressed,
    uncompressed,
    _num_rows,
    std::move(absolute),
    _column_sizes));
}

// ── compressed_device_representation ─────────────────────────────────────────

namespace {

// Resolve caller-relative column indices to absolute indices into the chunk's
// full column list, honoring any projection already applied. Shared by the host
// and device select_columns() implementations.
std::vector<std::size_t> resolve_absolute_indices(
  std::span<const std::size_t> indices,
  const std::optional<std::vector<std::size_t>>& existing_selection,
  std::size_t num_all_columns)
{
  std::vector<std::size_t> absolute;
  absolute.reserve(indices.size());
  for (auto idx : indices) {
    if (existing_selection.has_value()) {
      if (idx >= existing_selection->size()) {
        throw std::out_of_range(
          "[compressed_device_representation::select_columns] index out of range");
      }
      absolute.push_back((*existing_selection)[idx]);
    } else {
      if (idx >= num_all_columns) {
        throw std::out_of_range(
          "[compressed_device_representation::select_columns] index out of range");
      }
      absolute.push_back(idx);
    }
  }
  return absolute;
}

}  // namespace

compressed_device_representation::compressed_device_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<compressed_device_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _column_sizes(std::move(column_sizes))
{
}

compressed_device_representation::compressed_device_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<compressed_device_blob> blob,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::optional<std::vector<std::size_t>> selected_indices,
  std::shared_ptr<const per_column_byte_sizes> column_sizes)
  : cucascade::idata_representation(memory_space),
    _blob(std::move(blob)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _selected_indices(std::move(selected_indices)),
    _column_sizes(std::move(column_sizes))
{
}

const simpatico::compressed_table& compressed_device_representation::table() const noexcept
{
  return _blob->table;
}

std::unique_ptr<cucascade::idata_representation> compressed_device_representation::clone(
  rmm::cuda_stream_view /*stream*/)
{
  // Share the same cached blob — no byte copy needed.
  auto copy = std::unique_ptr<compressed_device_representation>(
    new compressed_device_representation(get_memory_space(),
                                         _blob,
                                         _column_names,
                                         _compressed_bytes,
                                         _uncompressed_bytes,
                                         _num_rows,
                                         _selected_indices,
                                         _column_sizes));
  // The request is indexed by the selected column list, which the clone shares.
  copy->set_pushdown_scan(_pushdown_scan);
  return copy;
}

std::unique_ptr<compressed_device_representation> compressed_device_representation::select_columns(
  std::span<const std::size_t> indices) const
{
  auto absolute = resolve_absolute_indices(indices, _selected_indices, _column_names.size());

  auto const [compressed, uncompressed] = projected_bytes(
    absolute, _column_sizes, _column_names.size(), _compressed_bytes, _uncompressed_bytes);

  // const_cast is safe: select_columns is logically const (it creates a
  // projection sharing the same blob) but the base-class constructor requires
  // a non-const memory_space& — the underlying object is non-const.
  return std::unique_ptr<compressed_device_representation>(new compressed_device_representation(
    const_cast<cucascade::memory::memory_space&>(get_memory_space()),
    _blob,
    _column_names,
    compressed,
    uncompressed,
    _num_rows,
    std::move(absolute),
    _column_sizes));
}

}  // namespace sirius
