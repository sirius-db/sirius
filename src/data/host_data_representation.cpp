/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Vendored from cuCascade (b4abc2d:src/data/cpu_data_representation.cpp); see
// src/include/data/host_data_representation.hpp for migration context.

#include <data/host_data_representation.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sirius {

namespace {

inline std::size_t align_up_8(std::size_t v) noexcept
{
  return (v + 7u) & ~static_cast<std::size_t>(7u);
}

std::size_t column_total_bytes(const memory::column_metadata& m)
{
  std::size_t bytes = 0;
  if (m.has_null_mask) bytes += m.null_mask_size;
  if (m.has_data) bytes += m.data_size;
  for (const auto& c : m.children) {
    bytes += column_total_bytes(c);
  }
  return bytes;
}

memory::column_metadata recompute_compact_offsets(const memory::column_metadata& src,
                                                  std::size_t& cursor)
{
  memory::column_metadata dst{};
  dst.type_id    = src.type_id;
  dst.num_rows   = src.num_rows;
  dst.null_count = src.null_count;
  dst.scale      = src.scale;

  dst.has_null_mask  = src.has_null_mask;
  dst.null_mask_size = src.null_mask_size;
  if (src.has_null_mask && src.null_mask_size > 0) {
    cursor               = align_up_8(cursor);
    dst.null_mask_offset = cursor;
    cursor += src.null_mask_size;
  } else {
    dst.null_mask_offset = 0;
  }

  dst.has_data  = src.has_data;
  dst.data_size = src.data_size;
  if (src.has_data && src.data_size > 0) {
    cursor          = align_up_8(cursor);
    dst.data_offset = cursor;
    cursor += src.data_size;
  } else {
    dst.data_offset = 0;
  }

  dst.children.reserve(src.children.size());
  for (const auto& child : src.children) {
    dst.children.push_back(recompute_compact_offsets(child, cursor));
  }
  return dst;
}

void copy_between_blocks(
  const memory::fixed_size_host_memory_resource::multiple_blocks_allocation& src,
  std::size_t src_offset,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& dst,
  std::size_t dst_offset,
  std::size_t size)
{
  if (size == 0) return;
  const std::size_t sb = src.block_size();
  const std::size_t db = dst.block_size();
  std::size_t s_idx    = src_offset / sb;
  std::size_t s_off    = src_offset % sb;
  std::size_t d_idx    = dst_offset / db;
  std::size_t d_off    = dst_offset % db;
  std::size_t copied   = 0;
  while (copied < size) {
    const std::size_t s_avail = sb - s_off;
    const std::size_t d_avail = db - d_off;
    const std::size_t chunk   = std::min({size - copied, s_avail, d_avail});
    std::memcpy(dst.at(d_idx).data() + d_off, src.at(s_idx).data() + s_off, chunk);
    copied += chunk;
    s_off += chunk;
    d_off += chunk;
    if (s_off == sb) {
      ++s_idx;
      s_off = 0;
    }
    if (d_off == db) {
      ++d_idx;
      d_off = 0;
    }
  }
}

void clone_column_buffers(
  const memory::column_metadata& src_meta,
  const memory::column_metadata& dst_meta,
  const memory::fixed_size_host_memory_resource::multiple_blocks_allocation& src,
  memory::fixed_size_host_memory_resource::multiple_blocks_allocation& dst)
{
  if (src_meta.has_null_mask && src_meta.null_mask_size > 0) {
    copy_between_blocks(
      src, src_meta.null_mask_offset, dst, dst_meta.null_mask_offset, src_meta.null_mask_size);
  }
  if (src_meta.has_data && src_meta.data_size > 0) {
    copy_between_blocks(src, src_meta.data_offset, dst, dst_meta.data_offset, src_meta.data_size);
  }
  for (std::size_t i = 0; i < src_meta.children.size(); ++i) {
    clone_column_buffers(src_meta.children[i], dst_meta.children[i], src, dst);
  }
}

}  // namespace

namespace memory {

// =============================================================================
// host_table_allocation
// =============================================================================

host_table_allocation::host_table_allocation(buffers_ptr buffers,
                                             std::vector<column_metadata> cols,
                                             std::size_t data_sz)
  : allocation(std::move(buffers)), columns(std::move(cols)), data_size(data_sz)
{
}

std::unique_ptr<host_table_allocation> host_table_allocation::create(
  fixed_multiple_blocks_allocation buffers,
  std::vector<column_metadata> columns,
  std::size_t data_size)
{
  buffers_ptr shared_buffers(std::move(buffers));
  return std::unique_ptr<host_table_allocation>(
    new host_table_allocation(std::move(shared_buffers), std::move(columns), data_size));
}

std::size_t host_table_allocation::column_size(std::size_t i) const
{
  if (i >= columns.size()) {
    throw std::out_of_range("host_table_allocation::column_size: index out of range");
  }
  return column_total_bytes(columns[i]);
}

std::unique_ptr<host_table_allocation> host_table_allocation::slice(
  std::span<const std::size_t> col_indices) const
{
  std::vector<column_metadata> sliced_cols;
  sliced_cols.reserve(col_indices.size());
  std::size_t sliced_size = 0;
  for (std::size_t idx : col_indices) {
    if (idx >= columns.size()) {
      throw std::out_of_range("host_table_allocation::slice: column index out of range");
    }
    sliced_cols.push_back(columns[idx]);
    sliced_size += column_total_bytes(columns[idx]);
  }
  return std::unique_ptr<host_table_allocation>(
    new host_table_allocation(allocation, std::move(sliced_cols), sliced_size));
}

std::unique_ptr<host_table_allocation> host_table_allocation::clone(memory_space& target) const
{
  auto* mr = target.get_memory_resource_as<fixed_size_host_memory_resource>();
  if (mr == nullptr) {
    throw std::runtime_error(
      "host_table_allocation::clone: target memory_space has no fixed_size_host_memory_resource");
  }

  std::size_t cursor = 0;
  std::vector<column_metadata> new_cols;
  new_cols.reserve(columns.size());
  for (const auto& c : columns) {
    new_cols.push_back(recompute_compact_offsets(c, cursor));
  }
  const std::size_t new_data_size = cursor;

  auto new_alloc = mr->allocate_multiple_blocks(new_data_size);

  if (allocation && new_alloc) {
    for (std::size_t i = 0; i < columns.size(); ++i) {
      clone_column_buffers(columns[i], new_cols[i], *allocation, *new_alloc);
    }
  }

  buffers_ptr shared_new(std::move(new_alloc));
  return std::unique_ptr<host_table_allocation>(
    new host_table_allocation(std::move(shared_new), std::move(new_cols), new_data_size));
}

}  // namespace memory

// =============================================================================
// host_data_representation
// =============================================================================

host_data_representation::host_data_representation(
  std::unique_ptr<memory::host_table_allocation> host_table,
  cucascade::memory::memory_space* memory_space)
  : idata_representation(*memory_space), _host_table(std::move(host_table))
{
}

std::size_t host_data_representation::get_size_in_bytes() const { return _host_table->data_size; }

std::size_t host_data_representation::get_uncompressed_data_size_in_bytes() const
{
  return get_size_in_bytes();
}

const std::unique_ptr<memory::host_table_allocation>& host_data_representation::get_host_table()
  const
{
  return _host_table;
}

std::size_t host_data_representation::num_columns() const noexcept
{
  return _host_table->num_columns();
}

std::size_t host_data_representation::column_size(std::size_t i) const
{
  return _host_table->column_size(i);
}

std::unique_ptr<host_data_representation> host_data_representation::slice(
  std::span<const std::size_t> col_indices) const
{
  auto sliced = _host_table->slice(col_indices);
  return std::make_unique<host_data_representation>(
    std::move(sliced), &const_cast<cucascade::memory::memory_space&>(get_memory_space()));
}

std::unique_ptr<idata_representation> host_data_representation::clone(
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
  auto cloned = _host_table->clone(get_memory_space());
  return std::make_unique<host_data_representation>(std::move(cloned), &get_memory_space());
}

// =============================================================================
// host_data_packed_representation
// =============================================================================

host_data_packed_representation::host_data_packed_representation(
  std::unique_ptr<memory::host_table_packed_allocation> host_table,
  cucascade::memory::memory_space* memory_space)
  : idata_representation(*memory_space), _host_table(std::move(host_table))
{
}

std::size_t host_data_packed_representation::get_size_in_bytes() const
{
  return _host_table->data_size;
}

std::size_t host_data_packed_representation::get_uncompressed_data_size_in_bytes() const
{
  return get_size_in_bytes();
}

const std::unique_ptr<memory::host_table_packed_allocation>&
host_data_packed_representation::get_host_table() const
{
  return _host_table;
}

std::unique_ptr<idata_representation> host_data_packed_representation::clone(
  [[maybe_unused]] rmm::cuda_stream_view stream)
{
  // Get the host memory resource from the memory space
  auto* host_mr = get_memory_space().get_memory_resource_of<memory::Tier::HOST>();
  if (!host_mr) {
    throw std::runtime_error(
      "Cannot clone host_data_packed_representation: no host memory resource");
  }

  // Allocate new blocks for the copy
  auto allocation_copy = host_mr->allocate_multiple_blocks(_host_table->data_size);

  // Copy data block by block
  const auto& src_blocks = _host_table->allocation->get_blocks();
  auto dst_blocks        = allocation_copy->get_blocks();
  std::size_t remaining  = _host_table->data_size;
  std::size_t block_size = _host_table->allocation->block_size();

  for (std::size_t i = 0; i < src_blocks.size() && remaining > 0; ++i) {
    std::size_t copy_size = std::min(remaining, block_size);
    std::memcpy(dst_blocks[i], src_blocks[i], copy_size);
    remaining -= copy_size;
  }

  // Clone the metadata
  auto metadata_copy = std::make_unique<std::vector<uint8_t>>(*_host_table->metadata);

  // Create the new host_table_packed_allocation
  auto host_table_copy = std::make_unique<memory::host_table_packed_allocation>(
    std::move(allocation_copy), std::move(metadata_copy), _host_table->data_size);

  return std::make_unique<host_data_packed_representation>(std::move(host_table_copy),
                                                           &get_memory_space());
}

}  // namespace sirius
