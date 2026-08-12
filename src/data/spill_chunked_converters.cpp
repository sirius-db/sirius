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

#include "data/spill_chunked_converters.hpp"

#include "data/chunked_spill_copy.hpp"
#include "log/logging.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cassert>
#include <cstdint>
#include <vector>

namespace sirius {
namespace spill {

namespace {

// Layout planning is a faithful transcription of the cucascade fast converter's planning pass
// (cucascade/src/cudf/representation_converter_builtins.cpp, convert_gpu_to_host_fast). Keeping
// the algorithm identical keeps the emitted column_metadata layout byte-identical, so the
// unchanged builtin HOST->GPU converter can restore from it.

/// @p alignment must be a power of two.
std::size_t align_up(std::size_t offset, std::size_t alignment) noexcept
{
  return (offset + alignment - 1u) & ~(alignment - 1u);
}

/// True for column types that store element data in a flat device buffer. Nested types carry
/// their payload in children instead (STRING's chars buffer is handled in plan_column_copy).
bool column_has_data_buffer(const cudf::column_view& col) noexcept
{
  switch (col.type().id()) {
    case cudf::type_id::STRING:
    case cudf::type_id::LIST:
    case cudf::type_id::STRUCT:
    case cudf::type_id::DICTIONARY32:
    case cudf::type_id::EMPTY: return false;
    default: return true;
  }
}

/**
 * @brief Recursively plan the buffer layout for one column, filling in column_metadata.
 *
 * @note Assumes the column view has offset == 0 (i.e. the table is not a slice) — same contract
 * as the builtin converter this replaces.
 */
cucascade::memory::column_metadata plan_column_copy(const cudf::column_view& col,
                                                    std::size_t& current_offset,
                                                    rmm::cuda_stream_view stream)
{
  assert(col.offset() == 0 && "column_view with non-zero offset is not supported");

  cucascade::memory::column_metadata meta{};
  meta.type_id    = static_cast<int32_t>(col.type().id());
  meta.num_rows   = col.size();
  meta.null_count = col.null_count();
  meta.scale      = 0;

  if (col.type().id() == cudf::type_id::DECIMAL32 || col.type().id() == cudf::type_id::DECIMAL64 ||
      col.type().id() == cudf::type_id::DECIMAL128) {
    meta.scale = col.type().scale();
  }

  if (col.nullable()) {
    meta.has_null_mask    = true;
    meta.null_mask_size   = cudf::bitmask_allocation_size_bytes(col.size());
    current_offset        = align_up(current_offset, 8u);
    meta.null_mask_offset = current_offset;
    current_offset += meta.null_mask_size;
  } else {
    meta.has_null_mask    = false;
    meta.null_mask_offset = 0;
    meta.null_mask_size   = 0;
  }

  if (col.type().id() == cudf::type_id::STRING) {
    // Offsets live in child(0) and may be INT32 or INT64 (large strings); chars_size() handles
    // both widths.
    if (col.size() > 0 && col.num_children() > 0 && col.data<char>() != nullptr) {
      auto const chars_bytes = cudf::strings_column_view(col).chars_size(stream);
      if (chars_bytes > 0) {
        current_offset   = align_up(current_offset, 8u);
        meta.has_data    = true;
        meta.data_offset = current_offset;
        meta.data_size   = static_cast<std::size_t>(chars_bytes);
        current_offset += meta.data_size;
      } else {
        meta.has_data    = false;
        meta.data_offset = 0;
        meta.data_size   = 0;
      }
    } else {
      meta.has_data    = false;
      meta.data_offset = 0;
      meta.data_size   = 0;
    }
  } else if (column_has_data_buffer(col) && col.size() > 0) {
    meta.has_data    = true;
    meta.data_size   = static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
    current_offset   = align_up(current_offset, 8u);
    meta.data_offset = current_offset;
    current_offset += meta.data_size;
  } else {
    meta.has_data    = false;
    meta.data_offset = 0;
    meta.data_size   = 0;
  }

  meta.children.reserve(static_cast<std::size_t>(col.num_children()));
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    meta.children.push_back(plan_column_copy(col.child(i), current_offset, stream));
  }

  return meta;
}

/**
 * @brief Queue copy ops for @p size device bytes into the host allocation.
 *
 * Split at host block boundaries so each op's dst is contiguous within one pinned block; the
 * batcher splits further at chunk boundaries.
 */
void collect_d2h_ops(
  const void* src,
  std::size_t size,
  std::size_t alloc_offset,
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  chunked_copy_batcher& batcher)
{
  if (size == 0 || src == nullptr) { return; }

  const std::size_t block_size = alloc.block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t src_off          = 0;

  while (src_off < size) {
    std::size_t remaining      = size - src_off;
    std::size_t space_in_block = block_size - block_off;
    std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block = alloc.at(block_idx);
    batcher.add(
      block.data() + block_off, static_cast<const uint8_t*>(src) + src_off, bytes_to_copy);
    src_off += bytes_to_copy;
    block_off += bytes_to_copy;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
}

/// Recursively queue D2H copy ops for a column's null mask, data buffer, and children.
void collect_column_d2h_ops(
  const cudf::column_view& col,
  const cucascade::memory::column_metadata& meta,
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& alloc,
  chunked_copy_batcher& batcher)
{
  if (meta.has_null_mask) {
    collect_d2h_ops(col.null_mask(), meta.null_mask_size, meta.null_mask_offset, alloc, batcher);
  }
  if (meta.has_data) {
    collect_d2h_ops(col.data<uint8_t>(), meta.data_size, meta.data_offset, alloc, batcher);
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    collect_column_d2h_ops(
      col.child(i), meta.children[static_cast<std::size_t>(i)], alloc, batcher);
  }
}

/// @brief Submit one chunk of D2H copies on @p stream. The dsts/srcs/sizes vectors are owned by
/// the caller and reused across chunks to avoid per-chunk reallocation.
void submit_chunk(std::span<const copy_op> ops,
                  rmm::cuda_stream_view stream,
                  std::vector<void*>& dsts,
                  std::vector<const void*>& srcs,
                  std::vector<std::size_t>& sizes)
{
  if (ops.empty()) { return; }
  dsts.clear();
  srcs.clear();
  sizes.clear();
  dsts.reserve(ops.size());
  srcs.reserve(ops.size());
  sizes.reserve(ops.size());
  for (auto const& op : ops) {
    dsts.push_back(op.dst);
    srcs.push_back(op.src);
    sizes.push_back(op.size);
  }
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  attr.flags          = cudaMemcpyFlagDefault;
  // cudaMemcpyBatchAsync requires a real (non-default) stream. CUDA 12.x takes a failIdx
  // parameter that was removed in CUDA 13.
#if CUDART_VERSION < 13000
  RMM_CUDA_TRY(cudaMemcpyBatchAsync(
    dsts.data(), srcs.data(), sizes.data(), dsts.size(), attr, nullptr, stream.value()));
#else
  RMM_CUDA_TRY(cudaMemcpyBatchAsync(
    dsts.data(), srcs.data(), sizes.data(), dsts.size(), attr, stream.value()));
#endif
#else
  // cudaMemcpyBatchAsync requires CUDA 12.8+.
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    RMM_CUDA_TRY(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.value()));
  }
#endif
}

/**
 * @brief Chunked GPU -> HOST conversion (drop-in replacement for the builtin fast converter).
 *
 * Same three passes as the builtin (plan layout, allocate pinned blocks, copy), but pass 3
 * submits copies every ~chunk_bytes instead of hoarding every op for one monolithic batched
 * call, so the DMA engine drains chunk k while the column-tree walk collects chunk k+1. The
 * final synchronize preserves the converter contract: source buffers are dead after return.
 */
std::unique_ptr<cucascade::idata_representation> convert_gpu_to_host_chunked(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  cucascade::memory::reservation* reservation,
  std::size_t chunk_bytes)
{
  nvtx3::scoped_range nvtx_range{"sirius::spill::gpu_to_host_chunked"};

  auto& gpu_source            = source.cast<cucascade::gpu_table_representation>();
  const cudf::table_view view = gpu_source.get_table_view();

  // Pass 1: plan the allocation layout.
  std::size_t current_offset = 0;
  std::vector<cucascade::memory::column_metadata> columns;
  columns.reserve(static_cast<std::size_t>(view.num_columns()));
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    columns.push_back(plan_column_copy(view.column(i), current_offset, stream));
  }
  const std::size_t total_size = current_offset;

  // Pass 2: allocate pinned host blocks (draws down the caller's reservation).
  auto mr = target_memory_space
              ->get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  auto allocation = mr->allocate_multiple_blocks(total_size, reservation);

  // Pass 3: collect D2H ops, flushing every ~chunk_bytes as the tree is walked.
  std::vector<void*> dsts;
  std::vector<const void*> srcs;
  std::vector<std::size_t> sizes;
  chunked_copy_batcher batcher(chunk_bytes, [&](std::span<const copy_op> ops) {
    submit_chunk(ops, stream, dsts, srcs, sizes);
  });
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    collect_column_d2h_ops(
      view.column(i), columns[static_cast<std::size_t>(i)], *allocation, batcher);
  }
  batcher.flush_pending();
  stream.synchronize();

  auto host_alloc = cucascade::memory::host_table_allocation::create(
    std::move(allocation), std::move(columns), total_size);

  return std::make_unique<cucascade::host_data_representation>(
    std::move(host_alloc), const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

}  // namespace

void register_chunked_spill_converters(cucascade::representation_converter_registry& registry,
                                       std::size_t chunk_bytes)
{
  if (chunk_bytes == 0) {
    SIRIUS_LOG_INFO(
      "[spill] chunked spill converter disabled (copy_chunk_bytes = 0); keeping the builtin "
      "monolithic GPU->HOST converter");
    return;
  }
  // register_converter throws on duplicate keys, so drop the existing registration first. This
  // also makes the call idempotent across context re-initializations.
  registry.unregister_converter<cucascade::gpu_table_representation,
                                cucascade::host_data_representation>();
  registry
    .register_converter<cucascade::gpu_table_representation, cucascade::host_data_representation>(
      [chunk_bytes](cucascade::idata_representation& source,
                    const cucascade::memory::memory_space* target_memory_space,
                    rmm::cuda_stream_view stream,
                    cucascade::memory::reservation* reservation) {
        return convert_gpu_to_host_chunked(
          source, target_memory_space, stream, reservation, chunk_bytes);
      });
  SIRIUS_LOG_INFO("[spill] registered chunked GPU->HOST spill converter (copy_chunk_bytes = {})",
                  chunk_bytes);
}

}  // namespace spill
}  // namespace sirius
