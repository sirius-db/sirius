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

#include "data/sirius_host_to_gpu_converter.hpp"

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/host_table.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <spdlog/spdlog.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::data {

namespace {

// =============================================================================
// Mirrors cucascade's file-private reconstruct_column logic, but issues every
// H2D copy on a CALLER-PROVIDED target_stream. cucascade's version consumes
// the original caller stream (which may be bound to a different device than
// the RAII target guard), producing cudaErrorInvalidValue under num_gpus=2.
// =============================================================================

// H2D copy of `size` bytes out of the pinned-host block allocation starting
// at alloc_offset, into a freshly allocated device_buffer on target_stream.
rmm::device_buffer alloc_and_copy_h2d(cucascade::memory::fixed_multiple_blocks_allocation& alloc,
                                      std::size_t alloc_offset,
                                      std::size_t size,
                                      rmm::cuda_stream_view target_stream,
                                      rmm::mr::device_memory_resource* mr)
{
  rmm::device_buffer buf(size, target_stream, mr);
  if (size == 0) { return buf; }
  if (!alloc || alloc->size() == 0) {
    throw std::invalid_argument(
      "sirius_host_to_gpu_converter: pinned host allocation is null or empty but copy size is "
      "non-zero");
  }

  const std::size_t block_size = alloc->block_size();
  std::size_t block_idx        = alloc_offset / block_size;
  std::size_t block_off        = alloc_offset % block_size;
  std::size_t dst_off          = 0;

  while (dst_off < size) {
    const std::size_t remaining      = size - dst_off;
    const std::size_t space_in_block = block_size - block_off;
    const std::size_t bytes_to_copy  = std::min(remaining, space_in_block);

    auto block             = alloc->at(block_idx);
    cudaError_t const cerr = cudaMemcpyAsync(static_cast<uint8_t*>(buf.data()) + dst_off,
                                             block.data() + block_off,
                                             bytes_to_copy,
                                             cudaMemcpyHostToDevice,
                                             target_stream.value());
    if (cerr != cudaSuccess) {
      // Consume sticky state before reporting.
      (void)cudaGetLastError();
      throw std::runtime_error(
        std::string("sirius_host_to_gpu_converter: cudaMemcpyAsync H2D failed: ")
        + cudaGetErrorString(cerr)
        + " (FIX-02; verify target_stream is bound to the current target device)");
    }
    dst_off += bytes_to_copy;
    block_off += bytes_to_copy;
    if (block_off == block_size) {
      ++block_idx;
      block_off = 0;
    }
  }
  return buf;
}

// Recursively reconstruct a cudf::column from column_metadata + pinned host
// blocks, using target_stream for all allocations and H2D copies.
//
// Mirrors the structure of cucascade's reconstruct_column (private function in
// cucascade/src/data/representation_converter.cpp:717) using only cucascade's
// PUBLIC host_table.hpp + fixed_size_host_memory_resource.hpp API plus cudf
// factories, so we do not depend on cucascade-internal helpers. Each branch
// handles the same cudf::type_id as cucascade; DECIMAL scale is propagated;
// nested types (STRING / LIST / STRUCT / DICTIONARY32) recurse.
std::unique_ptr<cudf::column> reconstruct_column_target_stream(
  const cucascade::memory::column_metadata& meta,
  cucascade::memory::fixed_multiple_blocks_allocation& alloc,
  rmm::cuda_stream_view target_stream,
  rmm::mr::device_memory_resource* mr)
{
  // Null mask first; cudf column factories read the null mask at construction
  // time, so it must land on device before the factory is called.
  rmm::device_buffer null_mask{};
  if (meta.has_null_mask) {
    null_mask =
      alloc_and_copy_h2d(alloc, meta.null_mask_offset, meta.null_mask_size, target_stream, mr);
    // Synchronize now so the null mask is visible to the factory call below.
    target_stream.synchronize();
  }
  const cudf::size_type null_count = meta.has_null_mask ? meta.null_count : 0;

  if (meta.type_id == cudf::type_id::STRING) {
    if (meta.children.empty()) {
      throw std::invalid_argument(
        "sirius_host_to_gpu_converter: STRING column metadata must have at least one child "
        "(offsets)");
    }
    auto offsets_col =
      reconstruct_column_target_stream(meta.children[0], alloc, target_stream, mr);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      // cudf's make_strings_column requires INT64 offsets for large-string
      // support. Sync before cast so the freshly-copied INT32 offsets buffer
      // is stable on device.
      target_stream.synchronize();
      offsets_col = cudf::cast(
        offsets_col->view(), cudf::data_type{cudf::type_id::INT64}, target_stream, mr);
    }
    rmm::device_buffer chars_buf{};
    if (meta.has_data && meta.data_size > 0) {
      chars_buf = alloc_and_copy_h2d(alloc, meta.data_offset, meta.data_size, target_stream, mr);
    }
    return cudf::make_strings_column(meta.num_rows,
                                     std::move(offsets_col),
                                     std::move(chars_buf),
                                     null_count,
                                     std::move(null_mask));
  }

  if (meta.type_id == cudf::type_id::LIST) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "sirius_host_to_gpu_converter: LIST column metadata must have two children (offsets, "
        "values)");
    }
    auto offsets_col =
      reconstruct_column_target_stream(meta.children[0], alloc, target_stream, mr);
    if (offsets_col->type().id() == cudf::type_id::INT32) {
      target_stream.synchronize();
      offsets_col = cudf::cast(
        offsets_col->view(), cudf::data_type{cudf::type_id::INT64}, target_stream, mr);
    }
    auto values_col =
      reconstruct_column_target_stream(meta.children[1], alloc, target_stream, mr);
    return cudf::make_lists_column(meta.num_rows,
                                   std::move(offsets_col),
                                   std::move(values_col),
                                   null_count,
                                   std::move(null_mask));
  }

  if (meta.type_id == cudf::type_id::STRUCT) {
    std::vector<std::unique_ptr<cudf::column>> fields;
    fields.reserve(meta.children.size());
    for (const auto& child_meta : meta.children) {
      fields.push_back(reconstruct_column_target_stream(child_meta, alloc, target_stream, mr));
    }
    // Construct the STRUCT column directly (matches cucascade's approach)
    // rather than via make_structs_column — our serialized null masks are
    // already consistent, so we skip the superimpose_nulls kernel.
    return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRUCT},
                                          meta.num_rows,
                                          rmm::device_buffer{},
                                          std::move(null_mask),
                                          null_count,
                                          std::move(fields));
  }

  if (meta.type_id == cudf::type_id::DICTIONARY32) {
    if (meta.children.size() < 2) {
      throw std::invalid_argument(
        "sirius_host_to_gpu_converter: DICTIONARY32 column metadata must have two children "
        "(indices, keys)");
    }
    // cucascade encoding: children[0] = indices, children[1] = keys.
    // cudf::make_dictionary_column signature: (keys, indices, null_mask, null_count).
    auto indices_col =
      reconstruct_column_target_stream(meta.children[0], alloc, target_stream, mr);
    auto keys_col = reconstruct_column_target_stream(meta.children[1], alloc, target_stream, mr);
    return cudf::make_dictionary_column(
      std::move(keys_col), std::move(indices_col), std::move(null_mask), null_count);
  }

  // Fixed-width leaf (including DECIMAL variants — scale carried in meta).
  const cudf::data_type dtype = cudf::is_fixed_point(cudf::data_type{meta.type_id})
                                  ? cudf::data_type{meta.type_id, meta.scale}
                                  : cudf::data_type{meta.type_id};
  rmm::device_buffer data_buf{};
  if (meta.has_data && meta.data_size > 0) {
    data_buf = alloc_and_copy_h2d(alloc, meta.data_offset, meta.data_size, target_stream, mr);
  }
  return std::make_unique<cudf::column>(
    dtype, meta.num_rows, std::move(data_buf), std::move(null_mask), null_count);
}

}  // namespace

std::unique_ptr<cucascade::idata_representation> sirius_host_fast_to_gpu_factory(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& host_source    = source.cast<cucascade::host_data_representation>();
  const auto& host_tbl = host_source.get_host_table();
  if (!host_tbl) {
    throw std::runtime_error(
      "sirius_host_to_gpu_converter: host table is null (source has no allocation)");
  }
  if (!host_tbl->allocation) {
    throw std::runtime_error(
      "sirius_host_to_gpu_converter: host table allocation is null but columns present");
  }

  // Sync caller's stream so any upstream work (e.g. the code path that
  // produced this host_data_representation) is flushed before we read the
  // pinned host blocks. No-op at the driver level when there is no
  // outstanding work on the caller's stream.
  stream.synchronize();

  // --- Switch to target device and acquire a TARGET-BOUND stream. ---
  // This is the fix: cucascade's body at representation_converter.cpp:849
  // uses the caller's stream for batch.flush(), which may be bound to a
  // non-target device under num_gpus=2 (e.g. the pipeline-executor stream
  // for GPU 0 while target_device_id == 1).
  const int target_device_id = target_memory_space->get_device_id();
  rmm::cuda_set_device_raii target_guard{rmm::cuda_device_id{target_device_id}};
  auto target_stream = target_memory_space->acquire_stream();
  auto* mr           = target_memory_space->get_default_allocator();

  // --- Reconstruct each top-level column on target_stream + target device. ---
  std::vector<std::unique_ptr<cudf::column>> gpu_columns;
  gpu_columns.reserve(host_tbl->columns.size());
  for (const auto& col_meta : host_tbl->columns) {
    gpu_columns.push_back(
      reconstruct_column_target_stream(col_meta, host_tbl->allocation, target_stream, mr));
  }

  // Assemble the cudf::table on target_stream. The table ctor does not issue
  // its own copies — it just takes ownership of the column tree we built —
  // but we pass the stream anyway for consistency with cucascade's shape.
  auto new_table = std::make_unique<cudf::table>(std::move(gpu_columns));

  // Sync target_stream so the caller observes a finished GPU table.
  target_stream.synchronize();

  // Consume any sticky CUDA state before returning so a later call-site does
  // not surface a stray error against us (matches Pattern 2 hygiene).
  (void)cudaGetLastError();

  spdlog::debug(
    "sirius_host_to_gpu_converter: converted host_data_representation "
    "({} columns, {} bytes) -> gpu_table_representation on GPU {} (FIX-02)",
    host_tbl->columns.size(),
    host_tbl->data_size,
    target_device_id);

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(new_table), *const_cast<cucascade::memory::memory_space*>(target_memory_space));
}

}  // namespace sirius::data
