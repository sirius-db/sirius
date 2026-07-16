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

#include "compression_converters.hpp"

#include "compressed_representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius {

namespace {
std::atomic<int> g_decompress_column_threads{1};
}  // namespace

void set_decompress_column_threads(int n) noexcept
{
  g_decompress_column_threads.store(n, std::memory_order_relaxed);
}
int decompress_column_threads() noexcept
{
  return g_decompress_column_threads.load(std::memory_order_relaxed);
}

namespace {

// Rebind a column's buffers (recursively) to `s` for their eventual async free.
// The parallel decompress overload allocates on internal pool streams that are
// destroyed on return; its data is already synced, so re-pointing the free
// stream to a long-lived one avoids a use-after-free at buffer teardown.
std::unique_ptr<cudf::column> rebind_column_stream(std::unique_ptr<cudf::column> col,
                                                   rmm::cuda_stream_view s)
{
  if (!col) { return col; }
  const auto type = col->type();
  const auto size = col->size();
  const auto nc   = col->null_count();
  auto contents   = col->release();
  if (contents.data) { contents.data->set_stream(s); }
  rmm::device_buffer null_mask =
    contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{};
  null_mask.set_stream(s);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(contents.children.size());
  for (auto& ch : contents.children) {
    children.push_back(rebind_column_stream(std::move(ch), s));
  }
  return std::make_unique<cudf::column>(
    type, size, std::move(*contents.data), std::move(null_mask), nc, std::move(children));
}

// Reconstruct + project + decompress a compressed_table into a GPU table
// representation. Shared by the host and device compression converters — only
// the byte transport (how `fetch` pulls the payload) differs between them.
std::unique_ptr<cucascade::idata_representation> reconstruct_and_decompress_to_gpu(
  std::span<const std::uint8_t> header,
  simpatico::payload_fetch_fn const& fetch,
  const std::optional<std::vector<std::size_t>>& selected_indices,
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  std::string read_error;
  simpatico::compressed_table ct = simpatico::read_compressed_table_from_memory(
    header, fetch, stream, rmm::mr::get_current_device_resource_ref(), &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[compression_converters] read_compressed_table_from_memory failed: " +
                             read_error);
  }

  // Project to the selected columns before decompressing to avoid
  // inflating memory with unrequested columns.
  simpatico::compressed_table subset;
  if (selected_indices.has_value()) {
    const auto& indices = *selected_indices;
    subset.columns.reserve(indices.size());
    for (auto idx : indices) {
      if (idx >= ct.columns.size()) {
        throw std::out_of_range(
          "[compression_converters] selected column index out of range during decompress");
      }
      subset.columns.push_back(std::move(ct.columns[idx]));
    }
  } else {
    subset = std::move(ct);
  }

  // Parallel per-column decode when >1 (capped at the column count); the
  // parallel overload owns and syncs its own stream pool, so the result is
  // resident before it is wrapped against `stream` below.
  const int n_threads =
    std::min(decompress_column_threads(), static_cast<int>(subset.columns.size()));
  auto decompressed =
    n_threads > 1
      ? simpatico::decompress(subset, n_threads, rmm::mr::get_current_device_resource_ref())
      : simpatico::decompress(subset, stream, rmm::mr::get_current_device_resource_ref());

  if (n_threads > 1) {
    // Re-point the parallel result's buffers off the (now-destroyed) internal
    // pool streams onto `stream` before it is used/freed downstream.
    auto cols = decompressed->release();
    for (auto& c : cols) {
      c = rebind_column_stream(std::move(c), stream);
    }
    decompressed = std::make_unique<cudf::table>(std::move(cols));
  }

  const cucascade::memory::memory_space* space =
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();

  SIRIUS_LOG_DEBUG("[compression_converters] decompressed cols={} rows={} → GPU device={}",
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
}

// compressed_host_representation (pinned host) → GPU.
std::unique_ptr<cucascade::idata_representation> decompress_host_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& rep = source.cast<compressed_host_representation>();

  // Pull each compressed leaf buffer straight from the pinned host payload into
  // device memory (block-aware, since the payload is a multi-block allocation).
  auto const& payload = rep.payload();
  simpatico::payload_fetch_fn fetch =
    [&payload](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      copy_pinned_blocks_to_device(payload, off, dst, sz, s);
    };

  return reconstruct_and_decompress_to_gpu(
    rep.header(), fetch, rep.selected_indices(), source, target_memory_space, stream);
}

// compressed_device_representation (device memory) → GPU.
std::unique_ptr<cucascade::idata_representation> decompress_device_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& rep = source.cast<compressed_device_representation>();

  // The payload is already a single contiguous device buffer — each leaf buffer
  // is one device→device copy at its offset.
  const auto* payload_base = static_cast<const std::byte*>(rep.payload_device_ptr());
  simpatico::payload_fetch_fn fetch =
    [payload_base](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      CUCASCADE_CUDA_TRY(
        cudaMemcpyAsync(dst, payload_base + off, sz, cudaMemcpyDeviceToDevice, s.value()));
    };

  return reconstruct_and_decompress_to_gpu(
    rep.header(), fetch, rep.selected_indices(), source, target_memory_space, stream);
}

}  // namespace

void register_compression_converters(cucascade::representation_converter_registry& registry)
{
  // Decompression paths used by prepare_for_processing / convert_to.
  if (!registry
         .has_converter<compressed_host_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_host_representation, cucascade::gpu_table_representation>(
        decompress_host_to_gpu);
  }
  if (!registry
         .has_converter<compressed_device_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_device_representation, cucascade::gpu_table_representation>(
        decompress_device_to_gpu);
  }
}

}  // namespace sirius
