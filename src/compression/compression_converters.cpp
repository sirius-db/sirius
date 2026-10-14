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

#include "compressed_disk_representation.hpp"
#include "compressed_representation.hpp"
#include "compressed_scan.hpp"
#include "device_compressed_blob.hpp"
#include "spill_compression.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/util/stream_pool.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>
#include <op/scan/decoded_batch_representation.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius {

namespace {

// Rebind a column's buffers (recursively) to `s` for ordered teardown.
// The decode's stream pool is long-lived (thread-local), but the caller's
// pipeline stream `s` is what orders the rest of the work downstream —
// re-pointing frees here ensures deallocation is not racing concurrent pipeline
// operations on `s`.
std::unique_ptr<cudf::column> rebind_column_stream(std::unique_ptr<cudf::column> col,
                                                   ::cuda::stream_ref s)
{
  if (!col) { return col; }
  const auto type = col->type();
  const auto size = col->size();
  const auto nc   = col->null_count();
  auto contents   = col->release();
  if (contents.data) { contents.data->set_stream(s); }
  auto null_mask = contents.null_mask ? std::move(*contents.null_mask)
                                      : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED);
  null_mask.set_stream(s);
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(contents.children.size());
  for (auto& ch : contents.children) {
    children.push_back(rebind_column_stream(std::move(ch), s));
  }
  return std::make_unique<cudf::column>(
    type, size, std::move(*contents.data), std::move(null_mask), nc, std::move(children));
}

// Wrap a decode's table as a GPU representation, re-pointing its buffers onto
// `stream` so pipeline teardown is ordered. What the decode did is a value on the
// representation when there is anything to report; the plain type is used
// otherwise, so an unfiltered decode is byte-identical to what it always was.
std::unique_ptr<cucascade::idata_representation> to_gpu_representation(
  decompress_result decoded,
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream)
{
  auto cols = decoded.table->release();
  for (auto& c : cols)
    c = rebind_column_stream(std::move(c), stream);
  auto decompressed = std::make_unique<cudf::table>(std::move(cols));

  auto* space = const_cast<cucascade::memory::memory_space*>(
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space());
  SIRIUS_LOG_DEBUG("[compression_converters] decompressed cols={} rows={} → GPU device={}",
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

  if (decoded.outcome.any()) {
    return std::make_unique<decompression_pushdown_batch_representation>(
      std::move(decompressed), *space, stream, decoded.outcome);
  }
  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *space, stream);
}

// Reconstruct + project + decompress a compressed_table into a GPU table
// representation. Shared by the host and device compression converters — only
// the byte transport (how `fetch` pulls the payload) differs between them.
std::unique_ptr<cucascade::idata_representation> reconstruct_and_decompress_to_gpu(
  std::span<const std::uint8_t> header,
  simpatico::payload_fetch_fn const& fetch,
  const std::optional<std::vector<std::size_t>>& selected_indices,
  decompression_pushdown_scan const* scan,
  decode_visibility_mask const& keep_mask,
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream)
{
  // Reconstruct only the requested columns. read_compressed_table_subset_from_memory
  // fetches just those columns' payload buffers, so serving a projection of a wide
  // pin does not pull every column's compressed bytes onto the GPU — that over-fetch
  // both wasted device memory and drove concurrent decode workers into the memory
  // adaptor's over-reservation path.
  std::string read_error;
  simpatico::compressed_table subset =
    selected_indices.has_value()
      ? simpatico::read_compressed_table_subset_from_memory(
          header,
          fetch,
          *selected_indices,
          stream,
          rmm::mr::get_current_device_resource_ref(),
          &read_error)
      : simpatico::read_compressed_table_from_memory(
          header, fetch, stream, rmm::mr::get_current_device_resource_ref(), &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[compression_converters] reconstruct failed: " + read_error);
  }

  // Decode across 4 pool streams, submitted from the calling thread — no worker
  // threads are spawned. The H2D fetch above ran on `stream`; sync it first so
  // pool-stream reads are ordered after all fetched bytes are resident.
  stream.sync();
  auto const mr = rmm::mr::get_current_device_resource_ref();
  // `subset` already holds only the projected columns, so the scan's request —
  // which is indexed by projected position — lines up with 0..num_columns.
  std::vector<std::size_t> selection(subset.num_columns());
  std::iota(selection.begin(), selection.end(), std::size_t{0});
  return to_gpu_representation(decompress_chunk(subset, selection, scan, keep_mask, stream, mr),
                               source,
                               target_memory_space,
                               stream);
}

// compressed_host_representation (pinned host) → GPU.
std::unique_ptr<cucascade::idata_representation> decompress_host_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::host_to_gpu"};
  auto& rep = source.cast<compressed_host_representation>();

  // Pull each compressed leaf buffer straight from the pinned host payload into
  // device memory (block-aware, since the payload is a multi-block allocation).
  auto const& payload = rep.payload();
  simpatico::payload_fetch_fn fetch =
    [&payload](std::uint64_t off, std::size_t sz, void* dst, ::cuda::stream_ref s) {
      copy_pinned_blocks_to_device(payload, off, dst, sz, s);
    };

  return reconstruct_and_decompress_to_gpu(rep.header(),
                                           fetch,
                                           rep.selected_indices(),
                                           rep.pushdown_scan().get(),
                                           rep.visibility_mask(),
                                           source,
                                           target_memory_space,
                                           stream);
}

// compressed_device_representation (device memory) → GPU.
// The compressed_table is already cached on device; decompress directly with no
// re-fetch. When a column projection is set, only the selected columns are decoded.
std::unique_ptr<cucascade::idata_representation> decompress_device_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::device_to_gpu"};
  auto& rep           = source.cast<compressed_device_representation>();
  auto const& indices = rep.selected_indices();
  auto const& ct      = rep.table();
  auto const mr       = rmm::mr::get_current_device_resource_ref();

  // Projected column count — what the scan's request is indexed by.
  auto const n_selected =
    indices.has_value() ? indices->size() : static_cast<std::size_t>(ct.num_columns());
  std::vector<std::size_t> identity_selection;
  std::span<const std::size_t> selected;
  if (indices.has_value()) {
    selected = *indices;
  } else {
    identity_selection.resize(n_selected);
    std::iota(identity_selection.begin(), identity_selection.end(), std::size_t{0});
    selected = identity_selection;
  }

  return to_gpu_representation(
    decompress_chunk(ct, selected, rep.pushdown_scan().get(), rep.visibility_mask(), stream, mr),
    source,
    target_memory_space,
    stream);
}

// compressed_disk_representation (.hpln file) → GPU. Spilled batches carry no
// projection, pushdown or visibility mask, so every column decodes in full.
std::unique_ptr<cucascade::idata_representation> decompress_disk_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::disk_to_gpu"};
  auto& rep     = source.cast<compressed_disk_representation>();
  auto const mr = rmm::mr::get_current_device_resource_ref();

  std::string read_error;
  auto const ct = simpatico::read_compressed_table(rep.path(), stream, mr, &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[compression_converters] reading '" + rep.path() +
                             "' failed: " + read_error);
  }
  // The file was read onto `stream`; the decode's pool streams are not ordered after it.
  stream.sync();

  std::vector<std::size_t> selected(ct.num_columns());
  std::iota(selected.begin(), selected.end(), std::size_t{0});
  return to_gpu_representation(
    decompress_chunk(ct, selected, nullptr, decode_visibility_mask{}, stream, mr),
    source,
    target_memory_space,
    stream);
}

// Whether a decode failure is the device running out of memory or streams, which
// passes once other work frees them, rather than a defect in the data.
bool is_transient_device_failure(std::exception const& e)
{
  if (dynamic_cast<std::bad_alloc const*>(&e) != nullptr) { return true; }
  std::string_view const what = e.what();
  return what.find("out of memory") != std::string_view::npos ||
         what.find("cudaErrorMemoryAllocation") != std::string_view::npos;
}

// Decoding a compressed batch onto the GPU allocates its full decoded size, so under
// memory pressure it can fail where the executor would succeed by waiting. Report
// such failures as rmm::out_of_memory, which the executor reschedules, instead of
// failing the query. Simpatico surfaces some of them as plan errors carrying the
// CUDA message, hence the message match.
template <auto Decode>
std::unique_ptr<cucascade::idata_representation> retryable_decode(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  cucascade::memory::reservation* reservation)
{
  try {
    return Decode(source, target_memory_space, stream, reservation);
  } catch (rmm::out_of_memory const&) {
    throw;
  } catch (std::exception const& e) {
    if (!is_transient_device_failure(e)) { throw; }
    // Clear the error so it does not resurface at the next unrelated runtime call.
    (void)cudaGetLastError();
    throw rmm::out_of_memory(std::string("[compression_converters] decode: ") + e.what());
  }
}

}  // namespace

void register_compression_converters(cucascade::representation_converter_registry& registry)
{
  // Decompression paths used by prepare_for_processing / convert_to.
  if (!registry
         .has_converter<compressed_host_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_host_representation, cucascade::gpu_table_representation>(
        retryable_decode<decompress_host_to_gpu>);
  }
  if (!registry
         .has_converter<compressed_device_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_device_representation, cucascade::gpu_table_representation>(
        decompress_device_to_gpu);
  }
  if (!registry
         .has_converter<compressed_disk_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_disk_representation, cucascade::gpu_table_representation>(
        retryable_decode<decompress_disk_to_gpu>);
  }

  // Compression paths the downgrade executor uses when spilling.
  register_spill_compression_converters(registry);
}

}  // namespace sirius
