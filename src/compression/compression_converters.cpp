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
#include "compressed_scan.hpp"
#include "device_compressed_blob.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/util/stream_pool.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>
#include <op/scan/decoded_batch_representation.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius {

namespace {

// Reconstruct + project + decompress a compressed_table into a GPU table
// representation. Shared by the host and device compression converters — only
// the byte transport (how `fetch` pulls the payload) differs between them.
std::unique_ptr<cucascade::idata_representation> reconstruct_and_decompress_to_gpu(
  std::span<const std::uint8_t> header,
  simpatico::payload_fetch_fn const& fetch,
  const std::optional<std::vector<std::size_t>>& selected_indices,
  decompression_pushdown_scan const* scan,
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
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
  stream.synchronize();
  auto const mr = rmm::mr::get_current_device_resource_ref();
  // `subset` already holds only the projected columns, so the scan's request —
  // which is indexed by projected position — lines up with 0..num_columns.
  std::vector<std::size_t> selection(subset.num_columns());
  std::iota(selection.begin(), selection.end(), std::size_t{0});
  auto decoded      = decompress_chunk(subset, selection, scan, stream, mr);
  auto decompressed = std::move(decoded.table);

  // Re-point decoded buffers onto `stream` so pipeline teardown is ordered.
  decompressed = rebind_table_stream(std::move(decompressed), stream);

  const cucascade::memory::memory_space* space =
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();

  SIRIUS_LOG_DEBUG("[compression_converters] decompressed cols={} rows={} → GPU device={}",
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

  // What the decode did is a value on the representation when there is
  // anything to report; the plain type is used otherwise, so an unfiltered
  // decode is byte-identical to what it always was.
  auto const& outcome = decoded.outcome;
  if (outcome.any()) {
    return std::make_unique<decompression_pushdown_batch_representation>(
      std::move(decompressed),
      *const_cast<cucascade::memory::memory_space*>(space),
      stream,
      outcome);
  }
  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
}

// compressed_host_representation (pinned host) → GPU.
std::unique_ptr<cucascade::idata_representation> decompress_host_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::host_to_gpu"};
  auto& rep = source.cast<compressed_host_representation>();

  // Pull each compressed leaf buffer straight from the pinned host payload into
  // device memory (block-aware, since the payload is a multi-block allocation).
  auto const& payload = rep.payload();
  simpatico::payload_fetch_fn fetch =
    [&payload](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      copy_pinned_blocks_to_device(payload, off, dst, sz, s);
    };

  // Zone maps ruled some of this chunk's 1024-row decode chunks out. Synthesize a header that
  // describes only the survivors and fetch only their bytes: the pruned rows then cost neither
  // H2D transfer nor decode, which is where a host-tier pin spends its time. The reader and the
  // decode see an ordinary, smaller table and need no knowledge that a subset is in play
  // (simpatico::build_chunk_subset_header).
  //
  // Every failure here degrades to serving the chunk whole, which is correct — just less
  // selective — so nothing below throws on a refusal.
  auto const survivors = rep.surviving_chunks();
  if (!survivors.empty()) {
    // Narrow to the read columns FIRST. The row subset rewrites every record of every column in
    // the header it is given, so over a wide pin it costs with the PIN's width while saving only
    // with the QUERY's — enough to make pruning a net loss. allocate_chunk_narrowed does the same
    // two-step for the .hpln file path.
    //
    // build_column_subset_header needs strictly ascending columns and emits them in that order, so
    // the projection is remapped onto the narrowed positions: the scan's own order (outputs, then
    // filter-only columns) is not ascending, and serving it reordered would hand the consumer a
    // neighbouring column.
    std::vector<std::uint8_t> column_header;
    std::vector<simpatico::gather_range> gather_columns;
    std::optional<std::vector<std::size_t>> projection = rep.selected_indices();
    std::vector<std::size_t> ascending;
    std::span<const std::uint8_t> working_header = rep.header();
    if (projection.has_value()) {
      ascending = *projection;
      std::ranges::sort(ascending);
      ascending.erase(std::ranges::unique(ascending).begin(), ascending.end());
      if (simpatico::build_column_subset_header(
            rep.header(), ascending, column_header, gather_columns)
            .empty()) {
        std::vector<std::size_t> remapped;
        remapped.reserve(projection->size());
        for (auto const col : *projection) {
          remapped.push_back(
            static_cast<std::size_t>(std::ranges::lower_bound(ascending, col) - ascending.begin()));
        }
        projection     = std::move(remapped);
        working_header = column_header;
      } else {
        // A refusal is correct, just wider: the row subset then runs over the whole header as it
        // always did, and the reader still projects.
        gather_columns.clear();
      }
    }

    std::vector<std::uint8_t> subset_header;
    // The row subset's offsets live in the COLUMN subset's destination space, so they are mapped
    // back before touching the payload; following them unmapped reads the wrong bytes silently.
    simpatico::payload_host_read_fn read_metadata =
      [&payload, &gather_columns](std::uint64_t off, std::uint64_t size, void* dst) {
        if (size == 0) { return true; }
        if (gather_columns.empty()) {
          copy_pinned_blocks_to_host(payload, off, dst, size);
          return true;
        }
        std::vector<simpatico::gather_range> const want{{off, size, 0}};
        auto const ranges = simpatico::compose_gathers(gather_columns, want);
        if (ranges.empty()) { return false; }
        for (auto const& r : ranges) {
          copy_pinned_blocks_to_host(
            payload, r.src_offset, static_cast<std::uint8_t*>(dst) + r.dst_offset, r.size);
        }
        return true;
      };
    std::vector<simpatico::gather_range> gather;
    std::vector<std::uint8_t> subsetted;
    auto const error = simpatico::build_chunk_subset_header(working_header,
                                                            survivors,
                                                            read_metadata,
                                                            subset_header,
                                                            gather,
                                                            /*max_gap_bytes=*/0,
                                                            /*out_payload_bytes=*/nullptr,
                                                            &subsetted);
    // A column the format cannot address per chunk is emitted whole, and a whole column next to a
    // compacted one has a different row count — the two cannot be one cudf::table. So the subset
    // is usable only when every column this scan READS was compacted; one that was not sends the
    // whole chunk down the ordinary path. After the narrowing above every column of
    // `working_header` is read, so the check covers all of them; it walks the projection only when
    // the narrowing was refused. Which column refused is worth naming — the answer is a property
    // of the compression plan, not of the query.
    auto const refusing_column = [&]() -> std::optional<std::size_t> {
      auto const refused = [&](std::size_t col) {
        return col >= subsetted.size() || subsetted[col] == 0;
      };
      if (gather_columns.empty() && projection.has_value()) {
        auto const it = std::ranges::find_if(*projection, refused);
        return it == projection->end() ? std::nullopt : std::optional{*it};
      }
      for (std::size_t col = 0; col < subsetted.size(); ++col) {
        if (refused(col)) { return col; }
      }
      return subsetted.empty() ? std::optional<std::size_t>{0} : std::nullopt;
    }();
    if (error.empty() && !refusing_column.has_value()) {
      // Both narrowings compose into one set of ranges against the ORIGINAL pinned payload.
      auto composed = gather_columns.empty() ? std::move(gather)
                                             : simpatico::compose_gathers(gather_columns, gather);
      if (!composed.empty() || gather.empty()) {
        SIRIUS_LOG_DEBUG(
          "[compression_converters] chunk subset: {} decode chunks, {} of {} columns, {} gather "
          "ranges, {} B moved",
          survivors.size(),
          subsetted.size(),
          rep.column_names().size(),
          composed.size(),
          std::accumulate(
            composed.begin(), composed.end(), std::uint64_t{0}, [](auto acc, auto const& g) {
              return acc + g.size;
            }));
        return reconstruct_and_decompress_to_gpu(
          subset_header,
          gathered_payload_fetch{payload, std::move(composed)},
          projection,
          rep.pushdown_scan().get(),
          source,
          target_memory_space,
          stream);
      }
      SIRIUS_LOG_WARN(
        "[compression_converters] chunk subset ranges did not compose, serving the whole chunk");
    } else if (error.empty()) {
      auto const& names = rep.column_names();
      // Report the FILE column, which is what a compression plan is written against.
      auto const col =
        gather_columns.empty()
          ? *refusing_column
          : (*refusing_column < ascending.size() ? ascending[*refusing_column] : *refusing_column);
      SIRIUS_LOG_DEBUG(
        "[compression_converters] chunk subset not used, serving the whole chunk: column {} ({}) "
        "is not chunk-addressable",
        col,
        col < names.size() ? names[col] : "?");
    } else {
      SIRIUS_LOG_WARN("[compression_converters] chunk subset refused, serving the whole chunk: {}",
                      error);
    }
  }

  return reconstruct_and_decompress_to_gpu(rep.header(),
                                           fetch,
                                           rep.selected_indices(),
                                           rep.pushdown_scan().get(),
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
  rmm::cuda_stream_view stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::device_to_gpu"};
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

  auto decoded      = decompress_chunk(ct, selected, rep.pushdown_scan().get(), stream, mr);
  auto decompressed = std::move(decoded.table);

  decompressed = rebind_table_stream(std::move(decompressed), stream);

  const cucascade::memory::memory_space* space =
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();

  SIRIUS_LOG_DEBUG("[compression_converters] decompressed cols={} rows={} → GPU device={}",
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

  // What the decode did is a value on the representation when there is
  // anything to report; the plain type is used otherwise.
  auto const& outcome = decoded.outcome;
  if (outcome.any()) {
    return std::make_unique<decompression_pushdown_batch_representation>(
      std::move(decompressed),
      *const_cast<cucascade::memory::memory_space*>(space),
      stream,
      outcome);
  }
  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
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
