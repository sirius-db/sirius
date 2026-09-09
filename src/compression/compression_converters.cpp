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

// Rebind a column's buffers (recursively) to `s` for ordered teardown.
// The decode's stream pool is long-lived (thread-local), but the caller's
// pipeline stream `s` is what orders the rest of the work downstream —
// re-pointing frees here ensures deallocation is not racing concurrent pipeline
// operations on `s`.
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
  auto cols = decompressed->release();
  for (auto& c : cols)
    c = rebind_column_stream(std::move(c), stream);
  decompressed = std::make_unique<cudf::table>(std::move(cols));

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

/// Serve a compacted payload out of the original one.
///
/// The reader asks for byte ranges of the payload the synthesized header describes; @p gather says
/// where each of those bytes lives in the ORIGINAL payload. Ranges are ascending and
/// non-overlapping in destination order, so answering a request is a walk from the first range
/// that reaches into it.
///
/// Destination bytes no gather range covers are real and deliberate: a bitpack `packed` buffer
/// declares decode guard words past its last live word (simpatico::kBitpackDecodeGuardWords), and
/// the decode loads them unconditionally without their values reaching an output row. They are
/// zeroed rather than left undefined so a decode never reads uninitialized device memory.
class gathered_payload_fetch {
 public:
  gathered_payload_fetch(
    cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation const& payload,
    std::vector<simpatico::gather_range> gather)
    : _payload(payload), _gather(std::move(gather))
  {
  }

  void operator()(std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) const
  {
    if (sz == 0) { return; }
    auto* out                 = static_cast<std::byte*>(dst);
    std::uint64_t const end   = off + sz;
    std::uint64_t filled_upto = off;
    // First range whose destination end is past `off`; the rest follow in order.
    auto it = std::ranges::lower_bound(
      _gather, off, {}, [](simpatico::gather_range const& g) { return g.dst_offset + g.size; });
    for (; it != _gather.end() && it->dst_offset < end; ++it) {
      std::uint64_t const seg_begin = std::max(it->dst_offset, off);
      std::uint64_t const seg_end   = std::min(it->dst_offset + it->size, end);
      if (seg_begin >= seg_end) { continue; }
      if (seg_begin > filled_upto) {
        CUCASCADE_CUDA_TRY(
          cudaMemsetAsync(out + (filled_upto - off), 0, seg_begin - filled_upto, s.value()));
      }
      copy_pinned_blocks_to_device(_payload,
                                   it->src_offset + (seg_begin - it->dst_offset),
                                   out + (seg_begin - off),
                                   seg_end - seg_begin,
                                   s);
      filled_upto = seg_end;
    }
    if (filled_upto < end) {
      CUCASCADE_CUDA_TRY(
        cudaMemsetAsync(out + (filled_upto - off), 0, end - filled_upto, s.value()));
    }
  }

 private:
  cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation const& _payload;
  std::vector<simpatico::gather_range> _gather;
};

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
    std::vector<std::uint8_t> subset_header;
    simpatico::payload_host_read_fn read_metadata =
      [&payload](std::uint64_t off, std::uint64_t size, void* dst) {
        copy_pinned_blocks_to_host(payload, off, dst, size);
        return true;
      };
    std::vector<simpatico::gather_range> gather;
    std::vector<std::uint8_t> subsetted;
    auto const error = simpatico::build_chunk_subset_header(rep.header(),
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
    // whole chunk down the ordinary path. Unread columns are free to be whole: the reader never
    // fetches their buffers. Which column refused is worth naming — the answer is a property of
    // the compression plan, not of the query.
    auto const refusing_column = [&]() -> std::optional<std::size_t> {
      auto const refused = [&](std::size_t col) {
        return col >= subsetted.size() || subsetted[col] == 0;
      };
      if (rep.selected_indices().has_value()) {
        auto const it = std::ranges::find_if(*rep.selected_indices(), refused);
        return it == rep.selected_indices()->end() ? std::nullopt : std::optional{*it};
      }
      for (std::size_t col = 0; col < subsetted.size(); ++col) {
        if (refused(col)) { return col; }
      }
      return subsetted.empty() ? std::optional<std::size_t>{0} : std::nullopt;
    }();
    if (error.empty() && !refusing_column.has_value()) {
      SIRIUS_LOG_DEBUG(
        "[compression_converters] chunk subset: {} decode chunks, {} gather ranges, {} B moved",
        survivors.size(),
        gather.size(),
        std::accumulate(
          gather.begin(), gather.end(), std::uint64_t{0}, [](auto acc, auto const& g) {
            return acc + g.size;
          }));
      return reconstruct_and_decompress_to_gpu(subset_header,
                                               gathered_payload_fetch{payload, std::move(gather)},
                                               rep.selected_indices(),
                                               rep.pushdown_scan().get(),
                                               source,
                                               target_memory_space,
                                               stream);
    }
    if (error.empty()) {
      auto const& names = rep.column_names();
      auto const col    = *refusing_column;
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

  auto cols = decompressed->release();
  for (auto& c : cols)
    c = rebind_column_stream(std::move(c), stream);
  decompressed = std::make_unique<cudf::table>(std::move(cols));

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
