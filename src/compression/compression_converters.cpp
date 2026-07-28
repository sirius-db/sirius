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
#include "device_compressed_blob.hpp"
#include "plan_register.hpp"
#include "simpatico_bridge.hpp"
#include "spill_context.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <explore/compression_explorer.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
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
// The parallel decompress overload allocates on cache-leased internal streams
// (kept alive process-wide, so this is no longer needed for safety); re-pointing
// the free stream onto the pipeline stream keeps buffer teardown ordered with the
// rest of the pipeline's work on `s`, which helps the async pool recycle memory.
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

  // Parallel per-column decode when >1 (capped at the column count); the
  // parallel overload owns and syncs its own stream pool, so the result is
  // resident before it is wrapped against `stream` below.
  const int n_threads =
    std::min(decompress_column_threads(), static_cast<int>(subset.columns.size()));
  std::unique_ptr<cudf::table> decompressed;
  if (n_threads > 1) {
    // The reconstruct above fetched every compressed leaf buffer to device on
    // `stream`. The parallel decode runs each column on its own pool stream, so
    // those reads are NOT ordered after the fetch on `stream`. Without a barrier
    // a worker's D2H read of a codec frame header (e.g. nvcomp's num_chunks)
    // races the still-in-flight H2D fetch and reads a garbage size — which then
    // sizes a std::vector and throws length_error/bad_alloc. Order the fetch
    // before the pool-stream reads. (The serial path below already runs on
    // `stream`, so it is stream-ordered and needs no barrier.)
    stream.synchronize();
    decompressed =
      simpatico::decompress(subset, n_threads, rmm::mr::get_current_device_resource_ref());
  } else {
    decompressed =
      simpatico::decompress(subset, stream, rmm::mr::get_current_device_resource_ref());
  }

  if (n_threads > 1) {
    // Re-point the parallel result's buffers off the internal cache streams onto
    // `stream` so teardown is ordered with the rest of the pipeline's work.
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

  return reconstruct_and_decompress_to_gpu(
    rep.header(), fetch, rep.selected_indices(), source, target_memory_space, stream);
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
  const int n_threads = std::min(decompress_column_threads(),
                                 static_cast<int>(indices ? indices->size() : ct.num_columns()));
  std::unique_ptr<cudf::table> decompressed;
  if (n_threads > 1) {
    decompressed = indices.has_value() ? simpatico::decompress(ct, *indices, n_threads, mr)
                                       : simpatico::decompress(ct, n_threads, mr);
    auto cols    = decompressed->release();
    for (auto& c : cols)
      c = rebind_column_stream(std::move(c), stream);
    decompressed = std::make_unique<cudf::table>(std::move(cols));
  } else {
    decompressed = indices.has_value() ? simpatico::decompress(ct, *indices, stream, mr)
                                       : simpatico::decompress(ct, stream, mr);
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

// ── Spill-path (compress) helpers ────────────────────────────────────────────

// Placeholder column names for a spilled batch. A pipeline batch carries no
// schema names (cudf tables have none), and the spill path always restores the
// whole batch, so names are never used for projection here — they only need to
// match the column count that compressed_*_representation records.
std::vector<std::string> synthetic_column_names(int n)
{
  std::vector<std::string> names;
  names.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    names.push_back("col_" + std::to_string(i));
  }
  return names;
}

/// Plan that stores a column as-is. Used for a column that has proved not worth
/// compressing: the compressed table must still carry every column for the batch
/// to round-trip, but there is no point running a codec that does not shrink it.
/// Safe for every dtype — identity on STRING decomposes via str_split and
/// round-trips through both the in-memory and the file path.
constexpr auto kPassthroughDsl = "input -> identity\n";

using column_state = compression::plan_register::column_plan_state;

// Resolve this edge's per-column plans, running Simpatico's beam-search explorer
// once (on the first batch to spill from this edge) and caching the result in the
// plan register for every later batch.
std::vector<column_state> resolve_or_explore_spill_plan(cudf::table_view table,
                                                        const compression::spill_context& ctx,
                                                        rmm::cuda_stream_view stream)
{
  using verdict = compression::plan_register::spill_plan_verdict;

  auto& reg           = compression::plan_register::global();
  const auto decision = reg.decide_spill_plan(ctx.repo, ctx.replan_after_uses);

  // A cached entry only applies if it still describes this schema; a differing
  // column count means the plans were explored for other data, so explore again.
  if (decision.verdict == verdict::use &&
      decision.columns.size() == static_cast<std::size_t>(table.num_columns())) {
    return decision.columns;
  }
  if (decision.verdict == verdict::skip) {
    // No column here compresses. convertible_data_batch normally checks this
    // before entering the converter, so reaching here means the entry changed
    // underneath us; bail out cheaply to the uncompressed path.
    throw std::runtime_error("[compression_converters] no column worth compressing");
  }

  nvtx3::scoped_range nvtx_range{"sirius::compression::explore_spill_plan"};

  simpatico::exploration_config ecfg;
  ecfg.beam_width        = ctx.explore_beam_width;
  ecfg.max_explore_bytes = ctx.explore_max_bytes;

  // The explorer already works one column at a time, so keep its results per
  // column rather than flattening them into a single "---"-joined plan.
  std::vector<std::string> per_column;
  per_column.reserve(static_cast<std::size_t>(table.num_columns()));
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto result = simpatico::explore_column_compression(
      table.column(i), ecfg, stream, rmm::mr::get_current_device_resource_ref());
    per_column.push_back(std::move(result.plan_dsl));
  }

  SIRIUS_LOG_DEBUG("[compression_converters] explored spill plans for repo={} cols={}",
                   static_cast<const void*>(ctx.repo),
                   table.num_columns());

  reg.set_spill_plan(ctx.repo, per_column);

  std::vector<column_state> states;
  states.reserve(per_column.size());
  for (auto& dsl : per_column) {
    states.push_back(column_state{std::move(dsl), /*viable=*/true, /*consecutive_errors=*/0});
  }
  return states;
}

// The spill context must be installed by convertible_data_batch::convert().
const compression::spill_context& require_spill_context()
{
  const auto* ctx = compression::current_spill_context();
  if (ctx == nullptr || ctx->repo == nullptr) {
    throw std::runtime_error("[compression_converters] no spill context installed");
  }
  return *ctx;
}

// Compress `table` with this edge's plan and build the .hpln header + payload
// buffer list. Throws when the compressed form does not save enough (the caller
// then falls back to an uncompressed spill).
struct staged_compression {
  simpatico::compressed_table table;
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
};

staged_compression compress_for_spill(cudf::table_view view,
                                      const compression::spill_context& ctx,
                                      std::size_t uncompressed_bytes,
                                      rmm::cuda_stream_view stream)
{
  using outcome_kind      = compression::plan_register::spill_attempt_outcome;
  const auto column_plans = resolve_or_explore_spill_plan(view, ctx, stream);
  const auto num_columns  = static_cast<std::size_t>(view.num_columns());
  auto const mr           = rmm::mr::get_current_device_resource_ref();

  // Report per-column outcomes exactly once, on every exit path. Everything
  // defaults to `failed`, so an exception anywhere below is reported as an error
  // rather than as a verdict on the data: the register absorbs a few of those
  // before writing a column off, since compression runs under memory pressure and
  // a throw is as likely to be a transient allocation failure as a real signal.
  struct outcome_guard {
    const cucascade::shared_data_repository* repo;
    std::uint64_t base_interval;
    std::uint32_t error_tolerance;
    std::vector<outcome_kind> per_column;
    ~outcome_guard()
    {
      compression::plan_register::global().conclude_spill_attempt(
        repo, per_column, base_interval, error_tolerance);
    }
  } outcome{ctx.repo,
            ctx.replan_after_uses,
            ctx.error_tolerance,
            std::vector<outcome_kind>(num_columns, outcome_kind::failed)};

  // Compress column by column so each can use its own plan. A column already
  // judged not worth compressing is stored raw instead: the compressed table must
  // still carry it for the batch to round-trip, but running a codec that does not
  // shrink it would be pure cost.
  staged_compression out;
  out.table.columns.reserve(num_columns);
  auto names = synthetic_column_names(view.num_columns());
  for (std::size_t i = 0; i < num_columns; ++i) {
    const auto col      = view.column(static_cast<cudf::size_type>(i));
    const bool compress = column_plans[i].viable;

    std::string err;
    auto tree = simpatico::compress_column(
      col, compress ? column_plans[i].dsl : kPassthroughDsl, stream, mr, &err);
    if (!tree) {
      throw std::runtime_error("[compression_converters] compress_column " + std::to_string(i) +
                               ": " + (err.empty() ? "failed" : err));
    }

    simpatico::compressed_column out_col;
    out_col.dtype     = col.type();
    out_col.num_rows  = view.num_rows();
    out_col.name      = names[i];
    out_col.plan_tree = std::move(tree);
    out.table.columns.push_back(std::move(out_col));
  }

  const std::string hdr_err = simpatico::build_compressed_table_header(
    out.table, out.header, out.buffers, out.payload_bytes, stream);
  if (!hdr_err.empty()) {
    throw std::runtime_error("[compression_converters] build_compressed_table_header: " + hdr_err);
  }

  // Judge each column on its own bytes. Compressibility is a per-column property,
  // so a single incompressible column must not disqualify its neighbours — nor
  // keep costing a compress attempt on every later batch.
  const auto descs = out.table.describe(stream);
  for (std::size_t i = 0; i < num_columns && i < descs.size(); ++i) {
    std::uint64_t col_compressed = 0;
    for (auto const& leaf : descs[i]) {
      for (auto const& buf : leaf.buffers) {
        col_compressed += buf.size_bytes;
      }
    }
    const auto col_original =
      simpatico::column_size_bytes_ex(view.column(static_cast<cudf::size_type>(i)), stream);

    const bool worth_it =
      col_original == 0 || static_cast<double>(col_compressed) <=
                             ctx.max_compressed_fraction * static_cast<double>(col_original);
    // A measurement, not an error: real evidence about this column's data, so it
    // applies immediately. A column stored raw this time is measured too, and
    // simply stays not-worth-it until the edge is re-explored.
    outcome.per_column[i] = worth_it ? outcome_kind::compressed : outcome_kind::not_worth_it;
  }

  // Final whole-batch check. With non-paying columns stored raw the total is
  // normally at or below the original, but the first batch from an edge compresses
  // every column speculatively and can come out worse; decline it rather than
  // store a compressed form that costs more to keep and to read back. The
  // per-column verdicts above are still recorded, so the next batch stores the
  // columns that did not pay raw — or skips the edge entirely if none of them did.
  const std::size_t compressed_bytes = out.header.size() + out.payload_bytes;
  if (uncompressed_bytes > 0 &&
      static_cast<double>(compressed_bytes) >
        ctx.max_compressed_fraction * static_cast<double>(uncompressed_bytes)) {
    SIRIUS_LOG_DEBUG(
      "[compression_converters] repo={} compressed {}B of {}B: below threshold; "
      "spilling uncompressed",
      static_cast<const void*>(ctx.repo),
      compressed_bytes,
      uncompressed_bytes);
    throw std::runtime_error("[compression_converters] compressed " +
                             std::to_string(compressed_bytes) + "B of " +
                             std::to_string(uncompressed_bytes) + "B original: below threshold");
  }

  return out;
}

// Resolve the memory space a converter should place its result in.
const cucascade::memory::memory_space* resolve_target_space(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  cucascade::memory::reservation* reservation)
{
  if (target_memory_space != nullptr) { return target_memory_space; }
  if (reservation != nullptr) { return &reservation->get_memory_space(); }
  return &source.get_memory_space();
}

// Write a .hpln file: the structural header followed by the payload bytes.
// This is exactly the on-disk layout build_compressed_table_header produces, so
// a pinned blob (header + payload) can be flushed verbatim with no re-compression.
void write_hpln_file(
  const std::string& path,
  std::span<const std::uint8_t> header,
  const cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& payload,
  std::uint64_t payload_bytes)
{
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) { throw std::runtime_error("[compression_converters] cannot open for write: " + path); }
  out.write(reinterpret_cast<const char*>(header.data()),
            static_cast<std::streamsize>(header.size()));

  const std::size_t bs  = payload.block_size();
  std::uint64_t written = 0;
  while (written < payload_bytes) {
    const std::size_t idx = static_cast<std::size_t>(written / bs);
    const std::size_t off = static_cast<std::size_t>(written % bs);
    const std::size_t chunk =
      static_cast<std::size_t>(std::min<std::uint64_t>(payload_bytes - written, bs - off));
    out.write(reinterpret_cast<const char*>(payload.at(idx).data() + off),
              static_cast<std::streamsize>(chunk));
    written += chunk;
  }
  out.close();
  if (!out) { throw std::runtime_error("[compression_converters] write failed: " + path); }
}

// gpu_table_representation → compressed_host_representation (compress on spill).
std::unique_ptr<cucascade::idata_representation> compress_gpu_to_host(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::gpu_to_host_compress"};
  const auto& ctx = require_spill_context();
  auto& rep       = source.cast<cucascade::gpu_table_representation>();
  auto view       = rep.get_table_view();

  const std::size_t uncompressed_bytes = source.get_size_in_bytes();
  auto staged                          = compress_for_spill(view, ctx, uncompressed_bytes, stream);

  const auto* space = resolve_target_space(source, target_memory_space, reservation);
  auto* space_mut   = const_cast<cucascade::memory::memory_space*>(space);
  auto* host_mr     = space_mut->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error(
      "[compression_converters] spill target has no fixed_size_host_memory_resource");
  }

  // Stage the compressed bytes into pinned host blocks. The reservation passed in
  // was sized for the uncompressed batch, so it comfortably covers the (smaller)
  // compressed payload.
  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->header        = std::move(staged.header);
  blob->payload       = host_mr->allocate_multiple_blocks(staged.payload_bytes, reservation);
  blob->payload_bytes = staged.payload_bytes;
  for (auto const& b : staged.buffers) {
    if (b.size_bytes > 0 && b.device_ptr != nullptr) {
      copy_device_to_pinned_blocks(
        b.device_ptr, *blob->payload, b.offset, static_cast<std::size_t>(b.size_bytes), stream);
    }
  }
  // `staged.table` owns the device buffers being read above; sync before it dies.
  stream.synchronize();

  SIRIUS_LOG_DEBUG("[compression_converters] spilled {}B → {}B compressed host (cols={} rows={})",
                   uncompressed_bytes,
                   staged.payload_bytes,
                   view.num_columns(),
                   view.num_rows());

  return std::make_unique<compressed_host_representation>(
    *space_mut,
    std::move(blob),
    synthetic_column_names(view.num_columns()),
    static_cast<std::size_t>(staged.payload_bytes),
    uncompressed_bytes,
    static_cast<std::int64_t>(view.num_rows()));
}

// gpu_table_representation → compressed_disk_representation (compress on spill).
std::unique_ptr<cucascade::idata_representation> compress_gpu_to_disk(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::gpu_to_disk_compress"};
  const auto& ctx = require_spill_context();
  auto& rep       = source.cast<cucascade::gpu_table_representation>();
  auto view       = rep.get_table_view();

  const std::size_t uncompressed_bytes = source.get_size_in_bytes();
  auto staged                          = compress_for_spill(view, ctx, uncompressed_bytes, stream);

  const auto* space = resolve_target_space(source, target_memory_space, reservation);
  auto* space_mut   = const_cast<cucascade::memory::memory_space*>(space);

  const std::string path =
    compression::make_compressed_temp_path(std::string(space_mut->get_disk_mount_path()));
  const std::string err = simpatico::write_compressed_table(staged.table, path, stream);
  if (!err.empty()) {
    throw std::runtime_error("[compression_converters] write_compressed_table: " + err);
  }

  std::error_code ec;
  const auto file_size = std::filesystem::file_size(path, ec);
  const std::size_t compressed_bytes =
    ec ? static_cast<std::size_t>(staged.header.size() + staged.payload_bytes)
       : static_cast<std::size_t>(file_size);

  SIRIUS_LOG_DEBUG("[compression_converters] spilled {}B → {}B compressed disk {}",
                   uncompressed_bytes,
                   compressed_bytes,
                   path);

  return std::make_unique<compressed_disk_representation>(
    *space_mut,
    path,
    compressed_bytes,
    uncompressed_bytes,
    static_cast<std::int64_t>(view.num_rows()),
    synthetic_column_names(view.num_columns()));
}

// compressed_host_representation → compressed_disk_representation (spill cascade).
// The pinned blob is already in .hpln layout, so this is a straight file flush —
// no decompress/re-compress round trip.
std::unique_ptr<cucascade::idata_representation> flush_host_to_disk(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  [[maybe_unused]] rmm::cuda_stream_view stream,
  cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::host_to_disk_flush"};
  auto& rep = source.cast<compressed_host_representation>();

  const auto* space = resolve_target_space(source, target_memory_space, reservation);
  auto* space_mut   = const_cast<cucascade::memory::memory_space*>(space);

  const std::string path =
    compression::make_compressed_temp_path(std::string(space_mut->get_disk_mount_path()));
  write_hpln_file(path, rep.header(), rep.payload(), rep.payload_bytes());

  const std::size_t compressed_bytes = rep.header().size() + rep.payload_bytes();

  SIRIUS_LOG_DEBUG(
    "[compression_converters] flushed compressed host chunk → {} ({}B)", path, compressed_bytes);

  return std::make_unique<compressed_disk_representation>(*space_mut,
                                                          path,
                                                          compressed_bytes,
                                                          rep.get_uncompressed_data_size_in_bytes(),
                                                          rep.num_rows(),
                                                          rep.column_names());
}

// compressed_disk_representation → GPU (decompress on restore).
std::unique_ptr<cucascade::idata_representation> decompress_disk_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::disk_to_gpu"};
  auto& rep     = source.cast<compressed_disk_representation>();
  auto const mr = rmm::mr::get_current_device_resource_ref();

  std::string read_error;
  simpatico::compressed_table ct =
    simpatico::read_compressed_table(rep.path(), stream, mr, &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[compression_converters] read_compressed_table failed: " +
                             read_error);
  }

  auto const& indices = rep.selected_indices();
  const int n_threads = std::min(decompress_column_threads(),
                                 static_cast<int>(indices ? indices->size() : ct.num_columns()));
  std::unique_ptr<cudf::table> decompressed;
  if (n_threads > 1) {
    // The read above filled device buffers on `stream`; the parallel decode runs
    // on its own pool streams, which are not ordered after it. Barrier first
    // (mirrors reconstruct_and_decompress_to_gpu).
    stream.synchronize();
    decompressed = indices.has_value() ? simpatico::decompress(ct, *indices, n_threads, mr)
                                       : simpatico::decompress(ct, n_threads, mr);
    auto cols    = decompressed->release();
    for (auto& c : cols) {
      c = rebind_column_stream(std::move(c), stream);
    }
    decompressed = std::make_unique<cudf::table>(std::move(cols));
  } else {
    decompressed = indices.has_value() ? simpatico::decompress(ct, *indices, stream, mr)
                                       : simpatico::decompress(ct, stream, mr);
  }

  const cucascade::memory::memory_space* space =
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();

  SIRIUS_LOG_DEBUG("[compression_converters] decompressed from disk cols={} rows={} → GPU {}",
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

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
  if (!registry
         .has_converter<compressed_disk_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_disk_representation, cucascade::gpu_table_representation>(
        decompress_disk_to_gpu);
  }

  // Spill (compress) paths. These require a spill_context installed by
  // convertible_data_batch::convert(); without one they throw and the caller
  // falls back to an uncompressed spill.
  if (!registry
         .has_converter<cucascade::gpu_table_representation, compressed_host_representation>()) {
    registry
      .register_converter<cucascade::gpu_table_representation, compressed_host_representation>(
        compress_gpu_to_host);
  }
  if (!registry
         .has_converter<cucascade::gpu_table_representation, compressed_disk_representation>()) {
    registry
      .register_converter<cucascade::gpu_table_representation, compressed_disk_representation>(
        compress_gpu_to_disk);
  }
  if (!registry.has_converter<compressed_host_representation, compressed_disk_representation>()) {
    registry.register_converter<compressed_host_representation, compressed_disk_representation>(
      flush_host_to_disk);
  }
}

}  // namespace sirius
