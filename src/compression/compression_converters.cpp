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
#include <cudf/utilities/traits.hpp>

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
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius {

namespace {
std::atomic<int> g_compression_column_threads{1};
}  // namespace

void set_compression_column_threads(int n) noexcept
{
  g_compression_column_threads.store(n, std::memory_order_relaxed);
}
int compression_column_threads() noexcept
{
  return g_compression_column_threads.load(std::memory_order_relaxed);
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
    std::min(compression_column_threads(), static_cast<int>(subset.columns.size()));
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
  auto const mr       = rmm::mr::get_current_device_resource_ref();
  // Reconstructs here for a blob staged by the output/downgrade tiers, which defer it;
  // for a pinned chunk the table is already built and this is a plain lookup.
  auto const& ct      = rep.table(stream, mr);
  const int n_threads = std::min(compression_column_threads(),
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

/// Default plan for a column with no offline plan to seed from.
///
/// A fixed choice costs nothing to make, so an edge can start compressing on its
/// first spill instead of paying for a beam search — which the SF100 sweep showed
/// is a fixed per-edge cost that ruins short queries (q22: 0.48s -> 8.70s, fixed
/// to 1.00x by defaulting instead of exploring).
///
/// bitpack rather than an entropy coder: bitcomp was tried here and compressed
/// well but ran far too slowly to be a default. On q21 — the one query that
/// spills heavily — it cost 8.6x, because the explored plans it replaced were
/// cheap bitpack/delta cascades. A default is applied to every un-seeded column
/// on every spilling edge, so its *speed* matters more than its ratio; the
/// explorer refines it later, once the edge has spilled enough to amortize a
/// search.
///
/// delta -> bitpack was also measured and is *worse* here (1.07x vs 0.95x
/// overall, and q21 0.98x vs 0.92x): the extra pass costs more than the width it
/// recovers, because partitioning has already narrowed TPC-H's key columns before
/// they spill.
///
/// Non-fixed-width columns (STRING, nested) are stored raw. A str_split cascade
/// was tried for STRING and is not obviously worth the risk as a blind default:
/// its cost profile on real data is unmeasured, and getting it wrong is expensive
/// on exactly the heavily-spilling queries this is meant to help.
std::string default_plan_for(cudf::data_type type)
{
  if (!cudf::is_fixed_width(type)) { return kPassthroughDsl; }
  return "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
}

// Compress every column of `view`, each with its own plan, and return the
// compressed_table. Above one thread this fans the columns across a stream pool
// exactly as the decompress converters do — one column per worker stream — so a
// batch's columns encode concurrently instead of queueing behind each other's
// per-plan-node stream syncs (simpatico's `compress_column` is single-stream by
// construction and syncs inside every variable-output codec, so cross-column
// overlap is the only overlap available).
//
// Like the parallel decode path, the pool streams do not observe `stream`, so the
// caller's work that produced `view` has to be ordered before the workers read it.
simpatico::compressed_table compress_columns_with_plans(cudf::table_view view,
                                                        std::vector<std::string> plans,
                                                        std::vector<std::string> names,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  const auto num_columns = static_cast<std::size_t>(view.num_columns());
  const int n_threads    = std::min(compression_column_threads(), static_cast<int>(num_columns));
  if (n_threads > 1) {
    stream.synchronize();
    return simpatico::compress_columns(view, plans, n_threads, mr, std::move(names));
  }

  simpatico::compressed_table out;
  out.columns.reserve(num_columns);
  for (std::size_t i = 0; i < num_columns; ++i) {
    const auto col = view.column(static_cast<cudf::size_type>(i));
    std::string err;
    auto tree = simpatico::compress_column(col, plans[i], stream, mr, &err);
    if (!tree) {
      throw std::runtime_error("[compression_converters] compress_column " + std::to_string(i) +
                               ": " + (err.empty() ? "failed" : err));
    }
    simpatico::compressed_column out_col;
    out_col.dtype     = col.type();
    out_col.num_rows  = view.num_rows();
    out_col.name      = std::move(names[i]);
    out_col.plan_tree = std::move(tree);
    out.columns.push_back(std::move(out_col));
  }
  return out;
}

using column_state = compression::plan_register::column_plan_state;

// Resolve this edge's per-column plans, running Simpatico's beam-search explorer
// once (on the first batch to spill from this edge) and caching the result in the
// plan register for every later batch.
std::vector<column_state> resolve_or_explore_spill_plan(cudf::table_view view,
                                                        const compression::spill_context& ctx,
                                                        rmm::cuda_stream_view stream)
{
  using verdict = compression::plan_register::spill_plan_verdict;

  auto& reg           = compression::plan_register::global();
  const auto decision = reg.decide_spill_plan(ctx.repo, ctx.replan_after_uses);
  const auto& table   = view;

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

  const auto num_cols = static_cast<std::size_t>(table.num_columns());

  // Prefer the plans already explored offline for these columns' base tables.
  // Exploring in-query costs a beam search on the downgrade thread — ~81% of query
  // time in the SF100 benchmark — so a plan that is merely good and free beats one
  // that is optimal and expensive. Columns with no lineage (aggregate results,
  // computed expressions) or whose table has no plan loaded are stored raw until
  // the edge's next scheduled replan.
  // First contact with an edge never explores. Seed each column from its base
  // table's offline plan where lineage reaches, and give the rest a general-
  // purpose default. Exploration is deferred to the edge's first expiry, so the
  // beam search is only paid for once an edge has proven it spills enough to
  // amortize it. Only an entry that has expired (or one that never got installed)
  // reaches the explorer below.
  if (!reg.resolve_spill_plan(ctx.repo).has_value()) {
    nvtx3::scoped_range seed_range{"sirius::compression::seed_spill_plan"};
    auto seeds = reg.seed_plans_from_lineage(ctx.repo, num_cols);

    std::vector<compression::plan_register::column_plan_candidate> initial;
    initial.reserve(num_cols);
    std::size_t hits = 0;
    for (std::size_t i = 0; i < num_cols; ++i) {
      if (seeds && (*seeds)[i].has_value()) {
        ++hits;
        initial.push_back({std::move(*(*seeds)[i]), /*ratio=*/1.0, /*c=*/0.0, /*d=*/0.0});
      } else {
        initial.push_back(
          {default_plan_for(view.column(static_cast<cudf::size_type>(i)).type()), 1.0, 0.0, 0.0});
      }
    }
    SIRIUS_LOG_DEBUG(
      "[compression_converters] repo={} initial plans: {}/{} seeded from table plans, "
      "rest defaulted",
      static_cast<const void*>(ctx.repo),
      hits,
      num_cols);

    reg.set_spill_plan(ctx.repo, std::move(initial), ctx.replan_change_threshold);
    const auto settled = reg.decide_spill_plan(ctx.repo, /*replan_after_uses=*/0);
    if (settled.verdict != verdict::skip) { return settled.columns; }
  }

  nvtx3::scoped_range nvtx_range{"sirius::compression::explore_spill_plan"};

  simpatico::exploration_config ecfg;
  ecfg.beam_width        = ctx.explore_beam_width;
  ecfg.max_explore_bytes = ctx.explore_max_bytes;
  // Explore a row prefix rather than the whole column. The beam search allocates
  // for hundreds of trial encodes, and the spill path runs it exactly when the
  // GPU is out of memory — on full columns it mostly throws bad_alloc. Sampling
  // cuts both the allocation and the search time by orders of magnitude.
  // Caveat from the explorer's own docs: a prefix misleads on sorted/monotonic
  // columns, whose best cascade exploits global structure. Set to 0 to disable.
  ecfg.sample_rows = ctx.explore_sample_rows;

  // The explorer already works one column at a time, so keep its results per
  // column rather than flattening them into a single "---"-joined plan. Its
  // measurements come along too: the register uses them to tell a genuinely
  // better plan from one that merely reads differently.
  std::vector<compression::plan_register::column_plan_candidate> candidates;
  candidates.reserve(static_cast<std::size_t>(table.num_columns()));
  try {
    for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
      auto result = simpatico::explore_column_compression(
        table.column(i), ecfg, stream, rmm::mr::get_current_device_resource_ref());
      candidates.push_back({std::move(result.plan_dsl),
                            result.compression_ratio,
                            result.compress_throughput_gbps,
                            result.decompress_throughput_gbps});
    }
  } catch (...) {
    // Record the failure against the edge before it propagates. Exploration
    // fails before any per-column state exists, so the outcome_guard in
    // compress_for_spill has not been constructed and conclude_spill_attempt
    // would find nothing to record — leaving every later spill to repeat this
    // beam search.
    reg.note_spill_explore_failure(ctx.repo, ctx.error_tolerance);
    SIRIUS_LOG_DEBUG("[compression_converters] repo={} exploration failed",
                     static_cast<const void*>(ctx.repo));
    throw;
  }

  SIRIUS_LOG_DEBUG("[compression_converters] explored spill plans for repo={} cols={}",
                   static_cast<const void*>(ctx.repo),
                   table.num_columns());

  // The register may keep a cached plan over an equivalent candidate, so read
  // back what it actually settled on rather than assuming the candidates won.
  reg.set_spill_plan(ctx.repo, std::move(candidates), ctx.replan_change_threshold);
  const auto settled = reg.decide_spill_plan(ctx.repo, /*replan_after_uses=*/0);
  if (settled.verdict == verdict::skip) {
    // Every column kept a cached "not worth it" verdict, because nothing the
    // explorer found this time performs materially differently.
    throw std::runtime_error("[compression_converters] no column worth compressing");
  }
  return settled.columns;
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
  std::vector<std::string> dsls;
  dsls.reserve(num_columns);
  for (std::size_t i = 0; i < num_columns; ++i) {
    dsls.push_back(column_plans[i].viable ? column_plans[i].dsl : kPassthroughDsl);
  }
  out.table = compress_columns_with_plans(
    view, std::move(dsls), synthetic_column_names(view.num_columns()), stream, mr);

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

// gpu_table_representation → compressed_device_representation (eager task-output
// compression). Distinct from the spill converters in what it is for: the batch
// stays on the GPU and stays usable, it is just held in a smaller form until a
// consumer materializes it. There is no explorer here — a column is compressed
// only where lineage already offers a plan whose measured characteristics clear
// the configured gate.
std::unique_ptr<cucascade::idata_representation> compress_gpu_to_device(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::gpu_to_device_compress"};

  const auto* ctx = compression::current_output_compression_context();
  if (ctx == nullptr || ctx->repo == nullptr) {
    throw std::runtime_error("[compression_converters] no output compression context installed");
  }

  auto& rep     = source.cast<cucascade::gpu_table_representation>();
  auto view     = rep.get_table_view();
  auto& reg     = compression::plan_register::global();
  auto const mr = rmm::mr::get_current_device_resource_ref();

  const auto num_columns               = static_cast<std::size_t>(view.num_columns());
  const std::size_t uncompressed_bytes = source.get_size_in_bytes();

  auto plans =
    reg.decide_output_plan(ctx->repo, num_columns, compression::output_compression_gate());
  if (!plans.has_value() || plans->size() != num_columns) {
    throw std::runtime_error("[compression_converters] no output column worth compressing");
  }

  // Compress column by column: a qualifying column uses its lineage plan, and
  // every other column is carried as cheaply as its dtype allows.
  //
  // A non-qualifying column cannot be carried by plain identity: the blob
  // reconstruct allocates every leaf with cudf::make_numeric_column
  // (compressed_table_io.cpp:281), which rejects DECIMAL and TIMESTAMP — so
  // `input -> identity` on those faults with "Invalid, non-numeric type".
  // default_plan_for is the spill path's answer to exactly that: bitpack for
  // fixed-width types (whose leaves are integral) and passthrough only for
  // STRING/nested, whose str_split leaves are INT32/INT8. Bitpack is cheap
  // enough that the spill sweep measured it at ~0.95x against no compression.
  std::vector<std::string> dsls;
  dsls.reserve(num_columns);
  std::vector<std::uint64_t> original_bytes(num_columns, 0);
  for (std::size_t i = 0; i < num_columns; ++i) {
    const auto col = view.column(static_cast<cudf::size_type>(i));
    dsls.push_back((*plans)[i].has_value() ? *(*plans)[i] : default_plan_for(col.type()));
    original_bytes[i] = simpatico::column_size_bytes_ex(col, stream);
  }
  simpatico::compressed_table ct = compress_columns_with_plans(
    view, std::move(dsls), synthetic_column_names(view.num_columns()), stream, mr);

  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  const std::string hdr_err =
    simpatico::build_compressed_table_header(ct, header, buffers, payload_bytes, stream);
  if (!hdr_err.empty()) {
    throw std::runtime_error("[compression_converters] output header build: " + hdr_err);
  }

  // Report what each plan actually achieved on *this* data before deciding
  // whether to keep the batch. A delta cascade admitted on a base-table ratio
  // measured over sorted rows is checked here against the shuffled reality of an
  // operator output, and dropped for later batches if it did not deliver.
  std::uint64_t selected_original   = 0;
  std::uint64_t selected_compressed = 0;
  {
    const auto descs = ct.describe(stream);
    std::vector<double> achieved(num_columns, 0.0);
    for (std::size_t i = 0; i < num_columns && i < descs.size(); ++i) {
      if (!(*plans)[i].has_value()) { continue; }  // stored raw: nothing to judge
      std::uint64_t compressed = 0;
      for (auto const& leaf : descs[i]) {
        for (auto const& buf : leaf.buffers) {
          compressed += buf.size_bytes;
        }
      }
      selected_original += original_bytes[i];
      selected_compressed += compressed;
      if (compressed > 0 && original_bytes[i] > 0) {
        achieved[i] = static_cast<double>(original_bytes[i]) / static_cast<double>(compressed);
      }
    }
    reg.conclude_output_attempt(ctx->repo, achieved, compression::output_compression_gate());
  }

  const std::size_t compressed_bytes = header.size() + payload_bytes;

  // Judge `max_compressed_fraction` on the columns we actually compressed, NOT on
  // the whole batch.
  //
  // Unlike the spill path — where every column is compressed, so the batch total
  // *is* the result — this path stores every non-qualifying column raw by design.
  // Including those untouched bytes measures a decision we did not make: a batch
  // that is 90% raw columns can never clear a 0.75 whole-batch bar however well
  // the selected columns did, so the check would decline essentially always.
  // (Measured on q3/SF100: 99.0 MB of 123.8 MB — 0.80 — declined, while the one
  // compressed column had done its job.)
  if (selected_original > 0 &&
      static_cast<double>(selected_compressed) >
        ctx->max_compressed_fraction * static_cast<double>(selected_original)) {
    SIRIUS_LOG_DEBUG(
      "[compression_converters] repo={} output: selected columns compressed {}B of {}B; "
      "below threshold, publishing uncompressed",
      static_cast<const void*>(ctx->repo),
      selected_compressed,
      selected_original);
    throw std::runtime_error("[compression_converters] output selected columns compressed " +
                             std::to_string(selected_compressed) + "B of " +
                             std::to_string(selected_original) + "B: below threshold");
  }

  // Whole-batch backstop: keeping a form that is no smaller than the original
  // costs memory and a decode for nothing. Deliberately a bare "did it shrink"
  // test rather than a fraction — the fraction is spent above, on the part of the
  // batch this path is responsible for.
  if (uncompressed_bytes > 0 && compressed_bytes >= uncompressed_bytes) {
    throw std::runtime_error("[compression_converters] output compressed " +
                             std::to_string(compressed_bytes) + "B of " +
                             std::to_string(uncompressed_bytes) + "B original: no saving");
  }

  const auto* space = resolve_target_space(source, target_memory_space, reservation);
  auto* space_mut   = const_cast<cucascade::memory::memory_space*>(space);

  // Release each column's compressed tree as soon as its buffers are staged.
  //
  // Without this the peak is U + 2C: the uncompressed source (held by convert_to
  // until this converter returns, so we cannot drop it), every column's
  // compressed output in `ct`, AND the contiguous payload copy of the same
  // bytes. Measured on q3/SF100 that is 124 + 46 + 46 = 216 MB resident to end
  // up holding 46 MB — and this runs during a downgrade, i.e. when the device
  // has nothing spare, which is where `launch_encode_fused_tree failed` comes
  // from. Dropping each tree once its bytes are in the payload takes the peak to
  // U + C + (largest single column), roughly a third off.
  //
  // Buffers are enumerated in column order by build_compressed_table_header, so
  // a running count per column maps buffer index -> owning column. If the counts
  // do not add up to the buffer list, the mapping is not what we assume and the
  // release is skipped entirely rather than freeing the wrong tree.
  std::vector<std::size_t> last_buffer_of_column;
  {
    const auto descs = ct.describe(stream);
    std::size_t seen = 0;
    for (auto const& col_leaves : descs) {
      for (auto const& leaf : col_leaves) {
        seen += leaf.buffers.size();
      }
      last_buffer_of_column.push_back(seen == 0 ? 0 : seen - 1);
    }
    if (seen != buffers.size() || descs.size() != num_columns) { last_buffer_of_column.clear(); }
  }

  buffer_copied_fn release_staged;
  if (!last_buffer_of_column.empty()) {
    release_staged = [&ct, &last_buffer_of_column](std::size_t buffer_index) {
      for (std::size_t c = 0; c < last_buffer_of_column.size(); ++c) {
        if (last_buffer_of_column[c] == buffer_index) {
          ct.columns[c].plan_tree.reset();
          break;
        }
      }
    };
  }

  // Everything that can legitimately decline has already happened: a failed
  // encode and a below-threshold result both throw ABOVE this line, with the
  // uncompressed source still intact, so they stay clean fall-backs. From here
  // the compressed table is complete and the only work left is staging it.
  //
  // So release the uncompressed source now. It is dead weight — the compress
  // loop was its last reader — and holding it is what made the peak U + 2C
  // (q3/SF100: 124 + 46 + 46 = 216 MB to end up holding 46 MB). Releasing frees
  // 124 MB immediately before asking for 46 MB, so the allocation below is made
  // against *more* free memory than we started with, not less. Peak becomes 2C.
  //
  // convert_to installs the new representation only on success, so a throw after
  // this point leaves the batch holding an emptied source — hence the retry.
  auto consumed_source = rep.release_table(stream);

  // The one remaining way this can fail is the payload allocation, and it fails
  // before a single byte has been copied — so every attempt starts from the same
  // intact `ct` and retrying is always sound. (Until the blob was made to
  // reconstruct lazily there was a second failure point, the re-read's decode
  // scratch, which threw *after* the copy loop had begun freeing per-column trees
  // and so could not be retried at all. That case is gone, along with the flag and
  // the terminal "source already released" branch that tracked it.)
  //
  // Bounded, not unbounded: on the downgrade path this runs ON the downgrade
  // thread, so spinning here would stall the very thread other work is waiting on
  // to free memory — the same livelock shape as a task retrying against a
  // downgrade that can never proceed. The task executor gets to spin 100 times
  // only because it *reschedules* and lets other work run; a converter holding an
  // exclusive batch lock cannot. 10 attempts over ~2.8 s is ample for transient
  // contention, given we just freed several times what we are asking for.
  //
  // The retry stays because the source is already released above: without it a
  // transient OOM on this one allocation is fatal for the batch, not a decline.
  constexpr int kMaxBlobAttempts       = 10;
  constexpr auto kBlobBackoffIncrement = std::chrono::milliseconds(50);
  std::shared_ptr<compressed_device_blob> blob;
  std::string last_error;
  for (int attempt = 1; attempt <= kMaxBlobAttempts; ++attempt) {
    try {
      // reconstruct_now=false: this batch is decompressed at most once, so the
      // re-read is deferred to that point rather than paid here, during a
      // downgrade, when the device has least to spare.
      blob = build_device_compressed_blob(header,
                                          buffers,
                                          stream,
                                          space_mut->get_default_allocator(),
                                          mr,
                                          /*reconstruct_now=*/false,
                                          release_staged);
      break;
    } catch (const std::exception& e) {
      last_error = e.what();
      if (attempt == kMaxBlobAttempts) { break; }
      SIRIUS_LOG_DEBUG(
        "[compression_converters] repo={} blob staging attempt {}/{} failed ({}); retrying",
        static_cast<const void*>(ctx->repo),
        attempt,
        kMaxBlobAttempts,
        e.what());
      std::this_thread::sleep_for(kBlobBackoffIncrement * attempt);
    }
  }
  if (!blob) {
    // The source is gone, so this is fatal for the batch rather than a decline.
    throw std::runtime_error("[compression_converters] output blob staging failed after " +
                             std::to_string(kMaxBlobAttempts) +
                             " attempts (source already released): " + last_error);
  }

  SIRIUS_LOG_DEBUG(
    "[compression_converters] repo={} compressed output {}B → {}B device (cols={} rows={})",
    static_cast<const void*>(ctx->repo),
    uncompressed_bytes,
    compressed_bytes,
    view.num_columns(),
    view.num_rows());

  return std::make_unique<compressed_device_representation>(
    *space_mut,
    std::move(blob),
    synthetic_column_names(view.num_columns()),
    compressed_bytes,
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

// compressed_device_representation → compressed_host_representation (spill an
// already-compressed batch).
//
// Without this the downgrade executor cannot evict a device-compressed batch at
// all: every converter from compressed_device_representation targeted the GPU, so
// convert() threw "No converter registered" and the batch stayed resident and
// un-evictable. Measured on the SF100 sweep: 86 failed downgrades, 74 of them in
// q9, whose spill traffic then *rose* from 6.13 GiB to 7.47 GiB as the executor
// evicted other batches instead.
//
// This is the cheapest spill in the system — cheaper than an uncompressed one.
// The bytes are already compressed and already contiguous (the device blob is
// laid out exactly as .hpln wants), so this is a straight D2H of the payload with
// no compression, no decompression, and no re-layout. Only the structural header
// is rebuilt, from the cached table, because the device representation does not
// retain one.
std::unique_ptr<cucascade::idata_representation> stage_device_to_host(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::device_to_host_stage"};
  auto& rep = source.cast<compressed_device_representation>();

  // A projection is applied at decode time, so it lives in the representation
  // rather than in the bytes. Carrying it across would need the private
  // projecting constructor; until something actually spills a projected pin,
  // decline rather than silently widen the batch back to all columns.
  if (rep.selected_indices().has_value()) {
    throw std::runtime_error(
      "[compression_converters] cannot stage a column-projected device chunk to host");
  }

  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  // Forces a deferred reconstruct: re-deriving the staged layout needs the table, and
  // the blob's stored header describes the pre-staging one. This runs on the eviction
  // path, so it asks for scratch under memory pressure — but a compressed chunk that
  // could not be staged out would be un-evictable, which is worse.
  const std::string hdr_err = simpatico::build_compressed_table_header(
    rep.table(stream, rmm::mr::get_current_device_resource_ref()),
    header,
    buffers,
    payload_bytes,
    stream);
  if (!hdr_err.empty()) {
    throw std::runtime_error("[compression_converters] device→host header build: " + hdr_err);
  }

  const auto* space = resolve_target_space(source, target_memory_space, reservation);
  auto* space_mut   = const_cast<cucascade::memory::memory_space*>(space);
  auto* host_mr     = space_mut->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error(
      "[compression_converters] spill target has no fixed_size_host_memory_resource");
  }

  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->header        = std::move(header);
  blob->payload       = host_mr->allocate_multiple_blocks(payload_bytes, reservation);
  blob->payload_bytes = payload_bytes;
  for (auto const& b : buffers) {
    if (b.size_bytes > 0 && b.device_ptr != nullptr) {
      copy_device_to_pinned_blocks(
        b.device_ptr, *blob->payload, b.offset, static_cast<std::size_t>(b.size_bytes), stream);
    }
  }
  // The device payload is owned by `source`, which convert_to only destroys after
  // this returns — but the copies must land before that happens.
  stream.synchronize();

  const std::size_t compressed_bytes = blob->header.size() + payload_bytes;
  SIRIUS_LOG_DEBUG("[compression_converters] staged compressed device chunk → host ({}B)",
                   compressed_bytes);

  return std::make_unique<compressed_host_representation>(*space_mut,
                                                          std::move(blob),
                                                          rep.column_names(),
                                                          compressed_bytes,
                                                          rep.get_uncompressed_data_size_in_bytes(),
                                                          rep.num_rows());
}

// compressed_device_representation → compressed_disk_representation.
//
// The host hop above is preferred (the downgrade executor tries HOST before
// DISK), but a full host tier must not make a compressed batch un-evictable —
// that is the same trap this whole pair of converters exists to close.
std::unique_ptr<cucascade::idata_representation> stage_device_to_disk(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  cucascade::memory::reservation* reservation)
{
  nvtx3::scoped_range nvtx_range{"sirius::compression::device_to_disk_stage"};
  auto& rep = source.cast<compressed_device_representation>();

  if (rep.selected_indices().has_value()) {
    throw std::runtime_error(
      "[compression_converters] cannot stage a column-projected device chunk to disk");
  }

  const auto* space = resolve_target_space(source, target_memory_space, reservation);
  auto* space_mut   = const_cast<cucascade::memory::memory_space*>(space);

  const std::string path =
    compression::make_compressed_temp_path(std::string(space_mut->get_disk_mount_path()));
  const std::string err = simpatico::write_compressed_table(
    rep.table(stream, rmm::mr::get_current_device_resource_ref()), path, stream);
  if (!err.empty()) {
    throw std::runtime_error("[compression_converters] device→disk write: " + err);
  }

  std::error_code ec;
  const auto file_size = std::filesystem::file_size(path, ec);
  const std::size_t compressed_bytes =
    ec ? rep.get_size_in_bytes() : static_cast<std::size_t>(file_size);

  SIRIUS_LOG_DEBUG("[compression_converters] staged compressed device chunk → disk {} ({}B)",
                   path,
                   compressed_bytes);

  return std::make_unique<compressed_disk_representation>(*space_mut,
                                                          path,
                                                          compressed_bytes,
                                                          rep.get_uncompressed_data_size_in_bytes(),
                                                          rep.num_rows(),
                                                          rep.column_names());
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
  const int n_threads = std::min(compression_column_threads(),
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

  // Eager task-output compression. Requires an output_compression_context
  // installed by the sink; without one it throws and the batch is published
  // uncompressed.
  if (!registry
         .has_converter<cucascade::gpu_table_representation, compressed_device_representation>()) {
    registry
      .register_converter<cucascade::gpu_table_representation, compressed_device_representation>(
        compress_gpu_to_device);
  }

  // Spill paths for an already-compressed device batch. These are what make a
  // device-compressed batch evictable at all; without them the downgrade
  // executor finds no converter and the batch is pinned in place for the rest of
  // the query. Neither compresses or decompresses — they re-stage bytes that are
  // already in .hpln layout.
  if (!registry.has_converter<compressed_device_representation, compressed_host_representation>()) {
    registry.register_converter<compressed_device_representation, compressed_host_representation>(
      stage_device_to_host);
  }
  if (!registry.has_converter<compressed_device_representation, compressed_disk_representation>()) {
    registry.register_converter<compressed_device_representation, compressed_disk_representation>(
      stage_device_to_disk);
  }
}

std::size_t estimated_materialization_bytes(const cucascade::idata_representation& data)
{
  const std::size_t uncompressed = data.get_uncompressed_data_size_in_bytes();

  // Every compressed representation decodes the same way — stage the compressed
  // payload on device, then build the table beside it — so all three carry the
  // compressed footprint as extra transient peak. See the header for why the two
  // coexist rather than replace one another.
  const bool compressed = dynamic_cast<const compressed_host_representation*>(&data) != nullptr ||
                          dynamic_cast<const compressed_device_representation*>(&data) != nullptr ||
                          dynamic_cast<const compressed_disk_representation*>(&data) != nullptr;

  return compressed ? uncompressed + data.get_size_in_bytes() : uncompressed;
}

}  // namespace sirius
