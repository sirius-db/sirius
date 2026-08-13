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
#include "compression_device_pool.hpp"
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

#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>

#include <absl/cleanup/cleanup.h>
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/util/stream_pool.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <explore/compression_explorer.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius {

namespace {

// Thread-local pool of 4 CUDA streams for per-column encode and decode.
//
// Work must be submitted from the calling thread: cuCascade's reservation state
// is thread_local (reservation_aware_resource_adaptor), so a spawned worker
// carries no reservation and its allocations are checked against the raw pool
// capacity, throwing LIMIT_EXCEEDED however much device memory is free. On
// SF300 that cost 36-39 failed compressions per run, independent of the
// downgrade trigger.
//
// One pool serves both directions; a thread never encodes and decodes at once,
// so run_column_workers' trailing sync_all() covers exactly that call's streams.
//
// 4 is not a configuration parameter — it matches the typical SM occupancy
// sweet spot for column-parallel work.
//
// The streams are distinct but columns do not run concurrently: simpatico's
// encoders and decoders each block the submitting thread mid-column (see
// compress_columns_with_plans), so column i completes before i+1 is submitted.
simpatico::stream_pool& column_pool()
{
  thread_local simpatico::stream_pool pool;
  if (pool.streams.empty()) {
    if (!pool.init(4)) throw std::runtime_error("[compression_converters] stream_pool init failed");
  }
  return pool;
}

// Rebind a column's buffers (recursively) to `s` for ordered teardown.
// Pool streams are long-lived (thread-local), but the caller's pipeline stream
// `s` is what orders the rest of the work downstream — re-pointing frees here
// ensures deallocation is not racing concurrent pipeline operations on `s`.
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

// Translate the representation's string-only pushdown into simpatico's decode
// directives, padded to `count` so it lines up 1:1 with the columns being
// decompressed. Returns empty when nothing is pushed down, which lets callers
// stay on the plain decompress overload.
std::vector<simpatico::decode_predicate> to_decode_predicates(
  decode_equality_pushdown const& pushdown, std::size_t count)
{
  bool const any =
    std::any_of(pushdown.begin(), pushdown.end(), [](auto const& v) { return !v.empty(); });
  if (!any) { return {}; }
  if (pushdown.size() > count) {
    throw std::runtime_error(
      "[compression_converters] equality pushdown wider than the projection");
  }
  std::vector<simpatico::decode_predicate> predicates(count);
  for (std::size_t i = 0; i < pushdown.size(); ++i) {
    predicates[i].equals_any = pushdown[i];
  }
  return predicates;
}

// Reconstruct + project + decompress a compressed_table into a GPU table
// representation. Shared by the host and device compression converters — only
// the byte transport (how `fetch` pulls the payload) differs between them.
std::unique_ptr<cucascade::idata_representation> reconstruct_and_decompress_to_gpu(
  std::span<const std::uint8_t> header,
  simpatico::payload_fetch_fn const& fetch,
  const std::optional<std::vector<std::size_t>>& selected_indices,
  decode_equality_pushdown const& equality_pushdown,
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
  auto& pool    = decode_pool();
  auto const mr = rmm::mr::get_current_device_resource_ref();
  // `subset` already holds only the projected columns, so the pushdown — which
  // is indexed by projected position — lines up with 0..num_columns.
  auto const predicates =
    to_decode_predicates(equality_pushdown, static_cast<std::size_t>(subset.num_columns()));
  std::unique_ptr<cudf::table> decompressed;
  if (predicates.empty()) {
    decompressed = simpatico::decompress(subset, pool, mr);
  } else {
    std::vector<std::size_t> all(subset.num_columns());
    std::iota(all.begin(), all.end(), std::size_t{0});
    decompressed = simpatico::decompress(subset, all, predicates, pool, mr);
  }
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

  // Per-column form: each entry is a complete 1-column .hpln, decoded on its own
  // and concatenated back into the batch. Projections pick entries directly
  // instead of going through the subset reader, since a column IS an artifact here.
  if (!rep.column_blobs().empty()) {
    auto const& blobs = rep.column_blobs();
    auto const& sel   = rep.selected_indices();
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(sel ? sel->size() : blobs.size());

    auto decode_one = [&](std::size_t idx) {
      auto const& b = *blobs.at(idx);
      simpatico::payload_fetch_fn fetch =
        [&b](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view st) {
          copy_pinned_blocks_to_device(*b.payload, off, dst, sz, st);
        };
      std::string read_error;
      auto one = simpatico::read_compressed_table_from_memory(
        b.header, fetch, stream, rmm::mr::get_current_device_resource_ref(), &read_error);
      if (!read_error.empty()) {
        throw std::runtime_error("[compression_converters] per-column reconstruct failed: " +
                                 read_error);
      }
      stream.synchronize();
      auto& pool = column_pool();
      auto tbl   = simpatico::decompress(one, pool, rmm::mr::get_current_device_resource_ref());
      auto parts = tbl->release();
      if (parts.empty()) {
        throw std::runtime_error("[compression_converters] per-column decode produced no column");
      }
      cols.push_back(rebind_column_stream(std::move(parts.front()), stream));
    };

    if (sel) {
      for (auto idx : *sel) {
        decode_one(idx);
      }
    } else {
      for (std::size_t i = 0; i < blobs.size(); ++i) {
        decode_one(i);
      }
    }

    auto table = std::make_unique<cudf::table>(std::move(cols));
    const cucascade::memory::memory_space* sp =
      (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();
    SIRIUS_LOG_DEBUG("[compression_converters] decompressed per-column cols={} rows={}",
                     table->num_columns(),
                     table->num_rows());
    return std::make_unique<cucascade::gpu_table_representation>(
      std::move(table), *const_cast<cucascade::memory::memory_space*>(sp), stream);
  }

  // Pull each compressed leaf buffer straight from the pinned host payload into
  // device memory (block-aware, since the payload is a multi-block allocation).
  auto const& payload = rep.payload();
  simpatico::payload_fetch_fn fetch =
    [&payload](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      copy_pinned_blocks_to_device(payload, off, dst, sz, s);
    };

  return reconstruct_and_decompress_to_gpu(rep.header(),
                                           fetch,
                                           rep.selected_indices(),
                                           rep.equality_pushdown(),
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
  auto const mr       = rmm::mr::get_current_device_resource_ref();
  // Reconstructs here for a blob staged by the output/downgrade tiers, which defer it;
  // for a pinned chunk the table is already built and this is a plain lookup.
  auto const& ct = rep.table(stream, mr);
  auto& pool     = decode_pool();

  // Projected column count — what the pushdown is indexed by.
  auto const n_selected =
    indices.has_value() ? indices->size() : static_cast<std::size_t>(ct.num_columns());
  auto const predicates = to_decode_predicates(rep.equality_pushdown(), n_selected);

  std::unique_ptr<cudf::table> decompressed;
  if (predicates.empty()) {
    decompressed = indices.has_value() ? simpatico::decompress(ct, *indices, pool, mr)
                                       : simpatico::decompress(ct, pool, mr);
  } else if (indices.has_value()) {
    decompressed = simpatico::decompress(ct, *indices, predicates, pool, mr);
  } else {
    std::vector<std::size_t> all(n_selected);
    std::iota(all.begin(), all.end(), std::size_t{0});
    decompressed = simpatico::decompress(ct, all, predicates, pool, mr);
  }
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

// Compress every column of `view` with its own plan, one column per pool stream.
//
// The pool streams do not observe `stream`, so work that produced `view` has to
// be ordered ahead of the encode — hence the barrier.
//
// Columns are submitted concurrently but do not execute concurrently: every
// encoder blocks the submitting thread mid-column to read a data-dependent
// output size back to the host (bitpack copies its live-word shards DtoH and
// syncs; nvcomp, ALP, dictionary and str_split each sync inside their
// variable-output paths). Overlapping columns requires deferring those readbacks
// to a single barrier per batch.
simpatico::compressed_table compress_columns_with_plans(cudf::table_view view,
                                                        std::vector<std::string> plans,
                                                        std::vector<std::string> names,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  stream.synchronize();
  auto& pool = column_pool();
  return simpatico::compress_columns(view, plans, pool, mr, std::move(names));
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
        table.column(i), ecfg, stream, compression::compression_device_mr());
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

/// @param owned_columns When non-null, the caller has handed over ownership of
///        the source columns and each is freed as soon as it has been encoded.
///        The batch is then committed to a compressed spill: the source no longer
///        exists to fall back to, so the whole-batch decline below is skipped and
///        a column that cannot even be stored raw is fatal rather than a decline.
staged_compression compress_for_spill(
  cudf::table_view view,
  const compression::spill_context& ctx,
  std::size_t uncompressed_bytes,
  rmm::cuda_stream_view stream,
  std::vector<std::unique_ptr<cudf::column>>* owned_columns = nullptr)
{
  using outcome_kind      = compression::plan_register::spill_attempt_outcome;
  using column_result     = compression::plan_register::spill_column_result;
  const auto column_plans = resolve_or_explore_spill_plan(view, ctx, stream);
  const auto num_columns  = static_cast<std::size_t>(view.num_columns());
  // The dedicated arena when one is configured, else the query's pool.
  // Everything allocated here is transient — the compressed table is staged out
  // and dropped — so it never has to come from the pool that owns query data.
  // See compression_device_pool.hpp for why sharing that pool is self-defeating.
  auto const mr = compression::compression_device_mr();

  // Report per-column outcomes exactly once, on every exit path. Everything
  // defaults to `failed`, so an exception anywhere below is reported as an error
  // rather than as a verdict on the data: the register absorbs a few of those
  // before writing a column off, since compression runs under memory pressure and
  // a throw is as likely to be a transient allocation failure as a real signal.
  struct outcome_guard {
    const cucascade::shared_data_repository* repo;
    std::uint64_t base_interval;
    std::uint32_t error_tolerance;
    std::vector<column_result> per_column;
    ~outcome_guard()
    {
      compression::plan_register::global().conclude_spill_attempt(
        repo, per_column, base_interval, error_tolerance);
    }
  } outcome{ctx.repo,
            ctx.replan_after_uses,
            ctx.error_tolerance,
            std::vector<column_result>(num_columns, column_result{})};

  // Compress column by column so each can use its own plan. A column already
  // judged not worth compressing is stored raw instead: the compressed table must
  // still carry it for the batch to round-trip, but running a codec that does not
  // shrink it would be pure cost.
  // Compressed one column at a time rather than handing the whole table to
  // compress_columns, so that a column which cannot be encoded costs only itself.
  //
  // Encoding the table in one call makes any failure fatal for the batch: the
  // exception unwinds past every column already done and the whole batch spills
  // raw. Under a tight device that is the common case, and the columns that
  // failed are usually the widest ones — precisely where the codec scratch, not
  // the data, was the problem. Per column, a failed encode falls back to storing
  // that column raw and its neighbours keep their compression.
  //
  // simpatico already encodes columns sequentially inside compress_columns (its
  // encoders block the submitting thread), so splitting the call costs no
  // parallelism; it only changes the blast radius of a failure.
  staged_compression out;
  out.table.columns.reserve(num_columns);
  const auto names = synthetic_column_names(view.num_columns());
  /// Columns stored raw because their encode threw, not because the data proved
  /// incompressible. Kept apart so the measurement loop below reports them as
  /// errors rather than as a verdict.
  std::vector<bool> failed_columns(num_columns, false);
  // Measured up front: once a column is freed below, view.column(i) dangles, and
  // the verdict loop further down needs the original size to judge the ratio.
  std::vector<std::uint64_t> original_bytes(num_columns, 0);
  for (std::size_t i = 0; i < num_columns; ++i) {
    original_bytes[i] =
      simpatico::column_size_bytes_ex(view.column(static_cast<cudf::size_type>(i)), stream);
  }

  for (std::size_t i = 0; i < num_columns; ++i) {
    const auto col        = view.column(static_cast<cudf::size_type>(i));
    const cudf::table_view one_col{{col}};
    std::vector<std::string> one_name{names[i]};

    auto encode = [&](const std::string& dsl) {
      return compress_columns_with_plans(one_col, {dsl}, one_name, stream, mr);
    };

    const std::string& plan = column_plans[i].viable ? column_plans[i].dsl : kPassthroughDsl;
    try {
      auto encoded = encode(plan);
      if (encoded.columns.empty()) {
        throw std::runtime_error("[compression_converters] encode produced no column for index " +
                                 std::to_string(i));
      }
      out.table.columns.push_back(std::move(encoded.columns.front()));
    } catch (const std::exception& e) {
      // Retry raw. `identity` needs no codec scratch, so it survives the memory
      // pressure that sank the real plan; if even this throws the batch genuinely
      // cannot be staged and the caller falls back to an uncompressed spill.
      if (plan == kPassthroughDsl) { throw; }
      SIRIUS_LOG_DEBUG(
        "[compression_converters] repo={} column {} encode failed ({}); storing it raw",
        static_cast<const void*>(ctx.repo),
        i,
        e.what());
      auto encoded = encode(kPassthroughDsl);
      if (encoded.columns.empty()) {
        throw std::runtime_error(
          "[compression_converters] raw fallback produced no column for index " +
          std::to_string(i));
      }
      out.table.columns.push_back(std::move(encoded.columns.front()));
      // Not a verdict on the data — the plan may be fine and the device merely
      // full — so it is reported as an error, which the register absorbs up to
      // its tolerance rather than writing the column off on one bad batch.
      failed_columns[i] = true;
    }

    // The encode was this column's last reader, so drop it now rather than
    // holding the whole batch until the converter returns. This is the memory the
    // downgrade is trying to reclaim, and releasing it here is what keeps the
    // encode's device footprint to one column rather than the whole batch.
    // NOT freed here. Freeing a column the moment it is encoded is the obvious
    // way to keep the encode's footprint to one column, and it is unsound: the
    // encode can still fail on a later column (an arena OOM is routine), and by
    // then the earlier columns are gone. The batch cannot be spilled
    // uncompressed and cannot be reconstructed — measured as zero-column batches
    // that surfaced much later as an out-of-range access during materialization.
    // The caller frees the whole set once the encode has succeeded, which is the
    // first moment nothing can fail.
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
    const auto col_original = original_bytes[i];

    const bool worth_it =
      col_original == 0 || static_cast<double>(col_compressed) <=
                             ctx.max_compressed_fraction * static_cast<double>(col_original);
    // A measurement, not an error: real evidence about this column's data, so it
    // applies immediately. A column stored raw this time is measured too, and
    // simply stays not-worth-it until the edge is re-explored.
    //
    // The achieved ratio travels with the verdict. It is what the plan currently
    // in use delivered on a whole batch, and it becomes the number a later
    // re-explore has to beat — without it the comparison is against the seed's
    // placeholder and any candidate wins.
    //
    // A column that fell back to raw because its encode threw is reported as
    // `failed` instead. Its bytes say nothing about compressibility — they are
    // the identity plan's — so recording them as a measurement would write the
    // column off for a transient allocation failure.
    if (failed_columns[i]) {
      outcome.per_column[i].outcome = outcome_kind::failed;
      continue;
    }
    outcome.per_column[i].outcome = worth_it ? outcome_kind::compressed : outcome_kind::not_worth_it;
    outcome.per_column[i].achieved_ratio =
      (col_original > 0 && col_compressed > 0)
        ? static_cast<double>(col_original) / static_cast<double>(col_compressed)
        : 0.0;
  }

  // Final whole-batch check. With non-paying columns stored raw the total is
  // normally at or below the original, but the first batch from an edge compresses
  // every column speculatively and can come out worse; decline it rather than
  // store a compressed form that costs more to keep and to read back. The
  // per-column verdicts above are still recorded, so the next batch stores the
  // columns that did not pay raw — or skips the edge entirely if none of them did.
  //
  // Skipped once the source columns have been freed: there is nothing left to
  // spill uncompressed, so declining here would destroy the batch rather than
  // fall back. The cost of not declining is bounded — columns that did not pay
  // are already stored raw via the identity plan, so the compressed form is at
  // worst the raw bytes plus a header — and the per-column verdicts above still
  // steer the next batch on this edge.
  const std::size_t compressed_bytes = out.header.size() + out.payload_bytes;
  if (owned_columns == nullptr && uncompressed_bytes > 0 &&
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

/// Holds a device reservation covering an encode's working memory, attached to
/// the calling thread for as long as it lives.
///
/// Only used when there is no compression arena. With an arena the encode
/// allocates from a pool carved off the device at startup, which is outside
/// cuCascade's accounting entirely — reserving as well would double-count.
/// Without one, the encode allocates from the query's own pool, during a
/// downgrade, and does so unreserved: it can push the pool past what
/// reservations promised, which then surfaces as an OOM in some unrelated
/// operator that did everything right. Reserving makes the demand visible and
/// gives it somewhere to fail cleanly.
///
/// Sirius runs the reservation adaptor in PER_THREAD tracking scope
/// (`per_stream_reservation` defaults false), so attaching binds to the calling
/// thread — which is why every encode must be submitted from this thread, the
/// same constraint the column pool already operates under.
class scoped_encode_reservation {
 public:
  /// @return an inactive guard when no reservation was needed or one could not
  ///         be granted; check `ok()` before proceeding.
  scoped_encode_reservation(cucascade::memory::memory_space* gpu_space,
                            std::size_t bytes,
                            rmm::cuda_stream_view stream)
    : _stream(stream)
  {
    if (gpu_space == nullptr || bytes == 0) {
      _ok = true;  // nothing to reserve; not a failure
      return;
    }
    auto reservation = gpu_space->make_reservation_or_null(bytes);
    if (!reservation) { return; }  // _ok stays false: caller declines
    _allocator = gpu_space->get_memory_resource_of<cucascade::memory::Tier::GPU>();
    if (_allocator == nullptr) {
      _ok = true;  // no adaptor to attach to; behave as before
      return;
    }
    _attached = _allocator->attach_reservation_to_tracker(stream, std::move(reservation));
    _ok       = true;
  }

  ~scoped_encode_reservation()
  {
    if (_attached && _allocator != nullptr) { _allocator->reset_stream_reservation(_stream); }
  }

  scoped_encode_reservation(const scoped_encode_reservation&)            = delete;
  scoped_encode_reservation& operator=(const scoped_encode_reservation&) = delete;
  scoped_encode_reservation(scoped_encode_reservation&&)                 = delete;
  scoped_encode_reservation& operator=(scoped_encode_reservation&&)      = delete;

  [[nodiscard]] bool ok() const noexcept { return _ok; }

 private:
  rmm::cuda_stream_view _stream;
  cucascade::memory::reservation_aware_resource_adaptor* _allocator{nullptr};
  bool _attached{false};
  bool _ok{false};
};

/// Bytes to reserve for encoding a unit of @p uncompressed_bytes, or 0 when no
/// reservation is wanted (an arena is installed, or the fraction is disabled).
std::size_t encode_reservation_bytes(const compression::spill_context& ctx,
                                     std::size_t uncompressed_bytes)
{
  if (compression::compression_device_pool_enabled()) { return 0; }
  if (!(ctx.encode_reserve_fraction > 0.0)) { return 0; }
  return static_cast<std::size_t>(static_cast<double>(uncompressed_bytes) *
                                  ctx.encode_reserve_fraction);
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

// Drain an iovec batch, tolerating short writes.
//
// writev is permitted to write fewer bytes than requested, so the caller cannot
// treat one call as one batch. Advance past whole entries the kernel consumed and
// re-issue the remainder, trimming the first partially-written entry in place.
void writev_all(int fd, iovec* iov, std::size_t count, const std::string& path)
{
  while (count > 0) {
    const ssize_t n = ::writev(fd, iov, static_cast<int>(count));
    if (n < 0) {
      if (errno == EINTR) { continue; }
      throw std::runtime_error("[compression_converters] writev failed on " + path + ": " +
                               std::strerror(errno));
    }
    auto remaining = static_cast<std::size_t>(n);
    while (count > 0 && remaining >= iov->iov_len) {
      remaining -= iov->iov_len;
      ++iov;
      --count;
    }
    if (count > 0 && remaining > 0) {
      iov->iov_base = static_cast<std::byte*>(iov->iov_base) + remaining;
      iov->iov_len -= remaining;
    }
  }
}

// Write a .hpln file: the structural header followed by the payload bytes.
// This is exactly the on-disk layout build_compressed_table_header produces, so
// a pinned blob (header + payload) can be flushed verbatim with no re-compression.
//
// The payload is a list of fixed-size blocks from the pinned host pool, not one
// contiguous region, so it is gathered with writev rather than a write per block:
// at a 1 MiB block size a multi-GB batch would otherwise cost thousands of
// syscalls. Batches are capped at IOV_MAX entries, which writev rejects beyond.
void write_hpln_file(
  const std::string& path,
  std::span<const std::uint8_t> header,
  const cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation& payload,
  std::uint64_t payload_bytes)
{
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    throw std::runtime_error("[compression_converters] cannot open for write: " + path + ": " +
                             std::strerror(errno));
  }
  absl::Cleanup close_fd = [fd] { ::close(fd); };

  static constexpr std::size_t kMaxIov = 1024;  // conservative floor for IOV_MAX
  std::vector<iovec> iov;
  iov.reserve(kMaxIov);

  auto flush = [&] {
    if (!iov.empty()) {
      writev_all(fd, iov.data(), iov.size(), path);
      iov.clear();
    }
  };
  auto push = [&](const void* base, std::size_t len) {
    if (len == 0) { return; }
    iov.push_back(iovec{const_cast<void*>(base), len});
    if (iov.size() == kMaxIov) { flush(); }
  };

  push(header.data(), header.size());

  const std::size_t bs  = payload.block_size();
  std::uint64_t written = 0;
  while (written < payload_bytes) {
    const std::size_t idx = static_cast<std::size_t>(written / bs);
    const std::size_t off = static_cast<std::size_t>(written % bs);
    const std::size_t chunk =
      static_cast<std::size_t>(std::min<std::uint64_t>(payload_bytes - written, bs - off));
    push(payload.at(idx).data() + off, chunk);
    written += chunk;
  }
  flush();
}


// Encode one column, stage it to pinned host, and free both its compressed form
// and its source before moving to the next.
//
// The ordering is the point. A whole-table spill only becomes durable once every
// column has been encoded and staged, so freeing sources during that loop leaves
// a window in which a later failure loses the columns already freed — their
// compressed forms live only in device memory and go with the exception. Staging
// each column as it is produced closes the window: by the time column i's source
// is freed, column i is already on the host, and a failure at column i+1 costs
// only that column, whose source is still intact.
//
// It also bounds the arena to one column instead of accumulating every column's
// compressed output, which is what forced the arena to 4 GB.
// @param column_plans  Resolved by the caller, before it took ownership of the
//        columns: resolution explores on first contact with an edge and the
//        explorer allocates, so under the pressure that triggered this spill it
//        can fail — and after the release a failure is no longer a decline.
std::vector<std::shared_ptr<pinned_compressed_blob>> encode_and_stage_per_column(
  cudf::table_view view,
  const compression::spill_context& ctx,
  const std::vector<column_state>& column_plans,
  std::vector<std::unique_ptr<cudf::column>>& owned_columns,
  cucascade::memory::fixed_size_host_memory_resource* host_mr,
  cucascade::memory::reservation* reservation,
  rmm::cuda_stream_view stream,
  std::uint64_t& out_compressed_bytes)
{
  const auto num_columns   = static_cast<std::size_t>(view.num_columns());
  auto const mr            = compression::compression_device_mr();
  const auto names         = synthetic_column_names(view.num_columns());
  out_compressed_bytes     = 0;

  std::vector<std::shared_ptr<pinned_compressed_blob>> blobs;
  blobs.reserve(num_columns);

  for (std::size_t i = 0; i < num_columns; ++i) {
    const cudf::table_view one_col{{view.column(static_cast<cudf::size_type>(i))}};
    std::vector<std::string> one_name{names[i]};

    auto encode = [&](const std::string& dsl) {
      return compress_columns_with_plans(one_col, {dsl}, one_name, stream, mr);
    };

    simpatico::compressed_table encoded;
    const std::string& plan = column_plans[i].viable ? column_plans[i].dsl : kPassthroughDsl;
    try {
      encoded = encode(plan);
    } catch (const std::exception& e) {
      // Raw is the floor: identity needs no codec scratch, so it survives the
      // pressure that sank the real plan. Only this column is affected.
      SIRIUS_LOG_DEBUG("[compression_converters] column {} encode failed ({}); storing it raw",
                       i,
                       e.what());
      // Raw is the floor and it has to succeed: by this point earlier columns have
      // been staged and their sources freed, so there is no intact batch to fall
      // back to and throwing would destroy it. identity needs no codec scratch,
      // and the previous column's compressed form was freed as soon as it was
      // staged, so the arena drains between columns — a failure here is transient
      // contention with the other downgrade threads, not a real shortage.
      bool staged_raw = false;
      for (int attempt = 1; attempt <= 20 && !staged_raw; ++attempt) {
        try {
          encoded    = encode(kPassthroughDsl);
          staged_raw = true;
        } catch (const std::exception& raw_err) {
          if (attempt == 20) {
            throw std::runtime_error(
              std::string("[compression_converters] column ") + std::to_string(i) +
              " could not be stored even raw after 20 attempts: " + raw_err.what());
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(25 * attempt));
        }
      }
    }

    std::vector<std::uint8_t> header;
    std::vector<simpatico::payload_buffer_ref> buffers;
    std::uint64_t payload_bytes = 0;
    const std::string hdr_err =
      simpatico::build_compressed_table_header(encoded, header, buffers, payload_bytes, stream);
    if (!hdr_err.empty()) {
      throw std::runtime_error("[compression_converters] header for column " + std::to_string(i) +
                               ": " + hdr_err);
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
    // The copies read `encoded`'s device buffers, so they must land before it dies.
    stream.synchronize();

    out_compressed_bytes += blob->header.size() + payload_bytes;
    blobs.push_back(std::move(blob));

    // Column i is now durable on the host. Its compressed form and its source are
    // both dead weight, and this is the first moment freeing either is safe.
    encoded.columns.clear();
    if (i < owned_columns.size()) { owned_columns[i].reset(); }
  }

  return blobs;
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

  // Reserve the encode's device working memory before anything else happens.
  // Ahead of the plan resolve and the source release, both because a decline
  // must still be a clean decline at this point and because a reservation is
  // cheap to fail and expensive to discover late.
  auto* gpu_space = const_cast<cucascade::memory::memory_space*>(&source.get_memory_space());
  scoped_encode_reservation encode_reservation(
    gpu_space, encode_reservation_bytes(ctx, uncompressed_bytes), stream);
  if (!encode_reservation.ok()) {
    SIRIUS_LOG_DEBUG(
      "[compression_converters] repo={} declining: cannot reserve {}B on device for the encode; "
      "spilling uncompressed",
      static_cast<const void*>(ctx.repo),
      encode_reservation_bytes(ctx, uncompressed_bytes));
    throw std::runtime_error(
      "[compression_converters] no device reservation available for the encode");
  }

  const auto* space = resolve_target_space(source, target_memory_space, reservation);
  auto* space_mut   = const_cast<cucascade::memory::memory_space*>(space);
  auto* host_mr_early = space_mut->get_memory_resource_of<cucascade::memory::Tier::HOST>();

  // Take ownership so each column can be freed as it is encoded. try_release_table,
  // not release_table: the latter deep-copies when the batch views externally
  // owned memory, adding a whole batch to the device at the moment we are trying
  // to free one. A null return means this batch is that kind, and it takes the
  // ordinary whole-table path below.
  //
  // Everything that can decline runs BEFORE this point, because the release is
  // the point of no return: the columns move out of `rep`, so from here on a
  // failure cannot fall back to an uncompressed spill — there is nothing left to
  // spill. Resolving the plan is such a step. It explores on first contact with
  // an edge, and the explorer allocates from exactly the memory that is scarce
  // during a spill; on q3/SF1000 with a 2 GB arena it OOM'd, the generic handler
  // declined to an uncompressed spill of the already-emptied representation, and
  // the resulting zero-column batch surfaced 20 s later as an out-of-range access
  // inside PARTITION.
  std::unique_ptr<cudf::table> owned;
  std::vector<std::unique_ptr<cudf::column>> owned_columns;
  std::vector<cudf::column_view> owned_views;
  std::vector<column_state> column_plans;
  if (ctx.release_columns_early && host_mr_early != nullptr) {
    column_plans = resolve_or_explore_spill_plan(view, ctx, stream);
    owned        = rep.try_release_table(stream);
    if (owned) {
      owned_columns = owned->release();
      owned_views.reserve(owned_columns.size());
      for (auto const& c : owned_columns) {
        owned_views.push_back(c->view());
      }
      view = cudf::table_view{owned_views};
    }
  }

  if (owned) {
    const auto num_columns = static_cast<std::size_t>(view.num_columns());
    const auto num_rows    = static_cast<std::int64_t>(view.num_rows());
    std::uint64_t compressed_bytes = 0;
    std::vector<std::shared_ptr<pinned_compressed_blob>> blobs;
    try {
      blobs = encode_and_stage_per_column(view,
                                          ctx,
                                          column_plans,
                                          owned_columns,
                                          host_mr_early,
                                          reservation,
                                          stream,
                                          compressed_bytes);
    } catch (const std::exception& e) {
      // The source is gone, so this is not a decline. Rethrown as
      // spill_source_consumed so the caller fails the downgrade instead of
      // spilling the emptied representation as a zero-column batch, which is
      // silent corruption that only surfaces when something materializes it.
      throw spill_source_consumed(
        std::string("[compression_converters] per-column spill failed after the source was "
                    "released: ") +
        e.what());
    }

    SIRIUS_LOG_DEBUG(
      "[compression_converters] spilled {}B → {}B compressed host per-column (cols={} rows={})",
      uncompressed_bytes,
      compressed_bytes,
      num_columns,
      num_rows);

    // The aggregate blob carries no bytes of its own; the data lives in `blobs`.
    auto aggregate = std::make_shared<pinned_compressed_blob>();
    auto out       = std::make_unique<compressed_host_representation>(
      *space_mut,
      std::move(aggregate),
      synthetic_column_names(static_cast<int>(num_columns)),
      static_cast<std::size_t>(compressed_bytes),
      uncompressed_bytes,
      num_rows);
    out->set_column_blobs(std::move(blobs));
    return out;
  }

  auto* host_mr = space_mut->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error(
      "[compression_converters] spill target has no fixed_size_host_memory_resource");
  }

  // Ordinary whole-table spill: encode every column into one .hpln, then stage it.
  // The source stays intact throughout, so a failure anywhere here is a clean
  // decline that falls back to an uncompressed spill.
  auto staged = compress_for_spill(view, ctx, uncompressed_bytes, stream);

  const auto num_columns = static_cast<std::size_t>(view.num_columns());
  const auto num_rows    = static_cast<std::int64_t>(view.num_rows());

  // Stage the compressed bytes into pinned host blocks. The reservation passed in
  // was sized for the uncompressed batch, so it comfortably covers the (smaller)
  // compressed payload.
  //
  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->header        = std::move(staged.header);
  blob->payload       = host_mr->allocate_multiple_blocks(staged.payload_bytes, reservation);
  blob->payload_bytes = staged.payload_bytes;

  // NOT released here, despite the device converter doing exactly that with its
  // own source. Releasing the uncompressed table before staging (rep.release_table
  // + reset) is an obvious-looking win — it frees a whole batch during a downgrade
  // — but measured on q3/SF1000 it wedged the query: downgrade requests went from
  // 65 to 82,661 while per-request yield collapsed from 7.8 GB to 4 MB, i.e. the
  // pressure the monitor is watching stopped resolving even though conversion
  // throughput was unchanged (4.5 GB/s either way). Something about emptying the
  // representation out from under convert_to leaves the space's view of itself
  // wrong; until that is understood, convert_to is left to retire the source when
  // it installs the replacement.
  // Staged in one pass, freeing nothing until the end.
  //
  // Releasing each column's compressed tree as its last buffer lands (what the
  // device converter does) looks like a straight win on peak — it would carry the
  // largest single column instead of all of them. It is not, here: these buffers
  // were allocated on the column-pool streams while the copies run on `stream`, so
  // each early release needs a stream.synchronize() first, and with four downgrade
  // threads that cost 27% of spill throughput (4.5 -> 3.3 GB/s) and ~13 s of wall
  // clock on q3/SF1000. The memory it saves is not the memory that is scarce: the
  // compressed form is a fraction of the batch and lives in the arena, while the
  // pressure is on the query pool.
  //
  // It becomes worth revisiting if the release can be made stream-ordered (freeing
  // on `stream` rather than syncing), or once arena capacity rather than query
  // memory is the binding constraint.
  for (auto const& b : staged.buffers) {
    if (b.size_bytes > 0 && b.device_ptr != nullptr) {
      copy_device_to_pinned_blocks(
        b.device_ptr, *blob->payload, b.offset, static_cast<std::size_t>(b.size_bytes), stream);
    }
  }
  // `staged.table` owns the device buffers being read above; sync before it dies.
  // Still required: the mapping may have been unusable, or a column may hold
  // buffers past its recorded last one.
  stream.synchronize();

  SIRIUS_LOG_DEBUG("[compression_converters] spilled {}B → {}B compressed host (cols={} rows={})",
                   uncompressed_bytes,
                   staged.payload_bytes,
                   num_columns,
                   num_rows);

  return std::make_unique<compressed_host_representation>(
    *space_mut,
    std::move(blob),
    synthetic_column_names(static_cast<cudf::size_type>(num_columns)),
    static_cast<std::size_t>(staged.payload_bytes),
    uncompressed_bytes,
    num_rows);
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

  // See compress_gpu_to_host: without an arena the encode is otherwise
  // unreserved pressure on the query's own pool.
  auto* gpu_space = const_cast<cucascade::memory::memory_space*>(&source.get_memory_space());
  scoped_encode_reservation encode_reservation(
    gpu_space, encode_reservation_bytes(ctx, uncompressed_bytes), stream);
  if (!encode_reservation.ok()) {
    SIRIUS_LOG_DEBUG(
      "[compression_converters] repo={} declining: cannot reserve device memory for the encode; "
      "spilling uncompressed",
      static_cast<const void*>(ctx.repo));
    throw std::runtime_error(
      "[compression_converters] no device reservation available for the encode");
  }

  auto staged = compress_for_spill(view, ctx, uncompressed_bytes, stream);

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

  // A per-column representation keeps its bytes in column_blobs(); the aggregate
  // blob is empty by construction, so the whole-table flush below would write a
  // zero-byte .hpln and silently destroy the batch. Each artifact is a complete
  // 1-column .hpln, so the disk form is one file per column: the same shape as
  // the host form, and each file stays readable by the ordinary reader.
  //
  // One file per column rather than a single indexed container, because a
  // container would need its own reader — the .hpln readers take either a path or
  // an in-memory image, so a sub-range of a file would have to be staged on the
  // host first, re-introducing the host copy this path exists to avoid.
  if (!rep.column_blobs().empty()) {
    auto const& blobs = rep.column_blobs();
    std::vector<std::string> paths;
    paths.reserve(blobs.size());
    std::size_t total_bytes    = 0;
    std::size_t max_artifact   = 0;
    // Partial output is not a leak: on a throw the paths written so far are
    // unlinked here, since no representation owns them yet.
    absl::Cleanup remove_partial = [&paths] {
      for (const auto& p : paths) {
        std::error_code rm_ec;
        std::filesystem::remove(p, rm_ec);
      }
    };
    for (auto const& blob : blobs) {
      const std::string col_path =
        compression::make_compressed_temp_path(std::string(space_mut->get_disk_mount_path()));
      paths.push_back(col_path);
      write_hpln_file(col_path, blob->header, *blob->payload, blob->payload_bytes);
      const std::size_t artifact_bytes = blob->header.size() + blob->payload_bytes;
      total_bytes += artifact_bytes;
      max_artifact = std::max(max_artifact, artifact_bytes);
    }
    std::move(remove_partial).Cancel();

    SIRIUS_LOG_DEBUG(
      "[compression_converters] flushed per-column compressed host chunk → {} files ({}B)",
      paths.size(),
      total_bytes);

    return std::make_unique<compressed_disk_representation>(
      *space_mut,
      std::move(paths),
      total_bytes,
      max_artifact,
      rep.get_uncompressed_data_size_in_bytes(),
      rep.num_rows(),
      rep.column_names());
  }

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

  // Per-column form: one complete 1-column .hpln per column, read and decoded one
  // at a time and assembled into the batch. A projection picks files directly —
  // a column IS an artifact here, so there is nothing for the subset reader to do.
  if (rep.is_per_column()) {
    auto const& paths = rep.column_paths();
    auto const& sel   = rep.selected_indices();
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(sel ? sel->size() : paths.size());

    auto decode_one = [&](std::size_t idx) {
      if (idx >= paths.size()) {
        throw std::runtime_error("[compression_converters] per-column disk index " +
                                 std::to_string(idx) + " out of range (" +
                                 std::to_string(paths.size()) + " artifacts)");
      }
      std::string err;
      simpatico::compressed_table one =
        simpatico::read_compressed_table(paths[idx], stream, mr, &err);
      if (!err.empty()) {
        throw std::runtime_error("[compression_converters] per-column disk read failed (" +
                                 paths[idx] + "): " + err);
      }
      // The read filled device buffers on `stream`; the pool streams are not
      // ordered after it, so barrier before decoding (as the whole-table path does).
      stream.synchronize();
      auto& pool = column_pool();
      auto tbl   = simpatico::decompress(one, pool, mr);
      auto parts = tbl->release();
      if (parts.empty()) {
        throw std::runtime_error(
          "[compression_converters] per-column disk decode produced no column");
      }
      cols.push_back(rebind_column_stream(std::move(parts.front()), stream));
    };

    if (sel) {
      for (auto idx : *sel) {
        decode_one(idx);
      }
    } else {
      for (std::size_t i = 0; i < paths.size(); ++i) {
        decode_one(i);
      }
    }

    auto table = std::make_unique<cudf::table>(std::move(cols));
    const cucascade::memory::memory_space* sp =
      (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();
    SIRIUS_LOG_DEBUG("[compression_converters] decompressed per-column from disk cols={} rows={}",
                     table->num_columns(),
                     table->num_rows());
    return std::make_unique<cucascade::gpu_table_representation>(
      std::move(table), *const_cast<cucascade::memory::memory_space*>(sp), stream);
  }

  std::string read_error;
  simpatico::compressed_table ct =
    simpatico::read_compressed_table(rep.path(), stream, mr, &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[compression_converters] read_compressed_table failed: " +
                             read_error);
  }

  auto const& indices = rep.selected_indices();
  // The read above filled device buffers on `stream`; the pool streams are not
  // ordered after it, so barrier first (mirrors reconstruct_and_decompress_to_gpu).
  stream.synchronize();
  auto& pool        = column_pool();
  auto decompressed = indices.has_value() ? simpatico::decompress(ct, *indices, pool, mr)
                                          : simpatico::decompress(ct, pool, mr);
  auto cols         = decompressed->release();
  for (auto& c : cols) {
    c = rebind_column_stream(std::move(c), stream);
  }
  decompressed = std::make_unique<cudf::table>(std::move(cols));

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
  //
  // A per-column batch stages one artifact at a time and frees it before the
  // next, so its transient is the largest column, not the sum; charging the sum
  // would over-reserve by roughly the column count on exactly the batches this
  // form exists to make cheaper.
  if (const auto* host = dynamic_cast<const compressed_host_representation*>(&data)) {
    return uncompressed + host->decode_transient_bytes();
  }
  if (const auto* disk = dynamic_cast<const compressed_disk_representation*>(&data)) {
    return uncompressed + disk->decode_transient_bytes();
  }
  if (dynamic_cast<const compressed_device_representation*>(&data) != nullptr) {
    return uncompressed + data.get_size_in_bytes();
  }
  return uncompressed;
}

}  // namespace sirius
