// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/cuda_check.hpp"
#include "codegen/util/nvtx.hpp"
#include "codegen/util/stream_pool.hpp"
#include "decode/decode_session.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace simpatico {

namespace {

// ── Internal helpers for the public compress/decompress API ───────────────────
// (Formerly compress_internals.hpp; this TU is the only consumer.)

class plan_error : public std::runtime_error {
 public:
  explicit plan_error(std::string const& msg) : std::runtime_error(msg) {}
};

std::string trim_plan_block(std::string s)
{
  while (!s.empty() &&
         (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
  size_t start = 0;
  while (start < s.size() && (s[start] == ' ' || s[start] == '\t'))
    ++start;
  return s.substr(start);
}

// Split a multi-column DSL string on "---" separators, skipping blank lines
// and comment lines (beginning with '#'). Each returned block is trimmed.
std::vector<std::string> split_plan_dsl_impl(std::string_view plan_dsl)
{
  std::vector<std::string> plans;
  std::string current;
  size_t i = 0;
  while (i < plan_dsl.size()) {
    size_t line_end = plan_dsl.find('\n', i);
    if (line_end == std::string_view::npos) line_end = plan_dsl.size();
    std::string_view line = plan_dsl.substr(i, line_end - i);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

    std::string_view trimmed = line;
    while (!trimmed.empty() && trimmed.front() == ' ')
      trimmed.remove_prefix(1);
    while (!trimmed.empty() && trimmed.back() == ' ')
      trimmed.remove_suffix(1);

    if (trimmed == "---") {
      auto block = trim_plan_block(current);
      if (!block.empty()) plans.push_back(std::move(block));
      current.clear();
    } else if (!trimmed.empty() && trimmed.front() != '#') {
      current.append(trimmed);
      current.push_back('\n');
    }
    i = (line_end == plan_dsl.size()) ? plan_dsl.size() : line_end + 1;
  }
  auto block = trim_plan_block(current);
  if (!block.empty()) plans.push_back(std::move(block));
  return plans;
}

void validate_plan_count(size_t plan_count, int table_columns)
{
  if (plan_count != static_cast<size_t>(table_columns)) {
    throw plan_error("plan count (" + std::to_string(plan_count) +
                     ") does not match table.num_columns() (" + std::to_string(table_columns) +
                     ")");
  }
}

void validate_column_names(std::vector<std::string> const& column_names, size_t num_columns)
{
  if (!column_names.empty() && column_names.size() != num_columns) {
    throw plan_error("column_names size (" + std::to_string(column_names.size()) +
                     ") does not match num_columns (" + std::to_string(num_columns) + ")");
  }
}

// Process-lifetime cache of CUDA streams for the internal `int column_threads`
// overloads. These overloads have no caller-owned pool, yet the objects they
// return (a cudf::table, or a compressed_table whose leaf buffers live in cudf
// columns) record the stream they were built on for their eventual async free.
// If that stream were a per-call stream_pool destroyed on return, freeing the
// result later would deallocate on a dangling stream handle — a use-after-free
// with an async memory resource. Leasing from a cache that NEVER destroys its
// streams keeps every recorded handle valid for the process lifetime, so the
// result is safe to free by any stream (including the RMM default) with no
// external rebinding. Streams are recycled between calls, so this also avoids
// per-call stream create/destroy churn.
// CUDA streams are device-bound, so recycled streams are keyed by device.
class stream_cache {
 public:
  // The caller must have `device` current when new streams are created.
  std::vector<cudaStream_t> checkout(int device, size_t n)
  {
    std::vector<cudaStream_t> out;
    out.reserve(n);
    std::lock_guard<std::mutex> lock(mu_);
    auto& free_list = free_[device];
    while (out.size() < n && !free_list.empty()) {
      out.push_back(free_list.back());
      free_list.pop_back();
    }
    while (out.size() < n) {
      cudaStream_t s{};
      if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) break;
      out.push_back(s);
    }
    return out;
  }

  // Return streams to the same device list they were checked out from.
  void check_in(int device, std::vector<cudaStream_t>& streams)
  {
    std::lock_guard<std::mutex> lock(mu_);
    auto& free_list = free_[device];
    free_list.insert(free_list.end(), streams.begin(), streams.end());
    streams.clear();
  }

 private:
  std::mutex mu_;
  std::map<int, std::vector<cudaStream_t>> free_;
};

stream_cache& global_stream_cache()
{
  static stream_cache cache;
  return cache;
}

// RAII lease of max(1, column_threads) cache streams into a stream_pool for the
// duration of an internal-parallel call. On destruction the streams are returned
// to the cache (NOT destroyed), so any buffer allocated on them stays valid for
// its eventual async free even after this pool is gone. Concurrent leases get
// disjoint streams (checkout is mutex-guarded and pops distinct handles), so each
// call's sync_all only touches its own streams.
// Capture the current device once for both checkout and check-in.
struct leased_pool {
  stream_pool pool;
  int device = 0;

  explicit leased_pool(int column_threads)
  {
    if (cudaGetDevice(&device) != cudaSuccess)
      throw plan_error("failed to query the current device for the internal stream lease");

    pool.streams =
      global_stream_cache().checkout(device, static_cast<size_t>(std::max(1, column_threads)));

    if (pool.streams.empty()) throw plan_error("failed to lease internal streams");
  }

  ~leased_pool()
  {
    // Completed compression/decode scopes already surfaced asynchronous errors.
    // A lease destructor cannot throw.
    (void)pool.sync_all();
    global_stream_cache().check_in(
      device, pool.streams);  // leaves pool.streams empty; ~stream_pool is a no-op
  }

  leased_pool(const leased_pool&)            = delete;
  leased_pool& operator=(const leased_pool&) = delete;
};

// Submit `body(i, stream)` for every index in [0, n_items) across the pool
// streams from the calling thread (round-robin), then synchronise all streams.
// No worker threads are spawned: CUDA stream submission is asynchronous, so
// the GPU can overlap column work across pool streams while the CPU submits
// serially. All allocations happen on the calling thread, keeping
// cuCascade's per-thread memory-reservation accounting correct.
template <typename Body>
void run_column_workers(size_t n_items, stream_pool& pool, Body&& body)
{
  size_t const n_streams = pool.streams.size();
  if (n_streams == 0) throw plan_error("stream_pool has no streams");
  std::exception_ptr first_exception;
  for (size_t i = 0; i < n_items; ++i) {
    ::cuda::stream_ref s{pool.streams[i % n_streams]};
    try {
      body(i, s);
    } catch (...) {
      if (!first_exception) first_exception = std::current_exception();
      break;
    }
  }
  cudaError_t sync_err = pool.sync_all();
  if (first_exception) std::rethrow_exception(first_exception);
  if (sync_err != cudaSuccess) {
    throw plan_error(std::string("column worker stream sync failed: ") +
                     cudaGetErrorString(sync_err));
  }
}

compressed_table compress_columns_parallel(cudf::table_view table,
                                           std::vector<std::string> const& plans,
                                           stream_pool& pool,
                                           rmm::device_async_resource_ref mr,
                                           std::vector<std::string> const& column_names)
{
  compressed_table out;
  out.columns.resize(plans.size());
  run_column_workers(plans.size(), pool, [&](size_t i, ::cuda::stream_ref stream) {
    std::string err;
    auto plan_tree =
      compress_column(table.column(static_cast<cudf::size_type>(i)), plans[i], stream, mr, &err);
    if (!plan_tree) throw plan_error(err.empty() ? "compress failed" : err);
    compressed_column col;
    col.dtype     = table.column(static_cast<cudf::size_type>(i)).type();
    col.num_rows  = table.num_rows();
    col.plan_tree = std::move(plan_tree);
    if (!column_names.empty()) col.name = column_names[i];
    out.columns[i] = std::move(col);
  });
  return out;
}

// Every table facade submits the same typed requests to one session. Streams and the explicit
// allocator are borrowed by the session.
std::unique_ptr<cudf::table> decode_table_columns(compressed_table const& table,
                                                  std::span<std::size_t const> selected,
                                                  std::span<decode_predicate const> predicates,
                                                  std::span<const ::cuda::stream_ref> streams,
                                                  rmm::device_async_resource_ref mr)
{
  for (auto index : selected) {
    if (index >= table.columns.size()) throw plan_error("selected column index out of range");
    if (!table.columns[index].plan_tree) throw plan_error("decompress: column missing plan tree");
  }
  decode_session session{streams, mr};
  for (std::size_t i = 0; i < selected.size(); ++i) {
    auto const& column = table.columns[selected[i]];
    column_decode_request request{std::cref(*column.plan_tree), value_result{column.dtype}, {}};
    if (!predicates.empty() && predicates[i].active()) {
      request.result = predicate_result{predicates[i], {}};
    }
    session.append(std::move(request));
  }
  return std::make_unique<cudf::table>(session.finish());
}

std::vector<std::size_t> all_columns(compressed_table const& table)
{
  std::vector<std::size_t> selected(table.num_columns());
  std::iota(selected.begin(), selected.end(), std::size_t{0});
  return selected;
}

// ── Filtering while decoding (env gate SIRIUS_EXP_FUSED_SCAN_FILTER) ────────
//
// Two waves inside one converter call:
//   wave 1: ballot the filter columns into mask words, round-robin on the pool
//           streams; stream 0 waits on the others (events, no host sync),
//           AND-combines, counts (per-chunk popcount + CUB scan -> chunk_offsets)
//           and D2H's the survivor count — the one added host sync, and it
//           gates wave-2 allocations.
//   wave 2: compactable columns decode straight to survivor width; the rest
//           decode full width and gather to survivor rows on their own
//           streams, in parallel, each waiting first for the gather map
//           (mask -> int32 row indices) built once on stream 0.
// Semantic preconditions and selectivity policy may decline to ordinary decode.
// Execution failures propagate after checked cleanup, preserving engine OOM recovery.
//
// Enable policy, in two stages:
//   Statically, before any device work: a column asking for a compacted route
//           must probe as that route; `full` is admitted for anything, with its
//           economics deferred below. Range-filter sources are exempt and
//           forced to bitpack_mask.
//   Dynamically, once the survivor count is known — both regimes report
//           declined_unselective, so the caller can remember it for the scan:
//           - any `full` output: proceed iff survivors/rows <= TIERB_MAX_SEL
//             (0.10);
//           - otherwise: give compaction up above MAX_SEL (0.35), unless a
//             dict_codes output is present (that gather wins at every
//             selectivity).
//           Giving up = masks dropped, ordinary decode, batch not row-filtered.

// The index walk is reachable from here (its launcher and the
// decode_selection.prefer_index_decode routing are both live). Effective kill
// switch: set SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL to a tiny value (the parse
// requires > 0).

// Both waves submit typed session requests through the same plan interpreter.
// probe_column describes row-route semantics, never asynchronous eligibility.

// A semantic precondition or completed selectivity decision may decline this
// optimization. Execution failures drain the phase and propagate unchanged;
// allocation, compilation, and device errors never become ordinary-decode retries.
std::optional<std::vector<std::unique_ptr<cudf::column>>> try_decompress_fused(
  compressed_table const& table,
  std::span<const std::size_t> selected,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  std::span<const ::cuda::stream_ref> streams,
  rmm::device_async_resource_ref mr)
{
  namespace sc = sirius::codegen;

  // Preconditions — all checked before touching the device. A refusal is a
  // normal-path decision (the caller runs the unfiltered decode, byte-identical
  // to what it would have run anyway), not an error; the reason line only
  // appears under SIRIUS_EXP_FUSED_SCAN_DIAG.
  //
  // Most of these are structural — index bounds, arity, chunk geometry, the
  // source cap — and protect the kernels from a request that would fault or
  // address outside a packed region.
  //
  // The four that call probe_column are different in kind and worth keeping
  // deliberately. The caller narrows its request with the SAME probe before
  // sending it, so these cannot disagree with it; they are assertions on the
  // boundary, not a second opinion. They stay because the failure they catch is
  // a WRONG MASK rather than a crash — render a range ballot over a plan that
  // is not bitpack-rooted and it reads the wrong bits and silently drops the
  // wrong rows — and because the caller's guarantee holds only through a chain
  // of reasoning across several files (a request reaches us only via
  // decompression_pushdown_scan::for_chunk, whose narrowing is what makes the claim true).
  // They are host-side plan-tree walks, once per batch, against device work
  // measured in milliseconds.
  auto refuse = [](char const* why) {
    if (sc::decompression_pushdown_diag_enabled())
      std::fprintf(stderr, "simpatico: filtered decode refused: %s\n", why);
    return std::nullopt;
  };
  if (!sc::decompression_pushdown_enabled()) return refuse("env gate off");
  size_t const k_range = request.filters.size();
  size_t const k_bool8 = request.bool8_filters.size();
  // Membership cap (drop-tail, sound — see max_membership_sources).
  size_t const k_member    = std::min(request.membership_filters.size(),
                                   sc::decompression_pushdown_max_membership_sources());
  size_t const k_total     = k_range + k_bool8 + k_member;
  result.source_generation = request.source_generation;  // echoed on every outcome
  if (k_member < request.membership_filters.size() && sc::decompression_pushdown_diag_enabled()) {
    std::fprintf(stderr,
                 "simpatico: filtered decode membership sources capped %zu -> %zu "
                 "(SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER)\n",
                 request.membership_filters.size(),
                 k_member);
  }
  if (k_total == 0) return refuse("no mask directives (no range/bool8/membership)");
  // The keep mask is a combine source, never a directive: with no real filter the
  // survivor rate is just the visible-row fraction, so compaction cannot pay off.
  bool const has_keep_mask = request.keep_mask_words != nullptr;
  if (k_total + (has_keep_mask ? 1 : 0) > 8) return refuse("more than 8 mask sources");
  if (request.routes.size() != selected.size())
    return refuse("request.routes not parallel to selected");
  if (streams.empty()) return refuse("no streams supplied");
  int64_t const num_rows = table.num_rows();
  if (num_rows <= 0) return refuse("num_rows <= 0");
  if (num_rows > std::numeric_limits<std::int32_t>::max())
    return refuse("num_rows > INT32_MAX (int32 row indices)");
  if (has_keep_mask && request.keep_mask_rows != num_rows)
    return refuse("keep mask row count != batch rows (positional mask misaligned)");
  for (auto const idx : selected) {
    if (idx >= table.columns.size()) return refuse("selected column index out of range");
    auto const& col = table.columns[idx];
    if (!col.plan_tree || col.num_rows != num_rows)
      return refuse("selected column missing plan_tree or row-count mismatch");
  }
  for (auto const& f : request.filters) {
    if (f.column >= selected.size()) return refuse("filter directive column out of range");
    if (f.pred.lo > f.pred.hi) return refuse("empty predicate range (lo > hi)");
    // can_produce_mask, not merely "decodes compacted": a dictionary column
    // decodes compacted but cannot ballot a numeric range, and one arriving as
    // a range source would be a latent wrong-mask gate.
    if (!probe_column(*table.columns[selected[f.column]].plan_tree).can_produce_mask())
      return refuse("filter column plan is not a bitpack-rooted range source");
  }
  for (auto const& b : request.bool8_filters) {
    if (b.column >= selected.size()) return refuse("bool8 directive column out of range");
    if (b.equals_any.empty()) return refuse("bool8 directive with empty equals_any");
    // The BOOL8 source rides the shipped dict-code pushdown: dictionary-rooted
    // plans only (the generic fallback would full-decode + compare — no win).
    if (!probe_column(*table.columns[selected[b.column]].plan_tree).can_answer_equality)
      return refuse("bool8 filter column plan not dictionary-rooted");
  }
  for (size_t mi = 0; mi < k_member; ++mi) {  // only the kept (capped) prefix
    auto const& m = request.membership_filters[mi];
    if (m.column >= selected.size()) return refuse("membership directive column out of range");
    if (!m.probe) return refuse("membership directive with an empty probe");
    // No plan-shape constraint: the key column decodes full width in wave 1
    // (any decodable plan) and takes its own tier in wave 2 like any output.
  }

  // A column answered off a dictionary delivers gather(bool8_fullwidth,
  // row_indices) — the compacted BOOL8 answer at the slot, no value decode, no
  // string re-compare downstream. These slots bypass the tier check below
  // (delivery is route-independent) and are EXCLUDED from the `full` selectivity
  // regime (a 1 B/row gather is cheap, not a full decode plus gather).
  std::vector<char> is_bool8_slot(selected.size(), 0);
  std::vector<int> bool8_of_slot(selected.size(), -1);
  for (size_t b = 0; b < k_bool8; ++b) {
    if (is_bool8_slot[request.bool8_filters[b].column])
      return refuse("duplicate bool8 directives on one column");
    is_bool8_slot[request.bool8_filters[b].column] = 1;
    bool8_of_slot[request.bool8_filters[b].column] = static_cast<int>(b);
  }

  // Static check, zero-cost: a requested compacted route must match the plan's
  // own; `full` is admitted for anything and takes the wave-2 full-decode +
  // survivor-gather path, with its economics enforced once the survivor count
  // is known (a full-width decode plus gather costs about the unfiltered path,
  // so only low-selectivity batches pay off; measured losses at high
  // selectivity: q1 +43.5%, q5 +6.2%). Range-filter source columns are exempt
  // from the check (probed bitpack-rooted above) and forced to bitpack_mask;
  // dictionary-answered sources get no exemption — they carry their own
  // route.
  std::vector<sc::decode_route> routes(request.routes.begin(), request.routes.end());
  for (auto const& f : request.filters)
    routes[f.column] = sc::decode_route::bitpack_mask;
  for (size_t i = 0; i < selected.size(); ++i) {
    if (is_bool8_slot[i]) continue;  // the slot's output is the compacted BOOL8 answer
    // `full` is always available — every plan decodes full width, and a
    // null-masked column declines the batch rather than corrupting it. Any
    // other requested route must be the one this plan
    // supports; one probe answers that, so a route and a capability cannot
    // disagree.
    if (routes[i] == sc::decode_route::full) { continue; }
    if (routes[i] != probe_column(*table.columns[selected[i]].plan_tree).compact_route) {
      return refuse("requested decode route does not match the column's plan shape");
    }
  }

  if (sc::decompression_pushdown_diag_enabled()) {
    std::string line = "simpatico: filtered decode wave-1 sources:";
    for (auto const& f : request.filters) {
      line += " range(col ";
      line += std::to_string(f.column) + " [" + std::to_string(f.pred.lo) + "," +
              std::to_string(f.pred.hi) + "])";
    }
    for (size_t mi = 0; mi < k_member; ++mi) {
      line += " member(col " + std::to_string(request.membership_filters[mi].column) + ")";
    }
    for (auto const& b : request.bool8_filters) {
      line += " bool8(col " + std::to_string(b.column) + " eq#" +
              std::to_string(b.equals_any.size()) + ")";
    }
    if (has_keep_mask) { line += " keep-mask"; }
    line += " rows=" + std::to_string(num_rows);
    std::fprintf(stderr, "%s\n", line.c_str());
  }

  // Declared before the try so the mid-flight catch can synchronize the streams BEFORE these
  // buffers unwind (their stream-ordered frees must not race the combine's cross-stream reads).
  std::vector<rmm::device_buffer> per_filter;
  rmm::device_buffer keep_mask_dev;
  // Full-width BOOL8 per bool8 source, retained for the wave-2 dual-delivery gather on the wave-1
  // lane that allocated it.
  std::vector<std::unique_ptr<cudf::column>> bool8_full(request.bool8_filters.size());
  std::vector<::cuda::stream_ref> bool8_lanes;
  // Probes that declined this chunk and were stood down to the AND identity.
  size_t declined_members = 0;
  auto reset_result       = [&](sc::scan_filter_status status) {
    result                   = sc::scan_filter_result{};
    result.status            = status;
    result.source_generation = request.source_generation;
  };
  try {
    int64_t const nc          = sc::selection_mask::ChunksFor(num_rows);
    int64_t const alloc_words = sc::selection_mask::AllocWordsFor(num_rows);
    auto const mask_bytes     = static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t);
    ::cuda::stream_ref s0     = streams.front();
    auto const same_stream    = [](::cuda::stream_ref stream) {
      return [stream](::cuda::stream_ref other) { return other.get() == stream.get(); };
    };

    result.num_rows   = num_rows;
    result.mask_words = rmm::device_buffer(mask_bytes, s0, mr);
    result.chunk_offsets =
      rmm::device_buffer(static_cast<std::size_t>(nc + 1) * sizeof(std::uint32_t), s0, mr);
    auto* combined = static_cast<std::uint32_t*>(result.mask_words.data());

    // ── Wave 1: one mask source per request, on the session's lanes in rotation. The first
    // request's lane is s0, so source 0 writes straight into the combined buffer (allocated on s0);
    // later sources write per-filter buffers allocated on the lane their request is assigned. Range
    // conjuncts run the range ballot; equality conjuncts run the BOOL8 pushdown and ballot it.
    per_filter.reserve(k_total > 1 ? k_total - 1 : 0);
    std::vector<std::uint32_t const*> mask_ptrs;
    mask_ptrs.reserve(k_total + 1);  // +1: the optional positional keep mask
    // Lanes other than s0 that produced a source, which the combine on s0 must wait for.
    std::vector<rmm::cuda_stream_view> producer_lanes;
    decode_session wave1{streams, mr};
    auto next_destination = [&] {
      auto const lane = wave1.next_stream();
      if (lane.get() != s0.get() &&
          std::none_of(producer_lanes.begin(), producer_lanes.end(), [&](auto other) {
            return other.value() == lane.get();
          }))
        producer_lanes.emplace_back(lane);
      auto* words =
        mask_ptrs.empty()
          ? combined
          : static_cast<std::uint32_t*>(per_filter.emplace_back(mask_bytes, lane, mr).data());
      mask_ptrs.push_back(words);
      return mask_destination{words, num_rows};
    };

    for (auto const& directive : request.filters) {
      auto const& column = table.columns[selected[directive.column]];
      if (wave1.append(mask_decode_request{
            *column.plan_tree, directive.pred, next_destination()}) != mask_source_status::ACCEPTED)
        throw plan_error("filtered decode: preflighted range source declined");
    }
    for (auto const& directive : request.bool8_filters) {
      auto const& column = table.columns[selected[directive.column]];
      bool8_lanes.push_back(wave1.next_stream());
      wave1.append(column_decode_request{
        std::cref(*column.plan_tree),
        predicate_result{decode_predicate{directive.equals_any}, next_destination()},
        {}});
    }
    // With another mask source to prime a prior from (a static range/bool8 strip or the
    // positional keep mask), membership probes run sequentially on s0 after the combine, so dead
    // rows skip the set/Bloom lookup. Otherwise they run concurrently here, prior-free.
    bool const sequential_membership = k_member > 0 && (k_range + k_bool8 > 0 || has_keep_mask);
    for (size_t m = 0; !sequential_membership && m < k_member; ++m) {
      auto const& directive = request.membership_filters[m];
      auto const& column    = table.columns[selected[directive.column]];
      auto const status     = wave1.append(mask_decode_request{
        *column.plan_tree, membership_source{directive.probe, column.dtype}, next_destination()});
      declined_members += status == mask_source_status::DECLINED;
    }

    // Order the combine on s0 after every other lane that produced a source.
    if (!producer_lanes.empty()) cudf::detail::join_streams(producer_lanes, s0);

    // ── Keep mask, uploaded before the combine so it primes the cascade's prior. No ballot
    // producer zeroes its tail (selection.hpp:52), so the gap between the host words and
    // alloc_words is memset here. Upload on s0 keeps the combine ordered behind it. When no source
    // has written `combined` yet, the keep mask lands there directly.
    if (has_keep_mask) {
      auto const host_words   = static_cast<std::size_t>((num_rows + 31) / 32);
      std::uint32_t* keep_dst = combined;
      if (!mask_ptrs.empty()) {
        keep_mask_dev =
          rmm::device_buffer(static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t), s0, mr);
        keep_dst = static_cast<std::uint32_t*>(keep_mask_dev.data());
      }
      if (static_cast<int64_t>(host_words) < alloc_words &&
          cudaMemsetAsync(
            keep_dst + host_words,
            0,
            (static_cast<std::size_t>(alloc_words) - host_words) * sizeof(std::uint32_t),
            s0.get()) != cudaSuccess) {
        throw plan_error("fused scan-filter: keep-mask padding memset failed");
      }
      if (cudaMemcpyAsync(keep_dst,
                          request.keep_mask_words,
                          host_words * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice,
                          s0.get()) != cudaSuccess) {
        throw plan_error("fused scan-filter: keep-mask upload failed");
      }
      mask_ptrs.push_back(keep_dst);
    }

    // ── Combine on stream 0.
    if (mask_ptrs.size() > 1) {
      sc::combine_masks_and(
        combined, mask_ptrs.data(), static_cast<int>(mask_ptrs.size()), alloc_words, s0);
    }

    // ── Sequential membership cascade, a single-lane session on s0 behind the combine. Each probe
    // takes `combined` as its prior and its result is ANDed back in, so the next probe sees the
    // tightened mask. A declining probe leaves `combined` untouched, the AND identity. The scratch
    // strip is written and read only on s0, so its stream-ordered free cannot race a reader.
    std::optional<decode_session> cascade;
    if (sequential_membership) {
      cascade.emplace(std::span{&s0, 1}, mr);
      rmm::device_buffer membership_scratch(mask_bytes, s0, mr);
      auto* member_words = static_cast<std::uint32_t*>(membership_scratch.data());
      std::array<std::uint32_t const*, 2> const and_sources{combined, member_words};
      for (size_t m = 0; m < k_member; ++m) {
        auto const& directive = request.membership_filters[m];
        auto const& column    = table.columns[selected[directive.column]];
        if (cascade->append(
              mask_decode_request{*column.plan_tree,
                                  membership_source{directive.probe, column.dtype, combined},
                                  {member_words, num_rows}}) == mask_source_status::DECLINED) {
          ++declined_members;
          continue;
        }
        sc::combine_masks_and(combined, and_sources.data(), 2, alloc_words, s0);
      }
    }

    // An all-ones source is the AND identity only while a real source still
    // zeroes the tail bits past num_rows, which CNT and the gather require. If
    // every source declined there is nothing left to filter by anyway, so take
    // the plain decode instead of counting a mask that is all ones. On a cascade
    // primed only by the keep mask, the survivor rate is just the visible-row
    // fraction, which the caller applies itself on the plain path.
    if (declined_members == k_total) {
      // Never count the padded all-ones masks: there is no real source to clear
      // the tail. This is an explicit policy decline after completed work.
      (void)wave1.finish();
      if (cascade) (void)cascade->finish();
      reset_result(sc::scan_filter_status::refused);
      return refuse("every membership source declined");
    }

    if (declined_members > 0 && sc::decompression_pushdown_diag_enabled()) {
      std::fprintf(stderr,
                   "simpatico: %zu of %zu membership probe(s) declined this chunk (rows=%lld); the "
                   "decode carries the rest and the join filters the remainder\n",
                   declined_members,
                   k_member,
                   static_cast<long long>(num_rows));
    }

    // ── CNT on stream 0. run_selection_cnt host-syncs s0 once (the survivor
    // count gates wave-2 allocations); after it returns, every wave-1 kernel,
    // the combine and the cascade have completed, so per_filter teardown is safe.
    sc::selection_mask sel{
      combined, num_rows, -1, static_cast<std::uint32_t*>(result.chunk_offsets.data())};
    sc::run_selection_cnt(sel, s0, mr);
    // CNT is the genuine host observation. finish also checks external phase work after request
    // submission before publishing the dual-delivery BOOL8 owners.
    bool8_full = wave1.finish();
    if (cascade) (void)cascade->finish();
    result.survivor_count = sel.survivor_count;
    per_filter.clear();
    keep_mask_dev = rmm::device_buffer{};  // combine consumed it; CNT synced s0

    // Selectivity guard, now that the survivor count is known; two regimes:
    //  * `full` outputs present: proceed only at sel <= the full-route threshold
    //    (default 0.10) — a full decode plus gather costs about the unfiltered
    //    path, so only a near-empty survivor set pays for the compacted batch.
    //  * no `full`: the 0.35 write-skip threshold (the mask walk costs about
    //    the plain decode at sel .5) covering bitpack_mask / delta_mask AND
    //    str_split (whose char-gather savings are dictionary-like in shape but
    //    weak at ~1-char widths — deliberately NOT exempt until measurements
    //    say otherwise), with only a dict_codes output
    //    exempting the batch (it wins 2.1-2.6x at ALL selectivities — the
    //    string-materialization savings are survivor-count-independent).
    // Both regimes explicitly decline after completed wave-1 work, reporting
    // declined_unselective for the rest of this scan/filter generation.
    bool any_dict_gather = false;
    bool any_full        = false;
    for (size_t i = 0; i < routes.size(); ++i) {
      if (is_bool8_slot[i])
        continue;  // dictionary-answered slots: a 1 B/row gather, excluded
                   // from the full-route regime
      any_dict_gather |= routes[i] == sc::decode_route::dict_codes;
      any_full |= routes[i] == sc::decode_route::full;
    }
    double const sel_frac = static_cast<double>(sel.survivor_count) / static_cast<double>(num_rows);
    bool const give_up =
      any_full ? sel_frac > sc::decompression_pushdown_full_route_max_selectivity()
               : (sel_frac > sc::decompression_pushdown_max_selectivity() && !any_dict_gather);
    if (give_up) {
      double const threshold = any_full ? sc::decompression_pushdown_full_route_max_selectivity()
                                        : sc::decompression_pushdown_max_selectivity();
      char const* env_name =
        any_full ? "SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL" : "SIRIUS_EXP_FUSED_SCAN_MAX_SEL";
      if (sc::decompression_pushdown_diag_enabled()) {
        std::fprintf(stderr,
                     "simpatico: filtered decode gave compaction up: sel=%.4f > %.4f (%s, "
                     "survivors=%lld/%lld)\n",
                     sel_frac,
                     threshold,
                     env_name,
                     static_cast<long long>(sel.survivor_count),
                     static_cast<long long>(num_rows));
      }
      reset_result(sc::scan_filter_status::declined_unselective);
      return std::nullopt;
    }

    // Per-batch enumeration pick for bitpack_mask outputs: walk the survivor
    // index list below the crossover, walk the mask bits above it. delta_mask
    // always walks the mask (the index walk rejects delta roots at render); the
    // dictionary route is unchanged.
    bool any_bitpack_mask = false;
    for (auto const t : routes)
      any_bitpack_mask |= t == sc::decode_route::bitpack_mask;
    bool const index_walk_pick =
      any_bitpack_mask && sel_frac <= sc::decompression_pushdown_index_walk_max_selectivity();
    if (sc::decompression_pushdown_diag_enabled()) {
      std::fprintf(stderr,
                   "simpatico: filtered decode row enumeration: %s (sel=%.4f, max=%.4f)\n",
                   index_walk_pick ? "index list" : "mask bits",
                   sel_frac,
                   sc::decompression_pushdown_index_walk_max_selectivity());
    }

    // ── Survivor index map on s0, built once per batch and shared by every consumer: the
    // full-route gathers, the BOOL8 gathers and the index-list decodes.
    result.routes        = routes;
    bool const any_bool8 = k_bool8 > 0;
    bool const indices_needed =
      (any_full || index_walk_pick || any_bool8) && sel.survivor_count > 0;
    cudf::column_view survivor_indices{
      cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0};
    if (indices_needed) {
      result.row_indices = rmm::device_buffer(
        static_cast<std::size_t>(sel.survivor_count) * sizeof(std::int32_t), s0, mr);
      sc::mask_to_row_indices(sel, static_cast<std::int32_t*>(result.row_indices.data()), s0);
      survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(sel.survivor_count),
                                           result.row_indices.data(),
                                           nullptr,
                                           0};
    }
    // Order a consuming lane after the indices kernel with one device-side wait; streams are FIFO,
    // so it covers every later launch on that lane.
    std::vector<::cuda::stream_ref> index_lanes{s0};
    auto const wait_for_indices = [&](::cuda::stream_ref lane) {
      if (!indices_needed || std::any_of(index_lanes.begin(), index_lanes.end(), same_stream(lane)))
        return;
      rmm::cuda_stream_view const producer{s0};
      cudf::detail::join_streams({&producer, 1}, lane);
      index_lanes.push_back(lane);
    };

    // ── Wave 2: every column decodes to survivor rows on its own lane. Compacted routes decode
    // compacted in the kernel; `full` routes decode full width and gather in the same request, so
    // the full-width column is released on its lane once the gather is queued. A slot answered off
    // a dictionary gathers its wave-1 BOOL8 on the lane that allocated it, never decoding values.
    std::vector<std::unique_ptr<cudf::column>> columns(selected.size());
    std::vector<std::size_t> decoded_positions;
    decoded_positions.reserve(selected.size());
    decode_session wave2{streams, mr};
    for (std::size_t i = 0; i < selected.size(); ++i) {
      if (is_bool8_slot[i]) {
        auto const slot = static_cast<std::size_t>(bool8_of_slot[i]);
        auto const lane = bool8_lanes[slot];
        wait_for_indices(lane);
        columns[i] = std::move(cudf::gather(cudf::table_view{{bool8_full[slot]->view()}},
                                            survivor_indices,
                                            cudf::out_of_bounds_policy::DONT_CHECK,
                                            lane,
                                            mr)
                                 ->release()
                                 .front());
        if (std::none_of(index_lanes.begin(), index_lanes.end(), same_stream(lane)))
          index_lanes.push_back(lane);
        // BOOL8 was excluded from full-value selectivity policy above; it is survivor-sized.
        result.routes[i] = sc::decode_route::full;
        continue;
      }
      auto const& column = table.columns[selected[i]];
      decode_selection selection;
      selection.mask               = &sel;
      selection.survivor_count     = sel.survivor_count;
      selection.survivor_indices   = survivor_indices;
      selection.route              = result.routes[i];
      selection.enumerate_by_index = index_walk_pick &&
                                     selection.route == sc::decode_route::bitpack_mask &&
                                     sel.survivor_count > 0;
      if (selection.route == sc::decode_route::full || selection.enumerate_by_index)
        wait_for_indices(wave2.next_stream());
      decoded_positions.push_back(i);
      wave2.append(column_decode_request{std::cref(*column.plan_tree),
                                         value_result{column.dtype},
                                         validated_selection(*column.plan_tree, selection)});
    }
    auto decoded = wave2.finish();
    for (std::size_t i = 0; i < decoded.size(); ++i)
      columns[decoded_positions[i]] = std::move(decoded[i]);
    // The session completed its own lanes; the indices kernel and the BOOL8 gathers are this
    // phase's work on s0 and the lanes in index_lanes.
    throw_if_cuda_error(synchronize_distinct(index_lanes), "filtered decode: gather completion");

    result.applied           = true;
    result.status            = sc::scan_filter_status::applied;
    result.keep_mask_applied = has_keep_mask;
    return columns;
  } catch (unsupported_nullable_selection const&) {
    // The existing row-selection policy excludes null-masked columns, which a full-route decode
    // discovers only once it runs: an explicit decline after a clean drain, not an execution
    // failure. A failed drain is a failure.
    if (auto const status = synchronize_distinct(streams); status != cudaSuccess) {
      reset_result(sc::scan_filter_status::failed);
      throw_if_cuda_error(status, "simpatico: filtered decode cleanup");
    }
    reset_result(sc::scan_filter_status::refused);
    return refuse("row selection on a null-masked column is not supported");
  } catch (...) {
    // Phase-owned masks outlive this drain, including work queued after session-owned decode
    // work. Preserve the original exception and OOM subtype.
    synchronize_distinct_or_log(streams, "simpatico: filtered decode cleanup failed");
    reset_result(sc::scan_filter_status::failed);
    throw;
  }
}

}  // namespace

// ── compressed_table ─────────────────────────────────────────────────────────

std::int64_t compressed_table::num_rows() const
{
  return columns.empty() ? 0 : columns.front().num_rows;
}

std::unique_ptr<cudf::table> compressed_table::decompress(::cuda::stream_ref stream,
                                                          rmm::device_async_resource_ref mr) const
{
  return simpatico::decompress(*this, stream, mr);
}

// ── split_plan_dsl ────────────────────────────────────────────────────────────

std::vector<std::string> split_plan_dsl(std::string_view plan_dsl)
{
  return split_plan_dsl_impl(plan_dsl);
}

// ── compress_with_plan ────────────────────────────────────────────────────────

namespace {
// Split the per-column plan DSL and validate it against the table + names.
// Shared preamble of all three compress_with_plan overloads.
std::vector<std::string> split_and_validate_plans(std::string_view plan_dsl,
                                                  cudf::table_view table,
                                                  std::vector<std::string> const& column_names)
{
  auto plans = split_plan_dsl_impl(plan_dsl);
  validate_plan_count(plans.size(), table.num_columns());
  validate_column_names(column_names, plans.size());
  // Sliced column views (offset != 0) are supported: every encode kernel reads
  // data<T>() (= head<T>() + offset) rather than head<T>() so the correct
  // elements are compressed regardless of the view's allocation base.
  return plans;
}
}  // namespace

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    ::cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  nvtx_scoped_range nvtx_range{"simpatico::compress_table[serial]"};
  auto plans = split_and_validate_plans(plan_dsl, table, column_names);

  compressed_table out;
  out.columns.reserve(plans.size());
  for (size_t i = 0; i < plans.size(); ++i) {
    std::string err;
    auto plan_tree =
      compress_column(table.column(static_cast<cudf::size_type>(i)), plans[i], stream, mr, &err);
    if (!plan_tree) throw plan_error(err.empty() ? "compress failed" : err);
    compressed_column col;
    col.dtype     = table.column(static_cast<cudf::size_type>(i)).type();
    col.num_rows  = table.num_rows();
    col.plan_tree = std::move(plan_tree);
    if (!column_names.empty()) col.name = column_names[i];
    out.columns.push_back(std::move(col));
  }
  return out;
}

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    int column_threads,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  nvtx_scoped_range nvtx_range{"simpatico::compress_table[threads]"};
  auto plans = split_and_validate_plans(plan_dsl, table, column_names);
  leased_pool lp(column_threads);
  return compress_columns_parallel(table, plans, lp.pool, mr, column_names);
}

compressed_table compress_with_plan(cudf::table_view table,
                                    std::string_view plan_dsl,
                                    simpatico::stream_pool& pool,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  nvtx_scoped_range nvtx_range{"simpatico::compress_table[pool]"};
  auto plans = split_and_validate_plans(plan_dsl, table, column_names);
  return compress_columns_parallel(table, plans, pool, mr, column_names);
}

// ── decompress ────────────────────────────────────────────────────────────────

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[serial]"};
  return decode_table_columns(table, all_columns(table), {}, {&stream, 1}, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        int column_threads,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[threads]"};
  leased_pool lp(column_threads);
  return decode_table_columns(table, all_columns(table), {}, lp.pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[pool]"};
  return decode_table_columns(table, all_columns(table), {}, pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        ::cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,serial]"};
  return decode_table_columns(table, selected_columns, {}, {&stream, 1}, mr);
}

namespace {

// The plan tree of one column, or nullptr with `error_out` set.
PlanTree const* column_plan(const compressed_table& table,
                            std::size_t column_index,
                            std::string* error_out,
                            char const* what)
{
  if (column_index >= table.columns.size()) {
    if (error_out) { *error_out = std::string(what) + ": column index out of range"; }
    return nullptr;
  }
  auto const& col = table.columns[column_index];
  if (!col.plan_tree) {
    if (error_out) { *error_out = std::string(what) + ": column has no plan tree"; }
    return nullptr;
  }
  return col.plan_tree.get();
}

std::unique_ptr<cudf::column> decode_complete_column(PlanTree const& plan,
                                                     cudf::data_type stored_type,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr,
                                                     decode_selection const* selection,
                                                     std::string* error_out)
{
  column_decode_request request{std::cref(plan), value_result{stored_type}, {}};
  if (selection) {
    try {
      request.selection.emplace(plan, *selection);
    } catch (std::invalid_argument const& error) {
      if (error_out) *error_out = error.what();
      return nullptr;
    }
  }
  return decode_one(request, stream, mr);
}

}  // namespace

std::unique_ptr<cudf::column> decompress_column_rows(const compressed_table& table,
                                                     std::size_t column_index,
                                                     sirius::codegen::chunk_row_set const& rows,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr,
                                                     std::string* error_out)
{
  auto const* tree = column_plan(table, column_index, error_out, "decompress_column_rows");
  if (tree == nullptr) { return nullptr; }
  if (rows.num_rows < 0 || rows.num_rows > std::numeric_limits<cudf::size_type>::max() ||
      !rows.valid()) {
    if (error_out) { *error_out = "decompress_column_rows: the row set is not valid"; }
    return nullptr;
  }
  // Only the bitpack compacted decode reads a row set, and discovering that
  // costs no device work.
  if (probe_column(*tree).compact_route != sirius::codegen::decode_route::bitpack_mask) {
    if (error_out) {
      *error_out = "decompress_column_rows: this plan has no random-access decode for a row set";
    }
    return nullptr;
  }

  decode_selection sel;
  sel.rows           = &rows;
  sel.survivor_count = rows.num_survivors;
  sel.route          = sirius::codegen::decode_route::bitpack_mask;

  return decode_complete_column(
    *tree, table.columns[column_index].dtype, stream, mr, &sel, error_out);
}

std::unique_ptr<cudf::column> decompress_column_compacted(
  const compressed_table& table,
  std::size_t column_index,
  sirius::codegen::selection_mask const& mask,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out)
{
  auto const* tree = column_plan(table, column_index, error_out, "decompress_column_compacted");
  if (tree == nullptr) { return nullptr; }
  if (mask.words == nullptr || mask.chunk_offsets == nullptr || mask.survivor_count < 0) {
    if (error_out) {
      *error_out = "decompress_column_compacted: the mask has not been counted (no chunk_offsets)";
    }
    return nullptr;
  }
  auto const route = probe_column(*tree).compact_route;
  if (route == sirius::codegen::decode_route::full) {
    if (error_out) { *error_out = "decompress_column_compacted: this plan has no compacted route"; }
    return nullptr;
  }

  decode_selection sel;
  sel.mask           = &mask;
  sel.survivor_count = mask.survivor_count;
  sel.route          = route;

  return decode_complete_column(
    *tree, table.columns[column_index].dtype, stream, mr, &sel, error_out);
}

std::unique_ptr<cudf::column> decompress_column_full(const compressed_table& table,
                                                     std::size_t column_index,
                                                     ::cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr,
                                                     std::string* error_out)
{
  auto const* tree = column_plan(table, column_index, error_out, "decompress_column_full");
  if (tree == nullptr) { return nullptr; }
  return decode_complete_column(
    *tree, table.columns[column_index].dtype, stream, mr, nullptr, error_out);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        int column_threads,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,threads]"};
  leased_pool lp(column_threads);
  return decode_table_columns(table, selected_columns, {}, lp.pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  return decompress(table, selected_columns, pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        std::span<const decode_predicate> predicates,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  return decompress(table, selected_columns, predicates, pool.refs(), mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        std::span<const ::cuda::stream_ref> streams,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,streams]"};
  return decode_table_columns(table, selected_columns, {}, streams, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        std::span<const decode_predicate> predicates,
                                        std::span<const ::cuda::stream_ref> streams,
                                        rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[selected,predicated,streams]"};
  if (predicates.size() != selected_columns.size()) {
    throw plan_error("decompress: predicates and selected_columns must be the same length");
  }
  return decode_table_columns(table, selected_columns, predicates, streams, mr);
}

std::unique_ptr<cudf::table> decompress_scan_filter(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  std::span<const ::cuda::stream_ref> streams,
  ::cuda::stream_ref /*stream*/,
  rmm::device_async_resource_ref mr)
{
  nvtx_scoped_range nvtx_range{"simpatico::decompress_table[scan_filter,streams]"};
  result = sirius::codegen::scan_filter_result{};
  if (auto cols = try_decompress_fused(table, selected_columns, request, result, streams, mr)) {
    if (sirius::codegen::decompression_pushdown_diag_enabled()) {
      int n_a = 0, n_delta = 0, n_dict = 0, n_str_split = 0, n_b = 0;
      for (auto const t : result.routes) {
        n_a += t == sirius::codegen::decode_route::bitpack_mask;
        n_delta += t == sirius::codegen::decode_route::delta_mask;
        n_dict += t == sirius::codegen::decode_route::dict_codes;
        n_str_split += t == sirius::codegen::decode_route::str_split;
        n_b += t == sirius::codegen::decode_route::full;
      }
      std::fprintf(stderr,
                   "simpatico: filtered decode applied: survivors=%lld/%lld "
                   "routes bitpack=%d delta=%d dict=%d str_split=%d full=%d sources=%zu\n",
                   static_cast<long long>(result.survivor_count),
                   static_cast<long long>(result.num_rows),
                   n_a,
                   n_delta,
                   n_dict,
                   n_str_split,
                   n_b,
                   request.filters.size() + request.bool8_filters.size());
    }
    // Every route, including `full` and the dictionary-answered BOOL8 slots, came back
    // survivor-sized and completed.
    return std::make_unique<cudf::table>(std::move(*cols));
  }
  // Gate off / nothing requested / explicit completed policy refusal:
  // exactly the unfiltered path — with one obligation: when the request routed
  // dictionary-answered equalities (bool8_filters) into the mask, the fallback
  // must NOT be a plain decode, or every declined batch would silently lose the
  // BOOL8-substitution win (q19 -21.4%). Re-express them as ordinary
  // decode_predicates so the fallback degrades to the substitution behaviour,
  // never below it.
  if (!request.bool8_filters.empty()) {
    std::vector<decode_predicate> predicates(selected_columns.size());
    for (auto const& b : request.bool8_filters) {
      if (b.column < predicates.size()) predicates[b.column].equals_any = b.equals_any;
    }
    return decode_table_columns(table, selected_columns, predicates, streams, mr);
  }
  return decode_table_columns(table, selected_columns, {}, streams, mr);
}

std::unique_ptr<cudf::table> decompress_scan_filter(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  simpatico::stream_pool& pool,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  std::string* /*error_out*/)
{
  return decompress_scan_filter(table, selected_columns, request, result, pool.refs(), stream, mr);
}

}  // namespace simpatico
