// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/selection/decompression_pushdown_policy.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
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
    // Destructors cannot throw; run_column_workers' own sync_all already
    // surfaced any async error before this cleanup sync runs.
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
    rmm::cuda_stream_view s{pool.streams[i % n_streams]};
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
  run_column_workers(plans.size(), pool, [&](size_t i, rmm::cuda_stream_view stream) {
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

// Restore a decoded column's logical type when it differs from the stored column
// dtype only in interpretation of identical bits (same physical width) — e.g. the
// INT64 storage a codec produced for a DECIMAL64 column back to DECIMAL64 with its
// scale. The codecs run on the underlying integer storage of fixed-point columns,
// so the bytes are already correct; this only re-tags the column. A no-op when the
// types already match.
std::unique_ptr<cudf::column> apply_stored_dtype(std::unique_ptr<cudf::column> col,
                                                 cudf::data_type stored)
{
  if (!col || col->type() == stored) return col;
  if (!cudf::is_fixed_width(col->type()) || !cudf::is_fixed_width(stored) ||
      cudf::size_of(col->type()) != cudf::size_of(stored)) {
    return col;
  }
  auto const n  = col->size();
  auto const nc = col->null_count();
  auto contents = col->release();
  rmm::device_buffer null_mask =
    contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{};
  return std::make_unique<cudf::column>(
    stored, n, std::move(*contents.data), std::move(null_mask), nc, std::move(contents.children));
}

std::unique_ptr<cudf::table> decompress_columns_parallel(compressed_table const& table,
                                                         stream_pool& pool,
                                                         rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols(table.num_columns());
  run_column_workers(
    static_cast<size_t>(table.num_columns()), pool, [&](size_t i, rmm::cuda_stream_view stream) {
      std::string err;
      auto col = decompress_column(*table.columns[i].plan_tree, stream, mr, &err);
      if (!col) throw plan_error(err.empty() ? "decompress failed" : err);
      cols[i] = apply_stored_dtype(std::move(col), table.columns[i].dtype);
    });
  return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> decompress_columns_parallel(
  compressed_table const& table,
  std::span<const std::size_t> selected,
  std::span<const decode_predicate> predicates,
  stream_pool& pool,
  rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols(selected.size());
  run_column_workers(selected.size(), pool, [&](size_t i, rmm::cuda_stream_view stream) {
    auto const idx = selected[i];
    if (idx >= table.columns.size()) throw plan_error("selected column index out of range");
    decode_predicate const* pred =
      (i < predicates.size() && predicates[i].active()) ? &predicates[i] : nullptr;
    std::string err;
    auto col = decompress_column(*table.columns[idx].plan_tree, stream, mr, &err, pred);
    if (!col) throw plan_error(err.empty() ? "decompress failed" : err);
    // A predicate result is BOOL8 by contract; re-tagging it with the column's
    // stored dtype would be a lie (and, for a same-width stored type, a silent
    // one), so the type restore only applies to a reconstructed column.
    cols[i] = pred ? std::move(col) : apply_stored_dtype(std::move(col), table.columns[idx].dtype);
  });
  return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> decompress_columns_parallel(compressed_table const& table,
                                                         std::span<const std::size_t> selected,
                                                         stream_pool& pool,
                                                         rmm::device_async_resource_ref mr)
{
  return decompress_columns_parallel(table, selected, {}, pool, mr);
}

// ── Filtering while decoding (env gate SIRIUS_EXP_FUSED_SCAN_FILTER) ────────
//
// Two waves inside one converter call:
//   wave 1: ballot the filter columns into mask words, round-robin on the pool
//           streams; stream 0 waits on the others (events, no host sync),
//           AND-combines, counts (per-chunk popcount + CUB scan -> chunk_offsets)
//           and D2H's the survivor count — the one added host sync, and it
//           gates wave-2 allocations.
//   wave 2: compactable columns decode straight to survivor width, the rest
//           decode plainly, in parallel; the full-width columns' gather map
//           (mask -> int32 row indices) is built on stream 0 concurrently.
// Any missing precondition => std::nullopt and the caller runs the unfiltered
// path byte-identically.
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

// Wave 1 and the compact-capability probe come from the entry points published
// in plan/decompress.cpp: probe_column + decompress_column_selection_mask. Wave 2 calls
// decompress_column(..., decode_selection const* sel) — compact_capable decodes to survivor width,
// and everything else decodes full width and is gathered inside the call.

// RAII CUDA events for the wave-1 -> combine cross-stream join.
struct event_set {
  std::vector<cudaEvent_t> events;

  cudaEvent_t make()
  {
    cudaEvent_t ev{};
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess)
      throw plan_error("filtered decode: cudaEventCreate failed");
    events.push_back(ev);
    return ev;
  }

  ~event_set()
  {
    for (auto ev : events)
      cudaEventDestroy(ev);
  }
};

// The two-wave filtered decode. Returns std::nullopt in two cases:
//   (a) a precondition fails BEFORE any device work — nothing was issued;
//   (b) anything fails mid-flight — the pool is synchronized, `result` is
//       reset, and the WHOLE batch is retried unfiltered by the caller (the
//       required fallback semantics; per-column errors from decompress_column
//       surface here as plan_error).
std::optional<std::vector<std::unique_ptr<cudf::column>>> try_decompress_fused(
  compressed_table const& table,
  std::span<const std::size_t> selected,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  stream_pool& pool,
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
  if (k_total > 8) return refuse("more than 8 mask sources");
  if (request.routes.size() != selected.size())
    return refuse("request.routes not parallel to selected");
  if (pool.streams.empty()) return refuse("stream pool empty");
  int64_t const num_rows = table.num_rows();
  if (num_rows <= 0) return refuse("num_rows <= 0");
  if (num_rows > std::numeric_limits<std::int32_t>::max())
    return refuse("num_rows > INT32_MAX (int32 row indices)");
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
    // `full` is always available — every plan decodes full width, and the
    // gather guards null-masked columns with a loud error rather than
    // corrupting. Any other requested route must be the one this plan
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
    line += " rows=" + std::to_string(num_rows);
    std::fprintf(stderr, "%s\n", line.c_str());
  }

  // Declared before the try so the mid-flight catch can pool.sync_all() BEFORE
  // these buffers/events unwind (their stream-ordered frees must not race the
  // combine's cross-stream reads).
  std::vector<rmm::device_buffer> per_filter;
  // Full-width BOOL8 per bool8 source, retained for the wave-2 dual-delivery
  // gather; declared before the try so the catch's pool.sync_all() runs
  // before any cross-stream consumer unwinds them.
  std::vector<std::unique_ptr<cudf::column>> bool8_full(request.bool8_filters.size());
  event_set join_events;
  bool unselective = false;  // distinguishes giving compaction up from a real failure

  try {
    int64_t const nc          = sc::selection_mask::ChunksFor(num_rows);
    int64_t const alloc_words = sc::selection_mask::AllocWordsFor(num_rows);
    size_t const n_streams    = pool.streams.size();
    rmm::cuda_stream_view s0{pool.streams[0]};

    result.num_rows = num_rows;
    result.mask_words =
      rmm::device_buffer(static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t), s0, mr);
    result.chunk_offsets =
      rmm::device_buffer(static_cast<std::size_t>(nc + 1) * sizeof(std::uint32_t), s0, mr);
    auto* combined = static_cast<std::uint32_t*>(result.mask_words.data());

    // ── Wave 1: mask sources round-robin on the pool streams. Source 0 writes
    // straight into the combined buffer on stream 0 (its allocation stream);
    // sources 1..k-1 into per-filter buffers allocated on the stream that
    // writes them. Range conjuncts run the range ballot; equality conjuncts run
    // the shipped BOOL8 pushdown then the packed-mask adapter.
    per_filter.reserve(k_total > 1 ? k_total - 1 : 0);
    std::vector<std::uint32_t const*> mask_ptrs;
    mask_ptrs.reserve(k_total);
    mask_ptrs.push_back(combined);

    auto submit_mask_source = [&](size_t s, auto&& produce) {
      rmm::cuda_stream_view stream =
        (s == 0) ? s0 : rmm::cuda_stream_view{pool.streams[s % n_streams]};
      std::uint32_t* dst = combined;
      if (s > 0) {
        per_filter.emplace_back(
          static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t), stream, mr);
        dst = static_cast<std::uint32_t*>(per_filter.back().data());
        mask_ptrs.push_back(dst);
      }
      produce(dst, stream);
      if (s > 0 && stream.value() != s0.value()) {
        // Publish this stream's mask to stream 0 without a host sync.
        cudaEvent_t ev = join_events.make();
        if (cudaEventRecord(ev, stream.value()) != cudaSuccess ||
            cudaStreamWaitEvent(s0.value(), ev, 0) != cudaSuccess) {
          throw plan_error("filtered decode: wave-1 stream join failed");
        }
      }
    };

    for (size_t f = 0; f < k_range; ++f) {
      submit_mask_source(f, [&](std::uint32_t* dst, rmm::cuda_stream_view stream) {
        auto const& directive = request.filters[f];
        auto const& col       = table.columns[selected[directive.column]];
        std::string err;
        if (!decompress_column_selection_mask(
              *col.plan_tree, directive.pred, dst, stream, mr, &err)) {
          throw plan_error(err.empty() ? "filtered decode: range ballot failed" : err);
        }
      });
    }
    for (size_t b = 0; b < k_bool8; ++b) {
      submit_mask_source(k_range + b, [&](std::uint32_t* dst, rmm::cuda_stream_view stream) {
        auto const& directive = request.bool8_filters[b];
        auto const& col       = table.columns[selected[directive.column]];
        decode_predicate pred;
        pred.equals_any = directive.equals_any;
        std::string err;
        auto flags = decompress_column(*col.plan_tree, stream, mr, &err, &pred);
        if (!flags)
          throw plan_error(err.empty() ? "filtered decode: bool8 predicate decode failed" : err);
        if (flags->type().id() != cudf::type_id::BOOL8 || flags->size() != num_rows)
          throw plan_error("filtered decode: bool8 predicate result shape mismatch");
        if (flags->null_count() != 0)
          throw plan_error("filtered decode: null-masked bool8 predicate result");
        sc::mask_from_bool8(flags->view().data<std::uint8_t>(), num_rows, dst, stream);
        // Keep the full-width BOOL8 alive — wave 2 gathers it at this
        // directive's slot (compacted BOOL8, 1 B/row) instead of decoding the
        // column's values. Contents are settled before wave-2
        // submission (the adapter kernel follows the fill on this stream, and
        // the CNT host sync transitively covers it via the join event).
        bool8_full[b] = std::move(flags);
      });
    }
    for (size_t m = 0; m < k_member; ++m) {
      submit_mask_source(
        k_range + k_bool8 + m, [&](std::uint32_t* dst, rmm::cuda_stream_view stream) {
          auto const& directive = request.membership_filters[m];
          auto const& col       = table.columns[selected[directive.column]];
          std::string err;
          // Full-width key decode (any plan shape), then the type-erased
          // device probe (in_list / cuco set / Bloom) -> BOOL8, then the
          // packed-mask adapter. Probe contract: all work on `stream`.
          auto keys = decompress_column(*col.plan_tree, stream, mr, &err);
          if (!keys)
            throw plan_error(err.empty() ? "filtered decode: membership key decode failed" : err);
          auto keys_typed = apply_stored_dtype(std::move(keys), col.dtype);
          auto flags      = directive.probe(keys_typed->view(), stream, mr);
          if (!flags || flags->type().id() != cudf::type_id::BOOL8 || flags->size() != num_rows)
            throw plan_error("filtered decode: membership probe result shape mismatch");
          if (flags->null_count() != 0)
            throw plan_error("filtered decode: null-masked membership probe result");
          sc::mask_from_bool8(flags->view().data<std::uint8_t>(), num_rows, dst, stream);
          // keys/flags die here: stream-ordered frees behind the adapter
          // kernel on the same stream.
        });
    }

    // ── Combine + CNT on stream 0. run_selection_cnt host-syncs s0 once (the
    // survivor count gates wave-2 allocations); after it returns, every wave-1
    // kernel and the combine have completed, so per_filter teardown is safe.
    if (k_total > 1) {
      sc::combine_masks_and(combined, mask_ptrs.data(), static_cast<int>(k_total), alloc_words, s0);
    }
    sc::selection_mask sel{
      combined, num_rows, -1, static_cast<std::uint32_t*>(result.chunk_offsets.data())};
    sc::run_selection_cnt(sel, s0, mr);
    result.survivor_count = sel.survivor_count;
    per_filter.clear();

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
    // Both regimes give up through the mid-flight machinery below, reporting
    // declined_unselective so the caller can remember it for the rest of the
    // scan (sync, drop masks, ordinary decode; batch NOT tagged ROW_FILTERED).
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
      unselective = true;
      throw plan_error("selectivity " + std::to_string(sel_frac) + " above " + env_name + " " +
                       std::to_string(threshold));
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

    // ── Survivor index map on stream 0, overlapping wave 2 (column 0's wave-2
    // work serializes behind it on s0; the other streams run free). Built ONCE
    // per batch and SHARED by every consumer: the full-width gathers and the
    // index-list decodes (the same int32 buffer).
    result.routes        = routes;
    bool const any_bool8 = k_bool8 > 0;  // dual-delivery gathers need the indices too
    cudf::column_view survivor_indices{
      cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0};
    if ((any_full || index_walk_pick || any_bool8) && sel.survivor_count > 0) {
      result.row_indices = rmm::device_buffer(
        static_cast<std::size_t>(sel.survivor_count) * sizeof(std::int32_t), s0, mr);
      sc::mask_to_row_indices(sel, static_cast<std::int32_t*>(result.row_indices.data()), s0);
      survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(sel.survivor_count),
                                           result.row_indices.data(),
                                           nullptr,
                                           0};
      // Index consumers (the full-width gathers, the index-list decodes) run on the other pool
      // streams; order them after the
      // indices kernel on s0 with a device-side wait (streams are FIFO, so one
      // up-front wait per stream covers every wave-2 launch on it).
      cudaEvent_t ev_idx = join_events.make();
      if (cudaEventRecord(ev_idx, s0.value()) != cudaSuccess)
        throw plan_error("filtered decode: indices event record failed");
      for (size_t si = 1; si < n_streams; ++si) {
        if (cudaStreamWaitEvent(pool.streams[si], ev_idx, 0) != cudaSuccess)
          throw plan_error("filtered decode: indices stream wait failed");
      }
    }

    // ── Wave 2: the compactable columns decode to survivor width and the rest
    // decode full width and are gathered, round-robin on the pool streams via
    // the published
    // decompress_column(..., decode_selection const*) contract. Everything
    // wave 2 consumes (mask, chunk_offsets) completed before the CNT host sync
    // above, so no cross-stream waits are needed.
    std::vector<std::unique_ptr<cudf::column>> cols(selected.size());
    run_column_workers(selected.size(), pool, [&](size_t i, rmm::cuda_stream_view stream) {
      auto const& col = table.columns[selected[i]];
      if (is_bool8_slot[i]) {
        // The slot's output is the wave-1 BOOL8 gathered to survivor rows
        // (1 B/row) — the column's values are never decoded and the downstream
        // residual sees the substitution column pre-compacted. No
        // apply_stored_dtype (BOOL8 by contract, like the unfiltered predicate
        // path).
        auto const& full = bool8_full[static_cast<size_t>(bool8_of_slot[i])];
        if (sel.survivor_count == 0) {
          cols[i] = cudf::make_fixed_width_column(
            cudf::data_type{cudf::type_id::BOOL8}, 0, cudf::mask_state::UNALLOCATED, stream, mr);
          return;
        }
        auto gathered = cudf::gather(cudf::table_view{{full->view()}},
                                     survivor_indices,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
        cols[i]       = std::move(gathered->release()[0]);
        return;
      }
      decode_selection dsel;
      dsel.mask             = &sel;
      dsel.survivor_count   = sel.survivor_count;
      dsel.survivor_indices = survivor_indices;
      // One route value, so the modes cannot contradict each other: the
      // bitpack/delta roots consume the mask in-kernel, the dictionary route
      // decodes codes under the mask and gathers surviving keys, and `full`
      // takes the gather path.
      dsel.route = result.routes[i];
      // Below the crossover, bitpack roots decode from the survivor
      // index list (populated above whenever index_walk_pick, per the
      // decode_selection contract); the dispatch silently keeps the mask walk
      // on any anomaly, delta roots ignore it, a dictionary gather is
      // unchanged.
      dsel.enumerate_by_index = index_walk_pick &&
                                result.routes[i] == sc::decode_route::bitpack_mask &&
                                sel.survivor_count > 0;
      std::string err;
      auto out = decompress_column(*col.plan_tree, stream, mr, &err, nullptr, &dsel);
      if (!out) throw plan_error(err.empty() ? "filtered decode: decompress failed" : err);
      cols[i] = apply_stored_dtype(std::move(out), col.dtype);
    });
    // run_column_workers ended with pool.sync_all(), which also covers the
    // mask_to_row_indices launch on s0.

    result.applied = true;
    result.status  = sc::scan_filter_status::applied;
    return cols;
  } catch (std::exception const& e) {
    std::fprintf(
      stderr, "simpatico: filtered decode fell back to the ordinary decode (%s)\n", e.what());
    pool.sync_all();  // quiesce in-flight wave kernels before buffers unwind
    result = sirius::codegen::scan_filter_result{};
    // A distinguishable outcome for the caller to remember: one batch too
    // unselective to compact predicts the scan's remaining batches — latched
    // per {operator, source_generation} (a newer filter set clears it).
    result.status =
      unselective ? sc::scan_filter_status::declined_unselective : sc::scan_filter_status::failed;
    result.source_generation = request.source_generation;
    return std::nullopt;
  }
}

}  // namespace

// ── compressed_table ─────────────────────────────────────────────────────────

std::int64_t compressed_table::num_rows() const
{
  return columns.empty() ? 0 : columns.front().num_rows;
}

std::unique_ptr<cudf::table> compressed_table::decompress(rmm::cuda_stream_view stream,
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
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr,
                                    std::vector<std::string> column_names)
{
  nvtx3::scoped_range nvtx_range{"simpatico::compress_table[serial]"};
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
  nvtx3::scoped_range nvtx_range{"simpatico::compress_table[threads]"};
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
  nvtx3::scoped_range nvtx_range{"simpatico::compress_table[pool]"};
  auto plans = split_and_validate_plans(plan_dsl, table, column_names);
  return compress_columns_parallel(table, plans, pool, mr, column_names);
}

// ── decompress ────────────────────────────────────────────────────────────────

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[serial]"};
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(table.num_columns());
  for (auto const& col : table.columns) {
    if (!col.plan_tree) throw plan_error("compressed_table column missing plan_tree");
    std::string err;
    auto c = decompress_column(*col.plan_tree, stream, mr, &err);
    if (!c) throw plan_error(err.empty() ? "decompress failed" : err);
    cols.push_back(apply_stored_dtype(std::move(c), col.dtype));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        int column_threads,
                                        rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[threads]"};
  leased_pool lp(column_threads);
  return decompress_columns_parallel(table, lp.pool, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[pool]"};
  return decompress_columns_parallel(table, pool, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[selected,serial]"};
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(selected_columns.size());
  for (auto const idx : selected_columns) {
    if (idx >= table.columns.size()) throw plan_error("selected column index out of range");
    auto const& col = table.columns[idx];
    if (!col.plan_tree) throw plan_error("compressed_table column missing plan_tree");
    std::string err;
    auto c = decompress_column(*col.plan_tree, stream, mr, &err);
    if (!c) throw plan_error(err.empty() ? "decompress failed" : err);
    cols.push_back(apply_stored_dtype(std::move(c), col.dtype));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        int column_threads,
                                        rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[selected,threads]"};
  leased_pool lp(column_threads);
  return decompress_columns_parallel(table, selected_columns, lp.pool, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[selected,pool]"};
  return decompress_columns_parallel(table, selected_columns, pool, mr);
}

std::unique_ptr<cudf::table> decompress(const compressed_table& table,
                                        std::span<const std::size_t> selected_columns,
                                        std::span<const decode_predicate> predicates,
                                        simpatico::stream_pool& pool,
                                        rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[selected,predicated,pool]"};
  if (predicates.size() != selected_columns.size()) {
    throw plan_error("decompress: predicates and selected_columns must be the same length");
  }
  return decompress_columns_parallel(table, selected_columns, predicates, pool, mr);
}

std::unique_ptr<cudf::table> decompress_scan_filter(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  simpatico::stream_pool& pool,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[scan_filter,pool]"};
  result = sirius::codegen::scan_filter_result{};
  if (auto cols = try_decompress_fused(table, selected_columns, request, result, pool, mr)) {
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
    // Reconcile the wave's ragged output into one uniformly survivor-sized
    // table before it leaves: the compacted routes came back survivor-sized and
    // the `full` ones full width, and nothing outside knows which is which.
    auto reconciled = compact_scan_filter_output(std::move(*cols), result, stream, mr, error_out);
    if (reconciled) { return reconciled; }
    // The assembly refused (a null-masked column, a mis-sized output). Fall
    // through to the unfiltered decode below and report nothing applied — the
    // caller must not see a half-filtered batch.
    result        = sirius::codegen::scan_filter_result{};
    result.status = sirius::codegen::scan_filter_status::failed;
  }
  // Gate off / nothing requested / policy refusal / mid-flight fallback:
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
    return decompress_columns_parallel(table, selected_columns, predicates, pool, mr);
  }
  return decompress_columns_parallel(table, selected_columns, pool, mr);
}

}  // namespace simpatico
