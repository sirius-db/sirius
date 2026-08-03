// SPDX-License-Identifier: Apache-2.0
#include "api/simpatico_codegen.hpp"

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

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
    pool.sync_all();
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
  pool.sync_all();
  if (first_exception) std::rethrow_exception(first_exception);
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

// ── Fused scan-filter orchestration (env gate SIRIUS_EXP_FUSED_SCAN_FILTER) ──
//
// Two-wave schedule inside one converter call (contract W4):
//   wave 1: K1 mask decodes for the filter columns, round-robin on the pool
//           streams; stream 0 waits on the others (events, no host sync),
//           AND-combines the masks, runs CNT (per-chunk popcount + CUB scan ->
//           chunk_offsets) and D2H's the survivor count — the single added
//           host sync, it gates wave-2 allocations.
//   wave 2: TierA K3 compacted decodes + TierB plain decodes in parallel on
//           the pool streams (run_column_workers, as today); the TierB gather
//           map (mask -> int32 row indices) is built on stream 0 concurrently.
// Any missing precondition ⇒ std::nullopt, and the caller runs today's path
// byte-identically (same kernels, same allocations, zero added syncs).
//
// Enable policy (measured on the M2/M3 + iteration-3/4 A/B matrices):
//   RULE 1 (static): compacted tiers (tier_a / tier_a_delta / tier_dict_k5)
//           verify against W3's plan_selection_tier classifier; tier_b outputs
//           are ADMITTED (full decode + survivor gather) with their economics
//           deferred to RULE 2. K1 range-filter sources are exempt from the
//           tag check and forced tierA.
//   RULE 2 (dynamic, post-CNT), two regimes, both bails memoized scan-wide via
//           scan_filter_status::bailed_high_selectivity:
//           - any tier_b output: proceed iff survivors/rows <=
//             SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL (default 0.10);
//           - otherwise: bail above SIRIUS_EXP_FUSED_SCAN_MAX_SEL (default
//             0.35) unless a tier_dict_k5 output is present (dict-string
//             masked gather wins at all selectivities).
//           Bail = masks dropped, classic decode, batch not ROW_FILTERED.

bool fused_scan_filter_enabled()
{
  // Gate semantics shared with the scan-side extraction: set and not exactly
  // "0" = on (cached; the gate check must stay cheap).
  static bool const enabled = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_FILTER");
    return v != nullptr && std::string_view{v} != "0";
  }();
  return enabled;
}

// RULE 2 threshold: bail out of the fused path after CNT when
// survivors/rows exceeds this (masks dropped, classic decode, batch NOT
// ROW_FILTERED). Measured: wins at sel <= .152, losses by .526; K3 ~ K0 at .5.
double fused_scan_max_selectivity()
{
  static double const threshold = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_MAX_SEL");
    if (v == nullptr || *v == '\0') return 0.35;
    char* end      = nullptr;
    double const d = std::strtod(v, &end);
    return (end != v && d > 0.0) ? d : 0.35;
  }();
  return threshold;
}

// TierB re-admission threshold (iteration 4, track A): when tier_b outputs are
// present the batch proceeds only at sel <= this (default 0.10) — a TierB
// full-decode + gather costs about the classic path, so the win is the
// compacted batch (and, whole-filter case, the skipped post-filter), which
// only pays off at low selectivity (q12 ~.005 is nearly free). Above it the
// batch takes the memoized RULE-2 bail like any other.
double fused_scan_tierb_max_selectivity()
{
  static double const threshold = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL");
    if (v == nullptr || *v == '\0') return 0.10;
    char* end      = nullptr;
    double const d = std::strtod(v, &end);
    return (end != v && d > 0.0) ? d : 0.10;
  }();
  return threshold;
}

// K4 crossover (iteration 4, track B): at survivors/rows <= this (default
// 0.15) the index-list payload decode (W1's K4,
// launch_decode_fused_tree_index_consume) beats the mask-walk K3 for bitpack
// tier_a outputs — microbench crossover sits at 15-50% depending on bits, so
// 0.15 is the conservative edge. Bitpack leaf roots only: delta roots always
// keep K3-delta, dict-k5 is unaffected.
double fused_scan_k4_max_selectivity()
{
  static double const threshold = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL");
    if (v == nullptr || *v == '\0') return 0.15;
    char* end      = nullptr;
    double const d = std::strtod(v, &end);
    return (end != v && d > 0.0) ? d : 0.15;
  }();
  return threshold;
}

// Track B dispatch: W1's K4 launcher + W3's decode_selection.prefer_index_decode
// routing are live — the pick below reaches the decode. Effective kill switch:
// set SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL to a tiny value (parse requires > 0).

// ── Pair-predicate wave-1 entry (iteration 5, stubbed until published) ───────
// W1 is building the K1m2 render variant (ONE kernel over two bitpack regions:
// comparison ballot + optional fused constant ranges per side); the
// PlanTree-binding wrapper belongs in W3's decompress.cpp, same split as the
// shipped K1. SIGNATURE NEEDED (published here + selection.hpp + STATUS-W4 so
// it cannot drift like the boolean adapter almost did):
//
//   bool decompress_column_pair_selection_mask(PlanTree const& tree_a,
//       PlanTree const& tree_b,
//       sirius::codegen::pair_predicate pred,
//       std::uint32_t* mask_words,   // AllocWordsFor(n); write all 32 words/chunk
//       rmm::cuda_stream_view stream,
//       rmm::device_async_resource_ref mr,
//       std::string* error_out);     // false on non-bitpack root / geometry / launch fail
//
// Until it lands, requests carrying pair_filters refuse in the preconditions
// (self-announcing under SIRIUS_EXP_FUSED_SCAN_DIAG); flip this constant and
// forward the stub when W1/W3 publish.
constexpr bool kPairMaskDecodeAvailable = false;

bool fused_decompress_column_pair_selection_mask(PlanTree const& /*tree_a*/,
                                                 PlanTree const& /*tree_b*/,
                                                 sirius::codegen::pair_predicate /*pred*/,
                                                 std::uint32_t* /*mask_words*/,
                                                 rmm::cuda_stream_view /*stream*/,
                                                 rmm::device_async_resource_ref /*mr*/,
                                                 std::string* error_out)
{
  if (error_out) *error_out = "pair (K1m2) mask decode not published yet (W1/W3)";
  return false;
}

// Env-gated fused scan-filter diagnostics: SIRIUS_EXP_FUSED_SCAN_DIAG set (and
// not "0") ⇒ one stderr line per fused-path decision (each precondition/RULE-1
// refusal with its reason, the wave-1 source list, RULE-2 bails, the applied
// summary); unset ⇒ zero output. Permanent triage tooling: an nsys capture
// shows WHICH kernels ran, these lines show WHY a batch took the path it took.
bool fused_scan_diag_enabled()
{
  static bool const enabled = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_DIAG");
    return v != nullptr && std::string_view{v} != "0";
  }();
  return enabled;
}

// Wave-1 K1 + the TierA/filter probe come from W3's published entry points in
// plan/decompress.cpp (plan_interpreter.hpp:148/157): plan_supports_selection_decode
// + decompress_column_selection_mask. Wave 2 calls the published
// decompress_column(..., decode_selection const* sel) (TierA = compact_capable,
// TierB = full decode + in-call gather over survivor_indices).

// RAII CUDA events for the wave-1 -> combine cross-stream join.
struct event_set {
  std::vector<cudaEvent_t> events;

  cudaEvent_t make()
  {
    cudaEvent_t ev{};
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess)
      throw plan_error("fused scan-filter: cudaEventCreate failed");
    events.push_back(ev);
    return ev;
  }

  ~event_set()
  {
    for (auto ev : events)
      cudaEventDestroy(ev);
  }
};

// The fused two-wave decode. Returns std::nullopt in two cases:
//   (a) a precondition fails BEFORE any device work — nothing was issued;
//   (b) anything fails mid-flight — the pool is synchronized, `result` is
//       reset, and the WHOLE batch is retried unfused by the caller (W3's
//       required fallback semantics; TierA/TierB errors from decompress_column
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
  // normal-path decision (the caller runs today's byte-identical decode), not
  // an error; the reason line only appears under SIRIUS_EXP_FUSED_SCAN_DIAG.
  auto refuse = [](char const* why) {
    if (fused_scan_diag_enabled())
      std::fprintf(stderr, "simpatico: fused scan-filter refused: %s\n", why);
    return std::nullopt;
  };
  if (!fused_scan_filter_enabled()) return refuse("env gate off");
  size_t const k_range = request.filters.size();
  size_t const k_pair  = request.pair_filters.size();
  size_t const k_total = k_range + k_pair + request.bool8_filters.size();
  if (k_total == 0) return refuse("no mask directives (no range, pair or bool8 conjuncts)");
  if (k_total > 8) return refuse("more than 8 mask sources (a pair counts once)");
  if (k_pair > 0 && !kPairMaskDecodeAvailable)
    return refuse("pair sources pending W1/W3 (K1m2 launcher + binding wrapper)");
  if (request.tiers.size() != selected.size())
    return refuse("request.tiers not parallel to selected");
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
    // Tier classifier, NOT the umbrella probe: plan_supports_selection_decode
    // is now bitpack OR dict-K5, but a K1 range source must be bitpack-rooted
    // (a dict column arriving as a numeric range source would be a latent
    // wrong-mask gate, even though W2 extracts numeric ranges only today).
    if (plan_selection_tier(*table.columns[selected[f.column]].plan_tree) !=
        sc::output_tier::tier_a)
      return refuse("filter column plan not a bitpack-rooted K1 source");
  }
  for (auto const& p : request.pair_filters) {
    if (p.column_a >= selected.size() || p.column_b >= selected.size())
      return refuse("pair directive column out of range");
    if (p.column_a == p.column_b) return refuse("pair directive compares a column to itself");
    if (p.pred.range_a.lo > p.pred.range_a.hi || p.pred.range_b.lo > p.pred.range_b.hi)
      return refuse("pair directive with an empty fused range");
    // Both sides must be bitpack roots: over the batch's shared num_rows that
    // fixes both to the identical 1024-row chunk geometry (the ONLY mismatch
    // vector is a nested/parented bitpack, which classifies != tier_a). The
    // extraction should have dropped such a conjunct (covers_whole_filter
    // false); if one leaks through, refusing the batch is never wrong.
    if (plan_selection_tier(*table.columns[selected[p.column_a]].plan_tree) !=
          sc::output_tier::tier_a ||
        plan_selection_tier(*table.columns[selected[p.column_b]].plan_tree) !=
          sc::output_tier::tier_a)
      return refuse("pair column not a bitpack root (chunk-geometry mismatch class)");
  }
  for (auto const& b : request.bool8_filters) {
    if (b.column >= selected.size()) return refuse("bool8 directive column out of range");
    if (b.equals_any.empty()) return refuse("bool8 directive with empty equals_any");
    // The BOOL8 source rides the shipped dict-code pushdown: dictionary-rooted
    // plans only (the generic fallback would full-decode + compare — no win).
    if (!plan_supports_predicate_decode(*table.columns[selected[b.column]].plan_tree))
      return refuse("bool8 filter column plan not dictionary-rooted");
  }

  // RULE 1 (static, zero-cost): compacted tags (tier_a / tier_a_delta /
  // tier_dict_k5) must match the plan classifier; tier_b outputs are ADMITTED
  // (iteration 4) and take the wave-2 full-decode + survivor-gather path, with
  // their economics enforced post-CNT (TierB threshold — full-width decode +
  // gather costs ~classic, so only low-sel batches pay off; measured losses at
  // high sel: q1 +43.5%, q5 +6.2%). Range-filter source columns are exempt
  // from the tag check (probed bitpack-rooted above) and forced tier_a; bool8
  // sources get no exemption (dict-rooted — they carry their own tier).
  std::vector<sc::output_tier> tiers(request.tiers.begin(), request.tiers.end());
  for (auto const& f : request.filters)
    tiers[f.column] = sc::output_tier::tier_a;
  for (auto const& p : request.pair_filters) {
    // Pair-source columns are classifier-verified bitpack roots (above) —
    // same exemption as range sources.
    tiers[p.column_a] = sc::output_tier::tier_a;
    tiers[p.column_b] = sc::output_tier::tier_a;
  }
  for (size_t i = 0; i < selected.size(); ++i) {
    switch (tiers[i]) {
      case sc::output_tier::tier_a:
        if (plan_selection_tier(*table.columns[selected[i]].plan_tree) !=
            sc::output_tier::tier_a)
          return refuse("RULE1: tier_a-tagged output not bitpack-rooted (would need tierB)");
        break;
      case sc::output_tier::tier_dict_k5:
        // W3's general dict-K5 route: mask->codes K3 + survivor-only key
        // gather, count-first strings out. Verify against the classifier.
        if (plan_selection_tier(*table.columns[selected[i]].plan_tree) !=
            sc::output_tier::tier_dict_k5)
          return refuse("RULE1: tier_dict_k5-tagged output not dict-K5 decodable");
        break;
      case sc::output_tier::tier_a_delta:
        // W1 delta variant + W3 dispatch live: delta->bitpack roots decode
        // compacted through the same launcher wave 2 already calls.
        if (plan_selection_tier(*table.columns[selected[i]].plan_tree) !=
            sc::output_tier::tier_a_delta)
          return refuse("RULE1: tier_a_delta-tagged output not a delta->bitpack root");
        break;
      case sc::output_tier::tier_str_k6:
        // W3's K6 masked-strings route (str_compact). Until W3's classifier
        // reports tier_str_k6 (it flips together with the umbrella when this
        // enum value is consumed end-to-end), any such tag refuses — safe.
        if (plan_selection_tier(*table.columns[selected[i]].plan_tree) !=
            sc::output_tier::tier_str_k6)
          return refuse("RULE1: tier_str_k6-tagged output not K6-decodable");
        break;
      case sc::output_tier::tier_b:
        // Iteration 4 (track A): tier_b outputs are ADMITTED — they take the
        // wave-2 full-decode + survivor-gather path (mask->indices + event
        // fence). Economics are enforced post-CNT: the batch proceeds only at
        // sel <= SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL, else memoized bail.
        // No probe needed: every plan can full-decode (W3's gather guards
        // null-masked columns with a loud error, never corruption).
        break;
      default: return refuse("RULE1: unknown output tier");
    }
  }

  if (fused_scan_diag_enabled()) {
    static constexpr char const* kOpNames[] = {"<", "<=", ">", ">=", "==", "!="};
    std::string line = "simpatico: fused scan-filter wave-1 sources:";
    for (auto const& f : request.filters) {
      line += " range(col " + std::to_string(f.column) + " [" + std::to_string(f.pred.lo) +
              "," + std::to_string(f.pred.hi) + "])";
    }
    for (auto const& p : request.pair_filters) {
      line += " pair(col " + std::to_string(p.column_a) + " " +
              kOpNames[static_cast<uint8_t>(p.pred.op) < 6 ? static_cast<uint8_t>(p.pred.op) : 0] +
              " col " + std::to_string(p.column_b) + ")";
    }
    for (auto const& b : request.bool8_filters) {
      line +=
        " bool8(col " + std::to_string(b.column) + " eq#" + std::to_string(b.equals_any.size()) +
        ")";
    }
    line += " rows=" + std::to_string(num_rows);
    std::fprintf(stderr, "%s\n", line.c_str());
  }

  // Declared before the try so the mid-flight catch can pool.sync_all() BEFORE
  // these buffers/events unwind (their stream-ordered frees must not race the
  // combine's cross-stream reads).
  std::vector<rmm::device_buffer> per_filter;
  event_set join_events;
  bool rule2_bailed = false;  // distinguishes the RULE-2 bail from real failures

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
    // writes them. Range conjuncts run the K1 decode; dict-code conjuncts run
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
        per_filter.emplace_back(static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t),
                                stream,
                                mr);
        dst = static_cast<std::uint32_t*>(per_filter.back().data());
        mask_ptrs.push_back(dst);
      }
      produce(dst, stream);
      if (s > 0 && stream.value() != s0.value()) {
        // Publish this stream's mask to stream 0 without a host sync.
        cudaEvent_t ev = join_events.make();
        if (cudaEventRecord(ev, stream.value()) != cudaSuccess ||
            cudaStreamWaitEvent(s0.value(), ev, 0) != cudaSuccess) {
          throw plan_error("fused scan-filter: wave-1 stream join failed");
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
          throw plan_error(err.empty() ? "fused scan-filter: K1 mask decode failed" : err);
        }
      });
    }
    for (size_t p = 0; p < k_pair; ++p) {
      submit_mask_source(k_range + p, [&](std::uint32_t* dst, rmm::cuda_stream_view stream) {
        auto const& directive = request.pair_filters[p];
        auto const& col_a     = table.columns[selected[directive.column_a]];
        auto const& col_b     = table.columns[selected[directive.column_b]];
        std::string err;
        if (!fused_decompress_column_pair_selection_mask(
              *col_a.plan_tree, *col_b.plan_tree, directive.pred, dst, stream, mr, &err)) {
          throw plan_error(err.empty() ? "fused scan-filter: K1m2 pair mask decode failed"
                                       : err);
        }
      });
    }
    for (size_t b = 0; b < request.bool8_filters.size(); ++b) {
      submit_mask_source(k_range + k_pair + b, [&](std::uint32_t* dst, rmm::cuda_stream_view stream) {
        auto const& directive = request.bool8_filters[b];
        auto const& col       = table.columns[selected[directive.column]];
        decode_predicate pred;
        pred.equals_any = directive.equals_any;
        std::string err;
        auto flags = decompress_column(*col.plan_tree, stream, mr, &err, &pred);
        if (!flags)
          throw plan_error(err.empty() ? "fused scan-filter: bool8 predicate decode failed"
                                       : err);
        if (flags->type().id() != cudf::type_id::BOOL8 || flags->size() != num_rows)
          throw plan_error("fused scan-filter: bool8 predicate result shape mismatch");
        if (flags->null_count() != 0)
          throw plan_error("fused scan-filter: null-masked bool8 predicate result");
        sc::mask_from_bool8(flags->view().data<std::uint8_t>(), num_rows, dst, stream);
        // `flags` dies here: its stream-ordered free follows the adapter kernel
        // on the same stream, so the read cannot race the release.
      });
    }

    // ── Combine + CNT on stream 0. run_selection_cnt host-syncs s0 once (the
    // survivor count gates wave-2 allocations); after it returns, every wave-1
    // kernel and the combine have completed, so per_filter teardown is safe.
    if (k_total > 1) {
      sc::combine_masks_and(
        combined, mask_ptrs.data(), static_cast<int>(k_total), alloc_words, s0);
    }
    sc::selection_mask sel{
      combined, num_rows, -1, static_cast<std::uint32_t*>(result.chunk_offsets.data())};
    sc::run_selection_cnt(sel, s0, mr);
    result.survivor_count = sel.survivor_count;
    per_filter.clear();

    // RULE 2 (dynamic, post-CNT guard), two regimes:
    //  * tier_b outputs present: proceed only at sel <= TIERB threshold
    //    (default 0.10) — TierB full-decode + gather costs ~classic, so only a
    //    near-empty survivor set pays for the compacted batch.
    //  * no tier_b: the shipped 0.35 write-skip threshold (K3 ~ K0 at sel .5)
    //    covering tier_a / tier_a_delta AND tier_str_k6 (iteration-5 policy:
    //    K6's char-gather savings are dict-like but weak at ~1-char widths —
    //    deliberately NOT exempt until q12/q22 measures say otherwise), with
    //    only a tier_dict_k5 output exempting the batch (F5 wins 2.1-2.6x at
    //    ALL selectivities — string-materialization savings are
    //    survivor-count-independent).
    // Both regimes bail via the mid-flight machinery below with the MEMOIZED
    // rule2_bailed status (one bail latches the whole scan; sync, drop masks,
    // classic decode; batch NOT tagged ROW_FILTERED).
    bool any_dict_k5 = false;
    bool any_tier_b  = false;
    for (auto const t : tiers) {
      any_dict_k5 |= t == sc::output_tier::tier_dict_k5;
      any_tier_b |= t == sc::output_tier::tier_b;
    }
    double const sel_frac =
      static_cast<double>(sel.survivor_count) / static_cast<double>(num_rows);
    bool const bail = any_tier_b
                        ? sel_frac > fused_scan_tierb_max_selectivity()
                        : (sel_frac > fused_scan_max_selectivity() && !any_dict_k5);
    if (bail) {
      double const threshold =
        any_tier_b ? fused_scan_tierb_max_selectivity() : fused_scan_max_selectivity();
      char const* env_name = any_tier_b ? "SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL"
                                        : "SIRIUS_EXP_FUSED_SCAN_MAX_SEL";
      if (fused_scan_diag_enabled()) {
        std::fprintf(stderr,
                     "simpatico: fused scan-filter RULE-2 bail: sel=%.4f > %.4f (%s, "
                     "survivors=%lld/%lld)\n",
                     sel_frac,
                     threshold,
                     env_name,
                     static_cast<long long>(sel.survivor_count),
                     static_cast<long long>(num_rows));
      }
      rule2_bailed = true;
      throw plan_error("selectivity " + std::to_string(sel_frac) + " above " + env_name +
                       " " + std::to_string(threshold));
    }

    // Track B: per-batch payload-variant pick for tier_a bitpack outputs —
    // K4 (index-list decode) below the crossover, K3 (mask walk) above.
    // tier_a_delta always K3-delta (K4 rejects delta roots at render); dict-k5
    // unchanged.
    bool any_tier_a = false;
    for (auto const t : tiers)
      any_tier_a |= t == sc::output_tier::tier_a;
    bool const k4_pick = any_tier_a && sel_frac <= fused_scan_k4_max_selectivity();
    if (fused_scan_diag_enabled()) {
      std::fprintf(stderr,
                   "simpatico: fused scan-filter payload pick: %s (sel=%.4f, k4_max=%.4f)\n",
                   k4_pick ? "k4" : "k3",
                   sel_frac,
                   fused_scan_k4_max_selectivity());
    }

    // ── Survivor index map on stream 0, overlapping wave 2 (column 0's wave-2
    // work serializes behind it on s0; the other streams run free). Built ONCE
    // per batch and SHARED by every consumer: TierB gathers and, once W3's
    // dispatch lands, K4 index-list decodes (same int32 buffer, W1 contract).
    result.tiers = tiers;
    cudf::column_view survivor_indices{cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0};
    if ((any_tier_b || k4_pick) && sel.survivor_count > 0) {
      result.row_indices = rmm::device_buffer(
        static_cast<std::size_t>(sel.survivor_count) * sizeof(std::int32_t), s0, mr);
      sc::mask_to_row_indices(sel, static_cast<std::int32_t*>(result.row_indices.data()), s0);
      survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(sel.survivor_count),
                                           result.row_indices.data(),
                                           nullptr,
                                           0};
      // Index consumers (TierB gathers, K4 decodes) run on the other pool
      // streams; order them after the
      // indices kernel on s0 with a device-side wait (streams are FIFO, so one
      // up-front wait per stream covers every wave-2 launch on it).
      cudaEvent_t ev_idx = join_events.make();
      if (cudaEventRecord(ev_idx, s0.value()) != cudaSuccess)
        throw plan_error("fused scan-filter: indices event record failed");
      for (size_t si = 1; si < n_streams; ++si) {
        if (cudaStreamWaitEvent(pool.streams[si], ev_idx, 0) != cudaSuccess)
          throw plan_error("fused scan-filter: indices stream wait failed");
      }
    }

    // ── Wave 2: TierA compacted (compact_capable) + TierB full-decode+gather,
    // round-robin on the pool streams via the published
    // decompress_column(..., decode_selection const*) contract. Everything
    // wave 2 consumes (mask, chunk_offsets) completed before the CNT host sync
    // above, so no cross-stream waits are needed.
    std::vector<std::unique_ptr<cudf::column>> cols(selected.size());
    run_column_workers(selected.size(), pool, [&](size_t i, rmm::cuda_stream_view stream) {
      auto const& col = table.columns[selected[i]];
      decode_selection dsel;
      dsel.mask             = &sel;
      dsel.survivor_count   = sel.survivor_count;
      dsel.survivor_indices = survivor_indices;
      // Route per tier: compact_capable = in-kernel mask consumption (bitpack
      // K3 and delta-root mask_consume); dict_compact = the dict-K5 arm
      // (mask->codes + survivor-only key gather). Mutually exclusive by
      // decode_selection contract; tier_b keeps both false (gather path).
      dsel.compact_capable = result.tiers[i] == sc::output_tier::tier_a ||
                             result.tiers[i] == sc::output_tier::tier_a_delta;
      dsel.dict_compact    = result.tiers[i] == sc::output_tier::tier_dict_k5;
      dsel.str_compact     = result.tiers[i] == sc::output_tier::tier_str_k6;
      // Track B: below the K4 crossover, tier_a bitpack roots take W1's
      // index-list decode over survivor_indices (populated above whenever
      // k4_pick, per the decode_selection contract); W3's dispatch silently
      // keeps K3 on any anomaly, delta roots ignore it, dict-k5 unchanged.
      dsel.prefer_index_decode = k4_pick && result.tiers[i] == sc::output_tier::tier_a &&
                                 sel.survivor_count > 0;
      std::string err;
      auto out = decompress_column(*col.plan_tree, stream, mr, &err, nullptr, &dsel);
      if (!out) throw plan_error(err.empty() ? "fused scan-filter: decompress failed" : err);
      cols[i] = apply_stored_dtype(std::move(out), col.dtype);
    });
    // run_column_workers ended with pool.sync_all(), which also covers the
    // mask_to_row_indices launch on s0.

    result.applied = true;
    result.status  = sc::scan_filter_status::applied;
    return cols;
  } catch (std::exception const& e) {
    std::fprintf(stderr,
                 "simpatico: fused scan-filter fell back to the classic decode (%s)\n",
                 e.what());
    pool.sync_all();  // quiesce in-flight wave kernels before buffers unwind
    result        = sirius::codegen::scan_filter_result{};
    // Distinguishable outcome for the scan side's bail memoization: one
    // high-selectivity bail predicts the scan's remaining batches.
    result.status = rule2_bailed ? sc::scan_filter_status::bailed_high_selectivity
                                 : sc::scan_filter_status::failed;
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

bool column_supports_predicate_decode(const compressed_table& table, std::size_t column_index)
{
  if (column_index >= table.columns.size()) { return false; }
  auto const& tree = table.columns[column_index].plan_tree;
  return tree && plan_supports_predicate_decode(*tree);
}

std::vector<std::unique_ptr<cudf::column>> decompress_scan_filter(
  const compressed_table& table,
  std::span<const std::size_t> selected_columns,
  sirius::codegen::scan_filter_request const& request,
  sirius::codegen::scan_filter_result& result,
  simpatico::stream_pool& pool,
  rmm::device_async_resource_ref mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::decompress_table[scan_filter,pool]"};
  result = sirius::codegen::scan_filter_result{};
  if (auto cols = try_decompress_fused(table, selected_columns, request, result, pool, mr)) {
    if (fused_scan_diag_enabled()) {
      int n_a = 0, n_delta = 0, n_k5 = 0, n_k6 = 0, n_b = 0;
      for (auto const t : result.tiers) {
        n_a += t == sirius::codegen::output_tier::tier_a;
        n_delta += t == sirius::codegen::output_tier::tier_a_delta;
        n_k5 += t == sirius::codegen::output_tier::tier_dict_k5;
        n_k6 += t == sirius::codegen::output_tier::tier_str_k6;
        n_b += t == sirius::codegen::output_tier::tier_b;
      }
      std::fprintf(stderr,
                   "simpatico: fused scan-filter applied: survivors=%lld/%lld "
                   "tiers a=%d delta=%d k5=%d k6=%d b=%d sources=%zu\n",
                   static_cast<long long>(result.survivor_count),
                   static_cast<long long>(result.num_rows),
                   n_a,
                   n_delta,
                   n_k5,
                   n_k6,
                   n_b,
                   request.filters.size() + request.pair_filters.size() +
                     request.bool8_filters.size());
    }
    return std::move(*cols);
  }
  // Gate off / no directives / policy refusal / mid-flight fallback: exactly
  // today's path — with one obligation: when the request routed dict-code
  // conjuncts (bool8_filters) into the mask, the fallback must NOT be a plain
  // decode, or every refused/failed/bailed batch would silently lose the
  // shipped dict BOOL8-substitution win (q19 -21.4%). Re-express them as
  // classic decode_predicates so the fallback degrades to today's
  // substitution behavior, never below it.
  if (!request.bool8_filters.empty()) {
    std::vector<decode_predicate> predicates(selected_columns.size());
    for (auto const& b : request.bool8_filters) {
      if (b.column < predicates.size()) predicates[b.column].equals_any = b.equals_any;
    }
    return decompress_columns_parallel(table, selected_columns, predicates, pool, mr)
      ->release();
  }
  return decompress_columns_parallel(table, selected_columns, pool, mr)->release();
}

}  // namespace simpatico
