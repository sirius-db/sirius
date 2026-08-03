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

bool fused_scan_filter_enabled()
{
  static bool const enabled = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_FILTER");
    return v != nullptr && std::string_view{v} == "1";
  }();
  return enabled;
}

// ── Wave-1 per-column entry point required from W3 (stubbed until published) ─
// W1 published the FusedTree-level K1 launcher (launch_decode_fused_tree_mask_out,
// codegen/decode/masked_launch.hpp), but the PlanTree -> FusedTree/LabeledBuffers
// binding lives in plan/decompress.cpp (W3). Needed there (tracked in
// scratchpad/fusebench/STATUS-W4.md):
//
//   // Wave 1 / K1: decode the root bitpack leaf fused with the inclusive
//   // decoded-domain range -> mask.words; NO column output. Must write all 32
//   // words of every 1024-row chunk (out-of-range lanes ballot to 0, so tail
//   // bits are zero); mask.words holds AllocWordsFor(num_rows) words.
//   bool decompress_column_selection_mask(PlanTree const& tree,
//                                         sirius::codegen::range_predicate pred,
//                                         sirius::codegen::selection_mask& mask,
//                                         rmm::cuda_stream_view stream,
//                                         rmm::device_async_resource_ref mr,
//                                         std::string* error_out);
//
// Wave 2 needs no stub: it calls the published decompress_column(...,
// decode_selection const* sel) (plan_interpreter.hpp; TierA = compact_capable,
// TierB = full decode + in-call gather over survivor_indices).

// Flip when W3's per-column K1 wrapper lands; false keeps the fused path
// rejected in the preconditions (zero device work, no per-batch log spam).
constexpr bool kSelectionMaskDecodeAvailable = false;

bool fused_decompress_column_selection_mask(PlanTree const& /*tree*/,
                                            sirius::codegen::range_predicate /*pred*/,
                                            sirius::codegen::selection_mask& /*mask*/,
                                            rmm::cuda_stream_view /*stream*/,
                                            rmm::device_async_resource_ref /*mr*/,
                                            std::string* error_out)
{
  if (error_out) *error_out = "K1 selection-mask decode not published yet (W3 wrapper over W1)";
  return false;
}

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

  // Preconditions — all checked before touching the device.
  if (!fused_scan_filter_enabled()) return std::nullopt;
  if (!kSelectionMaskDecodeAvailable) return std::nullopt;  // W3 K1 wrapper pending
  if (request.filters.empty()) return std::nullopt;
  if (request.tiers.size() != selected.size()) return std::nullopt;
  if (pool.streams.empty()) return std::nullopt;
  int64_t const num_rows = table.num_rows();
  if (num_rows <= 0) return std::nullopt;
  if (num_rows > std::numeric_limits<std::int32_t>::max()) return std::nullopt;
  if (request.filters.size() > 8) return std::nullopt;  // combine_masks_and limit
  for (auto const idx : selected) {
    if (idx >= table.columns.size()) return std::nullopt;
    auto const& col = table.columns[idx];
    if (!col.plan_tree || col.num_rows != num_rows) return std::nullopt;
  }
  for (auto const& f : request.filters) {
    if (f.column >= selected.size()) return std::nullopt;
    if (f.pred.lo > f.pred.hi) return std::nullopt;  // empty range: let the engine filter
  }

  // Declared before the try so the mid-flight catch can pool.sync_all() BEFORE
  // these buffers/events unwind (their stream-ordered frees must not race the
  // combine's cross-stream reads).
  std::vector<rmm::device_buffer> per_filter;
  event_set join_events;

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

    // ── Wave 1: K1 mask decodes, filter 0 straight into the combined buffer on
    // stream 0 (its allocation stream), filters 1..k-1 into per-filter buffers
    // allocated on the stream that writes them.
    size_t const k = request.filters.size();
    per_filter.reserve(k > 1 ? k - 1 : 0);
    std::vector<std::uint32_t const*> mask_ptrs;
    mask_ptrs.reserve(k);
    mask_ptrs.push_back(combined);

    for (size_t f = 0; f < k; ++f) {
      rmm::cuda_stream_view stream =
        (f == 0) ? s0 : rmm::cuda_stream_view{pool.streams[f % n_streams]};
      std::uint32_t* dst = combined;
      if (f > 0) {
        per_filter.emplace_back(static_cast<std::size_t>(alloc_words) * sizeof(std::uint32_t),
                                stream,
                                mr);
        dst = static_cast<std::uint32_t*>(per_filter.back().data());
        mask_ptrs.push_back(dst);
      }
      auto const& directive = request.filters[f];
      auto const& col       = table.columns[selected[directive.column]];
      sc::selection_mask fmask{dst, num_rows, -1, nullptr};
      std::string err;
      if (!fused_decompress_column_selection_mask(
            *col.plan_tree, directive.pred, fmask, stream, mr, &err)) {
        throw plan_error(err.empty() ? "fused scan-filter: K1 mask decode failed" : err);
      }
      if (f > 0 && stream.value() != s0.value()) {
        // Publish this stream's mask to stream 0 without a host sync.
        cudaEvent_t ev = join_events.make();
        if (cudaEventRecord(ev, stream.value()) != cudaSuccess ||
            cudaStreamWaitEvent(s0.value(), ev, 0) != cudaSuccess) {
          throw plan_error("fused scan-filter: wave-1 stream join failed");
        }
      }
    }

    // ── Combine + CNT on stream 0. run_selection_cnt host-syncs s0 once (the
    // survivor count gates wave-2 allocations); after it returns, every wave-1
    // kernel and the combine have completed, so per_filter teardown is safe.
    if (k > 1) {
      sc::combine_masks_and(combined, mask_ptrs.data(), static_cast<int>(k), alloc_words, s0);
    }
    sc::selection_mask sel{
      combined, num_rows, -1, static_cast<std::uint32_t*>(result.chunk_offsets.data())};
    sc::run_selection_cnt(sel, s0, mr);
    result.survivor_count = sel.survivor_count;
    per_filter.clear();

    // ── TierB gather map on stream 0, overlapping wave 2 (column 0's wave-2
    // work serializes behind it on s0; the other streams run free). Built ONCE
    // per batch; the same view is handed to every TierB column (W3 contract).
    result.tiers.assign(request.tiers.begin(), request.tiers.end());
    bool any_tier_b = false;
    for (auto const t : result.tiers)
      any_tier_b |= t == sc::output_tier::tier_b;
    cudf::column_view survivor_indices{cudf::data_type{cudf::type_id::INT32}, 0, nullptr, nullptr, 0};
    if (any_tier_b && sel.survivor_count > 0) {
      result.row_indices = rmm::device_buffer(
        static_cast<std::size_t>(sel.survivor_count) * sizeof(std::int32_t), s0, mr);
      sc::mask_to_row_indices(sel, static_cast<std::int32_t*>(result.row_indices.data()), s0);
      survivor_indices = cudf::column_view{cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(sel.survivor_count),
                                           result.row_indices.data(),
                                           nullptr,
                                           0};
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
      dsel.compact_capable  = result.tiers[i] == sc::output_tier::tier_a;
      std::string err;
      auto out = decompress_column(*col.plan_tree, stream, mr, &err, nullptr, &dsel);
      if (!out) throw plan_error(err.empty() ? "fused scan-filter: decompress failed" : err);
      cols[i] = apply_stored_dtype(std::move(out), col.dtype);
    });
    // run_column_workers ended with pool.sync_all(), which also covers the
    // mask_to_row_indices launch on s0.

    result.applied = true;
    return cols;
  } catch (std::exception const& e) {
    std::fprintf(stderr,
                 "simpatico: fused scan-filter failed (%s); retrying the batch unfused\n",
                 e.what());
    pool.sync_all();  // quiesce in-flight wave kernels before buffers unwind
    result = sirius::codegen::scan_filter_result{};
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
    return std::move(*cols);
  }
  // Gate off / no directives / precondition failed: exactly today's path.
  return decompress_columns_parallel(table, selected_columns, pool, mr)->release();
}

}  // namespace simpatico
