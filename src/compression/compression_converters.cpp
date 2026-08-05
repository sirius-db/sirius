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
#include "device_compressed_blob.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/selection/selection.hpp>
#include <codegen/selection/selection_capture.hpp>
#include <codegen/util/stream_pool.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>
#include <op/scan/row_filtered_table_representation.hpp>
#include <op/scan/selection_captured_table_representation.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius {

namespace {

// Deterministic decision tracing for the fused scan-filter pipeline
// (permanent tooling, not temp instrumentation): harness runs drop DEBUG
// lines at the duckdb sink, so SIRIUS_EXP_FUSED_SCAN_DIAG (set and not "0")
// promotes every [fused-diag] line to INFO — proven to reach the sirius log
// file. Unset: the trace stays at DEBUG (silent by default). Cached like the
// engine gate; same "set and not exactly 0" semantics.
bool fused_scan_diag_enabled()
{
  static bool const enabled = [] {
    char const* v = std::getenv("SIRIUS_EXP_FUSED_SCAN_DIAG");
    return v != nullptr && std::string_view{v} != "0";
  }();
  return enabled;
}

}  // namespace

// Routes one [fused-diag] line to INFO when the diag env is set, DEBUG
// otherwise. A macro (not a function) so the level dispatch keeps the
// call-site file/line and the lazy formatting of the underlying macros.
#define SIRIUS_FUSED_DIAG(...)                        \
  do {                                                \
    if (::sirius::fused_scan_diag_enabled()) {        \
      SIRIUS_LOG_INFO(__VA_ARGS__);                   \
    } else {                                          \
      SIRIUS_LOG_DEBUG(__VA_ARGS__);                  \
    }                                                 \
  } while (0)

namespace {

// Thread-local pool of 4 CUDA streams for cross-column decode parallelism.
// Work is submitted from the calling thread so cuCascade memory-reservation
// tracking (attached to the calling thread) sees all allocations.
// 4 is not a configuration parameter — it matches the typical SM occupancy
// sweet spot for column-parallel decode without thread-spawn overhead.
simpatico::stream_pool& decode_pool()
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

// The sirius-side tier mirror of sirius::codegen::output_tier (this header
// stays simpatico-free for the test TUs; the .cpp maps 1:1).
decode_output_tier to_local_tier(sirius::codegen::output_tier t)
{
  switch (t) {
    case sirius::codegen::output_tier::tier_a: return decode_output_tier::tier_a;
    case sirius::codegen::output_tier::tier_a_delta: return decode_output_tier::tier_a_delta;
    case sirius::codegen::output_tier::tier_dict_k5: return decode_output_tier::tier_dict_k5;
    case sirius::codegen::output_tier::tier_str_k6: return decode_output_tier::tier_str_k6;
    default: return decode_output_tier::tier_b;
  }
}

sirius::codegen::output_tier to_shared_tier(decode_output_tier t)
{
  switch (t) {
    case decode_output_tier::tier_a: return sirius::codegen::output_tier::tier_a;
    case decode_output_tier::tier_a_delta: return sirius::codegen::output_tier::tier_a_delta;
    case decode_output_tier::tier_dict_k5: return sirius::codegen::output_tier::tier_dict_k5;
    case decode_output_tier::tier_str_k6: return sirius::codegen::output_tier::tier_str_k6;
    default: return sirius::codegen::output_tier::tier_b;
  }
}

// Per-column probe for the fused scan-filter pipeline.
//
// `tier` defers to simpatico::plan_selection_tier — the ground-truth
// classifier next to the decode implementation, consistent by
// construction with the umbrella plan_supports_selection_decode probe that
// the RULE-1 gate re-checks. Keeping a single classifier means a new masked
// variant lights its tier up everywhere at once, with no shape-walk here to
// drift.
//
// `lane_ok`: the dtype decodes into a lane K1 can compare as signed int64 —
// uint64 is excluded because its upper half would misorder under a signed
// compare. Only meaningful for RANGE participation (wave-1 mask source),
// which additionally requires tier_a: K1 renders bitpack-leaf roots only
// (plan_interpreter.hpp's CAUTION on wave-1 callers).
struct fused_column_probe {
  bool lane_ok         = false;
  bool compact_capable = false;
  decode_output_tier tier = decode_output_tier::tier_b;

  [[nodiscard]] bool range_source_ok() const noexcept
  {
    return lane_ok && compact_capable && tier == decode_output_tier::tier_a;
  }
};

fused_column_probe probe_fused_column(simpatico::compressed_table const& table,
                                      std::size_t column_index)
{
  fused_column_probe probe;
  if (column_index >= table.columns.size()) { return probe; }
  auto const& column = table.columns[column_index];
  switch (column.dtype.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64: probe.lane_ok = true; break;
    default: break;
  }
  auto const* tree = column.plan_tree.get();
  if (tree == nullptr) { return probe; }
  probe.tier            = to_local_tier(simpatico::plan_selection_tier(*tree));
  probe.compact_capable = simpatico::plan_supports_selection_decode(*tree);
  return probe;
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

// Which representation class the converter must construct for a fused-attempt
// batch — the scan side keys its behavior off the class alone.
enum class fused_batch_tag : std::uint8_t {
  none,          ///< plain gpu_table_representation (classic or partial-mask output)
  row_filtered,  ///< whole conjunction applied ⇒ row_filtered_gpu_table_representation
  rule2_bailed,  ///< RULE-2 selectivity bail ⇒ rule2_bailed_gpu_table_representation,
                 ///< letting the scan latch and strip the pushdown for later batches
};

// Attempt the fused scan-filter decompress (SIRIUS_EXP_FUSED_SCAN_FILTER):
// wave-1 K1 masks on range columns + K1m2 pairs + dict-code BOOL8 masks +
// dynamic-membership probes (Phase A), wave-2 decode against the combined
// mask, output assembled to one uniformly survivor-sized table. Returns
// nullptr when the attempt is declined here (directives not buildable, shape
// checks) or the assembly refuses — the caller then runs the classic path
// (predicated when an equality pushdown exists). `tag` reports how to wrap a
// non-null return; `none` with a non-null return means the engine fell back
// internally and the table is the classic decode — with routed bool8_filters
// that fallback is PREDICATED (BOOL8 substitution columns at those slots, the
// dict win survives every outcome; decompress_scan_filter contract).
//
// DUAL DELIVERY (bool8 sources, W4 rev 18): a bool8 column's slot carries the
// wave-1 BOOL8 gathered to survivors (compacted BOOL8, never K5 values), so
// the scan's type inspection takes the substitution branch and the residual
// re-eval is a bare boolean AND. Membership sources are dynamic — the batch
// stays untagged regardless (the authoritative join still runs), and the
// post-scan DYNAMIC_FILTER operator's gate self-disables once it measures
// keep ≈ 1 on the already-masked stream.
std::unique_ptr<cudf::table> try_decompress_scan_filter(
  simpatico::compressed_table const& ct,
  std::span<const std::size_t> selected,
  decode_range_pushdown const& attached_ranges,
  bool all_conjuncts_convertible,
  decode_equality_pushdown const& equality_pushdown,
  decode_pair_pushdown const& attached_pairs,
  decode_membership_pushdown const& membership_pushdown,
  std::uint64_t membership_generation,
  simpatico::stream_pool& pool,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  fused_batch_tag& tag,
  bool capture_selection,
  std::shared_ptr<const late_mat::row_selection>& captured_out)
{
  // bool8 mask candidates = the shipped equality pushdown's non-empty entries
  // (attach already gates them per chunk on column_supports_predicate_decode,
  // i.e. dictionary-rooted pure-filter columns; W4 re-probes).
  bool has_bool8_candidates = false;
  for (std::size_t i = 0; i < equality_pushdown.size() && i < selected.size(); ++i) {
    if (!equality_pushdown[i].empty()) {
      has_bool8_candidates = true;
      break;
    }
  }
  // Membership sources (Phase A): dynamic-filter probes, one source per probe.
  std::size_t membership_sources = 0;
  for (std::size_t i = 0; i < membership_pushdown.size() && i < selected.size(); ++i) {
    membership_sources += membership_pushdown[i].probes.size();
  }
  auto const directives = build_fused_scan_directives(ct,
                                                      selected,
                                                      attached_ranges,
                                                      all_conjuncts_convertible,
                                                      has_bool8_candidates,
                                                      attached_pairs,
                                                      membership_sources > 0);
  if (!directives.enabled) { return nullptr; }

  // bool8 routing (dual delivery, W4 rev 18): the wave-1 BOOL8 is retained and
  // gathered to the column's slot (compacted BOOL8, RULE-1-bypassed, write-skip
  // cheap), so the residual re-eval is a bare boolean AND — bool8-ONLY masks
  // are now sound AND profitable (the iteration-4 +5.8% came from the old
  // single-delivery K5 string re-compare; that path is gone). No decline
  // remains beyond the shape checks: a REFUSED request with routed
  // bool8_filters still reruns classic PREDICATED (substitution floor, W4
  // rev 14 contract).
  std::size_t range_sources = 0;
  for (auto const& e : directives.ranges) {
    if (e.participates_in_scan_mask) { ++range_sources; }
  }
  std::size_t bool8_sources = 0;
  for (std::size_t i = 0; i < equality_pushdown.size() && i < selected.size(); ++i) {
    if (!equality_pushdown[i].empty()) { ++bool8_sources; }
  }
  // Shape checks: structurally impossible requests skip the engine round-trip.
  // Every source counts once (a pair once, each membership probe once).
  auto const total_sources =
    range_sources + directives.pairs.size() + bool8_sources + membership_sources;
  if (total_sources > 8 || ct.num_rows() > std::numeric_limits<std::int32_t>::max()) {
    SIRIUS_FUSED_DIAG(
      "[fused-diag] fused routing declined on shape ({} mask sources, {} rows) — classic path",
      total_sources,
      ct.num_rows());
    return nullptr;
  }
  bool const route_bool8 = has_bool8_candidates;

  // Build the wave request directly (not via make_scan_filter_request, whose
  // column_decode_directive collapses tiers to a boolean): W4's wave-2
  // dispatch and RULE 2's dict-K5 exemption key off the TRUE tier, so the
  // delta/dict unlocks only work when the request carries them.
  sirius::codegen::scan_filter_request request;
  request.tiers.reserve(selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    auto const& entry = directives.ranges[i];
    if (entry.active && entry.participates_in_scan_mask) {
      request.filters.push_back({i, {entry.lo, entry.hi}});
    }
    request.tiers.push_back(directives.compact_capable[i] != 0
                              ? to_shared_tier(directives.output_tiers[i])
                              : sirius::codegen::output_tier::tier_b);
  }
  // Pair sources: one K1m2 kernel per pair, with each side's constant range
  // folded in (an inactive side stays full-domain). The builder already
  // cleared those sides' standalone K1 participation.
  for (auto const& pair : directives.pairs) {
    sirius::codegen::pair_predicate pred;
    pred.op          = static_cast<sirius::codegen::pair_compare_op>(pair.op);
    auto const& ra   = directives.ranges[pair.column_a];
    auto const& rb   = directives.ranges[pair.column_b];
    if (ra.active) { pred.range_a = {ra.lo, ra.hi}; }
    if (rb.active) { pred.range_b = {rb.lo, rb.hi}; }
    request.pair_filters.push_back({pair.column_a, pair.column_b, pred});
  }
  if (route_bool8) {
    for (std::size_t i = 0; i < equality_pushdown.size() && i < selected.size(); ++i) {
      if (equality_pushdown[i].empty()) { continue; }
      request.bool8_filters.push_back({i, equality_pushdown[i]});
    }
  }
  // Membership sources (Phase A): the attach snapshotted the dynamic filter
  // set per batch; each probe closure co-owns its published device replica, so
  // copying the std::function here keeps the filter pinned for the call.
  // GLOBAL order = ascending expected keep-rate (kind_rank, then num_keys,
  // ties in channel order): the engine's membership cap keeps a PREFIX, so
  // the strong filters must come first (q21: the suppkey in_list must beat
  // the orders Bloom to the cap).
  {
    struct ordered_probe {
      std::size_t column;
      decode_membership_probe const* probe;
    };
    std::vector<ordered_probe> ordered;
    for (std::size_t i = 0; i < membership_pushdown.size() && i < selected.size(); ++i) {
      for (auto const& probe : membership_pushdown[i].probes) {
        ordered.push_back({i, &probe});
      }
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](auto const& a, auto const& b) {
      if (a.probe->kind_rank != b.probe->kind_rank) {
        return a.probe->kind_rank < b.probe->kind_rank;
      }
      // 0 = unknown key count: sort after known counts within the same kind.
      auto const a_keys = a.probe->num_keys == 0 ? std::numeric_limits<std::uint64_t>::max()
                                                 : a.probe->num_keys;
      auto const b_keys = b.probe->num_keys == 0 ? std::numeric_limits<std::uint64_t>::max()
                                                 : b.probe->num_keys;
      return a_keys < b_keys;
    });
    std::string order_echo;
    for (auto const& e : ordered) {
      request.membership_filters.push_back({e.column, e.probe->probe});
      order_echo += order_echo.empty() ? "" : ",";
      order_echo += std::to_string(e.probe->kind_rank) + ":c" + std::to_string(e.column);
    }
    if (!ordered.empty()) {
      SIRIUS_FUSED_DIAG("[fused-diag] membership cap order (ascending expected keep): [{}]",
                        order_echo);
    }
  }
  // Dynamic-filter-set version behind those probes (0 = static-only); echoed
  // on the result so the scan-side bail latch can clear when a later, tighter
  // set arrives (transitive targets).
  request.source_generation = membership_generation;

  sirius::codegen::scan_filter_result result;
  auto decoded = simpatico::decompress_scan_filter(ct, selected, request, result, pool, mr);
  std::string error;
  auto table =
    simpatico::compact_scan_filter_output(std::move(decoded), result, stream, mr, &error);
  // compact_scan_filter_output synchronized `stream`; re-point the selection
  // buffers there anyway so their teardown follows the batch's ordering.
  result.set_stream(stream);
  if (!table) {
    SIRIUS_FUSED_DIAG("[fused-diag] assembly REFUSED ({}); falling back to classic decompress",
                    error);
    return nullptr;
  }
  // row_filtered only when the mask carried EVERY restricting conjunct: a
  // partial mask must leave the batch untagged so post_filter_and_project
  // evaluates the residual (re-checking already-applied conjuncts on the
  // compacted rows is idempotent). A RULE-2 bail gets its own tag class —
  // classic full-width columns inside, but the scan latches on the type and
  // strips the range pushdown from the operator's remaining batches (bail
  // memoization: per-batch selectivity is uniform across a scan's batches).
  // refused/failed stay plain.
  if (result.status == sirius::codegen::scan_filter_status::bailed_high_selectivity) {
    tag = fused_batch_tag::rule2_bailed;
  } else if (result.applied && directives.covers_whole_filter) {
    tag = fused_batch_tag::row_filtered;
  }
  // Late-mat wave-seam capture (SIRIUS_EXP_LATE_MAT; gated on status ==
  // applied, NOT on the row_filtered tag): when the source representation
  // requested capture AND status == applied — the converter's promise that
  // EVERY emitted column is compacted to exactly survivor_count rows; the
  // conversion has no row-shaping step after compact_scan_filter_output, and
  // residual bool8/membership re-evaluation happens scan-side POST-emission
  // (it thins the batch as ordinary data, so the captured selection still
  // describes the rows as emitted here) — MOVE the wave-1 selection buffers
  // out of `result` instead of letting them die function-local. Capture
  // therefore also covers the membership-compacted and partial-coverage
  // batches that stay UNTAGGED by design (their tag semantics are unchanged
  // — the caller wraps them in the
  // metadata-only selection_captured carrier, never promotes them to
  // row_filtered). A RULE-2 bail / refusal / failure can never capture: their
  // status is never `applied` (the helper also re-checks). `range` is left
  // zeroed — the scan side fills it from the split's origin at harvest. The
  // helper set_stream-rebinds each moved buffer and leaves result's scalar
  // fields intact (the DIAG line below stays valid).
  if (capture_selection &&
      result.status == sirius::codegen::scan_filter_status::applied) {
    auto cap = sirius::codegen::capture_scan_filter_selection(std::move(result), stream);
    if (cap) {
      auto sel            = std::make_shared<late_mat::row_selection>();
      sel->kind           = late_mat::row_selection_kind::mask;
      sel->mask_words     = std::move(cap.mask_words);
      sel->chunk_offsets  = std::move(cap.chunk_offsets);
      sel->survivor_count = cap.survivor_count;
      captured_out        = std::move(sel);
    }
  }
  SIRIUS_FUSED_DIAG(
    "[fused-diag] decompress_scan_filter {} (status={} gen={}): range_filters={} pair_filters={} "
    "bool8_filters={} membership_filters={} survivors={}/{} column(s)={} covers_whole_filter={} "
    "tag={}",
    result.applied ? "APPLIED" : "NOT applied (classic output)",
    static_cast<int>(result.status),
    result.source_generation,
    request.filters.size(),
    request.pair_filters.size(),
    request.bool8_filters.size(),
    request.membership_filters.size(),
    result.survivor_count,
    result.num_rows,
    table->num_columns(),
    directives.covers_whole_filter,
    static_cast<int>(tag));
  return table;
}

// Reconstruct + project + decompress a compressed_table into a GPU table
// representation. Shared by the host and device compression converters — only
// the byte transport (how `fetch` pulls the payload) differs between them.
std::unique_ptr<cucascade::idata_representation> reconstruct_and_decompress_to_gpu(
  std::span<const std::uint8_t> header,
  simpatico::payload_fetch_fn const& fetch,
  const std::optional<std::vector<std::size_t>>& selected_indices,
  decode_equality_pushdown const& equality_pushdown,
  decode_range_pushdown const& range_pushdown,
  bool range_conjuncts_convertible,
  decode_membership_pushdown const& membership_pushdown,
  std::uint64_t membership_generation,
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream,
  bool capture_selection)
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
  fused_batch_tag tag = fused_batch_tag::none;
  // Fused scan-filter attempt — same contract as the device converter:
  // equality pushdowns route into the mask when admissible (substitution
  // dropped), decline ⇒ classic (predicated) path below.
  SIRIUS_FUSED_DIAG(
    "[fused-diag] host converter: n_columns={} equality_predicates={} range_entries={} "
    "range_gate={}",
    subset.num_columns(),
    predicates.size(),
    range_pushdown.size(),
    range_conjuncts_convertible);
  std::shared_ptr<const late_mat::row_selection> captured;
  if (!range_pushdown.empty() || !predicates.empty() || !membership_pushdown.empty()) {
    // `subset` holds exactly the projected columns, so selection is identity.
    std::vector<std::size_t> identity_selection(subset.num_columns());
    std::iota(identity_selection.begin(), identity_selection.end(), std::size_t{0});
    decompressed = try_decompress_scan_filter(subset,
                                              identity_selection,
                                              range_pushdown,
                                              range_conjuncts_convertible,
                                              equality_pushdown,
                                              decode_pair_pushdown{},  // pair carrier not yet threaded
                                              membership_pushdown,
                                              membership_generation,
                                              pool,
                                              stream,
                                              mr,
                                              tag,
                                              capture_selection,
                                              captured);
  }
  if (!decompressed) {
    if (predicates.empty()) {
      decompressed = simpatico::decompress(subset, pool, mr);
    } else {
      std::vector<std::size_t> all(subset.num_columns());
      std::iota(all.begin(), all.end(), std::size_t{0});
      decompressed = simpatico::decompress(subset, all, predicates, pool, mr);
    }
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

  // Tagged subclasses (see row_filtered_table_representation.hpp):
  // row_filtered ⇔ the fused pipeline applied the whole conjunction;
  // rule2_bailed ⇔ RULE-2 selectivity bail (classic columns inside — the scan
  // latches on the type and strips the pushdown for its remaining batches).
  if (tag == fused_batch_tag::row_filtered) {
    auto tagged = std::make_unique<row_filtered_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
    tagged->captured_selection = std::move(captured);  // empty unless requested+applied
    return tagged;
  }
  if (tag == fused_batch_tag::rule2_bailed) {
    return std::make_unique<rule2_bailed_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
  }
  if (captured) {
    // Untagged but applied (membership-compacted / partial-coverage batch):
    // metadata-only carrier — downstream filter semantics stay byte-identical
    // to a plain representation (the carrier adds a captured selection and
    // nothing else: no tag, no row shaping).
    auto neutral = std::make_unique<selection_captured_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
    neutral->captured_selection = std::move(captured);
    return neutral;
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

  return reconstruct_and_decompress_to_gpu(rep.header(),
                                           fetch,
                                           rep.selected_indices(),
                                           rep.equality_pushdown(),
                                           rep.range_pushdown(),
                                           rep.range_conjuncts_convertible(),
                                           rep.membership_pushdown(),
                                           rep.membership_generation(),
                                           source,
                                           target_memory_space,
                                           stream,
                                           rep.selection_capture_requested());
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
  auto& pool          = decode_pool();

  // Projected column count — what the pushdown is indexed by.
  auto const n_selected =
    indices.has_value() ? indices->size() : static_cast<std::size_t>(ct.num_columns());
  auto const predicates = to_decode_predicates(rep.equality_pushdown(), n_selected);

  std::unique_ptr<cudf::table> decompressed;
  fused_batch_tag tag = fused_batch_tag::none;
  // Fused scan-filter attempt (SIRIUS_EXP_FUSED_SCAN_FILTER; the env gate and
  // every per-plan precondition are re-checked inside decompress_scan_filter).
  // Iteration 4: string-equality pushdowns are composable — try_ routes them
  // into the mask (bool8_filters, substitution dropped) when the request is
  // admissible, and declines otherwise so the predicated overload below keeps
  // the shipped BOOL8-substitution behavior.
  SIRIUS_FUSED_DIAG(
    "[fused-diag] device converter: n_selected={} equality_predicates={} range_entries={} "
    "range_gate={}",
    n_selected,
    predicates.size(),
    rep.range_pushdown().size(),
    rep.range_conjuncts_convertible());
  std::vector<std::size_t> identity_selection;
  std::shared_ptr<const late_mat::row_selection> captured;
  if (!rep.range_pushdown().empty() || !predicates.empty() ||
      !rep.membership_pushdown().empty()) {
    std::span<const std::size_t> selected;
    if (indices.has_value()) {
      selected = *indices;
    } else {
      identity_selection.resize(n_selected);
      std::iota(identity_selection.begin(), identity_selection.end(), std::size_t{0});
      selected = identity_selection;
    }
    decompressed = try_decompress_scan_filter(ct,
                                              selected,
                                              rep.range_pushdown(),
                                              rep.range_conjuncts_convertible(),
                                              rep.equality_pushdown(),
                                              decode_pair_pushdown{},  // pair carrier not yet threaded
                                              rep.membership_pushdown(),
                                              rep.membership_generation(),
                                              pool,
                                              stream,
                                              mr,
                                              tag,
                                              rep.selection_capture_requested(),
                                              captured);
  }
  if (!decompressed) {
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

  // Tagged subclasses (see row_filtered_table_representation.hpp):
  // row_filtered is the scan's hard promise that the WHOLE table-filter
  // conjunction was applied during decode; rule2_bailed carries classic
  // full-width columns but lets the scan latch the per-operator bail flag and
  // strip the range pushdown from its remaining batches.
  if (tag == fused_batch_tag::row_filtered) {
    auto tagged = std::make_unique<row_filtered_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
    tagged->captured_selection = std::move(captured);  // empty unless requested+applied
    return tagged;
  }
  if (tag == fused_batch_tag::rule2_bailed) {
    return std::make_unique<rule2_bailed_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
  }
  if (captured) {
    // Untagged but applied (membership-compacted / partial-coverage batch):
    // metadata-only carrier — downstream filter semantics stay byte-identical
    // to a plain representation (the carrier adds a captured selection and
    // nothing else: no tag, no row shaping).
    auto neutral = std::make_unique<selection_captured_gpu_table_representation>(
      std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
    neutral->captured_selection = std::move(captured);
    return neutral;
  }
  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
}

}  // namespace

fused_scan_directives build_fused_scan_directives(const simpatico::compressed_table& table,
                                                  std::span<const std::size_t> selected_columns,
                                                  const decode_range_pushdown& attached_ranges,
                                                  bool all_conjuncts_convertible,
                                                  bool has_bool8_mask_sources,
                                                  const decode_pair_pushdown& attached_pairs,
                                                  bool has_membership_mask_sources)
{
  fused_scan_directives out;  // disabled/empty unless a mask source survives
  bool const has_external_sources = has_bool8_mask_sources || has_membership_mask_sources;
  // The [fused-diag] lines trace every accept/refuse decision of the fused
  // scan-filter pipeline. Kept at DEBUG: quiet by default, and raising the
  // level is the first move whenever a batch silently falls back to the
  // classic path (see the M2 hunt in STATUS-W2).
  bool const any_active = std::any_of(
    attached_ranges.begin(), attached_ranges.end(), [](auto const& e) { return e.active; });
  if (!any_active && !has_external_sources && attached_pairs.empty()) {
    SIRIUS_FUSED_DIAG("[fused-diag] directives: {} attached range entr(ies), NONE active, no "
                      "bool8/pair/membership sources — classic path",
                      attached_ranges.size());
    return out;
  }
  if (attached_ranges.size() > selected_columns.size()) {
    throw std::runtime_error("[compression_converters] range pushdown wider than the projection");
  }

  auto const count = selected_columns.size();
  out.ranges       = attached_ranges;
  out.ranges.resize(count);  // pad the inactive tail
  out.output_tiers.assign(count, decode_output_tier::tier_b);
  out.compact_capable.assign(count, 0);

  std::size_t mask_columns = 0;
  std::size_t compactable  = 0;
  bool dropped_conjunct    = false;
  for (std::size_t i = 0; i < count; ++i) {
    auto const probe = probe_fused_column(table, selected_columns[i]);
    auto const phys  = selected_columns[i];
    SIRIUS_FUSED_DIAG(
      "[fused-diag] directives col[{}] phys={} dtype={} tier={} lane_ok={} compact_capable={} "
      "range_active={} range=[{}, {}]",
      i,
      phys,
      phys < table.columns.size() ? type_id_to_name(table.columns[phys].dtype) : "OUT-OF-RANGE",
      static_cast<int>(probe.tier),
      probe.lane_ok,
      probe.compact_capable,
      out.ranges[i].active,
      out.ranges[i].lo,
      out.ranges[i].hi);
    out.output_tiers[i]    = probe.tier;
    out.compact_capable[i] = probe.compact_capable ? 1 : 0;
    if (probe.compact_capable) { ++compactable; }
    if (!out.ranges[i].active) { continue; }
    if (!probe.range_source_ok()) {
      // This chunk cannot evaluate the conjunct in-decode. Dropping it from
      // the mask is sound — mask conjuncts are conjunctive, so the mask only
      // under-filters and the residual filter still runs — but the mask no
      // longer covers the whole filter, so the batch must not be tagged
      // row-filtered.
      SIRIUS_FUSED_DIAG(
        "[fused-diag] directives: DROPPING range conjunct on selected pos {} (physical {}) — "
        "not a K1-capable bitpack leaf (tier={} lane_ok={} compact_capable={})",
        i,
        phys,
        static_cast<int>(probe.tier),
        probe.lane_ok,
        probe.compact_capable);
      out.ranges[i].active = false;
      dropped_conjunct     = true;
      continue;
    }
    out.ranges[i].participates_in_scan_mask = true;
    ++mask_columns;
  }

  // Pair validation (K1m2 sources): both sides must be K1-capable bitpack
  // leaves — same 1024-row chunk geometry, per the pair_predicate contract. A
  // kept pair CONSUMES its sides' standalone K1 participation (the pair kernel
  // fuses each side's constant range; one kernel, one mask). A bad pair is
  // DROPPED and clears whole-filter coverage per the contract in
  // selection.hpp — never emitted wrong.
  bool dropped_pair = false;
  for (auto const& pair : attached_pairs) {
    bool ok = pair.column_a < count && pair.column_b < count && pair.column_a != pair.column_b &&
              pair.op <= static_cast<std::uint8_t>(sirius::codegen::pair_compare_op::ne);
    if (ok) {
      auto const probe_a = probe_fused_column(table, selected_columns[pair.column_a]);
      auto const probe_b = probe_fused_column(table, selected_columns[pair.column_b]);
      ok = probe_a.range_source_ok() && probe_b.range_source_ok();
    }
    if (!ok) {
      SIRIUS_FUSED_DIAG(
        "[fused-diag] directives: DROPPING pair conjunct (sel {} vs {}, op={}) — sides not both "
        "K1m2-capable bitpack leaves",
        pair.column_a,
        pair.column_b,
        pair.op);
      dropped_pair = true;
      continue;
    }
    out.pairs.push_back(pair);
  }
  for (auto const& pair : out.pairs) {
    for (auto const idx : {pair.column_a, pair.column_b}) {
      if (out.ranges[idx].participates_in_scan_mask) {
        out.ranges[idx].participates_in_scan_mask = false;  // folded into the pair side
        --mask_columns;
      }
    }
  }

  if (mask_columns == 0 && out.pairs.empty() && !has_external_sources) {
    SIRIUS_FUSED_DIAG("[fused-diag] directives: no mask source survived — classic path");
    return {};
  }
  out.enabled = true;
  // Coverage: bool8 and membership sources never upgrade it (the extraction
  // gate only speaks for the static numeric view; dynamic filters are NEVER
  // whole-filter — the authoritative join must still run). Such batches stay
  // untagged and the post-decompress filter re-evaluates everything plus the
  // residual. Kept pairs do NOT affect it either — q12-class pair conjuncts
  // live in the FILTER operator ABOVE the scan, which runs regardless of the
  // scan's row-filtered tag, so masking them is a pure bonus restriction; a
  // DROPPED pair clears coverage per the selection.hpp contract (conservative
  // — costs only the tag).
  out.covers_whole_filter = all_conjuncts_convertible && !dropped_conjunct && !dropped_pair &&
                            (mask_columns > 0 || !out.pairs.empty()) && !has_external_sources;
  SIRIUS_FUSED_DIAG(
    "[fused-diag] directives ENABLED: {} range mask column(s), {} pair source(s), "
    "bool8_sources={}, membership_sources={}, {}/{} compact-capable output column(s), "
    "covers_whole_filter={}",
    mask_columns,
    out.pairs.size(),
    has_bool8_mask_sources,
    has_membership_mask_sources,
    compactable,
    count,
    out.covers_whole_filter);
  return out;
}

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
