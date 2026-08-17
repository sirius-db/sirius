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

#include "compressed_scan.hpp"

#include "decompression_pushdown_policy.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>

#include <api/simpatico_codegen.hpp>
#include <codegen/selection/selection.hpp>
#include <codegen/util/stream_pool.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius {

bool scan_decode_request::empty() const noexcept
{
  return std::all_of(columns.begin(), columns.end(), [](auto const& c) { return c.empty(); });
}

bool scan_decode_request::selects_rows() const noexcept
{
  return std::any_of(columns.begin(), columns.end(), [](auto const& c) {
    return c.range.has_value() || !c.membership.empty();
  });
}

namespace {

// Streams for cross-column decode parallelism, one pool per thread and device.
// Work is submitted from the calling thread so cuCascade memory-reservation
// tracking (attached to the calling thread) sees all allocations.
// 4 is not a configuration parameter — it matches the typical SM occupancy
// sweet spot for column-parallel decode without thread-spawn overhead.
constexpr std::size_t kDecodeStreams = 4;

simpatico::stream_pool& decode_pool()
{
  return simpatico::thread_device_stream_pool(kDecodeStreams);
}

//===----------------------------------------------------------------------===//
// Per-chunk capability probe
//===----------------------------------------------------------------------===//

/// What one column of a chunk can contribute to a filtered decode: the
/// decoder's own probe plus the one thing only this layer knows — whether the
/// dtype is comparable at all.
struct column_capability {
  /// The dtype decodes into a lane the row-selecting kernels can compare as a
  /// signed 64-bit integer. Every unsigned width is excluded: the JIT decode
  /// lane for an N-bit unsigned column is the same-width *signed* C++ type
  /// (see `dtype_to_cxx`), and the range/pair ballot widens that lane to
  /// int64 with a plain `static_cast`, which sign-extends. A stored value
  /// whose top bit is set (e.g. a uint32 above INT32_MAX) would decode to a
  /// negative int64 instead of its true unsigned value, silently corrupting
  /// the ballot for any such row.
  bool comparable_lane = false;
  simpatico::column_decode_caps decode;

  /// True iff the column can evaluate a range conjunct while it decodes.
  [[nodiscard]] bool can_select_rows() const noexcept
  {
    return comparable_lane && decode.can_produce_mask();
  }
};

column_capability probe_column(simpatico::compressed_table const& table, std::size_t column_index)
{
  column_capability capability;
  if (column_index >= table.columns.size()) { return capability; }
  auto const& column = table.columns[column_index];
  switch (column.dtype.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64: capability.comparable_lane = true; break;
    default: break;
  }
  auto const* tree = column.plan_tree.get();
  if (tree != nullptr) { capability.decode = simpatico::probe_column(*tree); }
  return capability;
}

//===----------------------------------------------------------------------===//
// Per-chunk plan
//===----------------------------------------------------------------------===//

/// A request narrowed to what one chunk can actually do with it.
struct chunk_decode_plan {
  /// One selected column's row-restricting bounds after the narrowing.
  struct range_slot {
    /// The scan asked for these bounds.
    bool requested = false;
    /// This chunk will evaluate them while decoding. False with @c requested
    /// true means the conjunct was dropped from the decode — still sound, the
    /// scan's own filter rejects those rows.
    bool selects = false;
    decode_range bounds;
  };

  /// At least one row-restricting conjunct survives on this chunk.
  bool enabled = false;
  /// The decode carries EVERY row-restricting conjunct of the scan's filter,
  /// so a batch it compacts needs no further filtering.
  bool covers_whole_filter = false;
  std::vector<range_slot> ranges;                     ///< parallel to selected columns
  std::vector<sirius::codegen::decode_route> routes;  ///< parallel to selected columns
};

/// Narrow @p request to what @p table can honour for @p selected_columns.
///
/// A conjunct on a column that cannot evaluate it while decoding is DROPPED
/// (sound: the decode then only under-filters, and the scan's residual filter
/// still runs) and clears @c covers_whole_filter. The plan is disabled only
/// when nothing survives.
///
/// @p has_external_selection declares that the caller contributes row-selecting
/// sources this narrowing does not see — an in-place equality answer folded
/// into the selection, or a dynamic join filter. Either lets the plan enable
/// with no range sources at all, and either forces @c covers_whole_filter
/// false: the coverage claim speaks only for the static numeric view, and a
/// dynamic filter is never the whole filter (the authoritative join still runs).
chunk_decode_plan plan_decode(simpatico::compressed_table const& table,
                              std::span<const std::size_t> selected_columns,
                              scan_decode_request const& request,
                              bool has_external_selection)
{
  chunk_decode_plan plan;
  bool const any_range = std::any_of(request.columns.begin(),
                                     request.columns.end(),
                                     [](auto const& c) { return c.range.has_value(); });
  if (!any_range && !has_external_selection) {
    SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
      "[decode-filter] plan: {} column entr(ies), no ranges, no in-place/membership "
      "sources — plain decode",
      request.columns.size());
    return plan;
  }
  if (request.columns.size() > selected_columns.size()) {
    throw std::runtime_error("[compressed_scan] decode request wider than the projection");
  }

  auto const count = selected_columns.size();
  plan.ranges.resize(count);
  plan.routes.assign(count, sirius::codegen::decode_route::full);
  for (std::size_t i = 0; i < request.columns.size(); ++i) {
    if (auto const& range = request.columns[i].range) {
      plan.ranges[i].requested = true;
      plan.ranges[i].bounds    = *range;
    }
  }

  std::size_t selecting_columns = 0;
  std::size_t compactable       = 0;
  bool dropped_conjunct         = false;
  for (std::size_t i = 0; i < count; ++i) {
    auto const capability = probe_column(table, selected_columns[i]);
    auto const physical   = selected_columns[i];
    SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
      "[decode-filter] plan col[{}] physical={} dtype={} route={} comparable_lane={} "
      "range_requested={} range=[{}, {}]",
      i,
      physical,
      physical < table.columns.size() ? type_id_to_name(table.columns[physical].dtype)
                                      : "OUT-OF-RANGE",
      static_cast<int>(capability.decode.compact_route),
      capability.comparable_lane,
      plan.ranges[i].requested,
      plan.ranges[i].bounds.lo,
      plan.ranges[i].bounds.hi);
    plan.routes[i] = capability.decode.compact_route;
    if (capability.decode.compact_route != sirius::codegen::decode_route::full) { ++compactable; }
    if (!plan.ranges[i].requested) { continue; }
    if (!capability.can_select_rows()) {
      SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
        "[decode-filter] plan: DROPPING range conjunct on selected pos {} (physical {}) — the "
        "column cannot evaluate it while decoding (route={} comparable_lane={})",
        i,
        physical,
        static_cast<int>(capability.decode.compact_route),
        capability.comparable_lane);
      plan.ranges[i].requested = false;
      dropped_conjunct         = true;
      continue;
    }
    plan.ranges[i].selects = true;
    ++selecting_columns;
  }

  if (selecting_columns == 0 && !has_external_selection) {
    SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
      "[decode-filter] plan: no row-selecting source survived — plain decode");
    return {};
  }
  plan.enabled             = true;
  plan.covers_whole_filter = request.ranges_cover_whole_filter && !dropped_conjunct &&
                             selecting_columns > 0 && !has_external_selection;
  SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
    "[decode-filter] plan ENABLED: {} row-selecting range column(s), "
    "external_sources={}, {}/{} column(s) decode compacted, covers_whole_filter={}",
    selecting_columns,
    has_external_selection,
    compactable,
    count,
    plan.covers_whole_filter);
  return plan;
}

//===----------------------------------------------------------------------===//
// Mechanism
//===----------------------------------------------------------------------===//

/// The in-place equality answers of @p request, padded to @p count so they line
/// up 1:1 with the columns being decoded. Empty when nothing is asked for,
/// which lets callers stay on the plain decompress overload.
std::vector<simpatico::decode_predicate> to_decode_predicates(scan_decode_request const& request,
                                                              std::size_t count)
{
  bool const any = std::any_of(request.columns.begin(), request.columns.end(), [](auto const& c) {
    return !c.equals_any.empty();
  });
  if (!any) { return {}; }
  if (request.columns.size() > count) {
    throw std::runtime_error("[compressed_scan] decode request wider than the projection");
  }
  std::vector<simpatico::decode_predicate> predicates(count);
  for (std::size_t i = 0; i < request.columns.size(); ++i) {
    predicates[i].equals_any = request.columns[i].equals_any;
  }
  return predicates;
}

/// One row-selecting source: a conjunct the decode evaluates in wave 1 to
/// produce one mask, whatever kind it is. Collecting all four kinds into one
/// list means the source count, the cap and the ordering are each computed
/// once, rather than per kind with rules that can drift apart.
struct selection_source {
  enum class kind : std::uint8_t { range, equality, membership };

  kind what                     = kind::range;
  std::size_t column            = 0;        ///< the source column
  membership_probe const* probe = nullptr;  ///< membership only

  /// Ascending EXPECTED keep-rate: the scarce mask slots go to the strongest
  /// source first, regardless of kind.
  ///
  /// Only the join filters carry a real signal (their kind rank, then the
  /// build-side key count). A range or equality conjunct has no
  /// statistics behind it, but it is exact and costs no probe launch, so it
  /// sorts ahead of every join filter and otherwise keeps request order — the
  /// same order they were emitted in before this list existed.
  [[nodiscard]] std::pair<int, std::uint64_t> order_key() const noexcept
  {
    if (what != kind::membership) { return {-1, 0}; }
    // 0 = unknown key count: sort after known counts within the same kind.
    return {probe->selectivity_rank,
            probe->num_keys == 0 ? std::numeric_limits<std::uint64_t>::max() : probe->num_keys};
  }
};

/// The most row-selecting sources one decode can carry. Beyond it the request
/// is structurally impossible and the round-trip is skipped.
constexpr std::size_t kMaxSelectionSources = 8;

/// Decode @p selected columns of @p chunk with @p request applied during the
/// decode. Returns a null table when the attempt is declined here (nothing to
/// plan, shape checks) or when the assembly refuses — the caller then decodes
/// plainly. @p outcome reports what a non-null return did.
///
/// An in-place equality answer survives EVERY outcome: whether the filtered
/// decode applies or falls back internally, the column's slot carries the BOOL8
/// answer (compacted to the surviving rows when the decode compacted), so the
/// scan's residual filter reads a boolean instead of re-comparing strings.
std::unique_ptr<cudf::table> decode_with_filters(simpatico::compressed_table const& chunk,
                                                 std::span<const std::size_t> selected,
                                                 scan_decode_request const& request,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr,
                                                 decode_outcome& outcome)
{
  // The conjuncts the chunk's compression plans cannot influence: an equality
  // answered off a dictionary, and the join filters. Collected first because
  // planning needs to know they exist — they let a chunk with no usable range
  // still be worth filtering.
  std::vector<selection_source> sources;
  for (std::size_t i = 0; i < request.columns.size() && i < selected.size(); ++i) {
    if (!request.columns[i].equals_any.empty()) {
      sources.push_back({selection_source::kind::equality, i});
    }
    for (auto const& probe : request.columns[i].membership) {
      sources.push_back({selection_source::kind::membership, i, &probe});
    }
  }

  auto const plan = plan_decode(chunk, selected, request, !sources.empty());
  if (!plan.enabled) { return nullptr; }

  // Then the conjuncts planning just decided this chunk can evaluate.
  for (std::size_t i = 0; i < plan.ranges.size(); ++i) {
    if (plan.ranges[i].selects) { sources.push_back({selection_source::kind::range, i}); }
  }

  if (sources.size() > kMaxSelectionSources ||
      chunk.num_rows() > std::numeric_limits<std::int32_t>::max()) {
    SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
      "[decode-filter] declined on shape ({} row-selecting sources, {} rows) — plain decode",
      sources.size(),
      chunk.num_rows());
    return nullptr;
  }

  // Strongest first. The decode keeps a PREFIX when it cannot carry every
  // source, so this order decides which ones survive.
  std::stable_sort(sources.begin(), sources.end(), [](auto const& a, auto const& b) {
    return a.order_key() < b.order_key();
  });

  sirius::codegen::scan_filter_request wave_request;
  wave_request.routes.reserve(selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    wave_request.routes.push_back(plan.routes[i]);
  }
  std::string order_echo;
  for (auto const& source : sources) {
    switch (source.what) {
      case selection_source::kind::range: {
        auto const& slot = plan.ranges[source.column];
        wave_request.filters.push_back({source.column, {slot.bounds.lo, slot.bounds.hi}});
        order_echo += " range(c" + std::to_string(source.column) + ")";
        break;
      }
      case selection_source::kind::equality:
        wave_request.bool8_filters.push_back(
          {source.column, request.columns[source.column].equals_any});
        order_echo += " equality(c" + std::to_string(source.column) + ")";
        break;
      case selection_source::kind::membership:
        wave_request.membership_filters.push_back({source.column, source.probe->probe});
        order_echo += " join(c" + std::to_string(source.column) + ",rank " +
                      std::to_string(source.probe->selectivity_rank) + ")";
        break;
    }
  }
  SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG("[decode-filter] wave-1 sources, ascending expected keep:{}",
                                     order_echo);
  wave_request.source_generation = request.membership_generation;

  sirius::codegen::scan_filter_result result;
  std::string error;
  auto table = simpatico::decompress_scan_filter(
    chunk, selected, wave_request, result, decode_pool(), stream, mr, &error);
  // The decode synchronized `stream`; re-point the selection buffers there
  // anyway so their teardown follows the batch's ordering.
  result.set_stream(stream);
  if (!error.empty()) {
    SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
      "[decode-filter] assembly REFUSED ({}); the batch decoded plainly", error);
  }
  // row_filtered only when the decode carried EVERY restricting conjunct: a
  // partially applied request must leave the batch untagged so the scan
  // evaluates the residual (re-checking already-applied conjuncts on the
  // surviving rows is idempotent).
  if (result.status == sirius::codegen::scan_filter_status::declined_unselective) {
    outcome.selection_unprofitable = true;
  } else if (result.applied && plan.covers_whole_filter) {
    outcome.row_filtered = true;
  }
  // Every equality the request carried was ANDed into the batch mask before
  // wave 2 ran, so on an applied decode the surviving rows already satisfy
  // them. On any other outcome the equality answers come from the plain
  // predicated rerun, which drops no rows.
  outcome.predicates_enforced = result.applied && !wave_request.bool8_filters.empty();
  SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
    "[decode-filter] decode {} (status={} generation={}): ranges={} equalities={} "
    "join_filters={} survivors={}/{} column(s)={} covers_whole_filter={} row_filtered={} "
    "selection_unprofitable={}",
    result.applied ? "APPLIED" : "NOT applied (plain output)",
    static_cast<int>(result.status),
    result.source_generation,
    wave_request.filters.size(),
    wave_request.bool8_filters.size(),
    wave_request.membership_filters.size(),
    result.survivor_count,
    result.num_rows,
    table->num_columns(),
    plan.covers_whole_filter,
    outcome.row_filtered,
    outcome.selection_unprofitable);
  return table;
}

}  // namespace

//===----------------------------------------------------------------------===//
// compressed_scan
//===----------------------------------------------------------------------===//

std::shared_ptr<const compressed_scan> compressed_scan::without_row_selection() const
{
  scan_decode_request narrowed;
  narrowed.columns.reserve(_request.columns.size());
  for (auto const& column : _request.columns) {
    narrowed.columns.push_back({column.equals_any, std::nullopt, {}});
  }
  if (narrowed.empty()) { return nullptr; }
  return std::make_shared<const compressed_scan>(std::move(narrowed));
}

std::shared_ptr<const compressed_scan> compressed_scan::with_membership_probes(
  std::vector<std::vector<membership_probe>> probes, std::uint64_t generation) const
{
  auto refreshed = _request;
  if (refreshed.columns.size() < probes.size()) { refreshed.columns.resize(probes.size()); }
  for (std::size_t i = 0; i < refreshed.columns.size(); ++i) {
    refreshed.columns[i].membership =
      i < probes.size() ? std::move(probes[i]) : std::vector<membership_probe>{};
  }
  refreshed.membership_generation = generation;
  return std::make_shared<const compressed_scan>(std::move(refreshed));
}

std::shared_ptr<const compressed_scan> compressed_scan::for_chunk(
  simpatico::compressed_table const& chunk, std::span<const std::size_t> selected) const
{
  // An in-place equality answer is only worth asking for where the plan can
  // resolve the predicate without materialising the column (a dictionary root);
  // pushing it into any other plan is correct but only moves the comparison.
  auto narrowed  = _request;
  bool any_asked = false;
  for (std::size_t i = 0; i < narrowed.columns.size(); ++i) {
    auto& column = narrowed.columns[i];
    if (!column.equals_any.empty() &&
        (i >= selected.size() || !probe_column(chunk, selected[i]).decode.can_answer_equality)) {
      column.equals_any.clear();
    }
    any_asked = any_asked || !column.empty();
  }
  if (!any_asked) { return nullptr; }
  return std::make_shared<const compressed_scan>(std::move(narrowed));
}

compressed_scan::compaction_forecast compressed_scan::forecast_compaction(
  simpatico::compressed_table const& chunk, std::span<const std::size_t> selected) const
{
  compaction_forecast forecast{};
  if (!decompression_pushdown_enabled()) { return forecast; }
  if (!_request.ranges_cover_whole_filter) { return forecast; }
  bool const any_range = std::any_of(_request.columns.begin(),
                                     _request.columns.end(),
                                     [](auto const& c) { return c.range.has_value(); });
  if (!any_range) { return forecast; }

  // Mirrors the planning above: every column this projection decodes must come
  // back compacted, else the decode runs plain and the caller must reserve the
  // full envelope. A column decoded through a dictionary gather also lifts the
  // selectivity ceiling — that shape wins at every selectivity, so the decode
  // never gives compaction up for it and the surviving row count is unbounded.
  bool any_unbounded = false;
  for (auto const index : selected) {
    if (index >= chunk.columns.size()) { return forecast; }
    auto const& plan_tree = chunk.columns[index].plan_tree;
    if (!plan_tree) { return forecast; }
    auto const route = simpatico::probe_column(*plan_tree).compact_route;
    if (route == sirius::codegen::decode_route::full) { return forecast; }
    any_unbounded = any_unbounded || route == sirius::codegen::decode_route::dict_codes;
  }
  forecast.compacts          = true;
  forecast.survivors_bounded = !any_unbounded;
  return forecast;
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

decode_result decode_compressed_chunk(simpatico::compressed_table const& chunk,
                                      std::span<const std::size_t> selected,
                                      compressed_scan const* scan,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  decode_result out;
  static scan_decode_request const no_request;
  auto const& request = scan != nullptr ? scan->request() : no_request;

  auto const predicates = to_decode_predicates(request, selected.size());
  SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(
    "[decode-filter] chunk: columns={} equalities={} request_empty={}",
    selected.size(),
    predicates.size(),
    request.empty());

  if (!request.empty()) {
    out.table = decode_with_filters(chunk, selected, request, stream, mr, out.outcome);
  }
  if (!out.table) {
    out.table = predicates.empty()
                  ? simpatico::decompress(chunk, selected, decode_pool(), mr)
                  : simpatico::decompress(chunk, selected, predicates, decode_pool(), mr);
  }
  // An active equality directive yields BOOL8 on every path — the filtered
  // decode and the plain rerun alike — so the substituted positions are exactly
  // the active entries.
  for (std::size_t i = 0; i < predicates.size(); ++i) {
    if (predicates[i].active()) { out.outcome.predicate_columns.push_back(i); }
  }
  return out;
}

}  // namespace sirius
