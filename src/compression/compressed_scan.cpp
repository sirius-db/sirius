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

#include "decode_filter_policy.hpp"

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
  return pairs.empty() &&
         std::all_of(columns.begin(), columns.end(), [](auto const& c) { return c.empty(); });
}

bool scan_decode_request::selects_rows() const noexcept
{
  if (!pairs.empty()) { return true; }
  return std::any_of(columns.begin(), columns.end(), [](auto const& c) {
    return c.range.has_value() || !c.membership.empty();
  });
}

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
    if (!pool.init(4)) { throw std::runtime_error("[compressed_scan] stream_pool init failed"); }
  }
  return pool;
}

//===----------------------------------------------------------------------===//
// Per-chunk capability probe
//===----------------------------------------------------------------------===//

/// What one column of a chunk can contribute to a filtered decode.
///
/// @c shape defers to @c simpatico::plan_selection_tier — the classifier that
/// lives next to the decode implementation — so a newly implemented decode
/// shape lights up here the moment it lands, with no plan-shape walk of our own
/// to drift from it.
struct column_capability {
  /// The dtype decodes into a lane the row-selecting kernels can compare as a
  /// signed 64-bit integer. uint64 is excluded: its upper half would misorder
  /// under a signed compare.
  bool comparable_lane = false;
  /// This build can decode the column straight into compacted (surviving-rows
  /// only) output.
  bool decodes_compacted = false;
  /// Which decode shape the column's plan falls into.
  sirius::codegen::output_tier shape = sirius::codegen::output_tier::tier_b;

  /// True iff the column can evaluate a range or a column-vs-column comparison
  /// while it decodes: the row-selecting kernels render bitpack-leaf roots
  /// only, so a compacted decode is necessary but not sufficient.
  [[nodiscard]] bool can_select_rows() const noexcept
  {
    return comparable_lane && decodes_compacted && shape == sirius::codegen::output_tier::tier_a;
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
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DECIMAL32:
    case cudf::type_id::DECIMAL64: capability.comparable_lane = true; break;
    default: break;
  }
  auto const* tree = column.plan_tree.get();
  if (tree == nullptr) { return capability; }
  capability.shape             = simpatico::plan_selection_tier(*tree);
  capability.decodes_compacted = simpatico::plan_supports_selection_decode(*tree);
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
    /// true means the conjunct was dropped from the decode (still sound — the
    /// scan's own filter rejects those rows) or was folded into a pair.
    bool selects = false;
    decode_range bounds;
  };

  /// At least one row-restricting conjunct survives on this chunk.
  bool enabled = false;
  /// The decode carries EVERY row-restricting conjunct of the scan's filter,
  /// so a batch it compacts needs no further filtering.
  bool covers_whole_filter = false;
  std::vector<range_slot> ranges;                    ///< parallel to selected columns
  std::vector<sirius::codegen::output_tier> shapes;  ///< parallel to selected columns
  std::vector<std::uint8_t> compacts;                ///< parallel to shapes; 0/1
  /// Column-vs-column conjuncts both of whose sides can select rows. A kept
  /// pair CONSUMES its sides' standalone range participation: the pair kernel
  /// folds each side's bounds in, so one kernel does both.
  std::vector<column_pair_conjunct> pairs;
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
  if (!any_range && !has_external_selection && request.pairs.empty()) {
    SIRIUS_DECODE_DIAG(
      "[decode-filter] plan: {} column entr(ies), no ranges, no in-place/pair/membership "
      "sources — plain decode",
      request.columns.size());
    return plan;
  }
  if (request.columns.size() > selected_columns.size()) {
    throw std::runtime_error("[compressed_scan] decode request wider than the projection");
  }

  auto const count = selected_columns.size();
  plan.ranges.resize(count);
  plan.shapes.assign(count, sirius::codegen::output_tier::tier_b);
  plan.compacts.assign(count, 0);
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
    SIRIUS_DECODE_DIAG(
      "[decode-filter] plan col[{}] physical={} dtype={} shape={} comparable_lane={} "
      "compacts={} range_requested={} range=[{}, {}]",
      i,
      physical,
      physical < table.columns.size() ? type_id_to_name(table.columns[physical].dtype)
                                      : "OUT-OF-RANGE",
      static_cast<int>(capability.shape),
      capability.comparable_lane,
      capability.decodes_compacted,
      plan.ranges[i].requested,
      plan.ranges[i].bounds.lo,
      plan.ranges[i].bounds.hi);
    plan.shapes[i]   = capability.shape;
    plan.compacts[i] = capability.decodes_compacted ? 1 : 0;
    if (capability.decodes_compacted) { ++compactable; }
    if (!plan.ranges[i].requested) { continue; }
    if (!capability.can_select_rows()) {
      SIRIUS_DECODE_DIAG(
        "[decode-filter] plan: DROPPING range conjunct on selected pos {} (physical {}) — the "
        "column cannot evaluate it while decoding (shape={} comparable_lane={} compacts={})",
        i,
        physical,
        static_cast<int>(capability.shape),
        capability.comparable_lane,
        capability.decodes_compacted);
      plan.ranges[i].requested = false;
      dropped_conjunct         = true;
      continue;
    }
    plan.ranges[i].selects = true;
    ++selecting_columns;
  }

  // Both sides of a pair must be able to select rows — same chunk geometry on
  // each side. A pair that cannot be evaluated is DROPPED and clears coverage
  // rather than being emitted wrong.
  bool dropped_pair = false;
  for (auto const& pair : request.pairs) {
    bool ok = pair.column_a < count && pair.column_b < count && pair.column_a != pair.column_b;
    if (ok) {
      ok = probe_column(table, selected_columns[pair.column_a]).can_select_rows() &&
           probe_column(table, selected_columns[pair.column_b]).can_select_rows();
    }
    if (!ok) {
      SIRIUS_DECODE_DIAG(
        "[decode-filter] plan: DROPPING pair conjunct (selected {} vs {}, op={}) — one side "
        "cannot evaluate a comparison while decoding",
        pair.column_a,
        pair.column_b,
        static_cast<int>(pair.op));
      dropped_pair = true;
      continue;
    }
    plan.pairs.push_back(pair);
  }
  for (auto const& pair : plan.pairs) {
    for (auto const index : {pair.column_a, pair.column_b}) {
      if (plan.ranges[index].selects) {
        plan.ranges[index].selects = false;  // folded into the pair
        --selecting_columns;
      }
    }
  }

  if (selecting_columns == 0 && plan.pairs.empty() && !has_external_selection) {
    SIRIUS_DECODE_DIAG("[decode-filter] plan: no row-selecting source survived — plain decode");
    return {};
  }
  plan.enabled = true;
  // Kept pairs do not affect coverage either way: a column-vs-column conjunct
  // lives in the FILTER operator above the scan, which runs regardless, so
  // masking it is a pure bonus restriction. A DROPPED pair clears coverage
  // (conservative — it costs only the ability to skip the residual filter).
  plan.covers_whole_filter = request.ranges_cover_whole_filter && !dropped_conjunct &&
                             !dropped_pair && (selecting_columns > 0 || !plan.pairs.empty()) &&
                             !has_external_selection;
  SIRIUS_DECODE_DIAG(
    "[decode-filter] plan ENABLED: {} row-selecting range column(s), {} pair source(s), "
    "external_sources={}, {}/{} column(s) decode compacted, covers_whole_filter={}",
    selecting_columns,
    plan.pairs.size(),
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
  std::size_t equality_sources   = 0;
  std::size_t membership_sources = 0;
  for (std::size_t i = 0; i < request.columns.size() && i < selected.size(); ++i) {
    if (!request.columns[i].equals_any.empty()) { ++equality_sources; }
    membership_sources += request.columns[i].membership.size();
  }
  auto const plan =
    plan_decode(chunk, selected, request, equality_sources > 0 || membership_sources > 0);
  if (!plan.enabled) { return nullptr; }

  std::size_t range_sources = 0;
  for (auto const& slot : plan.ranges) {
    if (slot.selects) { ++range_sources; }
  }
  // Every source counts once (a pair once, each membership probe once).
  auto const total_sources =
    range_sources + plan.pairs.size() + equality_sources + membership_sources;
  if (total_sources > kMaxSelectionSources ||
      chunk.num_rows() > std::numeric_limits<std::int32_t>::max()) {
    SIRIUS_DECODE_DIAG(
      "[decode-filter] declined on shape ({} row-selecting sources, {} rows) — plain decode",
      total_sources,
      chunk.num_rows());
    return nullptr;
  }

  sirius::codegen::scan_filter_request wave_request;
  wave_request.tiers.reserve(selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    auto const& slot = plan.ranges[i];
    if (slot.requested && slot.selects) {
      wave_request.filters.push_back({i, {slot.bounds.lo, slot.bounds.hi}});
    }
    wave_request.tiers.push_back(plan.compacts[i] != 0 ? plan.shapes[i]
                                                       : sirius::codegen::output_tier::tier_b);
  }
  // One kernel per pair, with each side's own bounds folded in (a side without
  // bounds stays full-domain). Planning already cleared those sides' standalone
  // participation.
  for (auto const& pair : plan.pairs) {
    sirius::codegen::pair_predicate predicate;
    predicate.op      = static_cast<sirius::codegen::pair_compare_op>(pair.op);
    auto const& left  = plan.ranges[pair.column_a];
    auto const& right = plan.ranges[pair.column_b];
    if (left.requested) { predicate.range_a = {left.bounds.lo, left.bounds.hi}; }
    if (right.requested) { predicate.range_b = {right.bounds.lo, right.bounds.hi}; }
    wave_request.pair_filters.push_back({pair.column_a, pair.column_b, predicate});
  }
  for (std::size_t i = 0; i < request.columns.size() && i < selected.size(); ++i) {
    if (request.columns[i].equals_any.empty()) { continue; }
    wave_request.bool8_filters.push_back({i, request.columns[i].equals_any});
  }
  // Join filters, strongest first: the decode keeps a PREFIX of the list when
  // it cannot carry them all, so the order decides which ones are kept.
  {
    struct ordered_probe {
      std::size_t column;
      membership_probe const* probe;
    };
    std::vector<ordered_probe> ordered;
    for (std::size_t i = 0; i < request.columns.size() && i < selected.size(); ++i) {
      for (auto const& probe : request.columns[i].membership) {
        ordered.push_back({i, &probe});
      }
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](auto const& a, auto const& b) {
      if (a.probe->selectivity_rank != b.probe->selectivity_rank) {
        return a.probe->selectivity_rank < b.probe->selectivity_rank;
      }
      // 0 = unknown key count: sort after known counts within the same kind.
      auto const a_keys =
        a.probe->num_keys == 0 ? std::numeric_limits<std::uint64_t>::max() : a.probe->num_keys;
      auto const b_keys =
        b.probe->num_keys == 0 ? std::numeric_limits<std::uint64_t>::max() : b.probe->num_keys;
      return a_keys < b_keys;
    });
    std::string order_echo;
    for (auto const& entry : ordered) {
      wave_request.membership_filters.push_back({entry.column, entry.probe->probe});
      order_echo += order_echo.empty() ? "" : ",";
      order_echo +=
        std::to_string(entry.probe->selectivity_rank) + ":c" + std::to_string(entry.column);
    }
    if (!ordered.empty()) {
      SIRIUS_DECODE_DIAG("[decode-filter] join filter order (ascending expected keep): [{}]",
                         order_echo);
    }
  }
  wave_request.source_generation = request.membership_generation;

  sirius::codegen::scan_filter_result result;
  auto decoded =
    simpatico::decompress_scan_filter(chunk, selected, wave_request, result, decode_pool(), mr);
  std::string error;
  auto table =
    simpatico::compact_scan_filter_output(std::move(decoded), result, stream, mr, &error);
  // compact_scan_filter_output synchronized `stream`; re-point the selection
  // buffers there anyway so their teardown follows the batch's ordering.
  result.set_stream(stream);
  if (!table) {
    SIRIUS_DECODE_DIAG("[decode-filter] assembly REFUSED ({}); falling back to a plain decode",
                       error);
    return nullptr;
  }
  // row_filtered only when the decode carried EVERY restricting conjunct: a
  // partially applied request must leave the batch untagged so the scan
  // evaluates the residual (re-checking already-applied conjuncts on the
  // surviving rows is idempotent).
  if (result.status == sirius::codegen::scan_filter_status::bailed_high_selectivity) {
    outcome.selection_unprofitable = true;
  } else if (result.applied && plan.covers_whole_filter) {
    outcome.row_filtered = true;
  }
  SIRIUS_DECODE_DIAG(
    "[decode-filter] decode {} (status={} generation={}): ranges={} pairs={} equalities={} "
    "join_filters={} survivors={}/{} column(s)={} covers_whole_filter={} row_filtered={} "
    "selection_unprofitable={}",
    result.applied ? "APPLIED" : "NOT applied (plain output)",
    static_cast<int>(result.status),
    result.source_generation,
    wave_request.filters.size(),
    wave_request.pair_filters.size(),
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
  bool any_asked = !narrowed.pairs.empty();
  for (std::size_t i = 0; i < narrowed.columns.size(); ++i) {
    auto& column = narrowed.columns[i];
    if (!column.equals_any.empty() &&
        (i >= selected.size() ||
         !simpatico::column_supports_predicate_decode(chunk, selected[i]))) {
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
  if (!decode_filtering_enabled()) { return forecast; }
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
    if (!simpatico::plan_supports_selection_decode(*plan_tree)) { return forecast; }
    any_unbounded = any_unbounded || simpatico::plan_selection_tier(*plan_tree) ==
                                       sirius::codegen::output_tier::tier_dict_k5;
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
  SIRIUS_DECODE_DIAG("[decode-filter] chunk: columns={} equalities={} request_empty={}",
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
