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

#include "scan_manager/late_mat_defer_policy.hpp"

#include "cudf/cudf_utils.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/utils.hpp"
#include "late_mat/defer_directive.hpp"
#include "late_mat/plan_deferral.hpp"
#include "log/logging.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "scan_manager/late_mat_resolver.hpp"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::scan_manager {

namespace {

bool defer_enabled()
{
  static const bool enabled = []() {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_DEFER");
    return v == nullptr || !(v[0] == '0' && v[1] == '\0');  // default ON under the late-mat gate
  }();
  return enabled;
}

bool compressed_origins_enabled()
{
  static const bool enabled = []() {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_COMPRESSED");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

std::size_t min_boundaries()
{
  static const std::size_t value = []() -> std::size_t {
    char const* v = std::getenv("SIRIUS_LATE_MAT_MIN_BOUNDARIES");
    if (v == nullptr) { return 4; }
    char* end    = nullptr;
    auto const n = std::strtoul(v, &end, 10);
    return end == v ? 4 : static_cast<std::size_t>(n);
  }();
  return value;
}

/// Deferred-value floor in B/row (SIRIUS_LATE_MAT_MIN_VALUE_BYTES, default 32):
/// install only when Σ(real deferred widths) − rowid_bytes ≥ this. Measured
/// basis (measured 2026-08-03): the port materialization prep is sorted-id
/// canonicalization sized by PORT rows — break-even Σw ≈ 60 B with the sort
/// path, ≈ 21 B with the planned uncompressed fast path; 32 kills every
/// measured loser (n_name 11.1 B +61 ms on q9's 800 M-row port, s_name 25 B)
/// and keeps the measured winners (customer bundle 154.6 B, supplier pair
/// 50 B at −2.8 ms). Lower toward 24 only after the fast path lands.
double min_value_bytes()
{
  static const double value = []() -> double {
    char const* v = std::getenv("SIRIUS_LATE_MAT_MIN_VALUE_BYTES");
    if (v == nullptr) { return 32.0; }
    char* end      = nullptr;
    double const d = std::strtod(v, &end);
    return (end == v || d < 0.0) ? 32.0 : d;
  }();
  return value;
}

/// Separate deferred-value floor for COMPRESSED-origin (D2) bundles
/// (SIRIUS_LATE_MAT_MIN_VALUE_COMPRESSED). Default = SIRIUS_LATE_MAT_MIN_VALUE_BYTES,
/// so behavior is unchanged until the arm-C experiment sets it (arm-C experiment:
/// compressed origins skip the scan-side decode of the deferred columns AND ride the
/// captured wave-1 selection, so their break-even Σw sits well below the
/// uncompressed sort-path floor — arm C measures it at 12).
double min_value_bytes_compressed()
{
  static const double value = []() -> double {
    char const* v = std::getenv("SIRIUS_LATE_MAT_MIN_VALUE_COMPRESSED");
    if (v == nullptr) { return min_value_bytes(); }
    char* end      = nullptr;
    double const d = std::strtod(v, &end);
    return (end == v || d < 0.0) ? min_value_bytes() : d;
  }();
  return value;
}

/// Count-on-deferred admit switch (SIRIUS_LATE_MAT_COUNT_DEFER, default OFF).
/// Measured neutral-to-noise on TPC-H SF1000 (q13, the only shape it fires
/// on: a ~4 B/row ride saving that repeated A/B runs could not separate from
/// that query's run-to-run spread), so it ships dark. The machinery stays —
/// outer-join lifetime modeling, pre-filter substitution and u32 emission are
/// shared with the group-by-rowid transform — and a workload with a heavier
/// count ride can re-enable it here.
bool count_on_deferred_enabled()
{
  static const bool enabled = []() {
    char const* v = std::getenv("SIRIUS_LATE_MAT_COUNT_DEFER");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

/// Group-by-rowid admit gate on GROUP-INPUT volume
/// (SIRIUS_LATE_MAT_GBR_MIN_GROUP_ROWS, default 0 = INERT). The mechanism is
/// retained because the ride's fixed costs (rowid hashing, per-group
/// materialization, port prep) can only pay on large aggregate inputs — a
/// 44.5 M-row group input measured -198 ms — but the suspected small-input
/// tax did NOT reproduce under a dedicated N=4 A/B on the shipping binary
/// (the 444 K-row shape measured +0.3/-1.0/+0.3/-1.3 ms, pooled ~0), so the
/// floor ships disabled and the shipping behavior equals the measured no-gate
/// arms exactly. Plan-time estimate of the FIRST ridden aggregate's input;
/// FAIL-OPEN — a missing/zero estimate always installs.
std::size_t min_gbr_group_rows()
{
  static const std::size_t value = []() -> std::size_t {
    char const* v = std::getenv("SIRIUS_LATE_MAT_GBR_MIN_GROUP_ROWS");
    if (v == nullptr) { return 0; }
    char* end    = nullptr;
    auto const n = std::strtoull(v, &end, 10);
    return end == v ? 0 : static_cast<std::size_t>(n);
  }();
  return value;
}

constexpr double kRowidBytes = 8.0;  // UINT64 pin-order rowid

/// Storage width of a fixed-width deferrable type, nullopt when unknown.
std::optional<double> fixed_type_width_bytes(sirius::logical_type const& t)
{
  switch (t.id()) {
    case sirius::type_id::TINYINT:
    case sirius::type_id::UTINYINT: return 1.0;
    case sirius::type_id::SMALLINT:
    case sirius::type_id::USMALLINT: return 2.0;
    case sirius::type_id::INTEGER:
    case sirius::type_id::UINTEGER:
    case sirius::type_id::FLOAT:
    case sirius::type_id::DATE: return 4.0;
    case sirius::type_id::BIGINT:
    case sirius::type_id::UBIGINT:
    case sirius::type_id::DOUBLE:
    case sirius::type_id::TIMESTAMP: return 8.0;
    case sirius::type_id::DECIMAL:
      if (t.decimal_precision() <= sirius::logical_type::decimal_max_precision_int16) {
        return 2.0;
      }
      if (t.decimal_precision() <= sirius::logical_type::decimal_max_precision_int32) {
        return 4.0;
      }
      if (t.decimal_precision() <= sirius::logical_type::decimal_max_precision_int64) {
        return 8.0;
      }
      return std::nullopt;
    default: return std::nullopt;
  }
}

/// Real per-row width of one pinned column (B/row): fixed types from the
/// type, VARCHAR measured from the pin storage itself (alloc bytes / rows —
/// chars + offsets + validity, the bytes that actually ride). nullopt = not
/// priceable (compressed strings have no cheap chars stat in v1) ⇒ the
/// candidate is dropped rather than guessed.
std::optional<double> pinned_column_width_bytes(pinned_entry const& entry,
                                                std::size_t entry_pos,
                                                sirius::logical_type const& t)
{
  if (t.id() != sirius::type_id::VARCHAR) { return fixed_type_width_bytes(t); }
  std::size_t bytes = 0;
  std::int64_t rows = 0;
  if (!entry.device_chunks.empty()) {
    for (auto const& chunk : entry.device_chunks) {
      if (chunk.compressed) { return std::nullopt; }
      if (entry_pos >= chunk.columns.size() || !chunk.columns[entry_pos]) { return std::nullopt; }
      bytes += chunk.columns[entry_pos]->alloc_size();
      rows += chunk.columns[entry_pos]->size();
    }
  } else if (!entry.data_batches_by_column.empty()) {
    auto const& names = entry.cache_info.column_names();
    if (entry_pos >= names.size()) { return std::nullopt; }
    auto const it = entry.data_batches_by_column.find(names[entry_pos]);
    if (it == entry.data_batches_by_column.end()) { return std::nullopt; }
    for (auto const& chunk : it->second) {
      if (!chunk) { return std::nullopt; }
      bytes += chunk->alloc_size();
      rows += chunk->size();
    }
  } else {
    return std::nullopt;
  }
  if (rows <= 0) { return std::nullopt; }
  return static_cast<double>(bytes) / static_cast<double>(rows);
}

/// Deferrable payload types: strings (the q10 class) and plain fixed-width
/// value types (the q9 class). Everything else is out (nested / HUGEINT /
/// wide DECIMAL / SQLNULL).
bool type_deferrable(sirius::logical_type const& t)
{
  switch (t.id()) {
    case sirius::type_id::VARCHAR:
    case sirius::type_id::TINYINT:
    case sirius::type_id::SMALLINT:
    case sirius::type_id::INTEGER:
    case sirius::type_id::BIGINT:
    case sirius::type_id::UTINYINT:
    case sirius::type_id::USMALLINT:
    case sirius::type_id::UINTEGER:
    case sirius::type_id::UBIGINT:
    case sirius::type_id::FLOAT:
    case sirius::type_id::DOUBLE:
    case sirius::type_id::DATE:
    case sirius::type_id::TIMESTAMP: return true;
    case sirius::type_id::DECIMAL:
      return t.decimal_precision() <= sirius::logical_type::decimal_max_precision_int64;
    default: return false;
  }
}

/// Bound column indices referenced by one side of a join's conditions.
std::unordered_set<std::size_t> condition_side_references(
  duckdb::vector<sirius::join_condition> const& conditions, bool left_side)
{
  std::unordered_set<std::size_t> refs;
  for (auto const& cond : conditions) {
    auto const& side = left_side ? cond.left : cond.right;
    if (!side) { continue; }
    sirius::ast::visit_references(*side, [&](sirius::ast::reference const& r) {
      refs.insert(static_cast<std::size_t>(r.column_index));
    });
  }
  return refs;
}

/// The scan must have NO static pushed-down filters in v1: a post-decode row
/// filter (or reader-side pushdown) would change batch row sets in ways only
/// the fused capture accounts for. (q10 customer and q9 lineitem carry none;
/// their filtering is dynamic-membership, which compacts either at the DF
/// operator — placeholders ride through it as data — or at fused decode,
/// where the capture supplies the survivor ids.)
bool scan_has_static_filters(op::scan::sirius_gpu_scan_operator const& op)
{
  auto const* pq = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(
    &const_cast<op::scan::sirius_gpu_scan_operator&>(op).get_ingestible().table_info());
  if (pq != nullptr) { return pq->table_filters != nullptr && !pq->table_filters->filters.empty(); }
  return true;  // non-parquet sources: refuse in v1 (duckdb pins carry MVCC machinery)
}

struct walk_stop {
  op::sirius_physical_operator* consumer{nullptr};
  op::sirius_physical_operator* producer_tail{nullptr};       // last op feeding the stop port
  std::unordered_map<std::size_t, std::size_t> position_map;  // scan output pos -> port pos
  std::size_t port_arity{0};
  std::size_t boundaries{0};
};

/// March from the scan's pipeline to the first non-transparent pipeline.
/// Returns nullopt on any shape the v1 walk cannot prove safe.
std::optional<walk_stop> walk_to_materialization_port(
  op::scan::sirius_gpu_scan_operator* scan_op,
  std::unordered_map<std::size_t, std::size_t> positions,  // candidate pos -> current pos
  std::size_t current_arity)
{
  using optype = op::SiriusPhysicalOperatorType;

  auto pipeline = scan_op->get_pipeline();
  if (!pipeline) { return std::nullopt; }
  // Operators after the scan inside its own pipeline: only DYNAMIC_FILTER is
  // transparent (identity mapping; placeholders ride its boolean-mask compact
  // as ordinary data).
  {
    auto ops  = pipeline->get_operators();
    bool seen = false;
    for (auto& op_ref : ops) {
      auto& op = op_ref.get();
      if (!seen) {
        seen = (&op == static_cast<op::sirius_physical_operator*>(scan_op));
        continue;
      }
      if (op.type != optype::DYNAMIC_FILTER) { return std::nullopt; }
    }
    if (!seen) { return std::nullopt; }
  }

  std::size_t boundaries = 0;
  // Hard cap so a malformed graph can never loop.
  for (int hops = 0; hops < 64; ++hops) {
    auto const& next_ports = pipeline->get_next_ports_after_sink();
    if (next_ports.size() != 1) { return std::nullopt; }
    auto* next_op        = next_ports.front().next_operator;
    auto const port_name = next_ports.front().next_operator_port_name;
    if (next_op == nullptr) { return std::nullopt; }
    // review-F2 hardening: the entered port must be fed by EXACTLY the pipeline we
    // came from, and no sibling input port of the entered op may share its
    // repository — a union-style CONCAT merging a deferred flow with another
    // producer (real data ⇒ loud concatenate throw; a second type-identical
    // deferred flow ⇒ silent cross-origin gather) must refuse deferral at
    // plan time instead.
    {
      auto* entered_port = next_op->get_port(port_name);
      if (entered_port == nullptr || entered_port->src_pipeline.get() != pipeline.get()) {
        return std::nullopt;
      }
      for (auto const other_id : next_op->get_port_ids()) {
        if (other_id == port_name) { continue; }
        auto* other = next_op->get_port(other_id);
        if (other != nullptr && other->repo != nullptr && other->repo == entered_port->repo) {
          return std::nullopt;
        }
      }
    }
    ++boundaries;

    auto next_pipeline = next_op->get_pipeline();
    if (!next_pipeline) { return std::nullopt; }
    auto next_ops = next_pipeline->get_operators();
    if (next_ops.empty() || &next_ops.front().get() != next_op) { return std::nullopt; }

    bool const single_op = next_ops.size() == 1;
    if (single_op && (next_op->type == optype::PARTITION || next_op->type == optype::CONCAT)) {
      // Identity mapping; keys (if any) are the downstream join's and are
      // checked when the walk reaches that join — a key collision there bails
      // the whole deferral before anything executes.
      pipeline = next_pipeline;
      continue;
    }
    if (single_op && next_op->type == optype::HASH_JOIN) {
      auto* join = dynamic_cast<op::sirius_physical_hash_join*>(next_op);
      if (join == nullptr) { return std::nullopt; }
      if (join->join_type != duckdb::JoinType::INNER) { return std::nullopt; }
      bool const entered_probe = port_name == "default";
      if (!entered_probe && port_name != "build") { return std::nullopt; }
      // Key columns leave candidacy (they must stay real data: the join reads
      // them, and any dynamic filter published back to the scan targets
      // exactly such key columns — the walk's single-consumer chain means
      // every join that can see this scan's columns is ON this path, so
      // dropping path-join key refs covers every possible DF target).
      auto const key_refs = condition_side_references(join->conditions, entered_probe);
      for (auto it = positions.begin(); it != positions.end();) {
        it = key_refs.contains(it->second) ? positions.erase(it) : std::next(it);
      }
      if (positions.empty()) { return std::nullopt; }
      // Remap through the join's output projection: [lhs selected..., rhs
      // selected...]. Candidates the join does not project simply leave
      // candidacy (their placeholders die with the join input).
      auto const& lhs        = join->lhs_output_columns.col_idxs;
      auto const& rhs        = join->rhs_output_columns.col_idxs;
      auto const& own        = entered_probe ? lhs : rhs;
      std::size_t const base = entered_probe ? 0 : lhs.size();
      std::unordered_map<std::size_t, std::size_t> remapped;
      for (auto const& [scan_pos, cur_pos] : positions) {
        auto const it = std::find(own.begin(), own.end(), static_cast<cudf::size_type>(cur_pos));
        if (it == own.end()) { continue; }
        remapped.emplace(scan_pos, base + static_cast<std::size_t>(std::distance(own.begin(), it)));
      }
      if (remapped.empty()) { return std::nullopt; }  // nothing left to materialize
      positions     = std::move(remapped);
      current_arity = lhs.size() + rhs.size();
      pipeline      = next_pipeline;
      continue;
    }

    // Non-transparent pipeline: materialize at its entry port. When that
    // pipeline is reached, `pipeline` is still the transparent chain's tail —
    // its last operator's planned types are the port schema (F1 hardening
    // input).
    // pipeline leads with a HASH_JOIN, drop candidates the join keys on from
    // OUR side — same DF-target reasoning as the pass-through case (the
    // custkey-class columns whose scan-side DYNAMIC_FILTER probes them must
    // ride as real data).
    if (next_op->type == optype::HASH_JOIN) {
      auto* join = dynamic_cast<op::sirius_physical_hash_join*>(next_op);
      if (join == nullptr) { return std::nullopt; }
      bool const entered_probe = port_name == "default";
      if (!entered_probe && port_name != "build") { return std::nullopt; }
      auto const key_refs = condition_side_references(join->conditions, entered_probe);
      for (auto it = positions.begin(); it != positions.end();) {
        it = key_refs.contains(it->second) ? positions.erase(it) : std::next(it);
      }
      if (positions.empty()) { return std::nullopt; }
    }
    walk_stop stop;
    stop.consumer = next_op;
    stop.producer_tail =
      pipeline->get_operators().empty() ? nullptr : &pipeline->get_operators().back().get();
    stop.position_map = std::move(positions);
    stop.port_arity   = current_arity;
    stop.boundaries   = boundaries;
    return stop;
  }
  return std::nullopt;
}

/// v2 count-on-deferred filter-purity check: a static filter may coexist with
/// the pre-filter rowid splice only when EVERY filter key is a pure-filter
/// column (materialized position >= output arity), i.e. the filter can never
/// read a deferred output position. Parquet-only in v2 (duckdb pins carry
/// MVCC machinery the deferral preconditions already exclude).
bool static_filters_only_on_pure_filter_columns(op::scan::sirius_gpu_scan_operator& op)
{
  auto const* pq =
    dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&op.get_ingestible().table_info());
  if (pq == nullptr || pq->table_filters == nullptr) { return false; }
  auto const order = op.get_ingestible().materialized_column_order();
  auto const n_out = op.get_types().size();
  for (auto const& [key, filter] : pq->table_filters->filters) {
    if (static_cast<std::size_t>(key) >= pq->column_ids.size()) { return false; }
    auto const primary = static_cast<std::size_t>(pq->column_ids[key].GetPrimaryIndex());
    auto const it      = std::find(order.begin(), order.end(), primary);
    if (it == order.end()) { return false; }
    if (static_cast<std::size_t>(std::distance(order.begin(), it)) < n_out) {
      return false;  // filter reads an OUTPUT column — refuse
    }
  }
  return true;
}

/// v2 count-on-deferred install (no-port shape). The rowid never
/// materializes: count(rowid) == count(col) for a non-null source, and an
/// outer join's NULLIFY nullifies the rowid exactly as it would the column.
/// Floors are count-specific (pure ride savings, zero materialize cost):
/// value = width − rowid_width >= 2 B/row, walk boundaries >= 2 (the
/// validation shape rides 3; same-pipeline counts stay out).
void try_install_count_only_deferral(op::scan::sirius_gpu_scan_operator* scan_op,
                                     pinned_entry const& entry,
                                     std::span<std::size_t const> selected_columns,
                                     duckdb::vector<sirius::logical_type> const& types,
                                     std::size_t scan_pos,
                                     late_mat::planned_column_deferral const& fact,
                                     bool has_static_filters)
{
  if (has_static_filters && !static_filters_only_on_pure_filter_columns(*scan_op)) {
    SIRIUS_LOG_INFO(
      "[late_mat] candidate REJECTED (count: static filter reads output columns): scan op {}",
      scan_op->get_operator_id());
    return;
  }
  std::unordered_map<std::size_t, std::size_t> positions{{scan_pos, scan_pos}};
  auto stop = walk_to_materialization_port(scan_op, std::move(positions), types.size());
  constexpr std::size_t kCountMinBoundaries = 2;
  if (!stop || stop->boundaries < kCountMinBoundaries) {
    SIRIUS_LOG_INFO(
      "[late_mat] candidate REJECTED (count: walk refused or ride too short): scan op {}",
      scan_op->get_operator_id());
    return;
  }
  if (fact.consumer == nullptr) { return; }
  auto const consumer_pipeline = fact.consumer->get_pipeline();
  auto const stop_pipeline     = stop->consumer->get_pipeline();
  if (!consumer_pipeline || !stop_pipeline || consumer_pipeline.get() != stop_pipeline.get()) {
    SIRIUS_LOG_INFO("[late_mat] v2 plan/walk DISAGREEMENT (count shape): scan op {} column {}",
                    scan_op->get_operator_id(),
                    scan_pos);
    return;
  }
  // Non-null gate: the resolver refuses nullable uncompressed sources;
  // simpatico-compressed pin columns are non-null by encode construction.
  late_mat::column_origin origin{entry.late_mat_handle,
                                 static_cast<std::uint32_t>(selected_columns[scan_pos]),
                                 entry.late_mat_handle->generation()};
  if (!resolve_pinned_column(origin)) { return; }
  auto const layout = resolve_pinned_layout(origin);
  if (!layout || layout->batch_row_start.empty()) { return; }
  bool const narrow =
    layout->batch_row_start.back() <= static_cast<std::int64_t>(std::uint64_t{1} << 32);
  auto const width = pinned_column_width_bytes(entry, selected_columns[scan_pos], types[scan_pos]);
  constexpr double kCountMinValue = 2.0;
  double const value              = width.value_or(0.0) - (narrow ? 4.0 : 8.0);
  if (value < kCountMinValue) {
    SIRIUS_LOG_INFO(
      "[late_mat] candidate REJECTED (count value): scan op {} width={:.1f} value={:.1f} < {:.1f}",
      scan_op->get_operator_id(),
      width.value_or(0.0),
      value,
      kCountMinValue);
    return;
  }
  auto directive              = std::make_shared<late_mat::deferred_scan_output>();
  directive->output_positions = {scan_pos};
  directive->narrow_rowid     = narrow;
  directive->pre_filter       = has_static_filters;
  SIRIUS_LOG_INFO(
    "[late_mat] deferral installed (count-on-deferred, no port): scan op {} -> consumer op {} "
    "({}), column {} width={:.1f}, rowid={} bit, pre_filter={}, {} boundary(ies)",
    scan_op->get_operator_id(),
    stop->consumer->get_operator_id(),
    stop->consumer->get_name(),
    scan_pos,
    width.value_or(0.0),
    narrow ? 32 : 64,
    has_static_filters,
    stop->boundaries);
  scan_op->late_mat_defer = std::move(directive);
}

/// Successful group-by-rowid ride extension: the port moves to the final
/// consumer's input, past the pass-modeled aggregate pipeline(s).
struct ride_extension {
  op::sirius_physical_operator* consumer{nullptr};       // port owner at the final pipeline
  op::sirius_physical_operator* producer_tail{nullptr};  // last op feeding that port
  op::sirius_physical_operator* fact_consumer{nullptr};  // the facts' final content reader
  std::vector<op::sirius_physical_operator*> chain;      // ridden aggregates, ride order
  std::size_t extra_boundaries{0};
  std::size_t group_by_stages{0};
  std::string unique_key_name;
};

/// Admission + pipeline hop for the group-by-rowid transform. Returns nullopt
/// on ANY unproven condition (the caller falls back to the sound stop-port
/// install). Requirements per the design's bijection argument:
///  - every stop-surviving candidate's fact rode through >=1 group-by to ONE
///    shared final consumer beyond the stop pipeline, null-free;
///  - a REAL-riding (non-deferred) column of the same origin with a pin-time
///    proven-unique fact is a planned group key at every aggregate on the
///    chain (rowid FD by planned keys ⇒ identical groups);
///  - the pipeline hops from the stop to the final consumer pass the same
///    single-producer checks the walk applies.
std::optional<ride_extension> try_extend_group_ride(op::scan::sirius_gpu_scan_operator* scan_op,
                                                    pinned_entry const& entry,
                                                    std::span<std::size_t const> selected_columns,
                                                    walk_stop const& stop)
{
  auto const& plan   = *scan_op->late_mat_plan;
  auto stop_pipeline = stop.consumer->get_pipeline();
  if (!stop_pipeline) { return std::nullopt; }
  std::unordered_map<std::size_t, late_mat::planned_column_deferral const*> fact_by_pos;
  for (auto const& f : plan.columns) {
    fact_by_pos.emplace(f.scan_output_position, &f);
  }
  op::sirius_physical_operator* final_consumer                = nullptr;
  std::vector<op::sirius_physical_operator*> const* key_chain = nullptr;
  for (auto const& [scan_pos, port_pos] : stop.position_map) {
    auto const it = fact_by_pos.find(scan_pos);
    if (it == fact_by_pos.end()) { return std::nullopt; }
    auto const* f = it->second;
    if (f->consumer == nullptr || f->group_key_at.empty() || f->nullified_on_ride ||
        f->consumed_as_count_only) {
      return std::nullopt;
    }
    auto const fp = f->consumer->get_pipeline();
    if (!fp || fp.get() == stop_pipeline.get()) { return std::nullopt; }
    if (final_consumer == nullptr) {
      final_consumer = f->consumer;
      key_chain      = &f->group_key_at;
    } else if (final_consumer != f->consumer) {
      return std::nullopt;
    }
  }
  if (final_consumer == nullptr || key_chain == nullptr) { return std::nullopt; }

  // Volume admit gate: the FIRST ridden aggregate's input estimate.
  if (min_gbr_group_rows() > 0 && !key_chain->empty()) {
    auto const* first_agg = key_chain->front();
    std::size_t const input_estimate =
      (first_agg != nullptr && !first_agg->children.empty() && first_agg->children[0])
        ? first_agg->children[0]->estimated_cardinality
        : 0;
    if (input_estimate > 0 && input_estimate < min_gbr_group_rows()) {
      SIRIUS_LOG_INFO(
        "[late_mat] group-by-rowid REFUSED (group input est. {} rows < admit floor {}): "
        "scan op {} — falling back to the pre-aggregate port",
        input_estimate,
        min_gbr_group_rows(),
        scan_op->get_operator_id());
      return std::nullopt;
    }
  }

  // Uniqueness proof (census line emitted by the caller): a non-deferred
  // column with a pin-proven-unique fact must be a group key at every ridden
  // aggregate.
  std::string unique_key_name;
  bool proven = false;
  for (auto const& f : plan.columns) {
    if (stop.position_map.contains(f.scan_output_position)) { continue; }  // must ride REAL
    if (f.group_key_at.empty()) { continue; }
    bool covers = true;
    for (auto* agg : *key_chain) {
      if (std::ranges::find(f.group_key_at, agg) == f.group_key_at.end()) {
        covers = false;
        break;
      }
    }
    if (!covers) { continue; }
    if (f.scan_output_position >= selected_columns.size()) { continue; }
    auto const entry_pos = selected_columns[f.scan_output_position];
    if (std::ranges::find(entry.unique_columns, static_cast<std::uint32_t>(entry_pos)) ==
        entry.unique_columns.end()) {
      continue;
    }
    proven          = true;
    unique_key_name = entry.cache_info.names[entry_pos];
    break;
  }
  if (!proven) {
    SIRIUS_LOG_INFO(
      "[late_mat] group-by-rowid REFUSED (no pin-proven-unique group key rides real): scan op "
      "{} — falling back to the pre-aggregate port",
      scan_op->get_operator_id());
    return std::nullopt;
  }

  // Hop pipelines to the final consumer with the walk's producer checks.
  auto pipeline             = stop_pipeline;
  auto const final_pipeline = final_consumer->get_pipeline();
  if (!final_pipeline) { return std::nullopt; }
  std::size_t extra_boundaries = 0;
  for (int hops = 0; hops < 16; ++hops) {
    auto const& next_ports = pipeline->get_next_ports_after_sink();
    if (next_ports.size() != 1) { return std::nullopt; }
    auto* next_op        = next_ports.front().next_operator;
    auto const port_name = next_ports.front().next_operator_port_name;
    if (next_op == nullptr) { return std::nullopt; }
    auto* entered = next_op->get_port(port_name);
    if (entered == nullptr || entered->src_pipeline.get() != pipeline.get()) {
      return std::nullopt;
    }
    for (auto const other_id : next_op->get_port_ids()) {
      if (other_id == port_name) { continue; }
      auto* other = next_op->get_port(other_id);
      if (other != nullptr && other->repo != nullptr && other->repo == entered->repo) {
        return std::nullopt;
      }
    }
    ++extra_boundaries;
    auto next_pipeline = next_op->get_pipeline();
    if (!next_pipeline) { return std::nullopt; }
    if (next_pipeline.get() == final_pipeline.get()) {
      ride_extension ride;
      ride.consumer = next_op;  // the port owner (first op of the final pipeline)
      ride.producer_tail =
        pipeline->get_operators().empty() ? nullptr : &pipeline->get_operators().back().get();
      ride.extra_boundaries = extra_boundaries;
      ride.group_by_stages  = key_chain->size();
      ride.unique_key_name  = unique_key_name;
      ride.fact_consumer    = final_consumer;
      ride.chain            = *key_chain;
      if (ride.producer_tail == nullptr) { return std::nullopt; }
      return ride;
    }
    pipeline = next_pipeline;
  }
  return std::nullopt;
}

/// One v3 rider origin ready for install: the port-side bundle, the rider
/// scan's own substitution positions, and the width it adds to the install's
/// arbitration value.
struct rider_plan {
  op::scan::sirius_gpu_scan_operator* scan{nullptr};
  late_mat::port_materialize_directive::origin_bundle bundle;
  std::vector<std::size_t> scan_positions;
  double width{0.0};
};

/// v3 FD closure + rider construction (SIRIUS_EXP_LATE_MAT_V3, dark
/// generality — no TPC-H perf claim). Seed: row(current origin), already
/// proven by the v2 admission (a pin-unique key of this origin is a planned
/// group key riding real). Transfer across the pass's INNER-join equality
/// edges; a rider origin's group keys drop iff its ROW becomes determined.
/// Every unprovable link contributes nothing — affected keys ride real. The
/// derivation-order trace is returned for the census verbatim.
std::vector<rider_plan> build_fd_riders(op::scan::sirius_gpu_scan_operator* scan_op,
                                        late_mat_defer_context const& context,
                                        late_mat::planned_fd_graph const& graph,
                                        ride_extension const& ride,
                                        std::string& trace)
{
  std::vector<rider_plan> riders;
  std::set<op::sirius_physical_operator*> determined_rows{scan_op};
  std::set<std::pair<op::sirius_physical_operator*, std::size_t>> determined_cols;
  trace          = "row(scan " + std::to_string(scan_op->get_operator_id()) + ")";
  auto unique_in = [&](op::sirius_physical_operator* scan, std::size_t pos) -> bool {
    auto* gpu_scan = dynamic_cast<op::scan::sirius_gpu_scan_operator*>(scan);
    if (gpu_scan == nullptr) { return false; }
    auto const it = context.by_scan.find(gpu_scan);
    if (it == context.by_scan.end() || pos >= it->second.columns.size()) { return false; }
    auto const entry_pos = static_cast<std::uint32_t>(it->second.columns[pos]);
    return std::ranges::find(it->second.entry->unique_columns, entry_pos) !=
           it->second.entry->unique_columns.end();
  };
  auto determined = [&](op::sirius_physical_operator* scan, std::size_t pos) {
    return determined_rows.contains(scan) || determined_cols.contains({scan, pos});
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto const& e : graph.edges) {
      for (int dir = 0; dir < 2; ++dir) {
        auto* from    = dir == 0 ? e.scan_a : e.scan_b;
        auto from_pos = dir == 0 ? e.pos_a : e.pos_b;
        auto* to      = dir == 0 ? e.scan_b : e.scan_a;
        auto to_pos   = dir == 0 ? e.pos_b : e.pos_a;
        if (!determined(from, from_pos) || determined(to, to_pos)) { continue; }
        determined_cols.insert({to, to_pos});
        changed = true;
        trace += " -> (scan " + std::to_string(to->get_operator_id()) + " col " +
                 std::to_string(to_pos) + ") ==join(op " +
                 std::to_string(e.join ? e.join->get_operator_id() : 0) + ")";
        if (!determined_rows.contains(to) && unique_in(to, to_pos)) {
          determined_rows.insert(to);
          trace += " [unique] -> row(scan " + std::to_string(to->get_operator_id()) + ")";
        }
      }
    }
  }

  // Rider keys: provenances at the ride's FIRST aggregate from OTHER,
  // row-determined origins, whose own lifetime facts rode the same chain to
  // the same final consumer.
  if (ride.chain.empty()) { return riders; }
  auto* first_agg = ride.chain.front();
  std::map<op::scan::sirius_gpu_scan_operator*,
           std::vector<late_mat::planned_fd_graph::key_provenance const*>>
    by_rider;
  for (auto const& kp : graph.key_provenances) {
    if (kp.aggregate != first_agg || kp.scan == scan_op) { continue; }
    auto* rider_scan = dynamic_cast<op::scan::sirius_gpu_scan_operator*>(kp.scan);
    if (rider_scan == nullptr || !determined_rows.contains(kp.scan)) { continue; }
    by_rider[rider_scan].push_back(&kp);
  }
  for (auto& [rider_scan, kps] : by_rider) {
    auto const ctx_it = context.by_scan.find(rider_scan);
    if (ctx_it == context.by_scan.end() || !rider_scan->late_mat_plan) { continue; }
    auto const& info = ctx_it->second;
    if (!info.entry->late_mat_handle) { continue; }
    rider_plan plan;
    plan.scan = rider_scan;
    std::optional<late_mat::pinned_table_layout> rider_layout;
    struct rider_col {
      std::size_t scan_pos;
      std::size_t port_pos;
    };
    std::vector<rider_col> cols;
    for (auto const* kp : kps) {
      // The rider column's own lifetime fact must have ridden the SAME chain
      // to the SAME final consumer, null-free.
      late_mat::planned_column_deferral const* fact = nullptr;
      for (auto const& f : rider_scan->late_mat_plan->columns) {
        if (f.scan_output_position == kp->scan_pos) {
          fact = &f;
          break;
        }
      }
      if (fact == nullptr || fact->consumer != ride.fact_consumer || fact->nullified_on_ride ||
          fact->consumed_as_count_only) {
        SIRIUS_LOG_INFO(
          "[late_mat] fd-chain REFUSED for rider scan {} column {}: lifetime fact does not "
          "reach the ride's final consumer null-free",
          rider_scan->get_operator_id(),
          kp->scan_pos);
        continue;
      }
      bool covers = true;
      for (auto* agg : ride.chain) {
        if (std::ranges::find(fact->group_key_at, agg) == fact->group_key_at.end()) {
          covers = false;
          break;
        }
      }
      if (!covers) { continue; }
      if (kp->scan_pos >= info.columns.size() || kp->scan_pos >= rider_scan->get_types().size()) {
        continue;
      }
      auto const width = pinned_column_width_bytes(
        *info.entry, info.columns[kp->scan_pos], rider_scan->get_types()[kp->scan_pos]);
      if (!width) { continue; }
      late_mat::column_origin origin{info.entry->late_mat_handle,
                                     static_cast<std::uint32_t>(info.columns[kp->scan_pos]),
                                     info.entry->late_mat_handle->generation()};
      auto view = resolve_pinned_column(origin);
      if (!view) { continue; }
      try {
        if (view->dtype.id() != cudf::type_id::EMPTY &&
            view->dtype != sirius::get_cudf_type(rider_scan->get_types()[kp->scan_pos])) {
          continue;  // narrow-stored rider column: same normalization-bypass guard
        }
      } catch (std::exception const&) {
        continue;
      }
      if (!rider_layout) {
        rider_layout = resolve_pinned_layout(origin);
        if (!rider_layout) { break; }
      }
      cols.push_back({kp->scan_pos, fact->final_position});
      plan.bundle.positions.push_back(fact->final_position);
      plan.bundle.origins.push_back(origin);
      plan.bundle.columns.push_back(std::move(*view));
      plan.width += *width;
    }
    if (plan.bundle.positions.empty() || !rider_layout) { continue; }
    plan.bundle.layout = std::move(*rider_layout);
    // Sort bundle vectors by port position; the rider's rowid rides at the
    // FIRST dropped scan position (its own scan-side directive convention).
    std::vector<std::size_t> order(plan.bundle.positions.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::sort(order, {}, [&](std::size_t i) { return plan.bundle.positions[i]; });
    late_mat::port_materialize_directive::origin_bundle sorted_bundle;
    sorted_bundle.layout = std::move(plan.bundle.layout);
    for (auto const i : order) {
      sorted_bundle.positions.push_back(plan.bundle.positions[i]);
      sorted_bundle.origins.push_back(plan.bundle.origins[i]);
      sorted_bundle.columns.push_back(std::move(plan.bundle.columns[i]));
    }
    for (auto const& c : cols) {
      plan.scan_positions.push_back(c.scan_pos);
    }
    std::ranges::sort(plan.scan_positions);
    // Rowid at the FIRST scan position -> its port position.
    for (auto const& c : cols) {
      if (c.scan_pos == plan.scan_positions.front()) {
        sorted_bundle.rowid_position = c.port_pos;
        break;
      }
    }
    plan.bundle = std::move(sorted_bundle);
    riders.push_back(std::move(plan));
  }
  return riders;
}

}  // namespace

void try_install_late_mat_deferral(op::scan::sirius_gpu_scan_operator* scan_op,
                                   pinned_entry const& entry,
                                   std::span<std::size_t const> selected_columns,
                                   late_mat_defer_context const* context)
{
  if (!late_mat::late_mat_enabled() || !defer_enabled()) { return; }
  if (scan_op == nullptr || !entry.late_mat_handle) { return; }
  bool const has_static_filters = scan_has_static_filters(*scan_op);
  // v1 refuses static filters outright; v2 defers the decision — the
  // count-on-deferred shape may substitute PRE-filter when the filter provably
  // never reads a deferred position (checked below). The normal bundle path
  // still refuses further down.
  if (!late_mat::late_mat_v2_enabled() && has_static_filters) { return; }

  // Compressed origins ride the wave-seam capture; hold them behind their own
  // gate until the wave-seam capture is live (a stamped scan whose fused batch lacks a
  // capture fails loudly at execute — never silently).
  bool const has_compressed_chunks = std::any_of(
    entry.device_chunks.begin(), entry.device_chunks.end(), [](sirius::device_pin_chunk const& c) {
      return c.compressed != nullptr;
    });
  if (has_compressed_chunks && !compressed_origins_enabled()) { return; }
  if (entry.tier != cucascade::memory::Tier::GPU) { return; }

  // Type-eligible candidates over the scan's OUTPUT layout (mapping
  // invariant: output position j == materialized slot j == selected_columns[j]).
  auto const& types = scan_op->get_types();
  std::unordered_map<std::size_t, std::size_t> positions;  // scan pos -> current pos
  for (std::size_t j = 0; j < types.size() && j < selected_columns.size(); ++j) {
    if (type_deferrable(types[j])) { positions.emplace(j, j); }
  }
  if (positions.empty()) { return; }

  // v2 (SIRIUS_EXP_LATE_MAT_V2): the planner's lifetime facts are the
  // authoritative CANDIDACY source — keep only columns the plan pass proved
  // are pure pass-throughs up to a recorded consumer. The pipeline walk below
  // still runs in full as the lowering/verification backend, and every
  // value/arbitration policy and runtime guard applies unchanged.
  std::unordered_map<std::size_t, late_mat::planned_column_deferral const*> planned_consumers;
  if (late_mat::late_mat_v2_enabled()) {
    if (!scan_op->late_mat_plan) {
      SIRIUS_LOG_INFO(
        "[late_mat] candidate REJECTED (v2: no plan annotation): scan op {} — lifetime pass "
        "recorded no deferrable column",
        scan_op->get_operator_id());
      return;
    }
    // Partition the facts: count-only candidates (NULL-extended rides are
    // sound here — COUNT_VALID counts the positionally-preserved null-ness)
    // vs normal-bundle candidacy (must be null-free: the port materializer
    // refuses NULL rowids by design).
    late_mat::planned_column_deferral const* count_fact = nullptr;
    std::size_t count_facts                             = 0;
    bool other_bundle_potential                         = false;
    for (auto const& fact : scan_op->late_mat_plan->columns) {
      if (fact.consumer == nullptr || !positions.contains(fact.scan_output_position)) { continue; }
      if (fact.consumed_as_count_only) {
        count_fact = &fact;
        ++count_facts;
        continue;
      }
      if (!fact.nullified_on_ride) {
        planned_consumers.emplace(fact.scan_output_position, &fact);
        // A fact whose consumer is a path join is key/expression-consumed en
        // route — the walk key-drops it, so it can never form a port bundle.
        if (fact.consumer->type != op::SiriusPhysicalOperatorType::HASH_JOIN) {
          other_bundle_potential = true;
        }
      }
    }
    // v2 count-on-deferred (no-port shape): exactly one count-only candidate
    // and nothing else that could form a (wider, precedence-winning) normal
    // bundle. The rowid IS the counted value — non-null source ⇒ identical
    // count; an outer join's NULLIFY nullifies it exactly as it would the
    // column.
    if (count_facts == 1 && !other_bundle_potential && count_on_deferred_enabled()) {
      try_install_count_only_deferral(scan_op,
                                      entry,
                                      selected_columns,
                                      types,
                                      count_fact->scan_output_position,
                                      *count_fact,
                                      has_static_filters);
      return;
    }
    for (auto it = positions.begin(); it != positions.end();) {
      it = planned_consumers.contains(it->first) ? std::next(it) : positions.erase(it);
    }
    if (positions.empty()) {
      SIRIUS_LOG_INFO(
        "[late_mat] candidate REJECTED (v2: no type-eligible column with a lifetime fact): "
        "scan op {}",
        scan_op->get_operator_id());
      return;
    }
  }
  if (has_static_filters) {
    // Census visibility: every refusal on the normal-bundle path logs a line.
    SIRIUS_LOG_INFO(
      "[late_mat] candidate REJECTED (static filters on scan op {} — normal-bundle path)",
      scan_op->get_operator_id());
    return;
  }

  auto stop = walk_to_materialization_port(scan_op, positions, types.size());
  // (positions was moved; candidate identity continues via stop->position_map.)
  if (!stop || stop->boundaries < min_boundaries()) {
    // Census visibility for formerly-silent walk refusals — v2-gated so the
    // banked v1 census line set stays byte-comparable.
    if (late_mat::late_mat_v2_enabled()) {
      SIRIUS_LOG_INFO(
        "[late_mat] candidate REJECTED (walk refused or ride shorter than {} boundaries): "
        "scan op {}",
        min_boundaries(),
        scan_op->get_operator_id());
    }
    return;
  }
  if (scan_op->late_mat_defer) { return; }

  // v2 group-by-rowid ride extension: when every stop-surviving candidate's
  // fact rode THROUGH the stop pipeline's group-by(s) to one shared final
  // consumer, move the materialization port past the aggregates (per-output-
  // group materialization). Admission needs the pin-time uniqueness proof; on
  // any refusal the bundle FALLS BACK to the stop-port install — early
  // materialization is always sound for a rode-through fact, so the banked
  // pre-aggregate behavior is preserved.
  op::sirius_physical_operator* target_consumer = stop->consumer;
  op::sirius_physical_operator* target_tail     = stop->producer_tail;
  std::size_t ride_boundaries                   = 0;
  bool group_ride                               = false;
  std::optional<ride_extension> active_ride;
  if (late_mat::late_mat_v2_enabled() && scan_op->late_mat_plan) {
    if (auto ride = try_extend_group_ride(scan_op, entry, selected_columns, *stop)) {
      active_ride     = *ride;
      target_consumer = ride->consumer;
      target_tail     = ride->producer_tail;
      ride_boundaries = ride->extra_boundaries;
      group_ride      = true;
      SIRIUS_LOG_INFO(
        "[late_mat] group-by-rowid ride: scan op {} -> final consumer op {} ({}), through {} "
        "group-by stage(s), unique key '{}' proof=pin-exact",
        scan_op->get_operator_id(),
        target_consumer->get_operator_id(),
        target_consumer->get_name(),
        ride->group_by_stages,
        ride->unique_key_name);
    }
  }

  // Resolve each surviving candidate's origin; drop what the resolver refuses
  // (nullable columns, unsupported storage). Sort by PORT position — the port
  // directive's vectors are parallel and ascending.
  auto const& handle = entry.late_mat_handle;
  struct resolved_candidate {
    std::size_t scan_pos;
    std::size_t port_pos;
    double width_bytes;
    late_mat::column_origin origin;
    late_mat::pinned_column_view view;
  };
  std::vector<resolved_candidate> resolved;
  std::optional<late_mat::pinned_table_layout> layout;
  for (auto const& [scan_pos, port_pos] : stop->position_map) {
    if (scan_pos >= selected_columns.size()) { continue; }
    // v2 verification: the walk's stop pipeline must contain the planner's
    // recorded consumer for this column — a disagreement means one of the two
    // analyses mis-modeled the plan, so the candidate fails closed.
    std::size_t use_port_pos = port_pos;
    if (!planned_consumers.empty()) {
      auto const planned = planned_consumers.find(scan_pos);
      if (planned == planned_consumers.end()) { continue; }
      auto const* fact             = planned->second;
      auto const consumer_pipeline = fact->consumer->get_pipeline();
      auto const stop_pipeline     = stop->consumer->get_pipeline();
      bool const same_pipeline =
        consumer_pipeline && stop_pipeline && consumer_pipeline.get() == stop_pipeline.get();
      // A fact that rode through the stop pipeline's group-by(s) is accepted
      // beyond it: materializing at the stop port is early-but-sound, and the
      // ride extension (when admitted) moves the port to the final consumer.
      if (!same_pipeline && fact->group_key_at.empty()) {
        SIRIUS_LOG_INFO(
          "[late_mat] v2 plan/walk DISAGREEMENT: scan op {} column {} — planned consumer's "
          "pipeline differs from the walk's stop pipeline; dropping candidate",
          scan_op->get_operator_id(),
          scan_pos);
        continue;
      }
      if (group_ride) { use_port_pos = fact->final_position; }
    }
    // Priceable-or-dropped: the value threshold below needs a REAL width
    // (measured: the earlier proxy rules installed eight losers); a column we cannot
    // price from pin metadata does not defer.
    auto const width =
      pinned_column_width_bytes(entry, selected_columns[scan_pos], types[scan_pos]);
    if (!width) { continue; }
    late_mat::column_origin origin{
      handle, static_cast<std::uint32_t>(selected_columns[scan_pos]), handle->generation()};
    auto view = resolve_pinned_column(origin);
    if (!view) { continue; }
    // Narrow-column-width interaction guard: a column STORED NARROW in the
    // pin must not defer — port-side materialization emits the stored
    // carrier and bypasses scan normalization, so downstream would see a
    // narrow type where the plan promises the native one. Only native-stored
    // columns defer; narrowing thus legitimately shrinks sum_width via the
    // columns that remain eligible.
    try {
      if (view->dtype.id() != cudf::type_id::EMPTY &&
          view->dtype != sirius::get_cudf_type(types[scan_pos])) {
        SIRIUS_LOG_INFO(
          "[late_mat] candidate REJECTED (narrow-stored): scan op {} column {} — stored "
          "carrier differs from the native type; materialization would bypass scan "
          "normalization",
          scan_op->get_operator_id(),
          scan_pos);
        continue;
      }
    } catch (std::exception const&) {
      continue;  // untranslatable native type: refuse rather than mis-type
    }
    if (!layout) {
      layout = resolve_pinned_layout(origin);
      if (!layout) { return; }
    }
    resolved.push_back({scan_pos, use_port_pos, *width, std::move(origin), std::move(*view)});
  }
  if (resolved.empty()) { return; }
  std::ranges::sort(resolved, {}, &resolved_candidate::port_pos);

  // Deferred-value threshold (measured-attribution fix, replaces "any string
  // pays"): the ride+materialize only beats the port-side prep when the
  // bundle is wide enough — Σ(real widths) − rowid ≥ T.
  double total_width = 0.0;
  for (auto const& c : resolved) {
    total_width += c.width_bytes;
  }
  double const bundle_value = total_width - kRowidBytes;
  double const value_floor =
    has_compressed_chunks ? min_value_bytes_compressed() : min_value_bytes();
  if (bundle_value < value_floor) {
    SIRIUS_LOG_INFO(
      "[late_mat] candidate REJECTED (value): scan op {} -> consumer op {} ({}), {} column(s), "
      "{} boundary(ies), sum_width={:.1f} B/row, value={:.1f} < threshold {:.1f}{}",
      scan_op->get_operator_id(),
      stop->consumer->get_operator_id(),
      stop->consumer->get_name(),
      resolved.size(),
      stop->boundaries,
      total_width,
      bundle_value,
      value_floor,
      has_compressed_chunks ? " (compressed floor)" : "");
    return;
  }

  // Consumer-slot arbitration (measured-attribution fix, replaces first-install-
  // wins): the WIDEST bundle per consumer holds the slot. Installs are
  // plan-time, pre-execution and single-threaded, so an eviction atomically
  // clears the loser's scan-side directive — a 25-row dimension string can
  // never lock out a payload bundle again.
  if (auto const& incumbent = target_consumer->late_mat_port_directive) {
    if (incumbent->bundle_value_bytes >= bundle_value) {
      SIRIUS_LOG_INFO(
        "[late_mat] candidate REJECTED (arbitration): scan op {} value={:.1f} <= incumbent "
        "value={:.1f} at consumer op {} ({})",
        scan_op->get_operator_id(),
        bundle_value,
        incumbent->bundle_value_bytes,
        target_consumer->get_operator_id(),
        target_consumer->get_name());
      return;
    }
    SIRIUS_LOG_INFO(
      "[late_mat] arbitration EVICTION at consumer op {} ({}): incumbent scan op {} "
      "(value={:.1f}) replaced by scan op {} (value={:.1f})",
      target_consumer->get_operator_id(),
      target_consumer->get_name(),
      incumbent->source_scan ? incumbent->source_scan->get_operator_id() : 0,
      incumbent->bundle_value_bytes,
      scan_op->get_operator_id(),
      bundle_value);
    if (incumbent->source_scan != nullptr) { incumbent->source_scan->late_mat_defer.reset(); }
    for (auto* rider : incumbent->rider_scans) {
      if (rider != nullptr) { rider->late_mat_defer.reset(); }
    }
    target_consumer->late_mat_port_directive.reset();
  }

  // Install the pair.
  auto scan_directive = std::make_shared<late_mat::deferred_scan_output>();
  {
    std::vector<std::size_t> scan_positions;
    scan_positions.reserve(resolved.size());
    for (auto const& c : resolved) {
      scan_positions.push_back(c.scan_pos);
    }
    std::sort(scan_positions.begin(), scan_positions.end());
    scan_directive->output_positions = std::move(scan_positions);
  }

  // F1 hardening input: the FULL expected post-substitution schema, from the
  // transparent chain tail's planned types. Any inconsistency (missing tail,
  // arity disagreement with the walk, untranslatable type) refuses the
  // install — the matcher must never run on a partial schema.
  std::vector<cudf::type_id> expected_types;
  if (target_tail == nullptr) { return; }
  std::size_t target_arity = 0;
  {
    auto const& tail_types = target_tail->get_types();
    // The walk's arity crosscheck applies at ITS stop port; a group ride's
    // port schema is the (pass-modeled) final producer tail's planned types.
    if (!group_ride && tail_types.size() != stop->port_arity) {
      SIRIUS_LOG_INFO(
        "[late_mat] candidate REJECTED (schema): scan op {} -> consumer op {}: producer tail "
        "arity {} != walk arity {}",
        scan_op->get_operator_id(),
        target_consumer->get_operator_id(),
        tail_types.size(),
        stop->port_arity);
      return;
    }
    target_arity = tail_types.size();
    expected_types.reserve(tail_types.size());
    // Narrow column widths (#1260): when the plan pass installed a physical
    // output schema on the tail, batches reach the port in THOSE carriers,
    // not the native mapping — record them so the apply-time signature
    // matcher agrees with what actually arrives. An empty override sidecar
    // means the native schema (get_physical_types() contract).
    if (target_tail->has_physical_overrides()) {
      auto const& phys = target_tail->get_physical_types();
      if (phys.size() != tail_types.size()) {
        return;  // malformed sidecar: refuse rather than half-match
      }
      for (auto const& dt : phys) {
        expected_types.push_back(dt.id());
      }
    } else {
      try {
        for (auto const& lt : tail_types) {
          expected_types.push_back(sirius::get_cudf_type(lt).id());
        }
      } catch (std::exception const&) {
        return;  // untranslatable planned type: refuse rather than half-match
      }
    }
  }

  auto port_directive                = std::make_shared<late_mat::port_materialize_directive>();
  port_directive->expected_arity     = target_arity;
  port_directive->bundle_value_bytes = bundle_value;
  port_directive->source_scan        = scan_op;
  // Nominated origin's bundle. The rowid rides at the FIRST deferred scan
  // position; find where that position landed at the port.
  late_mat::port_materialize_directive::origin_bundle nominated;
  nominated.layout          = std::move(*layout);
  auto const rowid_scan_pos = scan_directive->output_positions.front();
  for (auto const& c : resolved) {
    nominated.positions.push_back(c.port_pos);
    nominated.origins.push_back(c.origin);
    nominated.columns.push_back(c.view);
    if (c.scan_pos == rowid_scan_pos) { nominated.rowid_position = c.port_pos; }
  }
  auto const nominated_rowid_port_pos = nominated.rowid_position;
  port_directive->bundles.push_back(std::move(nominated));

  // v3 FD riders (dark generality — census-verbose, no perf claim): other
  // origins whose ROW the determination closure proves from this origin's
  // seed contribute rider bundles; their keys drop and materialize per
  // output group alongside the nominated bundle. Installed only after the
  // arbitration below (a rider whose scan still holds a foreign directive at
  // that point is dropped, never half-installed).
  std::vector<rider_plan> riders;
  std::string fd_trace;
  if (group_ride && active_ride && late_mat::late_mat_v3_enabled() && context != nullptr &&
      scan_op->late_mat_fd_graph) {
    riders =
      build_fd_riders(scan_op, *context, *scan_op->late_mat_fd_graph, *active_ride, fd_trace);
    for (auto const& r : riders) {
      port_directive->bundle_value_bytes += r.width;
    }
  }

  // Finalize riders now that the slot is clear: a rider scan still holding a
  // directive here belongs to a flow the arbitration did NOT evict — drop it
  // (its keys ride real; bundle and value adjusted).
  for (auto& r : riders) {
    if (r.scan->late_mat_defer) {
      SIRIUS_LOG_INFO("[late_mat] fd rider DROPPED (scan op {} holds a foreign directive)",
                      r.scan->get_operator_id());
      port_directive->bundle_value_bytes -= r.width;
      continue;
    }
    auto rider_directive              = std::make_shared<late_mat::deferred_scan_output>();
    rider_directive->output_positions = r.scan_positions;
    r.scan->late_mat_defer            = std::move(rider_directive);
    port_directive->rider_scans.push_back(r.scan);
    port_directive->bundles.push_back(std::move(r.bundle));
    SIRIUS_LOG_INFO(
      "[late_mat] group-by-rowid[v3] rider: scan op {} contributes {} key column(s) "
      "(width {:.1f} B/row); chain: {}",
      r.scan->get_operator_id(),
      port_directive->bundles.back().positions.size(),
      r.width,
      fd_trace);
  }
  // Swap every bundle's deferred positions to placeholder types (rowid
  // UINT64 at each bundle's rowid position, INT8 elsewhere) — the schema the
  // matcher must see, exactly.
  for (auto const& bundle : port_directive->bundles) {
    for (auto const pos : bundle.positions) {
      expected_types[pos] = cudf::type_id::INT8;
    }
    expected_types[bundle.rowid_position] = cudf::type_id::UINT64;
  }
  port_directive->expected_types = std::move(expected_types);

  SIRIUS_LOG_INFO(
    "[late_mat] deferral installed: scan op {} -> consumer op {} ({}), {} column(s), {} "
    "boundary(ies), rowid at scan pos {} / port pos {}, sum_width={:.1f} B/row, value={:.1f}",
    scan_op->get_operator_id(),
    target_consumer->get_operator_id(),
    target_consumer->get_name(),
    resolved.size(),
    stop->boundaries + ride_boundaries,
    rowid_scan_pos,
    nominated_rowid_port_pos,
    total_width,
    bundle_value);

  target_consumer->late_mat_port_directive = std::move(port_directive);
  scan_op->late_mat_defer                  = std::move(scan_directive);
}

}  // namespace sirius::scan_manager
