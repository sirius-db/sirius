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
#include "log/logging.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "scan_manager/late_mat_resolver.hpp"

#include <algorithm>
#include <cstdlib>
#include <optional>
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
    char* end     = nullptr;
    auto const n  = std::strtoul(v, &end, 10);
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
  std::size_t bytes  = 0;
  std::int64_t rows  = 0;
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
  op::sirius_physical_operator* producer_tail{nullptr};  // last op feeding the stop port
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
    auto ops    = pipeline->get_operators();
    bool seen   = false;
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
    if (single_op &&
        (next_op->type == optype::PARTITION || next_op->type == optype::CONCAT)) {
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
      auto const& lhs = join->lhs_output_columns.col_idxs;
      auto const& rhs = join->rhs_output_columns.col_idxs;
      auto const& own = entered_probe ? lhs : rhs;
      std::size_t const base = entered_probe ? 0 : lhs.size();
      std::unordered_map<std::size_t, std::size_t> remapped;
      for (auto const& [scan_pos, cur_pos] : positions) {
        auto const it = std::find(own.begin(), own.end(), static_cast<cudf::size_type>(cur_pos));
        if (it == own.end()) { continue; }
        remapped.emplace(scan_pos,
                         base + static_cast<std::size_t>(std::distance(own.begin(), it)));
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
    stop.consumer      = next_op;
    stop.producer_tail = pipeline->get_operators().empty()
                           ? nullptr
                           : &pipeline->get_operators().back().get();
    stop.position_map  = std::move(positions);
    stop.port_arity    = current_arity;
    stop.boundaries    = boundaries;
    return stop;
  }
  return std::nullopt;
}

}  // namespace

void try_install_late_mat_deferral(op::scan::sirius_gpu_scan_operator* scan_op,
                                   pinned_entry const& entry,
                                   std::span<std::size_t const> selected_columns)
{
  if (!late_mat::late_mat_enabled() || !defer_enabled()) { return; }
  if (scan_op == nullptr || !entry.late_mat_handle) { return; }
  if (scan_has_static_filters(*scan_op)) { return; }

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

  auto stop = walk_to_materialization_port(scan_op, positions, types.size());
  if (!stop || stop->boundaries < min_boundaries()) { return; }
  if (scan_op->late_mat_defer) { return; }

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
    if (!layout) {
      layout = resolve_pinned_layout(origin);
      if (!layout) { return; }
    }
    resolved.push_back({scan_pos, port_pos, *width, std::move(origin), std::move(*view)});
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
  if (auto const& incumbent = stop->consumer->late_mat_port_directive) {
    if (incumbent->bundle_value_bytes >= bundle_value) {
      SIRIUS_LOG_INFO(
        "[late_mat] candidate REJECTED (arbitration): scan op {} value={:.1f} <= incumbent "
        "value={:.1f} at consumer op {} ({})",
        scan_op->get_operator_id(),
        bundle_value,
        incumbent->bundle_value_bytes,
        stop->consumer->get_operator_id(),
        stop->consumer->get_name());
      return;
    }
    SIRIUS_LOG_INFO(
      "[late_mat] arbitration EVICTION at consumer op {} ({}): incumbent scan op {} "
      "(value={:.1f}) replaced by scan op {} (value={:.1f})",
      stop->consumer->get_operator_id(),
      stop->consumer->get_name(),
      incumbent->source_scan ? incumbent->source_scan->get_operator_id() : 0,
      incumbent->bundle_value_bytes,
      scan_op->get_operator_id(),
      bundle_value);
    if (incumbent->source_scan != nullptr) { incumbent->source_scan->late_mat_defer.reset(); }
    stop->consumer->late_mat_port_directive.reset();
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
  if (stop->producer_tail == nullptr) { return; }
  {
    auto const& tail_types = stop->producer_tail->get_types();
    if (tail_types.size() != stop->port_arity) {
      SIRIUS_LOG_INFO(
        "[late_mat] candidate REJECTED (schema): scan op {} -> consumer op {}: producer tail "
        "arity {} != walk arity {}",
        scan_op->get_operator_id(),
        stop->consumer->get_operator_id(),
        tail_types.size(),
        stop->port_arity);
      return;
    }
    expected_types.reserve(tail_types.size());
    try {
      for (auto const& lt : tail_types) {
        expected_types.push_back(sirius::get_cudf_type(lt).id());
      }
    } catch (std::exception const&) {
      return;  // untranslatable planned type: refuse rather than half-match
    }
  }

  auto port_directive               = std::make_shared<late_mat::port_materialize_directive>();
  port_directive->expected_arity    = stop->port_arity;
  port_directive->layout            = std::move(*layout);
  port_directive->bundle_value_bytes = bundle_value;
  port_directive->source_scan        = scan_op;
  // The rowid rides at the FIRST deferred scan position; find where that
  // position landed at the port.
  auto const rowid_scan_pos = scan_directive->output_positions.front();
  for (auto const& c : resolved) {
    port_directive->positions.push_back(c.port_pos);
    port_directive->origins.push_back(c.origin);
    port_directive->columns.push_back(c.view);
    if (c.scan_pos == rowid_scan_pos) { port_directive->rowid_position = c.port_pos; }
  }
  // Swap the deferred positions to their placeholder types (rowid UINT64,
  // INT8 elsewhere) — the schema the matcher must see, exactly.
  for (auto const& c : resolved) {
    expected_types[c.port_pos] = cudf::type_id::INT8;
  }
  expected_types[port_directive->rowid_position] = cudf::type_id::UINT64;
  port_directive->expected_types                 = std::move(expected_types);

  SIRIUS_LOG_INFO(
    "[late_mat] deferral installed: scan op {} -> consumer op {} ({}), {} column(s), {} "
    "boundary(ies), rowid at scan pos {} / port pos {}, sum_width={:.1f} B/row, value={:.1f}",
    scan_op->get_operator_id(),
    stop->consumer->get_operator_id(),
    stop->consumer->get_name(),
    resolved.size(),
    stop->boundaries,
    rowid_scan_pos,
    port_directive->rowid_position,
    total_width,
    bundle_value);

  stop->consumer->late_mat_port_directive = std::move(port_directive);
  scan_op->late_mat_defer                 = std::move(scan_directive);
}

}  // namespace sirius::scan_manager
