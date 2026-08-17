/*
 * Copyright 2025, Sirius Contributors.
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

#include "planner/sirius_plan_surrogate_groupby.hpp"

#include "cudf/cudf_utils.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/utils.hpp"
#include "log/logging.hpp"
#include "op/groupby_surrogate_deferral.hpp"
#include "op/groupby_surrogate_store.hpp"
#include "op/merge/gpu_surrogate_restore_impl.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <format>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

using sirius::op::gpu_surrogate_restore_impl;
using sirius::op::join_side;
using sirius::op::sirius_physical_grouped_aggregate;
using sirius::op::sirius_physical_hash_join;
using sirius::op::sirius_physical_operator;
using sirius::op::sirius_physical_projection;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::op::surrogate_deferral_store;
using sirius::op::surrogate_emit_plan;
using sirius::op::surrogate_restore_plan;

/// Result of `decline`, convertible to the failure value of any gate: false for the boolean
/// gates and nullopt for the optional-returning queries, so every early-out reads
/// `return decline(reason);`.
struct declined_t {
  operator bool() const noexcept { return false; }  // NOLINT(google-explicit-constructor)
  template <typename T>
  operator std::optional<T>() const noexcept  // NOLINT(google-explicit-constructor)
  {
    return std::nullopt;
  }
};

/// Log why the rewrite does not apply to this HASH_GROUP_BY. Every gate's early-out goes
/// through here so a declined query is always explainable from the DEBUG log.
declined_t decline(std::string const& reason)
{
  SIRIUS_LOG_DEBUG("groupby_surrogate_keys: declined for a HASH_GROUP_BY: {}", reason);
  return {};
}

/// C++20 stand-in for std::ranges::contains (C++23).
template <typename Range, typename T>
[[nodiscard]] bool contains(Range const& range, T const& value)
{
  return std::ranges::find(range, value) != std::ranges::end(range);
}

/// One operator on a traced key's path, with the key's column position at that operator's
/// OUTPUT. Used to patch declared types (and projection reference nodes / physical sidecars)
/// once a rewrite is committed.
struct trace_hop {
  sirius_physical_operator* op;
  cudf::size_type out_pos;
};

/// Where one string group key resolves to: the deepest INNER hash join it passes through.
/// Built wholesale by trace_string_key, so every field is valid by construction.
struct key_trace {
  sirius_physical_hash_join* join;
  join_side side;                    ///< side of `join` the column comes from
  cudf::size_type side_out_pos;      ///< position within that side's output column list
  cudf::size_type join_out_pos;      ///< position within the join's full output
  cudf::size_type source_input_col;  ///< column index within the side's child schema
  /// Operators strictly ABOVE the deferral join (deepest first excluded), each with the key's
  /// output position there. The deferral join itself is patched via side_out_pos/join_out_pos.
  std::vector<trace_hop> hops_above;
};

/// Trace a group-by input column down through pass-through operators to the deepest INNER
/// hash join it crosses. Returns nullopt when the column is not a pure pass-through of a
/// join side (computed projections, unsupported operators, non-INNER joins).
[[nodiscard]] std::optional<key_trace> trace_string_key(sirius_physical_operator* start,
                                                        cudf::size_type start_col)
{
  std::optional<key_trace> result;
  std::vector<trace_hop> hops;  // every op visited so far, deepest last
  sirius_physical_operator* cur = start;
  cudf::size_type col           = start_col;

  while (cur != nullptr) {
    hops.push_back(trace_hop{cur, col});
    if (cur->type == SiriusPhysicalOperatorType::PROJECTION) {
      auto& proj = cur->Cast<sirius_physical_projection>();
      // Only a plain reference is a pure pass-through; a computed expression (or an
      // out-of-range position) ends the trace.
      if (col < 0 || static_cast<std::size_t>(col) >= proj.select_list.size()) { break; }
      auto const* node = proj.select_list[static_cast<std::size_t>(col)].get();
      if (node == nullptr || !node->holds<sirius::ast::reference>()) { break; }
      col = static_cast<cudf::size_type>(node->get<sirius::ast::reference>().column_index);
      if (cur->children.size() != 1) { break; }
      cur = cur->children[0].get();
      continue;
    }
    if (cur->type == SiriusPhysicalOperatorType::HASH_JOIN) {
      auto& hj = cur->Cast<sirius_physical_hash_join>();
      // Only INNER joins never null-extend or drop a side's payload; delim-internal joins have
      // bespoke wiring. MARK joins append a column, so positions past the left side would be
      // the mark — excluded with the join-type gate.
      if (hj.join_type != duckdb::JoinType::INNER || hj.is_delim_join_inner()) { break; }
      auto const num_lhs = static_cast<cudf::size_type>(hj.lhs_output_columns.col_idxs.size());
      auto const num_rhs = static_cast<cudf::size_type>(hj.rhs_output_columns.col_idxs.size());
      if (col < 0 || col >= num_lhs + num_rhs) { break; }
      auto const side     = col < num_lhs ? join_side::left : join_side::right;
      auto const side_pos = side == join_side::left ? col : col - num_lhs;
      auto const src_col  = side == join_side::left
                              ? hj.lhs_output_columns.col_idxs[static_cast<std::size_t>(side_pos)]
                              : hj.rhs_output_columns.col_idxs[static_cast<std::size_t>(side_pos)];
      // Record this join as the (so far) deepest deferral candidate. Everything visited so far
      // EXCEPT this join is "above" it.
      result = key_trace{
        &hj, side, side_pos, col, src_col, std::vector<trace_hop>(hops.begin(), hops.end() - 1)};
      // Descend further only when this join does not READ the traced column: a deeper deferral
      // would hand this join a rowid where its key comparison (or a mixed-join AST predicate,
      // which may reference any input column) expects the string. As the deferral join itself
      // it is fine — emission changes, key inputs do not.
      if (!hj.all_conditions_are_equality_keys()) { break; }
      if (contains(hj.key_col_indices(side), src_col)) { break; }
      std::size_t const child_idx = side == join_side::left ? 0 : 1;
      if (child_idx >= cur->children.size()) { break; }
      cur = cur->children[child_idx].get();
      col = src_col;
      continue;
    }
    break;  // any other operator ends the trace
  }
  return result;
}

/// The traced column must have exactly ONE consumer at every operator above the deferral join:
/// the traced pass-through itself. Any other consumer (a computed projection expression, a
/// second join output slot reading the same input column, ...) would receive rowid/dummy
/// carriers where it expects the string. Returns false when such a consumer exists.
[[nodiscard]] bool trace_has_no_other_consumers(key_trace const& trace)
{
  for (auto const& hop : trace.hops_above) {
    if (hop.op->type == SiriusPhysicalOperatorType::PROJECTION) {
      auto& proj              = hop.op->Cast<sirius_physical_projection>();
      auto const traced_input = proj.select_list[static_cast<std::size_t>(hop.out_pos)]
                                  ->get<sirius::ast::reference>()
                                  .column_index;
      for (std::size_t o = 0; o < proj.select_list.size(); ++o) {
        if (static_cast<cudf::size_type>(o) == hop.out_pos) { continue; }
        auto const* entry = proj.select_list[o].get();
        if (!entry) { continue; }
        bool references_traced = false;
        sirius::ast::visit_references(*entry, [&](sirius::ast::reference const& ref) {
          if (ref.column_index == traced_input) { references_traced = true; }
        });
        if (references_traced) { return false; }
      }
    } else if (hop.op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
      auto& hj            = hop.op->Cast<sirius_physical_hash_join>();
      auto const num_lhs  = static_cast<cudf::size_type>(hj.lhs_output_columns.col_idxs.size());
      auto const side     = hop.out_pos < num_lhs ? join_side::left : join_side::right;
      auto const side_pos = side == join_side::left ? hop.out_pos : hop.out_pos - num_lhs;
      auto const& col_idxs =
        side == join_side::left ? hj.lhs_output_columns.col_idxs : hj.rhs_output_columns.col_idxs;
      auto const src_col = col_idxs[static_cast<std::size_t>(side_pos)];
      if (std::ranges::count(col_idxs, src_col) != 1) { return false; }
    }
  }
  return true;
}

/// Patch one operator's declared schema (and projection reference node / physical sidecar) so
/// position `pos` carries `new_type` instead of the original string type. Positions were
/// validated while tracing, so a miss here means the trace is corrupt — fail loudly rather than
/// mask a schema skew that would surface as wrong results downstream.
void patch_slot_type(sirius_physical_operator& op,
                     cudf::size_type pos,
                     sirius::logical_type const& new_type)
{
  auto const upos = static_cast<std::size_t>(pos);
  if (upos >= op.types.size()) {
    throw sirius::internal_exception(
      "groupby_surrogate_keys: patch position {} outside the {} declared output types (corrupt "
      "trace)",
      pos,
      op.types.size());
  }
  op.types[upos] = new_type;
  if (op.has_physical_overrides()) {
    auto phys = op.get_physical_types();
    if (upos >= phys.size()) {
      throw sirius::internal_exception(
        "groupby_surrogate_keys: patch position {} outside the {} physical sidecar entries "
        "(corrupt trace)",
        pos,
        phys.size());
    }
    phys[upos] = sirius::get_cudf_type(new_type);
    op.set_physical_types(std::move(phys));
  }
  if (op.type == SiriusPhysicalOperatorType::PROJECTION) {
    auto& proj = op.Cast<sirius_physical_projection>();
    if (upos >= proj.select_list.size() || !proj.select_list[upos] ||
        !proj.select_list[upos]->holds<sirius::ast::reference>()) {
      throw sirius::internal_exception(
        "groupby_surrogate_keys: projection position {} is not the traced pure reference "
        "(corrupt trace)",
        pos);
    }
    auto const idx = proj.select_list[upos]->get<sirius::ast::reference>().column_index;
    proj.select_list[upos] =
      std::make_unique<sirius::ast::node>(sirius::ast::reference(idx, new_type));
  }
  if (op.type == SiriusPhysicalOperatorType::HASH_JOIN) {
    auto& hj           = op.Cast<sirius_physical_hash_join>();
    auto const num_lhs = static_cast<cudf::size_type>(hj.lhs_output_columns.col_idxs.size());
    if (pos < num_lhs) {
      hj.lhs_output_columns.col_types[upos] = new_type;
    } else {
      auto const rhs_pos = static_cast<std::size_t>(pos - num_lhs);
      if (rhs_pos >= hj.rhs_output_columns.col_types.size()) {
        throw sirius::internal_exception(
          "groupby_surrogate_keys: join output position {} outside both sides' output columns "
          "(corrupt trace)",
          pos);
      }
      hj.rhs_output_columns.col_types[rhs_pos] = new_type;
    }
  }
}

//===----------------------------------------------------------------------===//
// try_rewrite_group_by gate sequence: pure predicates/queries
//===----------------------------------------------------------------------===//

/// Shape gates that need no tracing: not already rewritten, a single grouping set, no
/// COUNT(DISTINCT), the standard keys-then-aggregates output layout, a single child,
/// recomposable aggregate kinds (via gpu_surrogate_restore_impl, the finalizer's own
/// authority), and the minimum-rows knob.
[[nodiscard]] bool passes_structural_gates(sirius_physical_grouped_aggregate const& agg,
                                           sirius::operator_params const& op_params)
{
  if (agg.surrogate_restore()) { return decline("already rewritten"); }
  if (agg.grouping_sets.size() > 1) { return decline("multiple grouping sets"); }
  if (agg.has_count_distinct) {
    return decline("COUNT(DISTINCT) present");  // COLLECT_SET merge does not re-compose here
  }
  if (agg.types.size() != agg.group_idx.size() + agg.aggregate_slots.size()) {
    return decline("non-standard output layout");
  }
  if (agg.children.size() != 1 || !agg.children[0]) { return decline("no single child"); }
  for (auto kind : agg.cudf_aggregates) {
    if (!gpu_surrogate_restore_impl::is_recomposable_aggregate(kind)) {
      return decline("non-recomposable aggregate kind");
    }
  }
  if (agg.children[0]->estimated_cardinality < op_params.groupby_surrogate_min_rows) {
    return decline(std::format("estimated cardinality {} below groupby_surrogate_min_rows",
                               agg.children[0]->estimated_cardinality));
  }
  return true;
}

/// The group-by's STRING key slots, gated on the minimum-string-keys knob and on at least one
/// real (non-string) key remaining for partitioning and the distinct proof.
[[nodiscard]] std::optional<std::vector<std::size_t>> collect_string_key_slots(
  sirius_physical_grouped_aggregate const& agg, sirius::operator_params const& op_params)
{
  std::vector<std::size_t> string_slots;
  for (std::size_t slot = 0; slot < agg.group_idx.size(); ++slot) {
    if (agg.types[slot].is_varchar()) { string_slots.push_back(slot); }
  }
  if (string_slots.size() < op_params.groupby_surrogate_min_string_keys) {
    return decline(
      std::format("{} string key(s) below groupby_surrogate_min_string_keys", string_slots.size()));
  }
  if (string_slots.size() == agg.group_idx.size()) {
    return decline(
      "all keys are strings");  // need >= 1 real key slot for partitioning / distinct proof
  }
  return string_slots;
}

/// No aggregate may consume a deferred child column (it would aggregate rowids).
[[nodiscard]] bool no_aggregate_consumes(sirius_physical_grouped_aggregate const& agg,
                                         std::vector<std::size_t> const& string_slots)
{
  for (auto slot : string_slots) {
    int const child_col = agg.group_idx[slot];
    if (contains(agg.cudf_aggregate_idx, child_col)) {
      return decline("an aggregate consumes a deferred key column");
    }
    for (auto const& struct_cols : agg.cudf_aggregate_struct_col_indices) {
      if (contains(struct_cols, child_col)) {
        return decline("a struct aggregate consumes a deferred key column");
      }
    }
  }
  return true;
}

/// Every string key traced to one common deferral join.
struct traced_keys {
  sirius_physical_hash_join* deferral_join;
  std::vector<std::pair<std::size_t, key_trace>> traces;  ///< (group-by key slot, its trace)
};

/// Trace every string key slot to a single common INNER deferral join that is not already
/// deferring, with no other consumer of any traced column above it.
[[nodiscard]] std::optional<traced_keys> trace_all_to_common_join(
  sirius_physical_grouped_aggregate const& agg, std::vector<std::size_t> const& string_slots)
{
  traced_keys result{nullptr, {}};
  result.traces.reserve(string_slots.size());
  for (auto slot : string_slots) {
    auto trace =
      trace_string_key(agg.children[0].get(), static_cast<cudf::size_type>(agg.group_idx[slot]));
    if (!trace) {
      return decline(
        std::format("key slot {} (child col {}) does not trace to an INNER join pass-through",
                    slot,
                    agg.group_idx[slot]));
    }
    if (result.deferral_join == nullptr) {
      result.deferral_join = trace->join;
    } else if (result.deferral_join != trace->join) {
      return decline("string keys trace to different joins");
    }
    result.traces.emplace_back(slot, std::move(*trace));
  }
  if (result.deferral_join == nullptr) { return decline("no deferral join found"); }
  if (result.deferral_join->surrogate_emit()) {
    return decline("deferral join already carries a deferral");
  }
  for (auto const& [slot, trace] : result.traces) {
    if (!trace_has_no_other_consumers(trace)) {
      return decline("a deferred column has another consumer above the deferral join");
    }
  }
  return result;
}

/// Deduplicated ascending side-output positions chosen for deferral on each join side. The
/// smallest position of a side (`*begin()`) carries the rowid; the rest become dummies.
/// (Distinct key slots may trace to the same join output position; the sets deduplicate.)
struct deferred_positions {
  std::array<std::set<cudf::size_type>, 2> per_side;
  [[nodiscard]] std::set<cudf::size_type> const& operator[](join_side side) const noexcept
  {
    return per_side[static_cast<std::size_t>(side)];
  }
  [[nodiscard]] std::set<cudf::size_type>& operator[](join_side side) noexcept
  {
    return per_side[static_cast<std::size_t>(side)];
  }
};

[[nodiscard]] deferred_positions choose_deferred_positions(
  std::vector<std::pair<std::size_t, key_trace>> const& traces)
{
  deferred_positions positions;
  for (auto const& [slot, trace] : traces) {
    positions[trace.side].insert(trace.side_out_pos);
  }
  return positions;
}

/// Per-side estimated-row cap for the rowid address-space gate. The merge's finalize gather
/// addresses each deferral side with an INT32 cudf gather map, and reserve/commit dedupes by
/// batch id, so total registered rows per side is roughly that side's input rows; the /2
/// headroom absorbs a 2x cardinality-estimate error (value frozen).
constexpr std::size_t k_max_deferred_side_rows = std::numeric_limits<cudf::size_type>::max() / 2;

/// Decline when a deferred side's estimate approaches int32 rowid addressing (the runtime
/// reserve() throw remains as the hard backstop for estimate misses).
[[nodiscard]] bool passes_rowid_addressing_gate(sirius_physical_hash_join const& join,
                                                deferred_positions const& positions)
{
  if (!positions[join_side::left].empty() && join.children.size() >= 1 &&
      join.children[0]->estimated_cardinality > k_max_deferred_side_rows) {
    return decline("deferred probe side estimated cardinality exceeds int32 rowid addressing");
  }
  if (!positions[join_side::right].empty() && join.children.size() >= 2 &&
      join.children[1]->estimated_cardinality > k_max_deferred_side_rows) {
    return decline("deferred build side estimated cardinality exceeds int32 rowid addressing");
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Builders (pure, produce the immutable plan values)
//===----------------------------------------------------------------------===//

[[nodiscard]] surrogate_emit_plan build_emit_plan(deferred_positions const& positions,
                                                  std::shared_ptr<surrogate_deferral_store> store)
{
  auto const make_side =
    [](std::set<cudf::size_type> const& pos_set) -> std::optional<surrogate_emit_plan::side_plan> {
    if (pos_set.empty()) { return std::nullopt; }
    auto const rowid_pos = *pos_set.begin();
    std::vector<cudf::size_type> dummy_pos(std::next(pos_set.begin()), pos_set.end());
    return surrogate_emit_plan::side_plan{rowid_pos, std::move(dummy_pos)};
  };
  return surrogate_emit_plan{make_side(positions[join_side::left]),
                             make_side(positions[join_side::right]),
                             std::move(store)};
}

[[nodiscard]] std::shared_ptr<surrogate_restore_plan const> build_restore_plan(
  sirius_physical_grouped_aggregate const& agg,
  traced_keys const& traced,
  deferred_positions const& positions,
  std::shared_ptr<surrogate_deferral_store> store,
  sirius::operator_params const& op_params)
{
  std::vector<surrogate_restore_plan::restore_group> groups;
  for (auto const side : {join_side::left, join_side::right}) {
    auto const& pos_set = positions[side];
    if (pos_set.empty()) { continue; }
    auto const rowid_pos = *pos_set.begin();
    std::optional<int> rowid_key_slot;
    std::vector<surrogate_restore_plan::restored_key> keys;
    for (auto const& [slot, trace] : traced.traces) {
      if (trace.side != side) { continue; }
      if (!rowid_key_slot && trace.side_out_pos == rowid_pos) {
        rowid_key_slot = static_cast<int>(slot);
      }
      keys.push_back(surrogate_restore_plan::restored_key{
        static_cast<int>(slot), trace.source_input_col, agg.types[slot]});
    }
    if (!rowid_key_slot) {
      // Every non-empty side's rowid position was chosen from its own traces, so at least one
      // traced key must sit on it.
      throw sirius::internal_exception(
        "groupby_surrogate_keys: no traced key occupies the chosen rowid position on the {} side "
        "(corrupt trace)",
        sirius::op::to_string(side));
    }
    groups.emplace_back(side, *rowid_key_slot, std::move(keys));
  }

  std::vector<int> real_key_slots;
  auto const is_deferred_slot = [&traced](std::size_t slot) {
    return std::ranges::any_of(traced.traces,
                               [slot](auto const& entry) { return entry.first == slot; });
  };
  for (std::size_t slot = 0; slot < agg.group_idx.size(); ++slot) {
    if (!is_deferred_slot(slot)) { real_key_slots.push_back(static_cast<int>(slot)); }
  }

  return std::make_shared<surrogate_restore_plan const>(
    std::move(store),
    std::move(groups),
    std::move(real_key_slots),
    agg.types,
    op_params.groupby_surrogate_unique_fastpath);
}

//===----------------------------------------------------------------------===//
// The ONLY mutating step, unreachable until every gate passed
//===----------------------------------------------------------------------===//

/// Patch the declared schemas along every traced path (group-by slot, hops above the deferral
/// join, the join's own output slot — same order as the traces), log the activation, and
/// install the plan values on the join and the aggregate. No failure paths besides the
/// corrupt-trace throws in patch_slot_type — all validation is done.
void commit_rewrite(sirius_physical_grouped_aggregate& agg,
                    sirius_physical_hash_join& join,
                    traced_keys const& traced,
                    surrogate_emit_plan emit,
                    std::shared_ptr<surrogate_restore_plan const> restore)
{
  auto const bigint_type  = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto const tinyint_type = sirius::logical_type::make(sirius::type_id::TINYINT);
  for (auto const& [slot, trace] : traced.traces) {
    bool const is_rowid  = emit.side(trace.side)->rowid_out_pos() == trace.side_out_pos;
    auto const& new_type = is_rowid ? bigint_type : tinyint_type;
    // Group-by output slot.
    patch_slot_type(agg, static_cast<cudf::size_type>(slot), new_type);
    // Operators between the group-by child (inclusive) and the deferral join (exclusive).
    for (auto const& hop : trace.hops_above) {
      patch_slot_type(*hop.op, hop.out_pos, new_type);
    }
    // The deferral join's own output slot.
    patch_slot_type(join, trace.join_out_pos, new_type);
  }

  auto const& left  = emit.side(join_side::left);
  auto const& right = emit.side(join_side::right);
  SIRIUS_LOG_INFO(
    "groupby_surrogate_keys: deferring {} string group key slot(s) of a HASH_GROUP_BY to an "
    "upstream hash join [left rowid out-pos: {}, right rowid out-pos: {}]",
    traced.traces.size(),
    left ? std::to_string(left->rowid_out_pos()) : std::string("-"),
    right ? std::to_string(right->rowid_out_pos()) : std::string("-"));

  join.install_surrogate_emit(std::move(emit));
  agg.install_surrogate_restore(std::move(restore));
}

/// Attempt the rewrite on one HASH_GROUP_BY. Returns true when applied.
bool try_rewrite_group_by(sirius_physical_grouped_aggregate& agg,
                          const sirius::operator_params& op_params)
{
  // Operator ids are not assigned until pipeline conversion, so plan-time logs must not call
  // get_operator_id() (it throws on the unassigned sentinel).
  if (!passes_structural_gates(agg, op_params)) { return false; }
  auto const string_slots = collect_string_key_slots(agg, op_params);
  if (!string_slots) { return false; }
  if (!no_aggregate_consumes(agg, *string_slots)) { return false; }
  auto const traced = trace_all_to_common_join(agg, *string_slots);
  if (!traced) { return false; }
  auto const positions = choose_deferred_positions(traced->traces);
  if (!passes_rowid_addressing_gate(*traced->deferral_join, positions)) { return false; }

  auto store   = std::make_shared<surrogate_deferral_store>();
  auto emit    = build_emit_plan(positions, store);
  auto restore = build_restore_plan(agg, *traced, positions, std::move(store), op_params);
  commit_rewrite(agg, *traced->deferral_join, *traced, std::move(emit), std::move(restore));
  return true;
}

void rewrite_eligible_group_bys(duckdb::unique_ptr<sirius_physical_operator>& slot,
                                const sirius::operator_params& op_params)
{
  if (!slot) { return; }
  for (auto& child : slot->children) {
    rewrite_eligible_group_bys(child, op_params);
  }
  // DELIM JOIN internal subtrees (join / distinct_root) are deliberately NOT visited: their
  // group-bys have bespoke merge wiring.
  if (slot->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    try_rewrite_group_by(slot->Cast<sirius_physical_grouped_aggregate>(), op_params);
  }
}

}  // namespace

void apply_groupby_surrogate_keys(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan,
                                  const sirius::operator_params& op_params)
{
  if (!plan) { return; }
  if (!op_params.groupby_surrogate_keys) {
    SIRIUS_LOG_DEBUG("groupby_surrogate_keys: disabled by setting");
    return;
  }
  rewrite_eligible_group_bys(plan, op_params);
}

void apply_groupby_surrogate_keys(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan,
                                  duckdb::ClientContext& context)
{
  if (!plan || !context.registered_state) { return; }
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return; }
  // The retained source batches live on the GPU that ran the deferral join; the merge-side
  // gather must run on the same device. Single-GPU only for now.
  std::vector<int> gpu_ids;
  for (auto const* space :
       sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
    if (space != nullptr) { gpu_ids.push_back(space->get_device_id()); }
  }
  std::sort(gpu_ids.begin(), gpu_ids.end());
  gpu_ids.erase(std::unique(gpu_ids.begin(), gpu_ids.end()), gpu_ids.end());
  if (gpu_ids.size() != 1) {
    SIRIUS_LOG_DEBUG("groupby_surrogate_keys: disabled ({} GPUs; single-GPU only)", gpu_ids.size());
    return;
  }
  apply_groupby_surrogate_keys(plan, sirius_ctx->get_config().get_operator_params());
}

}  // namespace sirius::planner
