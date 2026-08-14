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
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "sirius_context.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

namespace sirius::planner {

namespace {

using sirius::op::sirius_physical_grouped_aggregate;
using sirius::op::sirius_physical_hash_join;
using sirius::op::sirius_physical_operator;
using sirius::op::sirius_physical_projection;
using sirius::op::SiriusPhysicalOperatorType;

/// One operator on a traced key's path, with the key's column position at that operator's
/// OUTPUT. Used to patch declared types (and projection reference nodes / physical sidecars)
/// once a rewrite is committed.
struct trace_hop {
  sirius_physical_operator* op = nullptr;
  cudf::size_type out_col      = -1;
};

/// Where one string group key resolves to: the deepest INNER hash join it passes through.
struct key_trace {
  bool ok = false;
  sirius_physical_hash_join* join = nullptr;
  bool from_left                  = true;  ///< side of `join` the column comes from
  cudf::size_type side_out_pos    = -1;    ///< position within that side's output column list
  cudf::size_type join_out_pos    = -1;    ///< position within the join's full output
  cudf::size_type source_input_col = -1;   ///< column index within the side's child schema
  /// Operators strictly ABOVE the deferral join (deepest first excluded), each with the key's
  /// output position there. The deferral join itself is patched via side_out_pos/join_out_pos.
  std::vector<trace_hop> hops_above;
};

/// Trace a group-by input column down through pass-through operators to the deepest INNER
/// hash join it crosses. Returns ok=false when the column is not a pure pass-through of a
/// join side (computed projections, unsupported operators, non-INNER joins).
key_trace trace_string_key(sirius_physical_operator* start, cudf::size_type start_col)
{
  key_trace result;
  std::vector<trace_hop> hops;  // every op visited so far, deepest last
  sirius_physical_operator* cur = start;
  cudf::size_type col           = start_col;

  while (cur != nullptr) {
    hops.push_back(trace_hop{cur, col});
    if (cur->type == SiriusPhysicalOperatorType::PROJECTION) {
      auto& proj = cur->Cast<sirius_physical_projection>();
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
      bool const from_left = col < num_lhs;
      auto const side_pos  = from_left ? col : col - num_lhs;
      auto const src_col   = from_left
                               ? hj.lhs_output_columns.col_idxs[static_cast<std::size_t>(side_pos)]
                               : hj.rhs_output_columns.col_idxs[static_cast<std::size_t>(side_pos)];
      // Record this join as the (so far) deepest deferral candidate. Everything visited so far
      // EXCEPT this join is "above" it.
      result.ok               = true;
      result.join             = &hj;
      result.from_left        = from_left;
      result.side_out_pos     = side_pos;
      result.join_out_pos     = col;
      result.source_input_col = src_col;
      result.hops_above.assign(hops.begin(), hops.end() - 1);
      // Descend further only when this join does not READ the traced column: a deeper deferral
      // would hand this join a rowid where its key comparison (or a mixed-join AST predicate,
      // which may reference any input column) expects the string. As the deferral join itself
      // it is fine — emission changes, key inputs do not.
      if (!hj.all_conditions_are_equality_keys()) { break; }
      auto const& key_cols =
        from_left ? hj.get_left_key_col_indices() : hj.get_right_key_col_indices();
      if (std::find(key_cols.begin(), key_cols.end(), src_col) != key_cols.end()) { break; }
      std::size_t const child_idx = from_left ? 0 : 1;
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
bool traced_column_is_sole_consumer(key_trace const& trace)
{
  for (auto const& hop : trace.hops_above) {
    if (hop.op->type == SiriusPhysicalOperatorType::PROJECTION) {
      auto& proj = hop.op->Cast<sirius_physical_projection>();
      auto const traced_input =
        proj.select_list[static_cast<std::size_t>(hop.out_col)]->get<sirius::ast::reference>()
          .column_index;
      for (std::size_t o = 0; o < proj.select_list.size(); ++o) {
        if (static_cast<cudf::size_type>(o) == hop.out_col) { continue; }
        auto const* entry = proj.select_list[o].get();
        if (!entry) { continue; }
        bool references_traced = false;
        sirius::ast::visit_references(*entry, [&](sirius::ast::reference const& ref) {
          if (ref.column_index == traced_input) { references_traced = true; }
        });
        if (references_traced) { return false; }
      }
    } else if (hop.op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
      auto& hj           = hop.op->Cast<sirius_physical_hash_join>();
      auto const num_lhs = static_cast<cudf::size_type>(hj.lhs_output_columns.col_idxs.size());
      bool const from_left = hop.out_col < num_lhs;
      auto const side_pos  = from_left ? hop.out_col : hop.out_col - num_lhs;
      auto const& col_idxs =
        from_left ? hj.lhs_output_columns.col_idxs : hj.rhs_output_columns.col_idxs;
      auto const src_col = col_idxs[static_cast<std::size_t>(side_pos)];
      if (std::count(col_idxs.begin(), col_idxs.end(), src_col) != 1) { return false; }
    }
  }
  return true;
}

/// Patch one operator's declared schema (and projection reference node / physical sidecar) so
/// position `pos` carries `new_type` instead of the original string type.
void patch_slot_type(sirius_physical_operator& op, cudf::size_type pos,
                     sirius::logical_type const& new_type)
{
  auto const upos = static_cast<std::size_t>(pos);
  if (upos < op.types.size()) { op.types[upos] = new_type; }
  if (op.has_physical_overrides()) {
    auto phys = op.get_physical_types();
    if (upos < phys.size()) {
      phys[upos] = sirius::get_cudf_type(new_type);
      op.set_physical_types(std::move(phys));
    }
  }
  if (op.type == SiriusPhysicalOperatorType::PROJECTION) {
    auto& proj = op.Cast<sirius_physical_projection>();
    if (upos < proj.select_list.size() && proj.select_list[upos] &&
        proj.select_list[upos]->holds<sirius::ast::reference>()) {
      auto const idx = proj.select_list[upos]->get<sirius::ast::reference>().column_index;
      proj.select_list[upos] =
        std::make_unique<sirius::ast::node>(sirius::ast::reference(idx, new_type));
    }
  }
  if (op.type == SiriusPhysicalOperatorType::HASH_JOIN) {
    auto& hj           = op.Cast<sirius_physical_hash_join>();
    auto const num_lhs = static_cast<cudf::size_type>(hj.lhs_output_columns.col_idxs.size());
    if (pos < num_lhs) {
      hj.lhs_output_columns.col_types[upos] = new_type;
    } else {
      auto const rhs_pos = static_cast<std::size_t>(pos - num_lhs);
      if (rhs_pos < hj.rhs_output_columns.col_types.size()) {
        hj.rhs_output_columns.col_types[rhs_pos] = new_type;
      }
    }
  }
}

/// Attempt the rewrite on one HASH_GROUP_BY. Returns true when applied.
bool try_rewrite_group_by(sirius_physical_grouped_aggregate& agg,
                          const sirius::operator_params& op_params)
{
  // Operator ids are not assigned until pipeline conversion, so plan-time logs must not call
  // get_operator_id() (it throws on the unassigned sentinel).
  auto const decline = [&](std::string const& reason) {
    SIRIUS_LOG_INFO("groupby_surrogate_keys: declined for a HASH_GROUP_BY: {}", reason);
    return false;
  };

  // ---- Structural gates -------------------------------------------------------------------
  if (agg.surrogate_spec) { return decline("already rewritten"); }
  if (agg.grouping_sets.size() > 1) { return decline("multiple grouping sets"); }
  if (agg.has_count_distinct) {
    return decline("COUNT(DISTINCT) present");  // COLLECT_SET merge does not re-compose here
  }
  if (agg.types.size() != agg.group_idx.size() + agg.aggregate_slots.size()) {
    return decline("non-standard output layout");
  }
  if (agg.children.size() != 1 || !agg.children[0]) { return decline("no single child"); }
  // Aggregates must re-compose under the conservative re-group.
  for (auto kind : agg.cudf_aggregates) {
    switch (kind) {
      case cudf::aggregation::Kind::MIN:
      case cudf::aggregation::Kind::MAX:
      case cudf::aggregation::Kind::SUM:
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: break;
      default: return decline("non-recomposable aggregate kind");
    }
  }
  if (agg.children[0]->estimated_cardinality < op_params.groupby_surrogate_min_rows) {
    return decline("estimated cardinality " +
                   std::to_string(agg.children[0]->estimated_cardinality) +
                   " below groupby_surrogate_min_rows");
  }

  // ---- Identify string key slots ----------------------------------------------------------
  std::vector<std::size_t> string_slots;
  for (std::size_t slot = 0; slot < agg.group_idx.size(); ++slot) {
    if (agg.types[slot].is_varchar()) { string_slots.push_back(slot); }
  }
  if (string_slots.size() < op_params.groupby_surrogate_min_string_keys) {
    return decline(std::to_string(string_slots.size()) +
                   " string key(s) below groupby_surrogate_min_string_keys");
  }
  if (string_slots.size() == agg.group_idx.size()) {
    return decline(
      "all keys are strings");  // need >= 1 real key slot for partitioning / distinct proof
  }

  // No aggregate may consume a deferred child column (it would aggregate rowids).
  for (auto slot : string_slots) {
    int const child_col = agg.group_idx[slot];
    if (std::find(agg.cudf_aggregate_idx.begin(), agg.cudf_aggregate_idx.end(), child_col) !=
        agg.cudf_aggregate_idx.end()) {
      return decline("an aggregate consumes a deferred key column");
    }
    for (auto const& struct_cols : agg.cudf_aggregate_struct_col_indices) {
      if (std::find(struct_cols.begin(), struct_cols.end(), child_col) != struct_cols.end()) {
        return decline("a struct aggregate consumes a deferred key column");
      }
    }
  }

  // ---- Trace every string key to a common deferral join ------------------------------------
  std::vector<key_trace> traces;
  traces.reserve(string_slots.size());
  sirius_physical_hash_join* deferral_join = nullptr;
  for (auto slot : string_slots) {
    auto trace = trace_string_key(agg.children[0].get(),
                                  static_cast<cudf::size_type>(agg.group_idx[slot]));
    if (!trace.ok) {
      return decline("key slot " + std::to_string(slot) + " (child col " +
                     std::to_string(agg.group_idx[slot]) +
                     ") does not trace to an INNER join pass-through");
    }
    if (deferral_join == nullptr) {
      deferral_join = trace.join;
    } else if (deferral_join != trace.join) {
      return decline("string keys trace to different joins");
    }
    traces.push_back(std::move(trace));
  }
  if (deferral_join == nullptr) { return decline("no deferral join found"); }
  if (deferral_join->surrogate_emit) {
    return decline("deferral join already carries a deferral");
  }
  for (auto const& trace : traces) {
    if (!traced_column_is_sole_consumer(trace)) {
      return decline("a deferred column has another consumer above the deferral join");
    }
  }

  // ---- Choose per-side rowid slots --------------------------------------------------------
  // Per join side, the deferred join-output position with the smallest index carries the rowid;
  // the rest become dummies. (Distinct key slots may trace to the same join output position;
  // the position sets below are deduplicated.)
  std::map<cudf::size_type, bool> left_positions, right_positions;  // side_out_pos -> is_rowid
  for (auto const& t : traces) {
    (t.from_left ? left_positions : right_positions).emplace(t.side_out_pos, false);
  }
  if (!left_positions.empty()) { left_positions.begin()->second = true; }
  if (!right_positions.empty()) { right_positions.begin()->second = true; }
  auto const is_rowid_pos = [&](bool from_left, cudf::size_type side_pos) {
    auto const& positions = from_left ? left_positions : right_positions;
    auto const it         = positions.find(side_pos);
    return it != positions.end() && it->second;
  };

  auto const bigint_type  = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto const tinyint_type = sirius::logical_type::make(sirius::type_id::TINYINT);

  // ---- Build the runtime specs -------------------------------------------------------------
  auto store = std::make_shared<sirius::op::surrogate_deferral_store>();

  sirius::op::surrogate_join_emit emit;
  emit.store = store;
  if (!left_positions.empty()) {
    sirius::op::surrogate_join_emit::side side;
    for (auto const& [pos, is_rowid] : left_positions) {
      if (is_rowid) {
        side.rowid_out_pos = pos;
      } else {
        side.dummy_out_pos.push_back(pos);
      }
    }
    emit.left = std::move(side);
  }
  if (!right_positions.empty()) {
    sirius::op::surrogate_join_emit::side side;
    for (auto const& [pos, is_rowid] : right_positions) {
      if (is_rowid) {
        side.rowid_out_pos = pos;
      } else {
        side.dummy_out_pos.push_back(pos);
      }
    }
    emit.right = std::move(side);
  }

  auto spec   = std::make_shared<sirius::op::surrogate_groupby_spec>();
  spec->store = store;
  spec->unique_fastpath       = op_params.groupby_surrogate_unique_fastpath;
  spec->original_output_types = agg.types;

  for (std::size_t side_pass = 0; side_pass < 2; ++side_pass) {
    bool const from_left = (side_pass == 0);
    sirius::op::surrogate_groupby_spec::restore_group group;
    group.from_left = from_left;
    for (std::size_t i = 0; i < traces.size(); ++i) {
      auto const& t = traces[i];
      if (t.from_left != from_left) { continue; }
      auto const slot = static_cast<int>(string_slots[i]);
      if (is_rowid_pos(from_left, t.side_out_pos) && group.rowid_key_slot < 0) {
        group.rowid_key_slot = slot;
      }
      group.restore_key_slots.push_back(slot);
      group.source_input_cols.push_back(t.source_input_col);
      group.restored_types.push_back(agg.types[string_slots[i]]);
    }
    if (group.restore_key_slots.empty()) { continue; }
    if (group.rowid_key_slot < 0) {
      // Defensive: every non-empty side has exactly one rowid position, and at least one traced
      // key sits on it.
      return false;
    }
    spec->groups.push_back(std::move(group));
  }

  for (std::size_t slot = 0; slot < agg.group_idx.size(); ++slot) {
    if (std::find(string_slots.begin(), string_slots.end(), slot) == string_slots.end()) {
      spec->real_key_slots.push_back(static_cast<int>(slot));
    }
  }

  // ---- Commit: patch schemas along every traced path ---------------------------------------
  // (No failure paths below — all validation is done.)
  for (std::size_t i = 0; i < traces.size(); ++i) {
    auto const& t        = traces[i];
    auto const slot      = string_slots[i];
    auto const& new_type = is_rowid_pos(t.from_left, t.side_out_pos) ? bigint_type : tinyint_type;
    // Group-by output slot.
    patch_slot_type(agg, static_cast<cudf::size_type>(slot), new_type);
    // Operators between the group-by child (inclusive) and the deferral join (exclusive).
    for (auto const& hop : t.hops_above) {
      patch_slot_type(*hop.op, hop.out_col, new_type);
    }
    // The deferral join's own output slot.
    patch_slot_type(*deferral_join, t.join_out_pos, new_type);
  }

  SIRIUS_LOG_INFO(
    "groupby_surrogate_keys: deferring {} string group key slot(s) of a HASH_GROUP_BY to an "
    "upstream hash join [left rowid out-pos: {}, right rowid out-pos: {}]",
    string_slots.size(),
    emit.left ? std::to_string(emit.left->rowid_out_pos) : std::string("-"),
    emit.right ? std::to_string(emit.right->rowid_out_pos) : std::string("-"));

  deferral_join->surrogate_emit = std::move(emit);
  agg.surrogate_spec            = std::move(spec);
  return true;
}

void walk(duckdb::unique_ptr<sirius_physical_operator>& slot,
          const sirius::operator_params& op_params)
{
  if (!slot) { return; }
  for (auto& child : slot->children) {
    walk(child, op_params);
  }
  // DELIM JOIN internal subtrees (join / distinct_root) are deliberately NOT visited: their
  // group-bys have bespoke merge wiring.
  if (slot->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    try_rewrite_group_by(slot->Cast<sirius_physical_grouped_aggregate>(), op_params);
  }
}

}  // namespace

void apply_groupby_surrogate_keys(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan,
                                  duckdb::ClientContext& context)
{
  if (!plan || !context.registered_state) { return; }
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return; }
  auto const& op_params = sirius_ctx->get_config().get_operator_params();
  if (!op_params.groupby_surrogate_keys) {
    SIRIUS_LOG_DEBUG("groupby_surrogate_keys: disabled by setting");
    return;
  }
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
    SIRIUS_LOG_INFO("groupby_surrogate_keys: disabled ({} GPUs; single-GPU only)", gpu_ids.size());
    return;
  }
  walk(plan, op_params);
}

}  // namespace sirius::planner
