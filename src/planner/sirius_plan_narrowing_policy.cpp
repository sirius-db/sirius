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

#include "duckdb/common/assert.hpp"
#include "expression/ast/constant_range.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/utils.hpp"
#include "helper/numeric_narrowing.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/sirius_plan_compressed_schema.hpp"

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// The tier narrowing policy: the analysis-only first pass of `apply_compressed_schema_passes`. It
// walks the pre-propagation tree with the same traversal skeleton and per-operator column maps as
// `propagate_compressed_schema`, tracing each candidate narrow scan column (a non-native sidecar
// entry of a scan marked `sidecar_from_gpu_tier_pin`) upward and accumulating four use flags per
// column: transport (survives into a hash-join payload output or an eligible grouped-aggregate key
// output), narrow_comparison (engages a narrow-domain comparison or BETWEEN), evaluator_restore
// (any other reference occurrence inside an expression), and boundary_restore (dies where
// propagation would insert a restore projection: join keys, value-sensitive aggregate inputs,
// ineligible or unmodeled operators, entry into a DELIM_JOIN sub-root, the plan root). The verdict
// per column is `keep iff transport or (not boundary_restore and (narrow_comparison or not
// evaluator_restore))`.
//
// narrow_comparison covers both comparison shapes the evaluator computes at the narrow width:
// against constants, and between two references sharing a carrier. The reference pair is the
// weaker of the two, because it holds only while BOTH carriers stay narrow and equal -- so in
// principle one side retracting could strand the other. No verdict coupling enforces that,
// because no reachable shape produces it: every modeled operator that boundaries one side either
// boundaries the other or gives it transport, and transport keeps it on its own merits. Were a
// stranded side to appear anyway, the evaluator simply declines the pair and restores both, which
// costs a widening cast rather than correctness.
//
// Retracted columns flip back to native in the scan sidecar before
// propagation runs. Because every case mirrors the propagation case of the same operator, an
// operator this file does not model is by construction one where propagation restores at the
// boundary. The conservative default is always native.

namespace sirius::planner {

namespace {

// One candidate narrow scan target: its identity plus the use flags accumulated over the walk.
struct narrow_candidate {
  sirius::op::sirius_physical_table_scan* scan;
  std::size_t column_index;  ///< Position in the scan output (and its sidecar).
  cudf::data_type carrier;   ///< The planned narrow carrier from the scan sidecar.
  bool transport         = false;
  bool boundary_restore  = false;
  bool evaluator_restore = false;
  bool narrow_comparison = false;
};

struct policy_state {
  // Seeded per scan in output-column order, so one scan's candidates are contiguous.
  std::vector<narrow_candidate> candidates;
};

// Output position -> candidate index for one subtree, sized to the operator's output width; an
// empty slot means that output column carries no candidate narrow carrier.
using carried_columns = std::vector<std::optional<std::size_t>>;

bool is_delim_join(sirius::op::sirius_physical_operator const& op) noexcept
{
  return op.type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
         op.type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN;
}

void mark_boundary_restore(policy_state& state, carried_columns const& columns)
{
  for (auto const& candidate : columns) {
    if (candidate) { state.candidates[*candidate].boundary_restore = true; }
  }
}

// Plan-time probe of the narrow-domain eligibility predicate shared with the evaluator,
// evaluated against the column's planned carrier instead of a runtime input column.
bool narrow_domain_engages(sirius::ast::reference const& ref,
                           cudf::data_type carrier,
                           std::initializer_list<sirius::ast::node const*> constant_operands)
{
  return sirius::ast::narrow_domain_carrier_eligible(ref.return_type(), carrier, constant_operands);
}

// The candidate carried by @p column_operand when it is a bare carried reference and
// @p constant_operands engage the narrow domain against its planned carrier.
std::optional<std::size_t> narrow_comparison_candidate(
  sirius::ast::node const* column_operand,
  std::initializer_list<sirius::ast::node const*> constant_operands,
  carried_columns const& inputs,
  policy_state const& state)
{
  if (!column_operand || !column_operand->holds<sirius::ast::reference>()) { return std::nullopt; }
  auto const& ref = column_operand->get<sirius::ast::reference>();
  if (ref.column_index >= inputs.size() || !inputs[ref.column_index]) { return std::nullopt; }
  auto const candidate = *inputs[ref.column_index];
  if (!narrow_domain_engages(ref, state.candidates[candidate].carrier, constant_operands)) {
    return std::nullopt;
  }
  return candidate;
}

// The two candidates of a reference-vs-reference comparison that both sit on the same planned
// carrier. Mirrors the evaluator's `narrow_domain_reference_pair_carrier` so plan-time keep/retract
// and evaluation-time widening cannot disagree.
std::optional<std::pair<std::size_t, std::size_t>> narrow_comparison_pair(
  sirius::ast::node const* lhs,
  sirius::ast::node const* rhs,
  carried_columns const& inputs,
  policy_state const& state)
{
  if (!lhs || !rhs || !lhs->holds<sirius::ast::reference>() ||
      !rhs->holds<sirius::ast::reference>()) {
    return std::nullopt;
  }
  auto const& lhs_ref = lhs->get<sirius::ast::reference>();
  auto const& rhs_ref = rhs->get<sirius::ast::reference>();
  if (lhs_ref.column_index >= inputs.size() || rhs_ref.column_index >= inputs.size()) {
    return std::nullopt;
  }
  if (!inputs[lhs_ref.column_index] || !inputs[rhs_ref.column_index]) { return std::nullopt; }
  auto const lhs_candidate = *inputs[lhs_ref.column_index];
  auto const rhs_candidate = *inputs[rhs_ref.column_index];
  if (!sirius::ast::narrow_domain_reference_pair_eligible(
        lhs_ref.return_type(),
        state.candidates[lhs_candidate].carrier,
        rhs_ref.return_type(),
        state.candidates[rhs_candidate].carrier)) {
    return std::nullopt;
  }
  return std::pair{lhs_candidate, rhs_candidate};
}

void classify_expression(sirius::ast::node const& expr,
                         carried_columns const& inputs,
                         policy_state& state);

void classify_child(std::unique_ptr<sirius::ast::node> const& child,
                    carried_columns const& inputs,
                    policy_state& state)
{
  if (child) { classify_expression(*child, inputs, state); }
}

// Classify every reference occurrence in an evaluated expression against the child map @p inputs.
// A comparison or BETWEEN whose shape engages the narrow domain marks its column
// narrow_comparison and classifies nothing further (the evaluator computes it at the narrow width
// with zero restoration); every other reference occurrence marks evaluator_restore, because
// `evaluate(reference)` restores a narrowed column unconditionally. The static_assert keeps the
// classification total: a new AST alternative fails to compile until it is given an arm here.
void classify_expression(sirius::ast::node const& expr,
                         carried_columns const& inputs,
                         policy_state& state)
{
  std::visit(
    [&](auto const& alt) {
      using T = std::decay_t<decltype(alt)>;

      if constexpr (std::is_same_v<T, sirius::ast::reference>) {
        if (alt.column_index < inputs.size() && inputs[alt.column_index]) {
          state.candidates[*inputs[alt.column_index]].evaluator_restore = true;
        }
      } else if constexpr (std::is_same_v<T, sirius::ast::constant>) {
        // leaf, no references
      } else if constexpr (std::is_same_v<T, sirius::ast::comparison>) {
        // Either operand order, mirroring the evaluator's two-sided probe.
        if (auto const candidate =
              narrow_comparison_candidate(alt.left.get(), {alt.right.get()}, inputs, state)) {
          state.candidates[*candidate].narrow_comparison = true;
          return;
        }
        if (auto const candidate =
              narrow_comparison_candidate(alt.right.get(), {alt.left.get()}, inputs, state)) {
          state.candidates[*candidate].narrow_comparison = true;
          return;
        }
        // Both operands carried narrow at one shared carrier: the comparison needs no restore, so
        // vote to keep both and stop -- descending would mark each operand evaluator_restore and
        // widen a pair that never had to widen.
        //
        // If a later verdict retracts one carrier, runtime eligibility declines the pair and
        // restores the remaining narrow input. That can miscost the keep decision, not the result.
        if (auto const pair =
              narrow_comparison_pair(alt.left.get(), alt.right.get(), inputs, state)) {
          state.candidates[pair->first].narrow_comparison  = true;
          state.candidates[pair->second].narrow_comparison = true;
          return;
        }
        classify_child(alt.left, inputs, state);
        classify_child(alt.right, inputs, state);
      } else if constexpr (std::is_same_v<T, sirius::ast::between>) {
        if (auto const candidate = narrow_comparison_candidate(
              alt.input.get(), {alt.lower.get(), alt.upper.get()}, inputs, state)) {
          state.candidates[*candidate].narrow_comparison = true;
          return;
        }
        classify_child(alt.input, inputs, state);
        classify_child(alt.lower, inputs, state);
        classify_child(alt.upper, inputs, state);
      } else if constexpr (std::is_same_v<T, sirius::ast::conjunction>) {
        for (auto const& child : alt.children) {
          classify_child(child, inputs, state);
        }
      } else if constexpr (std::is_same_v<T, sirius::ast::case_expr>) {
        for (auto const& when_then : alt.cases) {
          classify_child(when_then.when_, inputs, state);
          classify_child(when_then.then_, inputs, state);
        }
        classify_child(alt.else_, inputs, state);
      } else if constexpr (std::is_same_v<T, sirius::ast::cast>) {
        classify_child(alt.child, inputs, state);
      } else if constexpr (std::is_same_v<T, sirius::ast::unary_op>) {
        classify_child(alt.child, inputs, state);
      } else if constexpr (std::is_same_v<T, sirius::ast::coalesce>) {
        for (auto const& child : alt.children) {
          classify_child(child, inputs, state);
        }
      } else if constexpr (std::is_same_v<T, sirius::ast::in_list>) {
        classify_child(alt.probe, inputs, state);
        for (auto const& child : alt.values) {
          classify_child(child, inputs, state);
        }
      } else if constexpr (std::is_same_v<T, sirius::ast::function_call>) {
        for (auto const& child : alt.arguments()) {
          classify_child(child, inputs, state);
        }
      } else if constexpr (std::is_same_v<T, sirius::ast::aggregate>) {
        for (auto const& child : alt.arguments()) {
          classify_child(child, inputs, state);
        }
      } else {
        static_assert(sizeof(T) == 0,
                      "Unhandled sirius::ast alternative in the tier narrowing classifier -- "
                      "add an arm mirroring how the evaluator treats the new node kind");
      }
    },
    expr.v);
}

carried_columns analyze_subtree(sirius::op::sirius_physical_operator& op, policy_state& state);

// Analyze one DELIM_JOIN sub-tree. Propagation restores every child of @p sub_root in place and
// clears the sub-root's own sidecar, so the sub-root's operator case never runs here: it can
// award no transport and forward no carrier, and every carried column reaching the sub-root's
// input boundary is a boundary restore. A TABLE_SCAN sub-root has no inputs; its own seeded
// outputs die at the cleared sub-root output instead. A DELIM_JOIN sub-root recurses into its
// internal sub-trees the same way.
void analyze_delim_subtree(sirius::op::sirius_physical_operator& sub_root, policy_state& state)
{
  if (sub_root.type == sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    mark_boundary_restore(state, analyze_subtree(sub_root, state));
    return;
  }
  for (auto& child : sub_root.children) {
    if (child) { mark_boundary_restore(state, analyze_subtree(*child, state)); }
  }
  if (is_delim_join(sub_root)) {
    auto& delim = sub_root.Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join) { analyze_delim_subtree(*delim.join, state); }
    if (delim.distinct_root) { analyze_delim_subtree(*delim.distinct_root, state); }
  }
}

// Bottom-up walk returning the carried-column map of @p op 's output. Each case mirrors the
// propagation case of the same operator: a use is recorded on the origin candidate, then the map
// is transformed the way propagation forwards carriers. Falling out of the switch is the native
// boundary, exactly where propagation restores every narrowed child.
carried_columns analyze_subtree(sirius::op::sirius_physical_operator& op, policy_state& state)
{
  std::vector<carried_columns> child_maps;
  child_maps.reserve(op.children.size());
  for (auto& child : op.children) {
    child_maps.push_back(child ? analyze_subtree(*child, state) : carried_columns{});
  }

  if (is_delim_join(op)) {
    auto& delim = op.Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join) { analyze_delim_subtree(*delim.join, state); }
    if (delim.distinct_root) { analyze_delim_subtree(*delim.distinct_root, state); }
  }

  // The policy runs before propagation, so scans own the only sidecars in the tree.
  D_ASSERT(op.type == sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN ||
           !op.has_physical_overrides());

  switch (op.type) {
    case sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN: {
      // Dynamic-filter targets are not modeled: the propagation TABLE_SCAN case forces them
      // native after this pass. Both passes only flip narrow to native, so the outcomes commute;
      // a target this pass keeps may still end native (and is then counted as kept — the same
      // caveat scan_sidecars_installed documents). Targets are probe keys, which the HASH_JOIN
      // case marks boundary restores anyway.
      auto& scan = op.Cast<sirius::op::sirius_physical_table_scan>();
      carried_columns map(op.types.size());
      if (!scan.sidecar_from_gpu_tier_pin || !op.has_physical_overrides()) { return map; }
      auto const native    = native_physical_schema(op);
      auto const& physical = op.get_physical_types();
      D_ASSERT(native.size() == physical.size());
      if (native.size() != physical.size()) { return map; }
      for (std::size_t column_idx = 0; column_idx < physical.size(); ++column_idx) {
        if (physical[column_idx] == native[column_idx]) { continue; }
        map[column_idx] = state.candidates.size();
        state.candidates.push_back({&scan, column_idx, physical[column_idx]});
      }
      return map;
    }

    case sirius::op::SiriusPhysicalOperatorType::FILTER: {
      if (op.children.size() != 1) { break; }
      auto const& filter = op.Cast<sirius::op::sirius_physical_filter>();
      auto const& inputs = child_maps[0];
      carried_columns map;
      if (std::holds_alternative<sirius::op::passthrough>(filter.output_columns)) {
        if (op.children[0]->types.size() != op.types.size()) { break; }
        map = inputs;
      } else {
        auto const& output_indices = std::get<std::vector<cudf::size_type>>(filter.output_columns);
        if (output_indices.size() != op.types.size()) { break; }
        map.resize(output_indices.size());
        bool consistent = true;
        for (std::size_t output_idx = 0; output_idx < output_indices.size(); ++output_idx) {
          auto const input_idx = output_indices[output_idx];
          if (input_idx < 0 || static_cast<std::size_t>(input_idx) >= inputs.size()) {
            consistent = false;
            break;
          }
          map[output_idx] = inputs[static_cast<std::size_t>(input_idx)];
        }
        if (!consistent) { break; }
      }
      // Predicate references index the filter's input columns. Carried columns the output mask
      // drops die free: propagation inserts no restore for them.
      if (filter.expression) { classify_expression(*filter.expression, inputs, state); }
      return map;
    }

    case sirius::op::SiriusPhysicalOperatorType::PROJECTION: {
      if (op.children.size() != 1) { break; }
      auto const& projection = op.Cast<sirius::op::sirius_physical_projection>();
      if (projection.select_list.size() != op.types.size()) { break; }
      auto const& inputs = child_maps[0];
      carried_columns map(op.types.size());
      for (std::size_t output_idx = 0; output_idx < projection.select_list.size(); ++output_idx) {
        auto const& expression = projection.select_list[output_idx];
        if (!expression) { continue; }
        if (expression->holds<sirius::ast::reference>()) {
          // A bare-reference output forwards the carrier without evaluation.
          auto const input_idx = expression->get<sirius::ast::reference>().column_index;
          if (input_idx < inputs.size()) { map[output_idx] = inputs[input_idx]; }
          continue;
        }
        classify_expression(*expression, inputs, state);
      }
      return map;
    }

    case sirius::op::SiriusPhysicalOperatorType::LIMIT:
    case sirius::op::SiriusPhysicalOperatorType::STREAMING_LIMIT: {
      if (op.children.size() != 1 || op.children[0]->types != op.types) { break; }
      return child_maps[0];
    }

    case sirius::op::SiriusPhysicalOperatorType::HASH_JOIN: {
      if (op.children.size() != 2) { break; }
      auto const& join = op.Cast<sirius::op::sirius_physical_hash_join>();
      auto lhs         = child_maps[0];
      auto rhs         = child_maps[1];
      // Key columns are restored below the join before the output mapping, so they leave the map
      // with a boundary restore and can never earn transport through the join output.
      auto mark_keys = [&state](sirius::ast::node const* condition_side, carried_columns& side) {
        if (!condition_side) { return; }
        sirius::ast::visit_references(
          *condition_side, [&state, &side](sirius::ast::reference const& ref) {
            if (ref.column_index < side.size() && side[ref.column_index]) {
              state.candidates[*side[ref.column_index]].boundary_restore = true;
              side[ref.column_index]                                     = std::nullopt;
            }
          });
      };
      for (auto const& condition : join.conditions) {
        mark_keys(condition.left.get(), lhs);
        mark_keys(condition.right.get(), rhs);
      }

      carried_columns map(op.types.size());
      std::size_t output_idx  = 0;
      bool const collect_left = join.join_type != duckdb::JoinType::RIGHT_SEMI &&
                                join.join_type != duckdb::JoinType::RIGHT_ANTI;
      bool const collect_right = join.join_type != duckdb::JoinType::SEMI &&
                                 join.join_type != duckdb::JoinType::ANTI &&
                                 join.join_type != duckdb::JoinType::MARK;
      auto forward_payloads = [&state, &map, &output_idx](
                                std::vector<cudf::size_type> const& col_idxs,
                                carried_columns const& side) {
        for (auto const input_idx : col_idxs) {
          if (output_idx >= map.size()) { break; }
          if (input_idx >= 0 && static_cast<std::size_t>(input_idx) < side.size() &&
              side[static_cast<std::size_t>(input_idx)]) {
            auto const candidate                  = *side[static_cast<std::size_t>(input_idx)];
            state.candidates[candidate].transport = true;
            map[output_idx]                       = candidate;
          }
          ++output_idx;
        }
      };
      if (collect_left) { forward_payloads(join.lhs_output_columns.col_idxs, lhs); }
      if (collect_right) { forward_payloads(join.rhs_output_columns.col_idxs, rhs); }
      // Carried columns absent from the output maps die free.
      return map;
    }

    case sirius::op::SiriusPhysicalOperatorType::DENSE_COUNT_JOIN: {
      if (op.children.size() != 2) { break; }
      auto const& join             = op.Cast<sirius::op::sirius_physical_dense_count_join>();
      auto const preserved_key_idx = join.preserved_key_idx();
      auto const counted_key_idx   = join.counted_key_idx();
      if (preserved_key_idx >= child_maps[0].size() || counted_key_idx >= child_maps[1].size() ||
          (join.counted_value_idx() && *join.counted_value_idx() >= child_maps[1].size())) {
        break;
      }

      // Keys require native values; COUNT(col) uses only its validity mask. Output is native
      // [key, BIGINT] with no forwarded carrier.
      auto mark_key = [&state](carried_columns const& side, std::size_t key_idx) {
        if (side[key_idx]) { state.candidates[*side[key_idx]].boundary_restore = true; }
      };
      mark_key(child_maps[0], preserved_key_idx);
      mark_key(child_maps[1], counted_key_idx);
      return carried_columns(op.types.size());
    }

    case sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY: {
      // The exact eligibility ladder of the propagation case; any failing precondition falls
      // through to the native boundary below.
      if (op.children.size() != 1) { break; }
      auto const& aggregate = op.Cast<sirius::op::sirius_physical_grouped_aggregate>();
      if (aggregate.grouping_sets.size() > 1) { break; }
      if (aggregate.has_avg || aggregate.has_count_distinct) { break; }
      if (op.types.size() != aggregate.group_idx.size() + aggregate.aggregate_slots.size()) {
        break;
      }
      auto const child_width = op.children[0]->types.size();
      bool consistent        = true;
      for (auto const key_idx : aggregate.group_idx) {
        if (key_idx < 0 || static_cast<std::size_t>(key_idx) >= child_width) {
          consistent = false;
          break;
        }
      }
      if (!consistent) { break; }

      // Value-sensitive aggregate inputs are restored below the aggregate; COUNT_ALL carries a
      // placeholder input index and COUNT_VALID reads only the validity mask, so neither
      // constrains its input. A column that is both group key and value-sensitive input leaves
      // the map here and therefore goes native, like propagation's restored-then-forwarded key.
      auto inputs               = child_maps[0];
      auto mark_aggregate_input = [&state, &inputs](std::size_t child_idx) {
        if (child_idx < inputs.size() && inputs[child_idx]) {
          state.candidates[*inputs[child_idx]].boundary_restore = true;
          inputs[child_idx]                                     = std::nullopt;
        }
      };
      for (std::size_t i = 0; i < aggregate.cudf_aggregates.size(); ++i) {
        if (aggregate.cudf_aggregates[i] == cudf::aggregation::Kind::COUNT_ALL ||
            aggregate.cudf_aggregates[i] == cudf::aggregation::Kind::COUNT_VALID) {
          continue;
        }
        if (i < aggregate.cudf_aggregate_struct_col_indices.size() &&
            !aggregate.cudf_aggregate_struct_col_indices[i].empty()) {
          for (auto const struct_idx : aggregate.cudf_aggregate_struct_col_indices[i]) {
            if (struct_idx >= 0) { mark_aggregate_input(static_cast<std::size_t>(struct_idx)); }
          }
          continue;
        }
        if (i < aggregate.cudf_aggregate_idx.size() && aggregate.cudf_aggregate_idx[i] >= 0) {
          mark_aggregate_input(static_cast<std::size_t>(aggregate.cudf_aggregate_idx[i]));
        }
      }

      // Group keys earn transport: the aggregate's PARTITION/MERGE_GROUP_BY exchange moves narrow
      // key bytes. Other carried columns are unused by the aggregate and die free.
      carried_columns map(op.types.size());
      for (std::size_t key_pos = 0; key_pos < aggregate.group_idx.size(); ++key_pos) {
        if (key_pos >= map.size()) { break; }
        auto const child_idx = static_cast<std::size_t>(aggregate.group_idx[key_pos]);
        if (child_idx < inputs.size() && inputs[child_idx]) {
          state.candidates[*inputs[child_idx]].transport = true;
          map[key_pos]                                   = inputs[child_idx];
        }
      }
      return map;
    }

    default: break;
  }

  // Native boundary: propagation restores every narrowed child here, so every carried child
  // column is a boundary restore. This is the totality guarantee for unmodeled operators.
  for (auto const& child_map : child_maps) {
    mark_boundary_restore(state, child_map);
  }
  return carried_columns(op.types.size());
}

}  // namespace

std::size_t apply_tier_narrowing_policy(sirius::op::sirius_physical_operator& plan)
{
  policy_state state;
  // Carriers surviving to the plan root meet the final-result restore.
  mark_boundary_restore(state, analyze_subtree(plan, state));

  std::size_t total_retracted = 0;
  for (std::size_t begin = 0; begin < state.candidates.size();) {
    auto* scan      = state.candidates[begin].scan;
    std::size_t end = begin;
    while (end < state.candidates.size() && state.candidates[end].scan == scan) {
      ++end;
    }

    D_ASSERT(scan->has_physical_overrides());
    auto const native = native_physical_schema(*scan);
    auto physical     = scan->get_physical_types();
    if (native.size() != physical.size()) {
      begin = end;
      continue;
    }
    std::size_t retracted = 0;
    for (std::size_t i = begin; i < end; ++i) {
      auto const& candidate = state.candidates[i];
      bool const keep =
        candidate.transport || (!candidate.boundary_restore &&
                                (candidate.narrow_comparison || !candidate.evaluator_restore));
      if (keep) { continue; }
      physical[candidate.column_index] = native[candidate.column_index];
      ++retracted;
    }
    if (retracted > 0) {
      install_physical_schema(*scan, std::move(physical));
      total_retracted += retracted;
    }
    begin = end;
  }

  return total_retracted;
}

}  // namespace sirius::planner
