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

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression/join_condition.hpp"
#include "expression_evaluator/ast_supported_types.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/dynamic_filter/build_filter_evidence.hpp"
#include "planner/dynamic_filter/build_key_domain.hpp"
#include "planner/dynamic_filter/dynamic_filter_key_admission.hpp"
#include "planner/dynamic_filter/dynamic_filter_target_discovery.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace sirius::planner {

/// Returns a set of output column indices proven to form a unique key for the
/// given logical operator subtree, or an empty set if uniqueness cannot be proven.
static std::unordered_set<duckdb::idx_t> prove_unique_columns(duckdb::LogicalOperator& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET: {
      auto& get  = op.Cast<duckdb::LogicalGet>();
      auto table = get.GetTable();
      if (!table) { return {}; }

      const auto& constraints = table->GetConstraints();
      const auto& column_ids  = get.GetColumnIds();
      const auto& proj_ids    = get.projection_ids;

      // Build map: table column logical index → LogicalGet output position.
      std::unordered_map<duckdb::idx_t, duckdb::idx_t> col_to_output;
      if (proj_ids.empty()) {
        for (duckdb::idx_t i = 0; i < column_ids.size(); i++) {
          col_to_output[column_ids[i].GetPrimaryIndex()] = i;
        }
      } else {
        for (duckdb::idx_t i = 0; i < proj_ids.size(); i++) {
          col_to_output[column_ids[proj_ids[i]].GetPrimaryIndex()] = i;
        }
      }

      // Find the smallest PRIMARY KEY constraint whose columns are all present in the output.
      // Only PK is used (not plain UNIQUE) because PK guarantees NOT NULL — a nullable UNIQUE
      // column can contain duplicate NULLs which would violate distinct_hash_join's contract.
      std::unordered_set<duckdb::idx_t> best;
      for (const auto& constraint : constraints) {
        if (constraint->type != duckdb::ConstraintType::UNIQUE) { continue; }
        auto& unique = constraint->Cast<duckdb::UniqueConstraint>();
        if (!unique.IsPrimaryKey()) { continue; }
        auto logical_indexes = unique.GetLogicalIndexes(table->GetColumns());

        std::unordered_set<duckdb::idx_t> candidate;
        bool all_present = true;
        for (const auto& idx : logical_indexes) {
          auto it = col_to_output.find(idx.index);
          if (it == col_to_output.end()) {
            all_present = false;
            break;
          }
          candidate.insert(it->second);
        }
        if (all_present && !candidate.empty() && (best.empty() || candidate.size() < best.size())) {
          best = std::move(candidate);
        }
      }
      return best;
    }

    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
      auto& aggr = op.Cast<duckdb::LogicalAggregate>();
      // Only single grouping set — no CUBE/ROLLUP/GROUPING SETS.
      if (aggr.grouping_sets.size() > 1 || aggr.groups.empty()) { return {}; }
      std::unordered_set<duckdb::idx_t> unique_set;
      for (duckdb::idx_t i = 0; i < aggr.groups.size(); i++) {
        unique_set.insert(i);
      }
      return unique_set;
    }

    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N: {
      // These operators only truncate rows — no column remapping.
      if (op.children.empty()) { return {}; }
      return prove_unique_columns(*op.children[0]);
    }

    case duckdb::LogicalOperatorType::LOGICAL_FILTER: {
      if (op.children.empty()) { return {}; }
      auto child_unique = prove_unique_columns(*op.children[0]);
      if (child_unique.empty()) { return {}; }
      auto& filter = op.Cast<duckdb::LogicalFilter>();
      if (!filter.HasProjectionMap()) { return child_unique; }
      // Remap through projection_map: output[i] = child[projection_map[i]].
      std::unordered_set<duckdb::idx_t> remapped;
      for (duckdb::idx_t i = 0; i < filter.projection_map.size(); i++) {
        if (child_unique.count(filter.projection_map[i])) { remapped.insert(i); }
      }
      return (remapped.size() == child_unique.size()) ? remapped
                                                      : std::unordered_set<duckdb::idx_t>{};
    }

    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: {
      if (op.children.empty()) { return {}; }
      auto child_unique = prove_unique_columns(*op.children[0]);
      if (child_unique.empty()) { return {}; }
      auto& order = op.Cast<duckdb::LogicalOrder>();
      if (!order.HasProjectionMap()) { return child_unique; }
      std::unordered_set<duckdb::idx_t> remapped;
      for (duckdb::idx_t i = 0; i < order.projection_map.size(); i++) {
        if (child_unique.count(order.projection_map[i])) { remapped.insert(i); }
      }
      return (remapped.size() == child_unique.size()) ? remapped
                                                      : std::unordered_set<duckdb::idx_t>{};
    }

    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
      if (op.children.empty()) { return {}; }
      auto child_unique = prove_unique_columns(*op.children[0]);
      if (child_unique.empty()) { return {}; }

      auto& proj = op.Cast<duckdb::LogicalProjection>();
      // Remap child unique indices through projection expressions.
      // Only direct BoundReferenceExpression pass-throughs are safe.
      std::unordered_set<duckdb::idx_t> remapped;
      for (duckdb::idx_t i = 0; i < proj.expressions.size(); i++) {
        auto& expr = proj.expressions[i];
        if (expr->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { continue; }
        auto child_idx = expr->Cast<duckdb::BoundReferenceExpression>().index;
        if (child_unique.count(child_idx)) { remapped.insert(i); }
      }
      // All child unique columns must map through.
      if (remapped.size() == child_unique.size()) { return remapped; }
      return {};
    }

    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
      if (op.children.size() != 2) { return {}; }
      auto& join = op.Cast<duckdb::LogicalComparisonJoin>();

      auto left_unique  = prove_unique_columns(*op.children[0]);
      auto right_unique = prove_unique_columns(*op.children[1]);
      if (left_unique.empty() && right_unique.empty()) { return {}; }

      // Collect equality key columns on each side (only direct column refs).
      std::unordered_set<duckdb::idx_t> left_eq_keys, right_eq_keys;
      for (const auto& c : join.conditions) {
        if (c.comparison != duckdb::ExpressionType::COMPARE_EQUAL) { continue; }
        if (c.left->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          left_eq_keys.insert(c.left->Cast<duckdb::BoundReferenceExpression>().index);
        }
        if (c.right->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          right_eq_keys.insert(c.right->Cast<duckdb::BoundReferenceExpression>().index);
        }
      }

      // unique ⊆ keys → join keys include a unique key, so that side has no duplicates.
      auto unique_subset_of_keys = [](const std::unordered_set<duckdb::idx_t>& unique,
                                      const std::unordered_set<duckdb::idx_t>& keys) {
        if (unique.empty()) { return false; }
        for (auto col : unique) {
          if (!keys.count(col)) { return false; }
        }
        return true;
      };

      bool right_keys_unique = unique_subset_of_keys(right_unique, right_eq_keys);
      bool left_keys_unique  = unique_subset_of_keys(left_unique, left_eq_keys);

      // Which side's row-level uniqueness survives the join?
      bool left_preserved  = false;
      bool right_preserved = false;
      switch (join.join_type) {
        case duckdb::JoinType::INNER:
          // Each left row matches ≤1 right row iff right keys are unique (and vice versa).
          left_preserved  = right_keys_unique;
          right_preserved = left_keys_unique;
          break;
        case duckdb::JoinType::LEFT: left_preserved = right_keys_unique; break;
        case duckdb::JoinType::RIGHT: right_preserved = left_keys_unique; break;
        case duckdb::JoinType::SEMI:
        case duckdb::JoinType::ANTI:
          left_preserved = !left_unique.empty();  // output ⊆ left rows
          break;
        case duckdb::JoinType::RIGHT_SEMI:
        case duckdb::JoinType::RIGHT_ANTI: right_preserved = !right_unique.empty(); break;
        case duckdb::JoinType::MARK:
        case duckdb::JoinType::SINGLE: left_preserved = !left_unique.empty(); break;
        default: return {};  // FULL OUTER: NULL-padding can duplicate key values
      }
      if (!left_preserved && !right_preserved) { return {}; }

      // Remap child unique indices through a projection map to output positions.
      auto remap = [](const std::unordered_set<duckdb::idx_t>& child_unique,
                      const duckdb::vector<duckdb::idx_t>& proj_map,
                      duckdb::idx_t offset) -> std::unordered_set<duckdb::idx_t> {
        std::unordered_set<duckdb::idx_t> mapped;
        if (proj_map.empty()) {
          for (auto col : child_unique) {
            mapped.insert(offset + col);
          }
        } else {
          for (duckdb::idx_t i = 0; i < proj_map.size(); i++) {
            if (child_unique.count(proj_map[i])) { mapped.insert(offset + i); }
          }
          if (mapped.size() != child_unique.size()) { return {}; }
        }
        return mapped;
      };

      bool only_left =
        join.join_type == duckdb::JoinType::SEMI || join.join_type == duckdb::JoinType::ANTI;
      bool only_right = join.join_type == duckdb::JoinType::RIGHT_SEMI ||
                        join.join_type == duckdb::JoinType::RIGHT_ANTI;

      duckdb::idx_t left_output_count = join.left_projection_map.empty()
                                          ? op.children[0]->types.size()
                                          : join.left_projection_map.size();

      std::unordered_set<duckdb::idx_t> best;
      if (left_preserved && !only_right) {
        auto mapped = remap(left_unique, join.left_projection_map, 0);
        if (!mapped.empty() && (best.empty() || mapped.size() < best.size())) {
          best = std::move(mapped);
        }
      }
      if (right_preserved && !only_left) {
        duckdb::idx_t right_offset = only_right ? 0 : left_output_count;
        auto mapped                = remap(right_unique, join.right_projection_map, right_offset);
        if (!mapped.empty() && (best.empty() || mapped.size() < best.size())) {
          best = std::move(mapped);
        }
      }
      return best;
    }

    default: return {};
  }
}

/// Mirror the hash join's rule for routing mixed null-safe keys into the AST predicate.
static bool routes_null_safe_keys_to_predicate(const duckdb::LogicalComparisonJoin& op)
{
  if (op.join_type == duckdb::JoinType::MARK) { return false; }
  bool has_plain_equal = false;
  bool has_null_safe   = false;
  for (auto const& cond : op.conditions) {
    if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL) {
      has_plain_equal = true;
    } else if (cond.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      has_null_safe = true;
    }
  }
  return has_plain_equal && has_null_safe;
}

/// References and hash-key casts need no materialization. Routed predicate casts are trivial
/// only when cuDF AST supports their target type.
static bool is_trivial_key_side(const duckdb::Expression& expr, bool evaluated_as_ast_predicate)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) { return true; }
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CAST) {
    if (expr.Cast<duckdb::BoundCastExpression>().child->GetExpressionClass() !=
        duckdb::ExpressionClass::BOUND_REF) {
      return false;
    }
    if (!evaluated_as_ast_predicate) { return true; }
    return std::find(sirius::supported_ast_cast_types.begin(),
                     sirius::supported_ast_cast_types.end(),
                     expr.return_type.id()) != sirius::supported_ast_cast_types.end();
  }
  return false;
}

/// Materialize complex equality-key expressions into projected columns, then rewrite each
/// condition to reference the appended column. This lets PARTITION and hash join consume a
/// column index. Routed null-safe casts unsupported by cuDF AST use the same path.
static void materialize_expression_join_keys(
  duckdb::LogicalComparisonJoin& op,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& left,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& right)
{
  const bool routes_null_safe = routes_null_safe_keys_to_predicate(op);

  auto materialize_side = [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator>& child,
                              duckdb::vector<duckdb::idx_t>& projection_map,
                              bool is_left) {
    const std::size_t old_width = child->types.size();

    // Gather the complex, translatable sides on this child across all equality conditions.
    std::vector<std::size_t> cond_indices;
    duckdb::vector<std::unique_ptr<sirius::ast::node>> key_exprs;
    duckdb::vector<sirius::logical_type> key_types;
    for (std::size_t i = 0; i < op.conditions.size(); i++) {
      auto& cond = op.conditions[i];
      if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL &&
          cond.comparison != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        continue;  // inequality sides are evaluated inline as the mixed-join predicate
      }
      auto& side_expr = is_left ? cond.left : cond.right;
      const bool as_ast_predicate =
        routes_null_safe && cond.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM;
      if (is_trivial_key_side(*side_expr, as_ast_predicate)) { continue; }
      auto node = sirius::ast::from_duckdb(*side_expr);
      if (!node) { continue; }  // untranslatable: leave for the existing downstream throw
      cond_indices.push_back(i);
      key_exprs.push_back(std::move(node));
      key_types.push_back(sirius::from_duckdb(side_expr->return_type));
    }

    if (cond_indices.empty()) { return; }

    // Build the projection: identity passthrough of all original columns + the key expressions.
    duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
    duckdb::vector<sirius::logical_type> out_types;
    select_list.reserve(old_width + key_exprs.size());
    out_types.reserve(old_width + key_exprs.size());
    for (std::size_t c = 0; c < old_width; c++) {
      select_list.push_back(std::make_unique<sirius::ast::node>(
        sirius::ast::reference{static_cast<uint32_t>(c), child->types[c]}));
      out_types.push_back(child->types[c]);
    }
    for (std::size_t k = 0; k < key_exprs.size(); k++) {
      select_list.push_back(std::move(key_exprs[k]));
      out_types.push_back(key_types[k]);
    }

    const std::size_t cardinality = child->estimated_cardinality;
    child =
      push_projection(std::move(child), std::move(out_types), std::move(select_list), cardinality);

    // Rewrite each materialized condition side to reference its appended column.
    for (std::size_t k = 0; k < cond_indices.size(); k++) {
      const std::size_t new_index = old_width + k;
      auto& side_expr =
        is_left ? op.conditions[cond_indices[k]].left : op.conditions[cond_indices[k]].right;
      side_expr =
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(side_expr->return_type, new_index);
    }

    // Convert "all columns" into the original range so synthetic keys do not leak.
    if (projection_map.empty()) {
      projection_map.reserve(old_width);
      for (std::size_t c = 0; c < old_width; c++) {
        projection_map.push_back(static_cast<duckdb::idx_t>(c));
      }
    }
  };

  materialize_side(left, op.left_projection_map, /*is_left=*/true);
  materialize_side(right, op.right_projection_map, /*is_left=*/false);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_comparison_join(duckdb::LogicalComparisonJoin& op)
{
  // now visit the children
  D_ASSERT(op.children.size() == 2);

  // Reject nested join keys before planning either child.
  for (auto const& condition : op.conditions) {
    reject_nested_column_operation(*condition.left, "a join condition");
    reject_nested_column_operation(*condition.right, "a join condition");
  }

  // A MARK join mixing null-safe (IS NOT DISTINCT FROM) and plain keys has no correct GPU
  // encoding (see mark_join_mixes_null_safe_keys in sirius_physical_hash_join.cpp), and the NLJ
  // is no escape hatch: its MARK arm models the mixture but evaluates each (left batch x right
  // batch) pair independently, so a left side spread over several right batches emits one partial
  // mark per pair. Reject here so both routes are closed and DuckDB's CPU mark join answers.
  if (op.join_type == duckdb::JoinType::MARK) {
    bool has_null_safe = false;
    bool all_null_safe = !op.conditions.empty();
    for (auto const& condition : op.conditions) {
      if (condition.comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
        has_null_safe = true;
      } else {
        all_null_safe = false;
      }
    }
    if (has_null_safe && !all_null_safe) {
      throw duckdb::NotImplementedException(
        "MARK join mixing a null-safe (IS NOT DISTINCT FROM) key with a plain key not supported "
        "in GPU");
    }
  }

  std::size_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
  std::size_t rhs_cardinality = op.children[1]->EstimateCardinality(context);

  auto sirius_context = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  bool const dynamic_filter_enabled =
    sirius_context && sirius_context->get_config().get_operator_params().enable_dynamic_filter;

  bool const build_filtered = dynamic_filter_enabled && build_subtree_is_filtering(*op.children[1]);
  bool const build_opaque   = dynamic_filter_enabled && build_relation_is_opaque(*op.children[1]);
  bool const build_evidence = build_filtered || build_opaque;

  // Capture domain and uniqueness evidence before create_plan() moves the logical children.
  auto condition_domains =
    build_evidence ? build_key_domain_cardinalities(op, duckdb_base_table_cardinality{context})
                   : std::vector<std::size_t>{};

  auto build_side_unique_cols  = prove_unique_columns(*op.children[1]);
  auto left                    = create_plan(*op.children[0]);
  auto right                   = create_plan(*op.children[1]);
  left->estimated_cardinality  = lhs_cardinality;
  right->estimated_cardinality = rhs_cardinality;

  if (op.conditions.empty()) {
    throw duckdb::NotImplementedException("Cross product not supported in GPU");
    // no conditions: insert a cross product
    // return Make<PhysicalCrossProduct>(op.types, left, right, op.estimated_cardinality);
  }

  // Preserve key shape before materialization makes computed keys look like direct references.
  auto condition_key_shapes = classify_join_key_shapes(op.conditions);

  // Materialize computed equality keys before downstream admission and join planning.
  materialize_expression_join_keys(op, left, right);

  std::size_t has_range              = 0;
  [[maybe_unused]] bool has_equality = op.HasEquality(has_range);
  bool can_merge                     = has_range > 0;
  bool can_iejoin                    = has_range >= 2 && recursive_cte_tables.empty();
  switch (op.join_type) {
    case duckdb::JoinType::SEMI:
    case duckdb::JoinType::ANTI:
    case duckdb::JoinType::RIGHT_ANTI:
    case duckdb::JoinType::RIGHT_SEMI:
    case duckdb::JoinType::MARK:
      can_merge  = can_merge && op.conditions.size() == 1;
      can_iejoin = false;
      break;
    default: break;
  }
  //	TODO: Extend PWMJ to handle all comparisons and projection maps
  bool prefer_range_joins = duckdb::Settings::Get<duckdb::PreferRangeJoinsSetting>(context);
  prefer_range_joins      = prefer_range_joins && can_iejoin;

  // Check DuckDB's NLJ IsSupported here because it needs the raw `op.conditions`; wrapping the
  // conditions below drains them.
  const bool nlj_conditions_supported =
    duckdb::PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type);
  // DuckDB's check accepts join types the Sirius NLJ execute switch has no arm for
  // (RIGHT_SEMI / RIGHT_ANTI / SINGLE), which would throw mid-query instead of falling back.
  // The hash join screens the same way via are_conditions_supported, so SINGLE -- which no GPU
  // join implements -- is rejected on both routes.
  const bool nlj_join_type_supported =
    sirius::op::sirius_physical_nested_loop_join::is_join_type_supported(op.join_type);
  const bool nlj_is_supported = nlj_conditions_supported && nlj_join_type_supported;

  // Wrap once — subsequent checks and ctors consume from the wrapped vector.
  duckdb::vector<sirius::join_condition> conditions =
    sirius::wrap_join_conditions(std::move(op.conditions));

  bool is_supported_by_hash_join =
    sirius::op::sirius_physical_hash_join::are_conditions_supported(conditions, op.join_type);
  if (is_supported_by_hash_join && !prefer_range_joins) {
    const auto& op_params = sirius_context->get_config().get_operator_params();

    // Resolve placements before registering any producer channel.
    auto& memory_manager  = sirius_context->get_memory_manager();
    auto const gpu_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    auto const host_spaces =
      memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);

    // Only a single-column proof can establish per-key uniqueness.
    auto const build_side_unique_column =
      build_side_unique_cols.size() == 1
        ? std::optional<std::size_t>{static_cast<std::size_t>(*build_side_unique_cols.begin())}
        : std::nullopt;
    auto admitted_keys = admit_dynamic_filter_keys(
      conditions, condition_key_shapes, condition_domains, build_side_unique_column);

    // Prefer scan binding; each key uses one route.
    std::vector<sirius::op::dynamic_filter_publish_plan::probe_target> targets;
    std::size_t scan_target_count = 0;
    bool const discovery_runs     = build_evidence &&
                                op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
                                !gpu_spaces.empty() && !host_spaces.empty();
    if (discovery_runs) {
      bool const scan_bind_armed = scan_route_join_type_admissible(op.join_type);
      descent_policy const policy{.descend_build_blocks = true};
      std::unordered_map<sirius::op::sirius_physical_operator const*, std::size_t> target_by_scan;
      for (std::size_t key_index = 0; key_index < admitted_keys.size(); ++key_index) {
        auto const& key       = admitted_keys[key_index];
        auto const& condition = conditions[key.planner_condition_index];
        // Terminal ordinals are local to each terminal, not the probe entry schema.
        auto const terminals =
          trace_probe_key(*left, static_cast<std::size_t>(key.probe_key_ordinal), policy);
        bool scan_bound = false;
        for (auto const& terminal : terminals) {
          if (!scan_bind_armed ||
              terminal.node->type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
            continue;
          }
          auto& scan = terminal.node->Cast<sirius::op::sirius_physical_table_scan>();
          if (terminal.ordinal >= scan.types.size()) {
            SIRIUS_LOG_WARN(
              "[sirius_plan_comparison_join] dynamic filter key {}: scan-route terminal at scan "
              "'{}' carries exit ordinal {} outside the scan's {} output columns; skipping this "
              "binding.",
              key_index,
              scan.function.name,
              terminal.ordinal,
              scan.types.size());
            continue;
          }
          auto const [entry, inserted] = target_by_scan.try_emplace(terminal.node, targets.size());
          if (inserted) {
            // Reuse the scan channel so multiple producers share the consumer.
            if (!scan.sirius_dynamic_filters) {
              scan.sirius_dynamic_filters =
                std::make_shared<sirius::op::sirius_dynamic_filter_set>();
            }
            targets.push_back({.filter_set  = scan.sirius_dynamic_filters,
                               .route_class = sirius::op::dynamic_filter_route_class::scan,
                               .accepts_zone_map_filters = true,
                               .key_bindings             = {}});
            ++scan_target_count;
          }
          targets[entry->second].key_bindings.push_back(
            {.admitted_key_index   = key_index,
             .channel_push_ordinal = terminal.ordinal,
             .probe_storage_type   = sirius::try_get_cudf_type(scan.types[terminal.ordinal])
                                     .value_or(cudf::data_type{cudf::type_id::EMPTY})});
          scan_bound = true;
        }
        if (scan_bound) { continue; }
        // Admission proves build-block safety; placement preserves the reported site ordinal.
        if (!direct_route_admissible(op.join_type,
                                     condition.comparison,
                                     key.key_shape,
                                     key.probe_storage_type,
                                     key.storage_type)) {
          continue;
        }
        std::vector<std::shared_ptr<sirius::op::sirius_dynamic_filter_set>> site_channels;
        auto placed = place_endpoint(
          std::move(left),
          static_cast<std::size_t>(key.probe_key_ordinal),
          policy,
          [&site_channels, &op_params](sirius::op::sirius_physical_operator const& site)
            -> duckdb::unique_ptr<sirius::op::sirius_physical_operator> {
            auto channel  = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
            auto endpoint = duckdb::make_uniq<sirius::op::scan::sirius_physical_dynamic_filter>(
              site.types,
              site.estimated_cardinality,
              channel,
              op_params.dynamic_filter_keep_threshold,
              sirius::op::scan::dynamic_filter_apply_mode::membership_masks_only);
            site_channels.push_back(std::move(channel));
            return endpoint;
          });
        left = std::move(placed.subtree);
        if (site_channels.size() != placed.site_ordinals.size()) {
          SIRIUS_LOG_WARN(
            "[sirius_plan_comparison_join] dynamic filter key {}: direct-route walk placed {} "
            "endpoints but recorded {} site ordinals; skipping this key's bindings.",
            key_index,
            site_channels.size(),
            placed.site_ordinals.size());
          continue;
        }
        for (std::size_t site = 0; site < site_channels.size(); ++site) {
          targets.push_back({.filter_set  = std::move(site_channels[site]),
                             .route_class = sirius::op::dynamic_filter_route_class::direct,
                             .accepts_zone_map_filters = false,
                             .key_bindings             = {{.admitted_key_index   = key_index,
                                                           .channel_push_ordinal = placed.site_ordinals[site],
                                                           .probe_storage_type   = key.probe_storage_type}}});
        }
      }
      // Register after all keys bind so each declaration covers every planned push.
      for (auto const& target : targets) {
        std::vector<std::size_t> planned_columns;
        planned_columns.reserve(target.key_bindings.size());
        for (auto const& binding : target.key_bindings) {
          planned_columns.push_back(binding.channel_push_ordinal);
        }
        target.filter_set->register_producer(std::move(planned_columns));
      }
    }
    if (!targets.empty()) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_comparison_join] Wired hash join with {} dynamic-filter probe "
        "target(s) ({} scan-bound, {} join-edge; evidence={}; build est {} rows).",
        targets.size(),
        scan_target_count,
        targets.size() - scan_target_count,
        build_filtered ? (build_opaque ? "filtered+opaque" : "filtered") : "opaque",
        rhs_cardinality);
    } else if (build_evidence && op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
               !admitted_keys.empty() && (gpu_spaces.empty() || host_spaces.empty())) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_comparison_join] Not wiring dynamic filter(s): a GPU and HOST "
        "memory space are required for device-local replicas.");
    } else if (dynamic_filter_enabled && !build_evidence &&
               op.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
               !admitted_keys.empty()) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_comparison_join] Not wiring dynamic filter(s): build side carries "
        "neither filter nor opaque-build evidence (build est {} rows).",
        rhs_cardinality);
    }

    // Prefer a NUMA-local HOST staging space for each GPU and fall back to the first HOST space.
    std::vector<sirius::op::dynamic_filter_replica_space> filter_replica_spaces;
    if (!targets.empty()) {
      auto const& topology = sirius_context->get_config().get_hw_topology();
      for (auto const* gpu_space : gpu_spaces) {
        auto const gpu_id = gpu_space->get_device_id();
        auto const gpu_info =
          std::find_if(topology.gpus.begin(), topology.gpus.end(), [gpu_id](auto const& gpu) {
            return static_cast<int>(gpu.id) == gpu_id;
          });
        auto const numa_node = gpu_info == topology.gpus.end() ? -1 : gpu_info->numa_node;
        auto const local_host =
          std::find_if(host_spaces.begin(), host_spaces.end(), [numa_node](auto const* host) {
            return numa_node >= 0 && host->get_device_id() == numa_node;
          });
        auto const* host_space =
          local_host == host_spaces.end() ? host_spaces.front() : *local_host;
        auto* mutable_gpu_space =
          memory_manager.get_memory_space(cucascade::memory::Tier::GPU, gpu_id);
        if (mutable_gpu_space == nullptr) {
          throw std::logic_error(
            "[sirius_plan_comparison_join] Dynamic-filter GPU space disappeared during "
            "plan construction");
        }
        filter_replica_spaces.emplace_back(*mutable_gpu_space, *host_space);
      }
    }

    sirius::op::dynamic_filter_publish_plan filter_plan{
      std::move(admitted_keys),
      std::move(targets),
      std::move(filter_replica_spaces),
      {.emit_zone_map_filters     = op_params.enable_dynamic_zone_map_filter,
       .domain_coverage_threshold = op_params.dynamic_filter_domain_coverage_threshold,
       .inlist_max_l2_fraction    = op_params.dynamic_filter_inlist_max_l2_fraction}};

    auto join = duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(
      op,
      std::move(left),
      std::move(right),
      std::move(conditions),
      op.join_type,
      op.left_projection_map,
      op.right_projection_map,
      sirius::from_duckdb_vec(op.mark_types),
      op.estimated_cardinality,
      op_params.max_build_hash_table_bytes,
      std::move(filter_plan),
      op_params.hash_partition_bytes,
      op_params.max_broadcast_join_size,
      &sirius_context->get_dynamic_filter_stats());
    auto& hj                        = join->Cast<sirius::op::sirius_physical_hash_join>();
    hj.join_stats                   = std::move(op.join_stats);
    hj.mark_join_build_switch_ratio = op_params.mark_join_build_switch_ratio;
    hj.runtime_distinct_build_probe = op_params.enable_runtime_distinct_build_probe;

    // --- Detect build-side key uniqueness ---
    // Gate: only for pure equal conditions (not_distinct_from needs null_equality::EQUAL).
    bool all_compare_equal = true;
    for (const auto& c : hj.conditions) {
      if (c.comparison == sirius::comparison_type::equal) { continue; }
      if (c.comparison == sirius::comparison_type::not_distinct_from) {
        all_compare_equal = false;
        break;
      }
      // Inequality conditions are fine — only equality conditions drive the hash table.
    }
    if (all_compare_equal) {
      // Extract build-side (right) column indices from equality conditions.
      // Only accept direct BoundReferenceExpression (skip if any has a cast).
      std::unordered_set<duckdb::idx_t> build_key_cols;
      bool keys_extractable = true;
      for (const auto& c : hj.conditions) {
        if (c.comparison != sirius::comparison_type::equal) { continue; }
        auto right_expr = sirius::ast::to_duckdb(*c.right);
        if (right_expr->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
          keys_extractable = false;
          break;
        }
        build_key_cols.insert(right_expr->Cast<duckdb::BoundReferenceExpression>().index);
      }
      if (keys_extractable && !build_key_cols.empty()) {
        // build_side_unique_cols was computed before create_plan (which moves logical node data).
        // proven_unique ⊆ build_key_cols  →  build keys are unique.
        if (!build_side_unique_cols.empty()) {
          bool is_subset = true;
          for (auto col : build_side_unique_cols) {
            if (!build_key_cols.count(col)) {
              is_subset = false;
              break;
            }
          }
          if (is_subset) { hj.unique_build_keys = true; }
        }
      }
    }

    return join;
  }

  // D_ASSERT(op.left_projection_map.empty());
  // std::size_t nested_loop_join_threshold =
  //   duckdb::DBConfig::GetSetting<duckdb::NestedLoopJoinThresholdSetting>(context);
  // if (left->estimated_cardinality < nested_loop_join_threshold ||
  //     right->estimated_cardinality < nested_loop_join_threshold) {
  //   can_iejoin = false;
  //   can_merge  = false;
  // }

  // if (can_merge && can_iejoin) {
  //   std::size_t merge_join_threshold =
  //     duckdb::DBConfig::GetSetting<duckdb::MergeJoinThresholdSetting>(context);
  //   if (left->estimated_cardinality < merge_join_threshold ||
  //       right->estimated_cardinality < merge_join_threshold) {
  //     can_iejoin = false;
  //   }
  // }

  // if (can_iejoin) {
  //   throw duckdb::NotImplementedException("InequalityJoin not supported in GPU");
  //   // return Make<PhysicalIEJoin>(op, left, right, std::move(op.conditions), op.join_type,
  //   // op.estimated_cardinality,
  //   //                             std::move(op.filter_pushdown));
  // }
  // if (can_merge) {
  //   throw duckdb::NotImplementedException("Piecewise merge join not supported in GPU");
  //   // range join: use piecewise merge join
  //   // return Make<PhysicalPiecewiseMergeJoin>(op, left, right, std::move(op.conditions),
  //   // op.join_type,
  //   //                                         op.estimated_cardinality,
  //   //                                         std::move(op.filter_pushdown));
  // }
  if (nlj_is_supported) {
    // inequality join: use nested loop; pass projection maps so output column order matches plan
    auto join =
      duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(op,
                                                                      std::move(left),
                                                                      std::move(right),
                                                                      std::move(conditions),
                                                                      op.join_type,
                                                                      op.estimated_cardinality,
                                                                      op.left_projection_map,
                                                                      op.right_projection_map);
    return join;
  }

  // Neither GPU join implements this join type, so say so plainly: the generic "blockwise nested
  // loop" message below would misattribute the rejection to the conditions.
  if (!nlj_join_type_supported) {
    throw duckdb::NotImplementedException("Join type " + duckdb::JoinTypeToString(op.join_type) +
                                          " not supported in GPU");
  }

  throw duckdb::NotImplementedException("Blockwise nested loop join not supported in GPU");
  // for (auto &cond : op.conditions) {
  // 	RewriteJoinCondition(cond.right, left.types.size());
  // }
  // auto condition = JoinCondition::CreateExpression(std::move(op.conditions));
  // return Make<PhysicalBlockwiseNLJoin>(op, left, right, std::move(condition), op.join_type,
  // op.estimated_cardinality);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalComparisonJoin& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_ASOF_JOIN:
      // return plan_asof_join(op);
      throw duckdb::NotImplementedException("Asof join not supported in GPU");
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: return plan_comparison_join(op);
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN: return plan_delim_join(op);
    default:
      throw duckdb::InternalException("Unrecognized operator type for LogicalComparisonJoin");
  }
}

}  // namespace sirius::planner
