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
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
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
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "planner/duckdb_join_filter_candidate_adapter.hpp"
#include "planner/dynamic_filter_key_admission.hpp"
#include "planner/dynamic_filter_scan_routes.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <cassert>
#include <optional>
#include <span>
#include <stdexcept>

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

//===----------------------------------------------------------------------===//
// Dynamic Filters
//===----------------------------------------------------------------------===//
namespace {

/// Resolve @p binding through the logical subtree to the LogicalGet that produces it, or nullptr
/// when the column is computed (cast, expression, aggregate result) or not from a base scan.
duckdb::LogicalGet* trace_binding_to_get(duckdb::LogicalOperator& node,
                                         duckdb::ColumnBinding binding)
{
  switch (node.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET: {
      auto& get = node.Cast<duckdb::LogicalGet>();
      return get.table_index == binding.table_index ? &get : nullptr;
    }
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION: {
      auto& proj = node.Cast<duckdb::LogicalProjection>();
      if (proj.table_index != binding.table_index || node.children.empty() ||
          binding.column_index >= proj.expressions.size()) {
        return nullptr;
      }
      auto& expr = *proj.expressions[binding.column_index];
      if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
        return nullptr;
      }
      return trace_binding_to_get(*node.children[0],
                                  expr.Cast<duckdb::BoundColumnRefExpression>().binding);
    }
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
      auto& aggr = node.Cast<duckdb::LogicalAggregate>();
      if (binding.table_index != aggr.group_index || node.children.empty() ||
          binding.column_index >= aggr.groups.size()) {
        return nullptr;
      }
      auto& expr = *aggr.groups[binding.column_index];
      if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
        return nullptr;
      }
      return trace_binding_to_get(*node.children[0],
                                  expr.Cast<duckdb::BoundColumnRefExpression>().binding);
    }
    default: {
      // Table indexes are binder-unique, so recursing into every child finds the owning subtree
      // (or nothing, for rebinding operators this walk does not model).
      for (auto& child : node.children) {
        if (auto* get = trace_binding_to_get(*child, binding)) { return get; }
      }
      return nullptr;
    }
  }
}

/// Per join condition (original planner order): the unfiltered cardinality of the base table the
/// build key traces to, or 0 when untraceable -- the publish coverage gates are then off for that
/// key and the consumer-side gate is the backstop. Must run before create_plan, which moves data
/// out of the logical children. The admission helper re-emits these in admitted-key order.
///
/// NOTE: the BOUND_COLUMN_REF check below never matches after DuckDB's ColumnBindingResolver has
/// rewritten condition sides to BOUND_REF, so every cardinality is currently 0 and the coverage
/// gates are inert. The positional-trace fix is a follow-up unit of issue #1010.
std::vector<std::size_t> build_key_domain_cardinalities(duckdb::LogicalComparisonJoin& op,
                                                        duckdb::ClientContext& context)
{
  std::vector<std::size_t> domains(op.conditions.size(), 0);
  for (std::size_t condition_index = 0; condition_index < op.conditions.size(); ++condition_index) {
    auto& key = *op.conditions[condition_index].right;
    if (key.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) { continue; }
    auto* get =
      trace_binding_to_get(*op.children[1], key.Cast<duckdb::BoundColumnRefExpression>().binding);
    if (get == nullptr) { continue; }
    std::size_t card = 0;
    if (get->function.cardinality) {
      // The table function's own estimate is pre-filter (the optimizer-adjusted
      // estimated_cardinality field would undercount the domain for filtered GETs).
      auto stats = get->function.cardinality(context, get->bind_data.get());
      if (stats && stats->has_estimated_cardinality) { card = stats->estimated_cardinality; }
    }
    domains[condition_index] = card != 0 ? card : get->estimated_cardinality;
  }
  return domains;
}

}  // namespace
//===----------------------------------------------------------------------===//

/// A join equality-condition side that is a plain column reference (BOUND_REF) or a cast of one
/// (BOUND_CAST(BOUND_REF)) is already handled directly by the hash-join key extraction and by the
/// PARTITION operator's cast-alignment logic, so it needs no materialization.
static bool is_trivial_key_side(const duckdb::Expression& expr)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) { return true; }
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CAST) {
    return expr.Cast<duckdb::BoundCastExpression>().child->GetExpressionClass() ==
           duckdb::ExpressionClass::BOUND_REF;
  }
  return false;
}

/// Materialize complex expressions appearing in equality join conditions into real columns.
///
/// Sirius always feeds a hash join through a PARTITION -> CONCAT chain, and the PARTITION operator
/// - the first consumer of the join key - hashes columns by index and cannot evaluate expressions.
/// To support keys like `n_nationkey * 10`, we push a projection below the join that appends the
/// expression as a new column on that side's child, then rewrite the condition side to a plain
/// BoundReferenceExpression pointing at the appended column. PARTITION, CONCAT, and the hash join
/// then all see an ordinary column reference; the partitioning invariant (both sides hash the
/// identical materialized value) holds because DuckDB binds both sides of a comparison to a common
/// type, so the materialized column matches the type the other side hashes.
///
/// Only genuinely complex sides of equality (or IS-NOT-DISTINCT-FROM) conditions are materialized;
/// plain/cast-of-reference sides keep the existing fast path, and inequality-condition sides are
/// left untouched (they are evaluated inline as the mixed-join predicate). A side whose expression
/// cannot be translated to a Sirius AST node is left unchanged, preserving the existing
/// "unsupported join condition expression" behavior downstream.
static void materialize_expression_join_keys(
  duckdb::LogicalComparisonJoin& op,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& left,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& right)
{
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
      if (is_trivial_key_side(*side_expr)) { continue; }
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

    // Exclude the synthetic key column(s) from the join output. An empty projection map means
    // "all columns"; make it explicit over just the original columns so the appended key does
    // not leak into the join result or shift downstream column indices. A non-empty map already
    // lists only original indices (< old_width) and needs no change.
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

  std::size_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
  std::size_t rhs_cardinality = op.children[1]->EstimateCardinality(context);

  // Snapshot DuckDB's join-filter pushdown metadata into Sirius-owned values while the logical
  // join is intact. This is the only read of the optional DuckDB metadata in this function;
  // everything downstream consumes the snapshot.
  auto const filter_candidate = duckdb_join_filter_candidate_adapter::extract(op);

  // Gather per-condition domain evidence BEFORE create_plan, which moves data out of the logical
  // nodes (same constraint as prove_unique_columns below).
  auto condition_domains = filter_candidate.kind() == duckdb_candidate_kind::admitted
                             ? build_key_domain_cardinalities(op, context)
                             : std::vector<std::size_t>{};

  // Probe build-side uniqueness BEFORE create_plan, which moves data out of the logical nodes.
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

  // Classify every condition side BEFORE computed keys are materialized into plain references --
  // afterwards a computed key is indistinguishable from a direct one, and dynamic-filter
  // admission consumes this carried classification instead of re-deriving it.
  auto condition_key_shapes = classify_join_key_shapes(op.conditions);

  // Materialize any complex expressions in equality conditions into real columns below the join,
  // rewriting those condition sides to plain references. This must run before the conditions are
  // inspected/wrapped below so all downstream analysis sees plain column references.
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
  const bool nlj_is_supported =
    duckdb::PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type);

  // Wrap once — subsequent checks and ctors consume from the wrapped vector.
  duckdb::vector<sirius::join_condition> conditions =
    sirius::wrap_join_conditions(std::move(op.conditions));

  bool is_supported_by_hash_join =
    sirius::op::sirius_physical_hash_join::are_conditions_supported(conditions);
  if (is_supported_by_hash_join && !prefer_range_joins) {
    auto sirius_context   = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    const auto& op_params = sirius_context->get_config().get_operator_params();

    // Resolve scan endpoints from that snapshot and build the Sirius-owned
    // admitted-key publication plan before constructing the physical join. Routing and device
    // placement are plan-time decisions; runtime code only consumes the resulting
    // dynamic_filter_publish_plan value and never reads DuckDB metadata.
    //
    // The memory-space lookup stays here and is answered before resolve_scan_routes mints
    // anything: a producer must not register on channels it would then abandon for want of a
    // replica placement.
    auto& memory_manager  = sirius_context->get_memory_manager();
    auto const gpu_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    auto const host_spaces =
      memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    auto routes = resolve_scan_routes(
      filter_candidate,
      /*device_placement_available=*/!gpu_spaces.empty() && !host_spaces.empty(),
      [this](duckdb::DynamicTableFilterSet const* channel_identity) {
        return get_or_create_dynamic_filter_channel(channel_identity);
      });

    std::vector<sirius::op::dynamic_filter_replica_space> filter_replica_spaces;
    switch (routes.decision) {
      // A producer whose channels all mint null (the master switch being off, for instance) wires
      // nothing and says nothing, exactly as an absent candidate does.
      case dynamic_filter_routing_decision::no_duckdb_metadata:
      case dynamic_filter_routing_decision::no_live_channel: break;
      case dynamic_filter_routing_decision::metadata_malformed:
        SIRIUS_LOG_WARN(
          "[sirius_plan_comparison_join] Not wiring dynamic filter(s): DuckDB join-filter "
          "metadata failed structural validation.");
        break;
      case dynamic_filter_routing_decision::build_side_unfiltered:
        SIRIUS_LOG_INFO(
          "[sirius_plan_comparison_join] Not wiring dynamic filter(s): build side is "
          "unfiltered (build est {} rows).",
          rhs_cardinality);
        break;
      case dynamic_filter_routing_decision::no_device_placement:
        SIRIUS_LOG_INFO(
          "[sirius_plan_comparison_join] Not wiring dynamic filter(s): a GPU and HOST "
          "memory space are required for device-local replicas.");
        break;
      case dynamic_filter_routing_decision::wired: {
        // Resolve each replica's NUMA-local Sirius HOST space once at plan time. The transfer
        // path borrows fixed pinned blocks from that explicit space if peer DMA is unavailable.
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
        SIRIUS_LOG_INFO(
          "[sirius_plan_comparison_join] Wired hash join with {} dynamic-filter probe "
          "target(s) (build est {} rows).",
          routes.targets.size(),
          rhs_cardinality);
        break;
      }
    }

    // Static key admission: the producer-key admission boundary is one side-effect-free decision
    // over the wrapped conditions, carried shapes, and resolved targets. When DuckDB named a
    // condition set, admission is restricted to it so publication constructs exactly the filters
    // the runtime publisher constructed before admission moved to plan time.
    auto const hinted_condition_indexes =
      routes.hinted_condition_indexes.empty()
        ? std::optional<std::span<std::size_t const>>{}
        : std::optional<std::span<std::size_t const>>{
            std::span<std::size_t const>{routes.hinted_condition_indexes}};
    auto admission = admit_dynamic_filter_keys(conditions,
                                               condition_key_shapes,
                                               hinted_condition_indexes,
                                               routes.target_inputs,
                                               condition_domains);
    // Postcondition of admit_dynamic_filter_keys: one binding list per input target.
    assert(admission.per_target_key_bindings.size() == routes.targets.size());
    for (std::size_t target_index = 0; target_index < routes.targets.size(); ++target_index) {
      routes.targets[target_index].key_bindings =
        std::move(admission.per_target_key_bindings[target_index]);
    }
    sirius::op::dynamic_filter_publish_plan filter_plan{
      std::move(admission.admitted_keys),
      std::move(routes.targets),
      std::move(filter_replica_spaces),
      {.emit_zone_map_filters     = op_params.enable_dynamic_zone_map_filter,
       .domain_coverage_threshold = op_params.dynamic_filter_domain_coverage_threshold}};

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
      std::move(condition_key_shapes));
    auto& hj                        = join->Cast<sirius::op::sirius_physical_hash_join>();
    hj.join_stats                   = std::move(op.join_stats);
    hj.mark_join_build_switch_ratio = op_params.mark_join_build_switch_ratio;

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
