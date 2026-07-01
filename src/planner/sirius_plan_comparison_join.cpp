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
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

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

namespace {

/// Plan-time producer-side selectivity evidence for dynamic-filter wiring, gathered from the
/// build-side LOGICAL subtree (before create_plan moves data out of it).
struct build_side_filter_evidence {
  /// Any predicate anywhere in the subtree: a LOGICAL_FILTER node or GET-level table filters.
  /// An unfiltered build is (for FK-shaped joins) the whole key domain — its filter keeps every
  /// probe row by construction, so wiring a producer target for it only buys overhead.
  bool subtree_filtered = false;

  /// Largest UNFILTERED base-table cardinality among the subtree's GETs, from the table
  /// function's own statistics (parquet: exact row count from metadata) — a proxy for the build
  /// key's domain size. 0 when no GET reports statistics.
  std::size_t domain_cardinality = 0;
};

build_side_filter_evidence inspect_build_side_for_dynamic_filters(duckdb::LogicalOperator& node,
                                                                  duckdb::ClientContext& context)
{
  build_side_filter_evidence ev;
  if (node.type == duckdb::LogicalOperatorType::LOGICAL_FILTER) { ev.subtree_filtered = true; }
  if (node.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto& get = node.Cast<duckdb::LogicalGet>();
    if (!get.table_filters.filters.empty()) { ev.subtree_filtered = true; }
    std::size_t card = 0;
    if (get.function.cardinality) {
      // The table function's own estimate is pre-filter (the optimizer-adjusted
      // estimated_cardinality field would undercount the domain for filtered GETs).
      auto stats = get.function.cardinality(context, get.bind_data.get());
      if (stats && stats->has_estimated_cardinality) { card = stats->estimated_cardinality; }
    }
    if (card == 0) {
      // May include optimizer filter selectivity
      card = get.estimated_cardinality;
    }
    ev.domain_cardinality = std::max(ev.domain_cardinality, card);
  }
  for (auto& child : node.children) {
    auto child_ev         = inspect_build_side_for_dynamic_filters(*child, context);
    ev.subtree_filtered   = ev.subtree_filtered || child_ev.subtree_filtered;
    ev.domain_cardinality = std::max(ev.domain_cardinality, child_ev.domain_cardinality);
  }
  return ev;
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_comparison_join(duckdb::LogicalComparisonJoin& op)
{
  // now visit the children
  D_ASSERT(op.children.size() == 2);
  std::size_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
  std::size_t rhs_cardinality = op.children[1]->EstimateCardinality(context);

  // Inspect the build side for the dynamic-filter wiring decision BEFORE create_plan, which
  // moves data out of the logical nodes (same constraint as prove_unique_columns below).
  auto build_filter_evidence = op.filter_pushdown
                                 ? inspect_build_side_for_dynamic_filters(*op.children[1], context)
                                 : build_side_filter_evidence{};

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

  std::size_t has_range = 0;
  bool has_equality     = op.HasEquality(has_range);
  bool can_merge        = has_range > 0;
  bool can_iejoin       = has_range >= 2 && recursive_cte_tables.empty();
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
    const auto& op_params = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                              ->get_config()
                              .get_operator_params();
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
      std::move(op.filter_pushdown),
      op_params.max_build_hash_table_bytes);
    auto& hj                        = join->Cast<sirius::op::sirius_physical_hash_join>();
    hj.join_stats                   = std::move(op.join_stats);
    hj.mark_join_build_switch_ratio = op_params.mark_join_build_switch_ratio;

    //===----------Wire dynamic-filter producer targets----------===//
    // For each downstream scan DuckDB has paired with this join, look up the shared channel by
    // the DynamicTableFilterSet pointer (the route key) and stash it along with the per-key
    // consumer column indices on the join.
    //
    // Plan-time selectivity gate: an UNFILTERED build subtree is — for FK-shaped joins — the
    // whole key domain, so its membership filter keeps every probe row by construction.
    //
    // DuckDB is the source of truth for whether dynamic filter pushdown is valid for this join
    // type, gated here on the existence of a dynamic filter channel. See GenerateJoinFilters at
    // join_filter_pushdown_optimizer.cpp.
    if (hj.filter_pushdown) {
      if (!build_filter_evidence.subtree_filtered) {
        SIRIUS_LOG_INFO(
          "[sirius_plan_comparison_join] Not wiring dynamic filter(s): build subtree is "
          "unfiltered (build est {} rows, key-domain ~{} rows).",
          rhs_cardinality,
          build_filter_evidence.domain_cardinality);
      } else {
        hj.probe_targets.reserve(hj.filter_pushdown->probe_info.size());
        for (auto const& pi : hj.filter_pushdown->probe_info) {
          auto channel = get_or_create_dynamic_filter_channel(pi.dynamic_filters.get());
          if (!channel) { continue; }
          channel->register_producer();
          sirius::op::sirius_physical_hash_join::probe_target tgt{std::move(channel), {}};
          tgt.probe_col_idx.reserve(pi.columns.size());
          for (auto const& col : pi.columns) {
            tgt.probe_col_idx.push_back(col.probe_column_index.column_index);
          }
          hj.probe_targets.push_back(std::move(tgt));
        }
        hj.emit_zone_map_filters        = op_params.enable_dynamic_zone_map_filter;
        hj.build_key_domain_cardinality = build_filter_evidence.domain_cardinality;
        if (!hj.probe_targets.empty()) {
          SIRIUS_LOG_INFO(
            "[sirius_plan_comparison_join] Wired hash join with {} dynamic-filter probe "
            "target(s) (build est {} rows, key-domain ~{} rows).",
            hj.probe_targets.size(),
            rhs_cardinality,
            build_filter_evidence.domain_cardinality);
        }
      }
    }

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
