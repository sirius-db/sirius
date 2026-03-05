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

#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_comparison_join(duckdb::LogicalComparisonJoin& op)
{
  // now visit the children
  D_ASSERT(op.children.size() == 2);
  duckdb::idx_t lhs_cardinality = op.children[0]->EstimateCardinality(context);
  duckdb::idx_t rhs_cardinality = op.children[1]->EstimateCardinality(context);
  auto left                     = create_plan(*op.children[0]);
  auto right                    = create_plan(*op.children[1]);
  left->estimated_cardinality   = lhs_cardinality;
  right->estimated_cardinality  = rhs_cardinality;

  if (op.conditions.empty()) {
    throw duckdb::NotImplementedException("Cross product not supported in GPU");
    // no conditions: insert a cross product
    // return Make<PhysicalCrossProduct>(op.types, left, right, op.estimated_cardinality);
  }

  duckdb::idx_t has_range = 0;
  bool has_equality       = op.HasEquality(has_range);
  bool can_merge          = has_range > 0;
  bool can_iejoin         = has_range >= 2 && recursive_cte_tables.empty();
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
  bool prefer_range_joins = duckdb::DBConfig::GetSetting<duckdb::PreferRangeJoinsSetting>(context);
  prefer_range_joins      = prefer_range_joins && can_iejoin;

  bool is_supported_by_hash_join =
    sirius::op::sirius_physical_hash_join::are_conditions_supported(op.conditions);
  if (is_supported_by_hash_join && !prefer_range_joins) {
    // Equality join with small number of keys : possible perfect join optimization
    // auto &join = Make<PhysicalHashJoin>(op, left, right, std::move(op.conditions), op.join_type,
    //                                     op.left_projection_map, op.right_projection_map,
    //                                     std::move(op.mark_types), op.estimated_cardinality,
    //                                     std::move(op.filter_pushdown));
    // join.Cast<PhysicalHashJoin>().join_stats = std::move(op.join_stats);
    // return join;
    auto join =
      duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(op,
                                                               std::move(left),
                                                               std::move(right),
                                                               std::move(op.conditions),
                                                               op.join_type,
                                                               op.left_projection_map,
                                                               op.right_projection_map,
                                                               std::move(op.mark_types),
                                                               op.estimated_cardinality,
                                                               std::move(op.filter_pushdown));
    join->Cast<sirius::op::sirius_physical_hash_join>().join_stats = std::move(op.join_stats);
    return join;
  }

  // D_ASSERT(op.left_projection_map.empty());
  // duckdb::idx_t nested_loop_join_threshold =
  //   duckdb::DBConfig::GetSetting<duckdb::NestedLoopJoinThresholdSetting>(context);
  // if (left->estimated_cardinality < nested_loop_join_threshold ||
  //     right->estimated_cardinality < nested_loop_join_threshold) {
  //   can_iejoin = false;
  //   can_merge  = false;
  // }

  // if (can_merge && can_iejoin) {
  //   duckdb::idx_t merge_join_threshold =
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
  if (duckdb::PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type)) {
    // inequality join: use nested loop; pass projection maps so output column order matches plan
    auto join =
      duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(op,
                                                                      std::move(left),
                                                                      std::move(right),
                                                                      std::move(op.conditions),
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
