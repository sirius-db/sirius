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

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/operator/subtract.hpp"
#include "duckdb/execution/operator/join/perfect_hash_join_executor.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_iejoin.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_piecewise_merge_join.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "gpu_physical_hash_join.hpp"
#include "gpu_physical_nested_loop_join.hpp"
#include "gpu_physical_plan_generator.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::PlanComparisonJoin(
  LogicalComparisonJoin& op)
{
  // now visit the children
  D_ASSERT(op.children.size() == 2);
  idx_t lhs_cardinality        = op.children[0]->EstimateCardinality(context);
  idx_t rhs_cardinality        = op.children[1]->EstimateCardinality(context);
  auto left                    = CreatePlan(*op.children[0]);
  auto right                   = CreatePlan(*op.children[1]);
  left->estimated_cardinality  = lhs_cardinality;
  right->estimated_cardinality = rhs_cardinality;

  if (op.conditions.empty()) {
    throw NotImplementedException("Cross product not supported in GPU");
    // no conditions: insert a cross product
    // return Make<PhysicalCrossProduct>(op.types, left, right, op.estimated_cardinality);
  }

  idx_t has_range   = 0;
  bool has_equality = op.HasEquality(has_range);
  bool can_merge    = has_range > 0;
  bool can_iejoin   = has_range >= 2 && recursive_cte_tables.empty();
  switch (op.join_type) {
    case JoinType::SEMI:
    case JoinType::ANTI:
    case JoinType::RIGHT_ANTI:
    case JoinType::RIGHT_SEMI:
    case JoinType::MARK:
      can_merge  = can_merge && op.conditions.size() == 1;
      can_iejoin = false;
      break;
    default: break;
  }
  //	TODO: Extend PWMJ to handle all comparisons and projection maps
  bool prefer_range_joins = DBConfig::GetSetting<PreferRangeJoinsSetting>(context);
  prefer_range_joins      = prefer_range_joins && can_iejoin;
  if (has_equality && !prefer_range_joins) {
    // Equality join with small number of keys : possible perfect join optimization
    // auto &join = Make<PhysicalHashJoin>(op, left, right, std::move(op.conditions), op.join_type,
    //                                     op.left_projection_map, op.right_projection_map,
    //                                     std::move(op.mark_types), op.estimated_cardinality,
    //                                     std::move(op.filter_pushdown));
    // join.Cast<PhysicalHashJoin>().join_stats = std::move(op.join_stats);
    // return join;
    auto join                                    = make_uniq<GPUPhysicalHashJoin>(op,
                                               std::move(left),
                                               std::move(right),
                                               std::move(op.conditions),
                                               op.join_type,
                                               op.left_projection_map,
                                               op.right_projection_map,
                                               std::move(op.mark_types),
                                               op.estimated_cardinality,
                                               std::move(op.filter_pushdown));
    join->Cast<GPUPhysicalHashJoin>().join_stats = std::move(op.join_stats);
    return join;
  }

  D_ASSERT(op.left_projection_map.empty());
  idx_t nested_loop_join_threshold = DBConfig::GetSetting<NestedLoopJoinThresholdSetting>(context);
  if (left->estimated_cardinality < nested_loop_join_threshold ||
      right->estimated_cardinality < nested_loop_join_threshold) {
    can_iejoin = false;
    can_merge  = false;
  }

  if (can_merge && can_iejoin) {
    idx_t merge_join_threshold = DBConfig::GetSetting<MergeJoinThresholdSetting>(context);
    if (left->estimated_cardinality < merge_join_threshold ||
        right->estimated_cardinality < merge_join_threshold) {
      can_iejoin = false;
    }
  }

  if (can_iejoin) {
    throw NotImplementedException("InequalityJoin not supported in GPU");
    // return Make<PhysicalIEJoin>(op, left, right, std::move(op.conditions), op.join_type,
    // op.estimated_cardinality,
    //                             std::move(op.filter_pushdown));
  }
  if (can_merge) {
    throw NotImplementedException("Piecewise merge join not supported in GPU");
    // range join: use piecewise merge join
    // return Make<PhysicalPiecewiseMergeJoin>(op, left, right, std::move(op.conditions),
    // op.join_type,
    //                                         op.estimated_cardinality,
    //                                         std::move(op.filter_pushdown));
  }
  if (PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type)) {
    // inequality join: use nested loop
    // return Make<PhysicalNestedLoopJoin>(op, left, right, std::move(op.conditions), op.join_type,
    //                                     op.estimated_cardinality, std::move(op.filter_pushdown));
    auto join = make_uniq<GPUPhysicalNestedLoopJoin>(op,
                                                     std::move(left),
                                                     std::move(right),
                                                     std::move(op.conditions),
                                                     op.join_type,
                                                     op.estimated_cardinality);
    return join;
  }

  throw NotImplementedException("Blockwise nested loop join not supported in GPU");
  // for (auto &cond : op.conditions) {
  // 	RewriteJoinCondition(cond.right, left.types.size());
  // }
  // auto condition = JoinCondition::CreateExpression(std::move(op.conditions));
  // return Make<PhysicalBlockwiseNLJoin>(op, left, right, std::move(condition), op.join_type,
  // op.estimated_cardinality);
}

// unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::PlanComparisonJoin(
//   LogicalComparisonJoin& op)
// {
//   // now visit the children
//   D_ASSERT(op.children.size() == 2);
//   idx_t lhs_cardinality        = op.children[0]->EstimateCardinality(context);
//   idx_t rhs_cardinality        = op.children[1]->EstimateCardinality(context);
//   auto left                    = CreatePlan(*op.children[0]);
//   auto right                   = CreatePlan(*op.children[1]);
//   left->estimated_cardinality  = lhs_cardinality;
//   right->estimated_cardinality = rhs_cardinality;
//   D_ASSERT(left && right);

//   if (op.conditions.empty()) {
//     throw NotImplementedException("Cross product not supported in GPU");
//     // no conditions: insert a cross product
//     // return make_uniq<PhysicalCrossProduct>(op.types, std::move(left), std::move(right),
//     // op.estimated_cardinality);
//   }

//   idx_t has_range = 0;
//   // bool has_equality = HasEquality(op.conditions, has_range);
//   bool has_equality = op.HasEquality(has_range);
//   bool can_merge    = has_range > 0;
//   // bool can_iejoin = has_range >= 2 && recursive_cte_tables.empty();
//   bool can_iejoin = false;
//   switch (op.join_type) {
//     case JoinType::SEMI:
//     case JoinType::ANTI:
//     case JoinType::RIGHT_ANTI:
//     case JoinType::RIGHT_SEMI:
//     case JoinType::MARK:
//       can_merge  = can_merge && op.conditions.size() == 1;
//       can_iejoin = false;
//       break;
//     default: break;
//   }

//   auto& client_config = ClientConfig::GetConfig(context);

//   //	TODO: Extend PWMJ to handle all comparisons and projection maps
//   const auto prefer_range_joins = client_config.prefer_range_joins && can_iejoin;

//   unique_ptr<GPUPhysicalOperator> plan;
//   if (has_equality && !prefer_range_joins) {
//     // Equality join with small number of keys : possible perfect join optimization
//     // PerfectHashJoinStats perfect_join_stats;
//     // CheckForPerfectJoinOpt(op, perfect_join_stats);

//     plan                                         = make_uniq<GPUPhysicalHashJoin>(op,
//                                           std::move(left),
//                                           std::move(right),
//                                           std::move(op.conditions),
//                                           op.join_type,
//                                           op.left_projection_map,
//                                           op.right_projection_map,
//                                           std::move(op.mark_types),
//                                           op.estimated_cardinality,
//                                           std::move(op.filter_pushdown));
//     plan->Cast<GPUPhysicalHashJoin>().join_stats = std::move(op.join_stats);

//   } else {
//     // throw NotImplementedException("Non-equality join not supported in GPU");
//     // static constexpr const idx_t NESTED_LOOP_JOIN_THRESHOLD = 5;
//     // if (left->estimated_cardinality <= NESTED_LOOP_JOIN_THRESHOLD ||
//     //     right->estimated_cardinality <= NESTED_LOOP_JOIN_THRESHOLD) {
//     // 	can_iejoin = false;
//     // 	can_merge = false;
//     // }
//     D_ASSERT(op.left_projection_map.empty());
//     if (left->estimated_cardinality <= client_config.nested_loop_join_threshold ||
//         right->estimated_cardinality <= client_config.nested_loop_join_threshold) {
//       can_iejoin = false;
//       can_merge  = false;
//     }
//     if (can_merge && can_iejoin) {
//       if (left->estimated_cardinality <= client_config.merge_join_threshold ||
//           right->estimated_cardinality <= client_config.merge_join_threshold) {
//         can_iejoin = false;
//       }
//     }
//     if (can_iejoin) {
//       throw NotImplementedException("InequalityJoin not supported in GPU");
//       // plan = make_uniq<PhysicalIEJoin>(op, std::move(left), std::move(right),
//       // std::move(op.conditions),
//       //                                  op.join_type, op.estimated_cardinality);
//     } else if (can_merge) {
//       throw NotImplementedException("Piecewise merge join not supported in GPU");
//       // range join: use piecewise merge join
//       // plan =
//       //     make_uniq<PhysicalPiecewiseMergeJoin>(op, std::move(left), std::move(right),
//       //     std::move(op.conditions),
//       //                                           op.join_type, op.estimated_cardinality);
//     } else if (PhysicalNestedLoopJoin::IsSupported(op.conditions, op.join_type)) {
//       // throw NotImplementedException("Nested loop join not supported in GPU");
//       // inequality join: use nested loop
//       plan = make_uniq<GPUPhysicalNestedLoopJoin>(op,
//                                                   std::move(left),
//                                                   std::move(right),
//                                                   std::move(op.conditions),
//                                                   op.join_type,
//                                                   op.estimated_cardinality);
//     } else {
//       throw NotImplementedException("Blockwise nested loop join not supported in GPU");
//       // for (auto &cond : op.conditions) {
//       // 	RewriteJoinCondition(*cond.right, left->types.size());
//       // }
//       // auto condition = JoinCondition::CreateExpression(std::move(op.conditions));
//       // plan = make_uniq<PhysicalBlockwiseNLJoin>(op, std::move(left), std::move(right),
//       // std::move(condition),
//       //                                           op.join_type, op.estimated_cardinality);
//     }
//   }
//   return plan;
// }

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalComparisonJoin& op)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_ASOF_JOIN:
      // return PlanAsOfJoin(op);
      throw NotImplementedException("Asof join not supported in GPU");
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: return PlanComparisonJoin(op);
    case LogicalOperatorType::LOGICAL_DELIM_JOIN: return PlanDelimJoin(op);
    default: throw InternalException("Unrecognized operator type for LogicalComparisonJoin");
  }
}

}  // namespace duckdb
