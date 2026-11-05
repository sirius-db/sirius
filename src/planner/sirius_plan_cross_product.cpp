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

#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "helper/type_conversions.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::make_cross_product_join(
  duckdb::LogicalOperator& op,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> left,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> right,
  duckdb::JoinType join_type,
  std::size_t estimated_cardinality,
  duckdb::vector<std::size_t> left_projection_map,
  duckdb::vector<std::size_t> right_projection_map)
{
  // A cross product runs as a nested loop join without conditions, which splits each pair of
  // input batches into tasks small enough for cudf::cross_join.
  auto join = duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(
    op,
    std::move(left),
    std::move(right),
    duckdb::vector<sirius::join_condition>{},
    join_type,
    estimated_cardinality,
    std::move(left_projection_map),
    std::move(right_projection_map));
  // The output of each task aims for cross_join_task_bytes.
  if (auto sirius_context = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")) {
    join->set_cross_join_task_bytes(duckdb::query_operator_options(context)->cross_join_task_bytes);
  }
  return join;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalCrossProduct& op)
{
  D_ASSERT(op.children.size() == 2);

  auto const lhs_cardinality = op.children[0]->EstimateCardinality(context);
  auto const rhs_cardinality = op.children[1]->EstimateCardinality(context);
  auto left                  = create_plan(*op.children[0]);
  auto right                 = create_plan(*op.children[1]);
  // The estimate is the product of the child estimates. The generic create_plan() has already set
  // op.estimated_cardinality to the larger child when the join order optimizer set none, and it
  // copies op.estimated_cardinality onto the returned plan.
  op.estimated_cardinality = memory::saturating_mul(lhs_cardinality, rhs_cardinality);
  return make_cross_product_join(op,
                                 std::move(left),
                                 std::move(right),
                                 duckdb::JoinType::INNER,
                                 op.estimated_cardinality,
                                 /*left_projection_map=*/{},
                                 /*right_projection_map=*/{});
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalAnyJoin& op)
{
  D_ASSERT(op.children.size() == 2);
  D_ASSERT(op.condition);

  // DuckDB plans an ANY join as a blockwise nested loop join that evaluates the condition on each
  // (left, right) pair. For INNER that equals filtering the cross product by the condition, whose
  // references already index the left columns followed by the right columns. Outer, semi, anti and
  // mark joins must evaluate the condition while matching rows, which the GPU does not do yet.
  // The cost is that of the cross product: all |L|·|R| pairs are considered, and each slice of the
  // cross product is materialized before the filter drops the pairs that fail the condition.
  if (op.join_type != duckdb::JoinType::INNER) {
    throw duckdb::NotImplementedException("Any join with join type " +
                                          duckdb::JoinTypeToString(op.join_type) +
                                          " not supported in GPU");
  }
  // DuckDB's blockwise join ignores projection maps, so an ANY join never carries one.
  if (!op.left_projection_map.empty() || !op.right_projection_map.empty()) {
    throw duckdb::NotImplementedException("Any join with a projection map not supported in GPU");
  }
  reject_nested_column_operation(*op.condition, "a join predicate");
  auto predicate = sirius::ast::from_duckdb(*op.condition);
  if (!predicate) { throw duckdb::NotImplementedException("Unsupported any join condition"); }

  // op.estimated_cardinality stays DuckDB's estimate of the filtered join, which the generic
  // create_plan() copies onto the filter; only the nested loop join gets the cross product's.
  auto const estimated_cardinality = op.estimated_cardinality;
  auto const lhs_cardinality       = op.children[0]->EstimateCardinality(context);
  auto const rhs_cardinality       = op.children[1]->EstimateCardinality(context);
  auto left                        = create_plan(*op.children[0]);
  auto right                       = create_plan(*op.children[1]);
  auto cross_product =
    make_cross_product_join(op,
                            std::move(left),
                            std::move(right),
                            duckdb::JoinType::INNER,
                            memory::saturating_mul(lhs_cardinality, rhs_cardinality),
                            /*left_projection_map=*/{},
                            /*right_projection_map=*/{});
  auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(
    sirius::from_duckdb_vec(op.types), std::move(predicate), estimated_cardinality);
  filter->children.push_back(std::move(cross_product));
  return filter;
}

}  // namespace sirius::planner
