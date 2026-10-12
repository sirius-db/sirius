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
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalCrossProduct& op)
{
  D_ASSERT(op.children.size() == 2);

  // A cross product runs as a nested loop join without conditions, which splits each pair of
  // input batches into tasks small enough for cudf::cross_join. Its estimate is the product of the
  // child estimates. The generic create_plan() has already set op.estimated_cardinality to the
  // larger child when the join order optimizer set none.
  op.estimated_cardinality = memory::saturating_mul(op.children[0]->EstimateCardinality(context),
                                                    op.children[1]->EstimateCardinality(context));

  auto left  = create_plan(*op.children[0]);
  auto right = create_plan(*op.children[1]);
  auto join  = duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(
    op,
    std::move(left),
    std::move(right),
    duckdb::vector<sirius::join_condition>{},
    duckdb::JoinType::INNER,
    op.estimated_cardinality);
  // The output of each task aims for cross_join_task_bytes.
  if (auto sirius_context = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")) {
    join->set_cross_join_task_bytes(
      sirius_context->get_config().get_operator_params().cross_join_task_bytes);
  }
  return join;
}

}  // namespace sirius::planner
