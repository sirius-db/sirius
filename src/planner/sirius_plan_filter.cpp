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

#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_projection.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalFilter& op)
{
  D_ASSERT(op.children.size() == 1);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan = create_plan(*op.children[0]);
  if (!op.expressions.empty()) {
    D_ASSERT(plan->types.size() > 0);
    // create a filter if there is anything to filter
    auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(
      plan->types, std::move(op.expressions), op.estimated_cardinality);
    filter->children.push_back(std::move(plan));
    plan = std::move(filter);
  }
  if (op.HasProjectionMap()) {
    // there is a projection map, generate a physical projection
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    for (duckdb::idx_t i = 0; i < op.projection_map.size(); i++) {
      select_list.push_back(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(op.types[i], op.projection_map[i]));
    }
    auto proj = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
      op.types, std::move(select_list), op.estimated_cardinality);
    proj->children.push_back(std::move(plan));
    plan = std::move(proj);
  }
  return plan;
}

}  // namespace sirius::planner
