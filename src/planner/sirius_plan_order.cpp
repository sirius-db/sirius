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

#include "duckdb/planner/operator/logical_order.hpp"
#include "op/sirius_physical_order.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalOrder& op)
{
  D_ASSERT(op.children.size() == 1);

  auto plan = create_plan(*op.children[0]);
  if (!op.orders.empty()) {
    duckdb::vector<duckdb::idx_t> projection_map;
    if (op.HasProjectionMap()) {
      projection_map = std::move(op.projection_map);
    } else {
      for (duckdb::idx_t i = 0; i < plan->types.size(); i++) {
        projection_map.push_back(i);
      }
    }
    auto order = duckdb::make_uniq<sirius::op::sirius_physical_order>(
      op.types, std::move(op.orders), std::move(projection_map), op.estimated_cardinality);
    order->children.push_back(std::move(plan));
    plan = std::move(order);
  }
  return plan;
}

}  // namespace sirius::planner
