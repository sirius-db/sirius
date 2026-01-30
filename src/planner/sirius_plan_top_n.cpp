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

#include "duckdb/planner/operator/logical_top_n.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalTopN& op)
{
  D_ASSERT(op.children.size() == 1);

  auto plan = create_plan(*op.children[0]);

  auto top_n = duckdb::make_uniq<sirius::op::sirius_physical_top_n>(
    op.types,
    std::move(op.orders),
    duckdb::NumericCast<duckdb::idx_t>(op.limit),
    duckdb::NumericCast<duckdb::idx_t>(op.offset),
    std::move(op.dynamic_filter),
    op.estimated_cardinality);

  top_n->children.push_back(std::move(plan));
  return std::move(top_n);
}

}  // namespace sirius::planner
