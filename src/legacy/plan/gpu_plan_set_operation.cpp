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

#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "operator/gpu_physical_union.hpp"

namespace duckdb {

unique_ptr<GPUPhysicalOperator> GPUPhysicalPlanGenerator::CreatePlan(LogicalSetOperation& op)
{
  D_ASSERT(op.children.size() == 2);

  auto left  = CreatePlan(*op.children[0]);
  auto right = CreatePlan(*op.children[1]);

  if (left->types != right->types) {
    throw InvalidInputException("Type mismatch for SET OPERATION");
  }

  switch (op.type) {
    case LogicalOperatorType::LOGICAL_UNION: {
      auto result = make_uniq<GPUPhysicalUnion>(
        op.types, std::move(left), std::move(right), op.estimated_cardinality);
      return result;
    }
    default:
      throw NotImplementedException("Only UNION ALL is supported on GPU, not EXCEPT/INTERSECT");
  }
}

}  // namespace duckdb
