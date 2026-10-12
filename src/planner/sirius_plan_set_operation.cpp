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
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_union.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

namespace sirius::planner {

// The generator switch routes only `LOGICAL_UNION` to this builder — `EXCEPT` / `INTERSECT` keep
// their own throwing case — so the only discrimination left here is on `setop_all`.
duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalSetOperation& op)
{
  // A distinct UNION usually lowers to a LOGICAL_DISTINCT above this node, but not always: a
  // WITH RECURSIVE body with no self-reference degrades to a plain LogicalSetOperation carrying
  // the CTE's `union_all`, and nothing inserts a DistinctModifier on that path (duckdb
  // `bind_recursive_cte_node.cpp:124`-`:127`). For that shape this throw is the only thing between
  // a distinct UNION and duplicate rows.
  if (!op.setop_all) {
    throw duckdb::NotImplementedException(
      "UNION (distinct) not supported yet; only UNION ALL is on the GPU path");
  }

  // `allow_out_of_order == false` asks for strict left-to-right evaluation, which N independently
  // drained arms cannot honor. EXPORT DATABASE and deserialized plans carry it alongside
  // `setop_all == true`, so the guard above does not catch them, and the result would be silently
  // mis-ordered rather than an error.
  if (!op.allow_out_of_order) {
    throw duckdb::NotImplementedException(
      "UNION ALL with ordered arms (allow_out_of_order = false) not supported on the GPU path");
  }

  // N-ary: `a UNION ALL b UNION ALL c` binds to one node with three children.
  D_ASSERT(op.children.size() >= 2);
  if (op.children.size() < 2) {
    throw duckdb::NotImplementedException("UNION ALL with fewer than two inputs not supported");
  }

  auto union_op = duckdb::make_uniq<sirius::op::sirius_physical_union>(
    sirius::from_duckdb_vec(op.types), op.estimated_cardinality);

  for (auto& child : op.children) {
    // Re-check the binder's arity invariant on the *logical* arm, not on `create_plan`'s result:
    // a physical root's types are not always its output schema. `sirius_physical_cte` declares its
    // materialization side (`sirius_plan_cte.cpp`), so reading the physical plan declines a valid
    // query whose arm is a materialized CTE.
    if (child->types.size() != op.types.size()) {
      throw duckdb::NotImplementedException(
        "UNION ALL: input column count does not match the set operation output");
    }
    union_op->children.push_back(create_plan(*child));
  }

  return union_op;
}

}  // namespace sirius::planner
