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

//! `LogicalSetOperation` is DuckDB's single node for the whole set-operation family. The generator
//! switch routes only `LOGICAL_UNION` here — `EXCEPT` / `INTERSECT` keep their own throwing case —
//! so the only discrimination left is on `setop_all`.
duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalSetOperation& op)
{
  // A distinct UNION usually lowers to a LOGICAL_DISTINCT above this node, so declining here
  // rather than one node later only sharpens the message. Usually, not always: a WITH RECURSIVE
  // body with no self-reference degrades to a plain LogicalSetOperation carrying the CTE's
  // `union_all`, and nothing inserts a DistinctModifier on that path (duckdb
  // `bind_recursive_cte_node.cpp:124`-`:127`). For that shape this throw is the only thing between
  // a distinct UNION and duplicate rows.
  //
  // So lifting this once the DISTINCT builder lands is not a deletion: `create_plan` dispatches on
  // node type with no parent context, so this builder cannot tell whether a LOGICAL_DISTINCT sits
  // above it. The pair has to be recognised from the DISTINCT side and planned as one dedup over a
  // bag union, leaving this throw for the bare case.
  if (!op.setop_all) {
    throw duckdb::NotImplementedException(
      "UNION (distinct) not supported yet; only UNION ALL is on the GPU path");
  }

  // `allow_out_of_order == false` asks for the arms to be evaluated strictly left to right, which
  // N independently drained arms cannot honor. EXPORT DATABASE and deserialized plans are the
  // shapes that carry `false`, and they carry `setop_all == true`, so the guard above does not
  // catch them. Without this the result is silently mis-ordered rather than an error.
  if (!op.allow_out_of_order) {
    throw duckdb::NotImplementedException(
      "UNION ALL with ordered arms (allow_out_of_order = false) not supported on the GPU path");
  }

  // No Sirius-side schema reconciliation: the binder has already cast every arm to the common
  // super-type, and any cast needed is a LOGICAL_PROJECTION inside the arm's subtree. N-ary:
  // `a UNION ALL b UNION ALL c` binds to one node with three children, so this loops.
  D_ASSERT(op.children.size() >= 2);
  if (op.children.size() < 2) {
    throw duckdb::NotImplementedException("UNION ALL with fewer than two inputs not supported");
  }

  auto union_op = duckdb::make_uniq<sirius::op::sirius_physical_union>(
    sirius::from_duckdb_vec(op.types), op.estimated_cardinality);

  for (auto& child : op.children) {
    auto child_plan = create_plan(*child);
    // Re-check the binder's arity invariant: a mismatch means it was misread, and falling back to
    // the CPU beats emitting a differently-shaped batch on one arm.
    if (child_plan->types.size() != op.types.size()) {
      throw duckdb::NotImplementedException(
        "UNION ALL: input column count does not match the set operation output");
    }
    union_op->children.push_back(std::move(child_plan));
  }

  return union_op;
}

}  // namespace sirius::planner
