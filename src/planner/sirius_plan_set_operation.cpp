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

//! `LogicalSetOperation` is DuckDB's single node for the whole set-operation family; the inherited
//! `LogicalOperatorType` tag picks the operation and `setop_all` picks ALL vs distinct. The
//! generator switch routes only `LOGICAL_UNION` here — `EXCEPT` / `INTERSECT` keep their own
//! throwing case — so the only discrimination left to do is on `setop_all`.
duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalSetOperation& op)
{
  // Distinct UNION is `DISTINCT(UNION ALL)`: the binder rewrites "not ALL" into a modifier that
  // lowers to a separate LOGICAL_DISTINCT above this node. That node has no builder yet, so a
  // distinct UNION would fall back one node later anyway; rejecting here keeps the message
  // specific. Delete this throw once the DISTINCT builder lands.
  if (!op.setop_all) {
    throw duckdb::NotImplementedException(
      "UNION (distinct) not supported yet; only UNION ALL is on the GPU path");
  }

  // `allow_out_of_order == false` asks for the arms to be evaluated strictly left to right —
  // DuckDB's own PhysicalUnion adds pipeline dependencies to enforce it. This operator is built on
  // N independent arms drained in whatever order they produce, so it cannot honor that and must
  // decline the node. The main binder path always passes the `true` default; EXPORT DATABASE and
  // deserialized plans are the shapes that can carry `false`, and they carry `setop_all == true`,
  // so the guard above does not catch them. What this prevents is a silently mis-ordered result
  // rather than an exception, which is why it is a guard and not a reachability argument.
  if (!op.allow_out_of_order) {
    throw duckdb::NotImplementedException(
      "UNION ALL with ordered arms (allow_out_of_order = false) not supported on the GPU path");
  }

  // No Sirius-side schema reconciliation: the binder has already cast every arm to the per-column
  // common super-type and guaranteed `op.column_count` columns in matching order. Any cast that
  // was needed is already a LOGICAL_PROJECTION inside the arm's subtree.
  //
  // N-ary: `a UNION ALL b UNION ALL c` binds to one node with three children, and `UNION BY NAME`
  // and macro expansion can produce N-ary nodes too, so this loops rather than indexing 0 and 1.
  D_ASSERT(op.children.size() >= 2);
  if (op.children.size() < 2) {
    throw duckdb::NotImplementedException("UNION ALL with fewer than two inputs not supported");
  }

  auto union_op = duckdb::make_uniq<sirius::op::sirius_physical_union>(
    sirius::from_duckdb_vec(op.types), op.estimated_cardinality);

  for (auto& child : op.children) {
    auto child_plan = create_plan(*child);
    // The arity guarantee above is a binder invariant, not something re-derived here. Check it
    // anyway: a mismatch means the invariant was misread, and falling back to the CPU is better
    // than emitting a differently-shaped batch on one arm.
    if (child_plan->types.size() != op.types.size()) {
      throw duckdb::NotImplementedException(
        "UNION ALL: input column count does not match the set operation output");
    }
    union_op->children.push_back(std::move(child_plan));
  }

  return union_op;
}

}  // namespace sirius::planner
