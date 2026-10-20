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

#include <string>

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalSetOperation& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_UNION: return plan_union(op);
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      throw duckdb::NotImplementedException(
        "EXCEPT and INTERSECT are not supported on the GPU (falling back to CPU)");
    default: throw duckdb::InternalException("Unrecognized operator type for LogicalSetOperation");
  }
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> sirius_physical_plan_generator::plan_union(
  duckdb::LogicalSetOperation& op)
{
  // `UNION` and `UNION ALL` both arrive as this node. For `UNION`, DuckDB's binder plants a
  // DISTINCT above it and sets `setop_all`, so the node itself always concatenates. The exception
  // is a WITH RECURSIVE body that never references itself and has no USING KEY: it binds to this
  // node with `setop_all` false and nothing above it to deduplicate (duckdb
  // `Binder::BindNode(RecursiveCTENode&)`). This throw keeps that shape from returning duplicate
  // rows.
  if (!op.setop_all) {
    throw duckdb::NotImplementedException(
      "UNION that must deduplicate itself (a WITH RECURSIVE body with no self-reference) is not "
      "supported on the GPU (falling back to CPU)");
  }

  // `allow_out_of_order == false` requires output in arm order, which this operator does not
  // guarantee. It arrives with `setop_all == true`, so the guard above does not catch it, and the
  // result would be silently mis-ordered rather than an error.
  if (!op.allow_out_of_order) {
    throw duckdb::NotImplementedException(
      "UNION with ordered arms (allow_out_of_order = false) is not supported on the GPU (falling "
      "back to CPU)");
  }

  // N-ary: a chain of one kind, `a UNION b UNION c` or all `UNION ALL`, binds to one node with
  // three children. A mixed chain nests.
  D_ASSERT(op.children.size() >= 2);
  if (op.children.size() < 2) {
    throw duckdb::NotImplementedException(
      "UNION with fewer than two inputs is not supported on the GPU (falling back to CPU)");
  }

  auto declared_types = sirius::from_duckdb_vec(op.types);
  auto union_op =
    duckdb::make_uniq<sirius::op::sirius_physical_union>(declared_types, op.estimated_cardinality);

  for (duckdb::idx_t arm_index = 0; arm_index < op.children.size(); arm_index++) {
    auto arm                  = create_plan(*op.children[arm_index]);
    auto const& planned_types = arm->get_types();
    if (planned_types.size() != declared_types.size()) {
      throw duckdb::NotImplementedException(
        "UNION: input column count does not match the set operation output (falling back to "
        "CPU)");
    }
    // Arms are forwarded unconverted, so each must plan the declared type or, below an
    // aggregate, its `planned_aggregate_type`.
    for (duckdb::idx_t i = 0; i < declared_types.size(); i++) {
      if (plans_declared_type(planned_types[i], op.types[i])) { continue; }
      throw duckdb::NotImplementedException(
        "UNION: arm " + std::to_string(arm_index) + " plans " + planned_types[i].to_string() +
        " for column " + std::to_string(i) + " but the union declares " +
        declared_types[i].to_string() + " (falling back to CPU)");
    }
    union_op->children.push_back(std::move(arm));
  }

  return union_op;
}

}  // namespace sirius::planner
