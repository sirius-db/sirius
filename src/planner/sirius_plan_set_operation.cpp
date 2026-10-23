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

#include <numeric>
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
  // `Binder::BindNode(RecursiveCTENode&)`), so this builder adds the dedup itself, at the end.
  //
  // The dedup groups on every column, and a group key cannot be nested; refuse before planning.
  if (!op.setop_all) {
    for (duckdb::idx_t i = 0; i < op.types.size(); i++) {
      reject_nested_column_type(op.types[i], "column " + std::to_string(i), "UNION");
    }
  }

  // `allow_out_of_order == false` requires output in arm order, which this operator does not
  // guarantee, so the result would be silently mis-ordered rather than an error.
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
    auto arm = create_plan(*op.children[arm_index]);
    // Arms are forwarded unconverted, so each must plan the declared schema.
    require_planned_schema(arm->get_types(), op.types, "UNION arm " + std::to_string(arm_index));
    union_op->children.push_back(std::move(arm));
  }

  if (op.setop_all) { return union_op; }
  // Group on every column with no aggregates, as DuckDB's own planner does for `!setop_all`.
  duckdb::vector<duckdb::idx_t> key_columns(declared_types.size());
  std::iota(key_columns.begin(), key_columns.end(), duckdb::idx_t{0});
  return plan_dedup(std::move(union_op), key_columns, {}, op.estimated_cardinality);
}

}  // namespace sirius::planner
