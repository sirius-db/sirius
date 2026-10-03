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

#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "expression/ast/node.hpp"
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_union.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <string>
#include <string_view>

namespace sirius::planner {

// The generator switch routes only `LOGICAL_UNION` to this builder — `EXCEPT` / `INTERSECT` keep
// their own throwing case — so the UNION body below the fork checks only `setop_all` and
// `allow_out_of_order`.
duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalSetOperation& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT: return plan_except_intersect(op);
    case duckdb::LogicalOperatorType::LOGICAL_UNION: break;
    default: throw duckdb::InternalException("Unrecognized operator type for LogicalSetOperation");
  }

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

namespace {

//! The join a filtering set operation lowers to, and its SQL keyword for messages.
struct filtering_set_operation {
  duckdb::JoinType join_type;
  std::string_view name;
};

filtering_set_operation filtering_set_operation_of(duckdb::LogicalOperatorType type)
{
  switch (type) {
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      return {duckdb::JoinType::SEMI, "INTERSECT"};
    default: throw duckdb::InternalException("Unrecognized filtering set operation type");
  }
}

//! Whether DuckDB would wrap a key of @p type in a collation function before comparing it.
bool key_needs_collation(duckdb::ClientContext& context,
                         duckdb::LogicalType const& type,
                         duckdb::idx_t column)
{
  duckdb::unique_ptr<duckdb::Expression> key =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, column);
  return duckdb::ExpressionBinder::PushCollation(context, key, type);
}

//! One `IS NOT DISTINCT FROM` condition per column, pairing column `i` of both inputs.
duckdb::vector<duckdb::JoinCondition> null_safe_column_conditions(
  duckdb::vector<duckdb::LogicalType> const& types)
{
  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.reserve(types.size());
  for (duckdb::idx_t i = 0; i < types.size(); ++i) {
    duckdb::JoinCondition condition;
    condition.left       = duckdb::make_uniq<duckdb::BoundReferenceExpression>(types[i], i);
    condition.right      = duckdb::make_uniq<duckdb::BoundReferenceExpression>(types[i], i);
    condition.comparison = duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM;
    conditions.push_back(std::move(condition));
  }
  return conditions;
}

//! Refuses a planned input whose declared types differ from the set operation's output types.
void require_input_types(sirius::op::sirius_physical_operator const& input,
                         duckdb::idx_t input_index,
                         duckdb::vector<sirius::logical_type> const& output_types,
                         std::string const& name)
{
  if (input.types.size() != output_types.size()) {
    throw duckdb::NotImplementedException("%s input %d is planned with %d columns, not %d",
                                          name,
                                          input_index,
                                          static_cast<duckdb::idx_t>(input.types.size()),
                                          static_cast<duckdb::idx_t>(output_types.size()));
  }
  for (duckdb::idx_t i = 0; i < output_types.size(); ++i) {
    if (input.types[i] != output_types[i]) {
      throw duckdb::NotImplementedException("%s input %d plans column %d as %s, not %s",
                                            name,
                                            input_index,
                                            i,
                                            input.types[i].to_string(),
                                            output_types[i].to_string());
    }
  }
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_except_intersect(duckdb::LogicalSetOperation& op)
{
  auto const set_op = filtering_set_operation_of(op.type);
  std::string const name{set_op.name};

  // The ALL forms need a ROW_NUMBER window on each input, and there is no window operator.
  if (op.setop_all) {
    throw duckdb::NotImplementedException("%s ALL not supported on the GPU path", name);
  }

  D_ASSERT(op.children.size() == 2);
  if (op.children.size() != 2) {
    throw duckdb::NotImplementedException("%s requires exactly two inputs", name);
  }

  for (duckdb::idx_t i = 0; i < op.types.size(); ++i) {
    reject_nested_column_type(op.types[i], "column " + std::to_string(i), name);
    // DuckDB compares a collated key through its collation; the GPU would compare raw values.
    if (key_needs_collation(context, op.types[i], i)) {
      throw duckdb::NotImplementedException(
        "%s on column %d (%s): collated keys not supported on the GPU path",
        name,
        i,
        op.types[i].ToString());
    }
  }

  auto const output_types = sirius::from_duckdb_vec(op.types);
  auto conditions         = sirius::wrap_join_conditions(null_safe_column_conditions(op.types));
  if (!sirius::op::sirius_physical_hash_join::are_conditions_supported(conditions,
                                                                       set_op.join_type)) {
    throw duckdb::NotImplementedException("%s keys not supported by the GPU hash join", name);
  }

  // Input 0 is the probe and output side, input 1 the build side.
  auto left  = create_plan(*op.children[0]);
  auto right = create_plan(*op.children[1]);
  // The join and its wrappers read each input's declared `types`.
  require_input_types(*left, 0, output_types, name);
  require_input_types(*right, 1, output_types, name);

  // Without a SiriusContext the join takes default-constructed `operator_params`.
  sirius::operator_params op_params;
  auto const sirius_ctx = context.registered_state
                            ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                            : nullptr;
  if (sirius_ctx) { op_params = sirius_ctx->get_config().get_operator_params(); }

  return duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(
    op,
    std::move(left),
    std::move(right),
    std::move(conditions),
    set_op.join_type,
    duckdb::vector<std::size_t>{},
    duckdb::vector<std::size_t>{},
    duckdb::vector<sirius::logical_type>{},
    op.estimated_cardinality,
    op_params.max_build_hash_table_bytes,
    sirius::op::dynamic_filter_publish_plan{},
    op_params.hash_partition_bytes,
    op_params.max_broadcast_join_size,
    nullptr);
}

}  // namespace sirius::planner
