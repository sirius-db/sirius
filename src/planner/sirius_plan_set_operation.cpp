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
#include "expression/ast/utils.hpp"
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_replicate.hpp"
#include "op/sirius_physical_union.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::planner {

// The fork sends `EXCEPT` / `INTERSECT` to `plan_except_intersect`, so the UNION body below it
// checks only `setop_all` and `allow_out_of_order`.
duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalSetOperation& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
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
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT: return {duckdb::JoinType::ANTI, "EXCEPT"};
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

//! The types an ALL plan carries: per column, the type both planned inputs share, which must be
//! the declared type or, for an aggregate result, `planned_aggregate_type` of it.
duckdb::vector<sirius::logical_type> reconcile_input_types(
  std::array<duckdb::unique_ptr<sirius::op::sirius_physical_operator>, 2> const& inputs,
  duckdb::vector<duckdb::LogicalType> const& declared_types,
  std::string const& name)
{
  for (duckdb::idx_t input = 0; input < inputs.size(); ++input) {
    if (inputs[input]->types.size() != declared_types.size()) {
      throw duckdb::NotImplementedException("%s input %d is planned with %d columns, not %d",
                                            name,
                                            input,
                                            static_cast<duckdb::idx_t>(inputs[input]->types.size()),
                                            static_cast<duckdb::idx_t>(declared_types.size()));
    }
  }
  duckdb::vector<sirius::logical_type> types;
  for (duckdb::idx_t i = 0; i < declared_types.size(); ++i) {
    auto const declared = sirius::from_duckdb(declared_types[i]);
    auto const narrowed = sirius::from_duckdb(
      sirius_physical_plan_generator::planned_aggregate_type(declared_types[i]));
    for (duckdb::idx_t input = 0; input < inputs.size(); ++input) {
      auto const& planned = inputs[input]->types[i];
      if (planned != declared && planned != narrowed) {
        throw duckdb::NotImplementedException("%s input %d plans column %d as %s, not %s",
                                              name,
                                              input,
                                              i,
                                              planned.to_string(),
                                              declared.to_string());
      }
    }
    if (inputs[0]->types[i] != inputs[1]->types[i]) {
      throw duckdb::NotImplementedException("%s inputs plan column %d as %s and %s",
                                            name,
                                            i,
                                            inputs[0]->types[i].to_string(),
                                            inputs[1]->types[i].to_string());
    }
    types.push_back(inputs[0]->types[i]);
  }
  return types;
}

//! The connection's `operator_params`; defaults when no SiriusContext is registered.
sirius::operator_params current_operator_params(duckdb::ClientContext& context)
{
  auto const sirius_ctx = context.registered_state
                            ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                            : nullptr;
  return sirius_ctx ? sirius_ctx->get_config().get_operator_params() : sirius::operator_params{};
}

std::unique_ptr<sirius::ast::node> column_reference(std::size_t column, sirius::logical_type type)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::reference{static_cast<std::uint32_t>(column), std::move(type)});
}

//! Input @p input's tag values for an ALL form. Per group, the tag columns sum to `m - n` for
//! EXCEPT ALL, and to `m` and `n` for INTERSECT ALL, where `m` and `n` count the group's rows in
//! inputs 0 and 1.
std::vector<std::int8_t> input_tags(duckdb::LogicalOperatorType type, std::size_t input)
{
  bool const left = input == 0;
  switch (type) {
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
      return {left ? std::int8_t{1} : std::int8_t{-1}};
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      return {left ? std::int8_t{1} : std::int8_t{0}, left ? std::int8_t{0} : std::int8_t{1}};
    default: throw duckdb::InternalException("Unrecognized filtering set operation type");
  }
}

//! `CASE WHEN a <op> b THEN a ELSE b END` over `BIGINT` operands.
std::unique_ptr<sirius::ast::node> case_pick(sirius::comparison_type op,
                                             std::unique_ptr<sirius::ast::node> a,
                                             std::unique_ptr<sirius::ast::node> b)
{
  auto condition = std::make_unique<sirius::ast::node>(
    sirius::ast::comparison{op, sirius::ast::clone(*a), sirius::ast::clone(*b)});
  std::vector<sirius::ast::case_expr::when_then> cases;
  cases.push_back({std::move(condition), std::move(a)});
  return std::make_unique<sirius::ast::node>(sirius::ast::case_expr{
    std::move(cases), std::move(b), sirius::logical_type::make(sirius::type_id::BIGINT)});
}

std::unique_ptr<sirius::ast::node> greater_of(std::unique_ptr<sirius::ast::node> a,
                                              std::unique_ptr<sirius::ast::node> b)
{
  return case_pick(sirius::comparison_type::gt, std::move(a), std::move(b));
}

std::unique_ptr<sirius::ast::node> lesser_of(std::unique_ptr<sirius::ast::node> a,
                                             std::unique_ptr<sirius::ast::node> b)
{
  return case_pick(sirius::comparison_type::lt, std::move(a), std::move(b));
}

//! A group's copies from its tag sums at columns @p first_sum onward: `max(m - n, 0)` for EXCEPT
//! ALL, `min(m, n)` for INTERSECT ALL.
std::unique_ptr<sirius::ast::node> copy_count(duckdb::LogicalOperatorType type,
                                              std::size_t first_sum)
{
  auto const bigint = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto const sum = [&](std::size_t offset) { return column_reference(first_sum + offset, bigint); };
  switch (type) {
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
      return greater_of(sum(0),
                        std::make_unique<sirius::ast::node>(
                          sirius::ast::constant{sirius::value{std::int64_t{0}}, bigint}));
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT: return lesser_of(sum(0), sum(1));
    default: throw duckdb::InternalException("Unrecognized filtering set operation type");
  }
}

//! References to columns `0 .. types.size() - 1`, one per key.
duckdb::vector<std::unique_ptr<sirius::ast::node>> key_references(
  duckdb::vector<sirius::logical_type> const& types)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> references;
  for (std::size_t i = 0; i < types.size(); ++i) {
    references.push_back(column_reference(i, types[i]));
  }
  return references;
}

//! UNION ALL of the inputs, each projected to its key columns followed by its tags.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> union_tagged_inputs(
  duckdb::LogicalOperatorType type,
  std::array<duckdb::unique_ptr<sirius::op::sirius_physical_operator>, 2> inputs,
  duckdb::vector<sirius::logical_type> const& types,
  std::size_t estimated_cardinality)
{
  auto const tinyint = sirius::logical_type::make(sirius::type_id::TINYINT);
  auto union_types   = types;
  union_types.insert(union_types.end(), input_tags(type, 0).size(), tinyint);
  auto tagged_union =
    duckdb::make_uniq<sirius::op::sirius_physical_union>(union_types, estimated_cardinality);
  for (std::size_t input = 0; input < inputs.size(); ++input) {
    auto tag_list = key_references(types);
    for (auto const tag : input_tags(type, input)) {
      tag_list.push_back(
        std::make_unique<sirius::ast::node>(sirius::ast::constant{sirius::value{tag}, tinyint}));
    }
    auto const input_cardinality = inputs[input]->estimated_cardinality;
    tagged_union->children.push_back(push_projection(
      std::move(inputs[input]), union_types, std::move(tag_list), input_cardinality));
  }
  return tagged_union;
}

//! Groups @p tagged_union by its key columns and sums each of the @p tag_count tag columns after
//! them.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> sum_tags_per_group(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> tagged_union,
  duckdb::vector<sirius::logical_type> const& types,
  std::size_t tag_count,
  std::size_t estimated_cardinality)
{
  auto const tinyint   = sirius::logical_type::make(sirius::type_id::TINYINT);
  auto const bigint    = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto const key_count = types.size();
  duckdb::vector<std::unique_ptr<sirius::ast::node>> sum_list;
  for (std::size_t tag = 0; tag < tag_count; ++tag) {
    std::vector<std::unique_ptr<sirius::ast::node>> arguments;
    arguments.push_back(column_reference(key_count + tag, tinyint));
    sum_list.push_back(std::make_unique<sirius::ast::node>(sirius::ast::aggregate{
      sirius::aggregate_id::sum, std::move(arguments), bigint, /*distinct=*/false}));
  }
  auto aggregate_types = types;
  aggregate_types.insert(aggregate_types.end(), tag_count, bigint);
  auto aggregate = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate>(
    aggregate_types, std::move(sum_list), key_references(types), estimated_cardinality);
  aggregate->children.push_back(std::move(tagged_union));
  return aggregate;
}

//! Projects @p tag_sums to its key columns followed by the group's copy count.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> append_copy_count(
  duckdb::LogicalOperatorType type,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> tag_sums,
  duckdb::vector<sirius::logical_type> const& types,
  std::size_t estimated_cardinality)
{
  auto count_list = key_references(types);
  count_list.push_back(copy_count(type, types.size()));
  auto count_types = types;
  count_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  return push_projection(
    std::move(tag_sums), std::move(count_types), std::move(count_list), estimated_cardinality);
}

//! Lowers an ALL form over its planned inputs, as Spark does: tag each input's rows, UNION ALL,
//! sum the tags per group of every column, turn the sums into copies, and repeat each group's
//! row that many times.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan_set_operation_all(
  duckdb::LogicalOperatorType type,
  std::array<duckdb::unique_ptr<sirius::op::sirius_physical_operator>, 2> inputs,
  duckdb::vector<sirius::logical_type> const& types,
  std::size_t estimated_cardinality,
  sirius::op::gpu_replicate_impl::limits replicate_limits)
{
  auto const tag_count = input_tags(type, 0).size();
  auto tagged_union    = union_tagged_inputs(type, std::move(inputs), types, estimated_cardinality);
  auto tag_sums =
    sum_tags_per_group(std::move(tagged_union), types, tag_count, estimated_cardinality);
  auto count_projection =
    append_copy_count(type, std::move(tag_sums), types, estimated_cardinality);

  auto replicate = duckdb::make_uniq<sirius::op::sirius_physical_replicate>(
    types, static_cast<cudf::size_type>(types.size()), replicate_limits, estimated_cardinality);
  replicate->children.push_back(std::move(count_projection));
  return replicate;
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_except_intersect(duckdb::LogicalSetOperation& op)
{
  auto const set_op = filtering_set_operation_of(op.type);
  std::string const name =
    op.setop_all ? std::string{set_op.name} + " ALL" : std::string{set_op.name};

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
    // A group keeps one of its -0.0 / +0.0 rows; DuckDB's ALL forms return the input's own rows.
    auto const type_id = op.types[i].id();
    if (op.setop_all &&
        (type_id == duckdb::LogicalTypeId::FLOAT || type_id == duckdb::LogicalTypeId::DOUBLE)) {
      throw duckdb::NotImplementedException(
        "%s on column %d (%s): floating-point keys not supported on the GPU path",
        name,
        i,
        op.types[i].ToString());
    }
  }

  auto const op_params = current_operator_params(context);
  if (op.setop_all) {
    std::array inputs{create_plan(*op.children[0]), create_plan(*op.children[1])};
    // Each tag projection and the union read their input's planned `types`.
    auto const types = reconcile_input_types(inputs, op.types, name);
    return plan_set_operation_all(
      op.type,
      std::move(inputs),
      types,
      op.estimated_cardinality,
      {std::numeric_limits<cudf::size_type>::max(), op_params.concat_batch_bytes});
  }

  auto const output_types = sirius::from_duckdb_vec(op.types);

  auto conditions = sirius::wrap_join_conditions(null_safe_column_conditions(op.types));
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
