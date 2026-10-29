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
#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_empty_result.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_replicate.hpp"
#include "op/sirius_physical_union.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius/exception.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::planner {

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalSetOperation& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_UNION: return plan_union(op);
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT: return plan_except_intersect(op);
    default: throw sirius::internal_exception("Unrecognized operator type for LogicalSetOperation");
  }
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> sirius_physical_plan_generator::plan_union(
  duckdb::LogicalSetOperation& op)
{
  // Degenerate UNION ALL nodes need no concatenation. Preserve the declared
  // schema for an empty result and validate the schema when forwarding a child.
  if (op.setop_all && op.children.empty()) {
    return duckdb::make_uniq<sirius::op::sirius_physical_empty_result>(
      sirius::from_duckdb_vec(op.types), 0);
  }
  if (op.setop_all && op.children.size() == 1) {
    auto child = create_plan(*op.children[0]);
    require_planned_schema(child->get_types(), op.types, "UNION ALL child");
    return child;
  }

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
  return plan_dedup(std::move(union_op), key_columns, op.estimated_cardinality);
}

namespace {

//! The SQL keyword of a filtering set operation, for messages.
std::string_view set_operation_keyword(duckdb::LogicalOperatorType type)
{
  switch (type) {
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT: return "EXCEPT";
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT: return "INTERSECT";
    default: throw sirius::internal_exception("Unrecognized filtering set operation type");
  }
}

//! Whether DuckDB would wrap a key of @p type in a collation or normalization function before
//! comparing it.
bool key_needs_collation(duckdb::ClientContext& context,
                         duckdb::LogicalType const& type,
                         duckdb::idx_t column)
{
  duckdb::unique_ptr<duckdb::Expression> key =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, column);
  return duckdb::ExpressionBinder::PushCollation(context, key, type);
}

//! The types an ALL plan carries: per column, the type both planned inputs share, which must be
//! the declared type or, for an aggregate result, `planned_aggregate_type` of it.
duckdb::vector<sirius::logical_type> reconcile_input_types(
  std::array<duckdb::unique_ptr<sirius::op::sirius_physical_operator>, 2> const& arms,
  duckdb::vector<duckdb::LogicalType> const& declared_types,
  std::string const& name)
{
  auto const planned_types = [&](duckdb::idx_t arm) -> auto const& {
    return arms[arm]->get_output_types();
  };
  for (duckdb::idx_t arm = 0; arm < arms.size(); ++arm) {
    if (planned_types(arm).size() != declared_types.size()) {
      throw sirius::not_implemented_exception("{} input {} is planned with {} columns, not {}",
                                              name,
                                              arm,
                                              static_cast<duckdb::idx_t>(planned_types(arm).size()),
                                              static_cast<duckdb::idx_t>(declared_types.size()));
    }
  }
  duckdb::vector<sirius::logical_type> key_types;
  for (duckdb::idx_t i = 0; i < declared_types.size(); ++i) {
    auto const declared = sirius::from_duckdb(declared_types[i]);
    auto const narrowed = sirius::from_duckdb(
      sirius_physical_plan_generator::planned_aggregate_type(declared_types[i]));
    for (duckdb::idx_t arm = 0; arm < arms.size(); ++arm) {
      auto const& planned = planned_types(arm)[i];
      if (planned != declared && planned != narrowed) {
        throw sirius::not_implemented_exception("{} input {} plans column {} as {}, not {}",
                                                name,
                                                arm,
                                                i,
                                                planned.to_string(),
                                                declared.to_string());
      }
    }
    if (planned_types(0)[i] != planned_types(1)[i]) {
      throw sirius::not_implemented_exception("{} inputs plan column {} as {} and {}",
                                              name,
                                              i,
                                              planned_types(0)[i].to_string(),
                                              planned_types(1)[i].to_string());
    }
    key_types.push_back(planned_types(0)[i]);
  }
  return key_types;
}

//! The query's `operator_params`, including session overrides.
sirius::operator_params current_operator_params(duckdb::ClientContext& context)
{
  return *duckdb::query_operator_options(context);
}

std::unique_ptr<sirius::ast::node> column_reference(std::size_t column, sirius::logical_type type)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::reference{static_cast<std::uint32_t>(column), std::move(type)});
}

//! Arm @p arm's tag values for an ALL form. Per group, the tag columns sum to `m - n` for
//! EXCEPT ALL, and to `m` and `n` for INTERSECT ALL, where `m` and `n` count the group's rows in
//! arms 0 and 1.
std::vector<std::int8_t> input_tags(duckdb::LogicalOperatorType op_type, std::size_t arm)
{
  bool const left = arm == 0;
  switch (op_type) {
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
      return {left ? std::int8_t{1} : std::int8_t{-1}};
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      return {left ? std::int8_t{1} : std::int8_t{0}, left ? std::int8_t{0} : std::int8_t{1}};
    default: throw sirius::internal_exception("Unrecognized filtering set operation type");
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
std::unique_ptr<sirius::ast::node> copy_count(duckdb::LogicalOperatorType op_type,
                                              std::size_t first_sum)
{
  auto const bigint = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto const sum = [&](std::size_t offset) { return column_reference(first_sum + offset, bigint); };
  switch (op_type) {
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
      return greater_of(sum(0),
                        std::make_unique<sirius::ast::node>(
                          sirius::ast::constant{sirius::value{std::int64_t{0}}, bigint}));
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT: return lesser_of(sum(0), sum(1));
    default: throw sirius::internal_exception("Unrecognized filtering set operation type");
  }
}

//! References to columns `0 .. key_types.size() - 1`, one per key.
duckdb::vector<std::unique_ptr<sirius::ast::node>> key_references(
  duckdb::vector<sirius::logical_type> const& key_types)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> references;
  for (std::size_t i = 0; i < key_types.size(); ++i) {
    references.push_back(column_reference(i, key_types[i]));
  }
  return references;
}

//! UNION ALL of the inputs, each projected to its key columns followed by its tags.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> union_tagged_inputs(
  duckdb::LogicalOperatorType op_type,
  std::array<duckdb::unique_ptr<sirius::op::sirius_physical_operator>, 2> arms,
  duckdb::vector<sirius::logical_type> const& key_types,
  std::size_t estimated_cardinality)
{
  auto const tinyint = sirius::logical_type::make(sirius::type_id::TINYINT);
  auto union_types   = key_types;
  union_types.insert(union_types.end(), input_tags(op_type, 0).size(), tinyint);
  auto tagged_union =
    duckdb::make_uniq<sirius::op::sirius_physical_union>(union_types, estimated_cardinality);
  for (std::size_t arm = 0; arm < arms.size(); ++arm) {
    auto tag_list = key_references(key_types);
    for (auto const tag : input_tags(op_type, arm)) {
      tag_list.push_back(
        std::make_unique<sirius::ast::node>(sirius::ast::constant{sirius::value{tag}, tinyint}));
    }
    auto const arm_cardinality = arms[arm]->estimated_cardinality;
    tagged_union->children.push_back(
      push_projection(std::move(arms[arm]), union_types, std::move(tag_list), arm_cardinality));
  }
  return tagged_union;
}

//! Groups @p tagged_union by its key columns and sums each of the @p tag_count tag columns after
//! them.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> sum_tags_per_group(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> tagged_union,
  duckdb::vector<sirius::logical_type> const& key_types,
  std::size_t tag_count,
  std::size_t estimated_cardinality)
{
  auto const tinyint   = sirius::logical_type::make(sirius::type_id::TINYINT);
  auto const bigint    = sirius::logical_type::make(sirius::type_id::BIGINT);
  auto const key_count = key_types.size();
  duckdb::vector<std::unique_ptr<sirius::ast::node>> sum_list;
  for (std::size_t tag = 0; tag < tag_count; ++tag) {
    std::vector<std::unique_ptr<sirius::ast::node>> arguments;
    arguments.push_back(column_reference(key_count + tag, tinyint));
    sum_list.push_back(std::make_unique<sirius::ast::node>(sirius::ast::aggregate{
      sirius::aggregate_id::sum, std::move(arguments), bigint, /*distinct=*/false}));
  }
  auto aggregate_types = key_types;
  aggregate_types.insert(aggregate_types.end(), tag_count, bigint);
  auto aggregate = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate>(
    aggregate_types, std::move(sum_list), key_references(key_types), estimated_cardinality);
  aggregate->children.push_back(std::move(tagged_union));
  return aggregate;
}

//! Projects @p tag_sums to its key columns followed by the group's copy count.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> append_copy_count(
  duckdb::LogicalOperatorType op_type,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> tag_sums,
  duckdb::vector<sirius::logical_type> const& key_types,
  std::size_t estimated_cardinality)
{
  auto count_list = key_references(key_types);
  count_list.push_back(copy_count(op_type, key_types.size()));
  auto count_types = key_types;
  count_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  return push_projection(
    std::move(tag_sums), std::move(count_types), std::move(count_list), estimated_cardinality);
}

//! Lowers an ALL form by counting, as Spark does: a distinct row seen `m` times in arm 0 and `n` in
//! arm 1 is emitted `max(m - n, 0)` or `min(m, n)` times. Data flow: tagged arms -> UNION ALL ->
//! GROUP BY every column, SUM each tag (the counts) -> copy-count projection -> REPLICATE.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan_set_operation_all(
  duckdb::LogicalOperatorType op_type,
  std::array<duckdb::unique_ptr<sirius::op::sirius_physical_operator>, 2> arms,
  duckdb::vector<sirius::logical_type> const& key_types,
  std::size_t estimated_cardinality,
  sirius::op::gpu_replicate_impl::limits replicate_limits)
{
  auto const tag_count = input_tags(op_type, 0).size();
  auto tagged_union =
    union_tagged_inputs(op_type, std::move(arms), key_types, estimated_cardinality);
  auto tag_sums =
    sum_tags_per_group(std::move(tagged_union), key_types, tag_count, estimated_cardinality);
  auto count_projection =
    append_copy_count(op_type, std::move(tag_sums), key_types, estimated_cardinality);

  auto replicate = duckdb::make_uniq<sirius::op::sirius_physical_replicate>(
    key_types, replicate_limits, estimated_cardinality);
  replicate->children.push_back(std::move(count_projection));
  return replicate;
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::plan_except_intersect(duckdb::LogicalSetOperation& op)
{
  auto const keyword = set_operation_keyword(op.type);
  // The distinct forms need a DISTINCT above a semi / anti join; only the ALL forms lower here.
  if (!op.setop_all) {
    throw sirius::not_implemented_exception("{} (distinct) not supported on the GPU path",
                                            std::string{keyword});
  }
  std::string const name = std::string{keyword} + " ALL";

  D_ASSERT(op.children.size() == 2);
  if (op.children.size() != 2) {
    throw sirius::not_implemented_exception("{} requires exactly two inputs", name);
  }

  for (duckdb::idx_t i = 0; i < op.types.size(); ++i) {
    reject_nested_column_type(op.types[i], "column " + std::to_string(i), name);
    // DuckDB compares such a key through its collation or normalization; the GPU would compare raw
    // values.
    if (key_needs_collation(context, op.types[i], i)) {
      throw sirius::not_implemented_exception(
        "{} on column {} ({}): keys DuckDB compares through a collation or normalization not "
        "supported on the GPU path",
        name,
        i,
        op.types[i].ToString());
    }
    // A group keeps one of its -0.0 / +0.0 rows; DuckDB's ALL forms return the input's own rows.
    auto const type_id = op.types[i].id();
    if (type_id == duckdb::LogicalTypeId::FLOAT || type_id == duckdb::LogicalTypeId::DOUBLE) {
      throw sirius::not_implemented_exception(
        "{} on column {} ({}): floating-point keys not supported on the GPU path",
        name,
        i,
        op.types[i].ToString());
    }
  }

  auto const op_params = current_operator_params(context);
  std::array arms{create_plan(*op.children[0]), create_plan(*op.children[1])};
  // The ALL plan is typed from these, so each arm must actually produce them.
  auto const key_types = reconcile_input_types(arms, op.types, name);
  // concat_batch_bytes = 0 leaves batches whole, as in CONCAT: no byte cap, only the row cap.
  auto const max_bytes = op_params.concat_batch_bytes == 0 ? std::numeric_limits<std::size_t>::max()
                                                           : op_params.concat_batch_bytes;
  return plan_set_operation_all(
    op.type,
    std::move(arms),
    key_types,
    op.estimated_cardinality,
    {.max_rows = std::numeric_limits<cudf::size_type>::max(), .max_bytes = max_bytes});
}

}  // namespace sirius::planner
