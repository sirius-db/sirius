/*
 * Copyright 2026, Sirius Contributors.
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

#include "planner/sirius_plan_window.hpp"

#include "cudf/cudf_utils.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/aggregate_function.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/partition/partition_sizing.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_window.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace sirius::planner {
namespace {

/// Throw the NotImplementedException that sends the query to the CPU.
[[noreturn]] void decline(char const* reason)
{
  throw duckdb::NotImplementedException("Unsupported window: %s", reason);
}

/// The numeric types the window evaluator accepts as an aggregate argument.
bool is_supported_numeric_argument(duckdb::LogicalType const& type)
{
  using duckdb::LogicalTypeId;
  switch (type.id()) {
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::FLOAT:
    case LogicalTypeId::DOUBLE: return true;
    case LogicalTypeId::DECIMAL:
      return duckdb::DecimalType::GetWidth(type) >
             sirius::logical_type::decimal_max_precision_int16;
    default: return false;
  }
}

/// The types the window evaluator accepts as a partition key.
bool is_supported_partition_key(duckdb::LogicalType const& type)
{
  using duckdb::LogicalTypeId;
  switch (type.id()) {
    case LogicalTypeId::BOOLEAN:
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::DATE:
    case LogicalTypeId::TIMESTAMP:
    case LogicalTypeId::TIMESTAMP_MS:
    case LogicalTypeId::TIMESTAMP_NS:
    case LogicalTypeId::TIMESTAMP_SEC: return true;
    case LogicalTypeId::HUGEINT: return true;
    case LogicalTypeId::DECIMAL:
      return duckdb::DecimalType::GetWidth(type) >
             sirius::logical_type::decimal_max_precision_int16;
    case LogicalTypeId::VARCHAR: return duckdb::StringType::GetCollation(type).empty();
    default: return false;
  }
}

/// Whether COUNT, MIN and MAX accept an argument of @p type.
bool is_supported_count_min_max_argument(duckdb::LogicalType const& type)
{
  return is_supported_numeric_argument(type) || is_supported_partition_key(type);
}

/// Decline the window modifiers that no window function here supports.
void decline_unsupported_modifiers(duckdb::BoundWindowExpression const& expression)
{
  if (expression.Distinct()) { decline("DISTINCT"); }
  if (expression.Filter()) { decline("FILTER"); }
  if (expression.WindowExclude() != duckdb::WindowExcludeMode::NO_OTHER) { decline("EXCLUDE"); }
  if (!expression.ArgOrders().empty()) { decline("argument ORDER BY"); }
  if (expression.IgnoreNulls()) { decline("IGNORE NULLS"); }
  if (expression.StartExpr() || expression.EndExpr()) {
    decline("frame offsets or navigation arguments");
  }
}

/**
 * @brief Return the ranking function of @p expression, or `std::nullopt` when it is not one.
 *
 * Valid is `rank`, `dense_rank` or `row_number` with a window `ORDER BY`, with or without a
 * `PARTITION BY`, and without `DISTINCT`, `FILTER`, `EXCLUDE`, `IGNORE NULLS`, arguments or frame
 * offsets.
 */
std::optional<op::ranking_function> validate_ranking(
  duckdb::BoundWindowExpression const& expression)
{
  std::optional<op::ranking_function> function;
  switch (expression.GetExpressionType()) {
    case duckdb::ExpressionType::WINDOW_RANK: function = op::ranking_function::rank; break;
    case duckdb::ExpressionType::WINDOW_RANK_DENSE:
      function = op::ranking_function::dense_rank;
      break;
    case duckdb::ExpressionType::WINDOW_ROW_NUMBER:
      function = op::ranking_function::row_number;
      break;
    default: return std::nullopt;
  }
  decline_unsupported_modifiers(expression);
  if (!expression.GetChildren().empty()) { decline("ranking arguments"); }
  if (expression.GetReturnType() != duckdb::LogicalType::BIGINT) { decline("ranking result type"); }
  if (expression.OrderBy().empty()) { decline("ranking without ORDER BY"); }
  return function;
}

/// The types a window ORDER BY key accepts: the numeric types and the partition key types.
bool is_supported_order_key(duckdb::LogicalType const& type)
{
  return is_supported_partition_key(type) || is_supported_numeric_argument(type);
}

/// Whether two windows have the same ORDER BY keys, directions and NULL placement.
bool same_order(duckdb::BoundWindowExpression const& a, duckdb::BoundWindowExpression const& b)
{
  if (a.OrderBy().size() != b.OrderBy().size()) { return false; }
  for (std::size_t i = 0; i < a.OrderBy().size(); ++i) {
    auto const& left  = a.OrderBy()[i];
    auto const& right = b.OrderBy()[i];
    if (left.type != right.type || left.null_order != right.null_order ||
        !left.expression->Equals(*right.expression)) {
      return false;
    }
  }
  return true;
}

/// Record the order key slots, directions and NULL placements of the first ranking window.
template <typename Prepare>
void plan_ranking_order(window_plan& plan,
                        duckdb::BoundWindowExpression const& expression,
                        Prepare&& prepare)
{
  for (auto const& node : expression.OrderBy()) {
    if (node.null_order == duckdb::OrderByNullType::ORDER_DEFAULT) {
      decline("unresolved NULL order");
    }
    if (!is_supported_order_key(node.expression->GetReturnType())) {
      decline("order key type or collation");
    }
    plan.ranking_order.columns.push_back(static_cast<cudf::size_type>(prepare(*node.expression)));
    plan.ranking_order.directions.push_back(op::to_cudf_order(node.type));
    plan.ranking_order.nulls.push_back(op::to_cudf_null_order(node.type, node.null_order));
  }
}

/**
 * @brief Return the aggregate to run for @p expression, or decline it.
 *
 * Valid is a `SUM`, `AVG`, `COUNT`, `COUNT(*)`, `MIN` or `MAX` whose frame covers the whole
 * partition, without `DISTINCT`, `FILTER`, `EXCLUDE`, `IGNORE NULLS`, an argument `ORDER BY` or a
 * window `ORDER BY`. `SUM` and `AVG` take a numeric argument. `COUNT`, `MIN` and `MAX` also take
 * the types allowed as partition keys, and `MIN` and `MAX` keep the argument type.
 */
aggregate_id validate(duckdb::BoundWindowExpression const& expression)
{
  using duckdb::WindowBoundary;
  if (expression.GetExpressionType() != duckdb::ExpressionType::WINDOW_AGGREGATE ||
      !expression.AggregateFunction()) {
    decline("only SUM, AVG, COUNT, MIN and MAX aggregates are planned");
  }
  decline_unsupported_modifiers(expression);
  if (!expression.OrderBy().empty()) { decline("window ORDER BY"); }
  // Without ORDER BY all rows are peers. DuckDB retains CURRENT_ROW_RANGE for
  // the default frame. CURRENT_ROW_ROWS still ends at the physical current row.
  if (expression.WindowStart() != WindowBoundary::UNBOUNDED_PRECEDING ||
      (expression.WindowEnd() != WindowBoundary::UNBOUNDED_FOLLOWING &&
       expression.WindowEnd() != WindowBoundary::CURRENT_ROW_RANGE)) {
    decline("frame does not cover the whole partition");
  }
  auto const id =
    from_duckdb_aggregate_name(expression.AggregateFunction()->GetName().GetIdentifierName());
  // Every aggregate below also requires the result type DuckDB bound. Any other overload is
  // declined, so the result column keeps the type the rest of the plan expects.
  // DuckDB names the nullary COUNT overload `count`. Like count_star, it counts every row.
  if ((id == aggregate_id::count_star ||
       (id == aggregate_id::count && expression.GetChildren().empty())) &&
      expression.GetReturnType() == duckdb::LogicalType::BIGINT) {
    return aggregate_id::count_star;
  }
  if (expression.GetChildren().size() == 1) {
    auto const& argument = expression.GetChildren()[0]->GetReturnType();
    if (is_supported_count_min_max_argument(argument) && id == aggregate_id::count &&
        expression.GetReturnType() == duckdb::LogicalType::BIGINT) {
      return aggregate_id::count;
    }
    if (is_supported_count_min_max_argument(argument) && expression.GetReturnType() == argument) {
      if (id == aggregate_id::min) { return aggregate_id::min; }
      if (id == aggregate_id::max) { return aggregate_id::max; }
    }
  }
  if (expression.GetChildren().size() != 1 ||
      !is_supported_numeric_argument(expression.GetChildren()[0]->GetReturnType())) {
    decline("aggregate argument type or arity");
  }
  auto const& argument = expression.GetChildren()[0]->GetReturnType();
  if (id == aggregate_id::avg && expression.GetReturnType() == duckdb::LogicalType::DOUBLE) {
    return aggregate_id::avg;
  }
  // Statistics propagation replaces SUM with this internal implementation while retaining
  // the exact bound result type. We still evaluate it using checked, widened accumulation.
  if (id == aggregate_id::sum || id == aggregate_id::sum_no_overflow) {
    duckdb::LogicalType expected = duckdb::LogicalType::HUGEINT;
    if (argument.id() == duckdb::LogicalTypeId::DECIMAL) {
      expected = duckdb::LogicalType::DECIMAL(38, duckdb::DecimalType::GetScale(argument));
    } else if (argument.id() == duckdb::LogicalTypeId::FLOAT ||
               argument.id() == duckdb::LogicalTypeId::DOUBLE) {
      expected = duckdb::LogicalType::DOUBLE;
    }
    if (expression.GetReturnType() == expected) { return aggregate_id::sum; }
  }
  decline("aggregate function or result type");
}

// Plans the expressions of `window` listed in `indices` as one stage over an input of
// `input_types`. The input starts with the window child's columns, and a later stage of the same
// window also holds the results of the stages below it. The output is the input followed by the
// results of `indices`.
window_plan plan_window_stage(duckdb::LogicalWindow const& window,
                              std::vector<duckdb::idx_t> const& indices,
                              duckdb::vector<duckdb::LogicalType> const& input_types)
{
  if (window.children.size() != 1 || indices.empty()) {
    decline("expected one child and at least one expression");
  }
  auto const child_width = window.children[0]->types.size();
  window_plan plan;
  plan.input_width    = input_types.size();
  plan.prepared_types = input_types;
  plan.output_types   = input_types;
  if (window.types.size() != child_width + window.expressions.size() ||
      input_types.size() < child_width) {
    decline("output width");
  }
  for (duckdb::idx_t i = 0; i < child_width; ++i) {
    if (window.types[i] != input_types[i]) { decline("input column mapping"); }
  }
  for (auto index : indices) {
    plan.output_types.push_back(window.types[child_width + index]);
  }

  // Compare bound expressions before translation. Reuse references directly and
  // evaluate each deterministic hidden expression once for the entire stage.
  // A constant argument preserves row cardinality even when COUNT(*) is the only consumer
  // and DuckDB prunes every original input column. It is hidden from the SQL output.
  duckdb::BoundConstantExpression count_all_argument(duckdb::Value::INTEGER(1));
  std::vector<duckdb::Expression const*> hidden;
  auto prepare = [&](duckdb::Expression const& expression) -> duckdb::idx_t {
    if (expression.IsVolatile()) { decline("volatile key or argument"); }
    if (expression.GetExpressionType() == duckdb::ExpressionType::BOUND_REF) {
      auto const& ref = expression.Cast<duckdb::BoundReferenceExpression>();
      if (ref.Index() >= plan.input_width ||
          expression.GetReturnType() != plan.prepared_types[ref.Index()]) {
        decline("input reference");
      }
      return ref.Index();
    }
    for (std::size_t i = 0; i < hidden.size(); ++i) {
      if (expression.Equals(*hidden[i])) { return plan.input_width + i; }
    }
    auto translated = ast::from_duckdb(expression);
    if (!translated) { decline("key or argument expression cannot be translated"); }
    auto const slot = plan.prepared_types.size();
    hidden.push_back(&expression);
    plan.hidden_expressions.push_back(std::move(translated));
    plan.prepared_types.push_back(expression.GetReturnType());
    return slot;
  };

  duckdb::BoundWindowExpression const* organization         = nullptr;
  duckdb::BoundWindowExpression const* ranking_organization = nullptr;
  for (duckdb::idx_t i = 0; i < indices.size(); ++i) {
    auto const& bound = *window.expressions[indices[i]];
    if (bound.GetExpressionClass() != duckdb::ExpressionClass::BOUND_WINDOW) {
      decline("expected a bound window expression");
    }
    auto const& expression = bound.Cast<duckdb::BoundWindowExpression>();
    auto const ranking     = validate_ranking(expression);
    if (ranking && !plan.expressions.empty()) { decline("ranking mixed with aggregates"); }
    if (!ranking && !plan.rankings.empty()) { decline("ranking mixed with aggregates"); }
    auto const function = ranking ? aggregate_id::count_star : validate(expression);
    if (organization && (!organization->PartitionsAreEquivalent(expression) ||
                         !expression.PartitionsAreEquivalent(*organization))) {
      decline("incompatible partition specifications");
    }
    for (auto const& key : expression.Partitions()) {
      if (!is_supported_partition_key(key->GetReturnType())) {
        decline("partition key type or collation");
      }
      if (key->IsVolatile()) { decline("volatile key or argument"); }
    }
    if (!organization) {
      organization = &expression;
      for (auto const& key : expression.Partitions()) {
        plan.partition_slots.push_back(prepare(*key));
      }
    }
    auto const output_slot = plan.input_width + i;
    if (window.types[child_width + indices[i]] != expression.GetReturnType()) {
      decline("result column mapping");
    }
    if (ranking) {
      if (plan.rankings.empty()) {
        plan_ranking_order(plan, expression, prepare);
        ranking_organization = &expression;
      } else if (!same_order(*ranking_organization, expression)) {
        decline("incompatible ORDER BY specifications");
      }
      plan.rankings.push_back({*ranking, expression.GetReturnType(), output_slot});
      continue;
    }
    auto const& argument     = function == aggregate_id::count_star
                                 ? static_cast<duckdb::Expression const&>(count_all_argument)
                                 : *expression.GetChildren()[0];
    auto const argument_slot = prepare(argument);
    plan.expressions.push_back({function,
                                argument_slot,
                                plan.prepared_types[argument_slot],
                                expression.GetReturnType(),
                                expression.WindowStart(),
                                expression.WindowEnd(),
                                output_slot});
  }
  return plan;
}

/**
 * @brief Split the expressions of a window into stages.
 *
 * Two expressions share a stage when all of these hold:
 * - They are both ranking functions or both aggregates.
 * - Their `PARTITION BY` keys are equivalent.
 * - For ranking functions, their `ORDER BY` is the same.
 *
 * Stages are listed in the order of their first expression, and the indices of a stage are in
 * increasing order.
 *
 * @param window The window to split. Each of its expressions must be a bound window expression.
 * @return One list of indices into `window.expressions` for each stage. Every expression is in
 * exactly one stage.
 * @throws duckdb::NotImplementedException if an expression is not a bound window expression, or is
 * a ranking function in a form that `validate_ranking` refuses.
 */
std::vector<std::vector<duckdb::idx_t>> split_window_into_stages(
  duckdb::LogicalWindow const& window)
{
  // One stage while it is built: its first expression, which later expressions are compared with,
  // whether it holds ranking functions, and the indices of its expressions.
  struct stage {
    duckdb::BoundWindowExpression const* first;
    bool ranking;
    std::vector<duckdb::idx_t> indices;
  };

  std::vector<stage> stages;
  for (duckdb::idx_t i = 0; i < window.expressions.size(); ++i) {
    auto const& bound = *window.expressions[i];
    if (bound.GetExpressionClass() != duckdb::ExpressionClass::BOUND_WINDOW) {
      decline("expected a bound window expression");
    }
    auto const& expression = bound.Cast<duckdb::BoundWindowExpression>();
    bool const ranking     = validate_ranking(expression).has_value();
    auto const match       = std::find_if(stages.begin(), stages.end(), [&](stage const& s) {
      return s.ranking == ranking && s.first->PartitionsAreEquivalent(expression) &&
             expression.PartitionsAreEquivalent(*s.first) &&
             (!ranking || same_order(*s.first, expression));
    });
    if (match == stages.end()) {
      stages.push_back({&expression, ranking, {i}});
    } else {
      match->indices.push_back(i);
    }
  }
  std::vector<std::vector<duckdb::idx_t>> result;
  result.reserve(stages.size());
  for (auto& s : stages) {
    result.push_back(std::move(s.indices));
  }
  return result;
}

}  // namespace

window_plan plan_window(duckdb::LogicalWindow const& window)
{
  std::vector<duckdb::idx_t> indices(window.expressions.size());
  std::iota(indices.begin(), indices.end(), duckdb::idx_t{0});
  return plan_window_stage(window, indices, window.children[0]->types);
}

namespace {

/**
 * @brief Add a projection on @p child that appends the plan's hidden expressions as columns.
 *
 * The child's columns pass through unchanged. Each `PARTITION BY` key or aggregate argument that is
 * not a plain column reference is computed once and added after them, so that PARTITION and WINDOW
 * address every key and argument as a column slot of the prepared schema.
 *
 * @throws duckdb::InternalException if the child's schema differs from the plan's input schema.
 */
duckdb::unique_ptr<op::sirius_physical_operator> prepare_window_input(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  window_plan const& plan,
  std::size_t estimated_cardinality)
{
  if (!child || child->types.size() != plan.input_width ||
      plan.prepared_types.size() != plan.input_width + plan.hidden_expressions.size()) {
    throw duckdb::InternalException("Window preparation schema does not match its child");
  }
  auto prepared = sirius::from_duckdb_vec(plan.prepared_types);
  for (duckdb::idx_t i = 0; i < plan.input_width; ++i) {
    if (child->types[i] != prepared[i]) {
      throw duckdb::InternalException("Window child type mismatch");
    }
  }
  duckdb::vector<std::unique_ptr<ast::node>> hidden;
  for (auto const& expression : plan.hidden_expressions) {
    hidden.push_back(ast::clone(*expression));
  }
  duckdb::vector<sirius::logical_type> hidden_types(prepared.begin() + plan.input_width,
                                                    prepared.end());
  return push_appending_projection(
    std::move(child), std::move(hidden), std::move(hidden_types), estimated_cardinality);
}

}  // namespace

duckdb::unique_ptr<op::sirius_physical_operator> build_window_input(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  window_plan const& plan,
  std::size_t estimated_cardinality,
  uint64_t hash_partition_bytes)
{
  child = prepare_window_input(std::move(child), plan, estimated_cardinality);
  return build_window_exchange(
    std::move(child), plan.partition_slots, estimated_cardinality, hash_partition_bytes);
}

duckdb::unique_ptr<op::sirius_physical_operator> build_window_exchange(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  std::vector<duckdb::idx_t> const& partition_slots,
  std::size_t estimated_cardinality,
  uint64_t hash_partition_bytes)
{
  if (!child) { throw std::invalid_argument("Window exchange requires a prepared child"); }
  auto const prepared = child->types;
  if (hash_partition_bytes == 0) {
    throw std::invalid_argument("Partition byte target must be positive");
  }
  std::vector<cudf::size_type> keys;
  keys.reserve(partition_slots.size());
  for (auto slot : partition_slots) {
    if (slot >= prepared.size() ||
        slot > static_cast<duckdb::idx_t>(std::numeric_limits<cudf::size_type>::max())) {
      throw duckdb::NotImplementedException("Window partition slot exceeds the native index range");
    }
    keys.push_back(static_cast<cudf::size_type>(slot));
  }
  auto assembly = duckdb::make_uniq<op::sirius_physical_concat>(
    prepared, estimated_cardinality, /*concat_all=*/true);
  op::partition_spec spec;
  spec.mode = keys.empty() ? op::partition_mode::single : op::partition_mode::hash;
  for (auto key : keys) {
    spec.key_cast_types.push_back(sirius::require_cudf_type(prepared[key]));
  }
  spec.key_columns = std::move(keys);
  auto exchange    = duckdb::make_uniq<op::sirius_physical_partition>(
    prepared, estimated_cardinality, std::move(spec), op::MemoryBarrierType::FULL);
  if (partition_slots.empty()) {
    exchange->set_fixed_layout({op::partition_mode::single, op::partition_placement::unpinned(1)});
  } else {
    op::partition_exchange::bind(
      {exchange.get()},
      0,
      false,
      [target = hash_partition_bytes, receiver = assembly.get()](auto const& stats, bool) {
        auto count = op::natural_num_partitions(stats[0].bytes, target, receiver->num_gpus());
        return std::vector<op::partition_layout>{
          {op::partition_mode::hash, op::partition_placement::unpinned(count)}};
      });
  }
  exchange->children.push_back(std::move(child));
  assembly->children.push_back(std::move(exchange));
  return assembly;
}

namespace {

// Builds the WINDOW operator of one stage on top of its prepared input.
duckdb::unique_ptr<op::sirius_physical_operator> build_window_stage(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  window_plan const& bound_plan,
  std::size_t estimated_cardinality)
{
  child         = prepare_window_input(std::move(child), bound_plan, estimated_cardinality);
  auto prepared = sirius::from_duckdb_vec(bound_plan.prepared_types);
  std::vector<cudf::size_type> keys(bound_plan.partition_slots.begin(),
                                    bound_plan.partition_slots.end());
  if (!bound_plan.rankings.empty()) {
    std::vector<op::ranking_function> functions;
    for (auto const& ranking : bound_plan.rankings) {
      functions.push_back(ranking.function);
    }
    auto ranked = duckdb::make_uniq<op::sirius_physical_window>(
      sirius::from_duckdb_vec(bound_plan.output_types),
      std::move(prepared),
      bound_plan.input_width,
      std::move(keys),
      bound_plan.ranking_order,
      std::move(functions),
      estimated_cardinality);
    ranked->children.push_back(std::move(child));
    return ranked;
  }
  std::vector<op::whole_partition_aggregate> aggregates;
  for (auto const& expression : bound_plan.expressions) {
    aggregates.push_back(
      {expression.function, static_cast<cudf::size_type>(expression.argument_slot)});
  }
  auto result =
    duckdb::make_uniq<op::sirius_physical_window>(sirius::from_duckdb_vec(bound_plan.output_types),
                                                  std::move(prepared),
                                                  bound_plan.input_width,
                                                  std::move(keys),
                                                  std::move(aggregates),
                                                  estimated_cardinality);
  result->children.push_back(std::move(child));
  return result;
}

}  // namespace

duckdb::unique_ptr<op::sirius_physical_operator> sirius_physical_plan_generator::create_plan(
  duckdb::LogicalWindow& window)
{
  if (window.children.size() != 1 || window.expressions.empty()) {
    decline("expected one child and at least one expression");
  }
  // Plan every stage before the child. Planning the child can change the types of its logical
  // operator.
  auto const stages      = split_window_into_stages(window);
  auto const child_width = window.children[0]->types.size();
  auto types             = window.children[0]->types;
  std::vector<window_plan> stage_plans;
  for (auto const& indices : stages) {
    stage_plans.push_back(plan_window_stage(window, indices, types));
    types = stage_plans.back().output_types;
  }
  auto result = create_plan(*window.children[0]);
  for (auto const& stage_plan : stage_plans) {
    result = build_window_stage(std::move(result), stage_plan, window.estimated_cardinality);
  }
  // The stages append their results in stage order. Restore the order of the window's expressions.
  std::vector<duckdb::idx_t> source(window.expressions.size());
  duckdb::idx_t next = 0;
  for (auto const& indices : stages) {
    for (auto index : indices) {
      source[index] = child_width + next++;
    }
  }
  bool in_order = true;
  for (duckdb::idx_t i = 0; i < source.size(); ++i) {
    in_order &= source[i] == child_width + i;
  }
  if (in_order) { return result; }
  auto const output_types = sirius::from_duckdb_vec(window.types);
  duckdb::vector<std::unique_ptr<ast::node>> select;
  for (duckdb::idx_t i = 0; i < output_types.size(); ++i) {
    auto const slot = i < child_width ? i : source[i - child_width];
    select.push_back(
      std::make_unique<ast::node>(ast::reference{static_cast<uint32_t>(slot), output_types[i]}));
  }
  return push_projection(
    std::move(result), output_types, std::move(select), window.estimated_cardinality);
}

}  // namespace sirius::planner
