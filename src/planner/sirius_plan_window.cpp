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
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_window.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <variant>

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

/// Whether a running MIN or MAX accepts an argument of @p type.
bool is_cumulative_extremum_argument(duckdb::LogicalType const& type)
{
  using duckdb::LogicalTypeId;
  switch (type.id()) {
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::FLOAT:
    case LogicalTypeId::DOUBLE:
    case LogicalTypeId::DATE:
    case LogicalTypeId::TIMESTAMP:
    case LogicalTypeId::TIMESTAMP_MS:
    case LogicalTypeId::TIMESTAMP_NS:
    case LogicalTypeId::TIMESTAMP_SEC: return true;
    case LogicalTypeId::DECIMAL:
      return duckdb::DecimalType::GetWidth(type) >
             sirius::logical_type::decimal_max_precision_int16;
    default: return false;
  }
}

/// Decline the window modifiers that no window function here supports. Frame offsets are declined
/// too unless @p allow_frame_offsets is set, for an aggregate whose frame is translated.
void decline_unsupported_modifiers(duckdb::BoundWindowExpression const& expression,
                                   bool allow_frame_offsets = false)
{
  if (expression.Distinct()) { decline("DISTINCT"); }
  if (expression.Filter()) { decline("FILTER"); }
  if (expression.WindowExclude() != duckdb::WindowExcludeMode::NO_OTHER) { decline("EXCLUDE"); }
  if (!expression.ArgOrders().empty()) { decline("argument ORDER BY"); }
  if (expression.IgnoreNulls()) { decline("IGNORE NULLS"); }
  if (!allow_frame_offsets && (expression.StartExpr() || expression.EndExpr())) {
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
std::optional<op::ranking_function> build_ranking(duckdb::BoundWindowExpression const& expression)
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

/// Record the order key slots, directions and NULL placements of the stage's first window.
template <typename Prepare>
void plan_window_order(window_plan& plan,
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
    plan.order.columns.push_back(static_cast<cudf::size_type>(prepare(*node.expression)));
    plan.order.directions.push_back(op::to_cudf_order(node.type));
    plan.order.nulls.push_back(op::to_cudf_null_order(node.type, node.null_order));
  }
}

/**
 * @brief Return the aggregate function of @p expression, or decline it.
 *
 * Valid is a `SUM`, `AVG`, `COUNT`, `COUNT(*)`, `MIN` or `MAX` without `DISTINCT`, `FILTER`,
 * `EXCLUDE`, `IGNORE NULLS` or an argument `ORDER BY`. `SUM` and `AVG` take a numeric argument.
 * `COUNT`, `MIN` and `MAX` also take the types allowed as partition keys, and `MIN` and `MAX` keep
 * the argument type. The frame is not checked here.
 */
aggregate_id build_aggregate_function(duckdb::BoundWindowExpression const& expression)
{
  if (expression.GetExpressionType() != duckdb::ExpressionType::WINDOW_AGGREGATE ||
      !expression.AggregateFunction()) {
    decline("only SUM, AVG, COUNT, MIN and MAX aggregates are planned");
  }
  decline_unsupported_modifiers(expression, /*allow_frame_offsets=*/true);
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

/// Whether the frame of @p expression covers its whole partition whatever the row order. Without
/// ORDER BY all rows are peers, so DuckDB's default frame, which ends at CURRENT_ROW_RANGE, covers
/// it too. CURRENT_ROW_ROWS still ends at the physical current row.
bool covers_whole_partition(duckdb::BoundWindowExpression const& expression)
{
  using duckdb::WindowBoundary;
  if (expression.WindowStart() != WindowBoundary::UNBOUNDED_PRECEDING) { return false; }
  return expression.WindowEnd() == WindowBoundary::UNBOUNDED_FOLLOWING ||
         (expression.OrderBy().empty() &&
          expression.WindowEnd() == WindowBoundary::CURRENT_ROW_RANGE);
}

/**
 * @brief Translate the frame of @p expression, or decline it.
 *
 * `ROWS` offsets must be non-negative `BIGINT` constants once DuckDB's optimizer has folded them.
 * `RANGE` offsets and `GROUPS` frames are declined.
 */
op::window_frame translate_frame(duckdb::BoundWindowExpression const& expression)
{
  using duckdb::WindowBoundary;
  std::optional<op::frame_unit> unit;
  auto const set_unit = [&](op::frame_unit next) {
    // DuckDB does not bind a frame that mixes units. Decline one defensively.
    if (unit && *unit != next) { decline("frame mixes ROWS and RANGE bounds"); }
    unit = next;
  };
  auto const bound = [&](WindowBoundary boundary,
                         duckdb::Expression const* offset,
                         bool is_start) -> op::frame_bound {
    switch (boundary) {
      case WindowBoundary::UNBOUNDED_PRECEDING:
        if (!is_start) { decline("frame ends at UNBOUNDED PRECEDING"); }
        return op::frame_unbounded{};
      case WindowBoundary::UNBOUNDED_FOLLOWING:
        if (is_start) { decline("frame starts at UNBOUNDED FOLLOWING"); }
        return op::frame_unbounded{};
      case WindowBoundary::CURRENT_ROW_ROWS:
        set_unit(op::frame_unit::rows);
        return op::frame_current_row{};
      case WindowBoundary::CURRENT_ROW_RANGE:
        set_unit(op::frame_unit::range);
        return op::frame_current_row{};
      case WindowBoundary::EXPR_PRECEDING_ROWS:
      case WindowBoundary::EXPR_FOLLOWING_ROWS: {
        set_unit(op::frame_unit::rows);
        if (!offset) { decline("frame offset is missing"); }
        auto const rows = ast::bigint_constant(*offset);
        if (!rows) { decline("non-constant frame offset"); }
        if (*rows < 0) { decline("negative frame offset"); }
        return op::frame_row_offset{*rows, boundary == WindowBoundary::EXPR_PRECEDING_ROWS};
      }
      case WindowBoundary::EXPR_PRECEDING_RANGE:
      case WindowBoundary::EXPR_FOLLOWING_RANGE: decline("RANGE frame offsets");
      case WindowBoundary::CURRENT_ROW_GROUPS:
      case WindowBoundary::EXPR_PRECEDING_GROUPS:
      case WindowBoundary::EXPR_FOLLOWING_GROUPS: decline("GROUPS frames");
      default: decline("frame bound");
    }
  };
  auto start = bound(expression.WindowStart(), expression.StartExpr().get(), true);
  auto end   = bound(expression.WindowEnd(), expression.EndExpr().get(), false);
  // Both bounds unbounded covers the partition and is routed before translation.
  return {unit.value_or(op::frame_unit::rows), std::move(start), std::move(end)};
}

// An aggregate over a frame that is not the whole partition.
struct framed_function {
  aggregate_id function;
  op::window_frame frame;
};

/**
 * @brief Whether the running evaluator computes @p function for @p expression.
 *
 * It takes a window `ORDER BY` with the frame `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`,
 * `SUM`, `COUNT` and `COUNT(*)`, and `MIN` and `MAX` over an integer, decimal, floating point, date
 * or timestamp argument.
 */
bool runs_cumulatively(aggregate_id function, duckdb::BoundWindowExpression const& expression)
{
  using duckdb::WindowBoundary;
  if (expression.OrderBy().empty() ||
      expression.WindowStart() != WindowBoundary::UNBOUNDED_PRECEDING ||
      expression.WindowEnd() != WindowBoundary::CURRENT_ROW_ROWS) {
    return false;
  }
  switch (function) {
    case aggregate_id::sum:
    case aggregate_id::count:
    case aggregate_id::count_star: return true;
    case aggregate_id::min:
    case aggregate_id::max:
      return is_cumulative_extremum_argument(expression.GetChildren()[0]->GetReturnType());
    default: return false;
  }
}

// An aggregate and the kind of window that evaluates it.
struct classified_aggregate {
  std::variant<aggregate_id, framed_function> function;
  window_kind kind;
};

/**
 * @brief Return the aggregate, running aggregate or framed aggregate to run for @p expression, or
 * decline it.
 *
 * Aggregates whose frame covers the whole partition, with or without a window `ORDER BY`, are
 * evaluated without sorting: the order cannot change the result of these functions. Aggregates
 * that `runs_cumulatively` accepts are running aggregates, sorted within each hash bucket. Any
 * other frame needs a window `ORDER BY` and a `PARTITION BY`, and is translated by
 * `translate_frame`.
 */
classified_aggregate build_aggregate(duckdb::BoundWindowExpression const& expression)
{
  auto const function = build_aggregate_function(expression);
  if (covers_whole_partition(expression)) { return {function, window_kind::whole_partition}; }
  if (expression.OrderBy().empty()) { decline("frame does not cover the whole partition"); }
  if (runs_cumulatively(function, expression)) { return {function, window_kind::cumulative}; }
  if (expression.Partitions().empty()) { decline("framed window without PARTITION BY"); }
  return {framed_function{function, translate_frame(expression)}, window_kind::framed};
}

// One window expression, validated, with the function it computes and its kind.
struct classified_expression {
  duckdb::idx_t index;
  duckdb::BoundWindowExpression const* expression;
  std::variant<aggregate_id, op::ranking_function, framed_function> function;
  window_kind kind;
};

/**
 * @brief Validate each expression of @p window and record the function it computes.
 *
 * @return One entry for each expression of @p window, in order.
 * @throws duckdb::NotImplementedException if an expression is not a bound window expression or is
 * not a ranking function or aggregate that `build_ranking` or `build_aggregate` accepts.
 */
std::vector<classified_expression> classify_window_expressions(duckdb::LogicalWindow const& window)
{
  std::vector<classified_expression> result;
  result.reserve(window.expressions.size());
  for (duckdb::idx_t i = 0; i < window.expressions.size(); ++i) {
    auto const& bound = *window.expressions[i];
    if (bound.GetExpressionClass() != duckdb::ExpressionClass::BOUND_WINDOW) {
      decline("expected a bound window expression");
    }
    auto const& expression = bound.Cast<duckdb::BoundWindowExpression>();
    if (auto const ranking = build_ranking(expression)) {
      result.push_back({i, &expression, *ranking, window_kind::ranking});
      continue;
    }
    auto aggregate = build_aggregate(expression);
    std::visit(
      [&](auto const& function) { result.push_back({i, &expression, function, aggregate.kind}); },
      aggregate.function);
  }
  return result;
}

/**
 * @brief Split classified window expressions into stages.
 *
 * Two expressions share a stage when all of these hold:
 * - They are of the same kind: ranking functions, aggregates over the whole partition, running
 *   aggregates, or aggregates over other frames.
 * - Their `PARTITION BY` keys are equivalent.
 * - Except for aggregates over the whole partition, their `ORDER BY` is the same. Framed
 *   aggregates may differ in their frames.
 *
 * Stages are listed in the order of their first expression, and the expressions of a stage keep
 * their order.
 *
 * @param expressions The expressions of one window, from `classify_window_expressions`.
 * @return The expressions of each stage. Every expression is in exactly one stage.
 */
std::vector<std::vector<classified_expression>> split_window_into_stages(
  std::span<classified_expression const> expressions)
{
  std::vector<std::vector<classified_expression>> stages;
  for (auto const& entry : expressions) {
    auto const& expression = *entry.expression;
    auto const kind        = entry.kind;
    auto const match       = std::ranges::find_if(stages, [&](auto const& stage) {
      auto const& first = *stage.front().expression;
      return stage.front().kind == kind && first.PartitionsAreEquivalent(expression) &&
             (kind == window_kind::whole_partition || same_order(first, expression));
    });
    if (match == stages.end()) {
      stages.push_back({entry});
    } else {
      match->push_back(entry);
    }
  }
  return stages;
}

// Plans how the whole-partition evaluator computes the aggregates of @p plan, refusing the window
// when the evaluator cannot run them. The evaluator reads the partition keys and the arguments in
// their logical types, so only their slots need a cuDF type.
op::whole_partition_plan plan_aggregate_evaluation(window_plan const& plan)
{
  std::vector<cudf::data_type> types(plan.prepared_types.size(),
                                     cudf::data_type{cudf::type_id::EMPTY});
  auto const read = [&](duckdb::idx_t slot) {
    auto const type = sirius::get_cudf_type(sirius::from_duckdb(plan.prepared_types[slot]));
    if (!type) { decline("key or argument type"); }
    types[slot] = *type;
    return static_cast<cudf::size_type>(slot);
  };
  std::vector<cudf::size_type> keys;
  for (auto const slot : plan.partition_slots) {
    keys.push_back(read(slot));
  }
  std::vector<op::window_aggregate> aggregates;
  for (auto const& expression : plan.expressions) {
    aggregates.push_back({expression.function, read(expression.argument_slot)});
  }
  try {
    return op::plan_whole_partition_window(types, keys, aggregates);
  } catch (std::invalid_argument const&) {
    decline("aggregate the evaluator cannot run");
  }
}

// Plans how the framed evaluator computes the framed aggregates of @p plan, refusing the window
// when the evaluator cannot run them. Partition and order keys may reach the evaluator in narrower
// carriers; only the argument types are checked when it runs.
op::framed_window_plan plan_framed_evaluation(window_plan const& plan)
{
  std::vector<cudf::data_type> types(plan.prepared_types.size(),
                                     cudf::data_type{cudf::type_id::EMPTY});
  auto const read = [&](duckdb::idx_t slot) {
    auto const type = sirius::get_cudf_type(sirius::from_duckdb(plan.prepared_types[slot]));
    if (!type) { decline("key or argument type"); }
    types[slot] = *type;
    return static_cast<cudf::size_type>(slot);
  };
  std::vector<cudf::size_type> keys;
  for (auto const slot : plan.partition_slots) {
    keys.push_back(read(slot));
  }
  for (auto const slot : plan.order.columns) {
    read(static_cast<duckdb::idx_t>(slot));
  }
  std::vector<op::framed_aggregate> aggregates;
  for (auto const& expression : plan.framed) {
    aggregates.push_back({expression.function, read(expression.argument_slot), expression.frame});
  }
  try {
    return op::plan_framed_window(types, keys, plan.order, aggregates);
  } catch (std::invalid_argument const&) {
    decline("framed aggregate the evaluator cannot run");
  }
}

// Plans the expressions of one stage of `window` over an input of `input_types`. The expressions
// come from `split_window_into_stages`, so they share their kind, partitions and ranking order. The
// input starts with the window child's columns, and a later stage of the same window also holds the
// results of the stages below it. The output is the input followed by the results of `stage`.
window_plan plan_window_stage(duckdb::LogicalWindow const& window,
                              std::span<classified_expression const> stage,
                              duckdb::vector<duckdb::LogicalType> const& input_types)
{
  if (window.children.size() != 1 || stage.empty()) {
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
  for (auto const& entry : stage) {
    plan.output_types.push_back(window.types[child_width + entry.index]);
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

  // The expressions of a stage share their partitions and, except for aggregates over the whole
  // partition, their ORDER BY, so the first expression defines both.
  auto const& first = *stage.front().expression;
  for (auto const& key : first.Partitions()) {
    if (!is_supported_partition_key(key->GetReturnType())) {
      decline("partition key type or collation");
    }
    plan.partition_slots.push_back(prepare(*key));
  }
  plan.kind = stage.front().kind;
  if (plan.kind != window_kind::whole_partition) { plan_window_order(plan, first, prepare); }

  for (duckdb::idx_t i = 0; i < stage.size(); ++i) {
    auto const& entry      = stage[i];
    auto const& expression = *entry.expression;
    auto const output_slot = plan.input_width + i;
    if (window.types[child_width + entry.index] != expression.GetReturnType()) {
      decline("result column mapping");
    }
    if (auto const* ranking = std::get_if<op::ranking_function>(&entry.function)) {
      plan.rankings.push_back({*ranking, expression.GetReturnType(), output_slot});
      continue;
    }
    auto const* framed       = std::get_if<framed_function>(&entry.function);
    auto const aggregate     = framed ? framed->function : std::get<aggregate_id>(entry.function);
    auto const& argument     = aggregate == aggregate_id::count_star
                                 ? static_cast<duckdb::Expression const&>(count_all_argument)
                                 : *expression.GetChildren()[0];
    auto const argument_slot = prepare(argument);
    if (framed) {
      plan.framed.push_back({aggregate,
                             argument_slot,
                             plan.prepared_types[argument_slot],
                             expression.GetReturnType(),
                             framed->frame,
                             output_slot});
      continue;
    }
    plan.expressions.push_back({aggregate,
                                argument_slot,
                                plan.prepared_types[argument_slot],
                                expression.GetReturnType(),
                                expression.WindowStart(),
                                expression.WindowEnd(),
                                output_slot});
  }
  if (plan.kind == window_kind::whole_partition) {
    plan.aggregate_plan = plan_aggregate_evaluation(plan);
  }
  if (!plan.framed.empty()) { plan.framed_plan = plan_framed_evaluation(plan); }
  return plan;
}

}  // namespace

window_plan plan_window(duckdb::LogicalWindow const& window)
{
  auto const stages = split_window_into_stages(classify_window_expressions(window));
  if (stages.size() != 1) { decline("expressions that need more than one window stage"); }
  return plan_window_stage(window, stages.front(), window.children[0]->types);
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

duckdb::unique_ptr<op::sirius_physical_operator> build_window_sort_exchange(
  duckdb::unique_ptr<op::sirius_physical_operator> child,
  std::vector<duckdb::idx_t> const& partition_slots,
  op::window_order const& order,
  std::size_t estimated_cardinality,
  sort_exchange_options options)
{
  if (!child) { throw std::invalid_argument("Window exchange requires a prepared child"); }
  if (partition_slots.empty()) {
    throw std::invalid_argument("Sorted window exchange requires a PARTITION BY");
  }
  if (order.directions.size() != order.columns.size() ||
      order.nulls.size() != order.columns.size()) {
    throw std::invalid_argument("Window order has mismatched lengths");
  }
  auto const types = child->types;
  // The sort operators only read the slot of a reference. Its type is the carrier's.
  auto const reference = [&](duckdb::idx_t slot) {
    if (slot >= types.size()) {
      throw std::invalid_argument("Window sort key is outside its prepared schema");
    }
    return duckdb::make_uniq<duckdb::BoundReferenceExpression>(sirius::to_duckdb(types[slot]),
                                                               slot);
  };
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  // Any fixed order of the partition keys makes equal keys adjacent.
  for (auto const slot : partition_slots) {
    orders.emplace_back(
      duckdb::OrderType::ASCENDING, duckdb::OrderByNullType::NULLS_FIRST, reference(slot));
  }
  for (std::size_t i = 0; i < order.columns.size(); ++i) {
    if (order.columns[i] < 0) {
      throw std::invalid_argument("Window sort key is outside its prepared schema");
    }
    auto const sql_order = op::to_duckdb_order(order.directions[i], order.nulls[i]);
    orders.emplace_back(sql_order.type,
                        sql_order.null_order,
                        reference(static_cast<duckdb::idx_t>(order.columns[i])));
  }
  duckdb::vector<std::size_t> identity(types.size());
  std::iota(identity.begin(), identity.end(), 0);
  auto local_sort = duckdb::make_uniq<op::sirius_physical_order>(
    types, std::move(orders), std::move(identity), estimated_cardinality);
  local_sort->children.push_back(std::move(child));
  options.boundary_key_count = partition_slots.size();
  return build_sort_exchange(std::move(local_sort), options);
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
  if (bound_plan.kind == window_kind::framed) {
    if (!bound_plan.framed_plan) {
      throw duckdb::InternalException(
        "Window stage of framed aggregates lacks its evaluation plan");
    }
    auto framed = duckdb::make_uniq<op::sirius_physical_window>(
      sirius::from_duckdb_vec(bound_plan.output_types),
      std::move(prepared),
      bound_plan.input_width,
      *bound_plan.framed_plan,
      estimated_cardinality);
    framed->children.push_back(std::move(child));
    return framed;
  }
  if (bound_plan.kind == window_kind::ranking) {
    std::vector<op::ranking_function> functions;
    for (auto const& ranking : bound_plan.rankings) {
      functions.push_back(ranking.function);
    }
    auto ranked = duckdb::make_uniq<op::sirius_physical_window>(
      sirius::from_duckdb_vec(bound_plan.output_types),
      std::move(prepared),
      bound_plan.input_width,
      std::move(keys),
      bound_plan.order,
      std::move(functions),
      estimated_cardinality);
    ranked->children.push_back(std::move(child));
    return ranked;
  }
  if (bound_plan.kind == window_kind::cumulative) {
    std::vector<op::window_aggregate> running;
    for (auto const& expression : bound_plan.expressions) {
      running.push_back(
        {expression.function, static_cast<cudf::size_type>(expression.argument_slot)});
    }
    auto result = duckdb::make_uniq<op::sirius_physical_window>(
      sirius::from_duckdb_vec(bound_plan.output_types),
      std::move(prepared),
      bound_plan.input_width,
      std::move(keys),
      bound_plan.order,
      std::move(running),
      estimated_cardinality);
    result->children.push_back(std::move(child));
    return result;
  }
  if (!bound_plan.aggregate_plan) {
    throw duckdb::InternalException("Window stage of aggregates lacks its evaluation plan");
  }
  auto result =
    duckdb::make_uniq<op::sirius_physical_window>(sirius::from_duckdb_vec(bound_plan.output_types),
                                                  std::move(prepared),
                                                  bound_plan.input_width,
                                                  *bound_plan.aggregate_plan,
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
  auto const stages      = split_window_into_stages(classify_window_expressions(window));
  auto const child_width = window.children[0]->types.size();
  auto types             = window.children[0]->types;
  std::vector<window_plan> stage_plans;
  for (auto const& stage : stages) {
    stage_plans.push_back(plan_window_stage(window, stage, types));
    types = stage_plans.back().output_types;
  }
  auto result = create_plan(*window.children[0]);
  for (auto const& stage_plan : stage_plans) {
    result = build_window_stage(std::move(result), stage_plan, window.estimated_cardinality);
  }
  // The stages append their results in stage order. Restore the order of the window's expressions.
  std::vector<duckdb::idx_t> source(window.expressions.size());
  duckdb::idx_t next = 0;
  for (auto const& stage : stages) {
    for (auto const& entry : stage) {
      source[entry.index] = child_width + next++;
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
