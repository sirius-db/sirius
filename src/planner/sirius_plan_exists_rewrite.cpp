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

#include "cudf/cudf_utils.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace sirius::planner {

namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::ExpressionClass;
using duckdb::ExpressionType;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::LogicalType;
using duckdb::ProjectionIndex;
using duckdb::unique_ptr;

unique_ptr<Expression> column(LogicalType const& type, ColumnBinding binding)
{
  return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(type, binding);
}

bool is_column(Expression const& expression)
{
  return expression.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF;
}

ColumnBinding binding_of(Expression const& expression)
{
  return expression.Cast<duckdb::BoundColumnRefExpression>().Binding();
}

bool contains_delim_get(LogicalOperator const& op)
{
  if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET) { return true; }
  return std::ranges::any_of(op.children,
                             [](auto const& child) { return contains_delim_get(*child); });
}

/// The pieces of a correlated `EXISTS (... WHERE t.k = outer.k AND t.x <> outer.x)` as DuckDB
/// decorrelated it. Either comparison may also be its null-safe form (`IS [NOT] DISTINCT FROM`).
struct exists_match {
  LogicalOperator* replaced           = nullptr;  ///< The node being replaced
  duckdb::LogicalComparisonJoin* join = nullptr;  ///< The join of the outer rows and T
  std::size_t outer_child             = 0;        ///< Index of the outer rows among its children
  bool negated                        = false;    ///< NOT EXISTS
  unique_ptr<LogicalOperator>* t      = nullptr;  ///< Owning slot of T, the subquery rows
  ColumnBinding t_key;                            ///< Column of T equal to the key
  ColumnBinding t_value;                          ///< Column of T compared with `<>`
  ColumnBinding outer_key;
  ColumnBinding outer_value;
  LogicalType key_type;
  LogicalType value_type;
  bool key_null_safe   = false;       ///< `IS NOT DISTINCT FROM` rather than `=`
  bool value_null_safe = false;       ///< `IS DISTINCT FROM` rather than `<>`
  bool plain           = false;       ///< A plain SEMI or ANTI join rather than a delim join
  std::optional<ColumnBinding> mark;  ///< The mark a replaced FILTER(mark) consumed
};

enum class correlation { key, value, other };

/// Which side of the match a `t <comparison> outer` condition is, recording its null-safety.
correlation classify(ExpressionType comparison, exists_match& match)
{
  switch (comparison) {
    case ExpressionType::COMPARE_EQUAL: return correlation::key;
    case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
      match.key_null_safe = true;
      return correlation::key;
    case ExpressionType::COMPARE_NOTEQUAL: return correlation::value;
    case ExpressionType::COMPARE_DISTINCT_FROM:
      match.value_null_safe = true;
      return correlation::value;
    default: return correlation::other;
  }
}

/// @p binding followed down through plain-column @p projections (outermost first).
std::optional<ColumnBinding> see_through(
  std::vector<duckdb::LogicalProjection const*> const& projections, ColumnBinding binding)
{
  for (auto const* projection : projections) {
    if (!(binding.table_index == projection->table_index)) { continue; }
    if (binding.column_index >= projection->expressions.size() ||
        !is_column(*projection->expressions[binding.column_index])) {
      return std::nullopt;
    }
    binding = binding_of(*projection->expressions[binding.column_index]);
  }
  return binding;
}

/// Integer keys, and a value whose MIN/MAX order agrees with `<>`: integers, fixed-point and
/// temporal types. Floating point is excluded (NaN orders differently from how it compares), as
/// are strings and booleans.
bool has_supported_types(exists_match const& match)
{
  if (match.key_type.id() != duckdb::LogicalTypeId::INTEGER &&
      match.key_type.id() != duckdb::LogicalTypeId::BIGINT) {
    return false;
  }
  auto const value_cudf = sirius::get_cudf_type(sirius::from_duckdb(match.value_type));
  if (!value_cudf || value_cudf->id() == cudf::type_id::BOOL8) { return false; }
  return cudf::is_integral(*value_cudf) || cudf::is_fixed_point(*value_cudf) ||
         cudf::is_timestamp(*value_cudf) || cudf::is_duration(*value_cudf);
}

bool has_arbitrary_condition(duckdb::LogicalComparisonJoin const& join)
{
  return join.conditions.empty() || std::ranges::any_of(join.conditions, [](auto const& condition) {
           return !condition.IsComparison();
         });
}

/// Recognise the subquery side of @p delim, whose outer rows are child @p outer_idx: T joined to
/// the DELIM_GET on one key and one value comparison (`t.k = delim.k`, `t.x <> delim.x`). Every
/// check is structural; anything unexpected declines and leaves DuckDB's plan alone.
std::optional<exists_match> match_delim_subquery(duckdb::LogicalComparisonJoin& delim,
                                                 std::size_t outer_idx)
{
  if (has_arbitrary_condition(delim)) { return std::nullopt; }

  // Any number of projections may sit above the join, as long as the columns the delim join
  // reads are plain columns below them.
  auto* sub = delim.children[1 - outer_idx].get();
  std::vector<duckdb::LogicalProjection const*> projections;  // outermost first
  while (sub->type == LogicalOperatorType::LOGICAL_PROJECTION) {
    projections.push_back(&sub->Cast<duckdb::LogicalProjection>());
    sub = sub->children[0].get();
  }
  if (sub->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || sub->children.size() != 2) {
    return std::nullopt;
  }
  auto& inner = sub->Cast<duckdb::LogicalComparisonJoin>();
  if (inner.join_type != duckdb::JoinType::INNER || has_arbitrary_condition(inner)) {
    return std::nullopt;
  }
  std::size_t d_child = 2;
  for (std::size_t c = 0; c < 2; ++c) {
    if (inner.children[c]->type == LogicalOperatorType::LOGICAL_DELIM_GET) { d_child = c; }
  }
  if (d_child == 2 || contains_delim_get(*inner.children[1 - d_child])) { return std::nullopt; }
  auto const& delim_get = inner.children[d_child]->Cast<duckdb::LogicalDelimGet>();
  auto const width      = delim_get.chunk_types.size();
  if (width != delim.duplicate_eliminated_columns.size() || width != delim.conditions.size()) {
    return std::nullopt;
  }

  // Exactly one key and one value comparison, each between plain columns.
  exists_match match;
  match.join        = &delim;
  match.outer_child = outer_idx;
  match.t           = &inner.children[1 - d_child];
  std::optional<std::size_t> key_d, value_d;  // the delim scan's columns they compare with
  for (auto const& condition : inner.conditions) {
    if (!is_column(condition.GetLHS()) || !is_column(condition.GetRHS())) { return std::nullopt; }
    auto const& t_side = d_child == 1 ? condition.GetLHS() : condition.GetRHS();
    auto const& d_side = d_child == 1 ? condition.GetRHS() : condition.GetLHS();
    auto const d       = binding_of(d_side);
    if (!(d.table_index == delim_get.table_index) ||
        d_side.GetReturnType() != t_side.GetReturnType()) {
      return std::nullopt;
    }
    auto const kind = classify(condition.GetComparisonType(), match);
    if (kind == correlation::key && !key_d) {
      key_d          = d.column_index;
      match.t_key    = binding_of(t_side);
      match.key_type = t_side.GetReturnType();
    } else if (kind == correlation::value && !value_d) {
      value_d          = d.column_index;
      match.t_value    = binding_of(t_side);
      match.value_type = t_side.GetReturnType();
    } else {
      return std::nullopt;
    }
  }
  if (!key_d || !value_d || *key_d == *value_d || !has_supported_types(match)) {
    return std::nullopt;
  }

  // The delim join must equate each outer correlated column with the subquery's echo of the
  // matching delim column, null-safely, and nothing else. The echo is either the delim scan's own
  // column or T's key, which equals the delim key under the join's equality.
  std::vector<std::optional<ColumnBinding>> outer_for(width);
  for (auto const& condition : delim.conditions) {
    auto const& outer_side = outer_idx == 0 ? condition.GetLHS() : condition.GetRHS();
    auto const& sub_side   = outer_idx == 0 ? condition.GetRHS() : condition.GetLHS();
    if (condition.GetComparisonType() != ExpressionType::COMPARE_NOT_DISTINCT_FROM ||
        !is_column(outer_side) || !is_column(sub_side)) {
      return std::nullopt;
    }
    auto const origin = see_through(projections, binding_of(sub_side));
    if (!origin) { return std::nullopt; }
    std::size_t d = 0;
    if (origin->table_index == delim_get.table_index) {
      d = origin->column_index;
    } else if (*origin == match.t_key) {
      d = *key_d;
    } else {
      return std::nullopt;
    }
    if (d >= width || outer_for[d]) { return std::nullopt; }
    outer_for[d] = binding_of(outer_side);
  }
  // DuckDB builds the distinct set from the duplicate-eliminated columns in this order.
  for (std::size_t d = 0; d < width; ++d) {
    auto const& dup = *delim.duplicate_eliminated_columns[d];
    if (!outer_for[d] || !is_column(dup) || !(binding_of(dup) == *outer_for[d])) {
      return std::nullopt;
    }
  }
  match.outer_key   = *outer_for[*key_d];
  match.outer_value = *outer_for[*value_d];
  return match;
}

/// A plain SEMI or ANTI join of the outer rows and T on exactly one key and one value
/// comparison, as DuckDB leaves EXISTS once it removes the delim join, or as a client writes it.
std::optional<exists_match> match_plain_exists(LogicalOperator& node)
{
  using duckdb::JoinType;
  if (node.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || node.children.size() != 2) {
    return std::nullopt;
  }
  auto& join      = node.Cast<duckdb::LogicalComparisonJoin>();
  auto const type = join.join_type;
  if ((type != JoinType::SEMI && type != JoinType::ANTI && type != JoinType::RIGHT_SEMI &&
       type != JoinType::RIGHT_ANTI) ||
      has_arbitrary_condition(join) || join.conditions.size() != 2) {
    return std::nullopt;
  }
  exists_match match;
  match.replaced    = &node;
  match.join        = &join;
  match.plain       = true;
  match.outer_child = type == JoinType::SEMI || type == JoinType::ANTI ? 0 : 1;
  match.negated     = type == JoinType::ANTI || type == JoinType::RIGHT_ANTI;
  match.t           = &join.children[1 - match.outer_child];
  bool has_key = false, has_value = false;
  for (auto const& condition : join.conditions) {
    auto const& outer_side = match.outer_child == 0 ? condition.GetLHS() : condition.GetRHS();
    auto const& t_side     = match.outer_child == 0 ? condition.GetRHS() : condition.GetLHS();
    if (!is_column(outer_side) || !is_column(t_side) ||
        outer_side.GetReturnType() != t_side.GetReturnType()) {
      return std::nullopt;
    }
    auto const kind = classify(condition.GetComparisonType(), match);
    if (kind == correlation::key && !has_key) {
      has_key         = true;
      match.t_key     = binding_of(t_side);
      match.outer_key = binding_of(outer_side);
      match.key_type  = t_side.GetReturnType();
    } else if (kind == correlation::value && !has_value) {
      has_value         = true;
      match.t_value     = binding_of(t_side);
      match.outer_value = binding_of(outer_side);
      match.value_type  = t_side.GetReturnType();
    } else {
      return std::nullopt;
    }
  }
  if (!has_key || !has_value || !has_supported_types(match)) { return std::nullopt; }
  return match;
}

/// A SEMI or ANTI join (plain or DELIM) emitting the outer rows that have (or lack) a match, or
/// `FILTER([NOT] mark)` over a MARK DELIM_JOIN. The delim join's correlation conditions are
/// null-safe, so the mark is never NULL and the filter keeps exactly the rows a SEMI (or ANTI)
/// join would. A mark that anything but that filter reads declines.
std::optional<exists_match> match_exists(LogicalOperator& node,
                                         duckdb::column_binding_map_t<std::size_t> const& reads)
{
  using duckdb::JoinType;
  if (auto plain = match_plain_exists(node)) { return plain; }
  if (node.type == LogicalOperatorType::LOGICAL_DELIM_JOIN && node.children.size() == 2) {
    auto& delim      = node.Cast<duckdb::LogicalComparisonJoin>();
    bool const semi  = delim.join_type == JoinType::SEMI || delim.join_type == JoinType::RIGHT_SEMI;
    bool const anti  = delim.join_type == JoinType::ANTI || delim.join_type == JoinType::RIGHT_ANTI;
    auto const outer = delim.delim_flipped ? std::size_t{1} : std::size_t{0};
    bool const emits_lh = delim.join_type == JoinType::SEMI || delim.join_type == JoinType::ANTI;
    if ((!semi && !anti) || emits_lh != (outer == 0)) { return std::nullopt; }
    auto match = match_delim_subquery(delim, outer);
    if (!match) { return std::nullopt; }
    match->replaced = &delim;
    match->negated  = anti;
    return match;
  }

  if (node.type != LogicalOperatorType::LOGICAL_FILTER || node.expressions.size() != 1 ||
      node.children.size() != 1 ||
      node.children[0]->type != LogicalOperatorType::LOGICAL_DELIM_JOIN) {
    return std::nullopt;
  }
  auto& delim = node.children[0]->Cast<duckdb::LogicalComparisonJoin>();
  if (delim.join_type != JoinType::MARK || delim.delim_flipped || delim.children.size() != 2) {
    return std::nullopt;
  }
  Expression const* predicate = node.expressions[0].get();
  bool negated                = false;
  if (predicate->GetExpressionType() == ExpressionType::OPERATOR_NOT) {
    auto const& operands = predicate->Cast<duckdb::BoundOperatorExpression>().GetChildren();
    if (operands.size() != 1) { return std::nullopt; }
    predicate = operands[0].get();
    negated   = true;
  }
  ColumnBinding const mark(delim.mark_index, ProjectionIndex(0));
  auto const read = reads.find(mark);
  if (!is_column(*predicate) || !(binding_of(*predicate) == mark) || read == reads.end() ||
      read->second != 1) {
    return std::nullopt;
  }
  auto match = match_delim_subquery(delim, 0);
  if (!match) { return std::nullopt; }
  match->replaced = &node;
  match->negated  = negated;
  match->mark     = mark;
  return match;
}

/// Positions in @p outer's output of every column @p replaced emits, dropping a trailing @p mark.
/// Null when it emits anything else.
std::optional<duckdb::vector<ProjectionIndex>> output_map(LogicalOperator& replaced,
                                                          LogicalOperator& outer,
                                                          std::optional<ColumnBinding> mark)
{
  auto output = replaced.GetColumnBindings();
  if (mark && !output.empty() && output.back() == *mark) { output.pop_back(); }
  auto const available = outer.GetColumnBindings();
  duckdb::vector<ProjectionIndex> map;
  for (auto const& binding : output) {
    auto const found = std::find(available.begin(), available.end(), binding);
    if (found == available.end()) { return std::nullopt; }
    map.emplace_back(static_cast<duckdb::idx_t>(found - available.begin()));
  }
  return map;
}

unique_ptr<duckdb::BoundAggregateExpression> bind_extremum(duckdb::ClientContext& context,
                                                           char const* name,
                                                           LogicalType const& type,
                                                           unique_ptr<Expression> argument)
{
  auto& entry =
    duckdb::Catalog::GetSystemCatalog(context).GetEntry<duckdb::AggregateFunctionCatalogEntry>(
      context, DEFAULT_SCHEMA, name);
  auto function = entry.functions.GetFunctionByArguments(context, {type});
  duckdb::vector<unique_ptr<Expression>> children;
  children.push_back(std::move(argument));
  duckdb::FunctionBinder binder(context);
  return binder.BindAggregateFunction(std::move(function), std::move(children));
}

/// One subquery answered by the shared aggregate.
struct subquery {
  unique_ptr<Expression> value;      ///< The compared column, over T
  unique_ptr<Expression> predicate;  ///< AND of its own filters over T, or null
  ColumnBinding outer_value;
  bool negated         = false;
  bool value_null_safe = false;
};

unique_ptr<Expression> unary(ExpressionType type, unique_ptr<Expression> operand)
{
  auto result = duckdb::make_uniq<duckdb::BoundOperatorExpression>(type, LogicalType::BOOLEAN);
  result->GetChildrenMutable().push_back(std::move(operand));
  return result;
}

unique_ptr<Expression> both(ExpressionType type, unique_ptr<Expression> a, unique_ptr<Expression> b)
{
  if (!a) { return b; }
  return duckdb::make_uniq<duckdb::BoundConjunctionExpression>(type, std::move(a), std::move(b));
}

unique_ptr<Expression> compare(ExpressionType type,
                               unique_ptr<Expression> a,
                               unique_ptr<Expression> b)
{
  return duckdb::BoundComparisonExpression::Create(type, std::move(a), std::move(b));
}

/// Replace the matched nodes with the shape DuckDB itself produces for a correlated scalar
/// aggregate:
///
///   FILTER  EXISTS:     mn <> outer.x OR mx <> outer.x
///           NOT EXISTS: (that) IS NULL OR NOT (that)
///    └─ LEFT DELIM_JOIN on outer.k IS NOT DISTINCT FROM a.k
///        ├─ outer rows
///        └─ a = AGGREGATE(GROUP BY k: MIN(x), MAX(x) per subquery) ← INNER JOIN(T, DELIM_GET(k))
///
/// `EXISTS ... t.x <> outer.x` holds exactly when MIN or MAX of `t.x` over the key differs from
/// `outer.x`, with NULL comparisons never counting as different; NOT EXISTS keeps the rest,
/// including rows with no group. For `IS DISTINCT FROM`, the aggregate also records whether a NULL
/// `t.x` was seen: a NULL `outer.x` differs from any non-NULL `t.x`, and a non-NULL one from a
/// NULL `t.x` or from MIN or MAX. A subquery's own filters move into its aggregates as a CASE.
/// The filter's projection map restores exactly the columns the replaced node emitted.
unique_ptr<LogicalOperator> build(duckdb::ClientContext& context,
                                  duckdb::Binder& binder,
                                  unique_ptr<LogicalOperator> outer,
                                  ColumnBinding outer_key,
                                  unique_ptr<LogicalOperator> t,
                                  ColumnBinding t_key,
                                  LogicalType const& key_type,
                                  bool key_null_safe,
                                  std::vector<subquery> subqueries,
                                  duckdb::vector<ProjectionIndex> output,
                                  duckdb::idx_t output_estimate)
{
  auto const outer_estimate = outer->estimated_cardinality;

  auto delim_get = duckdb::make_uniq<duckdb::LogicalDelimGet>(
    binder.GenerateTableIndex(), duckdb::vector<LogicalType>{key_type});
  delim_get->SetEstimatedCardinality(outer_estimate);
  ColumnBinding const distinct_key(delim_get->table_index, ProjectionIndex(0));

  auto joined = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  joined->conditions.emplace_back(
    column(key_type, t_key),
    column(key_type, distinct_key),
    key_null_safe ? ExpressionType::COMPARE_NOT_DISTINCT_FROM : ExpressionType::COMPARE_EQUAL);
  joined->SetEstimatedCardinality(t->estimated_cardinality);
  joined->children.push_back(std::move(t));
  joined->children.push_back(std::move(delim_get));

  // MIN, MAX and, for a null-safe value, MAX(a NULL value was seen) per subquery.
  duckdb::vector<unique_ptr<Expression>> aggregates;
  std::vector<std::size_t> first;
  std::vector<LogicalType> types;
  for (auto& sub : subqueries) {
    first.push_back(aggregates.size());
    types.push_back(sub.value->GetReturnType());
    auto const& type = types.back();
    if (sub.value_null_safe) {
      auto seen = duckdb::make_uniq<duckdb::BoundCaseExpression>(
        both(ExpressionType::CONJUNCTION_AND,
             sub.predicate ? sub.predicate->Copy() : nullptr,
             unary(ExpressionType::OPERATOR_IS_NULL, sub.value->Copy())),
        duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1)),
        duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(0)));
      aggregates.push_back(bind_extremum(context, "max", LogicalType::INTEGER, std::move(seen)));
    }
    auto argument = std::move(sub.value);
    if (sub.predicate) {
      argument = duckdb::make_uniq<duckdb::BoundCaseExpression>(
        std::move(sub.predicate),
        std::move(argument),
        duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value(type)));
    }
    aggregates.push_back(bind_extremum(context, "min", type, argument->Copy()));
    aggregates.push_back(bind_extremum(context, "max", type, std::move(argument)));
  }
  auto const group_index     = binder.GenerateTableIndex();
  auto const aggregate_index = binder.GenerateTableIndex();
  auto grouped               = duckdb::make_uniq<duckdb::LogicalAggregate>(
    group_index, aggregate_index, std::move(aggregates));
  grouped->groups.push_back(column(key_type, t_key));
  grouped->grouping_sets.push_back(duckdb::GroupingSet{ProjectionIndex(0)});
  grouped->group_stats.resize(1);
  grouped->distinct_validity = duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES;
  grouped->SetEstimatedCardinality(outer_estimate);
  grouped->children.push_back(std::move(joined));

  auto left = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(
    duckdb::JoinType::LEFT, LogicalOperatorType::LOGICAL_DELIM_JOIN);
  left->conditions.emplace_back(column(key_type, outer_key),
                                column(key_type, ColumnBinding(group_index, ProjectionIndex(0))),
                                ExpressionType::COMPARE_NOT_DISTINCT_FROM);
  left->duplicate_eliminated_columns.push_back(column(key_type, outer_key));
  left->SetEstimatedCardinality(outer_estimate);
  left->children.push_back(std::move(outer));
  left->children.push_back(std::move(grouped));

  unique_ptr<Expression> predicate;
  for (std::size_t i = 0; i < subqueries.size(); ++i) {
    auto const& sub      = subqueries[i];
    auto const aggregate = [&](std::size_t offset, LogicalType const& type) {
      return column(type, ColumnBinding(aggregate_index, ProjectionIndex(first[i] + offset)));
    };
    auto const outer_x  = [&] { return column(types[i], sub.outer_value); };
    auto const extremum = sub.value_null_safe ? std::size_t{1} : std::size_t{0};
    auto const differs  = [&](std::size_t offset) {
      return compare(ExpressionType::COMPARE_NOTEQUAL, aggregate(offset, types[i]), outer_x());
    };
    auto condition = both(ExpressionType::CONJUNCTION_OR, differs(extremum), differs(extremum + 1));
    if (sub.value_null_safe) {
      auto null_seen =
        compare(ExpressionType::COMPARE_EQUAL,
                aggregate(0, LogicalType::INTEGER),
                duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1)));
      condition = both(
        ExpressionType::CONJUNCTION_OR,
        both(ExpressionType::CONJUNCTION_AND,
             unary(ExpressionType::OPERATOR_IS_NULL, outer_x()),
             unary(ExpressionType::OPERATOR_IS_NOT_NULL, aggregate(1, types[i]))),
        both(ExpressionType::CONJUNCTION_AND,
             unary(ExpressionType::OPERATOR_IS_NOT_NULL, outer_x()),
             both(ExpressionType::CONJUNCTION_OR, std::move(null_seen), std::move(condition))));
    }
    if (sub.negated) {
      condition = both(ExpressionType::CONJUNCTION_OR,
                       unary(ExpressionType::OPERATOR_IS_NULL, condition->Copy()),
                       unary(ExpressionType::OPERATOR_NOT, condition->Copy()));
    }
    predicate = both(ExpressionType::CONJUNCTION_AND, std::move(predicate), std::move(condition));
  }
  auto filter            = duckdb::make_uniq<duckdb::LogicalFilter>(std::move(predicate));
  filter->projection_map = std::move(output);
  filter->SetEstimatedCardinality(output_estimate);
  filter->children.push_back(std::move(left));
  return filter;
}

// ---------------------------------------------------------------------------------------------
// Merging sibling subqueries
//
// `EXISTS (... l2.k = l1.k AND l2.x <> l1.x) AND NOT EXISTS (... l3.k = l1.k AND l3.x <> l1.x AND
// <p>)` become two nested delim joins, each with a scan of the same table. When the scans read the
// same source, one scan serves both: each subquery's own filter turns into a CASE inside its
// MIN/MAX, so a single aggregate keyed by the shared correlation key answers every subquery.
// ---------------------------------------------------------------------------------------------

/// One subquery's scan with its filters peeled off.
struct peeled_scan {
  duckdb::LogicalGet* get                = nullptr;
  unique_ptr<LogicalOperator>* get_owner = nullptr;
  /// Every condition on the scan's rows, from filter nodes or pushed-down table filters, over the
  /// scan's own columns.
  duckdb::vector<unique_ptr<Expression>> predicates;
  std::vector<duckdb::LogicalProjection const*> projections;  ///< Above the scan, outermost first
};

/// The plain base-table column behind slot @p slot of @p gets column ids.
std::optional<duckdb::idx_t> base_column(duckdb::LogicalGet const& get, duckdb::idx_t slot)
{
  auto const& ids = get.GetColumnIds();
  if (slot >= ids.size() || !ids[slot].HasPrimaryIndex() || ids[slot].HasChildren() ||
      ids[slot].IsVirtualColumn() || ids[slot].GetPrimaryIndex() >= get.returned_types.size()) {
    return std::nullopt;
  }
  return ids[slot].GetPrimaryIndex();
}

/// Peel filters and plain-column projections down to a GET. Anything else between the
/// subquery's join and its scan declines.
std::optional<peeled_scan> peel_scan(unique_ptr<LogicalOperator>& root)
{
  peeled_scan scan;
  auto* owner = &root;
  for (;; owner = &(*owner)->children[0]) {
    if ((*owner)->type == LogicalOperatorType::LOGICAL_FILTER && (*owner)->children.size() == 1) {
      for (auto const& expression : (*owner)->expressions) {
        scan.predicates.push_back(expression->Copy());
      }
    } else if ((*owner)->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      scan.projections.push_back(&(*owner)->Cast<duckdb::LogicalProjection>());
    } else {
      break;
    }
  }
  bool resolved = true;
  for (auto& predicate : scan.predicates) {
    duckdb::ExpressionIterator::VisitExpressionClassMutable(
      predicate, ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& leaf) {
        auto const below = see_through(scan.projections, binding_of(*leaf));
        resolved         = resolved && below;
        if (below) { leaf = column(leaf->GetReturnType(), *below); }
      });
  }
  if (!resolved) { return std::nullopt; }
  if ((*owner)->type != LogicalOperatorType::LOGICAL_GET || !(*owner)->children.empty()) {
    return std::nullopt;
  }
  scan.get       = &(*owner)->Cast<duckdb::LogicalGet>();
  scan.get_owner = owner;
  if (scan.get->table_filters.HasMultiColumnFilters()) { return std::nullopt; }
  for (auto const& entry : scan.get->table_filters) {
    auto const slot = entry.GetIndex().GetIndex();
    auto const base = base_column(*scan.get, slot);
    if (!base) { return std::nullopt; }
    auto const tested = column(scan.get->returned_types[*base],
                               ColumnBinding(scan.get->table_index, ProjectionIndex(slot)));
    scan.predicates.push_back(entry.Filter().ToExpression(*tested));
  }
  return scan;
}

/// Whether two scans read the same view of the same source. This is the identity the transparent
/// execution path itself compares between plans (`capture_bound_read_view`), which a table
/// function's own bind-data equality does not reliably provide (parquet's does not, even for a
/// scan against itself).
bool same_source(duckdb::ClientContext& context,
                 duckdb::LogicalGet const& a,
                 duckdb::LogicalGet const& b)
{
  if (a.function.GetName() != b.function.GetName() || a.returned_types != b.returned_types ||
      a.names != b.names) {
    return false;
  }
  try {
    auto const view_a = sirius::op::scan::capture_bound_read_view(a, context);
    auto const view_b = sirius::op::scan::capture_bound_read_view(b, context);
    return view_a.identity && view_b.identity &&
           view_a.identity->fingerprint == view_b.identity->fingerprint;
  } catch (std::exception const&) {
    return false;
  }
}

/// The slot of @p get that outputs base column @p base, reading it if the scan does not already.
duckdb::idx_t widen(duckdb::LogicalGet& get, duckdb::idx_t base)
{
  auto& ids          = get.GetMutableColumnIds();
  duckdb::idx_t slot = 0;
  while (slot < ids.size() && base_column(get, slot) != base) {
    ++slot;
  }
  if (slot == ids.size()) { ids.push_back(duckdb::ColumnIndex(base)); }
  auto& projected = get.projection_ids;
  if (!projected.empty() &&
      std::find(projected.begin(), projected.end(), ProjectionIndex(slot)) == projected.end()) {
    projected.push_back(ProjectionIndex(slot));
  }
  return slot;
}

/// Whether every column @p expression reads is a plain base column of @p get.
bool reads_only(Expression const& expression, duckdb::LogicalGet const& get)
{
  bool ok = true;
  duckdb::ExpressionIterator::VisitExpressionClass(
    expression, ExpressionClass::BOUND_COLUMN_REF, [&](Expression const& leaf) {
      auto const binding = binding_of(leaf);
      ok = ok && binding.table_index == get.table_index && base_column(get, binding.column_index);
    });
  return ok;
}

/// Re-address @p expression from @p from's columns onto the same base columns of @p to.
unique_ptr<Expression> move_onto(unique_ptr<Expression> expression,
                                 duckdb::LogicalGet const& from,
                                 duckdb::LogicalGet& to)
{
  duckdb::ExpressionIterator::VisitExpressionClassMutable(
    expression, ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& leaf) {
      auto const base = *base_column(from, binding_of(*leaf).column_index);
      leaf            = column(leaf->GetReturnType(),
                    ColumnBinding(to.table_index, ProjectionIndex(widen(to, base))));
    });
  return expression;
}

/// Rewrite a chain of two or more matched joins (innermost first) over the same scan, one level
/// nested in the next through its outer rows. Returns null, consuming nothing, when the subqueries
/// cannot share one scan, or when they are plain joins estimated to fit in @p gpu_memory_budget
/// bytes (0: unknown).
unique_ptr<LogicalOperator> rewrite_merged(duckdb::ClientContext& context,
                                           duckdb::Binder& binder,
                                           std::vector<exists_match>& chain,
                                           std::uint64_t gpu_memory_budget)
{
  std::vector<peeled_scan> scans;
  for (auto& match : chain) {
    auto scan = peel_scan(*match.t);
    if (!scan) { return nullptr; }
    scans.push_back(std::move(*scan));
  }

  // One source and one correlation key; every column a subquery reads comes from its own scan, and
  // every outer value it compares is a column of the innermost outer rows.
  std::vector<ColumnBinding> t_keys, t_values;
  for (std::size_t i = 0; i < chain.size(); ++i) {
    auto const key   = see_through(scans[i].projections, chain[i].t_key);
    auto const value = see_through(scans[i].projections, chain[i].t_value);
    if (!key || !value) { return nullptr; }
    t_keys.push_back(*key);
    t_values.push_back(*value);
  }
  auto& innermost = chain.front();
  auto& outer     = innermost.join->children[innermost.outer_child];
  auto& shared    = *scans.front().get;
  if (!reads_only(*column(innermost.key_type, t_keys[0]), shared)) { return nullptr; }
  auto const key_base    = *base_column(shared, t_keys[0].column_index);
  auto const outer_names = outer->GetColumnBindings();
  for (std::size_t i = 0; i < chain.size(); ++i) {
    auto const& get = *scans[i].get;
    if (!same_source(context, shared, get) || chain[i].key_type != innermost.key_type ||
        !(chain[i].outer_key == innermost.outer_key) ||
        !reads_only(*column(chain[i].key_type, t_keys[i]), get) ||
        base_column(get, t_keys[i].column_index) != key_base ||
        !reads_only(*column(chain[i].value_type, t_values[i]), get) ||
        std::find(outer_names.begin(), outer_names.end(), chain[i].outer_value) ==
          outer_names.end() ||
        !std::ranges::all_of(scans[i].predicates,
                             [&](auto const& predicate) { return reads_only(*predicate, get); })) {
      return nullptr;
    }
  }
  auto output = output_map(*chain.back().replaced, *outer, chain.back().mark);
  if (!output) { return nullptr; }
  if (gpu_memory_budget > 0 &&
      std::ranges::all_of(chain, [](auto const& match) { return match.plain; })) {
    // The plain joins hash each subquery's rows on the GPU, which is cheaper than the aggregate
    // until those rows no longer fit and spill. The base-table row count is used because
    // estimates of the subqueries' filters can be far too low.
    std::uint64_t join_bytes = 0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
      auto const& get = *scans[i].get;
      if (!get.has_estimated_cardinality) { continue; }
      auto const row_bytes = duckdb::GetTypeIdSize(chain[i].key_type.InternalType()) +
                             duckdb::GetTypeIdSize(chain[i].value_type.InternalType());
      join_bytes = std::max<std::uint64_t>(join_bytes, get.estimated_cardinality * row_bytes);
    }
    bool const spills = join_bytes == 0 || join_bytes > gpu_memory_budget;
    SIRIUS_LOG_INFO(
      "[sirius_plan_exists_rewrite] {} {} sibling EXISTS/NOT EXISTS: their joins would hold an "
      "estimated {} bytes against a GPU budget of {} bytes",
      spills ? "Merging" : "Not merging",
      chain.size(),
      join_bytes,
      gpu_memory_budget);
    if (!spills) { return nullptr; }
  }

  // ---- Nothing below declines: consume the original nodes. ----
  // The shared scan's pushed-down filters have become its subquery's predicate, so it must return
  // every row.
  shared.table_filters.ClearFilters();
  ColumnBinding const shared_key(shared.table_index, ProjectionIndex(widen(shared, key_base)));
  // A NULL key forms a group when any subquery compares keys null-safely; the others skip it.
  bool const key_null_safe =
    std::ranges::any_of(chain, [](auto const& match) { return match.key_null_safe; });
  std::vector<subquery> subqueries;
  for (std::size_t i = 0; i < chain.size(); ++i) {
    subquery sub{move_onto(column(chain[i].value_type, t_values[i]), *scans[i].get, shared),
                 nullptr,
                 chain[i].outer_value,
                 chain[i].negated,
                 chain[i].value_null_safe};
    for (auto& predicate : scans[i].predicates) {
      sub.predicate = both(ExpressionType::CONJUNCTION_AND,
                           std::move(sub.predicate),
                           move_onto(std::move(predicate), *scans[i].get, shared));
    }
    if (key_null_safe && !chain[i].key_null_safe) {
      sub.predicate =
        both(ExpressionType::CONJUNCTION_AND,
             std::move(sub.predicate),
             unary(ExpressionType::OPERATOR_IS_NOT_NULL, column(innermost.key_type, shared_key)));
    }
    subqueries.push_back(std::move(sub));
  }
  return build(context,
               binder,
               std::move(outer),
               innermost.outer_key,
               std::move(*scans.front().get_owner),
               shared_key,
               innermost.key_type,
               key_null_safe,
               std::move(subqueries),
               std::move(*output),
               chain.back().replaced->estimated_cardinality);
}

/// The longest chain of matched joins directly nested through their outer child, innermost
/// first.
std::vector<exists_match> match_chain(LogicalOperator& node,
                                      duckdb::column_binding_map_t<std::size_t> const& reads)
{
  std::vector<exists_match> outermost_first;
  auto* current = &node;
  while (auto match = match_exists(*current, reads)) {
    outermost_first.push_back(*match);
    current = match->join->children[match->outer_child].get();
  }
  return {outermost_first.rbegin(), outermost_first.rend()};
}

void count_reads(LogicalOperator& op, duckdb::column_binding_map_t<std::size_t>& reads)
{
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression>* expression) {
    duckdb::ExpressionIterator::VisitExpressionClass(
      **expression, ExpressionClass::BOUND_COLUMN_REF, [&](Expression const& leaf) {
        ++reads[binding_of(leaf)];
      });
  });
  for (auto& child : op.children) {
    count_reads(*child, reads);
  }
}

void rewrite_tree(duckdb::ClientContext& context,
                  duckdb::Binder& binder,
                  unique_ptr<LogicalOperator>& slot,
                  duckdb::column_binding_map_t<std::size_t> const& reads,
                  std::uint64_t gpu_memory_budget,
                  std::size_t& rewrites)
{
  // Only sibling subqueries are rewritten: a lone one trades its join for a GROUP BY over the
  // same rows, which is rarely faster.
  if (auto chain = match_chain(*slot, reads); chain.size() >= 2) {
    if (auto merged = rewrite_merged(context, binder, chain, gpu_memory_budget)) {
      slot = std::move(merged);
      rewrites += chain.size();
      // The merged node's outer rows may hold further subqueries.
      rewrite_tree(
        context, binder, slot->children[0]->children[0], reads, gpu_memory_budget, rewrites);
      return;
    }
  }
  for (auto& child : slot->children) {
    rewrite_tree(context, binder, child, reads, gpu_memory_budget, rewrites);
  }
}

}  // namespace

std::size_t rewrite_correlated_inequality_exists(duckdb::ClientContext& context,
                                                 duckdb::Binder& binder,
                                                 duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                                                 std::uint64_t gpu_memory_budget)
{
  duckdb::column_binding_map_t<std::size_t> reads;
  count_reads(*plan, reads);
  std::size_t rewrites = 0;
  rewrite_tree(context, binder, plan, reads, gpu_memory_budget, rewrites);
  if (rewrites > 0) {
    SIRIUS_LOG_INFO(
      "[sirius_plan_exists_rewrite] Rewrote {} EXISTS/NOT EXISTS on a key and a value inequality "
      "into a MIN/MAX aggregate per key",
      rewrites);
  }
  return rewrites;
}

}  // namespace sirius::planner
