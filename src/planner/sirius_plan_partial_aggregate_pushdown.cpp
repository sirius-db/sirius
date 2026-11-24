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

#include "planner/sirius_plan_partial_aggregate_pushdown.hpp"

#include "duckdb/common/enums/tuple_data_layout_enums.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "log/logging.hpp"
#include "planner/aggregate_binding.hpp"
#include "planner/dynamic_filter/build_key_domain.hpp"
#include "transparent/parquet_footer_statistics.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::ExpressionClass;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::LogicalType;
using duckdb::ProjectionIndex;
using duckdb::TableIndex;
using duckdb::unique_ptr;

/// The share of its unfiltered rows a scan must be estimated to keep for a join against it to
/// count as keeping the rows of its other side.
constexpr double k_min_kept_share = 0.7;

/// The columns the pushed aggregate may group by beyond those it is joined on.
constexpr std::size_t k_max_extra_keys = 1;

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

duckdb::column_binding_set_t outputs_of(LogicalOperator& op)
{
  auto const bindings = op.GetColumnBindings();
  return {bindings.begin(), bindings.end()};
}

void append_columns_read(Expression const& expression, std::vector<ColumnBinding>& columns)
{
  duckdb::ExpressionIterator::VisitExpressionClass(
    expression, ExpressionClass::BOUND_COLUMN_REF, [&](Expression const& leaf) {
      columns.push_back(binding_of(leaf));
    });
}

void append_columns_read(duckdb::LogicalComparisonJoin const& join,
                         std::vector<ColumnBinding>& columns)
{
  for (auto const& condition : join.conditions) {
    if (condition.IsComparison()) {
      append_columns_read(condition.GetLHS(), columns);
      append_columns_read(condition.GetRHS(), columns);
    } else {
      append_columns_read(condition.GetJoinExpression(), columns);
    }
  }
}

/// The columns @p join's conditions read through an expression, not as a bare column.
void append_columns_computed_from(duckdb::LogicalComparisonJoin const& join,
                                  std::vector<ColumnBinding>& columns)
{
  auto const append = [&](Expression const& expression) {
    if (!is_column(expression)) { append_columns_read(expression, columns); }
  };
  for (auto const& condition : join.conditions) {
    if (condition.IsComparison()) {
      append(condition.GetLHS());
      append(condition.GetRHS());
    } else {
      append_columns_read(condition.GetJoinExpression(), columns);
    }
  }
}

/// Whether values of @p type that group together are the same value to every expression. Zeros of
/// either sign group together but `1 / x` tells them apart, and intervals and zoned times that
/// compare equal can be stored differently.
bool equal_means_identical(LogicalType const& type)
{
  switch (type.id()) {
    case duckdb::LogicalTypeId::BOOLEAN:
    case duckdb::LogicalTypeId::TINYINT:
    case duckdb::LogicalTypeId::SMALLINT:
    case duckdb::LogicalTypeId::INTEGER:
    case duckdb::LogicalTypeId::BIGINT:
    case duckdb::LogicalTypeId::HUGEINT:
    case duckdb::LogicalTypeId::UTINYINT:
    case duckdb::LogicalTypeId::USMALLINT:
    case duckdb::LogicalTypeId::UINTEGER:
    case duckdb::LogicalTypeId::UBIGINT:
    case duckdb::LogicalTypeId::UHUGEINT:
    case duckdb::LogicalTypeId::DECIMAL:
    case duckdb::LogicalTypeId::DATE:
    case duckdb::LogicalTypeId::TIME:
    case duckdb::LogicalTypeId::TIMESTAMP:
    case duckdb::LogicalTypeId::TIMESTAMP_SEC:
    case duckdb::LogicalTypeId::TIMESTAMP_MS:
    case duckdb::LogicalTypeId::TIMESTAMP_NS:
    case duckdb::LogicalTypeId::TIMESTAMP_TZ:
    case duckdb::LogicalTypeId::UUID: return true;
    default: return false;
  }
}

// ---------------------------------------------------------------------------------------------
// The aggregate's input
// ---------------------------------------------------------------------------------------------

using inlined_columns = duckdb::column_binding_map_t<unique_ptr<Expression>>;

/// @p expression with every column of @p columns replaced by its value.
unique_ptr<Expression> inline_columns(Expression const& expression, inlined_columns const& columns)
{
  auto copy = expression.Copy();
  duckdb::ExpressionIterator::VisitExpressionClassMutable(
    copy, ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& ref) {
      auto const found = columns.find(binding_of(*ref));
      if (found != columns.end()) { ref = found->second->Copy(); }
    });
  return copy;
}

/// The projections directly under an aggregate, folded away: `columns` gives each of their
/// columns as an expression over the first operator below them, which `input` holds. `input` is
/// null when a projection is volatile, since inlining it would evaluate it once per reader.
struct aggregate_input {
  unique_ptr<LogicalOperator>* input = nullptr;
  inlined_columns columns;
};

aggregate_input fold_projections(duckdb::LogicalAggregate& aggregate)
{
  std::vector<duckdb::LogicalProjection*> projections;
  auto* slot = &aggregate.children[0];
  while ((*slot)->type == LogicalOperatorType::LOGICAL_PROJECTION &&
         (*slot)->children.size() == 1) {
    projections.push_back(&(*slot)->Cast<duckdb::LogicalProjection>());
    slot = &(*slot)->children[0];
  }
  aggregate_input result;
  for (auto projection = projections.rbegin(); projection != projections.rend(); ++projection) {
    auto const& expressions = (*projection)->expressions;
    for (std::size_t i = 0; i < expressions.size(); ++i) {
      if (expressions[i]->IsVolatile()) { return {}; }
      result.columns[ColumnBinding((*projection)->table_index, ProjectionIndex(i))] =
        inline_columns(*expressions[i], result.columns);
    }
  }
  result.input = slot;
  return result;
}

/// One GROUP BY of every group: no grouping sets beyond the complete one, which DuckDB may leave
/// implicit.
bool groups_once(duckdb::LogicalAggregate const& aggregate)
{
  auto const& sets = aggregate.grouping_sets;
  return sets.empty() || (sets.size() == 1 && sets[0].size() == aggregate.groups.size());
}

// ---------------------------------------------------------------------------------------------
// Which aggregates can be computed in two steps
// ---------------------------------------------------------------------------------------------

/// The aggregate that combines results of the aggregate @p name over parts of a group into its
/// result over the whole group, or null when none is supported.
char const* combining_aggregate(std::string const& name)
{
  if (name == "sum" || name == "sum_no_overflow") { return "sum"; }
  if (name == "min") { return "min"; }
  if (name == "max") { return "max"; }
  return nullptr;
}

/// A SUM whose results add up to a result of the same type without an overflow the original could
/// not have: DuckDB sums a DECIMAL stored in at most 64 bits into a DECIMAL(38), and the sum of
/// such sums is the same DECIMAL(38) total. Integer sums combine through a 128-bit sum that the
/// GPU range-checks on every merge, and floating-point sums depend on the order of addition.
bool sum_combines_exactly(LogicalType const& argument)
{
  if (argument.id() != duckdb::LogicalTypeId::DECIMAL) { return false; }
  switch (argument.InternalType()) {
    case duckdb::PhysicalType::INT16:
    case duckdb::PhysicalType::INT32:
    case duckdb::PhysicalType::INT64: return true;
    default: return false;
  }
}

/// Whether @p expression is accepted on rows a join above would have removed: it cannot throw, or
/// it only adds, subtracts and multiplies such values. Accepting that arithmetic is Sirius's own
/// rule: it can still overflow on a row the unrewritten plan never evaluates it on, and DuckDB's
/// partial aggregate pushdown declines it unless column statistics rule the overflow out.
bool arithmetic_or_infallible(Expression const& expression)
{
  if (!expression.CanThrow()) { return true; }
  if (expression.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) { return false; }
  auto const& function = expression.Cast<duckdb::BoundFunctionExpression>();
  auto const& name     = function.Function().GetName().GetIdentifierName();
  if (name != "+" && name != "-" && name != "*" && name != "add" && name != "subtract" &&
      name != "multiply") {
    return false;
  }
  return std::ranges::all_of(function.GetChildren(),
                             [](auto const& child) { return arithmetic_or_infallible(*child); });
}

// ---------------------------------------------------------------------------------------------
// Whether the push pays off
// ---------------------------------------------------------------------------------------------

/// Whether a join against @p op is expected to keep nearly every row of its other side: @p op is
/// made of scans that keep most of their rows, joined to each other. A join against a filtered
/// side discards rows, and aggregating them first would be wasted. Scan estimates are compared
/// with the unfiltered row count; a filter the join itself implies is added after DuckDB
/// estimates the scan, so it does not count.
bool keeps_rows(duckdb::ClientContext& context, LogicalOperator const& op)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET: {
      auto const& get = op.Cast<duckdb::LogicalGet>();
      if (!get.children.empty() || !get.has_estimated_cardinality) { return false; }
      auto rows = transparent::parquet_scan_row_count(get);
      if (!rows) { rows = duckdb_base_table_cardinality{context}(get); }
      return rows && static_cast<double>(get.estimated_cardinality) >=
                       k_min_kept_share * static_cast<double>(*rows);
    }
    case LogicalOperatorType::LOGICAL_PROJECTION:
      return op.children.size() == 1 && keeps_rows(context, *op.children[0]);
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      return op.Cast<duckdb::LogicalComparisonJoin>().join_type == duckdb::JoinType::INNER &&
             std::ranges::all_of(op.children,
                                 [&](auto const& child) { return keeps_rows(context, *child); });
    default: return false;
  }
}

/// Whether @p op already has one row per group of some columns, so grouping it again by a join
/// key reduces little.
bool already_grouped(LogicalOperator const& op)
{
  auto const* node = &op;
  while ((node->type == LogicalOperatorType::LOGICAL_PROJECTION ||
          node->type == LogicalOperatorType::LOGICAL_FILTER) &&
         node->children.size() == 1) {
    node = node->children[0].get();
  }
  return node->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY ||
         node->type == LogicalOperatorType::LOGICAL_DISTINCT;
}

// ---------------------------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------------------------

/// A join between the aggregate and the input its arguments come from.
struct path_step {
  duckdb::LogicalComparisonJoin* join;
  std::size_t side;                      ///< The child the arguments come from
  std::vector<ColumnBinding> forwarded;  ///< That child's columns the join outputs
};

/// @p side's columns that @p join outputs, or nullopt when its projection map is out of range.
std::optional<std::vector<ColumnBinding>> forwarded_columns(duckdb::LogicalComparisonJoin& join,
                                                            std::size_t side)
{
  auto const columns = join.children[side]->GetColumnBindings();
  auto const& map    = side == 0 ? join.left_projection_map : join.right_projection_map;
  if (map.empty()) { return std::vector<ColumnBinding>{columns.begin(), columns.end()}; }
  std::vector<ColumnBinding> forwarded;
  for (auto const index : map) {
    if (index.GetIndex() >= columns.size()) { return std::nullopt; }
    forwarded.push_back(columns[index.GetIndex()]);
  }
  return forwarded;
}

struct pushdown {
  duckdb::LogicalAggregate* aggregate = nullptr;
  unique_ptr<LogicalOperator>* input  = nullptr;  ///< The aggregate's input below its projections
  std::vector<path_step> path;      ///< From that input down to the join the aggregate goes under
  std::vector<ColumnBinding> keys;  ///< The columns the pushed aggregate groups by
  std::vector<LogicalType> key_types;
  duckdb::vector<unique_ptr<Expression>> groups;  ///< The groups, over the joins' columns
  duckdb::vector<unique_ptr<Expression>> lower;   ///< The aggregates, over the pushed input
  duckdb::vector<unique_ptr<Expression>> upper;   ///< The aggregates combining those
  TableIndex lower_group_index;
  TableIndex lower_aggregate_index;
};

std::nullopt_t decline(char const* reason)
{
  SIRIUS_LOG_DEBUG(
    "[sirius_plan_partial_aggregate_pushdown] Not aggregating below a join that adds group-by "
    "payload: {}",
    reason);
  return std::nullopt;
}

/// Whether @p aggregate can be computed in two steps around a join that only adds payload, and
/// how. Changes nothing.
std::optional<pushdown> match(duckdb::ClientContext& context,
                              duckdb::Binder& binder,
                              duckdb::LogicalAggregate& aggregate)
{
  if (aggregate.children.size() != 1 || aggregate.groups.empty() || aggregate.expressions.empty() ||
      !aggregate.grouping_functions.empty() || !groups_once(aggregate)) {
    return std::nullopt;
  }
  auto folded = fold_projections(aggregate);
  if (!folded.input || (*folded.input)->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return std::nullopt;
  }

  pushdown result;
  result.aggregate = &aggregate;
  result.input     = folded.input;
  for (auto const& group : aggregate.groups) {
    auto value = inline_columns(*group, folded.columns);
    if (!is_column(*value)) { return std::nullopt; }
    result.groups.push_back(std::move(value));
  }

  // Each aggregate's argument over the joins' columns, and what combines its results.
  duckdb::vector<unique_ptr<Expression>> arguments;
  std::vector<char const*> combines;
  std::vector<ColumnBinding> argument_columns;
  for (auto const& expression : aggregate.expressions) {
    if (expression->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
      return std::nullopt;
    }
    auto const& original = expression->Cast<duckdb::BoundAggregateExpression>();
    auto const* combine  = combining_aggregate(original.Function().GetName().GetIdentifierName());
    if (!combine || original.IsDistinct() || original.GetFilter() || original.GetOrderBys() ||
        original.GetChildren().size() != 1) {
      return std::nullopt;
    }
    auto argument = inline_columns(*original.GetChildren()[0], folded.columns);
    if (argument->IsVolatile() || !arithmetic_or_infallible(*argument)) { return std::nullopt; }
    if (std::string_view{combine} == "sum" && !sum_combines_exactly(argument->GetReturnType())) {
      return std::nullopt;
    }
    auto const columns_before = argument_columns.size();
    append_columns_read(*argument, argument_columns);
    if (argument_columns.size() == columns_before) { return std::nullopt; }
    arguments.push_back(std::move(argument));
    combines.push_back(combine);
  }

  // Down through inner joins, on the side that holds every argument.
  auto* node = folded.input->get();
  while (node->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto& join = node->Cast<duckdb::LogicalComparisonJoin>();
    if (join.join_type != duckdb::JoinType::INNER || join.children.size() != 2 ||
        join.conditions.empty()) {
      break;
    }
    std::optional<std::size_t> side;
    for (std::size_t child = 0; child < 2 && !side; ++child) {
      auto const outputs = outputs_of(*join.children[child]);
      if (std::ranges::all_of(argument_columns,
                              [&](auto const& read) { return outputs.contains(read); })) {
        side = child;
      }
    }
    if (!side) { break; }
    auto forwarded = forwarded_columns(join, *side);
    if (!forwarded) { return std::nullopt; }
    result.path.push_back({&join, *side, std::move(*forwarded)});
    node = join.children[*side].get();
  }

  // The lowest join whose other side supplies a variable-width group column. Below it each input
  // row stands alone; from it upwards each carries that payload.
  std::optional<std::size_t> payload;
  for (std::size_t i = result.path.size(); i-- > 0 && !payload;) {
    auto const& step  = result.path[i];
    auto const others = outputs_of(*step.join->children[1 - step.side]);
    if (std::ranges::any_of(result.groups, [&](auto const& group) {
          return !duckdb::TypeIsConstantSize(group->GetReturnType().InternalType()) &&
                 others.contains(binding_of(*group));
        })) {
      payload = i;
    }
  }
  if (!payload) { return std::nullopt; }
  result.path.erase(result.path.begin() + static_cast<std::ptrdiff_t>(*payload) + 1,
                    result.path.end());
  auto const& under = result.path.back();
  auto& source      = *under.join->children[under.side];

  if (already_grouped(source)) { return decline("its input is already grouped"); }

  // The pushed aggregate groups by every column of its input that is read above it: by the joins
  // on the way up and by the aggregate's groups. The join it goes under comes first, so its keys
  // lead.
  auto const source_columns = source.GetColumnBindings();
  if (source_columns.size() != source.types.size()) { return std::nullopt; }
  auto const add_keys = [&](std::vector<ColumnBinding> const& read) {
    for (auto const& binding : read) {
      auto const found = std::ranges::find(source_columns, binding);
      if (found == source_columns.end() ||
          std::ranges::find(result.keys, binding) != result.keys.end()) {
        continue;
      }
      result.keys.push_back(binding);
      result.key_types.push_back(source.types[found - source_columns.begin()]);
    }
  };
  std::vector<ColumnBinding> read;
  append_columns_read(*under.join, read);
  add_keys(read);
  auto const join_keys = result.keys.size();
  read.clear();
  for (std::size_t i = 0; i + 1 < result.path.size(); ++i) {
    append_columns_read(*result.path[i].join, read);
  }
  for (auto const& group : result.groups) {
    read.push_back(binding_of(*group));
  }
  add_keys(read);

  if (result.keys.empty()) { return decline("the join reads no column of its input"); }
  if (!std::ranges::all_of(result.key_types, [](auto const& type) {
        return duckdb::TypeIsConstantSize(type.InternalType());
      })) {
    return decline("it would group by a variable-width column");
  }
  if (result.keys.size() >= aggregate.groups.size() ||
      result.keys.size() > join_keys + k_max_extra_keys) {
    return decline("it would group by too many columns");
  }
  // A join that computes its key sees one row of each group where it saw every row. That is the
  // same only when the rows of a group cannot differ in the columns the key is computed from.
  std::vector<ColumnBinding> computed_from;
  for (auto const& step : result.path) {
    append_columns_computed_from(*step.join, computed_from);
  }
  for (std::size_t k = 0; k < result.keys.size(); ++k) {
    if (!equal_means_identical(result.key_types[k]) &&
        std::ranges::find(computed_from, result.keys[k]) != computed_from.end()) {
      return decline("a join computes its key from a column whose equal values can differ");
    }
  }
  for (auto const& step : result.path) {
    if (!keeps_rows(context, *step.join->children[1 - step.side])) {
      return decline("a join above its input may discard the rows it would aggregate");
    }
  }

  result.lower_group_index     = binder.GenerateTableIndex();
  result.lower_aggregate_index = binder.GenerateTableIndex();
  for (std::size_t i = 0; i < aggregate.expressions.size(); ++i) {
    auto const& original = aggregate.expressions[i]->Cast<duckdb::BoundAggregateExpression>();
    auto lower           = original.Copy();
    lower->Cast<duckdb::BoundAggregateExpression>().GetChildrenMutable()[0] =
      std::move(arguments[i]);
    auto const type = lower->GetReturnType();
    auto upper      = bind_builtin_aggregate(
      context,
      combines[i],
      type,
      column(type, ColumnBinding(result.lower_aggregate_index, ProjectionIndex(i))));
    if (upper->GetReturnType() != original.GetReturnType()) {
      return decline("combining its results would change their type");
    }
    upper->SetAlias(original.GetAlias());
    result.lower.push_back(std::move(lower));
    result.upper.push_back(std::move(upper));
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Rewriting
// ---------------------------------------------------------------------------------------------

/// Insert the pushed aggregate and make the original combine its results. Cannot decline: a
/// failure here throws, and the caller plans without the rewrite.
void apply(pushdown& push)
{
  auto& aggregate         = *push.aggregate;
  auto const types_before = aggregate.types;
  auto const aggregates   = push.lower.size();
  auto& under             = push.path.back();
  auto& source            = under.join->children[under.side];
  auto const input_rows   = source->estimated_cardinality;

  auto lower = duckdb::make_uniq<duckdb::LogicalAggregate>(
    push.lower_group_index, push.lower_aggregate_index, std::move(push.lower));
  duckdb::GroupingSet every_key;
  duckdb::column_binding_map_t<ColumnBinding> moved;  ///< Each key to the pushed group column
  for (std::size_t k = 0; k < push.keys.size(); ++k) {
    lower->groups.push_back(column(push.key_types[k], push.keys[k]));
    every_key.insert(ProjectionIndex(k));
    moved.emplace(push.keys[k], ColumnBinding(push.lower_group_index, ProjectionIndex(k)));
  }
  lower->grouping_sets.push_back(std::move(every_key));
  lower->group_stats.resize(push.keys.size());
  lower->distinct_validity = duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES;
  // An upper bound: DuckDB has no estimate of the number of groups.
  if (source->has_estimated_cardinality) { lower->SetEstimatedCardinality(input_rows); }
  lower->children.push_back(std::move(source));
  source = std::move(lower);

  // Everything above that read a key reads the pushed aggregate's group column.
  auto const repoint = [&](unique_ptr<Expression>& expression) {
    duckdb::ExpressionIterator::VisitExpressionClassMutable(
      expression, ExpressionClass::BOUND_COLUMN_REF, [&](unique_ptr<Expression>& ref) {
        auto const found = moved.find(binding_of(*ref));
        if (found != moved.end()) {
          ref->Cast<duckdb::BoundColumnRefExpression>().BindingMutable() = found->second;
        }
      });
  };
  for (auto& step : push.path) {
    for (auto& condition : step.join->conditions) {
      if (condition.IsComparison()) {
        repoint(condition.LeftReference());
        repoint(condition.RightReference());
      } else {
        repoint(condition.JoinExpressionReference());
      }
    }
  }
  for (std::size_t g = 0; g < push.groups.size(); ++g) {
    repoint(push.groups[g]);
    aggregate.groups[g] = std::move(push.groups[g]);
  }
  aggregate.expressions = std::move(push.upper);
  if (push.input != &aggregate.children[0]) {
    // The projections under the aggregate are inlined into it.
    auto input            = std::move(*push.input);
    aggregate.children[0] = std::move(input);
  }

  // Each join on the way up forwarded some of its input's columns. It now forwards the same ones
  // under their new names, without the columns only the arguments read, plus the pushed results.
  for (std::size_t i = push.path.size(); i-- > 0;) {
    auto& step = push.path[i];
    auto& map  = step.side == 0 ? step.join->left_projection_map : step.join->right_projection_map;
    auto const now = step.join->children[step.side]->GetColumnBindings();
    std::vector<ColumnBinding> wanted;
    for (auto const& binding : step.forwarded) {
      if (auto const found = moved.find(binding); found != moved.end()) {
        wanted.push_back(found->second);
      } else if (std::ranges::find(now, binding) != now.end()) {
        wanted.push_back(binding);
      }
    }
    for (std::size_t a = 0; a < aggregates; ++a) {
      wanted.emplace_back(push.lower_aggregate_index, ProjectionIndex(a));
    }
    map.clear();
    bool every_column = wanted.size() == now.size();
    for (std::size_t w = 0; w < wanted.size(); ++w) {
      auto const found = std::ranges::find(now, wanted[w]);
      if (found == now.end()) {
        throw duckdb::InternalException(
          "partial aggregate pushdown: a join no longer outputs a column read above it");
      }
      auto const index = static_cast<duckdb::idx_t>(found - now.begin());
      every_column     = every_column && index == w;
      map.emplace_back(index);
    }
    if (every_column) { map.clear(); }
    // Its recorded scan targets and key statistics describe the join's old input.
    step.join->filter_pushdown.reset();
  }

  aggregate.ResolveOperatorTypes();
  if (aggregate.types != types_before) {
    throw duckdb::InternalException("partial aggregate pushdown changed the aggregate's types");
  }
  duckdb::ColumnBindingResolver verifier(true);
  verifier.VisitOperator(aggregate);

  SIRIUS_LOG_INFO(
    "[sirius_plan_partial_aggregate_pushdown] Aggregating {} value(s) by {} column(s) below {} "
    "join(s) that add group-by payload (input est {} rows)",
    aggregates,
    push.keys.size(),
    push.path.size(),
    input_rows);
}

/// Children first, so an aggregate is considered after everything below it, and the aggregates
/// this pass inserts are not revisited.
void visit(duckdb::ClientContext& context,
           duckdb::Binder& binder,
           LogicalOperator& op,
           std::size_t& pushed)
{
  for (auto& child : op.children) {
    visit(context, binder, *child, pushed);
  }
  if (op.type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { return; }
  auto push = match(context, binder, op.Cast<duckdb::LogicalAggregate>());
  if (!push) { return; }
  // ---- Nothing below declines. ----
  apply(*push);
  ++pushed;
}

}  // namespace

std::size_t push_partial_aggregates_below_joins(duckdb::ClientContext& context,
                                                duckdb::Binder& binder,
                                                duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  plan->ResolveOperatorTypes();
  std::size_t pushed = 0;
  visit(context, binder, *plan, pushed);
  return pushed;
}

}  // namespace sirius::planner
