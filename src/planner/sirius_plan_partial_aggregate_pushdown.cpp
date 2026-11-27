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

#include "duckdb/common/enums/function_errors.hpp"
#include "duckdb/common/enums/tuple_data_layout_enums.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/relation_statistics/relation_statistics_helper.hpp"
#include "duckdb/parser/group_by_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "log/logging.hpp"
#include "planner/aggregate_binding.hpp"
#include "planner/dynamic_filter/build_key_domain.hpp"
#include "planner/proven_unique_scan_columns.hpp"
#include "planner/sirius_plan_flat_tree.hpp"
#include "transparent/parquet_footer_statistics.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
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
using flat::binding_of;
using flat::column;
using flat::is_column;

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
/// columns as an expression over the first operator below them, which `input` holds. A volatile
/// projection cannot be folded, since inlining it would evaluate it once per reader: `is_volatile`
/// is then set and `columns` is incomplete. Nothing is folded when `input` is not a join, since
/// the rule then leaves the aggregate alone.
struct aggregate_input {
  unique_ptr<LogicalOperator>* input = nullptr;
  inlined_columns columns;
  bool is_volatile = false;
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
  result.input = slot;
  if ((*slot)->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return result; }
  for (auto projection = projections.rbegin(); projection != projections.rend(); ++projection) {
    auto const& expressions = (*projection)->expressions;
    for (std::size_t i = 0; i < expressions.size(); ++i) {
      if (expressions[i]->IsVolatile()) {
        result.is_volatile = true;
        return result;
      }
      result.columns[ColumnBinding((*projection)->table_index, ProjectionIndex(i))] =
        inline_columns(*expressions[i], result.columns);
    }
  }
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

/// Why the results of a SUM over @p argument are not added up, or null when they are. The sum of
/// sums of integers of up to 64 bits, or of DECIMALs of up to 18 digits, is the same total in the
/// same type, and the GPU never refuses it. It can refuse a 128-bit sum, and below the join one
/// huge value on a key that matches nothing would then send to the CPU a query it answers today.
///
/// A floating-point sum depends on the order of addition, which the GPU does not fix. Adding per
/// key first is one more order only while each group above takes its rows from a single key:
/// were several keys to share a group, each could overflow alone where their sum would not.
char const* sum_blocker(LogicalType const& argument)
{
  switch (argument.id()) {
    case duckdb::LogicalTypeId::DECIMAL:
      return argument.InternalType() == duckdb::PhysicalType::INT128 ? "sum:decimal128" : nullptr;
    case duckdb::LogicalTypeId::TINYINT:
    case duckdb::LogicalTypeId::SMALLINT:
    case duckdb::LogicalTypeId::INTEGER:
    case duckdb::LogicalTypeId::BIGINT:
    case duckdb::LogicalTypeId::FLOAT:
    case duckdb::LogicalTypeId::DOUBLE: return nullptr;
    case duckdb::LogicalTypeId::HUGEINT: return "sum:hugeint";
    default: return "sum:other";
  }
}

bool is_arithmetic(duckdb::BoundFunctionExpression const& function)
{
  auto const& name = function.Function().GetName().GetIdentifierName();
  return name == "+" || name == "-" || name == "*" || name == "add" || name == "subtract" ||
         name == "multiply";
}

/// Whether @p expression is safe to evaluate on rows a join above would have removed: it cannot
/// throw, or what can throw in it is only `+`, `-` and `*`, under CASE, AND, OR and functions that
/// cannot fail themselves, such as comparisons. Accepting that arithmetic is Sirius's own rule:
/// it can still overflow on a row the unrewritten plan never evaluates it on, and DuckDB's partial
/// aggregate pushdown declines it unless column statistics rule the overflow out.
bool fails_only_through_arithmetic(Expression const& expression)
{
  if (!expression.CanThrow()) { return true; }
  switch (expression.GetExpressionClass()) {
    case ExpressionClass::BOUND_FUNCTION: {
      auto const& function = expression.Cast<duckdb::BoundFunctionExpression>();
      if (!is_arithmetic(function) &&
          function.Function().GetErrorMode() == duckdb::FunctionErrors::CAN_THROW_RUNTIME_ERROR) {
        return false;
      }
      // A lambda's body is not among the children that are checked below.
      if (std::ranges::any_of(function.GetChildren(), [](auto const& child) {
            return child->GetExpressionClass() == ExpressionClass::BOUND_LAMBDA;
          })) {
        return false;
      }
      break;
    }
    case ExpressionClass::BOUND_CASE:
    case ExpressionClass::BOUND_CONJUNCTION: break;
    default: return false;
  }
  bool safe = true;
  duckdb::ExpressionIterator::EnumerateChildren(expression, [&](Expression const& child) {
    safe = safe && fails_only_through_arithmetic(child);
  });
  return safe;
}

// ---------------------------------------------------------------------------------------------
// Whether the push pays off
// ---------------------------------------------------------------------------------------------

/// What the sides joined to the aggregated input are expected to do to its rows.
struct other_sides {
  duckdb::ClientContext& context;
  double min_kept_share;
  /// Every scan is estimated to keep at least that share of its unfiltered rows, counting a
  /// filter directly above it.
  bool keep_rows = true;
  /// The smallest such share.
  std::optional<double> least_share;
};

/// Add to @p sides what a join against @p op is expected to do to the rows of its other side, and
/// return whether @p op is made of scans with a row estimate, joined to each other: anything
/// else, a filter above a scan included, is not trusted to keep rows. A join against a filtered
/// side discards rows, and aggregating them first would be wasted. Scan estimates are compared
/// with the unfiltered row count; a filter the join itself implies is added after DuckDB
/// estimates the scan, so it does not count. @p kept_above is the row estimate of a filter
/// directly above @p op, when there is one.
bool judge_other_side(other_sides& sides,
                      LogicalOperator const& op,
                      std::optional<double> kept_above = std::nullopt)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET: {
      auto const& get = op.Cast<duckdb::LogicalGet>();
      if (!get.children.empty() || !get.has_estimated_cardinality) { return false; }
      auto rows = transparent::parquet_scan_row_count(get);
      if (!rows) { rows = duckdb_base_table_cardinality{sides.context}(get); }
      if (!rows) { return false; }
      auto const scanned = static_cast<double>(get.estimated_cardinality);
      auto const total   = static_cast<double>(*rows);
      auto const kept    = kept_above ? std::min(scanned, *kept_above) : scanned;
      if (kept < sides.min_kept_share * total) { sides.keep_rows = false; }
      auto const share = total > 0 ? kept / total : 1.0;
      if (!sides.least_share || share < *sides.least_share) { sides.least_share = share; }
      return true;
    }
    case LogicalOperatorType::LOGICAL_PROJECTION:
      return op.children.size() == 1 && judge_other_side(sides, *op.children[0], kept_above);
    case LogicalOperatorType::LOGICAL_FILTER:
      // Not trusted, but what it is estimated to keep of the scan below it is still reported.
      if (op.children.size() == 1) {
        if (op.has_estimated_cardinality) {
          auto const kept = static_cast<double>(op.estimated_cardinality);
          kept_above      = kept_above ? std::min(kept, *kept_above) : kept;
        }
        judge_other_side(sides, *op.children[0], kept_above);
      }
      return false;
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
      if (op.Cast<duckdb::LogicalComparisonJoin>().join_type != duckdb::JoinType::INNER) {
        return false;
      }
      // Every child is judged, a trusted one after an untrusted one too: each adds its share.
      bool known = true;
      for (auto const& child : op.children) {
        known = judge_other_side(sides, *child) && known;
      }
      return known;
    }
    default: return false;
  }
}

/// @p op's row estimate, or the largest of its children's when it has none. Only reads:
/// `LogicalOperator::EstimateCardinality` would store the estimate it computes.
std::optional<duckdb::idx_t> estimated_rows(LogicalOperator const& op)
{
  if (op.has_estimated_cardinality) { return op.estimated_cardinality; }
  std::optional<duckdb::idx_t> rows;
  for (auto const& child : op.children) {
    rows = std::max(rows, estimated_rows(*child));
  }
  return rows;
}

/// Whether a filter pushed into a scan leaves the column it tests a single value: the column
/// equals a constant, alone or among conditions that must all hold.
bool keeps_one_value(Expression const& condition)
{
  if (condition.GetExpressionType() == duckdb::ExpressionType::CONJUNCTION_AND) {
    auto const& all = condition.Cast<duckdb::BoundConjunctionExpression>().GetChildren();
    return std::ranges::any_of(all, [](auto const& child) { return keeps_one_value(*child); });
  }
  if (!duckdb::BoundComparisonExpression::IsComparison(condition)) { return false; }
  auto comparison     = duckdb::ExpressionType::INVALID;
  auto const constant = duckdb::ExpressionFilter::TryGetColumnConstantComparison(
    condition.Cast<duckdb::BoundFunctionExpression>(), comparison);
  return constant && comparison == duckdb::ExpressionType::COMPARE_EQUAL;
}

/// A scan and one of its output columns.
struct scan_column {
  duckdb::LogicalGet* get;
  ColumnBinding binding;
};

/// The scan column that the column @p binding of @p op's output passes on unchanged, or nullopt
/// where an operator on the way computes it. With @p rows_only the way may hold only filters and
/// projections, which keep the scan's rows apart; otherwise any operator that forwards the column
/// from a child will do.
std::optional<scan_column> scan_column_behind(LogicalOperator& op,
                                              ColumnBinding binding,
                                              bool rows_only)
{
  auto* node = &op;
  while (node->type != LogicalOperatorType::LOGICAL_GET) {
    auto* below = node->children.size() == 1 ? node->children[0].get() : nullptr;
    if (node->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      auto const& projection = node->Cast<duckdb::LogicalProjection>();
      auto const index       = binding.column_index.GetIndex();
      if (!below || binding.table_index != projection.table_index ||
          index >= projection.expressions.size() || !is_column(*projection.expressions[index])) {
        return std::nullopt;
      }
      binding = binding_of(*projection.expressions[index]);
    } else if (rows_only) {
      if (!below || node->type != LogicalOperatorType::LOGICAL_FILTER) { return std::nullopt; }
    } else {
      auto const from = std::ranges::find_if(
        node->children, [&](auto const& child) { return outputs_of(*child).contains(binding); });
      if (from == node->children.end()) { return std::nullopt; }
      below = from->get();
    }
    node = below;
  }
  auto& get = node->Cast<duckdb::LogicalGet>();
  if (binding.table_index != get.table_index) { return std::nullopt; }
  return scan_column{&get, binding};
}

/// The most distinct values the column @p binding of @p op's output can hold, or nullopt when
/// nothing limits them. The limit is that of the scan column it comes from: a single value under
/// an equality filter; otherwise, for a native table, the distinct count DuckDB derives from its
/// statistics (the column's sketch, or an integer's range), and for a Parquet scan the integer
/// range in its footers; and never more than the rows the scan is estimated to return. Operators
/// between the scan and @p op drop values but add none, bar the NULL of an outer join.
///
/// DuckDB's statistics callback answers nothing for a Parquet scan here: the rule works on a copy
/// of the plan, and the copy of a scan's bind data drops the reader that callback reads through.
/// The ranges are the ones the footer statistics pass kept with the scan's row count and the copy
/// was given, so a scan that pass did not cover (one over a footer read limit, or planned with
/// footer statistics off) has none.
std::optional<double> most_distinct_values(duckdb::ClientContext& context,
                                           LogicalOperator& op,
                                           ColumnBinding const& binding)
{
  auto const behind = scan_column_behind(op, binding, /*rows_only=*/false);
  if (!behind || behind->binding.column_index.GetIndex() >= behind->get->GetColumnIds().size()) {
    return std::nullopt;
  }
  auto& get          = *behind->get;
  auto const& column = get.GetColumnIndex(behind->binding);
  if (column.IsRowIdColumn() || column.IsVirtualColumn()) { return std::nullopt; }
  try {
    auto const filter = get.table_filters.TryGetFilterByColumnIndex(behind->binding.column_index);
    if (filter && filter->filter_type == duckdb::TableFilterType::EXPRESSION_FILTER &&
        keeps_one_value(*filter->Cast<duckdb::ExpressionFilter>().expr)) {
      return 1.0;
    }
    std::optional<double> most;
    if (get.GetTable()) {
      // Where the statistics give no count DuckDB falls back to the scan's rows, which shows
      // nothing about how often a value repeats.
      auto const scan  = duckdb::RelationStatisticsHelper::ExtractGetStats(get, context);
      auto const found = scan.GetColumnStats(behind->binding);
      if (found && found->total_domain.distinct_count > 0 &&
          found->total_domain.source != duckdb::DistinctCountSource::CARDINALITY) {
        most = static_cast<double>(found->total_domain.distinct_count);
      }
    } else if (auto const span =
                 transparent::parquet_scan_value_span(get, column.GetPrimaryIndex())) {
      most = static_cast<double>(*span);
    }
    if (most && get.has_estimated_cardinality) {
      most = std::min(*most, std::max(1.0, static_cast<double>(get.estimated_cardinality)));
    }
    return most;
  } catch (...) {
    // A scan whose statistics cannot be read says nothing.
    return std::nullopt;
  }
}

/// The columns of @p get that are proven to hold no value twice, and no NULL: those its pinned
/// copy found distinct, or a native table's PRIMARY KEY.
std::vector<ColumnBinding> unique_columns(duckdb::ClientContext& context, duckdb::LogicalGet& get)
{
  std::vector<ColumnBinding> unique;
  try {
    auto const outputs = get.GetColumnBindings();
    for (auto const position : proven_unique_scan_outputs(context, get)) {
      if (position < outputs.size()) { unique.push_back(outputs[position]); }
    }
  } catch (...) {
    // A scan whose facts cannot be read proves nothing.
    unique.clear();
  }
  return unique;
}

/// The scan that @p op's rows come from through filters and projections alone, with the scan
/// column behind each of @p columns that it passes on unchanged; a column it computes is left
/// out. Filters and projections lead to one scan, so every column that reaches a scan reaches
/// that one.
struct scanned_columns {
  duckdb::LogicalGet* get = nullptr;
  std::vector<std::pair<ColumnBinding, ColumnBinding>> columns;  ///< Of @p op, and of the scan
};
scanned_columns scan_columns_behind(LogicalOperator& op, std::vector<ColumnBinding> const& columns)
{
  scanned_columns result;
  for (auto const& column : columns) {
    if (auto const behind = scan_column_behind(op, column, /*rows_only=*/true)) {
      result.get = behind->get;
      result.columns.emplace_back(column, behind->binding);
    }
  }
  return result;
}

/// Whether one of @p keys is proven to hold no value twice in @p source (see unique_columns), so
/// that an aggregate grouped by them would have a group for every row and combine nothing. The
/// proof holds only for the scan's own rows, under filters and projections, since a join below
/// @p source could repeat them.
bool key_never_repeats(duckdb::ClientContext& context,
                       LogicalOperator& source,
                       std::vector<ColumnBinding> const& keys)
{
  auto const scanned = scan_columns_behind(source, keys);
  if (!scanned.get) { return false; }
  auto const unique = unique_columns(context, *scanned.get);
  return std::ranges::any_of(scanned.columns, [&](auto const& column) {
    return std::ranges::find(unique, column.second) != unique.end();
  });
}

/// Adds to @p fixed the columns of @p side that one of its columns in @p fixed determines: where
/// @p side is a scan under filters and projections, and such a column is proven to hold no value
/// twice in it, a value of that column picks out one row, and with it a value of every other
/// column the scan passes on.
void add_columns_of_unique_row(duckdb::ClientContext& context,
                               LogicalOperator& side,
                               duckdb::column_binding_set_t& fixed)
{
  auto const scanned  = scan_columns_behind(side, side.GetColumnBindings());
  auto const is_fixed = [&](auto const& column) { return fixed.contains(column.first); };
  // The scan's facts cost a look-up of its files and their pin, so they are asked for only where
  // they could add a column: where some column of the side is fixed and some is not.
  if (!scanned.get || std::ranges::none_of(scanned.columns, is_fixed) ||
      std::ranges::all_of(scanned.columns, is_fixed)) {
    return;
  }
  auto const unique = unique_columns(context, *scanned.get);
  if (std::ranges::none_of(scanned.columns, [&](auto const& column) {
        return is_fixed(column) && std::ranges::find(unique, column.second) != unique.end();
      })) {
    return;
  }
  for (auto const& column : scanned.columns) {
    fixed.insert(column.first);
  }
}

/// The rows of @p source that one group of @p keys is estimated to combine, at least: its row
/// estimate over the most groups the keys can form. 1 when the statistics cannot show that keys
/// repeat, which includes a key nothing limits.
double least_rows_per_key(duckdb::ClientContext& context,
                          LogicalOperator& source,
                          std::vector<ColumnBinding> const& keys)
{
  auto const estimate = estimated_rows(source);
  if (!estimate || *estimate == 0 || keys.empty()) { return 1.0; }
  auto const rows = static_cast<double>(*estimate);
  auto groups     = 1.0;
  for (auto const& key : keys) {
    auto const most = most_distinct_values(context, source, key);
    if (!most) { return 1.0; }
    groups *= std::max(*most, 1.0);
    if (groups >= rows) { return 1.0; }
  }
  return rows / groups;
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
  /// An outer join that keeps every row of that child, and pads the other child's columns with
  /// NULL where a row finds no match.
  bool preserves = false;
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
  /// Each aggregate's argument over the joins' columns (null when it has none), and the aggregate
  /// that combines its results (null when none does).
  duckdb::vector<unique_ptr<Expression>> arguments;
  std::vector<char const*> combines;
  duckdb::vector<unique_ptr<Expression>> lower;  ///< The aggregates, over the pushed input
  duckdb::vector<unique_ptr<Expression>> upper;  ///< The aggregates combining those
  TableIndex lower_group_index;
  TableIndex lower_aggregate_index;
  /// The rows a group of the keys is estimated to combine, at least; unset until it is worked out.
  std::optional<double> rows_per_key;
};

/// Why an aggregate over a join is left alone: each reason once, in the order found.
struct blockers {
  duckdb::vector<std::string> reasons;
  std::optional<double> least_kept_share;  ///< Of the scans joined to the aggregated input

  void add(std::string_view reason)
  {
    if (std::ranges::find(reasons, reason) == reasons.end()) { reasons.emplace_back(reason); }
  }
};

/// Work out how @p aggregate, whose input below its projections is the join in @p folded, can be
/// computed in two steps around a join that only adds payload, and add to @p why every reason it
/// cannot be. Changes nothing. The analysis goes on past a reason, so that the reasons are
/// complete, and returns early only where nothing further can be worked out.
void analyze(duckdb::ClientContext& context,
             duckdb::LogicalAggregate& aggregate,
             aggregate_input const& folded,
             partial_aggregate_pushdown_options const& options,
             pushdown& result,
             blockers& why)
{
  if (aggregate.groups.empty()) { return why.add("shape:ungrouped"); }
  if (aggregate.expressions.empty()) { return why.add("shape:no-aggregates"); }
  if (!aggregate.grouping_functions.empty() || !groups_once(aggregate)) {
    return why.add("shape:grouping-sets");
  }
  if (folded.is_volatile) { return why.add("shape:volatile-projection"); }

  result.aggregate    = &aggregate;
  result.input        = folded.input;
  bool groups_payload = false;
  for (auto const& group : aggregate.groups) {
    auto value = inline_columns(*group, folded.columns);
    if (!is_column(*value)) { return why.add("shape:group-expression"); }
    groups_payload =
      groups_payload || !duckdb::TypeIsConstantSize(value->GetReturnType().InternalType());
    result.groups.push_back(std::move(value));
  }
  if (!groups_payload) { return why.add("path:no-payload"); }

  // Each aggregate's argument over the joins' columns, and what combines its results.
  std::vector<ColumnBinding> argument_columns;
  bool sums_floats = false;
  for (auto const& expression : aggregate.expressions) {
    if (expression->GetExpressionClass() != ExpressionClass::BOUND_AGGREGATE) {
      return why.add("shape:not-an-aggregate");
    }
    auto const& original = expression->Cast<duckdb::BoundAggregateExpression>();
    auto const& name     = original.Function().GetName().GetIdentifierName();
    auto const* combine  = combining_aggregate(name);
    auto const unary     = original.GetChildren().size() == 1;
    if (!combine || !unary) { why.add("aggregate:" + name); }
    if (original.IsDistinct()) { why.add("aggregate:distinct"); }
    if (original.GetFilter()) {
      why.add("aggregate:filter");
      append_columns_read(*inline_columns(*original.GetFilter(), folded.columns), argument_columns);
    }
    if (original.GetOrderBys()) { why.add("aggregate:order-by"); }

    auto const columns_before = argument_columns.size();
    unique_ptr<Expression> argument;
    for (auto const& child : original.GetChildren()) {
      auto value = inline_columns(*child, folded.columns);
      if (value->IsVolatile()) {
        why.add("argument:volatile");
      } else if (!fails_only_through_arithmetic(*value)) {
        why.add("argument:fallible");
      }
      append_columns_read(*value, argument_columns);
      if (unary) { argument = std::move(value); }
    }
    if (unary) {
      // AVG adds up its argument too, so it reports what stands in the way of that sum.
      if (name == "avg" || (combine && std::string_view{combine} == "sum")) {
        if (auto const* blocker = sum_blocker(argument->GetReturnType())) { why.add(blocker); }
        sums_floats = sums_floats || argument->GetReturnType().IsFloating();
      }
      if (combine && argument_columns.size() == columns_before) { why.add("argument:constant"); }
    }
    result.arguments.push_back(std::move(argument));
    result.combines.push_back(combine);
  }

  // Down through joins, on the side that holds every argument: inner joins, and outer joins that
  // keep every row of that side. Such an outer join repeats or pads the rows of the other side
  // only, so a key's rows still reach the aggregate above together, as one row per match.
  auto* node               = folded.input->get();
  char const* stopped_here = nullptr;
  while (node->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto& join = node->Cast<duckdb::LogicalComparisonJoin>();
    if (join.children.size() != 2 || join.conditions.empty()) {
      stopped_here = "join:no-condition";
      break;
    }
    std::optional<std::size_t> side;
    if (argument_columns.empty()) {
      // No aggregate reads a column (COUNT(*)): the side with more rows is the one to aggregate.
      auto const left  = estimated_rows(*join.children[0]);
      auto const right = estimated_rows(*join.children[1]);
      if (!left || !right || *left == *right) {
        stopped_here = "side:unknown";
        break;
      }
      side = *left > *right ? 0 : 1;
    } else {
      for (std::size_t child = 0; child < 2 && !side; ++child) {
        auto const outputs = outputs_of(*join.children[child]);
        if (std::ranges::all_of(argument_columns,
                                [&](auto const& read) { return outputs.contains(read); })) {
          side = child;
        }
      }
      if (!side) {
        stopped_here = "arguments:both-sides";
        break;
      }
    }
    bool preserves = false;
    switch (join.join_type) {
      case duckdb::JoinType::INNER: break;
      case duckdb::JoinType::LEFT:
      case duckdb::JoinType::RIGHT:
        preserves    = *side == (join.join_type == duckdb::JoinType::LEFT ? 0U : 1U);
        stopped_here = preserves ? nullptr : "join:outer-padded";
        break;
      case duckdb::JoinType::OUTER: stopped_here = "join:full"; break;
      case duckdb::JoinType::SEMI:
      case duckdb::JoinType::ANTI:
      case duckdb::JoinType::RIGHT_SEMI:
      case duckdb::JoinType::RIGHT_ANTI: stopped_here = "join:semi-anti"; break;
      case duckdb::JoinType::MARK: stopped_here = "join:mark"; break;
      case duckdb::JoinType::SINGLE: stopped_here = "join:single"; break;
      default: stopped_here = "join:other"; break;
    }
    if (stopped_here) { break; }
    auto forwarded = forwarded_columns(join, *side);
    if (!forwarded) { return why.add("plan:projection-map"); }
    result.path.push_back({&join, *side, std::move(*forwarded), preserves});
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
  if (!payload) { return why.add(stopped_here ? stopped_here : "path:no-payload"); }
  result.path.erase(result.path.begin() + static_cast<std::ptrdiff_t>(*payload) + 1,
                    result.path.end());
  auto const& under = result.path.back();
  auto& source      = *under.join->children[under.side];

  if (already_grouped(source)) { why.add("source:grouped"); }

  // The pushed aggregate groups by every column of its input that is read above it: by the joins
  // on the way up and by the aggregate's groups. The join it goes under comes first, so its keys
  // lead.
  auto const source_columns = source.GetColumnBindings();
  if (source_columns.size() != source.types.size()) { return why.add("plan:unresolved-source"); }
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

  // It would group by nothing, by a variable-width column, by as many columns as the aggregate
  // above it, or by too many columns beyond the join's.
  if (result.keys.empty()) { why.add("keys:none"); }
  if (!std::ranges::all_of(result.key_types, [](auto const& type) {
        return duckdb::TypeIsConstantSize(type.InternalType());
      })) {
    why.add("keys:variable-width");
  }
  if (result.keys.size() >= aggregate.groups.size()) { why.add("keys:not-fewer-than-groups"); }
  if (result.keys.size() - join_keys > options.max_extra_keys) { why.add("keys:too-many-extra"); }
  // It pays when the aggregate above groups as finely as it does: when the columns it groups by
  // fix the value of each of its keys. A key it groups by is fixed. So is a column that an inner
  // join on the way up equates with a fixed one, and every column of a joined table's row once a
  // fixed column picks that row out, being proven unique in the table: TPC-DS q11 groups by
  // c_customer_id, which fixes c_customer_sk and through the join ss_customer_sk. Every key
  // counts, a join's further up included. An outer join's equality does not: it pads the other
  // side's column with NULL where a key finds no match, so grouping by that column folds those
  // keys into one group.
  duckdb::column_binding_set_t grouped;
  for (auto const& group : result.groups) {
    grouped.insert(binding_of(*group));
  }
  // Each pass can fix columns that fix more, so repeat until one adds none.
  for (std::size_t before = 0; before != grouped.size();) {
    before = grouped.size();
    for (auto const& step : result.path) {
      if (step.preserves) { continue; }
      add_columns_of_unique_row(context, *step.join->children[1 - step.side], grouped);
      for (auto const& condition : step.join->conditions) {
        if (!condition.IsComparison() ||
            (condition.GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL &&
             condition.GetComparisonType() != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) ||
            !is_column(condition.GetLHS()) || !is_column(condition.GetRHS())) {
          continue;
        }
        auto const left  = binding_of(condition.GetLHS());
        auto const right = binding_of(condition.GetRHS());
        if (grouped.contains(right)) { grouped.insert(left); }
        if (grouped.contains(left)) { grouped.insert(right); }
      }
    }
  }
  // Failing that, it pays when the scan statistics show that its keys repeat often enough. A
  // floating-point sum is left to the grouping alone (see sum_blocker).
  result.rows_per_key = least_rows_per_key(context, source, result.keys);
  auto const repeat   = !sums_floats && options.min_rows_per_key > 0 &&
                      *result.rows_per_key >= options.min_rows_per_key;
  if (!repeat &&
      !std::ranges::all_of(result.keys, [&](auto const& key) { return grouped.contains(key); })) {
    why.add("keys:not-grouped");
  }
  if (key_never_repeats(context, source, result.keys)) { why.add("keys:unique"); }
  // A join that computes its key sees one row of each group where it saw every row. That is the
  // same only when the rows of a group cannot differ in the columns the key is computed from.
  std::vector<ColumnBinding> computed_from;
  for (auto const& step : result.path) {
    append_columns_computed_from(*step.join, computed_from);
  }
  for (std::size_t k = 0; k < result.keys.size(); ++k) {
    if (!equal_means_identical(result.key_types[k]) &&
        std::ranges::find(computed_from, result.keys[k]) != computed_from.end()) {
      why.add("key:computed-inexact");
    }
  }
  // A join above its input may discard the rows it would aggregate. One that keeps every row of
  // the aggregated side cannot.
  if (options.min_kept_share > 0) {
    other_sides sides{context, options.min_kept_share};
    bool known = true;
    for (auto const& step : result.path) {
      if (!step.preserves) {
        known = judge_other_side(sides, *step.join->children[1 - step.side]) && known;
      }
    }
    if (!known) { why.add("other-side:opaque"); }
    if (!sides.keep_rows) { why.add("other-side:filtered"); }
    why.least_kept_share = sides.least_share;
  }
}

/// Bind the aggregates @p result pushes and the ones that combine them. Adds a reason to @p why
/// when combining would change a result's type.
void bind(duckdb::ClientContext& context, duckdb::Binder& binder, pushdown& result, blockers& why)
{
  auto const& aggregate        = *result.aggregate;
  result.lower_group_index     = binder.GenerateTableIndex();
  result.lower_aggregate_index = binder.GenerateTableIndex();
  for (std::size_t i = 0; i < aggregate.expressions.size(); ++i) {
    auto const& original = aggregate.expressions[i]->Cast<duckdb::BoundAggregateExpression>();
    auto lower           = original.Copy();
    lower->Cast<duckdb::BoundAggregateExpression>().GetChildrenMutable()[0] =
      std::move(result.arguments[i]);
    auto const type = lower->GetReturnType();
    auto upper      = bind_builtin_aggregate(
      context,
      result.combines[i],
      type,
      column(type, ColumnBinding(result.lower_aggregate_index, ProjectionIndex(i))));
    if (upper->GetReturnType() != original.GetReturnType()) { return why.add("combine:type"); }
    upper->SetAlias(original.GetAlias());
    result.lower.push_back(std::move(lower));
    result.upper.push_back(std::move(upper));
  }
}

/// How to compute @p aggregate in two steps around a join that only adds payload, or nullopt
/// when it is left alone. An aggregate over a join that is left alone adds its reasons to
/// @p declined; any other aggregate is not the rule's concern and adds nothing. Changes nothing.
std::optional<pushdown> match(duckdb::ClientContext& context,
                              duckdb::Binder& binder,
                              duckdb::LogicalAggregate& aggregate,
                              partial_aggregate_pushdown_options const& options,
                              std::vector<std::string>& declined)
{
  if (aggregate.children.size() != 1) { return std::nullopt; }
  auto const folded = fold_projections(aggregate);
  if ((*folded.input)->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    return std::nullopt;
  }

  pushdown result;
  blockers why;
  analyze(context, aggregate, folded, options, result, why);
  if (why.reasons.empty()) { bind(context, binder, result, why); }
  if (why.reasons.empty()) { return result; }

  auto reasons = duckdb::StringUtil::Join(why.reasons, "; ");
  std::string measured;
  if (why.least_kept_share) {
    measured += std::format(" (least kept share {:.3f})", *why.least_kept_share);
  }
  if (result.rows_per_key) {
    measured += std::format(" (at least {:.2f} rows per key)", *result.rows_per_key);
  }
  SIRIUS_LOG_DEBUG(
    "[sirius_plan_partial_aggregate_pushdown] Not aggregating below a join that adds group-by "
    "payload: {}{}",
    reasons,
    measured);
  declined.push_back(std::move(reasons));
  return std::nullopt;
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
    "join(s) that add group-by payload (input est {} rows, at least {:.2f} per key)",
    aggregates,
    push.keys.size(),
    push.path.size(),
    input_rows,
    push.rows_per_key.value_or(1.0));
}

/// Children first, so an aggregate is considered after everything below it, and the aggregates
/// this pass inserts are not revisited.
void visit(duckdb::ClientContext& context,
           duckdb::Binder& binder,
           LogicalOperator& op,
           partial_aggregate_pushdown_options const& options,
           partial_aggregate_pushdown_result& result)
{
  for (auto& child : op.children) {
    visit(context, binder, *child, options, result);
  }
  if (op.type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { return; }
  auto push = match(context, binder, op.Cast<duckdb::LogicalAggregate>(), options, result.declined);
  if (!push) { return; }
  // ---- Nothing below declines. ----
  apply(*push);
  ++result.pushed;
}

}  // namespace

partial_aggregate_pushdown_result push_partial_aggregates_below_joins(
  duckdb::ClientContext& context,
  duckdb::Binder& binder,
  duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
  partial_aggregate_pushdown_options const& options)
{
  plan->ResolveOperatorTypes();
  partial_aggregate_pushdown_result result;
  visit(context, binder, *plan, options, result);
  return result;
}

}  // namespace sirius::planner
