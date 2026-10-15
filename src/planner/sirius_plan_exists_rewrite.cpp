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
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/aggregate/probe_accumulate_impl.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

using duckdb::LogicalOperatorType;

duckdb::unique_ptr<duckdb::Expression> ref(duckdb::LogicalType const& type, duckdb::idx_t index)
{
  return duckdb::make_uniq<duckdb::BoundReferenceExpression>(type, index);
}

bool contains_delim_get(duckdb::LogicalOperator const& op)
{
  if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET) { return true; }
  for (auto const& child : op.children) {
    if (contains_delim_get(*child)) { return true; }
  }
  return false;
}

/// The pieces of a correlated `EXISTS (... WHERE t.k = outer.k AND t.x <> outer.x)` that DuckDB
/// decorrelated into a DELIM_JOIN, everything addressed by position in the resolved plan.
struct exists_match {
  duckdb::LogicalComparisonJoin* delim = nullptr;  ///< The DELIM_JOIN being replaced
  std::size_t outer_child              = 0;        ///< Index of the outer rows among its children
  bool negated                         = false;    ///< NOT EXISTS
  duckdb::LogicalComparisonJoin* inner = nullptr;  ///< INNER join of T and the DELIM_GET
  std::size_t t_child                  = 0;        ///< Which child of `inner` is T
  std::size_t t_key                    = 0;        ///< Column of T equal to the delim key
  std::size_t t_value                  = 0;        ///< Column of T compared with `<>`
  std::size_t outer_key                = 0;        ///< Column of the outer rows for the key
  std::size_t outer_value              = 0;        ///< Column of the outer rows for the value
  duckdb::LogicalType key_type;
  duckdb::LogicalType value_type;
};

/// Recognise the decorrelated shape. Every check is structural; anything unexpected declines and
/// leaves DuckDB's plan alone.
std::optional<exists_match> match_exists(duckdb::LogicalOperator& node)
{
  using duckdb::ExpressionClass;
  using duckdb::ExpressionType;
  using duckdb::JoinType;

  if (node.type != LogicalOperatorType::LOGICAL_DELIM_JOIN || node.children.size() != 2) {
    return std::nullopt;
  }
  auto& delim = node.Cast<duckdb::LogicalComparisonJoin>();
  bool const semi_family =
    delim.join_type == JoinType::SEMI || delim.join_type == JoinType::RIGHT_SEMI;
  bool const anti_family =
    delim.join_type == JoinType::ANTI || delim.join_type == JoinType::RIGHT_ANTI;
  if (!semi_family && !anti_family) { return std::nullopt; }
  if (delim.predicate || delim.conditions.empty()) { return std::nullopt; }

  // The subquery side holds the DELIM_GET; the join emits the outer side's columns.
  std::size_t const sub_idx   = delim.delim_flipped ? 0 : 1;
  std::size_t const outer_idx = 1 - sub_idx;
  bool const emits_left = delim.join_type == JoinType::SEMI || delim.join_type == JoinType::ANTI;
  if (emits_left != (outer_idx == 0)) { return std::nullopt; }

  // Any number of projections may sit above the join; each must only reorder or drop columns.
  auto* sub = delim.children[sub_idx].get();
  std::vector<duckdb::LogicalProjection*> projections;  // outermost first
  while (sub->type == LogicalOperatorType::LOGICAL_PROJECTION) {
    projections.push_back(&sub->Cast<duckdb::LogicalProjection>());
    sub = projections.back()->children[0].get();
  }
  if (sub->type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN || sub->children.size() != 2) {
    return std::nullopt;
  }
  auto& inner = sub->Cast<duckdb::LogicalComparisonJoin>();
  if (inner.join_type != JoinType::INNER || inner.predicate || inner.conditions.empty()) {
    return std::nullopt;
  }

  // T joined to the DELIM_GET, in either order, with no other delim scan inside T.
  std::size_t d_child = 2;
  for (std::size_t c = 0; c < 2; ++c) {
    if (inner.children[c]->type == LogicalOperatorType::LOGICAL_DELIM_GET) { d_child = c; }
  }
  if (d_child == 2) { return std::nullopt; }
  std::size_t const t_child = 1 - d_child;
  if (contains_delim_get(*inner.children[t_child])) { return std::nullopt; }
  auto& delim_get = inner.children[d_child]->Cast<duckdb::LogicalDelimGet>();
  if (delim_get.chunk_types.size() != delim.duplicate_eliminated_columns.size()) {
    return std::nullopt;
  }

  // Positions in the inner join's output -> (child, column).
  auto const left_count  = inner.left_projection_map.empty() ? inner.children[0]->types.size()
                                                             : inner.left_projection_map.size();
  auto const right_count = inner.right_projection_map.empty() ? inner.children[1]->types.size()
                                                              : inner.right_projection_map.size();
  auto resolve_inner =
    [&](std::size_t position) -> std::optional<std::pair<std::size_t, std::size_t>> {
    if (position < left_count) {
      return std::pair{std::size_t{0},
                       inner.left_projection_map.empty()
                         ? position
                         : std::size_t{inner.left_projection_map[position]}};
    }
    position -= left_count;
    if (position >= right_count) { return std::nullopt; }
    return std::pair{std::size_t{1},
                     inner.right_projection_map.empty()
                       ? position
                       : std::size_t{inner.right_projection_map[position]}};
  };

  // Exactly one `t.k = delim.k` and one `t.x <> delim.x`, each between plain columns.
  exists_match match;
  match.delim       = &delim;
  match.outer_child = outer_idx;
  match.negated     = anti_family;
  match.inner       = &inner;
  match.t_child     = t_child;
  std::optional<std::size_t> key_delim_col, value_delim_col;
  for (auto const& condition : inner.conditions) {
    if (condition.left->GetExpressionClass() != ExpressionClass::BOUND_REF ||
        condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
      return std::nullopt;
    }
    auto const& left  = condition.left->Cast<duckdb::BoundReferenceExpression>();
    auto const& right = condition.right->Cast<duckdb::BoundReferenceExpression>();
    auto const& t_ref = d_child == 1 ? left : right;
    auto const& d_ref = d_child == 1 ? right : left;
    if (condition.comparison == ExpressionType::COMPARE_EQUAL) {
      if (key_delim_col) { return std::nullopt; }
      key_delim_col  = d_ref.index;
      match.t_key    = t_ref.index;
      match.key_type = t_ref.return_type;
      if (d_ref.return_type != t_ref.return_type) { return std::nullopt; }
    } else if (condition.comparison == ExpressionType::COMPARE_NOTEQUAL) {
      if (value_delim_col) { return std::nullopt; }
      value_delim_col  = d_ref.index;
      match.t_value    = t_ref.index;
      match.value_type = t_ref.return_type;
      if (d_ref.return_type != t_ref.return_type) { return std::nullopt; }
    } else {
      return std::nullopt;
    }
  }
  if (!key_delim_col || !value_delim_col || *key_delim_col == *value_delim_col) {
    return std::nullopt;
  }
  if (*key_delim_col >= delim_get.chunk_types.size() ||
      *value_delim_col >= delim_get.chunk_types.size()) {
    return std::nullopt;
  }

  // The fused operator takes integer keys and a MIN/MAX-able value.
  if (match.key_type.id() != duckdb::LogicalTypeId::INTEGER &&
      match.key_type.id() != duckdb::LogicalTypeId::BIGINT) {
    return std::nullopt;
  }
  auto const value_cudf = sirius::try_get_cudf_type(sirius::from_duckdb(match.value_type));
  if (!value_cudf || !op::probe_accumulate_supports_type(*value_cudf)) { return std::nullopt; }

  // The delim join must equate each outer correlated column with the subquery's echo of the
  // matching delim column, null-safely, and nothing else. `outer_for_delim[d]` is the outer
  // column the delim scan's column d was built from.
  if (delim.conditions.size() != delim_get.chunk_types.size()) { return std::nullopt; }
  std::vector<std::optional<std::size_t>> outer_for_delim(delim_get.chunk_types.size());
  for (auto const& condition : delim.conditions) {
    if (condition.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM ||
        condition.left->GetExpressionClass() != ExpressionClass::BOUND_REF ||
        condition.right->GetExpressionClass() != ExpressionClass::BOUND_REF) {
      return std::nullopt;
    }
    auto const& outer_ref =
      (outer_idx == 0 ? condition.left : condition.right)->Cast<duckdb::BoundReferenceExpression>();
    auto const& sub_ref =
      (outer_idx == 0 ? condition.right : condition.left)->Cast<duckdb::BoundReferenceExpression>();
    // Follow the subquery output column down through the projections to the inner join's output.
    std::size_t sub_position = sub_ref.index;
    for (auto const* projection : projections) {
      if (sub_position >= projection->expressions.size() ||
          projection->expressions[sub_position]->GetExpressionClass() !=
            ExpressionClass::BOUND_REF) {
        return std::nullopt;
      }
      sub_position =
        projection->expressions[sub_position]->Cast<duckdb::BoundReferenceExpression>().index;
    }
    auto const origin = resolve_inner(sub_position);
    if (!origin) { return std::nullopt; }
    // Either the delim scan's own column, or T's key column, which equals the delim key under
    // the join's equality and is what DuckDB echoes for the key.
    std::size_t d = 0;
    if (origin->first == d_child) {
      d = origin->second;
    } else if (origin->second == match.t_key) {
      d = *key_delim_col;
    } else {
      return std::nullopt;
    }
    if (d >= outer_for_delim.size() || outer_for_delim[d]) { return std::nullopt; }
    outer_for_delim[d] = outer_ref.index;
  }
  for (std::size_t d = 0; d < outer_for_delim.size(); ++d) {
    if (!outer_for_delim[d]) { return std::nullopt; }
    // DuckDB builds the distinct set from the duplicate-eliminated columns in this order.
    auto const& dup = delim.duplicate_eliminated_columns[d];
    if (dup->GetExpressionClass() != ExpressionClass::BOUND_REF ||
        dup->Cast<duckdb::BoundReferenceExpression>().index != *outer_for_delim[d]) {
      return std::nullopt;
    }
  }
  match.outer_key   = *outer_for_delim[*key_delim_col];
  match.outer_value = *outer_for_delim[*value_delim_col];
  return match;
}

duckdb::unique_ptr<duckdb::BoundAggregateExpression> bind_extremum(
  duckdb::ClientContext& context,
  char const* name,
  duckdb::LogicalType const& type,
  duckdb::unique_ptr<duckdb::Expression> argument)
{
  auto& entry =
    duckdb::Catalog::GetSystemCatalog(context).GetEntry<duckdb::AggregateFunctionCatalogEntry>(
      context, DEFAULT_SCHEMA, name);
  auto function = entry.functions.GetFunctionByArguments(context, {type});
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
  children.push_back(std::move(argument));
  duckdb::FunctionBinder binder(context);
  return binder.BindAggregateFunction(std::move(function), std::move(children));
}

/// `mn <> outer.x OR mx <> outer.x`: some row of the group differs from the outer value.
duckdb::unique_ptr<duckdb::Expression> some_row_differs(duckdb::LogicalType const& value_type,
                                                        duckdb::idx_t min_column,
                                                        duckdb::idx_t max_column,
                                                        duckdb::idx_t outer_value)
{
  return duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
    duckdb::ExpressionType::CONJUNCTION_OR,
    duckdb::make_uniq<duckdb::BoundComparisonExpression>(duckdb::ExpressionType::COMPARE_NOTEQUAL,
                                                         ref(value_type, min_column),
                                                         ref(value_type, outer_value)),
    duckdb::make_uniq<duckdb::BoundComparisonExpression>(duckdb::ExpressionType::COMPARE_NOTEQUAL,
                                                         ref(value_type, max_column),
                                                         ref(value_type, outer_value)));
}

/// Replace the matched DELIM_JOIN with
///
///   PROJECTION(outer columns) <- FILTER <- LEFT DELIM_JOIN(outer, A)
///   A = AGGREGATE(GROUP BY k: MIN(x), MAX(x)) <- INNER JOIN(T, DELIM_GET(k))
///
/// the shape DuckDB itself produces for a correlated scalar aggregate. The inner join is then a
/// join onto a one-column DELIM_GET (distinct, hence unique) under a group by its key, which the
/// probe-accumulate planner fuses.
duckdb::unique_ptr<duckdb::LogicalOperator> rewrite(duckdb::ClientContext& context,
                                                    exists_match const& match)
{
  auto& delim = *match.delim;
  auto& inner = *match.inner;

  auto outer                = std::move(delim.children[match.outer_child]);
  auto t                    = std::move(inner.children[match.t_child]);
  auto d                    = std::move(inner.children[1 - match.t_child]);
  auto const outer_types    = outer->types;
  auto const outer_width    = outer_types.size();
  auto const outer_estimate = outer->estimated_cardinality;
  auto const t_estimate     = t->estimated_cardinality;

  // The original output: the outer side's columns, through its projection map.
  auto const& output_map =
    match.outer_child == 0 ? delim.left_projection_map : delim.right_projection_map;
  duckdb::vector<duckdb::idx_t> output_columns;
  if (output_map.empty()) {
    for (duckdb::idx_t c = 0; c < outer_width; ++c) {
      output_columns.push_back(c);
    }
  } else {
    output_columns = output_map;
  }

  // The delim scan now carries the key alone.
  auto& delim_get       = d->Cast<duckdb::LogicalDelimGet>();
  delim_get.chunk_types = {match.key_type};
  delim_get.types       = {match.key_type};
  delim_get.SetEstimatedCardinality(outer_estimate);

  // T joined to the distinct keys, keeping only the value and the key.
  auto joined = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  duckdb::JoinCondition key_condition;
  key_condition.left       = ref(match.key_type, match.t_key);
  key_condition.right      = ref(match.key_type, 0);
  key_condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  joined->conditions.push_back(std::move(key_condition));
  joined->left_projection_map = {match.t_value};
  joined->types               = {match.value_type, match.key_type};
  joined->children.push_back(std::move(t));
  joined->children.push_back(std::move(d));
  joined->SetEstimatedCardinality(t_estimate);

  // MIN and MAX of the value per key.
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
  aggregates.push_back(bind_extremum(context, "min", match.value_type, ref(match.value_type, 0)));
  aggregates.push_back(bind_extremum(context, "max", match.value_type, ref(match.value_type, 0)));
  auto grouped = duckdb::make_uniq<duckdb::LogicalAggregate>(
    duckdb::idx_t{0}, duckdb::idx_t{0}, std::move(aggregates));
  grouped->groups.push_back(ref(match.key_type, 1));
  grouped->grouping_sets.push_back(duckdb::GroupingSet{0});
  grouped->group_stats.resize(1);
  grouped->distinct_validity = duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES;
  grouped->types             = {match.key_type, match.value_type, match.value_type};
  grouped->children.push_back(std::move(joined));
  grouped->SetEstimatedCardinality(outer_estimate);

  // outer LEFT JOIN A on the key, with the key still the duplicate-eliminated column.
  auto left = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(
    duckdb::JoinType::LEFT, LogicalOperatorType::LOGICAL_DELIM_JOIN);
  duckdb::JoinCondition delim_condition;
  delim_condition.left       = ref(match.key_type, match.outer_key);
  delim_condition.right      = ref(match.key_type, 0);
  delim_condition.comparison = duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM;
  left->conditions.push_back(std::move(delim_condition));
  left->duplicate_eliminated_columns.push_back(ref(match.key_type, match.outer_key));
  left->delim_flipped = false;
  left->types         = outer_types;
  left->types.push_back(match.key_type);
  left->types.push_back(match.value_type);
  left->types.push_back(match.value_type);
  left->children.push_back(std::move(outer));
  left->children.push_back(std::move(grouped));
  left->SetEstimatedCardinality(outer_estimate);

  // EXISTS keeps rows where some row differs. NOT EXISTS keeps the rest, including rows with no
  // group at all and rows whose comparison is unknown (a NULL value), hence "not true".
  auto const min_column = static_cast<duckdb::idx_t>(outer_width + 1);
  auto const max_column = static_cast<duckdb::idx_t>(outer_width + 2);
  duckdb::unique_ptr<duckdb::Expression> predicate =
    some_row_differs(match.value_type, min_column, max_column, match.outer_value);
  if (match.negated) {
    auto is_null = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
      duckdb::ExpressionType::OPERATOR_IS_NULL, duckdb::LogicalType::BOOLEAN);
    is_null->children.push_back(predicate->Copy());
    auto negation = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
      duckdb::ExpressionType::OPERATOR_NOT, duckdb::LogicalType::BOOLEAN);
    negation->children.push_back(std::move(predicate));
    predicate = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
      duckdb::ExpressionType::CONJUNCTION_OR, std::move(is_null), std::move(negation));
  }
  auto filter   = duckdb::make_uniq<duckdb::LogicalFilter>(std::move(predicate));
  filter->types = left->types;
  filter->children.push_back(std::move(left));
  filter->SetEstimatedCardinality(outer_estimate);

  // Back to exactly the columns the DELIM_JOIN used to emit.
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> columns;
  for (auto const column : output_columns) {
    columns.push_back(ref(outer_types[column], column));
  }
  auto projection =
    duckdb::make_uniq<duckdb::LogicalProjection>(duckdb::idx_t{0}, std::move(columns));
  projection->types = delim.types;
  projection->children.push_back(std::move(filter));
  projection->SetEstimatedCardinality(delim.estimated_cardinality);
  return projection;
}

// ---------------------------------------------------------------------------------------------
// Merging sibling subqueries
//
// `EXISTS (... l2.k = l1.k AND l2.x <> l1.x) AND NOT EXISTS (... l3.k = l1.k AND l3.x <> l1.x AND
// <p>)` become two nested delim joins, each with a scan of the same table. Rewritten one by one
// they still scan that table twice. When the scans read the same source, one scan can serve both:
// each subquery's own filter turns into a CASE inside its MIN/MAX, so a single aggregate keyed by
// the shared correlation key answers every subquery.
// ---------------------------------------------------------------------------------------------

/// One subquery's scan with its filters peeled off.
struct peeled_scan {
  duckdb::LogicalGet* get = nullptr;
  /// Output column of the peeled subtree -> output column of the scan, through the filter nodes'
  /// projection maps.
  duckdb::vector<duckdb::idx_t> scan_position;
  /// Every condition on the scan's rows, whether a filter node or a pushed-down table filter, with
  /// its column references holding base-table column numbers rather than output positions.
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> predicates;
};

/// Peel `FILTER* <- GET`. Anything else between the subquery's join and its scan declines.
std::optional<peeled_scan> peel_scan(duckdb::LogicalOperator& root);

/// The plain base-table column that output column @p position of @p get reads.
std::optional<duckdb::idx_t> base_column(duckdb::LogicalGet const& get, duckdb::idx_t position)
{
  auto const& ids    = get.GetColumnIds();
  duckdb::idx_t slot = position;
  if (!get.projection_ids.empty()) {
    if (position >= get.projection_ids.size()) { return std::nullopt; }
    slot = get.projection_ids[position];
  }
  if (slot >= ids.size()) { return std::nullopt; }
  auto const& column = ids[slot];
  if (!column.HasPrimaryIndex() || column.HasChildren() || column.IsVirtualColumn()) {
    return std::nullopt;
  }
  return column.GetPrimaryIndex();
}

/// Output position of base column @p column in @p get, if it is read at all.
std::optional<duckdb::idx_t> position_of(duckdb::LogicalGet const& get, duckdb::idx_t column)
{
  for (duckdb::idx_t p = 0; p < get.types.size(); ++p) {
    if (base_column(get, p) == column) { return p; }
  }
  return std::nullopt;
}

std::optional<peeled_scan> peel_scan(duckdb::LogicalOperator& root)
{
  peeled_scan scan;
  duckdb::vector<duckdb::reference<duckdb::Expression>> filter_expressions;
  duckdb::vector<duckdb::reference<duckdb::vector<duckdb::idx_t> const>> maps;  // outermost first
  auto* node = &root;
  while (node->type == LogicalOperatorType::LOGICAL_FILTER) {
    auto& filter = node->Cast<duckdb::LogicalFilter>();
    if (filter.children.size() != 1) { return std::nullopt; }
    if (!filter.projection_map.empty()) { maps.push_back(filter.projection_map); }
    for (auto const& expression : filter.expressions) {
      filter_expressions.push_back(*expression);
    }
    node = filter.children[0].get();
  }
  if (node->type != LogicalOperatorType::LOGICAL_GET || !node->children.empty()) {
    return std::nullopt;
  }
  scan.get = &node->Cast<duckdb::LogicalGet>();
  for (duckdb::idx_t p = 0; p < root.types.size(); ++p) {
    auto position = p;
    for (auto const& map : maps) {
      if (position >= map.get().size()) { return std::nullopt; }
      position = map.get()[position];
    }
    scan.scan_position.push_back(position);
  }

  // Filter nodes read the scan's output positions: translate them to base columns.
  for (auto const& expression : filter_expressions) {
    auto copy = expression.get().Copy();
    bool ok   = true;
    duckdb::ExpressionIterator::VisitExpressionClassMutable(
      copy, duckdb::ExpressionClass::BOUND_REF, [&](duckdb::unique_ptr<duckdb::Expression>& leaf) {
        auto const& reference = leaf->Cast<duckdb::BoundReferenceExpression>();
        auto const base       = base_column(*scan.get, reference.index);
        if (!base) {
          ok = false;
          return;
        }
        leaf = ref(reference.return_type, *base);
      });
    if (!ok) { return std::nullopt; }
    scan.predicates.push_back(std::move(copy));
  }

  // Pushed-down table filters are keyed by the base column they test.
  for (auto const& [base, table_filter] : scan.get->table_filters.filters) {
    if (base >= scan.get->returned_types.size()) { return std::nullopt; }
    auto column = ref(scan.get->returned_types[base], base);
    scan.predicates.push_back(table_filter->ToExpression(*column));
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
  if (a.function.name != b.function.name || a.returned_types != b.returned_types ||
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

/// Re-address @p expression, whose column references are base-table column numbers, onto output
/// positions given by @p position_of_base.
template <class BasePosition>
duckdb::unique_ptr<duckdb::Expression> to_positions(duckdb::Expression const& expression,
                                                    BasePosition const& position_of_base)
{
  auto copy = expression.Copy();
  duckdb::ExpressionIterator::VisitExpressionClassMutable(
    copy, duckdb::ExpressionClass::BOUND_REF, [&](duckdb::unique_ptr<duckdb::Expression>& leaf) {
      auto const& reference = leaf->Cast<duckdb::BoundReferenceExpression>();
      leaf                  = ref(reference.return_type, position_of_base(reference.index));
    });
  return copy;
}

/// Make @p get also output base column @p base, reading it if the scan does not already.
void widen(duckdb::LogicalGet& get, duckdb::idx_t base)
{
  auto& ids          = get.GetMutableColumnIds();
  duckdb::idx_t slot = ids.size();
  for (duckdb::idx_t i = 0; i < ids.size(); ++i) {
    if (ids[i].HasPrimaryIndex() && ids[i].GetPrimaryIndex() == base && !ids[i].HasChildren()) {
      slot = i;
      break;
    }
  }
  if (slot == ids.size()) { ids.push_back(duckdb::ColumnIndex(base)); }
  if (!get.projection_ids.empty()) { get.projection_ids.push_back(slot); }
  get.types.push_back(get.returned_types[base]);
}

/// Positions of the outer rows' columns, translated down through a chain of rewritten joins.
/// @p chain is innermost first; @p level is the level whose outer child's output @p position
/// addresses.
duckdb::idx_t translate_down(std::vector<exists_match> const& chain,
                             std::size_t level,
                             duckdb::idx_t position)
{
  for (std::size_t l = level; l-- > 0;) {
    auto const& match = chain[l];
    auto const& map =
      match.outer_child == 0 ? match.delim->left_projection_map : match.delim->right_projection_map;
    if (!map.empty()) { position = map[position]; }
  }
  return position;
}

/// Everything one subquery contributes to the merged aggregate, over the shared scan.
struct merged_sub {
  duckdb::unique_ptr<duckdb::Expression> value;      // the compared column, over the shared scan
  duckdb::unique_ptr<duckdb::Expression> predicate;  // AND of its filters, or null
  duckdb::idx_t outer_value = 0;                     // compared outer column, in O0 coordinates
  bool negated              = false;
};

/// Rewrite a chain of two or more matched delim joins (innermost first) over the same scan.
/// Returns null, consuming nothing, when the subqueries cannot share one scan.
duckdb::unique_ptr<duckdb::LogicalOperator> rewrite_merged(duckdb::ClientContext& context,
                                                           std::vector<exists_match>& chain)
{
  auto const levels = chain.size();
  if (levels < 2 || levels * 2 > op::probe_accumulate_max_specs) { return nullptr; }

  // Peel every subquery's scan.
  std::vector<peeled_scan> scans;
  for (auto& match : chain) {
    auto scan = peel_scan(*match.inner->children[match.t_child]);
    if (!scan) { return nullptr; }
    scans.push_back(std::move(*scan));
  }

  // One source, one correlation key (the same base column, the same outer column).
  auto const outer_key0 = chain[0].outer_key;
  auto const key_base   = base_column(*scans[0].get, scans[0].scan_position[chain[0].t_key]);
  if (!key_base) { return nullptr; }
  for (std::size_t i = 0; i < levels; ++i) {
    if (!same_source(context, *scans[0].get, *scans[i].get) ||
        chain[i].key_type != chain[0].key_type ||
        base_column(*scans[i].get, scans[i].scan_position[chain[i].t_key]) != key_base ||
        translate_down(chain, i, static_cast<duckdb::idx_t>(chain[i].outer_key)) != outer_key0) {
      return nullptr;
    }
  }

  // The shared scan is one of the existing ones: whichever reads every column the subqueries need.
  std::vector<duckdb::idx_t> needed;  // base columns
  auto need = [&](duckdb::LogicalGet const& get, duckdb::idx_t position) {
    auto const base = base_column(get, position);
    if (!base) { return false; }
    if (std::find(needed.begin(), needed.end(), *base) == needed.end()) { needed.push_back(*base); }
    return true;
  };
  for (std::size_t i = 0; i < levels; ++i) {
    if (!need(*scans[i].get, scans[i].scan_position[chain[i].t_key]) ||
        !need(*scans[i].get, scans[i].scan_position[chain[i].t_value])) {
      return nullptr;
    }
    for (auto const& predicate : scans[i].predicates) {
      auto scratch = predicate->Copy();
      duckdb::ExpressionIterator::EnumerateExpression(scratch, [&](duckdb::Expression& leaf) {
        if (leaf.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
          auto const base = leaf.Cast<duckdb::BoundReferenceExpression>().index;
          if (std::find(needed.begin(), needed.end(), base) == needed.end()) {
            needed.push_back(base);
          }
        }
      });
    }
  }
  // Any of the scans can serve once it also outputs the columns the others need, so take the one
  // that already outputs the most.
  std::size_t shared = 0;
  std::size_t best   = 0;
  for (std::size_t i = 0; i < levels; ++i) {
    std::size_t covered = 0;
    for (auto const column : needed) {
      covered += position_of(*scans[i].get, column).has_value() ? 1 : 0;
    }
    if (i == 0 || covered > best) {
      shared = i;
      best   = covered;
    }
  }
  auto& shared_get = *scans[shared].get;
  std::vector<duckdb::idx_t> missing;  // base columns the shared scan must start outputting
  for (auto const column : needed) {
    if (column >= shared_get.returned_types.size()) { return nullptr; }
    if (!position_of(shared_get, column)) { missing.push_back(column); }
  }
  auto const shared_width       = shared_get.types.size();
  auto const position_in_shared = [&](duckdb::idx_t base) -> duckdb::idx_t {
    if (auto const at = position_of(shared_get, base)) { return *at; }
    auto const found = std::find(missing.begin(), missing.end(), base);
    return static_cast<duckdb::idx_t>(shared_width + (found - missing.begin()));
  };

  // Each subquery over the shared scan.
  std::vector<merged_sub> subs;
  for (std::size_t i = 0; i < levels; ++i) {
    merged_sub sub;
    sub.negated     = chain[i].negated;
    sub.outer_value = translate_down(chain, i, static_cast<duckdb::idx_t>(chain[i].outer_value));
    auto const value_base = base_column(*scans[i].get, scans[i].scan_position[chain[i].t_value]);
    sub.value             = ref(chain[i].value_type, position_in_shared(*value_base));
    for (auto const& predicate : scans[i].predicates) {
      auto moved    = to_positions(*predicate, position_in_shared);
      sub.predicate = sub.predicate ? duckdb::unique_ptr<duckdb::Expression>(
                                        duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
                                          duckdb::ExpressionType::CONJUNCTION_AND,
                                          std::move(sub.predicate),
                                          std::move(moved)))
                                    : std::move(moved);
    }
    subs.push_back(std::move(sub));
  }
  auto const key_at = position_in_shared(*key_base);

  // ---- Nothing below declines: consume the original nodes. ----
  auto& outermost  = *chain.back().delim;
  auto const types = outermost.types;
  // The outermost join's output columns, in the innermost outer rows' coordinates.
  duckdb::vector<duckdb::idx_t> output_columns;
  {
    auto const& match = chain.back();
    auto const& map =
      match.outer_child == 0 ? match.delim->left_projection_map : match.delim->right_projection_map;
    auto const width = match.delim->children[match.outer_child]->types.size();
    for (duckdb::idx_t c = 0; c < (map.empty() ? width : map.size()); ++c) {
      auto const position = map.empty() ? c : map[c];
      output_columns.push_back(translate_down(chain, levels - 1, position));
    }
  }

  auto& innermost           = chain[0];
  auto outer                = std::move(innermost.delim->children[innermost.outer_child]);
  auto const outer_types    = outer->types;
  auto const outer_width    = outer_types.size();
  auto const outer_estimate = outer->estimated_cardinality;
  auto const key_type       = chain[0].key_type;

  // Reuse the first subquery's delim scan, carrying the key alone; take the shared scan out of its
  // own subquery (a filter chain around it is dropped with the rest of that subquery).
  // The shared scan now outputs everything the aggregates read, and its own pushed-down filters
  // have become CASE predicates, so it must return every row.
  for (auto const column : missing) {
    widen(shared_get, column);
  }
  shared_get.table_filters.filters.clear();

  auto delim_get         = std::move(chain[0].inner->children[1 - chain[0].t_child]);
  auto& delim_scan       = delim_get->Cast<duckdb::LogicalDelimGet>();
  delim_scan.chunk_types = {key_type};
  delim_scan.types       = {key_type};
  delim_scan.SetEstimatedCardinality(outer_estimate);

  // Locate the shared gets owning pointer: it is the leaf below its filter chain.
  duckdb::unique_ptr<duckdb::LogicalOperator>* slot =
    &chain[shared].inner->children[chain[shared].t_child];
  while ((*slot)->type == LogicalOperatorType::LOGICAL_FILTER) {
    slot = &(*slot)->children[0];
  }
  auto shared_owned        = std::move(*slot);
  auto const scan_estimate = shared_owned->estimated_cardinality;

  // Inner join of the shared scan and the distinct keys; keep only the columns the aggregates read.
  std::vector<duckdb::idx_t> kept;  // shared-scan positions, in output order
  auto keep = [&](duckdb::idx_t position) -> duckdb::idx_t {
    auto found = std::find(kept.begin(), kept.end(), position);
    if (found == kept.end()) {
      kept.push_back(position);
      return static_cast<duckdb::idx_t>(kept.size() - 1);
    }
    return static_cast<duckdb::idx_t>(found - kept.begin());
  };
  auto remap_to_kept = [&](duckdb::unique_ptr<duckdb::Expression> expression) {
    duckdb::ExpressionIterator::VisitExpressionClassMutable(
      expression,
      duckdb::ExpressionClass::BOUND_REF,
      [&](duckdb::unique_ptr<duckdb::Expression>& leaf) {
        auto const& reference = leaf->Cast<duckdb::BoundReferenceExpression>();
        leaf                  = ref(reference.return_type, keep(reference.index));
      });
    return expression;
  };

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> aggregates;
  duckdb::vector<duckdb::LogicalType> aggregate_types{key_type};
  for (auto& sub : subs) {
    auto const value_type                           = sub.value->return_type;
    duckdb::unique_ptr<duckdb::Expression> argument = sub.value->Copy();
    if (sub.predicate) {
      argument = duckdb::make_uniq<duckdb::BoundCaseExpression>(
        sub.predicate->Copy(),
        std::move(argument),
        duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value(value_type)));
    }
    argument = remap_to_kept(std::move(argument));
    aggregates.push_back(bind_extremum(context, "min", value_type, argument->Copy()));
    aggregates.push_back(bind_extremum(context, "max", value_type, std::move(argument)));
    aggregate_types.push_back(value_type);
    aggregate_types.push_back(value_type);
  }

  auto joined = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  duckdb::JoinCondition equal;
  equal.left       = ref(key_type, key_at);
  equal.right      = ref(key_type, 0);
  equal.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  joined->conditions.push_back(std::move(equal));
  for (auto const position : kept) {
    joined->left_projection_map.push_back(position);
    joined->types.push_back(shared_get.types[position]);
  }
  joined->types.push_back(key_type);
  joined->children.push_back(std::move(shared_owned));
  joined->children.push_back(std::move(delim_get));
  joined->SetEstimatedCardinality(scan_estimate);

  auto grouped = duckdb::make_uniq<duckdb::LogicalAggregate>(
    duckdb::idx_t{0}, duckdb::idx_t{0}, std::move(aggregates));
  grouped->groups.push_back(ref(key_type, static_cast<duckdb::idx_t>(kept.size())));
  grouped->grouping_sets.push_back(duckdb::GroupingSet{0});
  grouped->group_stats.resize(1);
  grouped->distinct_validity = duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES;
  grouped->types             = aggregate_types;
  grouped->children.push_back(std::move(joined));
  grouped->SetEstimatedCardinality(outer_estimate);

  auto left = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(
    duckdb::JoinType::LEFT, LogicalOperatorType::LOGICAL_DELIM_JOIN);
  duckdb::JoinCondition delim_condition;
  delim_condition.left       = ref(key_type, outer_key0);
  delim_condition.right      = ref(key_type, 0);
  delim_condition.comparison = duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM;
  left->conditions.push_back(std::move(delim_condition));
  left->duplicate_eliminated_columns.push_back(ref(key_type, outer_key0));
  left->delim_flipped = false;
  left->types         = outer_types;
  for (auto const& type : aggregate_types) {
    left->types.push_back(type);
  }
  left->children.push_back(std::move(outer));
  left->children.push_back(std::move(grouped));
  left->SetEstimatedCardinality(outer_estimate);

  // Every subquery's condition must hold: EXISTS some row differs, NOT EXISTS none is known to.
  duckdb::unique_ptr<duckdb::Expression> predicate;
  for (std::size_t i = 0; i < subs.size(); ++i) {
    auto const min_column = static_cast<duckdb::idx_t>(outer_width + 1 + 2 * i);
    auto const max_column = static_cast<duckdb::idx_t>(outer_width + 2 + 2 * i);
    auto condition =
      some_row_differs(subs[i].value->return_type, min_column, max_column, subs[i].outer_value);
    if (subs[i].negated) {
      auto is_null = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
        duckdb::ExpressionType::OPERATOR_IS_NULL, duckdb::LogicalType::BOOLEAN);
      is_null->children.push_back(condition->Copy());
      auto negation = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
        duckdb::ExpressionType::OPERATOR_NOT, duckdb::LogicalType::BOOLEAN);
      negation->children.push_back(std::move(condition));
      condition = duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
        duckdb::ExpressionType::CONJUNCTION_OR, std::move(is_null), std::move(negation));
    }
    predicate =
      predicate
        ? duckdb::unique_ptr<duckdb::Expression>(
            duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
              duckdb::ExpressionType::CONJUNCTION_AND, std::move(predicate), std::move(condition)))
        : std::move(condition);
  }
  auto filter   = duckdb::make_uniq<duckdb::LogicalFilter>(std::move(predicate));
  filter->types = left->types;
  filter->children.push_back(std::move(left));
  filter->SetEstimatedCardinality(outer_estimate);

  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> columns;
  for (auto const column : output_columns) {
    columns.push_back(ref(outer_types[column], column));
  }
  auto projection =
    duckdb::make_uniq<duckdb::LogicalProjection>(duckdb::idx_t{0}, std::move(columns));
  projection->types = types;
  projection->children.push_back(std::move(filter));
  projection->SetEstimatedCardinality(outermost.estimated_cardinality);
  return projection;
}

/// The longest chain of matched delim joins directly nested through their outer child, innermost
/// first. Only a chain of two or more is worth merging.
std::vector<exists_match> match_chain(duckdb::LogicalOperator& node)
{
  std::vector<exists_match> outermost_first;
  auto* current = &node;
  while (auto match = match_exists(*current)) {
    outermost_first.push_back(*match);
    current = outermost_first.back().delim->children[outermost_first.back().outer_child].get();
  }
  return {outermost_first.rbegin(), outermost_first.rend()};
}

/// Whether the projection a rewrite builds can reproduce the original node's output.
bool output_is_reproducible(exists_match const& match, duckdb::LogicalOperator const& original)
{
  auto const& output_map =
    match.outer_child == 0 ? match.delim->left_projection_map : match.delim->right_projection_map;
  auto const outer_width = match.delim->children[match.outer_child]->types.size();
  auto const expected    = output_map.empty() ? outer_width : output_map.size();
  if (expected != original.types.size()) { return false; }
  for (auto const column : output_map) {
    if (column >= outer_width) { return false; }
  }
  return true;
}

void rewrite_tree(duckdb::ClientContext& context,
                  duckdb::unique_ptr<duckdb::LogicalOperator>& slot,
                  std::size_t& rewrites)
{
  // Sibling subqueries first, so a chain is merged before its members are rewritten one by one.
  if (auto chain = match_chain(*slot); chain.size() >= 2) {
    bool reproducible = true;
    for (auto const& match : chain) {
      reproducible = reproducible && output_is_reproducible(match, *match.delim);
    }
    if (reproducible) {
      if (auto merged = rewrite_merged(context, chain)) {
        slot = std::move(merged);
        rewrites += chain.size();
        // The merged node's outer rows may hold further subqueries.
        auto& outer = slot->children[0]->children[0]->children[0];
        rewrite_tree(context, outer, rewrites);
        return;
      }
    }
  }

  for (auto& child : slot->children) {
    rewrite_tree(context, child, rewrites);
  }
  auto const match = match_exists(*slot);
  if (!match || !output_is_reproducible(*match, *slot)) { return; }
  slot = rewrite(context, *match);
  ++rewrites;
}

}  // namespace

std::size_t rewrite_correlated_inequality_exists(duckdb::ClientContext& context,
                                                 duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  std::size_t rewrites = 0;
  rewrite_tree(context, plan, rewrites);
  if (rewrites > 0) {
    SIRIUS_LOG_INFO(
      "[sirius_plan_exists_rewrite] Rewrote {} correlated EXISTS/NOT EXISTS with `=` and `<>` into "
      "a MIN/MAX aggregate per key",
      rewrites);
  }
  return rewrites;
}

}  // namespace sirius::planner
