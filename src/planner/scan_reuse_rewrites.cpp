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

#include "planner/scan_reuse_rewrites.hpp"

#include <duckdb/function/builtin_function_lookup.hpp>
#include <duckdb/function/function_binder.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/optimizer/column_binding_replacer.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/logical_operator_visitor.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_cteref.hpp>
#include <duckdb/planner/operator/logical_delim_get.hpp>
#include <duckdb/planner/operator/logical_filter.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_materialized_cte.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::planner {
namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::ProjectionIndex;
using duckdb::unique_ptr;

/// The binding @p expression reads when it is a plain column reference of this query level.
std::optional<ColumnBinding> column_ref(Expression const& expression)
{
  if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
    return std::nullopt;
  }
  auto const& ref = expression.Cast<duckdb::BoundColumnRefExpression>();
  if (ref.Depth() != 0) { return std::nullopt; }
  return ref.Binding();
}

/// Every column binding @p expression reads.
void collect_references(Expression const& expression, std::vector<ColumnBinding>& out)
{
  if (auto const ref = column_ref(expression)) {
    out.push_back(*ref);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    expression, [&](Expression const& child) { collect_references(child, out); });
}

std::optional<std::size_t> position_of(std::vector<ColumnBinding> const& bindings,
                                       ColumnBinding binding)
{
  auto const it = std::find(bindings.begin(), bindings.end(), binding);
  if (it == bindings.end()) { return std::nullopt; }
  return static_cast<std::size_t>(it - bindings.begin());
}

/// Why a rule left a plan alone, for the debug log.
void not_applied(char const* rule, std::string const& why)
{
  SIRIUS_LOG_DEBUG("[scan reuse] {} not applied: {}", rule, why);
}

bool produces(LogicalOperator& op, ColumnBinding binding)
{
  return position_of(op.GetColumnBindings(), binding).has_value();
}

/// A table column a binding carries unchanged: the scan and the column's primary index.
struct column_origin {
  duckdb::LogicalGet* get;
  duckdb::idx_t column;
};

/**
 * @brief The scan column @p binding carries, through projections of plain references, filters,
 * joins and aggregate group columns; nullopt for computed columns, row ids and virtual columns.
 */
std::optional<column_origin> trace(LogicalOperator& op, ColumnBinding binding)
{
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_GET: {
      auto& get = op.Cast<duckdb::LogicalGet>();
      if (get.table_index != binding.table_index) { return std::nullopt; }
      auto const& ids = get.GetColumnIds();
      auto const at   = binding.column_index.GetIndex();
      if (at >= ids.size()) { return std::nullopt; }
      auto const& id = ids[at];
      if (id.IsRowIdColumn() || id.IsVirtualColumn() || id.HasChildren()) { return std::nullopt; }
      return column_origin{&get, id.GetPrimaryIndex()};
    }
    case LogicalOperatorType::LOGICAL_PROJECTION: {
      auto& projection = op.Cast<duckdb::LogicalProjection>();
      if (projection.table_index != binding.table_index) { return std::nullopt; }
      auto const at = binding.column_index.GetIndex();
      if (at >= projection.expressions.size()) { return std::nullopt; }
      auto const ref = column_ref(*projection.expressions[at]);
      if (!ref) { return std::nullopt; }
      return trace(*projection.children[0], *ref);
    }
    case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
      auto& aggregate = op.Cast<duckdb::LogicalAggregate>();
      if (aggregate.group_index != binding.table_index) { return std::nullopt; }
      auto const at = binding.column_index.GetIndex();
      if (at >= aggregate.groups.size()) { return std::nullopt; }
      auto const ref = column_ref(*aggregate.groups[at]);
      if (!ref) { return std::nullopt; }
      return trace(*aggregate.children[0], *ref);
    }
    default:
      for (auto& child : op.children) {
        if (produces(*child, binding)) { return trace(*child, binding); }
      }
      return std::nullopt;
  }
}

/// The filters of @p get that change its rows, as text keyed by table column, in a fixed order.
/// Optional filters (dynamic join and top-N filters) only skip rows the plan drops later anyway.
std::vector<std::string> required_filters(duckdb::LogicalGet const& get)
{
  std::vector<std::string> out;
  auto const& ids = get.GetColumnIds();
  for (auto const& entry : get.table_filters) {
    auto const& filter = entry.Filter();
    if (duckdb::ExpressionFilter::IsOptionalFilter(filter)) { continue; }
    auto const at = entry.GetIndex().GetIndex();
    if (at >= ids.size()) {
      out.emplace_back("?");
      continue;
    }
    auto const column = ids[at].GetPrimaryIndex();
    auto const type   = column < get.returned_types.size() ? get.returned_types[column]
                                                           : duckdb::LogicalType::SQLNULL;
    duckdb::BoundColumnRefExpression placeholder(
      duckdb::Identifier("c" + std::to_string(column)),
      type,
      ColumnBinding(duckdb::TableIndex(0), ProjectionIndex(column)));
    out.push_back(std::to_string(column) + ":" + filter.ToExpression(placeholder)->ToString());
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// Whether two scans read the same rows: the same table, or the same table function with the same
/// arguments, and the same row filters.
bool same_rows(duckdb::LogicalGet const& a, duckdb::LogicalGet const& b)
{
  if (!(a.function.GetName() == b.function.GetName())) { return false; }
  auto const table = a.GetTable();
  if (table) {
    if (table.get() != b.GetTable().get()) { return false; }
  } else if (a.parameters.empty() || a.parameters != b.parameters ||
             a.named_parameters != b.named_parameters) {
    // Without arguments, a function's identity is only in its bind data (see DuckDB's
    // CommonSubplanOptimizer): equal arguments cannot prove the same rows.
    return false;
  }
  return required_filters(a) == required_filters(b);
}

/// A plain, non-DISTINCT, unfiltered, unordered aggregate @p name over one column reference.
/// "sum" also matches sum_no_overflow, which DuckDB's statistics put where a SUM cannot overflow.
std::optional<ColumnBinding> plain_aggregate_of(Expression const& expression, char const* name)
{
  if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
    return std::nullopt;
  }
  auto const& aggregate = expression.Cast<duckdb::BoundAggregateExpression>();
  auto const& function  = aggregate.Function().GetName();
  bool const named =
    function == name || (std::string_view(name) == "sum" && function == "sum_no_overflow");
  if (!named || aggregate.IsDistinct() || aggregate.GetFilter() || aggregate.GetOrderBys() ||
      aggregate.GetChildren().size() != 1) {
    return std::nullopt;
  }
  return column_ref(*aggregate.GetChildren()[0]);
}

/// A step from an operator down to one of its children.
struct path_step {
  LogicalOperator* op;
  std::size_t child;
};

/// Appends @p binding to @p op's projection map for child @p child when that map is not empty
/// (an empty map passes every child column through).
void keep_through(LogicalOperator& op, std::size_t child, ColumnBinding binding)
{
  auto map = duckdb::LogicalOperatorVisitor::GetProjectionMap(op, child);
  if (!map || map->empty()) { return; }
  auto const at = position_of(op.children[child]->GetColumnBindings(), binding);
  if (!at) { throw std::logic_error("[scan reuse] a carried column vanished below its parent"); }
  if (std::find(map->begin(), map->end(), ProjectionIndex(*at)) == map->end()) {
    map->push_back(ProjectionIndex(*at));
  }
}

//===----------------------------------------------------------------------===//
// Rule: reuse a SEMI join's grouped aggregate (TPC-H Q18)
//===----------------------------------------------------------------------===//

struct semi_join_match {
  /// Which side of the outer INNER join is the repeated scan.
  std::size_t scan_side;
  /// From the other side's root down to the SEMI join.
  std::vector<path_step> to_semi;
  duckdb::LogicalComparisonJoin* semi;
  /// From the SEMI join's build side down to the subquery's aggregate (projections and filters).
  std::vector<LogicalOperator*> to_aggregate;
  duckdb::LogicalAggregate* subquery;
  /// For each outer aggregate, the subquery aggregate (index into its expressions) it reuses.
  std::vector<std::size_t> reused;
};

std::optional<semi_join_match> match_semi_join_aggregate(duckdb::LogicalAggregate& outer)
{
  if (outer.grouping_sets.size() > 1 || !outer.grouping_functions.empty() ||
      outer.expressions.empty()) {
    return std::nullopt;
  }
  auto& below = *outer.children[0];
  if (below.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return std::nullopt; }
  auto& join = below.Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::INNER || join.conditions.size() != 1 ||
      !join.conditions[0].IsComparison() ||
      join.conditions[0].GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL) {
    return std::nullopt;
  }
  for (std::size_t side = 0; side < 2; ++side) {
    if (join.children[side]->type != LogicalOperatorType::LOGICAL_GET) { continue; }
    auto& scan            = join.children[side]->Cast<duckdb::LogicalGet>();
    auto const& condition = join.conditions[0];
    auto const scan_key   = column_ref(side == 0 ? condition.GetLHS() : condition.GetRHS());
    auto const other_key  = column_ref(side == 0 ? condition.GetRHS() : condition.GetLHS());
    if (!scan_key || !other_key) { continue; }
    auto const scan_key_origin = trace(scan, *scan_key);
    if (!scan_key_origin) { continue; }

    // Every outer aggregate is SUM(scan column), and nothing else reads the scan.
    std::vector<duckdb::idx_t> summed;
    bool sums_only = true;
    for (auto const& expression : outer.expressions) {
      auto const argument = plain_aggregate_of(*expression, "sum");
      auto const origin   = argument ? trace(scan, *argument) : std::nullopt;
      if (!origin) {
        sums_only = false;
        break;
      }
      summed.push_back(origin->column);
    }
    if (!sums_only) { continue; }
    std::vector<ColumnBinding> group_refs;
    for (auto const& group : outer.groups) {
      collect_references(*group, group_refs);
    }
    if (std::ranges::any_of(group_refs,
                            [&](auto const& ref) { return ref.table_index == scan.table_index; })) {
      continue;
    }

    // Down the other side, through INNER joins and filters, to the SEMI join on other_key.
    semi_join_match match{.scan_side = side};
    LogicalOperator* at                 = join.children[1 - side].get();
    duckdb::LogicalComparisonJoin* semi = nullptr;
    while (at != nullptr && semi == nullptr) {
      if (at->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
        auto& candidate = at->Cast<duckdb::LogicalComparisonJoin>();
        if (candidate.join_type == duckdb::JoinType::SEMI && candidate.conditions.size() == 1 &&
            candidate.conditions[0].IsComparison() &&
            candidate.conditions[0].GetComparisonType() == duckdb::ExpressionType::COMPARE_EQUAL &&
            column_ref(candidate.conditions[0].GetLHS()) == other_key) {
          semi = &candidate;
          break;
        }
        if (candidate.join_type != duckdb::JoinType::INNER) { break; }
      } else if (at->type != LogicalOperatorType::LOGICAL_FILTER) {
        break;
      }
      LogicalOperator* next = nullptr;
      for (std::size_t c = 0; c < at->children.size(); ++c) {
        if (produces(*at->children[c], *other_key)) {
          match.to_semi.push_back({at, c});
          next = at->children[c].get();
          break;
        }
      }
      at = next;
    }
    if (semi == nullptr) { continue; }

    // The build side: projections and filters over the subquery's aggregate, keyed on its group.
    auto key = column_ref(semi->conditions[0].GetRHS());
    if (!key) { continue; }
    LogicalOperator* build = semi->children[1].get();
    while (build->type == LogicalOperatorType::LOGICAL_PROJECTION ||
           build->type == LogicalOperatorType::LOGICAL_FILTER) {
      if (build->type == LogicalOperatorType::LOGICAL_PROJECTION) {
        auto& projection = build->Cast<duckdb::LogicalProjection>();
        if (projection.table_index != key->table_index) { break; }
        key = column_ref(*projection.expressions[key->column_index.GetIndex()]);
        if (!key) { break; }
      }
      match.to_aggregate.push_back(build);
      build = build->children[0].get();
    }
    if (!key || build->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { continue; }
    auto& subquery = build->Cast<duckdb::LogicalAggregate>();
    if (subquery.groups.size() != 1 || subquery.grouping_sets.size() > 1 ||
        !subquery.grouping_functions.empty() ||
        *key != ColumnBinding(subquery.group_index, ProjectionIndex(0))) {
      continue;
    }
    // Its input: projections of plain references over a scan of the same rows as the outer scan.
    LogicalOperator* input = subquery.children[0].get();
    while (input->type == LogicalOperatorType::LOGICAL_PROJECTION &&
           std::ranges::all_of(input->expressions,
                               [](auto const& e) { return column_ref(*e).has_value(); })) {
      input = input->children[0].get();
    }
    if (input->type != LogicalOperatorType::LOGICAL_GET ||
        !same_rows(scan, input->Cast<duckdb::LogicalGet>())) {
      continue;
    }
    auto const group_ref    = column_ref(*subquery.groups[0]);
    auto const group_origin = group_ref ? trace(*subquery.children[0], *group_ref) : std::nullopt;
    if (!group_origin || group_origin->column != scan_key_origin->column) { continue; }
    // Each outer SUM(c) has a SUM(c) in the subquery.
    for (auto const column : summed) {
      for (std::size_t j = 0; j < subquery.expressions.size(); ++j) {
        auto const argument = plain_aggregate_of(*subquery.expressions[j], "sum");
        auto const origin   = argument ? trace(*subquery.children[0], *argument) : std::nullopt;
        if (origin && origin->column == column) {
          match.reused.push_back(j);
          break;
        }
      }
    }
    if (match.reused.size() != summed.size()) { continue; }
    match.semi     = semi;
    match.subquery = &subquery;
    return match;
  }
  return std::nullopt;
}

bool reuse_semi_join_aggregate(duckdb::ClientContext& context,
                               duckdb::LogicalAggregate& outer,
                               removed_subtree_callback const& removed)
{
  auto match = match_semi_join_aggregate(outer);
  if (!match) { return false; }
  auto& subquery = *match->subquery;

  // Bind the new aggregates first: nothing changes unless they keep their result types.
  duckdb::FunctionBinder binder(context);
  std::vector<unique_ptr<Expression>> sums;
  std::vector<ColumnBinding> carried;  // the reused SUMs, as the SEMI join's build side emits them
  for (std::size_t i = 0; i < match->reused.size(); ++i) {
    auto const j    = match->reused[i];
    auto const type = subquery.expressions[j]->GetReturnType();
    // Placeholder binding: replaced below once the build side carries the column.
    std::vector<unique_ptr<Expression>> arguments;
    arguments.push_back(duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
      type, ColumnBinding(subquery.aggregate_index, ProjectionIndex(j))));
    auto sum = binder.BindAggregateFunction(
      duckdb::GetBuiltinAggregateFunction(context, duckdb::Identifier("sum"), {type}),
      std::move(arguments));
    if (sum->GetReturnType() != outer.expressions[i]->GetReturnType()) { return false; }
    sum->SetAlias(outer.expressions[i]->GetAlias());
    sums.push_back(std::move(sum));
  }

  // Carry each reused SUM up the build side: filters pass it through (keeping it in a non-empty
  // projection map), projections get a reference to it.
  for (auto const j : match->reused) {
    ColumnBinding binding(subquery.aggregate_index, ProjectionIndex(j));
    auto const type = subquery.expressions[j]->GetReturnType();
    for (auto it = match->to_aggregate.rbegin(); it != match->to_aggregate.rend(); ++it) {
      auto& op = **it;
      if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
        keep_through(op, 0, binding);
      } else {
        auto& projection = op.Cast<duckdb::LogicalProjection>();
        projection.expressions.push_back(
          duckdb::make_uniq<duckdb::BoundColumnRefExpression>(type, binding));
        binding =
          ColumnBinding(projection.table_index, ProjectionIndex(projection.expressions.size() - 1));
      }
    }
    carried.push_back(binding);
  }

  // The SEMI join becomes INNER (its build keys are unique), and the reused SUMs go up to the
  // outer aggregate.
  auto& semi     = *match->semi;
  semi.join_type = duckdb::JoinType::INNER;
  for (auto const& binding : carried) {
    keep_through(semi, 1, binding);
    for (auto it = match->to_semi.rbegin(); it != match->to_semi.rend(); ++it) {
      keep_through(*it->op, it->child, binding);
    }
  }
  for (std::size_t i = 0; i < sums.size(); ++i) {
    auto& sum = sums[i]->Cast<duckdb::BoundAggregateExpression>();
    sum.GetChildrenMutable()[0]->Cast<duckdb::BoundColumnRefExpression>().BindingMutable() =
      carried[i];
    outer.expressions[i] = std::move(sums[i]);
  }
  // The join with the repeated scan goes.
  if (removed) { removed(*outer.children[0]->children[match->scan_side]); }
  auto other        = std::move(outer.children[0]->children[1 - match->scan_side]);
  outer.children[0] = std::move(other);
  return true;
}

//===----------------------------------------------------------------------===//
// Inner-join regions
//===----------------------------------------------------------------------===//

/// A column of a join region by what it is, not where: its scan's identity (source and filters)
/// and the table column. A DELIM_GET column is {"", i}.
struct region_column {
  std::string leaf;
  duckdb::idx_t column;
  auto operator<=>(region_column const&) const = default;
};

/**
 * @brief A subtree of scans, INNER equality joins, filters and projections of plain references,
 * with at most one DELIM_GET: a set of rows that does not depend on its join order.
 */
struct join_region {
  LogicalOperator* root = nullptr;
  /// Each scan's identity, and the scan. A region with two scans of one identity is refused.
  std::map<std::string, duckdb::LogicalGet*> leaves;
  /// Each scan's identity by its table index.
  std::map<duckdb::idx_t, std::string> by_table;
  duckdb::LogicalDelimGet* delim = nullptr;
  /// Join equalities, by column.
  std::vector<std::pair<region_column, region_column>> equalities;
  /// Comparison of each equality with the DELIM_GET, by its column.
  std::map<duckdb::idx_t, duckdb::ExpressionType> delim_comparisons;
  /// FILTER predicates: the predicate's text with columns named by region_column, and the leaves
  /// it reads.
  std::vector<std::pair<std::string, std::set<std::string>>> filters;
};

std::string scan_identity(duckdb::LogicalGet const& get)
{
  std::string id = get.function.GetName().GetIdentifierName() + "|";
  if (auto const table = get.GetTable()) {
    id += "table@" + std::to_string(reinterpret_cast<std::uintptr_t>(table.get()));
  } else {
    if (get.parameters.empty()) { return {}; }
    for (auto const& value : get.parameters) {
      id += value.ToString() + ",";
    }
    for (auto const& [name, value] : get.named_parameters) {
      id += name + "=" + value.ToString() + ",";
    }
  }
  for (auto const& filter : required_filters(get)) {
    id += "|" + filter;
  }
  return id;
}

class region_builder {
 public:
  explicit region_builder(join_region& region) : _region(region) {}

  bool collect(LogicalOperator& op)
  {
    switch (op.type) {
      case LogicalOperatorType::LOGICAL_GET: {
        auto& get = op.Cast<duckdb::LogicalGet>();
        auto id   = scan_identity(get);
        if (id.empty() || !get.children.empty() || _region.leaves.contains(id)) { return false; }
        _region.by_table[get.table_index.index] = id;
        _region.leaves.emplace(std::move(id), &get);
        return true;
      }
      case LogicalOperatorType::LOGICAL_DELIM_GET:
        if (_region.delim != nullptr) { return false; }
        _region.delim = &op.Cast<duckdb::LogicalDelimGet>();
        return true;
      case LogicalOperatorType::LOGICAL_PROJECTION:
        return std::ranges::all_of(op.expressions,
                                   [](auto const& e) { return column_ref(*e).has_value(); }) &&
               collect(*op.children[0]);
      case LogicalOperatorType::LOGICAL_FILTER:
        if (!collect(*op.children[0])) { return false; }
        for (auto const& expression : op.expressions) {
          _filters.push_back({&op, expression.get()});
        }
        return true;
      case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
        auto& join = op.Cast<duckdb::LogicalComparisonJoin>();
        if (join.join_type != duckdb::JoinType::INNER || !collect(*op.children[0]) ||
            !collect(*op.children[1])) {
          return false;
        }
        for (auto const& condition : join.conditions) {
          if (!condition.IsComparison()) { return false; }
          auto const type = condition.GetComparisonType();
          auto const l    = column_ref(condition.GetLHS());
          auto const r    = column_ref(condition.GetRHS());
          if (!l || !r) { return false; }
          if (type != duckdb::ExpressionType::COMPARE_EQUAL &&
              type != duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
            return false;
          }
          _joins.push_back({&op, *l, *r, type});
        }
        return true;
      }
      default: return false;
    }
  }

  /// Resolves the collected joins and filters; false when one reads a computed column.
  bool finish(LogicalOperator& root)
  {
    _region.root = &root;
    for (auto const& [op, l, r, type] : _joins) {
      auto const a = column(*op, l);
      auto const b = column(*op, r);
      if (!a || !b) { return false; }
      bool const delim_a = a->leaf.empty(), delim_b = b->leaf.empty();
      if (delim_a && delim_b) { return false; }
      if (delim_a || delim_b) {
        _region.delim_comparisons[delim_a ? a->column : b->column] = type;
      } else if (type != duckdb::ExpressionType::COMPARE_EQUAL) {
        return false;
      }
      _region.equalities.emplace_back(*a, *b);
    }
    for (auto const& [op, expression] : _filters) {
      std::set<std::string> leaves;
      bool ok   = true;
      auto copy = expression->Copy();
      rename(*op, copy, leaves, ok);
      if (!ok) { return false; }
      _region.filters.emplace_back(copy->ToString(), std::move(leaves));
    }
    return true;
  }

  /// The region column that @p binding, read at @p op, holds.
  std::optional<region_column> column(LogicalOperator& op, ColumnBinding binding)
  {
    if (_region.delim != nullptr && binding.table_index == _region.delim->table_index) {
      return region_column{"", binding.column_index.GetIndex()};
    }
    auto const origin = trace(op, binding);
    if (!origin) { return std::nullopt; }
    auto const it = _region.by_table.find(origin->get->table_index.index);
    if (it == _region.by_table.end()) { return std::nullopt; }
    return region_column{it->second, origin->column};
  }

 private:
  /// Names every column @p expression reads by its region_column, so that equal predicates over
  /// equal columns print alike in two regions.
  void rename(LogicalOperator& op,
              unique_ptr<Expression>& expression,
              std::set<std::string>& leaves,
              bool& ok)
  {
    if (auto const ref = column_ref(*expression)) {
      auto const c = column(op, *ref);
      if (!c || c->leaf.empty()) {
        ok = false;
        return;
      }
      leaves.insert(c->leaf);
      expression = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        duckdb::Identifier("[" + c->leaf + "]." + std::to_string(c->column)),
        expression->GetReturnType(),
        ColumnBinding(duckdb::TableIndex(0), ProjectionIndex(0)));
      return;
    }
    duckdb::ExpressionIterator::EnumerateChildren(
      *expression, [&](unique_ptr<Expression>& child) { rename(op, child, leaves, ok); });
  }

  struct join_entry {
    LogicalOperator* op;
    ColumnBinding left, right;
    duckdb::ExpressionType type;
  };
  join_region& _region;
  std::vector<join_entry> _joins;
  std::vector<std::pair<LogicalOperator*, Expression*>> _filters;
};

std::optional<join_region> region_of(LogicalOperator& root)
{
  join_region region;
  region_builder builder(region);
  if (!builder.collect(root) || region.leaves.empty() || !builder.finish(root)) {
    return std::nullopt;
  }
  return region;
}

/// Union-find over region columns.
class column_classes {
 public:
  void join(region_column const& a, region_column const& b) { _parent[find(a)] = find(b); }
  bool same(region_column const& a, region_column const& b) { return find(a) == find(b); }
  region_column find(region_column const& c)
  {
    auto it = _parent.find(c);
    if (it == _parent.end()) {
      _parent.emplace(c, c);
      return c;
    }
    if (it->second == c) { return c; }
    auto root  = find(it->second);
    _parent[c] = root;
    return root;
  }

 private:
  std::map<region_column, region_column> _parent;
};

/// Every column binding read by an expression of an operator under @p op, outside @p skip.
void collect_plan_references(LogicalOperator& op,
                             LogicalOperator const* skip,
                             std::vector<ColumnBinding>& out)
{
  if (&op == skip) { return; }
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(
    op, [&](unique_ptr<Expression>* e) { collect_references(**e, out); });
  for (auto& child : op.children) {
    collect_plan_references(*child, skip, out);
  }
}

duckdb::idx_t next_table_index(LogicalOperator& op)
{
  duckdb::idx_t next = 0;
  for (auto const& index : op.GetTableIndex()) {
    if (index.IsValid()) { next = std::max(next, index.index + 1); }
  }
  if (op.type == LogicalOperatorType::LOGICAL_CTE_REF) {
    next = std::max(next, op.Cast<duckdb::LogicalCTERef>().cte_index.index + 1);
  }
  for (auto& child : op.children) {
    next = std::max(next, next_table_index(*child));
  }
  return next;
}

/// Finds the slot holding @p target under @p op.
unique_ptr<LogicalOperator>* slot_of(unique_ptr<LogicalOperator>& op, LogicalOperator const* target)
{
  if (op.get() == target) { return &op; }
  for (auto& child : op->children) {
    if (auto* found = slot_of(child, target)) { return found; }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Rule: a correlated MIN/MAX/AVG over the outer query's own rows (TPC-H Q17, Q2)
//===----------------------------------------------------------------------===//

struct correlated_match {
  std::size_t outer_side;
  /// The subquery's join region (with the DELIM_GET) and the outer region.
  join_region inner;
  join_region outer;
  /// For each DELIM_GET column, the outer column it was taken from (a binding of the outer).
  std::vector<ColumnBinding> delim_sources;
  /// The outer's join equalities: columns in one class hold equal values on every outer row.
  column_classes outer_classes;
};

std::optional<correlated_match> match_correlated_aggregate(duckdb::LogicalComparisonJoin& delim)
{
  auto const skip = [](char const* why) -> std::optional<correlated_match> {
    not_applied("correlated aggregate over the outer rows", why);
    return std::nullopt;
  };
  if (delim.type != LogicalOperatorType::LOGICAL_DELIM_JOIN) { return std::nullopt; }
  auto const outer_side = delim.delim_flipped ? std::size_t{1} : std::size_t{0};
  auto& outer_root      = *delim.children[outer_side];
  auto& subquery        = *delim.children[1 - outer_side];

  // The subquery: projections and filters over one aggregate of MIN, MAX and AVG only (they keep
  // their value when every row of a group repeats the same number of times), over its join region.
  LogicalOperator* at = &subquery;
  while (at->type == LogicalOperatorType::LOGICAL_PROJECTION ||
         at->type == LogicalOperatorType::LOGICAL_FILTER) {
    at = at->children[0].get();
  }
  if (at->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    return skip("no aggregate under the subquery's projections and filters");
  }
  auto& aggregate = at->Cast<duckdb::LogicalAggregate>();
  if (aggregate.grouping_sets.size() > 1 || !aggregate.grouping_functions.empty()) {
    return skip("grouping sets");
  }
  for (auto const& expression : aggregate.expressions) {
    if (expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
      return skip("an aggregate expression that is not an aggregate");
    }
    auto const& a    = expression->Cast<duckdb::BoundAggregateExpression>();
    auto const& name = a.Function().GetName();
    if (!(name == "min" || name == "max" || name == "avg") || a.IsDistinct() || a.GetFilter() ||
        a.GetOrderBys()) {
      return skip("an aggregate other than a plain MIN, MAX or AVG");
    }
  }
  auto inner = region_of(*aggregate.children[0]);
  auto outer = region_of(outer_root);
  if (!inner) { return skip("the subquery's input is not a join region"); }
  if (!outer) { return skip("the outer side is not a join region"); }
  if (inner->delim == nullptr || outer->delim != nullptr) {
    return skip("the DELIM_GET is not in the subquery's region");
  }
  correlated_match match{.outer_side = outer_side, .inner = *inner, .outer = *outer};

  // Each DELIM_GET column comes from an outer column, and its equality in the subquery.
  region_builder outer_columns(match.outer);
  std::map<duckdb::idx_t, region_column> delim_column;
  for (std::size_t i = 0; i < delim.duplicate_eliminated_columns.size(); ++i) {
    auto const ref = column_ref(*delim.duplicate_eliminated_columns[i]);
    auto const c   = ref ? outer_columns.column(outer_root, *ref) : std::nullopt;
    if (!c) { return skip("a delim column that is not an outer scan column"); }
    delim_column[i] = *c;
    match.delim_sources.push_back(*ref);
  }
  if (match.inner.delim_comparisons.size() != delim_column.size()) {
    return skip("a delim column the subquery does not join");
  }

  // The subquery's scans (R) are outer scans; the outer's other scans (E) join R only within the
  // equality classes of the delim columns.
  std::set<std::string> inner_leaves;
  for (auto const& [id, get] : match.inner.leaves) {
    if (!match.outer.leaves.contains(id)) { return skip("a subquery scan the outer side lacks"); }
    inner_leaves.insert(id);
  }
  auto const delim_as_outer = [&](region_column c) {
    return c.leaf.empty() ? delim_column.at(c.column) : c;
  };
  column_classes inner_classes, outer_classes;
  for (auto const& [a, b] : match.inner.equalities) {
    inner_classes.join(delim_as_outer(a), delim_as_outer(b));
  }
  for (auto const& [a, b] : match.outer.equalities) {
    outer_classes.join(a, b);
  }
  for (auto const& [a, b] : match.inner.equalities) {
    if (!outer_classes.same(delim_as_outer(a), delim_as_outer(b))) {
      return skip("a subquery join the outer side lacks");
    }
  }
  for (auto const& [a, b] : match.outer.equalities) {
    bool const ra = inner_leaves.contains(a.leaf), rb = inner_leaves.contains(b.leaf);
    if (!ra && !rb) { continue; }  // within E
    if (ra && rb) {
      if (!inner_classes.same(a, b)) { return skip("an outer join the subquery lacks"); }
      continue;
    }
    // R to E: only through a delim column's class, which the subquery joins to that R column.
    auto const& r            = ra ? a : b;
    auto const& e            = ra ? b : a;
    bool const through_delim = std::ranges::any_of(delim_column, [&](auto const& entry) {
      return outer_classes.same(e, entry.second) && inner_classes.same(r, entry.second);
    });
    if (!through_delim) { return skip("an outer join to R that is not on a delim column"); }
  }
  // Filters: the outer's filters on inner scans only are the subquery's; none reads both sides.
  std::multiset<std::string> outer_filters, inner_filters;
  for (auto const& [text, leaves] : match.outer.filters) {
    auto const on_inner =
      std::ranges::count_if(leaves, [&](auto const& l) { return inner_leaves.contains(l); });
    if (on_inner == 0) { continue; }
    if (static_cast<std::size_t>(on_inner) != leaves.size()) {
      return skip("an outer filter across both sides");
    }
    outer_filters.insert(text);
  }
  for (auto const& [text, leaves] : match.inner.filters) {
    inner_filters.insert(text);
  }
  if (outer_filters != inner_filters) { return skip("different filters"); }
  match.outer_classes = outer_classes;
  return match;
}

bool reuse_outer_rows(unique_ptr<LogicalOperator>& root,
                      unique_ptr<LogicalOperator>& slot,
                      scan_reuse_counts const& counts,
                      removed_subtree_callback const& removed)
{
  auto& delim = slot->Cast<duckdb::LogicalComparisonJoin>();
  auto match  = match_correlated_aggregate(delim);
  if (!match) { return false; }
  auto& outer_slot = delim.children[match->outer_side];
  auto* inner_root = match->inner.root;
  outer_slot->ResolveOperatorTypes();
  auto const outer_bindings = outer_slot->GetColumnBindings();
  auto const outer_types    = outer_slot->types;

  // Where each outer output column comes from, to find the subquery's columns among them.
  region_builder outer_columns(match->outer);
  std::vector<std::optional<region_column>> outer_output;
  for (auto const& binding : outer_bindings) {
    outer_output.push_back(outer_columns.column(*outer_slot, binding));
  }
  region_builder inner_columns(match->inner);
  std::vector<ColumnBinding> referenced;
  collect_plan_references(*root, inner_root, referenced);

  auto const tables = next_table_index(*root);
  duckdb::TableIndex const cte_index(tables), outer_ref(tables + 1), inner_ref(tables + 2);
  inner_root->ResolveOperatorTypes();
  auto const inner_bindings = inner_root->GetColumnBindings();
  // The outer output column holding @p c: c itself, else a column of its equality class (equal on
  // every outer row) of the same type.
  auto const position_in_outer =
    [&](region_column const& c, duckdb::LogicalType const& type) -> std::optional<std::size_t> {
    for (std::size_t j = 0; j < outer_output.size(); ++j) {
      if (outer_output[j] == c) { return j; }
    }
    for (std::size_t j = 0; j < outer_output.size(); ++j) {
      if (outer_output[j] && outer_types[j] == type &&
          match->outer_classes.same(*outer_output[j], c)) {
        return j;
      }
    }
    return std::nullopt;
  };
  // The subquery's columns read above its region, as the CTE's or the DELIM_GET's.
  std::vector<duckdb::ReplacementBinding> replacements;
  for (std::size_t k = 0; k < inner_bindings.size(); ++k) {
    auto const& binding = inner_bindings[k];
    if (std::find(referenced.begin(), referenced.end(), binding) == referenced.end()) { continue; }
    auto const c = inner_columns.column(*inner_root, binding);
    if (!c) {
      not_applied("correlated aggregate over the outer rows", "a subquery output not from a scan");
      return false;
    }
    if (c->leaf.empty()) {
      replacements.emplace_back(
        binding, ColumnBinding(match->inner.delim->table_index, ProjectionIndex(c->column)));
      continue;
    }
    auto const at = position_in_outer(*c, inner_root->types[k]);
    if (!at) {
      not_applied("correlated aggregate over the outer rows",
                  "a subquery column the outer side does not output: [" + c->leaf + "]." +
                    std::to_string(c->column));
      return false;
    }
    replacements.emplace_back(binding, ColumnBinding(inner_ref, ProjectionIndex(*at)));
  }
  // The delim keys, as CTE columns.
  std::vector<std::size_t> delim_positions;
  for (auto const& source : match->delim_sources) {
    auto const at = position_of(outer_bindings, source);
    if (!at) { return false; }
    delim_positions.push_back(*at);
  }

  // Nothing fails from here on.
  duckdb::vector<duckdb::Identifier> names;
  for (std::size_t j = 0; j < outer_types.size(); ++j) {
    names.emplace_back("c" + std::to_string(j));
  }
  auto* inner_slot = slot_of(slot, inner_root);
  if (removed) { removed(**inner_slot); }
  auto* delim_slot = slot_of(*inner_slot, match->inner.delim);
  auto delim_get   = std::move(*delim_slot);
  auto& delim_ref  = delim_get->Cast<duckdb::LogicalDelimGet>();
  auto joined      = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  for (std::size_t i = 0; i < delim_positions.size(); ++i) {
    auto const at = delim_positions[i];
    joined->conditions.emplace_back(
      duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        outer_types[at], ColumnBinding(inner_ref, ProjectionIndex(at))),
      duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        delim_ref.chunk_types[i], ColumnBinding(delim_ref.table_index, ProjectionIndex(i))),
      match->inner.delim_comparisons.at(i));
  }
  joined->children.push_back(
    duckdb::make_uniq<duckdb::LogicalCTERef>(inner_ref, cte_index, outer_types, names));
  joined->children.push_back(std::move(delim_get));
  joined->estimated_cardinality     = inner_root->estimated_cardinality;
  joined->has_estimated_cardinality = inner_root->has_estimated_cardinality;
  *inner_slot                       = std::move(joined);

  auto outer = std::move(outer_slot);
  outer_slot = duckdb::make_uniq<duckdb::LogicalCTERef>(outer_ref, cte_index, outer_types, names);
  outer_slot->estimated_cardinality     = outer->estimated_cardinality;
  outer_slot->has_estimated_cardinality = outer->has_estimated_cardinality;
  for (std::size_t j = 0; j < outer_bindings.size(); ++j) {
    replacements.emplace_back(outer_bindings[j], ColumnBinding(outer_ref, ProjectionIndex(j)));
  }
  auto* outer_raw = outer.get();
  auto cte        = duckdb::make_uniq<duckdb::LogicalMaterializedCTE>(
    duckdb::Identifier("scan_reuse_" + std::to_string(counts.total())),
    cte_index,
    outer_types.size(),
    std::move(outer),
    std::move(slot),
    duckdb::CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
  cte->estimated_cardinality     = cte->children[1]->estimated_cardinality;
  cte->has_estimated_cardinality = cte->children[1]->has_estimated_cardinality;
  slot                           = std::move(cte);

  duckdb::ColumnBindingReplacer replacer;
  replacer.replacement_bindings = std::move(replacements);
  replacer.stop_operator        = outer_raw;
  replacer.VisitOperator(*root);
  return true;
}

//===----------------------------------------------------------------------===//
// Rule: one CTE for two equal join regions (TPC-H Q11)
//===----------------------------------------------------------------------===//

/// Whether two regions hold the same rows: the same scans, the same join equalities (as classes,
/// whatever the join order) and the same filters.
bool same_region(join_region& a, join_region& b)
{
  if (a.delim != nullptr || b.delim != nullptr || a.leaves.size() != b.leaves.size()) {
    return false;
  }
  for (auto const& [id, get] : a.leaves) {
    if (!b.leaves.contains(id)) { return false; }
  }
  column_classes ca, cb;
  for (auto const& [x, y] : a.equalities) {
    ca.join(x, y);
  }
  for (auto const& [x, y] : b.equalities) {
    cb.join(x, y);
  }
  for (auto const& [x, y] : a.equalities) {
    if (!cb.same(x, y)) { return false; }
  }
  for (auto const& [x, y] : b.equalities) {
    if (!ca.same(x, y)) { return false; }
  }
  std::multiset<std::string> fa, fb;
  for (auto const& f : a.filters) {
    fa.insert(f.first);
  }
  for (auto const& f : b.filters) {
    fb.insert(f.first);
  }
  return fa == fb;
}

/// The first region with a join under @p op, through operators with one child.
LogicalOperator* region_below(LogicalOperator& op)
{
  LogicalOperator* at = &op;
  while (true) {
    if (at->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
      auto region = region_of(*at);
      return region && region->leaves.size() >= 2 && region->delim == nullptr ? at : nullptr;
    }
    if (at->children.size() != 1) { return nullptr; }
    at = at->children[0].get();
  }
}

/// Shares the regions under @p slot's two children when they are equal: one becomes a CTE read in
/// both places.
bool share_equal_regions(unique_ptr<LogicalOperator>& root,
                         unique_ptr<LogicalOperator>& slot,
                         scan_reuse_counts const& counts,
                         removed_subtree_callback const& removed)
{
  if (slot->children.size() != 2 || slot->type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
      slot->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE ||
      slot->type == LogicalOperatorType::LOGICAL_RECURSIVE_CTE) {
    return false;
  }
  LogicalOperator* roots[2] = {region_below(*slot->children[0]), region_below(*slot->children[1])};
  if (roots[0] == nullptr || roots[1] == nullptr) { return false; }
  auto regions = std::array{region_of(*roots[0]), region_of(*roots[1])};
  if (!same_region(*regions[0], *regions[1])) {
    not_applied("shared join", "the two joins differ");
    return false;
  }

  std::vector<ColumnBinding> referenced;
  for (auto const definition : {0, 1}) {
    auto const other = 1 - definition;
    auto& kept       = *roots[definition];
    auto& dropped    = *roots[other];
    kept.ResolveOperatorTypes();
    auto const kept_bindings = kept.GetColumnBindings();
    auto const kept_types    = kept.types;
    region_builder kept_columns(*regions[definition]);
    region_builder dropped_columns(*regions[other]);
    std::vector<std::optional<region_column>> kept_output;
    for (auto const& binding : kept_bindings) {
      kept_output.push_back(kept_columns.column(kept, binding));
    }
    referenced.clear();
    collect_plan_references(*root, &dropped, referenced);
    auto const tables = next_table_index(*root);
    duckdb::TableIndex const cte_index(tables), kept_ref(tables + 1), dropped_ref(tables + 2);
    std::vector<duckdb::ReplacementBinding> replacements;
    bool complete = true;
    for (auto const& binding : dropped.GetColumnBindings()) {
      if (std::find(referenced.begin(), referenced.end(), binding) == referenced.end()) {
        continue;
      }
      auto const c  = dropped_columns.column(dropped, binding);
      auto const at = c ? std::find(kept_output.begin(), kept_output.end(), c) : kept_output.end();
      if (at == kept_output.end()) {
        complete = false;
        break;
      }
      replacements.emplace_back(
        binding, ColumnBinding(dropped_ref, ProjectionIndex(at - kept_output.begin())));
    }
    if (!complete) { continue; }

    // Nothing fails from here on.
    duckdb::vector<duckdb::Identifier> names;
    for (std::size_t j = 0; j < kept_types.size(); ++j) {
      names.emplace_back("c" + std::to_string(j));
    }
    for (std::size_t j = 0; j < kept_bindings.size(); ++j) {
      replacements.emplace_back(kept_bindings[j], ColumnBinding(kept_ref, ProjectionIndex(j)));
    }
    auto* kept_slot    = slot_of(slot, &kept);
    auto* dropped_slot = slot_of(slot, &dropped);
    if (removed) { removed(**dropped_slot); }
    auto dropped_ref_op =
      duckdb::make_uniq<duckdb::LogicalCTERef>(dropped_ref, cte_index, kept_types, names);
    dropped_ref_op->estimated_cardinality     = dropped.estimated_cardinality;
    dropped_ref_op->has_estimated_cardinality = dropped.has_estimated_cardinality;
    *dropped_slot                             = std::move(dropped_ref_op);
    auto definition_op                        = std::move(*kept_slot);
    auto kept_ref_op =
      duckdb::make_uniq<duckdb::LogicalCTERef>(kept_ref, cte_index, kept_types, names);
    kept_ref_op->estimated_cardinality     = definition_op->estimated_cardinality;
    kept_ref_op->has_estimated_cardinality = definition_op->has_estimated_cardinality;
    *kept_slot                             = std::move(kept_ref_op);
    auto* definition_raw                   = definition_op.get();
    auto cte                               = duckdb::make_uniq<duckdb::LogicalMaterializedCTE>(
      duckdb::Identifier("scan_reuse_" + std::to_string(counts.total())),
      cte_index,
      kept_types.size(),
      std::move(definition_op),
      std::move(slot),
      duckdb::CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
    cte->estimated_cardinality     = cte->children[1]->estimated_cardinality;
    cte->has_estimated_cardinality = cte->children[1]->has_estimated_cardinality;
    slot                           = std::move(cte);
    duckdb::ColumnBindingReplacer replacer;
    replacer.replacement_bindings = std::move(replacements);
    replacer.stop_operator        = definition_raw;
    replacer.VisitOperator(*root);
    return true;
  }
  not_applied("shared join", "neither join outputs every column read from the other");
  return false;
}

void rewrite(duckdb::ClientContext& context,
             unique_ptr<LogicalOperator>& root,
             unique_ptr<LogicalOperator>& op,
             scan_reuse_counts& counts,
             removed_subtree_callback const& removed)
{
  for (auto& child : op->children) {
    rewrite(context, root, child, counts, removed);
  }
  if (op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
      reuse_semi_join_aggregate(context, op->Cast<duckdb::LogicalAggregate>(), removed)) {
    ++counts.semi_join_aggregates;
  } else if (op->type == LogicalOperatorType::LOGICAL_DELIM_JOIN &&
             reuse_outer_rows(root, op, counts, removed)) {
    ++counts.correlated_aggregates;
  } else if (share_equal_regions(root, op, counts, removed)) {
    ++counts.shared_regions;
  }
}

}  // namespace

scan_reuse_counts rewrite_repeated_scans(duckdb::ClientContext& context,
                                         unique_ptr<LogicalOperator>& plan,
                                         removed_subtree_callback const& removed)
{
  scan_reuse_counts counts;
  if (!plan) { return counts; }
  rewrite(context, plan, plan, counts, removed);
  if (counts.total() != 0) {
    SIRIUS_LOG_INFO(
      "[scan reuse] rewrote the plan: {} IN-subquery aggregate(s) reused, {} correlated "
      "aggregate(s) over the outer rows, {} join(s) shared",
      counts.semi_join_aggregates,
      counts.correlated_aggregates,
      counts.shared_regions);
  }
  return counts;
}

}  // namespace sirius::planner
