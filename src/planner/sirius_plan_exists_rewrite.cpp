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
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
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
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb_table_identity.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"
#include "planner/aggregate_binding.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_flat_tree.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
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

/// The pieces of a correlated `EXISTS (... WHERE t.k = outer.k AND t.x <> outer.x)` as DuckDB
/// decorrelated it. Either comparison may also be its null-safe form (`IS [NOT] DISTINCT FROM`).
struct exists_match {
  duckdb::LogicalComparisonJoin* join = nullptr;  ///< The join of the outer rows and T, replaced
  std::size_t outer_child             = 0;        ///< Index of the outer rows among its children
  bool negated                        = false;    ///< NOT EXISTS
  unique_ptr<LogicalOperator>* t      = nullptr;  ///< Owning slot of T, the subquery rows
  ColumnBinding t_key;                            ///< Column of T equal to the key
  ColumnBinding t_value;                          ///< Column of T compared with `<>`
  ColumnBinding outer_key;
  ColumnBinding outer_value;
  LogicalType key_type;
  LogicalType value_type;
  bool key_null_safe   = false;  ///< `IS NOT DISTINCT FROM` rather than `=`
  bool value_null_safe = false;  ///< `IS DISTINCT FROM` rather than `<>`
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

/// A SEMI or ANTI join of the outer rows and T on exactly one key and one value comparison, as
/// DuckDB's optimizer leaves a decorrelated EXISTS, or as a client writes it. Every check is
/// structural; anything unexpected declines and leaves DuckDB's plan alone.
std::optional<exists_match> match_exists(LogicalOperator& node)
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
  match.join        = &join;
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

/// Positions in @p outer's output of every column @p replaced emits. Null when it emits anything
/// else.
std::optional<duckdb::vector<ProjectionIndex>> output_map(LogicalOperator& replaced,
                                                          LogicalOperator& outer)
{
  auto const output    = replaced.GetColumnBindings();
  auto const available = outer.GetColumnBindings();
  duckdb::vector<ProjectionIndex> map;
  for (auto const& binding : output) {
    auto const found = std::find(available.begin(), available.end(), binding);
    if (found == available.end()) { return std::nullopt; }
    map.emplace_back(static_cast<duckdb::idx_t>(found - available.begin()));
  }
  return map;
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

/// A read of the CTE @p cte_index, which holds the rows of @p outer.
unique_ptr<duckdb::LogicalCTERef> read_rows(duckdb::TableIndex scan_index,
                                            duckdb::TableIndex cte_index,
                                            LogicalOperator const& outer)
{
  duckdb::vector<duckdb::Identifier> names;
  for (std::size_t i = 0; i < outer.types.size(); ++i) {
    names.emplace_back("__sirius_exists_rows_" + std::to_string(i));
  }
  auto scan = duckdb::make_uniq<duckdb::LogicalCTERef>(scan_index, cte_index, outer.types, names);
  scan->SetEstimatedCardinality(outer.estimated_cardinality);
  return scan;
}

/// The replacement of the matched nodes, and the bindings it gives the outer rows' columns.
struct rewritten {
  unique_ptr<LogicalOperator> node;
  LogicalOperator* outer = nullptr;  ///< The outer rows, now the CTE's definition
  duckdb::vector<ColumnBinding> outer_columns;
  duckdb::vector<ColumnBinding> read_columns;  ///< The same columns, read from the CTE
};

/// Replace the matched nodes with the shape DuckDB itself produces for a correlated scalar
/// aggregate under `delim_join_as_cte`:
///
///   CTE rows = outer rows
///    └─ FILTER  EXISTS:     mn <> rows.x OR mx <> rows.x
///               NOT EXISTS: (that) IS NULL OR NOT (that)
///        └─ LEFT JOIN on rows.k IS NOT DISTINCT FROM a.k
///            ├─ CTE_SCAN rows
///            └─ a = AGGREGATE(GROUP BY k: MIN(x), MAX(x) per subquery)
///                 ← SEMI JOIN(T, AGGREGATE(GROUP BY k) ← CTE_SCAN rows) on t.k = k
///
/// `EXISTS ... t.x <> outer.x` holds exactly when MIN or MAX of `t.x` over the key differs from
/// `outer.x`, with NULL comparisons never counting as different; NOT EXISTS keeps the rest,
/// including rows with no group. For `IS DISTINCT FROM`, the aggregate also records whether a NULL
/// `t.x` was seen: a NULL `outer.x` differs from any non-NULL `t.x`, and a non-NULL one from a
/// NULL `t.x` or from MIN or MAX. A subquery's own filters move into its aggregates as FILTERs.
/// The filter's projection map emits the outer columns the replaced node emitted, read from the
/// CTE.
rewritten build(duckdb::ClientContext& context,
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
  outer->ResolveOperatorTypes();
  auto const outer_estimate = outer->estimated_cardinality;
  auto const cte_index      = binder.GenerateTableIndex();
  auto const rows_index     = binder.GenerateTableIndex();
  auto const keys_index     = binder.GenerateTableIndex();

  rewritten result;
  result.outer         = outer.get();
  result.outer_columns = outer->GetColumnBindings();
  duckdb::column_binding_map_t<ProjectionIndex> position;
  for (std::size_t i = 0; i < result.outer_columns.size(); ++i) {
    position.emplace(result.outer_columns[i], ProjectionIndex(i));
    result.read_columns.emplace_back(rows_index, ProjectionIndex(i));
  }
  auto const rows_column = [&](ColumnBinding outer_column) {
    return ColumnBinding(rows_index, position.at(outer_column));
  };

  auto const distinct_index = binder.GenerateTableIndex();
  auto distinct             = duckdb::make_uniq<duckdb::LogicalAggregate>(
    distinct_index, binder.GenerateTableIndex(), duckdb::vector<unique_ptr<Expression>>{});
  distinct->groups.push_back(column(key_type, ColumnBinding(keys_index, position.at(outer_key))));
  distinct->grouping_sets.push_back(duckdb::GroupingSet{ProjectionIndex(0)});
  distinct->group_stats.resize(1);
  distinct->distinct_validity = duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES;
  distinct->SetEstimatedCardinality(outer_estimate);
  distinct->children.push_back(read_rows(keys_index, cte_index, *outer));

  auto joined = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::SEMI);
  joined->conditions.emplace_back(
    column(key_type, t_key),
    column(key_type, ColumnBinding(distinct_index, ProjectionIndex(0))),
    key_null_safe ? ExpressionType::COMPARE_NOT_DISTINCT_FROM : ExpressionType::COMPARE_EQUAL);
  joined->SetEstimatedCardinality(t->estimated_cardinality);
  joined->children.push_back(std::move(t));
  joined->children.push_back(std::move(distinct));

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
      aggregates.push_back(
        bind_builtin_aggregate(context, "max", LogicalType::INTEGER, std::move(seen)));
    }
    // As FILTERs, MIN and MAX share one filtered input column in the physical plan.
    aggregates.push_back(bind_builtin_aggregate(
      context, "min", type, sub.value->Copy(), sub.predicate ? sub.predicate->Copy() : nullptr));
    aggregates.push_back(
      bind_builtin_aggregate(context, "max", type, std::move(sub.value), std::move(sub.predicate)));
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

  auto left = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::LEFT);
  left->conditions.emplace_back(column(key_type, rows_column(outer_key)),
                                column(key_type, ColumnBinding(group_index, ProjectionIndex(0))),
                                ExpressionType::COMPARE_NOT_DISTINCT_FROM);
  left->SetEstimatedCardinality(outer_estimate);
  left->children.push_back(read_rows(rows_index, cte_index, *outer));
  left->children.push_back(std::move(grouped));

  unique_ptr<Expression> predicate;
  for (std::size_t i = 0; i < subqueries.size(); ++i) {
    auto const& sub      = subqueries[i];
    auto const aggregate = [&](std::size_t offset, LogicalType const& type) {
      return column(type, ColumnBinding(aggregate_index, ProjectionIndex(first[i] + offset)));
    };
    auto const outer_x  = [&] { return column(types[i], rows_column(sub.outer_value)); };
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

  auto const column_count = outer->types.size();
  auto cte                = duckdb::make_uniq<duckdb::LogicalMaterializedCTE>(
    duckdb::Identifier("__sirius_exists_rows_" + std::to_string(cte_index.index)),
    cte_index,
    column_count,
    std::move(outer),
    std::move(filter),
    duckdb::CTEMaterialize::CTE_MATERIALIZE_ALWAYS);
  cte->SetEstimatedCardinality(output_estimate);
  result.node = std::move(cte);
  result.node->ResolveOperatorTypes();
  return result;
}

// ---------------------------------------------------------------------------------------------
// Merging sibling subqueries
//
// `EXISTS (... l2.k = l1.k AND l2.x <> l1.x) AND NOT EXISTS (... l3.k = l1.k AND l3.x <> l1.x AND
// <p>)` become two nested SEMI/ANTI joins, each with a scan of the same table. When the scans read
// the same source, one scan serves both: each subquery's own filter turns into a CASE inside its
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

/// Whether @p get is served from a table pinned in GPU memory that holds every column it reads.
bool served_from_gpu_pin(duckdb::ClientContext& context, duckdb::LogicalGet const& get)
{
  auto const runtime = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!runtime) { return false; }
  auto& scan_manager = runtime->get_scan_manager();
  std::shared_ptr<sirius::scan_manager::pinned_entry const> entry;
  if (get.function.GetName() == "seq_scan") {
    auto const* bind = dynamic_cast<duckdb::TableScanBindData const*>(get.bind_data.get());
    if (bind == nullptr || !bind->table.IsDuckTable()) { return false; }
    auto& table = bind->table.Cast<duckdb::DuckTableEntry>();
    entry       = scan_manager.find_pinned_entry_for_duckdb_table(
      table.ParentCatalog().GetName().GetIdentifierName(),
      table.ParentSchema().name.GetIdentifierName(),
      table.name.GetIdentifierName(),
      sirius::duckdb_table_identity{table.oid, table.GetStorage().GetRowGroupCollection()});
  } else {
    // Other sources, such as Iceberg tables that map columns by field id, may refuse raw cached
    // rows even for the same files.
    auto const name = get.function.GetName().GetIdentifierName();
    if (name != "read_parquet" && name != "parquet_scan" && name != "sirius_read_parquet") {
      return false;
    }
    auto const files = resolve_parquet_scan_file_paths(name, get.bind_data.get(), get.parameters);
    if (files.empty()) { return false; }
    entry = scan_manager.find_pinned_entry_for_parquet_files(files);
  }
  return entry && entry->tier == cucascade::memory::Tier::GPU &&
         !entry->cache_info.column_projection_for(get.GetColumnIds()).empty();
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
/// nested in the next through its outer rows. Returns no node, consuming nothing, when the
/// subqueries cannot share one scan, or when their table is pinned in GPU memory and their joins
/// are estimated to fit in @p gpu_memory_budget bytes (0: unknown).
rewritten rewrite_merged(duckdb::ClientContext& context,
                         duckdb::Binder& binder,
                         std::vector<exists_match>& chain,
                         std::uint64_t gpu_memory_budget)
{
  std::vector<peeled_scan> scans;
  for (auto& match : chain) {
    auto scan = peel_scan(*match.t);
    if (!scan) { return {}; }
    scans.push_back(std::move(*scan));
  }

  // One source and one correlation key; every column a subquery reads comes from its own scan, and
  // every outer value it compares is a column of the innermost outer rows.
  std::vector<ColumnBinding> t_keys, t_values;
  for (std::size_t i = 0; i < chain.size(); ++i) {
    auto const key   = see_through(scans[i].projections, chain[i].t_key);
    auto const value = see_through(scans[i].projections, chain[i].t_value);
    if (!key || !value) { return {}; }
    t_keys.push_back(*key);
    t_values.push_back(*value);
  }
  auto& innermost = chain.front();
  auto& outer     = innermost.join->children[innermost.outer_child];
  auto& shared    = *scans.front().get;
  if (!reads_only(*column(innermost.key_type, t_keys[0]), shared)) { return {}; }
  auto const key_base    = *base_column(shared, t_keys[0].column_index);
  auto const outer_names = outer->GetColumnBindings();
  if (std::find(outer_names.begin(), outer_names.end(), innermost.outer_key) == outer_names.end()) {
    return {};
  }
  for (std::size_t i = 0; i < chain.size(); ++i) {
    auto const& get = *scans[i].get;
    if (!flat::same_source(context, shared, get) || chain[i].key_type != innermost.key_type ||
        !(chain[i].outer_key == innermost.outer_key) ||
        !reads_only(*column(chain[i].key_type, t_keys[i]), get) ||
        base_column(get, t_keys[i].column_index) != key_base ||
        !reads_only(*column(chain[i].value_type, t_values[i]), get) ||
        std::find(outer_names.begin(), outer_names.end(), chain[i].outer_value) ==
          outer_names.end() ||
        !std::ranges::all_of(scans[i].predicates,
                             [&](auto const& predicate) { return reads_only(*predicate, get); })) {
      return {};
    }
  }
  auto output = output_map(*chain.back().join, *outer);
  if (!output) { return {}; }
  if (gpu_memory_budget > 0) {
    // The joins each scan their subquery's table and hash its rows. That beats the merged
    // aggregate only when those scans are cheap, from a pin in GPU memory, and the rows fit
    // without spilling. The base-table row count is used because estimates of the subqueries'
    // filters can be far too low.
    bool pinned              = true;
    std::uint64_t join_bytes = 0;
    for (std::size_t i = 0; i < chain.size(); ++i) {
      auto const& get = *scans[i].get;
      pinned          = pinned && served_from_gpu_pin(context, get);
      if (!get.has_estimated_cardinality) { continue; }
      auto const row_bytes = duckdb::GetTypeIdSize(chain[i].key_type.InternalType()) +
                             duckdb::GetTypeIdSize(chain[i].value_type.InternalType());
      join_bytes = std::max<std::uint64_t>(join_bytes, get.estimated_cardinality * row_bytes);
    }
    bool const merge = !pinned || join_bytes == 0 || join_bytes > gpu_memory_budget;
    SIRIUS_LOG_INFO(
      "[sirius_plan_exists_rewrite] {} {} sibling EXISTS/NOT EXISTS: subquery table {} in GPU "
      "memory, their joins would hold an estimated {} bytes against a GPU budget of {} bytes",
      merge ? "Merging" : "Not merging",
      chain.size(),
      pinned ? "pinned" : "not pinned",
      join_bytes,
      gpu_memory_budget);
    if (!merge) { return {}; }
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
               chain.back().join->estimated_cardinality);
}

/// The longest chain of matched joins directly nested through their outer child, innermost
/// first.
std::vector<exists_match> match_chain(LogicalOperator& node)
{
  std::vector<exists_match> outermost_first;
  auto* current = &node;
  while (auto match = match_exists(*current)) {
    outermost_first.push_back(*match);
    current = match->join->children[match->outer_child].get();
  }
  return {outermost_first.rbegin(), outermost_first.rend()};
}

void rewrite_tree(duckdb::ClientContext& context,
                  duckdb::Binder& binder,
                  unique_ptr<LogicalOperator>& root,
                  unique_ptr<LogicalOperator>& slot,
                  std::uint64_t gpu_memory_budget,
                  std::size_t& rewrites)
{
  // Only sibling subqueries are rewritten: a lone one trades its join for a GROUP BY over the
  // same rows, which is rarely faster.
  if (auto chain = match_chain(*slot); chain.size() >= 2) {
    if (auto merged = rewrite_merged(context, binder, chain, gpu_memory_budget); merged.node) {
      slot = std::move(merged.node);
      rewrites += chain.size();
      // Everything above that read the outer rows' columns reads them from the CTE.
      duckdb::ColumnBindingReplacer replacer;
      replacer.AddReplacements(merged.outer_columns, merged.read_columns);
      replacer.stop_operator = merged.outer;
      replacer.VisitOperator(*root);
      // The outer rows, now the CTE's definition, may hold further subqueries.
      rewrite_tree(context, binder, root, slot->children[0], gpu_memory_budget, rewrites);
      return;
    }
  }
  for (auto& child : slot->children) {
    rewrite_tree(context, binder, root, child, gpu_memory_budget, rewrites);
  }
}

}  // namespace

std::size_t rewrite_correlated_inequality_exists(duckdb::ClientContext& context,
                                                 duckdb::Binder& binder,
                                                 duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                                                 std::uint64_t gpu_memory_budget)
{
  std::size_t rewrites = 0;
  rewrite_tree(context, binder, plan, plan, gpu_memory_budget, rewrites);
  if (rewrites > 0) {
    SIRIUS_LOG_INFO(
      "[sirius_plan_exists_rewrite] Rewrote {} EXISTS/NOT EXISTS on a key and a value inequality "
      "into a MIN/MAX aggregate per key",
      rewrites);
  }
  return rewrites;
}

}  // namespace sirius::planner
