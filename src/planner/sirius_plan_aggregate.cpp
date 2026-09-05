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

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <memory>
#include <optional>

namespace sirius::planner {

namespace {

// Translate a vector of DuckDB expressions into Sirius AST nodes at the planner
// boundary. The source vector is drained; size and order are preserved. If
// from_duckdb declines an unsupported shape (e.g. an ORDER BY aggregate rewritten
// to arg_min_null / create_sort_key), it returns null — which the downstream
// aggregate/projection operators cannot represent. Rather than build a GPU plan
// containing null nodes (which crashes at execution time), throw here so the
// query falls back to DuckDB's CPU execution.
duckdb::vector<std::unique_ptr<sirius::ast::node>> translate_expressions(
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> exprs)
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> out;
  out.reserve(exprs.size());
  for (auto& e : exprs) {
    auto translated = e ? sirius::ast::from_duckdb(*e) : nullptr;
    if (e && translated == nullptr) {
      throw duckdb::NotImplementedException(
        "Unsupported expression in aggregate (falling back to CPU): " + e->ToString());
    }
    out.push_back(std::move(translated));
  }
  return out;
}

// File-local helper (formerly sirius_physical_plan_generator::extract_aggregate_expressions).
// Pulls aggregate child / filter sub-expressions out of the aggregate list and groups into a
// projection fed upstream of the aggregate. Operates on raw DuckDB expressions so the hoist
// is straightforward; the caller translates the resulting groups/aggregates into Sirius AST
// nodes when constructing the aggregate operator.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> extract_aggregate_expressions(
  duckdb::ClientContext& context,
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> child,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& aggregates,
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& groups,
  duckdb::optional_ptr<duckdb::vector<duckdb::GroupingSet>> grouping_sets)
{
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> expressions;
  duckdb::vector<duckdb::LogicalType> types;

  // bind sorted aggregates
  for (auto& aggr : aggregates) {
    auto& bound_aggr = aggr->Cast<duckdb::BoundAggregateExpression>();
    if (bound_aggr.order_bys) {
      duckdb::FunctionBinder::BindSortedAggregate(context, bound_aggr, groups, grouping_sets);
    }
  }
  for (auto& group : groups) {
    auto ref =
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(group->return_type, expressions.size());
    types.push_back(group->return_type);
    expressions.push_back(std::move(group));
    group = std::move(ref);
  }
  for (auto& aggr : aggregates) {
    auto& bound_aggr = aggr->Cast<duckdb::BoundAggregateExpression>();
    for (auto& child_expr : bound_aggr.children) {
      auto ref = duckdb::make_uniq<duckdb::BoundReferenceExpression>(child_expr->return_type,
                                                                     expressions.size());
      types.push_back(child_expr->return_type);
      expressions.push_back(std::move(child_expr));
      child_expr = std::move(ref);
    }
    if (bound_aggr.filter) {
      auto& filter = bound_aggr.filter;
      auto ref     = duckdb::make_uniq<duckdb::BoundReferenceExpression>(filter->return_type,
                                                                     expressions.size());
      types.push_back(filter->return_type);
      expressions.push_back(std::move(filter));
      bound_aggr.filter = std::move(ref);
    }
  }
  if (expressions.empty()) { return child; }
  auto const estimated_cardinality = child->estimated_cardinality;
  return push_projection(std::move(child),
                         sirius::from_duckdb_vec(types),
                         translate_expressions(std::move(expressions)),
                         estimated_cardinality);
}

/// Column positions a fused COUNT-join reads from the two join inputs. Each index addresses the
/// output of the child it belongs to.
struct dense_count_join_detection {
  std::size_t preserved_child;
  std::size_t counted_child;
  std::size_t preserved_key_idx;
  std::size_t counted_key_idx;
  std::optional<std::size_t> counted_value_idx;  ///< nullopt for COUNT(*)
};

/// The number of output columns contributed by each side of a join
struct join_projection_layout {
  std::size_t left_output_count;
  std::size_t right_output_count;
};

static join_projection_layout join_output_layout(duckdb::LogicalComparisonJoin const& join)
{
  auto const left_width  = join.children[0]->types.size();
  auto const right_width = join.children[1]->types.size();
  D_ASSERT(std::ranges::all_of(join.left_projection_map,
                               [left_width](auto const index) { return index < left_width; }));
  D_ASSERT(std::ranges::all_of(join.right_projection_map,
                               [right_width](auto const index) { return index < right_width; }));
  return join_projection_layout{
    join.left_projection_map.empty() ? left_width : join.left_projection_map.size(),
    join.right_projection_map.empty() ? right_width : join.right_projection_map.size()};
}

// Convert a column position in the join's output [left, right] back into {child idx, original
// child col idx}.
static std::pair<std::size_t, std::size_t> resolve_join_output_column(
  duckdb::LogicalComparisonJoin const& join,
  join_projection_layout const& layout,
  std::size_t position)
{
  D_ASSERT(position < layout.left_output_count + layout.right_output_count);
  if (position < layout.left_output_count) {
    return {0, join.left_projection_map.empty() ? position : join.left_projection_map[position]};
  }
  auto const right_pos = position - layout.left_output_count;
  return {1, join.right_projection_map.empty() ? right_pos : join.right_projection_map[right_pos]};
}

// Whether a column-binding-resolved reference addresses an existing column of `types` and carries
// that column's type; assertion-only.
[[maybe_unused]] static bool ref_matches(duckdb::BoundReferenceExpression const& ref,
                                         duckdb::vector<duckdb::LogicalType> const& types)
{
  return ref.index < types.size() && ref.return_type == types[ref.index];
}

// Whether a subtree contains a LOGICAL_DELIM_JOIN at any depth.
static bool contains_delim_join(duckdb::LogicalOperator const& node)
{
  return node.type == duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN ||
         std::ranges::any_of(node.children,
                             [](auto const& child) { return contains_delim_join(*child); });
}

// Whether a join input can feed a DENSE_COUNT_JOIN port. Two rules apply to the input's logical
// subtree.
//
// Root rule: a DELIM_JOIN or MATERIALIZED_CTE root streams its consumer-side output into the
// enclosing pipeline, and a DELIM_GET or CTE_REF root is a routing-only scan fed by a producer
// outside the join input, so none of them delivers the input through a child-owned pipeline.
// Identity projections are elided during planning, so the root is found by looking through
// projections.
//
// Depth rule: a DELIM_JOIN is excluded anywhere in the subtree.
// sirius_physical_dense_count_join::get_next_task_hint directs the task creator into the source
// operator of an unfinished producer pipeline, which can happen before the delim subtree's sizing
// partitions have negotiated a join mode, and a MARK hash join polled before sizing throws in
// sirius_physical_hash_join::refresh_cross_schedule instead of deferring.
static bool can_feed_dense_count_join(duckdb::LogicalOperator const& input)
{
  if (contains_delim_join(input)) { return false; }
  auto const* root = &input;
  while (root->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    root = root->children[0].get();
  }
  switch (root->type) {
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET:
    case duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
    case duckdb::LogicalOperatorType::LOGICAL_CTE_REF: return false;
    default: return true;
  }
}

// Recognize an aggregate whose public shape is the built-in COUNT(col) or COUNT(*): a name that
// maps to a count aggregate_id, matching arity, a BIGINT result, and no bind data. Whether the
// bound callbacks are the host's is settled by is_host_builtin_count once every cheaper check has
// passed.
static std::optional<sirius::aggregate_id> builtin_count_candidate_id(
  duckdb::BoundAggregateExpression const& aggr)
{
  auto const aggregate_id = sirius::from_duckdb_aggregate_name(aggr.function.name);
  if (!aggregate_id || (*aggregate_id != sirius::aggregate_id::count &&
                        *aggregate_id != sirius::aggregate_id::count_star)) {
    return std::nullopt;
  }
  std::size_t const expected_arity = *aggregate_id == sirius::aggregate_id::count ? 1 : 0;
  if (aggr.children.size() != expected_arity || aggr.return_type != duckdb::LogicalType::BIGINT ||
      aggr.bind_info != nullptr) {
    return std::nullopt;
  }
  return aggregate_id;
}

// Confirm a candidate COUNT is the built-in one by comparing its callbacks against the host system
// catalog's overload of the same arity. The loadable extension carries its own hidden DuckDB copy,
// so callbacks obtained from this DSO (e.g. CountStarFun::GetFunction()) have different addresses
// from the host-bound ones; only two host-owned function objects compare equal. This also rejects
// user aggregates that reuse COUNT's name.
static bool is_host_builtin_count(duckdb::ClientContext& context,
                                  duckdb::AggregateFunction const& function,
                                  std::size_t arity)
{
  auto const& overloads =
    duckdb::Catalog::GetSystemCatalog(context)
      .GetEntry<duckdb::AggregateFunctionCatalogEntry>(context, DEFAULT_SCHEMA, "count")
      .functions.functions;
  auto const canonical = std::ranges::find_if(
    overloads, [arity](auto const& candidate) { return candidate.arguments.size() == arity; });
  return canonical != overloads.end() && function == *canonical;
}

// The join an aggregate counts over, looking through the column-only projections between them,
// plus the map from the aggregate's input columns to the join's output columns. A projection of
// bare references only renames and reorders; the StarRocks translator emits one under every
// AGGREGATION_NODE whose child is an exchange or a join (its PROJECT emit), so q13's
// COUNT-join arrives as AGGREGATE -> PROJECT -> JOIN and never fused. A projection that computes
// anything ends the search: the aggregate would then count a derived column.
struct counted_join {
  duckdb::LogicalComparisonJoin const* join;
  // aggregate input column -> join output column
  std::vector<duckdb::idx_t> to_join_column;
};

static std::optional<counted_join> join_under_pure_projections(duckdb::LogicalOperator const& child)
{
  auto const* node = &child;
  std::vector<duckdb::idx_t> map;
  bool identity = true;
  while (node->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    auto const& projection = node->Cast<duckdb::LogicalProjection>();
    std::vector<duckdb::idx_t> layer;
    layer.reserve(projection.expressions.size());
    for (auto const& expression : projection.expressions) {
      if (expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
        return std::nullopt;
      }
      layer.push_back(expression->Cast<duckdb::BoundReferenceExpression>().index);
    }
    if (identity) {
      map      = std::move(layer);
      identity = false;
    } else {
      for (auto& column : map) {
        if (column >= layer.size()) { return std::nullopt; }
        column = layer[column];
      }
    }
    node = node->children[0].get();
  }
  if (node->type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return std::nullopt; }
  auto const& join = node->Cast<duckdb::LogicalComparisonJoin>();
  if (identity) {
    map.resize(join.types.size());
    for (duckdb::idx_t i = 0; i < map.size(); ++i) {
      map[i] = i;
    }
  } else {
    for (auto column : map) {
      if (column >= join.types.size()) { return std::nullopt; }
    }
  }
  return counted_join{&join, std::move(map)};
}

static std::optional<dense_count_join_detection> detect_dense_count_join(
  duckdb::ClientContext& context, duckdb::LogicalAggregate const& op)
{
  if (op.groups.size() != 1 || op.grouping_sets.size() != 1 || op.grouping_sets[0].size() != 1 ||
      !op.grouping_functions.empty() || op.expressions.size() != 1) {
    return std::nullopt;
  }
  if (op.groups[0]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
      op.expressions[0]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
    return std::nullopt;
  }
  auto const& aggr        = op.expressions[0]->Cast<duckdb::BoundAggregateExpression>();
  auto const aggregate_id = builtin_count_candidate_id(aggr);
  if (!aggregate_id || aggr.IsDistinct() || aggr.filter || aggr.order_bys) { return std::nullopt; }
  bool const is_count = *aggregate_id == sirius::aggregate_id::count;
  if (is_count && aggr.children[0]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }

  auto const counted = join_under_pure_projections(*op.children[0]);
  if (!counted) { return std::nullopt; }
  auto const& join           = *counted->join;
  auto const& to_join_column = counted->to_join_column;
  D_ASSERT(join.children.size() == 2);
  if (join.join_type != duckdb::JoinType::LEFT && join.join_type != duckdb::JoinType::RIGHT) {
    return std::nullopt;
  }
  if (join.conditions.size() != 1 || join.predicate) { return std::nullopt; }
  for (auto const& child : join.children) {
    if (!can_feed_dense_count_join(*child)) { return std::nullopt; }
  }

  auto const& cond = join.conditions[0];
  if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL ||
      cond.left->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
      cond.right->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  auto const& left_ref  = cond.left->Cast<duckdb::BoundReferenceExpression>();
  auto const& right_ref = cond.right->Cast<duckdb::BoundReferenceExpression>();
  D_ASSERT(ref_matches(left_ref, join.children[0]->types));
  D_ASSERT(ref_matches(right_ref, join.children[1]->types));
  auto const key_type_id = left_ref.return_type.id();
  if (key_type_id != right_ref.return_type.id() || (key_type_id != duckdb::LogicalTypeId::INTEGER &&
                                                    key_type_id != duckdb::LogicalTypeId::BIGINT)) {
    return std::nullopt;
  }

  std::size_t const preserved_child = join.join_type == duckdb::JoinType::LEFT ? 0 : 1;
  std::size_t const counted_child   = 1 - preserved_child;
  auto const& preserved_ref         = preserved_child == 0 ? left_ref : right_ref;
  auto const& counted_ref           = preserved_child == 0 ? right_ref : left_ref;
  auto const layout                 = join_output_layout(join);
  D_ASSERT(join.types.size() == layout.left_output_count + layout.right_output_count);

  auto const& group_ref = op.groups[0]->Cast<duckdb::BoundReferenceExpression>();
  if (group_ref.index >= to_join_column.size()) { return std::nullopt; }
  auto const group_join_column = to_join_column[group_ref.index];
  D_ASSERT(group_join_column < join.types.size());
  auto const [group_child, group_col] = resolve_join_output_column(join, layout, group_join_column);
  if (group_child != preserved_child || group_col != preserved_ref.index) { return std::nullopt; }

  // COUNT(col): the argument must come from the counted side (a preserved-side or computed
  // argument has different NULL semantics under the outer join).
  std::optional<std::size_t> counted_value_idx;
  if (is_count) {
    auto const& count_ref = aggr.children[0]->Cast<duckdb::BoundReferenceExpression>();
    if (count_ref.index >= to_join_column.size()) { return std::nullopt; }
    auto const count_join_column = to_join_column[count_ref.index];
    D_ASSERT(count_join_column < join.types.size());
    auto const [count_child, count_col] =
      resolve_join_output_column(join, layout, count_join_column);
    if (count_child != counted_child) { return std::nullopt; }
    counted_value_idx = count_col;
  }

  if (!is_host_builtin_count(context, aggr.function, aggr.children.size())) { return std::nullopt; }
  return dense_count_join_detection{
    preserved_child, counted_child, preserved_ref.index, counted_ref.index, counted_value_idx};
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::try_plan_dense_count_join(duckdb::LogicalAggregate& op)
{
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  if (!sirius_ctx) { return nullptr; }
  auto const& op_params = sirius_ctx->get_config().get_operator_params();
  if (!op_params.enable_dense_count_join) { return nullptr; }

  auto const detection = detect_dense_count_join(context, op);
  if (!detection) { return nullptr; }
  // The projections between the aggregate and the join only renamed and reordered columns the
  // fused operator addresses by join-child index, so they are skipped along with the join.
  duckdb::LogicalOperator* under = op.children[0].get();
  while (under->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    under = under->children[0].get();
  }
  auto& join = under->Cast<duckdb::LogicalComparisonJoin>();

  // Mirror plan_comparison_join: capture cardinalities before create_plan drains the nodes.
  auto const preserved_cardinality =
    join.children[detection->preserved_child]->EstimateCardinality(context);
  auto const counted_cardinality =
    join.children[detection->counted_child]->EstimateCardinality(context);

  auto preserved                   = create_plan(*join.children[detection->preserved_child]);
  auto counted                     = create_plan(*join.children[detection->counted_child]);
  preserved->estimated_cardinality = preserved_cardinality;
  counted->estimated_cardinality   = counted_cardinality;

  SIRIUS_LOG_INFO(
    "[sirius_plan_aggregate] Fusing COUNT-join into DENSE_COUNT_JOIN: {} join, preserved child "
    "{} (key col {}, est {} rows), counted child {} (key col {}, {}, est {} rows)",
    join.join_type == duckdb::JoinType::LEFT ? "LEFT" : "RIGHT",
    detection->preserved_child,
    detection->preserved_key_idx,
    preserved_cardinality,
    detection->counted_child,
    detection->counted_key_idx,
    detection->counted_value_idx
      ? "COUNT(col " + std::to_string(*detection->counted_value_idx) + ")"
      : std::string("COUNT(*)"),
    counted_cardinality);

  auto fused = duckdb::make_uniq<sirius::op::sirius_physical_dense_count_join>(
    sirius::from_duckdb_vec(op.types),
    op.estimated_cardinality,
    detection->preserved_key_idx,
    detection->counted_key_idx,
    detection->counted_value_idx,
    op_params.dense_count_join_max_bytes,
    op_params.hash_partition_bytes);
  fused->children.push_back(std::move(preserved));
  fused->children.push_back(std::move(counted));
  return fused;
}

namespace {

static uint32_t required_bits_for_value(uint32_t n)
{
  std::size_t required_bits = 0;
  while (n > 0) {
    n >>= 1;
    required_bits++;
  }
  return duckdb::UnsafeNumericCast<uint32_t>(required_bits);
}

template <class T>
duckdb::hugeint_t get_range_hugeint(const duckdb::BaseStatistics& nstats)
{
  return duckdb::Hugeint::Convert(duckdb::NumericStats::GetMax<T>(nstats)) -
         duckdb::Hugeint::Convert(duckdb::NumericStats::GetMin<T>(nstats));
}

static bool can_use_partitioned_aggregate(duckdb::ClientContext& context,
                                          duckdb::LogicalAggregate& op,
                                          sirius::op::sirius_physical_operator& child,
                                          duckdb::vector<duckdb::column_t>& partition_columns)
{
  if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) { return false; }
  // check if the source is partitioned by the aggregate columns
  // figure out the columns we are grouping by
  for (auto& group_expr : op.groups) {
    // only support bound reference here
    if (group_expr->GetExpressionType() != duckdb::ExpressionType::BOUND_REF) { return false; }
    auto& ref = group_expr->Cast<duckdb::BoundReferenceExpression>();
    partition_columns.push_back(ref.index);
  }
  // traverse the children of the aggregate to find the source operator
  duckdb::reference<sirius::op::sirius_physical_operator> child_ref(child);
  while (child_ref.get().type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& child_op = child_ref.get();
    switch (child_op.type) {
      case sirius::op::SiriusPhysicalOperatorType::PROJECTION: {
        // recompute partition columns
        auto& projection = child_op.Cast<sirius::op::sirius_physical_projection>();
        duckdb::vector<duckdb::column_t> new_columns;
        for (auto& partition_col : partition_columns) {
          // we only support bound reference here
          auto const* expr = projection.select_list[partition_col].get();
          if (expr == nullptr) { return false; }
          if (!expr->holds<sirius::ast::reference>()) { return false; }
          new_columns.push_back(expr->get<sirius::ast::reference>().column_index);
        }
        // continue into child node with new columns
        partition_columns = std::move(new_columns);
        child_ref         = *child_op.children[0];
        break;
      }
      case sirius::op::SiriusPhysicalOperatorType::FILTER:
        // continue into child operators
        child_ref = *child_op.children[0];
        break;
      default:
        // unsupported operator for partition pass-through
        return false;
    }
  }
  auto& table_scan = child_ref.get().Cast<sirius::op::sirius_physical_table_scan>();
  if (!table_scan.function.get_partition_info) {
    // this source does not expose partition information - skip
    return false;
  }
  // get the base columns by projecting over the projection_ids/column_ids
  if (!table_scan.projection_ids.empty()) {
    for (auto& partition_col : partition_columns) {
      if (partition_col >= table_scan.projection_ids.size()) { return false; }
      partition_col = table_scan.projection_ids[partition_col];
    }
  }
  duckdb::vector<duckdb::column_t> base_columns;
  for (const auto& partition_idx : partition_columns) {
    auto col_idx = partition_idx;
    col_idx      = table_scan.column_ids[col_idx].GetPrimaryIndex();
    base_columns.push_back(col_idx);
  }
  // check if the source operator is partitioned by the grouping columns
  duckdb::TableFunctionPartitionInput input(table_scan.bind_data.get(), base_columns);
  auto partition_info = table_scan.function.get_partition_info(context, input);
  if (partition_info != duckdb::TablePartitionInfo::SINGLE_VALUE_PARTITIONS) {
    // we only support single-value partitions currently
    return false;
  }
  // we have single value partitions!
  return true;
}

static bool can_use_perfect_hash_aggregate(duckdb::ClientContext& context,
                                           duckdb::LogicalAggregate& op,
                                           duckdb::vector<std::size_t>& bits_per_group)
{
  if (op.grouping_sets.size() > 1 || !op.grouping_functions.empty()) { return false; }
  std::size_t perfect_hash_bits = 0;
  for (std::size_t group_idx = 0; group_idx < op.groups.size(); group_idx++) {
    auto& group = op.groups[group_idx];
    auto& stats = op.group_stats[group_idx];

    switch (group->return_type.InternalType()) {
      case duckdb::PhysicalType::INT8:
      case duckdb::PhysicalType::INT16:
      case duckdb::PhysicalType::INT32:
      case duckdb::PhysicalType::INT64:
      case duckdb::PhysicalType::UINT8:
      case duckdb::PhysicalType::UINT16:
      case duckdb::PhysicalType::UINT32:
      case duckdb::PhysicalType::UINT64: break;
      default:
        // we only support simple integer types for perfect hashing
        return false;
    }
    // check if the group has stats available
    auto& group_type = group->return_type;
    if (!stats) {
      // no stats, but we might still be able to use perfect hashing if the type is small enough
      // for small types we can just set the stats to [type_min, type_max]
      switch (group_type.InternalType()) {
        case duckdb::PhysicalType::INT8:
        case duckdb::PhysicalType::INT16:
        case duckdb::PhysicalType::UINT8:
        case duckdb::PhysicalType::UINT16: break;
        default:
          // type is too large and there are no stats: skip perfect hashing
          return false;
      }
      // construct stats with the min and max value of the type
      stats = duckdb::NumericStats::CreateUnknown(group_type).ToUnique();
      duckdb::NumericStats::SetMin(*stats, duckdb::Value::MinimumValue(group_type));
      duckdb::NumericStats::SetMax(*stats, duckdb::Value::MaximumValue(group_type));
    }
    auto& nstats = *stats;

    if (!duckdb::NumericStats::HasMinMax(nstats)) { return false; }

    if (duckdb::NumericStats::Max(*stats) < duckdb::NumericStats::Min(*stats)) {
      // May result in underflow
      return false;
    }

    // we have a min and a max value for the stats: use that to figure out how many bits we have
    // we add two here, one for the NULL value, and one to make the computation one-indexed
    // (e.g. if min and max are the same, we still need one entry in total)
    duckdb::hugeint_t range_h;
    switch (group_type.InternalType()) {
      case duckdb::PhysicalType::INT8: range_h = get_range_hugeint<int8_t>(nstats); break;
      case duckdb::PhysicalType::INT16: range_h = get_range_hugeint<int16_t>(nstats); break;
      case duckdb::PhysicalType::INT32: range_h = get_range_hugeint<int32_t>(nstats); break;
      case duckdb::PhysicalType::INT64: range_h = get_range_hugeint<int64_t>(nstats); break;
      case duckdb::PhysicalType::UINT8: range_h = get_range_hugeint<uint8_t>(nstats); break;
      case duckdb::PhysicalType::UINT16: range_h = get_range_hugeint<uint16_t>(nstats); break;
      case duckdb::PhysicalType::UINT32: range_h = get_range_hugeint<uint32_t>(nstats); break;
      case duckdb::PhysicalType::UINT64: range_h = get_range_hugeint<uint64_t>(nstats); break;
      default:
        throw duckdb::InternalException(
          "Unsupported type for perfect hash (should be caught before)");
    }

    uint64_t range;
    if (!duckdb::Hugeint::TryCast(range_h, range)) { return false; }

    // bail out on any range bigger than 2^32
    if (range >= duckdb::NumericLimits<int32_t>::Maximum()) { return false; }

    range += 2;
    // figure out how many bits we need
    std::size_t required_bits = required_bits_for_value(duckdb::UnsafeNumericCast<uint32_t>(range));
    bits_per_group.push_back(required_bits);
    perfect_hash_bits += required_bits;
    // check if we have exceeded the bits for the hash
    if (perfect_hash_bits > duckdb::Settings::Get<duckdb::PerfectHtThresholdSetting>(context)) {
      // too many bits for perfect hash
      return false;
    }
  }
  for (auto& expression : op.expressions) {
    auto& aggregate = expression->Cast<duckdb::BoundAggregateExpression>();
    if (aggregate.IsDistinct() || !aggregate.function.combine) {
      // distinct aggregates are not supported in perfect hash aggregates
      return false;
    }
  }
  return true;
}

/// cuDF does not support HUGEINT (int128). DuckDB widens aggregates like sum(int32) to HUGEINT
/// to avoid overflow. We downcast to BIGINT at the plan level so all downstream operators
/// (including the result collector) use the correct type.
static void downcast_hugeint_types(duckdb::vector<duckdb::LogicalType>& types,
                                   duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>& exprs)
{
  for (auto& type : types) {
    if (type == duckdb::LogicalType::HUGEINT) { type = duckdb::LogicalType::BIGINT; }
  }
  for (auto& expr : exprs) {
    if (expr->return_type == duckdb::LogicalType::HUGEINT) {
      expr->return_type = duckdb::LogicalType::BIGINT;
    }
  }
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalAggregate& op)
{
  D_ASSERT(op.children.size() == 1);

  // Downcast HUGEINT to BIGINT since cuDF does not support int128
  downcast_hugeint_types(op.types, op.expressions);

  // Reject nested GROUP BY keys before extract_aggregate_expressions rewrites
  // groups into bare references (which lose the name needed for the error).
  for (auto const& group : op.groups) {
    reject_nested_column_operation(*group, "GROUP BY");
  }

  if (auto fused = try_plan_dense_count_join(op)) { return fused; }

  auto plan = create_plan(*op.children[0]);

  plan = extract_aggregate_expressions(
    context, std::move(plan), op.expressions, op.groups, op.grouping_sets);
  bool can_use_simple_aggregation = true;
  for (auto& expression : op.expressions) {
    auto& aggregate = expression->Cast<duckdb::BoundAggregateExpression>();
    if (!aggregate.function.simple_update) {
      // unsupported aggregate for simple aggregation: use hash aggregation
      can_use_simple_aggregation = false;
      break;
    }
  }

  // Check if all groups are valid
  if (op.group_stats.empty()) { op.group_stats.resize(op.groups.size()); }
  auto group_validity = duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES;
  for (const auto& stats : op.group_stats) {
    if (stats && !stats->CanHaveNull()) { continue; }
    group_validity = duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES;
    break;
  }

  if (op.groups.empty() && op.grouping_sets.size() <= 1) {
    // no groups, check if we can use a simple aggregation
    // special case: aggregate entire columns together
    if (can_use_simple_aggregation) {
      auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                             sirius::op::sirius_physical_ungrouped_aggregate>(
        sirius::from_duckdb_vec(op.types),
        translate_expressions(std::move(op.expressions)),
        op.estimated_cardinality,
        op.distinct_validity);
      group_by->children.push_back(std::move(plan));
      return group_by;
    }
    throw duckdb::NotImplementedException("Non simple aggregation is not supported");
  }

  // groups! create a GROUP BY aggregator
  // use a partitioned or perfect hash aggregate if possible
  duckdb::vector<duckdb::column_t> partition_columns;
  duckdb::vector<std::size_t> required_bits;
  if (can_use_simple_aggregation &&
      can_use_partitioned_aggregate(context, op, *plan, partition_columns)) {
    auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                           sirius::op::sirius_physical_grouped_aggregate>(
      sirius::from_duckdb_vec(op.types),
      translate_expressions(std::move(op.expressions)),
      translate_expressions(std::move(op.groups)),
      std::move(op.grouping_sets),
      std::move(op.grouping_functions),
      op.estimated_cardinality,
      group_validity,
      op.distinct_validity);
    group_by->children.push_back(std::move(plan));
    return group_by;
  }

  if (can_use_perfect_hash_aggregate(context, op, required_bits)) {
    auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                           sirius::op::sirius_physical_grouped_aggregate>(
      sirius::from_duckdb_vec(op.types),
      translate_expressions(std::move(op.expressions)),
      translate_expressions(std::move(op.groups)),
      std::move(op.grouping_sets),
      std::move(op.grouping_functions),
      op.estimated_cardinality,
      group_validity,
      op.distinct_validity);
    group_by->children.push_back(std::move(plan));
    return group_by;
  }

  auto group_by = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                         sirius::op::sirius_physical_grouped_aggregate>(
    sirius::from_duckdb_vec(op.types),
    translate_expressions(std::move(op.expressions)),
    translate_expressions(std::move(op.groups)),
    std::move(op.grouping_sets),
    std::move(op.grouping_functions),
    op.estimated_cardinality,
    group_validity,
    op.distinct_validity);
  group_by->children.push_back(std::move(plan));
  return group_by;
}

}  // namespace sirius::planner
