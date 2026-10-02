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

//! Eager aggregation pushdown (Yan & Larson, "Eager Aggregation and Lazy
//! Aggregation", VLDB 1995).
//!
//! Pattern (q13 shape; DuckDB flips `customer LEFT JOIN orders` into
//! `orders RIGHT JOIN customer`, so the pushed side may be either child). The
//! PASS-THROUGH projection (PT) is optional — DuckDB's column pruning inserts
//! one on some shapes, e.g. INNER joins — and every one of its slots must be a
//! plain column ref:
//!
//!     AGGREGATE g=[keys from S], a=[AGG(col from R), ...]        (upper, A)
//!       [PROJECTION [colrefs from R and S]]                      (PT, optional)
//!         COMPARISON_JOIN INNER/LEFT/RIGHT on R.k1=S.e1 AND ...  (J)
//!           [R: pushed side]    [S: preserved / non-pushed side]
//!
//! becomes
//!
//!     PROJECTION [groups..., CAST(COALESCE(combined, 0))...]     (PX, only
//!       AGGREGATE g=[keys from S], a=[SUM(partial), ...]          when needed)
//!         [PROJECTION [kept S colrefs..., partials...]]          (PT, rewritten)
//!           COMPARISON_JOIN on AGG_R.k1=S.e1 AND ...
//!             AGGREGATE g=[R.k1,...], a=[AGG(col), ...]          (lower)
//!               [R]
//!             [S]
//!
//! PT's slots are rewritten in place: every slot referencing R is DROPPED (each
//! was only ever consumed as an aggregate input, and its join column no longer
//! exists below), the surviving S slots keep their order with the upper
//! aggregate's group keys remapped onto their new positions, and one slot per
//! partial is APPENDED so the combines route through PT.
//!
//! Soundness: every R row that joins a given S row carries the same join-key
//! tuple, so all of them land in exactly one lower-aggregate group and their
//! contribution to any upper group is combined losslessly — COUNT partials
//! combine by SUM, SUM partials by SUM, MIN/MAX by MIN/MAX. This holds for any
//! join multiplicity (an S key matching n R rows contributes the same combined
//! value once instead of n addends; duplicated S rows duplicate the partial
//! exactly as they duplicated the base rows). NULL join keys on R form a lower
//! group whose key never satisfies `=`, exactly as the underlying rows never
//! joined. For LEFT/RIGHT joins the preserved side's unmatched rows see one
//! NULL partial: SUM/MIN/MAX over them yields NULL exactly as over the original
//! NULL-extended rows, while COUNT would have yielded 0 — repaired by a
//! COALESCE(combined, 0) in PX. COUNT partials are never NULL (COUNT >= 0), so
//! matched groups need no repair.
//!
//! Everything above the join that referenced R must be rewritten or refused:
//! the upper aggregate's inputs are rewritten to the partials, the join
//! conditions' R sides are rewritten to the lower aggregate's keys, and the
//! matcher refuses any other reference to R (group keys, residual predicates,
//! non-column-ref condition sides). The caller additionally retries physical
//! planning with the untouched original plan if the rewritten copy fails any
//! later stage, so an unsupported rewritten shape costs only the optimization.

#include "planner/eager_agg_pushdown_plan_pass.hpp"

#include "log/logging.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "transparent/sirius_optimizer_extension.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/common/enums/expression_type.hpp>
#include <duckdb/common/enums/join_type.hpp>
#include <duckdb/function/function_binder.hpp>
#include <duckdb/optimizer/column_binding_replacer.hpp>
#include <duckdb/parser/constraints/unique_constraint.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_columnref_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>

#include <algorithm>
#include <exception>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::planner {

namespace {

// ---------------------------------------------------------------------------
// Settings (read per use from the connection's Sirius operator params, so a SET
// takes effect on the next query and tests can A/B in-process)
// ---------------------------------------------------------------------------

const sirius::operator_params* eager_agg_params(duckdb::ClientContext& context)
{
  if (!context.registered_state) { return nullptr; }
  auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!state) { return nullptr; }
  return &state->get_config().get_operator_params();
}

bool pass_enabled(const sirius::operator_params* params)
{
  return params == nullptr || params->enable_eager_agg_pushdown;
}

bool benefit_gate_bypassed(const sirius::operator_params* params)
{
  return params != nullptr && params->eager_agg_pushdown_force;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// Combine function applied above the join for each pushable aggregate, or
/// nullptr when the aggregate is not decomposable this way. COUNT partials are
/// non-NULL counts, so they combine by SUM like SUM partials do.
///
/// SUM over FLOAT / DOUBLE is refused: floating-point addition is not
/// associative, so `sum(per-key partials)` need not equal the single-pass
/// `sum` the unrewritten plan computes (e.g. addends 1e16, 1.0, -1e16 grouped
/// by join key yield 1.0 where the original order yields 0.0). Exact types
/// (integers, fixed-point DECIMAL, and COUNT's own BIGINT partials) reassociate
/// losslessly, so they stay in the set.
///
/// `sum_no_overflow` is deliberately absent: DuckDB installs it only from
/// SumPropagateStats, which needs the aggregate's input node to carry a
/// max-cardinality bound. A comparison join does not propagate one, so an
/// aggregate sitting on a join (the only shape this pass matches) never carries
/// it — accepting it would be untestable dead code.
const char* combine_function_name(const std::string& name, const duckdb::LogicalType& arg_type)
{
  if (name == "count") { return "sum"; }
  if (name == "sum") {
    switch (arg_type.InternalType()) {
      case duckdb::PhysicalType::FLOAT:
      case duckdb::PhysicalType::DOUBLE: return nullptr;
      default: return "sum";
    }
  }
  if (name == "min") { return "min"; }
  if (name == "max") { return "max"; }
  return nullptr;
}

std::unordered_set<duckdb::idx_t> output_table_indexes(duckdb::LogicalOperator& op)
{
  std::unordered_set<duckdb::idx_t> out;
  for (auto& binding : op.GetColumnBindings()) {
    out.insert(binding.table_index);
  }
  return out;
}

/// First half of the heuristic benefit gate: push only when the non-pushed side is a bare,
/// unfiltered table scan (modulo projections). Then the join is not expected
/// to discard most pushed-side rows, so the pre-aggregation's reduction
/// carries through to the join (q13: customer is a bare scan). A filtered or
/// composite non-pushed side (q18: a semi-join-reduced orders subtree) means
/// the join itself is selective and pre-aggregating the full pushed side would
/// mostly compute groups the join throws away.
bool non_pushed_side_is_bare_scan(const duckdb::LogicalOperator& side)
{
  const duckdb::LogicalOperator* node = &side;
  while (node->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    node = node->children[0].get();
  }
  if (node->type != duckdb::LogicalOperatorType::LOGICAL_GET) { return false; }
  return node->Cast<duckdb::LogicalGet>().table_filters.filters.empty();
}

/// The single base scan a row-preserving subtree reads, or nullptr when the
/// subtree is anything else (a join, a set operation, an aggregate, ...).
/// LOGICAL_PROJECTION is deliberately NOT traversed: it renumbers bindings, so
/// the caller's join-key bindings would no longer address the scan's outputs.
const duckdb::LogicalGet* sole_base_scan(const duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    return &op.Cast<duckdb::LogicalGet>();
  }
  if (op.children.size() != 1) { return nullptr; }
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY: return sole_base_scan(*op.children[0]);
    default: return nullptr;
  }
}

/// Second half of the benefit gate: the pre-aggregation only pays when it
/// actually reduces rows, and it provably does not when the join keys on the
/// pushed side contain a PRIMARY KEY of the scanned base table — then every
/// input row is already its own group and the lower aggregate is a full
/// PARTITION + MERGE_GROUP_BY for nothing (`customer JOIN orders ON c_custkey =
/// o_custkey` aggregating a customer column: 150M rows in, 150M groups out).
///
/// Only PRIMARY KEY is used, mirroring prove_unique_columns in
/// sirius_plan_comparison_join.cpp: a nullable UNIQUE column may repeat NULLs.
/// This is provable evidence only — no distinct-count statistics exist at this
/// point in planning, so a merely near-unique key is not caught.
bool pushed_keys_are_provably_unique(const duckdb::LogicalOperator& pushed_side,
                                     const std::unordered_set<duckdb::idx_t>& key_columns,
                                     duckdb::idx_t key_table_index)
{
  const auto* get = sole_base_scan(pushed_side);
  if (get == nullptr || get->table_index != key_table_index) { return false; }
  auto table = get->GetTable();
  if (!table) { return false; }

  // LogicalGet output position -> the table's logical column index.
  const auto& column_ids = get->GetColumnIds();
  const auto& proj_ids   = get->projection_ids;
  std::unordered_set<duckdb::idx_t> key_logical_indexes;
  for (auto position : key_columns) {
    if (proj_ids.empty()) {
      if (position >= column_ids.size()) { return false; }
      key_logical_indexes.insert(column_ids[position].GetPrimaryIndex());
    } else {
      if (position >= proj_ids.size() || proj_ids[position] >= column_ids.size()) { return false; }
      key_logical_indexes.insert(column_ids[proj_ids[position]].GetPrimaryIndex());
    }
  }

  for (const auto& constraint : table->GetConstraints()) {
    if (constraint->type != duckdb::ConstraintType::UNIQUE) { continue; }
    auto& unique = constraint->Cast<duckdb::UniqueConstraint>();
    if (!unique.IsPrimaryKey()) { continue; }
    auto logical_indexes = unique.GetLogicalIndexes(table->GetColumns());
    if (logical_indexes.empty()) { continue; }
    bool covered = true;
    for (const auto& idx : logical_indexes) {
      if (key_logical_indexes.count(idx.index) == 0) {
        covered = false;
        break;
      }
    }
    if (covered) { return true; }
  }
  return false;
}

struct match_info {
  duckdb::LogicalAggregate* aggregate = nullptr;
  duckdb::LogicalComparisonJoin* join = nullptr;
  /// Pure pass-through projection between aggregate and join, when present
  /// (DuckDB's column pruning inserts one on some shapes, e.g. INNER joins).
  duckdb::LogicalProjection* projection = nullptr;
  /// Which join child is the pushed (pre-aggregated) side.
  duckdb::idx_t pushed_slot = 0;
  /// Table indexes produced by the pushed side.
  std::unordered_set<duckdb::idx_t> pushed_tables;
  /// Per upper aggregate: combine function name ("sum"/"min"/"max").
  std::vector<const char*> combine_names;
  /// Per upper aggregate: repair 0-vs-NULL for unmatched preserved rows
  /// (COUNT under an outer join).
  std::vector<bool> needs_zero_fill;
};

/// Resolve @p expr (a reference in the aggregate) through the optional
/// pass-through projection to the column ref the JOIN emits. Returns nullptr
/// when the shape is out of contract (not a plain column ref at either level).
const duckdb::BoundColumnRefExpression* trace_to_join_output(
  const duckdb::Expression& expr, const duckdb::LogicalProjection* projection)
{
  if (expr.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) { return nullptr; }
  auto& colref = expr.Cast<duckdb::BoundColumnRefExpression>();
  if (projection == nullptr) { return &colref; }
  // With a projection in between, every aggregate-level reference must be one
  // of its outputs.
  if (colref.binding.table_index != projection->table_index ||
      colref.binding.column_index >= projection->expressions.size()) {
    return nullptr;
  }
  auto& below = *projection->expressions[colref.binding.column_index];
  if (below.GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) { return nullptr; }
  return &below.Cast<duckdb::BoundColumnRefExpression>();
}

/// Match @p op against the pushdown pattern. Purely read-only; returns
/// std::nullopt (refusal) unless every correctness gate and the benefit gate
/// hold.
std::optional<match_info> match_candidate(duckdb::LogicalOperator& op,
                                          const sirius::operator_params* params)
{
  if (op.type != duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    return std::nullopt;
  }
  auto& aggregate = op.Cast<duckdb::LogicalAggregate>();

  // --- upper aggregate shape ---
  // Exactly one plain grouping set over all group columns; no GROUPING() calls.
  if (!aggregate.grouping_functions.empty()) { return std::nullopt; }
  if (aggregate.groups.empty() || aggregate.expressions.empty()) { return std::nullopt; }
  if (aggregate.grouping_sets.size() != 1 ||
      aggregate.grouping_sets[0].size() != aggregate.groups.size()) {
    return std::nullopt;
  }

  // --- child must be a plain comparison join, optionally through ONE pure
  // pass-through projection (every slot a plain column ref) ---
  if (aggregate.children.size() != 1) { return std::nullopt; }
  match_info info;
  duckdb::LogicalOperator* below = aggregate.children[0].get();
  if (below->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    info.projection = &below->Cast<duckdb::LogicalProjection>();
    for (auto& slot : info.projection->expressions) {
      if (slot->GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
        return std::nullopt;
      }
    }
    below = info.projection->children[0].get();
  }
  if (below->type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return std::nullopt; }
  auto& join = below->Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::INNER && join.join_type != duckdb::JoinType::LEFT &&
      join.join_type != duckdb::JoinType::RIGHT) {
    return std::nullopt;
  }
  if (join.predicate || !join.duplicate_eliminated_columns.empty() || join.conditions.empty()) {
    return std::nullopt;
  }

  auto left_tables  = output_table_indexes(*join.children[0]);
  auto right_tables = output_table_indexes(*join.children[1]);

  // --- every aggregate is a decomposable single-column-ref over ONE side ---
  bool args_in_left  = false;
  bool args_in_right = false;
  for (auto& expr : aggregate.expressions) {
    if (expr->GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
      return std::nullopt;
    }
    auto& aggr = expr->Cast<duckdb::BoundAggregateExpression>();
    if (aggr.IsDistinct() || aggr.filter || (aggr.order_bys && !aggr.order_bys->orders.empty())) {
      return std::nullopt;
    }
    if (aggr.children.size() != 1) { return std::nullopt; }
    auto* traced = trace_to_join_output(*aggr.children[0], info.projection);
    if (traced == nullptr) { return std::nullopt; }
    const char* combine = combine_function_name(aggr.function.name, traced->return_type);
    if (combine == nullptr) { return std::nullopt; }
    auto table_index = traced->binding.table_index;
    if (left_tables.count(table_index) != 0) {
      args_in_left = true;
    } else if (right_tables.count(table_index) != 0) {
      args_in_right = true;
    } else {
      return std::nullopt;  // outer / lateral reference — not this join's column
    }
    info.combine_names.push_back(combine);
    info.needs_zero_fill.push_back(std::string_view{aggr.function.name} == "count" &&
                                   join.join_type != duckdb::JoinType::INNER);
  }
  if (args_in_left == args_in_right) { return std::nullopt; }  // both sides or neither
  info.pushed_slot = args_in_left ? 0 : 1;

  // The pushed side must not be the preserved side of an outer join: preserved
  // rows must reach the upper aggregate once each, not once per group.
  if (join.join_type == duckdb::JoinType::LEFT && info.pushed_slot != 1) { return std::nullopt; }
  if (join.join_type == duckdb::JoinType::RIGHT && info.pushed_slot != 0) { return std::nullopt; }

  info.pushed_tables        = info.pushed_slot == 0 ? left_tables : right_tables;
  const auto& pushed_tables = info.pushed_tables;

  // --- group keys must not touch the pushed side ---
  for (auto& group : aggregate.groups) {
    auto* traced = trace_to_join_output(*group, info.projection);
    if (traced == nullptr) { return std::nullopt; }
    if (pushed_tables.count(traced->binding.table_index) != 0) { return std::nullopt; }
  }

  // --- join conditions: all `=`, pushed side is a plain column ref ---
  std::unordered_set<duckdb::idx_t> pushed_key_columns;
  duckdb::idx_t pushed_key_table = 0;
  bool single_key_table          = true;
  for (auto& cond : join.conditions) {
    if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) { return std::nullopt; }
    auto& pushed_expr = info.pushed_slot == 0 ? cond.left : cond.right;
    if (pushed_expr->GetExpressionClass() != duckdb::ExpressionClass::BOUND_COLUMN_REF) {
      return std::nullopt;
    }
    const auto& binding = pushed_expr->Cast<duckdb::BoundColumnRefExpression>().binding;
    if (pushed_tables.count(binding.table_index) == 0) { return std::nullopt; }
    if (pushed_key_columns.empty()) {
      pushed_key_table = binding.table_index;
    } else if (binding.table_index != pushed_key_table) {
      single_key_table = false;
    }
    pushed_key_columns.insert(binding.column_index);
  }

  // --- benefit gate (heuristic; never affects correctness) ---
  if (!benefit_gate_bypassed(params)) {
    if (!non_pushed_side_is_bare_scan(*join.children[1 - info.pushed_slot])) {
      return std::nullopt;
    }
    // A provably unique pushed key means the lower aggregate emits one row per
    // input row: all cost, no reduction.
    if (single_key_table &&
        pushed_keys_are_provably_unique(
          *join.children[info.pushed_slot], pushed_key_columns, pushed_key_table)) {
      return std::nullopt;
    }
  }

  info.aggregate = &aggregate;
  info.join      = &join;
  return info;
}

// ---------------------------------------------------------------------------
// Rewriting
// ---------------------------------------------------------------------------

duckdb::idx_t max_table_index(const duckdb::LogicalOperator& op)
{
  duckdb::idx_t max_index = 0;
  for (auto index : op.GetTableIndex()) {
    max_index = std::max(max_index, index);
  }
  for (auto& child : op.children) {
    max_index = std::max(max_index, max_table_index(*child));
  }
  return max_index;
}

duckdb::unique_ptr<duckdb::BoundColumnRefExpression> make_column_ref(
  const duckdb::LogicalType& type, duckdb::idx_t table_index, duckdb::idx_t column_index)
{
  return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
    type, duckdb::ColumnBinding(table_index, column_index));
}

/// Bind `name(child)` against the system catalog, mirroring how DuckDB's own
/// optimizer rewrites bind replacement aggregates (sum_rewriter.cpp).
duckdb::unique_ptr<duckdb::Expression> bind_combine_aggregate(
  duckdb::ClientContext& context, const char* name, duckdb::unique_ptr<duckdb::Expression> child)
{
  duckdb::QueryErrorContext error_context;
  auto& entry = duckdb::Catalog::GetEntry<duckdb::AggregateFunctionCatalogEntry>(
    context, SYSTEM_CATALOG, DEFAULT_SCHEMA, name, error_context);
  auto function = entry.functions.GetFunctionByArguments(context, {child->return_type});
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> children;
  children.push_back(std::move(child));
  duckdb::FunctionBinder binder(context);
  return binder.BindAggregateFunction(std::move(function), std::move(children));
}

/// Apply the rewrite for a matched candidate. @p op_ref owns the upper
/// aggregate; it is replaced with a fix-up projection when one is needed.
/// @p root is the plan root, used to remap bindings above the new projection.
void apply_rewrite(duckdb::unique_ptr<duckdb::LogicalOperator>& op_ref,
                   duckdb::unique_ptr<duckdb::LogicalOperator>& root,
                   const match_info& info,
                   duckdb::ClientContext& context)
{
  auto& aggregate = *info.aggregate;
  auto& join      = *info.join;

  auto next_index              = max_table_index(*root) + 1;
  auto const lower_group_index = next_index++;
  auto const lower_aggr_index  = next_index++;
  auto const projection_index  = next_index++;
  auto const num_groups        = aggregate.groups.size();
  auto const num_aggregates    = aggregate.expressions.size();

  // --- lower aggregate: GROUP BY the pushed side's join keys, computing the
  // original aggregates with their inputs re-traced to below-join bindings
  // (identical bindings when no pass-through projection sits in between) ---
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> lower_aggregates;
  for (auto& expr : aggregate.expressions) {
    auto lower_expr        = expr->Copy();
    auto& lower_aggr       = lower_expr->Cast<duckdb::BoundAggregateExpression>();
    auto* traced           = trace_to_join_output(*lower_aggr.children[0], info.projection);
    lower_aggr.children[0] = make_column_ref(
      traced->return_type, traced->binding.table_index, traced->binding.column_index);
    lower_aggregates.push_back(std::move(lower_expr));
  }
  auto lower = duckdb::make_uniq<duckdb::LogicalAggregate>(
    lower_group_index, lower_aggr_index, std::move(lower_aggregates));
  duckdb::GroupingSet grouping_set;
  for (duckdb::idx_t k = 0; k < join.conditions.size(); k++) {
    auto& cond        = join.conditions[k];
    auto& pushed_expr = info.pushed_slot == 0 ? cond.left : cond.right;
    lower->groups.push_back(pushed_expr->Copy());
    grouping_set.insert(k);
  }
  lower->grouping_sets.push_back(std::move(grouping_set));
  auto& pushed_child = join.children[info.pushed_slot];
  // No cardinality is pinned on the lower aggregate: the copy this pass rewrites
  // comes out of LogicalOperator::Copy, which does not carry
  // estimated_cardinality, and create_plan recomputes it bottom-up anyway.
  lower->children.push_back(std::move(pushed_child));
  pushed_child = std::move(lower);

  // --- join now reads the lower aggregate's keys on the pushed side; its
  // former per-row output no longer exists, so the pushed-side projection map
  // is rebuilt to emit exactly the partials ---
  for (duckdb::idx_t k = 0; k < join.conditions.size(); k++) {
    auto& cond        = join.conditions[k];
    auto& pushed_expr = info.pushed_slot == 0 ? cond.left : cond.right;
    pushed_expr       = make_column_ref(pushed_expr->return_type, lower_group_index, k);
  }
  auto& pushed_projection_map =
    info.pushed_slot == 0 ? join.left_projection_map : join.right_projection_map;
  pushed_projection_map.clear();
  pushed_projection_map.reserve(num_aggregates);
  for (duckdb::idx_t i = 0; i < num_aggregates; i++) {
    // The lower aggregate emits [keys..., partials...]. Only the partials are
    // read above the join: the matcher proved nothing above it references the
    // pushed side's keys, which are consumed by the conditions themselves. An
    // empty map would mean "all columns" and carry every key through the build
    // payload and every output row for nothing.
    pushed_projection_map.push_back(join.conditions.size() + i);
  }

  // --- upper aggregate combines the partials ---
  std::vector<duckdb::LogicalType> original_types;
  original_types.reserve(num_aggregates);
  for (auto& expr : aggregate.expressions) {
    original_types.push_back(expr->return_type);
  }

  // With a pass-through projection in between, route the partials through it:
  // keep its non-pushed-side slots (remapping the group keys that use them),
  // drop the pushed-side slots (their join columns no longer exist — every one
  // of them was only consumed as an aggregate input), and append one slot per
  // partial for the upper aggregate to combine.
  auto partial_table         = lower_aggr_index;
  duckdb::idx_t partial_base = 0;
  if (info.projection != nullptr) {
    auto& projection = *info.projection;
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> kept_slots;
    std::unordered_map<duckdb::idx_t, duckdb::idx_t> slot_remap;
    for (duckdb::idx_t s = 0; s < projection.expressions.size(); s++) {
      auto& slot_ref = projection.expressions[s]->Cast<duckdb::BoundColumnRefExpression>();
      if (info.pushed_tables.count(slot_ref.binding.table_index) != 0) { continue; }
      slot_remap[s] = kept_slots.size();
      kept_slots.push_back(std::move(projection.expressions[s]));
    }
    partial_base = kept_slots.size();
    for (duckdb::idx_t i = 0; i < num_aggregates; i++) {
      kept_slots.push_back(make_column_ref(original_types[i], lower_aggr_index, i));
    }
    projection.expressions = std::move(kept_slots);
    partial_table          = projection.table_index;
    for (auto& group : aggregate.groups) {
      auto& group_ref                = group->Cast<duckdb::BoundColumnRefExpression>();
      group_ref.binding.column_index = slot_remap.at(group_ref.binding.column_index);
    }
  }

  bool needs_projection = false;
  for (duckdb::idx_t i = 0; i < num_aggregates; i++) {
    auto& expr       = aggregate.expressions[i];
    auto partial_ref = make_column_ref(original_types[i], partial_table, partial_base + i);
    auto combined = bind_combine_aggregate(context, info.combine_names[i], std::move(partial_ref));
    combined->SetAlias(expr->GetAlias());
    needs_projection =
      needs_projection || info.needs_zero_fill[i] || combined->return_type != original_types[i];
    expr = std::move(combined);
  }

  if (!needs_projection) {
    SIRIUS_LOG_INFO("Eager aggregation pushdown applied ({} keys, {} aggregates)",
                    join.conditions.size(),
                    num_aggregates);
    return;
  }

  // --- fix-up projection: restore COUNT's 0 for unmatched preserved rows and
  // the original output types, keeping the plan's schema byte-identical ---
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> projection_exprs;
  for (duckdb::idx_t g = 0; g < num_groups; g++) {
    projection_exprs.push_back(
      make_column_ref(aggregate.groups[g]->return_type, aggregate.group_index, g));
  }
  for (duckdb::idx_t i = 0; i < num_aggregates; i++) {
    auto const& combined_type = aggregate.expressions[i]->return_type;
    duckdb::unique_ptr<duckdb::Expression> expr =
      make_column_ref(combined_type, aggregate.aggregate_index, i);
    if (info.needs_zero_fill[i]) {
      auto coalesce = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
        duckdb::ExpressionType::OPERATOR_COALESCE, combined_type);
      coalesce->children.push_back(std::move(expr));
      coalesce->children.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value::Numeric(combined_type, 0)));
      expr = std::move(coalesce);
    }
    expr = duckdb::BoundCastExpression::AddCastToType(context, std::move(expr), original_types[i]);
    projection_exprs.push_back(std::move(expr));
  }

  auto projection =
    duckdb::make_uniq<duckdb::LogicalProjection>(projection_index, std::move(projection_exprs));
  if (aggregate.has_estimated_cardinality) {
    projection->SetEstimatedCardinality(aggregate.estimated_cardinality);
  }

  duckdb::ColumnBindingReplacer replacer;
  for (duckdb::idx_t g = 0; g < num_groups; g++) {
    replacer.replacement_bindings.emplace_back(duckdb::ColumnBinding(aggregate.group_index, g),
                                               duckdb::ColumnBinding(projection_index, g));
  }
  for (duckdb::idx_t i = 0; i < num_aggregates; i++) {
    replacer.replacement_bindings.emplace_back(
      duckdb::ColumnBinding(aggregate.aggregate_index, i),
      duckdb::ColumnBinding(projection_index, num_groups + i));
  }

  projection->children.push_back(std::move(op_ref));
  op_ref = std::move(projection);

  // Remap every reference above the aggregate to the projection's outputs. The
  // stop operator keeps the projection's own expressions (which legitimately
  // reference the aggregate) untouched; nothing below the aggregate can see
  // its bindings.
  replacer.stop_operator = op_ref.get();
  replacer.VisitOperator(*root);

  SIRIUS_LOG_INFO(
    "Eager aggregation pushdown applied ({} keys, {} aggregates, with fix-up projection)",
    join.conditions.size(),
    num_aggregates);
}

/// Bottom-up rewrite walk. Children first, so nested candidates are handled
/// before their ancestors; each node is visited exactly once, so a rewritten
/// aggregate (which still matches the pattern textually) is never re-pushed.
void visit(duckdb::unique_ptr<duckdb::LogicalOperator>& op_ref,
           duckdb::unique_ptr<duckdb::LogicalOperator>& root,
           duckdb::ClientContext& context,
           const sirius::operator_params* params,
           std::uint64_t& applied)
{
  for (auto& child : op_ref->children) {
    visit(child, root, context, params, applied);
  }
  if (auto info = match_candidate(*op_ref, params)) {
    apply_rewrite(op_ref, root, *info, context);
    applied++;
  }
}

bool contains_candidate(duckdb::LogicalOperator& op, const sirius::operator_params* params)
{
  if (match_candidate(op, params)) { return true; }
  for (auto& child : op.children) {
    if (contains_candidate(*child, params)) { return true; }
  }
  return false;
}

}  // namespace

eager_agg_pushdown_result try_eager_aggregation_pushdown(duckdb::LogicalOperator& plan,
                                                         duckdb::ClientContext& context)
{
  const auto* params = eager_agg_params(context);
  if (!pass_enabled(params)) { return {}; }

  // Cheap read-only scan first: the copy below is only paid when a provable
  // candidate exists (on TPC-H that is q13 alone). Every gate is structural, so
  // the pre-scan and the rewrite below decide identically — the copy carries the
  // same tree, and no gate reads metadata that LogicalOperator::Copy drops.
  if (!contains_candidate(plan, params)) { return {}; }

  duckdb::unique_ptr<duckdb::LogicalOperator> copy;
  try {
    // Preserves dynamic-filter metadata exactly like the transparent capture
    // path; a plan whose scans cannot be copied is simply not rewritten.
    copy = sirius::transparent::copy_logical_plan(plan, context);
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("Eager aggregation pushdown: plan not copyable, skipping: {}", e.what());
    return {};
  }

  std::uint64_t applied = 0;
  try {
    visit(copy, copy, context, params, applied);
  } catch (std::exception& e) {
    // E.g. no matching combine overload in the catalog. Fail closed.
    SIRIUS_LOG_DEBUG("Eager aggregation pushdown: rewrite failed, skipping: {}", e.what());
    return {};
  }
  if (applied == 0) { return {}; }
  return {std::move(copy), applied};
}

}  // namespace sirius::planner
