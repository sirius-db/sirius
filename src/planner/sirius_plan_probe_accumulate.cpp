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
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "expression/aggregate_id.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/aggregate/probe_accumulate_context.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_passthrough_sink.hpp"
#include "op/sirius_physical_probe_accumulate_emit.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace sirius::planner {

namespace {

/// One MIN/MAX/COUNT of the fused aggregate, with its argument re-addressed to the probe child.
struct accumulate_aggregate {
  op::accumulate_kind kind;
  /// Argument over the probe child's output columns; null for COUNT(*).
  duckdb::unique_ptr<duckdb::Expression> argument;
};

struct probe_accumulate_detection {
  std::size_t probe_key_idx;  ///< Key column of the probe child's output
  std::size_t build_key_idx;  ///< Key column of the build child's output
  /// The build side is a one-column DELIM_GET. DuckDB builds it from the distinct correlated
  /// tuples, so its single column is unique even though `prove_unique_columns` cannot say so.
  bool build_is_delim_get = false;
  std::vector<accumulate_aggregate> aggregates;
};

/// Output columns each side of the join contributes, before the aggregate reads them.
struct join_layout {
  std::size_t left_count;
  std::size_t right_count;
};

join_layout layout_of(duckdb::LogicalComparisonJoin const& join)
{
  return join_layout{join.left_projection_map.empty() ? join.children[0]->types.size()
                                                      : join.left_projection_map.size(),
                     join.right_projection_map.empty() ? join.children[1]->types.size()
                                                       : join.right_projection_map.size()};
}

/// {child, column within that child's output} of a position in the join's output.
std::pair<std::size_t, std::size_t> resolve_output(duckdb::LogicalComparisonJoin const& join,
                                                   join_layout const& layout,
                                                   std::size_t position)
{
  if (position < layout.left_count) {
    return {0, join.left_projection_map.empty() ? position : join.left_projection_map[position]};
  }
  auto const right = position - layout.left_count;
  return {1, join.right_projection_map.empty() ? right : join.right_projection_map[right]};
}

/// Re-address an expression over the join's output to the probe child's output. Returns null when
/// it reads the build side, or when the GPU expression evaluator cannot take the result.
duckdb::unique_ptr<duckdb::Expression> address_on_probe(duckdb::Expression const& expression,
                                                        duckdb::LogicalComparisonJoin const& join,
                                                        join_layout const& layout)
{
  auto copy = expression.Copy();
  bool ok   = true;
  duckdb::ExpressionIterator::VisitExpressionClassMutable(
    copy, duckdb::ExpressionClass::BOUND_REF, [&](duckdb::unique_ptr<duckdb::Expression>& leaf) {
      auto& ref = leaf->Cast<duckdb::BoundReferenceExpression>();
      if (ref.Index() >= layout.left_count + layout.right_count) {
        ok = false;
        return;
      }
      auto const [child, column] = resolve_output(join, layout, ref.Index());
      if (child != 0) {
        ok = false;
        return;
      }
      leaf = duckdb::make_uniq<duckdb::BoundReferenceExpression>(ref.GetReturnType(), column);
    });
  if (!ok || sirius::ast::from_duckdb(*copy) == nullptr) { return nullptr; }
  return copy;
}

std::optional<op::accumulate_kind> kind_of(sirius::aggregate_id id)
{
  switch (id) {
    case sirius::aggregate_id::min: return op::accumulate_kind::MIN;
    case sirius::aggregate_id::max: return op::accumulate_kind::MAX;
    case sirius::aggregate_id::count: return op::accumulate_kind::COUNT;
    case sirius::aggregate_id::count_star: return op::accumulate_kind::COUNT_STAR;
    default: return std::nullopt;
  }
}

/// Log why a shape that otherwise matched is left to the ordinary join and aggregate. Reported only
/// from checks made after the join and group key matched, so ordinary queries stay quiet.
std::nullopt_t decline(std::string const& reason)
{
  SIRIUS_LOG_INFO("[sirius_plan_probe_accumulate] not fusing: {}", reason);
  return std::nullopt;
}

/// Recognize `SELECT key, MIN/MAX/COUNT(...) ... GROUP BY key` over an INNER equi-join whose
/// build side is provably unique on the join key and whose aggregate arguments read only the
/// probe side. One projection may sit between the aggregate and the join.
std::optional<probe_accumulate_detection> detect_probe_accumulate(
  duckdb::LogicalAggregate const& aggregate)
{
  if (aggregate.groups.size() != 1 || aggregate.grouping_sets.size() != 1 ||
      aggregate.grouping_sets[0].size() != 1 || !aggregate.grouping_functions.empty() ||
      aggregate.expressions.empty() ||
      aggregate.expressions.size() > op::probe_accumulate_max_specs) {
    return std::nullopt;
  }
  if (aggregate.groups[0]->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }

  auto const* input                      = aggregate.children[0].get();
  duckdb::LogicalProjection const* below = nullptr;
  if (input->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION) {
    below = &input->Cast<duckdb::LogicalProjection>();
    input = below->children[0].get();
  }
  if (input->type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return std::nullopt; }
  auto const& join = input->Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::INNER || join.conditions.size() != 1 ||
      !join.conditions[0].IsComparison() || join.children.size() != 2) {
    return std::nullopt;
  }
  auto const& condition = join.conditions[0];
  if (condition.GetComparisonType() != duckdb::ExpressionType::COMPARE_EQUAL ||
      condition.GetLHS().GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF ||
      condition.GetRHS().GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  auto const& probe_ref = condition.GetLHS().Cast<duckdb::BoundReferenceExpression>();
  auto const& build_ref = condition.GetRHS().Cast<duckdb::BoundReferenceExpression>();
  auto const key_type   = probe_ref.GetReturnType().id();
  if (key_type != build_ref.GetReturnType().id() ||
      (key_type != duckdb::LogicalTypeId::INTEGER && key_type != duckdb::LogicalTypeId::BIGINT)) {
    return std::nullopt;
  }

  // The build side must hold each key at most once, or one slot would stand for several rows.
  auto const unique = prove_unique_columns(*join.children[1]);
  bool const build_is_delim_get =
    join.children[1]->type == duckdb::LogicalOperatorType::LOGICAL_DELIM_GET &&
    join.children[1]->types.size() == 1 && build_ref.Index() == 0;
  if (!build_is_delim_get && (unique.size() != 1 || unique.count(build_ref.Index()) == 0)) {
    return std::nullopt;
  }

  auto const layout = layout_of(join);
  auto const output = [&](duckdb::Expression const& expression) -> duckdb::Expression const* {
    // The expression the aggregate sees at a position of its child, in terms of the join output.
    if (expression.GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) { return nullptr; }
    auto const index = expression.Cast<duckdb::BoundReferenceExpression>().Index();
    if (below == nullptr) { return &expression; }
    if (index >= below->expressions.size()) { return nullptr; }
    return below->expressions[index].get();
  };

  // The group key must be the join key, from either side (they are equal under INNER).
  auto const key_expression = output(*aggregate.groups[0]);
  if (key_expression == nullptr ||
      key_expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
    return std::nullopt;
  }
  auto const key_position = key_expression->Cast<duckdb::BoundReferenceExpression>().Index();
  if (key_position >= layout.left_count + layout.right_count) { return std::nullopt; }
  auto const [key_child, key_column] = resolve_output(join, layout, key_position);
  if (!(key_child == 0 && key_column == probe_ref.Index()) &&
      !(key_child == 1 && key_column == build_ref.Index())) {
    return std::nullopt;
  }

  probe_accumulate_detection detection{
    probe_ref.Index(), build_ref.Index(), build_is_delim_get, {}};
  for (auto const& expression : aggregate.expressions) {
    if (expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
      return decline("aggregate list holds a non-aggregate expression");
    }
    auto const& aggr = expression->Cast<duckdb::BoundAggregateExpression>();
    if (aggr.IsDistinct() || aggr.GetFilter() || aggr.GetOrderBys() || aggr.BindInfo()) {
      return decline("DISTINCT, FILTER, ORDER BY or bound state on " +
                     aggr.Function().GetName().GetIdentifierName());
    }
    auto const id =
      sirius::from_duckdb_aggregate_name(aggr.Function().GetName().GetIdentifierName());
    auto const kind = id ? kind_of(*id) : std::nullopt;
    if (!kind) {
      return decline("aggregate " + aggr.Function().GetName().GetIdentifierName() +
                     " is not MIN/MAX/COUNT");
    }

    accumulate_aggregate entry{*kind, nullptr};
    if (*kind != op::accumulate_kind::COUNT_STAR) {
      if (aggr.GetChildren().size() != 1) { return decline("aggregate arity"); }
      // The argument is any expression over the aggregate's input. Re-express it over the join's
      // output, looking through the projection if there is one, then over the probe child.
      auto source   = aggr.GetChildren()[0]->Copy();
      bool resolved = true;
      duckdb::ExpressionIterator::VisitExpressionClassMutable(
        source,
        duckdb::ExpressionClass::BOUND_REF,
        [&](duckdb::unique_ptr<duckdb::Expression>& leaf) {
          if (below == nullptr) { return; }
          auto const index = leaf->Cast<duckdb::BoundReferenceExpression>().Index();
          if (index >= below->expressions.size()) {
            resolved = false;
            return;
          }
          leaf = below->expressions[index]->Copy();
        });
      if (!resolved) { return decline("aggregate argument references a missing column"); }
      entry.argument = address_on_probe(*source, join, layout);
      if (entry.argument == nullptr) {
        return decline("argument " + source->ToString() +
                       " reads the build side or is not translatable to the GPU");
      }

      if (*kind != op::accumulate_kind::COUNT) {
        // MIN/MAX keep their input type and need a device atomic on it.
        auto const cudf_type =
          sirius::get_cudf_type(sirius::from_duckdb(entry.argument->GetReturnType()));
        if (!cudf_type || !op::probe_accumulate_supports_type(*cudf_type) ||
            aggr.GetReturnType() != entry.argument->GetReturnType()) {
          return decline(aggr.Function().GetName().GetIdentifierName() + " over " +
                         entry.argument->GetReturnType().ToString() + " has no device atomic");
        }
      }
    } else if (!aggr.GetChildren().empty()) {
      return decline("COUNT(*) with arguments");
    }
    detection.aggregates.push_back(std::move(entry));
  }
  return detection;
}

/// Share of the smallest GPU's capacity the accumulator records may take. The records are
/// allocated up front, one per build row, beside the join's own hash table and the probe batches.
constexpr double k_record_capacity_fraction = 0.25;

/// Smallest GPU tier capacity in bytes, or nullopt when no space reports one.
std::optional<std::uint64_t> smallest_gpu_capacity(duckdb::SiriusContext& sirius_ctx)
{
  std::optional<std::uint64_t> capacity;
  for (auto const* space :
       sirius_ctx.get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
    if (space == nullptr) { continue; }
    auto const bytes = space->get_max_memory();
    capacity         = capacity ? std::min(*capacity, bytes) : bytes;
  }
  return capacity;
}

}  // namespace

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::try_plan_probe_accumulate(duckdb::LogicalAggregate& op)
{
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  if (!sirius_ctx) { return nullptr; }
  if (!duckdb::query_operator_options(context)->enable_probe_accumulate) { return nullptr; }
  // One build table, one set of accumulators, one device.
  if (sirius_ctx->get_memory_manager()
        .get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)
        .size() != 1) {
    return nullptr;
  }

  // MUST decide before touching the logical tree: planning drains it, so a decline afterwards
  // would leave the caller re-planning emptied nodes.
  auto detection = detect_probe_accumulate(op);
  if (!detection) { return nullptr; }

  auto const* below = op.children[0]->type == duckdb::LogicalOperatorType::LOGICAL_PROJECTION
                        ? op.children[0].get()
                        : nullptr;
  auto& join =
    (below ? *below->children[0] : *op.children[0]).Cast<duckdb::LogicalComparisonJoin>();

  auto const key_type          = join.conditions[0].GetLHS().GetReturnType();
  auto const build_key_type    = join.conditions[0].GetRHS().GetReturnType();
  auto const probe_cardinality = join.children[0]->EstimateCardinality(context);

  // One accumulator per aggregate; the probe column of each argument is numbered below, once the
  // probe projection that carries it exists.
  std::vector<op::accumulate_spec> specs;
  {
    cudf::size_type next_column = 1;  // column 0 of the probe projection is the join key
    for (auto const& aggregate : detection->aggregates) {
      op::accumulate_spec spec{aggregate.kind, 0, cudf::data_type{cudf::type_id::EMPTY}};
      if (aggregate.argument) {
        spec.probe_column = next_column++;
        auto const type =
          sirius::get_cudf_type(sirius::from_duckdb(aggregate.argument->GetReturnType()));
        if (type) { spec.value_type = *type; }
      }
      specs.push_back(spec);
    }
  }

  // The join must run as one single-partition BUILD_PROBE join, and the accumulators are
  // allocated up front for every build row. Decide from the estimate while the logical tree is
  // still intact; a join that cannot fit is better left on the ordinary path than failed at
  // sizing time.
  {
    auto const options      = duckdb::query_operator_options(context);
    auto const& params      = *options;
    auto const build_rows   = join.children[1]->EstimateCardinality(context);
    auto const key_bytes    = sirius::memory::saturating_mul(build_rows, std::size_t{8});
    auto const record_bytes = sirius::op::probe_accumulate_state(specs).record_bytes_for(1);
    auto const record_total = sirius::memory::saturating_mul(build_rows, record_bytes + 16);
    auto const capacity     = smallest_gpu_capacity(*sirius_ctx);
    bool const keys_fit     = key_bytes < params.max_build_hash_table_bytes;
    bool const records_fit =
      !capacity || static_cast<double>(record_total) <=
                     static_cast<double>(*capacity) * k_record_capacity_fraction;
    if (!keys_fit || !records_fit) {
      SIRIUS_LOG_INFO(
        "[sirius_plan_probe_accumulate] not fusing: build est {} rows needs {} key bytes (limit "
        "{}) "
        "and {} record bytes (limit {})",
        build_rows,
        key_bytes,
        params.max_build_hash_table_bytes,
        record_total,
        capacity
          ? static_cast<std::uint64_t>(static_cast<double>(*capacity) * k_record_capacity_fraction)
          : std::uint64_t{0});
      return nullptr;
    }
  }

  // The accumulate join reads its inputs by position, so give it a probe child that already
  // carries exactly [key, aggregate arguments...] and a one-column-per-side output.
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> projected;
  duckdb::vector<duckdb::LogicalType> projected_types;
  projected.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(key_type, detection->probe_key_idx));
  projected_types.push_back(key_type);
  for (auto& aggregate : detection->aggregates) {
    if (aggregate.argument == nullptr) { continue; }
    projected_types.push_back(aggregate.argument->GetReturnType());
    projected.push_back(std::move(aggregate.argument));
  }

  auto probe_projection =
    duckdb::make_uniq<duckdb::LogicalProjection>(duckdb::TableIndex(0), std::move(projected));
  probe_projection->types = projected_types;
  probe_projection->SetEstimatedCardinality(probe_cardinality);
  probe_projection->children.push_back(std::move(join.children[0]));
  join.children[0] = std::move(probe_projection);

  join.conditions[0].LeftReference() =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(key_type, 0);
  join.left_projection_map  = {duckdb::ProjectionIndex(0)};
  join.right_projection_map = {duckdb::ProjectionIndex(detection->build_key_idx)};
  join.types                = {key_type, build_key_type};

  auto plan = plan_comparison_join(join);
  if (plan->type != sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) {
    // Unreachable for a single INNER integer equality, but the logical tree is gone by now, so
    // surface it as the CPU-fallback signal rather than returning a half-built plan.
    throw duckdb::NotImplementedException(
      "probe-accumulate: the join did not plan to a hash join (falling back to CPU)");
  }
  auto& hash_join = plan->Cast<sirius::op::sirius_physical_hash_join>();
  if (!hash_join.unique_build_keys && detection->build_is_delim_get) {
    hash_join.unique_build_keys = true;
  }
  if (!hash_join.unique_build_keys) {
    throw duckdb::NotImplementedException(
      "probe-accumulate: build-key uniqueness was not carried to the join (falling back to CPU)");
  }

  auto context_state   = std::make_shared<sirius::op::probe_accumulate_context>(std::move(specs));
  hash_join.accumulate = context_state;

  SIRIUS_LOG_INFO(
    "[sirius_plan_probe_accumulate] Fusing GROUP BY join-key aggregate into a probe-accumulate "
    "join: {} aggregate(s), probe est {} rows",
    context_state->state.specs().size(),
    probe_cardinality);

  auto emit = duckdb::make_uniq<sirius::op::sirius_physical_probe_accumulate_emit>(
    sirius::from_duckdb_vec(op.types), op.estimated_cardinality, std::move(context_state));
  // A sink ends the join's pipeline, so the join's tasks forward their (empty) batches into the
  // emit operator's port and the emit operator starts a pipeline of its own that runs once, after
  // every probe task. Without it the emit would execute inside each join task, on whatever the
  // other tasks had accumulated so far.
  auto join_sink = duckdb::make_uniq<sirius::op::sirius_physical_passthrough_sink>(
    hash_join.types, hash_join.estimated_cardinality, "default");
  join_sink->children.push_back(std::move(plan));
  emit->children.push_back(std::move(join_sink));
  return emit;
}

}  // namespace sirius::planner
