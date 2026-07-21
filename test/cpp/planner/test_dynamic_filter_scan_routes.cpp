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

/**
 * @file test_dynamic_filter_scan_routes.cpp
 * @brief Arm-by-arm tests for sirius::planner::resolve_scan_routes -- the plan-time decision that
 *        turns one join's DuckDB-named probe targets into Sirius scan endpoints
 *
 * Candidates come from `duckdb_join_filter_candidate_adapter::extract` over synthetic logical
 * joins, so the inputs are the exact values the planner routes, and the channel factory is a stub
 * that records what routing asked for. Neither a GPU nor a `duckdb::ClientContext` is involved.
 *
 * Every routing decision has a case here: `no_duckdb_metadata`, `metadata_malformed`,
 * `build_side_unfiltered` from both of the candidate shapes that reach it, `no_device_placement`,
 * `no_live_channel`, and `wired`. Two properties recur because they are the ones a refactor is
 * likeliest to break: a producer is registered on a channel only when that producer really wires
 * it, and dropping a target keeps `targets` and `target_inputs` aligned.
 */

#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "planner/duckdb_join_filter_candidate_adapter.hpp"
#include "planner/dynamic_filter_scan_routes.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb/common/typedefs.hpp>
#include <duckdb/execution/operator/join/join_filter_pushdown.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using sirius::op::dynamic_filter_route_class;
using sirius::op::sirius_dynamic_filter_set;
using sirius::planner::dynamic_filter_routing_decision;
using sirius::planner::resolve_scan_routes;
using sirius::planner::resolved_scan_routes;
using sirius::planner::duckdb_join_filter_candidate_adapter::extract;

/// Channel push ordinals of target @p target_slot at DuckDB filter ordinal @p filter_ordinal, drawn
/// from a range no other coordinate in this file occupies: a routing bug that read a condition
/// index, a filter ordinal, or a target index where a push ordinal belongs yields a visibly
/// out-of-range value rather than a plausible one, and the per-target offset makes a target mix-up
/// equally visible.
constexpr std::size_t push_ordinal(std::size_t target_slot, std::size_t filter_ordinal)
{
  return 200 + (10 * target_slot) + filter_ordinal;
}

duckdb::JoinCondition make_condition()
{
  duckdb::JoinCondition condition;
  condition.left =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0ULL);
  condition.right =
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0ULL);
  condition.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  return condition;
}

/// A two-equality-condition join. Its pushdown hint, where a test attaches one, names conditions
/// `{1, 0}` -- permuted, so DuckDB's filter-ordinal order can never be mistaken for condition
/// order.
duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_join()
{
  auto join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  join->conditions.push_back(make_condition());
  join->conditions.push_back(make_condition());
  return join;
}

/// Attach pushdown metadata naming both conditions in permuted order.
void attach_pushdown_info(duckdb::LogicalComparisonJoin& join, bool build_side_has_filter = true)
{
  join.filter_pushdown                        = duckdb::make_uniq<duckdb::JoinFilterPushdownInfo>();
  join.filter_pushdown->join_condition        = duckdb::vector<duckdb::idx_t>{1, 0};
  join.filter_pushdown->build_side_has_filter = build_side_has_filter;
}

/// One probe target in slot @p target_slot: a column per filter ordinal, at this slot's push
/// ordinals, with @p storage_type as the probe-side storage type.
duckdb::JoinFilterPushdownFilter make_target(
  duckdb::shared_ptr<duckdb::DynamicTableFilterSet> channel_identity,
  std::size_t target_slot,
  duckdb::LogicalType storage_type = duckdb::LogicalType::INTEGER)
{
  duckdb::JoinFilterPushdownFilter target;
  target.dynamic_filters = std::move(channel_identity);
  for (std::size_t filter_ordinal = 0; filter_ordinal < 2; ++filter_ordinal) {
    duckdb::JoinFilterPushdownColumn column;
    column.probe_column_index = duckdb::ColumnBinding{0, push_ordinal(target_slot, filter_ordinal)};
    column.storage_type       = storage_type;
    target.columns.push_back(column);
  }
  return target;
}

/// Stands in for `sirius_physical_plan_generator::get_or_create_dynamic_filter_channel`: records
/// every identity routing asks about, and answers with the channel the test staged for that
/// identity or with null when none was staged (which is also how the disabled master switch
/// answers in production).
class stub_channel_factory {
 public:
  void stage(duckdb::DynamicTableFilterSet const* identity,
             std::shared_ptr<sirius_dynamic_filter_set> channel)
  {
    _staged.emplace(identity, std::move(channel));
  }

  std::shared_ptr<sirius_dynamic_filter_set> operator()(
    duckdb::DynamicTableFilterSet const* identity)
  {
    _requests.push_back(identity);
    auto const staged = _staged.find(identity);
    return staged == _staged.end() ? nullptr : staged->second;
  }

  [[nodiscard]] std::vector<duckdb::DynamicTableFilterSet const*> const& requests() const noexcept
  {
    return _requests;
  }

 private:
  std::unordered_map<duckdb::DynamicTableFilterSet const*,
                     std::shared_ptr<sirius_dynamic_filter_set>>
    _staged;
  std::vector<duckdb::DynamicTableFilterSet const*> _requests;
};

/// The shape every non-wiring decision shares: no endpoints in either vector.
void require_nothing_wired(resolved_scan_routes const& routes)
{
  REQUIRE(routes.targets.empty());
  REQUIRE(routes.target_inputs.empty());
}

}  // namespace

//===-----------------------------------------------------------------------------------------===//
// Arms that stop before any channel is minted
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("routing reports absent DuckDB metadata and asks for no channel",
          "[dynamic_filter][scan_routes]")
{
  auto join = make_join();  // no filter_pushdown attached at all
  stub_channel_factory factory;

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::no_duckdb_metadata);
  require_nothing_wired(routes);
  REQUIRE(routes.hinted_condition_indexes.empty());
  REQUIRE(factory.requests().empty());
}

TEST_CASE("routing reports malformed metadata and asks for no channel",
          "[dynamic_filter][scan_routes]")
{
  auto join = make_join();
  attach_pushdown_info(*join);
  // A hint naming a condition the join does not have: the adapter fails this candidate closed.
  join->filter_pushdown->join_condition = duckdb::vector<duckdb::idx_t>{1, 7};
  join->filter_pushdown->probe_info.push_back(
    make_target(duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>(), 0));
  stub_channel_factory factory;

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::metadata_malformed);
  require_nothing_wired(routes);
  // A malformed candidate carries no usable hint, so admission is left unrestricted.
  REQUIRE(routes.hinted_condition_indexes.empty());
  REQUIRE(factory.requests().empty());
}

TEST_CASE("routing reports an unfiltered build for a statistics-only candidate",
          "[dynamic_filter][scan_routes]")
{
  auto join = make_join();
  attach_pushdown_info(*join, /*build_side_has_filter=*/false);  // no probe target recorded
  stub_channel_factory factory;

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::build_side_unfiltered);
  require_nothing_wired(routes);
  REQUIRE(routes.hinted_condition_indexes.empty());
  REQUIRE(factory.requests().empty());
}

TEST_CASE("routing reports an unfiltered build when an admitted candidate carries no build hint",
          "[dynamic_filter][scan_routes]")
{
  auto channel_identity = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join();
  attach_pushdown_info(*join, /*build_side_has_filter=*/false);
  join->filter_pushdown->probe_info.push_back(make_target(channel_identity, 0));
  auto channel = std::make_shared<sirius_dynamic_filter_set>();
  stub_channel_factory factory;
  factory.stage(channel_identity.get(), channel);

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::build_side_unfiltered);
  require_nothing_wired(routes);
  // Unlike the statistics-only candidate above, DuckDB did name a condition set here, and
  // admission stays restricted to it even though this producer wires nothing.
  REQUIRE(routes.hinted_condition_indexes == std::vector<std::size_t>{1, 0});
  REQUIRE(factory.requests().empty());
  REQUIRE_FALSE(channel->has_producers());
}

TEST_CASE("routing stops before minting when device placement is unavailable",
          "[dynamic_filter][scan_routes]")
{
  auto channel_identity = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join();
  attach_pushdown_info(*join);
  join->filter_pushdown->probe_info.push_back(make_target(channel_identity, 0));
  auto channel = std::make_shared<sirius_dynamic_filter_set>();
  stub_channel_factory factory;
  factory.stage(channel_identity.get(), channel);

  auto const routes = resolve_scan_routes(extract(*join), false, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::no_device_placement);
  require_nothing_wired(routes);
  REQUIRE(routes.hinted_condition_indexes == std::vector<std::size_t>{1, 0});
  // The ordering that matters: placement is settled first, so no producer is left registered on a
  // channel this join then abandons.
  REQUIRE(factory.requests().empty());
  REQUIRE_FALSE(channel->has_producers());
}

//===-----------------------------------------------------------------------------------------===//
// Arms that reach the channel factory
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("routing wires every named target and registers this producer on each channel",
          "[dynamic_filter][scan_routes]")
{
  auto identity_a = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto identity_b = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join       = make_join();
  attach_pushdown_info(*join);
  join->filter_pushdown->probe_info.push_back(make_target(identity_a, 0));
  join->filter_pushdown->probe_info.push_back(make_target(identity_b, 1));
  auto channel_a = std::make_shared<sirius_dynamic_filter_set>();
  auto channel_b = std::make_shared<sirius_dynamic_filter_set>();
  stub_channel_factory factory;
  factory.stage(identity_a.get(), channel_a);
  factory.stage(identity_b.get(), channel_b);

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::wired);
  REQUIRE(routes.hinted_condition_indexes == std::vector<std::size_t>{1, 0});
  REQUIRE(factory.requests() ==
          std::vector<duckdb::DynamicTableFilterSet const*>{identity_a.get(), identity_b.get()});

  REQUIRE(routes.targets.size() == 2);
  REQUIRE(routes.target_inputs.size() == 2);
  REQUIRE(routes.targets[0].filter_set == channel_a);
  REQUIRE(routes.targets[1].filter_set == channel_b);
  REQUIRE(channel_a->has_producers());
  REQUIRE(channel_b->has_producers());

  for (auto const& target : routes.targets) {
    REQUIRE(target.route_class == dynamic_filter_route_class::scan);
    REQUIRE(target.accepts_zone_map_filters);
    // Bindings are admission's output; routing leaves them for the planner to move in.
    REQUIRE(target.key_bindings.empty());
  }

  for (std::size_t target_slot = 0; target_slot < routes.target_inputs.size(); ++target_slot) {
    auto const& columns = routes.target_inputs[target_slot].columns;
    REQUIRE(columns.size() == 2);
    for (std::size_t filter_ordinal = 0; filter_ordinal < columns.size(); ++filter_ordinal) {
      REQUIRE(columns[filter_ordinal].channel_push_ordinal ==
              push_ordinal(target_slot, filter_ordinal));
      REQUIRE(columns[filter_ordinal].probe_storage_type == cudf::data_type{cudf::type_id::INT32});
    }
  }
}

TEST_CASE("routing reports no live channel when every target mints null",
          "[dynamic_filter][scan_routes]")
{
  auto identity_a = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto identity_b = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join       = make_join();
  attach_pushdown_info(*join);
  join->filter_pushdown->probe_info.push_back(make_target(identity_a, 0));
  join->filter_pushdown->probe_info.push_back(make_target(identity_b, 1));
  stub_channel_factory factory;  // nothing staged: every mint answers null

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::no_live_channel);
  require_nothing_wired(routes);
  // Both identities were still offered to the factory -- the drop is the factory's answer, not a
  // decision routing took on its own.
  REQUIRE(factory.requests() ==
          std::vector<duckdb::DynamicTableFilterSet const*>{identity_a.get(), identity_b.get()});
  REQUIRE(routes.hinted_condition_indexes == std::vector<std::size_t>{1, 0});
}

TEST_CASE("routing drops a target that mints null and keeps its live sibling aligned",
          "[dynamic_filter][scan_routes]")
{
  auto identity_a = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto identity_b = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join       = make_join();
  attach_pushdown_info(*join);
  join->filter_pushdown->probe_info.push_back(make_target(identity_a, 0));
  join->filter_pushdown->probe_info.push_back(make_target(identity_b, 1));
  auto channel_b = std::make_shared<sirius_dynamic_filter_set>();
  stub_channel_factory factory;
  factory.stage(identity_b.get(), channel_b);  // target slot 0 mints null, slot 1 does not

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  REQUIRE(routes.decision == dynamic_filter_routing_decision::wired);
  REQUIRE(routes.targets.size() == 1);
  REQUIRE(routes.target_inputs.size() == 1);
  REQUIRE(routes.targets[0].filter_set == channel_b);
  REQUIRE(channel_b->has_producers());
  // The surviving entry carries slot 1's push ordinals: the drop compacted both vectors together
  // rather than leaving the surviving target described by its dropped sibling's coordinates.
  REQUIRE(routes.target_inputs[0].columns.size() == 2);
  REQUIRE(routes.target_inputs[0].columns[0].channel_push_ordinal == push_ordinal(1, 0));
  REQUIRE(routes.target_inputs[0].columns[1].channel_push_ordinal == push_ordinal(1, 1));
}

TEST_CASE("routing leaves an unmappable probe type EMPTY without dropping its target",
          "[dynamic_filter][scan_routes]")
{
  auto channel_identity = duckdb::make_shared_ptr<duckdb::DynamicTableFilterSet>();
  auto join             = make_join();
  attach_pushdown_info(*join);
  join->filter_pushdown->probe_info.push_back(
    make_target(channel_identity, 0, duckdb::LogicalType::SQLNULL));
  auto channel = std::make_shared<sirius_dynamic_filter_set>();
  stub_channel_factory factory;
  factory.stage(channel_identity.get(), channel);

  auto const routes = resolve_scan_routes(extract(*join), true, factory);

  // A probe type with no cuDF representation is a routine classification: the target is still
  // wired, and only zone maps for that binding are suppressed.
  REQUIRE(routes.decision == dynamic_filter_routing_decision::wired);
  REQUIRE(routes.target_inputs.size() == 1);
  REQUIRE(routes.target_inputs[0].columns.size() == 2);
  for (auto const& column : routes.target_inputs[0].columns) {
    REQUIRE(column.probe_storage_type == cudf::data_type{cudf::type_id::EMPTY});
  }
  REQUIRE(routes.target_inputs[0].columns[0].channel_push_ordinal == push_ordinal(0, 0));
}
