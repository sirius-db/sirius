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
 * @file test_dynamic_filter_freeze.cpp
 * @brief Contract tests for the dynamic-filter freeze seam (C1a-2): constructor key resolution
 *        into the builder, the one-shot freeze/verify entry point the engine calls, the
 *        zero-slots-changed guarantee when preparation fails, and planning-view/frozen-plan
 *        parity through a real hash join.
 */

#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <helper/type_conversions.hpp>
#include <op/dynamic_filter_publish_plan.hpp>
#include <op/sirius_dynamic_filter.hpp>
#include <op/sirius_physical_hash_join.hpp>
#include <sirius/exception.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

using sirius::op::duckdb_filter_ordinal;
using sirius::op::dynamic_filter_channel_id;
using sirius::op::dynamic_filter_frozen_descriptor;
using sirius::op::dynamic_filter_key_candidate;
using sirius::op::dynamic_filter_key_decision;
using sirius::op::dynamic_filter_publication_plan_id;
using sirius::op::dynamic_filter_publish_plan_builder;
using sirius::op::dynamic_filter_target_id;
using sirius::op::join_condition_index;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_key_ordinal;
using sirius::op::sirius_physical_hash_join;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

/// A join under test plus the logical node that must outlive it.
struct join_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius_physical_hash_join> hash_join;
};

/// One INTEGER-keyed INNER join over two minimal one-column children, with an optional
/// dynamic-filter builder — the smallest join whose constructor runs real equality-key
/// extraction and key resolution.
join_fixture make_join(std::unique_ptr<dynamic_filter_publish_plan_builder> builder)
{
  join_fixture fixture;
  fixture.logical_join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  fixture.logical_join->types =
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};

  auto left_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);
  auto right_child = duckdb::make_uniq<sirius_physical_operator>(
    SiriusPhysicalOperatorType::PROJECTION,
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
    0);

  duckdb::vector<duckdb::JoinCondition> conditions;
  duckdb::JoinCondition cond;
  cond.left  = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.right = duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0);
  cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
  conditions.push_back(std::move(cond));

  fixture.hash_join = duckdb::make_uniq<sirius_physical_hash_join>(
    *fixture.logical_join,
    std::move(left_child),
    std::move(right_child),
    sirius::wrap_join_conditions(std::move(conditions)),
    duckdb::JoinType::INNER,
    duckdb::vector<duckdb::idx_t>{},
    duckdb::vector<duckdb::idx_t>{},
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
    /*estimated_cardinality=*/100,
    std::move(builder));
  return fixture;
}

/// A producer builder with @p target_count scan targets and one equality candidate for the
/// join's single condition. Zero targets keeps the finalized plan disabled, which is enough for
/// the freeze-mechanics tests and needs no GPU replica spaces.
std::unique_ptr<dynamic_filter_publish_plan_builder> make_builder(
  std::size_t target_count,
  std::vector<sirius::op::dynamic_filter_replica_space> replica_spaces = {})
{
  std::vector<dynamic_filter_publish_plan_builder::scan_target_draft> drafts;
  for (std::size_t t = 0; t < target_count; ++t) {
    dynamic_filter_publish_plan_builder::scan_target_draft draft{
      dynamic_filter_target_id{static_cast<std::uint32_t>(t + 1)},
      dynamic_filter_channel_id{1},
      std::make_shared<sirius_dynamic_filter_set>(),
      {0},
      {cudf::data_type{cudf::type_id::INT32}}};
    drafts.push_back(std::move(draft));
  }
  return std::make_unique<dynamic_filter_publish_plan_builder>(
    dynamic_filter_publication_plan_id{7},
    /*wired=*/target_count > 0,
    std::move(drafts),
    /*emit_zone_map_filters=*/false,
    /*domain_coverage_threshold=*/0.9,
    std::move(replica_spaces),
    std::vector<dynamic_filter_key_candidate>{
      {duckdb_filter_ordinal{0}, join_condition_index{0}, /*is_equality=*/true}});
}

/// A builder that passes construction but FAILS finalize(): its single candidate claims DuckDB
/// ordinal 5 in a one-candidate list, which the validation ladder rejects. Used to prove that a
/// freeze that fails half-way through preparation changes no slot.
std::unique_ptr<dynamic_filter_publish_plan_builder> make_poisoned_builder()
{
  return std::make_unique<dynamic_filter_publish_plan_builder>(
    dynamic_filter_publication_plan_id{9},
    /*wired=*/false,
    std::vector<dynamic_filter_publish_plan_builder::scan_target_draft>{},
    /*emit_zone_map_filters=*/false,
    /*domain_coverage_threshold=*/0.9,
    std::vector<sirius::op::dynamic_filter_replica_space>{},
    std::vector<dynamic_filter_key_candidate>{
      {duckdb_filter_ordinal{5}, join_condition_index{0}, /*is_equality=*/true}});
}

void freeze(join_fixture& fixture)
{
  sirius_physical_hash_join* producers[] = {fixture.hash_join.get()};
  sirius::op::freeze_or_verify_dynamic_filter_plans(producers);
}

}  // namespace

//===-----------------------------------------------------------------------------------------===//
// Constructor key resolution
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("the constructor resolves builder candidates against its extracted keys",
          "[dynamic_filter][freeze]")
{
  auto fixture = make_join(make_builder(/*target_count=*/0));

  auto const view = fixture.hash_join->planning_view();
  REQUIRE(view.publication_plan_id == dynamic_filter_publication_plan_id{7});
  REQUIRE_FALSE(view.enabled);  // zero targets: registration-only producer
  REQUIRE(view.by_duckdb_ordinal.size() == 1);
  REQUIRE(view.by_duckdb_ordinal[0].decision == dynamic_filter_key_decision::admitted);
  REQUIRE(view.by_duckdb_ordinal[0].admitted_key.has_value());
  REQUIRE(view.by_duckdb_ordinal[0].admitted_key->ordinal == sirius_key_ordinal{0});
  REQUIRE(view.by_duckdb_ordinal[0].admitted_key->build_column_index == 0);
  REQUIRE(view.by_duckdb_ordinal[0].build_type == cudf::data_type{cudf::type_id::INT32});
}

TEST_CASE("planning_view on a non-producer join fails loudly", "[dynamic_filter][freeze]")
{
  auto fixture = make_join(nullptr);
  REQUIRE_THROWS_AS(fixture.hash_join->planning_view(), sirius::internal_exception);
}

//===-----------------------------------------------------------------------------------------===//
// Freeze mechanics
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("runtime plan access before the freeze is an internal error, not a silent disabled",
          "[dynamic_filter][freeze]")
{
  auto fixture = make_join(nullptr);
  REQUIRE_FALSE(fixture.hash_join->has_frozen_dynamic_filter_plan());
  REQUIRE_THROWS_AS(fixture.hash_join->dynamic_filter_plan(), sirius::internal_exception);
}

TEST_CASE("a non-producer join freezes to the canonical disabled plan", "[dynamic_filter][freeze]")
{
  auto fixture = make_join(nullptr);
  freeze(fixture);

  REQUIRE(fixture.hash_join->has_frozen_dynamic_filter_plan());
  auto const& plan = fixture.hash_join->dynamic_filter_plan();
  REQUIRE(plan != nullptr);
  REQUIRE_FALSE(plan->enabled());
  REQUIRE_FALSE(plan->publication_plan_id().is_valid());
  REQUIRE(plan->duckdb_key_count() == 0);
}

TEST_CASE("the freeze publishes the builder's finalized plan", "[dynamic_filter][freeze]")
{
  auto fixture = make_join(make_builder(/*target_count=*/0));
  freeze(fixture);

  auto const& plan = fixture.hash_join->dynamic_filter_plan();
  REQUIRE(plan->publication_plan_id() == dynamic_filter_publication_plan_id{7});
  REQUIRE(plan->duckdb_key_count() == 1);
  REQUIRE(plan->ordinals()[0].decision == dynamic_filter_key_decision::admitted);
  REQUIRE_FALSE(plan->enabled());  // zero targets
  // Full-arity all-zero domain cardinalities: the publish coverage gates stay off in C1a-2.
  REQUIRE(plan->build_key_domain_cardinalities() == std::vector<std::size_t>{0});
}

TEST_CASE("re-freezing a cached plan verifies and reuses; it never assigns twice",
          "[dynamic_filter][freeze]")
{
  auto fixture = make_join(make_builder(/*target_count=*/0));
  freeze(fixture);
  auto const* first = fixture.hash_join->dynamic_filter_plan().get();

  freeze(fixture);  // cached re-execution takes the verify branch
  REQUIRE(fixture.hash_join->dynamic_filter_plan().get() == first);  // the same object, reused
}

TEST_CASE("a mixed frozen/unfrozen enumeration is rejected", "[dynamic_filter][freeze]")
{
  auto frozen   = make_join(nullptr);
  auto unfrozen = make_join(nullptr);
  freeze(frozen);

  sirius_physical_hash_join* producers[] = {frozen.hash_join.get(), unfrozen.hash_join.get()};
  REQUIRE_THROWS_AS(sirius::op::freeze_or_verify_dynamic_filter_plans(producers),
                    sirius::internal_exception);
  REQUIRE_FALSE(unfrozen.hash_join->has_frozen_dynamic_filter_plan());
}

TEST_CASE("a freeze that fails preparation changes no slot", "[dynamic_filter][freeze]")
{
  // Producer 1 is healthy; producer 2's builder fails the validation ladder. Preparation mints
  // producer 1's token first, then throws on producer 2 — and the token's destructor must roll
  // producer 1's slot back so the failed freeze is invisible.
  auto healthy  = make_join(make_builder(/*target_count=*/0));
  auto poisoned = make_join(make_poisoned_builder());

  sirius_physical_hash_join* producers[] = {healthy.hash_join.get(), poisoned.hash_join.get()};
  REQUIRE_THROWS_AS(sirius::op::freeze_or_verify_dynamic_filter_plans(producers),
                    sirius::internal_exception);
  REQUIRE_FALSE(healthy.hash_join->has_frozen_dynamic_filter_plan());
  REQUIRE_FALSE(poisoned.hash_join->has_frozen_dynamic_filter_plan());

  // The healthy producers remain freezable once the poisoned one is out of the enumeration.
  freeze(healthy);
  REQUIRE(healthy.hash_join->has_frozen_dynamic_filter_plan());
}

//===-----------------------------------------------------------------------------------------===//
// Topology descriptors
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("planned and frozen descriptors agree after a freeze", "[dynamic_filter][freeze]")
{
  auto fixture                           = make_join(make_builder(/*target_count=*/0));
  sirius_physical_hash_join* producers[] = {fixture.hash_join.get()};
  freeze(fixture);

  auto const planned = sirius::op::describe_planned_dynamic_filter_topology(producers);
  auto const frozen  = sirius::op::describe_frozen_dynamic_filter_topology(producers);
  REQUIRE(planned == frozen);
  REQUIRE(planned.digest() == frozen.digest());
  // And the pair-form verify accepts them without touching any slot.
  sirius::op::verify_frozen_dynamic_filter_topology(frozen, planned);
}

TEST_CASE("verify rejects a topology that changed shape", "[dynamic_filter][freeze]")
{
  dynamic_filter_frozen_descriptor cached;
  cached.producers.push_back({dynamic_filter_publication_plan_id{1},
                              true,
                              {static_cast<std::uint8_t>(dynamic_filter_key_decision::admitted)},
                              {dynamic_filter_target_id{1}},
                              {dynamic_filter_channel_id{1}}});
  auto current                      = cached;
  current.producers[0].decisions[0] = static_cast<std::uint8_t>(dynamic_filter_key_decision::cast);

  REQUIRE_FALSE(cached == current);
  REQUIRE_THROWS_AS(sirius::op::verify_frozen_dynamic_filter_topology(cached, current),
                    sirius::internal_exception);
}

//===-----------------------------------------------------------------------------------------===//
// End-to-end parity on an enabled plan (GPU fixture)
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("an enabled producer freezes with planning-view/frozen-plan parity",
          "[dynamic_filter][freeze]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager(1);
  auto* gpu    = const_cast<cucascade::memory::memory_space*>(
    manager->get_memory_space(cucascade::memory::Tier::GPU, 0));
  auto const* host = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(gpu != nullptr);
  REQUIRE(host != nullptr);
  std::vector<sirius::op::dynamic_filter_replica_space> replicas;
  replicas.emplace_back(*gpu, *host);

  auto fixture    = make_join(make_builder(/*target_count=*/2, std::move(replicas)));
  auto const view = fixture.hash_join->planning_view();
  freeze(fixture);
  auto const& plan = fixture.hash_join->dynamic_filter_plan();

  REQUIRE(plan->enabled());
  REQUIRE(view.enabled == plan->enabled());
  REQUIRE(view.publication_plan_id == plan->publication_plan_id());
  REQUIRE(view.by_duckdb_ordinal.size() == plan->ordinals().size());
  REQUIRE(view.by_duckdb_ordinal[0].decision == plan->ordinals()[0].decision);
  REQUIRE(view.by_duckdb_ordinal[0].admitted_key->build_column_index ==
          plan->ordinals()[0].admitted_key->build_column_index);
  // The strong IDs the planner minted travel into the frozen targets.
  REQUIRE(plan->probe_targets().size() == 2);
  REQUIRE(plan->probe_targets()[0].target_id == dynamic_filter_target_id{1});
  REQUIRE(plan->probe_targets()[1].target_id == dynamic_filter_target_id{2});
  REQUIRE(plan->probe_targets()[0].channel_id == dynamic_filter_channel_id{1});
}
