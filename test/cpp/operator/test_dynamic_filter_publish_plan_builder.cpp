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
 * @file
 * @brief Contract tests for the validated C1a-2a planner-side publication model.
 */

#include <catch.hpp>
#include <op/dynamic_filter_publish_plan_builder.hpp>
#include <op/sirius_dynamic_filter.hpp>
#include <sirius/exception.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

using sirius::op::duckdb_filter_ordinal;
using sirius::op::dynamic_filter_identity_allocator;
using sirius::op::dynamic_filter_key_candidate;
using sirius::op::dynamic_filter_key_decision;
using sirius::op::dynamic_filter_key_plan;
using sirius::op::dynamic_filter_planning_view;
using sirius::op::dynamic_filter_publication_claim;
using sirius::op::dynamic_filter_publication_plan_id;
using sirius::op::dynamic_filter_publish_plan_builder;
using sirius::op::join_condition_index;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_key_ordinal;

static_assert(!std::is_constructible_v<dynamic_filter_publication_plan_id, std::uint32_t>);
static_assert(!std::is_copy_constructible_v<dynamic_filter_identity_allocator>);
static_assert(!std::is_move_constructible_v<dynamic_filter_identity_allocator>);
static_assert(!std::is_copy_constructible_v<dynamic_filter_publish_plan_builder>);
static_assert(!std::is_move_constructible_v<dynamic_filter_publish_plan_builder>);
static_assert(dynamic_filter_planning_view::required_publication_claim ==
              dynamic_filter_publication_claim::build_probe);

namespace sirius::op {

/// Test seam for the private resolve_keys (production resolution runs inside the hash-join
/// constructor, which this suite does not construct).
struct dynamic_filter_publish_plan_builder_test_access {
  static void resolve_keys(dynamic_filter_publish_plan_builder& builder,
                           std::vector<dynamic_filter_key_decision> decisions,
                           std::vector<dynamic_filter_key_plan> resolved_keys,
                           std::size_t build_input_column_count)
  {
    builder.resolve_keys(std::move(decisions), std::move(resolved_keys), build_input_column_count);
  }
};

}  // namespace sirius::op

using test_access = sirius::op::dynamic_filter_publish_plan_builder_test_access;

namespace {

dynamic_filter_key_candidate make_candidate(std::size_t ordinal,
                                            bool is_equality                   = true,
                                            bool has_direct_uncast_keys        = true,
                                            bool has_supported_membership_type = true)
{
  return {duckdb_filter_ordinal{ordinal},
          join_condition_index{ordinal},
          is_equality,
          has_direct_uncast_keys,
          has_supported_membership_type};
}

std::vector<dynamic_filter_key_candidate> make_candidates(std::size_t count)
{
  std::vector<dynamic_filter_key_candidate> result;
  result.reserve(count);
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    result.push_back(make_candidate(ordinal));
  }
  return result;
}

dynamic_filter_key_plan make_key(std::size_t sirius_ordinal,
                                 std::size_t duckdb_ordinal,
                                 std::size_t build_column_index = 0,
                                 cudf::type_id type             = cudf::type_id::INT32)
{
  return {sirius_key_ordinal{sirius_ordinal},
          duckdb_filter_ordinal{duckdb_ordinal},
          join_condition_index{duckdb_ordinal},
          build_column_index,
          cudf::data_type{type}};
}

dynamic_filter_publish_plan_builder::scan_target_draft make_target(
  dynamic_filter_identity_allocator& identities,
  std::size_t arity,
  std::shared_ptr<sirius_dynamic_filter_set> channel =
    std::make_shared<sirius_dynamic_filter_set>())
{
  dynamic_filter_publish_plan_builder::scan_target_draft target;
  target.target_id  = identities.mint_target_id();
  target.channel_id = identities.mint_channel_id();
  target.channel    = std::move(channel);
  for (std::size_t ordinal = 0; ordinal < arity; ++ordinal) {
    target.probe_col_idx.push_back(ordinal);
    target.probe_col_type.emplace_back(cudf::type_id::INT32);
  }
  return target;
}

dynamic_filter_publish_plan_builder::descriptor make_descriptor(
  dynamic_filter_identity_allocator& identities,
  std::vector<dynamic_filter_key_candidate> candidates,
  std::vector<dynamic_filter_publish_plan_builder::scan_target_draft> targets,
  bool build_subtree_has_filter_hint = false)
{
  dynamic_filter_publish_plan_builder::descriptor descriptor;
  descriptor.publication_plan_id           = identities.mint_publication_plan_id();
  descriptor.build_subtree_has_filter_hint = build_subtree_has_filter_hint;
  descriptor.join_condition_count          = candidates.size();
  descriptor.scan_targets                  = std::move(targets);
  descriptor.emit_zone_map_filters         = false;
  descriptor.domain_coverage_threshold     = 0.9;
  descriptor.key_candidates                = std::move(candidates);
  return descriptor;
}

std::unique_ptr<dynamic_filter_publish_plan_builder> construct(
  dynamic_filter_publish_plan_builder::descriptor descriptor)
{
  return std::make_unique<dynamic_filter_publish_plan_builder>(std::move(descriptor));
}

std::unique_ptr<dynamic_filter_publish_plan_builder> make_builder(
  dynamic_filter_identity_allocator& identities,
  std::vector<dynamic_filter_key_candidate> candidates,
  bool with_target                   = true,
  bool build_subtree_has_filter_hint = false)
{
  auto const arity = candidates.size();
  std::vector<dynamic_filter_publish_plan_builder::scan_target_draft> targets;
  if (with_target) { targets.push_back(make_target(identities, arity)); }
  return construct(make_descriptor(
    identities, std::move(candidates), std::move(targets), build_subtree_has_filter_hint));
}

}  // namespace

TEST_CASE("dynamic-filter entity IDs are allocator-only, nonzero, and category-distinct",
          "[dynamic_filter][publish_plan_builder]")
{
  dynamic_filter_identity_allocator identities;
  auto const publication_a = identities.mint_publication_plan_id();
  auto const publication_b = identities.mint_publication_plan_id();
  auto const target        = identities.mint_target_id();
  auto const channel       = identities.mint_channel_id();

  REQUIRE(publication_a.is_valid());
  REQUIRE(publication_b.is_valid());
  REQUIRE(target.is_valid());
  REQUIRE(channel.is_valid());
  REQUIRE(publication_a != publication_b);
  REQUIRE(publication_a.value() == 1);
  REQUIRE(target.value() == 1);
  REQUIRE(channel.value() == 1);
}

TEST_CASE("planning view is unavailable before single-shot key resolution",
          "[dynamic_filter][publish_plan_builder]")
{
  dynamic_filter_identity_allocator identities;
  auto builder = make_builder(identities, make_candidates(1));
  REQUIRE_THROWS_AS(builder->planning_view(), sirius::internal_exception);

  test_access::resolve_keys(*builder, {dynamic_filter_key_decision::admitted}, {make_key(0, 0)}, 1);
  REQUIRE_THROWS_AS(test_access::resolve_keys(
                      *builder, {dynamic_filter_key_decision::admitted}, {make_key(0, 0)}, 1),
                    sirius::internal_exception);
}

TEST_CASE("planning view exposes validated values but no mutable channels",
          "[dynamic_filter][publish_plan_builder]")
{
  dynamic_filter_identity_allocator identities;
  auto candidates           = make_candidates(2);
  candidates[1].is_equality = false;
  auto builder =
    make_builder(identities, std::move(candidates), true, /*build_subtree_has_filter_hint=*/false);

  test_access::resolve_keys(
    *builder,
    {dynamic_filter_key_decision::admitted, dynamic_filter_key_decision::non_equality},
    {make_key(0, 0)},
    2);
  auto const view = builder->planning_view();

  REQUIRE(view.publication_plan_id.is_valid());
  REQUIRE_FALSE(view.build_subtree_has_filter_hint);
  REQUIRE(view.enabled);
  REQUIRE(view.scan_targets.size() == 1);
  REQUIRE(view.scan_targets[0].target_id.is_valid());
  REQUIRE(view.scan_targets[0].channel_id.is_valid());
  REQUIRE((view.scan_targets[0].probe_col_idx == std::vector<std::size_t>{0, 1}));
  REQUIRE(view.by_duckdb_ordinal.size() == 2);
  REQUIRE(view.by_duckdb_ordinal[0].admitted_key->build_type ==
          cudf::data_type{cudf::type_id::INT32});
  REQUIRE_FALSE(view.by_duckdb_ordinal[1].admitted_key.has_value());
}

TEST_CASE("build-subtree hint is preserved as observation and never controls admission",
          "[dynamic_filter][publish_plan_builder]")
{
  dynamic_filter_identity_allocator identities;
  auto builder =
    make_builder(identities, make_candidates(1), true, /*build_subtree_has_filter_hint=*/false);
  test_access::resolve_keys(*builder, {dynamic_filter_key_decision::admitted}, {make_key(0, 0)}, 1);
  REQUIRE(builder->planning_view().enabled);
  REQUIRE_FALSE(builder->planning_view().build_subtree_has_filter_hint);
}

TEST_CASE("zero admitted keys produce a disabled canonical view",
          "[dynamic_filter][publish_plan_builder]")
{
  dynamic_filter_identity_allocator identities;
  auto builder = make_builder(identities, {make_candidate(0, false, true, true)});
  test_access::resolve_keys(*builder, {dynamic_filter_key_decision::non_equality}, {}, 1);

  auto const view = builder->planning_view();
  REQUIRE_FALSE(view.enabled);
  REQUIRE(view.scan_targets.size() == 1);
  REQUIRE_FALSE(view.by_duckdb_ordinal[0].admitted_key.has_value());
}

TEST_CASE("a statically eligible key may narrow to unresolved on physical drift",
          "[dynamic_filter][publish_plan_builder]")
{
  dynamic_filter_identity_allocator identities;
  auto builder = make_builder(identities, make_candidates(1));
  test_access::resolve_keys(*builder, {dynamic_filter_key_decision::unresolved}, {}, 1);
  REQUIRE_FALSE(builder->planning_view().enabled);
}
