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

#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "planner/dynamic_filter_key_admission.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/joinside.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using sirius::op::dynamic_filter_condition_shape;
using sirius::op::dynamic_filter_key_shape;
using sirius::op::dynamic_filter_publish_plan;
using sirius::planner::admit_dynamic_filter_keys;
using sirius::planner::classify_join_key_shapes;
using sirius::planner::direct_route_admissible;
using sirius::planner::dynamic_filter_scan_target_input;

constexpr auto kInt32 = cudf::data_type{cudf::type_id::INT32};
constexpr auto kInt64 = cudf::data_type{cudf::type_id::INT64};

constexpr dynamic_filter_condition_shape kDirectDirect{.probe = dynamic_filter_key_shape::direct,
                                                       .build = dynamic_filter_key_shape::direct};

duckdb::unique_ptr<duckdb::Expression> make_ref(
  duckdb::idx_t index, duckdb::LogicalType type = duckdb::LogicalType::INTEGER)
{
  return duckdb::make_uniq<duckdb::BoundReferenceExpression>(std::move(type), index);
}

duckdb::JoinCondition make_condition(
  duckdb::unique_ptr<duckdb::Expression> left,
  duckdb::unique_ptr<duckdb::Expression> right,
  duckdb::ExpressionType comparison = duckdb::ExpressionType::COMPARE_EQUAL)
{
  duckdb::JoinCondition condition;
  condition.left       = std::move(left);
  condition.right      = std::move(right);
  condition.comparison = comparison;
  return condition;
}

/// Wrapped equality conditions whose probe and build ordinals are given per condition, so the
/// condition index, the build ordinal, and the probe ordinal are three different numbers. Any
/// confusion between those spaces changes an asserted value.
duckdb::vector<sirius::join_condition> make_wrapped_equalities_at(
  std::vector<std::pair<duckdb::idx_t, duckdb::idx_t>> const& probe_build_ordinals,
  duckdb::LogicalType type = duckdb::LogicalType::INTEGER)
{
  duckdb::vector<duckdb::JoinCondition> conditions;
  for (auto const& [probe_ordinal, build_ordinal] : probe_build_ordinals) {
    conditions.push_back(
      make_condition(make_ref(probe_ordinal, type), make_ref(build_ordinal, type)));
  }
  return sirius::wrap_join_conditions(std::move(conditions));
}

/// Wrapped equality conditions over INTEGER references: condition i compares L(i) with R(i).
duckdb::vector<sirius::join_condition> make_wrapped_equalities(std::size_t count)
{
  duckdb::vector<duckdb::JoinCondition> conditions;
  for (std::size_t i = 0; i < count; ++i) {
    conditions.push_back(make_condition(make_ref(i), make_ref(i)));
  }
  return sirius::wrap_join_conditions(std::move(conditions));
}

dynamic_filter_publish_plan::admitted_key expected_key(
  std::size_t condition_index, dynamic_filter_condition_shape shape = kDirectDirect)
{
  return dynamic_filter_publish_plan::admitted_key{
    .planner_condition_index = condition_index,
    .build_key_ordinal       = static_cast<cudf::size_type>(condition_index),
    .storage_type            = kInt32,
    .key_shape               = shape};
}

}  // namespace

//===----------------------------------------------------------------------===//
// Coordinate-space separation
//===----------------------------------------------------------------------===//

TEST_CASE("admission reads the build ordinal from the build side, not the condition index",
          "[dynamic_filter][key_admission]")
{
  // Probe ordinals 9 and 3, build ordinals 4 and 7, hinted in reverse. Every coordinate differs
  // from every other, so reading the probe side, or substituting the condition index, fails.
  auto const conditions = make_wrapped_equalities_at({{9, 4}, {3, 7}});
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);
  std::vector<std::size_t> const hinted{1, 0};
  std::vector<dynamic_filter_scan_target_input> const targets{
    {.columns = {{.channel_push_ordinal = 21, .probe_storage_type = kInt32},
                 {.channel_push_ordinal = 8, .probe_storage_type = kInt32}}}};

  auto const result = admit_dynamic_filter_keys(
    conditions, shapes, std::span<std::size_t const>{hinted}, targets, {});

  REQUIRE(result.admitted_keys.size() == 2);
  REQUIRE(result.admitted_keys[0].planner_condition_index == 1);
  REQUIRE(result.admitted_keys[0].build_key_ordinal == 7);
  REQUIRE(result.admitted_keys[1].planner_condition_index == 0);
  REQUIRE(result.admitted_keys[1].build_key_ordinal == 4);
  REQUIRE(result.per_target_key_bindings[0] ==
          std::vector<dynamic_filter_publish_plan::key_binding>{
            {.admitted_key_index = 0, .channel_push_ordinal = 21, .probe_storage_type = kInt32},
            {.admitted_key_index = 1, .channel_push_ordinal = 8, .probe_storage_type = kInt32}});
}

TEST_CASE("admission records each build side's own storage type", "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(make_condition(make_ref(0), make_ref(0)));
  raw.push_back(make_condition(make_ref(1, duckdb::LogicalType::BIGINT),
                               make_ref(1, duckdb::LogicalType::BIGINT)));
  raw.push_back(
    make_condition(make_ref(2, duckdb::LogicalType::DATE), make_ref(2, duckdb::LogicalType::DATE)));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(3, kDirectDirect);

  auto const result = admit_dynamic_filter_keys(conditions, shapes, std::nullopt, {}, {});

  REQUIRE(result.admitted_keys.size() == 3);
  REQUIRE(result.admitted_keys[0].storage_type == kInt32);
  REQUIRE(result.admitted_keys[1].storage_type == kInt64);
  REQUIRE(result.admitted_keys[2].storage_type == cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
}

TEST_CASE("admission rejects a build ordinal outside the cuDF column range",
          "[dynamic_filter][key_admission]")
{
  auto const oversized =
    static_cast<duckdb::idx_t>(std::numeric_limits<cudf::size_type>::max()) + 1;
  auto const conditions = make_wrapped_equalities_at({{0, oversized}});
  std::vector<dynamic_filter_condition_shape> const shapes(1, kDirectDirect);

  REQUIRE_THROWS_AS(admit_dynamic_filter_keys(conditions, shapes, std::nullopt, {}, {}),
                    std::invalid_argument);
}

TEST_CASE("admission rejects a build side that is not a plain column reference",
          "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(make_condition(
    make_ref(0), duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1))));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(1, kDirectDirect);

  auto const result = admit_dynamic_filter_keys(conditions, shapes, std::nullopt, {}, {});
  REQUIRE(result.admitted_keys.empty());
}

//===----------------------------------------------------------------------===//
// classify_join_key_shapes
//===----------------------------------------------------------------------===//

TEST_CASE("key-shape classification distinguishes direct, cast, and computed sides",
          "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> conditions;
  // direct = direct
  conditions.push_back(make_condition(make_ref(0), make_ref(0)));
  // cast(direct) = direct
  conditions.push_back(make_condition(
    duckdb::BoundCastExpression::AddDefaultCastToType(make_ref(1), duckdb::LogicalType::BIGINT),
    make_ref(1, duckdb::LogicalType::BIGINT)));
  // computed (constant) = cast(computed)
  conditions.push_back(
    make_condition(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(42)),
                   duckdb::BoundCastExpression::AddDefaultCastToType(
                     duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(7)),
                     duckdb::LogicalType::BIGINT)));

  auto const shapes = classify_join_key_shapes(conditions);
  REQUIRE(shapes.size() == 3);
  REQUIRE(shapes[0] == kDirectDirect);
  REQUIRE(shapes[1] == dynamic_filter_condition_shape{.probe = dynamic_filter_key_shape::cast,
                                                      .build = dynamic_filter_key_shape::direct});
  REQUIRE(shapes[2] == dynamic_filter_condition_shape{.probe = dynamic_filter_key_shape::computed,
                                                      .build = dynamic_filter_key_shape::computed});
}

//===----------------------------------------------------------------------===//
// admit_dynamic_filter_keys
//===----------------------------------------------------------------------===//

TEST_CASE("admission binds reordered non-prefix hinted keys to their channel push ordinals",
          "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(3);
  std::vector<dynamic_filter_condition_shape> const shapes(3, kDirectDirect);
  // DuckDB discovery names conditions {2, 0}, in that filter-ordinal order.
  std::vector<std::size_t> const hinted{2, 0};
  std::vector<dynamic_filter_scan_target_input> const targets{
    {.columns = {{.channel_push_ordinal = 12, .probe_storage_type = kInt32},
                 {.channel_push_ordinal = 7, .probe_storage_type = kInt32}}}};

  auto const result = admit_dynamic_filter_keys(
    conditions, shapes, std::span<std::size_t const>{hinted}, targets, {});

  REQUIRE(result.admitted_keys ==
          std::vector<dynamic_filter_publish_plan::admitted_key>{expected_key(2), expected_key(0)});
  REQUIRE(result.per_target_key_bindings.size() == 1);
  REQUIRE(result.per_target_key_bindings[0] ==
          std::vector<dynamic_filter_publish_plan::key_binding>{
            {.admitted_key_index = 0, .channel_push_ordinal = 12, .probe_storage_type = kInt32},
            {.admitted_key_index = 1, .channel_push_ordinal = 7, .probe_storage_type = kInt32}});
}

TEST_CASE("admission keeps partially eligible composites correctly bound",
          "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(2);
  // Condition 0 carries a cast on its build side: excluded, matching the runtime publisher's cast
  // skip before producer-key admission moved to plan time. Condition 1 stays eligible and must
  // keep its own channel_push_ordinal (9) after admitted-key compaction.
  std::vector<dynamic_filter_condition_shape> const shapes{
    {.probe = dynamic_filter_key_shape::direct, .build = dynamic_filter_key_shape::cast},
    kDirectDirect};
  std::vector<std::size_t> const hinted{0, 1};
  std::vector<dynamic_filter_scan_target_input> const targets{
    {.columns = {{.channel_push_ordinal = 4, .probe_storage_type = kInt32},
                 {.channel_push_ordinal = 9, .probe_storage_type = kInt32}}}};

  auto const result = admit_dynamic_filter_keys(
    conditions, shapes, std::span<std::size_t const>{hinted}, targets, {});

  REQUIRE(result.admitted_keys ==
          std::vector<dynamic_filter_publish_plan::admitted_key>{expected_key(1)});
  REQUIRE(result.per_target_key_bindings[0] ==
          std::vector<dynamic_filter_publish_plan::key_binding>{
            {.admitted_key_index = 0, .channel_push_ordinal = 9, .probe_storage_type = kInt32}});
}

TEST_CASE("admission never admits an inequality condition", "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(make_condition(make_ref(0), make_ref(0)));
  raw.push_back(
    make_condition(make_ref(1), make_ref(1), duckdb::ExpressionType::COMPARE_GREATERTHAN));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  // DuckDB hints inequality comparisons too (`<`, `<=`, `>`, `>=`), so a hint naming a non-equality
  // condition is reachable. The pre-seam runtime skipped those by a bounds check that held only
  // because DuckDB reorders equalities to the front before emitting hint indexes; admission rejects
  // them on the comparison itself and depends on no such ordering.
  std::vector<std::size_t> const hinted{1};
  std::vector<dynamic_filter_scan_target_input> const targets{
    {.columns = {{.channel_push_ordinal = 3, .probe_storage_type = kInt32}}}};

  auto const result = admit_dynamic_filter_keys(
    conditions, shapes, std::span<std::size_t const>{hinted}, targets, {});
  REQUIRE(result.admitted_keys.empty());
  REQUIRE(result.per_target_key_bindings[0].empty());
}

TEST_CASE("admission works without DuckDB discovery", "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(2);
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  auto result = admit_dynamic_filter_keys(conditions, shapes, std::nullopt, {}, {});
  REQUIRE(result.admitted_keys ==
          std::vector<dynamic_filter_publish_plan::admitted_key>{expected_key(0), expected_key(1)});
  REQUIRE(result.per_target_key_bindings.empty());

  // Admitted keys without any target build a valid but disabled plan: publication is a no-op
  // until a target exists.
  dynamic_filter_publish_plan const plan{std::move(result.admitted_keys), {}, {}};
  REQUIRE_FALSE(plan.enabled());
}

TEST_CASE("admission re-emits domain cardinalities in admitted order",
          "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(3);
  std::vector<dynamic_filter_condition_shape> const shapes(3, kDirectDirect);
  std::vector<std::size_t> const hinted{2, 0};
  std::vector<dynamic_filter_scan_target_input> const targets{
    {.columns = {{.channel_push_ordinal = 1, .probe_storage_type = kInt32},
                 {.channel_push_ordinal = 0, .probe_storage_type = kInt32}}}};
  std::vector<std::size_t> const condition_domains{10, 20, 30};

  auto const result = admit_dynamic_filter_keys(
    conditions, shapes, std::span<std::size_t const>{hinted}, targets, condition_domains);
  // Recorded on each key rather than in a parallel array, so it cannot drift out of alignment.
  REQUIRE(result.admitted_keys[0].build_key_domain_cardinality == 30);
  REQUIRE(result.admitted_keys[1].build_key_domain_cardinality == 10);
}

TEST_CASE("admission records zero domains from an empty domain vector",
          "[dynamic_filter][key_admission]")
{
  // The empty vector is the no-domain-evidence encoding (a nonempty vector of the wrong length is
  // a programming error and throws): every admitted key's gate stays disabled.
  auto const conditions = make_wrapped_equalities(2);
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);
  std::vector<std::size_t> const hinted{1, 0};
  std::vector<dynamic_filter_scan_target_input> const targets{
    {.columns = {{.channel_push_ordinal = 1, .probe_storage_type = kInt32},
                 {.channel_push_ordinal = 0, .probe_storage_type = kInt32}}}};

  auto const result = admit_dynamic_filter_keys(
    conditions, shapes, std::span<std::size_t const>{hinted}, targets, {});

  REQUIRE(result.admitted_keys.size() == 2);
  REQUIRE(result.admitted_keys[0].build_key_domain_cardinality == 0);
  REQUIRE(result.admitted_keys[1].build_key_domain_cardinality == 0);
}

TEST_CASE("admission marks only the key whose build ordinal is the proven-unique column",
          "[dynamic_filter][key_admission]")
{
  // Build ordinals 4 and 7; the planner proved ordinal 7 unique. Only that key arms the gate --
  // and passing no proof arms nothing.
  auto const conditions = make_wrapped_equalities_at({{9, 4}, {3, 7}});
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  SECTION("a singleton proof marks the matching ordinal")
  {
    auto const result = admit_dynamic_filter_keys(
      conditions, shapes, std::nullopt, {}, {}, std::optional<std::size_t>{7});

    REQUIRE(result.admitted_keys.size() == 2);
    REQUIRE_FALSE(result.admitted_keys[0].build_key_proven_unique);
    REQUIRE(result.admitted_keys[1].build_key_proven_unique);
  }
  SECTION("no proof marks nothing")
  {
    auto const result = admit_dynamic_filter_keys(conditions, shapes, std::nullopt, {}, {});

    REQUIRE(result.admitted_keys.size() == 2);
    REQUIRE_FALSE(result.admitted_keys[0].build_key_proven_unique);
    REQUIRE_FALSE(result.admitted_keys[1].build_key_proven_unique);
  }
}

TEST_CASE("admission rejects inconsistent caller input", "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(2);
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);
  std::vector<std::size_t> const hinted{0};
  std::vector<dynamic_filter_scan_target_input> const targets{
    {.columns = {{.channel_push_ordinal = 3, .probe_storage_type = kInt32}}}};

  SECTION("misaligned shapes")
  {
    std::vector<dynamic_filter_condition_shape> const bad_shapes(1, kDirectDirect);
    REQUIRE_THROWS_AS(admit_dynamic_filter_keys(
                        conditions, bad_shapes, std::span<std::size_t const>{hinted}, targets, {}),
                      std::invalid_argument);
  }
  SECTION("targets without discovery")
  {
    REQUIRE_THROWS_AS(admit_dynamic_filter_keys(conditions, shapes, std::nullopt, targets, {}),
                      std::invalid_argument);
  }
  SECTION("hinted condition index out of range")
  {
    std::vector<std::size_t> const bad_hint{5};
    REQUIRE_THROWS_AS(admit_dynamic_filter_keys(
                        conditions, shapes, std::span<std::size_t const>{bad_hint}, targets, {}),
                      std::invalid_argument);
  }
  SECTION("target arity mismatched with the discovery")
  {
    std::vector<dynamic_filter_scan_target_input> const bad_targets{
      {.columns = {{.channel_push_ordinal = 3, .probe_storage_type = kInt32},
                   {.channel_push_ordinal = 4, .probe_storage_type = kInt32}}}};
    REQUIRE_THROWS_AS(admit_dynamic_filter_keys(
                        conditions, shapes, std::span<std::size_t const>{hinted}, bad_targets, {}),
                      std::invalid_argument);
  }
  SECTION("misaligned domain cardinalities")
  {
    std::vector<std::size_t> const bad_domains{1};
    REQUIRE_THROWS_AS(
      admit_dynamic_filter_keys(
        conditions, shapes, std::span<std::size_t const>{hinted}, targets, bad_domains),
      std::invalid_argument);
  }
}

//===----------------------------------------------------------------------===//
// direct_route_admissible
//===----------------------------------------------------------------------===//

TEST_CASE("join-edge route accepts only direct matching INT32/INT64 equality keys",
          "[dynamic_filter][key_admission]")
{
  auto const equal = sirius::comparison_type::equal;

  SECTION("positives")
  {
    REQUIRE(direct_route_admissible(duckdb::JoinType::INNER, equal, kDirectDirect, kInt32, kInt32));
    REQUIRE(direct_route_admissible(duckdb::JoinType::SEMI, equal, kDirectDirect, kInt64, kInt64));
  }
  SECTION("computed keys are rejected on either side, though the scan route admits them")
  {
    auto const probe_computed = dynamic_filter_condition_shape{
      .probe = dynamic_filter_key_shape::computed, .build = dynamic_filter_key_shape::direct};
    auto const build_computed = dynamic_filter_condition_shape{
      .probe = dynamic_filter_key_shape::direct, .build = dynamic_filter_key_shape::computed};
    REQUIRE_FALSE(
      direct_route_admissible(duckdb::JoinType::INNER, equal, probe_computed, kInt32, kInt32));
    REQUIRE_FALSE(
      direct_route_admissible(duckdb::JoinType::INNER, equal, build_computed, kInt32, kInt32));

    // The scan route keeps admitting the same computed shapes (materialized keys are
    // value-correct); assert both variants so the scopes cannot silently converge.
    auto const conditions = make_wrapped_equalities(2);
    std::vector<dynamic_filter_condition_shape> const shapes{probe_computed, build_computed};
    auto const result = admit_dynamic_filter_keys(conditions, shapes, std::nullopt, {}, {});
    REQUIRE(result.admitted_keys ==
            std::vector<dynamic_filter_publish_plan::admitted_key>{
              expected_key(0, probe_computed), expected_key(1, build_computed)});
  }
  SECTION("negatives")
  {
    REQUIRE_FALSE(
      direct_route_admissible(duckdb::JoinType::LEFT, equal, kDirectDirect, kInt32, kInt32));
    REQUIRE_FALSE(
      direct_route_admissible(duckdb::JoinType::ANTI, equal, kDirectDirect, kInt32, kInt32));
    REQUIRE_FALSE(direct_route_admissible(duckdb::JoinType::INNER,
                                          sirius::comparison_type::not_distinct_from,
                                          kDirectDirect,
                                          kInt32,
                                          kInt32));
    auto const cast_build = dynamic_filter_condition_shape{
      .probe = dynamic_filter_key_shape::direct, .build = dynamic_filter_key_shape::cast};
    REQUIRE_FALSE(
      direct_route_admissible(duckdb::JoinType::INNER, equal, cast_build, kInt32, kInt32));
    REQUIRE_FALSE(
      direct_route_admissible(duckdb::JoinType::INNER, equal, kDirectDirect, kInt32, kInt64));
    REQUIRE_FALSE(direct_route_admissible(duckdb::JoinType::INNER,
                                          equal,
                                          kDirectDirect,
                                          cudf::data_type{cudf::type_id::INT16},
                                          cudf::data_type{cudf::type_id::INT16}));
  }
}

//===----------------------------------------------------------------------===//
// dynamic_filter_publish_plan invariants (CPU-checkable subset)
//===----------------------------------------------------------------------===//

TEST_CASE("publish plan rejects inconsistent admitted-key metadata",
          "[dynamic_filter][key_admission]")
{
  SECTION("duplicate condition index")
  {
    REQUIRE_THROWS_AS(dynamic_filter_publish_plan({expected_key(0), expected_key(0)}, {}, {}),
                      std::invalid_argument);
  }
  SECTION("negative build ordinal")
  {
    auto key              = expected_key(0);
    key.build_key_ordinal = -1;
    REQUIRE_THROWS_AS(dynamic_filter_publish_plan({key}, {}, {}), std::invalid_argument);
  }
  SECTION("EMPTY storage type")
  {
    auto key         = expected_key(0);
    key.storage_type = cudf::data_type{cudf::type_id::EMPTY};
    REQUIRE_THROWS_AS(dynamic_filter_publish_plan({key}, {}, {}), std::invalid_argument);
  }
  SECTION("a transported threshold is accepted without re-validation")
  {
    // The threshold is validated once, where it enters the engine
    // (config::valid_domain_coverage_threshold). A plan is built for every GPU hash join, so it
    // must not be able to fail planning over a value it merely transports.
    REQUIRE_NOTHROW(
      dynamic_filter_publish_plan({expected_key(0)}, {}, {}, {.domain_coverage_threshold = 0.0}));
  }
  SECTION("equality is differential over every field")
  {
    // A structured binding fails to compile if a field is added without extending this section,
    // so the check cannot silently fall behind the struct.
    auto const& [planner_condition_index,
                 build_key_ordinal,
                 storage_type,
                 key_shape,
                 domain,
                 unique] = expected_key(3);
    (void)planner_condition_index;
    (void)build_key_ordinal;
    (void)storage_type;
    (void)key_shape;
    (void)domain;
    (void)unique;

    auto const base = expected_key(3);
    REQUIRE(base == expected_key(3));

    auto vary_planner_condition_index                    = base;
    vary_planner_condition_index.planner_condition_index = 4;
    REQUIRE_FALSE(base == vary_planner_condition_index);

    auto vary_build_key_ordinal              = base;
    vary_build_key_ordinal.build_key_ordinal = 9;
    REQUIRE_FALSE(base == vary_build_key_ordinal);

    auto vary_storage_type         = base;
    vary_storage_type.storage_type = kInt64;
    REQUIRE_FALSE(base == vary_storage_type);

    auto vary_key_shape            = base;
    vary_key_shape.key_shape.build = dynamic_filter_key_shape::computed;
    REQUIRE_FALSE(base == vary_key_shape);

    auto vary_domain                         = base;
    vary_domain.build_key_domain_cardinality = 1000;
    REQUIRE_FALSE(base == vary_domain);

    auto vary_proven_unique                    = base;
    vary_proven_unique.build_key_proven_unique = true;
    REQUIRE_FALSE(base == vary_proven_unique);
  }
}

//===----------------------------------------------------------------------===//
// Carried shapes stay aligned through the physical join's condition reorder
//===----------------------------------------------------------------------===//

namespace {

/// Minimal logical/physical join pair; the logical join must outlive the physical join because
/// the physical join stores op.types by reference.
struct hash_join_shapes_fixture {
  duckdb::unique_ptr<duckdb::LogicalComparisonJoin> logical_join;
  duckdb::unique_ptr<sirius::op::sirius_physical_hash_join> hash_join;
};

hash_join_shapes_fixture make_hash_join_with_shapes(
  duckdb::vector<duckdb::JoinCondition> conditions,
  std::vector<dynamic_filter_condition_shape> shapes)
{
  hash_join_shapes_fixture fixture;
  fixture.logical_join = duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  fixture.logical_join->types = duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER};

  auto make_child = [] {
    return duckdb::make_uniq<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::PROJECTION,
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER,
                                                                  duckdb::LogicalType::INTEGER}),
      0);
  };
  fixture.hash_join = duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(
    *fixture.logical_join,
    make_child(),
    make_child(),
    sirius::wrap_join_conditions(std::move(conditions)),
    duckdb::JoinType::INNER,
    duckdb::vector<duckdb::idx_t>{},
    duckdb::vector<duckdb::idx_t>{},
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{}),
    /*estimated_cardinality=*/1000,
    sirius::config::DEFAULT_MAX_BUILD_HASH_TABLE_BYTES,
    sirius::op::dynamic_filter_publish_plan{},
    sirius::config::DEFAULT_HASH_PARTITION_BYTES,
    sirius::config::DEFAULT_MAX_BROADCAST_JOIN_SIZE,
    std::move(shapes));
  return fixture;
}

}  // namespace

TEST_CASE("condition_key_shapes[i] describes conditions[i] after equality-first reordering",
          "[dynamic_filter][key_admission]")
{
  // Conditions in original order: eq(0), gt(0), eq(1) -- the reorder moves the inequality last.
  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.push_back(make_condition(make_ref(0), make_ref(0)));
  conditions.push_back(
    make_condition(make_ref(0), make_ref(0), duckdb::ExpressionType::COMPARE_GREATERTHAN));
  conditions.push_back(make_condition(make_ref(1), make_ref(1)));

  // Per-condition-unique shapes so any misalignment is observable.
  auto const shape_a = kDirectDirect;
  auto const shape_b = dynamic_filter_condition_shape{.probe = dynamic_filter_key_shape::cast,
                                                      .build = dynamic_filter_key_shape::cast};
  auto const shape_c = dynamic_filter_condition_shape{.probe = dynamic_filter_key_shape::computed,
                                                      .build = dynamic_filter_key_shape::direct};

  auto const fixture =
    make_hash_join_with_shapes(std::move(conditions), {shape_a, shape_b, shape_c});
  auto const& join = *fixture.hash_join;

  REQUIRE(join.conditions.size() == 3);
  REQUIRE(join.has_condition_key_shapes());
  // Reordered condition order is [eq(0), eq(1), gt(0)]; the shapes must have followed.
  REQUIRE(join.conditions[0].comparison == sirius::comparison_type::equal);
  REQUIRE(join.conditions[1].comparison == sirius::comparison_type::equal);
  REQUIRE(join.conditions[2].comparison == sirius::comparison_type::gt);
  REQUIRE(join.key_shape_of_condition(0) == shape_a);
  REQUIRE(join.key_shape_of_condition(1) == shape_c);
  REQUIRE(join.key_shape_of_condition(2) == shape_b);
  // Out of range yields the default shape rather than reading past the end.
  REQUIRE(join.key_shape_of_condition(3) == dynamic_filter_condition_shape{});
}

TEST_CASE("the physical join rejects misaligned condition_key_shapes",
          "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> conditions;
  conditions.push_back(make_condition(make_ref(0), make_ref(0)));

  REQUIRE_THROWS_AS(
    make_hash_join_with_shapes(std::move(conditions), {kDirectDirect, kDirectDirect}),
    std::invalid_argument);
}
