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

// `join_condition` only forward-declares `sirius::ast::node`, so destroying a
// `vector<join_condition>` needs the definition in this translation unit.
#include "expression/ast/node.hpp"
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "planner/dynamic_filter/dynamic_filter_key_admission.hpp"

#include <catch.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/joinside.hpp>

#include <cstddef>
#include <limits>
#include <optional>
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

// Build wrapped equalities with independently chosen probe and build ordinals.
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

// Build INTEGER equalities whose condition, probe, and build ordinals coincide.
duckdb::vector<sirius::join_condition> make_wrapped_equalities(std::size_t count)
{
  duckdb::vector<duckdb::JoinCondition> conditions;
  for (std::size_t i = 0; i < count; ++i) {
    conditions.push_back(make_condition(make_ref(i), make_ref(i)));
  }
  return sirius::wrap_join_conditions(std::move(conditions));
}

// Return the admitted key expected from make_wrapped_equalities.
dynamic_filter_publish_plan::admitted_key expected_key(
  std::size_t condition_index, dynamic_filter_condition_shape shape = kDirectDirect)
{
  return dynamic_filter_publish_plan::admitted_key{
    .planner_condition_index = condition_index,
    .build_key_ordinal       = static_cast<cudf::size_type>(condition_index),
    .probe_key_ordinal       = static_cast<cudf::size_type>(condition_index),
    .storage_type            = kInt32,
    .probe_storage_type      = kInt32,
    .key_shape               = shape};
}

}  // namespace

//===----------------------------------------------------------------------===//
// Coordinate-space separation
//===----------------------------------------------------------------------===//

TEST_CASE("admission reads build and probe ordinals from their own sides, not the condition index",
          "[dynamic_filter][key_admission]")
{
  // Probe ordinals 9 and 3, build ordinals 4 and 7. Every coordinate differs from every other, so
  // reading the probe side, or substituting the condition index, fails.
  auto const conditions = make_wrapped_equalities_at({{9, 4}, {3, 7}});
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});

  REQUIRE(admitted.size() == 2);
  REQUIRE(admitted[0].planner_condition_index == 0);
  REQUIRE(admitted[0].build_key_ordinal == 4);
  REQUIRE(admitted[1].planner_condition_index == 1);
  REQUIRE(admitted[1].build_key_ordinal == 7);
  // The probe ordinal is read from the probe (left) side, never the build side.
  REQUIRE(admitted[0].probe_key_ordinal == 9);
  REQUIRE(admitted[1].probe_key_ordinal == 3);
}

TEST_CASE("admission records each side's own storage type", "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(make_condition(make_ref(0), make_ref(0)));
  raw.push_back(make_condition(make_ref(1, duckdb::LogicalType::BIGINT),
                               make_ref(1, duckdb::LogicalType::BIGINT)));
  raw.push_back(
    make_condition(make_ref(2, duckdb::LogicalType::DATE), make_ref(2, duckdb::LogicalType::DATE)));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(3, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});

  REQUIRE(admitted.size() == 3);
  REQUIRE(admitted[0].storage_type == kInt32);
  REQUIRE(admitted[1].storage_type == kInt64);
  REQUIRE(admitted[2].storage_type == cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
  // Each side records its own type; `direct_route_admissible` compares the two.
  REQUIRE(admitted[0].probe_storage_type == kInt32);
  REQUIRE(admitted[1].probe_storage_type == kInt64);
  REQUIRE(admitted[2].probe_storage_type == cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
}

TEST_CASE("admission requires a probe-side bound reference", "[dynamic_filter][key_admission]")
{
  // Discovery traces start at the key's probe ordinal, so a condition without a probe-side
  // reference admits no key. (The carried shape stays direct so only the expression differs.)
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(make_condition(
    duckdb::BoundCastExpression::AddDefaultCastToType(make_ref(0), duckdb::LogicalType::BIGINT),
    make_ref(0, duckdb::LogicalType::BIGINT)));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(1, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});
  REQUIRE(admitted.empty());
}

TEST_CASE("admission rejects a build ordinal outside the cuDF column range",
          "[dynamic_filter][key_admission]")
{
  auto const oversized =
    static_cast<duckdb::idx_t>(std::numeric_limits<cudf::size_type>::max()) + 1;
  auto const conditions = make_wrapped_equalities_at({{0, oversized}});
  std::vector<dynamic_filter_condition_shape> const shapes(1, kDirectDirect);

  REQUIRE_THROWS_AS(admit_dynamic_filter_keys(conditions, shapes, {}), std::invalid_argument);
}

TEST_CASE("admission rejects a build side that is not a plain column reference",
          "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(make_condition(
    make_ref(0), duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::INTEGER(1))));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(1, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});
  REQUIRE(admitted.empty());
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

TEST_CASE("admission admits every legal condition in planner order",
          "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(3);
  std::vector<dynamic_filter_condition_shape> const shapes(3, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});
  REQUIRE(admitted == std::vector<dynamic_filter_publish_plan::admitted_key>{
                        expected_key(0), expected_key(1), expected_key(2)});
}

TEST_CASE("admission keeps partially eligible composites", "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(2);
  // The cast build key is ineligible; the independent direct key remains eligible.
  std::vector<dynamic_filter_condition_shape> const shapes{
    {.probe = dynamic_filter_key_shape::direct, .build = dynamic_filter_key_shape::cast},
    kDirectDirect};

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});
  REQUIRE(admitted == std::vector<dynamic_filter_publish_plan::admitted_key>{expected_key(1)});
}

TEST_CASE("admission never admits an inequality condition", "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(make_condition(make_ref(0), make_ref(0)));
  raw.push_back(
    make_condition(make_ref(1), make_ref(1), duckdb::ExpressionType::COMPARE_GREATERTHAN));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});
  REQUIRE(admitted == std::vector<dynamic_filter_publish_plan::admitted_key>{expected_key(0)});
}

TEST_CASE("admission never admits a null-equal condition", "[dynamic_filter][key_admission]")
{
  duckdb::vector<duckdb::JoinCondition> raw;
  raw.push_back(
    make_condition(make_ref(0), make_ref(0), duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM));
  auto const conditions = sirius::wrap_join_conditions(std::move(raw));
  std::vector<dynamic_filter_condition_shape> const shapes(1, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});
  REQUIRE(admitted.empty());
}

TEST_CASE("admitted keys without any target build a valid but disabled plan",
          "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(2);
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  auto admitted = admit_dynamic_filter_keys(conditions, shapes, {});
  REQUIRE(admitted ==
          std::vector<dynamic_filter_publish_plan::admitted_key>{expected_key(0), expected_key(1)});

  // Publication is a no-op until discovery binds a target.
  dynamic_filter_publish_plan const plan{std::move(admitted), {}, {}};
  REQUIRE_FALSE(plan.enabled());
}

TEST_CASE("admission re-emits domain cardinalities in admitted order",
          "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(3);
  std::vector<dynamic_filter_condition_shape> const shapes(3, kDirectDirect);
  std::vector<std::size_t> const condition_domains{10, 20, 30};

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, condition_domains);
  REQUIRE(admitted.size() == 3);
  REQUIRE(admitted[0].build_key_domain_cardinality == 10);
  REQUIRE(admitted[1].build_key_domain_cardinality == 20);
  REQUIRE(admitted[2].build_key_domain_cardinality == 30);
}

TEST_CASE("admission records zero domains from an empty domain vector",
          "[dynamic_filter][key_admission]")
{
  // The empty vector is the no-domain-evidence encoding (a nonempty vector of the wrong length is
  // a programming error and throws): every admitted key's gate stays disabled.
  auto const conditions = make_wrapped_equalities(2);
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});

  REQUIRE(admitted.size() == 2);
  REQUIRE(admitted[0].build_key_domain_cardinality == 0);
  REQUIRE(admitted[1].build_key_domain_cardinality == 0);
}

TEST_CASE("admission marks only the key whose build ordinal is the proven-unique column",
          "[dynamic_filter][key_admission]")
{
  // Build ordinals 4 and 7; the planner proved ordinal 7 unique.
  auto const conditions = make_wrapped_equalities_at({{9, 4}, {3, 7}});
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  SECTION("a singleton proof marks the matching ordinal")
  {
    auto const admitted =
      admit_dynamic_filter_keys(conditions, shapes, {}, std::optional<std::size_t>{7});

    REQUIRE(admitted.size() == 2);
    REQUIRE_FALSE(admitted[0].build_key_proven_unique);
    REQUIRE(admitted[1].build_key_proven_unique);
  }
  SECTION("no proof marks nothing")
  {
    auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});

    REQUIRE(admitted.size() == 2);
    REQUIRE_FALSE(admitted[0].build_key_proven_unique);
    REQUIRE_FALSE(admitted[1].build_key_proven_unique);
  }
}

TEST_CASE("admission rejects inconsistent caller input", "[dynamic_filter][key_admission]")
{
  auto const conditions = make_wrapped_equalities(2);
  std::vector<dynamic_filter_condition_shape> const shapes(2, kDirectDirect);

  SECTION("misaligned shapes")
  {
    std::vector<dynamic_filter_condition_shape> const bad_shapes(1, kDirectDirect);
    REQUIRE_THROWS_AS(admit_dynamic_filter_keys(conditions, bad_shapes, {}), std::invalid_argument);
  }
  SECTION("misaligned domain cardinalities")
  {
    std::vector<std::size_t> const bad_domains{1};
    REQUIRE_THROWS_AS(admit_dynamic_filter_keys(conditions, shapes, bad_domains),
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
    // value-correct).
    auto const conditions = make_wrapped_equalities(2);
    std::vector<dynamic_filter_condition_shape> const shapes{probe_computed, build_computed};
    auto const admitted = admit_dynamic_filter_keys(conditions, shapes, {});
    REQUIRE(admitted == std::vector<dynamic_filter_publish_plan::admitted_key>{
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
    // The plan assumes ingress validation and transports the threshold unchanged.
    REQUIRE_NOTHROW(
      dynamic_filter_publish_plan({expected_key(0)}, {}, {}, {.domain_coverage_threshold = 0.0}));
  }
  SECTION("equality is differential over every field")
  {
    // The structured binding fails to compile if a field is added without extending this section.
    auto const& [planner_condition_index,
                 build_key_ordinal,
                 probe_key_ordinal,
                 storage_type,
                 probe_storage_type,
                 key_shape,
                 domain,
                 unique] = expected_key(3);
    (void)planner_condition_index;
    (void)build_key_ordinal;
    (void)probe_key_ordinal;
    (void)storage_type;
    (void)probe_storage_type;
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

    auto vary_probe_key_ordinal              = base;
    vary_probe_key_ordinal.probe_key_ordinal = 9;
    REQUIRE_FALSE(base == vary_probe_key_ordinal);

    auto vary_storage_type         = base;
    vary_storage_type.storage_type = kInt64;
    REQUIRE_FALSE(base == vary_storage_type);

    auto vary_probe_storage_type               = base;
    vary_probe_storage_type.probe_storage_type = kInt64;
    REQUIRE_FALSE(base == vary_probe_storage_type);

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
