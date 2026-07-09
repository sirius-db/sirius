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
 * @file test_dynamic_filter_publish_plan_builder.cpp
 * @brief Contract tests for dynamic_filter_publish_plan_builder (C1a-2 piece 3): the single-shot
 *        resolve protocol, the sanctioned planning view, and every rung of the finalize()
 *        validation ladder. The ladder cases are freeze-boundary sentinels: each one proves a
 *        malformed builder cannot become a runtime plan.
 */

#include <op/dynamic_filter_publish_plan.hpp>
#include <op/sirius_dynamic_filter.hpp>
#include <sirius/exception.hpp>

#include "operator_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using sirius::op::duckdb_filter_ordinal;
using sirius::op::dynamic_filter_channel_id;
using sirius::op::dynamic_filter_key_candidate;
using sirius::op::dynamic_filter_key_decision;
using sirius::op::dynamic_filter_key_plan;
using sirius::op::dynamic_filter_publication_plan_id;
using sirius::op::dynamic_filter_publish_plan_builder;
using sirius::op::dynamic_filter_target_id;
using sirius::op::join_condition_index;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_key_ordinal;

namespace {

dynamic_filter_key_candidate make_candidate(std::size_t j, bool is_equality = true)
{
  return {duckdb_filter_ordinal{j}, join_condition_index{j}, is_equality};
}

std::vector<dynamic_filter_key_candidate> make_candidates(std::size_t count)
{
  std::vector<dynamic_filter_key_candidate> result;
  for (std::size_t j = 0; j < count; ++j) {
    result.push_back(make_candidate(j));
  }
  return result;
}

dynamic_filter_key_plan make_key(std::size_t compact_ordinal,
                                 std::size_t duckdb_ordinal,
                                 std::size_t build_column_index = 0)
{
  return {sirius_key_ordinal{compact_ordinal},
          duckdb_filter_ordinal{duckdb_ordinal},
          join_condition_index{duckdb_ordinal},
          build_column_index,
          cudf::data_type{cudf::type_id::INT32}};
}

dynamic_filter_publish_plan_builder::scan_target_draft make_draft(std::uint32_t target_id,
                                                                  std::uint32_t channel_id,
                                                                  std::size_t arity)
{
  dynamic_filter_publish_plan_builder::scan_target_draft draft;
  draft.target_id  = dynamic_filter_target_id{target_id};
  draft.channel_id = dynamic_filter_channel_id{channel_id};
  draft.channel    = std::make_shared<sirius_dynamic_filter_set>();
  for (std::size_t j = 0; j < arity; ++j) {
    draft.probe_col_idx.push_back(j);
    draft.probe_col_type.emplace_back(cudf::type_id::INT32);
  }
  return draft;
}

dynamic_filter_publish_plan_builder make_builder(
  std::vector<dynamic_filter_publish_plan_builder::scan_target_draft> targets,
  std::vector<dynamic_filter_key_candidate> candidates,
  std::uint32_t publication_plan_id                                    = 1,
  std::vector<sirius::op::dynamic_filter_replica_space> replica_spaces = {})
{
  return dynamic_filter_publish_plan_builder(
    dynamic_filter_publication_plan_id{publication_plan_id},
    /*wired=*/true,
    std::move(targets),
    /*emit_zone_map_filters=*/false,
    /*domain_coverage_threshold=*/0.9,
    std::move(replica_spaces),
    std::move(candidates));
}

/// The canonical two-candidate builder: candidate 0 admitted, candidate 1 non-equality.
dynamic_filter_publish_plan_builder make_resolved_builder(std::size_t target_count = 1)
{
  std::vector<dynamic_filter_publish_plan_builder::scan_target_draft> targets;
  for (std::size_t t = 0; t < target_count; ++t) {
    targets.push_back(make_draft(static_cast<std::uint32_t>(t + 1),
                                 static_cast<std::uint32_t>(t + 1),
                                 /*arity=*/2));
  }
  auto candidates = make_candidates(2);
  candidates[1].is_equality = false;

  auto builder = make_builder(std::move(targets), std::move(candidates));
  builder.resolve_keys(
    {dynamic_filter_key_decision::admitted, dynamic_filter_key_decision::non_equality},
    {make_key(0, 0)},
    /*build_input_column_count=*/4);
  return builder;
}

}  // namespace

//===-----------------------------------------------------------------------------------------===//
// Resolve protocol
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("resolve_keys is single-shot", "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_resolved_builder();
  REQUIRE_THROWS_AS(builder.resolve_keys({}, {}, 0), sirius::internal_exception);
}

TEST_CASE("planning_view and finalize require resolved keys",
          "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_builder({make_draft(1, 1, 1)}, make_candidates(1));
  REQUIRE_THROWS_AS(builder.planning_view(), sirius::internal_exception);
  REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
}

//===-----------------------------------------------------------------------------------------===//
// The sanctioned planning view
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("planning_view exposes one entry per DuckDB ordinal with keys engaged iff admitted",
          "[dynamic_filter][publish_plan_builder]")
{
  // Candidates 0..3: admitted, non_equality, cast, admitted — partial admission with a gap,
  // the exact shape C3 must read correctly.
  auto candidates           = make_candidates(4);
  candidates[1].is_equality = false;
  auto builder              = make_builder({make_draft(1, 1, 4)}, std::move(candidates));
  builder.resolve_keys({dynamic_filter_key_decision::admitted,
                        dynamic_filter_key_decision::non_equality,
                        dynamic_filter_key_decision::cast,
                        dynamic_filter_key_decision::admitted},
                       {make_key(0, 0, /*build_column_index=*/2), make_key(1, 3)},
                       /*build_input_column_count=*/4);

  auto const view = builder.planning_view();
  REQUIRE(view.publication_plan_id == dynamic_filter_publication_plan_id{1});
  REQUIRE(view.wired);
  REQUIRE(view.enabled);
  REQUIRE(view.by_duckdb_ordinal.size() == 4);

  for (std::size_t j = 0; j < 4; ++j) {
    REQUIRE(view.by_duckdb_ordinal[j].duckdb_ordinal == duckdb_filter_ordinal{j});
  }
  auto const& admitted_first = view.by_duckdb_ordinal[0];
  REQUIRE(admitted_first.decision == dynamic_filter_key_decision::admitted);
  REQUIRE(admitted_first.admitted_key.has_value());
  REQUIRE(admitted_first.admitted_key->ordinal == sirius_key_ordinal{0});
  REQUIRE(admitted_first.admitted_key->build_column_index == 2);
  REQUIRE(admitted_first.build_type == cudf::data_type{cudf::type_id::INT32});

  REQUIRE(view.by_duckdb_ordinal[1].decision == dynamic_filter_key_decision::non_equality);
  REQUIRE_FALSE(view.by_duckdb_ordinal[1].admitted_key.has_value());
  REQUIRE_FALSE(view.by_duckdb_ordinal[1].build_type.has_value());
  REQUIRE(view.by_duckdb_ordinal[2].decision == dynamic_filter_key_decision::cast);
  REQUIRE_FALSE(view.by_duckdb_ordinal[2].admitted_key.has_value());

  auto const& admitted_second = view.by_duckdb_ordinal[3];
  REQUIRE(admitted_second.decision == dynamic_filter_key_decision::admitted);
  REQUIRE(admitted_second.admitted_key->ordinal == sirius_key_ordinal{1});

  // The view is a stable immutable value for the rest of planning: repeated reads alias the
  // same builder-owned storage.
  auto const again = builder.planning_view();
  REQUIRE(again.by_duckdb_ordinal.data() == view.by_duckdb_ordinal.data());
}

TEST_CASE("planning_view reports a scan-only zero-admitted builder as disabled",
          "[dynamic_filter][publish_plan_builder]")
{
  auto candidates           = make_candidates(1);
  candidates[0].is_equality = false;
  auto builder              = make_builder({make_draft(1, 1, 1)}, std::move(candidates));
  builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);

  auto const view = builder.planning_view();
  REQUIRE(view.wired);
  REQUIRE_FALSE(view.enabled);  // wired but nothing publishable: registration side effects only
}

//===-----------------------------------------------------------------------------------------===//
// finalize(): the validation ladder, rung by rung
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("finalize rejects a decision/candidate count mismatch",
          "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_builder({}, make_candidates(2));
  builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);  // 1 decision, 2 cands
  REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
}

TEST_CASE("finalize rejects a key/admitted-decision count mismatch",
          "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_builder({}, make_candidates(1));
  builder.resolve_keys({dynamic_filter_key_decision::admitted}, {}, 4);  // admitted, no key
  REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
}

TEST_CASE("finalize rejects non-contiguous Sirius key ordinals",
          "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_builder({}, make_candidates(1));
  builder.resolve_keys({dynamic_filter_key_decision::admitted},
                       {make_key(/*compact_ordinal=*/1, /*duckdb_ordinal=*/0)},  // must start at 0
                       4);
  REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
}

TEST_CASE("finalize rejects an admitted key whose ordinals disagree with its candidate",
          "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_builder({}, make_candidates(2));
  builder.resolve_keys(
    {dynamic_filter_key_decision::admitted, dynamic_filter_key_decision::non_equality},
    {make_key(0, /*duckdb_ordinal=*/1)},  // candidate 0 admitted, but the key claims ordinal 1
    4);
  REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
}

TEST_CASE("finalize rejects a build column outside the captured build width",
          "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_builder({}, make_candidates(1));
  builder.resolve_keys({dynamic_filter_key_decision::admitted},
                       {make_key(0, 0, /*build_column_index=*/4)},
                       /*build_input_column_count=*/4);  // width 4: valid columns are 0..3
  REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
}

TEST_CASE("finalize rejects duplicate and out-of-range DuckDB ordinals",
          "[dynamic_filter][publish_plan_builder]")
{
  SECTION("duplicate")
  {
    std::vector<dynamic_filter_key_candidate> candidates{make_candidate(0), make_candidate(0)};
    auto builder = make_builder({}, std::move(candidates));
    builder.resolve_keys(
      {dynamic_filter_key_decision::non_equality, dynamic_filter_key_decision::non_equality},
      {},
      4);
    REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
  }
  SECTION("out of range")
  {
    std::vector<dynamic_filter_key_candidate> candidates{make_candidate(5)};  // 1 candidate: j<1
    auto builder = make_builder({}, std::move(candidates));
    builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);
    REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
  }
}

TEST_CASE("finalize rejects a scan target whose arity disagrees with the DuckDB key count",
          "[dynamic_filter][publish_plan_builder]")
{
  auto builder = make_builder({make_draft(1, 1, /*arity=*/1)}, make_candidates(2));
  builder.resolve_keys(
    {dynamic_filter_key_decision::non_equality, dynamic_filter_key_decision::non_equality}, {}, 4);
  REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
}

TEST_CASE("finalize rejects zero identity", "[dynamic_filter][publish_plan_builder]")
{
  SECTION("publication plan ID")
  {
    auto builder = make_builder({make_draft(1, 1, 1)},
                                make_candidates(1),
                                /*publication_plan_id=*/0);
    builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);
    REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
  }
  SECTION("target ID")
  {
    auto builder = make_builder({make_draft(/*target_id=*/0, 1, 1)}, make_candidates(1));
    builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);
    REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
  }
  SECTION("channel ID")
  {
    auto builder = make_builder({make_draft(1, /*channel_id=*/0, 1)}, make_candidates(1));
    builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);
    REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
  }
}

TEST_CASE("finalize rejects a null channel and duplicate target IDs",
          "[dynamic_filter][publish_plan_builder]")
{
  SECTION("null channel")
  {
    auto draft    = make_draft(1, 1, 1);
    draft.channel = nullptr;
    auto builder  = make_builder({std::move(draft)}, make_candidates(1));
    builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);
    REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
  }
  SECTION("duplicate target IDs")
  {
    auto builder =
      make_builder({make_draft(1, 1, 1), make_draft(/*target_id=*/1, 2, 1)}, make_candidates(1));
    builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);
    REQUIRE_THROWS_AS(builder.finalize(), sirius::internal_exception);
  }
}

//===-----------------------------------------------------------------------------------------===//
// finalize(): valid outcomes
//===-----------------------------------------------------------------------------------------===//

TEST_CASE("a disabled builder finalizes into a valid installable plan",
          "[dynamic_filter][publish_plan_builder]")
{
  SECTION("statistics-only shape: no targets, no candidates")
  {
    auto builder = make_builder({}, {});
    builder.resolve_keys({}, {}, 0);
    auto plan = builder.finalize();
    REQUIRE(plan != nullptr);
    REQUIRE_FALSE(plan->enabled());
    REQUIRE(plan->probe_targets().empty());
  }
  SECTION("scan-only zero-admitted shape: targets but no publishable key")
  {
    auto candidates           = make_candidates(1);
    candidates[0].is_equality = false;
    auto builder              = make_builder({make_draft(1, 1, 1)}, std::move(candidates));
    builder.resolve_keys({dynamic_filter_key_decision::non_equality}, {}, 4);
    auto plan = builder.finalize();
    REQUIRE(plan != nullptr);
    REQUIRE_FALSE(plan->enabled());  // no live target reaches the runtime plan
    REQUIRE(plan->probe_targets().empty());
    REQUIRE(plan->build_key_domain_cardinalities() == std::vector<std::size_t>{0});
  }
}

TEST_CASE("an enabled builder without replica spaces fails plan construction",
          "[dynamic_filter][publish_plan_builder]")
{
  // Rung 11 is deliberately delegated to dynamic_filter_publish_plan's constructor, which
  // requires at least one GPU replica space for an enabled plan.
  auto builder = make_resolved_builder();
  REQUIRE_THROWS_AS(builder.finalize(), std::invalid_argument);
}

TEST_CASE("an enabled builder finalizes with view/plan parity",
          "[dynamic_filter][publish_plan_builder]")
{
  auto manager = sirius::test::operator_utils::initialize_memory_manager(1);
  auto* gpu    = const_cast<cucascade::memory::memory_space*>(
    manager->get_memory_space(cucascade::memory::Tier::GPU, 0));
  auto const* host = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(gpu != nullptr);
  REQUIRE(host != nullptr);

  std::vector<sirius::op::dynamic_filter_replica_space> replicas;
  replicas.emplace_back(*gpu, *host);

  auto candidates           = make_candidates(2);
  candidates[1].is_equality = false;
  auto builder =
    make_builder({make_draft(1, 1, 2), make_draft(2, 1, 2)}, std::move(candidates), 7, replicas);
  builder.resolve_keys(
    {dynamic_filter_key_decision::admitted, dynamic_filter_key_decision::non_equality},
    {make_key(0, 0)},
    4);

  auto const view = builder.planning_view();
  auto const plan = builder.finalize();

  // The view and the frozen plan must agree on everything they share.
  REQUIRE(view.enabled == plan->enabled());
  REQUIRE(view.publication_plan_id == dynamic_filter_publication_plan_id{7});
  REQUIRE(plan->probe_targets().size() == 2);
  REQUIRE(plan->probe_targets()[0].probe_col_idx == std::vector<std::size_t>{0, 1});
  REQUIRE(plan->probe_targets()[0].probe_col_type.size() == view.by_duckdb_ordinal.size());
  // C1a-2 domain evidence is null: full-arity all-zero cardinalities keep the coverage gates off.
  REQUIRE(plan->build_key_domain_cardinalities() == std::vector<std::size_t>{0, 0});
  REQUIRE_FALSE(plan->emit_zone_map_filters());
  REQUIRE(plan->domain_coverage_threshold() == 0.9);
  REQUIRE(plan->replica_spaces().size() == 1);
}
