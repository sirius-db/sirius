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

#include "planner/connector_registry.hpp"
#include "sirius_extension.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <parquet_extension.hpp>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::planner::connector;
using sirius::planner::lookup_decline;
using sirius::planner::testing::scoped_connector_overrides;

connector const& registered_parquet()
{
  for (auto const& entry : sirius::planner::registered_connectors()) {
    if (entry.function_name == "parquet_scan") return entry;
  }
  throw std::logic_error("parquet_scan registry entry is missing");
}

duckdb::LogicalGet* find_get(duckdb::LogicalOperator& node)
{
  if (node.type == duckdb::LogicalOperatorType::LOGICAL_GET)
    return &node.Cast<duckdb::LogicalGet>();
  for (auto& child : node.children) {
    if (auto* get = find_get(*child)) return get;
  }
  return nullptr;
}

duckdb::LogicalGet& parquet_get(duckdb::LogicalOperator& plan)
{
  auto* get = find_get(plan);
  REQUIRE(get);
  REQUIRE(get->function.GetName() == "parquet_scan");
  return *get;
}

struct parquet_fixture {
  // Load the extension and its real catalog functions without allocating a GPU runtime.
  sirius::test::scoped_sirius_disable disable;
  sirius::test::scratch_dir files{"connector_overrides"};
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  parquet_fixture()
  {
    db.LoadStaticExtension<duckdb::SiriusExtension>();
    db.LoadStaticExtension<duckdb::ParquetExtension>();
    REQUIRE(db.ExtensionIsLoaded("sirius"));
    REQUIRE(db.ExtensionIsLoaded("parquet"));
    auto written = con.Query("COPY (SELECT 42 AS id) TO " + files.file_literal("data.parquet") +
                             " (FORMAT PARQUET)");
    REQUIRE(written);
    if (written->HasError()) UNSCOPED_INFO(written->GetError());
    REQUIRE_FALSE(written->HasError());
    con.BeginTransaction();
  }

  duckdb::unique_ptr<duckdb::LogicalOperator> bind()
  {
    auto plan =
      con.ExtractPlan("SELECT id FROM parquet_scan(" + files.file_literal("data.parquet") + ")");
    REQUIRE(plan);
    return plan;
  }

  connector const* lookup(duckdb::LogicalOperator& plan)
  {
    return sirius::planner::lookup_connector(parquet_get(plan), *con.context);
  }
};

thread_local unsigned matcher_calls  = 0;
thread_local unsigned verifier_calls = 0;
thread_local bool reject_matcher     = false;

bool counted_matcher(duckdb::FunctionData const* bind)
{
  ++matcher_calls;
  return !reject_matcher && registered_parquet().bind_data_matches(bind);
}

bool declining_verifier(duckdb::BaseTableFunction const&,
                        duckdb::FunctionData const*,
                        duckdb::ClientContext&)
{
  ++verifier_calls;
  return false;
}

}  // namespace

TEST_CASE("Connector override seam is linked and supplies the lookup entry",
          "[planner][connector_overrides]")
{
  REQUIRE(sirius::planner::testing::connector_overrides_linked());
  parquet_fixture fixture;
  auto const& registered       = registered_parquet();
  auto replacement             = registered;
  replacement.registry_profile = "test.parquet.override";

  scoped_connector_overrides scope(*fixture.db.instance, {replacement});
  auto plan         = fixture.bind();
  auto const* entry = fixture.lookup(*plan);
  REQUIRE(entry);
  CHECK(entry != &registered);
  CHECK(entry->registry_profile == replacement.registry_profile);
  CHECK(registered.registry_profile != replacement.registry_profile);
}

TEST_CASE("Connector override declines are authoritative", "[planner][connector_overrides]")
{
  reject_matcher = GENERATE(false, true);
  parquet_fixture fixture;
  auto replacement              = registered_parquet();
  replacement.bind_data_matches = counted_matcher;
  replacement.verify_binding    = declining_verifier;
  {
    scoped_connector_overrides scope(*fixture.db.instance, {replacement});
    auto plan     = fixture.bind();
    auto& get     = parquet_get(*plan);
    matcher_calls = verifier_calls = 0;
    auto const outcome = sirius::planner::lookup_connector_classified(get, *fixture.con.context);
    CHECK(outcome.entry == nullptr);
    CHECK(outcome.decline == lookup_decline::bind_data_mismatch);
    CHECK(matcher_calls == 1);
    CHECK(verifier_calls == (reject_matcher ? 0 : 1));
  }

  auto control = fixture.bind();
  CHECK(fixture.lookup(*control) == &registered_parquet());
}

TEST_CASE("Connector override scopes isolate databases and reject overlap",
          "[planner][connector_overrides]")
{
  parquet_fixture first;
  parquet_fixture second;
  auto replacement             = registered_parquet();
  replacement.registry_profile = "test.first.database";
  {
    scoped_connector_overrides scope(*first.db.instance, {replacement});
    auto first_plan         = first.bind();
    auto const* first_entry = first.lookup(*first_plan);
    REQUIRE(first_entry);
    REQUIRE(first_entry->registry_profile == replacement.registry_profile);
    {
      auto second_plan = second.bind();
      CHECK(second.lookup(*second_plan) == &registered_parquet());
    }
    CHECK_THROWS_AS(scoped_connector_overrides(*first.db.instance, {replacement}),
                    std::logic_error);
    CHECK(first.lookup(*first_plan) == first_entry);

    {
      auto other             = registered_parquet();
      other.registry_profile = "test.second.database";
      scoped_connector_overrides second_scope(*second.db.instance, {other});
      auto second_plan         = second.bind();
      auto const* second_entry = second.lookup(*second_plan);
      REQUIRE(second_entry);
      CHECK(second_entry->registry_profile == other.registry_profile);
      CHECK(first.lookup(*first_plan) == first_entry);
      CHECK(first_entry->registry_profile == replacement.registry_profile);
    }
    CHECK(first.lookup(*first_plan) == first_entry);
    {
      auto second_plan = second.bind();
      CHECK(second.lookup(*second_plan) == &registered_parquet());
    }
  }
  {
    auto control = first.bind();
    CHECK(first.lookup(*control) == &registered_parquet());
  }
  replacement.registry_profile = "test.reinstalled.scope";
  scoped_connector_overrides reinstalled(*first.db.instance, {replacement});
  auto plan         = first.bind();
  auto const* entry = first.lookup(*plan);
  REQUIRE(entry);
  CHECK(entry->registry_profile == replacement.registry_profile);
}

TEST_CASE("Connector override registration rejects malformed records without publishing a scope",
          "[planner][connector_overrides]")
{
  parquet_fixture fixture;
  std::vector<connector> invalid{registered_parquet()};
  SECTION("null mandatory matcher") { invalid.front().bind_data_matches = nullptr; }
  SECTION("unregistered name") { invalid.front().function_name = "unregistered_test_scan"; }
  SECTION("duplicate name") { invalid.push_back(registered_parquet()); }
  CHECK_THROWS_AS(scoped_connector_overrides(*fixture.db.instance, std::move(invalid)),
                  std::invalid_argument);

  auto replacement             = registered_parquet();
  replacement.registry_profile = "test.after.rejected.registration";
  scoped_connector_overrides scope(*fixture.db.instance, {replacement});
  auto plan         = fixture.bind();
  auto const* entry = fixture.lookup(*plan);
  REQUIRE(entry);
  CHECK(entry->registry_profile == replacement.registry_profile);
}

TEST_CASE("Connector override storage outlives the bound plans that observe it",
          "[planner][connector_overrides]")
{
  parquet_fixture fixture;
  auto replacement             = registered_parquet();
  replacement.registry_profile = "test.live.bound.plan";
  {
    scoped_connector_overrides scope(*fixture.db.instance, {replacement});
    auto plan            = fixture.bind();
    auto const* observed = fixture.lookup(*plan);
    REQUIRE(observed);
    REQUIRE(observed->registry_profile == replacement.registry_profile);
    {
      auto another_plan = fixture.bind();
      CHECK(fixture.lookup(*another_plan) == observed);
    }
    CHECK(observed->registry_profile == replacement.registry_profile);
    plan.reset();
    auto after_release = fixture.bind();
    CHECK(fixture.lookup(*after_release) == observed);
    CHECK(observed->registry_profile == replacement.registry_profile);
  }
  auto control = fixture.bind();
  CHECK(fixture.lookup(*control) == &registered_parquet());
}
