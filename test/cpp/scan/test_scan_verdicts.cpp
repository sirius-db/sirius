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

#include "op/scan/table_scan/scan_contract.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <set>

using namespace sirius::op::scan;

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a two-pass verdict records every scan before lowering",
                 "[scan][certificate][verdict][integration]")
{
  run_ok("CREATE TABLE r2a_pass(i INTEGER)");
  run_ok("INSERT INTO r2a_pass VALUES (1),(2)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=false");
  for (auto const& injected : {"unsupported", "incomplete", "unsupported@2"}) {
    CAPTURE(injected);
    run_ok(std::string("SET sirius_test_inject_scan_verdict='") + injected + "'");
    run_ok("BEGIN TRANSACTION READ ONLY");
    auto logical = con->ExtractPlan("SELECT i FROM r2a_pass UNION ALL SELECT i FROM r2a_pass");
    sirius::planner::sirius_physical_plan_generator generator(*con->context);
    auto before = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE_THROWS_AS(generator.create_plan(std::move(logical)), scan_verdict_declined);
    auto const& entries = generator.read_views->entries();
    REQUIRE(entries.size() == 2);
    auto declined = std::string(injected).find("@2") == std::string::npos ? 0 : 1;
    CHECK(entries[1 - declined].eligibility.verdict == eligibility_verdict::supported);
    CHECK(entries[declined].eligibility.verdict != eligibility_verdict::supported);
    auto after = sirius::test::get_transparent_execution_stats(*con);
    CHECK(after.scan_lowerings == before.scan_lowerings);
    CHECK(after.executions == before.executions);
    CHECK(after.window_tasks_started == before.window_tasks_started);
    for (auto const& entry : entries)
      CHECK(entry.eligibility.verdict != eligibility_verdict::not_evaluated);
    run_ok("ROLLBACK");
  }
}

TEST_CASE_METHOD(
  sirius::test::GpuExecutionFixture,
  "R2a certification budget counts in production mode and declines only under test flag",
  "[scan][certificate][verdict][integration]")
{
  run_ok("CREATE TABLE r2a_budget(i INTEGER)");
  run_ok("INSERT INTO r2a_budget VALUES (1)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=false");
  for (bool decline : {false, true}) {
    for (bool time : {false, true}) {
      CAPTURE(decline, time);
      run_ok(std::string("SET sirius_test_budget_declines=") + (decline ? "true" : "false"));
      run_ok(std::string("SET sirius_test_inject_certification_delay_ms=") + (time ? "51" : "0"));
      run_ok(std::string("SET sirius_test_inject_certification_bytes=") + (time ? "0" : "8388609"));
      run_ok("BEGIN TRANSACTION READ ONLY");
      auto logical =
        con->ExtractPlan("SELECT i FROM r2a_budget UNION ALL SELECT i FROM r2a_budget");
      sirius::planner::sirius_physical_plan_generator generator(*con->context);
      auto before = sirius::test::get_transparent_execution_stats(*con);
      if (decline)
        REQUIRE_THROWS_AS(generator.create_plan(std::move(logical)), scan_verdict_declined);
      else
        REQUIRE_NOTHROW(generator.create_plan(std::move(logical)));
      REQUIRE(generator.read_views->entries().size() == 2);
      for (auto const& entry : generator.read_views->entries()) {
        CHECK(entry.eligibility.verdict ==
              (decline ? eligibility_verdict::incomplete : eligibility_verdict::supported));
        CHECK(entry.eligibility.reason ==
              (decline ? (time ? verdict_reason::budget_time : verdict_reason::budget_bytes)
                       : verdict_reason::none));
        CHECK(entry.eligibility.cost.added_bytes > 0);
      }
      auto after = sirius::test::get_transparent_execution_stats(*con);
      CHECK(after.budget_exceeded[time ? 0 : 1] == before.budget_exceeded[time ? 0 : 1] + 1);
      if (decline) CHECK(after.scan_lowerings == before.scan_lowerings);
      run_ok("ROLLBACK");
    }
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a semantic lineage distinguishes export from semantic input",
                 "[scan][certificate][lineage][integration]")
{
  run_ok("CREATE TABLE r2a_lineage(a INTEGER, b INTEGER, c INTEGER)");
  run_ok("INSERT INTO r2a_lineage VALUES (1,2,3),(4,5,6)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=false");
  auto inspect = [&](std::string const& sql, bool expected) {
    CAPTURE(sql);
    run_ok("BEGIN TRANSACTION READ ONLY");
    auto logical = con->ExtractPlan(sql);
    sirius::planner::sirius_physical_plan_generator generator(*con->context);
    REQUIRE_NOTHROW(generator.create_plan(std::move(logical)));
    REQUIRE_FALSE(generator.read_views->entries().empty());
    bool marked = false;
    for (auto const& entry : generator.read_views->entries()) {
      REQUIRE_FALSE(entry.eligibility.semantic_columns.empty());
      for (bool bit : entry.eligibility.semantic_columns)
        marked |= bit;
    }
    CHECK(marked == expected);
    run_ok("ROLLBACK");
  };
  inspect("SELECT b,a FROM r2a_lineage", false);
  inspect("SELECT a+1 FROM r2a_lineage", true);
  inspect("SELECT a FROM r2a_lineage WHERE b=2", true);
  inspect("SELECT sum(b) FROM r2a_lineage GROUP BY a", true);
  inspect("SELECT a FROM r2a_lineage ORDER BY b", true);
  inspect("SELECT a FROM r2a_lineage LIMIT 1", false);
  inspect("SELECT a FROM r2a_lineage UNION ALL SELECT b FROM r2a_lineage", false);
  inspect(
    "SELECT a FROM r2a_lineage UNION ALL SELECT b FROM r2a_lineage UNION ALL SELECT c FROM "
    "r2a_lineage",
    false);
  inspect("WITH q AS MATERIALIZED (SELECT * FROM r2a_lineage) SELECT a FROM q", true);
  run_ok("SET sirius_test_lineage_unmodelled=true");
  inspect("SELECT a FROM r2a_lineage", true);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a lineage follows reordered join outputs and pure-filter inputs",
                 "[scan][verdict][lineage][integration]")
{
  run_ok("CREATE TABLE r2a_leaf(a INTEGER,b INTEGER,c INTEGER)");
  run_ok("INSERT INTO r2a_leaf VALUES (1,2,3),(2,1,4)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=false");
  auto inspect = [&](std::string const& sql, std::multiset<std::set<uint64_t>> expected) {
    CAPTURE(sql);
    run_ok("BEGIN TRANSACTION READ ONLY");
    auto logical = con->ExtractPlan(sql);
    sirius::planner::sirius_physical_plan_generator generator(*con->context);
    REQUIRE_NOTHROW(generator.create_plan(std::move(logical)));
    std::multiset<std::set<uint64_t>> actual;
    for (auto const& entry : generator.read_views->entries()) {
      std::set<uint64_t> used;
      for (std::size_t i = 0; i < entry.eligibility.semantic_columns.size(); ++i)
        if (entry.eligibility.semantic_columns[i])
          used.insert(entry.contract.columns.column_ids.at(i).GetPrimaryIndex());
      actual.insert(std::move(used));
    }
    CHECK(actual == expected);
    run_ok("ROLLBACK");
  };
  inspect("SELECT c,a FROM r2a_leaf", {{}});
  inspect("SELECT c FROM r2a_leaf WHERE b=2", {{1}});
  inspect("SELECT c,a+1 FROM r2a_leaf", {{0}});
  inspect("SELECT c FROM r2a_leaf ORDER BY b", {{1}});
  inspect("SELECT l.c,r.a+1 FROM r2a_leaf l JOIN r2a_leaf r ON l.a=r.b", {{0}, {0, 1}});
  inspect("SELECT l.c FROM r2a_leaf l WHERE EXISTS (SELECT 1 FROM r2a_leaf r WHERE r.b=l.a)",
          {{0}, {1}});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "R2a budget cannot overwrite established unsupported verdict",
                 "[scan][verdict][integration]")
{
  run_ok("CREATE TABLE r2a_priority(i INTEGER)");
  run_ok("INSERT INTO r2a_priority VALUES (1)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=false");
  run_ok("SET sirius_test_inject_scan_verdict='unsupported'");
  run_ok("SET sirius_test_inject_certification_bytes=8388609");
  run_ok("SET sirius_test_budget_declines=true");
  run_ok("BEGIN TRANSACTION READ ONLY");
  auto logical =
    con->ExtractPlan("SELECT i FROM r2a_priority UNION ALL SELECT i FROM r2a_priority");
  sirius::planner::sirius_physical_plan_generator generator(*con->context);
  REQUIRE_THROWS_AS(generator.create_plan(std::move(logical)), scan_verdict_declined);
  auto const& entries = generator.read_views->entries();
  REQUIRE(entries.size() == 2);
  CHECK(entries[0].eligibility.verdict == eligibility_verdict::unsupported);
  CHECK(entries[0].eligibility.reason == verdict_reason::carrier_missing);
  CHECK(entries[1].eligibility.verdict == eligibility_verdict::incomplete);
  CHECK(entries[1].eligibility.reason == verdict_reason::budget_bytes);
  run_ok("ROLLBACK");
}
