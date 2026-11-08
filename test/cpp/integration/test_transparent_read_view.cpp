/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/pipeline_conversion_test_utils.hpp"
#include "utils/read_view_test_observer.hpp"

#include <catch.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

class ReadViewFixture : public sirius::test::GpuExecutionFixture {
 public:
  ReadViewFixture()
    : original_optimizers(duckdb::DBConfig::GetConfig(*con->context).options.disabled_optimizers)
  {
  }
  ~ReadViewFixture()
  {
    // disabled_optimizers is database-global. RESET inside these tests clears
    // Sirius's load-time mask too; restore it before another fixture borrows
    // this database, including when an assertion unwinds the test early.
    duckdb::DBConfig::GetConfig(*con->context).options.disabled_optimizers =
      std::move(original_optimizers);
  }

 private:
  duckdb::set<duckdb::OptimizerType> original_optimizers;
};

namespace {
struct read_view_settings_guard {
  duckdb::Connection& con;
  ~read_view_settings_guard()
  {
    con.Query("ROLLBACK");
    con.Query("PRAGMA enable_optimizer");
    con.Query("RESET disabled_optimizers");
    con.Query("SET sirius_test_inject_pin_registry_change = false");
    con.Query("SET sirius_test_inject_read_view_mismatch = 'off'");
    con.Query("SET enable_duckdb_fallback = true");
    con.Query("SET gpu_execution = true");
  }
};

void query_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO("query failed: " << sql << "\n" << (result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
}

std::string scalar(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO("query failed: " << sql << "\n" << (result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == 1);
  return result->Collection().GetValue(0, 0).ToString();
}

void make_tables(duckdb::Connection& con)
{
  query_ok(con, "RESET disabled_optimizers");
  query_ok(con, "SET sirius_test_inject_read_view_mismatch = 'off'");
  query_ok(con, "SET sirius_test_inject_pin_registry_change = false");
  query_ok(con, "SET sirius_test_read_view_churn_path = ''");
  query_ok(con, "CREATE OR REPLACE TABLE read_view_a AS SELECT range AS i FROM range(10)");
  query_ok(con, "CREATE OR REPLACE TABLE read_view_b AS SELECT range AS i FROM range(10)");
  query_ok(con, "CHECKPOINT");
  query_ok(con, "SET gpu_execution = true");
  query_ok(con, "SET enable_duckdb_fallback = true");
}

using sirius::test::read_view_map_observer;
}  // namespace

TEST_CASE_METHOD(ReadViewFixture,
                 "finalize read-view mismatch declines before installing the GPU plan",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  make_tables(connection);
  query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'finalize'");
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");

  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
  CHECK(after.successful_rebinds == before.successful_rebinds);
  CHECK(after.fallbacks == before.fallbacks + 1);
  CHECK(after.executions == before.executions);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "copy-origin swapped bindings decline despite an equal fingerprint multiset",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  make_tables(connection);
  auto const join = "SELECT sum(a.i + b.i) FROM read_view_a a JOIN read_view_b b USING (i)";

  for (auto const fallback : {true, false}) {
    CAPTURE(fallback);
    query_ok(connection,
             std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"));

    auto before = sirius::test::get_transparent_execution_stats(connection);
    CHECK(scalar(connection, join) == "90");
    auto after = sirius::test::get_transparent_execution_stats(connection);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    CHECK(after.read_view_mismatches == before.read_view_mismatches);

    query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'swap'");
    before      = after;
    auto result = connection.Query(join);
    REQUIRE(result);
    if (fallback) {
      REQUIRE_FALSE(result->HasError());
      CHECK(result->Collection().GetValue(0, 0).ToString() == "90");
    } else {
      REQUIRE(result->HasError());
      CHECK(result->GetError().find("reason=binding_mismatch") != std::string::npos);
      CHECK(result->GetError().find("correspondence=table_index") != std::string::npos);
    }
    after = sirius::test::get_transparent_execution_stats(connection);
    sirius::test::require_transparent_execution_delta(before, after, 0, fallback ? 1 : 0, 0);
    CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
    query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'off'");
  }
}

TEST_CASE_METHOD(ReadViewFixture,
                 "execution rebuild compares read views before GPU work",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  make_tables(connection);
  query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'execute'");
  query_ok(connection, "SET sirius_test_inject_pin_registry_change = true");
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");

  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
  CHECK(after.successful_rebinds == before.successful_rebinds + 1);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "read-view mismatch is an error when CPU fallback is disabled",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  make_tables(connection);
  query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'finalize'");
  query_ok(connection, "SET enable_duckdb_fallback = false");

  auto result = connection.Query("SELECT sum(i) FROM read_view_a");
  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("read-view mismatch") != std::string::npos);
  CHECK(result->GetError().find("reason=fingerprint_mismatch") != std::string::npos);
  CHECK(result->GetError().find("correspondence=table_index") != std::string::npos);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "execution read-view mismatch errors before GPU work without fallback",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  make_tables(connection);
  query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'execute'");
  query_ok(connection, "SET sirius_test_inject_pin_registry_change = true");
  query_ok(connection, "SET enable_duckdb_fallback = false");
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  auto result = connection.Query("SELECT sum(i) FROM read_view_a");
  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("read-view mismatch") != std::string::npos);
  CHECK(result->GetError().find("reason=fingerprint_mismatch") != std::string::npos);
  CHECK(result->GetError().find("correspondence=table_index") != std::string::npos);
  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
  CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "SQL replan never infers multi-scan correspondence from traversal order",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  make_tables(connection);
  query_ok(connection, "SET disabled_optimizers = 'extension'");

  auto before = sirius::test::get_transparent_execution_stats(connection);
  CHECK(scalar(connection,
               "SELECT sum(a.i + b.i) FROM read_view_a a JOIN read_view_b b USING (i)") == "90");
  auto after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
  CHECK(after.fallbacks == before.fallbacks + 1);

  before = after;
  CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");
  after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
  CHECK(after.successful_rebinds == before.successful_rebinds + 1);
  CHECK(after.executions == before.executions + 1);
  query_ok(connection, "RESET disabled_optimizers");
}

TEST_CASE_METHOD(
  ReadViewFixture,
  "Hive partition pruning validates read views with the optimizer enabled or disabled",
  "[transparent][read_view][integration]")
{
  auto& connection = *con;
  read_view_settings_guard restore{connection};
  query_ok(connection, "RESET disabled_optimizers");
  query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'off'");
  query_ok(connection, "SET sirius_test_inject_pin_registry_change = false");
  query_ok(connection, "SET sirius_test_read_view_churn_path = ''");

  auto const hive_dir = std::filesystem::path(__FILE__).parent_path() / "data/hive_partitioned";
  REQUIRE(std::filesystem::exists(hive_dir));
  auto const scan = "read_parquet(" +
                    sirius::test::sql_literal((hive_dir / "**/*.parquet").string()) +
                    ", hive_partitioning=true)";
  auto const sql = "SELECT count(*) FROM " + scan + " WHERE year=2024";

  for (bool optimizer_enabled : {true, false}) {
    query_ok(connection,
             optimizer_enabled ? "PRAGMA enable_optimizer" : "PRAGMA disable_optimizer");
    query_ok(connection, "SET gpu_execution = false");
    REQUIRE(scalar(connection, "SELECT count(*) FROM " + scan) == "3");
    auto const expected = scalar(connection, sql);
    REQUIRE(expected == "2");
    query_ok(connection, "SET gpu_execution = true");

    for (bool fallback : {true, false}) {
      CAPTURE(optimizer_enabled, fallback);
      query_ok(connection,
               std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"));
      auto const before = sirius::test::get_transparent_execution_stats(connection);

      auto result = connection.Query(sql);
      REQUIRE(result);
      INFO((result->HasError() ? result->GetError() : ""));
      REQUIRE_FALSE(result->HasError());
      REQUIRE(result->RowCount() == 1);
      REQUIRE(result->ColumnCount() == 1);
      CHECK(result->Collection().GetValue(0, 0).ToString() == expected);
      auto const after = sirius::test::get_transparent_execution_stats(connection);
      CHECK(after.read_view_mismatches == before.read_view_mismatches);
      sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    }
  }
}

TEST_CASE_METHOD(ReadViewFixture,
                 "hooks-off SQL replan rejects an observed asymmetric join flip",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  read_view_settings_guard restore{connection};
  make_tables(connection);
  sirius::test::scratch_dir files("read_view_flip");
  query_ok(connection,
           "COPY (SELECT 0::BIGINT AS i, 10::BIGINT AS v) TO " +
             files.file_literal("small.parquet") + " (FORMAT PARQUET)");
  query_ok(connection,
           "COPY (SELECT range AS i, 100 + range AS v FROM range(1000)) TO " +
             files.file_literal("large.parquet") + " (FORMAT PARQUET)");
  query_ok(connection, "SET disabled_optimizers = 'extension'");
  auto reset_sequence = [&] {
    query_ok(connection, "CREATE OR REPLACE SEQUENCE read_view_flip_seq START 1");
  };
  auto const scan = "read_parquet(CASE WHEN nextval('read_view_flip_seq') % 4 IN (1, 0) THEN " +
                    files.file_literal("small.parquet") + " ELSE " +
                    files.file_literal("large.parquet") + " END)";
  auto const sql = "SELECT sum(a.v) FROM " + scan + " a JOIN " + scan + " b USING (i)";

  // Bind A=small/B=large, then A=large/B=small. Build/probe optimization flips the join;
  // both traversals still visit large,small, but their table indices have exchanged places.
  query_ok(connection, "SET gpu_execution = false");
  // ExtractPlan's automatic transaction is read-only. Creating the sequence in an explicit
  // write transaction allows bind-time nextval, just as an ordinary statement's binder does.
  query_ok(connection, "BEGIN");
  reset_sequence();
  auto original  = connection.ExtractPlan(sql);
  auto candidate = connection.ExtractPlan(sql);
  using binding  = std::pair<std::string, duckdb::idx_t>;
  std::vector<binding> observed;
  auto logical_scans = [&](auto&& visit, duckdb::LogicalOperator& op) -> void {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
      auto& get = op.Cast<duckdb::LogicalGet>();
      REQUIRE(get.parameters.size() == 1);
      observed.emplace_back(get.parameters[0].ToString(), get.table_index.index);
    }
    for (auto& child : op.children)
      visit(visit, *child);
  };
  logical_scans(logical_scans, *original);
  auto const original_order = observed;
  observed.clear();
  logical_scans(logical_scans, *candidate);
  REQUIRE(observed.size() == 2);
  REQUIRE(original_order.size() == 2);
  CHECK(observed[0].first == original_order[0].first);
  CHECK(observed[1].first == original_order[1].first);
  CHECK(observed[0].second == original_order[1].second);
  CHECK(observed[1].second == original_order[0].second);
  REQUIRE(original_order[0].second != original_order[1].second);
  CHECK(scalar(connection, "SELECT currval('read_view_flip_seq')") == "4");
  query_ok(connection, "COMMIT");
  reset_sequence();
  REQUIRE(scalar(connection, sql) == "10");
  REQUIRE(scalar(connection, sql) == "100");

  for (bool fallback : {true, false}) {
    reset_sequence();
    query_ok(connection, "SET gpu_execution = true");
    query_ok(connection,
             std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"));
    auto const before = sirius::test::get_transparent_execution_stats(connection);
    auto result       = connection.Query(sql);
    REQUIRE(result);
    if (fallback) {
      REQUIRE_FALSE(result->HasError());
      CHECK(result->Collection().GetValue(0, 0).ToString() == "10");
    } else {
      REQUIRE(result->HasError());
      CHECK(result->GetError().find("reason=no_correspondence") != std::string::npos);
      CHECK(result->GetError().find("correspondence=none") != std::string::npos);
    }
    auto const after = sirius::test::get_transparent_execution_stats(connection);
    CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
    sirius::test::require_transparent_execution_delta(before, after, 0, fallback ? 1 : 0, 0);
    query_ok(connection, "SET gpu_execution = false");
    CHECK(scalar(connection, "SELECT currval('read_view_flip_seq')") == "4");
  }
  query_ok(connection, "RESET disabled_optimizers");
}

TEST_CASE_METHOD(ReadViewFixture,
                 "SQL replan correspondence obeys fallback-off eligibility",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  make_tables(connection);
  query_ok(connection, "SET disabled_optimizers = 'extension'");
  query_ok(connection, "SET enable_duckdb_fallback = false");

  auto before = sirius::test::get_transparent_execution_stats(connection);
  auto result =
    connection.Query("SELECT sum(a.i + b.i) FROM read_view_a a JOIN read_view_b b USING (i)");
  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("reason=no_correspondence") != std::string::npos);
  CHECK(result->GetError().find("correspondence=none") != std::string::npos);
  auto after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
  sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0);

  before = after;
  CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");
  after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "copying a SQL-replanned template preserves replan correspondence",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  read_view_settings_guard restore{connection};
  make_tables(connection);
  query_ok(connection, "SET disabled_optimizers = 'extension'");
  query_ok(connection, "SET sirius_test_inject_pin_registry_change = true");
  for (bool fallback : {true, false}) {
    CAPTURE(fallback);
    query_ok(connection,
             std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"));
    auto const before = sirius::test::get_transparent_execution_stats(connection);
    CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");
    auto const after = sirius::test::get_transparent_execution_stats(connection);
    CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
    CHECK(after.read_view_mismatches == before.read_view_mismatches);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  }
  query_ok(connection, "RESET disabled_optimizers");
  query_ok(connection, "SET sirius_test_inject_pin_registry_change = false");
}

TEST_CASE_METHOD(ReadViewFixture,
                 "failed execution copy changes correspondence to replan",
                 "[transparent][read_view][integration][iceberg_multiscan]")
{
  auto& connection = *con;
  read_view_settings_guard restore{connection};
  make_tables(connection);
  query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'execute_copy_fails'");
  query_ok(connection, "SET sirius_test_inject_pin_registry_change = true");
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  {
    read_view_map_observer observed{connection};
    CHECK(scalar(connection,
                 "SELECT sum(a.i + b.i) FROM read_view_a a JOIN read_view_b b USING (i)") == "90");
    observed.require_equal_replan_map(2);
  }

  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);

  auto const single_before = after;
  CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");
  auto const single_after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(single_after.execution_rebuilds == single_before.execution_rebuilds + 1);
  CHECK(single_after.read_view_mismatches == single_before.read_view_mismatches);
  CHECK(single_after.runtime_fallbacks == single_before.runtime_fallbacks);
  sirius::test::require_transparent_execution_delta(single_before, single_after, 1, 0, 1);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "failed execution copy proves multi-scan correspondence with fallback disabled",
                 "[transparent][read_view][integration][iceberg_multiscan]")
{
  auto& connection = *con;
  read_view_settings_guard restore{connection};
  make_tables(connection);
  query_ok(connection, "SET sirius_test_inject_read_view_mismatch = 'execute_copy_fails'");
  query_ok(connection, "SET sirius_test_inject_pin_registry_change = true");
  query_ok(connection, "SET enable_duckdb_fallback = false");
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  {
    read_view_map_observer observed{connection};
    CHECK(scalar(connection,
                 "SELECT sum(a.i + b.i) FROM read_view_a a JOIN read_view_b b USING (i)") == "90");
    observed.require_equal_replan_map(2);
  }

  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + 1);
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);

  CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");
  auto const single_after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(single_after.execution_rebuilds == after.execution_rebuilds + 1);
  CHECK(single_after.read_view_mismatches == after.read_view_mismatches);
  sirius::test::require_transparent_execution_delta(after, single_after, 1, 0, 1);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "Multi-scan SQL replans match CPU rows at finalize and execute",
                 "[transparent][read_view][integration][iceberg_multiscan]")
{
  auto const at_finalize = GENERATE(false, true);
  auto const three       = GENERATE(false, true);
  auto& connection       = *con;
  read_view_settings_guard restore{connection};
  make_tables(connection);
  auto const sql = three ? "SELECT a.i, b.i, c.i FROM read_view_a a JOIN read_view_b b ON a.i=b.i "
                           "JOIN read_view_a c ON c.i=a.i+1 WHERE a.i<7 AND c.i>=3 ORDER BY a.i"
                         : "SELECT a.i, b.i FROM read_view_a a JOIN read_view_b b ON a.i=b.i "
                           "WHERE a.i<7 AND b.i>=2 ORDER BY a.i";
  query_ok(connection, "SET gpu_execution=false");
  auto cpu = connection.Query(sql);
  REQUIRE(cpu);
  REQUIRE_FALSE(cpu->HasError());
  REQUIRE(cpu->RowCount() == 5);
  auto const expected = collect_rows(*cpu, false);
  query_ok(connection, "SET gpu_execution=true");
  query_ok(connection, "SET enable_duckdb_fallback=false");
  if (!at_finalize) {
    query_ok(connection, "SET sirius_test_inject_read_view_mismatch='execute_copy_fails'");
    query_ok(connection, "SET sirius_test_inject_pin_registry_change=true");
  }

  read_view_map_observer observed{connection, at_finalize};
  auto const before = sirius::test::get_transparent_execution_stats(connection);
  auto gpu          = connection.Query(sql);
  REQUIRE(gpu);
  INFO((gpu->HasError() ? gpu->GetError() : ""));
  REQUIRE_FALSE(gpu->HasError());
  CHECK(gpu->GetTypes() == cpu->GetTypes());
  CHECK(collect_rows(*gpu, false) == expected);
  observed.require_equal_replan_map(three ? 3 : 2);
  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + (at_finalize ? 0 : 1));
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "Multi-scan replan correspondence is published in both pipeline dumps",
                 "[transparent][read_view][integration][iceberg_multiscan]")
{
  using namespace sirius::op::scan;
  using namespace sirius::transparent;
  auto const mismatch = GENERATE(false, true);
  auto& connection    = *con;
  read_view_settings_guard restore{connection};
  make_tables(connection);
  query_ok(connection, "SET gpu_execution=false");
  auto const sql = "SELECT a.i, b.i FROM read_view_a a JOIN read_view_b b USING (i)";
  sirius::test::with_conversion_result(connection, sql, [&](auto& result) {
    std::shared_ptr<read_view_registry> registry;
    std::size_t scans = 0;
    for (auto const& pipeline : result.scheduled_pipelines) {
      auto source = pipeline->get_source();
      if (!source || source->type != sirius::op::SiriusPhysicalOperatorType::GPU_SCAN) continue;
      auto& scan = source->template Cast<sirius_gpu_scan_operator>();
      if (registry) CHECK(registry == scan.read_views());
      registry = scan.read_views();
      ++scans;
    }
    REQUIRE(scans == 2);
    REQUIRE(registry);
    REQUIRE(registry->candidate_bindings().size() == 2);
    auto original_plan = connection.ExtractPlan(sql);
    logical_bound_read_view_capture logical{
      0, capture_bound_read_views(*original_plan, *con->context)};
    duckdb::PhysicalPlanGenerator cpu_generator(*con->context);
    auto cpu_plan = cpu_generator.Plan(std::move(original_plan));
    auto physical = capture_bound_read_views(cpu_plan->Root(), *con->context);
    REQUIRE(physical.size() == 2);
    if (mismatch) registry->inject_mismatch_for_testing(true);
    auto const comparison =
      compare_read_views(candidate_origin::replan, &logical, physical, *registry);
    REQUIRE(comparison.equal == !mismatch);
    if (mismatch) {
      CHECK(comparison.reason == "no_correspondence");
      CHECK(comparison.correspondence == "none");
    } else {
      REQUIRE(comparison.correspondence == "table_index_replan");
      share_equal_read_view_identities(&logical, physical, *registry);
      registry->publish_correspondence(
        certificate_evidence_scope::binding_correspondence, comparison.correspondence, physical);
    }
    for (auto const& entry : registry->entries()) {
      CHECK(entry.eligibility.correspondence == (mismatch ? "" : "table_index_replan"));
      CHECK(entry.eligibility.evidence_scope ==
            (mismatch ? certificate_evidence_scope::none
                      : certificate_evidence_scope::binding_correspondence));
    }
    std::set<scan_contract_id> expected_contracts;
    for (auto const& entry : registry->entries())
      expected_contracts.insert(entry.contract.contract_id);
    for (auto const& dump : {sirius::pipeline::dump_pipeline_conversion_result(result),
                             sirius::pipeline::dump_pipeline_schedule_raw(result)}) {
      std::set<scan_contract_id> dumped_contracts;
      std::istringstream lines(dump);
      std::string line;
      std::string const prefix = "contract: handle=";
      while (std::getline(lines, line)) {
        auto const position = line.find(prefix);
        if (position == std::string::npos) continue;
        auto const id = std::stoull(line.substr(position + prefix.size()));
        REQUIRE(expected_contracts.count(id) == 1);
        dumped_contracts.insert(id);
        CHECK(line.find(mismatch ? " correspondence=none "
                                 : " correspondence=table_index_replan ") != std::string::npos);
        CHECK(line.find(mismatch ? " evidence_scope=none "
                                 : " evidence_scope=binding_correspondence ") != std::string::npos);
      }
      CHECK(dumped_contracts == expected_contracts);
    }
  });
}

TEST_CASE_METHOD(ReadViewFixture,
                 "Multi-scan replan callers refuse mismatches with fallback disabled",
                 "[transparent][read_view][integration][iceberg_multiscan]")
{
  auto const at_execute = GENERATE(false, true);
  auto& connection      = *con;
  read_view_settings_guard restore{connection};
  make_tables(connection);
  query_ok(connection, "SET enable_duckdb_fallback=false");
  query_ok(connection,
           std::string("SET sirius_test_inject_read_view_mismatch='") +
             (at_execute ? "execute" : "finalize") + "'");
  if (at_execute) query_ok(connection, "SET sirius_test_inject_pin_registry_change=true");
  read_view_map_observer observed{connection, true};
  auto const before = sirius::test::get_transparent_execution_stats(connection);
  auto result       = connection.Query(
    "SELECT a.i, b.i FROM read_view_a a JOIN read_view_b b ON a.i=b.i ORDER BY a.i");
  REQUIRE(result);
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("reason=fingerprint_mismatch") != std::string::npos);
  CHECK(result->GetError().find("correspondence=table_index_replan") != std::string::npos);
  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
  CHECK(after.execution_rebuilds == before.execution_rebuilds + (at_execute ? 1 : 0));
  sirius::test::require_transparent_execution_delta(
    before, after, at_execute ? 1 : 0, 0, at_execute ? 1 : 0);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "SQL replan detects a parquet glob inventory that changed after CPU bind",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  sirius::test::scratch_dir files("read_view_churn");
  query_ok(
    connection,
    "COPY (SELECT 1::INTEGER AS i) TO " + files.file_literal("a.parquet") + " (FORMAT PARQUET)");
  query_ok(connection, "SET gpu_execution = true");
  query_ok(connection, "SET enable_duckdb_fallback = true");
  query_ok(connection, "SET disabled_optimizers = 'extension'");
  query_ok(connection, "SET sirius_test_read_view_churn_path = " + files.file_literal("b.parquet"));
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  auto const glob = sirius::test::sql_literal((files.path() / "*.parquet").string());
  CHECK(scalar(connection, "SELECT count(*) FROM read_parquet(" + glob + ")") == "1");

  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(std::filesystem::exists(files.file("b.parquet")));
  CHECK(after.read_view_mismatches == before.read_view_mismatches + 1);
  CHECK(after.fallbacks == before.fallbacks + 1);
  query_ok(connection, "SET sirius_test_read_view_churn_path = ''");
  query_ok(connection, "RESET disabled_optimizers");
}

TEST_CASE_METHOD(ReadViewFixture,
                 "prepared parquet glob rebind compares the current inventory each time",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  sirius::test::scratch_dir files("read_view_prepared");
  query_ok(
    connection,
    "COPY (SELECT 1::INTEGER AS i) TO " + files.file_literal("a.parquet") + " (FORMAT PARQUET)");
  query_ok(connection, "SET gpu_execution = true");
  query_ok(connection, "SET enable_duckdb_fallback = true");
  auto const glob = sirius::test::sql_literal((files.path() / "*.parquet").string());
  auto prepared   = connection.Prepare("SELECT count(*) FROM read_parquet(" + glob + ")");
  REQUIRE(prepared);
  REQUIRE_FALSE(prepared->HasError());
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  for (int expected = 1; expected <= 3; ++expected) {
    if (expected > 1) {
      query_ok(
        connection,
        "COPY (SELECT " + std::to_string(expected) + "::INTEGER AS i) TO " +
          files.file_literal(std::string(1, static_cast<char>('a' + expected - 1)) + ".parquet") +
          " (FORMAT PARQUET)");
    }
    auto const execution_before = sirius::test::get_transparent_execution_stats(connection);
    auto result                 = prepared->Execute();
    REQUIRE(result);
    auto const error = result->HasError() ? result->GetError() : std::string{};
    INFO(error);
    REQUIRE_FALSE(result->HasError());
    auto chunk = result->Fetch();
    REQUIRE(chunk);
    REQUIRE(chunk->size() == 1);
    CHECK(chunk->GetValue(0, 0).ToString() == std::to_string(expected));
    auto const execution_after = sirius::test::get_transparent_execution_stats(connection);
    CHECK(execution_after.successful_rebinds == execution_before.successful_rebinds + 1);
    CHECK(execution_after.fallbacks == execution_before.fallbacks);
    CHECK(execution_after.executions == execution_before.executions + 1);
    CHECK(execution_after.runtime_fallbacks == execution_before.runtime_fallbacks);
  }

  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
  CHECK(after.successful_rebinds == before.successful_rebinds + 3);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "parquet plan copy preserves bound glob metacharacter paths",
                 "[transparent][read_view][integration]")
{
  auto& connection = *con;
  sirius::test::scratch_dir files("read_view_metachar");
  query_ok(connection,
           "COPY (SELECT 11::INTEGER AS i) TO " + files.file_literal("a[1].parquet") +
             " (FORMAT PARQUET)");
  query_ok(
    connection,
    "COPY (SELECT 22::INTEGER AS i) TO " + files.file_literal("a1.parquet") + " (FORMAT PARQUET)");
  auto const scan = "SELECT sum(i) FROM read_parquet(" + files.file_literal("*.parquet") + ")";

  query_ok(connection, "SET gpu_execution = false");
  auto const expected = scalar(connection, scan);
  REQUIRE(expected == "33");
  query_ok(connection, "SET gpu_execution = true");
  auto const before = sirius::test::get_transparent_execution_stats(connection);
  CHECK(scalar(connection, scan) == expected);
  auto const after = sirius::test::get_transparent_execution_stats(connection);

  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
}

TEST_CASE_METHOD(ReadViewFixture,
                 "ordinary transparent scans do not produce read-view mismatches",
                 "[transparent][read_view][integration][guard]")
{
  auto& connection = *con;
  make_tables(connection);
  sirius::test::scratch_dir files("read_view_guard");
  query_ok(connection,
           "COPY (SELECT range::INTEGER AS i FROM range(5)) TO " +
             files.file_literal("guard.parquet") + " (FORMAT PARQUET)");
  auto const before = sirius::test::get_transparent_execution_stats(connection);

  CHECK(scalar(connection, "SELECT sum(i) FROM read_view_a") == "45");
  CHECK(scalar(connection,
               "SELECT sum(a.i + b.i) FROM read_view_a a JOIN read_view_b b USING (i)") == "90");
  CHECK(scalar(connection,
               "SELECT sum(i) FROM read_parquet(" + files.file_literal("guard.parquet") + ")") ==
        "10");

  auto const after = sirius::test::get_transparent_execution_stats(connection);
  CHECK(after.read_view_mismatches == before.read_view_mismatches);
  CHECK(after.successful_rebinds == before.successful_rebinds + 3);
  CHECK(after.executions == before.executions + 3);
}
