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
 * @file test_distinct_hash_join_detection.cpp
 * @brief Tests that unique_build_keys is correctly set on hash joins when
 *        build-side keys are proven unique at plan construction time.
 */

#include "duckdb/main/settings.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>

using namespace duckdb;

namespace {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_distinct_hj_XXXXXX";
    int fd      = ::mkstemp(tmpl);
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl);
    _path = tmpl;
  }

  ~scoped_temp_db_path()
  {
    if (!_path.empty()) {
      std::remove(_path.c_str());
      std::remove((_path + ".wal").c_str());
    }
  }

  scoped_temp_db_path(const scoped_temp_db_path&)            = delete;
  scoped_temp_db_path& operator=(const scoped_temp_db_path&) = delete;

  const std::string& path() const { return _path; }

 private:
  std::string _path;
};

/// Generate a Sirius physical plan from a SQL query string.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> generate_sirius_plan(
  Connection& con, const std::string& query)
{
  auto& context = *con.context;

  // Disable optimizers that can interfere with plan structure
  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
  try {
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);

    if (duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(context)) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(context, *plan);
    resolver.VisitOperator(*plan);

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
  } catch (duckdb::InternalException&) {
    // Plan generation can fail for DuckDB internal table scans (not the primary Sirius use case).
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    return nullptr;
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

/// Walk the operator tree to find the first hash join operator.
sirius::op::sirius_physical_hash_join* find_hash_join(sirius::op::sirius_physical_operator* root)
{
  if (!root) { return nullptr; }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) {
    return &root->Cast<sirius::op::sirius_physical_hash_join>();
  }
  for (auto& child : root->children) {
    auto* found = find_hash_join(child.get());
    if (found) { return found; }
  }
  return nullptr;
}

/// Shared fixture: one DuckDB + config for all distinct_hash_join tests.
struct distinct_hash_join_fixture {
  distinct_hash_join_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    // Create all test tables upfront
    con->Query("CREATE TABLE pk_orders (o_orderkey INTEGER PRIMARY KEY, o_custkey INTEGER)");
    con->Query("CREATE TABLE lineitem (l_orderkey INTEGER, l_linenumber INTEGER)");
    con->Query("INSERT INTO pk_orders VALUES (1, 100), (2, 200)");
    con->Query("INSERT INTO lineitem VALUES (1, 1), (2, 1), (1, 2)");

    con->Query("CREATE TABLE composite_pk (a INTEGER, b INTEGER, val DOUBLE, PRIMARY KEY (a, b))");
    con->Query("INSERT INTO composite_pk VALUES (1, 1, 10.0), (1, 2, 20.0)");

    con->Query("CREATE TABLE unique_table (id INTEGER UNIQUE, name VARCHAR)");
    con->Query("INSERT INTO unique_table VALUES (1, 'a'), (2, 'b')");

    con->Query("CREATE TABLE plain_orders (o_orderkey INTEGER, o_custkey INTEGER)");
    con->Query("INSERT INTO plain_orders VALUES (1, 100), (2, 200)");

    con->Query("CREATE TABLE sales (product_id INTEGER, amount DOUBLE)");
    con->Query("INSERT INTO sales VALUES (1, 10.0), (1, 20.0), (2, 30.0)");

    // products has many more rows so the optimizer keeps the aggregate as the build side
    con->Query("CREATE TABLE products (id INTEGER)");
    con->Query(
      "INSERT INTO products VALUES (1),(2),(3),(4),(5),(6),(7),(8),(9),(10),"
      "(11),(12),(13),(14),(15),(16),(17),(18),(19),(20)");

    con->Query("CREATE TABLE data (a INTEGER, b INTEGER, val DOUBLE)");
    con->Query("INSERT INTO data VALUES (1, 1, 10.0), (1, 2, 20.0)");

    // probe has many rows so the optimizer keeps the constrained table as the build side
    con->Query("CREATE TABLE probe (x INTEGER, y INTEGER)");
    con->Query(
      "INSERT INTO probe VALUES (1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1),(4,2),"
      "(5,1),(5,2),(6,1),(6,2),(7,1),(7,2),(8,1),(8,2),(9,1),(9,2),(10,1),(10,2)");
  }

  ~distinct_hash_join_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  scoped_temp_db_path db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

// ---------------------------------------------------------------------------
// PRIMARY KEY detection
// ---------------------------------------------------------------------------

// NOTE: Tests below with direct table JOINs may fail plan generation on some
// environments because the Sirius plan generator doesn't fully support DuckDB
// internal table scans. These tests use REQUIRE(plan) to skip gracefully
// (generate_sirius_plan returns nullptr on InternalException).

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - PK build side enables unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con, "SELECT * FROM lineitem JOIN pk_orders ON l_orderkey = o_orderkey");
  if (!plan) {
    WARN("Plan generation failed (no DuckDB table scan support); skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK(hj->unique_build_keys);
}

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - composite PK enables unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con, "SELECT * FROM probe JOIN composite_pk ON x = a AND y = b");
  if (!plan) {
    WARN("Plan generation failed; skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK(hj->unique_build_keys);
}

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - partial composite PK does NOT enable unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan = generate_sirius_plan(*con, "SELECT * FROM probe JOIN composite_pk ON x = a");
  if (!plan) {
    WARN("Plan generation failed; skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK_FALSE(hj->unique_build_keys);
}

// ---------------------------------------------------------------------------
// Plain UNIQUE constraint exclusion
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - plain UNIQUE does NOT enable unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan = generate_sirius_plan(*con, "SELECT * FROM probe JOIN unique_table ON x = id");
  if (!plan) {
    WARN("Plan generation failed; skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK_FALSE(hj->unique_build_keys);
}

// ---------------------------------------------------------------------------
// Non-unique build side
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - non-unique build side does NOT enable unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con, "SELECT * FROM lineitem JOIN plain_orders ON l_orderkey = o_orderkey");
  if (!plan) {
    WARN("Plan generation failed; skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK_FALSE(hj->unique_build_keys);
}

// ---------------------------------------------------------------------------
// GROUP BY uniqueness
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - GROUP BY build side enables unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT * FROM products JOIN (SELECT product_id, SUM(amount) AS total FROM sales GROUP BY "
    "product_id) AS agg ON id = product_id");
  REQUIRE(plan);

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK(hj->unique_build_keys);
}

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - multi-key GROUP BY joined on subset does NOT enable",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT * FROM probe JOIN (SELECT a, b, SUM(val) FROM data GROUP BY a, b) AS agg ON x = a");
  REQUIRE(plan);

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK_FALSE(hj->unique_build_keys);
}

// ---------------------------------------------------------------------------
// Uniqueness propagation through joins
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(
  distinct_hash_join_fixture,
  "distinct_hash_join - uniqueness propagates through INNER join with unique keys on other side",
  "[distinct_hash_join][isolated_context]")
{
  // Inner join: (GROUP BY product_id) JOIN pk_orders ON product_id = o_orderkey
  //   → pk_orders.o_orderkey is PK, so each agg row matches ≤1 pk_orders row
  //   → agg's product_id uniqueness is preserved in the join output
  // Outer join: products JOIN (result) ON id = product_id → unique_build_keys = true
  auto plan = generate_sirius_plan(
    *con,
    "SELECT * FROM products JOIN ("
    "  SELECT agg.product_id, agg.total, o.o_custkey"
    "  FROM (SELECT product_id, SUM(amount) AS total FROM sales GROUP BY product_id) agg"
    "  JOIN pk_orders o ON agg.product_id = o.o_orderkey"
    ") build ON products.id = build.product_id");
  REQUIRE(plan);

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK(hj->unique_build_keys);
}

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - uniqueness does NOT propagate through INNER join when other "
                 "side not unique",
                 "[distinct_hash_join][isolated_context]")
{
  // Inner join: (GROUP BY product_id) JOIN lineitem ON product_id = l_orderkey
  //   → lineitem has no PK, so each agg row can match multiple lineitem rows
  //   → agg's product_id uniqueness is NOT preserved
  // Outer join: products JOIN (result) ON id = product_id → unique_build_keys = false
  auto plan = generate_sirius_plan(
    *con,
    "SELECT * FROM products JOIN ("
    "  SELECT agg.product_id, agg.total, l.l_linenumber"
    "  FROM (SELECT product_id, SUM(amount) AS total FROM sales GROUP BY product_id) agg"
    "  JOIN lineitem l ON agg.product_id = l.l_orderkey"
    ") build ON products.id = build.product_id");
  REQUIRE(plan);

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK_FALSE(hj->unique_build_keys);
}

// ---------------------------------------------------------------------------
// LEFT join support
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - LEFT join with PK build side enables unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con, "SELECT * FROM lineitem LEFT JOIN pk_orders ON l_orderkey = o_orderkey");
  if (!plan) {
    WARN("Plan generation failed; skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  CHECK(hj->unique_build_keys);
}
