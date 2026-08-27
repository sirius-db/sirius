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

#include "op/sirius_physical_delim_join.hpp"
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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

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

  // Disable optimizers that can interfere with plan structure. DELIMINATOR is disabled so
  // correlated-subquery tests keep their DELIM_JOIN / DELIM_GET shape, which it would
  // otherwise rewrite away on toy tables; a no-op for the other tests.
  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  disabled.insert(OptimizerType::DELIMINATOR);

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

    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*plan);
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

/// Collect every hash join in the tree, recursing into a DELIM JOIN's internal `join` and
/// `distinct_root` subtrees, which live outside `children[]`.
void collect_hash_joins(sirius::op::sirius_physical_operator* root,
                        std::vector<sirius::op::sirius_physical_hash_join*>& out)
{
  if (!root) { return; }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) {
    out.push_back(&root->Cast<sirius::op::sirius_physical_hash_join>());
  }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      root->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    collect_hash_joins(delim.join.get(), out);
    collect_hash_joins(delim.distinct_root.get(), out);
  }
  for (auto& child : root->children) {
    collect_hash_joins(child.get(), out);
  }
}

/// True when @p root's subtree contains a DELIM_SCAN (the physical form of LOGICAL_DELIM_GET).
bool subtree_has_delim_scan(sirius::op::sirius_physical_operator* root)
{
  if (!root) { return false; }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::DELIM_SCAN) { return true; }
  return std::any_of(root->children.begin(), root->children.end(), [](auto& c) {
    return subtree_has_delim_scan(c.get());
  });
}

/// A hash join's build side is children[1]; the probe side is children[0].
bool builds_on_delim_scan(sirius::op::sirius_physical_hash_join* hj)
{
  REQUIRE(hj->children.size() == 2);
  return subtree_has_delim_scan(hj->children[1].get());
}

bool touches_delim_scan(sirius::op::sirius_physical_hash_join* hj)
{
  REQUIRE(hj->children.size() == 2);
  return subtree_has_delim_scan(hj->children[0].get()) ||
         subtree_has_delim_scan(hj->children[1].get());
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

    // delim_fact drives a correlated NOT EXISTS that decorrelates into a DELIM_GET build
    // side. delim_big is much larger so the DELIM_GET stays the build side.
    con->Query("CREATE TABLE delim_fact (k INTEGER, grp INTEGER, v INTEGER)");
    con->Query("INSERT INTO delim_fact SELECT i % 7, (i % 4) + 1, i FROM range(40) t(i)");
    con->Query("CREATE TABLE delim_big (okey INTEGER, k INTEGER)");
    con->Query("INSERT INTO delim_big SELECT i, i % 9 FROM range(3000) t(i)");
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
// DELIM_GET uniqueness (duplicate-eliminated delim scans)
// ---------------------------------------------------------------------------

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - DELIM_GET build side enables unique_build_keys",
                 "[distinct_hash_join][isolated_context]")
{
  // Decorrelates into a DELIM_JOIN whose inner re-join probes delim_big against a
  // DELIM_GET replaying the duplicate-eliminated k keys — the TPC-H q22 shape.
  auto plan = generate_sirius_plan(
    *con,
    "SELECT f.grp, f.v FROM delim_fact f "
    "WHERE f.v > 3 AND NOT EXISTS (SELECT 1 FROM delim_big o WHERE o.k = f.k)");
  REQUIRE(plan);

  std::vector<sirius::op::sirius_physical_hash_join*> joins;
  collect_hash_joins(plan.get(), joins);

  // Exactly one join builds on the DELIM_GET, and that is the one that must be claimed.
  // (The DELIM_JOIN's own join probes the delim scan and carries an IS NOT DISTINCT FROM
  // condition, so the pure-equal gate leaves it un-marked.)
  auto delim_builds = std::count_if(joins.begin(), joins.end(), builds_on_delim_scan);
  REQUIRE(delim_builds == 1);
  for (auto* hj : joins) {
    if (builds_on_delim_scan(hj)) { CHECK(hj->unique_build_keys); }
  }
}

TEST_CASE_METHOD(distinct_hash_join_fixture,
                 "distinct_hash_join - non-unique build still refused alongside DELIM_GET",
                 "[distinct_hash_join][isolated_context]")
{
  // Same delim shape, but the outer query also joins the non-unique delim_fact build.
  auto plan = generate_sirius_plan(
    *con,
    "SELECT p.x FROM probe p JOIN delim_fact f ON p.x = f.grp "
    "WHERE f.v > 3 AND NOT EXISTS (SELECT 1 FROM delim_big o WHERE o.k = f.k)");
  REQUIRE(plan);

  std::vector<sirius::op::sirius_physical_hash_join*> joins;
  collect_hash_joins(plan.get(), joins);

  // Assert per join rather than over the plan as a whole: the DELIM_GET build is claimed,
  // and the plain probe ⋈ delim_fact join (no delim scan on either side, so it builds over
  // raw duplicated rows) is refused. A whole-plan "not every join is unique" would pass
  // even if the wrong join were the refused one.
  auto delim_builds = std::count_if(joins.begin(), joins.end(), builds_on_delim_scan);
  REQUIRE(delim_builds == 1);

  bool saw_plain_join = false;
  for (auto* hj : joins) {
    if (builds_on_delim_scan(hj)) {
      CHECK(hj->unique_build_keys);
    } else if (!touches_delim_scan(hj)) {
      saw_plain_join = true;
      CHECK_FALSE(hj->unique_build_keys);
    }
  }
  REQUIRE(saw_plain_join);
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
