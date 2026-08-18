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
 * @file test_dense_count_join_detection.cpp
 * @brief Plan-time gates for the DENSE_COUNT_JOIN rewrite: the q13 shape (COUNT over an outer
 *        equi-join grouped by the preserved-side key) fuses into one operator with no HASH_JOIN
 *        and no inner HASH_GROUP_BY, while every off-shape variant (inner join, non-key group,
 *        multiple aggregates, DISTINCT, non-COUNT aggregate, disabled knob) keeps the normal
 *        join + aggregate plan.
 */

#include "op/sirius_physical_dense_count_join.hpp"
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
#include <string>
#include <vector>

using namespace duckdb;

namespace {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_dense_count_join_XXXXXX";
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
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

std::vector<sirius::op::sirius_physical_operator*> collect(
  sirius::op::sirius_physical_operator* root, sirius::op::SiriusPhysicalOperatorType type)
{
  std::vector<sirius::op::sirius_physical_operator*> out;
  if (!root) { return out; }
  if (root->type == type) { out.push_back(root); }
  for (auto& child : root->children) {
    auto sub = collect(child.get(), type);
    out.insert(out.end(), sub.begin(), sub.end());
  }
  return out;
}

struct dense_count_join_fixture {
  dense_count_join_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    // q13 shape: customers (preserved) vs orders (counted); orders is larger so the join
    // orientation matches TPC-H (either LEFT or RIGHT is accepted by the gate).
    con->Query("CREATE TABLE cust (c_id INTEGER, c_grp INTEGER)");
    con->Query("INSERT INTO cust SELECT range, range % 3 FROM range(20)");
    // A NULL c_grp keeps DuckDB's optimizer from rewriting count(c_grp) into count_star()
    // (which WOULD be a legitimate fusion target); the preserved-side-count decline test
    // below needs the count(col) shape to survive optimization.
    con->Query("INSERT INTO cust VALUES (100, NULL)");
    con->Query("CREATE TABLE ord (o_id BIGINT, o_cust INTEGER, o_note VARCHAR)");
    con->Query(
      "INSERT INTO ord SELECT range, (range * 7) % 30, concat('n', range) FROM range(200)");
  }

  ~dense_count_join_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  bool has_dense_count_join(const std::string& query)
  {
    auto plan = generate_sirius_plan(*con, query);
    REQUIRE(plan);
    using T          = sirius::op::SiriusPhysicalOperatorType;
    auto const fused = collect(plan.get(), T::DENSE_COUNT_JOIN);
    if (fused.empty()) { return false; }
    // Whenever the rewrite fires the join and the inner group-by must be gone.
    REQUIRE(fused.size() == 1);
    REQUIRE(collect(plan.get(), T::HASH_JOIN).empty());
    REQUIRE(collect(plan.get(), T::NESTED_LOOP_JOIN).empty());
    REQUIRE(fused[0]->children.size() == 2);
    return true;
  }

  scoped_temp_db_path db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join fires on COUNT(col) grouped by the preserved LEFT-join key",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  REQUIRE(has_dense_count_join(query));
  // The fused plan has no HASH_GROUP_BY at all: the only aggregate was the rewritten one.
  auto plan = generate_sirius_plan(*con, query);
  REQUIRE(collect(plan.get(), sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY).empty());
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join fires on COUNT(*) and on the RIGHT-join orientation",
                 "[dense_count_join][plan]")
{
  REQUIRE(has_dense_count_join(
    "SELECT c_id, count(*) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
  REQUIRE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM ord RIGHT JOIN cust ON o_cust = c_id GROUP BY c_id"));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join fires inside the two-level q13 distribution shape",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_count, count(*) AS custdist FROM ("
    "  SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"
    ") t GROUP BY c_count";
  REQUIRE(has_dense_count_join(query));
  // Exactly the outer distribution group-by survives.
  auto plan = generate_sirius_plan(*con, query);
  REQUIRE(collect(plan.get(), sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY).size() == 1);
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines off-shape aggregates and joins",
                 "[dense_count_join][plan]")
{
  // INNER join: no preserved side.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM cust JOIN ord ON c_id = o_cust GROUP BY c_id"));
  // Grouped by a non-key column.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_grp, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_grp"));
  // Two aggregates.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id), max(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY "
    "c_id"));
  // DISTINCT count.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(DISTINCT o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
  // Non-COUNT aggregate.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, sum(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
  // COUNT of a nullable preserved-side column (different NULL semantics; a NON-null
  // preserved-side count is rewritten to count_star() by DuckDB itself and legitimately fuses).
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(c_grp) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join respects the enable knob",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  REQUIRE(has_dense_count_join(query));
  auto off = con->Query("SET enable_dense_count_join = false");
  REQUIRE_FALSE(off->HasError());
  CHECK_FALSE(has_dense_count_join(query));
  auto on = con->Query("SET enable_dense_count_join = true");
  REQUIRE_FALSE(on->HasError());
  REQUIRE(has_dense_count_join(query));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines filtered aggregates, extra conditions, and "
                 "non-plain keys",
                 "[dense_count_join][plan]")
{
  // Aggregate FILTER clause: the filtered count is not a plain per-group match count.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FILTER (WHERE o_id > 0) FROM cust LEFT JOIN ord ON c_id = o_cust "
    "GROUP BY c_id"));
  // More than one join condition.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust AND c_grp = o_cust "
    "GROUP BY c_id"));
  // Mixed key types (INTEGER vs BIGINT): the binder inserts a CAST, which the BOUND_REF gate
  // refuses.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_cust) FROM cust LEFT JOIN ord ON c_id = o_id GROUP BY c_id"));
  // IS NOT DISTINCT FROM (NULL-matches-NULL semantics must never take the fused path).
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id IS NOT DISTINCT FROM o_cust "
    "GROUP BY c_id"));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines intervening operators and non-linear join children",
                 "[dense_count_join][plan]")
{
  // A residual WHERE referencing both sides stays a FILTER between the aggregate and the
  // join (not pushable into either child, not NULL-rejecting for o_id).
  CHECK_FALSE(
    has_dense_count_join("SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust "
                         "WHERE (o_id IS NULL OR c_grp = 0) GROUP BY c_id"));
  // Grouping by the COUNTED side's key is not the preserved-group shape.
  CHECK_FALSE(has_dense_count_join(
    "SELECT o_cust, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY o_cust"));
  // A join-rooted counted child is not a linear GET/FILTER/PROJECTION chain (and an identity
  // projection above it would be elided by push_projection — the chain gate must refuse).
  CHECK_FALSE(
    has_dense_count_join("SELECT c_id, count(o.o_id) FROM cust LEFT JOIN ("
                         "  SELECT o1.o_id, o1.o_cust FROM ord o1 JOIN ord o2 ON o1.o_id = o2.o_id"
                         ") o ON c_id = o.o_cust GROUP BY c_id"));
}
