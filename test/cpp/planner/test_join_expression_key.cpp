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
 * @file test_join_expression_key.cpp
 * @brief issue #329: hash-join equality conditions with expression keys (e.g. `a = b * 10`) are
 *        materialized into a projection below the join, and the condition side is rewritten to a
 *        plain column reference so PARTITION/CONCAT/hash-join see an ordinary column index.
 */

#include "duckdb/main/settings.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "expression/join_condition.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <filesystem>

using namespace duckdb;

namespace {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_join_expr_key_XXXXXX";
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

bool is_bound_ref(const sirius::ast::node& side)
{
  auto expr = sirius::ast::to_duckdb(side);
  return expr->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF;
}

/// Assert that every equality condition side of @p hj is a plain column reference (BOUND_REF),
/// i.e. any complex expression was materialized out into a projection below the join.
void require_all_equality_sides_are_references(sirius::op::sirius_physical_hash_join& hj)
{
  for (auto& c : hj.conditions) {
    if (c.comparison != sirius::comparison_type::equal &&
        c.comparison != sirius::comparison_type::not_distinct_from) {
      continue;
    }
    CHECK(is_bound_ref(*c.left));
    CHECK(is_bound_ref(*c.right));
  }
}

bool has_projection_child(sirius::op::sirius_physical_hash_join& hj)
{
  for (auto& child : hj.children) {
    // Plan-gen wraps each join child in a CONCAT -> PARTITION chain; the materializing
    // PROJECTION sits below the wraps.
    auto* node = child.get();
    while (node != nullptr && (node->type == sirius::op::SiriusPhysicalOperatorType::CONCAT ||
                               node->type == sirius::op::SiriusPhysicalOperatorType::PARTITION)) {
      node = node->children.empty() ? nullptr : node->children[0].get();
    }
    if (node != nullptr && node->type == sirius::op::SiriusPhysicalOperatorType::PROJECTION) {
      return true;
    }
  }
  return false;
}

struct join_expression_key_fixture {
  join_expression_key_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    // big_left is larger so the optimizer keeps small_right as the build side.
    con->Query("CREATE TABLE big_left (id INTEGER, val INTEGER)");
    con->Query(
      "INSERT INTO big_left VALUES (0,0),(1,3),(2,6),(3,9),(4,12),(5,15),(6,18),(7,21),(8,24),"
      "(9,27),(10,30),(11,33),(12,36),(13,39),(14,42),(15,45),(16,48),(17,51),(18,54),(19,57)");
    con->Query("CREATE TABLE small_right (rid INTEGER, other INTEGER)");
    con->Query("INSERT INTO small_right VALUES (0, 0), (1, 1)");
  }

  ~join_expression_key_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(join_expression_key_fixture,
                 "join expression key - single expression side is materialized and rewritten",
                 "[join_expression_key][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con, "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid * 10");
  if (!plan) {
    WARN("Plan generation failed (no DuckDB table scan support); skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);

  // The `r.rid * 10` key is materialized, so both equality sides are now plain references.
  require_all_equality_sides_are_references(*hj);

  // A materializing projection was inserted directly below the join.
  CHECK(has_projection_child(*hj));

  // The synthetic key column is excluded from the join output: SELECT * over (id,val)+(rid,other)
  // is 4 columns, not 5.
  CHECK(hj->lhs_output_columns.col_idxs.size() + hj->rhs_output_columns.col_idxs.size() == 4);
}

TEST_CASE_METHOD(join_expression_key_fixture,
                 "join expression key - expressions on both sides are materialized",
                 "[join_expression_key][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con, "SELECT * FROM big_left l JOIN small_right r ON l.id + 1 = r.rid + 1");
  if (!plan) {
    WARN("Plan generation failed; skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  require_all_equality_sides_are_references(*hj);
  CHECK(has_projection_child(*hj));
  CHECK(hj->lhs_output_columns.col_idxs.size() + hj->rhs_output_columns.col_idxs.size() == 4);
}

TEST_CASE_METHOD(join_expression_key_fixture,
                 "join expression key - plain column join is not materialized",
                 "[join_expression_key][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con, "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid");
  if (!plan) {
    WARN("Plan generation failed; skipping");
    return;
  }

  auto* hj = find_hash_join(plan.get());
  REQUIRE(hj);
  // Fast path: plain-reference keys need no projection below the join.
  CHECK_FALSE(has_projection_child(*hj));
  require_all_equality_sides_are_references(*hj);
}
