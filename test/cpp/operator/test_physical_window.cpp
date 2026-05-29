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

#include "op/sirius_physical_window.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace duckdb;

namespace {

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path(SIRIUS_PROJECT_ROOT);
#else
  return std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path();
#endif
}

void require_ok(Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> generate_sirius_plan(
  Connection& con, const std::string& query)
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  require_ok(con, "BEGIN TRANSACTION");

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
    auto sirius_plan = gen.create_plan(std::move(plan));

    require_ok(con, "COMMIT");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    return sirius_plan;
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }
}

sirius::op::sirius_physical_window* find_window_operator(sirius::op::sirius_physical_operator* root)
{
  if (!root) { return nullptr; }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::WINDOW) {
    return &root->Cast<sirius::op::sirius_physical_window>();
  }
  for (auto& child : root->children) {
    if (auto* found = find_window_operator(child.get())) { return found; }
  }
  return nullptr;
}

class WindowPlanFixture {
 public:
  WindowPlanFixture()
  {
    auto cfg = project_root() / "test" / "cpp" / "config" / "data" / "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(nullptr);
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    require_ok(*con,
               "CREATE TABLE window_rank_input ("
               "  grp INTEGER,"
               "  subgroup INTEGER,"
               "  metric INTEGER,"
               "  id INTEGER"
               ")");
    require_ok(*con,
               "INSERT INTO window_rank_input VALUES "
               "(1, 10, 100, 1),"
               "(1, 10, 100, 2),"
               "(1, 10,  80, 3),"
               "(1, 20,  70, 4),"
               "(2, 10,  50, 5),"
               "(2, 10,  50, 6),"
               "(2, 20, NULL, 7),"
               "(NULL, 10, 40, 8),"
               "(NULL, 10, 30, 9)");

    require_ok(*con,
               "CREATE TABLE window_hugeint_input ("
               "  grp INTEGER,"
               "  id INTEGER,"
               "  huge_value HUGEINT"
               ")");
    require_ok(*con,
               "INSERT INTO window_hugeint_input VALUES "
               "(1, 1, 170141183460469231731687303715884105727::HUGEINT),"
               "(1, 2, 2::HUGEINT)");
  }

  ~WindowPlanFixture()
  {
    unsetenv("SIRIUS_CONFIG_FILE");
    unsetenv("SIRIUS_DISABLE");
  }

  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(WindowPlanFixture,
                 "sirius_physical_window accepts shared Phase 1 ranking expressions",
                 "[window][plan]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT "
    "  grp,"
    "  id,"
    "  row_number() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST, id ASC) AS rn,"
    "  rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST, id ASC) AS rnk,"
    "  dense_rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST, id ASC) AS dr "
    "FROM window_rank_input");

  auto* window = find_window_operator(plan.get());
  REQUIRE(window);
  CHECK(sirius::op::sirius_physical_window::TYPE == sirius::op::SiriusPhysicalOperatorType::WINDOW);
  CHECK(window->type == sirius::op::SiriusPhysicalOperatorType::WINDOW);
  CHECK(window->is_sink());
  CHECK(window->is_source());
  CHECK(window->source_order() == sirius::OrderPreservationType::NO_ORDER);

  std::vector<int> expected_partition_keys{0};
  CHECK(window->get_partition_key_indices() == expected_partition_keys);
}

TEST_CASE_METHOD(WindowPlanFixture,
                 "sirius_physical_window accepts global row_number without order",
                 "[window][plan]")
{
  auto plan = generate_sirius_plan(*con,
                                   "SELECT id, row_number() OVER () AS rn "
                                   "FROM window_rank_input");

  auto* window = find_window_operator(plan.get());
  REQUIRE(window);
  CHECK(window->get_partition_key_indices().empty());
  CHECK(window->is_sink());
  CHECK(window->is_source());
  CHECK(window->source_order() == sirius::OrderPreservationType::NO_ORDER);
}

TEST_CASE_METHOD(WindowPlanFixture,
                 "sirius_physical_window rejects unsupported window shapes with NotImplemented",
                 "[window][plan][negative]")
{
  struct NegativeCase {
    std::string name;
    std::string sql;
  };

  const std::vector<NegativeCase> cases{
    {"rank_without_order", "SELECT rank() OVER (PARTITION BY grp) AS rnk FROM window_rank_input"},
    {"dense_rank_without_order", "SELECT dense_rank() OVER () AS dr FROM window_rank_input"},
    {"heterogeneous_partition",
     "SELECT "
     "  row_number() OVER (PARTITION BY grp ORDER BY metric) AS rn1,"
     "  row_number() OVER (PARTITION BY subgroup ORDER BY metric) AS rn2 "
     "FROM window_rank_input"},
    {"aggregate_window",
     "SELECT sum(metric) OVER (PARTITION BY grp ORDER BY id) AS running_sum "
     "FROM window_rank_input"},
    {"exclude_clause",
     "SELECT row_number() OVER ("
     "  PARTITION BY grp ORDER BY metric "
     "  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW "
     "  EXCLUDE CURRENT ROW"
     ") AS rn FROM window_rank_input"},
    {"non_bound_partition_key",
     "SELECT row_number() OVER (PARTITION BY grp + 1 ORDER BY metric) AS rn "
     "FROM window_rank_input"},
    {"hugeint_child_column",
     "SELECT huge_value, row_number() OVER (PARTITION BY grp ORDER BY id) AS rn "
     "FROM window_hugeint_input"}};

  for (const auto& negative_case : cases) {
    DYNAMIC_SECTION(negative_case.name)
    {
      REQUIRE_THROWS_AS(generate_sirius_plan(*con, negative_case.sql),
                        duckdb::NotImplementedException);
    }
  }
}
