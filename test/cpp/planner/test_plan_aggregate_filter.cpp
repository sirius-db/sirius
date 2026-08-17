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
 * @file test_plan_aggregate_filter.cpp
 * @brief Plan-time handling of the aggregate FILTER clause: supported filtered aggregates
 *        convert with their inputs mask-wrapped as CASE projections below the aggregate (and
 *        `count(*) FILTER` lowered to `count(mask)`), while DISTINCT and `first` with FILTER
 *        throw duckdb::NotImplementedException so the query falls back to CPU.
 */

#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/case_expr.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cudf/aggregation.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_plan_aggregate_filter_XXXXXX";
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

/// Generate a Sirius physical plan from a SQL query string. Throws on any failure (after
/// rolling back and restoring the optimizer settings) so a planner regression fails the
/// test instead of silently skipping it -- and so REQUIRE_THROWS_AS can assert the
/// NotImplementedException the FILTER gate raises for rejected shapes.
duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(Connection& con,
                                                                  const std::string& query)
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius_physical_operator> result;
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

sirius_physical_operator* find_first(sirius_physical_operator* root,
                                     SiriusPhysicalOperatorType type)
{
  if (!root) { return nullptr; }
  if (root->type == type) { return root; }
  for (auto& child : root->children) {
    if (auto* found = find_first(child.get(), type)) { return found; }
  }
  return nullptr;
}

std::size_t count_case_entries(sirius::op::sirius_physical_projection const& projection)
{
  std::size_t count = 0;
  for (auto const& entry : projection.select_list) {
    if (entry && entry->holds<sirius::ast::case_expr>()) { count++; }
  }
  return count;
}

struct plan_aggregate_filter_fixture {
  plan_aggregate_filter_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    con->Query("CREATE TABLE events (id INTEGER, val INTEGER)");
    con->Query("INSERT INTO events SELECT range, range * 3 % 7 FROM range(100)");
  }

  ~plan_aggregate_filter_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - grouped filtered aggregates convert to masked CASE inputs",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(
    *con,
    "SELECT val % 2 AS g, sum(val) FILTER (WHERE val < 5), count(*) FILTER (WHERE val < 5) "
    "FROM events GROUP BY g");

  auto* group_by = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
  REQUIRE(group_by != nullptr);

  // count(*) FILTER lowers to count(mask) -> COUNT_VALID; no COUNT_ALL may survive, since
  // COUNT_ALL would count masked-out rows.
  auto& aggregate = group_by->Cast<sirius::op::sirius_physical_grouped_aggregate>();
  std::vector<cudf::aggregation::Kind> const expected{cudf::aggregation::Kind::SUM,
                                                      cudf::aggregation::Kind::COUNT_VALID};
  CHECK(aggregate.cudf_aggregates == expected);

  // The projection below the aggregate carries the mask rewrites: the sum input wrapped as
  // CASE WHEN filter THEN val ELSE NULL END and the count(*) boolean mask column.
  auto* projection = find_first(group_by, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection != nullptr);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 2);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - ungrouped count(*) FILTER converts to count(mask)",
                 "[aggregate_filter][isolated_context]")
{
  auto plan = generate_sirius_plan(*con, "SELECT count(*) FILTER (WHERE val < 5) FROM events");

  auto* ungrouped = find_first(plan.get(), SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  REQUIRE(ungrouped != nullptr);

  auto& aggregate = ungrouped->Cast<sirius::op::sirius_physical_ungrouped_aggregate>();
  REQUIRE(aggregate.aggregates.size() == 1);
  REQUIRE(aggregate.aggregates[0]->holds<sirius::ast::aggregate>());
  auto const& agg = aggregate.aggregates[0]->get<sirius::ast::aggregate>();
  CHECK(agg.function() == sirius::aggregate_id::count);
  CHECK_FALSE(agg.distinct());
  REQUIRE(agg.arguments().size() == 1);
  CHECK(agg.arguments()[0]->holds<sirius::ast::reference>());

  // The boolean mask column (CASE WHEN filter THEN true ELSE NULL END) lives in the
  // projection below the aggregate.
  auto* projection = find_first(ungrouped, SiriusPhysicalOperatorType::PROJECTION);
  REQUIRE(projection != nullptr);
  CHECK(count_case_entries(projection->Cast<sirius::op::sirius_physical_projection>()) == 1);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - DISTINCT with FILTER is rejected for CPU fallback",
                 "[aggregate_filter][isolated_context]")
{
  REQUIRE_THROWS_AS(
    generate_sirius_plan(
      *con, "SELECT val % 2 AS g, count(DISTINCT id) FILTER (WHERE id < 5) FROM events GROUP BY g"),
    duckdb::NotImplementedException);
}

TEST_CASE_METHOD(plan_aggregate_filter_fixture,
                 "aggregate filter - first with FILTER is rejected for CPU fallback",
                 "[aggregate_filter][isolated_context]")
{
  REQUIRE_THROWS_AS(
    generate_sirius_plan(*con, "SELECT first(val) FILTER (WHERE val = 3) FROM events"),
    duckdb::NotImplementedException);
}
