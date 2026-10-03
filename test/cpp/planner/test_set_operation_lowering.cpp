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
 * @file test_set_operation_lowering.cpp
 * @brief Tests `plan_except_intersect` and the set-operation fork in `create_plan`.
 *
 * The generator's dispatch switch still refuses `LOGICAL_EXCEPT` and `LOGICAL_INTERSECT`, so these
 * tests find the set operation in an optimized logical plan and call
 * `create_plan(LogicalSetOperation&)` on it through `set_operation_planner`.
 */

#include "expression/ast/node.hpp"
#include "expression/join_condition.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/scoped_temp_directory.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/operator/logical_set_operation.hpp>
#include <duckdb/planner/planner.hpp>

#include <filesystem>
#include <memory>
#include <string>

namespace {

using sirius::op::sirius_physical_hash_join;
using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

//! Exposes the protected `create_plan` overloads, including `create_plan(LogicalSetOperation&)`.
struct set_operation_planner : sirius::planner::sirius_physical_plan_generator {
  using sirius_physical_plan_generator::create_plan;
  using sirius_physical_plan_generator::sirius_physical_plan_generator;
};

//! Masks the optimizers Sirius masks at load and holds a transaction open while planning.
class planning_scope {
 public:
  explicit planning_scope(duckdb::Connection& con)
    : _con(con), _saved(duckdb::DBConfig::GetConfig(*con.context).options.disabled_optimizers)
  {
    auto& disabled = duckdb::DBConfig::GetConfig(*con.context).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    disabled.insert(duckdb::OptimizerType::LATE_MATERIALIZATION);
    sirius::test::query(_con, "BEGIN TRANSACTION");
  }

  ~planning_scope()
  {
    _con.Query("ROLLBACK");
    duckdb::DBConfig::GetConfig(*_con.context).options.disabled_optimizers = _saved;
  }

  planning_scope(planning_scope const&)            = delete;
  planning_scope& operator=(planning_scope const&) = delete;

 private:
  duckdb::Connection& _con;
  duckdb::set<duckdb::OptimizerType> _saved;
};

//! Sets `default_collation` for its lifetime.
class scoped_default_collation {
 public:
  scoped_default_collation(duckdb::Connection& con, std::string const& collation) : _con(con)
  {
    sirius::test::query(_con, "SET default_collation = '" + collation + "'");
  }
  ~scoped_default_collation() { _con.Query("RESET default_collation"); }

  scoped_default_collation(scoped_default_collation const&)            = delete;
  scoped_default_collation& operator=(scoped_default_collation const&) = delete;

 private:
  duckdb::Connection& _con;
};

duckdb::LogicalSetOperation* find_set_operation(duckdb::LogicalOperator& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_UNION:
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      return &op.Cast<duckdb::LogicalSetOperation>();
    default: break;
  }
  for (auto& child : op.children) {
    if (auto* found = find_set_operation(*child)) { return found; }
  }
  return nullptr;
}

//! Plans the first set operation in @p query's optimized logical plan; throws what the builder
//! throws.
duckdb::unique_ptr<sirius_physical_operator> plan_set_operation(duckdb::Connection& con,
                                                                std::string const& query)
{
  auto& context = *con.context;
  planning_scope const scope(con);

  duckdb::Parser parser(context.GetParserOptions());
  parser.ParseQuery(query);
  REQUIRE(parser.statements.size() == 1);

  duckdb::Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));
  REQUIRE(planner.plan);
  auto plan = std::move(planner.plan);

  duckdb::Optimizer optimizer(*planner.binder, context);
  plan = optimizer.Optimize(std::move(plan));
  plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  resolver.VisitOperator(*plan);

  auto* set_operation = find_set_operation(*plan);
  REQUIRE(set_operation != nullptr);
  set_operation_planner generator(context);
  return generator.create_plan(*set_operation);
}

duckdb::vector<sirius::logical_type> sirius_types(duckdb::vector<duckdb::LogicalType> const& types)
{
  return sirius::from_duckdb_vec(types);
}

//! Requires @p plan to be a hash join of @p join_type whose output and inputs are typed @p types.
sirius_physical_hash_join& require_hash_join(sirius_physical_operator& plan,
                                             duckdb::JoinType join_type,
                                             duckdb::vector<sirius::logical_type> const& types)
{
  REQUIRE(plan.type == SiriusPhysicalOperatorType::HASH_JOIN);
  auto& join = plan.Cast<sirius_physical_hash_join>();
  CHECK(join.join_type == join_type);
  REQUIRE(join.children.size() == 2);
  CHECK(join.types == types);
  CHECK(join.children[0]->types == types);
  CHECK(join.children[1]->types == types);
  return join;
}

//! Requires @p condition to be `column IS NOT DISTINCT FROM column` over references of @p type.
void require_null_safe_column_key(sirius::join_condition const& condition,
                                  uint32_t column,
                                  sirius::logical_type const& type)
{
  CHECK(condition.comparison == sirius::comparison_type::not_distinct_from);
  for (auto const* side : {condition.left.get(), condition.right.get()}) {
    REQUIRE(side != nullptr);
    REQUIRE(side->is_reference());
    CHECK(side->as_reference().column_index == column);
    CHECK(side->return_type() == type);
  }
}

class set_operation_lowering_fixture {
 public:
  set_operation_lowering_fixture()
  {
    auto const config = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" /
                        "data" / "minimal.yaml";
    db  = sirius::test::open_sirius_db(db_dir.path.c_str(), config);
    con = std::make_unique<duckdb::Connection>(*db);

    sirius::test::query(*con, "CREATE TABLE ia (k INTEGER, v VARCHAR)");
    sirius::test::query(*con, "CREATE TABLE ib (k INTEGER, v VARCHAR)");
    sirius::test::query(*con, "CREATE TABLE iwide (k BIGINT)");
    sirius::test::query(*con, "CREATE TABLE icollated (s VARCHAR COLLATE NOCASE)");
    sirius::test::query(*con, "CREATE TABLE iduration (i INTERVAL)");
    sirius::test::query(*con, "CREATE TABLE ilist (l INTEGER[])");
    sirius::test::query(*con, "INSERT INTO ia VALUES (1, 'a'), (2, 'b'), (3, 'c'), (NULL, 'n')");
    sirius::test::query(*con, "INSERT INTO ib VALUES (3, 'c'), (4, 'd'), (NULL, 'n')");
    sirius::test::query(*con, "INSERT INTO iwide VALUES (100), (200)");
    sirius::test::query(*con, "INSERT INTO icollated VALUES ('a'), ('A')");
    sirius::test::query(*con, "INSERT INTO iduration VALUES (INTERVAL 1 MONTH), (INTERVAL 30 DAY)");
    sirius::test::query(*con, "INSERT INTO ilist VALUES ([1, 2]), ([3])");
  }

  duckdb::unique_ptr<sirius_physical_operator> lower(std::string const& query)
  {
    return plan_set_operation(*con, query);
  }

  sirius::test::scoped_temp_directory db_dir;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
};

using Catch::Matchers::ContainsSubstring;

}  // namespace

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - the fork leaves UNION ALL on the union operator",
                 "[planner][set_operation][isolated_context]")
{
  auto const plan = lower("SELECT k FROM ia UNION ALL SELECT k FROM ib");
  CHECK(plan->type == SiriusPhysicalOperatorType::UNION);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - INTERSECT lowers to a null-safe SEMI hash join",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::INTEGER});
  auto const plan  = lower("SELECT k FROM ia INTERSECT SELECT k FROM ib");
  auto& join       = require_hash_join(*plan, duckdb::JoinType::SEMI, types);
  REQUIRE(join.conditions.size() == 1);
  require_null_safe_column_key(join.conditions[0], 0, types[0]);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - INTERSECT keys every column in column order",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR});
  auto const plan  = lower("SELECT k, v FROM ia INTERSECT SELECT k, v FROM ib");
  auto& join       = require_hash_join(*plan, duckdb::JoinType::SEMI, types);
  REQUIRE(join.conditions.size() == 2);
  require_null_safe_column_key(join.conditions[0], 0, types[0]);
  require_null_safe_column_key(join.conditions[1], 1, types[1]);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - INTERSECT plans both inputs at the common super-type",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::BIGINT});
  auto const plan  = lower("SELECT k FROM ia INTERSECT SELECT k FROM iwide");
  auto& join       = require_hash_join(*plan, duckdb::JoinType::SEMI, types);
  REQUIRE(join.conditions.size() == 1);
  require_null_safe_column_key(join.conditions[0], 0, types[0]);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - INTERSECT ALL is refused",
                 "[planner][set_operation][isolated_context]")
{
  REQUIRE_THROWS_WITH(lower("SELECT k FROM ia INTERSECT ALL SELECT k FROM ib"),
                      ContainsSubstring("INTERSECT ALL not supported"));
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - a CTE input whose definition is wider than its body is refused",
                 "[planner][set_operation][isolated_context]")
{
  // MATERIALIZED keeps the CTE from being inlined; the self-join keeps `v` in the definition.
  REQUIRE_THROWS_WITH(lower("SELECT k FROM ia "
                            "INTERSECT "
                            "(WITH m AS MATERIALIZED (SELECT k, v FROM ib) "
                            " SELECT m1.k FROM m m1 JOIN m m2 ON m1.v = m2.v)"),
                      ContainsSubstring("INTERSECT input 1 is planned with 2 columns, not 1"));
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - a CTE input whose definition matches its body is accepted",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR});
  auto const plan  = lower(
    "SELECT k, v FROM ia "
     "INTERSECT "
     "(WITH m AS MATERIALIZED (SELECT k, v FROM ib) SELECT k, v FROM m)");
  auto& join = require_hash_join(*plan, duckdb::JoinType::SEMI, types);
  CHECK(join.children[1]->type == SiriusPhysicalOperatorType::CTE);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - INTERSECT on a key DuckDB collates is refused",
                 "[planner][set_operation][isolated_context]")
{
  SECTION("a VARCHAR column with a collation")
  {
    REQUIRE_THROWS_WITH(lower("SELECT s FROM icollated INTERSECT SELECT s FROM icollated"),
                        ContainsSubstring("INTERSECT on column 0"));
  }
  SECTION("a plain VARCHAR column under a default collation")
  {
    scoped_default_collation const nocase(*con, "nocase");
    REQUIRE_THROWS_WITH(lower("SELECT v FROM ia INTERSECT SELECT v FROM ib"),
                        ContainsSubstring("INTERSECT on column 0"));
  }
  SECTION("an INTERVAL column")
  {
    REQUIRE_THROWS_WITH(lower("SELECT i FROM iduration INTERSECT SELECT i FROM iduration"),
                        ContainsSubstring("INTERSECT on column 0"));
  }
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - INTERSECT on a nested key is refused",
                 "[planner][set_operation][isolated_context]")
{
  REQUIRE_THROWS_WITH(lower("SELECT l FROM ilist INTERSECT SELECT l FROM ilist"),
                      ContainsSubstring("nested column operation on column 'column 0'"));
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an input planned narrower than the output type is refused",
                 "[planner][set_operation][isolated_context]")
{
  // DuckDB types sum(INTEGER) as HUGEINT; the Sirius aggregate plans it as BIGINT.
  REQUIRE_THROWS_WITH(lower("SELECT sum(k) FROM ia INTERSECT SELECT sum(k) FROM ib"),
                      ContainsSubstring("INTERSECT input 0 plans column 0 as BIGINT, not HUGEINT"));
}
