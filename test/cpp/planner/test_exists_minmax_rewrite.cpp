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

#include "expression/ast/case_expr.hpp"
#include "expression/ast/node.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_projection.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/scoped_sirius_setting.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;
using T = sirius::op::SiriusPhysicalOperatorType;

namespace {

/// Parse and optimize @p query, apply the rewrite when @p rewrite is set, then resolve it and hand
/// the logical plan to @p consume inside a transaction. Returns the number of subqueries rewritten.
template <class Consume>
std::size_t with_resolved_plan(Connection& con,
                               std::string const& query,
                               bool rewrite,
                               Consume&& consume,
                               std::uint64_t gpu_memory_budget = 0)
{
  std::size_t rewrites = 0;
  auto& context        = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");
  try {
    Parser parser(context);
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
    if (rewrite) {
      rewrites = sirius::planner::rewrite_correlated_inequality_exists(
        context, *planner.binder, plan, gpu_memory_budget);
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver verifier(true);
    verifier.VisitOperator(*plan);
    ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    consume(context, plan);
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }
  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return rewrites;
}

std::vector<sirius::op::sirius_physical_operator*> collect(
  sirius::op::sirius_physical_operator* root, T type)
{
  std::vector<sirius::op::sirius_physical_operator*> out;
  if (!root) { return out; }
  if (root->type == type) { out.push_back(root); }
  for (auto& child : root->children) {
    auto sub = collect(child.get(), type);
    out.insert(out.end(), sub.begin(), sub.end());
  }
  // A delim join keeps its inner join outside `children`.
  if (root->type == T::LEFT_DELIM_JOIN || root->type == T::RIGHT_DELIM_JOIN) {
    auto const& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    auto sub          = collect(delim.join.get(), type);
    out.insert(out.end(), sub.begin(), sub.end());
  }
  return out;
}

struct exists_rewrite_fixture {
  exists_rewrite_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    // The GPU seq_scan ingestible requires a single-file block manager: an on-disk database.
    db  = sirius::test::open_sirius_db(db_dir.file("exists_rewrite.db").c_str(), cfg);
    con = std::make_unique<Connection>(*db);
    set_flag(true);

    con->Query("CREATE TABLE dim (id BIGINT, tag INTEGER)");
    con->Query("INSERT INTO dim SELECT range, range % 5 FROM range(1, 30)");
    con->Query(
      "CREATE TABLE fact (k BIGINT, v INTEGER, w INTEGER, x DOUBLE, s VARCHAR, late BOOLEAN, "
      "d DATE)");
    con->Query(
      "INSERT INTO fact SELECT (range * 7) % 61, (range * 13) % 1000, "
      "CASE WHEN range % 4 = 0 THEN NULL ELSE range % 97 END, range * 0.5, concat('s', range), "
      "range % 3 = 0, DATE '2020-01-01' + (range % 50)::INTEGER FROM range(5000)");
  }

  void set_flag(bool on)
  {
    auto result =
      con->Query(std::string("SET enable_exists_minmax_rewrite = ") + (on ? "true" : "false"));
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(result->HasError());
    rewrite_enabled = on;
  }

  /// Subqueries the rewrite replaces in @p query's optimized plan, independent of the setting.
  std::size_t rewrites(std::string const& query, std::uint64_t gpu_memory_budget = 0)
  {
    return with_resolved_plan(*con, query, true, [](ClientContext&, auto&) {}, gpu_memory_budget);
  }

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> physical_plan(std::string const& query)
  {
    duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
    with_resolved_plan(*con, query, rewrite_enabled, [&](ClientContext& context, auto& plan) {
      sirius::planner::sirius_physical_plan_generator gen(context);
      result = gen.create_plan(std::move(plan));
    });
    REQUIRE(result);
    return result;
  }

  sirius::test::scratch_dir db_dir{"exists_rewrite"};
  bool rewrite_enabled = false;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

/// Join type of every delim join's inner join in @p plan, of either orientation.
std::vector<duckdb::JoinType> delim_join_types(sirius::op::sirius_physical_operator* plan)
{
  std::vector<duckdb::JoinType> types;
  for (auto const type : {T::LEFT_DELIM_JOIN, T::RIGHT_DELIM_JOIN}) {
    for (auto* delim : collect(plan, type)) {
      auto const& join = delim->Cast<sirius::op::sirius_physical_delim_join>().join;
      types.push_back(join->Cast<sirius::op::sirius_physical_hash_join>().join_type);
    }
  }
  return types;
}

constexpr char const* q21_shape_query =
  "SELECT f.k, count(*) FROM fact f WHERE f.late "
  "AND EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v) "
  "AND NOT EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v AND g.late) "
  "GROUP BY f.k";

/// An EXISTS and a NOT EXISTS over `fact`, correlated on @p first and @p second.
std::string siblings(std::string const& first, std::string const& second)
{
  return "SELECT f.k FROM fact f WHERE f.late AND EXISTS (SELECT 1 FROM fact g WHERE " + first +
         ") AND NOT EXISTS (SELECT 1 FROM fact g WHERE " + second + ")";
}

}  // namespace

TEST_CASE_METHOD(exists_rewrite_fixture,
                 "exists rewrite merges sibling subqueries over the same table into one scan",
                 "[exists_rewrite][plan]")
{
  CHECK(rewrites(q21_shape_query) == 2);

  auto merged = physical_plan(q21_shape_query);
  // One LEFT delim join and one aggregate answer both subqueries.
  CHECK(delim_join_types(merged.get()) == std::vector{duckdb::JoinType::LEFT});

  set_flag(false);
  auto unrewritten = physical_plan(q21_shape_query);
  CHECK(delim_join_types(unrewritten.get()).size() == 2);
  // The outer rows and one shared subquery scan, against one scan per subquery.
  CHECK(collect(merged.get(), T::GPU_SCAN).size() == 2);
  CHECK(collect(unrewritten.get(), T::GPU_SCAN).size() == 3);
}

TEST_CASE_METHOD(exists_rewrite_fixture,
                 "exists rewrite computes a filtered subquery value once for MIN and MAX",
                 "[exists_rewrite][plan]")
{
  auto const query = siblings("g.k = f.k AND g.v <> f.v", "g.k = f.k AND g.v <> f.v AND g.late");
  REQUIRE(rewrites(query) == 2);
  auto plan = physical_plan(query);
  // Only the NOT EXISTS has a filter of its own: one masked `v` feeds both its MIN and MAX.
  std::size_t masked = 0;
  for (auto* projection : collect(plan.get(), T::PROJECTION)) {
    for (auto const& entry :
         projection->Cast<sirius::op::sirius_physical_projection>().select_list) {
      if (entry && entry->holds<sirius::ast::case_expr>()) { ++masked; }
    }
  }
  CHECK(masked == 1);
}

TEST_CASE_METHOD(exists_rewrite_fixture,
                 "exists rewrite covers null-safe comparisons and temporal values",
                 "[exists_rewrite][plan]")
{
  CHECK(rewrites(siblings("g.k = f.k AND g.d <> f.d", "g.k = f.k AND g.d <> f.d AND g.v > 3")) ==
        2);
  CHECK(rewrites(siblings("g.k = f.k AND g.w IS DISTINCT FROM f.w",
                          "g.k IS NOT DISTINCT FROM f.k AND g.w IS DISTINCT FROM f.w")) == 2);
}

TEST_CASE_METHOD(exists_rewrite_fixture,
                 "exists rewrite leaves lone and unsupported subqueries alone",
                 "[exists_rewrite][plan]")
{
  // A lone subquery, as a delim join or as the plain SEMI join DuckDB leaves for simple shapes.
  CHECK(rewrites("SELECT f.k FROM fact f WHERE f.w > f.v AND EXISTS "
                 "(SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v)") == 0);
  CHECK(rewrites("SELECT f.k FROM fact f SEMI JOIN fact g ON g.k = f.k AND g.v <> f.v") == 0);
  // Siblings of which one has no inequality, one that is not <>, two inequalities, a
  // floating-point or string value, or a non-integer key.
  auto const supported = "g.k = f.k AND g.v <> f.v";
  for (char const* other : {"g.k = f.k",
                            "g.k = f.k AND g.v < f.v",
                            "g.k = f.k AND g.v <> f.v AND g.w <> f.w",
                            "g.k = f.k AND g.x <> f.x",
                            "g.k = f.k AND g.s <> f.s",
                            "g.s = f.s AND g.v <> f.v"}) {
    CAPTURE(other);
    CHECK(rewrites(siblings(supported, other)) == 0);
  }
  // Siblings over different tables.
  CHECK(rewrites("SELECT f.k FROM fact f WHERE f.late "
                 "AND EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v) "
                 "AND NOT EXISTS (SELECT 1 FROM dim d WHERE d.id = f.k AND d.tag <> f.v)") == 0);
}

TEST_CASE_METHOD(exists_rewrite_fixture,
                 "exists rewrite merges the plain joins left under delim_join_as_cte",
                 "[exists_rewrite][plan]")
{
  // DuckDB removes q21's delim joins, comparing keys with `=` in one subquery and
  // IS NOT DISTINCT FROM in the other; the two still share one scan.
  sirius::test::scoped_sirius_setting as_cte{*con, "delim_join_as_cte", true};
  CHECK(rewrites(q21_shape_query) == 2);
  auto merged = physical_plan(q21_shape_query);
  CHECK(delim_join_types(merged.get()) == std::vector{duckdb::JoinType::LEFT});
  CHECK(collect(merged.get(), T::GPU_SCAN).size() == 2);
}

TEST_CASE_METHOD(exists_rewrite_fixture,
                 "exists rewrite leaves plain joins over a GPU pin that fits unmerged",
                 "[exists_rewrite][plan]")
{
  // 5000 rows of a BIGINT key and an INTEGER value: 60,000 bytes per join.
  std::uint64_t const fits = 1 << 20, spills = 1 << 10;
  {
    sirius::test::scoped_sirius_setting as_cte{*con, "delim_join_as_cte", true};
    CHECK(rewrites(q21_shape_query, fits) == 2);
    REQUIRE_FALSE(con->Query("CHECKPOINT")->HasError());
    auto const pin = con->Query("CALL pin_table(format='duckdb', name='fact', tier='gpu')");
    INFO(pin->GetError());
    REQUIRE_FALSE(pin->HasError());
    CHECK(rewrites(q21_shape_query, fits) == 0);
    CHECK(rewrites(q21_shape_query, spills) == 2);
  }
  // Delim joins are merged regardless.
  CHECK(rewrites(q21_shape_query, fits) == 2);
  REQUIRE_FALSE(con->Query("CALL unpin_table('fact')")->HasError());
}

TEST_CASE_METHOD(exists_rewrite_fixture,
                 "exists rewrite merges plain joins over a GPU pin missing a column they read",
                 "[exists_rewrite][plan]")
{
  sirius::test::scoped_sirius_setting as_cte{*con, "delim_join_as_cte", true};
  REQUIRE_FALSE(con->Query("CHECKPOINT")->HasError());
  // Both subqueries also compare `v`, which the pin lacks, so their scans read storage.
  auto const pin =
    con->Query("CALL pin_table(format='duckdb', name='fact', tier='gpu', cols=['k'])");
  INFO(pin->GetError());
  REQUIRE_FALSE(pin->HasError());
  CHECK(rewrites(q21_shape_query, std::uint64_t{1} << 20) == 2);
  REQUIRE_FALSE(con->Query("CALL unpin_table('fact')")->HasError());
}
