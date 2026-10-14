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

// [pin_unique_join] — a pinned parquet column proven distinct at pin time is a join-key proof.
// GPU required.
//
// Parquet scans declare no PRIMARY KEY, so without SIRIUS_EXP_PIN_UNIQUE_JOIN every join over
// them takes cuDF's two-pass hash join. With it, the pin observes every column automatically and
// the planner takes the single-pass distinct hash join for a build key the pin proved distinct.
//
// Each case checks two things: the plan (did the proof fire, or correctly not fire) and the
// answer against DuckDB on the CPU. A wrong proof drops matches, so the answer catches it even
// where the plan check alone would not.

#include "op/sirius_physical_hash_join.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t dim_rows  = 20'000;
constexpr std::int64_t fact_rows = 200'000;

/// Sets one environment variable for the scope of a case, and restores it afterwards.
struct scoped_env {
  std::string name;
  bool had_previous{false};
  std::string previous;

  scoped_env(std::string var, char const* value) : name(std::move(var))
  {
    char const* old = std::getenv(name.c_str());
    had_previous    = old != nullptr;
    if (had_previous) { previous = old; }
    if (value == nullptr) {
      ::unsetenv(name.c_str());
    } else {
      ::setenv(name.c_str(), value, 1);
    }
  }
  ~scoped_env()
  {
    if (had_previous) {
      ::setenv(name.c_str(), previous.c_str(), 1);
    } else {
      ::unsetenv(name.c_str());
    }
  }
};

void run_generator(duckdb::Connection& gen, std::string const& sql)
{
  auto result = gen.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("generator error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

/// `unique_dim.k` is distinct and `dup_dim.k` repeats every value twice. Both are written in key
/// order, so each pinned chunk is sorted, which is what the automatic probe requires.
void generate_parquet(fs::path const& unique_dim, fs::path const& dup_dim, fs::path const& fact)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  run_generator(gen,
                "COPY (SELECT range AS k, range * 3 AS payload FROM range(" +
                  std::to_string(dim_rows) + ")) TO " +
                  sirius::test::sql_literal(unique_dim.string()) + " (FORMAT PARQUET);");
  run_generator(gen,
                "COPY (SELECT range // 2 AS k, range AS payload FROM range(" +
                  std::to_string(dim_rows) + ")) TO " +
                  sirius::test::sql_literal(dup_dim.string()) + " (FORMAT PARQUET);");
  run_generator(gen,
                "COPY (SELECT range AS f_id, range % " + std::to_string(dim_rows + 100) +
                  " AS f_k, range % 7 AS v FROM range(" + std::to_string(fact_rows) + ")) TO " +
                  sirius::test::sql_literal(fact.string()) + " (FORMAT PARQUET);");
}

std::string join_query(fs::path const& fact, fs::path const& dim)
{
  return "SELECT count(*), sum(f.v + d.payload) FROM read_parquet(" +
         sirius::test::sql_literal(fact.string()) + ") f JOIN read_parquet(" +
         sirius::test::sql_literal(dim.string()) + ") d ON f.f_k = d.k";
}

std::vector<std::string> rows_of(duckdb::QueryResult& result)
{
  std::vector<std::string> rows;
  auto result_rows = result.Collection().GetRows();
  for (duckdb::idx_t row_idx = 0; row_idx < result_rows.size(); ++row_idx) {
    std::string row;
    for (duckdb::idx_t col_idx = 0; col_idx < result.ColumnCount(); ++col_idx) {
      row += result_rows.GetValue(col_idx, row_idx).ToString() + "|";
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

/// The ground truth: the same query on DuckDB with Sirius disabled.
std::vector<std::string> cpu_answer(std::string const& sql)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto result = con.Query(sql);
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  return rows_of(*result);
}

std::vector<std::string> gpu_answer(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return rows_of(*result);
}

/// Plans @p sql with the Sirius plan generator on @p con 's context, without running it.
duckdb::unique_ptr<sirius::op::sirius_physical_operator> sirius_plan(duckdb::Connection& con,
                                                                     std::string const& sql)
{
  auto& context = *con.context;
  con.Query("BEGIN TRANSACTION");
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
  try {
    duckdb::Parser parser(context);
    parser.ParseQuery(sql);
    REQUIRE(!parser.statements.empty());
    duckdb::Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);
    auto plan = std::move(planner.plan);
    if (duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(context)) {
      duckdb::Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    sirius::planner::sirius_physical_plan_generator generator(context);
    result = generator.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    throw;
  }
  con.Query("COMMIT");
  return result;
}

void collect_hash_joins(sirius::op::sirius_physical_operator* root,
                        std::vector<sirius::op::sirius_physical_hash_join*>& out)
{
  if (root == nullptr) { return; }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::HASH_JOIN) {
    out.push_back(&root->Cast<sirius::op::sirius_physical_hash_join>());
  }
  for (auto& child : root->children) {
    collect_hash_joins(child.get(), out);
  }
}

/// Whether the plan of @p sql has exactly one hash join, and that join's build keys are proven
/// unique.
bool join_proven_unique(duckdb::Connection& con, std::string const& sql)
{
  auto plan = sirius_plan(con, sql);
  REQUIRE(plan);
  std::vector<sirius::op::sirius_physical_hash_join*> joins;
  collect_hash_joins(plan.get(), joins);
  REQUIRE(joins.size() == 1);
  return joins.front()->unique_build_keys;
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.4\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 4\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n";
}

void pin(duckdb::Connection& con, fs::path const& file, std::string const& name)
{
  auto result = con.Query("CALL pin_table(" + sirius::test::sql_literal(file.string()) +
                          ", tier='gpu', name='" + name + "');");
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("pin_table error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

}  // namespace

TEST_CASE("a pinned distinct parquet column proves a join's build keys unique",
          "[integration][pin_unique_join]")
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  sirius::test::scratch_dir scratch{"pin_unique_join"};
  auto const& tmp       = scratch.path();
  auto const unique_dim = tmp / "unique_dim.parquet";
  auto const dup_dim    = tmp / "dup_dim.parquet";
  auto const fact       = tmp / "fact.parquet";
  generate_parquet(unique_dim, dup_dim, fact);

  auto const yaml_path = tmp / "pin_unique_join.yaml";
  write_config(yaml_path);
  sirius::test::shared_test_env local_env(yaml_path);
  auto con      = local_env.make_connection();
  auto fallback = con.Query("SET enable_duckdb_fallback = false;");
  REQUIRE(fallback);
  REQUIRE_FALSE(fallback->HasError());

  // Automatic selection only: no explicit column list.
  scoped_env const no_explicit_columns{"SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", nullptr};
  {
    scoped_env const join_proof{"SIRIUS_EXP_PIN_UNIQUE_JOIN", "1"};
    pin(con, unique_dim, "pin_unique_join_unique_dim");
    pin(con, dup_dim, "pin_unique_join_dup_dim");

    SECTION("a distinct key takes the distinct hash join and answers right")
    {
      auto const sql = join_query(fact, unique_dim);
      CHECK(join_proven_unique(con, sql));
      CHECK(gpu_answer(con, sql) == cpu_answer(sql));
    }

    SECTION("a repeated key is not proven and keeps every match")
    {
      auto const sql = join_query(fact, dup_dim);
      CHECK_FALSE(join_proven_unique(con, sql));
      CHECK(gpu_answer(con, sql) == cpu_answer(sql));
    }
  }

  SECTION("without the flag the planner ignores the pin's facts")
  {
    CHECK_FALSE(join_proven_unique(con, join_query(fact, unique_dim)));
  }

  for (auto const* name : {"pin_unique_join_unique_dim", "pin_unique_join_dup_dim"}) {
    con.Query("CALL unpin_table('" + std::string{name} + "');");
  }
}
