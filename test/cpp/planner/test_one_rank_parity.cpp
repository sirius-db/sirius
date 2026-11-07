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

// One rank is not distributed: with sirius_exchange_peers unset or naming one peer, Sirius plans
// and runs a query as it does without the distributed settings, and connects to nothing.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <sirius_context.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void write_tables(fs::path const& dir)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto r =
    con.Query("COPY (SELECT range AS k, (range * 7) % 1000 AS v FROM range(20000)) TO " +
              sirius::test::sql_literal((dir / "t.parquet").string()) + " (FORMAT PARQUET);");
  REQUIRE_FALSE(r->HasError());
  r = con.Query("COPY (SELECT range AS k, 'n' || (range % 37) AS name FROM range(5000)) TO " +
                sirius::test::sql_literal((dir / "d.parquet").string()) + " (FORMAT PARQUET);");
  REQUIRE_FALSE(r->HasError());
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.2\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 4000000000\n"
       "      initial_number_pools: 2\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 2\n"
       "    task_creator:\n"
       "      num_threads: 1\n"
       "    downgrade:\n"
       "      num_threads: 1\n";
}

void pause_shared_envs()
{
  for (auto* env : {sirius::test::g_shared_env,
                    sirius::test::g_integration_env,
                    sirius::test::g_integration_env_2gpu}) {
    if (env && env->is_active()) { env->pause(); }
  }
}

class physical_planner : public sirius::planner::sirius_physical_plan_generator {
 public:
  using sirius_physical_plan_generator::sirius_physical_plan_generator;

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan(
    duckdb::unique_ptr<duckdb::LogicalOperator> logical)
  {
    logical->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*logical);
    return create_plan(*logical);
  }
};

void append_shape(sirius::op::sirius_physical_operator const& op, std::string& out)
{
  out += sirius::op::SiriusPhysicalOperatorToString(op.type);
  if (op.children.empty()) { return; }
  out += '(';
  for (auto const& child : op.children) {
    append_shape(*child, out);
    out += ' ';
  }
  out += ')';
}

/// The operator tree Sirius lowers @p sql to, with every GPU pipeline operator in place.
std::string lowered_shape(duckdb::Connection& con, std::string const& sql)
{
  REQUIRE_FALSE(con.Query("SET gpu_execution = false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION READ ONLY")->HasError());
  physical_planner planner(*con.context);
  auto plan = planner.plan(con.ExtractPlan(sql));
  planner.insert_gpu_pipeline_operators(plan);
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
  std::string shape;
  append_shape(*plan, shape);
  return shape;
}

/// @p sql's result, run on the GPU.
std::string gpu_result(duckdb::Connection& con, std::string const& sql)
{
  REQUIRE_FALSE(con.Query("SET gpu_execution = true")->HasError());
  auto result = con.Query(sql);
  REQUIRE_FALSE(result->HasError());
  return result->ToString();
}

}  // namespace

TEST_CASE("one peer plans and runs like no peer, and connects to nothing",
          "[planner][one_rank_parity]")
{
  pause_shared_envs();
  sirius::test::scratch_dir scratch{"one_rank_parity"};
  write_tables(scratch.path());
  auto const config = scratch.path() / "sirius.yaml";
  write_config(config);
  sirius::test::shared_test_env env(config);
  auto con = env.make_connection();

  auto const table = [&](char const* file) {
    return "read_parquet(" + sirius::test::sql_literal((scratch.path() / file).string()) + ")";
  };
  auto const t = table("t.parquet");
  auto const d = table("d.parquet");
  // A filtered join with a group by and a top-N, a full ORDER BY (SORT_SAMPLE), and a semi join.
  std::vector<std::string> const queries{
    "SELECT d.name, sum(t.v) AS s FROM " + t + " t JOIN " + d +
      " d ON t.k = d.k WHERE d.k < 4000 GROUP BY d.name ORDER BY s DESC, d.name LIMIT 10",
    "SELECT k, v FROM " + t + " ORDER BY v, k",
    "SELECT count(*) FROM " + t + " WHERE k IN (SELECT k FROM " + d + " WHERE k % 3 = 0)"};

  std::vector<std::string> shapes;
  std::vector<std::string> results;
  for (auto const& sql : queries) {
    shapes.push_back(lowered_shape(con, sql));
    results.push_back(gpu_result(con, sql));
  }

  REQUIRE_FALSE(con.Query("SET sirius_exchange_peers = '127.0.0.1:62999'")->HasError());
  REQUIRE_FALSE(con.Query("SET sirius_exchange_rank = 0")->HasError());
  for (std::size_t i = 0; i < queries.size(); ++i) {
    INFO(queries[i]);
    CHECK(lowered_shape(con, queries[i]) == shapes[i]);
    CHECK(gpu_result(con, queries[i]) == results[i]);
  }

  auto ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(ctx);
  CHECK(ctx->distributed() == nullptr);
  REQUIRE_FALSE(con.Query("SET sirius_exchange_peers = ''")->HasError());
}
