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

#include <absl/cleanup/cleanup.h>
#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace {

using sirius::test::scoped_sirius_setting;

using result_rows = std::vector<std::vector<std::string>>;

std::size_t occurrences(std::string const& text, std::string const& word)
{
  std::size_t count = 0;
  for (auto at = text.find(word); at != std::string::npos; at = text.find(word, at + word.size())) {
    ++count;
  }
  return count;
}

/// The result of @p sql, which must succeed.
auto run_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query failed: " << sql << ": " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return result;
}

/// DuckDB's plan for @p query as it is planned for the GPU, Sirius's optimizer hooks included.
std::string gpu_plan(duckdb::Connection& con, std::string const& query)
{
  scoped_sirius_setting gpu(con, "gpu_execution", true);
  return run_ok(con, "EXPLAIN " + query)->ToString();
}

result_rows cpu_rows(duckdb::Connection& con, std::string const& query)
{
  scoped_sirius_setting cpu_only(con, "gpu_execution", false);
  return sirius::test::collect_rows(*run_ok(con, query));
}

/// @p query runs on the GPU, without a fallback, and returns the CPU's rows.
void compare_gpu_vs_cpu(duckdb::Connection& con, std::string const& query)
{
  INFO(query);
  auto const expected = cpu_rows(con, query);
  scoped_sirius_setting gpu(con, "gpu_execution", true);
  auto const before = sirius::test::get_transparent_execution_stats(con);
  auto const result = run_ok(con, query);
  sirius::test::require_transparent_execution_delta(
    before, sirius::test::get_transparent_execution_stats(con), 1, 0, 1);
  REQUIRE(sirius::test::collect_rows(*result) == expected);
}

void create_tables(duckdb::Connection& con)
{
  scoped_sirius_setting cpu_only(con, "gpu_execution", false);
  run_ok(con,
         "CREATE TABLE parts AS SELECT (range % 50)::INTEGER AS pk, "
         "concat('p', range % 3) AS pname FROM range(100);");
  run_ok(con, "INSERT INTO parts VALUES (NULL, 'p1'), (NULL, 'p2'), (1, NULL);");
  // Keys 0..39. Every item of key 7 has a NULL qty: its sum is NULL although it has rows, so
  // "has a group" and "has a value" differ for it.
  run_ok(con,
         "CREATE TABLE items AS SELECT range::INTEGER AS id, (range % 40)::INTEGER AS fk, "
         "CASE WHEN range % 40 = 7 THEN NULL ELSE (range % 5)::INTEGER END AS qty "
         "FROM range(3000);");
  run_ok(con, "INSERT INTO items VALUES (3000, NULL, 1), (3001, NULL, NULL);");
  // Keys 0..29: the keys 30..39 have items and no stock, the keys 40..49 have neither.
  run_ok(con,
         "CREATE TABLE stock AS SELECT (range % 30)::INTEGER AS sfk, range::INTEGER AS bin "
         "FROM range(60);");
  run_ok(con, "INSERT INTO stock VALUES (NULL, 0);");
  run_ok(con, "CHECKPOINT;");
}

// TPC-Havoc Q05_V01 shape: the subquery joins two tables on its correlation column, and the
// EXISTS reads the same rows.
std::string const rows       = " FROM items i, stock s WHERE i.fk = s.sfk AND i.fk = p.pk";
std::string const outer_rows = " FROM parts p WHERE p.pname = 'p1'";
std::string const exists     = " AND EXISTS (SELECT 1" + rows + ")";
std::string const scalar     = "SELECT p.pk, p.pname, (SELECT sum(i.qty)" + rows + ") AS total";
/// The query both rules rewrite.
std::string const twin = scalar + outer_rows + exists;

/// The rules change the plan of `twin`, and no combination of them changes a result. Shared by the
/// one-GPU and the two-GPU case.
void check_rewrites(duckdb::Connection& con)
{
  create_tables(con);

  SECTION("the rules change the plan of these queries")
  {
    std::size_t without_exists_rule = 0;
    std::size_t without_carry       = 0;
    {
      scoped_sirius_setting off(con, "enable_exists_from_aggregate", false);
      without_exists_rule = occurrences(gpu_plan(con, twin), "items");
    }
    {
      scoped_sirius_setting off(con, "enable_subquery_key_carry", false);
      without_carry = occurrences(gpu_plan(con, twin), "SEMI");
    }
    // Both rules are on by default.
    auto const plan = gpu_plan(con, twin);
    INFO(plan);
    // The subquery's tables are read once instead of twice, and its keys reach one more table.
    CHECK(occurrences(plan, "items") < without_exists_rule);
    CHECK(occurrences(plan, "SEMI") > without_carry);
  }

  SECTION("results")
  {
    // Loops, not GENERATE: a generator would run this function, and build the tables, once per
    // combination.
    for (bool const answer_exists : {true, false}) {
      for (bool const carry_keys : {true, false}) {
        CAPTURE(answer_exists, carry_keys);
        scoped_sirius_setting exists_switch(con, "enable_exists_from_aggregate", answer_exists);
        scoped_sirius_setting carry_switch(con, "enable_subquery_key_carry", carry_keys);

        // Keys restricted on both tables; outer rows without subquery rows keep a NULL total.
        compare_gpu_vs_cpu(con, scalar + outer_rows);
        // The EXISTS answered from the subquery: key 7 stays, with a NULL total.
        compare_gpu_vs_cpu(con, twin);
        // The subquery inside an expression, and the mark of the removed join read by the query.
        compare_gpu_vs_cpu(con,
                           "SELECT p.pk, coalesce((SELECT sum(i.qty)" + rows + "), -1) AS total" +
                             outer_rows + exists);
        // Shapes the EXISTS rule declines stay as they were.
        compare_gpu_vs_cpu(con, scalar + outer_rows + " AND NOT EXISTS (SELECT 1" + rows + ")");
        compare_gpu_vs_cpu(con,
                           "SELECT p.pk, (SELECT count(*)" + rows + ") AS n" + outer_rows + exists);
      }
    }
  }
}

}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - the delim CTE rewrites keep results",
                 "[integration][gpu_execution][delim_cte_rewrites]")
{
  check_rewrites(*con);
}

TEST_CASE("gpu_execution - the delim CTE rewrites keep results on two GPUs",
          "[integration][gpu_execution][delim_cte_rewrites][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  // The `[integration][multi_gpu]` tags make the `shared_env_listener` in `unittest.cpp` activate
  // this environment; resuming it again here would replace its live SiriusContext.
  auto* env = sirius::test::acquire_integration_env_for(2);
  REQUIRE(env != nullptr);
  REQUIRE(env->is_active());
  auto const database = std::filesystem::temp_directory_path() /
                        ("sirius_delim_cte_rewrites_mgpu_" + std::to_string(::getpid()) + ".db");
  absl::Cleanup remove_database = [&database] {
    std::error_code ignored;
    std::filesystem::remove(database, ignored);
    std::filesystem::remove(database.string() + ".wal", ignored);
  };
  auto con = std::make_unique<duckdb::Connection>(env->make_connection());
  run_ok(*con, "ATTACH '" + database.string() + "' AS delim_rewrites_mgpu;");
  run_ok(*con, "USE delim_rewrites_mgpu;");
  check_rewrites(*con);
  con->Query("USE memory;");
  con->Query("DETACH delim_rewrites_mgpu;");
}

// The rewrites are made for the GPU, before DuckDB optimizes the plan, so the CPU plan DuckDB
// builds for a fallback comes from the rewritten plan. When Sirius then does not run the query,
// the CPU must run the plan DuckDB makes alone.
TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - a fallback of a rewritten query plans it again for the CPU",
                 "[integration][gpu_execution][delim_cte_rewrites]")
{
  create_tables(*con);
  auto replans = [&] {
    return sirius::test::get_transparent_execution_stats(*con).fallback_replans;
  };

  SECTION("GPU planning turns the query down")
  {
    scoped_sirius_setting inject{*con, "sirius_test_inject_read_view_mismatch", "finalize"};
    auto const before = replans();
    expect_plan_fallback_matches_cpu(twin);
    CHECK(replans() == before + 1);
    // A query no rule rewrites keeps the CPU plan DuckDB already built.
    expect_plan_fallback_matches_cpu("SELECT count(*), sum(qty) FROM items");
    CHECK(replans() == before + 1);
  }

  SECTION("the GPU fails while running the query")
  {
    scoped_sirius_setting inject{*con, "sirius_test_inject_transparent_gpu_error", "injected"};
    auto const before = replans();
    expect_runtime_fallback_matches_cpu(twin);
    CHECK(replans() == before + 1);
    expect_runtime_fallback_matches_cpu("SELECT count(*), sum(qty) FROM items");
    CHECK(replans() == before + 1);
  }
}
