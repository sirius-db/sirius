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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <cstddef>
#include <string>

namespace {

std::size_t occurrences(std::string const& text, std::string const& word)
{
  std::size_t count = 0;
  for (auto at = text.find(word); at != std::string::npos; at = text.find(word, at + word.size())) {
    ++count;
  }
  return count;
}

/// DuckDB's plan for @p query as it is planned for the GPU, Sirius's optimizer hooks included.
std::string gpu_plan(duckdb::Connection& con, std::string const& query)
{
  sirius::test::scoped_sirius_setting gpu(con, "gpu_execution", true);
  auto result = con.Query("EXPLAIN " + query);
  REQUIRE(result);
  INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  return result->ToString();
}

}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - the delim CTE rewrites keep results",
                 "[integration][gpu_execution][delim_cte_rewrites]")
{
  run_ok("SET gpu_execution = false;");
  run_ok(
    "CREATE TABLE parts AS SELECT (range % 50)::INTEGER AS pk, "
    "concat('p', range % 3) AS pname FROM range(100);");
  run_ok("INSERT INTO parts VALUES (NULL, 'p1'), (NULL, 'p2'), (1, NULL);");
  // Keys 0..39. Every item of key 7 has a NULL qty: its sum is NULL although it has rows, so
  // "has a group" and "has a value" differ for it.
  run_ok(
    "CREATE TABLE items AS SELECT range::INTEGER AS id, (range % 40)::INTEGER AS fk, "
    "CASE WHEN range % 40 = 7 THEN NULL ELSE (range % 5)::INTEGER END AS qty FROM range(3000);");
  run_ok("INSERT INTO items VALUES (3000, NULL, 1), (3001, NULL, NULL);");
  // Keys 0..29: the keys 30..39 have items and no stock, the keys 40..49 have neither.
  run_ok(
    "CREATE TABLE stock AS SELECT (range % 30)::INTEGER AS sfk, range::INTEGER AS bin "
    "FROM range(60);");
  run_ok("INSERT INTO stock VALUES (NULL, 0);");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_sirius_setting native_cte(*con, "delim_join_as_cte", true);

  // TPC-Havoc Q05_V01 shape: the subquery joins two tables on its correlation column, and the
  // EXISTS reads the same rows.
  std::string const rows       = " FROM items i, stock s WHERE i.fk = s.sfk AND i.fk = p.pk";
  std::string const outer_rows = " FROM parts p WHERE p.pname = 'p1'";
  std::string const exists     = " AND EXISTS (SELECT 1" + rows + ")";
  std::string const scalar     = "SELECT p.pk, p.pname, (SELECT sum(i.qty)" + rows + ") AS total";

  SECTION("the rules change the plan of these queries")
  {
    auto const twin                 = scalar + outer_rows + exists;
    std::size_t without_exists_rule = 0;
    std::size_t without_carry       = 0;
    {
      sirius::test::scoped_sirius_setting off(*con, "enable_exists_from_aggregate", false);
      without_exists_rule = occurrences(gpu_plan(*con, twin), "items");
    }
    {
      sirius::test::scoped_sirius_setting off(*con, "enable_subquery_key_carry", false);
      without_carry = occurrences(gpu_plan(*con, twin), "SEMI");
    }
    auto const plan = gpu_plan(*con, twin);
    INFO(plan);
    // The subquery's tables are read once instead of twice, and its keys reach one more table.
    CHECK(occurrences(plan, "items") < without_exists_rule);
    CHECK(occurrences(plan, "SEMI") > without_carry);
  }

  SECTION("results")
  {
    bool const answer_exists = GENERATE(true, false);
    bool const carry_keys    = GENERATE(true, false);
    CAPTURE(answer_exists, carry_keys);
    sirius::test::scoped_sirius_setting exists_switch(
      *con, "enable_exists_from_aggregate", answer_exists);
    sirius::test::scoped_sirius_setting carry_switch(*con, "enable_subquery_key_carry", carry_keys);

    // Keys restricted on both tables; outer rows without subquery rows keep a NULL total.
    compare_gpu_vs_cpu(scalar + outer_rows);
    // The EXISTS answered from the subquery: key 7 stays, with a NULL total.
    compare_gpu_vs_cpu(scalar + outer_rows + exists);
    // The subquery inside an expression, and the mark of the removed join read by the query.
    compare_gpu_vs_cpu("SELECT p.pk, coalesce((SELECT sum(i.qty)" + rows + "), -1) AS total" +
                       outer_rows + exists);
    // Shapes the EXISTS rule declines stay as they were.
    compare_gpu_vs_cpu(scalar + outer_rows + " AND NOT EXISTS (SELECT 1" + rows + ")");
    compare_gpu_vs_cpu("SELECT p.pk, (SELECT count(*)" + rows + ") AS n" + outer_rows + exists);
  }
}
