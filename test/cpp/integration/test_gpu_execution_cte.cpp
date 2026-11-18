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

#include <string>

namespace {

class CteFixture : public sirius::test::GpuExecutionFixture {
 public:
  CteFixture()
  {
    run_ok(
      "CREATE TABLE ct AS SELECT range::BIGINT AS k, (range % 5)::BIGINT AS g FROM range(50);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

// Regression: statistics propagation removes every CTE reference, which previously hung.
TEST_CASE_METHOD(CteFixture,
                 "gpu_execution materialized CTE with every reference folded away",
                 "[integration][gpu_execution][cte]")
{
  compare_gpu_vs_cpu(
    "WITH c AS MATERIALIZED (SELECT g AS c0, max(k) AS c1 FROM ct WHERE (673 IS NULL) "
    "GROUP BY g HAVING max(k) < 188) "
    "SELECT c1, count(*) FROM c WHERE c1 IS NULL GROUP BY c1");

  // The outer CTE is read only from inside the unread one.
  compare_gpu_vs_cpu(
    "WITH o AS MATERIALIZED (SELECT k FROM ct), "
    "c AS MATERIALIZED (SELECT k AS c0, max(k) AS c1 FROM o WHERE (673 IS NULL) "
    "GROUP BY k HAVING max(k) < 188) "
    "SELECT c1, count(*) FROM c WHERE c1 IS NULL GROUP BY c1");

  // Control: the same CTE read normally.
  compare_gpu_vs_cpu(
    "WITH c AS MATERIALIZED (SELECT g AS c0, max(k) AS c1 FROM ct GROUP BY g) "
    "SELECT c1, count(*) FROM c GROUP BY c1");
}

// The consumer folds to an empty result but DuckDB keeps the definition's scan, which the GPU
// plan drops. Read-view validation must accept that scan rather than reject the GPU plan.
TEST_CASE_METHOD(CteFixture,
                 "gpu_execution materialized CTE whose skipped definition scans a table",
                 "[integration][gpu_execution][cte]")
{
  compare_gpu_vs_cpu(
    "WITH c AS MATERIALIZED (SELECT k FROM ct) "
    "SELECT min(k) AS m FROM c HAVING min(k) > 100");

  // The outer CTE's scan is reachable only through the unread inner CTE.
  compare_gpu_vs_cpu(
    "WITH o AS MATERIALIZED (SELECT k FROM ct), c AS MATERIALIZED (SELECT k FROM o) "
    "SELECT min(k) AS m FROM c HAVING min(k) > 100");

  // A planned scan of the same table still corresponds to its own original.
  compare_gpu_vs_cpu(
    "WITH c AS MATERIALIZED (SELECT k FROM ct) "
    "SELECT min(k) AS m FROM c HAVING min(k) > 100 "
    "UNION ALL SELECT max(g) FROM ct");
}

// An unreferenced modifying CTE must still execute. If Sirius cannot plan its write, the
// statement must fall back to DuckDB rather than silently discarding the CTE definition.
TEST_CASE_METHOD(CteFixture,
                 "gpu_execution keeps side effects of an unread CTE",
                 "[integration][gpu_execution][cte]")
{
  run_ok("CREATE TABLE cte_write_target (k INTEGER);");
  run_ok("WITH ins AS (INSERT INTO cte_write_target VALUES (42) RETURNING k) SELECT 1;");

  auto result = con->Query("SELECT count(*), min(k) FROM cte_write_target;");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  auto rows = collect_rows(*result);
  REQUIRE(rows.size() == 1);
  REQUIRE(rows[0].size() == 2);
  REQUIRE(rows[0][0] == "1");
  REQUIRE(rows[0][1] == "42");
}

// Sirius has no operators for the delim joins DuckDB makes of correlated subqueries when
// `delim_join_as_cte` is off, so a query it would run is an error there, not a silent CPU run.
TEST_CASE_METHOD(CteFixture,
                 "gpu_execution rejects delim_join_as_cte = false",
                 "[integration][gpu_execution][cte]")
{
  std::string const expected = "delim_join_as_cte = true";
  std::string const plain    = "SELECT count(*) FROM ct WHERE k > 10";
  std::string const correlated =
    "SELECT count(*) FROM ct o WHERE k > (SELECT avg(k) FROM ct i WHERE i.g = o.g)";

  run_ok("PREPARE under_default AS " + plain + ";");
  run_ok("EXECUTE under_default;");
  // Refused with or without a subquery, and whether or not a CPU fallback is allowed.
  for (bool const fallback : {true, false}) {
    CAPTURE(fallback);
    sirius::test::scoped_sirius_setting allow{*con, "enable_duckdb_fallback", fallback};
    sirius::test::scoped_sirius_setting off{*con, "delim_join_as_cte", false};
    CHECK(query_error(plain).find(expected) != std::string::npos);
    CHECK(query_error(correlated).find(expected) != std::string::npos);
  }
  {
    sirius::test::scoped_sirius_setting off{*con, "delim_join_as_cte", false};
    // A statement prepared under the default is planned again when it runs.
    CHECK(query_error("EXECUTE under_default;").find(expected) != std::string::npos);

    // Statements Sirius does not run are unaffected: DML, and any query once the GPU is off.
    run_ok("CREATE TABLE cte_copy AS " + correlated + ";");
    run_ok("SET gpu_execution = false;");
    run_ok(correlated + ";");
    run_ok("SET gpu_execution = true;");
  }
  // The session recovers once the setting is back.
  compare_gpu_vs_cpu(plain);
  compare_gpu_vs_cpu(correlated);
  run_ok("EXECUTE under_default;");
}
