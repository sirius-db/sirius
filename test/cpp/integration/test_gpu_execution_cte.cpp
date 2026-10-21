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

// GPU-vs-CPU correctness for materialized CTEs. The plan-shape side of the unread-CTE case is
// pinned in test/cpp/pipeline/test_cte_build_pipelines.cpp.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

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

// Statistics propagation proves the CTE empty and folds its only reference into an empty result,
// leaving the materialization unread. Planning it used to strand the consumer: the query hung.
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
