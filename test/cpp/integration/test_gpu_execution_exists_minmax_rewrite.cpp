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

#include <optional>
#include <string>

namespace {

// Sibling correlated EXISTS / NOT EXISTS with an equality and an inequality over the same table,
// which Sirius merges into one scan and a MIN/MAX aggregate per key. `w` carries NULLs, some keys
// have a single row and some are NULL, so every branch of the semantics is exercised.
class ExistsMinmaxRewriteFixture : public sirius::test::GpuExecutionFixture {
 public:
  ExistsMinmaxRewriteFixture()
  {
    enable_guard.emplace(*con, "enable_exists_minmax_rewrite", true);
    run_ok(
      "CREATE TABLE fact (k BIGINT, v INTEGER, w INTEGER, d DATE, amount DECIMAL(12, 2), "
      "late BOOLEAN);");
    run_ok(
      "INSERT INTO fact SELECT (range * 7) % 61, (range * 13) % 1000 - 500, "
      "CASE WHEN range % 4 = 0 THEN NULL ELSE (range * 31) % 97 END, "
      "DATE '2020-01-01' + ((range * 11) % 30)::INTEGER, "
      "((range * 17) % 10000)::DECIMAL(12, 2) / 100, range % 3 = 0 FROM range(4000);");
    // Keys with exactly one row: EXISTS another row is false, NOT EXISTS true.
    run_ok(
      "INSERT INTO fact VALUES (1000, 1, 1, DATE '2020-01-01', 1.00, true), "
      "(1001, 2, NULL, DATE '2020-01-02', 2.00, true);");
    // NULL keys, which only null-safe key comparisons match.
    run_ok(
      "INSERT INTO fact VALUES (NULL, 3, NULL, DATE '2020-01-03', 3.00, true), "
      "(NULL, 4, 5, DATE '2020-01-04', 4.00, true), (NULL, 4, 6, DATE '2020-01-04', 4.00, false);");
    run_ok("CHECKPOINT;");
  }

  // Rewritten against the CPU, and the unrewritten GPU plan against the CPU, under both of
  // DuckDB's representations of the delim join.
  void compare_rewritten_and_not(std::string const& query)
  {
    for (bool const as_cte : {false, true}) {
      CAPTURE(as_cte);
      sirius::test::scoped_sirius_setting representation{*con, "delim_join_as_cte", as_cte};
      compare_gpu_vs_cpu(query);
      sirius::test::scoped_sirius_setting off{*con, "enable_exists_minmax_rewrite", false};
      compare_gpu_vs_cpu(query);
    }
  }

  std::optional<sirius::test::scoped_sirius_setting> enable_guard;
};

}  // namespace

TEST_CASE_METHOD(ExistsMinmaxRewriteFixture,
                 "gpu_execution EXISTS and NOT EXISTS together, in the shape of TPC-H q21",
                 "[integration][gpu_execution][exists_rewrite]")
{
  compare_rewritten_and_not(
    "SELECT f.k, count(*) AS n FROM fact f WHERE f.late "
    "AND EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v) "
    "AND NOT EXISTS (SELECT 1 FROM fact g WHERE g.k = f.k AND g.v <> f.v AND g.late) "
    "GROUP BY f.k");
}

TEST_CASE_METHOD(ExistsMinmaxRewriteFixture,
                 "gpu_execution sibling subqueries over nullable, temporal and decimal values",
                 "[integration][gpu_execution][exists_rewrite]")
{
  for (char const* column : {"w", "d", "amount"}) {
    CAPTURE(column);
    for (char const* value : {"<>", "IS DISTINCT FROM"}) {
      CAPTURE(value);
      for (char const* key : {"=", "IS NOT DISTINCT FROM"}) {
        CAPTURE(key);
        auto const differs = std::string(" g.") + column + " " + value + " f." + column;
        compare_rewritten_and_not(std::string("SELECT f.k, f.") + column +
                                  " FROM fact f WHERE f.v > 0 AND EXISTS (SELECT 1 FROM fact g "
                                  "WHERE g.k = f.k AND" +
                                  differs + ") AND NOT EXISTS (SELECT 1 FROM fact g WHERE g.k " +
                                  key + " f.k AND" + differs + " AND g.w > 40)");
      }
    }
  }
}
