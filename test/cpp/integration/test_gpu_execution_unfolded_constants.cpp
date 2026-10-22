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

// DuckDB's statistics propagation reduces coalesce(<non-null constant>, x) to the constant after
// constant folding has run, so the GPU receives calls such as month(DATE '2020-02-01') whose
// operands are all constants. These tests check that such calls run on the GPU and match the CPU.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <algorithm>
#include <cctype>
#include <string>

namespace {

class UnfoldedConstantsFixture : public sirius::test::GpuExecutionFixture {
 public:
  UnfoldedConstantsFixture()
  {
    run_ok(
      "CREATE TABLE t AS SELECT i::INTEGER AS i,"
      " CASE WHEN i % 7 = 0 THEN NULL ELSE (i % 100)::INTEGER END AS x,"
      " CASE WHEN i % 5 = 0 THEN NULL ELSE 'v' || (i % 13)::VARCHAR || 'a' END AS s,"
      " CASE WHEN i % 11 = 0 THEN NULL"
      " ELSE TIMESTAMP '2001-02-03 04:05:06' + to_seconds(i * 3607) END AS ts,"
      " CASE WHEN i % 3 = 0 THEN NULL ELSE (i * 0.5)::DOUBLE END AS f,"
      " CASE WHEN i % 4 = 0 THEN NULL ELSE i % 2 = 0 END AS b,"
      " CASE WHEN i % 9 = 0 THEN NULL ELSE (i * 0.25)::DECIMAL(10, 2) END AS dd,"
      " NULL::DATE AS null_date"
      " FROM range(2000) r(i);");
    run_ok("CHECKPOINT;");
  }

  /// Requires that DuckDB's optimized plan no longer contains the coalesce, so @p query reaches the
  /// GPU with an all-constant call.
  void require_coalesce_removed(std::string const& query)
  {
    auto result = con->Query("EXPLAIN " + query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    auto plan = result->ToString();
    std::transform(plan.begin(), plan.end(), plan.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    REQUIRE(plan.find("coalesce") == std::string::npos);
  }

  void compare_all_strategies(std::string const& query)
  {
    CAPTURE(query);
    require_coalesce_removed(query);
    for (auto const* strategy : {"materialize", "ast_interpret", "ast_jit"}) {
      CAPTURE(strategy);
      run_ok(std::string("SET expression_evaluator_strategy = '") + strategy + "';");
      compare_gpu_vs_cpu(query);
    }
    run_ok("SET expression_evaluator_strategy = 'ast_interpret';");
  }
};

}  // namespace

TEST_CASE_METHOD(UnfoldedConstantsFixture,
                 "date part of a constant left by coalesce simplification",
                 "[integration][gpu_execution][unfolded_constants]")
{
  // Fuzz finding 010-gpu_error-5f584789. Without `i` the scan projects no columns.
  compare_all_strategies(
    "SELECT month(coalesce(DATE '2020-02-01', CASE WHEN s LIKE '%a%' THEN DATE '2000-12-05'"
    " ELSE DATE '2019-01-01' END)) FROM t");
  compare_all_strategies(
    "SELECT i, month(coalesce(DATE '2020-02-01', CASE WHEN s LIKE '%a%' THEN DATE '2000-12-05'"
    " ELSE DATE '2019-01-01' END)) FROM t");
  compare_all_strategies("SELECT i, year(coalesce(DATE '2020-02-01', ts::DATE)) FROM t");
  compare_all_strategies(
    "SELECT i, hour(coalesce(TIMESTAMP '2020-02-01 13:14:15', ts)),"
    " day(coalesce(TIMESTAMP '2020-02-01 13:14:15', ts)) FROM t");
  compare_all_strategies("SELECT i, month(coalesce(null_date, NULL::DATE)) FROM t");
  compare_all_strategies(
    "SELECT i, date_trunc('hour', coalesce(TIMESTAMP '2020-02-01 13:14:15', ts)) FROM t");
}

TEST_CASE_METHOD(UnfoldedConstantsFixture,
                 "string functions of a constant left by coalesce simplification",
                 "[integration][gpu_execution][unfolded_constants]")
{
  compare_all_strategies("SELECT i, strlen(coalesce('apple', s)) FROM t");
  compare_all_strategies("SELECT i, length(coalesce('apple', s)) FROM t");
  compare_all_strategies(
    "SELECT i, upper(coalesce('apple', s)), lower(coalesce('APPLE', s)) FROM t");
  compare_all_strategies("SELECT i, substring(coalesce('apple', s), 2, 3) FROM t");
  // DuckDB rewrites `LIKE '%lit%'` to contains(), so these patterns keep the LIKE handler.
  compare_all_strategies(
    "SELECT i, coalesce('apple', s) LIKE '%pp%le%', coalesce('apple', s) NOT LIKE '%pp%x%',"
    " coalesce('apple', s) LIKE 'a_ple' FROM t");
  compare_all_strategies(
    "SELECT i, contains(coalesce('apple', s), 'pl'), prefix(coalesce('apple', s), 'ap'),"
    " suffix(coalesce('apple', s), 'le') FROM t");
  compare_all_strategies("SELECT i, regexp_replace(coalesce('apple', s), 'p', 'x') FROM t");
}

TEST_CASE_METHOD(UnfoldedConstantsFixture,
                 "arithmetic and IN over constants left by coalesce simplification",
                 "[integration][gpu_execution][unfolded_constants]")
{
  compare_all_strategies("SELECT i, coalesce(2, x) + 1 FROM t");
  compare_all_strategies("SELECT i, (coalesce(2, x) + 1) * coalesce(3, x) FROM t");
  compare_all_strategies("SELECT i, coalesce(2.5::DOUBLE, f) / 2 FROM t");
  compare_all_strategies("SELECT i, coalesce(2, x) IN (1, 2), coalesce(2, x) NOT IN (2, 3) FROM t");
  compare_all_strategies("SELECT i, coalesce('apple', s) IN ('apple', 'pear') FROM t");
  compare_all_strategies("SELECT i, coalesce(2, x) IN (x, i) FROM t");
  // DECIMAL arithmetic has no AST form, so every strategy takes the materialize path.
  compare_all_strategies("SELECT i, coalesce(2.5, dd) + 1 FROM t");
}

TEST_CASE_METHOD(UnfoldedConstantsFixture,
                 "predicates over constants left by coalesce simplification",
                 "[integration][gpu_execution][unfolded_constants]")
{
  compare_all_strategies(
    "SELECT i, coalesce('apple', s) = 'apple', coalesce('apple', s) < 'pear' FROM t");
  compare_all_strategies(
    "SELECT i, coalesce('apple', s) IS DISTINCT FROM 'pear',"
    " coalesce(2, x) IS NOT DISTINCT FROM 2 FROM t");
  compare_all_strategies("SELECT i, coalesce(2, x) BETWEEN 1 AND 3 FROM t");
  compare_all_strategies("SELECT i, NOT coalesce(true, b) FROM t");
  compare_all_strategies("SELECT i, coalesce(2, x) IS NULL, coalesce(2, x) IS NOT NULL FROM t");
}
