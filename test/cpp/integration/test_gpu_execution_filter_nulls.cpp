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

// GPU-vs-CPU correctness for NULL semantics in filters and scalar projections
// (issue #1095): three-valued predicate logic, IS [NOT] NULL, IS [NOT] DISTINCT
// FROM, BETWEEN / IN with NULLs, and NULL propagation through COALESCE, NULLIF,
// CASE, CAST, arithmetic, string, and date expressions.
//
// Every query goes through the shared file-backed GpuExecutionFixture, which
// runs it once on the GPU (asserting a real GPU execution with no fallback) and
// once on DuckDB CPU, then compares the results.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>
#include <vector>

namespace {

// One nullable column per data type under test, plus a never-null `id` so every
// row stays individually identifiable in the comparison.
class NullDataFixture : public sirius::test::GpuExecutionFixture {
 public:
  NullDataFixture()
  {
    run_ok(
      "CREATE TABLE nt ("
      "  id  INTEGER,"
      "  i   INTEGER,"
      "  b   BIGINT,"
      "  dec DECIMAL(10,2),"
      "  dbl DOUBLE,"
      "  s   VARCHAR,"
      "  dt  DATE);");

    // Mix of: fully-populated rows, an all-NULL-payload row, rows with a NULL in
    // some columns, duplicates (for equality/DISTINCT), and edge values
    // (zero, negative, empty string).
    run_ok(
      "INSERT INTO nt VALUES "
      "(1, 10,  100,  10.50,  1.5, 'apple',  DATE '2021-01-15'),"
      "(2, NULL, NULL, NULL,  NULL, NULL,    NULL),"
      "(3, 10,  NULL, 20.00,  NULL, 'apple', DATE '2022-06-30'),"
      "(4, NULL, 100, NULL,   2.5,  NULL,    DATE '2021-01-15'),"
      "(5, 20,  200,  30.25,  3.5,  'banana', NULL),"
      "(6, 0,  -50,  -5.00,   0.0,  '',      DATE '2020-12-31'),"
      "(7, 10,  100,  10.50,  1.5,  'apple', DATE '2021-01-15'),"
      "(8, NULL, NULL, 40.00, 4.5,  'cherry', DATE '2023-03-03');");
    run_ok("CHECKPOINT;");
  }
};

const std::vector<std::string> kNullableColumns = {"i", "b", "dec", "dbl", "s", "dt"};

}  // namespace

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution IS NULL / IS NOT NULL per column type",
                 "[integration][gpu_execution][filter][nulls]")
{
  for (const auto& col : kNullableColumns) {
    DYNAMIC_SECTION(col << " IS NULL")
    {
      compare_gpu_vs_cpu("SELECT id FROM nt WHERE " + col + " IS NULL");
    }
    DYNAMIC_SECTION(col << " IS NOT NULL")
    {
      compare_gpu_vs_cpu("SELECT id FROM nt WHERE " + col + " IS NOT NULL");
    }
  }
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution comparison predicates filter NULL results (three-valued)",
                 "[integration][gpu_execution][filter][nulls]")
{
  // A comparison against NULL is UNKNOWN, so those rows must be excluded by WHERE
  // for every operator.
  const std::vector<std::string> ops = {"=", "<>", "<", "<=", ">", ">="};
  for (const auto& op : ops) {
    DYNAMIC_SECTION("i " << op << " 10")
    {
      compare_gpu_vs_cpu("SELECT id FROM nt WHERE i " + op + " 10");
    }
    DYNAMIC_SECTION("dec " << op << " 10.50")
    {
      compare_gpu_vs_cpu("SELECT id FROM nt WHERE dec " + op + " 10.50");
    }
  }
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution IS [NOT] DISTINCT FROM treats NULL as a value",
                 "[integration][gpu_execution][filter][nulls]")
{
  // vs a constant
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i IS NOT DISTINCT FROM 10");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i IS DISTINCT FROM 10");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i IS NOT DISTINCT FROM NULL");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE s IS NOT DISTINCT FROM NULL");
  // column vs column (NULL vs NULL must be "not distinct")
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i IS NOT DISTINCT FROM b");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i IS DISTINCT FROM b");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution three-valued AND / NOT",
                 "[integration][gpu_execution][filter][nulls]")
{
  // Project the boolean result rather than filtering on it: a WHERE clause
  // collapses FALSE and UNKNOWN together, so it cannot tell whether TRUE AND NULL
  // yields NULL (correct) or FALSE. These rows span TRUE / FALSE / NULL outcomes.
  compare_gpu_vs_cpu("SELECT id, (i = 10 AND b = 100) AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, (NOT (i = 10)) AS r FROM nt");
}

// SQL three-valued logic: TRUE OR UNKNOWN = TRUE, so a row where one branch is
// TRUE and the other is NULL is kept.
TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution three-valued OR, TRUE-OR-NULL branch",
                 "[integration][gpu_execution][filter][nulls]")
{
  // Row (i=10, b=NULL) satisfies `i = 10`, so it is kept even though `b = 200` is NULL.
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i = 10 OR b = 200");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution three-valued OR, IS-NULL-OR-match branch",
                 "[integration][gpu_execution][filter][nulls]")
{
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE (i IS NULL) OR (b = 100)");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution BETWEEN and IN with NULLs",
                 "[integration][gpu_execution][filter][nulls]")
{
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i BETWEEN 0 AND 15");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE dec BETWEEN 0.00 AND 25.00");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i IN (10, 20)");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE i NOT IN (10, 20)");
  compare_gpu_vs_cpu("SELECT id FROM nt WHERE s IN ('apple', 'cherry')");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution COALESCE / NULLIF / CASE projections propagate NULL",
                 "[integration][gpu_execution][projection][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, COALESCE(i, -1) AS c FROM nt");
  compare_gpu_vs_cpu("SELECT id, COALESCE(i, b, 0) AS c FROM nt");
  compare_gpu_vs_cpu("SELECT id, COALESCE(s, 'none') AS c FROM nt");
  compare_gpu_vs_cpu("SELECT id, NULLIF(i, 10) AS c FROM nt");
  compare_gpu_vs_cpu(
    "SELECT id, CASE WHEN i IS NULL THEN 'na' WHEN i = 10 THEN 'ten' ELSE 'other' END AS c "
    "FROM nt");
  compare_gpu_vs_cpu("SELECT id, CASE WHEN b > 0 THEN b ELSE NULL END AS c FROM nt");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution arithmetic propagates NULL",
                 "[integration][gpu_execution][projection][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, i + b AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, i - 5 AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, dec * 2 AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, dbl / 2 AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, i + b + dec AS r FROM nt");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution CAST propagates NULL",
                 "[integration][gpu_execution][projection][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, CAST(i AS BIGINT) AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, CAST(dec AS DOUBLE) AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, CAST(b AS DOUBLE) AS r FROM nt");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution string functions propagate NULL",
                 "[integration][gpu_execution][projection][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, length(s) AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, substring(s, 1, 2) AS r FROM nt");
}

// KNOWN GPU DIVERGENCE (issue #1218):
// DuckDB's concat() ignores NULL arguments (concat(NULL, '_x') = '_x'), but
// Sirius's GPU concat propagates NULL (returns NULL) -- it behaves like the `||`
// operator instead. Tagged [!shouldfail] so it documents the bug without failing
// CI; remove the tag once GPU concat matches DuckDB's NULL-as-empty semantics.
TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution concat NULL-handling [known divergence]",
                 "[integration][gpu_execution][projection][nulls][!shouldfail]")
{
  compare_gpu_vs_cpu("SELECT id, concat(s, '_x') AS r FROM nt");
}

TEST_CASE_METHOD(NullDataFixture,
                 "gpu_execution date functions propagate NULL",
                 "[integration][gpu_execution][projection][nulls]")
{
  compare_gpu_vs_cpu("SELECT id, year(dt) AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, month(dt) AS r FROM nt");
  compare_gpu_vs_cpu("SELECT id, day(dt) AS r FROM nt");
}
