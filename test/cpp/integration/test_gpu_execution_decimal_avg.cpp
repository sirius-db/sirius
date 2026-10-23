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

/**
 * @file test_gpu_execution_decimal_avg.cpp
 * @brief Verifies that AVG over DECIMAL rounds to the same DOUBLE as DuckDB.
 *
 * DuckDB divides the unscaled integer sum by `count * 10^scale`. Each group below holds `g` copies
 * of one value, so every group's average is that value. Grouping on the averages must therefore
 * give one row, which also catches a last-bit difference between groups that the tolerant result
 * comparison would accept.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

using DecimalAvgFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Group g holds g rows of @p value cast to @p type, for g in 1..20.
void create_equal_value_groups(DecimalAvgFixture& fx,
                               std::string const& type,
                               std::string const& value)
{
  fx.run_ok("CREATE TABLE eq AS SELECT g::INTEGER g, CAST(" + value + " AS " + type +
            ") v FROM range(1, 21) a(g), range(20) b(i) WHERE i < g;");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(DecimalAvgFixture,
                 "decimal avg - equal grouped averages over DECIMAL(18,4) are one value",
                 "[integration][gpu_execution][aggregate][decimal_avg]")
{
  create_equal_value_groups(*this, "DECIMAL(18,4)", "626554.3479");
  compare_gpu_vs_cpu("SELECT a FROM (SELECT g, avg(v) a FROM eq GROUP BY g) GROUP BY a;");
}

TEST_CASE_METHOD(DecimalAvgFixture,
                 "decimal avg - equal grouped averages over DECIMAL(9,3) are one value",
                 "[integration][gpu_execution][aggregate][decimal_avg]")
{
  create_equal_value_groups(*this, "DECIMAL(9,3)", "123456.789");
  compare_gpu_vs_cpu("SELECT a FROM (SELECT g, avg(v) a FROM eq GROUP BY g) GROUP BY a;");
}

TEST_CASE_METHOD(DecimalAvgFixture,
                 "decimal avg - grouped AVG over DECIMAL keeps NULLs and NULL-only groups",
                 "[integration][gpu_execution][aggregate][decimal_avg]")
{
  run_ok(
    "CREATE TABLE eqn AS SELECT (i % 4)::INTEGER g, "
    "CASE WHEN i % 4 = 3 OR i % 5 = 0 THEN NULL ELSE 626554.3479 END::DECIMAL(18,4) v "
    "FROM range(40) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT a FROM (SELECT g, avg(v) a FROM eqn GROUP BY g) GROUP BY a;");
}
