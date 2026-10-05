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
 * @file test_gpu_execution_decimal_sum_overflow.cpp
 * @brief Verifies that SUM and AVG over DECIMAL columns do not overflow the input width.
 *
 * DECIMAL(7,2) is stored as a 32-bit decimal and DECIMAL(18,2) as a 64-bit decimal. The sums
 * below exceed those widths, so they are only correct when computed in a wider type.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <set>

using DecimalSumFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Rows per group: 20000 x about 99999.99 is about 2 x 10^11 cents, beyond the 2^31 - 1 cents a
/// DECIMAL(7,2) sum can hold. For DECIMAL(18,2), 20 x about 9 x 10^15 is about 1.8 x 10^19
/// cents, beyond 2^63 - 1.
void create_decimal_tables(DecimalSumFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE d32 AS SELECT (i % 3)::INTEGER g, (99999.99 - (i % 7))::DECIMAL(7,2) v "
    "FROM range(60000) r(i);");
  fx.run_ok(
    "CREATE TABLE d64 AS SELECT (i % 2)::INTEGER g, "
    "(9000000000000000.00 - i)::DECIMAL(18,2) v FROM range(40) r(i);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - grouped SUM and AVG over DECIMAL(7,2)",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_decimal_tables(*this);
  compare_gpu_vs_cpu_approx(
    "SELECT g, sum(v) s, avg(v) a, count(v) c FROM d32 GROUP BY g;", std::set<size_t>{2}, 1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - grouped SUM next to MIN and MAX of the same column",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_decimal_tables(*this);
  compare_gpu_vs_cpu("SELECT g, min(v) mn, sum(v) s, max(v) mx FROM d32 GROUP BY g;");
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - ungrouped AVG over DECIMAL(7,2)",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_decimal_tables(*this);
  compare_gpu_vs_cpu_approx("SELECT avg(v) a, sum(v) s FROM d32;", std::set<size_t>{0}, 1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - grouped SUM and AVG over DECIMAL(18,2)",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_decimal_tables(*this);
  compare_gpu_vs_cpu_approx(
    "SELECT g, sum(v) s, avg(v) a FROM d64 GROUP BY g;", std::set<size_t>{2}, 1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - ungrouped AVG over DECIMAL(18,2)",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_decimal_tables(*this);
  compare_gpu_vs_cpu_approx("SELECT avg(v) a FROM d64;", std::set<size_t>{0}, 1e-12);
}
