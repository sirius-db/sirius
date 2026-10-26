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
#include <string>

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

// A SUM only widens when rows * max|value| could exceed the storage width, so the cases below sit
// on both sides of that bound. Each uses a single group so a missing widening shows as a wrapped
// sum. DECIMAL(7,2) holds at most 9999999 unscaled against a 32-bit limit of 2147483647, so 214
// rows still fit (2139999786) and 216 do not.

TEST_CASE_METHOD(
  DecimalSumFixture,
  "decimal sum overflow - DECIMAL(7,2) just inside and just outside the 32-bit bound",
  "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE fits AS SELECT 0::INTEGER g, 99999.99::DECIMAL(7,2) v "
    "FROM range(214) r(i);");
  run_ok(
    "CREATE TABLE spills AS SELECT 0::INTEGER g, 99999.99::DECIMAL(7,2) v "
    "FROM range(216) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT sum(v) s FROM (SELECT v FROM fits);");
  compare_gpu_vs_cpu("SELECT g, sum(v) s, count(v) c, min(v) mn FROM fits GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu(
    "SELECT g, sum(v) s, count(v) c, min(v) mn FROM spills GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu("SELECT sum(v) s FROM spills;");
}

TEST_CASE_METHOD(
  DecimalSumFixture,
  "decimal sum overflow - DECIMAL(18,2) just inside and just outside the 64-bit bound",
  "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  // 9999999999999999.99 is 999999999999999999 unscaled: 9 rows reach 8.99e18 < 2^63 - 1 and 10
  // rows reach 9.99e18, which wraps.
  run_ok(
    "CREATE TABLE fits64 AS SELECT 0::INTEGER g, 9999999999999999.99::DECIMAL(18,2) v "
    "FROM range(9) r(i);");
  run_ok(
    "CREATE TABLE spills64 AS SELECT 0::INTEGER g, 9999999999999999.99::DECIMAL(18,2) v "
    "FROM range(10) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT g, sum(v) s FROM fits64 GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu("SELECT g, sum(v) s FROM spills64 GROUP BY g ORDER BY g;");
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - large negative values widen too",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE neg AS SELECT 0::INTEGER g, -99999.99::DECIMAL(7,2) v "
    "FROM range(300) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT g, sum(v) s FROM neg GROUP BY g ORDER BY g;");
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - narrow and wide sums in one aggregate",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  // narrow stays at the input width while wide needs the next one, in the same request list.
  run_ok(
    "CREATE TABLE mixed AS SELECT (i % 3)::INTEGER g, (i % 100)::DECIMAL(15,2) narrow, "
    "9999999999999999.99::DECIMAL(18,2) wide FROM range(30) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu(
    "SELECT g, sum(narrow) sn, min(narrow) mn, sum(wide) sw, max(wide) mw, "
    "sum(narrow * 2) se FROM mixed GROUP BY g ORDER BY g;");
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - TPC-H-like DECIMAL(15,2) over many rows stays exact",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE narrow AS SELECT (i % 4)::INTEGER g, ((i * 7919) % 10494950)::DECIMAL(15,2) v "
    "FROM range(300000) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu(
    "SELECT g, sum(v) s, count(v) c, min(v) mn, max(v) mx FROM narrow "
    "GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu_approx(
    "SELECT g, sum(v) s, avg(v) a FROM narrow GROUP BY g ORDER BY g;", std::set<size_t>{2}, 1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - nulls, all-null groups and empty input",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE nulls AS SELECT (i % 3)::INTEGER g, "
    "CASE WHEN i % 3 = 2 THEN NULL ELSE 99999.99 END::DECIMAL(7,2) v FROM range(600) r(i);");
  run_ok(
    "CREATE TABLE allnull AS SELECT (i % 2)::INTEGER g, NULL::DECIMAL(7,2) v "
    "FROM range(100) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT g, sum(v) s, count(v) c FROM nulls GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu("SELECT g, sum(v) s FROM allnull GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu("SELECT g, sum(v) s FROM nulls WHERE g > 100 GROUP BY g;");
}

// The ungrouped path applies the same proof: widen only if rows * max|value| could exceed the
// storage width, otherwise reduce at the input width and widen the one-row result. AVG's count
// and the SUM partial must keep their types either way.

TEST_CASE_METHOD(
  DecimalSumFixture,
  "decimal sum overflow - ungrouped SUM and AVG DECIMAL(7,2) around the 32-bit bound",
  "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok("CREATE TABLE fits AS SELECT 99999.99::DECIMAL(7,2) v FROM range(214) r(i);");
  run_ok("CREATE TABLE spills AS SELECT 99999.99::DECIMAL(7,2) v FROM range(216) r(i);");
  run_ok("CREATE TABLE neg AS SELECT -99999.99::DECIMAL(7,2) v FROM range(300) r(i);");
  run_ok("CHECKPOINT;");
  for (char const* table : {"fits", "spills", "neg"}) {
    compare_gpu_vs_cpu_approx(
      std::string("SELECT sum(v) s, avg(v) a, count(v) c FROM ") + table + ";",
      std::set<size_t>{1},
      1e-12);
  }
}

TEST_CASE_METHOD(
  DecimalSumFixture,
  "decimal sum overflow - ungrouped SUM and AVG DECIMAL(18,2) around the 64-bit bound",
  "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok("CREATE TABLE fits64 AS SELECT 9999999999999999.99::DECIMAL(18,2) v FROM range(9) r(i);");
  run_ok(
    "CREATE TABLE spills64 AS SELECT 9999999999999999.99::DECIMAL(18,2) v FROM range(10) r(i);");
  run_ok("CREATE TABLE neg64 AS SELECT -9999999999999999.99::DECIMAL(18,2) v FROM range(12) r(i);");
  run_ok("CHECKPOINT;");
  for (char const* table : {"fits64", "spills64", "neg64"}) {
    compare_gpu_vs_cpu_approx(
      std::string("SELECT sum(v) s, avg(v) a FROM ") + table + ";", std::set<size_t>{1}, 1e-12);
  }
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - ungrouped narrow and wide sums in one aggregate",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE mixed AS SELECT (i % 100)::DECIMAL(15,2) narrow, "
    "9999999999999999.99::DECIMAL(18,2) wide, 99999.99::DECIMAL(7,2) small "
    "FROM range(300) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu_approx(
    "SELECT sum(narrow) sn, avg(narrow) an, sum(wide) sw, avg(wide) aw, "
    "sum(small) ss, avg(small) asm FROM mixed;",
    std::set<size_t>{1, 3, 5},
    1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - ungrouped TPC-H-like DECIMAL(15,2) stays exact",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE narrow AS SELECT ((i * 7919) % 10494950)::DECIMAL(15,2) v "
    "FROM range(300000) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu_approx(
    "SELECT sum(v) s, avg(v) a, min(v) mn, max(v) mx FROM narrow;", std::set<size_t>{1}, 1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - ungrouped nulls, all-null column and empty input",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE nulls AS SELECT (i % 3)::INTEGER g, "
    "CASE WHEN i % 3 = 2 THEN NULL ELSE 99999.99 END::DECIMAL(7,2) v FROM range(600) r(i);");
  run_ok("CREATE TABLE allnull AS SELECT NULL::DECIMAL(7,2) v FROM range(100) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu_approx(
    "SELECT sum(v) s, avg(v) a, count(v) c FROM nulls;", std::set<size_t>{1}, 1e-12);
  compare_gpu_vs_cpu_approx("SELECT sum(v) s, avg(v) a FROM allnull;", std::set<size_t>{1}, 1e-12);
  compare_gpu_vs_cpu_approx(
    "SELECT sum(v) s, avg(v) a FROM nulls WHERE g > 100;", std::set<size_t>{1}, 1e-12);
}

namespace {

/// Values whose statistics prove no batch can overflow: q in [1.00, 50.00] and p in
/// [0.00, 1008.99] as DECIMAL(15,2), f in [0.00, 7.50] as DECIMAL(9,2). The planner's bound
/// decides these without the per-batch scan.
void create_small_decimal_tables(DecimalSumFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE small AS SELECT (i % 4)::INTEGER g, ((i % 50) + 1)::DECIMAL(15,2) q, "
    "((i % 1000) * 1.01)::DECIMAL(15,2) p FROM range(20000) r(i);");
  fx.run_ok(
    "CREATE TABLE groups AS SELECT i::INTEGER g, (i * 2.5)::DECIMAL(9,2) f FROM range(4) r(i);");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - statistics-bounded grouped SUM and AVG",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_small_decimal_tables(*this);
  compare_gpu_vs_cpu_approx(
    "SELECT g, sum(q) s, avg(q) a, sum(p) t, count(*) c FROM small GROUP BY g;",
    std::set<size_t>{1, 2, 3},
    1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - statistics-bounded ungrouped SUM and AVG",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_small_decimal_tables(*this);
  compare_gpu_vs_cpu_approx("SELECT sum(q) s, avg(p) a, sum(p) t, count(q) c FROM small;",
                            std::set<size_t>{0, 1, 2},
                            1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - statistics bound traced through a join to both sides",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_small_decimal_tables(*this);
  compare_gpu_vs_cpu_approx(
    "SELECT s.g, sum(s.q) sq, sum(r.f) sf FROM small s JOIN groups r ON s.g = r.g "
    "WHERE s.p >= 0 GROUP BY s.g;",
    std::set<size_t>{1, 2},
    1e-12);
}

TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - unbounded computed input measured beside a bounded column",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  create_small_decimal_tables(*this);
  // q + q is DECIMAL(16,2), a widening candidate without a statistics bound, so the batch is
  // measured for it while q itself is decided from its bound.
  compare_gpu_vs_cpu_approx(
    "SELECT g, sum(q + q) s2, sum(q) s FROM small GROUP BY g;", std::set<size_t>{1, 2}, 1e-12);
  compare_gpu_vs_cpu_approx(
    "SELECT sum(q + q) s2, avg(q) a FROM small;", std::set<size_t>{0, 1}, 1e-12);
}

// DuckDB v2 types DECIMAL(18,4) - DECIMAL(18,4) as DECIMAL(19,4), which the GPU holds as
// DECIMAL128. A grouped SUM over it runs over DECIMAL64 when rows * max|value| fits 64 bits and
// stays 128-bit otherwise; both must match the CPU exactly.
TEST_CASE_METHOD(DecimalSumFixture,
                 "decimal sum overflow - DECIMAL128 sums narrow only when 64 bits suffice",
                 "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  // The TPC-H Q9 shape: price * (1 - discount) - cost * quantity, small values, many rows.
  run_ok(
    "CREATE TABLE profit AS SELECT (i % 5)::INTEGER g, "
    "((i * 7919) % 10494950)::DECIMAL(15,2) price, ((i % 11) / 100)::DECIMAL(15,2) disc, "
    "((i * 31) % 100000)::DECIMAL(15,2) cost, ((i % 50) + 1)::DECIMAL(15,2) qty "
    "FROM range(200000) r(i);");
  // Sirius cannot scan DECIMAL128 table columns, so the 128-bit inputs below are computed. v - w
  // is DECIMAL(19,4) with every value 2 * (10^18 - 1) unscaled, which fits 64 bits. Four such
  // values still sum within 2^63 - 1 and narrow; eight in one group would wrap, so that batch
  // stays 128-bit.
  run_ok(
    "CREATE TABLE near4 AS SELECT 0::INTEGER g, 99999999999999.9999::DECIMAL(18,4) v, "
    "-99999999999999.9999::DECIMAL(18,4) w FROM range(4) r(i);");
  run_ok(
    "CREATE TABLE near8 AS SELECT 0::INTEGER g, 99999999999999.9999::DECIMAL(18,4) v, "
    "-99999999999999.9999::DECIMAL(18,4) w FROM range(8) r(i);");
  // v::DECIMAL(38,0) * scale is DECIMAL(38,0) with values beyond 64 bits.
  run_ok(
    "CREATE TABLE beyond AS SELECT (i % 2)::INTEGER g, (100000000000000000 + i)::DECIMAL(18,0) v, "
    "1000000000000::DECIMAL(18,0) scale FROM range(8) r(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu(
    "SELECT g, sum(price * (1 - disc) - cost * qty) s FROM profit GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu("SELECT g, sum(v - w) s FROM near4 GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu("SELECT g, sum(v - w) s, min(v - w) mn FROM near8 GROUP BY g ORDER BY g;");
  compare_gpu_vs_cpu(
    "SELECT g, sum(v::DECIMAL(38,0) * scale) s, max(v::DECIMAL(38,0) * scale) mx FROM beyond "
    "GROUP BY g ORDER BY g;");
}

// The ungrouped aggregate narrows DECIMAL128 SUM and AVG inputs the same way, per batch, and casts
// each one-row partial back to DECIMAL128 for the merge.
TEST_CASE_METHOD(
  DecimalSumFixture,
  "decimal sum overflow - ungrouped DECIMAL128 sums narrow only when 64 bits suffice",
  "[integration][gpu_execution][aggregate][decimal_sum_overflow]")
{
  run_ok(
    "CREATE TABLE profit AS SELECT ((i * 7919) % 10494950)::DECIMAL(15,2) price, "
    "((i % 11) / 100)::DECIMAL(15,2) disc, ((i * 31) % 100000)::DECIMAL(15,2) cost, "
    "((i % 50) + 1)::DECIMAL(15,2) qty FROM range(200000) r(i);");
  // price is NULL on every seventh row; the null-only variant has no valid value at all.
  run_ok(
    "CREATE TABLE profit_nulls AS SELECT CASE WHEN i % 7 = 0 THEN NULL "
    "ELSE ((i * 7919) % 10494950)::DECIMAL(15,2) END price, "
    "((i % 11) / 100)::DECIMAL(15,2) disc, ((i * 31) % 100000)::DECIMAL(15,2) cost, "
    "((i % 50) + 1)::DECIMAL(15,2) qty FROM range(200000) r(i);");
  // v - w is DECIMAL(19,4) with every value 2 * (10^18 - 1) unscaled: four still sum within
  // 2^63 - 1, eight would wrap.
  run_ok(
    "CREATE TABLE near4 AS SELECT 99999999999999.9999::DECIMAL(18,4) v, "
    "-99999999999999.9999::DECIMAL(18,4) w FROM range(4) r(i);");
  run_ok(
    "CREATE TABLE near8 AS SELECT 99999999999999.9999::DECIMAL(18,4) v, "
    "-99999999999999.9999::DECIMAL(18,4) w FROM range(8) r(i);");
  run_ok(
    "CREATE TABLE beyond AS SELECT (100000000000000000 + i)::DECIMAL(18,0) v, "
    "1000000000000::DECIMAL(18,0) scale FROM range(8) r(i);");
  run_ok("CHECKPOINT;");

  compare_gpu_vs_cpu("SELECT sum(price * (1 - disc) - cost * qty) s FROM profit;");
  compare_gpu_vs_cpu_approx(
    "SELECT avg(price * (1 - disc) - cost * qty) a, sum(price * (1 - disc) - cost * qty) s "
    "FROM profit;",
    std::set<size_t>{0},
    1e-12);
  compare_gpu_vs_cpu(
    "SELECT sum(price * (1 - disc) - cost * qty) s, count(price) c FROM profit_nulls;");
  compare_gpu_vs_cpu(
    "SELECT sum(price * (1 - disc) - cost * qty) s FROM profit_nulls WHERE price IS NULL;");
  compare_gpu_vs_cpu("SELECT sum(price * (1 - disc) - cost * qty) s FROM profit WHERE qty < 0;");
  compare_gpu_vs_cpu("SELECT sum(v - w) s FROM near4;");
  compare_gpu_vs_cpu("SELECT sum(v - w) s, min(v - w) mn FROM near8;");
  compare_gpu_vs_cpu("SELECT sum(v::DECIMAL(38,0) * scale) s FROM beyond;");
}
