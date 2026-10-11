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
 * @file test_gpu_execution_decimal_divide.cpp
 * @brief sirius_decimal_divide() on the CPU and the GPU matches an exact reference.
 *
 * DuckDB divides decimals in DOUBLE; sirius_decimal_divide(a, b, s) returns the exact quotient as
 * DECIMAL(38, s), rounded half away from zero, as StarRocks computes `/` and AVG. The expected
 * values were computed with Python's decimal module (ROUND_HALF_UP). Each case runs once with GPU
 * execution, asserting the query ran on the GPU, and once on the CPU.
 *
 * The scan does not decode DECIMAL(38,x) storage, so 128-bit operands are built in the query as
 * hi * 10^k + lo from DECIMAL(18,x) columns.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>
#include <utility>
#include <vector>

using DecimalDivideFixture = sirius::test::GpuExecutionFixture;

namespace {

using expected_rows = std::vector<std::pair<int, std::string>>;

/// Runs @p query (columns: id, quotient) with GPU execution, requiring that it ran on the GPU, and
/// on the CPU, and checks both against @p expected in id order. With @p on_gpu false, only the CPU
/// runs it.
void check_quotients(DecimalDivideFixture& fx,
                     std::string const& query,
                     expected_rows const& expected,
                     bool on_gpu = true)
{
  auto const ordered = query + " ORDER BY id";
  for (bool const gpu : {true, false}) {
    if (gpu && !on_gpu) { continue; }
    CAPTURE(query, gpu);
    fx.run_ok(gpu ? "SET gpu_execution = true;" : "SET gpu_execution = false;");
    auto const before = sirius::test::get_transparent_execution_stats(*fx.con);
    auto result       = fx.con->Query(ordered);
    auto const after  = sirius::test::get_transparent_execution_stats(*fx.con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    if (gpu) { sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1); }
    REQUIRE(result->RowCount() == expected.size());
    for (duckdb::idx_t r = 0; r < result->RowCount(); ++r) {
      CAPTURE(r);
      CHECK(result->Collection().GetValue(0, r).ToString() == std::to_string(expected[r].first));
      CHECK(result->Collection().GetValue(1, r).ToString() == expected[r].second);
    }
  }
  fx.run_ok("SET gpu_execution = true;");
}

/// Requires @p query to fail on the CPU, and with GPU execution to fail on the GPU, replay on the
/// CPU and raise the same error.
void check_overflow(DecimalDivideFixture& fx, std::string const& query)
{
  CAPTURE(query);
  fx.run_ok("SET gpu_execution = false;");
  auto cpu_result = fx.con->Query(query);
  REQUIRE(cpu_result);
  REQUIRE(cpu_result->HasError());
  CHECK_THAT(cpu_result->GetError(), Catch::Matchers::ContainsSubstring("more than 38 digits"));

  fx.run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*fx.con);
  auto gpu_result   = fx.con->Query(query);
  auto const after  = sirius::test::get_transparent_execution_stats(*fx.con);
  REQUIRE(gpu_result);
  if (!gpu_result->HasError()) { UNSCOPED_INFO("GPU returned: " << gpu_result->ToString()); }
  REQUIRE(gpu_result->HasError());
  REQUIRE(gpu_result->GetError() == cpu_result->GetError());
  REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks + 1);
}

/// DECIMAL(38,4) built from DECIMAL(18,4) columns: hi * 10^18 + lo.
constexpr char const* kWide4 =
  "(CAST(hi AS DECIMAL(38,4)) * CAST(1000000000000000000 AS DECIMAL(38,0)) + lo)";

/// DECIMAL(38,2) values built from DECIMAL(18,2) columns: hi * 10^12 + lo.
constexpr char const* kWide2A =
  "(CAST(a_hi AS DECIMAL(38,2)) * CAST(1000000000000 AS DECIMAL(38,0)) + a_lo)";
constexpr char const* kWide2B =
  "(CAST(b_hi AS DECIMAL(38,2)) * CAST(1000000000000 AS DECIMAL(38,0)) + b_lo)";

}  // namespace

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide DECIMAL(15,2) / DECIMAL(15,2) at scale 8",
                 "[integration][gpu_execution][projection][decimal_divide]")
{
  // Signs, ties at the last digit (0.01 / 2000000.00 = 0.000000005), the largest DECIMAL(15,2)
  // over the smallest, a zero divisor and NULLs.
  run_ok(
    "CREATE TABLE d15 AS SELECT id::INTEGER id, a::DECIMAL(15,2) a, b::DECIMAL(15,2) b FROM (VALUES"
    " (1, 1.00, 3.00), (2, 2.00, 3.00), (3, -2.00, 3.00), (4, 2.00, -3.00), (5, -2.00, -3.00),"
    " (6, 0.01, 2000000.00), (7, -0.01, 2000000.00), (8, 0.03, 2000000.00),"
    " (9, 0.05, 2000000.00), (10, -0.05, 2000000.00), (11, 9999999999999.99, 0.01),"
    " (12, -9999999999999.99, 0.03), (13, 0.00, 5.00), (14, 5.00, 0.00), (15, NULL, 3.00),"
    " (16, 3.00, NULL), (17, 123.45, 0.07)) v(id, a, b);");
  run_ok("CHECKPOINT;");
  check_quotients(*this,
                  "SELECT id, sirius_decimal_divide(a, b, 8) FROM d15",
                  {{1, "0.33333333"},
                   {2, "0.66666667"},
                   {3, "-0.66666667"},
                   {4, "-0.66666667"},
                   {5, "0.66666667"},
                   {6, "0.00000001"},
                   {7, "-0.00000001"},
                   {8, "0.00000002"},
                   {9, "0.00000003"},
                   {10, "-0.00000003"},
                   {11, "999999999999999.00000000"},
                   {12, "-333333333333333.00000000"},
                   {13, "0.00000000"},
                   {14, "NULL"},
                   {15, "NULL"},
                   {16, "NULL"},
                   {17, "1763.57142857"}});
}

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide DECIMAL(38,4) / BIGINT at scale 10 (AVG's sum / count)",
                 "[integration][gpu_execution][projection][decimal_divide]")
{
  // Rows 1 and 2 scale the dividend by 10^6 past 2^128.
  run_ok(
    "CREATE TABLE avg_t AS SELECT id::INTEGER id, hi::DECIMAL(18,4) hi, lo::DECIMAL(18,4) lo,"
    " n::BIGINT n FROM (VALUES"
    " (1, 12345678901234.5678, 90123456789012.3456, 9999),"
    " (2, -12345678901234.5678, -90123456789012.3456, 9999), (3, 0, 12345.6789, 7),"
    " (4, 0, 0.0001, 3), (5, 0, 0.0005, 1000000000000000000), (6, 0, 1.0000, 0),"
    " (7, NULL, NULL, 5), (8, 0, 0.0001, 2000000), (9, 0, -0.0001, 2000000),"
    " (10, 0.1234, 56789012345678.9012, 3)) v(id, hi, lo, n);");
  run_ok("CHECKPOINT;");
  check_quotients(*this,
                  std::string("SELECT id, sirius_decimal_divide(") + kWide4 + ", n, 10) FROM avg_t",
                  {{1, "1234691359259382727285074186.3198665467"},
                   {2, "-1234691359259382727285074186.3198665467"},
                   {3, "1763.6684142857"},
                   {4, "0.0000333333"},
                   {5, "0.0000000000"},
                   {6, "NULL"},
                   {7, "NULL"},
                   {8, "0.0000000001"},
                   {9, "-0.0000000001"},
                   {10, "41152263004115226.3004000000"}});
}

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide DECIMAL(38,2) / DECIMAL(38,2) at scale 12",
                 "[integration][gpu_execution][projection][decimal_divide]")
{
  // Rows 1 to 3 scale the dividend by 10^12 past 2^128; row 6's divisor needs more than 64 bits.
  run_ok(
    "CREATE TABLE wide2 AS SELECT id::INTEGER id, a_hi::DECIMAL(18,2) a_hi, a_lo::DECIMAL(18,2) "
    "a_lo,"
    " b_hi::DECIMAL(18,2) b_hi, b_lo::DECIMAL(18,2) b_lo FROM (VALUES"
    " (1, 12345678901234.56, 7890123456.78, 0.00, 3.00),"
    " (2, -12345678901234.56, -7890123456.78, 0.00, -7.00),"
    " (3, 99999999999999.99, 9999999999.99, 99999999999999.99, 9999999999.98),"
    " (4, 0.00, 1.00, 0.00, 3.00), (5, 0.00, 0.00, 0.00, 0.00),"
    " (6, 0.00, 5.00, 12345678901234.56, 7890123456.78), (7, 0.00, -1.00, 0.00, 0.08))"
    " v(id, a_hi, a_lo, b_hi, b_lo);");
  run_ok("CHECKPOINT;");
  check_quotients(*this,
                  std::string("SELECT id, sirius_decimal_divide(") + kWide2A + ", " + kWide2B +
                    ", 12) FROM wide2",
                  {{1, "4115226300411522630041152.260000000000"},
                   {2, "1763668414462081127160493.825714285714"},
                   {3, "1.000000000000"},
                   {4, "0.333333333333"},
                   {5, "NULL"},
                   {6, "0.000000000000"},
                   {7, "-12.500000000000"}});
}

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide with integer operands and fewer result digits",
                 "[integration][gpu_execution][projection][decimal_divide]")
{
  run_ok(
    "CREATE TABLE ints AS SELECT id::INTEGER id, i::INTEGER i, j::BIGINT j, b::DECIMAL(15,2) b,"
    " t::TINYINT t FROM (VALUES (1, 5, 2, 0.40, 5), (2, -5, 2, -3.00, -5), (3, 7, -2, 0.03, 7),"
    " (4, 1, 0, 0.00, 1), (5, 2147483647, 1, 7.00, 127), (6, NULL, 1, 1.00, NULL),"
    " (7, 3, NULL, NULL, 3)) v(id, i, j, b, t);");
  run_ok(
    "CREATE TABLE negexp AS SELECT id::INTEGER id, a::DECIMAL(15,6) a, k::INTEGER k FROM (VALUES"
    " (1, 1.234567, 1), (2, 1.235000, 1), (3, -1.235000, 1), (4, 1.234999, 1), (5, -0.005000, 1),"
    " (6, 0.004999, 1), (7, 9.990000, 3)) v(id, a, k);");
  run_ok("CHECKPOINT;");
  check_quotients(
    *this,
    "SELECT id, sirius_decimal_divide(i, j, 0) FROM ints",
    {{1, "3"}, {2, "-3"}, {3, "-4"}, {4, "NULL"}, {5, "2147483647"}, {6, "NULL"}, {7, "NULL"}});
  check_quotients(*this,
                  "SELECT id, sirius_decimal_divide(i, b, 3) FROM ints",
                  {{1, "12.500"},
                   {2, "1.667"},
                   {3, "233.333"},
                   {4, "NULL"},
                   {5, "306783378.143"},
                   {6, "NULL"},
                   {7, "NULL"}});
  check_quotients(
    *this,
    "SELECT id, sirius_decimal_divide(t, j, 1) FROM ints",
    {{1, "2.5"}, {2, "-2.5"}, {3, "-3.5"}, {4, "NULL"}, {5, "127.0"}, {6, "NULL"}, {7, "NULL"}});
  // The result keeps fewer fractional digits than the dividend: the divisor is scaled instead.
  check_quotients(
    *this,
    "SELECT id, sirius_decimal_divide(a, k, 2) FROM negexp",
    {{1, "1.23"}, {2, "1.24"}, {3, "-1.24"}, {4, "1.23"}, {5, "-0.01"}, {6, "0.00"}, {7, "3.33"}});
}

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide fails where the quotient needs more than 38 digits",
                 "[integration][gpu_execution][projection][decimal_divide]")
{
  run_ok(
    "CREATE TABLE avg_t AS SELECT 12345678901234.5678::DECIMAL(18,4) hi,"
    " 90123456789012.3456::DECIMAL(18,4) lo, 1000::BIGINT n, 1234567890123456789::BIGINT x,"
    " 0.5::DECIMAL(18,18) h;");
  run_ok(
    "CREATE TABLE wide2 AS SELECT -12345678901234.56::DECIMAL(18,2) a_hi,"
    " -7890123456.78::DECIMAL(18,2) a_lo, 0.00::DECIMAL(18,2) b_hi, 0.07::DECIMAL(18,2) b_lo;");
  run_ok("CHECKPOINT;");
  check_overflow(*this,
                 std::string("SELECT sirius_decimal_divide(") + kWide4 + ", n, 10) FROM avg_t");
  check_overflow(
    *this,
    std::string("SELECT sirius_decimal_divide(") + kWide2A + ", " + kWide2B + ", 12) FROM wide2");
  // About 1.2 * 10^37 / DECIMAL(38,38) at scale 38 scales the dividend by 10^76, past 2^256.
  check_overflow(*this,
                 "SELECT sirius_decimal_divide(CAST(x AS DECIMAL(38,0)) *"
                 " CAST(10000000000000000000 AS DECIMAL(38,0)), CAST(h AS DECIMAL(38,38)), 38)"
                 " FROM avg_t");
}

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide over generated rows matches DuckDB and rounds exactly",
                 "[integration][gpu_execution][projection][decimal_divide]")
{
  // 20000 rows with zeros and NULLs in both operands. hi * 10^18 + lo is a DECIMAL(38,4) of up to
  // about 5 * 10^30, which a scale of 10 or 12 scales past 2^128; m keeps those quotients within
  // 38 digits.
  run_ok(
    "CREATE TABLE gen AS SELECT i::INTEGER id,"
    " CASE WHEN i % 101 = 0 THEN NULL ELSE ((i * 7919 % 2000003) - 1000000) / 100 END"
    "::DECIMAL(15,2) a,"
    " CASE WHEN i % 103 = 0 THEN NULL ELSE ((i * 104729 % 20011) - 10000) / 100 END"
    "::DECIMAL(15,2) b,"
    " (i % 97)::BIGINT n, (i * 31 % 1999 - 999)::INTEGER k,"
    " CASE WHEN i % 89 = 0 THEN 0 ELSE 1000 + i % 97 END::BIGINT m,"
    " ((i * 7919 % 10007 - 5003) * 1000000000.1234)::DECIMAL(18,4) hi,"
    " ((i * 40503 % 99999999) / 10000)::DECIMAL(18,4) lo"
    " FROM range(20000) r(i);");
  run_ok("CHECKPOINT;");
  auto const wide = std::string(kWide4);
  for (auto const& query :
       {std::string("SELECT id, sirius_decimal_divide(a, b, 8) FROM gen"),
        std::string("SELECT id, sirius_decimal_divide(a, b, 0) FROM gen"),
        std::string("SELECT id, sirius_decimal_divide(a, n, 8) FROM gen"),
        std::string("SELECT id, sirius_decimal_divide(k, b, 4) FROM gen"),
        std::string("SELECT id, sirius_decimal_divide(a, 7, 8) FROM gen"),
        std::string("SELECT id, sirius_decimal_divide(1000, b, 2) FROM gen"),
        // A quotient below 1 in magnitude fits 38 digits at scale 38.
        std::string("SELECT id, sirius_decimal_divide(a, b, 38) FROM gen WHERE a * a < b * b"),
        "SELECT id, sirius_decimal_divide(" + wide + ", m, 10) FROM gen",
        "SELECT id, sirius_decimal_divide(" + wide + ", " + wide + " - 1, 12) FROM gen"}) {
    CAPTURE(query);
    compare_gpu_vs_cpu(query);
  }

  // An independent check of the rounding, in exact HUGEINT arithmetic: with N = a * 10^10 and
  // D = b * 10^2 in units, Q = the quotient in units of 10^-8, R = N - Q * D satisfies
  // 2|R| < |D|, or 2|R| = |D| with |Q * D| > |N| (a tie rounded away from zero).
  // The GPU results equal the CPU results above, so the check runs on the CPU, which has HUGEINT.
  run_ok("SET gpu_execution = false;");
  auto checked = con->Query(
    "WITH q AS (SELECT a, b, sirius_decimal_divide(a, b, 8) q FROM gen),"
    " u AS (SELECT (a * 100)::HUGEINT * 100000000 n, (b * 100)::HUGEINT d,"
    " (q * 100000000)::HUGEINT qq FROM q WHERE q IS NOT NULL)"
    " SELECT count(*) FILTER (WHERE NOT (2 * abs(n - qq * d) < abs(d) OR"
    " (2 * abs(n - qq * d) = abs(d) AND abs(qq * d) > abs(n)))) AS wrong, count(*) AS total"
    " FROM u");
  REQUIRE(checked);
  if (checked->HasError()) { UNSCOPED_INFO(checked->GetError()); }
  REQUIRE_FALSE(checked->HasError());
  CHECK(checked->Collection().GetValue(0, 0).GetValue<int64_t>() == 0);
  CHECK(checked->Collection().GetValue(1, 0).GetValue<int64_t>() > 19000);
  auto nulls = con->Query(
    "SELECT count(*) FILTER (WHERE (sirius_decimal_divide(a, b, 8) IS NULL) <>"
    " (a IS NULL OR b IS NULL OR b = 0)) FROM gen");
  REQUIRE(nulls);
  REQUIRE_FALSE(nulls->HasError());
  CHECK(nulls->Collection().GetValue(0, 0).GetValue<int64_t>() == 0);
}

TEST_CASE_METHOD(
  DecimalDivideFixture,
  "sirius_decimal_divide over DECIMAL32, narrow DECIMAL, UBIGINT and HUGEINT operands",
  "[integration][gpu_execution][projection][decimal_divide]")
{
  // DECIMAL(9,3) and DECIMAL(5,1) are 32-bit on the GPU. UBIGINT runs past 2^63 and the bind
  // casts it to DECIMAL(20,0); HUGEINT becomes DECIMAL(38,0). The GPU scan does not read HUGEINT
  // storage, so the GPU cases make HUGEINT operands from BIGINT in the query.
  run_ok(
    "CREATE TABLE typed AS SELECT i::INTEGER id,"
    " CASE WHEN i % 17 = 0 THEN NULL ELSE ((i * 7919 % 1999999) - 999999) / 1000 END"
    "::DECIMAL(9,3) d9,"
    " CASE WHEN i % 19 = 0 THEN NULL ELSE ((i * 104729 % 19999) - 9999) / 10 END::DECIMAL(5,1) d5,"
    " (((i * 37 % 199) - 99) / 100)::DECIMAL(4,2) d4,"
    " CASE WHEN i % 23 = 0 THEN NULL ELSE 18446744073709551615 - i * 1000003 END::UBIGINT u,"
    " (i * 1000000007 - 1000000000000)::HUGEINT h, (i * 1000000007 - 1000000000000)::BIGINT hb"
    " FROM range(2000) r(i);");
  run_ok("CHECKPOINT;");
  for (auto const* query :
       {"SELECT id, sirius_decimal_divide(d9, d5, 6) FROM typed",
        "SELECT id, sirius_decimal_divide(d5, d9, 9) FROM typed",
        "SELECT id, sirius_decimal_divide(d9, CAST(0.07 AS DECIMAL(4,2)), 4) FROM typed",
        "SELECT id, sirius_decimal_divide(u, d5, 3) FROM typed",
        "SELECT id, sirius_decimal_divide(d9, u, 30) FROM typed",
        "SELECT id, sirius_decimal_divide(CAST(hb AS HUGEINT), d9, 2) FROM typed",
        "SELECT id, sirius_decimal_divide(d5, CAST(hb AS HUGEINT), 20) FROM typed"}) {
    CAPTURE(query);
    compare_gpu_vs_cpu(query);
  }
  // A HUGEINT column is planned on the CPU, with the same results.
  expect_plan_fallback_matches_cpu("SELECT id, sirius_decimal_divide(h, d9, 2) FROM typed");
  // A DECIMAL(4,2) column is stored as INT16, which the GPU scan does not read; the CPU computes
  // it.
  check_quotients(*this,
                  "SELECT id, sirius_decimal_divide(d4, 3, 4) FROM typed WHERE id IN (1, 2, 50)",
                  {{1, "-0.2067"}, {2, "-0.0833"}, {50, "-0.1333"}},
                  /*on_gpu=*/false);
}

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide over a LIMIT/OFFSET slice with NULLs",
                 "[integration][gpu_execution][projection][decimal_divide]")
{
  run_ok(
    "CREATE TABLE gen AS SELECT i::INTEGER id,"
    " CASE WHEN i % 7 = 0 THEN NULL ELSE ((i * 7919 % 2000003) - 1000000) / 100 END"
    "::DECIMAL(15,2) a,"
    " CASE WHEN i % 13 = 0 THEN NULL WHEN i % 11 = 0 THEN 0 ELSE (i * 31 % 1999) - 999 END"
    "::INTEGER b FROM range(5000) r(i);");
  run_ok("CHECKPOINT;");
  // 333 rows from offset 37: a mask offset that isn't a multiple of 32 and a 13-row tail word.
  compare_gpu_vs_cpu(
    "SELECT id, sirius_decimal_divide(a, b, 8) FROM"
    " (SELECT * FROM gen ORDER BY id LIMIT 333 OFFSET 37)");
  compare_gpu_vs_cpu(
    "SELECT id, sirius_decimal_divide(a, b, 8) FROM (SELECT * FROM gen LIMIT 333 OFFSET 37)");
}

TEST_CASE_METHOD(DecimalDivideFixture,
                 "sirius_decimal_divide rejects a non-constant scale and non-numeric operands",
                 "[integration][gpu_execution][decimal_divide]")
{
  run_ok("CREATE TABLE t AS SELECT 1.00::DECIMAL(15,2) a, 2::INTEGER s, 1.5::DOUBLE d;");
  for (auto const* query : {"SELECT sirius_decimal_divide(a, a, s) FROM t",
                            "SELECT sirius_decimal_divide(a, a, 39) FROM t",
                            "SELECT sirius_decimal_divide(a, a, -1) FROM t",
                            "SELECT sirius_decimal_divide(a, a, NULL) FROM t",
                            "SELECT sirius_decimal_divide(d, a, 2) FROM t",
                            "SELECT sirius_decimal_divide(a, 'x', 2) FROM t"}) {
    CAPTURE(query);
    auto result = con->Query(query);
    REQUIRE(result);
    REQUIRE(result->HasError());
    CHECK_THAT(result->GetError(), Catch::Matchers::ContainsSubstring("sirius_decimal_divide"));
  }
  // A NULL operand gives a typed NULL.
  auto typed = con->Query(
    "SELECT typeof(sirius_decimal_divide(NULL, a, 5)), typeof(sirius_decimal_divide(a, 2::BIGINT, "
    "8::BIGINT)) FROM t");
  REQUIRE(typed);
  REQUIRE_FALSE(typed->HasError());
  CHECK(typed->Collection().GetValue(0, 0).ToString() == "DECIMAL(38,5)");
  CHECK(typed->Collection().GetValue(1, 0).ToString() == "DECIMAL(38,8)");
}
