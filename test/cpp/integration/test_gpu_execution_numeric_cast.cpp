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
 * @file test_gpu_execution_numeric_cast.cpp
 * @brief GPU casts to DECIMAL and integer types match DuckDB.
 *
 * cudf::cast and the cuDF AST CAST_TO_INT64 truncate toward zero and wrap values that do not fit.
 * DuckDB rounds: DOUBLE 1 - 0.07 is 0.9299999999999999, which casts to 0.93 in DECIMAL(16,2);
 * FLOAT and DOUBLE round to integers with ties to even (0.93 -> 1, 2.5 -> 2, -2.675 -> -3); DECIMAL
 * rounds half away from zero (2.5000 -> 3). Values that do not fit fail CAST and give NULL from
 * TRY_CAST. Covers ties, values just below a tie, signed values, values at and beyond each
 * target's range, NaN, infinities and NULL, for 32-, 64- and 128-bit DECIMAL and all integer
 * targets, through the materialized and the AST evaluation paths and over parquet.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>
#include <vector>

using NumericCastFixture = sirius::test::GpuExecutionFixture;

namespace {

using names = std::vector<std::string>;

/// DECIMAL targets for FLOAT, DOUBLE and DECIMAL sources, with at least 7 integer digits so every
/// value of `fits_t` fits. DECIMAL with a precision of 4 or less has no cuDF type.
names const kDecimalTargets{"DECIMAL(9,2)",
                            "DECIMAL(9,0)",
                            "DECIMAL(12,3)",
                            "DECIMAL(16,2)",
                            "DECIMAL(18,0)",
                            "DECIMAL(18,6)",
                            "DECIMAL(38,2)",
                            "DECIMAL(38,10)"};

/// DECIMAL targets for integer sources.
names const kIntegerToDecimalTargets{"DECIMAL(9,0)",
                                     "DECIMAL(9,2)",
                                     "DECIMAL(18,0)",
                                     "DECIMAL(18,8)",
                                     "DECIMAL(38,0)",
                                     "DECIMAL(38,30)"};

names const kSignedTargets{"TINYINT", "SMALLINT", "INTEGER", "BIGINT"};
names const kUnsignedTargets{"UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"};

/// Compares `SELECT id, <cast>(<column> AS <target>) AS c FROM <from>` on the GPU and the CPU for
/// every column and target.
void compare_casts(NumericCastFixture& fx,
                   std::string const& cast,
                   names const& columns,
                   names const& targets,
                   std::string const& from)
{
  for (auto const& column : columns) {
    for (auto const& target : targets) {
      CAPTURE(column, target);
      fx.compare_gpu_vs_cpu("SELECT id, " + cast + "(" + column + " AS " + target + ") AS c FROM " +
                            from);
    }
  }
}

/// Runs @p query on the CPU and on the GPU: both must fail with the same error, the GPU through a
/// runtime fallback that replays the query on the CPU. The GPU must not return a wrapped or
/// truncated value.
void expect_same_error(NumericCastFixture& fx, std::string const& query)
{
  CAPTURE(query);
  fx.run_ok("SET gpu_execution = false;");
  auto cpu_result = fx.con->Query(query);
  REQUIRE(cpu_result);
  REQUIRE(cpu_result->HasError());

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

/// Tables of DOUBLE `d`, FLOAT `f` and DECIMAL(18,4) `dec`: `fits_t` holds values below 10^7 in
/// magnitude, also as FLOAT, and `limits_t` values at and beyond the DECIMAL precision limits plus
/// NaN and infinities. The scan does not decode DECIMAL(38,x) storage, so 128-bit sources are cast
/// in the query.
void create_decimal_tables(NumericCastFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE fits_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f,"
    " TRY_CAST(d AS DECIMAL(18,4)) AS dec FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " (1::DOUBLE - 0.07::DOUBLE), (1::DOUBLE - 0.29::DOUBLE), ('0.9299999999999999'::DOUBLE),"
    " ('1.005'::DOUBLE), ('-1.005'::DOUBLE), ('2.675'::DOUBLE), ('-2.675'::DOUBLE),"
    " ('0.125'::DOUBLE), ('-0.125'::DOUBLE), ('0.5'::DOUBLE), ('-0.5'::DOUBLE), ('1.5'::DOUBLE),"
    " ('2.5'::DOUBLE), ('-2.5'::DOUBLE), ('0.005'::DOUBLE), ('-0.005'::DOUBLE),"
    " ('0.00499999999999'::DOUBLE), ('0.49999999999999994'::DOUBLE), ('0.0'::DOUBLE),"
    " ('-0.0'::DOUBLE), ('5e-324'::DOUBLE), ('1234567.125'::DOUBLE), ('-1234567.125'::DOUBLE),"
    " ('9999999.49'::DOUBLE), ('-9999999.49'::DOUBLE), ('9999999.4949'::DOUBLE), (NULL)) v(d)"
    " UNION ALL SELECT 100 + i, (i * 7919 % 100003) / ((i % 977) + 3.0) - 1000 FROM range(3000) "
    "t(i)"
    " UNION ALL SELECT 4000 + i, 1::DOUBLE - (i % 11) / 100.0 FROM range(1000) t(i));");
  fx.run_ok(
    "CREATE TABLE limits_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f,"
    " TRY_CAST(d AS DECIMAL(18,4)) AS dec FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " ('9999999.994999'::DOUBLE), ('9999999.996'::DOUBLE), ('-9999999.996'::DOUBLE),"
    " ('99999999.995'::DOUBLE), ('999999999.4'::DOUBLE), ('999999999.5'::DOUBLE),"
    " ('99999999999.9995'::DOUBLE), ('99999999999999.99'::DOUBLE), ('1e16'::DOUBLE),"
    " ('999999999999999872'::DOUBLE), ('1e18'::DOUBLE), ('-1e18'::DOUBLE), ('1e27'::DOUBLE),"
    " ('9.999999999999999e27'::DOUBLE), ('1e36'::DOUBLE), ('9.99999999999999e37'::DOUBLE),"
    " ('1e38'::DOUBLE), ('-1e38'::DOUBLE), ('1.7976931348623157e308'::DOUBLE),"
    " ('inf'::DOUBLE), ('-inf'::DOUBLE), ('nan'::DOUBLE), ('0.125'::DOUBLE), (NULL),"
    " ('9999999.99'::DOUBLE), ('9999999.995'::DOUBLE)) v(d));");
  fx.run_ok("CHECKPOINT;");
}

/// `vals_t` holds DOUBLE `d`, FLOAT `f` and DECIMAL(18,4) `dec`: ties, values next to each integer
/// type's bounds, NaN, infinities and NULL. `ints_t` holds BIGINT `b` and its INTEGER, SMALLINT,
/// TINYINT and UBIGINT casts at and beyond each integer type's bounds.
void create_integer_tables(NumericCastFixture& fx)
{
  fx.run_ok(
    "CREATE TABLE vals_t AS SELECT id, d, TRY_CAST(d AS FLOAT) AS f,"
    " TRY_CAST(d AS DECIMAL(18,4)) AS dec FROM ("
    " SELECT row_number() OVER () AS id, d FROM (VALUES"
    " ('0.93'::DOUBLE), ('-0.93'::DOUBLE), ('2.5'::DOUBLE), ('-2.5'::DOUBLE), ('1.5'::DOUBLE),"
    " ('0.5'::DOUBLE), ('-0.5'::DOUBLE), ('3.5'::DOUBLE), ('2.675'::DOUBLE), ('-2.675'::DOUBLE),"
    " ('0.49999999999999994'::DOUBLE), ('-0.4'::DOUBLE), ('0.0'::DOUBLE), ('-0.0'::DOUBLE),"
    " ('5e-324'::DOUBLE), ('-5e-324'::DOUBLE), ('127.4'::DOUBLE), ('126.5'::DOUBLE),"
    " ('-128.4'::DOUBLE), ('-128.5'::DOUBLE), ('-128.6'::DOUBLE), ('128.0'::DOUBLE),"
    " ('127.5'::DOUBLE), ('127.6'::DOUBLE), ('255.5'::DOUBLE), ('-0.6'::DOUBLE),"
    " ('-2147483648.5'::DOUBLE), ('-2147483648.6'::DOUBLE), ('-32768.6'::DOUBLE),"
    " ('254.5'::DOUBLE), ('255.0'::DOUBLE), ('256.0'::DOUBLE), ('32767.0'::DOUBLE),"
    " ('-32768.5'::DOUBLE), ('32768.0'::DOUBLE), ('65535.25'::DOUBLE), ('65536.0'::DOUBLE),"
    " ('2147483646.5'::DOUBLE), ('-2147483648.0'::DOUBLE), ('-2147483648.4'::DOUBLE),"
    " ('-2147483647.5'::DOUBLE), ('2147483648.0'::DOUBLE), ('4294967294.5'::DOUBLE),"
    " ('4294967296.0'::DOUBLE), ('1e10'::DOUBLE), ('-1e10'::DOUBLE), ('9.2e18'::DOUBLE),"
    " ('-9.2e18'::DOUBLE), ('9223372036854775808.0'::DOUBLE), ('-9223372036854775808.0'::DOUBLE),"
    " ('1.8e19'::DOUBLE), ('18446744073709551616.0'::DOUBLE), ('1e30'::DOUBLE), ('-1e30'::DOUBLE),"
    " ('nan'::DOUBLE), ('inf'::DOUBLE), ('-inf'::DOUBLE), (NULL)) v(d)"
    " UNION ALL SELECT 100 + i, i / 4.0 - 100 FROM range(800) t(i)"
    " UNION ALL SELECT 5000 + i, (i * 7919 % 100003) / ((i % 977) + 3.0) - 1000"
    " FROM range(3000) t(i));");
  fx.run_ok(
    "CREATE TABLE ints_t AS SELECT id, b, TRY_CAST(b AS INTEGER) AS i,"
    " TRY_CAST(b AS SMALLINT) AS s, TRY_CAST(b AS TINYINT) AS t, TRY_CAST(b AS UBIGINT) AS u FROM ("
    " SELECT row_number() OVER () AS id, b FROM (VALUES"
    " (0::BIGINT), (1), (-1), (127), (128), (-128), (-129), (255), (256), (32767), (32768),"
    " (-32768), (-32769), (65535), (65536), (2147483647), (2147483648), (-2147483648),"
    " (-2147483649), (4294967295), (4294967296), (9999999), (10000000), (999999999),"
    " (1000000000), (-1000000000), (10000000000), (99999999999999999), (100000000000000000),"
    " (999999999999999999), (1000000000000000000), (9223372036854775807),"
    " (-9223372036854775808), (NULL)) v(b)"
    " UNION ALL SELECT 100 + i, (i - 1000) * (i - 1000) * (i - 1000) FROM range(2000) t(i));");
  fx.run_ok("CHECKPOINT;");
}

}  // namespace

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST of DOUBLE, FLOAT and DECIMAL to DECIMAL rounds like DuckDB",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  create_decimal_tables(*this);
  compare_casts(*this,
                "CAST",
                {"d", "f", "dec", "CAST(dec AS DECIMAL(38,4))", "TRY_CAST(dec AS DECIMAL(9,4))"},
                kDecimalTargets,
                "fits_t");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST of FLOAT to DECIMAL keeps every digit of the rounded value",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  // DuckDB 2.0 rounds x * 10^scale in DOUBLE and converts that DOUBLE (DoubleToDecimalCast); 1.x
  // narrowed it back to FLOAT first and lost digits beyond 2^24: 1234567.125::FLOAT is
  // 1234567.13 in DECIMAL(9,2), not 1234567.12. The values sit near rounding boundaries and at the
  // precision limits of FLOAT and of the targets: ties exact in binary, values just below a
  // decimal tie, integers around 2^24, powers of ten that FLOAT rounds (1e38 is 9.99999968e37),
  // the largest FLOAT, subnormals, signed zeros, NaN, infinities and NULL.
  run_ok(
    "CREATE TABLE float_t AS SELECT row_number() OVER () AS id, f::FLOAT AS f FROM (VALUES"
    " ('1234567.125'), ('-1234567.125'), ('99999999.995'), ('12345678'), ('-12345678'),"
    " ('16777215'), ('16777216'), ('16777217'), ('-16777217'), ('8388607.5'), ('-8388607.5'),"
    " ('8388608.5'), ('9999999'), ('9999999.5'), ('123456789'), ('0.125'), ('-0.125'),"
    " ('0.375'), ('0.5'), ('-0.5'), ('2.5'), ('-2.5'), ('1.005'), ('2.675'), ('-2.675'),"
    " ('0.015'), ('0.005'), ('-0.005'), ('1.0000001'), ('7.0000005'), ('4294967296'),"
    " ('123456780000'), ('1e15'), ('9.999999e17'), ('1e18'), ('1e27'), ('1e38'), ('-1e38'),"
    " ('3.4028235e38'), ('-3.4028235e38'), ('1.17549435e-38'), ('1e-45'), ('0'), ('-0'),"
    " ('inf'), ('-inf'), ('nan'), (NULL)) v(f);");
  run_ok("CHECKPOINT;");
  compare_casts(*this, "TRY_CAST", {"f"}, kDecimalTargets, "float_t");
  compare_casts(*this, "CAST", {"f"}, kDecimalTargets, "float_t WHERE f > -1e7 AND f < 1e7");
  compare_casts(*this, "CAST", {"f"}, {"DECIMAL(18,2)"}, "float_t WHERE f > -1e16 AND f < 1e16");
  compare_casts(*this, "CAST", {"f"}, {"DECIMAL(38,0)"}, "float_t WHERE f > -1e38 AND f < 1e38");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution TRY_CAST to DECIMAL gives NULL where the value does not fit",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  create_decimal_tables(*this);
  compare_casts(*this,
                "TRY_CAST",
                {"d", "f", "dec", "TRY_CAST(d AS DECIMAL(38,4))"},
                kDecimalTargets,
                "limits_t");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST to DECIMAL fails like DuckDB where the value does not fit",
                 "[integration][gpu_execution][projection][decimal_cast]")
{
  create_decimal_tables(*this);
  // Each query holds one value that fails the cast.
  for (auto const* query :
       {"SELECT CAST(d AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)",
        "SELECT CAST(d AS DECIMAL(18,0)) FROM limits_t WHERE id IN (10, 11)",
        "SELECT CAST(d AS DECIMAL(38,0)) FROM limits_t WHERE id IN (16, 17)",
        "SELECT CAST(f AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)",
        "SELECT CAST(d AS DECIMAL(16,2)) FROM limits_t WHERE id IN (20, 23, 24)",
        "SELECT CAST(d AS DECIMAL(16,2)) FROM limits_t WHERE id IN (22, 23, 24)",
        "SELECT CAST(dec AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)",
        "SELECT CAST(dec AS DECIMAL(12,0)) FROM limits_t WHERE id IN (6, 8)",
        "SELECT CAST(dec AS DECIMAL(12,6)) FROM limits_t WHERE id IN (5, 23)",
        "SELECT CAST(CAST(d AS DECIMAL(38,4)) AS DECIMAL(9,2)) FROM limits_t WHERE id IN (1, 2)"}) {
    expect_same_error(*this, query);
  }
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution TPC-H revenue factor CAST(1 - discount AS DECIMAL) matches DuckDB",
                 "[integration][gpu_execution][aggregate][decimal_cast]")
{
  // The StarRocks plans evaluate 1 - l_discount in DOUBLE and cast it back to DECIMAL(16,2).
  run_ok(
    "CREATE TABLE lineitem_t AS SELECT (i % 11 / 100)::DECIMAL(15,2) AS l_discount,"
    " (900 + i % 9000 / 7)::DECIMAL(15,2) AS l_extendedprice FROM range(20000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu(
    "SELECT CAST((1::DOUBLE - l_discount::DOUBLE) AS DECIMAL(16,2)) AS c, count(*) AS n"
    " FROM lineitem_t GROUP BY 1");
  compare_gpu_vs_cpu(
    "SELECT l_discount, sum(l_extendedprice * CAST((1::DOUBLE - l_discount::DOUBLE) AS"
    " DECIMAL(16,2))) AS revenue FROM lineitem_t GROUP BY 1");
  // DuckDB widens the DECIMAL(12,2) operand to DECIMAL(15,2), a cast that cannot fail.
  compare_gpu_vs_cpu(
    "SELECT l_extendedprice + CAST(l_discount AS DECIMAL(12,2)) AS s FROM lineitem_t");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST of DOUBLE, FLOAT and DECIMAL to integers rounds like DuckDB",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_integer_tables(*this);
  // Each range keeps the values every target of its signedness accepts.
  compare_casts(
    *this, "CAST", {"d", "f", "dec"}, kSignedTargets, "vals_t WHERE d BETWEEN -128 AND 127");
  compare_casts(
    *this, "CAST", {"d", "f", "dec"}, kUnsignedTargets, "vals_t WHERE d BETWEEN 0 AND 255");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST of DOUBLE and FLOAT to integers checks the rounded value",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  // DuckDB 2.0 checks the range after rounding: these values lie outside the target, but round
  // into it, so CAST succeeds. DuckDB 1.x checked the unrounded value.
  create_integer_tables(*this);
  compare_casts(*this, "CAST", {"d", "f"}, {"TINYINT"}, "vals_t WHERE d IN (-128.4, -128.5, 0.93)");
  compare_casts(
    *this, "CAST", {"d", "f"}, {"UTINYINT"}, "vals_t WHERE d IN (-0.4, -0.5, 254.5, 0.93)");
  compare_casts(*this, "CAST", {"d", "f"}, {"SMALLINT"}, "vals_t WHERE d IN (-32768.5, 0.93)");
  compare_casts(
    *this, "CAST", {"d"}, {"INTEGER"}, "vals_t WHERE d IN (-2147483648.4, -2147483648.5)");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution TRY_CAST to integers gives NULL where the value does not fit",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_integer_tables(*this);
  for (auto const* targets : {&kSignedTargets, &kUnsignedTargets}) {
    compare_casts(
      *this, "TRY_CAST", {"d", "f", "dec", "CAST(dec AS DECIMAL(38,4))"}, *targets, "vals_t");
    compare_casts(*this, "TRY_CAST", {"b", "i", "s", "t"}, *targets, "ints_t");
  }
  // Unsigned sources reach the GPU only for targets that hold their whole range.
  compare_casts(
    *this, "TRY_CAST", {"CAST(t AS UTINYINT)"}, {"UBIGINT", "UINTEGER"}, "ints_t WHERE t >= 0");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST of integers to DECIMAL checks the precision like DuckDB",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_integer_tables(*this);
  names const columns{"b", "i", "s", "t", "u"};
  compare_casts(*this, "TRY_CAST", columns, kIntegerToDecimalTargets, "ints_t");
  compare_casts(*this,
                "CAST",
                columns,
                kIntegerToDecimalTargets,
                "ints_t WHERE b BETWEEN -9999999 AND 9999999");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST to integers and DECIMAL fails like DuckDB where it overflows",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_integer_tables(*this);
  // Each query holds one value that fails the cast.
  for (auto const* query : {
         "SELECT CAST(b AS DECIMAL(9,0)) FROM ints_t WHERE b IN (0, 10000000000)",
         "SELECT CAST(b AS DECIMAL(9,0)) FROM ints_t WHERE b IN (0, 1000000000)",
         "SELECT CAST(b AS DECIMAL(18,8)) FROM ints_t WHERE b IN (0, 10000000000)",
         "SELECT CAST(u AS DECIMAL(18,0)) FROM ints_t WHERE b IN (0, 1000000000000000000)",
         "SELECT CAST(b AS INTEGER) FROM ints_t WHERE b IN (0, 2147483648)",
         "SELECT CAST(b AS UINTEGER) FROM ints_t WHERE b IN (0, -1)",
         "SELECT CAST(i AS SMALLINT) FROM ints_t WHERE b IN (0, -32769)",
         "SELECT CAST(t AS UBIGINT) FROM ints_t WHERE b IN (0, -1)",
         "SELECT CAST(d AS INTEGER) FROM vals_t WHERE d IN (0.93, 2147483648)",
         "SELECT CAST(d AS INTEGER) FROM vals_t WHERE d IN (0.93, -2147483648.6)",
         "SELECT CAST(d AS TINYINT) FROM vals_t WHERE d IN (0.93, 127.6)",
         "SELECT CAST(f AS UTINYINT) FROM vals_t WHERE d IN (0.93, 255.5)",
         "SELECT CAST(d AS BIGINT) FROM vals_t WHERE d IN (0.93, 9223372036854775808.0)",
         "SELECT CAST(d AS BIGINT) FROM vals_t WHERE d IN (0.93, 'inf'::DOUBLE)",
         "SELECT CAST(d AS UTINYINT) FROM vals_t WHERE d IN (0.93, -0.6)",
         "SELECT CAST(f AS SMALLINT) FROM vals_t WHERE d IN (0.93, 32768)",
         "SELECT CAST(dec AS TINYINT) FROM vals_t WHERE d IN (0.93, -128.5)",
         "SELECT CAST(dec AS INTEGER) FROM vals_t WHERE d IN (0.93, 1e10)",
       }) {
    expect_same_error(*this, query);
  }
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST to BIGINT inside cuDF AST expressions rounds like DuckDB",
                 "[integration][gpu_execution][projection][filter][integer_cast]")
{
  create_integer_tables(*this);
  // Enough operators for the AST path: a cast from DOUBLE or DECIMAL must not lower to
  // CAST_TO_INT64, which truncates.
  compare_gpu_vs_cpu(
    "SELECT id, CAST(d AS BIGINT) * 2 + id AS c, TRY_CAST(f AS BIGINT) - id AS t FROM vals_t"
    " WHERE d BETWEEN -1e6 AND 1e6");
  compare_gpu_vs_cpu(
    "SELECT id, CAST(dec AS BIGINT) * 2 + id AS c FROM vals_t WHERE d BETWEEN -1e6 AND 1e6");
  compare_gpu_vs_cpu("SELECT id FROM vals_t WHERE CAST(dec AS BIGINT) = -3");
  compare_gpu_vs_cpu("SELECT id FROM vals_t WHERE TRY_CAST(d AS BIGINT) + 1 = 2");
  // Exact integer casts keep the AST path.
  compare_gpu_vs_cpu("SELECT id, CAST(i AS BIGINT) * 2 + CAST(t AS BIGINT) AS c FROM ints_t");
  compare_gpu_vs_cpu(
    "SELECT id, CAST(i AS DOUBLE) * 2 + d AS c FROM ints_t JOIN vals_t USING (id)");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution CAST of DOUBLE to integers over parquet rounds like DuckDB",
                 "[integration][gpu_execution][parquet][integer_cast]")
{
  create_integer_tables(*this);
  sirius::test::scratch_dir dir("integer_cast");
  auto const file = dir.file_literal("vals.parquet");
  run_ok("SET gpu_execution = false;");
  run_ok("COPY vals_t TO " + file + " (FORMAT parquet);");
  run_ok("SET gpu_execution = true;");
  auto const scan = "read_parquet(" + file + ")";
  compare_gpu_vs_cpu(
    "SELECT id, CAST(d AS BIGINT) AS b, CAST(d AS INTEGER) AS i,"
    " CAST(dec AS INTEGER) AS di FROM " +
    scan + " WHERE d BETWEEN -1e6 AND 1e6");
  compare_gpu_vs_cpu("SELECT id, TRY_CAST(d AS INTEGER) AS i FROM " + scan);
  compare_gpu_vs_cpu("SELECT id FROM " + scan + " WHERE CAST(dec AS BIGINT) = 1");
  compare_gpu_vs_cpu("SELECT id FROM " + scan + " WHERE TRY_CAST(d AS BIGINT) = -3");
  compare_gpu_vs_cpu("SELECT count(*) FROM " + scan + " WHERE TRY_CAST(d AS SMALLINT) IS NULL");
}

TEST_CASE_METHOD(NumericCastFixture,
                 "gpu_execution TRY_CAST to HUGEINT replays on the CPU beyond BIGINT",
                 "[integration][gpu_execution][projection][integer_cast]")
{
  create_integer_tables(*this);
  // HUGEINT runs as INT64 on the GPU.
  compare_gpu_vs_cpu(
    "SELECT id, TRY_CAST(d AS HUGEINT) AS h FROM vals_t WHERE d BETWEEN -1e6 AND 1e6");
  auto const beyond = std::string("SELECT TRY_CAST(d AS HUGEINT) AS h FROM vals_t WHERE d = 1e30");
  expect_gpu_fallback(beyond);
  run_ok("SET gpu_execution = false;");
  auto cpu_result = con->Query(beyond);
  run_ok("SET gpu_execution = true;");
  auto gpu_result = con->Query(beyond);
  REQUIRE(cpu_result);
  REQUIRE(gpu_result);
  REQUIRE_FALSE(cpu_result->HasError());
  REQUIRE_FALSE(gpu_result->HasError());
  REQUIRE(gpu_result->ToString() == cpu_result->ToString());
}
