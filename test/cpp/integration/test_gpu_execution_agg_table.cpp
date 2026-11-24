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
 * @file test_gpu_execution_agg_table.cpp
 * @brief GROUP BY correctness with the table path (src/cuda/agg_table) off and on.
 *
 * Every `on_off*` call runs its query on DuckDB's CPU execution, on the GPU with the table path
 * off and on the GPU with it on, and requires the same column types and rows from all three (see
 * utils/agg_table_test_utils.hpp). The suite does not know which queries the table path accepts:
 * a query it rejects must still return the right rows through the existing path, and a query it
 * accepts must return them through the table.
 *
 * Sizes. DuckDB stores 122880 rows per row group, and `scan_task_batch_size = 1` gives every row
 * group a scan batch of its own, so a table of N rows reaches the aggregate in N / 122880 batches
 * on the four pipeline threads of integration.yaml. Results with many groups are checked through
 * an outer aggregate, an ORDER BY ... LIMIT and a sampled HAVING, so no test stringifies more
 * than a few tens of thousands of rows.
 *
 * SQL semantics covered (plan section 7.1): keys -1 and -2 (the table's free and claimed
 * markers) in every key type, NULL keys, NaN / +-inf / -0.0 in float MIN and MAX, BIGINT sums
 * past INT64 (exact HUGEINT), DECIMAL sums that need 128 bits.
 */

#include <catch.hpp>
#include <utils/agg_table_test_utils.hpp>

#include <cstdint>
#include <string>

using sirius::test::scoped_sirius_setting;
using sirius::test::agg_table::AggTableFixture;
using sirius::test::agg_table::expected_path;
using sirius::test::agg_table::kBudgetSetting;
using sirius::test::agg_table::kPartitionBytes;
using sirius::test::agg_table::kScanBatch;
using sirius::test::agg_table::setting_exists;
using sirius::test::agg_table::subst;

#define AGG_TABLE_TAGS "[integration][gpu_execution][aggregate][agg_table]"

//===----------------------------------------------------------------------===//
// The plain case: what the table path exists for
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - one BIGINT key, integer aggregates, several batches",
                 AGG_TABLE_TAGS)
{
  // No NULLs, no negative keys, sums far inside INT64: nothing here gives the engine a reason to
  // leave the table path, so these checks name it as the expected path.
  run_ok(
    "CREATE TABLE plain AS SELECT (hash(i) % 5000)::BIGINT AS k, (i % 1000)::BIGINT AS v, "
    "(i % 77)::INTEGER AS w FROM range(1000000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off("SELECT k, count(*) AS c FROM plain GROUP BY k", expected_path::table);
  on_off("SELECT k, sum(v) AS s FROM plain GROUP BY k", expected_path::table);
  on_off(
    "SELECT k, count(*) AS c, count(v) AS cv, sum(v) AS s, min(v) AS mn, max(v) AS mx, "
    "sum(w) AS sw, min(w) AS mnw, max(w) AS mxw FROM plain GROUP BY k",
    expected_path::table);
  on_off_approx("SELECT k, avg(v) AS a, avg(w) AS aw FROM plain GROUP BY k", {1, 2}, 1e-12);
  on_off_ordered("SELECT k, sum(v) AS s FROM plain GROUP BY k ORDER BY k");
}

//===----------------------------------------------------------------------===//
// Key types
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - every fixed-width key type, with NULL, extreme and reserved keys",
                 AGG_TABLE_TAGS)
{
  // Each key column holds NULLs, the type's extremes, and the values whose 64-bit pattern is -1
  // or -2 either directly (BIGINT, UBIGINT 2^64-1 and 2^64-2) or after a sign extension
  // (narrower signed types, dates and timestamps just before the epoch).
  run_ok(
    "CREATE TABLE kt AS SELECT "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 127 WHEN i % 1009 = 2 THEN -128 "
    "  ELSE i % 100 - 50 END::TINYINT AS k_i8, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 32767 WHEN i % 1009 = 2 THEN -32768 "
    "  ELSE i % 2000 - 1000 END::SMALLINT AS k_i16, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 2147483647 "
    "  WHEN i % 1009 = 2 THEN -2147483648 ELSE i % 5000 - 2500 END::INTEGER AS k_i32, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 9223372036854775807 "
    "  WHEN i % 1009 = 2 THEN -9223372036854775808 ELSE i % 7000 - 3500 END::BIGINT AS k_i64, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 255 WHEN i % 1009 = 2 THEN 254 "
    "  ELSE i % 200 END::UTINYINT AS k_u8, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 65535 WHEN i % 1009 = 2 THEN 65534 "
    "  ELSE i % 3000 END::USMALLINT AS k_u16, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 4294967295 "
    "  WHEN i % 1009 = 2 THEN 4294967294 ELSE i % 4000 END::UINTEGER AS k_u32, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN 18446744073709551615 "
    "  WHEN i % 1009 = 2 THEN 18446744073709551614 WHEN i % 1009 = 3 THEN 9223372036854775808 "
    "  ELSE i % 6000 END::UBIGINT AS k_u64, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN DATE '1969-12-31' "
    "  WHEN i % 1009 = 2 THEN DATE '1969-12-30' WHEN i % 1009 = 3 THEN DATE '1970-01-01' "
    "  ELSE DATE '1992-01-01' + (i % 2500)::INTEGER END AS k_date, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN TIMESTAMP '1969-12-31 23:59:59.999999' "
    "  WHEN i % 1009 = 2 THEN TIMESTAMP '1969-12-31 23:59:59.999998' "
    "  WHEN i % 1009 = 3 THEN TIMESTAMP '1970-01-01 00:00:00' "
    "  ELSE TIMESTAMP '2000-01-01 00:00:00' + INTERVAL (i % 3600) SECOND END AS k_ts, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN TIMESTAMP_S '1969-12-31 23:59:59' "
    "  WHEN i % 1009 = 2 THEN TIMESTAMP_S '1969-12-31 23:59:58' "
    "  ELSE (TIMESTAMP '2000-01-01 00:00:00' + INTERVAL (i % 3600) SECOND)::TIMESTAMP_S "
    "  END AS k_ts_s, "
    "CASE WHEN i % 97 = 0 THEN NULL WHEN i % 1009 = 1 THEN TIMESTAMP_MS '1969-12-31 23:59:59.999' "
    "  WHEN i % 1009 = 2 THEN TIMESTAMP_MS '1969-12-31 23:59:59.998' "
    "  ELSE (TIMESTAMP '2000-01-01 00:00:00' + INTERVAL (i % 3600) SECOND)::TIMESTAMP_MS "
    "  END AS k_ts_ms, "
    "CASE WHEN i % 97 = 0 THEN NULL "
    "  WHEN i % 1009 = 1 THEN TIMESTAMP_NS '1969-12-31 23:59:59.999999999' "
    "  WHEN i % 1009 = 2 THEN TIMESTAMP_NS '1969-12-31 23:59:59.999999998' "
    "  ELSE (TIMESTAMP '2000-01-01 00:00:00' + INTERVAL (i % 3600) SECOND)::TIMESTAMP_NS "
    "  END AS k_ts_ns, "
    "CASE WHEN i % 11 = 0 THEN NULL ELSE i END::BIGINT AS v, (i % 1000 - 500)::INTEGER AS w "
    "FROM range(300000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  for (std::string const column : {"k_i8",
                                   "k_i16",
                                   "k_i32",
                                   "k_i64",
                                   "k_u8",
                                   "k_u16",
                                   "k_u32",
                                   "k_u64",
                                   "k_date",
                                   "k_ts",
                                   "k_ts_s",
                                   "k_ts_ms",
                                   "k_ts_ns"}) {
    INFO("key column " << column);
    on_off(
      subst("SELECT $1, count(*) AS c, count(v) AS cv, sum(v) AS s, min(v) AS mn, "
            "max(v) AS mx, sum(w) AS sw FROM kt GROUP BY $1",
            column));
    on_off_approx(
      subst("SELECT $1, avg(v) AS a, avg(w) AS aw FROM kt GROUP BY $1", column), {1, 2}, 1e-12);
    on_off_ordered(subst(
      "SELECT $1, count(*) AS c, max(w) AS mx FROM kt GROUP BY $1 ORDER BY $1 NULLS FIRST LIMIT 25",
      column));
    on_off_ordered(subst(
      "SELECT $1, min(v) AS mn FROM kt GROUP BY $1 ORDER BY $1 DESC NULLS LAST LIMIT 25", column));
  }
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - keys the table does not take keep working: other types, several keys",
                 AGG_TABLE_TAGS)
{
  run_ok(
    "CREATE TABLE rj AS SELECT (i % 5000 - 2500)::INTEGER AS k_i32, (i % 37)::SMALLINT AS k_i16, "
    "(i % 6000)::BIGINT AS k_i64, 'k' || (i % 977)::VARCHAR AS k_str, "
    "CASE WHEN i % 50 = 0 THEN NULL ELSE 'name-' || (i % 211)::VARCHAR END AS k_str_null, "
    "(i % 3 = 0) AS k_bool, ((i % 977)::DECIMAL(9,2) / 4)::DECIMAL(9,2) AS k_dec9, "
    "((i % 977)::DECIMAL(18,4) / 8)::DECIMAL(18,4) AS k_dec18, "
    "((i % 977) * 0.5)::DOUBLE AS k_f64, ((i % 977) * 0.5)::FLOAT AS k_f32, "
    "DATE '1995-01-01' + (i % 400)::INTEGER AS k_date, i::BIGINT AS v, (i * 0.25)::DOUBLE AS d "
    "FROM range(300000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};

  // Strings.
  on_off(
    "SELECT k_str, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM rj "
    "GROUP BY k_str",
    expected_path::legacy);
  on_off("SELECT k_str_null, count(*) AS c, sum(v) AS s FROM rj GROUP BY k_str_null",
         expected_path::legacy);

  // Several key columns, fixed-width or not.
  on_off(
    "SELECT k_i32, k_i16, count(*) AS c, sum(v) AS s FROM rj WHERE k_i32 % 8 = 0 "
    "GROUP BY k_i32, k_i16",
    expected_path::legacy);
  on_off("SELECT k_i64, k_date, count(*) AS c, min(v) AS mn FROM rj GROUP BY k_i64, k_date",
         expected_path::legacy);
  on_off("SELECT k_i16, k_str, count(*) AS c, max(v) AS mx FROM rj GROUP BY k_i16, k_str",
         expected_path::legacy);
  on_off("SELECT k_i16, k_bool, k_date, sum(v) AS s FROM rj GROUP BY ALL", expected_path::legacy);

  // Other single keys. Whether the table takes them is the engine's choice.
  on_off("SELECT k_bool, count(*) AS c, sum(v) AS s, min(v) AS mn FROM rj GROUP BY k_bool");
  on_off("SELECT k_dec9, count(*) AS c, sum(v) AS s FROM rj GROUP BY k_dec9");
  on_off("SELECT k_dec18, count(*) AS c, max(v) AS mx FROM rj GROUP BY k_dec18");
  on_off_canonical("SELECT k_f64, count(*) AS c, sum(v) AS s FROM rj GROUP BY k_f64");
  on_off_canonical("SELECT k_f32, count(*) AS c, min(v) AS mn FROM rj GROUP BY k_f32");

  // A key that is an expression, and a key under a cast.
  on_off("SELECT k_i32 + 1 AS k, count(*) AS c, sum(v) AS s FROM rj GROUP BY k_i32 + 1");
  on_off("SELECT k_i64::INTEGER AS k, count(*) AS c, sum(v) AS s FROM rj GROUP BY k_i64::INTEGER");
  on_off(
    "SELECT year(k_date) AS y, month(k_date) AS m, count(*) AS c FROM rj "
    "GROUP BY year(k_date), month(k_date)");

  // Grouping sets re-aggregate the partial rows for the coarser sets.
  on_off("SELECT k_i16, sum(v) AS s, count(*) AS c FROM rj GROUP BY ROLLUP(k_i16)");
}

TEST_CASE_METHOD(
  AggTableFixture,
  "agg table - aggregates the table does not take keep working next to an eligible key",
  AGG_TABLE_TAGS)
{
  run_ok(
    "CREATE TABLE mixed AS SELECT (i % 3000)::BIGINT AS k, i::BIGINT AS v, "
    "'s' || (i % 1013)::VARCHAR AS s, (i * 0.5)::DOUBLE AS d, "
    "DATE '1994-01-01' + (i % 900)::INTEGER AS dt, (i % 251)::SMALLINT AS j "
    "FROM range(300000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off("SELECT k, min(s) AS mn, max(s) AS mx FROM mixed GROUP BY k");
  on_off("SELECT k, sum(v) AS sv, min(s) AS mn, count(*) AS c FROM mixed GROUP BY k");
  on_off("SELECT k, min(dt) AS mn, max(dt) AS mx, count(dt) AS c FROM mixed GROUP BY k");
  on_off("SELECT k, count(DISTINCT j) AS dj FROM mixed GROUP BY k");
  on_off("SELECT k, count(DISTINCT j) AS dj, sum(v) AS sv FROM mixed GROUP BY k");
  on_off(
    "SELECT k, sum(v) FILTER (WHERE v % 2 = 0) AS se, count(*) FILTER (WHERE j > 100) AS c "
    "FROM mixed GROUP BY k");
  on_off_approx(
    "SELECT k, sum(d) AS sd, avg(d) AS ad, sum(v) AS sv FROM mixed GROUP BY k", {1, 2}, 1e-9);
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - plans the GPU rejects fall back to the CPU with the table path on",
                 AGG_TABLE_TAGS)
{
  // The fallback must not depend on the setting: these are rejected before any aggregate runs.
  run_ok(
    "CREATE TABLE nested AS SELECT [i % 5, i % 3] AS k_list, {'a': i % 5, 'b': i % 3} AS k_struct, "
    "(i % 977)::HUGEINT AS k_huge, TIME '00:00:00' + INTERVAL (i % 3600) SECOND AS k_time, "
    "to_days((i % 9)::INTEGER) AS k_interval, (i % 13)::INTEGER AS k, i::BIGINT AS v, "
    "(i * 0.5)::DOUBLE AS d, (i % 2 = 0) AS b FROM range(20000) r(i)");
  run_ok("CHECKPOINT");
  on_off_fallback("SELECT k_list, count(*) AS c, sum(v) AS s FROM nested GROUP BY k_list");
  on_off_fallback("SELECT k_struct, count(*) AS c, sum(v) AS s FROM nested GROUP BY k_struct");
  on_off_fallback("SELECT k_huge, count(*) AS c, sum(v) AS s FROM nested GROUP BY k_huge");
  on_off_fallback("SELECT k_time, count(*) AS c, sum(v) AS s FROM nested GROUP BY k_time");
  on_off_fallback("SELECT k_interval, count(*) AS c, sum(v) AS s FROM nested GROUP BY k_interval");
  on_off_fallback("SELECT k, bool_and(b) AS a, bool_or(b) AS o FROM nested GROUP BY k");
  on_off_fallback("SELECT k, stddev(d) AS sd FROM nested GROUP BY k");
}

//===----------------------------------------------------------------------===//
// Aggregate inputs
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - COUNT, SUM, MIN, MAX and AVG over every integer width",
                 AGG_TABLE_TAGS)
{
  // Group 13 holds only NULL values; every other group mixes NULLs with both signs.
  run_ok(
    "CREATE TABLE ints AS SELECT (i % 14)::INTEGER AS g, "
    "CASE WHEN i % 14 = 13 OR i % 7 = 0 THEN NULL ELSE i % 256 - 128 END::TINYINT AS a, "
    "CASE WHEN i % 14 = 13 OR i % 5 = 0 THEN NULL ELSE i % 65536 - 32768 END::SMALLINT AS b, "
    "CASE WHEN i % 14 = 13 OR i % 3 = 0 THEN NULL ELSE 2147483647 - i END::INTEGER AS c, "
    "CASE WHEN i % 14 = 13 OR i % 9 = 0 THEN NULL ELSE i * 1000003 - 150000000000 END::BIGINT AS "
    "d, "
    "CASE WHEN i % 14 = 13 THEN NULL ELSE i % 256 END::UTINYINT AS ua, "
    "CASE WHEN i % 14 = 13 THEN NULL ELSE i % 65536 END::USMALLINT AS ub, "
    "CASE WHEN i % 14 = 13 THEN NULL ELSE 4294967295 - i END::UINTEGER AS uc "
    "FROM range(300000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  for (std::string const column : {"a", "b", "c", "d", "ua", "ub", "uc"}) {
    INFO("value column " << column);
    on_off(
      subst("SELECT g, count(*) AS n, count($1) AS c, sum($1) AS s, min($1) AS mn, "
            "max($1) AS mx FROM ints GROUP BY g",
            column));
    on_off_approx(subst("SELECT g, avg($1) AS av FROM ints GROUP BY g", column), {1}, 1e-12);
  }
  // Every width in one query: twenty-one aggregates in one state row.
  on_off(
    "SELECT g, sum(a) AS sa, min(a) AS na, max(a) AS xa, sum(b) AS sb, min(b) AS nb, "
    "max(b) AS xb, sum(c) AS sc, min(c) AS nc, max(c) AS xc, sum(d) AS sd, min(d) AS n_d, "
    "max(d) AS xd, sum(ua) AS sua, min(ua) AS nua, max(ua) AS xua, sum(ub) AS sub, "
    "max(ub) AS xub, sum(uc) AS suc, min(uc) AS nuc, count(*) AS n FROM ints GROUP BY g");
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - state widths: one, two, four, eight and more aggregates",
                 AGG_TABLE_TAGS)
{
  // The table's kernels are specialised on 1, 2, 4 and 8 state words.
  run_ok(
    "CREATE TABLE widths AS SELECT (hash(i) % 20000)::BIGINT AS k, (i % 1000 - 500)::BIGINT AS v, "
    "(i % 91)::INTEGER AS w, (hash(i + 7) % 100000)::BIGINT AS x FROM range(600000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off("SELECT k FROM widths GROUP BY k");
  on_off("SELECT DISTINCT k FROM widths");
  on_off("SELECT k, count(*) AS c FROM widths GROUP BY k");
  on_off("SELECT k, min(v) AS mn FROM widths GROUP BY k");
  on_off("SELECT k, max(x) AS mx FROM widths GROUP BY k");
  on_off("SELECT k, sum(v) AS s, count(*) AS c FROM widths GROUP BY k");
  on_off("SELECT k, sum(v) AS s, min(v) AS mn, max(v) AS mx, count(*) AS c FROM widths GROUP BY k");
  on_off(
    "SELECT k, sum(v) AS s, min(v) AS mn, max(v) AS mx, count(v) AS c, sum(w) AS sw, "
    "min(w) AS mnw, max(w) AS mxw, count(*) AS n FROM widths GROUP BY k");
  on_off(
    "SELECT k, sum(v) AS s, min(v) AS mn, max(v) AS mx, count(v) AS c, sum(w) AS sw, "
    "min(w) AS mnw, max(w) AS mxw, sum(x) AS sx, min(x) AS mnx, max(x) AS mxx, "
    "count(x) AS cx, count(*) AS n FROM widths GROUP BY k");
  // The same aggregate twice, aggregates before the key, the key twice, and no key in the output.
  on_off("SELECT sum(v) AS s1, k, sum(v) AS s2, k AS k2, max(w) AS mx FROM widths GROUP BY k");
  on_off("SELECT sum(v) AS s, count(*) AS c FROM widths GROUP BY k");
  // Aggregates over expressions, and an aggregate inside an expression.
  on_off(
    "SELECT k, sum(v * 2 + w) AS s, max(v - w) AS mx, min(abs(v)) AS mn FROM widths "
    "GROUP BY k");
  on_off("SELECT k, sum(v) + count(*) AS sc, max(x) - min(x) AS spread FROM widths GROUP BY k");
  on_off(
    "SELECT k, sum(CASE WHEN w < 40 THEN v ELSE 0 END) AS low, "
    "count(CASE WHEN w >= 40 THEN 1 END) AS high FROM widths GROUP BY k");
  // A filter under the aggregate.
  on_off("SELECT k, sum(v) AS s, count(*) AS c FROM widths WHERE w < 30 AND v <> 0 GROUP BY k");
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - DECIMAL inputs of 7 to 18 digits, with NULLs and both signs",
                 AGG_TABLE_TAGS)
{
  run_ok(
    "CREATE TABLE decs AS SELECT (i % 9)::INTEGER AS g, "
    "CASE WHEN i % 13 = 0 THEN NULL ELSE ((i % 1999) - 999) / 10.0 END::DECIMAL(7,1) AS d7, "
    "CASE WHEN i % 11 = 0 THEN NULL ELSE ((i % 199999) - 99999) / 100.0 END::DECIMAL(9,2) AS d9, "
    "CASE WHEN i % 7 = 0 THEN NULL ELSE (i * 7919 - 1000000000) / 100.0 END::DECIMAL(15,2) AS d15, "
    "CASE WHEN i % 5 = 0 THEN NULL ELSE (i * 104729 - 15000000000) / 10000.0 "
    "  END::DECIMAL(18,4) AS d18 FROM range(300000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  for (std::string const column : {"d7", "d9", "d15", "d18"}) {
    INFO("value column " << column);
    on_off(
      subst("SELECT g, count($1) AS c, sum($1) AS s, min($1) AS mn, max($1) AS mx "
            "FROM decs GROUP BY g",
            column));
    on_off_approx(
      subst("SELECT g, avg($1) AS av, count(*) AS n FROM decs GROUP BY g", column), {1}, 1e-12);
  }
  on_off(
    "SELECT g, sum(d7) AS s7, sum(d9) AS s9, sum(d15) AS s15, sum(d18) AS s18, "
    "min(d9) AS mn9, max(d15) AS mx15 FROM decs GROUP BY g");
  // TPC-H's shape: a product of decimals summed per key.
  on_off(
    "SELECT g, sum(d9 * (1 - d7)) AS revenue, sum(d9 * (1 - d7) * (1 + d7)) AS charge "
    "FROM decs GROUP BY g");
}

TEST_CASE_METHOD(AggTableFixture, "agg table - DECIMAL sums that need 128 bits", AGG_TABLE_TAGS)
{
  // 9e15 with two decimals is 9e17 unscaled, so eleven rows already pass INT64. Group 0 only
  // grows, group 1 only shrinks, group 2 alternates signs and stays small, group 3 is small.
  run_ok(
    "CREATE TABLE dec_wide AS SELECT (i % 4)::INTEGER AS g, "
    "CASE i % 4 WHEN 0 THEN (9000000000000000.00 - i)::DECIMAL(18,2) "
    "  WHEN 1 THEN (-9000000000000000.00 + i)::DECIMAL(18,2) "
    "  WHEN 2 THEN (CASE WHEN (i // 4) % 2 = 0 THEN 8999999999999999.99::DECIMAL(18,2) "
    "    ELSE -8999999999999999.98::DECIMAL(18,2) END) "
    "  ELSE (i % 100000)::DECIMAL(18,2) / 100 END::DECIMAL(18,2) AS v "
    "FROM range(400000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off("SELECT g, sum(v) AS s FROM dec_wide GROUP BY g");
  on_off(
    "SELECT g, sum(v) AS s, min(v) AS mn, max(v) AS mx, count(*) AS c FROM dec_wide "
    "GROUP BY g");
  on_off_approx("SELECT g, avg(v) AS av FROM dec_wide GROUP BY g", {1}, 1e-12);
  on_off("SELECT sum(s) AS total FROM (SELECT g, sum(v) AS s FROM dec_wide GROUP BY g) t");
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - DOUBLE and FLOAT inputs: infinities, signed zeros and NULLs",
                 AGG_TABLE_TAGS)
{
  // One group per case. 0: ordinary values. 1: both infinities and a number. 2: only -0.0 and
  // +0.0, which are equal, so either may come back. 3: only NULL. 4: NULL, -inf and numbers.
  // 5: negative numbers. 6: only +inf. 7: one value in the first row, NULL after it.
  run_ok(
    "CREATE TABLE fl AS SELECT (i % 8)::INTEGER AS g, "
    "CASE i % 8 WHEN 0 THEN (i % 1000) * 0.5 "
    "  WHEN 1 THEN (CASE (i // 8) % 3 WHEN 0 THEN 'inf'::DOUBLE WHEN 1 THEN '-inf'::DOUBLE "
    "    ELSE 1.5 END) "
    "  WHEN 2 THEN (CASE WHEN (i // 8) % 2 = 0 THEN '-0.0'::DOUBLE ELSE 0.0 END) "
    "  WHEN 3 THEN NULL "
    "  WHEN 4 THEN (CASE (i // 8) % 5 WHEN 0 THEN NULL WHEN 2 THEN '-inf'::DOUBLE "
    "    ELSE -1.0 * i END) "
    "  WHEN 5 THEN -0.125 * (i % 100) - 1 "
    "  WHEN 6 THEN 'inf'::DOUBLE "
    "  ELSE (CASE WHEN i = 7 THEN 42.5 ELSE NULL END) END::DOUBLE AS x FROM range(300000) r(i)");
  run_ok("CREATE TABLE fl32 AS SELECT g, x::FLOAT AS y FROM fl");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off_canonical(
    "SELECT g, min(x) AS mn, max(x) AS mx, count(x) AS c, count(*) AS n FROM fl "
    "GROUP BY g");
  on_off_canonical(
    "SELECT g, min(y) AS mn, max(y) AS mx, count(y) AS c, count(*) AS n FROM fl32 "
    "GROUP BY g");
  on_off_canonical("SELECT g, max(x) AS mx FROM fl GROUP BY g");
  on_off_canonical("SELECT g, min(x) AS mn FROM fl GROUP BY g");
  // Sums of the special groups are an infinity, NaN (inf - inf), NULL or a zero, whatever the
  // order of addition.
  on_off_canonical("SELECT g, sum(x) AS s FROM fl WHERE g IN (1, 2, 3, 4, 6, 7) GROUP BY g");
  // Sums of the ordinary groups depend on the order of addition in the last bits.
  on_off_approx(
    "SELECT g, sum(x) AS s, avg(x) AS a FROM fl WHERE g IN (0, 5) GROUP BY g", {1, 2}, 1e-9);
  on_off_approx(
    "SELECT g, sum(y) AS s, avg(y) AS a FROM fl32 WHERE g IN (0, 5) GROUP BY g", {1, 2}, 1e-4);
  // MIN and MAX of integers next to MIN and MAX of doubles in one state row.
  on_off_canonical(
    "SELECT g, min(x) AS mn, max(x) AS mx, min(g) AS mng, max(g + 1) AS mxg, "
    "count(*) AS n FROM fl GROUP BY g");
}

// NaN in MIN and MAX. DuckDB orders NaN above every other value: MAX of a group that holds a NaN
// is NaN, and MIN is NaN only when the group holds nothing else. The table path must return
// exactly that (plan section 7.1). The existing cudf::groupby path does not: as of this
// writing it returns an arbitrary non-NaN member of such a group, a different one from run to
// run, for grouped and ungrouped MIN and MAX alike. `on_off_nan` therefore requires the CPU's
// rows only with the table path on and reports, without failing, what the path returns with it
// off. If the engine keeps float MIN / MAX off the table path, these checks fail for the
// existing defect and not for the table; that is the one place in this suite where the two
// cannot be told apart from SQL.
TEST_CASE_METHOD(AggTableFixture,
                 "agg table - NaN is the greatest value in DOUBLE and FLOAT MIN and MAX",
                 AGG_TABLE_TAGS "[agg_table_nan]")
{
  // 0: NaN among numbers. 1: only NaN. 2: NULL, NaN, -inf and numbers. 3: NaN and +inf.
  // 4: no NaN at all. 5: one NaN in the last row of the table, numbers before it.
  run_ok(
    "CREATE TABLE fn AS SELECT (i % 6)::INTEGER AS g, "
    "CASE i % 6 WHEN 0 THEN (CASE WHEN i % 12 = 0 THEN 'nan'::DOUBLE ELSE i * 0.25 END) "
    "  WHEN 1 THEN 'nan'::DOUBLE "
    "  WHEN 2 THEN (CASE (i // 6) % 5 WHEN 0 THEN NULL WHEN 1 THEN 'nan'::DOUBLE "
    "    WHEN 2 THEN '-inf'::DOUBLE ELSE -1.0 * i END) "
    "  WHEN 3 THEN (CASE WHEN (i // 6) % 2 = 0 THEN 'nan'::DOUBLE ELSE 'inf'::DOUBLE END) "
    "  WHEN 4 THEN i * 0.5 "
    "  ELSE (CASE WHEN i = 299999 THEN 'nan'::DOUBLE ELSE i % 1000 END) END::DOUBLE AS x "
    "FROM range(300000) r(i)");
  run_ok("CREATE TABLE fn32 AS SELECT g, x::FLOAT AS y FROM fn");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off_nan("SELECT g, min(x) AS mn, max(x) AS mx, count(x) AS c FROM fn GROUP BY g");
  on_off_nan("SELECT g, max(x) AS mx FROM fn GROUP BY g");
  on_off_nan("SELECT g, min(x) AS mn FROM fn GROUP BY g");
  on_off_nan("SELECT g, min(y) AS mn, max(y) AS mx, count(y) AS c FROM fn32 GROUP BY g");
  on_off_nan(
    "SELECT g, min(x) AS mn, max(x) AS mx, min(g) AS mng, count(*) AS n FROM fn "
    "GROUP BY g");
  // A sum that meets a NaN is NaN on every path.
  on_off_canonical("SELECT g, sum(x) AS s FROM fn WHERE g <> 4 GROUP BY g");
}

//===----------------------------------------------------------------------===//
// Keys the table reserves, and NULL
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - BIGINT keys -1 and -2 and NULL: hot, rare, late and in runs",
                 AGG_TABLE_TAGS)
{
  // -1 and -2 mark a free and a claimed slot in the table; a NULL key is carried as -1 plus the
  // validity bit. All three are groups like any other in SQL.
  run_ok(
    "CREATE TABLE rk_hot AS SELECT CASE WHEN i % 10 < 5 THEN -1 WHEN i % 10 = 5 THEN -2 "
    "  WHEN i % 10 = 6 THEN NULL WHEN i % 10 = 7 THEN i % 1000 WHEN i % 10 = 8 THEN -(i % 1000) - "
    "3 "
    "  WHEN i % 20 = 9 THEN 9223372036854775807 ELSE -9223372036854775808 END::BIGINT AS k, "
    "(i % 1000)::BIGINT AS v, CASE WHEN i % 3 = 0 THEN NULL ELSE i % 17 END::INTEGER AS w "
    "FROM range(600000) r(i)");
  // One row of each in 600000.
  run_ok(
    "CREATE TABLE rk_rare AS SELECT CASE i WHEN 17 THEN -1 WHEN 300000 THEN -2 WHEN 599999 THEN "
    "NULL "
    "  ELSE i % 5000 END::BIGINT AS k, (i % 1000)::BIGINT AS v, (i % 17)::INTEGER AS w "
    "FROM range(600000) r(i)");
  // Only in the last row group, after the table has seen four batches without them.
  run_ok(
    "CREATE TABLE rk_late AS SELECT CASE WHEN i < 590000 THEN i % 5000 WHEN i % 3 = 0 THEN -1 "
    "  WHEN i % 3 = 1 THEN -2 ELSE NULL END::BIGINT AS k, (i % 1000)::BIGINT AS v, "
    "(i % 17)::INTEGER AS w FROM range(600000) r(i)");
  // Runs of 50000 equal keys: -1, 7, -2, NULL, repeated.
  run_ok(
    "CREATE TABLE rk_runs AS SELECT CASE (i // 50000) % 4 WHEN 0 THEN -1 WHEN 1 THEN 7 "
    "  WHEN 2 THEN -2 ELSE NULL END::BIGINT AS k, (i % 1000)::BIGINT AS v, (i % 17)::INTEGER AS w "
    "FROM range(600000) r(i)");
  // Nothing but reserved keys.
  run_ok(
    "CREATE TABLE rk_only AS SELECT CASE i % 3 WHEN 0 THEN -1 WHEN 1 THEN -2 ELSE NULL "
    "  END::BIGINT AS k, (i % 1000)::BIGINT AS v, (i % 17)::INTEGER AS w FROM range(300000) r(i)");
  run_ok("CREATE TABLE rk_m1 AS SELECT (-1)::BIGINT AS k, v, w FROM rk_only");
  run_ok("CREATE TABLE rk_null AS SELECT NULL::BIGINT AS k, v, w FROM rk_only");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  for (std::string const table :
       {"rk_hot", "rk_rare", "rk_late", "rk_runs", "rk_only", "rk_m1", "rk_null"}) {
    INFO("table " << table);
    on_off(
      subst("SELECT k, count(*) AS c, count(w) AS cw, sum(v) AS s, min(v) AS mn, "
            "max(v) AS mx, sum(w) AS sw FROM $1 GROUP BY k",
            table));
    on_off(subst("SELECT k, count(*) AS c FROM $1 GROUP BY k", table));
    on_off_ordered(
      subst("SELECT k, sum(v) AS s FROM $1 GROUP BY k ORDER BY k NULLS FIRST LIMIT 10", table));
    on_off_approx(subst("SELECT k, avg(v) AS a FROM $1 GROUP BY k", table), {1}, 1e-12);
    on_off(subst("SELECT k, count(*) AS c FROM $1 GROUP BY k HAVING k IS NULL OR k < 0", table));
    on_off(
      subst("SELECT k IS NULL AS is_null, k = -1 AS is_m1, k = -2 AS is_m2, count(*) AS c, "
            "sum(v) AS s FROM (SELECT k, count(*) AS c, sum(v) AS v FROM $1 GROUP BY k) t "
            "GROUP BY ALL",
            table));
  }
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - NULL values: all-NULL groups, all-NULL columns, NULL keys",
                 AGG_TABLE_TAGS)
{
  run_ok(
    "CREATE TABLE nl AS SELECT CASE WHEN i % 6 = 5 THEN NULL ELSE i % 6 END::INTEGER AS g, "
    "CASE WHEN i % 6 IN (1, 5) THEN NULL WHEN i % 4 = 0 THEN NULL ELSE i END::BIGINT AS v, "
    "NULL::BIGINT AS all_null, NULL::DOUBLE AS all_null_d, "
    "CASE WHEN i % 6 = 2 AND i > 6 THEN NULL ELSE (i % 50)::DECIMAL(10,2) END AS dv "
    "FROM range(300000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  // Group 1 and the NULL group have only NULL values: SUM, MIN and MAX are NULL, COUNT is 0.
  on_off(
    "SELECT g, count(*) AS n, count(v) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx "
    "FROM nl GROUP BY g");
  on_off(
    "SELECT g, count(all_null) AS c, sum(all_null) AS s, min(all_null) AS mn, "
    "max(all_null) AS mx FROM nl GROUP BY g");
  on_off(
    "SELECT g, sum(all_null_d) AS s, min(all_null_d) AS mn, avg(all_null_d) AS a "
    "FROM nl GROUP BY g");
  on_off("SELECT g, avg(all_null) AS a, avg(v) IS NULL AS avg_is_null FROM nl GROUP BY g");
  // Group 2 has one non-NULL decimal, in its first row.
  on_off(
    "SELECT g, count(dv) AS c, sum(dv) AS s, min(dv) AS mn, max(dv) AS mx FROM nl "
    "GROUP BY g");
  on_off_approx("SELECT g, avg(v) AS a, avg(dv) AS ad FROM nl GROUP BY g", {1, 2}, 1e-12);
  on_off_ordered("SELECT g, count(v) AS c FROM nl GROUP BY g ORDER BY g NULLS FIRST");
  on_off_ordered("SELECT g, count(v) AS c FROM nl GROUP BY g ORDER BY g DESC NULLS LAST");
}

//===----------------------------------------------------------------------===//
// Integer sums past INT64
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - BIGINT sums past INT64 return the exact HUGEINT",
                 AGG_TABLE_TAGS)
{
  // Group 0 passes INT64_MAX within three rows, group 1 passes INT64_MIN, group 2 alternates
  // signs (every prefix of three same-sign rows overflows, the total is small), group 3 is small.
  run_ok(
    "CREATE TABLE ov AS SELECT (i % 4)::INTEGER AS g, "
    "CASE i % 4 WHEN 0 THEN 4000000000000000000 WHEN 1 THEN -4000000000000000000 "
    "  WHEN 2 THEN (CASE WHEN (i // 4) % 2 = 0 THEN 4000000000000000000 "
    "    ELSE -3999999999999999999 END) ELSE i % 1000 END::BIGINT AS x "
    "FROM range(400000) r(i)");
  // Totals exactly at, and one past, both ends of INT64.
  run_ok(
    "CREATE TABLE edges AS SELECT * FROM (VALUES (0, 4611686018427387904::BIGINT), "
    "(0, 4611686018427387903::BIGINT), (1, (-4611686018427387904)::BIGINT), "
    "(1, (-4611686018427387904)::BIGINT), (2, 9223372036854775807::BIGINT), (2, 1::BIGINT), "
    "(3, (-9223372036854775808)::BIGINT), (3, (-1)::BIGINT), (4, 9223372036854775807::BIGINT), "
    "(4, (-9223372036854775808)::BIGINT), (5, NULL::BIGINT), (5, 9223372036854775807::BIGINT), "
    "(5, 9223372036854775807::BIGINT)) t(g, x)");
  // Twelve INT64_MIN and eleven INT64_MAX per key.
  run_ok(
    "CREATE TABLE extremes AS SELECT (i % 3)::INTEGER AS g, "
    "CASE WHEN i % 23 < 12 THEN -9223372036854775808 ELSE 9223372036854775807 END::BIGINT AS x "
    "FROM range(69) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off("SELECT g, sum(x) AS s FROM ov GROUP BY g");
  on_off("SELECT g, sum(x) AS s, count(*) AS c, min(x) AS mn, max(x) AS mx FROM ov GROUP BY g");
  on_off_approx("SELECT g, avg(x) AS a, sum(x) AS s FROM ov GROUP BY g", {1}, 1e-12);
  on_off("SELECT g, sum(x) AS s FROM edges GROUP BY g");
  on_off_approx("SELECT g, avg(x) AS a FROM edges GROUP BY g", {1}, 1e-12);
  on_off("SELECT g, sum(x) AS s, count(x) AS c FROM extremes GROUP BY g");
  // A sum over the wide sums.
  on_off("SELECT sum(s) AS total FROM (SELECT g, sum(x) AS s FROM ov GROUP BY g) t");

  // The result type is HUGEINT whether or not this group overflowed.
  for (bool const enabled : {false, true}) {
    scoped_sirius_setting const table_path{*con, sirius::test::agg_table::kEnableSetting, enabled};
    auto result = con->Query("SELECT g, sum(x) AS s FROM edges GROUP BY g ORDER BY g");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->GetTypes()[1] == duckdb::LogicalType::HUGEINT);
    CHECK(result->Collection().GetValue(1, 0).ToString() == "9223372036854775807");
    CHECK(result->Collection().GetValue(1, 1).ToString() == "-9223372036854775808");
    CHECK(result->Collection().GetValue(1, 2).ToString() == "9223372036854775808");
    CHECK(result->Collection().GetValue(1, 3).ToString() == "-9223372036854775809");
    CHECK(result->Collection().GetValue(1, 4).ToString() == "-1");
    CHECK(result->Collection().GetValue(1, 5).ToString() == "18446744073709551614");
  }
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - BIGINT sums that fit in every batch and overflow across batches",
                 AGG_TABLE_TAGS)
{
  // 2e13 per row: a row group of 122880 rows sums to 2.5e18, inside INT64, but a key's 600000
  // rows reach 1.2e19. The bound breaks in the fifth batch, after the table has folded four
  // batches narrow. Key 1 alternates signs and stays small; key 2 holds one row per batch.
  run_ok(
    "CREATE TABLE creep AS SELECT 0::BIGINT AS k, 20000000000000::BIGINT AS x "
    "FROM range(600000) r(i)");
  run_ok(
    "CREATE TABLE creep3 AS SELECT CASE WHEN i % 100000 = 0 THEN 2 ELSE i % 2 END::BIGINT AS k, "
    "CASE WHEN i % 2 = 0 THEN 40000000000000 WHEN (i // 2) % 2 = 0 THEN 39999999999999 "
    "  ELSE -39999999999999 END::BIGINT AS x FROM range(1200000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off("SELECT k, sum(x) AS s FROM creep GROUP BY k");
  on_off("SELECT k, sum(x) AS s, count(*) AS c FROM creep3 GROUP BY k");
  on_off("SELECT k, sum(x) AS s, min(x) AS mn, max(x) AS mx, count(x) AS c FROM creep3 GROUP BY k");
  on_off_approx("SELECT k, avg(x) AS a FROM creep3 GROUP BY k", {1}, 1e-12);
  // The INT64 path where DuckDB's statistics prove the sum fits.
  run_ok(
    "CREATE TABLE small AS SELECT (i % 7)::BIGINT AS k, (i % 1000)::BIGINT AS x, "
    "(i % 1000)::INTEGER AS y FROM range(300000) r(i)");
  run_ok("CHECKPOINT");
  on_off("SELECT k, sum(x) AS s, sum(y) AS t FROM small GROUP BY k");
}

//===----------------------------------------------------------------------===//
// Input shapes
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - empty input, filtered-out input, one row and one group",
                 AGG_TABLE_TAGS)
{
  run_ok("CREATE TABLE empty_t (k BIGINT, v BIGINT, d DECIMAL(12,2), x DOUBLE)");
  run_ok(
    "CREATE TABLE filled AS SELECT (i % 100)::BIGINT AS k, i::BIGINT AS v, "
    "(i % 10)::DECIMAL(12,2) AS d, (i * 0.5)::DOUBLE AS x FROM range(300000) r(i)");
  run_ok("CREATE TABLE one_row AS SELECT 7::BIGINT AS k, 11::BIGINT AS v");
  run_ok(
    "CREATE TABLE one_group AS SELECT 5::BIGINT AS k, (i % 1000)::BIGINT AS v "
    "FROM range(500000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  // A grouped aggregate over no rows returns no rows.
  on_off(
    "SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM empty_t GROUP BY k");
  on_off("SELECT k, sum(d) AS s, avg(x) AS a FROM empty_t GROUP BY k");
  on_off("SELECT count(*) AS groups FROM (SELECT k FROM empty_t GROUP BY k) t");
  // Every row filtered out, in every batch, and in all but the last batch.
  on_off("SELECT k, count(*) AS c, sum(v) AS s FROM filled WHERE v < 0 GROUP BY k");
  on_off("SELECT k, count(*) AS c, sum(v) AS s FROM filled WHERE v >= 299990 GROUP BY k");
  on_off("SELECT k, count(*) AS c, sum(v) AS s FROM filled WHERE v = 150000 GROUP BY k");
  on_off(
    "SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM one_row GROUP BY k");
  on_off(
    "SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM one_group "
    "GROUP BY k");
  on_off_approx("SELECT k, avg(v) AS a FROM one_group GROUP BY k", {1}, 1e-12);
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - every key distinct: two million groups in sorted and in random order",
                 AGG_TABLE_TAGS)
{
  // Seventeen batches of new keys only: the table grows in every batch. `sorted` arrives in
  // ascending key order, which the existing sorted-keys path must keep taking.
  run_ok(
    "CREATE TABLE sorted AS SELECT (i * 7919 + 13)::BIGINT AS k, (i % 1000)::BIGINT AS v "
    "FROM range(2000000) r(i)");
  run_ok("CREATE TABLE shuffled AS SELECT k, v FROM sorted ORDER BY hash(k)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  for (std::string const table : {"sorted", "shuffled"}) {
    INFO("table " << table);
    on_off(
      subst("SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s, min(mn) AS mn, "
            "max(mx) AS mx, min(k) AS k0, max(k) AS k1, sum(k % 1000) AS ks FROM "
            "(SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM $1 "
            "GROUP BY k) t",
            table));
    on_off_ordered(
      subst("SELECT k, count(*) AS c, sum(v) AS s FROM $1 GROUP BY k ORDER BY k LIMIT 50", table));
    on_off_ordered(subst(
      "SELECT k, count(*) AS c, sum(v) AS s FROM $1 GROUP BY k ORDER BY k DESC LIMIT 50", table));
    on_off(subst("SELECT k, count(*) AS c, sum(v) AS s FROM $1 GROUP BY k HAVING k % 100003 = 0",
                 table));
    on_off(subst("SELECT count(*) AS c FROM (SELECT k FROM $1 GROUP BY k HAVING count(*) > 1) t",
                 table));
  }
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - many groups: a million keys seen twice, batches apart",
                 AGG_TABLE_TAGS)
{
  // Each key's two rows are a million rows apart, so the second meets a table that has grown
  // eight times since the first.
  run_ok(
    "CREATE TABLE twice AS SELECT (i % 1000000)::BIGINT AS k, i::BIGINT AS v, "
    "(i % 3)::INTEGER AS w FROM range(2000000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off(
    "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s, sum(mx - mn) AS spread, "
    "min(c) AS c0, max(c) AS c1 FROM (SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, "
    "max(v) AS mx FROM twice GROUP BY k) t");
  on_off_ordered(
    "SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, max(v) AS mx FROM twice "
    "GROUP BY k ORDER BY k LIMIT 100");
  on_off("SELECT k, sum(v) AS s, sum(w) AS sw FROM twice GROUP BY k HAVING k % 9973 = 5");
  // A small partition target makes the existing path exchange its partial rows over many
  // partitions; the table path emits into the same merge.
  scoped_sirius_setting const partitions{*con, kPartitionBytes, std::uint64_t{1} << 20};
  on_off(
    "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s FROM (SELECT k, count(*) AS c, "
    "sum(v) AS s FROM twice GROUP BY k) t");
  on_off("SELECT k, sum(v) AS s, sum(w) AS sw FROM twice GROUP BY k HAVING k % 9973 = 5");
}

TEST_CASE_METHOD(
  AggTableFixture,
  "agg table - heavy skew: one key on 99% of the rows, Zipf-like keys, a few hot keys",
  AGG_TABLE_TAGS)
{
  // Many threads of one launch update the same slot.
  run_ok(
    "CREATE TABLE one_hot AS SELECT CASE WHEN i % 100 = 0 THEN 100 + hash(i) % 20000 ELSE 42 "
    "  END::BIGINT AS k, (i % 1000)::BIGINT AS v, (hash(i) % 100000)::BIGINT AS x "
    "FROM range(2000000) r(i)");
  // u^8 over 50000 ids: key 0 holds about a quarter of the rows and the tail is long.
  run_ok(
    "CREATE TABLE zipf AS SELECT floor(50000 * pow((hash(i) % 1000003) / 1000003.0, 8))::BIGINT "
    "  AS k, (i % 1000)::BIGINT AS v, (hash(i + 1) % 100000)::BIGINT AS x "
    "FROM range(2000000) r(i)");
  // Sixteen keys on 90% of the rows over a uniform tail.
  run_ok(
    "CREATE TABLE hot16 AS SELECT CASE WHEN i % 10 = 0 THEN 1000 + hash(i) % 30000 "
    "  ELSE hash(i) % 16 END::BIGINT AS k, (i % 1000)::BIGINT AS v, "
    "(hash(i + 2) % 100000)::BIGINT AS x FROM range(2000000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  for (std::string const table : {"one_hot", "zipf", "hot16"}) {
    INFO("table " << table);
    on_off(subst("SELECT k, count(*) AS c FROM $1 GROUP BY k", table));
    on_off(
      subst("SELECT k, count(*) AS c, sum(v) AS s, min(x) AS mn, max(x) AS mx FROM $1 "
            "GROUP BY k",
            table));
    on_off(
      subst("SELECT k, sum(v) AS s, min(v) AS mnv, max(v) AS mxv, count(v) AS cv, "
            "sum(x) AS sx, min(x) AS mn, max(x) AS mx, count(*) AS c FROM $1 GROUP BY k",
            table));
    on_off_ordered(
      subst("SELECT k, count(*) AS c FROM $1 GROUP BY k ORDER BY c DESC, k LIMIT 20", table));
  }
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - keys in long runs: ascending, descending, repeating and with noise",
                 AGG_TABLE_TAGS)
{
  run_ok(
    "CREATE TABLE runs AS SELECT (i // 50000)::BIGINT AS up, (29 - i // 50000)::BIGINT AS down, "
    "((i // 50000) % 5)::BIGINT AS again, "
    "CASE WHEN i % 1000 = 0 THEN 100 + hash(i) % 100 ELSE i // 50000 END::BIGINT AS noisy, "
    "(i // 700)::BIGINT AS short_runs, (i % 1000)::BIGINT AS v, "
    "(hash(i) % 100000)::BIGINT AS x FROM range(1500000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  for (std::string const column : {"up", "down", "again", "noisy", "short_runs"}) {
    INFO("key column " << column);
    on_off(subst("SELECT $1, count(*) AS c, sum(v) AS s FROM runs GROUP BY $1", column));
    on_off(
      subst("SELECT $1, count(*) AS c, sum(v) AS s, min(x) AS mn, max(x) AS mx FROM runs "
            "GROUP BY $1",
            column));
    on_off(
      subst("SELECT $1, sum(v) AS s, min(v) AS mnv, max(v) AS mxv, count(v) AS cv, "
            "sum(x) AS sx, min(x) AS mn, max(x) AS mx, count(*) AS c FROM runs GROUP BY $1",
            column));
  }
}

//===----------------------------------------------------------------------===//
// Query shapes around the aggregate
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - HAVING, ORDER BY and LIMIT over the aggregate",
                 AGG_TABLE_TAGS)
{
  run_ok(
    "CREATE TABLE sales AS SELECT CASE WHEN i % 500 = 0 THEN NULL ELSE hash(i) % 30000 "
    "  END::BIGINT AS k, (i % 1000 - 300)::BIGINT AS v, ((i % 5000) / 100.0)::DECIMAL(12,2) AS p "
    "FROM range(1000000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  on_off("SELECT k, sum(v) AS s FROM sales GROUP BY k HAVING sum(v) > 9000");
  on_off("SELECT k, sum(v) AS s FROM sales GROUP BY k HAVING count(*) < 25 AND min(v) >= -250");
  on_off("SELECT k, count(*) AS c FROM sales GROUP BY k HAVING max(p) - min(p) < 45.00");
  on_off("SELECT k FROM sales GROUP BY k HAVING sum(v) IS NOT NULL AND k % 7 = 3");
  on_off_ordered("SELECT k, sum(v) AS s FROM sales GROUP BY k ORDER BY s DESC, k LIMIT 40");
  on_off_ordered(
    "SELECT k, sum(p) AS s, count(*) AS c FROM sales GROUP BY k "
    "ORDER BY c, s DESC, k NULLS FIRST LIMIT 40");
  on_off_ordered("SELECT k, max(v) AS mx FROM sales GROUP BY k ORDER BY k NULLS FIRST LIMIT 30");
  on_off_ordered(
    "SELECT k, max(v) AS mx FROM sales GROUP BY k ORDER BY k DESC NULLS FIRST "
    "LIMIT 30 OFFSET 17");
  on_off_ordered(
    "SELECT k, sum(v) AS s FROM sales GROUP BY k HAVING sum(v) < 0 "
    "ORDER BY s, k LIMIT 15");
  on_off_ordered("SELECT k, sum(v) AS s FROM sales GROUP BY k ORDER BY k NULLS LAST");
  on_off("SELECT count(*) AS groups FROM (SELECT k FROM sales GROUP BY k HAVING sum(v) > 0) t");
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - the aggregate under joins, subqueries, CTEs and unions",
                 AGG_TABLE_TAGS)
{
  run_ok(
    "CREATE TABLE fact AS SELECT (hash(i) % 40000)::BIGINT AS fk, (i % 200000)::BIGINT AS ok, "
    "(i % 50 + 1)::BIGINT AS qty, ((i % 9000) / 100.0 + 1)::DECIMAL(12,2) AS price "
    "FROM range(1000000) r(i)");
  run_ok(
    "CREATE TABLE dim AS SELECT i::BIGINT AS pk, (i % 25)::INTEGER AS region, "
    "(i % 7 = 0) AS flagged FROM range(40000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};

  // Join below the aggregate: group by the probe key, and by a build-side column.
  on_off(
    "SELECT f.fk, sum(f.qty) AS q, count(*) AS c FROM fact f JOIN dim d ON f.fk = d.pk "
    "WHERE d.flagged GROUP BY f.fk");
  on_off(
    "SELECT d.region, sum(f.qty) AS q, sum(f.price) AS p, count(*) AS c FROM fact f "
    "JOIN dim d ON f.fk = d.pk GROUP BY d.region");
  // Aggregate below the join, on either side, and on both.
  on_off(
    "SELECT d.region, sum(t.q) AS q, count(*) AS c FROM (SELECT fk, sum(qty) AS q FROM fact "
    "GROUP BY fk) t JOIN dim d ON t.fk = d.pk GROUP BY d.region");
  on_off(
    "SELECT a.fk, a.q, b.c FROM (SELECT fk, sum(qty) AS q FROM fact GROUP BY fk) a "
    "JOIN (SELECT fk, count(*) AS c FROM fact WHERE qty > 25 GROUP BY fk) b ON a.fk = b.fk "
    "WHERE a.fk % 997 = 0");
  // TPC-H Q13: an aggregate over an outer join, then a count of the counts.
  on_off(
    "SELECT c, count(*) AS dist FROM (SELECT d.pk, count(f.ok) AS c FROM dim d "
    "LEFT JOIN fact f ON d.pk = f.fk AND f.qty > 45 GROUP BY d.pk) t GROUP BY c");
  // TPC-H Q15: the maximum of a grouped sum, joined back to the groups.
  on_off(
    "WITH revenue AS (SELECT fk, sum(price * qty) AS total FROM fact GROUP BY fk) "
    "SELECT d.pk, d.region, r.total FROM dim d JOIN revenue r ON d.pk = r.fk "
    "WHERE r.total = (SELECT max(total) FROM revenue)");
  // TPC-H Q17: rows below a fraction of their own key's average.
  on_off_approx(
    "SELECT sum(f.price) / 7.0 AS avg_yearly FROM fact f JOIN dim d ON f.fk = d.pk "
    "WHERE d.region = 3 AND f.qty < (SELECT 0.2 * avg(f2.qty) FROM fact f2 "
    "WHERE f2.fk = f.fk)",
    {0},
    1e-9);
  // TPC-H Q18: keys whose grouped sum passes a threshold, as an IN list for another aggregate.
  on_off_ordered(
    "SELECT f.ok, sum(f.qty) AS q FROM fact f WHERE f.ok IN (SELECT ok FROM fact "
    "GROUP BY ok HAVING sum(qty) > 200) GROUP BY f.ok ORDER BY q DESC, f.ok LIMIT 100");
  // A scalar subquery over a grouped subquery.
  on_off(
    "SELECT count(*) AS above FROM (SELECT fk, sum(qty) AS q FROM fact GROUP BY fk) t "
    "WHERE q > (SELECT avg(q) FROM (SELECT fk, sum(qty) AS q FROM fact GROUP BY fk) u)");
  // Two aggregates of one CTE, and a union of two aggregates.
  on_off(
    "WITH per_key AS MATERIALIZED (SELECT fk, sum(qty) AS q, count(*) AS c FROM fact "
    "GROUP BY fk) SELECT a.fk, a.q, b.c FROM per_key a JOIN per_key b ON a.fk = b.fk "
    "WHERE a.fk % 1009 = 1");
  on_off(
    "SELECT fk AS k, sum(qty) AS q FROM fact WHERE fk < 100 GROUP BY fk "
    "UNION ALL SELECT ok AS k, sum(qty) AS q FROM fact WHERE ok < 100 GROUP BY ok");
  // EXISTS over a grouped subquery.
  on_off(
    "SELECT d.region, count(*) AS c FROM dim d WHERE EXISTS (SELECT 1 FROM fact f "
    "WHERE f.fk = d.pk GROUP BY f.fk HAVING sum(f.qty) > 700) GROUP BY d.region");
}

//===----------------------------------------------------------------------===//
// A table too small for the groups
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - a budget smaller than the groups: unapplied rows take the old path",
                 AGG_TABLE_TAGS)
{
  sirius::test::agg_table::require_enable_setting(*con);
  if (!setting_exists(*con, kBudgetSetting)) {
    SKIP("setting '" << kBudgetSetting
                     << "' is not registered; set kBudgetSetting in "
                        "utils/agg_table_test_utils.hpp to the engine's name for the table budget");
  }
  // A million groups of two rows a million rows apart (the table cannot hold the first copy of a
  // key until the second arrives), and a skewed input whose hot keys fit while the tail does not.
  run_ok(
    "CREATE TABLE twice AS SELECT (i % 1000000)::BIGINT AS k, i::BIGINT AS v, "
    "(i % 3)::INTEGER AS w FROM range(2000000) r(i)");
  run_ok(
    "CREATE TABLE hot16 AS SELECT CASE WHEN i % 10 = 0 THEN 1000 + hash(i) % 300000 "
    "  ELSE hash(i) % 16 END::BIGINT AS k, (i % 1000)::BIGINT AS v, (i % 3)::INTEGER AS w "
    "FROM range(2000000) r(i)");
  run_ok(
    "CREATE TABLE special AS SELECT CASE WHEN i % 50 = 0 THEN -1 WHEN i % 50 = 1 THEN -2 "
    "  WHEN i % 50 = 2 THEN NULL ELSE hash(i) % 400000 END::BIGINT AS k, "
    "CASE WHEN i % 4 = 0 THEN 4000000000000000000 ELSE i % 1000 END::BIGINT AS v, "
    "(i % 3)::INTEGER AS w FROM range(1000000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  // From "holds a fraction of the groups" down to "holds nothing": 16 MiB, 1 MiB, 64 KiB, 1 byte.
  for (std::uint64_t const budget : {std::uint64_t{16} << 20,
                                     std::uint64_t{1} << 20,
                                     std::uint64_t{64} << 10,
                                     std::uint64_t{1}}) {
    INFO(kBudgetSetting << " = " << budget);
    scoped_sirius_setting const table_budget{*con, kBudgetSetting, budget};
    on_off(
      "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s, sum(mx - mn) AS spread, "
      "sum(sw) AS sw FROM (SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn, "
      "max(v) AS mx, sum(w) AS sw FROM twice GROUP BY k) t");
    on_off("SELECT k, count(*) AS c, sum(v) AS s FROM twice GROUP BY k HAVING k % 9973 = 5");
    on_off(
      "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s, max(mx) AS mx FROM "
      "(SELECT k, count(*) AS c, sum(v) AS s, max(v) AS mx FROM hot16 GROUP BY k) t");
    on_off_ordered(
      "SELECT k, count(*) AS c, sum(v) AS s, min(v) AS mn FROM hot16 GROUP BY k "
      "ORDER BY c DESC, k LIMIT 40");
    on_off(
      "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s FROM (SELECT k, count(*) AS c, "
      "sum(v) AS s FROM special GROUP BY k) t");
    on_off(
      "SELECT k, count(*) AS c, sum(v) AS s, min(w) AS mn FROM special GROUP BY k "
      "HAVING k IS NULL OR k < 0 OR k % 4999 = 0");
  }
}

TEST_CASE_METHOD(AggTableFixture,
                 "agg table - a table that stops absorbing is flushed at a batch boundary",
                 AGG_TABLE_TAGS)
{
  sirius::test::agg_table::require_enable_setting(*con);
  if (!setting_exists(*con, kBudgetSetting)) {
    SKIP("setting '" << kBudgetSetting << "' is not registered");
  }
  // Three million distinct keys in random order, 25 batches, and a 16 MiB table: the first
  // batches fill it, every later one is handed back whole, and the engine gives the table up.
  // Its rows then leave it as the output of an ordinary task, in the middle of the input, and
  // the batches after that take the old path.
  run_ok(
    "CREATE TABLE distinct3m AS SELECT ((i * 7919) % 3000017)::BIGINT AS k, (i % 1000)::BIGINT "
    "AS v FROM range(3000000) r(i)");
  run_ok("CHECKPOINT");
  scoped_sirius_setting const scan_batches{*con, kScanBatch, std::uint64_t{1}};
  scoped_sirius_setting const table_budget{*con, kBudgetSetting, std::uint64_t{16} << 20};
  auto const before = sirius::test::agg_table::table_stats();
  on_off(
    "SELECT count(*) AS groups, sum(c) AS n, sum(s) AS s, sum(k) AS sk FROM (SELECT k, "
    "count(*) AS c, sum(v) AS s FROM distinct3m GROUP BY k) t");
  on_off("SELECT k, count(*) AS c, sum(v) AS s FROM distinct3m GROUP BY k HAVING k % 9973 = 5");
  auto const delta = sirius::test::agg_table::table_stats().since(before);
  INFO("folded batches " << delta.folded_batches << ", boundary flushes " << delta.boundary_flushes
                         << ", batches on the old path " << delta.legacy_batches);
  CHECK(delta.folded_batches > 0);
  CHECK(delta.boundary_flushes > 0);
  CHECK(delta.legacy_batches > 0);
}
