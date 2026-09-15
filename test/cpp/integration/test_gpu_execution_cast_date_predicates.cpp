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

// GPU-vs-CPU correctness for cast-shaped DATE predicates — the shapes DuckDB's
// constant folding produces for qgen-style date arithmetic:
//
//   d <= DATE '1998-12-01' - INTERVAL '72' DAY
//     ⇒ table filter  CAST(d AS TIMESTAMP) <= TIMESTAMP '1998-09-20 00:00:00'
//
// These arrive as EXPRESSION_FILTERs, which scan_filter_analysis.cpp lowers to
// stored-day bounds. An off-by-one bound silently changes results, so every
// comparison op runs at midnight and non-midnight constants against rows
// sitting exactly on the cutoffs, on three scan paths:
//
//  * plain DuckDB table and DuckDB-format GPU pin: the filter is evaluated by
//    the residual GPU cast (cudf::cast), the fold is not consulted;
//  * Parquet GPU pin compressed with a bitpack plan: the only path on which
//    analyze_scan_filters runs AND the decoder applies its ranges while
//    decoding, dropping the residual when coverage is full. This is the path
//    the change is for, and the one whose bounds these tests actually exercise.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/pinned_entry_census.hpp>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Arm the gate before any test latches its function-local static, without
// overriding an explicit setting. The fused path's contract is byte-identical
// output, so this is behavior-neutral for the rest of the binary.
struct fused_gate_armer {
  fused_gate_armer()
  {
    setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", /*overwrite=*/0);
    // The fused decode hands compaction back to the ordinary decode + residual
    // filter once a range keeps more than 35% (10% on the full route) of a
    // chunk's rows. These tables are a handful of rows on purpose, so nearly
    // every predicate is above that; lift the caps so the in-decode mask, whose
    // bounds are under test, is what actually answers.
    setenv("SIRIUS_EXP_FUSED_SCAN_MAX_SEL", "1", /*overwrite=*/0);
    setenv("SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL", "1", /*overwrite=*/0);
  }
};
[[maybe_unused]] fused_gate_armer const arm_fused_gate{};

// Rows sit exactly on / around every cutoff used below: the qgen q1 cutoff
// 1998-09-20 (= DATE '1998-12-01' - 72 days), the 1994 year-range bounds, the
// epoch boundary (negative stored days), plus NULLs and duplicates.
class CastDatePredicateFixture : public sirius::test::GpuExecutionFixture {
 public:
  CastDatePredicateFixture()
  {
    run_ok("CREATE TABLE t (id INTEGER, d DATE);");
    run_ok(
      "INSERT INTO t VALUES "
      "(1,  DATE '1998-09-19'),"  // cutoff - 1
      "(2,  DATE '1998-09-20'),"  // exactly the q1 cutoff
      "(3,  DATE '1998-09-20'),"  // duplicate on the cutoff
      "(4,  DATE '1998-09-21'),"  // cutoff + 1
      "(5,  DATE '1998-12-01'),"
      "(6,  DATE '1993-12-31'),"   // year-range lower bound - 1
      "(7,  DATE '1994-01-01'),"   // year-range lower bound
      "(8,  DATE '1994-12-31'),"   // last day inside the year range
      "(9,  DATE '1995-01-01'),"   // year-range upper bound (excluded by <)
      "(10, DATE '1969-12-31'),"   // stored day -1: pre-epoch floor/ceil sides
      "(11, DATE '1970-01-01'),"   // stored day 0
      "(12, NULL),"                // must be dropped by every predicate
      "(13, DATE '2262-04-11');")  // far future, still castable to every flavor
      ;
    // DATE ±infinity rows. DuckDB casts these straight to TIMESTAMP ±infinity
    // rather than through day arithmetic; the day-domain bound must still
    // agree with it (see lower_timestamp_to_days). Kept in a separate table so
    // the main one's stored-day span stays narrow and its pinned bitpack plan
    // stays representative.
    run_ok("CREATE TABLE t_inf (id INTEGER, d DATE);");
    run_ok(
      "INSERT INTO t_inf VALUES "
      "(1, DATE '-infinity'),"
      "(2, DATE '1969-12-31'),"
      "(3, DATE '1998-09-20'),"
      "(4, DATE '1998-09-21'),"
      "(5, NULL),"
      "(6, DATE '2262-04-11'),"
      "(7, DATE 'infinity');");
    // A finite date whose midnight overflows TIMESTAMP's int64 microseconds
    // (year 300000 > the ~292,277-year limit). DuckDB raises on the cast.
    run_ok("CREATE TABLE t_far (id INTEGER, d DATE);");
    run_ok(
      "INSERT INTO t_far VALUES "
      "(1, DATE '2000-01-01'),"
      "(2, DATE '300000-01-01');");
    run_ok("CHECKPOINT;");

    // Parquet twins of the three tables, each behind a view of the same-named
    // pinned entry, so `FROM p_*` resolves to the pinned Parquet scan. Without
    // the NULL row: the bitpack compressor refuses nullable input, and an
    // uncompressed pin would silently fall back to the residual path.
    for (auto const* name : {"t", "t_inf", "t_far"}) {
      auto const pq = dir_.file(std::string(name) + ".parquet");
      run_ok("COPY (SELECT * FROM " + std::string(name) + " WHERE d IS NOT NULL) TO '" + pq +
             "' (FORMAT PARQUET);");
      run_ok("CREATE VIEW p_" + std::string(name) + " AS SELECT * FROM read_parquet('" + pq +
             "');");
      // One bitpack block per column (id, d) in schema order.
      std::ofstream plan(dir_.file("p_" + std::string(name) + ".txt"));
      plan << "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n---\n"
              "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
    }
  }

  ~CastDatePredicateFixture() { con->Query("SET pin_table_compression = false;"); }

  /// GPU-pin the Parquet twin of @p table under a bitpack plan and prove the
  /// pin really compressed, since an uncompressed entry would silently take
  /// the residual path and make the assertions vacuous.
  void pin_compressed_parquet(const std::string& table)
  {
    auto const name = "p_" + table;
    run_ok("SET pin_table_compression = true;");
    run_ok("SET pin_table_compression_min_batch_size_bytes = 0;");
    // A handful of rows never shrinks under bitpack headers; keep the
    // compressed form regardless of size so the decode path is exercised.
    run_ok("SET pin_table_compression_max_compressed_fraction = 1000;");
    run_ok("SET pin_table_input_compression_plan_dir = '" + dir_.path().string() + "';");
    run_ok("CALL pin_table('" + dir_.file(table + ".parquet") + "', tier='gpu', name='" + name +
           "');");
    auto const census = sirius::test::census_entry(*con, name);
    REQUIRE(census.chunks > 0);
    REQUIRE(census.compressed_chunks == census.chunks);
  }

  void unpin_parquet(const std::string& table) { run_ok("CALL unpin_table('p_" + table + "');"); }

  void compare_all(const std::vector<std::string>& predicates, const std::string& table = "t")
  {
    for (const auto& pred : predicates) {
      DYNAMIC_SECTION(pred)
      {
        compare_gpu_vs_cpu("SELECT id FROM " + table + " WHERE " + pred);
        // Aggregate shape too: a wrong decode-time bound that only miscounts
        // (rather than mis-selects ids) would still show here.
        compare_gpu_vs_cpu("SELECT count(*), sum(id) FROM " + table + " WHERE " + pred);
      }
    }
  }

  /// The query raises on CPU (DuckDB refuses to cast the far date) but the GPU,
  /// which cannot raise from a decode-time range and whose residual cast would
  /// not raise either, answers by the instant each date denotes. Documents that
  /// divergence rather than hiding it.
  void expect_cpu_raises_gpu_answers(const std::string& query,
                                     const std::vector<std::vector<std::string>>& expected_rows)
  {
    run_ok("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    if (!cpu_result->HasError()) {
      UNSCOPED_INFO("expected DuckDB to raise a conversion error on the far date");
    }
    REQUIRE(cpu_result->HasError());

    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu_result   = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(gpu_result);
    if (gpu_result->HasError()) {
      UNSCOPED_INFO("transparent GPU execution error: " << gpu_result->GetError());
    }
    REQUIRE_FALSE(gpu_result->HasError());
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    auto rows = collect_rows(gpu_result->Cast<duckdb::MaterializedQueryResult>(), true);
    CHECK(rows == expected_rows);
  }

 private:
  sirius::test::scratch_dir dir_{"cast_date"};
};

// Every comparison op, midnight and non-midnight, both operand orders, plus
// the folded-arithmetic originals. Written as SQL text so DuckDB itself
// performs the constant folding that produces the cast-shaped table filters.
const std::vector<std::string> kFoldedPredicates = {
  // qgen q1 shape (folds to CAST(d AS TIMESTAMP) <= TIMESTAMP '1998-09-20 00:00:00')
  "d <= DATE '1998-12-01' - INTERVAL '72' DAY",
  "d <  DATE '1998-12-01' - INTERVAL '72' DAY",
  "d >= DATE '1998-12-01' - INTERVAL '72' DAY",
  "d >  DATE '1998-12-01' - INTERVAL '72' DAY",
  "d =  DATE '1998-12-01' - INTERVAL '72' DAY",
  // explicit midnight timestamp literal
  "d <= TIMESTAMP '1998-09-20 00:00:00'",
  "d <  TIMESTAMP '1998-09-20 00:00:00'",
  "d >= TIMESTAMP '1998-09-20 00:00:00'",
  "d >  TIMESTAMP '1998-09-20 00:00:00'",
  "d =  TIMESTAMP '1998-09-20 00:00:00'",
  // non-midnight constants: the instant falls strictly between two days
  "d <= TIMESTAMP '1998-09-20 12:00:00'",
  "d <  TIMESTAMP '1998-09-20 12:00:00'",
  "d >= TIMESTAMP '1998-09-20 12:00:00'",
  "d >  TIMESTAMP '1998-09-20 12:00:00'",
  "d =  TIMESTAMP '1998-09-20 12:00:00'",  // can never match a DATE: constant false
  "d <  TIMESTAMP '1998-09-20 00:00:00.000001'",
  "d >= TIMESTAMP '1998-09-20 00:00:00.000001'",
  "d <= TIMESTAMP '1998-09-19 23:59:59.999999'",
  "d >  TIMESTAMP '1998-09-19 23:59:59.999999'",
  // constant on the left
  "TIMESTAMP '1998-09-20 00:00:00' >= d",
  "TIMESTAMP '1998-09-20 00:00:00' <  d",
  "TIMESTAMP '1998-09-20 12:00:00' >  d",
  "TIMESTAMP '1998-09-20 12:00:00' =  d",
  // pre-epoch boundaries (stored day -1 / 0)
  "d <= TIMESTAMP '1969-12-31 00:00:00'",
  "d <  TIMESTAMP '1969-12-31 00:00:00.000001'",
  "d >= TIMESTAMP '1969-12-31 00:00:00.000001'",
  "d >= TIMESTAMP '1969-12-31 12:00:00'",
  // q6-style year range (both conjuncts fold to cast comparisons)
  "d >= DATE '1994-01-01' AND d < DATE '1994-01-01' + INTERVAL '1' YEAR",
  "d BETWEEN DATE '1994-01-01' AND DATE '1998-12-01' - INTERVAL '72' DAY",
  // interval months (q4/q10/q14/q15/q20 shapes)
  "d >= DATE '1994-01-01' AND d < DATE '1994-01-01' + INTERVAL '3' MONTH",
  // plain DATE constants for contrast (ConstantFilter path, already fused)
  "d <= DATE '1998-09-20'",
  "d =  DATE '1998-09-20'",
};

// Against t_inf: finite constants must sort the ±infinity rows to the right
// side. Correct on every path, since CAST(DATE ±infinity) is ±infinity on CPU
// and the fold's day domain keeps ±INT32_MAX beyond every finite bound.
const std::vector<std::string> kFiniteConstantsOnInfinityRows = {
  "d <= TIMESTAMP '1998-09-20 00:00:00'",
  "d <  TIMESTAMP '1998-09-20 00:00:00'",
  "d >= TIMESTAMP '1998-09-20 00:00:00'",
  "d >  TIMESTAMP '1998-09-20 00:00:00'",
  "d =  TIMESTAMP '1998-09-20 00:00:00'",
  "d >= TIMESTAMP '1998-09-20 12:00:00'",
  "d <  TIMESTAMP '1998-09-20 12:00:00'",
  "d <= DATE '1998-12-01' - INTERVAL '72' DAY",
  "d <= TIMESTAMP_NS '1998-09-20 00:00:00'",
};

// ±infinity constants lower to the infinite dates' stored days (`< 'infinity'`
// keeps everything but +infinity, `>= 'infinity'` only it, ...). Run on the
// fold path only: the residual cudf::cast that the other paths evaluate
// multiplies ±INT32_MAX days into int64 micros, which wraps, so those paths
// mis-answer these predicates today (pre-existing, independent of the fold).
const std::vector<std::string> kInfinityConstants = {
  // +infinity constants
  "d <  TIMESTAMP 'infinity'",
  "d <= TIMESTAMP 'infinity'",
  "d >= TIMESTAMP 'infinity'",
  "d >  TIMESTAMP 'infinity'",
  "d =  TIMESTAMP 'infinity'",
  "TIMESTAMP 'infinity' > d",
  // -infinity constants
  "d >  TIMESTAMP '-infinity'",
  "d >= TIMESTAMP '-infinity'",
  "d <= TIMESTAMP '-infinity'",
  "d <  TIMESTAMP '-infinity'",
  "d =  TIMESTAMP '-infinity'",
  // (Not `d > '-infinity' AND d < 'infinity'`: DuckDB merges two cast
  // comparisons on one column into a single BETWEEN expression, a shape the
  // fold does not recognize, so it takes the residual path.)
  // other flavors (every finite row of t_inf fits TIMESTAMP_NS's range)
  "d <  TIMESTAMP_NS 'infinity'",
  "d >= TIMESTAMP_NS '-infinity'",
  "d <= TIMESTAMP_S 'infinity'",
  // (TIMESTAMP_MS '-infinity' is left out: DuckDB itself fails to bind that
  // constant, "Could not convert Timestamp(MS) to Timestamp(US)".)
};

}  // namespace

TEST_CASE_METHOD(CastDatePredicateFixture,
                 "gpu_execution cast-shaped DATE predicates match CPU (plain scan)",
                 "[integration][gpu_execution][filter][fused_scan_filter][cast_date]")
{
  compare_all(kFoldedPredicates);
}

TEST_CASE_METHOD(CastDatePredicateFixture,
                 "gpu_execution cast-shaped DATE predicates match CPU (gpu-pinned duckdb table)",
                 "[integration][gpu_execution][filter][fused_scan_filter][cast_date][pin_table]")
{
  // The DuckDB-native ingestible does not analyze its filter, so this pin
  // still evaluates the residual GPU cast; kept as coverage of that path.
  run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu');");
  compare_all(kFoldedPredicates);
  run_ok("CALL unpin_table('t');");
}

TEST_CASE_METHOD(CastDatePredicateFixture,
                 "gpu_execution cast-shaped DATE predicates match CPU (compressed parquet pin)",
                 "[integration][gpu_execution][filter][fused_scan_filter][cast_date][pin_table]")
{
  // Pinned Parquet entries route through sirius_scan_manager's pushdown
  // request and, with the column bitpacked, through the fused in-decode
  // masking whose bounds this change produces.
  pin_compressed_parquet("t");
  compare_all(kFoldedPredicates, "p_t");
  unpin_parquet("t");
}

TEST_CASE_METHOD(CastDatePredicateFixture,
                 "gpu_execution cast-shaped DATE predicates match CPU on ±infinity dates",
                 "[integration][gpu_execution][filter][fused_scan_filter][cast_date]")
{
  SECTION("plain scan") { compare_all(kFiniteConstantsOnInfinityRows, "t_inf"); }
  SECTION("gpu-pinned duckdb table")
  {
    run_ok("CALL pin_table(format='duckdb', name='t_inf', tier='gpu');");
    compare_all(kFiniteConstantsOnInfinityRows, "t_inf");
    run_ok("CALL unpin_table('t_inf');");
  }
  SECTION("compressed parquet pin")
  {
    pin_compressed_parquet("t_inf");
    compare_all(kFiniteConstantsOnInfinityRows, "p_t_inf");
    compare_all(kInfinityConstants, "p_t_inf");
    unpin_parquet("t_inf");
  }
}

TEST_CASE_METHOD(CastDatePredicateFixture,
                 "gpu_execution cast-shaped DATE predicates on dates beyond TIMESTAMP's range",
                 "[integration][gpu_execution][filter][fused_scan_filter][cast_date]")
{
  // DuckDB: CAST(DATE '300000-01-01' AS TIMESTAMP) overflows int64 micros and
  // raises for the whole query. The fold answers by the instant each date
  // denotes: the far date is above any finite cutoff and below +infinity.
  // (The residual cudf::cast path, taken by unpinned and DuckDB-format scans,
  // wraps the overflow instead and is not what this pins.)
  pin_compressed_parquet("t_far");
  expect_cpu_raises_gpu_answers("SELECT id FROM p_t_far WHERE d <= TIMESTAMP '2000-06-01'",
                                {{"1"}});
  expect_cpu_raises_gpu_answers("SELECT id FROM p_t_far WHERE d >  TIMESTAMP '2000-06-01'",
                                {{"2"}});
  expect_cpu_raises_gpu_answers("SELECT id FROM p_t_far WHERE d <  TIMESTAMP 'infinity'",
                                {{"1"}, {"2"}});
  unpin_parquet("t_far");
}
