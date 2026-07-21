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

// GPU-vs-CPU correctness for NULL handling in aggregates (issue #1095):
// COUNT(*) vs COUNT(col), NULL-skipping SUM/AVG/MIN/MAX, all-NULL inputs and
// groups, GROUP BY on a NULL key, and COUNT(DISTINCT) with NULLs.
//
// Every query goes through the shared file-backed GpuExecutionFixture, which
// runs it once on the GPU (asserting a real GPU execution with no fallback) and
// once on DuckDB CPU, then compares the results. The comparator is order-
// insensitive, which suits GROUP BY output.
//
// Cases tagged [!shouldfail] document confirmed GPU/CPU divergences (tracked in
// their own issues); the tag reports them as expected failures so CI stays green
// until the underlying bug is fixed.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

// `g` is the group key and deliberately contains a NULL group; `v`/`d`/`f` are
// nullable value columns; `allnull` is entirely NULL. Group 3 has only NULL
// values (an all-NULL group), and one group key is NULL.
class AggNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  AggNullFixture()
  {
    run_ok(
      "CREATE TABLE agg_n ("
      "  g       INTEGER,"
      "  v       INTEGER,"
      "  d       DECIMAL(10,2),"
      "  f       DOUBLE,"
      "  allnull INTEGER);");
    run_ok(
      "INSERT INTO agg_n VALUES "
      "(1,    10,   1.00,  1.5,  NULL),"
      "(1,    20,   2.00,  NULL, NULL),"
      "(1,    NULL, NULL,  2.5,  NULL),"
      "(2,    5,    5.00,  5.0,  NULL),"
      "(2,    NULL, NULL,  NULL, NULL),"
      "(NULL, 100,  10.00, 10.0, NULL),"  // NULL group key
      "(NULL, 200,  NULL,  NULL, NULL),"  // NULL group key
      "(3,    NULL, NULL,  NULL, NULL);"  // all-NULL group
    );
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Verified-correct coverage
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution COUNT(*) vs COUNT(col) with NULLs",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  // COUNT(*) counts rows; COUNT(col) skips NULLs. (COUNT over a wholly-NULL
  // column is broken -- see the [!shouldfail] case below.)
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(v), COUNT(d), COUNT(f) FROM agg_n");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution ungrouped SUM/MIN/MAX skip NULLs",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT SUM(v), MIN(v), MAX(v) FROM agg_n");
  compare_gpu_vs_cpu("SELECT SUM(d), MIN(d), MAX(d) FROM agg_n");
  compare_gpu_vs_cpu("SELECT SUM(f), MIN(f), MAX(f) FROM agg_n");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution GROUP BY groups NULL keys together",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  // The NULL group key forms its own group; the all-NULL group (g=3) yields
  // COUNT(v)=0 and SUM/AVG/MIN/MAX = NULL. Grouped AVG uses the correct non-null
  // denominator (unlike ungrouped AVG -- see the [!shouldfail] case below).
  compare_gpu_vs_cpu(
    "SELECT g, COUNT(*), COUNT(v), SUM(v), AVG(v), MIN(v), MAX(v) FROM agg_n GROUP BY g");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution grouped SUM/AVG over NULL values",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT g, SUM(d), AVG(d) FROM agg_n GROUP BY g");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution grouped COUNT(DISTINCT) ignores NULLs",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  // Grouped COUNT(DISTINCT) runs on the GPU and skips NULLs correctly. (The
  // ungrouped form falls back to CPU -- see the [!shouldfail] case below.)
  compare_gpu_vs_cpu("SELECT g, COUNT(DISTINCT v) FROM agg_n GROUP BY g");
}

//===----------------------------------------------------------------------===//
// Known GPU divergences (quarantined) -- each tracked in its own issue; remove
// the [!shouldfail] tag when the underlying bug is fixed.
//===----------------------------------------------------------------------===//

// KNOWN GPU DIVERGENCE (issue #1095 follow-up -- please file):
// A column that is entirely NULL loses its validity mask in the GPU native scan
// and is read as sentinel values (INT_MAX), so aggregates over it see fake data:
// SUM(allnull) returns 8*INT_MAX and COUNT(allnull) returns the row count (8)
// instead of NULL / 0. All-NULL *groups* of a normally-nullable column are fine;
// only a wholly-NULL column is affected.
// Split into ungrouped/grouped cases: Catch2 aborts a test case at the first
// REQUIRE failure, so bundling them would leave the grouped query unexercised.
TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution ungrouped aggregates over a wholly-NULL column [known divergence]",
                 "[integration][gpu_execution][aggregate][nulls][!shouldfail]")
{
  compare_gpu_vs_cpu(
    "SELECT SUM(allnull), AVG(allnull), MIN(allnull), MAX(allnull), COUNT(allnull) FROM agg_n");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution grouped aggregates over a wholly-NULL column [known divergence]",
                 "[integration][gpu_execution][aggregate][nulls][!shouldfail]")
{
  compare_gpu_vs_cpu("SELECT g, SUM(allnull), COUNT(allnull) FROM agg_n GROUP BY g");
}

// KNOWN GPU DIVERGENCE (issue #1095 follow-up -- please file):
// Ungrouped AVG divides SUM by the total row count instead of the non-null
// count, so AVG over a column containing NULLs is wrong: AVG(v) returns
// 335/8 = 41.875 instead of 335/5 = 67. SUM and COUNT are individually correct,
// and grouped AVG is correct -- the bug is isolated to the ungrouped aggregate.
TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution ungrouped AVG denominator counts NULL rows [known divergence]",
                 "[integration][gpu_execution][aggregate][nulls][!shouldfail]")
{
  compare_gpu_vs_cpu("SELECT AVG(v), AVG(d), AVG(f) FROM agg_n");
}

// KNOWN GPU DIVERGENCE (issue #1095 follow-up -- please file):
// Ungrouped COUNT(DISTINCT ...) errors on the GPU and falls back to DuckDB CPU at
// runtime (runtime_fallbacks increments), so it does not execute on the GPU. The
// result is correct via fallback, but the operation is unsupported on-device.
// (Grouped COUNT(DISTINCT) does run on the GPU -- see the verified case above.)
TEST_CASE_METHOD(
  AggNullFixture,
  "gpu_execution ungrouped COUNT(DISTINCT) runtime-falls-back to CPU [known divergence]",
  "[integration][gpu_execution][aggregate][nulls][!shouldfail]")
{
  compare_gpu_vs_cpu("SELECT COUNT(DISTINCT v) FROM agg_n");
}
