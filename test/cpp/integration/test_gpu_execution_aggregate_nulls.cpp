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
  // COUNT(*) counts rows; COUNT(col) skips NULLs.
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
  // COUNT(v)=0 and SUM/AVG/MIN/MAX = NULL.
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
  // Grouped COUNT(DISTINCT) runs on the GPU and skips NULLs correctly (the
  // ungrouped form falls back to CPU -- see the next case).
  compare_gpu_vs_cpu("SELECT g, COUNT(DISTINCT v) FROM agg_n GROUP BY g");
}

// Not a result divergence: ungrouped COUNT(DISTINCT) is unsupported on the GPU
// and forces a runtime fallback to DuckDB CPU (the result is still correct).
// Asserted with expect_gpu_fallback rather than abusing [!shouldfail] on the
// no-fallback comparator. Tracked in issue #1218.
TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution ungrouped COUNT(DISTINCT) falls back to CPU",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  expect_gpu_fallback("SELECT COUNT(DISTINCT v) FROM agg_n");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution ungrouped AVG skips NULLs (non-null denominator)",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  // AVG divides SUM by the count of non-null values, not the row count, so AVG
  // over a NULL-containing column matches DuckDB: AVG(v) = 335/5 = 67.
  compare_gpu_vs_cpu("SELECT AVG(v), AVG(d), AVG(f) FROM agg_n");
}

// A wholly-NULL column checkpoints to CONSTANT all-null validity; the native
// scan synthesizes its null mask, so aggregates must see NULLs rather than
// sentinel values. Split into ungrouped/grouped cases: Catch2 aborts a test
// case at the first REQUIRE failure, so bundling them would leave the grouped
// query unexercised.
TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution ungrouped aggregates over a wholly-NULL column",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu(
    "SELECT SUM(allnull), AVG(allnull), MIN(allnull), MAX(allnull), COUNT(allnull) FROM agg_n");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution grouped aggregates over a wholly-NULL column",
                 "[integration][gpu_execution][aggregate][nulls]")
{
  compare_gpu_vs_cpu("SELECT g, SUM(allnull), COUNT(allnull) FROM agg_n GROUP BY g");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution stddev_samp NULL and numeric semantics",
                 "[integration][gpu_execution][stddev_samp]")
{
  compare_gpu_vs_cpu_approx(
    "SELECT stddev_samp(v), stddev_samp(d), stddev_samp(f), stddev_samp(allnull), avg(v), count(v) "
    "FROM agg_n",
    {0, 1, 2, 3, 4});
  compare_gpu_vs_cpu_approx(
    "SELECT g, stddev_samp(v), stddev_samp(d), stddev_samp(f), avg(v), count(v) FROM agg_n GROUP "
    "BY g",
    {1, 2, 3, 4});
  compare_gpu_vs_cpu("SELECT stddev_samp(v) FROM agg_n WHERE g = 2");    // singleton
  compare_gpu_vs_cpu("SELECT stddev_samp(v) FROM agg_n WHERE g = 999");  // empty
  compare_gpu_vs_cpu("SELECT g, stddev_samp(v) FROM agg_n WHERE g = 999 GROUP BY g");
  compare_gpu_vs_cpu_approx(
    "SELECT g, stddev_samp(v)/avg(v) FROM agg_n GROUP BY g HAVING stddev_samp(v) > 0", {1});
  expect_plan_fallback_matches_cpu("SELECT g, stddev_samp(DISTINCT v) FROM agg_n GROUP BY g");
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution stddev_samp over joined measures",
                 "[integration][gpu_execution][stddev_samp]")
{
  // Q17's relevant shape: join several measures before grouping, with COUNT/AVG/STDDEV
  // and coefficient-of-variation projections on each measure. Keep the data small and
  // include constant, singleton and wholly-NULL measures in nonempty result groups.
  run_ok("CREATE TABLE products (id INTEGER, name VARCHAR)");
  run_ok("INSERT INTO products VALUES (1, 'widget'), (2, 'gadget')");
  run_ok("CREATE TABLE purchases (event INTEGER, product INTEGER, region VARCHAR, units INTEGER)");
  run_ok(
    "INSERT INTO purchases VALUES (1, 1, 'east', 2), (2, 1, 'east', 4), "
    "(3, 1, 'east', NULL), (4, 1, 'west', 9), (5, 1, 'west', 9), "
    "(6, 2, NULL, 6), (7, 1, 'unmatched', 100)");
  run_ok("CREATE TABLE refunds (event INTEGER, product INTEGER, units INTEGER)");
  run_ok(
    "INSERT INTO refunds VALUES (1, 1, 1), (2, 1, 3), (3, 1, 5), "
    "(4, 1, 1), (5, 1, 1), (6, 2, NULL)");
  run_ok("CREATE TABLE deliveries (event INTEGER, product INTEGER, units INTEGER)");
  run_ok(
    "INSERT INTO deliveries VALUES (1, 1, 4), (2, 1, NULL), (3, 1, 8), "
    "(4, 1, NULL), (5, 1, NULL), (6, 2, 0)");
  run_ok("CHECKPOINT");
  std::string const query = R"SQL(
SELECT p.name, s.region,
       count(s.units), avg(s.units), stddev_samp(s.units), stddev_samp(s.units)/avg(s.units),
       count(r.units), avg(r.units), stddev_samp(r.units), stddev_samp(r.units)/avg(r.units),
       count(d.units), avg(d.units), stddev_samp(d.units), stddev_samp(d.units)/avg(d.units)
FROM purchases s
JOIN products p ON s.product = p.id
JOIN refunds r ON s.event = r.event AND s.product = r.product
JOIN deliveries d ON s.event = d.event AND s.product = d.product
GROUP BY p.name, s.region
ORDER BY p.name, s.region NULLS FIRST
LIMIT 10
)SQL";
  run_ok("SET gpu_execution = false");
  auto expected = con->Query(query);
  REQUIRE_FALSE(expected->HasError());
  REQUIRE(expected->RowCount() == 3);
  compare_gpu_vs_cpu_approx(query, {3, 4, 5, 7, 8, 9, 11, 12, 13});
}

TEST_CASE_METHOD(AggNullFixture,
                 "gpu_execution stddev_samp filtered aggregate CTE self join",
                 "[integration][gpu_execution][stddev_samp]")
{
  run_ok(
    "CREATE TABLE inv_sample (warehouse INTEGER, item INTEGER, month INTEGER, quantity INTEGER)");
  run_ok(
    "INSERT INTO inv_sample VALUES (1, 1, 1, 0), (1, 1, 1, 0), (1, 1, 1, 90), "
    "(1, 1, 2, 0), (1, 1, 2, 0), (1, 1, 2, 60), "
    "(1, 2, 1, NULL), (1, 2, 2, NULL), "
    "(1, 3, 1, 0), (1, 3, 1, 0), (1, 3, 2, 0), (1, 3, 2, 0), "
    "(1, 4, 1, 10), (1, 4, 1, 10), (1, 4, 2, 10), (1, 4, 2, 10)");
  run_ok("CHECKPOINT");
  std::string const query = R"SQL(
WITH statistics AS (
  SELECT warehouse, item, month, stddev_samp(quantity) spread, avg(quantity) mean
  FROM inv_sample GROUP BY warehouse, item, month
), variable_stock AS (
  SELECT warehouse, item, month, mean, spread/nullif(mean, 0) relative_spread
  FROM statistics WHERE mean > 0 AND spread > mean
)
SELECT a.warehouse, a.item, a.mean, a.relative_spread, b.mean, b.relative_spread
FROM variable_stock a JOIN variable_stock b
  ON a.item = b.item AND a.warehouse = b.warehouse
WHERE a.month = 1 AND b.month = 2
)SQL";
  run_ok("SET gpu_execution = false");
  auto expected = con->Query(query);
  REQUIRE_FALSE(expected->HasError());
  REQUIRE(expected->RowCount() == 1);
  compare_gpu_vs_cpu_approx(query, {2, 3, 4, 5});
}
