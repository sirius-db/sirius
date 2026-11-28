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

// Ungrouped aggregates handed to Meld (SET meld_units = true), against DuckDB on the CPU, over a
// lineitem of doubles. Every value is dyadic and small, so every sum is exact whatever order the
// batches are added in, and the answers compare exactly.

#include "meld/meld_unit.hpp"

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>

namespace {

class MeldUnitsFixture : public sirius::test::GpuExecutionFixture {
 public:
  MeldUnitsFixture()
  {
    run_ok(
      "CREATE TABLE lineitem AS SELECT "
      "(i % 50)::DOUBLE AS l_quantity, "
      "((i % 97) * 0.25)::DOUBLE AS l_extendedprice, "
      "((i % 11) * 0.0078125)::DOUBLE AS l_discount "
      "FROM range(200000) t(i);");
    run_ok("CHECKPOINT;");
    run_ok("SET meld_units = true;");
  }

  ~MeldUnitsFixture() { con->Query("SET meld_units = false;"); }

  /// Runs @p query on the GPU and on the CPU and compares them, and checks Meld ran batches of it.
  void compare_in_meld(std::string const& query)
  {
    auto const before = sirius::meld::meld_unit::batches_run();
    compare_gpu_vs_cpu(query);
    CHECK(sirius::meld::meld_unit::batches_run() > before);
  }

  /// The same, for a query Meld does not support.
  void compare_without_meld(std::string const& query)
  {
    auto const before = sirius::meld::meld_unit::batches_run();
    compare_gpu_vs_cpu(query);
    CHECK(sirius::meld::meld_unit::batches_run() == before);
  }

  sirius::test::scratch_dir dir{"meld_units"};
};

std::string const q6 =
  "SELECT sum(l_extendedprice * l_discount) AS revenue FROM {} "
  "WHERE l_quantity < 24 AND l_discount BETWEEN 0.0390625 AND 0.0625";

std::string over(std::string const& query, std::string const& table)
{
  auto text = query;
  text.replace(text.find("{}"), 2, table);
  return text;
}

}  // namespace

TEST_CASE_METHOD(MeldUnitsFixture, "Meld runs Q6 over doubles", "[meld_units][integration]")
{
  compare_in_meld(over(q6, "lineitem"));
}

TEST_CASE_METHOD(MeldUnitsFixture,
                 "Meld runs the filter DuckDB leaves above the scan",
                 "[meld_units][integration]")
{
  run_ok("SET disabled_optimizers = 'filter_pushdown';");
  compare_in_meld(over(q6, "lineitem"));
  run_ok("RESET disabled_optimizers;");
}

TEST_CASE_METHOD(MeldUnitsFixture,
                 "Meld runs COUNT(*), AVG, MIN and MAX beside a SUM",
                 "[meld_units][integration]")
{
  compare_in_meld(
    "SELECT count(*), sum(l_extendedprice), avg(l_discount), min(l_quantity), "
    "max(l_extendedprice * l_discount) FROM lineitem WHERE l_quantity >= 10");
}

TEST_CASE_METHOD(MeldUnitsFixture,
                 "Meld runs Q6 over a Parquet table pinned on the GPU",
                 "[meld_units][integration]")
{
  auto const parquet = dir.file("lineitem.parquet");
  run_ok("COPY lineitem TO '" + parquet + "' (FORMAT PARQUET);");
  run_ok("CREATE VIEW p_lineitem AS SELECT * FROM read_parquet('" + parquet + "');");
  run_ok("CALL pin_table('" + parquet + "', tier='gpu', name='p_lineitem');");
  compare_in_meld(over(q6, "p_lineitem"));
  run_ok("CALL unpin_table('p_lineitem');");
}

namespace {

/// Doubles with NaN and NULL among them, and dates, with the filters left above the scan for Meld.
class MeldSpecialValuesFixture : public MeldUnitsFixture {
 public:
  MeldSpecialValuesFixture()
  {
    run_ok(
      "CREATE TABLE special AS SELECT "
      "CASE WHEN i % 7 = 0 THEN 'NaN'::DOUBLE WHEN i % 13 = 0 THEN NULL "
      "ELSE ((i % 41) - 20)::DOUBLE END AS p, "
      "CASE WHEN i % 5 = 0 THEN 'NaN'::DOUBLE ELSE (i % 3)::DOUBLE END AS q, "
      "DATE '1994-01-01' + (i % 730)::INTEGER AS d "
      "FROM range(100000) t(i);");
    run_ok("CHECKPOINT;");
    run_ok("SET disabled_optimizers = 'filter_pushdown';");
  }

  ~MeldSpecialValuesFixture() { con->Query("RESET disabled_optimizers;"); }
};

}  // namespace

TEST_CASE_METHOD(MeldSpecialValuesFixture,
                 "Meld compares NaN as DuckDB does",
                 "[meld_units][integration]")
{
  auto const condition = GENERATE(as<std::string>{},
                                  "p > 20",
                                  "p >= 20",
                                  "p < 20",
                                  "p <= 20",
                                  "p = 'NaN'::DOUBLE",
                                  "p <> 1",
                                  "p < 'NaN'::DOUBLE",
                                  "p > q",
                                  "p >= q",
                                  "p < q",
                                  "p <= q",
                                  "p = q",
                                  "p <> q",
                                  "p BETWEEN -5 AND 5",
                                  "p BETWEEN q AND 10",
                                  "q BETWEEN p AND 'NaN'::DOUBLE");
  CAPTURE(condition);
  compare_in_meld("SELECT count(*), count(p), sum(q) FROM special WHERE " + condition);
}

TEST_CASE_METHOD(MeldSpecialValuesFixture,
                 "Meld orders NaN above every number in MIN and MAX",
                 "[meld_units][integration]")
{
  compare_in_meld("SELECT min(p), max(p), min(q), max(q) FROM special WHERE d > DATE '1994-03-01'");
  // Only NaN is above every number, so these see only NaNs.
  compare_in_meld("SELECT min(p), max(p), count(p) FROM special WHERE p > 1e300");
}

TEST_CASE_METHOD(MeldSpecialValuesFixture,
                 "Meld skips NULL in aggregates and filters",
                 "[meld_units][integration]")
{
  compare_in_meld("SELECT count(*), count(p), sum(p), avg(p) FROM special WHERE q < 2");
  compare_in_meld("SELECT count(*), sum(q) FROM special WHERE p < 5 OR p > 10");
}

TEST_CASE_METHOD(MeldSpecialValuesFixture, "Meld filters on dates", "[meld_units][integration]")
{
  compare_in_meld(
    "SELECT count(*), sum(q), min(p) FROM special "
    "WHERE d >= DATE '1994-06-01' AND d < DATE '1995-01-01'");
}

TEST_CASE_METHOD(MeldSpecialValuesFixture,
                 "Meld answers a filter no row passes",
                 "[meld_units][integration]")
{
  // Only a NaN is above 2, and it is not below 3; DuckDB's statistics do not see that.
  compare_in_meld(
    "SELECT count(*), count(p), sum(p), avg(q), min(p), max(q) FROM special "
    "WHERE q > 2 AND q < 3");
}

TEST_CASE_METHOD(MeldUnitsFixture,
                 "Meld does not support an integer sum",
                 "[meld_units][integration]")
{
  compare_without_meld("SELECT sum(l_quantity::INTEGER) FROM lineitem WHERE l_discount > 0.03");
}

TEST_CASE_METHOD(MeldUnitsFixture,
                 "Meld is not used when meld_units is off",
                 "[meld_units][integration]")
{
  run_ok("SET meld_units = false;");
  compare_without_meld(over(q6, "lineitem"));
}
