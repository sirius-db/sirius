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
 * @file test_gpu_execution_decimal_pushdown.cpp
 * @brief Values decoded from decimal parquet scans whose filter is pushed into the reader
 *
 * `test/cpp/scan/test_parquet_decimal_pushdown.cpp` pins which row groups survive pruning. That is
 * metadata: it would still pass if the surviving rows decoded to the wrong values. These tests run
 * the queries and compare against DuckDB's own answer, so a defect that pruned correctly but
 * mis-decoded the rows it kept is caught here.
 *
 * The distinction matters for the specific defect at issue. If FIXED_LEN_BYTE_ARRAY decimal
 * statistics stop sign-extending, negative amounts read as large positive ones, which can show up
 * either as wrongly pruned row groups (caught by the scan test) or as wrong decoded values (caught
 * here). Every predicate below therefore selects across the sign boundary in the fixtures, whose
 * first five row groups hold only negative amounts.
 *
 * The fixtures store identical data at FIXED_LEN_BYTE_ARRAY widths 4, 8 and 16, because sign
 * extension is applied per stored width and DuckDB can only produce the 16-byte one. See
 * `test/cpp/scan/data/generate_flba_decimal_bands.py`.
 */

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path fixture_dir()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT} / "test/cpp/scan/data";
#else
  return std::filesystem::current_path() / "test/cpp/scan/data";
#endif
}

/// Views over the checked-in fixtures, one per FIXED_LEN_BYTE_ARRAY width, so a query can name a
/// width directly. The views read the parquet files in place; nothing is copied into the database.
class FlbaDecimalFixture : public sirius::test::GpuExecutionFixture {
 public:
  FlbaDecimalFixture()
  {
    create_view("flba4", "flba4_decimal_bands.parquet");
    create_view("flba8", "flba8_decimal_bands.parquet");
    create_view("flba16", "flba16_decimal_bands.parquet");
  }

 private:
  void create_view(std::string const& view_name, std::string const& file_name)
  {
    auto const path = fixture_dir() / file_name;
    REQUIRE(std::filesystem::exists(path));
    run_ok("CREATE VIEW " + view_name + " AS SELECT * FROM read_parquet('" + path.string() + "');");
  }
};

/// The three widths, named as the views above.
std::vector<std::string> const kWidths = {"flba4", "flba8", "flba16"};

}  // namespace

TEST_CASE_METHOD(FlbaDecimalFixture,
                 "gpu_execution decimal pushdown returns correct rows across the sign boundary",
                 "[integration][gpu_execution][filter][decimal]")
{
  for (auto const& width : kWidths) {
    DYNAMIC_SECTION(width << " negative range")
    {
      compare_gpu_vs_cpu("SELECT id, amount FROM " + width + " WHERE amount < -5");
    }
    DYNAMIC_SECTION(width << " positive range")
    {
      compare_gpu_vs_cpu("SELECT id, amount FROM " + width + " WHERE amount > 35000");
    }
    DYNAMIC_SECTION(width << " range spanning the sign boundary")
    {
      compare_gpu_vs_cpu("SELECT id, amount FROM " + width +
                         " WHERE amount BETWEEN -15000 AND 15000");
    }
    DYNAMIC_SECTION(width << " equality on a negative amount")
    {
      compare_gpu_vs_cpu("SELECT id, amount FROM " + width + " WHERE amount = -30000");
    }
    DYNAMIC_SECTION(width << " nulls")
    {
      compare_gpu_vs_cpu("SELECT id FROM " + width + " WHERE amount IS NULL");
    }
  }
}

TEST_CASE_METHOD(FlbaDecimalFixture,
                 "gpu_execution decimal pushdown aggregates the amounts it decoded",
                 "[integration][gpu_execution][filter][decimal]")
{
  // Aggregating the decoded amounts, rather than selecting them, catches a defect in a single
  // decoded value that a row-count comparison would miss.
  for (auto const& width : kWidths) {
    DYNAMIC_SECTION(width << " aggregate over negative amounts")
    {
      compare_gpu_vs_cpu("SELECT count(*), sum(id), min(amount), max(amount), sum(amount) FROM " +
                         width + " WHERE amount < -5");
    }
    DYNAMIC_SECTION(width << " aggregate grouped by sign")
    {
      // The sum widens the amount first. A grouped SUM over a decimal32 column (any parquet
      // DECIMAL of precision 9 or less, whatever its physical type) currently wraps at 32 bits
      // and returns a wrong total; that defect is in the grouped aggregate rather than in the
      // scan, it reproduces on INT32-backed decimals that never went through this probe, and
      // widening here keeps this test measuring the scan instead of failing on it.
      // Tracked: "Grouped SUM over DECIMAL32/64 accumulates at input width and wraps before
      // widening" -- remove this CAST when the aggregate is fixed.
      compare_gpu_vs_cpu(
        "SELECT amount < 0 AS is_negative, count(*), min(amount), max(amount),"
        " sum(CAST(amount AS DECIMAL(38,4))) FROM " +
        width + " WHERE amount IS NOT NULL GROUP BY 1");
    }
  }
}
