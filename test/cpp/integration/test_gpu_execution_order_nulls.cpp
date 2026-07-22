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

// GPU-vs-CPU correctness for ORDER BY / TOP-N with NULLs in the sort key,
// across every (ASC|DESC) x (NULLS FIRST|LAST) combination (issue #1095).
//
// Uses the shared file-backed GpuExecutionFixture so the source table is read
// through the real GPU DuckDB-native scan, and compares position-sensitively
// (compare_gpu_vs_cpu_ordered) so NULL placement is actually verified rather
// than sorted away.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

struct null_order_case {
  std::string sort_direction;
  std::string null_position;
};

const null_order_case kNullOrderCases[] = {
  {"ASC", "FIRST"},
  {"ASC", "LAST"},
  {"DESC", "FIRST"},
  {"DESC", "LAST"},
};

std::string case_name(const null_order_case& order_case)
{
  return order_case.sort_direction + " NULLS " + order_case.null_position;
}

// Sets a runtime setting for a scope and restores its PREVIOUS value on
// destruction (not the default), so the setting survives a REQUIRE failure that
// unwinds out of the test without clobbering an outer override.
class scoped_setting {
 public:
  scoped_setting(sirius::test::GpuExecutionFixture& fixture,
                 std::string name,
                 const std::string& value)
    : fixture(fixture), name(std::move(name)), old_value(read_setting(fixture, this->name))
  {
    fixture.run_ok("SET " + this->name + " = " + value + ";");
  }
  ~scoped_setting() { fixture.con->Query("SET " + name + " = " + old_value + ";"); }

  scoped_setting(const scoped_setting&)            = delete;
  scoped_setting& operator=(const scoped_setting&) = delete;

 private:
  static std::string read_setting(sirius::test::GpuExecutionFixture& fixture,
                                  const std::string& name)
  {
    auto result =
      fixture.con->Query("SELECT value FROM duckdb_settings() WHERE name = '" + name + "'");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    return result->GetValue(0, 0).ToString();
  }

  sirius::test::GpuExecutionFixture& fixture;
  std::string name;
  std::string old_value;
};

class OrderNullsGPUExecutionFixture : public sirius::test::GpuExecutionFixture {
 public:
  OrderNullsGPUExecutionFixture()
  {
    // Every 7th row has a NULL key; keys otherwise cycle 0..99 so ties exercise
    // the secondary `id` ordering. Persist to disk for the native GPU scan.
    run_ok(
      "CREATE TABLE ord_n AS "
      "SELECT CASE WHEN i % 7 = 0 THEN NULL ELSE CAST(i % 100 AS INTEGER) END AS k, "
      "       CAST(i AS INTEGER) AS id "
      "FROM range(30000) AS t(i);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(OrderNullsGPUExecutionFixture,
                 "gpu_execution ORDER BY places NULLs correctly for ASC and DESC",
                 "[integration][gpu_execution][order_by][nulls]")
{
  // Shrink the sort partition so the 30k rows span multiple partitions and the
  // merge path (not just a single-partition sort) is exercised with NULL keys.
  scoped_setting sort_partition(*this, "max_sort_partition_bytes", "65536");

  for (const auto& order_case : kNullOrderCases) {
    DYNAMIC_SECTION(case_name(order_case))
    {
      compare_gpu_vs_cpu_ordered(
        "SELECT k, id "
        "FROM ord_n "
        "ORDER BY k " +
        order_case.sort_direction + " NULLS " + order_case.null_position + ", id ASC");
    }
  }
}

TEST_CASE_METHOD(OrderNullsGPUExecutionFixture,
                 "gpu_execution TOP-N single-key places NULLs correctly for ASC and DESC",
                 "[integration][gpu_execution][top_n][nulls]")
{
  for (const auto& order_case : kNullOrderCases) {
    DYNAMIC_SECTION(case_name(order_case))
    {
      // Single sort key only (no tie-break): with LIMIT 50 the boundary
      // k-value has far more than 50 ties, so the emitted k-values are
      // deterministic even though which rows fill the ties is not.
      compare_gpu_vs_cpu_ordered(
        "SELECT k "
        "FROM ord_n "
        "ORDER BY k " +
        order_case.sort_direction + " NULLS " + order_case.null_position +
        " "
        "LIMIT 50");
    }
  }
}

TEST_CASE_METHOD(OrderNullsGPUExecutionFixture,
                 "gpu_execution TOP-N multi-key places NULLs correctly for ASC and DESC",
                 "[integration][gpu_execution][top_n][nulls]")
{
  for (const auto& order_case : kNullOrderCases) {
    DYNAMIC_SECTION(case_name(order_case))
    {
      compare_gpu_vs_cpu_ordered(
        "SELECT k, id "
        "FROM ord_n "
        "ORDER BY k " +
        order_case.sort_direction + " NULLS " + order_case.null_position +
        ", id ASC "
        "LIMIT 50");
    }
  }
}
