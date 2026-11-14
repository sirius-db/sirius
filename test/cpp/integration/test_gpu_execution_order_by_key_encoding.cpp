/*
 * Copyright 2025, Sirius Contributors.
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

// ORDER BY through SQL with the per-batch sort's key encoding on (`sort_key_encoding = true`, the
// default) and off. Each query has a total order and runs on DuckDB's CPU execution and on the
// GPU with both values of the setting; the three results must be equal row by row.

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <string>
#include <vector>

namespace {

class OrderByKeyEncodingFixture : public sirius::test::GpuExecutionFixture {
 public:
  void create_table(const std::string& name, const std::string& select)
  {
    run_ok("CREATE TABLE " + name + " AS " + select + ";");
    run_ok("CHECKPOINT;");
  }

  std::vector<std::vector<std::string>> run_on_gpu(const std::string& query, bool encoding)
  {
    INFO("sort_key_encoding = " << encoding);
    sirius::test::scoped_sirius_setting setting(*con, "sort_key_encoding", encoding);
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    // Exactly one GPU execution and no fallback.
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    return collect_rows(*result, false);
  }

  /// Requires the GPU results of @p query, with the key encoding on and off, to equal DuckDB's.
  void check_order(const std::string& query)
  {
    INFO(query);
    run_ok("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    REQUIRE_FALSE(cpu_result->HasError());
    auto const cpu = collect_rows(*cpu_result, false);
    cpu_result.reset();
    for (const bool encoding : {true, false}) {
      auto const gpu = run_on_gpu(query, encoding);
      REQUIRE(gpu.size() == cpu.size());
      for (std::size_t r = 0; r < cpu.size(); r++) {
        if (gpu[r] != cpu[r]) {
          FAIL("sort_key_encoding = " << encoding << ": row " << r << " differs from DuckDB's");
        }
      }
    }
  }
};

}  // namespace

TEST_CASE_METHOD(OrderByKeyEncodingFixture,
                 "sort_key_encoding: ORDER BY string and multi-column keys matches DuckDB",
                 "[integration][gpu_execution][order_by_key_encoding]")
{
  // s: 18 values of 18 bytes sharing a 16-byte prefix. t: close to unique, lengths 1 to 40, some
  // a prefix of others, some with multi-byte characters. a repeats; b is close to unique.
  create_table(
    "oke",
    "SELECT CAST(i AS BIGINT) AS id, "
    "       CASE WHEN i % 29 = 0 THEN NULL "
    "            ELSE '0000000000000000' || lpad(CAST(hash(i) % 18 AS VARCHAR), 2, '0') END AS s, "
    "       CASE WHEN i % 31 = 0 THEN NULL "
    "            WHEN i % 7 = 0 THEN '' "
    "            WHEN i % 11 = 0 THEN repeat('é', CAST(hash(i + 5) % 20 AS INTEGER)) "
    "            ELSE substr(md5(CAST(hash(i + 1) % 60000 AS VARCHAR)) || 'tail', 1, "
    "                        1 + CAST(hash(i + 2) % 36 AS INTEGER)) END AS t, "
    "       CASE WHEN i % 23 = 0 THEN NULL ELSE CAST(hash(i + 3) % 500 AS INTEGER) - 250 END AS a, "
    "       CASE WHEN i % 37 = 0 THEN NULL "
    "            ELSE CAST(hash(i + 4) >> 1 AS BIGINT) - 4611686018427387904 END AS b, "
    "       CASE WHEN hash(i + 6) % 10 = 0 THEN 'NaN'::DOUBLE "
    "            WHEN hash(i + 6) % 10 = 1 THEN NULL "
    "            ELSE CAST(hash(i + 7) % 2000 AS DOUBLE) / 8 - 100 END AS d, "
    "       CAST(CAST(hash(i + 8) % 100000 AS BIGINT) - 50000 AS DECIMAL(15, 2)) AS w "
    "FROM range(150000) AS t(i)");

  check_order("SELECT s, id FROM oke ORDER BY s, id");
  check_order("SELECT s, id FROM oke ORDER BY s DESC NULLS LAST, id DESC");
  check_order("SELECT t, id FROM oke ORDER BY t ASC NULLS FIRST, id");
  check_order("SELECT t, id FROM oke ORDER BY t DESC NULLS FIRST, id");
  check_order("SELECT a, b, t, id FROM oke ORDER BY a, b, t, id");
  check_order(
    "SELECT a, s, d, id FROM oke ORDER BY a DESC NULLS LAST, s ASC NULLS LAST, d DESC, id");
  check_order("SELECT d, a, id FROM oke ORDER BY d ASC NULLS FIRST, a DESC NULLS FIRST, id DESC");
  check_order("SELECT w, t, id FROM oke ORDER BY w DESC, t, id");
}

TEST_CASE_METHOD(OrderByKeyEncodingFixture,
                 "sort_key_encoding: default, SET and RESET",
                 "[integration][gpu_execution][order_by_key_encoding]")
{
  auto value = [&] {
    auto result = con->Query("SELECT current_setting('sort_key_encoding')");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    auto const rows = collect_rows(*result, false);
    REQUIRE(rows.size() == 1);
    return rows[0][0] == "true";
  };
  run_ok("RESET sort_key_encoding;");
  CHECK(value());
  run_ok("SET sort_key_encoding = false;");
  CHECK_FALSE(value());
  run_ok("RESET sort_key_encoding;");
  CHECK(value());
}
