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

// ORDER BY through SQL with run coalescing (`sort_coalesce_runs`): the local sort concatenates
// groups of scan batches and sorts each group as one run. Every query has a total order and runs
// on DuckDB's CPU execution, on the GPU with coalescing off, and on the GPU with coalescing on at
// several run sizes; all results must be identical row for row. The ORDER_BY counters show that
// the runs really were coalesced (or not).
//
// Tables are file-backed (GpuExecutionFixture) and scanned in small batches, so the local sort
// sees many input batches.

#include <catch.hpp>
#include <op/sirius_physical_order.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <memory>
#include <string>
#include <vector>

namespace {

class OrderByCoalesceFixture : public sirius::test::GpuExecutionFixture {
 public:
  OrderByCoalesceFixture()
  {
    // Small scan batches: the local sort receives many of them.
    set("scan_task_batch_size", "2097152");
  }

  ~OrderByCoalesceFixture()
  {
    // Newest first, so a setting changed twice ends at the value it had before the fixture.
    while (!settings.empty()) {
      settings.pop_back();
    }
  }

  /// Sets a setting until the fixture is destroyed, which restores the value it had before.
  void set(const std::string& name, const std::string& value)
  {
    settings.push_back(std::make_unique<sirius::test::scoped_sirius_setting>(*con, name, value));
  }

  /// Creates and persists a table, so the GPU native scan reads it from disk.
  void create_table(const std::string& name, const std::string& select)
  {
    run_ok("CREATE TABLE " + name + " AS " + select + ";");
    run_ok("CHECKPOINT;");
  }

  /// What the local sorts of one GPU run did.
  struct gpu_run {
    std::vector<std::vector<std::string>> rows;
    sirius::op::order_by_counters sorts;
  };

  gpu_run run_on_gpu(const std::string& query)
  {
    sirius::op::sirius_physical_order::reset_process_counters();
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    // Exactly one GPU execution and no fallback.
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    return {collect_rows(*result, false), sirius::op::sirius_physical_order::process_counters()};
  }

  /**
   * @brief Requires the GPU results of @p query, with and without coalescing, to equal DuckDB's
   * row for row. The query's order must be total.
   *
   * @param expect_coalesced Whether the coalesced run must have sorted several batches as one
   *        run (false for queries whose sort key keeps the local sort per batch).
   */
  void check_order(const std::string& query, bool expect_coalesced = true)
  {
    INFO(query);
    run_ok("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) { UNSCOPED_INFO("CPU error: " << cpu_result->GetError()); }
    REQUIRE_FALSE(cpu_result->HasError());
    auto const cpu = collect_rows(*cpu_result, false);
    cpu_result.reset();

    {
      INFO("sort_coalesce_runs = false");
      run_ok("SET sort_coalesce_runs = false;");
      auto const per_batch = run_on_gpu(query);
      REQUIRE(per_batch.rows == cpu);
      // Every input batch is its own run.
      CHECK(per_batch.sorts.runs == per_batch.sorts.input_batches);
      CHECK(per_batch.sorts.concatenated_bytes == 0);
    }
    run_ok("SET sort_coalesce_runs = true;");
    // From "the whole table is one run" down to runs of a few scan batches or a single one,
    // whatever the size of the test GPU.
    for (const char* fraction : {"1.0", "0.001", "0.0001", "0.00001", "0.000001"}) {
      INFO("sort_coalesce_memory_fraction = " << fraction);
      run_ok(std::string("SET sort_coalesce_memory_fraction = ") + fraction + ";");
      auto const coalesced = run_on_gpu(query);
      REQUIRE(coalesced.rows == cpu);
      CHECK(coalesced.sorts.runs <= coalesced.sorts.input_batches);
      if (expect_coalesced && std::string(fraction) == "1.0" && coalesced.sorts.input_batches > 1) {
        CHECK(coalesced.sorts.runs < coalesced.sorts.input_batches);
        CHECK(coalesced.sorts.concatenated_bytes > 0);
      }
      if (!expect_coalesced) { CHECK(coalesced.sorts.concatenated_bytes == 0); }
    }
    run_ok("RESET sort_coalesce_memory_fraction;");
  }

 private:
  std::vector<std::unique_ptr<sirius::test::scoped_sirius_setting>> settings;
};

}  // namespace

TEST_CASE_METHOD(OrderByCoalesceFixture,
                 "coalesced ORDER BY: fixed-width keys",
                 "[integration][gpu_execution][order_by_coalesce]")
{
  create_table("oc_int",
               "SELECT CASE WHEN i % 13 = 0 THEN NULL "
               "            ELSE CAST(hash(i) % 100003 AS INTEGER) - 50000 END AS ki, "
               "       CASE WHEN i % 17 = 0 THEN NULL "
               "            ELSE CAST(hash(i + 7) >> 1 AS BIGINT) - 4611686018427387904 END AS kb, "
               "       CAST(hash(i + 3) % 2000 AS DOUBLE) / 8 - 100 AS d, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(1500000) AS t(i)");

  check_order("SELECT ki, id FROM oc_int ORDER BY ki ASC NULLS FIRST, id");
  check_order("SELECT ki, id FROM oc_int ORDER BY ki DESC NULLS LAST, id DESC");
  check_order("SELECT kb, ki, id FROM oc_int ORDER BY kb DESC, id");
  check_order("SELECT d, ki, id FROM oc_int ORDER BY d, ki DESC NULLS FIRST, id");
  // The key is not part of the output, and the output is narrower than the table.
  check_order("SELECT id FROM oc_int ORDER BY kb, id DESC");
  // Already sorted and reverse sorted input.
  check_order("SELECT id, ki FROM oc_int ORDER BY id");
  check_order("SELECT id, ki FROM oc_int ORDER BY id DESC");
}

TEST_CASE_METHOD(OrderByCoalesceFixture,
                 "coalesced ORDER BY: string payload and string keys",
                 "[integration][gpu_execution][order_by_coalesce]")
{
  create_table("oc_str",
               "SELECT CAST(hash(i) % 5000 AS INTEGER) AS k, "
               "       CASE WHEN i % 19 = 0 THEN NULL "
               "            WHEN i % 31 = 0 THEN '' "
               "            ELSE repeat(chr(CAST(97 + hash(i) % 3 AS INTEGER)), "
               "                        CAST((hash(i) >> 8) % 40 AS INTEGER)) "
               "                 || CAST(hash(i + 1) % 100000 AS VARCHAR) END AS s, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(600000) AS t(i)");

  // A fixed-width key with a string payload: the strings are concatenated with their rows.
  check_order("SELECT k, s, id FROM oc_str ORDER BY k, id");
  check_order("SELECT s, id FROM oc_str ORDER BY k DESC, id DESC");
  // String keys are coalesced as well.
  check_order("SELECT s, id FROM oc_str ORDER BY s NULLS FIRST, id");
  check_order("SELECT k, s, id FROM oc_str ORDER BY k, s DESC NULLS LAST, id");
  // With sort_coalesce_string_keys off a string key keeps the local sort per batch; a
  // fixed-width key with a string payload is still coalesced.
  set("sort_coalesce_string_keys", "false");
  check_order("SELECT s, id FROM oc_str ORDER BY s NULLS FIRST, id", false);
  check_order("SELECT k, s, id FROM oc_str ORDER BY k, s DESC NULLS LAST, id", false);
  check_order("SELECT k, s, id FROM oc_str ORDER BY k, id");
}

TEST_CASE_METHOD(OrderByCoalesceFixture,
                 "coalesced ORDER BY: string keys are coalesced only with the encoded sort",
                 "[integration][gpu_execution][order_by_coalesce]")
{
  create_table("oc_enc",
               "SELECT CAST(hash(i) % 5000 AS INTEGER) AS k, "
               "       CAST(hash(i + 1) % 100000 AS VARCHAR) AS s, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(600000) AS t(i)");

  // Encoded sort on (the default): a string key is coalesced.
  check_order("SELECT s, id FROM oc_enc ORDER BY s, id");
  // Encoded sort off: a sort with a string key keeps the local sort per batch, whether the
  // string is the only key or one of several; a fixed-width key is still coalesced.
  set("sort_key_encoding", "false");
  check_order("SELECT s, id FROM oc_enc ORDER BY s, id", false);
  check_order("SELECT k, s, id FROM oc_enc ORDER BY k, s DESC, id", false);
  check_order("SELECT k, s, id FROM oc_enc ORDER BY k, id");
}

TEST_CASE_METHOD(OrderByCoalesceFixture,
                 "coalesced ORDER BY: under a LIMIT, an OFFSET, a filter and an aggregate",
                 "[integration][gpu_execution][order_by_coalesce]")
{
  create_table("oc_mix",
               "SELECT CAST(hash(i) % 1000 AS INTEGER) AS a, "
               "       CAST((i * 7919) % 1000003 AS BIGINT) AS b, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(1200000) AS t(i)");

  // A LIMIT too large for TOP_N keeps the full sort below it.
  check_order("SELECT a, id FROM oc_mix ORDER BY a, id LIMIT 900000");
  check_order("SELECT a, id FROM oc_mix ORDER BY a DESC, id LIMIT 500000 OFFSET 300000");
  check_order("SELECT b, a, id FROM oc_mix WHERE a % 3 = 1 ORDER BY b DESC, id");
  // The sort's input is an aggregate, not a scan.
  check_order("SELECT a, count(*) AS n, sum(b) AS s FROM oc_mix GROUP BY a ORDER BY n DESC, a");
  // No rows reach the sort.
  check_order("SELECT a, id FROM oc_mix WHERE a < 0 ORDER BY a, id");
  // A sort inside a subquery whose result is sorted again.
  check_order(
    "SELECT id, a FROM (SELECT * FROM oc_mix ORDER BY b, id LIMIT 700000) AS t ORDER BY a, id");
}

TEST_CASE_METHOD(OrderByCoalesceFixture,
                 "coalesced ORDER BY: settings",
                 "[integration][gpu_execution][order_by_coalesce]")
{
  auto const current = [&](const std::string& name) -> duckdb::Value {
    auto result = con->Query("SELECT current_setting('" + name + "')");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    return result->Collection().GetValue(0, 0);
  };
  CHECK(current("sort_coalesce_runs").GetValue<bool>());
  CHECK(current("sort_coalesce_string_keys").GetValue<bool>());
  CHECK(current("sort_coalesce_memory_fraction").GetValue<double>() == Approx(0.25));

  for (const char* bad : {"0", "-0.5", "1.5", "'nan'::DOUBLE"}) {
    auto result = con->Query(std::string("SET sort_coalesce_memory_fraction = ") + bad);
    REQUIRE(result);
    INFO(bad);
    CHECK(result->HasError());
  }
  CHECK(current("sort_coalesce_memory_fraction").GetValue<double>() == Approx(0.25));

  run_ok("SET sort_coalesce_memory_fraction = 0.5;");
  CHECK(current("sort_coalesce_memory_fraction").GetValue<double>() == Approx(0.5));
  run_ok("RESET sort_coalesce_memory_fraction;");
  CHECK(current("sort_coalesce_memory_fraction").GetValue<double>() == Approx(0.25));
}
