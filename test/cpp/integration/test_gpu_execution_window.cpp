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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace {

void require_ok(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

std::string strip_trailing_sql(const std::string& sql)
{
  auto clean = sql;
  while (!clean.empty() && (clean.back() == ';' || clean.back() == ' ')) {
    clean.pop_back();
  }
  return clean;
}

std::string sql_string_literal(const std::string& value)
{
  std::string escaped = "'";
  for (char c : value) {
    if (c == '\'') { escaped += '\''; }
    escaped += c;
  }
  escaped += "'";
  return escaped;
}

std::string query_single_value(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return result->GetValue(0, 0).ToString();
}

class runtime_setting_guard {
 public:
  runtime_setting_guard(duckdb::Connection& con, std::string setting, uint64_t value)
    : con(con),
      setting(std::move(setting)),
      old_value(query_single_value(con, "SELECT current_setting('" + this->setting + "')"))
  {
    require_ok(con, "SET " + this->setting + " = " + std::to_string(value));
  }

  ~runtime_setting_guard() { restore(); }

 private:
  void restore()
  {
    auto result = con.Query("SET " + setting + " = " + old_value);
    if (!result || result->HasError()) {
      WARN("Failed to restore " << setting << " to " << old_value
                                << (result ? ": " + result->GetError() : ""));
    }
  }

  duckdb::Connection& con;
  std::string setting;
  std::string old_value;
};

class window_runtime_settings_guard {
 public:
  window_runtime_settings_guard(duckdb::Connection& con,
                                uint64_t scan_task_batch_size,
                                uint64_t hash_partition_bytes)
    : scan_task_batch_size(con, "scan_task_batch_size", scan_task_batch_size),
      hash_partition_bytes(con, "hash_partition_bytes", hash_partition_bytes)
  {
  }

 private:
  runtime_setting_guard scan_task_batch_size;
  runtime_setting_guard hash_partition_bytes;
};

class WindowGPUExecutionFixture {
 public:
  WindowGPUExecutionFixture()
  {
    REQUIRE(sirius::test::g_integration_env != nullptr);
    REQUIRE(sirius::test::g_integration_env->is_active());
    con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());

    require_ok(*con, "SET enable_duckdb_fallback = false");
    require_ok(*con,
               "CREATE TEMP TABLE window_rank_input ("
               "  grp INTEGER,"
               "  subgroup INTEGER,"
               "  metric INTEGER,"
               "  id INTEGER"
               ")");
    require_ok(*con,
               "INSERT INTO window_rank_input VALUES "
               "(1, 10, 100, 1),"
               "(1, 10, 100, 2),"
               "(1, 10,  80, 3),"
               "(1, 20,  70, 4),"
               "(2, 10,  50, 5),"
               "(2, 10,  50, 6),"
               "(2, 20, NULL, 7),"
               "(NULL, 10, 40, 8),"
               "(NULL, 10, 30, 9)");
  }

  void materialize_gpu_and_cpu(const std::string& query)
  {
    auto clean_query = strip_trailing_sql(query);
    require_ok(*con, "DROP TABLE IF EXISTS window_gpu_result");
    require_ok(*con, "DROP TABLE IF EXISTS window_cpu_result");
    require_ok(*con,
               "CREATE TEMP TABLE window_gpu_result AS SELECT * FROM gpu_execution(" +
                 sql_string_literal(clean_query) + ")");
    require_ok(*con, "CREATE TEMP TABLE window_cpu_result AS " + clean_query);
  }

  void require_empty(const std::string& sql, const std::string& direction)
  {
    auto result = con->Query("SELECT count(*) FROM (" + sql + ") diff");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(direction << ": " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());

    auto count = result->GetValue(0, 0).GetValue<int64_t>();
    if (count != 0) { UNSCOPED_INFO(direction << " returned " << count << " unexpected rows"); }
    REQUIRE(count == 0);
  }

  void compare_materialized_results(
    const std::string& gpu_select = "SELECT * FROM window_gpu_result",
    const std::string& cpu_select = "SELECT * FROM window_cpu_result")
  {
    require_empty(gpu_select + " EXCEPT ALL " + cpu_select, "GPU minus CPU");
    require_empty(cpu_select + " EXCEPT ALL " + gpu_select, "CPU minus GPU");
  }

  void compare_gpu_vs_cpu(const std::string& query)
  {
    materialize_gpu_and_cpu(query);
    compare_materialized_results();
  }

  void require_gpu_row_count(int64_t expected_count)
  {
    auto result = con->Query("SELECT count(*) FROM window_gpu_result");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
    REQUIRE_FALSE(result->HasError());

    auto count = result->GetValue(0, 0).GetValue<int64_t>();
    if (count != expected_count) {
      UNSCOPED_INFO("GPU result row count " << count << " != expected " << expected_count);
    }
    REQUIRE(count == expected_count);
  }

  void compare_gpu_vs_cpu_and_require_rows(const std::string& query, int64_t expected_count)
  {
    materialize_gpu_and_cpu(query);
    require_gpu_row_count(expected_count);
    compare_materialized_results();
  }

  std::unique_ptr<duckdb::Connection> con;
};

}  // namespace

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window row_number partitions and NULL groups",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT grp, id, "
    "       row_number() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST, id ASC) AS rn "
    "FROM window_rank_input");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window rank and dense_rank ties",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT id, "
    "       rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST) AS rnk, "
    "       dense_rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST) AS dr "
    "FROM window_rank_input "
    "WHERE grp = 1");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window ASC NULLS FIRST ordering",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT id, "
    "       rank() OVER (PARTITION BY grp ORDER BY metric ASC NULLS FIRST) AS rnk, "
    "       dense_rank() OVER (PARTITION BY grp ORDER BY metric ASC NULLS FIRST) AS dr "
    "FROM window_rank_input "
    "WHERE grp = 2");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window multi-column partition and order keys",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT grp, subgroup, id, "
    "       row_number() OVER ("
    "         PARTITION BY grp, subgroup "
    "         ORDER BY metric DESC NULLS LAST, id DESC"
    "       ) AS rn "
    "FROM window_rank_input "
    "WHERE grp IS NOT NULL");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window ranking ignores frame clause",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT id, "
    "       rank() OVER ("
    "         PARTITION BY grp "
    "         ORDER BY metric DESC "
    "         ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"
    "       ) AS rnk, "
    "       dense_rank() OVER ("
    "         PARTITION BY grp "
    "         ORDER BY metric DESC "
    "         ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"
    "       ) AS dr "
    "FROM window_rank_input "
    "WHERE grp = 1");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window row_number without order preserves numbering invariant",
                 "[integration][window][gpu]")
{
  materialize_gpu_and_cpu(
    "SELECT grp, row_number() OVER (PARTITION BY grp) AS rn FROM window_rank_input");

  compare_materialized_results(
    "SELECT coalesce(grp, -1) AS grp_key, "
    "       min(rn) AS min_rn, "
    "       max(rn) AS max_rn, "
    "       count(DISTINCT rn) AS distinct_rn, "
    "       count(*) AS row_count "
    "FROM window_gpu_result "
    "GROUP BY grp_key",
    "SELECT coalesce(grp, -1) AS grp_key, "
    "       min(rn) AS min_rn, "
    "       max(rn) AS max_rn, "
    "       count(DISTINCT rn) AS distinct_rn, "
    "       count(*) AS row_count "
    "FROM window_cpu_result "
    "GROUP BY grp_key");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window supports independent nested LogicalWindow nodes",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT grp, subgroup, id, rn_by_grp, "
    "       row_number() OVER ("
    "         PARTITION BY subgroup "
    "         ORDER BY rn_by_grp DESC, id ASC"
    "       ) AS rn_by_subgroup "
    "FROM ("
    "  SELECT grp, subgroup, metric, id, "
    "         row_number() OVER ("
    "           PARTITION BY grp "
    "           ORDER BY metric DESC NULLS LAST, id ASC"
    "         ) AS rn_by_grp "
    "  FROM window_rank_input"
    ") ranked_by_grp");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window pure row_number top-N per group query",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT grp, subgroup, metric, id, rn "
    "FROM ("
    "  SELECT grp, subgroup, metric, id, "
    "         row_number() OVER ("
    "           PARTITION BY grp, subgroup "
    "           ORDER BY metric DESC NULLS LAST, id ASC"
    "         ) AS rn "
    "  FROM window_rank_input"
    ") ranked "
    "WHERE rn <= 2");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window rank dense_rank multi-partition mixed order",
                 "[integration][window][gpu]")
{
  compare_gpu_vs_cpu(
    "SELECT grp, subgroup, metric, id, "
    "       rank() OVER ("
    "         PARTITION BY grp, subgroup "
    "         ORDER BY metric DESC NULLS LAST, id ASC"
    "       ) AS rnk, "
    "       dense_rank() OVER ("
    "         PARTITION BY grp, subgroup "
    "         ORDER BY metric DESC NULLS LAST, id ASC"
    "       ) AS dr "
    "FROM window_rank_input "
    "WHERE grp IS NOT NULL");
}

TEST_CASE_METHOD(WindowGPUExecutionFixture,
                 "gpu_execution window multi-batch and multi-partition ranking",
                 "[integration][window][gpu]")
{
  constexpr int64_t kRows = 30000;
  window_runtime_settings_guard settings(*con, 2048, 65536);

  std::string create_multibatch_input =
    "CREATE TEMP TABLE window_multibatch_input AS "
    "SELECT CAST(i % 50 AS INTEGER) AS grp, "
    "       CAST(i % 1000 AS INTEGER) AS metric, "
    "       CAST(i AS INTEGER) AS id "
    "FROM range(";
  create_multibatch_input += std::to_string(kRows);
  create_multibatch_input += ") AS t(i)";
  require_ok(*con, create_multibatch_input);

  std::string create_multibatch_null_input =
    "CREATE TEMP TABLE window_multibatch_null_input AS "
    "SELECT CASE WHEN i % 17 = 0 THEN NULL ELSE CAST(i % 50 AS INTEGER) END AS grp, "
    "       CASE WHEN i % 19 = 0 THEN NULL ELSE CAST(i % 1000 AS INTEGER) END "
    "         AS metric, "
    "       CAST(i AS INTEGER) AS id "
    "FROM range(";
  create_multibatch_null_input += std::to_string(kRows);
  create_multibatch_null_input += ") AS t(i)";
  require_ok(*con, create_multibatch_null_input);

  {
    INFO("row_number uses a deterministic mixed ASC/DESC order across scan batches");
    compare_gpu_vs_cpu_and_require_rows(
      "SELECT grp, metric, id, "
      "       row_number() OVER ("
      "         PARTITION BY grp "
      "         ORDER BY metric DESC NULLS LAST, id ASC"
      "       ) AS rn "
      "FROM window_multibatch_input",
      kRows);
  }

  {
    INFO("rank and dense_rank keep peer groups that span scan batches");
    compare_gpu_vs_cpu_and_require_rows(
      "SELECT grp, metric, id, "
      "       rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST) AS rnk, "
      "       dense_rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST) AS dr "
      "FROM window_multibatch_input",
      kRows);
  }

  {
    INFO("NULL partition keys and NULL order values survive multi-batch row_number");
    compare_gpu_vs_cpu_and_require_rows(
      "SELECT grp, metric, id, "
      "       row_number() OVER ("
      "         PARTITION BY grp "
      "         ORDER BY metric ASC NULLS FIRST, id DESC"
      "       ) AS rn "
      "FROM window_multibatch_null_input",
      kRows);
  }

  {
    INFO("NULL partition keys and NULL order values survive multi-batch rank and dense_rank");
    compare_gpu_vs_cpu_and_require_rows(
      "SELECT grp, metric, id, "
      "       rank() OVER (PARTITION BY grp ORDER BY metric ASC NULLS FIRST) AS rnk, "
      "       dense_rank() OVER (PARTITION BY grp ORDER BY metric ASC NULLS FIRST) AS dr "
      "FROM window_multibatch_null_input",
      kRows);
  }

  {
    INFO("NULL order values survive multi-batch DESC NULLS LAST rank and dense_rank");
    compare_gpu_vs_cpu_and_require_rows(
      "SELECT grp, metric, id, "
      "       rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST) AS rnk, "
      "       dense_rank() OVER (PARTITION BY grp ORDER BY metric DESC NULLS LAST) AS dr "
      "FROM window_multibatch_null_input",
      kRows);
  }
}
