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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kWarmupRuns  = 2;
constexpr int kMeasureRuns = 5;

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
  if (!result) { throw std::runtime_error("Query returned a null result"); }
  if (result->HasError()) { throw std::runtime_error(result->GetError()); }
  return result->GetValue(0, 0).ToString();
}

void query_checked(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query(sql);
  if (!result) { throw std::runtime_error("Query returned a null result"); }
  if (result->HasError()) { throw std::runtime_error(result->GetError()); }
}

double time_query_once_ms(duckdb::Connection& con, const std::string& sql)
{
  const auto start = std::chrono::steady_clock::now();
  auto result      = con.Query(sql);
  const auto end   = std::chrono::steady_clock::now();

  if (!result) { throw std::runtime_error("Query returned a null result"); }
  if (result->HasError()) { throw std::runtime_error(result->GetError()); }

  return std::chrono::duration<double, std::milli>(end - start).count();
}

double median_query_ms(duckdb::Connection& con, const std::string& sql)
{
  for (int i = 0; i < kWarmupRuns; i++) {
    query_checked(con, sql);
  }

  std::vector<double> samples;
  samples.reserve(kMeasureRuns);
  for (int i = 0; i < kMeasureRuns; i++) {
    samples.push_back(time_query_once_ms(con, sql));
  }

  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::string gpu_call_sql(const std::string& query)
{
  return "CALL gpu_execution(" + sql_string_literal(strip_trailing_sql(query)) + ")";
}

std::string gpu_table_sql(const std::string& query)
{
  return "SELECT * FROM gpu_execution(" + sql_string_literal(strip_trailing_sql(query)) + ")";
}

int64_t count_query_rows(duckdb::Connection& con, const std::string& sql)
{
  auto result = con.Query("SELECT count(*) FROM (" + sql + ") rows_to_count");
  if (!result) { throw std::runtime_error("count query returned a null result"); }
  if (result->HasError()) { throw std::runtime_error(result->GetError()); }
  return result->GetValue(0, 0).GetValue<int64_t>();
}

std::string lower_copy(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

std::string classify_error(const std::string& error)
{
  const auto lower = lower_copy(error);
  if (lower.find("out of memory") != std::string::npos || lower.find("oom") != std::string::npos) {
    return "OOM";
  }
  if (lower.find("not implemented") != std::string::npos ||
      lower.find("notimplemented") != std::string::npos) {
    return "NotImplemented";
  }
  return "ERROR";
}

std::string compact_error(const std::string& error)
{
  auto compact = error;
  std::replace(compact.begin(), compact.end(), '\n', ' ');
  if (compact.size() > 180) {
    compact.resize(177);
    compact += "...";
  }
  return compact;
}

std::string format_ms(double value)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << value;
  return out.str();
}

std::string format_speedup(double cpu_ms, double gpu_ms)
{
  if (gpu_ms <= 0.0) { return "n/a"; }
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << (cpu_ms / gpu_ms) << "x";
  return out.str();
}

fs::path tpcds_db_path()
{
  const char* env = std::getenv("SIRIUS_TPCDS_TEST_DB_PATH");
  if (!env) { env = std::getenv("SIRIUS_INTEGRATION_TEST_DB_PATH"); }
  return env ? fs::path(env) : fs::path(__FILE__).parent_path() / "data/duckdb/tpcds.duckdb";
}

std::string q44_ranking_query()
{
  return R"SQL(
SELECT low_side.rnk,
       i1.i_product_name best_performing,
       i2.i_product_name worst_performing
FROM
  (SELECT *
   FROM
     (SELECT item_sk,
             rank() OVER (ORDER BY rank_col ASC) rnk
      FROM
        (SELECT ss_item_sk item_sk,
                avg(ss_net_profit) rank_col
         FROM store_sales ss1
         WHERE ss_store_sk = 1
         GROUP BY ss_item_sk
         HAVING avg(ss_net_profit) > 0.9 *
           (SELECT avg(ss_net_profit) rank_col
            FROM store_sales
            WHERE ss_store_sk = 1
              AND ss_addr_sk IS NULL
            GROUP BY ss_store_sk)) v1) v11
   WHERE rnk < 11) low_side,
  (SELECT *
   FROM
     (SELECT item_sk,
             rank() OVER (ORDER BY rank_col DESC) rnk
      FROM
        (SELECT ss_item_sk item_sk,
                avg(ss_net_profit) rank_col
         FROM store_sales ss1
         WHERE ss_store_sk = 1
         GROUP BY ss_item_sk
         HAVING avg(ss_net_profit) > 0.9 *
           (SELECT avg(ss_net_profit) rank_col
            FROM store_sales
            WHERE ss_store_sk = 1
              AND ss_addr_sk IS NULL
            GROUP BY ss_store_sk)) v2) v21
   WHERE rnk < 11) high_side,
  item i1,
  item i2
WHERE low_side.rnk = high_side.rnk
  AND i1.i_item_sk = low_side.item_sk
  AND i2.i_item_sk = high_side.item_sk
ORDER BY low_side.rnk
LIMIT 100
)SQL";
}

std::string synthetic_ranking_query()
{
  return "SELECT grp, id, "
         "       row_number() OVER (PARTITION BY grp ORDER BY metric DESC, id) AS rn "
         "FROM bench_w";
}

class WindowBenchmarkFixture {
 public:
  WindowBenchmarkFixture()
  {
    REQUIRE(sirius::test::g_integration_env != nullptr);
    REQUIRE(sirius::test::g_integration_env->is_active());
    con = std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    query_checked(*con, "SET enable_duckdb_fallback = false");
  }

  bool attach_tpcds_if_available()
  {
    const auto path = tpcds_db_path();
    if (!fs::exists(path)) {
      std::cout << "Q44 | SKIP | fixture_missing=" << path.string() << '\n';
      return false;
    }

    query_checked(
      *con, "ATTACH IF NOT EXISTS " + sql_string_literal(path.string()) + " AS tpcds (READ_ONLY)");
    query_checked(*con, "USE tpcds");
    return true;
  }

  void print_config_note()
  {
    std::cout << "config | scan_task_batch_size="
              << query_single_value(*con, "SELECT current_setting('scan_task_batch_size')")
              << " | hash_partition_bytes="
              << query_single_value(*con, "SELECT current_setting('hash_partition_bytes')") << '\n';
  }

  void create_synthetic_table(int64_t rows)
  {
    query_checked(*con, "DROP TABLE IF EXISTS bench_w");
    std::string sql =
      "CREATE TEMP TABLE bench_w AS "
      "SELECT CAST(i % 100 AS INTEGER) AS grp, "
      "       CAST(i % 1000 AS INTEGER) AS metric, "
      "       CAST(i AS BIGINT) AS id "
      "FROM range(";
    sql += std::to_string(rows);
    sql += ") AS t(i)";
    query_checked(*con, sql);
  }

  void create_skew_table(int64_t rows)
  {
    query_checked(*con, "DROP TABLE IF EXISTS bench_w");
    std::string sql =
      "CREATE TEMP TABLE bench_w AS "
      "SELECT 0::INTEGER AS grp, "
      "       CAST(i % 1000 AS INTEGER) AS metric, "
      "       CAST(i AS BIGINT) AS id "
      "FROM range(";
    sql += std::to_string(rows);
    sql += ") AS t(i)";
    query_checked(*con, sql);
  }

  std::unique_ptr<duckdb::Connection> con;
};

void print_cpu_gpu_row(const std::string& label,
                       duckdb::Connection& con,
                       const std::string& query,
                       int64_t expected_rows)
{
  try {
    const auto cpu_ms = median_query_ms(con, query);
    const auto gpu_ms = median_query_ms(con, gpu_call_sql(query));

    if (expected_rows >= 0) {
      const auto gpu_rows = count_query_rows(con, gpu_table_sql(query));
      CHECK(gpu_rows == expected_rows);
    }

    std::cout << label << " | " << format_ms(cpu_ms) << " | " << format_ms(gpu_ms) << " | "
              << format_speedup(cpu_ms, gpu_ms) << '\n';
  } catch (const std::exception& e) {
    std::cout << label << " | " << classify_error(e.what()) << "@" << label << " | clean-error | "
              << compact_error(e.what()) << '\n';
  }
}

void print_skew_row(const std::string& label,
                    duckdb::Connection& con,
                    const std::string& query,
                    int64_t expected_rows)
{
  try {
    const auto gpu_ms   = median_query_ms(con, gpu_call_sql(query));
    const auto gpu_rows = count_query_rows(con, gpu_table_sql(query));
    CHECK(gpu_rows == expected_rows);

    std::cout << label << " | OK | " << format_ms(gpu_ms) << " ms | rows=" << gpu_rows << '\n';
  } catch (const std::exception& e) {
    std::cout << label << " | clean-error | " << classify_error(e.what()) << "@" << label << " | "
              << compact_error(e.what()) << '\n';
  }
}

}  // namespace

TEST_CASE_METHOD(WindowBenchmarkFixture,
                 "window ranking CPU vs GPU end-to-end benchmark",
                 "[.][integration][benchmark][window]")
{
  print_config_note();
  std::cout << "window_cpu_gpu_e2e | warmup=" << kWarmupRuns << " | runs=" << kMeasureRuns
            << " | median_ms\n";
  std::cout << "N_or_query | CPU_ms | GPU_e2e_ms | speedup\n";

  for (const auto rows : std::vector<int64_t>{50000, 500000, 5000000, 50000000}) {
    try {
      create_synthetic_table(rows);
      print_cpu_gpu_row(std::to_string(rows), *con, synthetic_ranking_query(), rows);
    } catch (const std::exception& e) {
      std::cout << rows << " | " << classify_error(e.what()) << "@" << rows << " | clean-error | "
                << compact_error(e.what()) << '\n';
    }
  }

  try {
    if (attach_tpcds_if_available()) {
      print_cpu_gpu_row("Q44_SF0.01", *con, q44_ranking_query(), -1);
    }
  } catch (const std::exception& e) {
    std::cout << "Q44_SF0.01 | " << classify_error(e.what()) << " | clean-error | "
              << compact_error(e.what()) << '\n';
  }
}

TEST_CASE_METHOD(WindowBenchmarkFixture,
                 "window ranking skew benchmark",
                 "[.][integration][benchmark][window][skew]")
{
  print_config_note();
  std::cout << "window_skew_hot_partition | warmup=" << kWarmupRuns << " | runs=" << kMeasureRuns
            << " | median_ms\n";
  std::cout << "N | behavior | GPU_e2e_ms_or_error | rows\n";

  for (const auto rows : std::vector<int64_t>{5000000, 50000000}) {
    try {
      create_skew_table(rows);
      print_skew_row(std::to_string(rows), *con, synthetic_ranking_query(), rows);
    } catch (const std::exception& e) {
      std::cout << rows << " | clean-error | " << classify_error(e.what()) << "@" << rows << " | "
                << compact_error(e.what()) << '\n';
    }
  }
}
