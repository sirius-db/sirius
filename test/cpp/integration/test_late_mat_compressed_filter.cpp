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

// A deferral over a filtered scan of a COMPRESSED pin. Rows can be dropped in
// two places: inside the fused decode, and again by the conjuncts the decode
// could not carry. The pin-order rowid is only correct when both stages are
// composed.
//
// Which stages run depends on SIRIUS_EXP_FUSED_SCAN_FILTER, whose value caches
// on first read and so cannot be varied within one binary. The case holds under
// either setting; with the gate on it also covers the decode-compacted stage.
//
// Requires SIRIUS_EXP_LATE_MAT=1 in the ENVIRONMENT: the gate is read per process.

#include <catch.hpp>
#include <compression/decompression_pushdown_policy.hpp>
#include <duckdb.hpp>
#include <late_mat/column_origin.hpp>
#include <late_mat/defer_directive.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/pinned_entry_census.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kCustomers = 20'000;
constexpr std::int64_t kOrders    = 60'000;
constexpr std::int64_t kLines     = 120'000;

/// Three wide strings read only by the aggregate at the far end, two joins to
/// clear the crossing floor, and a range predicate on the pinned table so the
/// scan restricts rows. The predicate is a bare range on the key, which a
/// bitpack-rooted plan can answer inside the decode.
constexpr char const* kQuery =
  "SELECT c.c_custkey, c.c_name, c.c_address, c.c_comment, count(*) AS n "
  "FROM read_parquet('{C}') c "
  "JOIN read_parquet('{O}') o ON c.c_custkey = o.o_custkey "
  "JOIN read_parquet('{L}') l ON o.o_orderkey = l.l_orderkey "
  "WHERE c.c_custkey < 4000 {X}"
  "GROUP BY c.c_custkey, c.c_name, c.c_address, c.c_comment "
  "ORDER BY c.c_custkey";

/// A conjunct on an identity-planned string column, which no decode route can
/// answer. It leaves the range pushed down but the request no longer covering
/// the whole filter, so the decode compacts and the residual filter then runs
/// over what it kept — the two stages whose positions have to be composed.
constexpr char const* kResidualConjunct = "AND c.c_name LIKE '%7' ";

std::string query_for(fs::path const& customer,
                      fs::path const& orders,
                      fs::path const& lines,
                      bool residual = false)
{
  std::string sql = kQuery;
  sql.replace(sql.find("{X}"), 3, residual ? kResidualConjunct : "");
  sql.replace(sql.find("{C}"), 3, customer.string());
  sql.replace(sql.find("{O}"), 3, orders.string());
  sql.replace(sql.find("{L}"), 3, lines.string());
  return sql;
}

std::vector<std::string> rows_of(duckdb::MaterializedQueryResult& result)
{
  std::vector<std::string> rows;
  for (duckdb::idx_t i = 0; i < result.RowCount(); ++i) {
    std::string row;
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      row += result.GetValue(c, i).ToString() + "|";
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

void generate_parquet(fs::path const& customer, fs::path const& orders, fs::path const& lines)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  auto r = gen.Query(
    "COPY (SELECT range AS c_custkey, "
    "             'Customer#' || lpad(CAST(range AS VARCHAR), 12, '0') AS c_name, "
    "             'address-' || repeat(CAST(range % 97 AS VARCHAR), 6) AS c_address, "
    "             'comment for customer ' || CAST(range AS VARCHAR) AS c_comment "
    "      FROM range(" +
    std::to_string(kCustomers) + ")) TO " + sirius::test::sql_literal(customer.string()) +
    " (FORMAT PARQUET);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  r = gen.Query(
    "COPY (SELECT range AS o_orderkey, "
    "             range % " +
    std::to_string(kCustomers) +
    " AS o_custkey "
    "      FROM range(" +
    std::to_string(kOrders) + ")) TO " + sirius::test::sql_literal(orders.string()) +
    " (FORMAT PARQUET);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  r = gen.Query("COPY (SELECT range AS l_linekey, range % " + std::to_string(kOrders) +
                " AS l_orderkey "
                "      FROM range(" +
                std::to_string(kLines) + ")) TO " + sirius::test::sql_literal(lines.string()) +
                " (FORMAT PARQUET);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
}

/// The same query on DuckDB with Sirius disabled.
std::vector<std::string> cpu_answer(fs::path const& customer,
                                    fs::path const& orders,
                                    fs::path const& lines,
                                    bool residual)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto r = con.Query(query_for(customer, orders, lines, residual));
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  return rows_of(*r);
}

/// A bitpack block for the key, so its decode produces a range-answering mask —
/// the plan shape that lets the decode drop rows at all. The string payloads sit
/// on identity and take the full-decode route, compacted by the survivor gather.
void write_plan_file(fs::path const& plan_dir, std::string const& table)
{
  fs::create_directories(plan_dir);
  std::ofstream f(plan_dir / (table + ".txt"));
  f << "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
       "---\n"
       "input -> identity\n"
       "---\n"
       "input -> identity\n"
       "---\n"
       "input -> identity\n";
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.4\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 4\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: 100000000\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

}  // namespace

TEST_CASE("a deferral over a filtered compressed pin rebuilds its rowid from both filter stages",
          "[late_mat][compression][compressed_filter]")
{
  if (!sirius::late_mat::late_mat_enabled()) {
    WARN("SIRIUS_EXP_LATE_MAT unset; skipping the filtered compressed-pin case");
    return;
  }
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  sirius::test::scratch_dir scratch{"late_mat_compressed_filter"};
  auto const& tmp     = scratch.path();
  auto const customer = tmp / "lm_comp_customer.parquet";
  auto const orders   = tmp / "orders.parquet";
  auto const lines    = tmp / "lineitem.parquet";
  generate_parquet(customer, orders, lines);
  auto const expected          = cpu_answer(customer, orders, lines, /*residual=*/false);
  auto const expected_composed = cpu_answer(customer, orders, lines, /*residual=*/true);
  REQUIRE_FALSE(expected.empty());
  REQUIRE_FALSE(expected_composed.empty());

  auto yaml_path = tmp / "late_mat_compressed_filter.yaml";
  write_config(yaml_path);

  sirius::test::shared_test_env local_env(yaml_path);
  auto con = local_env.make_connection();
  auto fb  = con.Query("SET enable_duckdb_fallback = false;");
  REQUIRE(fb);
  REQUIRE_FALSE(fb->HasError());

  auto const plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "lm_comp_customer");
  for (auto const& setting :
       {std::string{"SET pin_table_compression = true;"},
        std::string{"SET pin_table_compression_min_batch_size_bytes = 0;"},
        std::string{"SET pin_table_compression_max_compressed_fraction = 1.5;"},
        "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';"}) {
    auto r = con.Query(setting);
    REQUIRE(r);
    if (r->HasError()) { UNSCOPED_INFO(setting << " error: " << r->GetError()); }
    REQUIRE_FALSE(r->HasError());
  }

  ::setenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", "c_custkey", 1);

  auto pin = con.Query("CALL pin_table(" + sirius::test::sql_literal(customer.string()) +
                       ", tier='gpu', name='lm_comp_customer');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // Identity blocks on the payloads barely shrink the chunk, so the savings
  // gate is what decides whether it stays compressed at all.
  REQUIRE(sirius::test::census_entry(con, "lm_comp_customer").compressed_chunks > 0);

  // Whole filter in the decode: the batch arrives compacted and nothing filters
  // it again, so the rowid comes from the decode's positions alone.
  auto before = sirius::late_mat::deferrals_installed();
  auto res    = con.Query(query_for(customer, orders, lines, /*residual=*/false));
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO("query error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());
  REQUIRE(rows_of(*res) == expected);
  // A query that deferred nothing returns the same answer, so the count is what
  // makes the comparison above mean anything.
  REQUIRE(sirius::late_mat::deferrals_installed() > before);

  // Part of the filter in the decode and the rest after it: the two stages
  // report positions in different coordinate systems and must be composed.
  before = sirius::late_mat::deferrals_installed();
  res    = con.Query(query_for(customer, orders, lines, /*residual=*/true));
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO("composed query error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());
  REQUIRE(rows_of(*res) == expected_composed);
  REQUIRE(sirius::late_mat::deferrals_installed() > before);

  if (!sirius::codegen::decompression_pushdown_enabled()) {
    WARN("SIRIUS_EXP_FUSED_SCAN_FILTER unset; the decode-compacted stage was not exercised");
  }

  ::unsetenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS");
  auto unpin = con.Query("CALL unpin_table('lm_comp_customer');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());
}
