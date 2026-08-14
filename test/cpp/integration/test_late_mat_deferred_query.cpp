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

// A deferral installed and executed by a real query, end to end. GPU required,
// and SIRIUS_EXP_LATE_MAT must be set in the ENVIRONMENT — the gate is read
// once per process, so this case skips rather than lying when it is off.
//
// Everything else in the suite tests a half: the walk decides on hand-built
// plans, the port materializes against a hand-built pin. Only here does a plan
// the generator built carry a payload across the wrap chain it actually
// inserts, substituted by the scan that served it and restored by the operator
// that reads it.
//
// THE ASSERTION IS TWO-PART, and the second half is what makes the first mean
// anything: the answer must match the CPU's, AND a deferral must actually have
// installed. A query that deferred nothing also returns the right answer.
//
// Shape: three wide strings from a pinned table ride an INNER join and are read
// by an aggregate that is not grouped on them. Grouping on them instead would
// make them partition keys, which the walk refuses — restoring a group key
// needs the rowid to survive the group-by, which is a different feature.

#include <catch.hpp>
#include <duckdb.hpp>
#include <late_mat/column_origin.hpp>
#include <late_mat/defer_directive.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kCustomers = 20'000;
constexpr std::int64_t kOrders    = 60'000;
constexpr std::int64_t kLines     = 120'000;

/// The payload columns are wide enough to clear the policy's value floor
/// (3 x 24 B estimated, less the 8 B rowid) and are touched only by the
/// aggregate at the far end. TWO joins, because the floor is four port
/// crossings and one join's wrap chain is three: a join streams into the
/// group-by that consumes it, so there is no port between those two.
constexpr char const* kQuery =
  "SELECT o.o_status, max(c.c_name), max(c.c_address), max(c.c_comment) "
  "FROM read_parquet('{C}') c "
  "JOIN read_parquet('{O}') o ON c.c_custkey = o.o_custkey "
  "JOIN read_parquet('{L}') l ON o.o_orderkey = l.l_orderkey "
  "GROUP BY o.o_status ORDER BY o.o_status";

std::string query_for(fs::path const& customer, fs::path const& orders, fs::path const& lines)
{
  std::string sql = kQuery;
  sql.replace(sql.find("{C}"), 3, customer.string());
  sql.replace(sql.find("{O}"), 3, orders.string());
  sql.replace(sql.find("{L}"), 3, lines.string());
  return sql;
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
    " AS o_custkey, "
    "             CAST(range % 3 AS VARCHAR) AS o_status "
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

/// The ground truth: the same query on DuckDB with Sirius disabled.
std::vector<std::string> cpu_answer(fs::path const& customer,
                                    fs::path const& orders,
                                    fs::path const& lines)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto r = con.Query(query_for(customer, orders, lines));
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  std::vector<std::string> rows;
  for (duckdb::idx_t i = 0; i < r->RowCount(); ++i) {
    std::string row;
    for (duckdb::idx_t c = 0; c < r->ColumnCount(); ++c) {
      row += r->GetValue(c, i).ToString() + "|";
    }
    rows.push_back(std::move(row));
  }
  return rows;
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

TEST_CASE("a deferred payload rides a real plan and comes back right", "[late_mat][deferred_query]")
{
  if (!sirius::late_mat::late_mat_enabled()) {
    WARN("SIRIUS_EXP_LATE_MAT unset; skipping the end-to-end deferral case");
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

  sirius::test::scratch_dir scratch{"late_mat_query"};
  auto const& tmp     = scratch.path();
  auto const customer = tmp / "customer.parquet";
  auto const orders   = tmp / "orders.parquet";
  auto const lines    = tmp / "lineitem.parquet";
  generate_parquet(customer, orders, lines);
  auto const expected = cpu_answer(customer, orders, lines);
  REQUIRE_FALSE(expected.empty());

  auto yaml_path = tmp / "late_mat_query.yaml";
  write_config(yaml_path);

  sirius::test::shared_test_env local_env(yaml_path);
  auto con = local_env.make_connection();
  auto fb  = con.Query("SET enable_duckdb_fallback = false;");
  REQUIRE(fb);
  REQUIRE_FALSE(fb->HasError());

  // GPU tier: a deferral addresses rows by their position in device-resident
  // pinned storage, so nothing installs against a host pin.
  auto pin = con.Query("CALL pin_table(" + sirius::test::sql_literal(customer.string()) +
                       ", tier='gpu', name='late_mat_customer');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  auto const before = sirius::late_mat::deferrals_installed();
  auto res          = con.Query(query_for(customer, orders, lines));
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO("query error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());

  std::vector<std::string> got;
  for (duckdb::idx_t i = 0; i < res->RowCount(); ++i) {
    std::string row;
    for (duckdb::idx_t c = 0; c < res->ColumnCount(); ++c) {
      row += res->GetValue(c, i).ToString() + "|";
    }
    got.push_back(std::move(row));
  }
  REQUIRE(got == expected);

  // Without this the case above passes just as well on a query that deferred
  // nothing, which is the failure mode a correctness test is least able to see.
  REQUIRE(sirius::late_mat::deferrals_installed() > before);

  auto unpin = con.Query("CALL unpin_table('late_mat_customer');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());
}
