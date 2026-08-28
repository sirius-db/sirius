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

// The same end-to-end shape as test_late_mat_deferred_query.cpp, over a pin that
// lives on the HOST tier. GPU required, and SIRIUS_EXP_LATE_MAT must be set in
// the ENVIRONMENT — the gate is read once per process, so this case skips rather
// than lying when it is off.
//
// What is different from the GPU case, and why the payload looks the way it
// does:
//
//  * The payload columns are FIXED WIDTH. A host chunk stores its columns over
//    pinned blocks that are not contiguous, so a variable-width column has no
//    offsets the gather can rebuild and is refused; the GPU case's three wide
//    strings would install nothing here.
//  * There are many of them. Materializing from the host costs more than from
//    the device, so the value floor is multiplied (host_tier_cost_multiplier),
//    and the bundle has to be correspondingly wider to clear it. The count is
//    sized to clear the default floor even at the minimum crossing count.
//
// THE ASSERTION IS TWO-PART, and the second half is what makes the first mean
// anything: the answer must match the CPU's, AND a deferral must actually have
// installed. A query that deferred nothing also returns the right answer.

#include <catch.hpp>
#include <duckdb.hpp>
#include <late_mat/column_origin.hpp>
#include <late_mat/defer_directive.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kCustomers = 5'000;
constexpr std::int64_t kOrders    = 15'000;
constexpr std::int64_t kLines     = 30'000;

/// Payload width. The bundle must clear the host-tier product floor, which is
/// 128 times a multiplier the startup probe measures (13 on a GB300). The ride
/// carries one rowid plus a placeholder per remaining column, so N BIGINTs save
/// 8N - (8 + N - 1) per row, and this shape crosses 6 ports. 56 columns save 385
/// B/row for 2310, clearing a 1664 floor; 40 would sit just under at 1638.
constexpr int kPayloadColumns = 56;

std::string payload_names(char const* prefix)
{
  std::string out;
  for (int i = 0; i < kPayloadColumns; ++i) {
    out += std::string(", ") + prefix + "p" + std::to_string(i);
  }
  return out;
}

std::string payload_definitions()
{
  std::string out;
  for (int i = 0; i < kPayloadColumns; ++i) {
    // Distinct per column and per row, so a gather that read the wrong column or
    // the wrong row cannot coincide with the right answer.
    out += ", CAST(range * 100 + " + std::to_string(i) + " AS BIGINT) AS p" + std::to_string(i);
  }
  return out;
}

/// Two INNER joins, then a GROUP BY on the key and the whole payload — the
/// group-by-rowid shape, materializing one row per group at the far end.
std::string query_for(fs::path const& customer, fs::path const& orders, fs::path const& lines)
{
  return "SELECT c.c_custkey" + payload_names("c.") +
         ", count(*) AS n "
         "FROM read_parquet('" +
         customer.string() +
         "') c "
         "JOIN read_parquet('" +
         orders.string() +
         "') o ON c.c_custkey = o.o_custkey "
         "JOIN read_parquet('" +
         lines.string() +
         "') l ON o.o_orderkey = l.l_orderkey "
         "GROUP BY c.c_custkey" +
         payload_names("c.") + " ORDER BY c.c_custkey";
}

void generate_parquet(fs::path const& customer, fs::path const& orders, fs::path const& lines)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  auto r = gen.Query("COPY (SELECT range AS c_custkey" + payload_definitions() + " FROM range(" +
                     std::to_string(kCustomers) + ")) TO " +
                     sirius::test::sql_literal(customer.string()) + " (FORMAT PARQUET);");
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
  return rows_of(*r);
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

TEST_CASE("a deferred payload rides a real plan off a host-tier pin",
          "[late_mat][deferred_query][host_pin]")
{
  if (!sirius::late_mat::late_mat_enabled()) {
    WARN("SIRIUS_EXP_LATE_MAT unset; skipping the end-to-end host-tier deferral case");
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

  sirius::test::scratch_dir scratch{"late_mat_host_query"};
  auto const& tmp     = scratch.path();
  auto const customer = tmp / "customer.parquet";
  auto const orders   = tmp / "orders.parquet";
  auto const lines    = tmp / "lineitem.parquet";
  generate_parquet(customer, orders, lines);
  auto const expected = cpu_answer(customer, orders, lines);
  REQUIRE_FALSE(expected.empty());

  auto yaml_path = tmp / "late_mat_host_query.yaml";
  write_config(yaml_path);

  sirius::test::shared_test_env local_env(yaml_path);
  auto con = local_env.make_connection();
  auto fb  = con.Query("SET enable_duckdb_fallback = false;");
  REQUIRE(fb);
  REQUIRE_FALSE(fb->HasError());

  // c_custkey rides REAL — it is the join key and a group key — so proving it
  // distinct over the pin is what admits the ride past the aggregates. The probe
  // runs on the GPU table each batch is materialized as, before it is copied to
  // the host, so a host pin proves its columns exactly as a GPU pin does.
  ::setenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", "c_custkey", 1);

  auto pin = con.Query("CALL pin_table(" + sirius::test::sql_literal(customer.string()) +
                       ", tier='host', name='late_mat_host_customer');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  auto const before = sirius::late_mat::deferrals_installed();
  auto res          = con.Query(query_for(customer, orders, lines));
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO("query error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());
  REQUIRE(rows_of(*res) == expected);

  // Without this the case above passes just as well on a query that deferred
  // nothing, which is the failure mode a correctness test is least able to see.
  REQUIRE(sirius::late_mat::deferrals_installed() > before);

  ::unsetenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS");
  auto unpin = con.Query("CALL unpin_table('late_mat_host_customer');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());
}
