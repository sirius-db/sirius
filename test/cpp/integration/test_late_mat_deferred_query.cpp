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
// Shape: three wide strings from a pinned table ride two INNER joins and the
// aggregate that GROUPS on them, materializing one row per group at the far end
// — q10's shape in miniature. Reading them as aggregate inputs instead puts the
// port above a join fan-out, which the plan pass now refuses: the gather would
// run over the joined rows rather than the ones the scan produced.
//
// The pinned table is FOUR parquet files, so that on two GPUs a small batch
// size spreads its chunks over both and the port has to gather some of them
// from the GPU it is not on. The [multi_gpu] case asserts that it did.

#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <late_mat/column_origin.hpp>
#include <late_mat/defer_directive.hpp>
#include <late_mat/materialize.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kCustomerFiles    = 4;
constexpr std::int64_t kCustomersPerFile = 20'000;
constexpr std::int64_t kCustomers        = kCustomerFiles * kCustomersPerFile;
constexpr std::int64_t kOrders           = 60'000;
constexpr std::int64_t kLines            = 120'000;

/// The payload columns are wide enough to clear the policy's value floor
/// (3 x 24 B estimated, less the 8 B rowid) and are touched only by the
/// aggregate at the far end. TWO joins, because the floor is four port
/// crossings and one join's wrap chain is three: a join streams into the
/// group-by that consumes it, so there is no port between those two.
constexpr char const* kQuery =
  "SELECT c.c_custkey, c.c_name, c.c_address, c.c_comment, count(*) AS n "
  "FROM read_parquet('{C}') c "
  "JOIN read_parquet('{O}') o ON c.c_custkey = o.o_custkey "
  "JOIN read_parquet('{L}') l ON o.o_orderkey = l.l_orderkey "
  "GROUP BY c.c_custkey, c.c_name, c.c_address, c.c_comment "
  "ORDER BY c.c_custkey";

std::string query_for(std::string const& customer_glob,
                      fs::path const& orders,
                      fs::path const& lines)
{
  std::string sql = kQuery;
  sql.replace(sql.find("{C}"), 3, customer_glob);
  sql.replace(sql.find("{O}"), 3, orders.string());
  sql.replace(sql.find("{L}"), 3, lines.string());
  return sql;
}

/// The customer table as kCustomerFiles files of consecutive key ranges, so
/// c_custkey is distinct over the whole pin and each file can become its own
/// chunk.
void generate_parquet(fs::path const& customer_dir, fs::path const& orders, fs::path const& lines)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  fs::create_directories(customer_dir);
  for (std::int64_t file = 0; file < kCustomerFiles; ++file) {
    auto const path  = customer_dir / ("customer_" + std::to_string(file) + ".parquet");
    auto const first = file * kCustomersPerFile;
    auto r           = gen.Query(
      "COPY (SELECT range AS c_custkey, "
                "             'Customer#' || lpad(CAST(range AS VARCHAR), 12, '0') AS c_name, "
                "             'address-' || repeat(CAST(range % 97 AS VARCHAR), 6) AS c_address, "
                "             'comment for customer ' || CAST(range AS VARCHAR) AS c_comment "
                "      FROM range(" +
      std::to_string(first) + ", " + std::to_string(first + kCustomersPerFile) + ")) TO " +
      sirius::test::sql_literal(path.string()) + " (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }

  auto r = gen.Query(
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

std::vector<std::string> rows_of(duckdb::QueryResult& r)
{
  std::vector<std::string> rows;
  auto r_rows = r.Collection().GetRows();
  for (duckdb::idx_t i = 0; i < r_rows.size(); ++i) {
    std::string row;
    for (duckdb::idx_t c = 0; c < r.ColumnCount(); ++c) {
      row += r_rows.GetValue(c, i).ToString() + "|";
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

/// The ground truth: the same query on DuckDB with Sirius disabled.
std::vector<std::string> cpu_answer(std::string const& customer_glob,
                                    fs::path const& orders,
                                    fs::path const& lines)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto r = con.Query(query_for(customer_glob, orders, lines));
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
  return rows_of(*r);
}

void write_config(fs::path const& yaml_path, int num_gpus, std::uint64_t scan_task_batch_size)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: "
    << num_gpus
    << "\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.4\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n";
  // Only the sirius scan backend serves more than one GPU.
  if (num_gpus > 1) {
    f << "    scan_manager:\n"
         "      backend: sirius\n";
  }
  f << "    pipeline:\n"
       "      num_threads: 4\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: "
    << scan_task_batch_size
    << "\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

/// GPUs the pin's chunks landed on.
std::set<int> pin_devices(duckdb::Connection& con, std::string const& pin_name)
{
  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);
  std::set<int> devices;
  sirius_ctx->get_scan_manager().visit_pinned_entries(
    [&](std::string_view name, auto const& entry) {
      if (name != pin_name) { return false; }
      for (auto const* space : entry.chunk_memory_spaces) {
        if (space != nullptr) { devices.insert(space->get_device_id()); }
      }
      for (auto const& chunk : entry.device_chunks) {
        if (chunk.memory_space != nullptr) { devices.insert(chunk.memory_space->get_device_id()); }
      }
      return true;
    });
  return devices;
}

/// Pin, run, compare with the CPU, and prove a deferral installed. With more
/// than one GPU, also prove the pin spread and that the port gathered from a
/// GPU it was not on.
void run_deferred_query(int num_gpus, std::uint64_t scan_task_batch_size, char const* tag)
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  sirius::test::scratch_dir scratch{tag};
  auto const& tmp          = scratch.path();
  auto const customer_dir  = tmp / "customer";
  auto const customer_glob = (customer_dir / "*.parquet").string();
  auto const orders        = tmp / "orders.parquet";
  auto const lines         = tmp / "lineitem.parquet";
  generate_parquet(customer_dir, orders, lines);
  auto const expected = cpu_answer(customer_glob, orders, lines);
  REQUIRE_FALSE(expected.empty());

  auto yaml_path = tmp / "late_mat_query.yaml";
  write_config(yaml_path, num_gpus, scan_task_batch_size);

  sirius::test::shared_test_env local_env(yaml_path);
  auto con = local_env.make_connection();
  auto fb  = con.Query("SET enable_duckdb_fallback = false;");
  REQUIRE(fb);
  REQUIRE_FALSE(fb->HasError());

  // c_custkey rides REAL — it is the join key and a group key — so proving it
  // distinct over the pin is what admits the ride past the aggregates. The
  // probe reads this per pin, so setting it here is enough.
  ::setenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS", "c_custkey", 1);

  // GPU tier: a deferral addresses rows by their position in device-resident
  // pinned storage, so nothing installs against a host pin.
  auto pin = con.Query("CALL pin_table(" + sirius::test::sql_literal(customer_glob) +
                       ", tier='gpu', name='late_mat_customer');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());
  if (num_gpus > 1) {
    // Without chunks on both GPUs there is nothing remote for the port to gather.
    auto const devices = pin_devices(con, "late_mat_customer");
    INFO("pin landed on " << devices.size() << " GPU(s)");
    REQUIRE(devices.size() >= 2);
  }

  auto const before        = sirius::late_mat::deferrals_installed();
  auto const remote_before = sirius::late_mat::remote_batches_materialized();
  auto res                 = con.Query(query_for(customer_glob, orders, lines));
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO("query error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());
  REQUIRE(rows_of(*res) == expected);

  // Without this the case above passes just as well on a query that deferred
  // nothing, which is the failure mode a correctness test is least able to see.
  REQUIRE(sirius::late_mat::deferrals_installed() > before);
  if (num_gpus > 1) {
    // And without this it passes just as well when every port task happened to
    // run on the GPU holding the rows it asked for.
    REQUIRE(sirius::late_mat::remote_batches_materialized() > remote_before);
  }

  ::unsetenv("SIRIUS_EXP_LATE_MAT_PIN_UNIQUE_COLS");
  auto unpin = con.Query("CALL unpin_table('late_mat_customer');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());
}

}  // namespace

TEST_CASE("a deferred payload rides a real plan and comes back right", "[late_mat][deferred_query]")
{
  if (!sirius::late_mat::late_mat_enabled()) {
    WARN("SIRIUS_EXP_LATE_MAT unset; skipping the end-to-end deferral case");
    return;
  }
  run_deferred_query(/*num_gpus=*/1, /*scan_task_batch_size=*/100'000'000, "late_mat_query");
}

TEST_CASE("a deferred payload pinned across two GPUs is gathered from both",
          "[late_mat][deferred_query][multi_gpu]")
{
  if (!sirius::late_mat::late_mat_enabled()) {
    WARN("SIRIUS_EXP_LATE_MAT unset; skipping the two-GPU deferral case");
    return;
  }
  if (!sirius::test::has_gpus(2)) { return; }
  // 1 MB batches: each ~1.4 MB customer file becomes its own chunk, and the
  // pin's per-batch round-robin lands them on both GPUs.
  run_deferred_query(/*num_gpus=*/2, /*scan_task_batch_size=*/1'000'000, "late_mat_query_2gpu");
}
