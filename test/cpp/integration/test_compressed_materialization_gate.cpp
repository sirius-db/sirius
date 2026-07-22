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

// End-to-end coverage of the compressed-materialization residency gate: a
// narrow physical sidecar is installed on a table scan only when the pinned
// cache will serve that scan with columns already narrowed at pin time.
// The three residency states (pinned-narrow, unpinned, pinned-native) plus the
// cols-subset serve-ability boundary are each discriminated through the
// plan-time scan_sidecars_installed counter and the serve-time
// scan_columns_narrowed / scan_columns_restored counters; a second test proves
// zero-benefit pruning stays observable on pinned-backed sidecars.

#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

// 8192 rows x (k BIGINT, v BIGINT, d DECIMAL(18,2)) in 2048-row row groups
// (DuckDB rounds parquet row-group sizes up to vector-size multiples, so 2048
// is the smallest real group); the 16 KiB scan batches make each row group its
// own pin chunk (>= 3, asserted).
constexpr std::int64_t kFactRows = 8192;
// Dimension side for the join-key pruning case: k only, same value recipe.
constexpr std::int64_t kDimRows   = 1200;
constexpr std::size_t kBatchBytes = 16u << 10;

// Value recipes keep every chunk's exact range equal to the file-level footer
// range: k takes the multiples of 7 in [0, 294] (period 43 rows) and v the
// multiples of 11 in [0, 286] (period 27 rows), so both pick INT16 per chunk
// and at plan time; d's unscaled values stay under 29000, picking DECIMAL32
// everywhere. Chunk carrier == plan target for all chunks, so a pinned-narrow
// serve performs zero casts and the restored counter can be asserted flat.
constexpr char const* kPayloadQuery =
  "SELECT k, v, d, d + CAST('0.00' AS DECIMAL(3,2)) AS d_copy "
  "FROM t WHERE k <= 150 ORDER BY k, v;";

void require_ok(duckdb::unique_ptr<duckdb::MaterializedQueryResult> const& r, char const* what)
{
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO(what << " error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
}

void generate_fact_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r = gen.Query(
      "COPY (SELECT (range * 7) % 301 AS k, (range * 11) % 297 AS v, "
      "CAST(((range * 13) % 29000) / 100.0 AS DECIMAL(18,2)) AS d "
      "FROM range(" +
      std::to_string(kFactRows) + ")) TO '" + path.string() +
      "' (FORMAT PARQUET, ROW_GROUP_SIZE 2048);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

void generate_dim_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r =
      gen.Query("COPY (SELECT (range * 7) % 301 AS k FROM range(" + std::to_string(kDimRows) +
                ")) TO '" + path.string() + "' (FORMAT PARQUET, ROW_GROUP_SIZE 400);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_bytes: "
    << (2ull << 30)
    << "\n"
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
       "    scan_task_batch_size: "
    << kBatchBytes
    << "\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

void pause_shared_envs()
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
}

sirius::scan_manager::pinned_entry const* find_entry(duckdb::SiriusContext& sirius_ctx,
                                                     std::string_view entry_name)
{
  sirius::scan_manager::pinned_entry const* entry_ptr = nullptr;
  sirius_ctx.get_scan_manager().visit_pinned_entries(
    [&entry_ptr, entry_name](std::string_view name, auto const& e) {
      if (name == entry_name) {
        entry_ptr = &e;
        return false;  // found: stop iterating
      }
      return true;  // keep iterating
    });
  return entry_ptr;
}

std::size_t entry_chunk_count(sirius::scan_manager::pinned_entry const& e)
{
  if (e.tier == cucascade::memory::Tier::HOST) { return e.host_chunks.size(); }
  return e.data_batches_by_column.empty() ? 0 : e.data_batches_by_column.begin()->second.size();
}

/// Run @p query on the GPU (a failure surfaces loudly: duckdb fallback is
/// disabled by every test below), then on the CPU, and compare the two result
/// sets as sorted stringified rows (the payload's ORDER BY has ties).
void compare_gpu_vs_cpu(duckdb::Connection& con, std::string const& query)
{
  auto gpu_result = con.Query(query);
  require_ok(gpu_result, "gpu query");

  require_ok(con.Query("SET gpu_execution = false;"), "disable gpu");
  auto cpu_result = con.Query(query);
  require_ok(cpu_result, "cpu query");
  require_ok(con.Query("SET gpu_execution = true;"), "re-enable gpu");

  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

  auto collect_rows = [](duckdb::MaterializedQueryResult& result) {
    std::vector<std::vector<std::string>> rows;
    for (duckdb::idx_t r = 0; r < result.RowCount(); r++) {
      std::vector<std::string> row;
      row.reserve(result.ColumnCount());
      for (duckdb::idx_t c = 0; c < result.ColumnCount(); c++) {
        row.push_back(result.GetValue(c, r).ToString());
      }
      rows.push_back(std::move(row));
    }
    std::sort(rows.begin(), rows.end());
    return rows;
  };
  auto gpu_rows = collect_rows(*gpu_result);
  auto cpu_rows = collect_rows(*cpu_result);
  for (std::size_t r = 0; r < gpu_rows.size(); r++) {
    for (std::size_t c = 0; c < gpu_rows[r].size(); c++) {
      if (gpu_rows[r][c] != cpu_rows[r][c]) {
        UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_rows[r][c]
                             << "] CPU=[" << cpu_rows[r][c] << "]");
      }
      REQUIRE(gpu_rows[r][c] == cpu_rows[r][c]);
    }
  }
}

}  // namespace

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own
// SiriusContext from a small-batch yaml and manages (pauses) the shared envs
// itself, mirroring the other isolated-context pin tests.
TEST_CASE("gpu_execution - compressed materialization residency gate states end to end",
          "[gpu_execution][parquet][compressed_materialization_gate]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-gate-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "gate.parquet";
  generate_fact_parquet(parquet_path);

  auto yaml_path = tmp / "compmat_gate.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // A gate regression must fail loudly instead of silently falling back.
    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + parquet_path.string() + "');"),
      "create view");

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);

    SECTION("pinned-narrow serves narrow without serve-time casts")
    {
      for (auto const* tier : {"gpu", "host"}) {
        DYNAMIC_SECTION("tier = " << tier)
        {
          require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
          auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
          auto pin = con.Query("CALL pin_table('" + parquet_path.string() + "', tier='" +
                               std::string(tier) + "', name='t');");
          require_ok(pin, "pin_table");
          auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
          REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

          auto const* entry_ptr = find_entry(*sirius_ctx, "t");
          REQUIRE(entry_ptr != nullptr);
          REQUIRE(entry_chunk_count(*entry_ptr) >= 3);

          // The gate installs a sidecar (the pin serves every column, narrowed
          // in every chunk) and the serve is cast-free: the chunk carriers
          // already equal the plan targets, so nothing narrows or restores
          // during scan normalization.
          auto const before = sirius::test::get_compressed_materialization_stats(con);
          compare_gpu_vs_cpu(con, kPayloadQuery);
          auto const after = sirius::test::get_compressed_materialization_stats(con);
          REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
          REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
          REQUIRE(after.scan_columns_restored == before.scan_columns_restored);

          // Flag-off contrast — the discriminator that narrow data actually
          // flowed through the sidecar above: with the flag off no sidecar
          // exists, so the same narrow resident chunks restore to native
          // during scan normalization.
          require_ok(con.Query("SET enable_compressed_materialization = false;"), "disable flag");
          auto const off_before = sirius::test::get_compressed_materialization_stats(con);
          compare_gpu_vs_cpu(con, kPayloadQuery);
          auto const off_after = sirius::test::get_compressed_materialization_stats(con);
          REQUIRE(off_after.scan_sidecars_installed == off_before.scan_sidecars_installed);
          REQUIRE(off_after.scan_columns_restored > off_before.scan_columns_restored);
          require_ok(con.Query("SET enable_compressed_materialization = true;"), "restore flag");

          require_ok(con.Query("CALL unpin_table('t');"), "unpin");
        }
      }
    }

    SECTION("pinned-native installs no narrow targets")
    {
      // Pinning with the flag off stores native carriers with all-native
      // markers; a later flag-on query must not narrow them at serve time as a
      // recurring per-query cost, so the whole sidecar is dropped.
      require_ok(con.Query("SET enable_compressed_materialization = false;"), "flag off for pin");
      auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
      auto pin =
        con.Query("CALL pin_table('" + parquet_path.string() + "', tier='gpu', name='t');");
      require_ok(pin, "pin_table");
      auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(pin_after.pin_columns_narrowed == pin_before.pin_columns_narrowed);

      require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, kPayloadQuery);
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed == before.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored == before.scan_columns_restored);

      require_ok(con.Query("CALL unpin_table('t');"), "unpin");
    }

    SECTION("unpinned installs nothing")
    {
      // State 2: with no pinned entry the flag-on fresh scan is byte-identical
      // to feature-off — no sidecar, no statistics work, no casts, no restores.
      require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, kPayloadQuery);
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed == before.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored == before.scan_columns_restored);
    }

    SECTION("cols-subset pin gates per serve-ability")
    {
      require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
      auto pin = con.Query("CALL pin_table('" + parquet_path.string() +
                           "', tier='gpu', name='t', cols=['k']);");
      require_ok(pin, "pin k only");

      // The pin cannot serve v: empty serve-projection means no sidecar and a
      // fresh native disk read.
      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, "SELECT k, v FROM t WHERE k <= 150 ORDER BY k, v;");
      auto const mid = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(mid.scan_sidecars_installed == before.scan_sidecars_installed);
      REQUIRE(mid.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(mid.scan_columns_restored == before.scan_columns_restored);

      // A k-only scan is served from the pin, so the gate installs; the serve
      // stays cast-free. (No restored assertion: k is filter+order-only here,
      // so zero-benefit pruning legitimately restores it at the scan.)
      compare_gpu_vs_cpu(con, "SELECT k FROM t WHERE k <= 150 ORDER BY k;");
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed > mid.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == mid.scan_columns_narrowed);

      require_ok(con.Query("CALL unpin_table('t');"), "unpin");
    }
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("gpu_execution - zero-benefit pruning stays discriminating on pinned-backed sidecars",
          "[gpu_execution][parquet][compressed_materialization_gate]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-prune-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto fact_path = tmp / "gate.parquet";
  generate_fact_parquet(fact_path);
  auto dim_path = tmp / "gate_dim.parquet";
  generate_dim_parquet(dim_path);

  auto yaml_path = tmp / "compmat_prune.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + fact_path.string() + "');"),
      "create fact view");
    require_ok(
      con.Query("CREATE VIEW o AS SELECT * FROM read_parquet('" + dim_path.string() + "');"),
      "create dim view");
    require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");

    // At runtime, "installed then pruned to native" and "cleared by
    // propagation" are deliberately indistinguishable — that equivalence is
    // the point of pruning. These sections prove the composite user-visible
    // contract: a narrow-eligible pinned scan whose only uses are zero-benefit
    // emits native at the scan (resident narrow chunks restore during scan
    // normalization), while the residency-gate test's payload shape stays
    // narrow. WHICH pass produces the native sidecar is pinned down by the
    // planner unit tests in
    // test/cpp/planner/test_compressed_schema_propagation.cpp ("pruning
    // removes only zero-benefit restores").
    SECTION("aggregate-only pinned scan ends native at the scan")
    {
      auto pin = con.Query("CALL pin_table('" + fact_path.string() + "', tier='gpu', name='t');");
      require_ok(pin, "pin_table");

      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, "SELECT sum(v), sum(d) FROM t;");
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored > before.scan_columns_restored);

      require_ok(con.Query("CALL unpin_table('t');"), "unpin");
    }

    SECTION("join-key-only pinned scans end native at both scans")
    {
      auto pin_fact =
        con.Query("CALL pin_table('" + fact_path.string() + "', tier='gpu', name='t');");
      require_ok(pin_fact, "pin fact");
      auto pin_dim =
        con.Query("CALL pin_table('" + dim_path.string() + "', tier='gpu', name='o');");
      require_ok(pin_dim, "pin dim");

      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, "SELECT count(*) FROM t, o WHERE t.k = o.k;");
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed >= before.scan_sidecars_installed + 2);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored > before.scan_columns_restored);

      require_ok(con.Query("CALL unpin_table('o');"), "unpin dim");
      require_ok(con.Query("CALL unpin_table('t');"), "unpin fact");
    }
  }

  fs::remove_all(tmp, ec);
}
