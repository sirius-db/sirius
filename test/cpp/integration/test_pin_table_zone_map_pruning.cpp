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

// End-to-end zone-map pruning of pinned-table chunks (parquet + duckdb-native,
// HOST + GPU tiers). A clustered table is pinned with a small
// scan_task_batch_size so it spans many chunks; the tests assert the pinned
// entry carries zone maps, that a selective filter provably prunes chunks
// (result equality alone is not evidence pruning ran — handoff §2.3), that
// selective and selects-nothing queries return exact results (the latter via
// the sentinel chunk, guarding the zero-batch pipeline hang), and that
// SET enable_pinned_zone_map_pruning = false preserves results.

#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace {

// 4M rows x 2 int64 columns -> 64 MiB decoded; 8 MiB scan batches -> ~8 chunks.
constexpr std::int64_t kRows      = 4'000'000;
constexpr std::size_t kBatchBytes = 8ull << 20;
// Keeps the top ~1/8 of the (clustered) key range: most chunks are provably empty.
constexpr std::int64_t kSelectiveLo = 3'500'000;
constexpr std::int64_t kExpectCount = kRows - kSelectiveLo;
// sum(v) = sum(2i for i in [lo, kRows)) = (kRows - lo) * (lo + kRows - 1).
constexpr std::int64_t kExpectSum = kExpectCount * (kSelectiveLo + kRows - 1);

void require_ok(duckdb::unique_ptr<duckdb::MaterializedQueryResult> const& r, char const* what)
{
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO(what << " error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
}

void generate_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r =
      gen.Query("COPY (SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRows) +
                ") ORDER BY k) TO '" + path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

void generate_native_db(fs::path const& db_path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(db_path.string().c_str());
    duckdb::Connection gen(gen_db);
    auto r = gen.Query("CREATE TABLE zm_native_t AS SELECT range AS k, range * 2 AS v FROM range(" +
                       std::to_string(kRows) + ");");
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
       "    default_scan_task_varchar_size: 256\n"
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
        return true;
      }
      return false;
    });
  return entry_ptr;
}

std::size_t entry_chunk_count(sirius::scan_manager::pinned_entry const& e)
{
  if (e.tier == cucascade::memory::Tier::HOST) { return e.host_chunks.size(); }
  return e.data_batches_by_column.empty() ? 0 : e.data_batches_by_column.begin()->second.size();
}

// Probe the live entry with the same filter shape the SQL below pushes down
// (k >= kSelectiveLo, BIGINT); a positive pruned count proves the zone maps
// engaged rather than the query merely returning correct results.
std::size_t probe_pruned_count(sirius::scan_manager::pinned_entry const& e)
{
  duckdb::TableFilterSet fs;
  duckdb::unique_ptr<duckdb::TableFilter> f = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::BIGINT(kSelectiveLo));
  fs.filters[0] = std::move(f);
  duckdb::vector<duckdb::ColumnIndex> qcols;
  qcols.emplace_back(duckdb::ColumnIndex(0));
  return sirius::scan_manager::build_cached_scan_plan(e, &fs, &qcols).pruned;
}

void run_pruning_assertions(duckdb::Connection& con,
                            duckdb::SiriusContext& sirius_ctx,
                            std::string const& entry_name,
                            std::string const& relation)
{
  auto const* entry_ptr = find_entry(sirius_ctx, entry_name);
  REQUIRE(entry_ptr != nullptr);
  auto const n_chunks = entry_chunk_count(*entry_ptr);
  UNSCOPED_INFO("pinned entry '" << entry_name << "' has " << n_chunks << " chunks");
  REQUIRE(n_chunks >= 3);
  REQUIRE(entry_ptr->zone_maps.has_stats());
  auto const pruned = probe_pruned_count(*entry_ptr);
  UNSCOPED_INFO("plan probe pruned " << pruned << "/" << n_chunks << " chunks");
  REQUIRE(pruned > 0);

  auto const selective = "SELECT count(*), sum(v) FROM " + relation +
                         " WHERE k >= " + std::to_string(kSelectiveLo) + ";";
  auto sel = con.Query(selective);
  require_ok(sel, "selective query");
  REQUIRE(sel->GetValue(0, 0).ToString() == std::to_string(kExpectCount));
  REQUIRE(sel->GetValue(1, 0).ToString() == std::to_string(kExpectSum));

  auto full = con.Query("SELECT count(*) FROM " + relation + ";");
  require_ok(full, "unfiltered count");
  REQUIRE(full->GetValue(0, 0).ToString() == std::to_string(kRows));

  // Selects-nothing: every chunk is provably empty, so the scan serves only
  // the sentinel chunk; the query must return 0 rows and must not hang.
  auto none_agg = con.Query("SELECT count(*) FROM " + relation +
                            " WHERE k >= " + std::to_string(kRows * 10) + ";");
  require_ok(none_agg, "selects-nothing aggregate");
  REQUIRE(none_agg->GetValue(0, 0).ToString() == "0");

  auto none_rows = con.Query("SELECT k FROM " + relation + " WHERE k < 0;");
  require_ok(none_rows, "selects-nothing projection");
  REQUIRE(none_rows->RowCount() == 0);

  require_ok(con.Query("SET enable_pinned_zone_map_pruning = false;"), "disable pruning");
  auto sel_off = con.Query(selective);
  require_ok(sel_off, "selective query, pruning off");
  REQUIRE(sel_off->GetValue(0, 0).ToString() == std::to_string(kExpectCount));
  REQUIRE(sel_off->GetValue(1, 0).ToString() == std::to_string(kExpectSum));
  require_ok(con.Query("SET enable_pinned_zone_map_pruning = true;"), "re-enable pruning");
}

}  // namespace

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own
// SiriusContext from a small-batch yaml and manages (pauses) the shared envs
// itself, mirroring the other isolated-context pin tests.
TEST_CASE("gpu_execution - pinned zone maps prune clustered parquet chunks end to end",
          "[gpu_execution][parquet][pin_table_zone_map]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-pin-zonemap-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "clustered.parquet";
  generate_parquet(parquet_path);

  auto yaml_path = tmp / "pin_zonemap.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + parquet_path.string() + "');"),
      "create view");

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);

    for (auto const* tier : {"host", "gpu"}) {
      DYNAMIC_SECTION("tier = " << tier)
      {
        auto pin = con.Query("CALL pin_table('" + parquet_path.string() + "', tier='" +
                             std::string(tier) + "', name='t');");
        require_ok(pin, "pin_table");

        run_pruning_assertions(con, *sirius_ctx, "t", "t");

        require_ok(con.Query("CALL unpin_table('t');"), "unpin");
      }
    }

    SECTION("capture disabled at pin time yields a statless entry")
    {
      require_ok(con.Query("SET enable_pinned_zone_map_pruning = false;"), "disable before pin");
      auto pin =
        con.Query("CALL pin_table('" + parquet_path.string() + "', tier='gpu', name='t');");
      require_ok(pin, "pin with capture disabled");
      auto const* entry_ptr = find_entry(*sirius_ctx, "t");
      REQUIRE(entry_ptr != nullptr);
      REQUIRE(entry_chunk_count(*entry_ptr) >= 3);
      REQUIRE_FALSE(entry_ptr->zone_maps.has_stats());

      // Re-enabling the flag does not resurrect stats for an already-pinned
      // entry: queries stay correct but nothing prunes...
      require_ok(con.Query("SET enable_pinned_zone_map_pruning = true;"), "re-enable after pin");
      auto sel = con.Query(
        "SELECT count(*), sum(v) FROM t WHERE k >= " + std::to_string(kSelectiveLo) + ";");
      require_ok(sel, "selective query on statless entry");
      REQUIRE(sel->GetValue(0, 0).ToString() == std::to_string(kExpectCount));
      REQUIRE(sel->GetValue(1, 0).ToString() == std::to_string(kExpectSum));
      REQUIRE(probe_pruned_count(*entry_ptr) == 0);

      // ...until a re-pin: the same-name GPU re-pin takes the equal-row-count
      // merge and the statless entry adopts the fresh capture.
      auto repin =
        con.Query("CALL pin_table('" + parquet_path.string() + "', tier='gpu', name='t');");
      require_ok(repin, "re-pin with capture enabled");
      run_pruning_assertions(con, *sirius_ctx, "t", "t");

      require_ok(con.Query("CALL unpin_table('t');"), "unpin");
    }
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("gpu_execution - pinned zone maps prune clustered duckdb-native chunks end to end",
          "[gpu_execution][duckdb_native][pin_table_zone_map]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-pin-zonemap-nat-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto yaml_path = tmp / "pin_zonemap_native.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  auto db_file = tmp / "zm_native.duckdb";
  generate_native_db(db_file);

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET gpu_execution = false;"), "disable gpu for setup");
    require_ok(con.Query("ATTACH '" + db_file.string() + "' AS zm_native;"), "attach");
    require_ok(con.Query("USE zm_native;"), "use attached db");
    require_ok(con.Query("SET gpu_execution = true;"), "enable gpu");
    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);

    for (auto const* tier : {"host", "gpu"}) {
      DYNAMIC_SECTION("tier = " << tier)
      {
        auto pin = con.Query("CALL pin_table(format='duckdb', name='zm_native_t', tier='" +
                             std::string(tier) + "');");
        require_ok(pin, "pin_table");

        run_pruning_assertions(con, *sirius_ctx, "zm_native_t", "zm_native_t");

        require_ok(con.Query("CALL unpin_table('zm_native_t');"), "unpin");
      }
    }
  }

  fs::remove_all(tmp, ec);
}
