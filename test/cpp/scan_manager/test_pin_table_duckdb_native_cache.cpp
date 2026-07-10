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

// Regression test for the duckdb-native pinned-cache identity under the tree-based
// pipeline build (USE_TREE_BASED_PIPELINE_BUILD, the default).
//
// The tree-path table-info builder (build_duckdb_native_table_info) omitted the
// qualified-name identity fields (catalog_name / schema_name / table_name) that
// cache_entry_info::can_serve_with_columns compares, so every query after
// pin_table(format='duckdb', ...) missed the pinned cache and silently fell back to
// scanning table storage. Result equality cannot catch that — both paths return
// correct rows — so this test asserts the cache HIT itself: pin the table, mutate the
// base table, and require the query to keep returning the PINNED snapshot. A silent
// fall-through to storage (the bug) sees the mutation instead.

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

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
       "    default_scan_task_varchar_size: 256\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

void require_ok(duckdb::unique_ptr<duckdb::MaterializedQueryResult> const& r, char const* what)
{
  REQUIRE(r);
  if (r->HasError()) { UNSCOPED_INFO(what << " error: " << r->GetError()); }
  REQUIRE_FALSE(r->HasError());
}

// Throwaway, Sirius-disabled DuckDB creates the file-backed table: the duckdb-native
// scan path requires a single-file block manager, so an in-memory table cannot back it.
void generate_native_db(fs::path const& db_path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(db_path.string().c_str());
    duckdb::Connection gen(gen_db);
    auto r = gen.Query(
      "CREATE TABLE pin_native_t AS SELECT range AS k, range * 2 AS v FROM range(100000);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

}  // namespace

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own SiriusContext and
// manages (pauses) the shared envs itself, mirroring the other isolated-context pin tests.
TEST_CASE("pin_table - duckdb-native pin serves the cache under the tree pipeline build",
          "[pin_table][scan_manager][duckdb_native]")
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

  auto tmp = fs::temp_directory_path() / ("sirius-pin-ddbnative-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto yaml_path = tmp / "pin_ddbnative.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  auto db_file = tmp / "pin_native.duckdb";
  generate_native_db(db_file);

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // DDL/DML run on the CPU path; GPU interception is re-enabled per SELECT below.
    require_ok(con.Query("SET gpu_execution = false;"), "disable gpu for setup");
    require_ok(con.Query("ATTACH '" + db_file.string() + "' AS pin_native;"), "attach");
    require_ok(con.Query("USE pin_native;"), "use attached db");

    require_ok(con.Query("SET gpu_execution = true;"), "enable gpu");
    // Force GPU execution so a cache miss cannot hide behind a CPU replay.
    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");

    for (auto const* tier : {"host", "gpu"}) {
      DYNAMIC_SECTION("pin tier = " << tier)
      {
        auto pin = con.Query("CALL pin_table(format='duckdb', name='pin_native_t', tier='" +
                             std::string(tier) + "');");
        require_ok(pin, "pin_table");

        // Sanity: the pinned snapshot matches the table as pinned.
        auto pre = con.Query("SELECT max(k) FROM pin_native_t;");
        require_ok(pre, "pre-mutation query");
        REQUIRE(pre->GetValue(0, 0).GetValue<int64_t>() == 99999);

        // Mutate the base table (CPU path). A cached scan must keep serving the pinned
        // snapshot; the bug's silent fall-through to table storage sees the new rows.
        require_ok(con.Query("SET gpu_execution = false;"), "disable gpu for insert");
        require_ok(con.Query("INSERT INTO pin_native_t "
                             "SELECT range + 100000 AS k, 0 AS v FROM range(100000);"),
                   "insert");
        // The duckdb-native scan reads checkpointed storage segments; flush the WAL so
        // the fresh rows are scannable post-unpin (un-checkpointed appends surface as
        // zero-size segments to gpu_decode_table).
        require_ok(con.Query("CHECKPOINT;"), "checkpoint after insert");
        require_ok(con.Query("SET gpu_execution = true;"), "re-enable gpu");

        auto cached = con.Query("SELECT max(k) FROM pin_native_t;");
        require_ok(cached, "post-mutation query");
        INFO(
          "expected pinned-snapshot max(k)=99999 "
          "(fall-through to table storage would be 199999)");
        REQUIRE(cached->GetValue(0, 0).GetValue<int64_t>() == 99999);

        require_ok(con.Query("CALL unpin_table('pin_native_t');"), "unpin_table");

        // Unpinned, the scan falls through to storage and sees the mutation.
        auto post = con.Query("SELECT max(k) FROM pin_native_t;");
        require_ok(post, "post-unpin query");
        REQUIRE(post->GetValue(0, 0).GetValue<int64_t>() == 199999);

        // Restore the fixture for the next tier.
        require_ok(con.Query("SET gpu_execution = false;"), "disable gpu for reset");
        require_ok(con.Query("DELETE FROM pin_native_t WHERE k >= 100000;"), "reset rows");
        require_ok(con.Query("CHECKPOINT;"), "checkpoint after reset");
        require_ok(con.Query("SET gpu_execution = true;"), "re-enable gpu after reset");
      }
    }
  }

  fs::remove_all(tmp, ec);
}
