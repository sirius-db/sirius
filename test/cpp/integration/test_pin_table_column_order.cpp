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

// Regression test for the pinned-cache column-ordering bug.
//
// The pinned/cached scan path served columns in column_ids order, while the rest of the
// scan operator (the pushed-down filter's batch-position refs and the projection in
// post_filter_and_project) assumes the disk-decode "materialized" order — output columns
// first, then pure-filter columns. The two orders diverge whenever a column is read only to
// evaluate a pushed-down filter (a "pure-filter" column): on disk it is appended last, but in
// column_ids order it can sit anywhere. On the cached path the divergence mis-binds the
// filter/projection column references, which surfaces as a cuDF "non-matching operand types"
// crash (parquet AST path) or silently wrong results (duckdb-native prefix projection).
//
// Fixture: a table [k (BIGINT), a (DATE), b (DATE)]. The query
//   SELECT count(*) WHERE k >= 0 AND a < b
// makes `k` a pure-filter column (the `k >= 0` predicate is pushed down) while the two DATE
// columns are emitted for the column-vs-column `a < b` filter. So the disk-decode order is
// [a, b, k] but column_ids order is [k, a, b] — the exact C-order != D-order case. Before the
// fix the cached scan binds the date filter onto `k` (BIGINT) and cuDF rejects it.
//
// Pinning the FULL column set [k, a, b] guarantees a cache hit (cache_entry_info is a
// superset), so the query exercises the cached-scan path on both pin tiers.

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kRows = 100'000;

// The query whose plan produces a pure-filter column ahead (in column_ids order) of the
// emitted date columns. `k >= 0` is a pushed-down table filter (k is pure-filter); `a < b`
// is a column-vs-column predicate evaluated as a downstream FILTER over the scan output.
constexpr char const* kWhere = "k >= 0 AND a < b";

// Throwaway, Sirius-disabled DuckDB writes the parquet so the extension callback does not
// build a SiriusContext on it. Columns: k (BIGINT), a (DATE), b (DATE) with a < b for every
// row except the first (range 0 → a == b).
void generate_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r = gen.Query(
      "COPY (SELECT range AS k, "
      "             DATE '1990-01-01' + CAST(range AS INTEGER) AS a, "
      "             DATE '1990-01-01' + CAST(range * 2 AS INTEGER) AS b "
      "      FROM range(" +
      std::to_string(kRows) + ")) TO '" + path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

// CPU (DuckDB, Sirius disabled) baseline for the same query — the ground truth the GPU result
// must match.
std::string cpu_count(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  std::string out;
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto r = con.Query("SELECT count(*) FROM read_parquet('" + path.string() + "') WHERE " +
                       std::string(kWhere) + ";");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    out = r->GetValue(0, 0).ToString();
  }
  unsetenv("SIRIUS_DISABLE");
  return out;
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

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own SiriusContext and
// manages (pauses) the shared envs itself, mirroring the other isolated-context pin tests.
TEST_CASE("pin_table - cached scan serves columns in materialized order (column-order regression)",
          "[pin_table][scan_manager][column_order]")
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

  auto tmp = fs::temp_directory_path() / ("sirius-pin-colorder-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "kab.parquet";
  generate_parquet(parquet_path);

  auto const expected = cpu_count(parquet_path);

  auto yaml_path = tmp / "pin_colorder.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // Force GPU execution so the bug surfaces (crash / wrong result) instead of silently
    // falling back to CPU.
    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    // The cached-scan column ordering is tier-agnostic (both tiers share
    // cached_databatch_provider), so exercise both. 'host' matches the originally reported
    // configuration; 'gpu' is the tier the other pin tests use.
    for (auto const* tier : {"host", "gpu"}) {
      DYNAMIC_SECTION("pin tier = " << tier)
      {
        auto pin = con.Query("CALL pin_table('" + parquet_path.string() + "', tier='" + tier +
                             "', name='colorder', cols=['k', 'a', 'b']);");
        REQUIRE(pin);
        if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
        REQUIRE_FALSE(pin->HasError());

        auto res = con.Query("SELECT count(*) FROM read_parquet('" + parquet_path.string() +
                             "') WHERE " + std::string(kWhere) + ";");
        REQUIRE(res);
        if (res->HasError()) { UNSCOPED_INFO("query error: " << res->GetError()); }
        REQUIRE_FALSE(res->HasError());  // before the fix: cuDF non-matching-operand crash
        REQUIRE(res->GetValue(0, 0).ToString() == expected);  // and the count must be correct

        auto unpin = con.Query("CALL unpin_table('colorder');");
        REQUIRE(unpin);
        REQUIRE_FALSE(unpin->HasError());
      }
    }
  }

  fs::remove_all(tmp, ec);
}
