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

// Streaming host-tier table pin: a host pin must NOT require the whole table to fit in GPU
// memory. We generate a parquet table whose decoded size (~640 MiB) far exceeds a deliberately
// small GPU budget (256 MiB), with a small scan_task_batch_size so the table spans many
// batches. Pinning it on the host tier must succeed — the pin materializes one batch on GPU,
// copies it to pinned host memory, and frees it before the next, so peak GPU residency stays
// at ~one batch. Before the streaming change, pin_table materialized every batch GPU-resident
// at once and could not fit the budget.
//
// Asserting BOTH (a) the pin succeeds under a budget smaller than the full table and (b) the
// GPU allocation high-water mark stays far below the full-table size makes the test robust
// regardless of whether an over-budget allocation hard-throws or merely balloons memory:
//   - hard cap: pre-streaming code throws while materializing the whole table -> (a) fails.
//   - soft cap: pre-streaming code succeeds but its high-water mark ~= full table -> (b) fails.
// The streaming code satisfies both. Self-contained (no external dataset): the parquet is
// generated with a throwaway DuckDB, mirroring test_table_gpu_cache_warm_mgpu.cpp.

#include "memory/sirius_memory_reservation_manager.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

// 40M rows x 2 int64 columns -> ~640 MiB decoded, comfortably above the GPU budget.
constexpr std::int64_t kRows          = 40'000'000;
constexpr std::size_t kFullTableBytes = static_cast<std::size_t>(kRows) * 2 * sizeof(std::int64_t);
constexpr std::size_t kGpuBudgetBytes = 256ull
                                        << 20;   // 256 MiB: < full table, comfortably > 1 batch
constexpr std::size_t kBatchBytes = 8ull << 20;  // 8 MiB scan batches -> ~80 batches

// A throwaway, Sirius-disabled DuckDB writes the parquet so the extension callback does not
// build a SiriusContext on it — the real (tiny-budget) instance is created later from the yaml.
void generate_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r = gen.Query("COPY (SELECT range AS k, range * 2 AS v FROM range(" +
                       std::to_string(kRows) + ")) TO '" + path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

// Mirror integration.yaml, but cap GPU memory well below the table size and shrink the scan
// batch so the pin must stream many small batches through GPU rather than the whole table.
void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_bytes: "
    << kGpuBudgetBytes
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
       "    duckdb_scan:\n"
       "      num_threads: 2\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period_ms: 10\n"
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

}  // namespace

// NB: no [integration]/[shared_context] tag — those make the Catch2 listener bind a shared
// env, which would fight this test's own tiny-budget local_env. Like the other isolated-context
// integration tests, this TEST_CASE manages (pauses) the shared envs itself.
TEST_CASE("gpu_execution - pin_table host tier streams without fitting the whole table in GPU",
          "[gpu_execution][parquet][pin_table_host_streaming]")
{
  // This TEST_CASE builds its own SiriusContext from a tiny-GPU-budget config, so pause any
  // shared env still holding the extension lock (mirrors test_table_gpu_cache_warm_mgpu.cpp).
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  auto tmp = fs::temp_directory_path() / ("sirius-pin-host-stream-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "wide.parquet";
  generate_parquet(parquet_path);

  auto yaml_path = tmp / "pin_host_stream.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    // shared_test_env's constructor creates the DuckDB + SiriusContext from SIRIUS_CONFIG_FILE,
    // which it sets to our tiny-budget yaml.
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // Force GPU execution so a failure surfaces instead of silently falling back to CPU.
    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    // The view over the generated parquet is the queryable relation; the pinned cache entry is
    // keyed by the same resolved file path, so a scan of `t` is served from the host cache.
    auto v =
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + parquet_path.string() + "');");
    REQUIRE(v);
    REQUIRE_FALSE(v->HasError());

    // Reach the GPU memory space to read its allocation high-water mark.
    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);
    auto& mem_mgr   = sirius_ctx->get_memory_manager();
    auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    REQUIRE_FALSE(gpu_spaces.empty());
    auto* gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
    auto* adaptor =
      gpu0->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(adaptor != nullptr);

    // Pin a ~640 MiB table on host under a 256 MiB GPU budget. Streaming one ~8 MiB batch
    // through GPU at a time, this must succeed; materializing the whole table on GPU (the
    // pre-streaming behavior) could not fit the budget.
    auto pin_result =
      con.Query("CALL pin_table('" + parquet_path.string() + "', tier='host', name='t');");
    REQUIRE(pin_result);
    if (pin_result->HasError()) { UNSCOPED_INFO("pin_table error: " << pin_result->GetError()); }
    REQUIRE_FALSE(pin_result->HasError());

    // The GPU high-water mark during the pin must stay far below the full decoded table size —
    // confirming only ~one batch was GPU-resident at a time, not the whole table.
    std::size_t const peak = adaptor->get_peak_total_allocated_bytes();
    UNSCOPED_INFO("GPU peak during host pin = " << peak << " bytes; full table ~= "
                                                << kFullTableBytes << " bytes");
    REQUIRE(peak < kFullTableBytes / 2);

    // The host-pinned data must serve correct results on a scan hit.
    auto count_result = con.Query("SELECT count(*) FROM t;");
    REQUIRE(count_result);
    REQUIRE_FALSE(count_result->HasError());
    REQUIRE(count_result->GetValue(0, 0).GetValue<std::int64_t>() == kRows);

    // sum(v) = sum(2*i for i in [0, N)) = N*(N-1); stays within int64.
    auto sum_result = con.Query("SELECT sum(v) FROM t;");
    REQUIRE(sum_result);
    REQUIRE_FALSE(sum_result->HasError());
    REQUIRE(sum_result->GetValue(0, 0).ToString() == std::to_string(kRows * (kRows - 1)));

    auto unpin_result = con.Query("CALL unpin_table('t');");
    REQUIRE(unpin_result);
    REQUIRE_FALSE(unpin_result->HasError());
  }

  fs::remove_all(tmp, ec);
}
