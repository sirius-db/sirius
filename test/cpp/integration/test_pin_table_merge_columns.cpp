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

// Single-GPU regression test for the same-row-count merge path of
// sirius_scan_manager::insert_pinned_entry. Pinning the same table twice with
// overlapping-but-different column subsets ([k,v] then [k,w]) merges the new
// column's data into the existing entry. The merge MUST also extend
// entry.cache_info (column_ids + names) to the UNION of pinned columns, because
// the cache-hit match (cache_entry_info::can_serve_with_columns) keys off
// cache_info.column_ids. Before the fix, cache_info kept only the first pin's
// columns: the merged column 'w' sat in data_batches_by_column unservable, so a
// later `SELECT w` missed the cache and re-read from disk, wasting the merge.
//
// The existing merge guard (test_pin_table_multi_gpu.cpp) asserts the same
// cache_info union, but is gated behind require_two_gpus() and is skipped on
// single-GPU CI — which is exactly how this bug slipped through. This test runs
// on a single GPU so the regression is caught in standard CI.

#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace {

// Small fixture: 100k rows x 3 int64 columns -> ~2.4 MiB decoded, trivially
// GPU-resident. The merge-union assertion does not need a multi-chunk surface.
constexpr std::int64_t kRows = 100'000;

// A throwaway, Sirius-disabled DuckDB writes the parquet so the extension callback does not
// build a SiriusContext on it — the real instance is created later from the yaml.
void generate_parquet(fs::path const& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r = gen.Query("COPY (SELECT range AS k, range * 2 AS v, range * 3 AS w FROM range(" +
                       std::to_string(kRows) + ")) TO '" + path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

// Single-GPU config with a generous GPU budget (the fixture is tiny) and a large
// scan batch so each pin yields a deterministic chunk layout — two pins of the
// same file therefore produce identical chunk_memory_spaces, satisfying the
// merge path's alignment invariant.
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

// NB: no [integration]/[shared_context] tag — those make the Catch2 listener bind a shared
// env, which would fight this test's own local_env. Like the other isolated-context
// integration tests, this TEST_CASE manages (pauses) the shared envs itself.
TEST_CASE("pin_table - same-row-count merge extends cache_info to the column union (single GPU)",
          "[pin_table][merge][scan_manager]")
{
  // This TEST_CASE builds its own SiriusContext, so pause any shared env still holding the
  // extension lock (mirrors test_pin_table_host_streaming.cpp).
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  auto tmp = fs::temp_directory_path() / ("sirius-pin-merge-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "kvw.parquet";
  generate_parquet(parquet_path);

  auto yaml_path = tmp / "pin_merge.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // Force GPU execution so a cache miss / fallback surfaces instead of silently
    // returning a correct-but-uncached result.
    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    // First pin: columns [k, v].
    auto pin1 = con.Query("CALL pin_table('" + parquet_path.string() +
                          "', tier='gpu', name='merge_pin', cols=['k', 'v']);");
    REQUIRE(pin1);
    if (pin1->HasError()) { UNSCOPED_INFO("pin_table 1 error: " << pin1->GetError()); }
    REQUIRE_FALSE(pin1->HasError());

    // Second pin: columns [k, w] — same file, same row count → the merge path
    // installs w and (after the fix) extends cache_info to {k, v, w}.
    auto pin2 = con.Query("CALL pin_table('" + parquet_path.string() +
                          "', tier='gpu', name='merge_pin', cols=['k', 'w']);");
    REQUIRE(pin2);
    if (pin2->HasError()) { UNSCOPED_INFO("pin_table 2 error: " << pin2->GetError()); }
    REQUIRE_FALSE(pin2->HasError());

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    auto const& mgr = sirius_ctx->get_scan_manager();

    bool found                                    = false;
    const sirius::scan_manager::pinned_entry* ptr = nullptr;
    mgr.visit_pinned_entries([&found, &ptr](std::string_view name, const auto& entry) {
      if (name == "merge_pin") {
        found = true;
        ptr   = &entry;
        return true;  // stop iteration
      }
      return false;  // continue
    });
    REQUIRE(found);
    auto const& entry = *ptr;

    // The regression assertion: cache_info must reflect the UNION {k, v, w}, not
    // just the first pin's {k, v}. With the bug, column_names() has size 2.
    auto const& names = entry.cache_info.column_names();
    REQUIRE(names.size() == 3u);
    REQUIRE(entry.cache_info.column_ids.size() == names.size());  // stay aligned 1:1
    std::set<std::string> name_set(names.begin(), names.end());
    REQUIRE(name_set == std::set<std::string>{"k", "v", "w"});

    // Every column listed in cache_info must have backing data in the entry.
    for (auto const& col_name : names) {
      auto it = entry.data_batches_by_column.find(col_name);
      INFO("col_name=" << col_name);
      REQUIRE(it != entry.data_batches_by_column.end());
      REQUIRE_FALSE(it->second.empty());
    }

    // Serving check: a scan requesting the newly merged column 'w' must now be
    // satisfiable from the cache (can_serve_with_columns matches) and return the
    // correct value. sum(w) = 3 * sum(0..N-1) = 3 * N*(N-1)/2.
    std::int64_t const expected_w_sum = static_cast<std::int64_t>(3) * (kRows * (kRows - 1) / 2);
    auto sum_w = con.Query("SELECT sum(w) FROM read_parquet('" + parquet_path.string() + "');");
    REQUIRE(sum_w);
    if (sum_w->HasError()) { UNSCOPED_INFO("sum(w) error: " << sum_w->GetError()); }
    REQUIRE_FALSE(sum_w->HasError());
    REQUIRE(sum_w->GetValue(0, 0).ToString() == std::to_string(expected_w_sum));

    auto unpin = con.Query("CALL unpin_table('merge_pin');");
    REQUIRE(unpin);
    REQUIRE_FALSE(unpin->HasError());
  }

  fs::remove_all(tmp, ec);
}
