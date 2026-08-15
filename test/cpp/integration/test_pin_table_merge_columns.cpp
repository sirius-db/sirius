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
       "    enable_compressed_materialization: true\n"
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

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    auto const stats_before = sirius_ctx->get_compressed_materialization_stats();

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
    auto const stats_after = sirius_ctx->get_compressed_materialization_stats();
    REQUIRE(stats_after.pin_columns_narrowed > stats_before.pin_columns_narrowed);

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

    // The chunk-major storage metadata must merge in the same column order as
    // cache_info. All fixture values fit INT32 while the logical columns are
    // BIGINT, so every cached column in every chunk is physically narrow and its
    // recorded carrier is the stored INT32.
    REQUIRE(entry.column_storage.size() == entry.chunk_memory_spaces.size());
    for (auto const& chunk : entry.column_storage) {
      REQUIRE(chunk.size() == names.size());
      for (auto const& column : chunk) {
        REQUIRE(column.narrowed);
        REQUIRE(column.carrier == cudf::data_type{cudf::type_id::INT32});
      }
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

    // A re-pin whose chunk shape disagrees with the existing entry must report the merge
    // mismatch itself. A host pin holds no chunk_memory_spaces, so re-pinning the same name on
    // the GPU tier fails the very first merge guard; the storage-matrix shape check runs after
    // the guards, so the diagnosis names the boundary disagreement and not a matrix shape.
    auto host_pin = con.Query("CALL pin_table('" + parquet_path.string() +
                              "', tier='host', name='tier_swap', cols=['k']);");
    REQUIRE(host_pin);
    if (host_pin->HasError()) { UNSCOPED_INFO("host pin error: " << host_pin->GetError()); }
    REQUIRE_FALSE(host_pin->HasError());

    auto gpu_repin = con.Query("CALL pin_table('" + parquet_path.string() +
                               "', tier='gpu', name='tier_swap', cols=['k']);");
    REQUIRE(gpu_repin);
    REQUIRE(gpu_repin->HasError());
    UNSCOPED_INFO("re-pin error: " << gpu_repin->GetError());
    REQUIRE(gpu_repin->GetError().find("merge mismatch") != std::string::npos);
    REQUIRE(gpu_repin->GetError().find("column_storage") == std::string::npos);

    auto unpin_swap = con.Query("CALL unpin_table('tier_swap');");
    REQUIRE(unpin_swap);
    REQUIRE_FALSE(unpin_swap->HasError());

    auto unpin = con.Query("CALL unpin_table('merge_pin');");
    REQUIRE(unpin);
    REQUIRE_FALSE(unpin->HasError());
  }

  fs::remove_all(tmp, ec);
}

namespace {

/// Copy of a pinned entry's cache_info column names + mvcc presence, taken
/// inside the visitor so no reference escapes the pin-table lock.
struct identity_probe {
  bool found    = false;
  bool has_mvcc = false;
  std::set<std::string> column_names;
  std::string table_name;
};

identity_probe probe_identity(duckdb::SiriusContext& ctx, std::string const& pin_name)
{
  identity_probe out;
  ctx.get_scan_manager().visit_pinned_entries(
    [&](std::string_view name, sirius::scan_manager::pinned_entry const& entry) {
      if (name != pin_name) { return true; }  // keep scanning
      out.found    = true;
      out.has_mvcc = entry.mvcc != nullptr;
      auto names   = entry.cache_info.column_names();
      out.column_names.insert(names.begin(), names.end());
      out.table_name = entry.cache_info.table_name;
      return false;  // stop
    });
  return out;
}

}  // namespace

// The pinned-entry map (and the MVCC metadata attach) is keyed by the bare
// user-supplied pin name. Two live same-named pins that resolve to DIFFERENT
// sources — bare-name duckdb pins from two ATTACHed databases, or two parquet
// file sets pinned under one name — used to collide in the merge/replace path:
// the second pin silently spliced its columns (and its MVCC metadata) into the
// first source's entry, so a query on the first table could serve the second
// table's data. The name stays the user-facing handle (unique per name); a
// same-named pin of a different resolved identity is now rejected loudly.
TEST_CASE("pin_table - same pin name refuses a different resolved identity",
          "[pin_table][scan_manager]")
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

  auto tmp = fs::temp_directory_path() / ("sirius-pin-identity-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto yaml_path = tmp / "pin_identity.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  constexpr std::int64_t kTableRows = 100'000;
  // sum(range(N)) and the per-database b multipliers used below.
  std::int64_t const range_sum = kTableRows * (kTableRows - 1) / 2;

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    auto run_ok = [&](std::string const& sql) {
      auto r = con.Query(sql);
      REQUIRE(r);
      if (r->HasError()) { UNSCOPED_INFO("SQL '" << sql << "' error: " << r->GetError()); }
      REQUIRE_FALSE(r->HasError());
      return r;
    };

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);

    // Two ATTACHed databases with a same-named table of the SAME shape and row
    // count (the exact precondition under which the old merge path proceeded)
    // but DIFFERENT values, so any cross-database serving is observable.
    auto db1 = tmp / "identity_db1.db";
    auto db2 = tmp / "identity_db2.db";
    run_ok("ATTACH '" + db1.string() + "' AS identity_db1;");
    run_ok("ATTACH '" + db2.string() + "' AS identity_db2;");
    run_ok("CREATE TABLE identity_db1.main.t AS SELECT range AS a, range * 2 AS b FROM range(" +
           std::to_string(kTableRows) + ");");
    run_ok("CREATE TABLE identity_db2.main.t AS SELECT range AS a, range * 3 AS b FROM range(" +
           std::to_string(kTableRows) + ");");
    run_ok("CHECKPOINT identity_db1;");
    run_ok("CHECKPOINT identity_db2;");

    // Pin db1's table under the bare name 't' (resolved via the search path).
    run_ok("USE identity_db1;");
    run_ok("CALL pin_table(format='duckdb', name='t', tier='gpu', cols=['a']);");

    // Same bare pin name from db2 resolves to a DIFFERENT table: must be
    // rejected loudly. Pre-fix this silently merged db2's column 'b' into
    // db1's entry and re-bound the entry's MVCC metadata to db2's snapshot.
    run_ok("USE identity_db2;");
    auto collide = con.Query("CALL pin_table(format='duckdb', name='t', tier='gpu', cols=['b']);");
    REQUIRE(collide);
    REQUIRE(collide->HasError());
    UNSCOPED_INFO("colliding pin error: " << collide->GetError());
    REQUIRE(collide->GetError().find("already bound") != std::string::npos);
    REQUIRE(collide->GetError().find("identity_db1") != std::string::npos);

    // db1's entry is untouched: still exactly its own column, still its own
    // MVCC metadata slot.
    auto probe = probe_identity(*sirius_ctx, "t");
    REQUIRE(probe.found);
    REQUIRE(probe.column_names == std::set<std::string>{"a"});
    REQUIRE(probe.has_mvcc);
    REQUIRE(probe.table_name == "t");

    // The supported shape: a distinct pin name for the second source. Both
    // pins are then LIVE SIMULTANEOUSLY and each table serves its own values.
    run_ok("CALL pin_table(format='duckdb', name='identity_db2.main.t', tier='gpu');");

    auto sum_a1 = run_ok("SELECT sum(a) FROM identity_db1.main.t;");
    REQUIRE(sum_a1->GetValue(0, 0).ToString() == std::to_string(range_sum));
    auto sum_b1 = run_ok("SELECT sum(b) FROM identity_db1.main.t;");
    REQUIRE(sum_b1->GetValue(0, 0).ToString() == std::to_string(2 * range_sum));
    auto sum_b2 = run_ok("SELECT sum(b) FROM identity_db2.main.t;");
    REQUIRE(sum_b2->GetValue(0, 0).ToString() == std::to_string(3 * range_sum));

    run_ok("CALL unpin_table('identity_db2.main.t');");
    run_ok("CALL unpin_table('t');");
    run_ok("USE memory;");

    // Parquet variant: one pin name over two different (same-shape) file sets
    // trips the same guard; releasing the name frees it for the other set.
    auto pq1 = tmp / "identity_one.parquet";
    auto pq2 = tmp / "identity_two.parquet";
    run_ok("COPY (SELECT range AS a FROM range(" + std::to_string(kTableRows) + ")) TO '" +
           pq1.string() + "' (FORMAT PARQUET);");
    run_ok("COPY (SELECT range * 5 AS a FROM range(" + std::to_string(kTableRows) + ")) TO '" +
           pq2.string() + "' (FORMAT PARQUET);");
    run_ok("CALL pin_table('" + pq1.string() + "', tier='gpu', name='pq_pin');");
    auto pq_collide =
      con.Query("CALL pin_table('" + pq2.string() + "', tier='gpu', name='pq_pin');");
    REQUIRE(pq_collide);
    REQUIRE(pq_collide->HasError());
    UNSCOPED_INFO("parquet colliding pin error: " << pq_collide->GetError());
    REQUIRE(pq_collide->GetError().find("already bound") != std::string::npos);

    auto pq_sum = run_ok("SELECT sum(a) FROM read_parquet('" + pq2.string() + "');");
    REQUIRE(pq_sum->GetValue(0, 0).ToString() == std::to_string(5 * range_sum));

    run_ok("CALL unpin_table('pq_pin');");
    run_ok("CALL pin_table('" + pq2.string() + "', tier='gpu', name='pq_pin');");
    run_ok("CALL unpin_table('pq_pin');");
  }

  fs::remove_all(tmp, ec);
}
