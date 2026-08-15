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

// Tests for the Phase 2 cached-table compression infrastructure:
//
//  [compression][plan_register]  — plan_register unit tests (no GPU required)
//  [compression][pin_table]      — end-to-end SQL tests (GPU required):
//    * compressed pin → cached scan result equality vs. uncompressed pin
//    * column-subset projection correctness
//    * compressed footprint < uncompressed logical size
//    * fallback when no plan / chunk below threshold

#include <catch.hpp>
#include <compression/plan_register.hpp>

// standard library
#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// ─── plan_register unit tests (no GPU required) ─────────────────────────────

TEST_CASE("plan_register - global singleton is the same object", "[compression][plan_register]")
{
  auto& a = sirius::compression::plan_register::global();
  auto& b = sirius::compression::plan_register::global();
  REQUIRE(&a == &b);
}

TEST_CASE("plan_register - table plan round-trips correctly", "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  REQUIRE_FALSE(reg.resolve_table_plan("lineitem").has_value());

  const std::string dsl = "identity\n---\nidentity\n---\nidentity";
  reg.set_table_plan("lineitem", dsl);
  auto result = reg.resolve_table_plan("lineitem");
  REQUIRE(result.has_value());
  REQUIRE(result.value() == dsl);

  reg.clear_table_plan("lineitem");
  REQUIRE_FALSE(reg.resolve_table_plan("lineitem").has_value());

  reg.clear_all();
}

TEST_CASE("plan_register - table plans are independent per table", "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  reg.set_table_plan("orders", "lz4");
  reg.set_table_plan("lineitem", "zstd");

  REQUIRE(reg.resolve_table_plan("orders").value() == "lz4");
  REQUIRE(reg.resolve_table_plan("lineitem").value() == "zstd");
  REQUIRE_FALSE(reg.resolve_table_plan("customer").has_value());

  reg.clear_all();
}

TEST_CASE("plan_register - clear_all removes both table and column plans",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  reg.set_table_plan("t", "identity");
  reg.set_plan("t", "col_a", "lz4");

  reg.clear_all();

  REQUIRE_FALSE(reg.resolve_table_plan("t").has_value());

  reg.clear_all();
}

TEST_CASE("plan_register - per-column plans are independent from table plans",
          "[compression][plan_register]")
{
  auto& reg = sirius::compression::plan_register::global();
  reg.clear_all();

  // Setting a per-column plan does not create a table-level entry
  reg.set_plan("t", "col_a", "lz4");
  REQUIRE_FALSE(reg.resolve_table_plan("t").has_value());

  reg.clear_plan("t", "col_a");
  reg.clear_all();
}

TEST_CASE("plan_register - get_or_load_table_plan loads exactly once per key",
          "[compression][plan_register][concurrency]")
{
  // Register E5: the pin path's resolve-miss-load-set-resolve was a
  // check-then-act across three critical sections; two concurrent callers
  // could both miss and overwrite each other. get_or_load_table_plan runs the
  // whole lookup-miss-then-populate under ONE exclusive critical section:
  // exactly one loader runs per key, and every caller returns the value that
  // actually ended up registered.
  auto& reg             = sirius::compression::plan_register::global();
  const std::string key = "e5_get_or_load_once";
  struct key_cleanup {
    std::string k;
    ~key_cleanup() { sirius::compression::plan_register::global().clear_table_plan(k); }
  } cleanup{key};
  reg.clear_table_plan(key);

  constexpr int n_threads = 8;
  std::atomic<int> loads{0};
  std::vector<std::string> results(n_threads);
  std::vector<std::thread> threads;
  threads.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      auto value = reg.get_or_load_table_plan(key, [&]() -> std::optional<std::string> {
        loads.fetch_add(1);
        // Widen the pre-fix both-miss window; under the single critical
        // section this only delays the losers, it cannot double-load.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return "loaded-by-thread-" + std::to_string(t);
      });
      // Catch2 assertions are not thread-safe: collect, assert on the main
      // thread below.
      results[static_cast<std::size_t>(t)] = value.value_or("");
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  CHECK(loads.load() == 1);
  for (int t = 0; t < n_threads; ++t) {
    CHECK(results[static_cast<std::size_t>(t)] == results[0]);
  }
  CHECK(results[0].rfind("loaded-by-thread-", 0) == 0);
  REQUIRE(reg.resolve_table_plan(key) == results[0]);
}

TEST_CASE("plan_register - get_or_load_table_plan stores nothing on a loader miss",
          "[compression][plan_register]")
{
  auto& reg             = sirius::compression::plan_register::global();
  const std::string key = "e5_get_or_load_miss";
  struct key_cleanup {
    std::string k;
    ~key_cleanup() { sirius::compression::plan_register::global().clear_table_plan(k); }
  } cleanup{key};
  reg.clear_table_plan(key);

  int loads = 0;
  // A nullopt loader result registers nothing (matching the old semantics:
  // no plan file found => no entry, a later pin rescans the directory).
  auto miss = reg.get_or_load_table_plan(key, [&]() -> std::optional<std::string> {
    ++loads;
    return std::nullopt;
  });
  CHECK_FALSE(miss.has_value());
  CHECK_FALSE(reg.resolve_table_plan(key).has_value());
  CHECK(loads == 1);

  // The next call therefore loads again; a real value now sticks.
  auto hit = reg.get_or_load_table_plan(key, [&]() -> std::optional<std::string> {
    ++loads;
    return "the real plan";
  });
  REQUIRE(hit.has_value());
  CHECK(*hit == "the real plan");
  CHECK(loads == 2);

  // And a third call is a pure hit: the loader must not run.
  auto cached = reg.get_or_load_table_plan(key, [&]() -> std::optional<std::string> {
    FAIL("loader ran on a registered key");
    return std::nullopt;
  });
  REQUIRE(cached.has_value());
  CHECK(*cached == "the real plan");
  CHECK(loads == 2);
}

TEST_CASE("select_plan_blocks - picks blocks by full-table index in pinned order",
          "[compression][plan_register]")
{
  using sirius::compression::select_plan_blocks;
  // Four full-table columns, one block each. select_plan_blocks re-joins the
  // selected blocks with "\n---\n" (blocks are trimmed on split).
  const std::string full =
    "input -> lz4\n---\ninput -> snappy\n---\ninput -> deflate\n---\ninput "
    "-> bitcomp";

  SECTION("subset preserves the requested (pinned) order")
  {
    // Pin columns [2, 0] (full-table indices), in that pinned order.
    auto s = select_plan_blocks(full, {2, 0});
    REQUIRE(s.has_value());
    REQUIRE(*s == "input -> deflate\n---\ninput -> lz4");
  }

  SECTION("identity over all columns returns every block in order")
  {
    auto s = select_plan_blocks(full, {0, 1, 2, 3});
    REQUIRE(s.has_value());
    REQUIRE(*s ==
            "input -> lz4\n---\ninput -> snappy\n---\ninput -> deflate\n---\ninput -> bitcomp");
  }

  SECTION("out-of-range index yields nullopt (plan does not cover the column)")
  {
    REQUIRE_FALSE(select_plan_blocks(full, {4}).has_value());
    REQUIRE_FALSE(select_plan_blocks(full, {0, 9}).has_value());
  }
}

// ─── End-to-end SQL tests require a GPU (Simpatico JIT) ─────────────────────
//
// These tests gate on GPU availability and use an isolated DuckDB with a
// minimal single-GPU Sirius config.  They:
//   1. Create a small in-memory table.
//   2. Pin it with tier='host' and compression enabled.
//   3. SELECT from the cached table and compare against the raw table.
//   4. Verify compressed_bytes < uncompressed_bytes in the pinned_entry.
//   5. Verify fallback (no plan / small chunk) does not error.

#include "operator/mgpu_test_utils.hpp"
#include "sirius_context.hpp"

#include <cuda_runtime.h>

#include <compression/compressed_representation.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <unistd.h>
#include <utils/pinned_entry_census.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace {

// Build a minimal single-GPU Sirius YAML with compression enabled. The scan
// batch size is parameterized so fixtures that need one pin chunk per parquet
// row group can shrink it.
void write_compression_yaml(const fs::path& yaml_path, std::size_t scan_batch_bytes = 100000000)
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
       "      capacity_bytes: 2000000000\n"
       "      initial_number_pools: 4\n"
       "      pool_size: 128\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 2\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: "
    << scan_batch_bytes
    << "\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

bool has_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  return count >= 1;
}

fs::path make_comp_tmp(const std::string& tag)
{
  return fs::temp_directory_path() / ("sirius-comp-test-" + tag + "-" + std::to_string(::getpid()));
}

// Skip guard shared by every GPU test: emits the standard warning and returns
// true when no GPU is present, so callers can write `if (no_gpu()) return;`.
bool no_gpu()
{
  if (has_gpu()) { return false; }
  WARN("Compression test requires a GPU — skipping");
  return true;
}

// A fresh, empty tmp dir tagged @p tag with a single-GPU compression yaml
// written into it. The caller owns cleanup (fs::remove_all(dir)).
struct comp_env_paths {
  fs::path dir;
  fs::path yaml;
};

comp_env_paths make_comp_env(const std::string& tag, std::size_t scan_batch_bytes = 100000000)
{
  auto dir = make_comp_tmp(tag);
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto yaml = dir / "comp.yaml";
  write_compression_yaml(yaml, scan_batch_bytes);
  return {dir, yaml};
}

// Assert that a completed query @p res succeeded, attaching the DuckDB error to
// the failing assertion on error.
void require_ok(const duckdb::unique_ptr<duckdb::MaterializedQueryResult>& res,
                const std::string& what)
{
  REQUIRE(res);
  if (res->HasError()) { UNSCOPED_INFO(what << " error: " << res->GetError()); }
  REQUIRE_FALSE(res->HasError());
}

// Run @p sql on @p con, assert it succeeded, and return the result for
// inspection.
duckdb::unique_ptr<duckdb::MaterializedQueryResult> run_ok(duckdb::Connection& con,
                                                           const std::string& sql,
                                                           const std::string& what)
{
  auto res = con.Query(sql);
  require_ok(res, what);
  return res;
}

}  // namespace

namespace {

// Write a plan file for table @p table_name into @p plan_dir containing @p dsl.
void write_plan_file(const fs::path& plan_dir,
                     const std::string& table_name,
                     const std::string& dsl)
{
  fs::create_directories(plan_dir);
  std::ofstream f(plan_dir / (table_name + ".txt"));
  f << dsl;
}

// One bitpack block per pinned column, the plan shape the stacking fixtures share: bitpack
// genuinely shrinks small-range columns, so chunks pass the max_compressed_fraction gate and pin in
// the compressed form (a plan whose output stays input-sized would fall back to uncompressed
// storage instead).
std::string bitpack_plan(std::size_t columns)
{
  const std::string block = "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
  std::string dsl;
  for (std::size_t i = 0; i < columns; ++i) {
    if (i > 0) { dsl += "---\n"; }
    dsl += block;
  }
  return dsl;
}

}  // anonymous namespace

TEST_CASE("pin_table compression - result equality vs uncompressed pin",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("eq");

  // SELECT range AS k, range * 3 AS v FROM range(10000):
  //   SUM(k) = 9999*10000/2 = 49995000
  //   SUM(v) = 3 * SUM(k)   = 149985000
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS k, range * 3 AS v FROM range(10000)", /*num_files=*/1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  // Write a plan file for the table that matches the parquet glob name
  auto plan_dir = tmp / "plans";
  write_plan_file(
    plan_dir, "t_comp", "input -> delta -> differences\n---\ninput -> delta -> differences\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin_comp = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_comp');");
  require_ok(pin_comp, "pin");

  // Query via gpu_execution so Sirius serves the result from the compressed cache.
  const std::string select_sql = "SELECT SUM(k), SUM(v) FROM read_parquet('" + glob + "')";
  auto sum_comp                = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(sum_comp, "select");
  REQUIRE(sum_comp->RowCount() == 1);
  REQUIRE(sum_comp->GetValue(0, 0) == duckdb::Value::BIGINT(49995000LL));
  REQUIRE(sum_comp->GetValue(1, 0) == duckdb::Value::BIGINT(149985000LL));

  run_ok(con, "CALL unpin_table('t_comp');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - device tier result equality vs uncompressed pin",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("eqdev");

  // Same surface as the host case: SUM(k)=49995000, SUM(v)=149985000.
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS k, range * 3 AS v FROM range(10000)", /*num_files=*/1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  write_plan_file(
    plan_dir, "t_comp_dev", "input -> delta -> differences\n---\ninput -> delta -> differences\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  // tier='gpu' → compressed payload kept in device memory, decompressed on query.
  auto pin_comp = con.Query("CALL pin_table('" + glob + "', tier='gpu', name='t_comp_dev');");
  require_ok(pin_comp, "pin");

  const std::string select_sql = "SELECT SUM(k), SUM(v) FROM read_parquet('" + glob + "')";
  auto sum_comp                = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(sum_comp, "select");
  REQUIRE(sum_comp->RowCount() == 1);
  REQUIRE(sum_comp->GetValue(0, 0) == duckdb::Value::BIGINT(49995000LL));
  REQUIRE(sum_comp->GetValue(1, 0) == duckdb::Value::BIGINT(149985000LL));

  run_ok(con, "CALL unpin_table('t_comp_dev');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - device tier column-subset projection correctness",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("projdev");

  // SUM(b) = 2*(0+1+...+4999) = 24995000.
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS a, range * 2 AS b, range * 3 AS c FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir,
                  "t_proj_dev",
                  "input -> delta -> differences\n---\ninput -> delta -> differences\n---\ninput "
                  "-> delta -> differences\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='gpu', name='t_proj_dev');");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT SUM(b) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(24995000LL));

  run_ok(con, "CALL unpin_table('t_proj_dev');", "unpin");

  fs::remove_all(tmp);
}

// ─── Decode-time predicate pushdown ──────────────────────────────────────────
//
// The TPC-H q19 shape: a low-cardinality string column that is only ever
// equality-tested and never projected. On a GPU-tier dictionary pin the scan
// hands the predicate to decompression, which answers it against the four-entry
// key set and returns a BOOL8 mask instead of gathering the decoded chars. The
// observable contract is that the answer does not change — including for the
// shapes that must decline the substitution.

TEST_CASE("pin_table compression - dictionary predicate pushdown preserves results",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("dictpred");

  // Four distinct labels over 5000 rows, mirroring l_shipinstruct. Rows with
  // s = 'DELIVER IN PERSON' are range % 4 == 0 (1250 rows) and carry
  // v = range * 2, so SUM(v) = 2 * 4 * (0 + 1 + ... + 1249) = 6245000.
  sirius::test::mgpu::generate_parquet_surface(
    tmp,
    "SELECT CASE range % 4 WHEN 0 THEN 'DELIVER IN PERSON' WHEN 1 THEN 'COLLECT COD' "
    "WHEN 2 THEN 'NONE' ELSE 'TAKE BACK RETURN' END AS s, range * 2 AS v FROM range(5000)",
    1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  // The exact l_shipinstruct plan from the SF1000 lineitem plan file — depth 3,
  // with both keys_offsets and indices bitpacked — so the interception is
  // exercised against the shape production actually decodes, where the indices
  // reach the dictionary rep from a JIT bitpack inverse rather than raw storage.
  // v stays raw so it can be summed.
  write_plan_file(plan_dir,
                  "t_dictpred",
                  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
                  "dictionary.keys_offsets -> bitpack -> chunk_min, chunk_count, chunk_bits, "
                  "packed\n"
                  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
                  "---\ninput -> identity\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='gpu', name='t_dictpred');");
  require_ok(pin, "pin dictpred");

  // s is filter-only — never in the select list — so the scan may replace it
  // with the decode-time mask. This is the case the optimization exists for.
  auto eq = con.Query("CALL gpu_execution(\"SELECT SUM(v) FROM read_parquet('" + glob +
                      "') WHERE s = 'DELIVER IN PERSON'\");");
  require_ok(eq, "equality pushdown");
  REQUIRE(eq->RowCount() == 1);
  REQUIRE(eq->GetValue(0, 0) == duckdb::Value::BIGINT(6245000LL));

  // IN over two keys exercises the multi-value LUT: range % 4 in {0, 2} →
  // SUM(v) = 2 * ((0+4+...+4996) + (2+6+...+4998)) = 2 * (3122500 + 3125000).
  auto in_res = con.Query("CALL gpu_execution(\"SELECT SUM(v) FROM read_parquet('" + glob +
                          "') WHERE s IN ('DELIVER IN PERSON', 'NONE')\");");
  require_ok(in_res, "in pushdown");
  REQUIRE(in_res->RowCount() == 1);
  REQUIRE(in_res->GetValue(0, 0) == duckdb::Value::BIGINT(12495000LL));

  // A *projected* dictionary column must not be substituted — the mask would
  // replace the very values the query selects.
  auto proj = con.Query("CALL gpu_execution(\"SELECT COUNT(s), MIN(s) FROM read_parquet('" + glob +
                        "') WHERE s = 'DELIVER IN PERSON'\");");
  require_ok(proj, "projected dictionary column");
  REQUIRE(proj->RowCount() == 1);
  REQUIRE(proj->GetValue(0, 0) == duckdb::Value::BIGINT(1250LL));
  REQUIRE(proj->GetValue(1, 0).ToString() == "DELIVER IN PERSON");

  // A non-equality filter is not a candidate and must still evaluate normally.
  auto ne = con.Query("CALL gpu_execution(\"SELECT COUNT(*) FROM read_parquet('" + glob +
                      "') WHERE s <> 'DELIVER IN PERSON'\");");
  require_ok(ne, "inequality falls back");
  REQUIRE(ne->RowCount() == 1);
  REQUIRE(ne->GetValue(0, 0) == duckdb::Value::BIGINT(3750LL));

  run_ok(con, "CALL unpin_table('t_dictpred');", "unpin dictpred");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - predicate pushdown mixes substituted and decoded columns",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("dictpredmix");

  // The literal q19 shape: two filter-only string columns, only one of which a
  // dictionary can answer. l_shipinstruct-alike `s` is dictionary-compressed and
  // substitutes; l_shipmode-alike `m` is str_split and must still be decoded and
  // compared. Rows matching both are range % 12 == 0 (417 rows), v = range * 2 →
  // SUM(v) = 2081664. This is the case that would break if one ineligible
  // candidate disabled the pushdown for its eligible sibling.
  sirius::test::mgpu::generate_parquet_surface(
    tmp,
    "SELECT CASE range % 4 WHEN 0 THEN 'DELIVER IN PERSON' WHEN 1 THEN 'COLLECT COD' "
    "WHEN 2 THEN 'NONE' ELSE 'TAKE BACK RETURN' END AS s, "
    "CASE range % 3 WHEN 0 THEN 'AIR' WHEN 1 THEN 'RAIL' ELSE 'SHIP' END AS m, "
    "range * 2 AS v FROM range(5000)",
    1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir,
                  "t_dictpred_mix",
                  "input -> dictionary\n---\ninput -> str_split -> offsets, chars\n---\ninput -> "
                  "identity\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='gpu', name='t_dictpred_mix');");
  require_ok(pin, "pin mixed");

  auto res = con.Query("CALL gpu_execution(\"SELECT SUM(v) FROM read_parquet('" + glob +
                       "') WHERE s = 'DELIVER IN PERSON' AND m = 'AIR'\");");
  require_ok(res, "mixed substitution");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(2081664LL));

  run_ok(con, "CALL unpin_table('t_dictpred_mix');", "unpin mixed");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - predicate pushdown declines a non-dictionary plan",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("dictpredfb");

  // str_split cannot answer a predicate from its compressed form, so the scan
  // must decode and compare as before. Same 1250-matching-row surface:
  // SUM(v) = 6245000.
  sirius::test::mgpu::generate_parquet_surface(
    tmp,
    "SELECT CASE range % 4 WHEN 0 THEN 'AIR' WHEN 1 THEN 'RAIL' WHEN 2 THEN 'SHIP' "
    "ELSE 'TRUCK' END AS s, range * 2 AS v FROM range(5000)",
    1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  write_plan_file(
    plan_dir, "t_dictpred_fb", "input -> str_split -> offsets, chars\n---\ninput -> identity\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='gpu', name='t_dictpred_fb');");
  require_ok(pin, "pin fallback");

  auto res = con.Query("CALL gpu_execution(\"SELECT SUM(v) FROM read_parquet('" + glob +
                       "') WHERE s = 'AIR'\");");
  require_ok(res, "str_split fallback");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(6245000LL));

  run_ok(con, "CALL unpin_table('t_dictpred_fb');", "unpin fallback");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - column-subset projection correctness",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("proj");

  // SELECT range AS a, range * 2 AS b, range * 3 AS c FROM range(5000):
  //   SUM(b) = 2 * (0+1+...+4999) = 2 * 4999*5000/2 = 24995000
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS a, range * 2 AS b, range * 3 AS c FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir,
                  "t_proj",
                  "input -> delta -> differences\n---\ninput -> delta -> differences\n---\ninput "
                  "-> delta -> differences\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_proj');");
  require_ok(pin, "pin");

  // Project only column b; Sirius must decompress only that column.
  // SUM(b) = 2*(0+1+...+4999) = 2*4999*5000/2 = 24995000
  const std::string select_sql = "SELECT SUM(b) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(24995000LL));

  run_ok(con, "CALL unpin_table('t_proj');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - pinned column subset selects matching plan blocks",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("subset");

  // Four full-table columns; the plan file has one block per column (schema order).
  //   SUM(a) = 0+1+...+4999               = 12497500
  //   SUM(c) = 3*(0+1+...+4999)           = 37492500
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS a, range * 2 AS b, range * 3 AS c, range * 4 AS d FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  // Distinct op per column so a wrong block↔column mapping would compress with the
  // wrong plan; all ops are valid for INT64 so a correct mapping round-trips cleanly.
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir,
                  "t_sub",
                  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n---\n"
                  "input -> rle -> runs, values\n---\n"
                  "input -> delta -> differences\n---\n"
                  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  // Pin a reordered subset [c, a] (full-table indices 2, 0) — exercises per-column
  // block selection (blocks 2 and 0, in that order).
  auto pin =
    con.Query("CALL pin_table('" + glob + "', tier='host', name='t_sub', cols=['c','a']);");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT SUM(c), SUM(a) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(37492500LL));
  REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(12497500LL));

  run_ok(con, "CALL unpin_table('t_sub');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - decimal columns round-trip with scale restored",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("decimal");

  // A DECIMAL(15,2) column is physically INT64 (unscaled) + scale; the integer
  // codecs compress that storage losslessly and the scale is restored on decode.
  //   SUM(k) = 0+1+...+4999                       = 12497500
  //   d      = (range % 100) * 0.25 (2 d.p.)      -> SUM(d) = 50*4950*0.25 = 61875.00
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS k, ((range % 100) * 0.25)::DECIMAL(15,2) AS d FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  // Integer codegen ops — valid for INT64 and, via storage reinterpret, for DECIMAL64.
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir,
                  "t_dec",
                  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n---\n"
                  "input -> delta -> differences\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_dec');");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT SUM(k), SUM(d) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(12497500LL));
  // The decimal sum must come back with the correct scale (61875.00, not 6187500).
  REQUIRE(res->GetValue(1, 0).ToString() == "61875.00");

  run_ok(con, "CALL unpin_table('t_dec');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - fallback when no plan file for table",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("noplan");

  // Plan dir exists but has no file matching 't_noplan' — must pin uncompressed.
  // SUM(k) = 0+1+...+999 = 999*1000/2 = 499500
  auto plan_dir = tmp / "plans";
  fs::create_directories(plan_dir);

  sirius::test::mgpu::generate_parquet_surface(tmp, "SELECT range AS k FROM range(1000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_noplan');");
  require_ok(pin, "pin");

  // Must still scan correctly via fallback raw host rep
  const std::string select_sql = "SELECT SUM(k) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(499500LL));

  run_ok(con, "CALL unpin_table('t_noplan');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - fallback when batch is below min_batch_size_bytes threshold",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("threshold");

  // Plan file exists but threshold is far above the tiny chunk size — forces
  // fallback to uncompressed host rep.
  // SUM(k) = 0+1+...+99 = 99*100/2 = 4950
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_threshold", "input -> delta -> differences\n");

  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS k FROM range(100)", 1);  // tiny chunk (~800 B uncompressed)

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");
  // Threshold (1 GiB) far above the tiny chunk (~800 B) — forces fallback
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 1000000000;", "set min_batch");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_threshold');");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT SUM(k) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(4950LL));

  run_ok(con, "CALL unpin_table('t_threshold');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - fallback when compression saves too little",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("ratio");

  // A well-compressing plan on well-compressing data, but an impossibly strict
  // max-compressed-fraction (1%) forces the compressed form to be discarded and
  // the batch pinned uncompressed. Result must still be correct.
  // SUM(k) = 0+1+...+999 = 499500
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_ratio", "input -> delta -> differences\n");

  sirius::test::mgpu::generate_parquet_surface(tmp, "SELECT range AS k FROM range(1000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");
  // Require a 99% saving — no realistic plan meets this, so the compressed form
  // is discarded and the batch is pinned uncompressed.
  run_ok(con, "SET pin_table_compression_max_compressed_fraction = 0.01;", "set max_fraction");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_ratio');");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT SUM(k) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(499500LL));

  run_ok(con, "CALL unpin_table('t_ratio');", "unpin");

  fs::remove_all(tmp);
}

// ─── Per-operator sweeps ──────────────────────────────────────────────────────
//
// Each group pins the same parquet surface repeatedly, once per operator, and
// verifies aggregate correctness after decompression.  Codegen (fused JIT) ops
// require explicit output-channel declarations in the plan DSL so the fused
// region materialises transformed streams into recoverable leaf buffers.
// Terminal nvCOMP ops need no output declaration.
//
// Operators intentionally excluded from these sweeps:
//   identity       — excluded from all_compressor_names() (explorable=false)
//   nvcomp_cascaded— excluded from all_compressor_names() (explorable=false)
//   bitextract_f32/f64 — require a float column and a bitfield spec; covered
//                    by test_operator_sweep in the simpatico unit suite
//   str_split      — structural op requiring channel routing; see the
//                    dedicated str_split cascade test below

namespace {

// Every operator an integer plan block can name, with the plan DSL that exercises it. Shared by
// the native-width and narrowed-carrier sweeps so the two cannot drift apart.
struct op_case {
  const char* tag;
  const char* plan;
};

constexpr op_case kOps[] = {
  {"delta", "input -> delta -> differences"},
  {"rle", "input -> rle -> runs, values"},
  {"bitpack", "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed"},
  {"for", "input -> for -> deltas, references"},
  {"zigzag", "input -> zigzag -> zigzag"},
  {"lz4", "input -> lz4"},
  {"snappy", "input -> snappy"},
  {"deflate", "input -> deflate"},
  {"ans", "input -> ans"},
  {"bitcomp", "input -> bitcomp"},
};

}  // namespace

TEST_CASE("pin_table compression - single-op sweep over all INT64 operators",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("sweep_i64");

  // SUM(k) = 0+1+...+4999 = 12497500
  sirius::test::mgpu::generate_parquet_surface(tmp, "SELECT range AS k FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  const std::string select_sql = "SELECT SUM(k) FROM read_parquet('" + glob + "')";

  for (auto const& tc : kOps) {
    CAPTURE(tc.tag);
    std::string tname = std::string("t_sw_") + tc.tag;
    write_plan_file(plan_dir, tname, tc.plan);

    auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='" + tname + "');");
    require_ok(pin, std::string("pin:") + tc.tag);

    auto res = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
    require_ok(res, std::string("select:") + tc.tag);
    REQUIRE(res->RowCount() == 1);
    REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(12497500LL));

    run_ok(con, "CALL unpin_table('" + tname + "');", std::string("unpin:") + tc.tag);
  }

  fs::remove_all(tmp);
}

// The narrowed-carrier counterpart of the INT64 sweep: the same operators against a column the pin
// narrowed, once per carrier width narrowing can select. Where the INT64 sweep asserts only query
// correctness, this one censuses the entry per operator — a pin that silently failed to compress
// still answers SUM(k) correctly, so correctness alone would not notice an operator losing support
// for a carrier width. CAPTURE(tag) names the failing operator and SECTION names the width.
TEST_CASE("pin_table compression - single-op sweep over narrowed carriers",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  // Range expression, the carrier the chooser selects for it from a BIGINT column, and SUM(k) over
  // the 5000 generated rows.
  struct carrier_case {
    const char* label;
    const char* select;
    cudf::type_id carrier;
    std::int64_t sum;
  };
  // Each fixture cycles four values, ten rows each, so the column carries runs every operator in
  // kOps can act on: an all-distinct column expands under rle and would fail the census for a
  // reason unrelated to the carrier. Two of the four values are excursions towards the carrier's
  // limits, which is what makes the sweep sensitive rather than merely green:
  //
  //   * The excursions push zigzag's output past the carrier's signed maximum (int8 reaches 146
  //     against 127, int16 reaches 33097 against 32767), so the stored element goes negative and
  //     the decoder has to narrow it to the element's exact width before the inverse shift. A
  //     decoder that widens a signed element straight to 32 bits sign-extends garbage and returns
  //     wrong values with no error, which correctness assertions alone would catch only here.
  //   * Their span needs the full element width (146 spans 8 bits, 32998 spans 16), so bitpack
  //     takes its flush-width path where elements sit on word boundaries instead of the general
  //     scatter. The int32 case cannot reach that path without a span over 2^31, which would put
  //     the steps below out of range, so it exercises the general path.
  //   * The step between adjacent values stays inside the carrier (81, 16597, 5000097), so delta
  //     re-emits differences the element type can hold.
  auto const cases = std::vector<carrier_case>{
    {"int8",
     "SELECT CASE (range // 10) % 4 WHEN 1 THEN 64 + (range // 10) % 10 "
     "WHEN 3 THEN -64 - (range // 10) % 10 ELSE (range // 10) % 10 END AS k FROM range(5000)",
     cudf::type_id::INT8,
     10000},
    {"int16",
     "SELECT CASE (range // 10) % 4 WHEN 1 THEN 16400 + (range // 10) % 50 "
     "WHEN 3 THEN -16500 - (range // 10) % 50 ELSE (range // 10) % 50 END AS k FROM range(5000)",
     cudf::type_id::INT16,
     -65000},
    {"int32",
     "SELECT CASE (range // 10) % 4 WHEN 1 THEN 5000000 + (range // 10) % 50 "
     "WHEN 3 THEN -5000000 - (range // 10) % 50 ELSE (range // 10) % 50 END AS k FROM range(5000)",
     cudf::type_id::INT32,
     60000},
  };

  for (auto const& cc : cases) {
    DYNAMIC_SECTION(cc.label)
    {
      auto [tmp, yaml_path] = make_comp_env(std::string("sweep_narrow_") + cc.label);
      sirius::test::mgpu::generate_parquet_surface(tmp, cc.select, 1);

      sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
      auto con  = env.make_connection();
      auto glob = sirius::test::mgpu::parquet_glob(tmp);

      run_ok(con, "SET pin_table_compression = true;", "set compression");
      run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
      // delta, zigzag and for re-emit one element per input element, so they are size-neutral or
      // slightly expanding by construction and the default fraction gate would store their chunks
      // uncompressed. Accept any compressed size: this sweep measures encodability, not ratio.
      run_ok(con, "SET pin_table_compression_max_compressed_fraction = 1.5;", "set fraction");
      run_ok(con, "SET enable_compressed_materialization = true;", "set narrowing");

      auto plan_dir = tmp / "plans";
      run_ok(con,
             "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';",
             "set plan_dir");

      const std::string select_sql = "SELECT SUM(k) FROM read_parquet('" + glob + "')";

      for (auto const& tc : kOps) {
        CAPTURE(cc.label, tc.tag);
        std::string tname = std::string("t_nw_") + cc.label + "_" + tc.tag;
        write_plan_file(plan_dir, tname, tc.plan);

        auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='" + tname + "');");
        require_ok(pin, std::string("pin:") + tc.tag);

        auto const census = sirius::test::census_entry(con, tname);
        REQUIRE(census.first_chunk_carriers ==
                std::vector<cudf::data_type>{cudf::data_type{cc.carrier}});
        REQUIRE(census.compressed_chunks == census.chunks);

        auto res = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
        require_ok(res, std::string("select:") + tc.tag);
        REQUIRE(res->RowCount() == 1);
        REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(cc.sum));

        run_ok(con, "CALL unpin_table('" + tname + "');", std::string("unpin:") + tc.tag);
      }

      fs::remove_all(tmp);
    }
  }
}

// A natively 16-bit column reaches the compressor at INT16 whatever the narrowing setting is: no
// carrier is narrower than SMALLINT for these values, so the chooser proposes nothing and the
// column keeps its native carrier. This guards Simpatico's 16-bit support on its own, with carrier
// narrowing entirely uninvolved.
TEST_CASE("pin_table compression - a native SMALLINT column compresses",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("smallint");
  // Values exceed INT8, so the column stays at its native INT16 carrier. 1000 distinct values,
  // ten rows each; SUM(k) = 10 * (0+1+...+999).
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT (range % 1000)::SMALLINT AS k FROM range(10000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  run_ok(con, "SET pin_table_compression_max_compressed_fraction = 1.5;", "set fraction");
  // enable_compressed_materialization stays at its default (off): this case is about the native
  // carrier, not a narrowed one.

  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_smallint", bitpack_plan(1));
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_smallint');");
  require_ok(pin, "pin smallint");

  auto const census = sirius::test::census_entry(con, "t_smallint");
  REQUIRE(census.compressed_chunks == census.chunks);
  REQUIRE_FALSE(census.any_marker_true);
  REQUIRE(census.first_chunk_carriers ==
          std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT16}});

  auto res =
    con.Query("CALL gpu_execution(\"SELECT COUNT(*), SUM(k) FROM read_parquet('" + glob + "')\");");
  require_ok(res, "select smallint");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(10000LL));
  REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(4995000LL));

  run_ok(con, "CALL unpin_table('t_smallint');", "unpin smallint");
  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - single-op sweep over float operators (ALP / ALP-RD)",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("sweep_float");

  // DOUBLE col d: integer-valued doubles — ideal for ALP (exact representation,
  //   no exceptions).  SUM(d) = 12497500.0 exactly (< 2^53).
  // FLOAT col f: same values cast to FLOAT32.  ALP-RD targets FLOAT32.
  //   12497500 < 2^24 so it is exactly representable; SUM is exact.
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT CAST(range AS DOUBLE) AS d, CAST(range AS FLOAT) AS f FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  // Column block 0 (d, DOUBLE) → alp; block 1 (f, FLOAT) → alp_rd.
  // Bare forms: all sub-channels become terminal leaves inside the rep.
  write_plan_file(plan_dir, "t_sw_float", "input -> alp\n---\ninput -> alp_rd\n");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_sw_float');");
  require_ok(pin, "pin float");

  const std::string select_sql = "SELECT SUM(d), SUM(f) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select float");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::DOUBLE(12497500.0));
  REQUIRE(res->GetValue(1, 0) == duckdb::Value::FLOAT(12497500.0f));

  run_ok(con, "CALL unpin_table('t_sw_float');", "unpin float");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - single-op sweep over string operators (dictionary)",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("sweep_str");

  // 10-char zero-padded strings: '0000000000'..'0000004999'.
  // MAX(s) = '0000004999' (lexicographic max of zero-padded decimal).
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT printf('%010d', range) AS s FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  write_plan_file(plan_dir, "t_sw_dict", "input -> dictionary\n");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_sw_dict');");
  require_ok(pin, "pin dict");

  const std::string select_sql = "SELECT COUNT(*), MAX(s) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select dict");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(5000LL));
  REQUIRE(res->GetValue(1, 0).ToString() == "0000004999");

  run_ok(con, "CALL unpin_table('t_sw_dict');", "unpin dict");

  fs::remove_all(tmp);
}

// ─── Fused cascade + entropy tail ────────────────────────────────────────────
//
// These tests exercise the fused-region entropy-tail code path: the JIT fused
// region (delta → rle → bitpack) emits `packed` as a raw passthrough channel,
// which the compress walk detects as having a downstream non-fused consumer
// (bitcomp / ans).  The RawFused leaf strips its `data` buffer and hands it to
// the entropy codec; on decode the entropy codec decompresses first, then the
// JIT inverse kernel reconstructs the original values using the stored offsets.
//
// `rle.runs` and `bitpack.{chunk_min,chunk_count,chunk_bits}` are terminal
// within the fused region and stored in the fused rep's internal buffers;
// they do not require explicit routing lines in the plan.

TEST_CASE("pin_table compression - fused delta->rle->bitpack with bitcomp entropy tail",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("fused_bitcomp");

  // SUM(k) = 12497500
  sirius::test::mgpu::generate_parquet_surface(tmp, "SELECT range AS k FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  write_plan_file(plan_dir,
                  "t_fused_bitcomp",
                  "input -> delta -> differences\n"
                  "delta.differences -> rle -> runs, values\n"
                  "rle.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
                  "bitpack.packed -> bitcomp\n");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_fused_bitcomp');");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT SUM(k) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(12497500LL));

  run_ok(con, "CALL unpin_table('t_fused_bitcomp');", "unpin");

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - fused delta->rle->bitpack with ANS entropy tail",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("fused_ans");

  // SUM(k) = 12497500
  sirius::test::mgpu::generate_parquet_surface(tmp, "SELECT range AS k FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  write_plan_file(plan_dir,
                  "t_fused_ans",
                  "input -> delta -> differences\n"
                  "delta.differences -> rle -> runs, values\n"
                  "rle.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
                  "bitpack.packed -> ans\n");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_fused_ans');");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT SUM(k) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(12497500LL));

  run_ok(con, "CALL unpin_table('t_fused_ans');", "unpin");

  fs::remove_all(tmp);
}

// ─── str_split cascade ────────────────────────────────────────────────────────
//
// str_split decomposes a STRING column into {offsets (INT32), chars (UINT8)}.
// The offsets sub-plan uses a fused codegen cascade (delta → rle → bitpack):
// for fixed-length-10 strings, all offsets differ by exactly 10, so delta
// produces a constant-10 differences stream, RLE collapses it to one run, and
// bitpack packs the tiny values into minimal bits.  The chars sub-plan uses
// snappy (byte-oriented entropy codec).  Together they exercise the "structural
// op with heterogeneous child plans" codepath.

TEST_CASE("pin_table compression - str_split cascade (snappy chars, delta->rle->bitpack offsets)",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("strsplit");

  // Fixed-length 10-char strings: '0000000000'..'0000004999'.
  // COUNT(*) = 5000, MAX(s) = '0000004999'.
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT printf('%010d', range) AS s FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");

  auto plan_dir = tmp / "plans";
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  // offsets: 0,10,20,...,50000 — constant delta=10, one RLE run, 4-bit bitpack.
  // chars:   raw ASCII bytes    — snappy on the UINT8 byte stream.
  write_plan_file(plan_dir,
                  "t_strsplit",
                  "input -> str_split -> offsets, chars\n"
                  "str_split.offsets -> delta -> differences\n"
                  "delta.differences -> rle -> runs, values\n"
                  "rle.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
                  "str_split.chars -> snappy\n");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_strsplit');");
  require_ok(pin, "pin");

  const std::string select_sql = "SELECT COUNT(*), MAX(s) FROM read_parquet('" + glob + "')";
  auto res                     = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(5000LL));
  REQUIRE(res->GetValue(1, 0).ToString() == "0000004999");

  run_ok(con, "CALL unpin_table('t_strsplit');", "unpin");

  fs::remove_all(tmp);
}

// ---------------------------------------------------------------------------
// Composition with compressed materialization (carrier narrowing)
// ---------------------------------------------------------------------------

// Stacking contract: narrowing runs in the pin driver before compression, so a
// compression-enabled pin stores NARROW compressed chunks and Simpatico's
// round-trip contract makes decompression reproduce the narrow carriers
// directly. The residency gate reads the recorded storage metadata, installs a
// narrow sidecar, and the serve is cast-free — scan_columns_restored stays flat
// while the chunks are compressed representations.
TEST_CASE("pin_table compression - narrowing stacks with compression (decompress to narrow)",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("bothflags");

  // k = range % 1000 and v = (range * 3) % 2000 both fit INT16 but not INT8, so both select an
  // INT16 carrier; w spans 0..59999 as UBIGINT, which selects UINT16 and carries the unsigned
  // 16-bit carrier through the same narrow-then-compress path. k takes 1000 distinct values, ten
  // rows each.
  sirius::test::mgpu::generate_parquet_surface(tmp,
                                               "SELECT range % 1000 AS k, (range * 3) % 2000 AS v, "
                                               "((range * 7) % 60000)::UBIGINT AS w "
                                               "FROM range(10000)",
                                               1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  run_ok(con, "SET enable_compressed_materialization = true;", "set narrowing");

  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_both", bitpack_plan(3));
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
  auto pin              = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_both');");
  require_ok(pin, "pin both-on");
  auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
  REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

  // Narrowed AND compressed: every chunk is a compressed representation and the recorded metadata
  // shows every column narrowed to a 16-bit carrier, signed and unsigned alike. This triple is the
  // primary guard on Simpatico's 16-bit support — a regression there throws during the pin, latches
  // compression off, and every chunk stores uncompressed, so compressed_chunks == chunks fails.
  auto const both = sirius::test::census_entry(con, "t_both");
  REQUIRE(both.compressed_chunks == both.chunks);
  REQUIRE(both.all_columns_narrowed);
  REQUIRE(both.first_chunk_carriers ==
          std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT16},
                                       cudf::data_type{cudf::type_id::INT16},
                                       cudf::data_type{cudf::type_id::UINT16}});

  // The gate reads the recorded carriers and installs the sidecar; the serve
  // decompresses straight into the narrow carriers, so nothing narrows or
  // restores at scan time (the cast-free happy path) and results match DuckDB.
  // Grouping by k gives the carrier a transport use (group keys stay narrow
  // through the aggregate), so no restore lands at the scan — bare aggregate
  // inputs like SUM(k) would be zero-benefit-pruned back to native at plan time
  // and widen at the scan instead (a legitimate, separately tested path).
  auto const before = sirius::test::get_compressed_materialization_stats(con);
  const std::string select_sql =
    "SELECT COUNT(*), MIN(cnt), MAX(cnt) FROM (SELECT k, COUNT(*) AS cnt FROM read_parquet('" +
    glob + "') GROUP BY k)";
  auto res = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select");
  REQUIRE(res->RowCount() == 1);
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1000LL));
  REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(10LL));
  REQUIRE(res->GetValue(2, 0) == duckdb::Value::BIGINT(10LL));
  auto const after = sirius::test::get_compressed_materialization_stats(con);
  REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
  REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
  REQUIRE(after.scan_columns_restored == before.scan_columns_restored);

  run_ok(con, "CALL unpin_table('t_both');", "unpin both");

  fs::remove_all(tmp);
}

// The compression-only contrast to the stacking case above: with narrowing off, a
// compression-enabled pin stores NATIVE carriers inside its compressed chunks, and a later
// narrowing-on query installs no sidecar over them (cached native columns are never narrowed at
// serve time). The discriminating fact is the recorded carriers, not the stored bytes: bitpack
// derives its width from the data, so a narrow and a native input compress to near-identical
// payloads.
TEST_CASE("pin_table compression - compression without narrowing stores native carriers",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("componly");
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range % 1000 AS k, (range * 3) % 2000 AS v FROM range(10000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  run_ok(con, "SET enable_compressed_materialization = false;", "narrowing off");
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_only", bitpack_plan(2));
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin_only = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_only');");
  require_ok(pin_only, "pin compression-only");

  auto const only = sirius::test::census_entry(con, "t_only");
  REQUIRE(only.compressed_chunks == only.chunks);
  REQUIRE_FALSE(only.any_marker_true);
  REQUIRE_FALSE(only.all_columns_narrowed);
  // The columns are BIGINT in the parquet surface, so native means INT64 everywhere.
  for (auto const carrier : only.first_chunk_carriers) {
    REQUIRE(carrier == cudf::data_type{cudf::type_id::INT64});
  }

  run_ok(con, "SET enable_compressed_materialization = true;", "narrowing on");
  auto const before = sirius::test::get_compressed_materialization_stats(con);
  const std::string select_sql =
    "SELECT COUNT(*), MIN(cnt), MAX(cnt) FROM (SELECT k, COUNT(*) AS cnt FROM read_parquet('" +
    glob + "') GROUP BY k)";
  auto res = con.Query("CALL gpu_execution(\"" + select_sql + "\");");
  require_ok(res, "select compression-only");
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1000LL));
  auto const after = sirius::test::get_compressed_materialization_stats(con);
  REQUIRE(after.scan_sidecars_installed == before.scan_sidecars_installed);
  REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);

  run_ok(con, "CALL unpin_table('t_only');", "unpin only");

  fs::remove_all(tmp);
}

// GPU-tier stacking: the compressed payload stays in device memory, the recorded
// metadata still drives the gate, and the tier narrowing policy now applies
// (sidecar_from_gpu_tier_pin). A group-key column has a transport use and serves
// cast-free from the compressed chunk; a filter-and-order-only column retracts
// at plan time and widens during scan normalization.
TEST_CASE("pin_table compression - narrowing stacks on the GPU tier and tier policy retracts",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("bothflags-gpu");
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range % 1000 AS k, (range * 3) % 2000 AS v FROM range(10000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  run_ok(con, "SET enable_compressed_materialization = true;", "set narrowing");
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_gpu", bitpack_plan(2));
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
  auto pin              = con.Query("CALL pin_table('" + glob + "', tier='gpu', name='t_gpu');");
  require_ok(pin, "pin gpu both-on");
  auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
  REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

  auto const census = sirius::test::census_entry(con, "t_gpu");
  REQUIRE(census.compressed_chunks == census.chunks);
  REQUIRE(census.all_columns_narrowed);
  for (auto const carrier : census.first_chunk_carriers) {
    REQUIRE(carrier == cudf::data_type{cudf::type_id::INT16});
  }

  // Group keys are a transport use, so k stays narrow through the aggregate on
  // GPU tier too, and the compressed chunk decompresses straight to the narrow
  // carrier — cast-free.
  {
    auto const before = sirius::test::get_compressed_materialization_stats(con);
    auto res          = con.Query(
      "CALL gpu_execution(\"SELECT COUNT(*), MIN(cnt), MAX(cnt) FROM (SELECT k, COUNT(*) AS cnt "
               "FROM read_parquet('" +
      glob + "') GROUP BY k)\");");
    require_ok(res, "narrow-kept select");
    REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1000LL));
    REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(10LL));
    REQUIRE(res->GetValue(2, 0) == duckdb::Value::BIGINT(10LL));
    auto const after = sirius::test::get_compressed_materialization_stats(con);
    REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
    REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);
    REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
    REQUIRE(after.scan_columns_restored == before.scan_columns_restored);
  }

  // A filter-and-order-only k has no transport use, so the GPU-tier policy
  // retracts the target at plan time and the decompressed narrow chunk widens
  // during scan normalization.
  {
    auto const before = sirius::test::get_compressed_materialization_stats(con);
    auto res          = con.Query("CALL gpu_execution(\"SELECT k FROM read_parquet('" + glob +
                         "') WHERE k < 5 ORDER BY k\");");
    require_ok(res, "retracted select");
    REQUIRE(res->RowCount() == 50);
    REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(0LL));
    REQUIRE(res->GetValue(0, 49) == duckdb::Value::BIGINT(4LL));
    auto const after = sirius::test::get_compressed_materialization_stats(con);
    REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
    REQUIRE(after.scan_narrow_targets_retracted > before.scan_narrow_targets_retracted);
    REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
  }

  // Pin-on / query-off over compressed chunks: the entry keeps its narrow carriers, the query
  // installs no sidecar at all, and the chunk that decompresses narrow is restored to native by
  // scan normalization. Same restore path the retraction leg takes, reached without any sidecar.
  {
    run_ok(con, "SET enable_compressed_materialization = false;", "narrowing off");
    auto const before = sirius::test::get_compressed_materialization_stats(con);
    auto res          = con.Query(
      "CALL gpu_execution(\"SELECT COUNT(*), MIN(cnt), MAX(cnt) FROM (SELECT k, COUNT(*) AS cnt "
               "FROM read_parquet('" +
      glob + "') GROUP BY k)\");");
    require_ok(res, "pin-on/query-off select");
    REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1000LL));
    REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(10LL));
    REQUIRE(res->GetValue(2, 0) == duckdb::Value::BIGINT(10LL));
    auto const after = sirius::test::get_compressed_materialization_stats(con);
    REQUIRE(after.scan_sidecars_installed == before.scan_sidecars_installed);
    REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
    REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
    run_ok(con, "SET enable_compressed_materialization = true;", "narrowing back on");
  }

  run_ok(con, "CALL unpin_table('t_gpu');", "unpin");
  fs::remove_all(tmp);
}

namespace {

// Write @p select_sql to @p file as parquet with @p row_group_size rows per row
// group (Sirius disabled so generation never touches the GPU under test).
void generate_parquet_row_groups(const fs::path& file,
                                 const std::string& select_sql,
                                 std::size_t row_group_size)
{
  fs::create_directories(file.parent_path());
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r =
      gen.Query("COPY (" + select_sql + ") TO '" + file.string() +
                "' (FORMAT PARQUET, ROW_GROUP_SIZE " + std::to_string(row_group_size) + ");");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

}  // namespace

// Heterogeneous chunk widths under compression: each chunk's blob records its
// own carriers, the plan target is the widest across chunks, and a chunk stored
// narrower widens right after decode (scan_columns_restored counts it).
TEST_CASE("pin_table compression - heterogeneous narrow widths widen post-decode",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  // 8 KiB scan batches (under one 2048-row BIGINT row group's 16 KiB) make each
  // row group its own pin chunk.
  auto [tmp, yaml_path] = make_comp_env("hetero", /*scan_batch_bytes=*/8u << 10);

  // Chunk 0 (rows < 2048) fits INT8; the remaining chunks need INT32 (values
  // over 32767 rule out INT16). The plan target is the widest carrier, INT32,
  // so chunk 0 widens after decode.
  generate_parquet_row_groups(tmp / "hetero.parquet",
                              "SELECT CASE WHEN range < 2048 THEN range % 100 "
                              "ELSE 40000 + range % 1000 END AS k FROM range(8192)",
                              2048);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con        = env.make_connection();
  auto const file = (tmp / "hetero.parquet").string();

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  // The INT8 chunk's 7-bit values barely compress, so let it keep the
  // compressed form anyway: this fixture is about heterogeneous widths inside
  // compressed chunks, not the fraction gate (the fail-soft test covers mixed
  // storage forms).
  run_ok(con, "SET pin_table_compression_max_compressed_fraction = 1.5;", "set fraction");
  run_ok(con, "SET enable_compressed_materialization = true;", "set narrowing");
  auto plan_dir = tmp / "plans";
  write_plan_file(
    plan_dir, "t_hetero", "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + file + "', tier='host', name='t_hetero');");
  require_ok(pin, "pin hetero");

  auto const census = sirius::test::census_entry(con, "t_hetero");
  REQUIRE(census.chunks >= 2);
  REQUIRE(census.compressed_chunks == census.chunks);
  REQUIRE(census.all_columns_narrowed);
  REQUIRE(census.first_chunk_carriers ==
          std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT8}});

  // Grouping by k gives the carrier a transport use, so the plan target stays
  // the widest recorded carrier (INT32) instead of being zero-benefit-pruned
  // back to native. 1100 distinct values (100 small + 1000 large); the small
  // ones appear 20-21 times (2048 rows) and the large ones 6-7 times (6144).
  auto const before = sirius::test::get_compressed_materialization_stats(con);
  auto res          = con.Query(
    "CALL gpu_execution(\"SELECT COUNT(*), MIN(cnt), MAX(cnt) FROM (SELECT k, COUNT(*) AS cnt "
             "FROM read_parquet('" +
    file + "') GROUP BY k)\");");
  require_ok(res, "select hetero");
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1100LL));
  REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(6LL));
  REQUIRE(res->GetValue(2, 0) == duckdb::Value::BIGINT(21LL));
  auto const after = sirius::test::get_compressed_materialization_stats(con);
  REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
  // Only the chunks stored narrower than the INT32 target widen post-decode:
  // more than zero restores, but strictly fewer than one per chunk (which is
  // what an all-native plan target would produce).
  auto const restored = after.scan_columns_restored - before.scan_columns_restored;
  REQUIRE(restored > 0);
  REQUIRE(restored < census.chunks);
  REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);

  run_ok(con, "CALL unpin_table('t_hetero');", "unpin");
  fs::remove_all(tmp);
}

// Compression engages only when the plan resolver finds a plan for the pinned table, not from
// pin_table_compression alone. With the setting on but no plan file covering this table, nothing
// hands the pin to Simpatico, so no chunk may end up compressed while narrowing still proceeds
// independently and stores its selected carrier.
TEST_CASE("pin_table compression - no plan file for the table pins uncompressed",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("noplan");
  // k fits INT16 but not INT8; 1000 distinct values, ten rows each.
  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range % 1000 AS k FROM range(10000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  run_ok(con, "SET enable_compressed_materialization = true;", "set narrowing");
  // A plan directory holding a plan for some OTHER table: the resolver finds no plan for this
  // one, so compression never engages.
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "some_other_table", bitpack_plan(2));
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_noplan');");
  require_ok(pin, "pin no-plan");

  auto const census = sirius::test::census_entry(con, "t_noplan");
  REQUIRE(census.compressed_chunks == 0);
  REQUIRE(census.all_columns_narrowed);
  REQUIRE(census.first_chunk_carriers ==
          std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT16}});

  // The INT16 carrier serves correctly end to end.
  auto res = con.Query(
    "CALL gpu_execution(\"SELECT COUNT(*), MIN(cnt), MAX(cnt) FROM (SELECT k, COUNT(*) AS cnt "
    "FROM read_parquet('" +
    glob + "') GROUP BY k)\");");
  require_ok(res, "select no-plan");
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1000LL));
  REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(10LL));

  run_ok(con, "CALL unpin_table('t_noplan');", "unpin");
  fs::remove_all(tmp);
}

// A width-explicit packed op (bitextract with a 64-bit field spec) cannot encode
// a column narrowed to a different width: that batch's compression fails,
// pin_table warns and latches compression off, and the pin falls back to
// UNCOMPRESSED NARROW chunks — markers intact, results correct.
TEST_CASE("pin_table compression - width-explicit op on a narrowed column fails soft",
          "[compression][pin_table][isolated_context]")
{
  if (no_gpu()) { return; }

  auto [tmp, yaml_path] = make_comp_env("widthop");
  // k fits INT16; 1000 distinct values, five rows each.
  sirius::test::mgpu::generate_parquet_surface(tmp, "SELECT range % 1000 AS k FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con  = env.make_connection();
  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  run_ok(con, "SET pin_table_compression = true;", "set compression");
  run_ok(con, "SET pin_table_compression_min_batch_size_bytes = 0;", "set min_batch");
  run_ok(con, "SET enable_compressed_materialization = true;", "set narrowing");
  auto plan_dir = tmp / "plans";
  // Valid against the native INT64 column (32 + 32 field bits), unencodable
  // against the INT16 carrier the narrowing stores.
  write_plan_file(plan_dir, "t_widthop", "input -> bitextract_32hi_32lo -> hi, lo\n");
  run_ok(
    con, "SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';", "set plan_dir");

  auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_widthop');");
  require_ok(pin, "pin widthop");
  auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
  REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

  // Fail-soft outcome: no compressed chunks, but the narrowing survived.
  auto const census = sirius::test::census_entry(con, "t_widthop");
  REQUIRE(census.compressed_chunks == 0);
  REQUIRE(census.all_columns_narrowed);
  REQUIRE(census.first_chunk_carriers ==
          std::vector<cudf::data_type>{cudf::data_type{cudf::type_id::INT16}});

  // Grouping by k keeps the carrier narrow (transport use, see the stacking
  // test), so the uncompressed-narrow serve is cast-free.
  auto const before = sirius::test::get_compressed_materialization_stats(con);
  auto res          = con.Query(
    "CALL gpu_execution(\"SELECT COUNT(*), MIN(cnt), MAX(cnt) FROM (SELECT k, COUNT(*) AS cnt "
             "FROM read_parquet('" +
    glob + "') GROUP BY k)\");");
  require_ok(res, "select widthop");
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1000LL));
  REQUIRE(res->GetValue(1, 0) == duckdb::Value::BIGINT(5LL));
  REQUIRE(res->GetValue(2, 0) == duckdb::Value::BIGINT(5LL));
  auto const after = sirius::test::get_compressed_materialization_stats(con);
  REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
  REQUIRE(after.scan_columns_restored == before.scan_columns_restored);

  run_ok(con, "CALL unpin_table('t_widthop');", "unpin");
  fs::remove_all(tmp);
}
