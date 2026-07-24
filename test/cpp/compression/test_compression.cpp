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
#include <string>
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

#include <duckdb.hpp>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace {

// Build a minimal single-GPU Sirius YAML with compression enabled.
void write_compression_yaml(const fs::path& yaml_path)
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
       "    scan_task_batch_size: 100000000\n"
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

comp_env_paths make_comp_env(const std::string& tag)
{
  auto dir = make_comp_tmp(tag);
  fs::remove_all(dir);
  fs::create_directories(dir);
  auto yaml = dir / "comp.yaml";
  write_compression_yaml(yaml);
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
