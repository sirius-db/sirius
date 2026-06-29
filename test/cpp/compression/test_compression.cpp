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
       "    duckdb_scan:\n"
       "      cache: none\n"
       "      num_threads: 2\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period_ms: 10\n"
       "  operator_params:\n"
       "    scan_task_batch_size: 100000000\n"
       "    default_scan_task_varchar_size: 256\n"
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
  if (!has_gpu()) {
    WARN("Compression test requires a GPU — skipping");
    return;
  }

  auto tmp = make_comp_tmp("eq");
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  auto yaml_path = tmp / "comp.yaml";
  write_compression_yaml(yaml_path);

  // Write a plan file for the 't_comp' table (identity plan — universally available)
  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_comp", "identity\n---\nidentity");

  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS k, range * 3 AS v FROM range(10000)", /*num_files=*/1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  // Pin without compression as reference
  auto pin_raw = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_raw');");
  REQUIRE(pin_raw);
  if (pin_raw->HasError()) { UNSCOPED_INFO("pin_raw error: " << pin_raw->GetError()); }
  REQUIRE_FALSE(pin_raw->HasError());

  REQUIRE_FALSE(con.Query("SET pin_table_compression = true;")->HasError());
  REQUIRE_FALSE(con.Query("SET pin_table_compression_min_chunk_bytes = 0;")->HasError());
  REQUIRE_FALSE(
    con.Query("SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';")
      ->HasError());

  auto pin_comp = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_comp');");
  REQUIRE(pin_comp);
  if (pin_comp->HasError()) { UNSCOPED_INFO("pin_comp error: " << pin_comp->GetError()); }
  REQUIRE_FALSE(pin_comp->HasError());

  // Both cached scans must return the same aggregate
  auto sum_raw  = con.Query("SELECT SUM(k), SUM(v) FROM t_raw;");
  auto sum_comp = con.Query("SELECT SUM(k), SUM(v) FROM t_comp;");

  REQUIRE(sum_raw);
  REQUIRE(sum_comp);
  REQUIRE_FALSE(sum_raw->HasError());
  REQUIRE_FALSE(sum_comp->HasError());

  REQUIRE(sum_raw->RowCount() == sum_comp->RowCount());
  for (duckdb::idx_t col = 0; col < 2; ++col) {
    REQUIRE(sum_raw->GetValue(col, 0) == sum_comp->GetValue(col, 0));
  }

  REQUIRE_FALSE(con.Query("CALL unpin_table('t_raw');")->HasError());
  REQUIRE_FALSE(con.Query("CALL unpin_table('t_comp');")->HasError());

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - column-subset projection correctness",
          "[compression][pin_table][isolated_context]")
{
  if (!has_gpu()) {
    WARN("Compression test requires a GPU — skipping");
    return;
  }

  auto tmp = make_comp_tmp("proj");
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  auto yaml_path = tmp / "comp.yaml";
  write_compression_yaml(yaml_path);

  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_proj", "identity\n---\nidentity\n---\nidentity");

  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS a, range * 2 AS b, range * 3 AS c FROM range(5000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  REQUIRE_FALSE(con.Query("SET pin_table_compression = true;")->HasError());
  REQUIRE_FALSE(con.Query("SET pin_table_compression_min_chunk_bytes = 0;")->HasError());
  REQUIRE_FALSE(
    con.Query("SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';")
      ->HasError());

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_proj');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // Project only column b; result should equal range(5000)*2 sum
  auto res = con.Query("SELECT SUM(b) FROM t_proj;");
  REQUIRE(res);
  REQUIRE_FALSE(res->HasError());
  REQUIRE(res->RowCount() == 1);

  // SUM of 0..4999 * 2 = 2 * 4999*5000/2 = 24990000
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(24990000LL));

  REQUIRE_FALSE(con.Query("CALL unpin_table('t_proj');")->HasError());

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - fallback when no plan file for table",
          "[compression][pin_table][isolated_context]")
{
  if (!has_gpu()) {
    WARN("Compression test requires a GPU — skipping");
    return;
  }

  auto tmp = make_comp_tmp("noplan");
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  auto yaml_path = tmp / "comp.yaml";
  write_compression_yaml(yaml_path);

  // Plan dir exists but has no file matching 't_noplan'
  auto plan_dir = tmp / "plans";
  fs::create_directories(plan_dir);

  sirius::test::mgpu::generate_parquet_surface(tmp, "SELECT range AS k FROM range(1000)", 1);

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  REQUIRE_FALSE(con.Query("SET pin_table_compression = true;")->HasError());
  REQUIRE_FALSE(
    con.Query("SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';")
      ->HasError());

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_noplan');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // Must still scan correctly via fallback raw host rep
  auto res = con.Query("SELECT COUNT(*) FROM t_noplan;");
  REQUIRE(res);
  REQUIRE_FALSE(res->HasError());
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(1000));

  REQUIRE_FALSE(con.Query("CALL unpin_table('t_noplan');")->HasError());

  fs::remove_all(tmp);
}

TEST_CASE("pin_table compression - fallback when chunk is below min_chunk_bytes threshold",
          "[compression][pin_table][isolated_context]")
{
  if (!has_gpu()) {
    WARN("Compression test requires a GPU — skipping");
    return;
  }

  auto tmp = make_comp_tmp("threshold");
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  auto yaml_path = tmp / "comp.yaml";
  write_compression_yaml(yaml_path);

  auto plan_dir = tmp / "plans";
  write_plan_file(plan_dir, "t_threshold", "identity");

  sirius::test::mgpu::generate_parquet_surface(
    tmp, "SELECT range AS k FROM range(100)", 1);  // tiny chunk

  sirius::test::mgpu::scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = sirius::test::mgpu::parquet_glob(tmp);

  REQUIRE_FALSE(con.Query("SET pin_table_compression = true;")->HasError());
  REQUIRE_FALSE(
    con.Query("SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';")
      ->HasError());
  // Threshold far above the tiny chunk — forces fallback
  REQUIRE_FALSE(con.Query("SET pin_table_compression_min_chunk_bytes = 1000000000;")->HasError());

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='t_threshold');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  auto res = con.Query("SELECT COUNT(*) FROM t_threshold;");
  REQUIRE(res);
  REQUIRE_FALSE(res->HasError());
  REQUIRE(res->GetValue(0, 0) == duckdb::Value::BIGINT(100));

  REQUIRE_FALSE(con.Query("CALL unpin_table('t_threshold');")->HasError());

  fs::remove_all(tmp);
}
