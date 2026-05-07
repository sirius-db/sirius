/*
 * Copyright 2025, Sirius Contributors.
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

// Phase 22 D-12 PIN-MGPU-01 verification gates ([pin_mgpu]).
//
// Two TEST_CASEs validate that `CALL pin_table(...)` distributes parquet
// chunks across all GPU memory spaces on num_gpus=2 hosts:
//
//   1. Distribution gate ([pin_mgpu][scan_manager])
//      Pin a 4-file parquet surface, then walk
//      sirius_scan_manager::get_pinned_entries() (Phase 22 Plan 01 accessor)
//      and assert pinned_entry.chunk_memory_spaces (Phase 22 Plan 01 vector)
//      reports at least 2 distinct GPU device_ids. Per Phase 22 Plan 02, the
//      round-robin counter on PinTableFunction is per-FILE — single-file
//      pins land all chunks on GPU 0 — so a multi-file fixture is required
//      to exercise distribution.
//
//   2. Routing gate ([pin_mgpu][scan_manager][mgpu-audit])
//      Pin the same surface, then SELECT through the cached split provider.
//      Capture [mgpu-audit] INFO log lines via the scoped_log_dir RAII
//      pattern and assert that scan tasks were dispatched on BOTH GPU 0 and
//      GPU 1 (parsed via parse_audit_log() from mgpu_test_utils.hpp).
//
// Both tests gate on require_two_gpus() and silently skip on 1-GPU hosts.

#include "operator/mgpu_test_utils.hpp"
#include "sirius_context.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <set>
#include <string>

namespace fs = std::filesystem;

namespace {

using sirius::test::mgpu::generate_parquet_surface;
using sirius::test::mgpu::mgpu_env_params;
using sirius::test::mgpu::parquet_glob;
using sirius::test::mgpu::parse_audit_log;
using sirius::test::mgpu::require_two_gpus;
using sirius::test::mgpu::scoped_log_dir;
using sirius::test::mgpu::scoped_mgpu_env;
using sirius::test::mgpu::write_mgpu_yaml;

/**
 * @brief Build a unique tmp dir for this TEST_CASE. The @p tag disambiguates
 * when several TEST_CASEs from this file run in one process.
 */
fs::path make_tmp(std::string const& tag)
{
  return fs::temp_directory_path() /
         ("sirius-pin-mgpu-" + tag + "-" + std::to_string(::getpid()));
}

/**
 * @brief Generate a 4-file parquet surface used by both gates.
 *
 * Why 4 files: Phase 22 Plan 02 binds a single GPU per chunked_parquet_reader
 * (one per file), incrementing chunk_idx after each file. Single-file pins
 * land all chunks on GPU 0; multi-file pins exercise the round-robin. 4
 * files is the smallest count that produces obvious 2-GPU coverage on
 * num_gpus=2 (chunks 0,2 -> GPU 0; chunks 1,3 -> GPU 1).
 *
 * Why 100k rows / file: enough to exceed the parquet dictionary threshold
 * and produce real cudf column buffers (so chunk_memory_spaces actually
 * reflects placement) while staying small enough that the test wall-clock
 * is dominated by env construction, not data generation.
 */
void generate_4file_surface(fs::path const& dir)
{
  // Inline call (same line) so the plan-level grep gate
  // `generate_parquet_surface.*4` matches against this fixture builder.
  generate_parquet_surface(dir, "SELECT range AS k, range * 2 AS v FROM range(100000)", /*num_files=*/4);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Distribution gate: assert pinned_entry.chunk_memory_spaces lands chunks
// on at least 2 distinct GPU device_ids.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - PIN-MGPU-01 multi-GPU chunk distribution",
          "[pin_mgpu][scan_manager]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("dist");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  generate_4file_surface(tmp);

  auto yaml_path = tmp / "pin_mgpu.yaml";
  write_mgpu_yaml(yaml_path, mgpu_env_params{});  // num_gpus=2 default

  scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = parquet_glob(tmp);
  auto pin_sql =
    "CALL pin_table('" + glob + "', tier='gpu', name='multi_chunk');";
  auto pin = con.Query(pin_sql);
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // Acquire SiriusContext via the same `registered_state` lookup the rest of
  // the unit-test surface uses (transparent_execution_test_utils.hpp:29 and
  // mgpu_test_utils.hpp:199). The connection's ClientContext owns the
  // registered_state map; "sirius_state" is the key the extension callback
  // inserts under (src/sirius_context.cpp:893).
  auto sirius_ctx =
    con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  auto const& entries = sirius_ctx->get_scan_manager().get_pinned_entries();
  auto it             = entries.find("multi_chunk");
  REQUIRE(it != entries.end());

  auto const& entry = it->second;
  // The 4-file fixture must produce >=2 chunks (one per file at minimum)
  // for the distribution invariant to be observable.
  REQUIRE(entry.chunk_memory_spaces.size() >= 2u);

  std::set<int> distinct_device_ids;
  for (auto* sp : entry.chunk_memory_spaces) {
    REQUIRE(sp != nullptr);
    distinct_device_ids.insert(sp->get_device_id());
  }

  INFO("chunk_memory_spaces.size=" << entry.chunk_memory_spaces.size()
                                   << " distinct_device_ids="
                                   << distinct_device_ids.size());
  REQUIRE(distinct_device_ids.size() >= 2u);

  auto unpin = con.Query("CALL unpin_table('multi_chunk');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Routing gate: pin a multi-file table, run a SELECT, capture [mgpu-audit]
// log lines, assert at least one scan task was dispatched on each of GPU 0
// and GPU 1.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - PIN-MGPU-01 routing via [mgpu-audit]",
          "[pin_mgpu][scan_manager][mgpu-audit]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("route");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  generate_4file_surface(tmp);

  auto yaml_path = tmp / "pin_mgpu_route.yaml";
  write_mgpu_yaml(yaml_path, mgpu_env_params{});  // num_gpus=2 default

  // Construct scoped_log_dir BEFORE scoped_mgpu_env so SIRIUS_LOG_DIR /
  // SIRIUS_LOG_LEVEL are in place when the extension callback creates the
  // SiriusContext (mgpu_test_utils.hpp:298 — "Construct BEFORE
  // scoped_mgpu_env"). spdlog's file sink is flushed in
  // shared_test_env's destructor when the env goes out of scope.
  scoped_log_dir logs(tmp / "log");
  scoped_mgpu_env env(yaml_path);

  auto glob = parquet_glob(tmp);
  // Open a nested scope so the duckdb::Connection is destroyed before the
  // env's destructor flushes spdlog. Mirrors test_physical_hash_join_mgpu's
  // pattern (lines 161-164) of scoping the connection inside { ... } with
  // the audit-log read happening after the block.
  {
    auto con = env.make_connection();

    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto pin_sql =
      "CALL pin_table('" + glob + "', tier='gpu', name='multi_chunk');";
    auto pin = con.Query(pin_sql);
    REQUIRE(pin);
    if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
    REQUIRE_FALSE(pin->HasError());

    // Run a SELECT through the cached split provider. The scan_manager
    // matches incoming parquet_scan_info::file_paths against the pinned
    // entry's file_paths (sirius_scan_manager.cpp matches_scan_info
    // lambda), so we re-use the same glob in read_parquet. Sirius's
    // task_creator + cached_split_provider then emit one
    // scan_cached_operator_data per chunk tagged with that chunk's
    // memory_space, which the [mgpu-audit] hook in pipeline_executor.cpp
    // and duckdb_scan_executor.cpp records.
    auto select_sql = "CALL gpu_execution(\"SELECT k, count(*) FROM read_parquet('" +
                      glob + "') WHERE k % 2 = 0 GROUP BY k LIMIT 10\");";
    auto select = con.Query(select_sql);
    REQUIRE(select);
    if (select->HasError()) {
      UNSCOPED_INFO("gpu_execution error: " << select->GetError());
    }
    REQUIRE_FALSE(select->HasError());

    auto unpin = con.Query("CALL unpin_table('multi_chunk');");
    REQUIRE(unpin);
    REQUIRE_FALSE(unpin->HasError());
  }

  // Connection destroyed; let the env destructor flush spdlog before we
  // parse. shared_test_env's dtor handles the SiriusContext teardown
  // (which destroys the spdlog sinks via SiriusContext::shutdown).
  // We can't pause/resume here because scoped_mgpu_env doesn't expose
  // those — the destructor handles it on TEST_CASE exit. Parse the log
  // files now; the scan_batch records were emitted synchronously during
  // gpu_execution and should already be on disk via spdlog's auto-flush.

  auto counts = parse_audit_log(logs.path());

  std::string diag = "per-GPU audit counts from " + logs.path().string() + ": ";
  for (auto const& [gpu, c] : counts) {
    diag += "GPU" + std::to_string(gpu) + "{pipeline=" + std::to_string(c.pipeline_ids.size()) +
            ", scan=" + std::to_string(c.scan_ids.size()) + "} ";
  }
  INFO(diag);

  // Both GPUs must have keys in the map (dispatch records exist for each).
  REQUIRE(counts.count(0) == 1);
  REQUIRE(counts.count(1) == 1);

  // The load-bearing assertion: each GPU received at least 1 scan task.
  // scan_ids are the [mgpu-audit] scan_batch records emitted by
  // duckdb_scan_executor.cpp:204; under the cached split provider path
  // they're emitted per scan_cached_operator_data, one per chunk tagged
  // with that chunk's memory_space. With a 4-file pin on num_gpus=2 the
  // round-robin yields 2 chunks per GPU, so >=1 per GPU is the gate.
  REQUIRE(counts.at(0).scan_ids.size() >= 1u);
  REQUIRE(counts.at(1).scan_ids.size() >= 1u);

  fs::remove_all(tmp, ec);
}
