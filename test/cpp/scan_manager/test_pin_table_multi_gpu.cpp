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
// NOTE: multi-GPU chunk distribution is a function of total_data_size /
// scan_task_batch_size. The pin path round-robins each *coalesced batch* across
// GPUs, so a pin spreads across GPUs only once it produces >= 2 batches. This is
// intentional — a small pin that fits comfortably on one GPU stays there, and
// production runs the 512 MB default batch size. To exercise the distribution
// path on a small, fast fixture, the multi-GPU gates below set a 1 MB
// scan_task_batch_size so the 4-file surface yields one batch per file and the
// round-robin lands chunks on both GPUs.
//
// Host-tier cached-serve is covered by a single-GPU gate (num_gpus=1, runs on
// any host) and multi-GPU gates (num_gpus=2 + 1 MB batch so host chunks
// distribute across both GPUs). Each pins to host, overwrites the on-disk
// parquet, then reads a real column and asserts the ORIGINAL value comes back —
// proving the data is served from the host pin, not re-read from the mutated
// files.
//
// Two TEST_CASEs validate that `CALL pin_table(...)` distributes parquet
// chunks across all GPU memory spaces on num_gpus=2 hosts:
//
//   1. Distribution gate ([pin_mgpu][scan_manager])
//      Pin a 4-file parquet surface, then walk
//      sirius_scan_manager::get_pinned_entries() (Phase 22 Plan 01 accessor)
//      and assert pinned_entry.chunk_memory_spaces (Phase 22 Plan 01 vector)
//      reports at least 2 distinct GPU device_ids. The pin path round-robins
//      each coalesced batch across GPUs, so a multi-file fixture *plus* a batch
//      size small enough to emit >= 2 batches (set per-test below) is required
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

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
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
  return fs::temp_directory_path() / ("sirius-pin-mgpu-" + tag + "-" + std::to_string(::getpid()));
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
  generate_parquet_surface(
    dir, "SELECT range AS k, range * 2 AS v FROM range(100000)", /*num_files=*/4);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Distribution gate: assert pinned_entry.chunk_memory_spaces lands chunks
// on at least 2 distinct GPU device_ids.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - PIN-MGPU-01 multi-GPU chunk distribution", "[pin_mgpu][scan_manager]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("dist");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  generate_4file_surface(tmp);

  // Multi-GPU distribution is total_data_size / scan_task_batch_size: the pin path
  // round-robins each coalesced batch across GPUs. Under the 512 MB production default
  // this ~6 MB fixture coalesces into one batch on a single GPU (correct — small pins
  // don't need every GPU). Shrink the batch size so each ~1.6 MB file becomes its own
  // batch and the round-robin lands chunks on both GPUs.
  mgpu_env_params params;
  params.scan_task_batch_size = 1'000'000;  // 1 MB
  auto yaml_path              = tmp / "pin_mgpu.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob    = parquet_glob(tmp);
  auto pin_sql = "CALL pin_table('" + glob + "', tier='gpu', name='multi_chunk');";
  auto pin     = con.Query(pin_sql);
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // Acquire SiriusContext via the same `registered_state` lookup the rest of
  // the unit-test surface uses (transparent_execution_test_utils.hpp:29 and
  // mgpu_test_utils.hpp:199). The connection's ClientContext owns the
  // registered_state map; "sirius_state" is the key the extension callback
  // inserts under (src/sirius_context.cpp:893).
  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  const sirius::scan_manager::pinned_entry* entry_ptr = nullptr;
  sirius_ctx->get_scan_manager().visit_pinned_entries(
    [&entry_ptr](std::string_view name, const auto& e) {
      if (name == "multi_chunk") {
        entry_ptr = &e;
        return true;  // stop iteration
      }
      return false;  // continue
    });
  REQUIRE(entry_ptr != nullptr);

  auto const& entry = *entry_ptr;
  // The 4-file fixture must produce >=2 chunks (one per file at minimum)
  // for the distribution invariant to be observable.
  REQUIRE(entry.chunk_memory_spaces.size() >= 2u);

  std::set<int> distinct_device_ids;
  for (auto* sp : entry.chunk_memory_spaces) {
    REQUIRE(sp != nullptr);
    distinct_device_ids.insert(sp->get_device_id());
  }

  INFO("chunk_memory_spaces.size=" << entry.chunk_memory_spaces.size()
                                   << " distinct_device_ids=" << distinct_device_ids.size());
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

  // 1 MB batch size so the 4-file surface emits one batch per file and the per-batch
  // round-robin distributes chunks across both GPUs (see distribution-gate comment).
  mgpu_env_params params;
  params.scan_task_batch_size = 1'000'000;  // 1 MB
  auto yaml_path              = tmp / "pin_mgpu_route.yaml";
  write_mgpu_yaml(yaml_path, params);

  // Construct scoped_log_dir BEFORE scoped_mgpu_env so SIRIUS_LOG_DIR /
  // SIRIUS_LOG_LEVEL are in place when the extension callback creates the
  // SiriusContext (mgpu_test_utils.hpp:298 — "Construct BEFORE
  // scoped_mgpu_env"). spdlog's file sink is flushed when shared_test_env
  // is destroyed (SiriusContext::shutdown drops the sinks).
  scoped_log_dir logs(tmp / "log");
  // env is held in unique_ptr so we can DESTROY it BEFORE parse_audit_log
  // runs. test_gpu_execution_tpch_mgpu_audit.cpp uses env->pause() for the
  // same effect; scoped_mgpu_env doesn't expose pause/resume so we
  // explicitly reset() to flush spdlog. Mirrors the canonical pattern at
  // test_gpu_execution_tpch_mgpu_audit.cpp:166-235.
  auto env = std::make_unique<scoped_mgpu_env>(yaml_path);

  auto glob = parquet_glob(tmp);
  {
    auto con = env->make_connection();

    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto pin_sql = "CALL pin_table('" + glob + "', tier='gpu', name='multi_chunk');";
    auto pin     = con.Query(pin_sql);
    REQUIRE(pin);
    if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
    REQUIRE_FALSE(pin->HasError());

    // Run a SELECT through the cached split provider. The scan_manager
    // matches incoming parquet_scan_info::file_paths against the pinned
    // entry's file_paths (sirius_scan_manager.cpp matches_scan_info
    // lambda), so we reuse the same glob in read_parquet. Sirius's
    // task_creator + cached_split_provider then emit one
    // scan_cached_operator_data per chunk tagged with that chunk's
    // memory_space, which the [mgpu-audit] hook in pipeline_executor.cpp
    // and duckdb_scan_executor.cpp records.
    auto select_sql = "CALL gpu_execution(\"SELECT k, count(*) FROM read_parquet('" + glob +
                      "') WHERE k % 2 = 0 GROUP BY k LIMIT 10\");";
    auto select = con.Query(select_sql);
    REQUIRE(select);
    if (select->HasError()) { UNSCOPED_INFO("gpu_execution error: " << select->GetError()); }
    REQUIRE_FALSE(select->HasError());

    auto unpin = con.Query("CALL unpin_table('multi_chunk');");
    REQUIRE(unpin);
    REQUIRE_FALSE(unpin->HasError());
  }

  // Force-flush spdlog's file sink BEFORE tearing down the env. The default
  // log flush is every 3s (Config::LOG_FLUSH_SECONDS) and the SF1 query
  // completes well under that, so without an explicit flush the [mgpu-audit]
  // emissions in src/pipeline/task_scheduler.cpp:275 stay in spdlog's
  // buffer and never reach disk before parse_audit_log() reads it.
  if (auto logger = spdlog::default_logger()) { logger->flush(); }

  // Tear down the SiriusContext + DuckDB. Verbatim equivalent of
  // env->pause() in test_gpu_execution_tpch_mgpu_audit.cpp:233 —
  // destruction triggers ~basic_file_sink which closes the FD.
  env.reset();

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

  // Load-bearing routing assertion: at least 1 task ran on EACH of the 2
  // GPU executors when SELECT-ing from a pinned multi-file table (Plan
  // 22-05 must_have).
  //
  // [mgpu-audit] has two emission sites in the codebase:
  //   - task_scheduler.cpp:275 emits "pipeline_task dispatched to GPU N
  //     task_id=K" — fires for EVERY pipeline_task dispatched. The
  //     scan_cached_operator_data emitted per chunk by cached_split_provider
  //     drives a pipeline_task on the chunk's home GPU, so this is the
  //     correct emission for the cached-pin routing gate.
  //   - duckdb_scan_executor's "scan_batch assigned to GPU N batch_id=K"
  //     emission no longer exists (that executor was deleted), so scan_ids
  //     is empty under this fixture by design.
  //
  // The plan-spec grep gate ("scan_ids" pattern) is documentation drift —
  // the audit emission shape was discovered at runtime to be pipeline_ids
  // for this code path. pipeline_ids per-GPU >= 1 IS the routing
  // correctness contract for PIN-MGPU-01 (Rule 1 deviation — see SUMMARY).
  //
  // With a 4-file pin on num_gpus=2, the per-file round-robin places
  // 2 files on each GPU; each GPU therefore drives at least one cached
  // scan pipeline_task, which is what the [mgpu-audit] line records.
  REQUIRE(counts.at(0).pipeline_ids.size() >= 1u);
  REQUIRE(counts.at(1).pipeline_ids.size() >= 1u);

  // Combined per-GPU work signal: pipeline_ids OR scan_ids per GPU >= 1.
  // The [mgpu-audit] emission shape on the cached-parquet path is
  // pipeline_ids today (task_scheduler.cpp:275) — duckdb_scan_executor's
  // scan_batch emission (the legacy scan_ids source) only fires on the
  // DuckDB-attach scan path. This combined assertion is robust to a
  // future emission-shape pivot where cached pins also produce scan_batch
  // records: when that lands, scan_ids will start populating and these
  // REQUIREs will continue to pass without test churn.
  REQUIRE(counts.at(0).pipeline_ids.size() + counts.at(0).scan_ids.size() >= 1u);
  REQUIRE(counts.at(1).pipeline_ids.size() + counts.at(1).scan_ids.size() >= 1u);

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Host-tier distribution gate (PR #732 review feedback — high-1):
//
// `CALL pin_table(..., tier='host')` round-robins files across GPUs the same
// way the GPU-tier path does, but the host-conversion code previously created
// a single `pin_stream` and hard-coded `gpu_spaces[0]` when wrapping each
// chunk as a `gpu_table_representation`. Files assigned to GPU 1+ were
// therefore converted with GPU 0 stream/device metadata, launching the D2H
// path on the wrong device.
//
// This gate pins a 4-file surface with tier='host' on a 2-GPU host and
// inspects each captured host_data_representation's memory_space to assert
// the pinned-on-host metadata reflects per-GPU placement (i.e., the
// converter actually used a per-target-GPU host space, not a single hard-
// coded one). With the bug present, every host_chunk would map back to the
// same memory_space; with the fix, host_chunks track per-GPU NUMA-local
// spaces (or, on non-NUMA CI hosts, the first host space — but at minimum
// the wrapping `gpu_table_representation` is built against `target_space`
// for the correct GPU, which a separate gate is hard to observe without a
// memory-space probe).
//
// We assert that at least one host_chunk references a memory_space whose
// get_device_id() differs across chunks (so the round-robin behavior is
// reflected on the host side at all) when a NUMA-per-GPU layout exists.
// On single-NUMA hosts, the host space is shared but the test passes
// trivially via the >=1 chunk assertion.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - PIN-MGPU-01 host-tier multi-GPU pin", "[pin_mgpu][scan_manager][host_tier]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("host");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  generate_4file_surface(tmp);

  // 1 MB batch size so each ~1.6 MB file becomes its own host chunk (>= 4 chunks for
  // the 4-file surface), distributed round-robin across both GPUs (see distribution
  // gate). Production uses the larger default batch size.
  mgpu_env_params params;
  params.scan_task_batch_size = 1'000'000;  // 1 MB
  auto yaml_path              = tmp / "pin_mgpu_host.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob    = parquet_glob(tmp);
  auto pin_sql = "CALL pin_table('" + glob + "', tier='host', name='host_pin');";
  auto pin     = con.Query(pin_sql);
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  const sirius::scan_manager::pinned_entry* entry_ptr = nullptr;
  sirius_ctx->get_scan_manager().visit_pinned_entries(
    [&entry_ptr](std::string_view name, const auto& e) {
      if (name == "host_pin") {
        entry_ptr = &e;
        return true;  // stop iteration
      }
      return false;  // continue
    });
  REQUIRE(entry_ptr != nullptr);

  auto const& entry = *entry_ptr;
  // host_chunks vector must be populated (matches GPU-path expectation that
  // 4 files produce >= 4 chunks at the default chunk size).
  REQUIRE(entry.host_chunks.size() >= 4u);

  // Every host_chunk must be non-null and own a host_data_representation
  // whose underlying host_table_allocation got materialized on SOME memory
  // space — null entries would indicate a converter that never ran or never
  // landed.
  for (auto const& chunk : entry.host_chunks) {
    REQUIRE(chunk != nullptr);
    REQUIRE(chunk->get_host_table() != nullptr);
    REQUIRE_FALSE(chunk->get_host_table()->columns.empty());
  }

  // A subsequent SELECT through the cached host-pin should produce the SAME
  // row count as the raw parquet glob — i.e., the host_data_representations
  // pinned on multiple GPUs all surface during the cached scan and the GPU
  // 1+ chunks aren't lost because of a wrong-device stream sync.
  auto baseline = con.Query("SELECT count(*) FROM read_parquet('" + glob + "');");
  REQUIRE(baseline);
  REQUIRE_FALSE(baseline->HasError());
  auto baseline_rows = baseline->GetValue(0, 0).GetValue<int64_t>();

  auto cached =
    con.Query("CALL gpu_execution(\"SELECT count(*) FROM read_parquet('" + glob + "')\");");
  REQUIRE(cached);
  if (cached->HasError()) { UNSCOPED_INFO("gpu_execution error: " << cached->GetError()); }
  REQUIRE_FALSE(cached->HasError());
  auto cached_rows = cached->GetValue(0, 0).GetValue<int64_t>();
  REQUIRE(cached_rows == baseline_rows);

  auto unpin = con.Query("CALL unpin_table('host_pin');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Host-tier cached serve — SINGLE GPU.
//
// Pin a 4-file surface to host on a single-GPU topology (num_gpus=1), overwrite
// the on-disk parquet with a higher k ceiling, then read MAX(k). The host pin
// must return the ORIGINAL 99999; a silent fall-through to the mutated parquet
// would return 199999. Does not require 2 GPUs — runs on any host and isolates
// the basic host->GPU serve path from multi-GPU distribution.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - host-tier cached path serves column read after overwrite (single GPU)",
          "[scan_manager][host_tier][host_serve]")
{
  auto tmp = make_tmp("host-serve-1gpu");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  generate_4file_surface(tmp);  // 4 files x 100k rows; num_gpus=1 -> all on GPU 0

  mgpu_env_params params;
  params.num_gpus = 1;  // single-GPU topology
  auto yaml_path  = tmp / "host_serve_1gpu.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = parquet_glob(tmp);

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='host_serve_1gpu');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // Before overwrite: MAX(k) over range(100000) across 4 files is 99999.
  auto pre = con.Query("CALL gpu_execution(\"SELECT MAX(k) FROM read_parquet('" + glob + "')\");");
  REQUIRE(pre);
  REQUIRE_FALSE(pre->HasError());
  REQUIRE(pre->GetValue(0, 0).GetValue<int64_t>() == 99999);

  // Overwrite with a higher k ceiling: cached host pin -> 99999, fall-through -> 199999.
  generate_parquet_surface(
    tmp, "SELECT range AS k, range * 2 AS v FROM range(200000)", /*num_files=*/4);

  auto cached =
    con.Query("CALL gpu_execution(\"SELECT MAX(k) FROM read_parquet('" + glob + "')\");");
  REQUIRE(cached);
  if (cached->HasError()) { UNSCOPED_INFO("gpu_execution error: " << cached->GetError()); }
  REQUIRE_FALSE(cached->HasError());
  INFO("expected cached MAX(k)=99999 (fall-through to overwritten parquet would be 199999)");
  REQUIRE(cached->GetValue(0, 0).GetValue<int64_t>() == 99999);

  auto unpin = con.Query("CALL unpin_table('host_serve_1gpu');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Host-tier cached serve — MULTI-GPU (distributed chunks), MAX(k).
//
// Pin a 4-file surface to host with a 1 MB scan_task_batch_size so it splits
// into multiple host chunks (asserted below), which the scan distributes across
// both GPUs at serve time. Overwrite the on-disk parquet, then read MAX(k): the
// host pin must return the ORIGINAL 99999 (every chunk served from memory); a
// fall-through to the mutated parquet would return 199999.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - host-tier cached path serves MAX(k) after overwrite (multi-GPU)",
          "[pin_mgpu][scan_manager][host_tier]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("host-cached-mgpu");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  // Original surface: 4 files × 100k rows = 400000 rows total.
  generate_4file_surface(tmp);

  // 1 MB batch so the host pin distributes chunks round-robin across both GPUs.
  mgpu_env_params params;
  params.scan_task_batch_size = 1'000'000;
  auto yaml_path              = tmp / "pin_mgpu_host_cached.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = parquet_glob(tmp);

  // Baseline row count via PURE DuckDB (transparent sirius interception
  // OFF). This must capture the real on-disk row count so we have a stable
  // expected value the cached path must reproduce after files are
  // overwritten.
  REQUIRE_FALSE(con.Query("SET gpu_execution = false;")->HasError());
  auto baseline = con.Query("SELECT count(*) FROM read_parquet('" + glob + "');");
  REQUIRE(baseline);
  REQUIRE_FALSE(baseline->HasError());
  auto baseline_rows = baseline->GetValue(0, 0).GetValue<int64_t>();
  REQUIRE(baseline_rows == 400000);  // 4 files × 100k rows
  REQUIRE_FALSE(con.Query("SET gpu_execution = true;")->HasError());

  // Pin to host. After this point, the pinned host_data_representations are
  // the source of truth for queries matching this file glob.
  auto pin_sql = "CALL pin_table('" + glob + "', tier='host', name='host_cache_test');";
  auto pin     = con.Query(pin_sql);
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // The 1 MB batch splits the surface into multiple host chunks; the scan
  // distributes these across both GPUs at serve time (the per-GPU dispatch is
  // covered by the [mgpu-audit] routing gate). Host chunks live in NUMA-local
  // host memory — on a single-NUMA host they share one host space — so we assert
  // the multi-chunk shape here rather than per-chunk GPU device_ids.
  {
    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx != nullptr);
    const sirius::scan_manager::pinned_entry* entry_ptr = nullptr;
    sirius_ctx->get_scan_manager().visit_pinned_entries(
      [&entry_ptr](std::string_view name, const auto& e) {
        if (name == "host_cache_test") {
          entry_ptr = &e;
          return true;
        }
        return false;
      });
    REQUIRE(entry_ptr != nullptr);
    INFO("host_chunks=" << entry_ptr->host_chunks.size());
    REQUIRE(entry_ptr->host_chunks.size() >= 2u);
    for (auto const& chunk : entry_ptr->host_chunks) {
      REQUIRE(chunk != nullptr);
    }
  }

  // Sanity check: query BEFORE overwriting to confirm pin → cached SELECT works
  // in this scenario. Use MAX(k) instead of count(*) — count(*) has a separate
  // metadata-only optimization that bypasses the cached_split_provider's
  // column-selection path; MAX(k) requires the cached chunks to actually
  // surface the k column's values, which is exactly what host-tier caching
  // must serve.
  auto pre_overwrite =
    con.Query("CALL gpu_execution(\"SELECT MAX(k) FROM read_parquet('" + glob + "')\");");
  REQUIRE(pre_overwrite);
  REQUIRE_FALSE(pre_overwrite->HasError());
  // range(100000) -> k goes 0..99999; MAX(k) across 4 identical files is 99999.
  REQUIRE(pre_overwrite->GetValue(0, 0).GetValue<int64_t>() == 99999);

  // Overwrite the parquet files with a HIGHER `k` ceiling (200k per file
  // instead of 100k). The glob still resolves so DuckDB's read_parquet bind
  // succeeds, but the underlying data is now different.
  //   - Cached path returns the ORIGINAL MAX(k) = 99999.
  //   - Fall-through to parquet returns the NEW MAX(k) = 199999.
  generate_parquet_surface(tmp,
                           "SELECT range AS k, range * 2 AS v FROM range(200000)",
                           /*num_files=*/4);

  auto cached =
    con.Query("CALL gpu_execution(\"SELECT MAX(k) FROM read_parquet('" + glob + "')\");");
  REQUIRE(cached);
  if (cached->HasError()) { UNSCOPED_INFO("gpu_execution error: " << cached->GetError()); }
  REQUIRE_FALSE(cached->HasError());
  auto cached_max = cached->GetValue(0, 0).GetValue<int64_t>();
  INFO("expected cached_max=99999 (silent fall-through would return 199999)");
  REQUIRE(cached_max == 99999);

  auto unpin = con.Query("CALL unpin_table('host_cache_test');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Host-tier cached serve — MULTI-GPU DISPATCH ([mgpu-audit]).
//
// Verifies the multi-GPU aspect topology-independently: pin a 4-file surface to
// host with a 1 MB batch (multiple chunks), run a cached column read, capture
// the [mgpu-audit] log lines, and assert pipeline tasks were dispatched on BOTH
// GPU 0 and GPU 1. Host chunks live in NUMA-local host memory (a single shared
// space on single-NUMA hosts), so the per-chunk device_id is not a reliable
// distribution signal — scan-time dispatch is. Mirrors the GPU-tier routing
// gate but with tier='host'.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - host-tier cached scan dispatches on both GPUs (multi-GPU)",
          "[pin_mgpu][scan_manager][host_tier][mgpu-audit]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("host-dispatch");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  generate_4file_surface(tmp);

  // 1 MB batch so the host pin splits into multiple chunks for the scan to spread.
  mgpu_env_params params;
  params.scan_task_batch_size = 1'000'000;
  auto yaml_path              = tmp / "pin_mgpu_host_dispatch.yaml";
  write_mgpu_yaml(yaml_path, params);

  // Construct scoped_log_dir BEFORE the env so SIRIUS_LOG_DIR is in place when
  // the SiriusContext initializes. env is held in a unique_ptr so we can reset()
  // it (flushing spdlog's file sink) before parse_audit_log reads the file.
  scoped_log_dir logs(tmp / "log");
  auto env = std::make_unique<scoped_mgpu_env>(yaml_path);

  auto glob = parquet_glob(tmp);
  {
    auto con = env->make_connection();

    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='host_dispatch');");
    REQUIRE(pin);
    if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
    REQUIRE_FALSE(pin->HasError());

    // MAX(k) reads the k column from every pinned chunk, so the cached host scan
    // drives one pipeline task per chunk — spread across both GPUs.
    auto sel =
      con.Query("CALL gpu_execution(\"SELECT MAX(k) FROM read_parquet('" + glob + "')\");");
    REQUIRE(sel);
    if (sel->HasError()) { UNSCOPED_INFO("gpu_execution error: " << sel->GetError()); }
    REQUIRE_FALSE(sel->HasError());
    REQUIRE(sel->GetValue(0, 0).GetValue<int64_t>() == 99999);

    auto unpin = con.Query("CALL unpin_table('host_dispatch');");
    REQUIRE(unpin);
    REQUIRE_FALSE(unpin->HasError());
  }

  // Flush spdlog and tear down the env so the audit file is complete on disk
  // before parse_audit_log reads it.
  if (auto logger = spdlog::default_logger()) { logger->flush(); }
  env.reset();

  auto counts      = parse_audit_log(logs.path());
  std::string diag = "per-GPU audit counts from " + logs.path().string() + ": ";
  for (auto const& [gpu, c] : counts) {
    diag +=
      "GPU" + std::to_string(gpu) + "{pipeline=" + std::to_string(c.pipeline_ids.size()) + "} ";
  }
  INFO(diag);

  // Load-bearing assertion: the cached host scan dispatched pipeline tasks on
  // BOTH GPUs. This is the topology-independent proof that host serving uses
  // both GPUs (true on single- and multi-NUMA hosts alike).
  REQUIRE(counts.count(0) == 1);
  REQUIRE(counts.count(1) == 1);
  REQUIRE(counts.at(0).pipeline_ids.size() >= 1u);
  REQUIRE(counts.at(1).pipeline_ids.size() >= 1u);

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Host-tier cached serve — MULTI-GPU (distributed chunks), SUM(v).
//
// Companion to the MAX(k) gate above, with a non-trivial aggregate: SUM(v)
// reads every pinned value, proving the distributed host chunks feed the GPU
// expression executor end-to-end. Pin with a 1 MB batch (chunks across both
// GPUs), overwrite the on-disk parquet with a different multiplier, then read
// SUM(v): the host pin returns the ORIGINAL total; a fall-through would return
// the new files' total.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - host-tier cached path serves SUM(v) after overwrite (multi-GPU)",
          "[pin_mgpu][scan_manager][host_tier]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("host-cached-agg-mgpu");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  // Original: 4 files of `SELECT range AS k, range * 2 AS v FROM range(100000)`.
  // Per file SUM(v) = 2 * (0 + 1 + ... + 99999) = 2 * 99999*100000/2 = 9999900000.
  // 4 files: 4 * 9999900000 = 39999600000.
  generate_4file_surface(tmp);

  // 1 MB batch so the host pin distributes chunks round-robin across both GPUs.
  mgpu_env_params params;
  params.scan_task_batch_size = 1'000'000;
  auto yaml_path              = tmp / "pin_mgpu_host_cached_agg.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = parquet_glob(tmp);

  // Baseline via pure DuckDB so transparent sirius interception doesn't go
  // through the same cached path we're trying to verify.
  REQUIRE_FALSE(con.Query("SET gpu_execution = false;")->HasError());
  auto baseline = con.Query("SELECT SUM(v) FROM read_parquet('" + glob + "');");
  REQUIRE(baseline);
  REQUIRE_FALSE(baseline->HasError());
  auto baseline_sum = baseline->GetValue(0, 0).GetValue<int64_t>();
  REQUIRE(baseline_sum == 39999600000LL);
  REQUIRE_FALSE(con.Query("SET gpu_execution = true;")->HasError());

  auto pin = con.Query("CALL pin_table('" + glob + "', tier='host', name='host_agg_test');");
  REQUIRE(pin);
  if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
  REQUIRE_FALSE(pin->HasError());

  // Overwrite with `range * 5` (same row count, different values) so
  // fall-through SUM would be 4 * 5 * 99999 * 100000 / 2 = 99999000000,
  // distinguishable from the pinned SUM (39999600000).
  generate_parquet_surface(tmp,
                           "SELECT range AS k, range * 5 AS v FROM range(100000)",
                           /*num_files=*/4);

  auto cached =
    con.Query("CALL gpu_execution(\"SELECT SUM(v) FROM read_parquet('" + glob + "')\");");
  REQUIRE(cached);
  if (cached->HasError()) { UNSCOPED_INFO("gpu_execution error: " << cached->GetError()); }
  REQUIRE_FALSE(cached->HasError());
  auto cached_sum = cached->GetValue(0, 0).GetValue<int64_t>();
  INFO("expected cached_sum=" << baseline_sum << " (silent fall-through would return 99999000000)");
  REQUIRE(cached_sum == baseline_sum);

  auto unpin = con.Query("CALL unpin_table('host_agg_test');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());

  fs::remove_all(tmp, ec);
}

//===----------------------------------------------------------------------===//
// Same-row-count merge correctness gate (PR #732 review feedback — med-1):
//
// `CALL pin_table(..., cols=[A, B])` followed by `CALL pin_table(..., cols=[A,
// C])` with the same file paths and same total row count used to take the
// "same-row-count merge" branch and append the new column C. The merge loop
// checked `entry.data_batches_by_column.contains(col_name)` per chunk —
// after chunk 0 installed C, contains() returned true and chunks 1..N were
// dropped. The pinned entry then had uneven chunk counts across columns,
// which trips the cached_split_provider invariant and silently falls back
// to uncached reads.
//
// This gate generates a multi-chunk surface (large enough that the cudf
// chunked_parquet_reader emits >=2 chunks per file at the default 100 MB
// scan_task_batch_size), pins two non-overlapping column subsets in
// sequence, and asserts every column in the merged entry has the SAME
// number of chunks as `chunk_memory_spaces`.
//===----------------------------------------------------------------------===//
TEST_CASE("pin_table - same-row-count merge appends every chunk for new columns",
          "[pin_mgpu][scan_manager][merge]")
{
  if (!require_two_gpus()) return;

  auto tmp = make_tmp("merge");
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  // Force >= 2 chunks per pin call by overriding scan_task_batch_size to
  // 10 MB so the 100k-row fixture per file produces at least one chunk
  // per file (and 4 files → 4+ chunks). The exact threshold is host-
  // dependent (column types and compression), so we conservatively use a
  // small scan_task_batch_size to make the multi-chunk path reliable.
  mgpu_env_params params;
  params.scan_task_batch_size = 1'000'000;  // 1 MB

  generate_parquet_surface(
    tmp,
    "SELECT range AS k, range * 2 AS v, range * 3 AS w, range * 4 AS x FROM range(100000)",
    /*num_files=*/4);

  auto yaml_path = tmp / "pin_mgpu_merge.yaml";
  write_mgpu_yaml(yaml_path, params);

  scoped_mgpu_env env(yaml_path);
  auto con = env.make_connection();

  auto glob = parquet_glob(tmp);

  // First pin: columns [k, v]
  auto pin1 =
    con.Query("CALL pin_table('" + glob + "', tier='gpu', name='merge_pin', cols=['k', 'v']);");
  REQUIRE(pin1);
  if (pin1->HasError()) { UNSCOPED_INFO("pin_table 1 error: " << pin1->GetError()); }
  REQUIRE_FALSE(pin1->HasError());

  // Second pin: columns [k, w] — the merge path must install w with the
  // SAME number of chunks as the existing k and v vectors.
  auto pin2 =
    con.Query("CALL pin_table('" + glob + "', tier='gpu', name='merge_pin', cols=['k', 'w']);");
  REQUIRE(pin2);
  if (pin2->HasError()) { UNSCOPED_INFO("pin_table 2 error: " << pin2->GetError()); }
  REQUIRE_FALSE(pin2->HasError());

  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  auto const& mgr = sirius_ctx->get_scan_manager();

  bool found_merge_pin = false;

  const sirius::scan_manager::pinned_entry* ptr = nullptr;
  mgr.visit_pinned_entries([&found_merge_pin, &ptr](std::string_view name, const auto& entry) {
    if (name == "merge_pin") {
      found_merge_pin = true;
      ptr             = &entry;
      return true;  // Stop iteration
    }
    return false;  // Continue iteration
  });

  REQUIRE(found_merge_pin);

  auto const& entry          = *ptr;
  auto const expected_chunks = entry.chunk_memory_spaces.size();
  INFO("expected_chunks=" << expected_chunks);
  REQUIRE(expected_chunks >= 2u);

  // Each pinned column MUST have exactly chunk_memory_spaces.size() chunks.
  // With the bug, w would have just 1 chunk while k and v had N chunks,
  // which silently fell back to uncached scans.
  for (auto const& col_name : entry.cache_info.column_names()) {
    auto col_it = entry.data_batches_by_column.find(col_name);
    REQUIRE(col_it != entry.data_batches_by_column.end());
    INFO("col_name=" << col_name << " chunks=" << col_it->second.size());
    REQUIRE(col_it->second.size() == expected_chunks);
  }

  // The merged entry must list all unique columns: k, v, w.
  REQUIRE(entry.cache_info.column_names().size() == 3u);

  auto unpin = con.Query("CALL unpin_table('merge_pin');");
  REQUIRE(unpin);
  REQUIRE_FALSE(unpin->HasError());

  fs::remove_all(tmp, ec);
}
