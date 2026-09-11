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

// Shared helpers for per-operator multi-GPU integration tests.
//
// The per-operator unit tests under test/cpp/operator/ construct operators
// directly and call execute() in a single-GPU process — they can't reach the
// task_creator routing layer where the SCHED-00/01/02 pinning decisions
// happen, so they miss cross-GPU correctness bugs.
//
// The helpers here spin up a real 2-GPU SiriusContext scoped to a single
// TEST_CASE so each operator gets a mini TPC-H-like integration test. Call
// require_two_gpus() first, then construct scoped_mgpu_env with a generated
// yaml, then issue queries through the returned connection. The [mgpu-audit]
// and [mgpu-probe] breadcrumbs land in SIRIUS_LOG_DIR — parse_audit_log()
// returns a per-GPU scan/pipeline count so tests can assert distribution.

#pragma once

#include "pipeline/task_scheduler.hpp"
#include "sirius_context.hpp"
#include "utils/sirius_test_env.hpp"

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace sirius::test::mgpu {

/**
 * @brief Config knobs for the generated MGPU yaml. Defaults match the
 * integration-2gpu.yaml except `cache` which flips to `table_gpu` — most
 * operator MGPU tests run a cold iteration and care about distribution,
 * not caching, so they set cache=none explicitly.
 */
struct mgpu_env_params {
  int num_gpus                        = 2;
  std::string cache                   = "none";  // none | table_gpu | table_host | parquet
  uint64_t scan_task_batch_size       = 100'000'000;
  uint64_t hash_partition_bytes       = 10'000'000;
  uint64_t concat_batch_bytes         = 100'000'000;
  uint64_t max_build_hash_table_bytes = 90'000'000;
  double usage_limit_fraction         = 0.4;
  // Absolute per-GPU usage cap in bytes. When non-zero this is emitted as
  // `usage_limit_bytes` (mutually exclusive with usage_limit_fraction per
  // sirius_config.cpp:201) so a test can impose a small fixed budget that
  // forces OOM/reschedule regardless of the host GPU's physical capacity —
  // essential on big-memory parts (e.g. GB200 ~186 GB) where a fraction
  // would never trigger pressure at unit-test data scales.
  uint64_t usage_limit_bytes   = 0;
  int pipeline_num_threads     = 4;
  int task_creator_num_threads = 4;
  int duckdb_scan_num_threads  = 2;
};

/**
 * @brief Emit a Sirius yaml at @p yaml_path with the given params.
 */
inline void write_mgpu_yaml(std::filesystem::path const& yaml_path,
                            mgpu_env_params const& params = {})
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: "
    << params.num_gpus
    << "\n"
       "  memory:\n"
       "    gpu:\n";
  if (params.usage_limit_bytes != 0) {
    f << "      usage_limit_bytes: " << params.usage_limit_bytes << "\n";
  } else {
    f << "      usage_limit_fraction: " << params.usage_limit_fraction << "\n";
  }
  f << "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    scan_manager:\n"
       "      backend: sirius\n"
       "    pipeline:\n"
       "      num_threads: "
    << params.pipeline_num_threads
    << "\n"
       "    task_creator:\n"
       "      num_threads: "
    << params.task_creator_num_threads
    << "\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: "
    << params.scan_task_batch_size
    << "\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: "
    << params.hash_partition_bytes
    << "\n"
       "    concat_batch_bytes: "
    << params.concat_batch_bytes
    << "\n"
       "    max_build_hash_table_bytes: "
    << params.max_build_hash_table_bytes << "\n";
}

/**
 * @brief Skip the rest of the TEST_CASE if fewer than 2 GPUs are visible.
 * Matches the Catch2 v2 WARN+return convention used by the other MGPU tests.
 *
 * @return true if the host has >=2 GPUs; the caller MUST still `return;`
 *         when this returns false.
 */
inline bool require_two_gpus()
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("MGPU operator test requires >=2 GPUs; single-GPU host — skipping");
    return false;
  }
  return true;
}

/**
 * @brief RAII wrapper that pauses any active shared test env (so our local
 * env has the Sirius extension lock to itself) and constructs a local
 * shared_test_env backed by @p yaml_path. On destruction the local env
 * is released, restoring SIRIUS_CONFIG_FILE; the Catch2 listener will
 * re-resume shared envs on the next TEST_CASE as usual.
 */
class scoped_mgpu_env {
 public:
  explicit scoped_mgpu_env(std::filesystem::path const& yaml_path)
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
    _env = std::make_unique<sirius::test::shared_test_env>(yaml_path);
  }

  scoped_mgpu_env(scoped_mgpu_env const&)            = delete;
  scoped_mgpu_env& operator=(scoped_mgpu_env const&) = delete;

  duckdb::Connection make_connection() { return _env->make_connection(); }

  /**
   * @brief Test-only accessor for the underlying task_scheduler.
   *
   * Returns a non-owning reference to the task_scheduler instance owned
   * by the SiriusContext that this fixture wraps. The returned reference
   * is valid for the lifetime of the scoped_mgpu_env instance.
   *
   * The connection argument is the route into the `registered_state`
   * map where the SiriusContext is registered under "sirius_state";
   * `scoped_mgpu_env` itself does not own a connection so the caller
   * passes one (typically the one already created via
   * `make_connection()`).
   */
  sirius::pipeline::task_scheduler& get_task_scheduler(duckdb::Connection& con)
  {
    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    return sirius_ctx->get_task_scheduler();
  }

 private:
  std::unique_ptr<sirius::test::shared_test_env> _env;
};

/**
 * @brief Make @p num_files parquet files at @p dir using a vanilla DuckDB
 * instance (no Sirius). @p create_select_sql is the body of a `CREATE TABLE t
 * AS <create_select_sql>;` statement so each file gets the same contents
 * (tests only care about stable schema + enough rows to force multi-batch
 * scan). Returns the tmp directory path; caller constructs globs with
 * parquet_glob().
 */
inline void generate_parquet_surface(std::filesystem::path const& dir,
                                     std::string const& create_select_sql,
                                     int num_files)
{
  std::filesystem::create_directories(dir);
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto create = gen.Query("CREATE TABLE t AS " + create_select_sql + ";");
    REQUIRE(create);
    REQUIRE_FALSE(create->HasError());
    for (int i = 0; i < num_files; ++i) {
      auto copy =
        gen.Query("COPY t TO '" + (dir / ("p" + std::to_string(i) + ".parquet")).string() +
                  "' (FORMAT PARQUET);");
      REQUIRE(copy);
      REQUIRE_FALSE(copy->HasError());
    }
  }
  unsetenv("SIRIUS_DISABLE");
}

/**
 * @brief Return a glob matching every parquet file under @p dir, for use
 * inside a read_parquet(...) call.
 */
inline std::string parquet_glob(std::filesystem::path const& dir)
{
  return (dir / "*.parquet").string();
}

/**
 * @brief Per-GPU audit counts populated by parse_audit_log().
 *
 * task_id / batch_id are tracked as std::set<std::string> rather than
 * std::set<int> so retry-derived duplicates collapse and the threshold
 * assertion measures UNIQUE work.
 */
struct audit_counts {
  std::set<std::string> pipeline_ids;
  std::set<std::string> scan_ids;
};

/**
 * @brief Walk @p log_dir and return the per-GPU audit counts observed in
 * [mgpu-audit] log lines. Mirrors the parse_audit_log() in
 * test_gpu_execution_tpch_mgpu_audit.cpp verbatim so tests share one
 * grep-stable contract with pipeline_executor.cpp:255 and
 * duckdb_scan_executor.cpp:216.
 */
inline std::map<int, audit_counts> parse_audit_log(std::filesystem::path const& log_dir)
{
  std::map<int, audit_counts> by_gpu;
  std::regex pipeline_re(R"(\[mgpu-audit\] pipeline_task dispatched to GPU (\d+) task_id=(\S+))");
  std::regex scan_re(R"(\[mgpu-audit\] scan_batch assigned to GPU (\d+) batch_id=(\S+))");

  if (!std::filesystem::exists(log_dir)) return by_gpu;
  for (auto const& entry : std::filesystem::directory_iterator(log_dir)) {
    if (!entry.is_regular_file()) continue;
    std::ifstream f(entry.path());
    std::string line;
    while (std::getline(f, line)) {
      std::smatch m;
      if (std::regex_search(line, m, pipeline_re)) {
        by_gpu[std::stoi(m[1])].pipeline_ids.insert(m[2]);
      } else if (std::regex_search(line, m, scan_re)) {
        by_gpu[std::stoi(m[1])].scan_ids.insert(m[2]);
      }
    }
  }
  return by_gpu;
}

/**
 * @brief RAII wrapper around SIRIUS_LOG_DIR / SIRIUS_LOG_LEVEL env vars so
 * each MGPU TEST_CASE routes Sirius logs into a unique tmp dir that
 * parse_audit_log() can scan after the test. The previous values (if any)
 * are restored on destruction.
 *
 * Construct BEFORE scoped_mgpu_env — the SiriusContext reads these env
 * vars in its constructor (src/sirius_context.cpp:569-571), so they must
 * be in place when the extension callback fires.
 */
class scoped_log_dir {
 public:
  explicit scoped_log_dir(std::filesystem::path const& log_dir, std::string const& level = "info")
    : _log_dir(log_dir)
  {
    std::filesystem::create_directories(log_dir);
    if (char const* prev = std::getenv("SIRIUS_LOG_DIR")) {
      _prev_dir     = prev;
      _had_prev_dir = true;
    }
    if (char const* prev = std::getenv("SIRIUS_LOG_LEVEL")) {
      _prev_level     = prev;
      _had_prev_level = true;
    }
    setenv("SIRIUS_LOG_DIR", log_dir.c_str(), 1);
    setenv("SIRIUS_LOG_LEVEL", level.c_str(), 1);
  }

  ~scoped_log_dir()
  {
    if (_had_prev_dir) {
      setenv("SIRIUS_LOG_DIR", _prev_dir.c_str(), 1);
    } else {
      unsetenv("SIRIUS_LOG_DIR");
    }
    if (_had_prev_level) {
      setenv("SIRIUS_LOG_LEVEL", _prev_level.c_str(), 1);
    } else {
      unsetenv("SIRIUS_LOG_LEVEL");
    }
  }

  scoped_log_dir(scoped_log_dir const&)            = delete;
  scoped_log_dir& operator=(scoped_log_dir const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const { return _log_dir; }

 private:
  std::filesystem::path _log_dir;
  std::string _prev_dir;
  std::string _prev_level;
  bool _had_prev_dir{false};
  bool _had_prev_level{false};
};

/**
 * @brief Compare @p inner_query through `gpu_execution` with a reference execution.
 *
 * The reference may be transparently intercepted by Sirius unless
 * @p force_cpu_reference is true. Disables fallback so GPU errors are not
 * silently hidden.
 *
 * The comparison mirrors the one in test_gpu_execution_tpch.cpp's
 * compare_gpu_vs_cpu but stripped down to what the operator MGPU tests
 * need: same row count, same string-ToString on every value after a
 * normalized ORDER BY.
 */
inline void require_gpu_matches_cpu(duckdb::Connection& con,
                                    std::string const& inner_query,
                                    bool force_cpu_reference = false)
{
  auto fb = con.Query("SET enable_duckdb_fallback = false;");
  REQUIRE(fb);
  REQUIRE_FALSE(fb->HasError());

  auto clean_query = inner_query;
  while (!clean_query.empty() && (clean_query.back() == ';' || clean_query.back() == ' ')) {
    clean_query.pop_back();
  }

  auto gpu_result = con.Query("CALL gpu_execution(\"" + clean_query + "\")");
  REQUIRE(gpu_result);
  if (gpu_result->HasError()) { UNSCOPED_INFO("gpu_execution error: " << gpu_result->GetError()); }
  REQUIRE_FALSE(gpu_result->HasError());

  // The reference run. By default the plain SELECT is transparently intercepted
  // and run on the GPU again (a GPU-vs-GPU consistency check). When
  // force_cpu_reference is set, disable interception around it so it executes on
  // DuckDB CPU — a true correctness oracle, and immune to GPU memory pressure
  // (so a tight usage_limit_bytes can't turn the reference into a false
  // failure). Mirrors compare_gpu_vs_cpu's SET gpu_execution=false pattern.
  if (force_cpu_reference) {
    auto off = con.Query("SET gpu_execution=false;");
    REQUIRE(off);
    REQUIRE_FALSE(off->HasError());
  }
  auto cpu_result = con.Query(clean_query + ";");
  if (force_cpu_reference) {
    auto on = con.Query("SET gpu_execution=true;");
    REQUIRE(on);
    REQUIRE_FALSE(on->HasError());
  }
  REQUIRE(cpu_result);
  REQUIRE_FALSE(cpu_result->HasError());

  REQUIRE(gpu_result->ColumnCount() == cpu_result->ColumnCount());
  REQUIRE(gpu_result->RowCount() == cpu_result->RowCount());

  // Materialize both result sets into row vectors of stringified cells, sort
  // lexicographically, and compare. We do the sort client-side because sirius
  // registers `gpu_execution` as a CALL/pragma function only — not as a scan
  // function — so `SELECT * FROM gpu_execution(...) ORDER BY ...` is rejected
  // by DuckDB ("Unsupported scan function: gpu_execution") before any GPU code
  // runs. Sorting in C++ also avoids re-executing the GPU query just to sort.
  auto const ncols = gpu_result->ColumnCount();
  auto const nrows = gpu_result->RowCount();
  std::vector<std::vector<std::string>> gpu_rows;
  std::vector<std::vector<std::string>> cpu_rows;
  gpu_rows.reserve(nrows);
  cpu_rows.reserve(nrows);
  for (duckdb::idx_t r = 0; r < nrows; ++r) {
    std::vector<std::string> g_row;
    std::vector<std::string> c_row;
    g_row.reserve(ncols);
    c_row.reserve(ncols);
    for (duckdb::idx_t c = 0; c < ncols; ++c) {
      g_row.push_back(gpu_result->GetValue(c, r).ToString());
      c_row.push_back(cpu_result->GetValue(c, r).ToString());
    }
    gpu_rows.push_back(std::move(g_row));
    cpu_rows.push_back(std::move(c_row));
  }
  std::sort(gpu_rows.begin(), gpu_rows.end());
  std::sort(cpu_rows.begin(), cpu_rows.end());

  for (duckdb::idx_t r = 0; r < nrows; ++r) {
    for (duckdb::idx_t c = 0; c < ncols; ++c) {
      auto const& gpu_str = gpu_rows[r][c];
      auto const& cpu_str = cpu_rows[r][c];
      if (gpu_str != cpu_str) {
        UNSCOPED_INFO("Row " << r << " Col " << c << " mismatch: GPU=[" << gpu_str << "] CPU=["
                             << cpu_str << "]");
      }
      REQUIRE(gpu_str == cpu_str);
    }
  }
}

}  // namespace sirius::test::mgpu
