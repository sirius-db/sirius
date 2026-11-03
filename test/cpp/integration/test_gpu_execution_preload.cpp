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

// The preload executor end to end: spill-heavy parquet queries through Sirius on a GPU limited to
// 1 GiB, with preload off and on, against plain DuckDB. Results must match the CPU's with preload
// on and off, and with preload on, the engine's spills and pops must reach preload and preload
// must promote.

#include "catch.hpp"
#include "preload/preload_executor.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <duckdb.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using sirius::preload::preload_counters;

/// Rows of the test table: about 1.6 GB of id and value columns once on a GPU.
constexpr std::uint64_t table_rows = 100'000'000;

/// HOST capacity. A small one makes the GPU spill straight to DISK, and HOST to DISK.
constexpr std::uint64_t large_host = 32'000'000'000;
constexpr std::uint64_t small_host = 1'500'000'000;

/// A one-GPU Sirius configuration with the GPU limited to 1 GiB, a DISK space under @p disk_dir
/// and, when @p preload, a device and a host preload budget. The device ceiling is the GPU's
/// downgrade stop fraction (0.6 by default): the highest that never starts a downgrade. Two
/// pipeline threads, so the queue-depth gate opens once at most two tasks are left for the GPU.
void write_config(fs::path const& path,
                  fs::path const& disk_dir,
                  std::uint64_t host_bytes,
                  bool preload)
{
  std::ofstream out(path);
  out << "sirius:\n"
         "  topology:\n"
         "    num_gpus: 1\n"
         "  memory:\n"
         "    gpu:\n"
         "      usage_limit_bytes: 1073741824\n"
         "      reservation_limit_fraction: 1.0\n"
         "    host:\n"
         "      capacity_bytes: "
      << host_bytes
      << "\n"
         "      initial_number_pools: 10\n"
         "      pool_size: 512\n"
         "      block_size: 1048576\n"
         "    disk:\n"
         "      disk_id: 0\n"
         "      capacity_bytes: 16000000000\n"
         "      downgrade_root_dirs: \""
      << disk_dir.string()
      << "\"\n"
         "  executor:\n"
         "    pipeline:\n"
         "      num_threads: 2\n"
         "    task_creator:\n"
         "      num_threads: 2\n"
         "    downgrade:\n"
         "      num_threads: 1\n"
         "      monitor_period: 10ms\n"
      << (preload ? "    preload:\n"
                    "      device_budget: 0.2\n"
                    "      device_ceiling: 0.6\n"
                    "      host_budget: 0.1\n"
                  : "")
      << "  operator_params:\n"
         "    scan_task_batch_size: 33554432\n"
         "    max_sort_partition_bytes: 67108864\n"
         "    hash_partition_bytes: 33554432\n"
         "    concat_batch_bytes: 33554432\n"
         "    max_build_hash_table_bytes: 33554432\n";
}

/// A Sirius-enabled DuckDB on its own configuration. One Sirius context runs per process at a
/// time, so the shared environments are paused first; the test listener resumes the integration
/// environment for the next test that needs it.
struct preload_db {
  std::unique_ptr<sirius::test::shared_test_env> env;
  std::unique_ptr<duckdb::Connection> con;

  explicit preload_db(fs::path const& config)
  {
    for (auto* shared : {sirius::test::g_shared_env,
                         sirius::test::g_integration_env,
                         sirius::test::g_integration_env_2gpu}) {
      if (shared != nullptr && shared->is_active()) { shared->pause(); }
    }
    env = std::make_unique<sirius::test::shared_test_env>(config);
    con = std::make_unique<duckdb::Connection>(env->make_connection());
  }
  ~preload_db()
  {
    con.reset();
    env.reset();
  }
  preload_db(preload_db const&)            = delete;
  preload_db& operator=(preload_db const&) = delete;

  void run_ok(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    INFO(sql << " -> " << (result->HasError() ? result->GetError() : std::string{"ok"}));
    REQUIRE_FALSE(result->HasError());
  }

  [[nodiscard]] sirius::preload::preload_executor* preload()
  {
    return sirius::test::get_registered_sirius_context(*con)->get_preload_executor();
  }
};

/// A result's row count and a hash of its rows, in order or as a multiset.
struct digest {
  std::uint64_t rows{0};
  std::uint64_t hash{0};
  bool operator==(digest const&) const = default;
};

std::uint64_t mix(std::uint64_t h, std::uint64_t x)
{
  return h ^ (x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

/// One spill-heavy query, and whether its result order is defined.
struct spill_query {
  std::string sql;
  bool ordered{false};
};

/// Digest of @p q's result, hashed as it streams. BIGINT columns hash their values, other
/// columns their text.
digest query_digest(duckdb::Connection& con, spill_query const& q)
{
  auto result = con.Query(q.sql);
  REQUIRE(result);
  INFO(q.sql << " -> " << (result->HasError() ? result->GetError() : std::string{"ok"}));
  REQUIRE_FALSE(result->HasError());
  digest d;
  while (auto chunk = result->Fetch()) {
    if (chunk->size() == 0) { break; }
    chunk->Flatten();
    for (duckdb::idx_t r = 0; r < chunk->size(); ++r) {
      std::uint64_t h = 0;
      for (duckdb::idx_t c = 0; c < chunk->ColumnCount(); ++c) {
        auto& column = chunk->data[c];
        if (duckdb::FlatVector::IsNull(column, r)) {
          h = mix(h, 0x5bd1e995ULL);
        } else if (column.GetType().id() == duckdb::LogicalTypeId::BIGINT) {
          h = mix(h,
                  static_cast<std::uint64_t>(duckdb::FlatVector::GetData<std::int64_t>(column)[r]));
        } else {
          h = mix(h, std::hash<std::string>{}(column.GetValue(r).ToString()));
        }
      }
      d.hash = q.ordered ? mix(d.hash, h) : d.hash + mix(0, h);
      ++d.rows;
    }
  }
  return d;
}

// Preload promotes only inputs of queued tasks that are off the GPU: the downgrade executor spilled
// them out of the queued task, or they were spilled from a repository before the task was queued
// with them. Both queries below spill under memory pressure while their tasks queue.

/// Joins the table with itself on its unique id: both sides are partitioned and spill, and the
/// join tasks queue with their inputs.
spill_query const self_join{"SELECT count(*), sum(a.v), sum(b.v) FROM t a JOIN t b ON a.id = b.id"};

/// A grouped aggregate whose groups are then sorted and merged, returned in order.
spill_query const sorted_aggregate{
  "SELECT id % 1000000 AS g, count(*) AS c, max(v) AS m FROM t GROUP BY g ORDER BY m DESC, g",
  true};

/// The test table: a unique id, a value with about a million distinct values, and a short string.
std::string write_table(sirius::test::scratch_dir const& dir)
{
  auto const file   = dir.file("t.parquet");
  auto const config = dir.path() / "generate.yaml";
  write_config(config, dir.path(), large_host, false);
  preload_db db(config);
  db.run_ok("SET gpu_execution = false;");
  db.run_ok(
    "COPY (SELECT range AS id, (range * 7919) % 1000003 AS v, 'k' || (range % 101) AS s "
    "FROM range(" +
    std::to_string(table_rows) + ")) TO " + sirius::test::sql_literal(file) +
    " (FORMAT PARQUET, ROW_GROUP_SIZE 500000);");
  return file;
}

std::string report(preload_counters const& c)
{
  return std::format(
    "pops={} to_gpu={} ({}B) to_host={} ({}B) gated={} no_candidate={} no_room={} "
    "not_needed={} failed={} stale_event={}",
    c.pops,
    c.promoted_to_gpu,
    c.bytes_to_gpu,
    c.promoted_to_host,
    c.bytes_to_host,
    c.gated,
    c.no_candidate,
    c.no_room,
    c.not_needed,
    c.failed,
    c.stale_event);
}

/// Runs @p queries with preload off and then on, on a configuration with @p host_bytes of HOST.
/// Checks that every result matches DuckDB's both ways, that no query fell back to the CPU, that
/// preload received both events from the engine, and that it promoted inputs.
void run_off_and_on(sirius::test::scratch_dir const& dir,
                    std::string const& file,
                    std::uint64_t host_bytes,
                    std::vector<spill_query> const& queries)
{
  auto const create_view =
    "CREATE VIEW t AS SELECT * FROM read_parquet(" + sirius::test::sql_literal(file) + ");";
  std::vector<digest> cpu;
  {
    auto const config = dir.path() / "off.yaml";
    write_config(config, dir.path(), host_bytes, false);
    preload_db db(config);
    REQUIRE(db.preload() == nullptr);
    db.run_ok(create_view);
    auto const before = sirius::test::get_transparent_execution_stats(*db.con);
    for (auto const& q : queries) {
      INFO(q.sql);
      db.run_ok("SET gpu_execution = false;");
      cpu.push_back(query_digest(*db.con, q));
      db.run_ok("SET gpu_execution = true;");
      CHECK(query_digest(*db.con, q) == cpu.back());
    }
    auto const after = sirius::test::get_transparent_execution_stats(*db.con);
    CHECK(after.executions - before.executions == queries.size());
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  }

  auto const config = dir.path() / "on.yaml";
  write_config(config, dir.path(), host_bytes, true);
  preload_db db(config);
  auto* preload = db.preload();
  REQUIRE(preload != nullptr);
  db.run_ok(create_view);
  db.run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*db.con);
  for (std::size_t i = 0; i < queries.size(); ++i) {
    INFO(queries[i].sql);
    CHECK(query_digest(*db.con, queries[i]) == cpu[i]);
  }
  auto const after = sirius::test::get_transparent_execution_stats(*db.con);
  CHECK(after.executions - before.executions == queries.size());
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  // Let the workers finish the credits already delivered; the counters are then at least what the
  // queries' pops caused so far.
  REQUIRE(preload->wait_idle_for_testing(std::chrono::seconds(30)));
  // A consistent snapshot: no event handled between the two reads, so the events handled are
  // exactly the counted pops plus the task_downgraded events.
  std::uint64_t handled = 0;
  preload_counters counters;
  do {
    handled  = preload->events_handled_for_testing();
    counters = preload->counters();
  } while (preload->events_handled_for_testing() != handled);
  INFO(report(counters) << " events_handled=" << handled);
  // The scheduler's pops reached preload, and so did the task_downgraded events of the downgrade
  // executor and the scheduler, which made candidates that some pops promoted: both queries queue
  // tasks with spilled inputs, many of them spilled from repositories before their task existed.
  CHECK(counters.pops > 0);
  CHECK(handled > counters.pops);
  CHECK(counters.promoted_to_gpu + counters.promoted_to_host > 0);
}

}  // namespace

TEST_CASE("preload keeps spill-heavy query results and sees the engine's spills and pops",
          "[integration][preload]")
{
  sirius::test::scratch_dir dir{"preload"};
  auto const file = write_table(dir);
  run_off_and_on(dir, file, large_host, {self_join, sorted_aggregate});
}

TEST_CASE("preload keeps the result when a small HOST sends spills to DISK",
          "[integration][preload]")
{
  sirius::test::scratch_dir dir{"preload_disk"};
  auto const file = write_table(dir);
  // HOST holds about half of what the GPU spills: the rest goes to DISK, from where preload may
  // stage it back to HOST.
  run_off_and_on(dir, file, small_host, {self_join});
}
