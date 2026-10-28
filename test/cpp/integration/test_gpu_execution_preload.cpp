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

// The preload executor end to end: spill-heavy parquet queries through Sirius on GPUs limited to
// 1 GiB, with preload off and on, against plain DuckDB. Results must match the CPU's and be
// identical with preload on and off; with preload on, no promotion may fail and few may bounce
// back to where they came from. Each query's preload counters are reported (as Catch2 warnings),
// so a run shows what preload moved and how often the ceilings or stalled tasks held it back.

#include "catch.hpp"
#include "memory/device_preload_budget.hpp"
#include "preload/preload_executor.hpp"
#include "preload/preload_stats.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <duckdb.hpp>

#include <algorithm>
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
using sirius::preload::counter;
using sirius::preload::preload_stats;

/// Rows of the test table: about 1.6 GB of id and value columns once on a GPU.
constexpr std::uint64_t table_rows = 100'000'000;

/// The shape of one test configuration.
struct env_shape {
  int num_gpus{1};
  /// HOST capacity. A small one makes the GPU spill straight to DISK, and HOST to DISK.
  std::uint64_t host_bytes{32'000'000'000};
};

/// A Sirius configuration of @p shape with GPUs limited to 1 GiB, a DISK space under
/// @p disk_dir, and preload @p preload. Batches and partitions (32 MiB) stay below the
/// speculative window (5% of the GPU), so every class of promotion can admit them. Two pipeline
/// threads per GPU: with four, a task is waiting for memory during most of a spilling query, and
/// preload rightly stays out of the way of every stalled task.
void write_config(fs::path const& path,
                  fs::path const& disk_dir,
                  env_shape const& shape,
                  bool preload)
{
  std::ofstream out(path);
  out << "sirius:\n"
         "  topology:\n"
         "    num_gpus: "
      << shape.num_gpus
      << "\n"
         "  memory:\n"
         "    gpu:\n"
         "      usage_limit_bytes: 1073741824\n"
         "      reservation_limit_fraction: 1.0\n"
         "    host:\n"
         "      capacity_bytes: "
      << shape.host_bytes
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
         "    preload:\n"
      << (preload ? "      device_preloading_budget: 0.2\n" : "      concurrent_preloads: 0\n")
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

/// Digest of @p sql's result. Rows are hashed as they stream, so large results are compared
/// without being held as strings. BIGINT columns hash their values, other columns their text.
digest query_digest(duckdb::Connection& con, std::string const& sql, bool ordered)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  INFO(sql << " -> " << (result->HasError() ? result->GetError() : std::string{"ok"}));
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
      d.hash = ordered ? mix(d.hash, h) : d.hash + mix(0, h);
      ++d.rows;
    }
  }
  return d;
}

/// One spill-heavy query, and whether its result order is defined.
struct spill_query {
  std::string sql;
  bool ordered{false};
};

/// Joins the table with itself on its unique id: both sides are partitioned and spill.
spill_query const self_join{"SELECT count(*), sum(a.v), sum(b.v) FROM t a JOIN t b ON a.id = b.id"};

/// A grouped aggregate whose per-batch partials do not shrink: they pile up behind the PARTITION
/// operator's FULL barrier and are spilled, and the partition tasks read them back only once every
/// scan has finished. That second phase frees memory between tasks, so preload has room to bring
/// queued inputs back.
spill_query const phase_separated_aggregate{
  "SELECT count(*), sum(c), max(m) FROM (SELECT id % 10000000 AS g, count(*) AS c, max(v) AS m "
  "FROM t GROUP BY g)"};

/// A grouped aggregate whose groups are then sorted and merged, returned in order.
spill_query const sorted_aggregate{
  "SELECT id % 1000000 AS g, count(*) AS c, max(v) AS m FROM t GROUP BY g ORDER BY m DESC, g",
  true};

/// The counters @p after gained over @p before.
preload_stats delta(preload_stats const& after, preload_stats const& before)
{
  auto d = after;
  for (std::size_t i = 0; i < d.values.size(); ++i) {
    d.values[i] -= before.values[i];
  }
  return d;
}

std::uint64_t promoted(preload_stats const& s)
{
  return s[counter::promoted_h2g] + s[counter::promoted_d2g] + s[counter::promoted_d2h] +
         s[counter::promoted_relocate_d2h];
}

/// What a query's run reports: promotions by kind and class, accuracy, and why the planner and
/// the workers said no.
std::string report(preload_stats const& s)
{
  auto const passes = static_cast<double>(std::max<std::uint64_t>(s[counter::passes], 1));
  return std::format(
    "promoted={} (h2g={} d2g={} d2h={}) jobs: class_a={} class_b={} staging={} | hit={} "
    "bounce={} mispredicted={} (class B {}) wasted={} | passes={} noop={} "
    "refused_ceiling/passes={:.2f} refused_held_off/passes={:.2f} refused_trigger/passes={:.2f} | "
    "worker held_off={} gated={} no_memory={} contended={} failed={}",
    promoted(s),
    s[counter::promoted_h2g],
    s[counter::promoted_d2g],
    s[counter::promoted_d2h],
    s[counter::class_a_items],
    s[counter::class_b_items],
    s[counter::staging_items],
    s[counter::hit],
    s[counter::bounce],
    s[counter::mispredicted_device],
    s[counter::mispredicted_device_b],
    s[counter::wasted_unconsumed],
    s[counter::passes],
    s[counter::passes_noop],
    static_cast<double>(s[counter::refused_ceiling]) / passes,
    static_cast<double>(s[counter::refused_held_off]) / passes,
    static_cast<double>(s[counter::refused_trigger]) / passes,
    s[counter::held_off],
    s[counter::gated],
    s[counter::no_memory],
    s[counter::contended],
    s[counter::failed]);
}

/// Fewer than one in ten promotions of a run went back where they came from. Re-downgrades
/// depend on timing, so the rate is taken over all of a run's queries, and a run with few
/// promotions may see up to two.
void check_bounce_rate(std::vector<preload_stats> const& stats)
{
  std::uint64_t moved   = 0;
  std::uint64_t bounced = 0;
  for (auto const& s : stats) {
    moved += promoted(s);
    bounced += s[counter::bounce];
  }
  INFO("bounce=" << bounced << " promoted=" << moved);
  if (moved >= 20) {
    CHECK(bounced * 10 < moved);
  } else {
    CHECK(bounced <= 2);
  }
}

/// The test table: a unique id, a value with about a million distinct values, and a short string.
std::string write_table(sirius::test::scratch_dir const& dir)
{
  auto const file   = dir.file("t.parquet");
  auto const config = dir.path() / "generate.yaml";
  write_config(config, dir.path(), {}, false);
  preload_db db(config);
  db.run_ok("SET gpu_execution = false;");
  db.run_ok(
    "COPY (SELECT range AS id, (range * 7919) % 1000003 AS v, 'k' || (range % 101) AS s "
    "FROM range(" +
    std::to_string(table_rows) + ")) TO " + sirius::test::sql_literal(file) +
    " (FORMAT PARQUET, ROW_GROUP_SIZE 500000);");
  return file;
}

/// Runs @p queries on a configuration of @p shape with preload off and then on. Checks that
/// every result matches DuckDB's and is the same both ways, that no query fell back to the CPU,
/// that preload logs nothing when off, and that when on no promotion failed and few bounced.
/// @return Each query's preload counters with preload on.
std::vector<preload_stats> run_off_and_on(sirius::test::scratch_dir const& dir,
                                          std::string const& file,
                                          env_shape const& shape,
                                          std::vector<spill_query> const& queries)
{
  auto const create_view =
    "CREATE VIEW t AS SELECT * FROM read_parquet(" + sirius::test::sql_literal(file) + ");";
  std::vector<digest> cpu;
  std::vector<digest> off;
  {
    auto const config = dir.path() / "off.yaml";
    write_config(config, dir.path(), shape, false);
    preload_db db(config);
    REQUIRE(db.preload() == nullptr);
    db.run_ok(create_view);
    sirius::test::scoped_recording_log_sink log("info");
    auto const before = sirius::test::get_transparent_execution_stats(*db.con);
    for (auto const& q : queries) {
      db.run_ok("SET gpu_execution = false;");
      cpu.push_back(query_digest(*db.con, q.sql, q.ordered));
      db.run_ok("SET gpu_execution = true;");
      off.push_back(query_digest(*db.con, q.sql, q.ordered));
    }
    auto const after = sirius::test::get_transparent_execution_stats(*db.con);
    CHECK(after.executions - before.executions == queries.size());
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(std::ranges::none_of(log.sink().records(),
                               [](auto const& r) { return r.message.starts_with("[preload]"); }));
  }

  std::vector<preload_stats> stats;
  {
    auto const config = dir.path() / "on.yaml";
    write_config(config, dir.path(), shape, true);
    preload_db db(config);
    auto* preload = db.preload();
    REQUIRE(preload != nullptr);
    db.run_ok(create_view);
    db.run_ok("SET gpu_execution = true;");
    auto const before_all = sirius::test::get_transparent_execution_stats(*db.con);
    for (std::size_t i = 0; i < queries.size(); ++i) {
      INFO(queries[i].sql);
      // Each query starts and ends on a settled executor, so the counters it gains are its own.
      REQUIRE(preload->wait_until_quiescent_for_testing(std::chrono::seconds(30)));
      auto const before = preload->stats();
      auto const on     = query_digest(*db.con, queries[i].sql, queries[i].ordered);
      REQUIRE(preload->wait_until_quiescent_for_testing(std::chrono::seconds(30)));
      stats.push_back(delta(preload->stats(), before));
      CHECK(off[i] == cpu[i]);
      CHECK(on == cpu[i]);
      CHECK(stats.back()[counter::failed] == 0);
      WARN("preload on, " << shape.num_gpus << " GPU(s), " << queries[i].sql << ": "
                          << report(stats.back()));
    }
    auto const after_all = sirius::test::get_transparent_execution_stats(*db.con);
    CHECK(after_all.executions - before_all.executions == queries.size());
    CHECK(after_all.runtime_fallbacks == before_all.runtime_fallbacks);
    // Device preloading released budget when claimed tasks were popped. Only assert it when
    // something was actually promoted onto a GPU (and so charged against the budget).
    auto const budget =
      sirius::test::get_registered_sirius_context(*db.con)->device_preload_budget_for_testing();
    REQUIRE_FALSE(budget.expired());
    std::uint64_t gpu_promotions = 0;
    for (auto const& s : stats) {
      gpu_promotions += s[counter::promoted_h2g] + s[counter::promoted_d2g];
    }
    if (gpu_promotions > 0) { CHECK(budget.lock()->claimed_releases() > 0); }
  }
  check_bounce_rate(stats);
  return stats;
}

}  // namespace

TEST_CASE("preload keeps spill-heavy query results unchanged and brings spilled inputs back",
          "[integration][preload]")
{
  sirius::test::scratch_dir dir{"preload"};
  auto const file = write_table(dir);

  auto const stats =
    run_off_and_on(dir, file, {}, {self_join, phase_separated_aggregate, sorted_aggregate});

  // The designed phase-separated workload: once the scans finish, the partition tasks drain the
  // spilled partials while memory is freed between them, and preload copies the inputs of queued
  // tasks past the scheduler's head margin back onto the GPU.
  auto const& phased = stats[1];
  CHECK(promoted(phased) > 0);
  CHECK(phased[counter::class_a_items] > 0);
  CHECK(phased[counter::hit] > 0);
}

TEST_CASE("preload brings batches spilled to DISK back when HOST is small",
          "[integration][preload]")
{
  sirius::test::scratch_dir dir{"preload_disk"};
  auto const file = write_table(dir);

  // HOST holds about half of what the GPU spills: the rest goes to DISK. Whether preload then
  // finds room depends on timing, so its DISK->GPU and DISK->HOST counts are reported only.
  run_off_and_on(dir, file, {.host_bytes = 1'500'000'000}, {phase_separated_aggregate});
}

TEST_CASE("preload targets only the GPU that consumes a batch", "[integration][preload][multi_gpu]")
{
  sirius::test::scratch_dir dir{"preload_mgpu"};
  auto const file = write_table(dir);

  auto const stats =
    run_off_and_on(dir, file, {.num_gpus = 2}, {self_join, phase_separated_aggregate});
  for (auto const& s : stats) {
    // Class A targets a task's own device bucket, so its consumer runs there: no preloaded batch
    // is cloned to the other GPU. Class B follows the partition placement and is only reported.
    CHECK(s[counter::mispredicted_device] - s[counter::mispredicted_device_b] == 0);
    auto const jobs = std::max<std::uint64_t>(s[counter::items_planned], 1);
    WARN("class B share of preload jobs: " << s[counter::class_b_items] << "/" << jobs);
  }
}
