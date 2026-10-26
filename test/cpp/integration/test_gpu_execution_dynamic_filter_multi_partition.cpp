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

#include <absl/cleanup/cleanup.h>
#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::test::scoped_sirius_setting;

using result_rows = std::vector<std::vector<std::string>>;

struct observed_run {
  result_rows rows;
  sirius::op::dynamic_filter_stats_snapshot before;
  sirius::op::dynamic_filter_stats_snapshot after;
};

observed_run run_on_gpu(duckdb::Connection& con, std::string const& query)
{
  auto const before_execution = sirius::test::get_transparent_execution_stats(con);
  auto const before           = sirius::test::get_dynamic_filter_stats_snapshot(con);
  auto result                 = con.Query(query);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("GPU query failed: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  sirius::test::require_transparent_execution_delta(
    before_execution, sirius::test::get_transparent_execution_stats(con), 1, 0, 1);
  return {sirius::test::collect_rows(*result),
          before,
          sirius::test::get_dynamic_filter_stats_snapshot(con)};
}

void require_complete_publication(observed_run const& run, result_rows const& cpu_rows)
{
  REQUIRE(run.rows == cpu_rows);
  REQUIRE(run.after.accumulations_started == run.before.accumulations_started + 1);
  auto const expected =
    run.after.accumulation_expected_contributions - run.before.accumulation_expected_contributions;
  auto const completed = run.after.accumulation_completed_contributions -
                         run.before.accumulation_completed_contributions;
  REQUIRE(expected > 1);
  REQUIRE(completed == expected);
  REQUIRE(run.after.accumulation_publications_finished ==
          run.before.accumulation_publications_finished + 1);
  REQUIRE(run.after.accumulations_incomplete == run.before.accumulations_incomplete);
  REQUIRE(run.after.accumulations_skipped_admission == run.before.accumulations_skipped_admission);
  REQUIRE(run.after.accumulations_skipped_error == run.before.accumulations_skipped_error);
  REQUIRE(run.after.accumulations_skipped_transient == run.before.accumulations_skipped_transient);
  REQUIRE(run.after.accumulations_abandoned == run.before.accumulations_abandoned);
  REQUIRE(run.after.filters_pushed > run.before.filters_pushed);
  REQUIRE(run.after.publications_skipped_build_not_whole ==
          run.before.publications_skipped_build_not_whole);
}

/**
 * @brief Checks a run whose accumulation declined at start because a pair of its GPUs has no
 * working peer DMA: the results still match the CPU and nothing else ended the attempt.
 */
void require_peerless_decline(observed_run const& run, result_rows const& cpu_rows)
{
  REQUIRE(run.rows == cpu_rows);
  REQUIRE(run.after.accumulations_started == run.before.accumulations_started);
  REQUIRE(run.after.accumulations_skipped_admission ==
          run.before.accumulations_skipped_admission + 1);
  REQUIRE(run.after.accumulations_skipped_error == run.before.accumulations_skipped_error);
  REQUIRE(run.after.accumulation_publications_finished ==
          run.before.accumulation_publications_finished);
}

void run_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query failed: " << sql << ": " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

result_rows run_on_cpu(duckdb::Connection& con, std::string const& query)
{
  scoped_sirius_setting cpu_only(con, "gpu_execution", false);
  auto result = con.Query(query);
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  return sirius::test::collect_rows(*result);
}

/**
 * @brief Checks accumulation through `UNION ALL`: every arm of a build-side union feeds its own
 * port of the union operator, so the build PARTITION still has one finished source pipeline, and a
 * probe-side union exposes one scan target per arm.
 *
 * @param peer_dma Whether every pair of the session's GPUs has working peer DMA; without it the
 * accumulation declines by design and only the results and the decline are checked
 */
void check_union_all_accumulation(duckdb::Connection& con, bool peer_dma = true)
{
  auto const require_outcome = [peer_dma](observed_run const& run, result_rows const& cpu_rows) {
    if (peer_dma) {
      require_complete_publication(run, cpu_rows);
    } else {
      require_peerless_decline(run, cpu_rows);
    }
  };
  scoped_sirius_setting gpu_on(con, "gpu_execution", true);
  scoped_sirius_setting master_on(con, "enable_dynamic_filter", true);
  scoped_sirius_setting accumulation_on(con, "enable_dynamic_filter_multi_partition", true);
  scoped_sirius_setting native_keys(con, "enable_compressed_materialization", false);
  sirius::test::disabled_optimizers_guard shape(
    con, "statistics_propagation,join_order,build_side_probe_side");
  sirius::test::coverage_gate_disable_guard gate_off(con);
  sirius::test::scoped_setting no_broadcast(con, "max_broadcast_join_size", 1);
  sirius::test::scoped_setting small_partitions(con, "hash_partition_bytes", 1024 * 1024);
  sirius::test::scoped_setting small_build(con, "max_build_hash_table_bytes", 512 * 1024);
  sirius::test::scoped_setting small_batches(con, "scan_task_batch_size", 256 * 1024);
  {
    scoped_sirius_setting cpu_only(con, "gpu_execution", false);
    run_ok(con,
           "CREATE TABLE df_union_build AS SELECT (i * 17)::BIGINT AS k, (i * 31 + 5)::BIGINT AS "
           "payload, (i % 5)::INTEGER AS marker FROM range(262144) AS t(i);");
    run_ok(con,
           "CREATE TABLE df_union_probe AS SELECT (i * 17)::BIGINT AS k FROM range(524288) AS "
           "t(i);");
    run_ok(con, "CHECKPOINT;");
  }

  SECTION("a build side that is a UNION ALL of two scans accumulates once")
  {
    std::string const query =
      "SELECT count(*), sum(p.k), max(p.k), sum(b.payload) FROM df_union_probe p JOIN "
      "(SELECT k, payload, marker FROM df_union_build WHERE marker < 2 UNION ALL "
      "SELECT k, payload, marker FROM df_union_build WHERE marker >= 2) b "
      "ON p.k = b.k WHERE b.marker <> 0";
    auto const cpu_rows = run_on_cpu(con, query);
    require_outcome(run_on_gpu(con, query), cpu_rows);
  }
  SECTION("a probe side that is a UNION ALL of two scans receives the filter in both arms")
  {
    std::string const query =
      "SELECT count(*), sum(p.k), max(p.k), sum(b.payload) FROM "
      "(SELECT k FROM df_union_probe WHERE k % 2 = 0 UNION ALL "
      "SELECT k FROM df_union_probe WHERE k % 2 <> 0) p "
      "JOIN df_union_build b ON p.k = b.k WHERE b.marker <> 0";
    auto const cpu_rows = run_on_cpu(con, query);
    auto const run      = run_on_gpu(con, query);
    require_outcome(run, cpu_rows);
    if (peer_dma) { REQUIRE(run.after.filters_pushed - run.before.filters_pushed >= 2); }
  }
}

}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - multi-partition accumulation through UNION ALL matches the CPU",
                 "[integration][gpu_execution][dynamic_filter][multi_partition]")
{
  check_union_all_accumulation(*con);
}

TEST_CASE("gpu_execution - multi-partition accumulation through UNION ALL on two GPUs",
          "[integration][gpu_execution][dynamic_filter][multi_partition][mgpu][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  // The `[integration][multi_gpu]` tags make the `shared_env_listener` in `unittest.cpp` activate
  // this environment; resuming it again here would replace its live SiriusContext.
  auto* env = sirius::test::acquire_integration_env_for(2);
  REQUIRE(env != nullptr);
  REQUIRE(env->is_active());
  // Accumulation reduces the per-GPU arrays over peer DMA and declines to start without it.
  bool const peer_dma =
    cucascade::memory::probe_peer_dma_works(0, 1) && cucascade::memory::probe_peer_dma_works(1, 0);
  if (!peer_dma) {
    WARN("no working peer DMA between GPUs 0 and 1; checking only results and the decline");
  }

  auto const database = std::filesystem::temp_directory_path() /
                        ("sirius_df_union_mgpu_" + std::to_string(::getpid()) + ".db");
  absl::Cleanup remove_database = [&database] {
    std::error_code ignored;
    std::filesystem::remove(database, ignored);
    std::filesystem::remove(database.string() + ".wal", ignored);
  };
  {
    auto con = std::make_unique<duckdb::Connection>(env->make_connection());
    run_ok(*con, "ATTACH '" + database.string() + "' AS df_union_mgpu;");
    run_ok(*con, "USE df_union_mgpu;");
    check_union_all_accumulation(*con, peer_dma);
    con->Query("USE memory;");
    con->Query("DETACH df_union_mgpu;");
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "gpu_execution - complete multi-partition dynamic filters match the CPU",
                 "[integration][gpu_execution][dynamic_filter][multi_partition]")
{
  std::string const key_type = GENERATE(std::string{"INTEGER"}, std::string{"BIGINT"});
  CAPTURE(key_type);
  scoped_sirius_setting gpu_on(*con, "gpu_execution", true);
  scoped_sirius_setting master_on(*con, "enable_dynamic_filter", true);
  scoped_sirius_setting native_keys(*con, "enable_compressed_materialization", false);
  sirius::test::disabled_optimizers_guard shape(
    *con, "statistics_propagation,join_order,build_side_probe_side");
  sirius::test::coverage_gate_disable_guard gate_off(*con);
  sirius::test::scoped_setting no_broadcast(*con, "max_broadcast_join_size", 1);
  sirius::test::scoped_setting small_partitions(*con, "hash_partition_bytes", 1024 * 1024);
  sirius::test::scoped_setting small_build(*con, "max_build_hash_table_bytes", 512 * 1024);
  sirius::test::scoped_setting small_batches(*con, "scan_task_batch_size", 256 * 1024);
  sirius::test::scoped_setting bloom_cap(
    *con, "max_dynamic_filter_bloom_bytes_per_gpu", 256ULL * 1024 * 1024);

  // Disjoint matching keys expose a partial union if consumed; deterministic membership tests
  // cover completeness independently of whether this scan reaches its final checkpoint first.
  result_rows cpu_rows;
  std::string const query =
    "SELECT count(*), sum(p.k), max(p.k), sum(b.payload) "
    "FROM df_probe p JOIN df_build b ON p.k = b.k WHERE b.marker <> 0";
  {
    scoped_sirius_setting cpu_only(*con, "gpu_execution", false);
    run_ok("CREATE TABLE df_build AS SELECT (i * 17)::" + key_type +
           " AS k, (i * 31 + 5)::BIGINT AS payload, (i % 5)::INTEGER AS marker "
           "FROM range(262144) AS t(i);");
    run_ok("CREATE TABLE df_probe AS SELECT (i * 17)::" + key_type +
           " AS k FROM range(524288) AS t(i);");
    run_ok("CHECKPOINT;");
    auto result = con->Query(query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->Collection().GetValue(0, 0).GetValue<std::int64_t>() == 209715);
    cpu_rows = sirius::test::collect_rows(*result);
  }

  SECTION("disabled accumulation retains one-shot-only behavior")
  {
    scoped_sirius_setting accumulation_off(*con, "enable_dynamic_filter_multi_partition", false);
    auto const run = run_on_gpu(*con, query);
    REQUIRE(run.rows == cpu_rows);
    REQUIRE(run.after.accumulations_started == run.before.accumulations_started);
    REQUIRE(run.after.filters_pushed == run.before.filters_pushed);
    REQUIRE(run.after.publications_skipped_build_not_whole >
            run.before.publications_skipped_build_not_whole);
  }

  SECTION("enabled accumulation publishes every original contribution")
  {
    scoped_sirius_setting accumulation_on(*con, "enable_dynamic_filter_multi_partition", true);
    require_complete_publication(run_on_gpu(*con, query), cpu_rows);
    require_complete_publication(run_on_gpu(*con, query), cpu_rows);
  }

  SECTION("a LIMIT above an ORDER BY over the accumulating join completes normally")
  {
    scoped_sirius_setting accumulation_on(*con, "enable_dynamic_filter_multi_partition", true);
    std::string const limited =
      "SELECT k, payload FROM (SELECT p.k AS k, b.payload AS payload FROM df_probe p "
      "JOIN df_build b ON p.k = b.k WHERE b.marker <> 0 ORDER BY p.k) LIMIT 10";
    result_rows cpu_limited;
    {
      scoped_sirius_setting cpu_only(*con, "gpu_execution", false);
      auto result = con->Query(limited);
      REQUIRE(result);
      REQUIRE_FALSE(result->HasError());
      cpu_limited = sirius::test::collect_rows(*result);
    }
    auto const run = run_on_gpu(*con, limited);
    REQUIRE(run.rows == cpu_limited);
    REQUIRE(run.after.accumulations_started == run.before.accumulations_started + 1);
    REQUIRE(run.after.accumulations_skipped_error == run.before.accumulations_skipped_error);
    REQUIRE(run.after.accumulations_abandoned == run.before.accumulations_abandoned);
  }

  SECTION("the master switch disables accumulated publication")
  {
    scoped_sirius_setting accumulation_on(*con, "enable_dynamic_filter_multi_partition", true);
    scoped_sirius_setting master_off(*con, "enable_dynamic_filter", false);
    auto const run = run_on_gpu(*con, query);
    REQUIRE(run.rows == cpu_rows);
    REQUIRE(run.after.producers_enabled == run.before.producers_enabled);
    REQUIRE(run.after.accumulations_started == run.before.accumulations_started);
    REQUIRE(run.after.filters_pushed == run.before.filters_pushed);
  }

  SECTION("pinned narrow payloads retain native accumulated join keys")
  {
    scoped_sirius_setting compression_on(*con, "enable_compressed_materialization", true);
    scoped_sirius_setting accumulation_on(*con, "enable_dynamic_filter_multi_partition", true);
    absl::Cleanup unpin = [&] {
      con->Query("CALL unpin_table('df_probe');");
      con->Query("CALL unpin_table('df_build');");
    };
    auto const before_pin = sirius::test::get_compressed_materialization_stats(*con);
    run_ok("CALL pin_table(format='duckdb', name='df_build', tier='host', compression=false);");
    run_ok("CALL pin_table(format='duckdb', name='df_probe', tier='host', compression=false);");
    auto const pinned = sirius::test::get_compressed_materialization_stats(*con);
    REQUIRE(pinned.pin_columns_narrowed > before_pin.pin_columns_narrowed);
    auto const run = run_on_gpu(*con, query);
    require_complete_publication(run, cpu_rows);
    REQUIRE(run.after.keys_skipped_type_mismatch == run.before.keys_skipped_type_mismatch);
    auto const after = sirius::test::get_compressed_materialization_stats(*con);
    REQUIRE(after.scan_sidecars_installed > pinned.scan_sidecars_installed);
    REQUIRE(after.partition_narrow_columns > pinned.partition_narrow_columns);
    if (key_type == "BIGINT") {
      REQUIRE(after.scan_columns_restored > pinned.scan_columns_restored);
    }
  }

  SECTION("disabling accumulation does not change results")
  {
    scoped_sirius_setting accumulation_off(*con, "enable_dynamic_filter_multi_partition", false);
    auto const run = run_on_gpu(*con, query);
    REQUIRE(run.rows == cpu_rows);
    REQUIRE(run.after.publication_attempts == run.before.publication_attempts);
    REQUIRE(run.after.accumulations_started == run.before.accumulations_started);
    REQUIRE(run.after.accumulations_skipped_inventory ==
            run.before.accumulations_skipped_inventory);
    REQUIRE(run.after.accumulations_skipped_admission ==
            run.before.accumulations_skipped_admission);
    REQUIRE(run.after.keys_considered == run.before.keys_considered);
    REQUIRE(run.after.keys_skipped_bloom_size_gate == run.before.keys_skipped_bloom_size_gate);
    REQUIRE(run.after.keys_skipped_bloom_unsupported == run.before.keys_skipped_bloom_unsupported);
    REQUIRE(run.after.keys_skipped_type_mismatch == run.before.keys_skipped_type_mismatch);
    REQUIRE(run.after.membership_filters_built == run.before.membership_filters_built);
    REQUIRE(run.after.filters_pushed == run.before.filters_pushed);
    REQUIRE(run.after.accumulation_publications_finished ==
            run.before.accumulation_publications_finished);
  }

  SECTION("disabling accumulation leaves one-shot Bloom publication enabled")
  {
    scoped_sirius_setting accumulation_off(*con, "enable_dynamic_filter_multi_partition", false);
    sirius::test::scoped_setting whole_partition(*con, "hash_partition_bytes", 64 * 1024 * 1024);
    sirius::test::scoped_setting whole_build(*con, "max_build_hash_table_bytes", 64 * 1024 * 1024);
    sirius::test::scoped_setting force_bloom(*con, "dynamic_filter_inlist_max_l2_fraction", 0);
    auto const run = run_on_gpu(*con, query);
    REQUIRE(run.rows == cpu_rows);
    REQUIRE(run.after.accumulations_started == run.before.accumulations_started);
    REQUIRE(run.after.keys_skipped_bloom_size_gate == run.before.keys_skipped_bloom_size_gate);
    REQUIRE(run.after.membership_filters_built > run.before.membership_filters_built);
    REQUIRE(run.after.filters_pushed > run.before.filters_pushed);
  }
}
