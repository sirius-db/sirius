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

// End-to-end coverage of the compressed-materialization residency gate: a
// narrow physical sidecar is installed on a table scan only when the pinned
// cache will serve that scan with columns already narrowed at pin time.
// The three residency states (pinned-narrow, unpinned, pinned-native) plus the
// cols-subset serve-ability boundary are each discriminated through the
// plan-time scan_sidecars_installed counter and the serve-time
// scan_columns_narrowed / scan_columns_restored counters; a multi-file test
// proves the gate engages identically when the parquet identity spans several
// files (plan targets derive from the pinned entry's stored chunk carriers,
// never from source statistics); a further test proves zero-benefit pruning
// stays observable on pinned-backed sidecars. GPU-tier pins additionally
// engage the tier narrowing policy — columns whose only uses are restorations
// retract to native at plan time (scan_narrow_targets_retracted) and widen
// during scan normalization instead — while HOST-tier pins are structurally
// invisible to that pass, so their serves stay cast-free.

#include "compressed_materialization_test_common.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>

namespace fs = std::filesystem;

using namespace sirius::test::compmat;

namespace {

// 8192 rows x (k BIGINT, v BIGINT, d DECIMAL(18,2)) in 2048-row row groups
// (DuckDB rounds parquet row-group sizes up to vector-size multiples, so 2048
// is the smallest real group); the 16 KiB scan batches make each row group its
// own pin chunk (>= 3, asserted).
constexpr std::int64_t kFactRows = 8192;
// Dimension side for the join-key pruning case: k only, same value recipe.
constexpr std::int64_t kDimRows   = 1200;
constexpr std::size_t kBatchBytes = 16u << 10;

// Value recipes keep every chunk's exact range equal to the file-level footer
// range: k takes the multiples of 7 in [0, 294] (period 43 rows) and v the
// multiples of 11 in [0, 286] (period 27 rows), so both pick INT16 per chunk
// and at plan time; d's unscaled values stay under 29000, picking DECIMAL32
// everywhere. Chunk carrier == plan target for all chunks, so a pinned-narrow
// serve performs zero casts and the restored counter can be asserted flat.
// The periods divide the 2048-row chunk size many times over, so ANY 2048-row
// slice of the recipe — in particular every chunk of every part of a
// multi-part fixture — picks the same carriers (INT16, INT16, DECIMAL32).
constexpr char const* kPayloadQuery =
  "SELECT k, v, d, d + CAST('0.00' AS DECIMAL(3,2)) AS d_copy "
  "FROM t WHERE k <= 150 ORDER BY k, v;";

// Both exchange-related sizes stay large so this fixture never partitions; only the scan batch
// size is small enough to split the file into per-row-group pin chunks.
constexpr config_values kConfigValues{.scan_batch_bytes     = kBatchBytes,
                                      .hash_partition_bytes = 100000000,
                                      .concat_batch_bytes   = 100000000};

// Write the fact rows [first, first + count) — one slice of the single value
// recipe — so single-file and multi-part fixtures share byte-identical values
// over the same row range.
void generate_fact_parquet_part(fs::path const& path, std::int64_t first, std::int64_t count)
{
  generate_parquet(path,
                   "SELECT (range * 7) % 301 AS k, (range * 11) % 297 AS v, "
                   "CAST(((range * 13) % 29000) / 100.0 AS DECIMAL(18,2)) AS d "
                   "FROM range(" +
                     std::to_string(first) + ", " + std::to_string(first + count) + ")",
                   2048);
}

void generate_fact_parquet(fs::path const& path) { generate_fact_parquet_part(path, 0, kFactRows); }

void generate_dim_parquet(fs::path const& path)
{
  generate_parquet(
    path, "SELECT (range * 7) % 301 AS k FROM range(" + std::to_string(kDimRows) + ")", 400);
}

sirius::scan_manager::pinned_entry const* find_entry(duckdb::SiriusContext& sirius_ctx,
                                                     std::string_view entry_name)
{
  sirius::scan_manager::pinned_entry const* entry_ptr = nullptr;
  sirius_ctx.get_scan_manager().visit_pinned_entries(
    [&entry_ptr, entry_name](std::string_view name, auto const& e) {
      if (name == entry_name) {
        entry_ptr = &e;
        return false;  // found: stop iterating
      }
      return true;  // keep iterating
    });
  return entry_ptr;
}

std::size_t entry_chunk_count(sirius::scan_manager::pinned_entry const& e)
{
  if (e.tier == cucascade::memory::Tier::HOST) { return e.host_chunks.size(); }
  return e.data_batches_by_column.empty() ? 0 : e.data_batches_by_column.begin()->second.size();
}

}  // namespace

// Narrowing is the materialization every pinned numeric scan column gets without anyone asking
// for it, so the enabled default is part of the contract rather than a convenience. The SQL
// setting registers its default from this same struct member, which is why pinning the member
// pins both.
TEST_CASE("compressed materialization is enabled by default", "[compressed_materialization_gate]")
{
  REQUIRE(sirius::operator_params{}.enable_compressed_materialization);
}

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own
// SiriusContext from a small-batch yaml and manages (pauses) the shared envs
// itself, mirroring the other isolated-context pin tests.
TEST_CASE("gpu_execution - compressed materialization residency gate states end to end",
          "[gpu_execution][parquet][compressed_materialization_gate]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-gate-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "gate.parquet";
  generate_fact_parquet(parquet_path);

  auto yaml_path = tmp / "compmat_gate.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // Read before any SET on this connection, so it is the registered default that answers.
    auto registered_default =
      con.Query("SELECT current_setting('enable_compressed_materialization')::BOOLEAN;");
    require_ok(registered_default, "read registered default");
    REQUIRE(registered_default->GetValue(0, 0).GetValue<bool>());

    // A gate regression must fail loudly instead of silently falling back.
    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + parquet_path.string() + "');"),
      "create view");

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);

    SECTION("pinned-narrow serve is cast-free wherever the plan stays narrow")
    {
      for (auto const* tier : {"gpu", "host"}) {
        DYNAMIC_SECTION("tier = " << tier)
        {
          require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
          auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
          auto pin = con.Query("CALL pin_table('" + parquet_path.string() + "', tier='" +
                               std::string(tier) + "', name='t');");
          require_ok(pin, "pin_table");
          auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
          REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

          auto const* entry_ptr = find_entry(*sirius_ctx, "t");
          REQUIRE(entry_ptr != nullptr);
          REQUIRE(entry_chunk_count(*entry_ptr) >= 3);

          // The gate installs a sidecar (the pin serves every column, narrowed
          // in every chunk). The serve outcome is then tier-dependent: this
          // query's narrow columns all die at the ORDER_BY boundary, so on
          // GPU tier the narrowing policy retracts every target (no upload
          // means no benefit without transport) and the narrow resident
          // chunks widen during scan normalization, while on HOST tier the
          // policy is inert and the serve is cast-free — the chunk carriers
          // already equal the plan targets, so nothing narrows or restores.
          bool const gpu_tier = std::string_view(tier) == "gpu";
          auto const before   = sirius::test::get_compressed_materialization_stats(con);
          compare_gpu_vs_cpu(con, kPayloadQuery);
          auto const after = sirius::test::get_compressed_materialization_stats(con);
          REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
          REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
          if (gpu_tier) {
            REQUIRE(after.scan_narrow_targets_retracted > before.scan_narrow_targets_retracted);
            REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
          } else {
            REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);
            REQUIRE(after.scan_columns_restored == before.scan_columns_restored);
          }

          // Flag-off contrast — the discriminator that narrow data actually
          // flowed through the sidecar above: with the flag off no sidecar
          // exists, so the same narrow resident chunks restore to native
          // during scan normalization.
          require_ok(con.Query("SET enable_compressed_materialization = false;"), "disable flag");
          auto const off_before = sirius::test::get_compressed_materialization_stats(con);
          compare_gpu_vs_cpu(con, kPayloadQuery);
          auto const off_after = sirius::test::get_compressed_materialization_stats(con);
          REQUIRE(off_after.scan_sidecars_installed == off_before.scan_sidecars_installed);
          REQUIRE(off_after.scan_columns_restored > off_before.scan_columns_restored);
          require_ok(con.Query("SET enable_compressed_materialization = true;"), "restore flag");

          require_ok(con.Query("CALL unpin_table('t');"), "unpin");
        }
      }
    }

    SECTION("pinned-native installs no narrow targets")
    {
      // Pinning with the flag off stores native carriers with all-native
      // markers; a later flag-on query must not narrow them at serve time as a
      // recurring per-query cost, so the whole sidecar is dropped.
      require_ok(con.Query("SET enable_compressed_materialization = false;"), "flag off for pin");
      auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
      auto pin =
        con.Query("CALL pin_table('" + parquet_path.string() + "', tier='gpu', name='t');");
      require_ok(pin, "pin_table");
      auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(pin_after.pin_columns_narrowed == pin_before.pin_columns_narrowed);

      require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, kPayloadQuery);
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed == before.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored == before.scan_columns_restored);

      require_ok(con.Query("CALL unpin_table('t');"), "unpin");
    }

    SECTION("unpinned installs nothing")
    {
      // State 2: with no pinned entry the flag-on fresh scan is byte-identical
      // to feature-off — no sidecar, no statistics work, no casts, no restores.
      require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, kPayloadQuery);
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed == before.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored == before.scan_columns_restored);
    }

    SECTION("cols-subset pin gates per serve-ability")
    {
      require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
      auto pin = con.Query("CALL pin_table('" + parquet_path.string() +
                           "', tier='gpu', name='t', cols=['k']);");
      require_ok(pin, "pin k only");

      // The pin cannot serve v: empty serve-projection means no sidecar and a
      // fresh native disk read.
      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, "SELECT k, v FROM t WHERE k <= 150 ORDER BY k, v;");
      auto const mid = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(mid.scan_sidecars_installed == before.scan_sidecars_installed);
      REQUIRE(mid.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(mid.scan_columns_restored == before.scan_columns_restored);

      // A k-only scan is served from the pin, so the gate installs; nothing
      // ever narrows at serve time. k is filter+order-only on a GPU-tier pin,
      // so the tier policy retracts its target at plan time (no restored
      // assertion: the resident narrow chunks legitimately widen at the scan).
      compare_gpu_vs_cpu(con, "SELECT k FROM t WHERE k <= 150 ORDER BY k;");
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed > mid.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == mid.scan_columns_narrowed);
      REQUIRE(after.scan_narrow_targets_retracted > mid.scan_narrow_targets_retracted);

      require_ok(con.Query("CALL unpin_table('t');"), "unpin");
    }
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("gpu_execution - multi-file pinned-narrow serve installs the residency sidecar",
          "[gpu_execution][parquet][compressed_materialization_gate]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-multi-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp / "multi");

  // Two-part fixture: the parquet identity is MULTIPLE_FILES, the shape whose
  // sidecar installation must not depend on source-statistics availability.
  generate_fact_parquet_part(tmp / "multi" / "part0.parquet", 0, kFactRows / 2);
  generate_fact_parquet_part(tmp / "multi" / "part1.parquet", kFactRows / 2, kFactRows / 2);

  auto yaml_path = tmp / "compmat_multi.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    auto const glob = (tmp / "multi" / "*.parquet").string();
    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + glob + "');"),
               "create view");

    auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(sirius_ctx);

    for (auto const* tier : {"gpu", "host"}) {
      DYNAMIC_SECTION("tier = " << tier)
      {
        require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
        auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
        auto pin =
          con.Query("CALL pin_table('" + glob + "', tier='" + std::string(tier) + "', name='t');");
        require_ok(pin, "pin_table");
        auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
        REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

        // Pin the repro shape against fixture edits: two resolved files with
        // at least two pin chunks per file side.
        auto const* entry_ptr = find_entry(*sirius_ctx, "t");
        REQUIRE(entry_ptr != nullptr);
        REQUIRE(entry_ptr->cache_info.resolved_file_paths.size() == 2);
        REQUIRE(entry_chunk_count(*entry_ptr) >= 4);

        // The gate installs a sidecar (targets derive from the entry's stored
        // chunk carriers) regardless of source-statistics availability. As in
        // the single-file test, the query's narrow columns die at the
        // ORDER_BY boundary, so GPU-tier pins retract them (normalization
        // widens the resident chunks) while HOST-tier serves stay cast-free.
        bool const gpu_tier = std::string_view(tier) == "gpu";
        auto const before   = sirius::test::get_compressed_materialization_stats(con);
        compare_gpu_vs_cpu(con, kPayloadQuery);
        auto const after = sirius::test::get_compressed_materialization_stats(con);
        REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
        REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
        if (gpu_tier) {
          REQUIRE(after.scan_narrow_targets_retracted > before.scan_narrow_targets_retracted);
          REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
        } else {
          REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);
          REQUIRE(after.scan_columns_restored == before.scan_columns_restored);
        }

        // Flag-off contrast — the discriminator that narrow data actually
        // resides and flowed through the sidecar above: with the flag off no
        // sidecar exists, so the same narrow resident chunks restore to
        // native during scan normalization.
        require_ok(con.Query("SET enable_compressed_materialization = false;"), "disable flag");
        auto const off_before = sirius::test::get_compressed_materialization_stats(con);
        compare_gpu_vs_cpu(con, kPayloadQuery);
        auto const off_after = sirius::test::get_compressed_materialization_stats(con);
        REQUIRE(off_after.scan_sidecars_installed == off_before.scan_sidecars_installed);
        REQUIRE(off_after.scan_columns_restored > off_before.scan_columns_restored);
        require_ok(con.Query("SET enable_compressed_materialization = true;"), "restore flag");

        require_ok(con.Query("CALL unpin_table('t');"), "unpin");
      }
    }
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("gpu_execution - tier policy retracts restore-only columns on GPU tier only",
          "[gpu_execution][parquet][compressed_materialization_gate]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-tier-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto parquet_path = tmp / "gate.parquet";
  generate_fact_parquet(parquet_path);

  auto yaml_path = tmp / "compmat_tier.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + parquet_path.string() + "');"),
      "create view");
    require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");

    // Q1 shape: no transport and no narrow-domain comparison anywhere — k feeds modulo
    // arithmetic, v feeds arithmetic and aggregates, d is an input of the AVG-ineligible
    // aggregate — so every narrow-carrier use is a restoration.
    constexpr char const* kQ1ShapeQuery =
      "SELECT sum(v * 2) AS sv, avg(v) AS av, sum(d) AS sd FROM t WHERE (k % 2) = 0;";

    for (auto const* tier : {"gpu", "host"}) {
      DYNAMIC_SECTION("tier = " << tier)
      {
        require_ok(con.Query("CALL pin_table('" + parquet_path.string() + "', tier='" +
                             std::string(tier) + "', name='t');"),
                   "pin_table");

        // Both tiers install the sidecar and never narrow at serve time. On GPU tier the
        // narrowing policy retracts every column (a restoration-only plan is pure cost when the
        // serve pays no upload), so the resident narrow chunks widen during scan normalization;
        // on HOST tier the plan keeps every narrow target and the serve is cast-free — the
        // restorations run downstream, in the evaluator and at the aggregate boundary.
        bool const gpu_tier = std::string_view(tier) == "gpu";
        auto const before   = sirius::test::get_compressed_materialization_stats(con);
        compare_gpu_vs_cpu(con, kQ1ShapeQuery);
        auto const after = sirius::test::get_compressed_materialization_stats(con);
        REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
        REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
        if (gpu_tier) {
          REQUIRE(after.scan_narrow_targets_retracted > before.scan_narrow_targets_retracted);
          REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
        } else {
          REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);
          REQUIRE(after.scan_columns_restored == before.scan_columns_restored);
        }

        require_ok(con.Query("CALL unpin_table('t');"), "unpin");
      }
    }
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("gpu_execution - zero-benefit pruning stays discriminating on pinned-backed sidecars",
          "[gpu_execution][parquet][compressed_materialization_gate]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-prune-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto fact_path = tmp / "gate.parquet";
  generate_fact_parquet(fact_path);
  auto dim_path = tmp / "gate_dim.parquet";
  generate_dim_parquet(dim_path);

  auto yaml_path = tmp / "compmat_prune.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + fact_path.string() + "');"),
      "create fact view");
    require_ok(
      con.Query("CREATE VIEW o AS SELECT * FROM read_parquet('" + dim_path.string() + "');"),
      "create dim view");
    require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");

    // At runtime, "installed then pruned to native" and "cleared by
    // propagation" are deliberately indistinguishable — that equivalence is
    // the point of pruning. These sections prove the composite user-visible
    // contract: a narrow-eligible pinned scan whose only uses are zero-benefit
    // emits native at the scan (resident narrow chunks restore during scan
    // normalization), while the residency-gate test's payload shape stays
    // narrow. HOST-tier pins keep the tier narrowing policy structurally
    // inert (asserted through the flat retraction counter), so the native
    // sidecar here is pruning's own work; WHICH pass produces it is pinned
    // down by the planner unit tests in
    // test/cpp/planner/test_compressed_schema_propagation.cpp ("pruning
    // removes only zero-benefit restores").
    SECTION("aggregate-only pinned scan ends native at the scan")
    {
      auto pin = con.Query("CALL pin_table('" + fact_path.string() + "', tier='host', name='t');");
      require_ok(pin, "pin_table");

      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, "SELECT sum(v), sum(d) FROM t;");
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
      REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);

      require_ok(con.Query("CALL unpin_table('t');"), "unpin");
    }

    SECTION("join-key-only pinned scans end native at both scans")
    {
      auto pin_fact =
        con.Query("CALL pin_table('" + fact_path.string() + "', tier='host', name='t');");
      require_ok(pin_fact, "pin fact");
      auto pin_dim =
        con.Query("CALL pin_table('" + dim_path.string() + "', tier='host', name='o');");
      require_ok(pin_dim, "pin dim");

      auto const before = sirius::test::get_compressed_materialization_stats(con);
      compare_gpu_vs_cpu(con, "SELECT count(*) FROM t, o WHERE t.k = o.k;");
      auto const after = sirius::test::get_compressed_materialization_stats(con);
      REQUIRE(after.scan_sidecars_installed >= before.scan_sidecars_installed + 2);
      REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
      REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
      REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);

      require_ok(con.Query("CALL unpin_table('o');"), "unpin dim");
      require_ok(con.Query("CALL unpin_table('t');"), "unpin fact");
    }
  }

  fs::remove_all(tmp, ec);
}
