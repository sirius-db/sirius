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

/**
 * @file test_gpu_execution_top_n_pinned_serve.cpp
 * @brief End-to-end pinned-serve consumption flip (main doc, "Pinned-cache-served scans").
 *
 * A pinned parquet scan runs no reader, so dynamic filters published to it can only be consumed
 * by the scan's post-decode wrapper after the prepare-time `read_time_filter_bypass` promotes it
 * to `include_ast_row_masks`. The bracket here is two-sided: on a Top-N-only channel (no join, so
 * no membership filters) the `post_decode_apply_rows_in/out` counters move on a pinned serve only
 * if the flip engaged, and stay exactly zero on a fresh read where the reader consumes the
 * boundary -- so an under-flip fails the pinned leg and an over-flip fails the fresh-read leg.
 * Counter deltas follow the documented delivery-time contract (publication races batch arrival),
 * hence the bounded retry loop on the pinned leg and the deterministic zero on the fresh one.
 */

#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <unistd.h>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

using rows_t = std::vector<std::vector<std::string>>;

// 1M rows x 2 int64 columns -> 16 MiB decoded; 1 MiB scan batches -> ~16 pinned chunks, so a
// boundary established by an early batch still has most of the table in front of it.
constexpr std::int64_t kRows           = 1'000'000;
constexpr std::uint64_t kBatchBytes    = 1ull << 20;
constexpr std::size_t kMinPinnedChunks = 8;

// Enable the Top-N flag for one scope and restore the previous setting.
struct top_n_flag_guard {
  explicit top_n_flag_guard(duckdb::Connection& c, bool enabled) : con(c)
  {
    original = sirius::test::get_registered_sirius_context(c)
                 ->get_config()
                 .get_operator_params()
                 .enable_top_n_dynamic_filter;
    con.Query(std::string{"SET enable_top_n_dynamic_filter = "} + (enabled ? "true" : "false") +
              ";");
  }
  ~top_n_flag_guard()
  {
    con.Query(std::string{"SET enable_top_n_dynamic_filter = "} + (original ? "true" : "false") +
              ";");
  }

  top_n_flag_guard(const top_n_flag_guard&)            = delete;
  top_n_flag_guard& operator=(const top_n_flag_guard&) = delete;

  duckdb::Connection& con;
  bool original = false;
};

//! Shrink scan batches for one scope so both the pin and the query span many chunks. pin_table
//! reads the live setting, so this must be constructed before pinning.
struct scan_batch_size_guard {
  explicit scan_batch_size_guard(duckdb::Connection& c, std::uint64_t bytes) : con(c)
  {
    con.Query("SET scan_task_batch_size = " + std::to_string(bytes) + ";");
  }
  ~scan_batch_size_guard() { con.Query("RESET scan_task_batch_size"); }

  scan_batch_size_guard(const scan_batch_size_guard&)            = delete;
  scan_batch_size_guard& operator=(const scan_batch_size_guard&) = delete;

  duckdb::Connection& con;
};

// Enable zone-map filters for one scope and restore the previous setting.
struct zone_map_switch_guard {
  explicit zone_map_switch_guard(duckdb::Connection& c)
    : con(c),
      original(sirius::test::get_registered_sirius_context(c)
                 ->get_config()
                 .get_operator_params()
                 .enable_dynamic_zone_map_filter)
  {
    con.Query("SET enable_dynamic_zone_map_filter = true;");
  }
  ~zone_map_switch_guard()
  {
    con.Query(std::string{"SET enable_dynamic_zone_map_filter = "} + (original ? "true" : "false") +
              ";");
  }

  zone_map_switch_guard(const zone_map_switch_guard&)            = delete;
  zone_map_switch_guard& operator=(const zone_map_switch_guard&) = delete;

  duckdb::Connection& con;
  bool original;
};

//! Best-effort unpin on scope exit, so a failing assertion cannot leak a pinned entry into the
//! shared environment (destructor results are not asserted -- Catch2 macros throw).
struct scoped_pin {
  scoped_pin(duckdb::Connection& c, std::string const& path, std::string name, char const* tier)
    : con(c), entry_name(std::move(name))
  {
    auto pin =
      con.Query("CALL pin_table('" + path + "', tier='" + tier + "', name='" + entry_name + "');");
    REQUIRE(pin);
    if (pin->HasError()) { UNSCOPED_INFO("pin_table error: " << pin->GetError()); }
    REQUIRE_FALSE(pin->HasError());
  }
  ~scoped_pin() { con.Query("CALL unpin_table('" + entry_name + "');"); }

  scoped_pin(const scoped_pin&)            = delete;
  scoped_pin& operator=(const scoped_pin&) = delete;

  duckdb::Connection& con;
  std::string entry_name;
};

std::size_t pinned_entry_chunk_count(duckdb::Connection& con, std::string const& entry_name)
{
  auto sirius_ctx    = sirius::test::get_registered_sirius_context(con);
  std::size_t chunks = 0;
  bool found         = false;
  sirius_ctx->get_scan_manager().visit_pinned_entries(
    [&](std::string_view name, sirius::scan_manager::pinned_entry const& e) {
      if (name != entry_name) { return false; }
      found = true;
      chunks =
        e.tier == cucascade::memory::Tier::HOST
          ? e.host_chunks.size()
          : (e.data_batches_by_column.empty() ? 0
                                              : e.data_batches_by_column.begin()->second.size());
      return true;
    });
  REQUIRE(found);
  return chunks;
}

rows_t query_rows(duckdb::Connection& con, const std::string& query)
{
  auto result = con.Query(query);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
}

//! Result equivalence across the three legs the flip must not disturb: GPU with the Top-N flag
//! on, GPU with it off, and the CPU baseline -- all under whatever serve path is currently live.
void require_flag_equivalence(duckdb::Connection& con, const std::string& query)
{
  con.Query("SET gpu_execution = true;");
  rows_t flag_on;
  {
    top_n_flag_guard flag(con, true);
    flag_on = query_rows(con, query);
  }
  rows_t flag_off;
  {
    top_n_flag_guard flag(con, false);
    flag_off = query_rows(con, query);
  }
  con.Query("SET gpu_execution = false;");
  auto const cpu = query_rows(con, query);
  con.Query("SET gpu_execution = true;");

  REQUIRE(flag_on == flag_off);
  REQUIRE(flag_on == cpu);
}

}  // namespace

TEST_CASE("gpu_execution - top-n dynamic filter on a pinned-cache-served scan",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  con.Query("SET gpu_execution = true;");

  auto tmp = fs::temp_directory_path() / ("sirius-topn-pinned-serve-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  auto const parquet_path = (tmp / "pinned_facts.parquet").string();
  std::string const entry = "pinned_facts_" + std::to_string(::getpid());

  // Clustered ascending on the ORDER BY key, so the first batch's boundary prunes almost every
  // later batch and strict rows_out < rows_in is safe once any apply ran.
  {
    auto r = con.Query("COPY (SELECT range AS id, range AS v FROM range(" + std::to_string(kRows) +
                       ") ORDER BY v) TO '" + parquet_path + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }

  // Constructed before any pin: pin_table shapes its chunks from the live setting.
  scan_batch_size_guard batches(con, kBatchBytes);

  std::string const topn_query =
    "SELECT id, v FROM read_parquet('" + parquet_path + "') ORDER BY v LIMIT 10";

  for (auto const* tier : {"gpu", "host"}) {
    DYNAMIC_SECTION("tier = " << tier)
    {
      scoped_pin pin(con, parquet_path, entry, tier);
      // Premise: the SET above must shape the pin into many chunks, or the boundary would have
      // nothing left to prune and every counter assertion below would be vacuous. If this fires,
      // move the fixture to a dedicated-YAML environment (zone-map test precedent).
      auto const n_chunks = pinned_entry_chunk_count(con, entry);
      UNSCOPED_INFO("pinned entry '" << entry << "' has " << n_chunks << " chunks");
      REQUIRE(n_chunks >= kMinPinnedChunks);

      SECTION("pinned serve consumes the top-n boundary post-decode")
      {
        require_flag_equivalence(con, topn_query);

        top_n_flag_guard flag(con, true);
        con.Query("SET gpu_execution = true;");
        // The channel is Top-N-only (no join in the query), so these counters cannot move
        // without the flip. Publication races batch arrival by design; retry within a small
        // bound rather than asserting one run's schedule.
        auto const tbefore = sirius::test::get_transparent_execution_stats(con);
        auto const before  = sirius::test::get_dynamic_filter_stats_snapshot(con);
        std::uint64_t runs = 0;
        for (; runs < 5; ++runs) {
          auto const rows = query_rows(con, topn_query);
          REQUIRE(rows.size() == 10);
          auto const now = sirius::test::get_dynamic_filter_stats_snapshot(con);
          if (now.post_decode_apply_rows_in > before.post_decode_apply_rows_in) {
            ++runs;
            break;
          }
        }
        auto const after  = sirius::test::get_dynamic_filter_stats_snapshot(con);
        auto const tafter = sirius::test::get_transparent_execution_stats(con);
        sirius::test::require_transparent_execution_delta(tbefore, tafter, runs, 0, runs);

        REQUIRE(after.top_n_revisions_published - before.top_n_revisions_published >= 1);
        REQUIRE(after.post_decode_apply_rows_in > before.post_decode_apply_rows_in);
        REQUIRE(after.post_decode_apply_rows_out - before.post_decode_apply_rows_out <
                after.post_decode_apply_rows_in - before.post_decode_apply_rows_in);
      }

      SECTION("fresh read keeps the post-decode counters at zero")
      {
        // The other half of the bracket: with the entry unpinned the reader consumes the
        // boundary, and a membership-only wrapper on a Top-N-only channel applies nothing --
        // an unconditional promotion (over-flip) fails here, deterministically.
        auto unpin = con.Query("CALL unpin_table('" + entry + "');");
        REQUIRE(unpin);
        REQUIRE_FALSE(unpin->HasError());

        top_n_flag_guard flag(con, true);
        con.Query("SET gpu_execution = true;");
        auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
        auto const rows   = query_rows(con, topn_query);
        auto const after  = sirius::test::get_dynamic_filter_stats_snapshot(con);
        REQUIRE(rows.size() == 10);
        REQUIRE(after.post_decode_apply_rows_in == before.post_decode_apply_rows_in);
        REQUIRE(after.post_decode_apply_rows_out == before.post_decode_apply_rows_out);
        require_flag_equivalence(con, topn_query);
      }

      SECTION("flag off moves nothing on a pinned serve")
      {
        // The flip cannot manufacture consumption without a producer: a channel-less scan is
        // never wrapped, so the latch has no consumer and every counter stays flat.
        top_n_flag_guard flag(con, false);
        con.Query("SET gpu_execution = true;");
        auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
        auto const rows   = query_rows(con, topn_query);
        auto const after  = sirius::test::get_dynamic_filter_stats_snapshot(con);
        REQUIRE(rows.size() == 10);
        REQUIRE(after.top_n_producers_eligible == before.top_n_producers_eligible);
        REQUIRE(after.top_n_offers == before.top_n_offers);
        REQUIRE(after.top_n_revisions_published == before.top_n_revisions_published);
        REQUIRE(after.post_decode_apply_rows_in == before.post_decode_apply_rows_in);
        REQUIRE(after.post_decode_apply_rows_out == before.post_decode_apply_rows_out);
      }

      SECTION("join zone map applies on a pinned-served scan")
      {
        // The join-path leg (zone maps are AST-lowerable only). Honest note: this section is
        // not a flip discriminator -- the join also publishes membership filters, which move the
        // same counters in either mode; the discriminators are the Top-N-only section above and
        // the operator-level cases. Build precedes probe (FULL barrier), so publication precedes
        // every probe batch and no retry loop is needed.
        zone_map_switch_guard zone_maps_on(con);
        con.Query("SET gpu_execution = true;");

        // File-backed scratch db: an in-memory native table would fall the query back to CPU.
        auto const dim_db = (tmp / "dim.duckdb").string();
        auto attach       = con.Query("ATTACH IF NOT EXISTS '" + dim_db + "' AS pinned_dim_db;");
        REQUIRE(attach);
        REQUIRE_FALSE(attach->HasError());
        // A narrow runtime-derived id band: integer division defeats static transitive pushdown,
        // and ~1k of 1M keys keeps the wrapper's keep ratio far under the 0.9 gate threshold.
        auto create = con.Query(
          "CREATE OR REPLACE TABLE pinned_dim_db.dim AS "
          "SELECT range AS did FROM range(4500, 6500);");
        REQUIRE(create);
        REQUIRE_FALSE(create->HasError());
        con.Query("CHECKPOINT pinned_dim_db;");

        std::string const join_query =
          "SELECT count(*), sum(f.v) FROM read_parquet('" + parquet_path +
          "') f JOIN (SELECT did FROM pinned_dim_db.dim WHERE did // 1000 = 5) d ON f.id = d.did";

        auto const tbefore = sirius::test::get_transparent_execution_stats(con);
        auto const before  = sirius::test::get_dynamic_filter_stats_snapshot(con);
        auto const gpu     = query_rows(con, join_query);
        auto const after   = sirius::test::get_dynamic_filter_stats_snapshot(con);
        auto const tafter  = sirius::test::get_transparent_execution_stats(con);
        sirius::test::require_transparent_execution_delta(tbefore, tafter, 1, 0, 1);

        con.Query("SET gpu_execution = false;");
        auto const cpu = query_rows(con, join_query);
        con.Query("SET gpu_execution = true;");
        REQUIRE(gpu == cpu);

        REQUIRE(after.zone_map_filters_built > before.zone_map_filters_built);
        REQUIRE(after.post_decode_apply_rows_in > before.post_decode_apply_rows_in);

        con.Query("DROP TABLE IF EXISTS pinned_dim_db.dim;");
        con.Query("DETACH pinned_dim_db;");
        std::error_code dim_ec;
        fs::remove(dim_db, dim_ec);
        fs::remove(dim_db + ".wal", dim_ec);
      }
    }
  }

  fs::remove_all(tmp, ec);
}
