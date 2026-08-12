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

// Stage-1 Top-N threshold refinement, end to end: eligible producers arm the coordinator and
// prefilter their own input (sink self-consumption); nothing is published. Counter assertions
// are deltas/directions only -- the transparent path constructs the plan twice per query, and
// batch arrival is not deterministic.

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using rows_t = std::vector<std::vector<std::string>>;

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

//! Force the scan to produce many small batches for one scope, so a boundary established by an
//! early batch is still in front of later ones. With a single batch there is nothing left to
//! prune by the time any evidence exists, and a prefilter test would pass vacuously.
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

rows_t query_rows(duckdb::Connection& con, const std::string& query)
{
  auto result = con.Query(query);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
}

//! Result equivalence across the three legs the feature must not disturb: GPU with the flag on,
//! GPU with the flag off, and the CPU baseline.
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

// Per-process scratch database holding the file-backed test tables: the Sirius duckdb-native
// scan requires a single-file block manager, so in-memory tables would fall every query back to
// DuckDB. The PID suffix keeps concurrent runs on one host from colliding on DuckDB's file lock,
// and the destructor detaches and removes the file so no state survives the run.
struct scoped_topn_db {
  explicit scoped_topn_db(duckdb::Connection& c)
    : con(c),
      path(std::filesystem::temp_directory_path() /
           ("sirius_topn_dynamic_filter_stage1." + std::to_string(::getpid()) + ".duckdb"))
  {
    auto attach = con.Query("ATTACH IF NOT EXISTS '" + path.string() + "' AS topn_db;");
    REQUIRE(attach);
    REQUIRE_FALSE(attach->HasError());
    auto use_db = con.Query("USE topn_db;");
    REQUIRE(use_db);
    REQUIRE_FALSE(use_db->HasError());
  }
  ~scoped_topn_db()
  {
    // Best-effort teardown from a destructor: results are not asserted (Catch2 macros throw).
    con.Query("USE memory;");
    con.Query("DETACH topn_db;");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + ".wal", ec);
  }

  scoped_topn_db(const scoped_topn_db&)            = delete;
  scoped_topn_db& operator=(const scoped_topn_db&) = delete;

  duckdb::Connection& con;
  std::filesystem::path path;
};

// Parquet copy of one scratch table, removed on destruction. The publication sections read this
// instead of the native table: the duckdb-native scan has no read-time filter path, so the siting
// rule binds no Top-N target there and the publication counters those sections assert cannot
// move. Parquet is the consumer that skips reads. The native shapes stay covered by the sections
// whose subject is refusal or self-consumption, which are scan-format-independent.
struct scoped_topn_parquet {
  scoped_topn_parquet(duckdb::Connection& con, std::string const& table)
    : path(std::filesystem::temp_directory_path() /
           ("sirius_topn_" + table + "." + std::to_string(::getpid()) + ".parquet"))
  {
    auto r = con.Query("COPY " + table + " TO '" + path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  ~scoped_topn_parquet()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  scoped_topn_parquet(const scoped_topn_parquet&)            = delete;
  scoped_topn_parquet& operator=(const scoped_topn_parquet&) = delete;

  [[nodiscard]] std::string scan() const { return "read_parquet('" + path.string() + "')"; }

  std::filesystem::path path;
};

void create_test_tables(duckdb::Connection& con)
{
  // v is a permutation (10007 is prime and coprime to 37), so single- and multi-key orders over
  // it are deterministic; w duplicates within v-ties never surface because v never ties.
  auto r = con.Query(
    "CREATE OR REPLACE TABLE topn_facts AS "
    "SELECT CAST(i AS INTEGER) AS id, "
    "       CAST((i * 37) % 10007 AS INTEGER) AS v, "
    "       CAST((i * 11) % 97 AS INTEGER) AS w, "
    "       CAST(i % 13 AS INTEGER) AS grp, "
    "       'p' || CAST(i % 50 AS VARCHAR) AS pay "
    "FROM range(10000) t(i);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  r = con.Query(
    "CREATE OR REPLACE TABLE topn_dim AS "
    "SELECT CAST(i AS INTEGER) AS dk, CAST((i * 5) % 997 AS INTEGER) AS dv "
    "FROM range(10000) t(i);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  r = con.Query(
    "CREATE OR REPLACE TABLE topn_small AS "
    "SELECT CAST(i AS INTEGER) AS id, CAST(i * 3 AS INTEGER) AS v "
    "FROM range(20) t(i);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  // More than K null keys, so NULLS FIRST makes every K-row boundary head null.
  r = con.Query(
    "CREATE OR REPLACE TABLE topn_nulls AS "
    "SELECT CAST(i AS INTEGER) AS id, "
    "       CASE WHEN i % 3 = 0 THEN NULL ELSE CAST(i AS INTEGER) END AS v, "
    "       CAST(i % 7 AS INTEGER) AS w "
    "FROM range(50) t(i);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  // Grouped shape for the Stage-5 group-key producer. `grp` cycles every 13 rows, so with
  // `ORDER BY grp LIMIT 5` the boundary group (`grp = 4`) has rows in every batch the scan
  // produces -- which is what makes a one-comparison-too-strict predicate corrupt its `sum`
  // instead of merely pruning less. The table is large enough that later batches are certain to
  // meet a boundary established by earlier ones.
  // `sub` advances every 13 rows, so each (grp, sub) pair recurs every 39 rows and the multi-key
  // boundary tuple is likewise duplicated throughout the table.
  r = con.Query(
    "CREATE OR REPLACE TABLE topn_groups AS "
    "SELECT CAST(i AS INTEGER) AS id, "
    "       CAST(i % 13 AS INTEGER) AS grp, "
    "       CAST((i / 13) % 3 AS INTEGER) AS sub, "
    "       CAST((i * 37) % 10007 AS INTEGER) AS v "
    "FROM range(200000) t(i);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());

  // Flush the fresh tables out of the WAL: the native decoder reads checkpointed segments.
  r = con.Query("CHECKPOINT;");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
}

}  // namespace

TEST_CASE("gpu_execution - top-n dynamic filter sink self-consumption",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  scoped_topn_db scratch_db(con);
  create_test_tables(con);
  scoped_topn_parquet facts_pq(con, "topn_facts");

  SECTION("eligible multi-key query: equivalence plus producer and offer movement")
  {
    std::string const query = "SELECT id, v, w FROM " + facts_pq.scan() + " ORDER BY v, w LIMIT 10";
    require_flag_equivalence(con, query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    auto const tbefore = sirius::test::get_transparent_execution_stats(con);
    auto const before  = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows    = query_rows(con, query);
    auto const after   = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const tafter  = sirius::test::get_transparent_execution_stats(con);
    // The counted run must have executed on the GPU -- a silent fallback would void every
    // counter assertion below.
    sirius::test::require_transparent_execution_delta(tbefore, tafter, 1, 0, 1);
    REQUIRE(rows.size() == 10);
    REQUIRE(after.top_n_producers_eligible > before.top_n_producers_eligible);
    REQUIRE(after.top_n_producers_rejected == before.top_n_producers_rejected);
    REQUIRE(after.top_n_offers > before.top_n_offers);
    // Pruning direction only: batch arrival is not deterministic.
    REQUIRE(after.top_n_prefilter_rows_out - before.top_n_prefilter_rows_out <=
            after.top_n_prefilter_rows_in - before.top_n_prefilter_rows_in);

    // Publication is producer-side, so it does not race the scan: both keys come from one table,
    // so the all-keys trace binds that scan and every accepted offer installs a revision. The
    // scan is the parquet copy -- the siting rule binds only a consumer that skips reads, and the
    // native table would leave every publication counter below flat. The *effect* on the scan is
    // what races, and is therefore never asserted here (R2).
    REQUIRE(after.top_n_lex_scan_targets > before.top_n_lex_scan_targets);
    REQUIRE(after.top_n_revisions_published > before.top_n_revisions_published);
    REQUIRE(after.top_n_lex_filters_pushed > before.top_n_lex_filters_pushed);
    REQUIRE(after.top_n_revisions_failed == before.top_n_revisions_failed);
    // The LEX site subsumes the first-key bound at the same scan, so no separate RANGE target.
    REQUIRE(after.top_n_first_key_subsumed_by_lex > before.top_n_first_key_subsumed_by_lex);
    REQUIRE(after.top_n_first_key_scan_targets == before.top_n_first_key_scan_targets);
  }

  SECTION("split keys across a join: sited endpoint publishes and results stay exact")
  {
    // Scenario 10, executed rather than only planned. Keys come from opposite sides of a join, so
    // both traces stop at the join output; the residual filter makes that site material, so the
    // inclusive first-key bound is published through a *sited Top-N endpoint above a HASH_JOIN* --
    // a placement the join path never produces, and the only runtime shape that exercises the
    // endpoint's execute path, its generation-domain gate, and a multi-key FIRST_KEY filter.
    std::string const query =
      "SELECT f.v, d.dv FROM topn_facts f JOIN topn_dim d ON f.id = d.dk "
      "WHERE f.v + d.dv <> 1234567 ORDER BY f.v, d.dv LIMIT 10";

    // Correctness first: identical to flag-off and to CPU.
    require_flag_equivalence(con, query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    auto const tbefore = sirius::test::get_transparent_execution_stats(con);
    auto const before  = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows    = query_rows(con, query);
    auto const after   = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const tafter  = sirius::test::get_transparent_execution_stats(con);
    sirius::test::require_transparent_execution_delta(tbefore, tafter, 1, 0, 1);
    REQUIRE(rows.size() == 10);

    // An endpoint was sited and it published; the LEX site was deferred, not cost-gate skipped.
    REQUIRE(after.top_n_endpoint_sites_placed > before.top_n_endpoint_sites_placed);
    REQUIRE(after.top_n_lex_endpoint_sites_deferred > before.top_n_lex_endpoint_sites_deferred);
    REQUIRE(after.top_n_first_key_filters_pushed > before.top_n_first_key_filters_pushed);
    REQUIRE(after.top_n_revisions_published > before.top_n_revisions_published);
    REQUIRE(after.top_n_revisions_failed == before.top_n_revisions_failed);
    REQUIRE(after.top_n_revisions_ignored == before.top_n_revisions_ignored);
    // Neither trace reached a scan.
    REQUIRE(after.top_n_lex_scan_targets == before.top_n_lex_scan_targets);
    REQUIRE(after.top_n_first_key_scan_targets == before.top_n_first_key_scan_targets);
    // Pruning is direction-only: the endpoint races the sink by design (R2).
    REQUIRE(after.top_n_prefilter_rows_out - before.top_n_prefilter_rows_out <=
            after.top_n_prefilter_rows_in - before.top_n_prefilter_rows_in);
  }

  SECTION("flag off moves no Top-N counter")
  {
    top_n_flag_guard flag(con, false);
    con.Query("SET gpu_execution = true;");
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    (void)query_rows(con, "SELECT id, v, w FROM topn_facts ORDER BY v, w LIMIT 10");
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.top_n_producers_eligible == before.top_n_producers_eligible);
    REQUIRE(after.top_n_producers_rejected == before.top_n_producers_rejected);
    REQUIRE(after.top_n_producers_first_key_only == before.top_n_producers_first_key_only);
    REQUIRE(after.top_n_offers == before.top_n_offers);
    REQUIRE(after.top_n_prefilter_rows_in == before.top_n_prefilter_rows_in);
  }
}

//! Stage-5 group-key producer, end to end. The failure mode this suite defends against is not
//! "no pruning" but *quietly wrong aggregate values*: a boundary one comparison too strict drops
//! rows of a group that is in the answer and lowers that group's `sum`. Equivalence therefore
//! comes first in every section, and the counters only say the path was exercised.
TEST_CASE("gpu_execution - top-n dynamic filter group-key producer",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  scoped_topn_db scratch_db(con);
  create_test_tables(con);
  scoped_topn_parquet groups_pq(con, "topn_groups");

  SECTION("grouping-key order: the producer witnesses distinct keys and prunes its own input")
  {
    // Scenario 15. Every ORDER BY key is a grouping key and nothing filters in between, so the
    // group-key producer is admitted, roots below the aggregate, and prunes the aggregate's input.
    scan_batch_size_guard batches(con, 65536);
    std::string const query =
      "SELECT grp, min(v) AS m FROM " + groups_pq.scan() + " GROUP BY grp ORDER BY grp LIMIT 5";
    require_flag_equivalence(con, query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    auto const tbefore = sirius::test::get_transparent_execution_stats(con);
    auto const before  = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows    = query_rows(con, query);
    auto const after   = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const tafter  = sirius::test::get_transparent_execution_stats(con);
    sirius::test::require_transparent_execution_delta(tbefore, tafter, 1, 0, 1);
    REQUIRE(rows.size() == 5);

    REQUIRE(after.top_n_group_producers_eligible > before.top_n_group_producers_eligible);
    REQUIRE(after.top_n_group_producers_rejected == before.top_n_group_producers_rejected);
    REQUIRE(after.top_n_group_offers > before.top_n_group_offers);
    // Thirteen distinct groups against K = 5, so the very first offer fills the witness set.
    REQUIRE(after.top_n_group_witness_set_full > before.top_n_group_witness_set_full);
    REQUIRE(after.top_n_revisions_published > before.top_n_revisions_published);
    REQUIRE(after.top_n_revisions_failed == before.top_n_revisions_failed);
    // Prefiltering is direction-only. Whether a given batch meets a boundary at all depends on
    // batch scheduling, and the scan below may already have applied the published filter to the
    // rows that reach the sink, so neither a nonzero row count nor strict pruning is a property
    // of a correct run (R3). The deterministic proof that the predicate keeps boundary ties lives
    // in the operator test, `[physical_grouped_aggregate][top_n]`.
    REQUIRE(after.top_n_group_prefilter_rows_out - before.top_n_group_prefilter_rows_out <=
            after.top_n_group_prefilter_rows_in - before.top_n_group_prefilter_rows_in);
  }

  SECTION("boundary ties are kept, so the boundary group's aggregate stays exact")
  {
    // Scenario 18, the test that catches real damage rather than a flag. The boundary is the
    // fifth-best distinct `grp`, whose rows recur in every batch; a strict predicate would drop
    // every one of them after the first batch and `sum` for that group would come out low while
    // the query still returns five rows and no error. Only exact equality with CPU sees it.
    scan_batch_size_guard batches(con, 65536);
    std::string const query =
      "SELECT grp, sum(v) AS s FROM " + groups_pq.scan() + " GROUP BY grp ORDER BY grp LIMIT 5";
    require_flag_equivalence(con, query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows   = query_rows(con, query);
    auto const after  = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(rows.size() == 5);
    // The boundary group is present and is the last row of an ascending order.
    REQUIRE(rows.back().front() == "4");
    // A boundary existed and was published, so the inclusive predicate really did face these
    // rows somewhere. Which site pruned them, and how many, races the scan (R3).
    REQUIRE(after.top_n_group_offers > before.top_n_group_offers);
    REQUIRE(after.top_n_group_witness_set_full > before.top_n_group_witness_set_full);
    REQUIRE(after.top_n_revisions_published > before.top_n_revisions_published);
    REQUIRE(after.top_n_revisions_failed == before.top_n_revisions_failed);
  }

  SECTION("two grouping keys publish an inclusive LEX predicate and stay exact")
  {
    // The multi-key group path, which a single-column GROUP BY never reaches: both ordinals
    // survive to the scan together, so the producer plans a LEX target and publishes the
    // *inclusive* full-tuple predicate there. A strict one would drop rows tying the whole
    // boundary tuple -- rows of a group that is in the answer -- and lower that group's sum.
    scan_batch_size_guard batches(con, 65536);
    std::string const query = "SELECT grp, sub, sum(v) AS s FROM " + groups_pq.scan() +
                              " GROUP BY grp, sub ORDER BY grp, sub LIMIT 5";
    require_flag_equivalence(con, query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    auto const tbefore = sirius::test::get_transparent_execution_stats(con);
    auto const before  = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows    = query_rows(con, query);
    auto const after   = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const tafter  = sirius::test::get_transparent_execution_stats(con);
    sirius::test::require_transparent_execution_delta(tbefore, tafter, 1, 0, 1);
    REQUIRE(rows.size() == 5);
    // Thirty-nine group tuples against K = 5, so the boundary tuple is (1, 1) and its rows recur
    // in every batch.
    REQUIRE(rows.back().front() == "1");
    REQUIRE(rows.back()[1] == "1");

    REQUIRE(after.top_n_group_producers_eligible > before.top_n_group_producers_eligible);
    REQUIRE(after.top_n_group_offers > before.top_n_group_offers);
    REQUIRE(after.top_n_group_witness_set_full > before.top_n_group_witness_set_full);
    // A LEX target was planned and received the revision -- the site a single-key group shape
    // never creates. The first-key bound is subsumed at the same scan.
    REQUIRE(after.top_n_lex_scan_targets > before.top_n_lex_scan_targets);
    REQUIRE(after.top_n_lex_filters_pushed > before.top_n_lex_filters_pushed);
    REQUIRE(after.top_n_first_key_subsumed_by_lex > before.top_n_first_key_subsumed_by_lex);
    REQUIRE(after.top_n_revisions_published > before.top_n_revisions_published);
    REQUIRE(after.top_n_revisions_failed == before.top_n_revisions_failed);
  }

  SECTION("an aggregate-output order key admits no group-key producer")
  {
    // Scenario 16 (the Q3 shape): no input row's value determines `s`, so no input row can be
    // ruled out by a bound on it.
    std::string const query =
      "SELECT grp, sum(v) AS s FROM topn_groups GROUP BY grp ORDER BY s LIMIT 5";
    require_flag_equivalence(con, query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    (void)query_rows(con, query);
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.top_n_group_producers_rejected > before.top_n_group_producers_rejected);
    REQUIRE(after.top_n_group_producers_eligible == before.top_n_group_producers_eligible);
    REQUIRE(after.top_n_group_offers == before.top_n_group_offers);
  }

  SECTION("a filter between the aggregate and the top-n admits no group-key producer")
  {
    // Scenario 17: HAVING lowers to a FILTER above the aggregate, which can eliminate whole
    // groups, so a witnessed distinct key no longer proves a surviving group.
    std::string const query =
      "SELECT grp, min(v) AS m FROM topn_groups GROUP BY grp HAVING min(v) > 0 "
      "ORDER BY grp LIMIT 5";
    require_flag_equivalence(con, query);

    top_n_flag_guard flag(con, true);
    con.Query("SET gpu_execution = true;");
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    (void)query_rows(con, query);
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.top_n_group_producers_rejected > before.top_n_group_producers_rejected);
    REQUIRE(after.top_n_group_producers_eligible == before.top_n_group_producers_eligible);
    REQUIRE(after.top_n_group_offers == before.top_n_group_offers);
  }

  SECTION("flag off moves no group-key counter")
  {
    top_n_flag_guard flag(con, false);
    con.Query("SET gpu_execution = true;");
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    (void)query_rows(con,
                     "SELECT grp, min(v) AS m FROM topn_groups GROUP BY grp ORDER BY grp LIMIT 5");
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.top_n_group_producers_eligible == before.top_n_group_producers_eligible);
    REQUIRE(after.top_n_group_producers_rejected == before.top_n_group_producers_rejected);
    REQUIRE(after.top_n_group_offers == before.top_n_group_offers);
    REQUIRE(after.top_n_group_prefilter_rows_in == before.top_n_group_prefilter_rows_in);
    // Correctness is asserted here too, so this section leads with results like every other one.
    require_flag_equivalence(
      con, "SELECT grp, min(v) AS m FROM topn_groups GROUP BY grp ORDER BY grp LIMIT 5");
  }
}

//! Repeated-execution freshness: channels and coordinators are minted per physical-plan
//! construction and the transparent path constructs a fresh plan per execution, so no dynamic
//! filter state may leak across runs -- equal per-run counter deltas and identical results pin
//! that, for plain re-execution and for a prepared statement executed twice.
TEST_CASE("gpu_execution - dynamic filter state never leaks across executions",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  scoped_topn_db scratch_db(con);
  create_test_tables(con);

  top_n_flag_guard flag(con, true);
  con.Query("SET gpu_execution = true;");
  std::string const query = "SELECT id, v, w FROM topn_facts ORDER BY v, w LIMIT 10";

  struct run_observation {
    rows_t rows;
    std::uint64_t eligible = 0;
    std::uint64_t offers   = 0;
  };
  auto const run_once = [&]() -> run_observation {
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto rows         = query_rows(con, query);
    auto const after  = sirius::test::get_dynamic_filter_stats_snapshot(con);
    return {std::move(rows),
            after.top_n_producers_eligible - before.top_n_producers_eligible,
            after.top_n_offers - before.top_n_offers};
  };

  auto const first  = run_once();
  auto const second = run_once();
  REQUIRE(first.rows == second.rows);
  REQUIRE(first.eligible == second.eligible);
  // `offers` counts batches whose local heap reached K, which depends on how the scan partitions
  // across pipeline threads -- a direction, never an equality (R2). Freshness is what is under
  // test: a leaked coordinator would carry a boundary into run two and suppress its offers.
  REQUIRE(first.offers > 0);
  REQUIRE(second.offers > 0);

  SECTION("prepared statement executed twice")
  {
    auto prepared = con.Prepare(query);
    REQUIRE(prepared);
    REQUIRE_FALSE(prepared->HasError());
    auto const execute_prepared = [&]() -> run_observation {
      auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
      auto result       = prepared->Execute();
      REQUIRE(result);
      REQUIRE_FALSE(result->HasError());
      rows_t rows;
      while (auto chunk = result->Fetch()) {
        for (duckdb::idx_t r = 0; r < chunk->size(); ++r) {
          std::vector<std::string> row;
          for (duckdb::idx_t c = 0; c < chunk->ColumnCount(); ++c) {
            row.push_back(chunk->GetValue(c, r).ToString());
          }
          rows.push_back(std::move(row));
        }
      }
      std::sort(rows.begin(), rows.end());
      auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
      return {std::move(rows),
              after.top_n_producers_eligible - before.top_n_producers_eligible,
              after.top_n_offers - before.top_n_offers};
    };

    auto const first_prepared  = execute_prepared();
    auto const second_prepared = execute_prepared();
    REQUIRE(first_prepared.rows == second_prepared.rows);
    REQUIRE(first_prepared.rows == first.rows);
    REQUIRE(first_prepared.eligible == second_prepared.eligible);
    // Direction only, for the same scheduling reason as the plain re-execution above.
    REQUIRE(first_prepared.offers > 0);
    REQUIRE(second_prepared.offers > 0);
  }
}

TEST_CASE("gpu_execution - top-n dynamic filter negative shapes",
          "[integration][gpu_execution][dynamic_filter]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  scoped_topn_db scratch_db(con);
  create_test_tables(con);

  top_n_flag_guard flag(con, true);
  con.Query("SET gpu_execution = true;");

  SECTION("LIMIT 0: no producer, no offers, empty result")
  {
    std::string const query = "SELECT v FROM topn_facts ORDER BY v LIMIT 0";
    auto const before       = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows         = query_rows(con, query);
    auto const after        = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(rows.empty());
    REQUIRE(after.top_n_producers_eligible == before.top_n_producers_eligible);
    REQUIRE(after.top_n_offers == before.top_n_offers);
    require_flag_equivalence(con, query);
  }

  SECTION("tie-preserving rank shapes never form a plain Top-N producer")
  {
    // DuckDB's TopN optimizer only forms LogicalTopN from a plain constant LIMIT [OFFSET], so a
    // rank shape (the tie-preserving equivalent) must move no producer counter.
    std::string const query =
      "SELECT v FROM (SELECT v, RANK() OVER (ORDER BY v) AS r FROM topn_facts) WHERE r <= 10";
    auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
    (void)query_rows(con, query);
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.top_n_producers_eligible == before.top_n_producers_eligible);
    REQUIRE(after.top_n_offers == before.top_n_offers);
    require_flag_equivalence(con, query);
  }

  SECTION("VARCHAR head key rejects the producer")
  {
    std::string const query = "SELECT pay, v FROM topn_facts ORDER BY pay, v LIMIT 10";
    auto const before       = sirius::test::get_dynamic_filter_stats_snapshot(con);
    (void)query_rows(con, query);
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.top_n_producers_rejected > before.top_n_producers_rejected);
    REQUIRE(after.top_n_producers_eligible == before.top_n_producers_eligible);
    require_flag_equivalence(con, query);
  }

  SECTION("VARCHAR tail key only degrades the producer to first-key")
  {
    std::string const query = "SELECT v, pay FROM topn_facts ORDER BY v, pay LIMIT 10";
    auto const before       = sirius::test::get_dynamic_filter_stats_snapshot(con);
    (void)query_rows(con, query);
    auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(after.top_n_producers_eligible > before.top_n_producers_eligible);
    REQUIRE(after.top_n_producers_first_key_only > before.top_n_producers_first_key_only);
    REQUIRE(after.top_n_producers_rejected == before.top_n_producers_rejected);
    require_flag_equivalence(con, query);
  }

  SECTION("LIMIT with OFFSET pins K = limit + offset through the offer condition")
  {
    // Offers fire only when a local result owns exactly K rows, so any offer proves the
    // planner's checked K and the operator's kept-row count agree at K = 12.
    std::string const query = "SELECT v FROM topn_facts ORDER BY v LIMIT 7 OFFSET 5";
    auto const before       = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows         = query_rows(con, query);
    auto const after        = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(rows.size() == 7);
    REQUIRE(after.top_n_offers > before.top_n_offers);
    require_flag_equivalence(con, query);
  }

  SECTION("table smaller than K publishes nothing")
  {
    std::string const query = "SELECT v FROM topn_small ORDER BY v LIMIT 100";
    auto const before       = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows         = query_rows(con, query);
    auto const after        = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(rows.size() == 20);
    REQUIRE(after.top_n_producers_eligible > before.top_n_producers_eligible);
    REQUIRE(after.top_n_offers == before.top_n_offers);
    require_flag_equivalence(con, query);
  }

  SECTION("null boundary head is unsupported yet exact")
  {
    std::string const query = "SELECT v FROM topn_nulls ORDER BY v NULLS FIRST LIMIT 10";
    auto const before       = sirius::test::get_dynamic_filter_stats_snapshot(con);
    auto const rows         = query_rows(con, query);
    auto const after        = sirius::test::get_dynamic_filter_stats_snapshot(con);
    REQUIRE(rows.size() == 10);
    REQUIRE(after.top_n_offers_unsupported > before.top_n_offers_unsupported);
    require_flag_equivalence(con, query);
  }
}
