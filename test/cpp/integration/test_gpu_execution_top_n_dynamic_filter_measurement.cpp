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
 * @file test_gpu_execution_top_n_dynamic_filter_measurement.cpp
 * @brief Wall-clock cost and benefit of `enable_top_n_dynamic_filter`, measured against itself
 *
 * These cases produce the two numbers the enable-by-default decision needs and nothing else: what
 * the feature costs on a shape it cannot help, and what it wins on shapes it can. They are hidden
 * from the default suite (`[.]`) because they are a measurement, not a gate -- run them
 * deliberately, on a quiet device, with `sirius_unittest "[measurement]"`.
 *
 * Every number here is a *ratio between two arms of one process*.
 * `sirius::test::run_interleaved_ab` alternates the arms pair by pair, which this feature has
 * already been shown to require: a run that took all the flag-off samples first and all the flag-on
 * samples second reported a 24% regression, and re-running the identical queries interleaved
 * reduced it to 1.9%. Reading the scheduling of a shared host as an effect of the treatment is the
 * failure mode these cases are built to avoid, and it is why the control case below measures
 * flag-off against flag-off: until that A/A cell says how wide the instrument's own noise band is,
 * no A/B number in the file can be called an effect.
 *
 * ## What proves the measurement describes the feature
 *
 * A timing is only evidence if the run it came from took the GPU path and armed the producer the
 * shape is supposed to engage, so every execution is checked for both and any pair containing a
 * failed execution is dropped whole:
 *
 * - `duckdb::SiriusContext::transparent_execution_stats` must move by exactly one rebind and one
 *   execution with no fallback. A query that silently fell back to DuckDB's CPU engine would
 *   otherwise contribute a timing in which the feature does not exist.
 * - `sirius::op::dynamic_filter_stats` must show the shape's producer eligible and offering on the
 *   flag-on arm, and must show it untouched on the flag-off arm.
 *
 * `EXPLAIN` cannot serve as that proof and is deliberately not used. DuckDB renders an EXPLAIN
 * during physical planning of the explained statement, and `duckdb::SiriusContext` only swaps in a
 * GPU plan for `SELECT_STATEMENT`, so the text describes the CPU plan DuckDB would have run and
 * names DuckDB's operators. Its `PERFECT_HASH_GROUP_BY` in particular proves nothing either way
 * about the group-key producer: `sirius_physical_plan_generator` builds from the *logical* plan,
 * and every grouped `duckdb::LogicalAggregate` becomes a `sirius_physical_grouped_aggregate`
 * whatever DuckDB's own physical planner would have chosen. The counter
 * `top_n_group_producers_eligible` moving is the direct observation that
 * `install_group_key_producer` found that operator in the plan that was actually timed.
 *
 * ## The corpus, and why it is split across two scan backends
 *
 * Arming the producer and being *able to benefit* are different things, and the corpus separates
 * them because the engine does.
 *
 * The duckdb-native tables measure **cost**. The native scan has no read-time dynamic-filter path,
 * so a published boundary is applied post-decode, by a separate pass over rows that have already
 * been read, decoded, and copied to the device -- and `sirius_physical_top_n` has already
 * prefiltered its own input with the same predicate. Every counter says the feature is armed, and
 * every row it prunes was paid for anyway. A native shape can bound what the feature costs; it
 * cannot show what it wins, and reading a native regression as a verdict on the feature would be
 * the same error as measuring a shape the feature never touches.
 *
 * The parquet shapes measure **benefit**. There the boundary is merged into the reader AST and
 * cuDF excludes row groups on their statistics before reading them, which is the one place a
 * boundary turns into work not done. That requires the key to be clustered in storage, so the
 * corpus writes one file with an ascending key and reads it in both directions: ascending is the
 * winning shape, descending the adversary on the same bytes.
 *
 * Every shape lowers `scan_task_batch_size` so that one query spans enough batches for a boundary
 * established early to still be in front of later ones. With the default 100 MB batch these tables
 * scan in two tasks and there is nothing left to prune by the time any evidence exists.
 *
 * The adversary shapes are what enable-by-default hinges on. Their key ascends in storage order
 * and they are ordered descending, so every batch holds better rows than the boundary the batch
 * before it offered: the boundary is re-published continuously and prunes nothing. How adversarial
 * a run really was is measured rather than assumed -- the reported keep ratio says so, and a keep
 * ratio far below 100% would mean the scan visited batches out of storage order and the shape was
 * not adversarial that time.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/measurement_harness.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

using rows_t   = std::vector<std::vector<std::string>>;
using snapshot = sirius::op::dynamic_filter_stats_snapshot;

//! Scan batches a measured query should span. A boundary can only prune what the scan has not
//! reached yet, so one query has to produce enough batches for an early boundary to still be in
//! front of most of them; too many, and per-batch overhead rather than data movement is what the
//! measurement compares.
constexpr std::uint64_t k_target_batches_per_query = 24;

//! Bytes the common two-column projection reads per row, used to size a scan task from the row
//! count so the batch count stays put as the corpus is scaled.
constexpr std::uint64_t k_projected_bytes_per_row = 8;

//! The same, for the parquet shapes: a 4-byte key plus eight 8-byte payload columns.
constexpr std::uint64_t k_parquet_projected_bytes_per_row = 68;

//! K for every measured shape. Under `top_n_group_key_producer::k_max_admitted_k`, so the group
//! shape is admitted rather than refused on K.
constexpr int k_limit = 100;

/// Reads a positive integer from the environment, falling back to @p fallback. Corpus size and
/// repetition count are tunable this way because the scale at which this feature can be measured
/// at all is an open question: at a corpus small enough for a query to be dominated by fixed
/// per-query overhead, no amount of pruning shows up in wall clock, and the size where that stops
/// being true is found by re-running rather than by reasoning.
[[nodiscard]] std::uint64_t env_value(const char* name, std::uint64_t fallback)
{
  char const* raw = std::getenv(name);
  if (raw == nullptr) { return fallback; }
  auto const parsed = std::strtoull(raw, nullptr, 10);
  return parsed > 0 ? parsed : fallback;
}

/// Which producer a shape is meant to engage, and therefore which counters describe it.
enum class producer_kind { row, group };

/**
 * @brief Whether a shape is expected to arm its producer, asserted either way
 *
 * A measurement corpus contains two kinds of query, and both have to be measurable. Shapes the
 * feature acts on must arm, and a run of one that did not is worthless -- its timing describes an
 * execution in which the feature did nothing, and averaging it in would report the feature's
 * absence as its cost. Shapes the feature does not act on must *not* arm, and their timings are
 * the whole point of a no-regression claim: "this query is untouched" is a result, not a reason to
 * discard the sample.
 *
 * So arming is a declared property of the shape that the harness checks, never a precondition for
 * keeping a sample. Both directions are traps and both stay caught: a shape declared @ref armed
 * whose counters never moved, and a shape declared @ref unarmed whose counters did -- the latter
 * being eligibility creep, where a query silently comes into the feature's scope and a
 * no-regression corpus stops covering what it claims to.
 */
enum class arming_expectation {
  armed,   ///< The producer must be eligible and offering on the flag-on arm
  unarmed  ///< No Top-N counter may move on either arm; the shape is outside the feature's reach
};

/// A measured query together with what makes its timings mean something.
struct measurement_shape {
  std::string name;
  std::string intent;
  std::string query;
  producer_kind producer;
  std::size_t expected_rows;
  //! Scan task size to hold for this cell's two arms. It differs per shape because it is chosen to
  //! give a comparable batch count, and a shape's bytes per row depend on how wide its projection
  //! is. Both arms of a cell always see the same value, so it cannot bias a ratio.
  std::uint64_t scan_task_batch_bytes = 0;
  //! What the counters must show. Defaults to @ref arming_expectation::armed so that a shape added
  //! to exercise the feature cannot silently measure nothing; a no-regression shape states
  //! @ref arming_expectation::unarmed explicitly, which is also the assertion that it stays that
  //! way.
  arming_expectation arming = arming_expectation::armed;
};

/// Counter movement summed over one arm's executions of one cell, with the row-producer and
/// group-producer counters folded into common names by @ref producer_kind so a shape's report
/// reads the same whichever producer it engages.
struct arming_evidence {
  std::uint64_t executions          = 0;
  std::uint64_t producers_eligible  = 0;
  std::uint64_t producers_rejected  = 0;
  std::uint64_t offers              = 0;
  std::uint64_t revisions_published = 0;
  std::uint64_t revisions_failed    = 0;
  std::uint64_t scan_targets        = 0;
  //! Revisions a scan consumer accepted. Distinct from @ref scan_targets, which is only the
  //! plan-time fact that a site was bound: on the parquet path this is the observation that a
  //! boundary reached the reader, which is where a row group can be excluded before it is read.
  std::uint64_t filters_pushed     = 0;
  std::uint64_t prefilter_rows_in  = 0;
  std::uint64_t prefilter_rows_out = 0;

  void accumulate(producer_kind producer, const snapshot& before, const snapshot& after)
  {
    ++executions;
    revisions_published += after.top_n_revisions_published - before.top_n_revisions_published;
    revisions_failed += after.top_n_revisions_failed - before.top_n_revisions_failed;
    scan_targets += (after.top_n_first_key_scan_targets - before.top_n_first_key_scan_targets) +
                    (after.top_n_lex_scan_targets - before.top_n_lex_scan_targets);
    filters_pushed +=
      (after.top_n_first_key_filters_pushed - before.top_n_first_key_filters_pushed) +
      (after.top_n_lex_filters_pushed - before.top_n_lex_filters_pushed);
    if (producer == producer_kind::row) {
      producers_eligible += after.top_n_producers_eligible - before.top_n_producers_eligible;
      producers_rejected += after.top_n_producers_rejected - before.top_n_producers_rejected;
      offers += after.top_n_offers - before.top_n_offers;
      prefilter_rows_in += after.top_n_prefilter_rows_in - before.top_n_prefilter_rows_in;
      prefilter_rows_out += after.top_n_prefilter_rows_out - before.top_n_prefilter_rows_out;
    } else {
      producers_eligible +=
        after.top_n_group_producers_eligible - before.top_n_group_producers_eligible;
      producers_rejected +=
        after.top_n_group_producers_rejected - before.top_n_group_producers_rejected;
      offers += after.top_n_group_offers - before.top_n_group_offers;
      prefilter_rows_in +=
        after.top_n_group_prefilter_rows_in - before.top_n_group_prefilter_rows_in;
      prefilter_rows_out +=
        after.top_n_group_prefilter_rows_out - before.top_n_group_prefilter_rows_out;
    }
  }

  /// Fraction of rows the measured prefilters kept, or -1 when no prefilter ran at all -- the
  /// distinction between "kept everything" and "never faced a row" decides whether an adversarial
  /// shape really was adversarial.
  [[nodiscard]] double keep_fraction() const
  {
    return prefilter_rows_in == 0
             ? -1.0
             : static_cast<double>(prefilter_rows_out) / static_cast<double>(prefilter_rows_in);
  }

  [[nodiscard]] std::string describe() const
  {
    std::string text =
      "executions " + std::to_string(executions) + ", eligible +" +
      std::to_string(producers_eligible) + ", rejected +" + std::to_string(producers_rejected) +
      ", offers +" + std::to_string(offers) + ", scan targets +" + std::to_string(scan_targets) +
      ", filters pushed to scans +" + std::to_string(filters_pushed) + ", revisions +" +
      std::to_string(revisions_published) + " (failed " + std::to_string(revisions_failed) + ")";
    if (keep_fraction() < 0.0) {
      text += ", prefilter never ran";
    } else {
      text += ", prefilter " + std::to_string(prefilter_rows_in) + " -> " +
              std::to_string(prefilter_rows_out) + " rows (" +
              sirius::test::detail::fixed(keep_fraction() * 100.0, 3) + "% kept)";
    }
    return text;
  }
};

/// One arm of a cell: the name that appears in the report and the flag setting it stands for.
struct arm_spec {
  std::string name;
  bool flag_enabled = false;
};

/**
 * @brief Performs one execution of one arm and reports whether that execution may be counted
 *
 * The timed interval covers only `duckdb::Connection::Query`. Setting the flag and reading the
 * counters happen around it and are excluded, so a difference between the arms cannot be an
 * artifact of the measurement's own bookkeeping.
 *
 * An execution is rejected, rather than silently timed, when the transparent path did not run
 * exactly one GPU execution without falling back, when the query failed or returned the wrong row
 * count, or when the shape's producer did not match the arm: armed on the flag-on arm, untouched
 * on the flag-off arm. Rejecting is the point -- a timing whose `top_n_offers` never moved
 * describes a run in which the feature did nothing, and averaging it into a result would report
 * the absence of the feature as its cost.
 */
class arm_executor {
 public:
  arm_executor(duckdb::Connection& con,
               const measurement_shape& shape,
               std::array<arm_spec, 2> arms)
    : con_(con), shape_(shape), arms_(std::move(arms))
  {
  }

  sirius::test::timed_run operator()(int arm)
  {
    auto const& spec = arms_[static_cast<std::size_t>(arm)];
    sirius::test::timed_run run;

    auto set_flag = con_.Query(std::string{"SET enable_top_n_dynamic_filter = "} +
                               (spec.flag_enabled ? "true" : "false") + ";");
    if (!set_flag || set_flag->HasError()) {
      run.validity = {false, "could not set enable_top_n_dynamic_filter"};
      return run;
    }

    auto const transparent_before = sirius::test::get_transparent_execution_stats(con_);
    auto const counters_before    = sirius::test::get_dynamic_filter_stats_snapshot(con_);

    auto const start = std::chrono::steady_clock::now();
    auto result      = con_.Query(shape_.query);
    run.elapsed      = std::chrono::steady_clock::now() - start;

    auto const counters_after    = sirius::test::get_dynamic_filter_stats_snapshot(con_);
    auto const transparent_after = sirius::test::get_transparent_execution_stats(con_);

    if (!result || result->HasError()) {
      run.validity = {false, "query failed: " + (result ? result->GetError() : "null result")};
      return run;
    }
    if (result->RowCount() != shape_.expected_rows) {
      run.validity = {false,
                      "returned " + std::to_string(result->RowCount()) + " rows, expected " +
                        std::to_string(shape_.expected_rows)};
      return run;
    }
    if (transparent_after.successful_rebinds != transparent_before.successful_rebinds + 1 ||
        transparent_after.executions != transparent_before.executions + 1 ||
        transparent_after.fallbacks != transparent_before.fallbacks ||
        transparent_after.runtime_fallbacks != transparent_before.runtime_fallbacks) {
      run.validity = {false, "did not take the GPU path exactly once"};
      return run;
    }

    arming_evidence delta;
    delta.accumulate(shape_.producer, counters_before, counters_after);
    auto const armed = delta.producers_eligible != 0 || delta.offers != 0;
    // The flag-off arm is the same assertion for both expectations: with the feature off nothing
    // may move, whatever the shape is.
    if (!spec.flag_enabled && armed) {
      run.validity = {false, "flag off but a Top-N counter moved"};
      return run;
    }
    if (spec.flag_enabled && shape_.arming == arming_expectation::armed &&
        (delta.producers_eligible == 0 || delta.offers == 0)) {
      run.validity = {false,
                      "shape expects to arm but the producer did not (eligible/offers did not "
                      "move); its timing would describe a run without the feature"};
      return run;
    }
    if (spec.flag_enabled && shape_.arming == arming_expectation::unarmed && armed) {
      run.validity = {false,
                      "shape expects no producer but one armed -- eligibility creep has brought "
                      "this query into the feature's scope and the no-regression corpus no longer "
                      "covers what it claims to"};
      return run;
    }

    evidence_[static_cast<std::size_t>(arm)].accumulate(
      shape_.producer, counters_before, counters_after);
    return run;
  }

  [[nodiscard]] const arming_evidence& evidence(int arm) const noexcept
  {
    return evidence_[static_cast<std::size_t>(arm)];
  }

 private:
  duckdb::Connection& con_;
  const measurement_shape& shape_;
  std::array<arm_spec, 2> arms_;
  std::array<arming_evidence, 2> evidence_{};
};

/// Per-process scratch database holding the corpus. The Sirius duckdb-native scan needs a
/// single-file block manager, so an in-memory table would fall every query back to DuckDB; the PID
/// suffix keeps concurrent runs on one host off each other's file lock.
struct scoped_measurement_db {
  explicit scoped_measurement_db(duckdb::Connection& c)
    : con(c),
      path(std::filesystem::temp_directory_path() /
           ("sirius_topn_measurement." + std::to_string(::getpid()) + ".duckdb")),
      parquet_path(std::filesystem::temp_directory_path() /
                   ("sirius_topn_measurement." + std::to_string(::getpid()) + ".parquet"))
  {
    auto attach = con.Query("ATTACH IF NOT EXISTS '" + path.string() + "' AS topn_measure_db;");
    REQUIRE(attach);
    REQUIRE_FALSE(attach->HasError());
    auto use_db = con.Query("USE topn_measure_db;");
    REQUIRE(use_db);
    REQUIRE_FALSE(use_db->HasError());
  }
  ~scoped_measurement_db()
  {
    // Best-effort teardown from a destructor: Catch2 assertion macros throw, so results are
    // discarded rather than asserted.
    con.Query("USE memory;");
    con.Query("DETACH topn_measure_db;");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + ".wal", ec);
    std::filesystem::remove(parquet_path, ec);
  }

  scoped_measurement_db(const scoped_measurement_db&)            = delete;
  scoped_measurement_db& operator=(const scoped_measurement_db&) = delete;

  duckdb::Connection& con;
  std::filesystem::path path;
  std::filesystem::path parquet_path;
};

void execute(duckdb::Connection& con, const std::string& statement)
{
  auto result = con.Query(statement);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("statement error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

/// Builds the corpus. `48271` is coprime to the prime `2147483647`, so `(i * 48271) % 2147483647`
/// is injective over `i` in `[1, 20000000]` -- `m_scan.v` is a genuine permutation with no ordering
/// relationship to storage position, and `m_ascending.v` is its opposite by construction.
void create_corpus(duckdb::Connection& con, std::uint64_t corpus_rows)
{
  auto const rows = std::to_string(corpus_rows);

  execute(con,
          "CREATE OR REPLACE TABLE m_scan AS "
          "SELECT CAST(i AS INTEGER) AS id, "
          "       CAST((i * 48271) % 2147483647 AS INTEGER) AS v, "
          "       CAST((i * 48271) % 4099 AS INTEGER) AS w "
          "FROM range(1, " +
            rows + " + 1) t(i);");

  execute(con,
          "CREATE OR REPLACE TABLE m_ascending AS "
          "SELECT CAST(i AS INTEGER) AS id, "
          "       CAST(i AS INTEGER) AS v, "
          "       CAST(i % 4099 AS INTEGER) AS w "
          "FROM range(1, " +
            rows + " + 1) t(i);");

  // 200003 distinct groups: far more than K, so the boundary is a real bound rather than the whole
  // key domain, and the aggregate is large enough that its input is worth pruning.
  execute(con,
          "CREATE OR REPLACE TABLE m_groups AS "
          "SELECT CAST(i AS INTEGER) AS id, "
          "       CAST((i * 48271) % 200003 AS INTEGER) AS grp, "
          "       CAST((i * 48271) % 2147483647 AS INTEGER) AS v "
          "FROM range(1, " +
            rows + " + 1) t(i);");

  // The native decoder reads checkpointed segments, so the fresh tables must leave the WAL.
  execute(con, "CHECKPOINT;");
}

/**
 * @brief Write the parquet corpus: the only backend on which a boundary can avoid reading data
 *
 * The duckdb-native tables above can never show this feature winning wall clock, and that is a
 * property of the code rather than of the corpus: the native scan has no read-time dynamic-filter
 * path, so a published boundary is applied post-decode by a separate
 * `sirius_physical_dynamic_filter` pass over rows that were already read, decoded, and copied to
 * the device -- and `sirius_physical_top_n` has already prefiltered its own input with the
 * identical predicate. The only saving left there is downstream row reduction, against a full extra
 * compaction pass per batch. The native shapes are therefore measured as a *cost* bound, not as a
 * benefit.
 *
 * `parquet_gpu_ingestible` merges the boundary into the reader AST instead, and cuDF prunes row
 * groups against their statistics before reading them, so rows are never fetched or decoded at all.
 * That only helps when row-group statistics can separate rows: the key is written in ascending
 * order here, so each row group covers a narrow, disjoint range of `v` and a boundary near the low
 * end excludes almost all of them. A randomly ordered key -- `m_scan.v` above -- gives every row
 * group min/max spanning the whole domain, and no row-level selectivity, however extreme, prunes a
 * single one.
 *
 * The payload columns are the point of the shape as much as the key is: what a skipped row group
 * saves is its bytes, so a two-column table has almost nothing to save even when every row group is
 * skipped.
 */
void create_parquet_corpus(duckdb::Connection& con,
                           const std::filesystem::path& parquet_path,
                           std::uint64_t corpus_rows)
{
  // A row-group size well below the row count, so one file holds many independently prunable
  // groups; DuckDB writes them in the order the rows are produced, which is `v` ascending.
  execute(con,
          "COPY (SELECT CAST(i AS INTEGER) AS v, "
          "             CAST(i * 7 AS BIGINT) AS p0, CAST(i * 11 AS BIGINT) AS p1, "
          "             CAST(i * 13 AS BIGINT) AS p2, CAST(i * 17 AS BIGINT) AS p3, "
          "             CAST(i * 19 AS BIGINT) AS p4, CAST(i * 23 AS BIGINT) AS p5, "
          "             CAST(i * 29 AS BIGINT) AS p6, CAST(i * 31 AS BIGINT) AS p7 "
          "      FROM range(1, " +
            std::to_string(corpus_rows) + " + 1) t(i)) TO '" + parquet_path.string() +
            "' (FORMAT PARQUET, ROW_GROUP_SIZE 122880);");
}

rows_t query_rows(duckdb::Connection& con, const std::string& query)
{
  auto result = con.Query(query);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  return sirius::test::collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
}

/// Result equivalence across the three legs a measured shape must not disturb: a faster wrong
/// answer is not a win, so C1 evidence for the measured corpus is gathered here rather than assumed
/// from the correctness suite's much smaller tables.
void require_equivalence(duckdb::Connection& con, const std::string& query)
{
  execute(con, "SET gpu_execution = true;");
  execute(con, "SET enable_top_n_dynamic_filter = true;");
  auto const flag_on = query_rows(con, query);
  execute(con, "SET enable_top_n_dynamic_filter = false;");
  auto const flag_off = query_rows(con, query);
  execute(con, "SET gpu_execution = false;");
  auto const cpu = query_rows(con, query);
  execute(con, "SET gpu_execution = true;");

  REQUIRE(flag_on == flag_off);
  REQUIRE(flag_on == cpu);
}

/**
 * @brief Check one shape's results, then time it, both under the shape's own batch size
 *
 * Equivalence is checked at the batching the timings use rather than at the default, because
 * batching is what decides how much a boundary can prune: at the default size these tables scan in
 * two tasks, almost nothing meets a boundary, and a predicate one comparison too strict would
 * return correct results anyway. The batching that makes the measurement interesting is also the
 * batching that makes the correctness check sensitive.
 *
 * @param check_equivalence Whether to run the correctness legs first. The A/A control re-measures a
 *                          shape already checked by its own A/B cell, so it skips them rather than
 *                          paying for a duplicate CPU leg.
 *
 * The report is emitted through `WARN` so that it reaches the console on a passing run: these cases
 * exist to produce numbers, and a number Catch2 only prints on failure is not a deliverable.
 */
void measure(duckdb::Connection& con,
             const measurement_shape& shape,
             std::array<arm_spec, 2> arms,
             const sirius::test::interleaved_ab_plan& plan,
             bool check_equivalence = true)
{
  sirius::test::scoped_setting batch_size(con, "scan_task_batch_size", shape.scan_task_batch_bytes);
  if (check_equivalence) { require_equivalence(con, shape.query); }

  arm_executor executor(con, shape, arms);
  auto const result = sirius::test::run_interleaved_ab(
    {arms[0].name, arms[1].name}, [&executor](int arm) { return executor(arm); }, plan);

  std::string report = "\n=== " + shape.name + " ===\n  " + shape.intent +
                       "\n  query: " + shape.query + "\n" + result.report();
  report += "  " + arms[0].name + " counters: " + executor.evidence(0).describe() + "\n";
  report += "  " + arms[1].name + " counters: " + executor.evidence(1).describe() + "\n";
  if (!result.conditions_clean()) {
    report +=
      "  CONDITIONS NOT CLEAN -- this cell's numbers are not evidence; see discards and gpu "
      "bracket above\n";
  }
  WARN(report);

  // A cell that kept no pairs measured nothing. `CHECK` rather than `REQUIRE` so one unmeasurable
  // shape is recorded as a failure without stopping the cells after it -- the corpus is built once
  // for the whole run and a shape that cannot arm is itself a result worth having alongside the
  // others.
  CHECK(result.arms[0].size() > 0);
  CHECK(result.arms[1].size() > 0);
}

}  // namespace

//! Wall-clock cost and benefit of the Top-N dynamic filter. Hidden from the default suite: this is
//! a measurement whose output is its report, not a pass/fail gate on behaviour.
TEST_CASE("top-n dynamic filter measurement - interleaved feature-off vs feature-on",
          "[.][integration][gpu_execution][dynamic_filter][measurement]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  scoped_measurement_db scratch_db(con);

  auto const corpus_rows = env_value("SIRIUS_TOPN_MEASURE_ROWS", 20'000'000);
  // The effect the corpus has to be able to see. Enable-by-default is argued in single-digit
  // percentages -- a no-regression claim in particular is worthless if the interval behind it is
  // wider than the regression it is meant to exclude -- so the pair count is derived from this
  // rather than chosen, and a cell that cannot reach it says so instead of reporting a number.
  auto const resolution =
    static_cast<double>(env_value("SIRIUS_TOPN_MEASURE_RESOLUTION_BP", 200)) / 10000.0;
  auto const min_pairs   = static_cast<int>(env_value("SIRIUS_TOPN_MEASURE_MIN_PAIRS", 21));
  auto const max_pairs   = static_cast<int>(env_value("SIRIUS_TOPN_MEASURE_MAX_PAIRS", 401));
  auto const batch_bytes = std::max<std::uint64_t>(
    4ULL << 20U, corpus_rows * k_projected_bytes_per_row / k_target_batches_per_query);
  auto const parquet_batch_bytes = std::max<std::uint64_t>(
    4ULL << 20U, corpus_rows * k_parquet_projected_bytes_per_row / k_target_batches_per_query);
  create_corpus(con, corpus_rows);
  create_parquet_corpus(con, scratch_db.parquet_path, corpus_rows);
  auto const parquet_scan = "read_parquet('" + scratch_db.parquet_path.string() + "')";

  auto const opening = sirius::test::observe_gpu_occupancy();
  WARN("\ngpu occupancy before the corpus was measured: " + opening.describe() +
       "\ncorpus rows per table: " + std::to_string(corpus_rows) + "\nscan_task_batch_size: " +
       std::to_string(batch_bytes) + " bytes native, " + std::to_string(parquet_batch_bytes) +
       " bytes parquet (about " + std::to_string(k_target_batches_per_query) +
       " batches per query)" + "\ntarget resolution: +/-" +
       sirius::test::detail::fixed(resolution * 100.0, 2) + "% (pairs per cell derived, " +
       std::to_string(min_pairs) + " to " + std::to_string(max_pairs) + ")\n");

  execute(con, "SET gpu_execution = true;");

  arm_spec const flag_off{"flag-off", false};
  arm_spec const flag_on{"flag-on", true};
  sirius::test::interleaved_ab_plan const plan{.warmup_pairs      = 3,
                                               .target_resolution = resolution,
                                               .min_pairs         = min_pairs,
                                               .max_pairs         = max_pairs};

  measurement_shape const s_scan{
    .name = "S-scan",
    .intent =
      "single-key Top-N over a permuted key; the boundary tightens early and should prune "
      "most of the scan",
    .query                 = "SELECT id, v FROM m_scan ORDER BY v LIMIT " + std::to_string(k_limit),
    .producer              = producer_kind::row,
    .expected_rows         = k_limit,
    .scan_task_batch_bytes = batch_bytes};

  measurement_shape const s_lex{
    .name = "S-lex",
    .intent =
      "two-key Top-N over the same table; exercises the LEX scan target and the "
      "prefix-disjunction cost the single-key shape never pays",
    .query         = "SELECT id, v, w FROM m_scan ORDER BY v, w LIMIT " + std::to_string(k_limit),
    .producer      = producer_kind::row,
    .expected_rows = k_limit,
    .scan_task_batch_bytes = batch_bytes};

  measurement_shape const s_group{
    .name = "S-group",
    .intent =
      "grouping-key order over 200003 groups; the group-key producer prunes the "
      "aggregate's input",
    .query = "SELECT grp, min(v) AS m FROM m_groups GROUP BY grp ORDER BY grp LIMIT " +
             std::to_string(k_limit),
    .producer              = producer_kind::group,
    .expected_rows         = k_limit,
    .scan_task_batch_bytes = batch_bytes};

  measurement_shape const s_adversary{
    .name = "S-adversary",
    .intent =
      "key ascending in storage order, ordered descending: every batch beats the previous "
      "boundary, so publication runs continuously and prunes nothing -- the bound on what "
      "the feature costs when it cannot help",
    .query    = "SELECT id, v FROM m_ascending ORDER BY v DESC LIMIT " + std::to_string(k_limit),
    .producer = producer_kind::row,
    .expected_rows         = k_limit,
    .scan_task_batch_bytes = batch_bytes};

  // The shape class a no-regression claim is actually about: no Top-N anywhere in the plan, so no
  // producer exists to arm and the feature should cost exactly nothing. Most of a TPC-H corpus
  // looks like this to this feature, and a harness that could not measure it could not answer
  // "does enabling the flag hurt the queries it does not help".
  measurement_shape const s_untouched{
    .name = "S-untouched",
    .intent =
      "no Top-N in the plan at all: the flag must be free here, and the counters must "
      "confirm nothing armed rather than the timing being taken on trust",
    .query                 = "SELECT count(*) AS c, sum(v) AS s FROM m_scan",
    .producer              = producer_kind::row,
    .expected_rows         = 1,
    .scan_task_batch_bytes = batch_bytes,
    .arming                = arming_expectation::unarmed};

  std::string const parquet_payload = "v, p0, p1, p2, p3, p4, p5, p6, p7";

  measurement_shape const s_parquet_clustered{
    .name = "S-parquet-clustered",
    .intent =
      "parquet whose key is written in ascending order, with a wide payload: the boundary reaches "
      "the reader AST and cuDF drops whole row groups on their statistics, so the rows are never "
      "read or decoded -- the only mechanism in the engine that turns a boundary into avoided work",
    .query = "SELECT " + parquet_payload + " FROM " + parquet_scan + " ORDER BY v LIMIT " +
             std::to_string(k_limit),
    .producer              = producer_kind::row,
    .expected_rows         = k_limit,
    .scan_task_batch_bytes = parquet_batch_bytes};

  measurement_shape const s_parquet_adversary{
    .name = "S-parquet-adversary",
    .intent =
      "the same file ordered descending: the boundary starts at the low end of an ascending key, "
      "so no row group can be excluded ahead of the scan and the reader-side filter is pure cost",
    .query = "SELECT " + parquet_payload + " FROM " + parquet_scan + " ORDER BY v DESC LIMIT " +
             std::to_string(k_limit),
    .producer              = producer_kind::row,
    .expected_rows         = k_limit,
    .scan_task_batch_bytes = parquet_batch_bytes};

  // Cells run in one body rather than in Catch2 sections, because a section re-executes the whole
  // body and would rebuild the corpus for each cell. Every cell interleaves its own arms, so the
  // order cells run in cannot bias any of their ratios.

  // A/A first. Nothing distinguishes its arms, so whatever separation it reports is the
  // instrument's own noise band; no A/B ratio narrower than this band is an effect, and it doubles
  // as extra warm-up for the cells after it.
  measure(con, s_scan, {flag_off, arm_spec{"flag-off (control)", false}}, plan, false);

  // A shape the feature never touches, measured rather than assumed to be free.
  measure(con, s_untouched, {flag_off, flag_on}, plan);

  // The adversary next: it is the shape enable-by-default hinges on, so it is measured while the
  // opening GPU bracket is still fresh rather than after the winning shapes.
  measure(con, s_adversary, {flag_off, flag_on}, plan);
  measure(con, s_scan, {flag_off, flag_on}, plan);
  measure(con, s_lex, {flag_off, flag_on}, plan);
  measure(con, s_group, {flag_off, flag_on}, plan);

  // The parquet pair last, because it is the only pair that can show a benefit: everything above
  // measures what the feature costs on a backend that cannot skip reads.
  measure(con, s_parquet_clustered, {flag_off, flag_on}, plan);
  measure(con, s_parquet_adversary, {flag_off, flag_on}, plan);

  auto const closing = sirius::test::observe_gpu_occupancy();
  WARN("\ngpu occupancy after the corpus was measured: " + closing.describe() + "\n");
}

/**
 * @brief Whether the parquet reader survives repeated reads under a published Top-N boundary
 *
 * Separate from the measurement above because its result is delivered by the process surviving or
 * not, which no assertion inside the process can report. A boundary published to a parquet scan
 * becomes a `cudf::ast` predicate handed to `cudf::io::read_parquet`, and a fault while cuDF walks
 * that predicate kills the run before any timing is written -- so the arms have to be separated
 * into different processes and compared by exit status rather than by a counter.
 *
 * Run it twice and compare:
 *
 *     SIRIUS_TOPN_STRESS_FLAG=off sirius_unittest "[topn_parquet_stress]"
 *     SIRIUS_TOPN_STRESS_FLAG=on  sirius_unittest "[topn_parquet_stress]"
 *
 * Identical queries, identical file, identical iteration count; the only difference is whether a
 * boundary reaches the reader. If the `off` process completes and the `on` process faults, the
 * boundary is the cause, and no further attribution argument is needed.
 */
TEST_CASE("top-n dynamic filter measurement - parquet reader stability under a published boundary",
          "[.][integration][gpu_execution][dynamic_filter][measurement][topn_parquet_stress]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  scoped_measurement_db scratch_db(con);

  auto const corpus_rows = env_value("SIRIUS_TOPN_MEASURE_ROWS", 20'000'000);
  auto const iterations  = static_cast<int>(env_value("SIRIUS_TOPN_STRESS_ITERATIONS", 200));
  char const* flag_env   = std::getenv("SIRIUS_TOPN_STRESS_FLAG");
  bool const flag_on     = flag_env == nullptr || std::string{flag_env} != "off";
  // Descending over an ascending key republishes the boundary on nearly every batch instead of
  // settling after the first few, so it drives the publish-during-read window this case exists to
  // probe far harder than the ascending order does.
  char const* order_env = std::getenv("SIRIUS_TOPN_STRESS_ORDER");
  bool const descending = order_env != nullptr && std::string{order_env} == "desc";

  create_parquet_corpus(con, scratch_db.parquet_path, corpus_rows);
  auto const batch_bytes = std::max<std::uint64_t>(
    4ULL << 20U, corpus_rows * k_parquet_projected_bytes_per_row / k_target_batches_per_query);
  sirius::test::scoped_setting batch_size(con, "scan_task_batch_size", batch_bytes);
  execute(con, "SET gpu_execution = true;");
  execute(con,
          std::string{"SET enable_top_n_dynamic_filter = "} + (flag_on ? "true" : "false") + ";");

  std::string const query = "SELECT v, p0, p1, p2, p3, p4, p5, p6, p7 FROM read_parquet('" +
                            scratch_db.parquet_path.string() + "') ORDER BY v" +
                            (descending ? " DESC" : "") + " LIMIT " + std::to_string(k_limit);
  WARN("\nparquet stability: flag " + std::string{flag_on ? "on" : "off"} + ", order " +
       std::string{descending ? "desc" : "asc"} + ", " + std::to_string(iterations) +
       " iterations over " + std::to_string(corpus_rows) + " rows\nquery: " + query + "\n");

  auto const before = sirius::test::get_dynamic_filter_stats_snapshot(con);
  for (int i = 0; i < iterations; ++i) {
    auto result = con.Query(query);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("iteration " << i << ": " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->RowCount() == static_cast<duckdb::idx_t>(k_limit));
  }
  auto const after = sirius::test::get_dynamic_filter_stats_snapshot(con);

  // The flag-off arm has to be shown to be a real control rather than an accidentally identical
  // run, and the flag-on arm has to be shown to have actually reached the reader; otherwise a
  // surviving `on` process would prove only that no boundary was ever published.
  auto const pushed =
    (after.top_n_first_key_filters_pushed - before.top_n_first_key_filters_pushed) +
    (after.top_n_lex_filters_pushed - before.top_n_lex_filters_pushed);
  WARN("\ncompleted " + std::to_string(iterations) + " iterations; offers +" +
       std::to_string(after.top_n_offers - before.top_n_offers) + ", filters pushed to scans +" +
       std::to_string(pushed) + "\n");
  if (flag_on) {
    REQUIRE(after.top_n_offers > before.top_n_offers);
  } else {
    REQUIRE(after.top_n_offers == before.top_n_offers);
  }
}
