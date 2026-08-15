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
 * @file test_twin_scan_fusion.cpp
 * @brief Contract tests for the twin-scan fusion pass (`fuse_twin_scans`): the multi-key
 *        positive match, the `fuse_twin_scans` setting toggle, one pinned rejection reason
 *        per refusal shape from the `twin_scan_fusion_report`, and GPU execution equivalence
 *        between the fused and unfused plans.
 *
 * Every case follows the positive-control discipline: before asserting the fusion outcome it
 * asserts the precondition shape it claims to exercise, so a DuckDB plan-shape drift fails the
 * test loudly instead of letting a refusal pass vacuously. The canonical schema mirrors TPC-H
 * q21's structure without its names: chained EXISTS / NOT EXISTS over one table, correlated on
 * an equality (k) plus a non-equality (s) -- a multi-column delim key tuple -- with the NOT
 * EXISTS side carrying the residual predicate d2 > d1.
 */

#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "plan_test_harness.hpp"
#include "planner/sirius_plan_twin_scan_fusion.hpp"
#include "sirius_engine.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/pipeline_conversion_test_utils.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::planner::twin_scan_fusion_report;
using sirius::planner::twin_scan_rejection_reason;
using sirius::test::collect;
using sirius::test::find_first;
using sirius::test::generate_sirius_plan;
using sirius::test::scoped_temp_db_path;
using sirius::test::tree_to_string;

namespace {

// Column declaration order is load-bearing: the prefix property compares column_ids in table
// order, so the EXISTS side's columns {k, s} must precede the NOT EXISTS side's residual-only
// columns {d1, d2}; `pad` exists to break the prefix in the non-prefix refusal case.
constexpr const char* kCreateTwinTable =
  "CREATE TABLE t(k BIGINT, s BIGINT, d1 DATE, d2 DATE, pad INTEGER)";

// Deterministic rows (range + modular values) so plans are stable: enough distinct k groups
// for the delim distinct to matter, s varying within each k group so the <> correlations are
// selective, and d1/d2 phases making the residual d2 > d1 keep roughly half the rows.
constexpr const char* kPopulateTwinTable =
  "INSERT INTO t SELECT range % 50, range % 7, DATE '2024-01-01' + (range % 30)::INTEGER, "
  "DATE '2024-01-01' + (range % 45)::INTEGER, (range % 11)::INTEGER FROM range(400)";

// The positive query: q21's structure with no TPC-H names. A = the bare EXISTS scan needing
// {k, s}; B = the NOT EXISTS scan needing {k, s, d1, d2} with the residual d2 > d1. The
// correlation is the multi-key delim tuple (equality on k, <> on s). The OUTER residual
// o.d2 > o.d1 (q21's own l1 filter) is load-bearing: without it the deliminator flattens the
// EXISTS into a plain semi join and no delim chain exists to fuse over.
constexpr const char* kPositiveQuery =
  "SELECT count(*) FROM t o "
  "WHERE o.d2 > o.d1 "
  "  AND EXISTS     (SELECT 1 FROM t i WHERE i.k = o.k AND i.s <> o.s) "
  "  AND NOT EXISTS (SELECT 1 FROM t j WHERE j.k = o.k AND j.s <> o.s AND j.d2 > j.d1)";

/// `fuse_twin_scans` is a per-connection DuckDB setting, so the toggle is driven with
/// `SET`/`RESET` on the connection the plan is generated on.
class twin_fusion_flag_guard {
 public:
  twin_fusion_flag_guard(duckdb::Connection& con, bool enabled) : _con(con)
  {
    _con.Query(std::string("SET fuse_twin_scans = ") + (enabled ? "true" : "false") + ";");
  }

  ~twin_fusion_flag_guard() { _con.Query("RESET fuse_twin_scans;"); }

  twin_fusion_flag_guard(const twin_fusion_flag_guard&)            = delete;
  twin_fusion_flag_guard& operator=(const twin_fusion_flag_guard&) = delete;

 private:
  duckdb::Connection& _con;
};

bool contains_reason(const twin_scan_fusion_report& report, twin_scan_rejection_reason reason)
{
  return std::any_of(report.same_table_rejections.begin(),
                     report.same_table_rejections.end(),
                     [&](const auto& rejection) { return rejection.reason == reason; });
}

std::string report_to_string(const twin_scan_fusion_report& report)
{
  std::string out = "fused_pairs=" + std::to_string(report.fused_pairs) + " rejections=[";
  for (const auto& rejection : report.same_table_rejections) {
    out += std::string(sirius::planner::to_string(rejection.reason)) + " ";
  }
  out += "]";
  return out;
}

/// Whether any delim join in the plan carries a distinct over at least two group columns --
/// the positive control that the correlation produced a MULTI-key delim tuple (the shipped
/// single-key-only draft bug would have made the positive case below pass for the wrong
/// reason without this).
bool has_multi_key_delim_distinct(sirius_physical_operator* root)
{
  bool found = false;
  sirius::test::for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type != SiriusPhysicalOperatorType::LEFT_DELIM_JOIN &&
        op->type != SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
      return;
    }
    auto& delim = op->Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.distinct != nullptr && delim.distinct->group_idx.size() >= 2) { found = true; }
  });
  return found;
}

struct twin_scan_fusion_fixture {
  twin_scan_fusion_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    con->Query(kCreateTwinTable);
    con->Query(kPopulateTwinTable);
  }

  ~twin_scan_fusion_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  /// Plan @p query on the fixture connection and capture the generator's fusion report.
  duckdb::unique_ptr<sirius_physical_operator> plan_with_report(const std::string& query,
                                                                twin_scan_fusion_report& report)
  {
    return generate_sirius_plan(*con, query, &report);
  }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - multi-key tuple pair fuses into split + ref",
                 "[twin_scan_fusion][isolated_context]")
{
  // Knob-off baseline: the twin shape must exist unfused before the knob-on plan may claim a
  // fusion (positive control against plan-shape drift).
  std::size_t unfused_scan_count = 0;
  {
    twin_fusion_flag_guard guard(*con, /*enabled=*/false);
    twin_scan_fusion_report report;
    auto plan = plan_with_report(kPositiveQuery, report);
    INFO(tree_to_string(plan.get()));
    REQUIRE(report.fused_pairs == 0);
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).empty());

    // Producer-backed channels on at least the two twin scans: each is rewritten into a
    // GPU_SCAN wrapped by a DYNAMIC_FILTER only when its channel has producers.
    REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).size() >= 2);
    // The multi-key positive control: the correlation (k equality + s <>) must have produced
    // a delim distinct over >= 2 group columns.
    REQUIRE(has_multi_key_delim_distinct(plan.get()));

    unfused_scan_count = collect(plan.get(), SiriusPhysicalOperatorType::GPU_SCAN).size();
    REQUIRE(unfused_scan_count >= 3);  // o, i, and j all scan t
  }

  twin_fusion_flag_guard guard(*con, /*enabled=*/true);
  twin_scan_fusion_report report;
  auto plan = plan_with_report(kPositiveQuery, report);
  INFO(tree_to_string(plan.get()));
  INFO(report_to_string(report));

  REQUIRE(report.fused_pairs == 1);
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).size() == 1);
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_REF).size() == 1);
  // The fused plan decodes t one time fewer.
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::GPU_SCAN).size() == unfused_scan_count - 1);
  // The fused pipeline shape: the split consumes the shared GPU_SCAN -> DYNAMIC_FILTER stream.
  auto* split = find_first(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT);
  REQUIRE(split != nullptr);
  CHECK(find_first(split, SiriusPhysicalOperatorType::GPU_SCAN) != nullptr);
  CHECK(find_first(split, SiriusPhysicalOperatorType::DYNAMIC_FILTER) != nullptr);
  CHECK(has_multi_key_delim_distinct(plan.get()));
}

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - the fuse_twin_scans setting toggles the pass",
                 "[twin_scan_fusion][isolated_context]")
{
  {
    twin_fusion_flag_guard guard(*con, /*enabled=*/false);
    twin_scan_fusion_report report;
    auto plan = plan_with_report(kPositiveQuery, report);
    INFO(tree_to_string(plan.get()));
    // Knob off: the pass never ran, so the report is the default -- no fusions AND no
    // rejection records.
    CHECK(report.fused_pairs == 0);
    CHECK(report.same_table_rejections.empty());
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).empty());
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_REF).empty());
  }
  {
    twin_fusion_flag_guard guard(*con, /*enabled=*/true);
    twin_scan_fusion_report report;
    auto plan = plan_with_report(kPositiveQuery, report);
    INFO(tree_to_string(plan.get()));
    CHECK(report.fused_pairs == 1);
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).size() == 1);
  }
}

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - refusal: multi-target channel",
                 "[twin_scan_fusion][isolated_context]")
{
  // Two single-key equality joins into the outer scan make o's channel plan filters on two
  // columns -- q21's own l2 x l1 near-miss shape, where the outer's residual-filtered scan is
  // itself a candidate B whose channel is fed by two unrelated producer joins. The u1/u2
  // predicates matter: producers are only wired for filtered (or derived) build sides, since
  // an unfiltered base-table build covers the whole key domain. They filter on `w` (never a
  // join key) so no transitive predicate leaks into the t-scans' static filters. The
  // legitimate (i, j) pair must still fuse; the (i, o) pair must be refused with the pinned
  // reason.
  con->Query("CREATE TABLE u1(k BIGINT, w BIGINT)");
  con->Query("INSERT INTO u1 SELECT range % 50, range FROM range(20)");
  con->Query("CREATE TABLE u2(s BIGINT, w BIGINT)");
  con->Query("INSERT INTO u2 SELECT range % 7, range FROM range(20)");

  const std::string query =
    "SELECT count(*) FROM t o JOIN u1 ON o.k = u1.k JOIN u2 ON o.s = u2.s "
    "WHERE o.d2 > o.d1 AND u1.w < 100 AND u2.w < 100 "
    "  AND EXISTS     (SELECT 1 FROM t i WHERE i.k = o.k AND i.s <> o.s) "
    "  AND NOT EXISTS (SELECT 1 FROM t j WHERE j.k = o.k AND j.s <> o.s AND j.d2 > j.d1)";

  twin_fusion_flag_guard guard(*con, /*enabled=*/true);
  twin_scan_fusion_report report;
  auto plan = plan_with_report(query, report);
  INFO(tree_to_string(plan.get()));
  INFO(report_to_string(report));

  // Positive control: besides the fused pipeline's shared channel, o's scan carries its own
  // producer-backed channel (a scan is wrapped by a DYNAMIC_FILTER only then) -- so the
  // multi-target rejection below judged a real two-producer channel, not a missing one.
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).size() >= 2);

  // The rejection reason is pinned: geometry passes (o's {k, s} is a prefix of j's columns),
  // so the pair fails at the channel checks, on the multi-target condition.
  CHECK(contains_reason(report, twin_scan_rejection_reason::channel_multi_target));
  // fused_pairs counts only the legitimate pair.
  CHECK(report.fused_pairs == 1);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).size() == 1);
}

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - refusal: non-nested key sets",
                 "[twin_scan_fusion][isolated_context]")
{
  // The chain is broken: the subqueries correlate on different keys (equality on k vs
  // equality on s), so keys(B) within keys(A) is not provable. j references k before s so
  // both scans keep the declared column order and the pair survives the geometry stage --
  // this case must fail in the PROOF stage.
  const std::string query =
    "SELECT count(*) FROM t o "
    "WHERE o.d2 > o.d1 "
    "  AND EXISTS     (SELECT 1 FROM t i WHERE i.k = o.k AND i.s <> o.s) "
    "  AND NOT EXISTS (SELECT 1 FROM t j WHERE j.k <> o.k AND j.s = o.s AND j.d2 > j.d1)";

  twin_fusion_flag_guard guard(*con, /*enabled=*/true);
  twin_scan_fusion_report report;
  auto plan = plan_with_report(query, report);
  INFO(tree_to_string(plan.get()));
  INFO(report_to_string(report));

  // Positive control: the twin shape (two producer-backed channels, a residual FILTER) still
  // exists; only the proof fails.
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).size() >= 2);
  REQUIRE(!collect(plan.get(), SiriusPhysicalOperatorType::FILTER).empty());

  CHECK(report.fused_pairs == 0);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).empty());
  // Pinned by the check order: the diverging keys surface first at the channel obligations
  // (i's channel targets k, j's targets s), before the delim-chain walk would also fail.
  CHECK(contains_reason(report, twin_scan_rejection_reason::channel_targets_differ));
}

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - refusal: multi-equality producer join",
                 "[twin_scan_fusion][isolated_context]")
{
  // The NOT EXISTS side correlates on two equalities (k and s, plus a <> on d1 keeping the
  // delim chain retained): its channel plans filters for both key columns, which the channel
  // checks refuse before the single-equality producer check is ever reached.
  const std::string query =
    "SELECT count(*) FROM t o "
    "WHERE o.d2 > o.d1 "
    "  AND EXISTS     (SELECT 1 FROM t i WHERE i.k = o.k AND i.s <> o.s) "
    "  AND NOT EXISTS (SELECT 1 FROM t j WHERE j.k = o.k AND j.s = o.s AND j.d1 <> o.d1 "
    "                  AND j.d2 > j.d1)";

  twin_fusion_flag_guard guard(*con, /*enabled=*/true);
  twin_scan_fusion_report report;
  auto plan = plan_with_report(query, report);
  INFO(tree_to_string(plan.get()));
  INFO(report_to_string(report));

  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).size() >= 2);

  CHECK(report.fused_pairs == 0);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).empty());
  // Pinned by the check order: the two-equality correlation surfaces as a two-column channel
  // before the single-equality producer check could see it.
  CHECK(contains_reason(report, twin_scan_rejection_reason::channel_multi_target));
}

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - refusal: columns not a strict prefix",
                 "[twin_scan_fusion][isolated_context]")
{
  // The EXISTS side additionally needs `pad` (declared AFTER d1/d2), so A's columns
  // {k, s, pad} are not a prefix of B's {k, s, d1, d2}.
  const std::string query =
    "SELECT count(*) FROM t o "
    "WHERE o.d2 > o.d1 "
    "  AND EXISTS     (SELECT 1 FROM t i WHERE i.k = o.k AND i.s <> o.s AND i.pad = o.pad) "
    "  AND NOT EXISTS (SELECT 1 FROM t j WHERE j.k = o.k AND j.s <> o.s AND j.d2 > j.d1)";

  twin_fusion_flag_guard guard(*con, /*enabled=*/true);
  twin_scan_fusion_report report;
  auto plan = plan_with_report(query, report);
  INFO(tree_to_string(plan.get()));
  INFO(report_to_string(report));

  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).size() >= 2);

  CHECK(report.fused_pairs == 0);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).empty());
  CHECK(contains_reason(report, twin_scan_rejection_reason::columns_not_strict_prefix));
}

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - refusal: static pushed filters differ",
                 "[twin_scan_fusion][isolated_context]")
{
  // Positive control first: a constant comparison on this schema is pushed into the scan's
  // table_filters, not planned as a FILTER node -- so `j.pad > 0` below lands as a static
  // filter on j's scan only.
  {
    auto control = generate_sirius_plan(*con, "SELECT count(*) FROM t WHERE pad > 0");
    INFO(tree_to_string(control.get()));
    REQUIRE(collect(control.get(), SiriusPhysicalOperatorType::FILTER).empty());
  }

  const std::string query =
    "SELECT count(*) FROM t o "
    "WHERE o.d2 > o.d1 "
    "  AND EXISTS     (SELECT 1 FROM t i WHERE i.k = o.k AND i.s <> o.s) "
    "  AND NOT EXISTS (SELECT 1 FROM t j WHERE j.k = o.k AND j.s <> o.s AND j.d2 > j.d1 "
    "                  AND j.pad > 0)";

  twin_fusion_flag_guard guard(*con, /*enabled=*/true);
  twin_scan_fusion_report report;
  auto plan = plan_with_report(query, report);
  INFO(tree_to_string(plan.get()));
  INFO(report_to_string(report));

  CHECK(report.fused_pairs == 0);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).empty());
  CHECK(contains_reason(report, twin_scan_rejection_reason::static_filters_differ));
}

TEST_CASE_METHOD(twin_scan_fusion_fixture,
                 "twin scan fusion - refusal: both scans carry residuals",
                 "[twin_scan_fusion][isolated_context]")
{
  // Both subqueries carry a residual, so neither site matches the bare A shape and the pair
  // is never collected: no fusion, no crash; an empty rejection list is acceptable because no
  // bare-plus-filtered candidate pair exists.
  const std::string query =
    "SELECT count(*) FROM t o "
    "WHERE o.d2 > o.d1 "
    "  AND EXISTS     (SELECT 1 FROM t i WHERE i.k = o.k AND i.s <> o.s AND i.d2 > i.d1) "
    "  AND NOT EXISTS (SELECT 1 FROM t j WHERE j.k = o.k AND j.s <> o.s AND j.d2 > j.d1)";

  twin_fusion_flag_guard guard(*con, /*enabled=*/true);
  twin_scan_fusion_report report;
  auto plan = plan_with_report(query, report);
  INFO(tree_to_string(plan.get()));
  INFO(report_to_string(report));

  // Positive control: both residual FILTERs exist over producer-backed scan pipelines.
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::FILTER).size() >= 2);
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).size() >= 2);

  CHECK(report.fused_pairs == 0);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT).empty());
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TWIN_SCAN_REF).empty());
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "twin scan fusion - GPU execution equivalence between fused and unfused plans",
                 "[twin_scan_fusion][integration][gpu_execution]")
{
  run_ok(kCreateTwinTable);
  // The residual d2 > d1 is tied to s so the NOT EXISTS stays satisfiable AND selective: with
  // residual-passing rows spread across every s value, every k group would contain a
  // different-s passer and the count would degenerate to zero (guarded below). Here only
  // s = 0 passes everywhere and s = 1 passes in half the k groups, so the anti join keeps the
  // s = 0 rows of the other half (358 of 5000) and genuinely drops the rest (715 rows).
  run_ok(
    "INSERT INTO t SELECT range % 50, range % 7, DATE '2024-01-01' + 10, DATE '2024-01-01' + "
    "(CASE WHEN range % 7 = 0 THEN 20 WHEN range % 7 = 1 AND range % 50 < 25 THEN 20 ELSE 5 "
    "END)::INTEGER, (range % 11)::INTEGER FROM range(5000)");
  run_ok("CHECKPOINT");

  // Positive control: under this connection's production planning the query actually fuses --
  // otherwise the off/on comparison below would compare two identical unfused runs and never
  // exercise the converter wiring, the PARTIAL edges, or sink() routing.
  {
    run_ok("SET fuse_twin_scans = true;");
    sirius::test::with_initialized_engine(*con, kPositiveQuery, [&](sirius::sirius_engine& engine) {
      bool has_split = false;
      for (const auto& pipeline : engine.new_scheduled) {
        for (const auto& op : pipeline->get_operators()) {
          if (op.get().type == SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT) { has_split = true; }
        }
      }
      REQUIRE(has_split);
    });
  }

  auto run_on_gpu = [&](bool fused) {
    run_ok(std::string("SET fuse_twin_scans = ") + (fused ? "true" : "false") + ";");
    // Proves the run executed on the GPU with no fallback AND matches DuckDB CPU -- a
    // plan-time throw would silently fall back and make the off/on comparison vacuous.
    compare_gpu_vs_cpu(kPositiveQuery);
    run_ok("SET gpu_execution = true;");
    auto result = con->Query(kPositiveQuery);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    return collect_rows(result->Cast<duckdb::MaterializedQueryResult>());
  };

  auto rows_unfused = run_on_gpu(/*fused=*/false);
  auto rows_fused   = run_on_gpu(/*fused=*/true);
  con->Query("RESET fuse_twin_scans;");

  REQUIRE(rows_fused == rows_unfused);
  // Guard against a degenerate dataset: the count must be a real, non-zero survivor set.
  REQUIRE(rows_fused.size() == 1);
  REQUIRE(std::stoll(rows_fused[0][0]) > 0);
}
