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

// Runtime coverage for `enable_runtime_size_estimation`. test_data_size_estimator.cpp
// covers the estimator's arithmetic against a synthetic DAG; this covers the wiring —
// real scans feeding real pipeline_memory_history, a real PARTITION consuming the
// projection, and a real partition count coming out.
//
// The correctness stake: a partition count is frozen once chosen, since `murmur3(key) % N`
// decides a key's bucket and a later N would split one group across two buckets and emit
// it twice. So a projection that picks a different N than the measured path must still
// produce the same *rows* — hence the comparisons against DuckDB CPU below.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using sirius::test::get_size_estimation_stats;

// Wide-ish rows and enough of them that a scan reports a meaningful byte total and the
// group-by has real work to partition. `g` has 512 distinct values so the partition has
// something to spread; `payload` pads the row so the projected byte total is not noise.
class SizeEstimationFixture : public sirius::test::GpuExecutionFixture {
 public:
  SizeEstimationFixture()
  {
    run_ok(
      "CREATE TABLE facts AS "
      "SELECT (i % 512)::INTEGER AS g, "
      "       (i % 97)::INTEGER  AS k, "
      "       (i * 3)::BIGINT    AS v, "
      "       repeat('x', 48)    AS payload "
      "FROM range(200000) t(i);");
    run_ok(
      "CREATE TABLE dims AS "
      "SELECT i::INTEGER AS k, ('d' || i) AS label FROM range(97) t(i);");
    run_ok("CHECKPOINT;");
  }

  void set_estimation(bool on)
  {
    run_ok("SET gpu_execution = true;");
    run_ok(std::string("SET enable_runtime_size_estimation = ") + (on ? "true;" : "false;"));
  }

  /// Run @p query on the GPU and return its rows sorted, so two runs can be compared as
  /// multisets. Asserts the query really executed on the GPU rather than falling back.
  std::vector<std::vector<std::string>> gpu_rows(const std::string& query)
  {
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(query);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU execution error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    auto const after = sirius::test::get_transparent_execution_stats(*con);
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    return collect_rows(result->Cast<duckdb::MaterializedQueryResult>(), /*sort=*/true);
  }

  /// Partitions that fixed their count from a number the estimator produced, by either
  /// anchor. Which of the two a given run lands on is a race between the scan finishing
  /// and the partition asking, so no test can pin it down; that they sum to non-zero is
  /// the timing-independent statement of "the estimator produced a usable answer".
  static uint64_t estimator_driven(const duckdb::SiriusContext::size_estimation_stats& s)
  {
    return s.partitions_sized_from_projection + s.partitions_sized_from_upstream_complete;
  }
};

constexpr const char* kGroupByQuery =
  "SELECT g, COUNT(*) AS n, SUM(v) AS total FROM facts GROUP BY g";

constexpr const char* kGroupByOverJoinQuery =
  "SELECT d.label, COUNT(*) AS n, SUM(f.v) AS total "
  "FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.label";

// A DISTINCT aggregate, which the planner lowers through a delim join whose distinct root is a
// PARTITION of its own. That partition's tree parent is a grouped_aggregate_merge — typed
// MERGE_GROUP_BY, exactly like the one above — but it is never constructed with estimation
// enabled, so resolve_barrier must leave its ingress FULL under both settings. Covered here
// because the type test alone cannot tell the two shapes apart.
constexpr const char* kDistinctAggregateQuery =
  "SELECT g, COUNT(DISTINCT k) AS ks, SUM(v) AS total FROM facts GROUP BY g";

}  // namespace

TEST_CASE_METHOD(SizeEstimationFixture,
                 "gpu_execution grouped aggregation sizes from an estimate when enabled",
                 "[integration][gpu_execution][size_estimation]")
{
  // With the feature off, the partition has no projection to consult and sizes from the
  // bytes it has drained — the pre-feature behaviour.
  set_estimation(false);
  auto const before_off = get_size_estimation_stats(*con);
  auto off_result       = con->Query(kGroupByQuery);
  REQUIRE(off_result);
  REQUIRE_FALSE(off_result->HasError());
  auto const after_off = get_size_estimation_stats(*con);

  CHECK(after_off.partitions_sized_from_measured > before_off.partitions_sized_from_measured);
  CHECK(estimator_driven(after_off) == estimator_driven(before_off));

  // With it on, the same query's partition sizes from the estimator instead. This is the
  // assertion that would have caught a silently-disengaged feature: every other signal
  // (results, timing, absence of errors) looks identical either way.
  set_estimation(true);
  auto const before_on = get_size_estimation_stats(*con);
  auto on_result       = con->Query(kGroupByQuery);
  REQUIRE(on_result);
  REQUIRE_FALSE(on_result->HasError());
  auto const after_on = get_size_estimation_stats(*con);

  CHECK(estimator_driven(after_on) > estimator_driven(before_on));
  CHECK(after_on.partitions_sized_from_measured == before_on.partitions_sized_from_measured);
}

TEST_CASE_METHOD(SizeEstimationFixture,
                 "gpu_execution grouped aggregation is correct with size estimation on",
                 "[integration][gpu_execution][size_estimation]")
{
  set_estimation(true);
  // A wrong N duplicates group rows rather than erroring, so the row-level comparison
  // against CPU is the real check here.
  compare_gpu_vs_cpu(kGroupByQuery);
  compare_gpu_vs_cpu("SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM facts GROUP BY g");
  // A filtered scan: the projection is scaled by a learned ratio rather than read off
  // scan metadata, so this exercises the ratio path rather than the exact-total path.
  compare_gpu_vs_cpu("SELECT g, COUNT(*) AS n FROM facts WHERE v % 7 = 0 GROUP BY g");
}

TEST_CASE_METHOD(SizeEstimationFixture,
                 "gpu_execution grouped aggregation above a join is correct with estimation on",
                 "[integration][gpu_execution][size_estimation]")
{
  set_estimation(true);
  // Reaching this group-by's input means walking through the join pipeline — the fan-in
  // branch, which forms its ratio from probe bytes the join counts itself.
  //
  // Engagement is deliberately not asserted: the fan-in ratio is withheld until the build
  // side is complete and until min_fan_in_ratio_samples join tasks have finished, and on a
  // table this size the join may finish before either condition is interesting. What must
  // hold regardless is that the answer is right.
  compare_gpu_vs_cpu(kGroupByOverJoinQuery);
  compare_gpu_vs_cpu(
    "SELECT d.label, COUNT(DISTINCT f.g) AS groups "
    "FROM facts f JOIN dims d ON f.k = d.k WHERE f.v > 100000 GROUP BY d.label");
}

TEST_CASE_METHOD(SizeEstimationFixture,
                 "gpu_execution size estimation produces the same rows either way",
                 "[integration][gpu_execution][size_estimation]")
{
  // Direct GPU-vs-GPU comparison. The two settings can legitimately choose different
  // partition counts, and the point of the feature is that this is invisible in the output.
  // Comparing the row multisets (not sets) is deliberate: a group split across two buckets
  // by a stale N comes back as a *duplicated* group row, which set comparison would hide.
  //
  // kDistinctAggregateQuery is here for the opposite reason: its delim-join PARTITION must be
  // untouched by the setting, and a duplicated group row is exactly how a barrier wrongly
  // relaxed on that path would surface.
  for (auto const* query : {kGroupByQuery, kGroupByOverJoinQuery, kDistinctAggregateQuery}) {
    set_estimation(false);
    auto const off = gpu_rows(query);
    set_estimation(true);
    auto const on = gpu_rows(query);

    UNSCOPED_INFO("query: " << query);
    REQUIRE(off.size() == on.size());
    for (std::size_t r = 0; r < off.size(); ++r) {
      REQUIRE(off[r] == on[r]);
    }
  }
}
