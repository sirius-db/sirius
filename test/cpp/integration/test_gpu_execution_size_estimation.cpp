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

// End-to-end coverage for estimator wiring and grouped-aggregation partition sizing.
// Changing a chosen partition count can split a group across buckets, so tests compare rows,
// not only successful execution.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

// Use enough distinct, wide rows to exercise partition sizing.
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

  /// Run on the GPU and return sorted rows for multiset comparison.
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
};

constexpr const char* kGroupByQuery =
  "SELECT g, COUNT(*) AS n, SUM(v) AS total FROM facts GROUP BY g";

constexpr const char* kGroupByOverJoinQuery =
  "SELECT d.label, COUNT(*) AS n, SUM(f.v) AS total "
  "FROM facts f JOIN dims d ON f.k = d.k GROUP BY d.label";

// Its delim-join partition also sits below MERGE_GROUP_BY but must not enable estimation.
constexpr const char* kDistinctAggregateQuery =
  "SELECT g, COUNT(DISTINCT k) AS ks, SUM(v) AS total FROM facts GROUP BY g";

}  // namespace

TEST_CASE_METHOD(SizeEstimationFixture,
                 "gpu_execution grouped aggregation is correct with size estimation on",
                 "[integration][gpu_execution][size_estimation]")
{
  set_estimation(true);
  // A wrong partition count duplicates groups rather than raising an error.
  compare_gpu_vs_cpu(kGroupByQuery);
  compare_gpu_vs_cpu("SELECT g, MIN(v) AS lo, MAX(v) AS hi FROM facts GROUP BY g");
  // A filter exercises the learned-ratio path.
  compare_gpu_vs_cpu("SELECT g, COUNT(*) AS n FROM facts WHERE v % 7 = 0 GROUP BY g");
}

TEST_CASE_METHOD(SizeEstimationFixture,
                 "gpu_execution grouped aggregation above a join is correct with estimation on",
                 "[integration][gpu_execution][size_estimation]")
{
  set_estimation(true);
  // Engagement is timing-dependent because fan-in estimates require a completed build and
  // enough ratio samples; correctness is unconditional.
  compare_gpu_vs_cpu(kGroupByOverJoinQuery);
  compare_gpu_vs_cpu(
    "SELECT d.label, COUNT(DISTINCT f.g) AS groups "
    "FROM facts f JOIN dims d ON f.k = d.k WHERE f.v > 100000 GROUP BY d.label");
}

TEST_CASE_METHOD(SizeEstimationFixture,
                 "gpu_execution size estimation produces the same rows either way",
                 "[integration][gpu_execution][size_estimation]")
{
  // Compare multisets so a group duplicated across partitions is not hidden.
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
