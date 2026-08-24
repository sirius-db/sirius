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

// End-to-end reachability of the fold guard (INV-FOLD, op/fold_limits.hpp). The partition-sizing
// floor bounds the fold on the side it measured; a side that folds but is never measured -- a
// FULL OUTER join's probe, whose partition count is decided from the build -- has no floor and is
// bounded solely by gpu_merge_impl::concat's guard. That is exactly the residual this suite
// exercises, at a data volume CI can run: max_concat_fold_rows lowers the limit, and
// scan_task_batch_size keeps the probe in several batches so a fold actually happens.
//
// The point of the case is that the failure is Sirius's, attributable, and loud: with
// enable_duckdb_fallback off it surfaces as a query error carrying the stable "[fold_limit]"
// marker rather than as cuDF's "column size limit" message or a silent 20x CPU fallback.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

// DuckDB writes 122880-row row groups, and the GPU scan batches at row-group granularity, so a
// 300k-row table spans three row groups. With scan_task_batch_size at its minimum each becomes
// its own batch, which is what makes the probe side a multi-batch fold rather than a pass-through
// of one batch.
constexpr int kProbeRows = 300000;

class FoldLimitFixture : public sirius::test::GpuExecutionFixture {
 public:
  FoldLimitFixture()
  {
    run_ok("CREATE TABLE small_build (k INTEGER, tag VARCHAR);");
    run_ok("INSERT INTO small_build VALUES (1, 'a'), (2, 'b'), (3, 'c');");
    run_ok(
      "CREATE TABLE wide_probe AS SELECT (i % 3 + 1)::INTEGER AS k, i::INTEGER AS v "
      "FROM range(" +
      std::to_string(kProbeRows) + ") t(i);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(FoldLimitFixture,
                 "gpu_execution an unmeasured fold over the row limit fails as [fold_limit]",
                 "[integration][gpu_execution][fold_limit]")
{
  // A FULL OUTER join folds BOTH sides, but its partition count is sized from the build, so the
  // probe fold is the one nothing measures. Three rows of build keep the count at one partition,
  // which puts the whole 300k-row probe into a single fold.
  const std::string query =
    "SELECT count(*) FROM small_build FULL OUTER JOIN wide_probe ON small_build.k = wide_probe.k";

  {
    sirius::test::scoped_setting scan_batches(*con, "scan_task_batch_size", 1);
    sirius::test::scoped_setting fold_rows(*con, "max_concat_fold_rows", 64);

    run_ok("SET enable_duckdb_fallback = false;");
    auto result = con->Query(query);
    con->Query("SET enable_duckdb_fallback = true;");

    REQUIRE(result);
    REQUIRE(result->HasError());
    INFO("query error: " << result->GetError());
    CHECK(result->GetError().find("[fold_limit]") != std::string::npos);
  }

  // The session survives the refusal, and the same query answers correctly once the limit is back
  // to what cuDF can actually address.
  compare_gpu_vs_cpu(query);
}
