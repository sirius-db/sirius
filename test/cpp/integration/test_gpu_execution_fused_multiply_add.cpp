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
 * @file test_gpu_execution_fused_multiply_add.cpp
 * @brief The `nvrtc_jit_fuse_multiply_add` setting decides whether `nvrtc_jit` rounds a DOUBLE
 * product before it adds to it.
 *
 * The table holds DOUBLE columns `a` and `b` and DuckDB's rounded product `c`, so `a * b - c` is
 * exactly 0 on DuckDB and wherever the two operations are rounded separately, and the rounding
 * error of the product where they are fused. By default a projection and a filter return DuckDB's
 * rows; with the setting they return fused ones; without it again they return DuckDB's again. Which
 * operations NVRTC fuses is its choice, so with the setting the test asks only that some row of the
 * projection and some row of the filter show a fused result, which the NVRTC of the pinned CUDA
 * toolkit gives.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/expression_evaluator_strategies.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <cstdint>
#include <string>

using FusedMultiplyAddFixture = sirius::test::GpuExecutionFixture;

TEST_CASE_METHOD(FusedMultiplyAddFixture,
                 "gpu_execution nvrtc_jit fuses a multiplication and an addition only on request",
                 "[integration][gpu_execution][projection][filter][nvrtc_jit][floating_point]")
{
  run_ok("SET gpu_execution = false;");
  run_ok(
    "CREATE TABLE fma_t AS SELECT id, a, b, a * b AS c FROM ("
    " SELECT i AS id, 1 + (i * 7919 % 100003) / 100003.0 AS a,"
    " 1 + (i * 104729 % 99991) / 99991.0 AS b FROM range(4096) t(i));");
  run_ok("CHECKPOINT;");
  run_ok("SET gpu_execution = true;");
  sirius::test::scoped_expression_evaluator_strategy const strategy{
    *con, sirius::expression_evaluator_strategy::NVRTC_JIT};

  std::string const projection = "SELECT id, a * b - c AS r FROM fma_t";
  std::string const filter     = "SELECT id FROM fma_t WHERE a * b - c <> 0";
  auto const rows_on_gpu       = [&](std::string const& query) {
    auto result = run_on_path(query, sirius::test::execution_path::gpu);
    REQUIRE_FALSE(result->HasError());
    return collect_rows(*result);
  };
  auto const matches_duckdb = [&] {
    expect_result_matches_cpu(projection, sirius::test::execution_path::gpu);
    expect_result_matches_cpu(filter, sirius::test::execution_path::gpu);
    CHECK(rows_on_gpu(filter).empty());
  };

  matches_duckdb();
  {
    sirius::test::scoped_sirius_setting const fuse{*con, "nvrtc_jit_fuse_multiply_add", true};
    std::size_t not_cancelled = 0;
    for (auto const& row : rows_on_gpu(projection)) {
      not_cancelled += row.at(1) != "0.0" ? 1 : 0;
    }
    CHECK(not_cancelled > 0);
    CHECK_FALSE(rows_on_gpu(filter).empty());
  }
  matches_duckdb();
}
