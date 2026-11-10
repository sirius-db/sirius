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
 * @file test_gpu_execution_string_constants.cpp
 * @brief Verifies string constants in projections and filters that reach the GPU in many batches.
 */

#include "codegen/jit/kernel_cache.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/expression_evaluator_strategies.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>

using StringConstantsFixture = sirius::test::GpuExecutionFixture;

namespace {

/**
 * Writes `t(i INTEGER, s VARCHAR)` with 20,000 rows in small row groups, so that with a 16 KiB scan
 * target the scan delivers it in many batches. `s` cycles through an empty string, a 300-byte
 * string, a string with an embedded NUL byte, UTF-8 text, short keys, and NULL.
 */
void create_string_table(StringConstantsFixture& fx, sirius::test::scratch_dir const& dir)
{
  fx.run_ok(
    "COPY (SELECT i::INTEGER AS i, CASE i % 8"
    " WHEN 0 THEN ''"
    " WHEN 1 THEN repeat('L', 300)"
    " WHEN 2 THEN 'a' || chr(0) || 'b'"
    " WHEN 3 THEN chr(233) || 't' || chr(233)"
    " WHEN 4 THEN 'k07'"
    " WHEN 5 THEN 'x' || chr(233) || 't' || chr(233) || 'y'"
    " WHEN 6 THEN NULL"
    " ELSE 'k' || (i % 40)::VARCHAR END AS s FROM range(20000) r(i)) TO " +
    dir.file_literal("t.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048);");
  fx.run_ok("CREATE VIEW t AS SELECT * FROM read_parquet(" + dir.file_literal("t.parquet") + ");");
}

}  // namespace

TEST_CASE_METHOD(StringConstantsFixture,
                 "gpu_execution string constants match the CPU across many batches",
                 "[integration][gpu_execution][string_constant_cache]")
{
  auto const strategy = GENERATE(sirius::test::each_expression_evaluator_strategy());
  CAPTURE(strategy);
  sirius::test::scratch_dir const dir{"string_constants"};
  create_string_table(*this, dir);
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 16384};
  sirius::test::scoped_expression_evaluator_strategy const strategy_guard{*con, strategy};

  // Empty, long, embedded-NUL and UTF-8 constants in one projection.
  compare_gpu_vs_cpu(
    "SELECT i, s = '', s = repeat('L', 300), s = 'a' || chr(0) || 'b', contains(s, chr(233)),"
    " s = chr(233) || 't' || chr(233), s = 'k07' FROM t");
  // Many constants in one filter.
  std::string many_constants = "contains(s, chr(233)) OR s = repeat('L', 300)";
  for (int k = 0; k < 40; k += 3) {
    many_constants += " OR s = 'k" + std::to_string(k) + "'";
  }
  compare_gpu_vs_cpu("SELECT count(*), sum(i) FROM t WHERE " + many_constants);
}

TEST_CASE_METHOD(StringConstantsFixture,
                 "gpu_execution nvrtc_jit reuses one kernel for different string literals",
                 "[integration][gpu_execution][string_constant_cache][nvrtc_jit]")
{
  sirius::test::scratch_dir const dir{"string_constant_literals"};
  create_string_table(*this, dir);
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 16384};
  sirius::test::scoped_expression_evaluator_strategy const strategy{*con, "nvrtc_jit"};

  compare_gpu_vs_cpu("SELECT count(*), sum(i) FROM t WHERE s = 'k07' OR contains(s, chr(233))");
  auto const kernels = codegen::jit::KernelCache::instance().size();
  compare_gpu_vs_cpu(
    "SELECT count(*), sum(i) FROM t WHERE s = 'k39' OR contains(s, 'a' || chr(0))");
  CHECK(codegen::jit::KernelCache::instance().size() == kernels);
}
