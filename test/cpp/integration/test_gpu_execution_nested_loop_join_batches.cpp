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
 * @file test_gpu_execution_nested_loop_join_batches.cpp
 * @brief Verifies nested loop joins whose inputs reach the join in several batches.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <string>

using NestedLoopJoinBatchesFixture = sirius::test::GpuExecutionFixture;

TEST_CASE_METHOD(NestedLoopJoinBatchesFixture,
                 "nested loop join - inputs split into several batches match the CPU",
                 "[integration][gpu_execution][nested_loop_join]")
{
  // l holds 0..17999 and r the even numbers below 18000, in small row groups. With 16 KiB scan and
  // CONCAT targets, each side spans several batches unless CONCAT folds it. r is the smaller side,
  // so DuckDB keeps SEMI and ANTI instead of flipping them to RIGHT_SEMI and RIGHT_ANTI.
  sirius::test::scratch_dir const dir{"nested_loop_join_batches"};
  run_ok("COPY (SELECT i::INTEGER x FROM range(18000) t(i)) TO " + dir.file_literal("l.parquet") +
         " (FORMAT PARQUET, ROW_GROUP_SIZE 4096);");
  run_ok("COPY (SELECT (2 * i)::INTEGER y FROM range(9000) t(i)) TO " +
         dir.file_literal("r.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048);");
  run_ok("CREATE VIEW l AS SELECT * FROM read_parquet(" + dir.file_literal("l.parquet") + ");");
  run_ok("CREATE VIEW r AS SELECT * FROM read_parquet(" + dir.file_literal("r.parquet") + ");");
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 16384};
  sirius::test::scoped_setting const concat_batch{*con, "concat_batch_bytes", 16384};

  // Each x matches the one even y in {x, x + 1}, except x = 17999.
  std::string const band = "l.x <= r.y AND r.y <= l.x + 1";
  SECTION("INNER")
  {
    compare_gpu_vs_cpu("SELECT count(*), sum(x), sum(y) FROM l JOIN r ON " + band + ";");
  }
  SECTION("LEFT")
  {
    compare_gpu_vs_cpu("SELECT count(*), count(y), sum(x) FROM l LEFT JOIN r ON " + band + ";");
  }
  SECTION("RIGHT")
  {
    compare_gpu_vs_cpu("SELECT count(*), count(x), sum(y) FROM l RIGHT JOIN r ON " + band + ";");
  }
  SECTION("FULL OUTER")
  {
    compare_gpu_vs_cpu("SELECT count(*), count(x), count(y) FROM l FULL OUTER JOIN r ON " + band +
                       ";");
  }
  SECTION("SEMI")
  {
    compare_gpu_vs_cpu("SELECT count(*), sum(x) FROM l SEMI JOIN r ON l.x < r.y;");
  }
  SECTION("ANTI")
  {
    compare_gpu_vs_cpu("SELECT count(*), sum(x) FROM l ANTI JOIN r ON l.x < r.y;");
  }
}
