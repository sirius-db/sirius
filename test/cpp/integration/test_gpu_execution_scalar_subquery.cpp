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

#include <catch.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "scalar subquery FIRST ignores empty partials without discarding real NULLs",
                 "[integration][gpu_execution][scalar_subquery][first_empty_sql]")
{
  sirius::test::scratch_dir const dir{"scalar_first"};
  run_ok("SET gpu_execution = false");
  run_ok("SET threads = 1");
  run_ok(
    "COPY (SELECT i::INTEGER AS id, CASE WHEN i = 8191 THEN NULL ELSE 42 END::BIGINT AS x "
    "FROM range(8192) r(i) ORDER BY i) TO " +
    dir.file_literal("input.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  auto layout = con->Query("SELECT row_group_num_rows FROM parquet_metadata(" +
                           dir.file_literal("input.parquet") +
                           ") WHERE path_in_schema = 'id' ORDER BY row_group_id");
  REQUIRE(layout);
  REQUIRE_FALSE(layout->HasError());
  REQUIRE(layout->RowCount() == 4);
  for (duckdb::idx_t i = 0; i < 4; ++i) {
    REQUIRE(layout->Collection().GetValue(0, i).GetValue<int64_t>() == 2048);
  }
  run_ok("CREATE VIEW scalar_first_input AS SELECT * FROM read_parquet(" +
         dir.file_literal("input.parquet") + ")");

  // Keep row groups in separate scan batches and the filter above the scan,
  // so rejected rows reach FIRST as empty batches instead of being pruned at IO.
  sirius::test::scoped_setting const scan_batch{*con, "scan_task_batch_size", 1};
  sirius::test::scoped_setting const concat_batch{*con, "concat_batch_bytes", 1};
  sirius::test::disabled_optimizers_guard const optimizers{
    *con, "filter_pushdown,statistics_propagation"};
  for (auto id : {4095, 8191, 8192}) {
    CAPTURE(id);
    auto query = "SELECT (SELECT x FROM scalar_first_input WHERE id = " + std::to_string(id) +
                 ") AS scalar_value";
    // Establish the SQL oracle explicitly: one value, one actual NULL row, and
    // no matching rows all produce exactly one scalar result row.
    run_ok("SET gpu_execution = false");
    auto expected = con->Query(query);
    REQUIRE(expected);
    REQUIRE_FALSE(expected->HasError());
    REQUIRE(expected->ColumnCount() == 1);
    REQUIRE(expected->RowCount() == 1);
    auto value = expected->Collection().GetValue(0, 0);
    REQUIRE(value.IsNull() == (id != 4095));
    if (id == 4095) REQUIRE(value.GetValue<int64_t>() == 42);
    // Checks full rows/types, successful GPU execution, and zero CPU fallback.
    compare_gpu_vs_cpu(query);
  }
}
