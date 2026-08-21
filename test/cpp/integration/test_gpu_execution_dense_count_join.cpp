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

// GPU-vs-CPU correctness for the DENSE_COUNT_JOIN rewrite (TPC-H q13 shape): COUNT over an
// outer equi-join grouped by the preserved-side key. The tables carry every hazard the rewrite
// must preserve exactly: preserved keys with zero matches (the q13 c_count=0 bucket), duplicate
// preserved keys, NULL keys on both sides, NULL COUNT(col) arguments, and counted keys that
// match nothing. Each query runs through the full engine (planner rewrite, port wiring, task
// scheduling) on GPU and is compared against DuckDB CPU.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

class DenseCountJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  DenseCountJoinFixture()
  {
    run_ok("SET enable_dense_count_join = true;");
    run_ok("CREATE TABLE cust (c_id INTEGER, c_grp INTEGER);");
    // Keys 1..8 with 3 duplicated, plus two NULL-key rows; 4 and 6..8 have no orders.
    run_ok(
      "INSERT INTO cust VALUES (1, 0), (2, 1), (3, 0), (3, 1), (4, 0), (5, 1), (6, 0), (7, 1), "
      "(8, 0), (NULL, 0), (NULL, 1);");
    run_ok("CREATE TABLE ord (o_id BIGINT, o_cust INTEGER, o_val INTEGER);");
    // o_cust 42 matches nothing; one NULL o_cust; o_val NULL on two rows (COUNT(o_val) path).
    run_ok(
      "INSERT INTO ord VALUES (100, 2, 10), (101, 2, NULL), (102, 3, 11), (103, 5, 12), "
      "(104, 5, NULL), (105, 5, 13), (106, 42, 14), (107, NULL, 15);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: COUNT(col) grouped by the LEFT-join key",
                 "[integration][gpu_execution][dense_count_join]")
{
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: COUNT(*) and nullable COUNT(col) semantics",
                 "[integration][gpu_execution][dense_count_join]")
{
  // COUNT(*): unmatched and NULL-key preserved rows count one each.
  compare_gpu_vs_cpu(
    "SELECT c_id, count(*) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
  // COUNT(o_val): matched rows with NULL o_val are excluded.
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_val) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY "
    "c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: RIGHT-join orientation",
                 "[integration][gpu_execution][dense_count_join]")
{
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM ord RIGHT JOIN cust ON o_cust = c_id GROUP BY "
    "c_id");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: full q13 distribution shape with ORDER BY",
                 "[integration][gpu_execution][dense_count_join]")
{
  compare_gpu_vs_cpu_ordered(
    "SELECT c_count, count(*) AS custdist FROM ("
    "  SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY "
    "c_id"
    ") t GROUP BY c_count ORDER BY custdist DESC, c_count DESC");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: sparse strategy under a tiny histogram budget",
                 "[integration][gpu_execution][dense_count_join]")
{
  run_ok("SET dense_count_join_max_bytes = 8;");
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
  compare_gpu_vs_cpu(
    "SELECT c_id, count(*) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
  run_ok("SET dense_count_join_max_bytes = 2147483648;");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: disabled knob keeps the join plan correct",
                 "[integration][gpu_execution][dense_count_join]")
{
  run_ok("SET enable_dense_count_join = false;");
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id");
  run_ok("SET enable_dense_count_join = true;");
}

TEST_CASE_METHOD(DenseCountJoinFixture,
                 "gpu_execution dense count-join: runtime-empty sides",
                 "[integration][gpu_execution][dense_count_join]")
{
  // Subquery WHERE clauses keep the tables statically non-empty (the optimizer cannot fold
  // them to EMPTY_RESULT), so these exercise the operator's runtime empty-side paths through
  // the scheduler: an empty counted side must emit every preserved key with count 0, an empty
  // preserved side must drain the counted batches and emit zero groups, and empty-both must
  // finish the zero-task pipeline with an empty result.
  compare_gpu_vs_cpu(
    "SELECT c_id, count(o.o_id) AS c_count FROM cust "
    "LEFT JOIN (SELECT * FROM ord WHERE o_val > 1000000) o ON c_id = o.o_cust GROUP BY c_id");
  compare_gpu_vs_cpu(
    "SELECT c.c_id, count(o_id) AS c_count FROM (SELECT * FROM cust WHERE c_grp > 1000000) c "
    "LEFT JOIN ord ON c.c_id = o_cust GROUP BY c.c_id");
  compare_gpu_vs_cpu(
    "SELECT c.c_id, count(o.o_id) AS c_count FROM (SELECT * FROM cust WHERE c_grp > 1000000) c "
    "LEFT JOIN (SELECT * FROM ord WHERE o_val > 1000000) o ON c.c_id = o.o_cust "
    "GROUP BY c.c_id");
}
