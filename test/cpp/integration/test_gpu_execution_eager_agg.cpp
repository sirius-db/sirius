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
 * @file test_gpu_execution_eager_agg.cpp
 * @brief End-to-end GPU-vs-CPU correctness for the eager-aggregation-pushdown
 *        pass (src/planner/eager_agg_pushdown_plan_pass.cpp) on the transparent
 *        execution path — the path where the rewrite actually runs in
 *        production. Each "fires" case additionally asserts the pass really
 *        fired (via its applied counter), so a silently-refused rewrite cannot
 *        make these tests vacuous; each "refused" case asserts the counter did
 *        NOT move and the results are still correct.
 *
 * The data is built so every NULL/no-match edge of the rewrite is exercised:
 * customers with zero orders (COUNT must be 0, not NULL, under LEFT/RIGHT
 * joins), NULL join keys on the pushed side (rows that never match), NULLs in
 * the aggregated column (COUNT skips them, SUM/MIN/MAX ignore them), duplicate
 * keys on BOTH sides (N:M multiplicity), and an empty pushed side.
 */

#include "planner/eager_agg_pushdown_plan_pass.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdlib>
#include <string>

namespace {

/// RAII environment variable override.
struct scoped_env {
  scoped_env(const char* name, const char* value) : _name(name) { setenv(name, value, 1); }
  ~scoped_env() { unsetenv(_name); }
  scoped_env(const scoped_env&)            = delete;
  scoped_env& operator=(const scoped_env&) = delete;

 private:
  const char* _name;
};

class EagerAggFixture : public sirius::test::GpuExecutionFixture {
 public:
  EagerAggFixture()
  {
    // cust: c_id 4 is duplicated (N:M multiplicity through the join); c_id 5
    // and 6 have no orders (LEFT/RIGHT no-match rows -> COUNT 0 / SUM NULL).
    run_ok("CREATE TABLE cust (c_id INTEGER, c_grp INTEGER);");
    run_ok("INSERT INTO cust VALUES (1, 1), (2, 0), (3, 1), (4, 0), (4, 0), (5, 1), (6, 0);");
    // ord: duplicate join keys (three rows for c_id 1), a NULL join key (never
    // matches), NULLs in o_val (COUNT counts non-NULLs only), negatives for
    // MIN/MAX, and o_grp for multi-key joins.
    run_ok("CREATE TABLE ord (o_cid INTEGER, o_grp INTEGER, o_key INTEGER, o_val INTEGER);");
    run_ok(
      "INSERT INTO ord VALUES "
      "(1, 1, 100, 10), (1, 1, 101, NULL), (1, 0, 102, -7), "
      "(2, 0, 103, 20), (2, 0, 104, 20), "
      "(3, 1, 105, NULL), "
      "(4, 0, 106, -1), (4, 1, 107, 42), "
      "(NULL, 1, 108, 99);");
    // ordempty: an empty pushed side.
    run_ok("CREATE TABLE ordempty (o_cid INTEGER, o_key INTEGER);");
    run_ok("CHECKPOINT;");
  }

  /// compare_gpu_vs_cpu, requiring that the pass fired during the GPU planning
  /// (the CPU run disables transparent execution and never plans on Sirius).
  void compare_fired(const std::string& query)
  {
    auto before = sirius::planner::eager_agg_pushdown_applied_count();
    compare_gpu_vs_cpu(query);
    auto after = sirius::planner::eager_agg_pushdown_applied_count();
    CHECK(after > before);
  }

  /// compare_gpu_vs_cpu, requiring that the pass did NOT fire.
  void compare_refused(const std::string& query)
  {
    auto before = sirius::planner::eager_agg_pushdown_applied_count();
    compare_gpu_vs_cpu(query);
    auto after = sirius::planner::eager_agg_pushdown_applied_count();
    CHECK(after == before);
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Fired shapes: rewritten GPU plan must match the CPU results exactly
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - LEFT join grouped COUNT (q13 shape)",
                 "[integration][gpu_execution][eager_agg]")
{
  // Customers 5/6 have no orders: COUNT must be 0 (COALESCE repair), not NULL.
  // Customer 3's only order has o_val NULL: count(o_val) = 0 for a MATCHED row.
  compare_fired(
    "SELECT c_id, count(o_key), count(o_val) FROM cust LEFT JOIN ord ON c_id = o_cid "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - RIGHT join (DuckDB's flipped q13 plan)",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired("SELECT c_id, count(o_key) FROM ord RIGHT JOIN cust ON o_cid = c_id GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - INNER join COUNT and SUM",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired(
    "SELECT c_id, count(o_key), sum(o_val) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - LEFT join SUM/MIN/MAX keep NULL semantics",
                 "[integration][gpu_execution][eager_agg]")
{
  // Unmatched customers must stay NULL for SUM/MIN/MAX (no COALESCE), and
  // customer 3 (only NULL o_val) must also be NULL.
  compare_fired(
    "SELECT c_id, sum(o_val), min(o_val), max(o_val) FROM cust LEFT JOIN ord ON c_id = o_cid "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - multi-key equi join",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid AND c_grp = o_grp "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - full q13 shape (second GROUP BY over counts)",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired(
    "SELECT c_count, count(*) AS custdist FROM ("
    "  SELECT c_id, count(o_key) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cid "
    "  GROUP BY c_id) GROUP BY c_count");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - empty pushed side yields COUNT 0 everywhere",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ordempty ON c_id = o_cid GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - forced pushdown on a filtered preserved side",
                 "[integration][gpu_execution][eager_agg]")
{
  // The default benefit gate refuses a filtered non-pushed side; FORCE bypasses
  // only the benefit heuristic, so the result must still be exact.
  scoped_env force("SIRIUS_EAGER_AGG_FORCE", "1");
  compare_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid WHERE c_grp = 1 "
    "GROUP BY c_id");
}

//===----------------------------------------------------------------------===//
// Refused shapes: pass must not fire, results must still be correct
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - refusals stay refused and correct",
                 "[integration][gpu_execution][eager_agg]")
{
  SECTION("count(*) counts join rows")
  {
    compare_refused("SELECT c_id, count(*) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("avg is not decomposed")
  {
    compare_refused("SELECT c_id, avg(o_val) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("DISTINCT aggregate")
  {
    compare_refused(
      "SELECT c_id, count(DISTINCT o_grp) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("group key on the pushed side")
  {
    compare_refused("SELECT o_grp, count(o_key) FROM cust JOIN ord ON c_id = o_cid GROUP BY o_grp");
  }
  SECTION("aggregates over both sides")
  {
    compare_refused(
      "SELECT c_id, count(o_key), sum(c_grp) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("filtered preserved side fails the default benefit gate")
  {
    compare_refused(
      "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid WHERE c_grp = 1 "
      "GROUP BY c_id");
  }
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - kill switch",
                 "[integration][gpu_execution][eager_agg]")
{
  scoped_env off("SIRIUS_EAGER_AGG_PUSHDOWN", "0");
  compare_refused(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
}
