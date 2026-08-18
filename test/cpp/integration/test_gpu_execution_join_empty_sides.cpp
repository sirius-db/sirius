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

// GPU-vs-CPU correctness for hash joins whose build side is empty or emptied
// at runtime, and for MARK (IN / NOT IN) three-valued semantics driven by the
// mark_build_kind classification: `x IN (S)` is FALSE for every x — including
// NULL x — when S is empty, while NULL semantics apply only to non-empty sets.
// Also pins the stacked-MARK scheduling shape, where the outer join's probe
// chain forwards a task-creation hint into the inner MARK join before that
// join has been sized (refresh_cross_schedule's MARK tripwire tolerates this
// pre-sizing state, so the hint reports wait-for-build).
//
// All cases run over file-backed native tables, whose scans always deliver at
// least one (possibly 0-row) batch. The zero-BATCH side shapes (a
// zero-row-group parquet scan feeding a BUILD_PROBE join, served by
// plan_build_probe_reclaim's orphan build task) are exercised by the
// [build_probe] and [build_probe_orphan] unit tests instead: a
// zero-row-group parquet scan can nondeterministically fail task creation
// (a scan-side defect, tracked separately), and its runtime fallback leaves
// the shared SiriusContext unable to run later GPU queries — so that trigger
// cannot be used in a shared-context test suite.
//
// Every query goes through the shared file-backed GpuExecutionFixture, which
// runs it once on the GPU (asserting a real GPU execution with no fallback)
// and once on DuckDB CPU, then compares the results (order-insensitive).

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

/// Native-table fixture: `t` (100k ints), `marks` (a NULL-bearing 3-value
/// IN-list), `probe_n` (10 ints + one NULL), `b100` (0..99, filter-emptiable),
/// and `empty_t` (an empty on-disk table, which scans as one 0-row batch).
class JoinEmptySideFixture : public sirius::test::GpuExecutionFixture {
 public:
  JoinEmptySideFixture()
  {
    run_ok("CREATE TABLE t AS SELECT CAST((range * 7) % 1000 AS INTEGER) x FROM range(100000);");
    run_ok("CREATE TABLE marks(y INTEGER);");
    run_ok("INSERT INTO marks VALUES (1), (2), (NULL);");
    run_ok("CREATE TABLE probe_n(x INTEGER);");
    run_ok("INSERT INTO probe_n SELECT CAST(range AS INTEGER) FROM range(10);");
    run_ok("INSERT INTO probe_n VALUES (NULL);");
    run_ok("CREATE TABLE b100(y INTEGER);");
    run_ok("INSERT INTO b100 SELECT CAST(range AS INTEGER) FROM range(100);");
    run_ok("CREATE TABLE empty_t(y INTEGER);");
    run_ok("CHECKPOINT;");
  }
};

/// Applies a `disabled_optimizers` setting and restores the default (none disabled) when the
/// enclosing scope ends — including through a failed assertion, so a leaked setting cannot bleed
/// into later queries on the same connection.
class disabled_optimizers_guard {
 public:
  disabled_optimizers_guard(sirius::test::GpuExecutionFixture& fixture, const std::string& value)
    : fixture_(fixture)
  {
    fixture_.run_ok("SET disabled_optimizers='" + value + "';");
  }
  ~disabled_optimizers_guard() { fixture_.con->Query("SET disabled_optimizers='';"); }

  disabled_optimizers_guard(const disabled_optimizers_guard&)            = delete;
  disabled_optimizers_guard& operator=(const disabled_optimizers_guard&) = delete;

 private:
  sirius::test::GpuExecutionFixture& fixture_;
};

}  // namespace

TEST_CASE_METHOD(JoinEmptySideFixture,
                 "gpu_execution stacked MARK joins schedule through a pre-sizing hint walk",
                 "[integration][gpu_execution][join][empty_sides]")
{
  // Two IN-subquery MARK joins over ordinary non-empty tables: the outer join's probe chain
  // forwards a task-creation hint into the inner join before the inner join has been sized.
  compare_gpu_vs_cpu(
    "SELECT sum(CASE WHEN x IN (SELECT y FROM marks) THEN 1 END) s1, "
    "sum(CASE WHEN x IN (SELECT y FROM marks) THEN x END) s2 FROM t");
}

TEST_CASE_METHOD(JoinEmptySideFixture,
                 "gpu_execution MARK join against a filter-emptied build set",
                 "[integration][gpu_execution][join][empty_sides]")
{
  // The predicate empties the build at runtime (a 0-row batch still flows), so the mark must be a
  // definite FALSE for every probe row — including the NULL probe key — and NOT IN keeps all 11.
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM probe_n WHERE x NOT IN (SELECT y FROM b100 WHERE y > 1000)");
  // The same emptied set projected as a mark column: all-FALSE, no NULLs.
  compare_gpu_vs_cpu("SELECT x, x IN (SELECT y FROM b100 WHERE y > 1000) AS m FROM probe_n");
}

TEST_CASE_METHOD(JoinEmptySideFixture,
                 "gpu_execution MARK join three-valued marks over non-empty build sets",
                 "[integration][gpu_execution][join][empty_sides]")
{
  // NULL-bearing build over a NULL-bearing probe: matches are TRUE, everything else is NULL.
  compare_gpu_vs_cpu("SELECT x, x IN (SELECT y FROM marks) AS m FROM probe_n");
  compare_gpu_vs_cpu("SELECT x, x NOT IN (SELECT y FROM marks) AS m FROM probe_n");
  // NULL-free build: unmatched non-NULL probes are FALSE; the NULL probe key stays NULL.
  compare_gpu_vs_cpu("SELECT x, x IN (SELECT y FROM marks WHERE y IS NOT NULL) AS m FROM probe_n");
  compare_gpu_vs_cpu(
    "SELECT x, x NOT IN (SELECT y FROM marks WHERE y IS NOT NULL) AS m FROM probe_n");
  // NULL-free probe against both build shapes.
  compare_gpu_vs_cpu("SELECT y, y IN (SELECT m.y FROM marks m) AS mk FROM b100");
  compare_gpu_vs_cpu(
    "SELECT y, y IN (SELECT m.y FROM marks m WHERE m.y IS NOT NULL) AS mk FROM b100");
}

TEST_CASE_METHOD(JoinEmptySideFixture,
                 "gpu_execution MARK join against an empty native table (one 0-row batch)",
                 "[integration][gpu_execution][join][empty_sides]")
{
  // An empty on-disk table scans as ONE 0-row batch (not zero batches), so the join schedules
  // normally; statistics propagation is disabled so the empty build is not folded away at plan
  // time. The empty set makes every mark a definite FALSE, including for the NULL probe key.
  disabled_optimizers_guard const guard{*this, "statistics_propagation"};
  compare_gpu_vs_cpu("SELECT x, x IN (SELECT y FROM empty_t) AS m FROM probe_n");
  compare_gpu_vs_cpu("SELECT count(*) FROM probe_n WHERE x NOT IN (SELECT y FROM empty_t)");
}

TEST_CASE_METHOD(JoinEmptySideFixture,
                 "gpu_execution INNER join against an empty native table completes with zero rows",
                 "[integration][gpu_execution][join][empty_sides]")
{
  disabled_optimizers_guard const guard{*this, "statistics_propagation"};
  compare_gpu_vs_cpu("SELECT count(*) FROM t JOIN empty_t e ON t.x = e.y");
}
