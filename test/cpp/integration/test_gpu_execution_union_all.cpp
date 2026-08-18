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

// GPU-vs-CPU correctness for `UNION ALL` (sirius_physical_union +
// sirius_physical_passthrough_sink).
//
// Every query goes through the shared GpuExecutionFixture, which runs it once
// transparently on the GPU -- asserting exactly one GPU execution and ZERO
// fallbacks -- and once on DuckDB CPU, then compares. That zero-fallback
// assertion is what makes these tests meaningful: a UNION ALL that silently
// CPU-fell-back would still produce the right answer, so a plain result check
// would pass whether or not the operator existed.
//
// The rest of the set-operation family is asserted the other way round, with
// expect_gpu_fallback: those must still leave the GPU path cleanly.
//
// NOT covered here (needs a fixture with a small scan_task_batch_size; the
// shared integration config uses 100 MB, so these fixtures are one batch per
// arm): multi-batch streaming, unequal-length arms across many batches, and
// multi-GPU device placement.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

// Overlapping keys between a and b (k = 3) so multiset semantics are
// observable, a disjoint third table for N-ary chains, an empty table, and a
// BIGINT-keyed table so the binder has a super-type to reconcile to.
class UnionAllFixture : public sirius::test::GpuExecutionFixture {
 public:
  UnionAllFixture()
  {
    run_ok("CREATE TABLE ua (k INTEGER, v VARCHAR);");
    run_ok("CREATE TABLE ub (k INTEGER, v VARCHAR);");
    run_ok("CREATE TABLE uc (k INTEGER, v VARCHAR);");
    run_ok("CREATE TABLE uempty (k INTEGER, v VARCHAR);");
    run_ok("CREATE TABLE uwide (k BIGINT, v VARCHAR);");

    run_ok("INSERT INTO ua VALUES (1, 'a'), (2, 'b'), (3, 'c'), (NULL, 'n');");
    run_ok("INSERT INTO ub VALUES (3, 'c'), (4, 'd');");
    run_ok("INSERT INTO uc VALUES (5, 'e');");
    run_ok("INSERT INTO uwide VALUES (100, 'w'), (200, 'x');");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL two arms",
                 "[integration][gpu_execution][union_all]")
{
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub");
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL SELECT k FROM ub");
  // Arm order reversed: the operator must not depend on which arm is children[0].
  compare_gpu_vs_cpu("SELECT k, v FROM ub UNION ALL SELECT k, v FROM ua");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL is a multiset, not a set",
                 "[integration][gpu_execution][union_all]")
{
  // (3, 'c') is in both inputs and must survive twice. The count comparison is
  // the assertion that no de-duplication happened.
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub) t");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub) t WHERE k = 3");
  // A table unioned with itself: every row appears exactly twice.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM ua");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL with an empty arm",
                 "[integration][gpu_execution][union_all]")
{
  // An empty arm's pipeline finishes without ever pushing a batch, so the
  // operator must fall through it to the other arm rather than reporting
  // exhausted.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM uempty");
  compare_gpu_vs_cpu("SELECT k, v FROM uempty UNION ALL SELECT k, v FROM ua");
  compare_gpu_vs_cpu("SELECT k FROM uempty UNION ALL SELECT k FROM uempty");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL is N-ary",
                 "[integration][gpu_execution][union_all]")
{
  // At the pinned DuckDB a chain binds to ONE set-operation node with N
  // children rather than a left-deep tree of binary nodes, which is why the
  // operator loops over children instead of indexing 0 and 1.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub UNION ALL SELECT k, v FROM uc");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub UNION ALL SELECT k FROM uc "
    "UNION ALL SELECT k FROM uempty) t");
  // Five arms, with the empty one in the middle rather than at the end.
  compare_gpu_vs_cpu(
    "SELECT k FROM ua UNION ALL SELECT k FROM ub UNION ALL SELECT k FROM uempty UNION ALL "
    "SELECT k FROM uc UNION ALL SELECT k FROM ua");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL arms of unequal length",
                 "[integration][gpu_execution][union_all]")
{
  // A short arm drains first. The base task-driver contract would strand the
  // longer arm's remaining rows once the short one finished; the row counts
  // here are what catches that.
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k FROM uc UNION ALL SELECT k FROM ua) t");
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM uc) t");
  compare_gpu_vs_cpu("SELECT k FROM uc UNION ALL SELECT k FROM ua");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL reconciles types in the binder",
                 "[integration][gpu_execution][union_all]")
{
  // INTEGER + BIGINT reconcile to BIGINT. Sirius does no type work: the cast is
  // already a projection inside the INTEGER arm's subtree.
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL SELECT k FROM uwide");
  compare_gpu_vs_cpu("SELECT k, v FROM uwide UNION ALL SELECT k, v FROM ua");
  // A literal arm forces a second reconciliation shape.
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL SELECT 999");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL preserves NULLs",
                 "[integration][gpu_execution][union_all]")
{
  // Bag union has no key and no comparison, so a NULL is just another value —
  // it must be forwarded, never dropped or coalesced.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t WHERE k IS NULL");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL composes with downstream operators",
                 "[integration][gpu_execution][union_all]")
{
  // Aggregate downstream: UNION becomes a pipeline sink under the group-by's
  // hash PARTITION, which is the is_sink() == true shape.
  compare_gpu_vs_cpu(
    "SELECT k, count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t GROUP BY k");
  compare_gpu_vs_cpu("SELECT sum(k) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t");

  // Ordering downstream, where UNION's NO_ORDER source_order() decides the
  // plan's order preservation.
  compare_gpu_vs_cpu_ordered(
    "SELECT k FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t ORDER BY k NULLS LAST");
  compare_gpu_vs_cpu_ordered(
    "SELECT k FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t ORDER BY k DESC NULLS LAST "
    "LIMIT 3");

  // Filter and projection downstream.
  compare_gpu_vs_cpu("SELECT k * 2 FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t WHERE k > 2");

  // Probe side of a join.
  compare_gpu_vs_cpu(
    "SELECT t.k, b.v FROM (SELECT k FROM ua UNION ALL SELECT k FROM uc) t JOIN ub b ON t.k = b.k");
  // Build side of a join.
  compare_gpu_vs_cpu(
    "SELECT a.k, a.v FROM ua a JOIN (SELECT k FROM ub UNION ALL SELECT k FROM uc) t ON a.k = t.k");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL nested inside another UNION ALL arm",
                 "[integration][gpu_execution][union_all]")
{
  // An arm whose own subtree is a UNION: the inner UNION's passthrough sinks
  // and the outer one's must not collide, since port names are per-operator.
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k FROM ua UNION ALL "
    "SELECT k FROM (SELECT k FROM ub UNION ALL SELECT k FROM uc) inner_t) t");
  compare_gpu_vs_cpu(
    "SELECT k FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) l UNION ALL "
    "SELECT k FROM (SELECT k FROM uc UNION ALL SELECT k FROM uempty) r");
  // An arm that is itself an aggregate: that arm's root is an unconditional
  // sink, so the passthrough sink ends up alone in its pipeline with an input
  // port, rather than inline at the end of a scan's pipeline.
  compare_gpu_vs_cpu(
    "SELECT k FROM (SELECT k FROM ua GROUP BY k) g UNION ALL SELECT k FROM ub");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution UNION ALL over a narrowed scan",
                 "[integration][gpu_execution][union_all]")
{
  // The compressed-schema pass treats UNION as a native carrier boundary and
  // restores every arm before the wrap pass runs, which is what stops two arms
  // presenting different physical carriers for the same logical column. This
  // exercises that path: uwide.k is BIGINT with small values (a narrowing
  // candidate) unioned against an INTEGER arm.
  compare_gpu_vs_cpu("SELECT k FROM uwide UNION ALL SELECT k FROM ua");
  compare_gpu_vs_cpu(
    "SELECT k, count(*) FROM (SELECT k FROM uwide UNION ALL SELECT k FROM ua) t GROUP BY k");
}

TEST_CASE_METHOD(UnionAllFixture,
                 "gpu_execution declines the rest of the set-operation family",
                 "[integration][gpu_execution][union_all]")
{
  // Distinct UNION, EXCEPT and INTERSECT are still unsupported. They must leave
  // the GPU path cleanly via a fallback -- not error, and not silently produce a
  // bag union.
  expect_gpu_fallback("SELECT k FROM ua UNION SELECT k FROM ub");
  expect_gpu_fallback("SELECT k FROM ua EXCEPT SELECT k FROM ub");
  expect_gpu_fallback("SELECT k FROM ua INTERSECT SELECT k FROM ub");

  // And the results are still right on the CPU.
  auto distinct_result = con->Query(
    "SELECT count(*) FROM (SELECT k FROM ua UNION SELECT k FROM ub) t WHERE k IS NOT NULL");
  REQUIRE(distinct_result);
  REQUIRE_FALSE(distinct_result->HasError());
  REQUIRE(distinct_result->GetValue(0, 0).ToString() == "4");
}
