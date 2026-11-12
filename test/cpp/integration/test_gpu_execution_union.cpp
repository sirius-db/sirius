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

// GPU-vs-CPU correctness for `UNION ALL`: sirius_physical_union + sirius_physical_passthrough_sink,
// and for distinct `UNION`, which runs as the DISTINCT lowering over that same union.
//
// GpuExecutionFixture runs each query on the GPU -- asserting one GPU execution and ZERO fallbacks
// -- and again on DuckDB CPU, then compares. The zero-fallback assertion is what makes these tests
// meaningful: a silent CPU fallback would still produce the right answer, so a plain result check
// would pass whether or not the operator existed. Distinct EXCEPT and INTERSECT also use this
// comparator to verify that their hash join and DISTINCT execute on the GPU.
//
// The shared integration config uses a 100 MB scan_task_batch_size, so the fixture tables are one
// batch per arm. Only the many-batch distinct EXCEPT / INTERSECT case lowers it. NOT covered here:
// multi-batch UNION streaming, unequal-length arms across many batches, and multi-GPU device
// placement.

#include <catch.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/pinned_table_guard.hpp>
#include <utils/scoped_sirius_setting.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <utility>

namespace {

// Overlapping keys between a and b (k = 3) so multiset semantics are observable, a disjoint third
// table for N-ary chains, an empty table, and a BIGINT-keyed table for super-type reconciliation.
class UnionFixture : public sirius::test::GpuExecutionFixture {
 public:
  UnionFixture()
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

  //! DuckDB's physical plan text for @p query, explained with GPU execution off.
  std::string cpu_plan_text(std::string const& query)
  {
    run_ok("SET gpu_execution = false;");
    auto const result = con->Query("EXPLAIN " + query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    std::string text;
    for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
      for (duckdb::idx_t column = 0; column < result->ColumnCount(); ++column) {
        text += result->Collection().GetValue(column, row).ToString();
        text += '\n';
      }
    }
    return text;
  }
};

}  // namespace

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL two arms",
                 "[integration][gpu_execution][union][union_all]")
{
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub");
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL SELECT k FROM ub");
  // Arm order reversed: the operator must not depend on which arm is children[0].
  compare_gpu_vs_cpu("SELECT k, v FROM ub UNION ALL SELECT k, v FROM ua");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL is a multiset, not a set",
                 "[integration][gpu_execution][union][union_all]")
{
  // (3, 'c') is in both inputs and must survive twice: the count is what catches de-duplication.
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub) t");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub) t WHERE k = 3");
  // A table unioned with itself: every row appears exactly twice.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM ua");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL with an empty arm",
                 "[integration][gpu_execution][union][union_all]")
{
  // An empty arm's pipeline finishes without pushing a batch, so the operator must fall through
  // to the other arm rather than reporting exhausted. The base's all-ports readiness test can
  // never hold here, so these cases are the regression test for excusing a finished arm.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM uempty");
  compare_gpu_vs_cpu("SELECT k, v FROM uempty UNION ALL SELECT k, v FROM ua");
  compare_gpu_vs_cpu("SELECT k FROM uempty UNION ALL SELECT k FROM uempty");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL is N-ary",
                 "[integration][gpu_execution][union][union_all]")
{
  // At the pinned DuckDB a chain binds to ONE set-operation node with N children rather than a
  // left-deep tree of binary nodes, which is why the operator loops over children.
  compare_gpu_vs_cpu(
    "SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub UNION ALL SELECT k, v FROM uc");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub UNION ALL SELECT k FROM uc "
    "UNION ALL SELECT k FROM uempty) t");
  // Five arms, with the empty one in the middle rather than at the end.
  compare_gpu_vs_cpu(
    "SELECT k FROM ua UNION ALL SELECT k FROM ub UNION ALL SELECT k FROM uempty UNION ALL "
    "SELECT k FROM uc UNION ALL SELECT k FROM ua");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL arms of unequal length",
                 "[integration][gpu_execution][union][union_all]")
{
  // A row-count check on asymmetric inputs, and no more than that. Arm stranding needs an arm
  // with a second batch to strand, which the shared 100 MB scan_task_batch_size cannot produce:
  // at one batch per arm the base task-driver contract answers these correctly too.
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k FROM uc UNION ALL SELECT k FROM ua) t");
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM uc) t");
  compare_gpu_vs_cpu("SELECT k FROM uc UNION ALL SELECT k FROM ua");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL reconciles types in the binder",
                 "[integration][gpu_execution][union][union_all]")
{
  // INTEGER + BIGINT reconcile to BIGINT; the cast is already a projection inside the INTEGER arm.
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL SELECT k FROM uwide");
  compare_gpu_vs_cpu("SELECT k, v FROM uwide UNION ALL SELECT k, v FROM ua");
  // A literal arm forces a second reconciliation shape.
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL SELECT 999");
  // BY NAME: ub's columns are reordered and ua's missing `v` is filled with NULL, inside the arms.
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL BY NAME SELECT v, k FROM ub");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL over HUGEINT aggregates",
                 "[integration][gpu_execution][union][union_all]")
{
  // DuckDB types sum(INTEGER) as HUGEINT; the union and its aggregate arms all plan that type.
  compare_gpu_vs_cpu("SELECT sum(k) AS s FROM ua UNION ALL SELECT sum(k) FROM ub");
  compare_gpu_vs_cpu(
    "SELECT s + 1 FROM (SELECT sum(k) AS s FROM ua UNION ALL SELECT sum(k) FROM ub) t");
  compare_gpu_vs_cpu_ordered(
    "SELECT sum(k) AS s FROM ua UNION ALL SELECT sum(k) FROM ub ORDER BY s");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL preserves NULLs",
                 "[integration][gpu_execution][union][union_all]")
{
  // A union has no key and no comparison, so a NULL must be forwarded, not dropped or coalesced.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t WHERE k IS NULL");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL composes with downstream operators",
                 "[integration][gpu_execution][union][union_all]")
{
  // Aggregate downstream: UNION becomes a pipeline sink under the group-by's hash PARTITION,
  // which is the is_sink() == true shape.
  compare_gpu_vs_cpu(
    "SELECT k, count(*) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t GROUP BY k");
  compare_gpu_vs_cpu("SELECT sum(k) FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t");

  // An explicit downstream ORDER BY establishes the union result's sequence.
  compare_gpu_vs_cpu_ordered(
    "SELECT k FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t ORDER BY k NULLS LAST");
  compare_gpu_vs_cpu_ordered(
    "SELECT k FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t ORDER BY k DESC NULLS LAST "
    "LIMIT 3");

  // Filter and projection downstream.
  compare_gpu_vs_cpu(
    "SELECT k * 2 FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) t WHERE k > 2");

  // Probe side of a join.
  compare_gpu_vs_cpu(
    "SELECT t.k, b.v FROM (SELECT k FROM ua UNION ALL SELECT k FROM uc) t JOIN ub b ON t.k = b.k");
  // Build side of a join.
  compare_gpu_vs_cpu(
    "SELECT a.k, a.v FROM ua a JOIN (SELECT k FROM ub UNION ALL SELECT k FROM uc) t ON a.k = t.k");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL nested inside another UNION ALL arm",
                 "[integration][gpu_execution][union][union_all]")
{
  // An arm whose own subtree is a UNION: the inner and outer passthrough sinks must not collide,
  // since port names are per-operator.
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k FROM ua UNION ALL "
    "SELECT k FROM (SELECT k FROM ub UNION ALL SELECT k FROM uc) inner_t) t");
  compare_gpu_vs_cpu(
    "SELECT k FROM (SELECT k FROM ua UNION ALL SELECT k FROM ub) l UNION ALL "
    "SELECT k FROM (SELECT k FROM uc UNION ALL SELECT k FROM uempty) r");
  // An arm that is itself an aggregate: its root is an unconditional sink, so the passthrough
  // sink ends up alone in its pipeline with an input port rather than inline in a scan's.
  compare_gpu_vs_cpu("SELECT k FROM (SELECT k FROM ua GROUP BY k) g UNION ALL SELECT k FROM ub");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL over a narrowed scan",
                 "[integration][gpu_execution][union][union_all]")
{
  // The compressed-schema pass treats UNION as a native carrier boundary and restores every arm,
  // so two arms never present different carriers for one logical column. uwide.k is BIGINT with
  // small values (a narrowing candidate) against an INTEGER arm, and must be pinned for a sidecar
  // to be installed at all (`scan_physical_schema` in sirius_plan_get.cpp).
  sirius::test::pinned_table_guard pin{*con, "uwide"};

  auto const before = sirius::test::get_compressed_materialization_stats(*con);

  compare_gpu_vs_cpu("SELECT k FROM uwide UNION ALL SELECT k FROM ua");
  compare_gpu_vs_cpu(
    "SELECT k, count(*) FROM (SELECT k FROM uwide UNION ALL SELECT k FROM ua) t GROUP BY k");

  auto const after = sirius::test::get_compressed_materialization_stats(*con);
  // Both halves of the claim: the arm scans narrow, the boundary restores. scan_columns_narrowed
  // stays 0 by design -- nothing survives narrow past the boundary.
  REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
  REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct EXCEPT and INTERSECT match the CPU",
                 "[integration][gpu_execution][set_operation]")
{
  compare_gpu_vs_cpu("SELECT k FROM ua EXCEPT SELECT k FROM ub");
  compare_gpu_vs_cpu("SELECT k FROM ua INTERSECT SELECT k FROM ub");
  for (std::string const keyword : {"EXCEPT", "INTERSECT"}) {
    compare_gpu_vs_cpu("SELECT k FROM ua " + keyword +
                       " (WITH m AS MATERIALIZED (SELECT k, v FROM ub) "
                       "SELECT m1.k FROM m m1 JOIN m m2 ON m1.v = m2.v)");
  }
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct set operations deduplicate surviving composite rows",
                 "[integration][gpu_execution][set_operation][setop_distinct]")
{
  std::string keyword;
  SECTION("EXCEPT") { keyword = "EXCEPT"; }
  SECTION("INTERSECT") { keyword = "INTERSECT"; }

  run_ok("CREATE TABLE setop_left (k INTEGER, v VARCHAR);");
  run_ok("CREATE TABLE setop_right (k INTEGER, v VARCHAR);");
  run_ok(
    "INSERT INTO setop_left VALUES "
    "(1, 'a'), (1, 'a'), (2, 'b'), (2, 'b'), (NULL, 'n'), (NULL, 'n'), "
    "(NULL, NULL), (NULL, NULL), (3, NULL), (3, NULL);");
  run_ok(
    "INSERT INTO setop_right VALUES "
    "(1, 'a'), (1, 'a'), (NULL, 'n'), (NULL, 'n'), (NULL, NULL), (3, 'other');");
  run_ok("CHECKPOINT;");

  compare_gpu_vs_cpu("SELECT k, v FROM setop_left " + keyword + " SELECT k, v FROM setop_right");
  compare_gpu_vs_cpu("SELECT k, v FROM setop_left " + keyword + " SELECT k, v FROM uempty");
  compare_gpu_vs_cpu("SELECT k, v FROM uempty " + keyword + " SELECT k, v FROM setop_right");
}

// The DISTINCT over a distinct set operation plans no aggregate of its own, so the set operation's
// deduplication alone must remove the duplicates. DISTINCTs that can still remove rows keep theirs.
TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution DISTINCT over set operations removes every duplicate",
                 "[integration][gpu_execution][set_operation][setop_distinct]")
{
  run_ok("CREATE TABLE dup_left (k INTEGER, v VARCHAR);");
  run_ok("CREATE TABLE dup_right (k INTEGER, v VARCHAR);");
  run_ok(
    "INSERT INTO dup_left VALUES (1, 'a'), (1, 'a'), (1, 'b'), (2, 'a'), (2, 'a'), "
    "(NULL, 'n'), (NULL, 'n'), (3, NULL), (3, NULL), (4, 'x');");
  run_ok("INSERT INTO dup_right VALUES (1, 'b'), (4, 'x'), (4, 'x'), (5, 'y');");
  run_ok("CHECKPOINT;");

  for (std::string const keyword : {"EXCEPT", "INTERSECT"}) {
    CAPTURE(keyword);
    std::string const setop =
      "SELECT k, v FROM dup_left " + keyword + " SELECT k, v FROM dup_right";
    compare_gpu_vs_cpu(setop);
    compare_gpu_vs_cpu("SELECT count(*) FROM (" + setop + ")");
    compare_gpu_vs_cpu_ordered(setop + " ORDER BY k NULLS FIRST, v NULLS FIRST");
    compare_gpu_vs_cpu("SELECT v, k FROM (" + setop + ")");
    compare_gpu_vs_cpu("SELECT DISTINCT k FROM (" + setop + ")");
    compare_gpu_vs_cpu("SELECT DISTINCT * FROM (SELECT k, v FROM dup_left " + keyword +
                       " ALL SELECT k, v FROM dup_right)");
  }
  compare_gpu_vs_cpu(
    "(SELECT k, v FROM dup_left EXCEPT SELECT k, v FROM dup_right) "
    "INTERSECT SELECT k, v FROM dup_left");
}

// The fixture tables above fit in one batch and take the whole-build BUILD_PROBE join. Here both
// inputs span many batches and hash partitions, and a zero hash-table cap moves the join to its
// partitioned STANDARD path. Two keys carry NULLs and the DOUBLE column mixes -0.0, 0.0 and NaN.
TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct EXCEPT and INTERSECT over many batches and partitions",
                 "[integration][gpu_execution][set_operation][setop_distinct]")
{
  bool const standard_join = GENERATE(false, true);
  INFO("STANDARD join path: " << standard_join);

  run_ok(
    "CREATE TABLE sdl AS SELECT "
    "  CASE WHEN i % 97 = 0 THEN NULL ELSE i % 500 END AS k1, "
    "  CASE WHEN i % 89 = 0 THEN NULL ELSE i % 7 END AS k2, "
    "  CASE i % 11 WHEN 0 THEN -(0.0::DOUBLE) WHEN 1 THEN 0.0::DOUBLE "
    "    WHEN 2 THEN 'NaN'::DOUBLE ELSE (i % 11)::DOUBLE END AS d "
    "FROM range(200000) t(i);");
  run_ok(
    "CREATE TABLE sdr AS SELECT "
    "  CASE WHEN i % 83 = 0 THEN NULL ELSE (i * 3) % 400 END AS k1, "
    "  CASE WHEN i % 79 = 0 THEN NULL ELSE i % 5 END AS k2, "
    "  CASE i % 13 WHEN 0 THEN 0.0::DOUBLE WHEN 1 THEN 'NaN'::DOUBLE "
    "    WHEN 2 THEN -(0.0::DOUBLE) ELSE (i % 13)::DOUBLE END AS d "
    "FROM range(150000) t(i);");
  run_ok("CREATE TABLE sdk AS SELECT i AS k1 FROM range(0, 500, 3) t(i);");
  run_ok("CHECKPOINT;");

  sirius::test::scoped_sirius_setting const scan_batches{
    *con, "scan_task_batch_size", std::uint64_t{65536}};
  sirius::test::scoped_sirius_setting const partitions{
    *con, "hash_partition_bytes", std::uint64_t{65536}};
  sirius::test::scoped_sirius_setting const build_cap{
    *con, "max_build_hash_table_bytes", standard_join ? std::uint64_t{0} : std::uint64_t{1} << 30};

  std::string const left  = "SELECT k1, k2, d FROM sdl";
  std::string const right = "SELECT k1, k2, d FROM sdr";
  for (auto const& keywords :
       std::array{std::pair{"EXCEPT", "INTERSECT"}, std::pair{"INTERSECT", "EXCEPT"}}) {
    std::string const keyword = keywords.first;
    std::string const other   = keywords.second;
    CAPTURE(keyword);
    auto const setop = [&](std::string const& lhs, std::string const& rhs) {
      return lhs + " " + keyword + " " + rhs;
    };
    compare_gpu_vs_cpu_canonical(setop(left, right));
    compare_gpu_vs_cpu_canonical(setop(right, left));
    compare_gpu_vs_cpu_canonical(setop(left, right + " WHERE false"));
    // A join above the set operation consumes its partitioned output.
    compare_gpu_vs_cpu("SELECT s.k2, count(*), count(s.d) FROM (" + setop(left, right) +
                       ") s JOIN sdk ON s.k1 = sdk.k1 GROUP BY s.k2");
    // A set operation as the left input of the other one.
    compare_gpu_vs_cpu_canonical("(" + setop(left, right) + ") " + other + " " + left +
                                 " WHERE k2 < 4");
  }
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution deduplicates a non-recursive WITH RECURSIVE UNION",
                 "[integration][gpu_execution][union]")
{
  // With no self-reference the binder plans a bare distinct UNION with no DISTINCT above it, so the
  // set-operation builder deduplicates it.
  std::string const cte = "WITH RECURSIVE c AS (SELECT k FROM ua UNION SELECT k FROM ub) ";
  compare_gpu_vs_cpu(cte + "SELECT * FROM c");
  compare_gpu_vs_cpu(
    "WITH RECURSIVE c AS (SELECT k, v FROM ua UNION SELECT k, v FROM ub) "
    "SELECT * FROM c");
  compare_gpu_vs_cpu(cte + "SELECT count(*) FROM c");

  // Four distinct keys and one NULL out of six input rows: the shape does need a dedup.
  run_ok("SET gpu_execution = false;");
  auto const count = con->Query(cte + "SELECT count(*) FROM c");
  run_ok("SET gpu_execution = true;");
  REQUIRE(count);
  REQUIRE_FALSE(count->HasError());
  REQUIRE(count->Collection().GetValue(0, 0).GetValue<int64_t>() == 5);
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution non-recursive WITH RECURSIVE UNION ignores default_collation",
                 "[integration][gpu_execution][union]")
{
  // DuckDB groups this shape on the raw values with no collation pushed, so under nocase 'A' and
  // 'a' stay apart on the CPU, and the GPU must keep them apart too.
  run_ok("CREATE TABLE ucase (v VARCHAR);");
  run_ok("INSERT INTO ucase VALUES ('A'), ('a'), ('A');");
  run_ok("CHECKPOINT;");
  sirius::test::scoped_sirius_setting collation{*con, "default_collation", "nocase"};
  compare_gpu_vs_cpu(
    "WITH RECURSIVE c AS (SELECT v FROM ucase UNION SELECT v FROM ucase) "
    "SELECT * FROM c");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION ALL over a materialized CTE arm",
                 "[integration][gpu_execution][union][union_all]")
{
  // Regression for the arity guard in sirius_plan_set_operation.cpp, which reads the arm's planned
  // `types`. Before DuckDB v2 a materialized CTE declared its materialization side, so a guard
  // reading that width declined an arm whose definition and body differ in width.
  //
  // Two traps when reshaping these. The CTE must stay MATERIALIZED, or a single-reference one is
  // inlined away and no CTE node reaches the planner. And the definition's width must be forced
  // apart from the body's: DuckDB's unused-column optimizer prunes the materialization down to
  // what the body reads, so a body that merely projects a subset leaves the two equal and tests
  // nothing.

  // Definition wider, 2 against 1. The self-join on v is what keeps v from being pruned.
  compare_gpu_vs_cpu(
    "SELECT k FROM ua "
    "UNION ALL "
    "(WITH m AS MATERIALIZED (SELECT k, v FROM ub) "
    " SELECT m1.k FROM m m1 JOIN m m2 ON m1.v = m2.v)");

  // Definition narrower, 1 against 2; the body widens with arithmetic. Direction does not matter.
  compare_gpu_vs_cpu(
    "SELECT k, k * 2 FROM ua "
    "UNION ALL "
    "(WITH m AS MATERIALIZED (SELECT k FROM ub) SELECT k, k * 2 FROM m)");

  // Widths coincide at 2, so this passed before the fix too. Kept so the guard cannot regress into
  // rejecting it.
  compare_gpu_vs_cpu(
    "SELECT k, v FROM ua "
    "UNION ALL "
    "(WITH m AS MATERIALIZED (SELECT k, v FROM ub) SELECT k, v FROM m)");

  // The CTE as arm 0, mirroring the "arm order reversed" case above: arm position must not matter.
  compare_gpu_vs_cpu(
    "(WITH m AS MATERIALIZED (SELECT k FROM ub) SELECT k, k * 2 FROM m) "
    "UNION ALL "
    "SELECT k, k * 2 FROM ua");

  // Control: written above the union, the CTE is its parent and never reaches the guard.
  compare_gpu_vs_cpu(
    "WITH m AS MATERIALIZED (SELECT k, v FROM ub) "
    "SELECT k FROM ua "
    "UNION ALL "
    "SELECT k FROM m");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION deduplicates across arms",
                 "[integration][gpu_execution][union]")
{
  // (3, 'c') is in both ua and ub, so each of these has a row to drop.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION SELECT k, v FROM ub");
  compare_gpu_vs_cpu("SELECT k FROM ua UNION SELECT k FROM ub");
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION BY NAME SELECT v, k FROM ub");
  // The second arm's NULL-filled `k` turns (NULL, 'n') into a row both arms produce.
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION BY NAME SELECT v FROM ua");
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k, v FROM ua UNION BY NAME SELECT v FROM ua) t");
  // A chain of distinct UNIONs merges into one N-ary union; a mixed chain does not.
  compare_gpu_vs_cpu("SELECT k FROM ua UNION SELECT k FROM ub UNION SELECT k FROM uc");
  compare_gpu_vs_cpu("SELECT k FROM ua UNION ALL SELECT k FROM ub UNION SELECT k FROM uc");
  compare_gpu_vs_cpu_ordered("SELECT k FROM ua UNION SELECT k FROM ub ORDER BY k NULLS LAST");
  compare_gpu_vs_cpu_ordered(
    "SELECT k FROM ua UNION SELECT k FROM ub ORDER BY k DESC NULLS LAST LIMIT 3");
  // The same set written as a user's DISTINCT over a UNION ALL.
  compare_gpu_vs_cpu(
    "SELECT DISTINCT * FROM (SELECT k, v FROM ua UNION ALL SELECT k, v FROM ub) t");
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k, v FROM ua UNION SELECT k, v FROM ub) t");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION as an arm of UNION ALL",
                 "[integration][gpu_execution][union][union_all]")
{
  // The arm's root is the DISTINCT's merge, an unconditional sink feeding a passthrough sink.
  compare_gpu_vs_cpu("(SELECT k FROM ua UNION SELECT k FROM ub) UNION ALL SELECT k FROM ua");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM ((SELECT k FROM ua UNION SELECT k FROM ub) "
    "UNION ALL SELECT k FROM ua) t");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION reconciles types and keeps NULL",
                 "[integration][gpu_execution][union]")
{
  compare_gpu_vs_cpu("SELECT k FROM ua UNION SELECT k FROM uwide");
  compare_gpu_vs_cpu("SELECT k FROM ua UNION SELECT 3");
  // A self-union collapses to ua's distinct keys, with one NULL.
  compare_gpu_vs_cpu("SELECT k FROM ua UNION SELECT k FROM ua");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION with an empty arm",
                 "[integration][gpu_execution][union]")
{
  compare_gpu_vs_cpu("SELECT k, v FROM ua UNION SELECT k, v FROM uempty");
  compare_gpu_vs_cpu("SELECT k, v FROM uempty UNION SELECT k, v FROM ua");
  // Every arm empty: the dedup runs over zero input rows.
  compare_gpu_vs_cpu("SELECT k FROM uempty UNION SELECT k FROM uempty");
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k FROM uempty UNION SELECT k FROM uempty) t");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution UNION casts DECIMAL and date-time arms to the union type",
                 "[integration][gpu_execution][union][union_all]")
{
  // The binder casts each arm to the union's type inside the arm: DECIMAL(15,2) and DECIMAL(10,4)
  // meet at DECIMAL(17,4), DATE widens to TIMESTAMP. Each table pairs a value equal across arms
  // after the cast with one that is not, so the distinct UNION must merge exactly one.
  run_ok("CREATE TABLE udec2 (d DECIMAL(15,2));");
  run_ok("CREATE TABLE udec4 (d DECIMAL(10,4));");
  run_ok("CREATE TABLE udate (t DATE);");
  run_ok("CREATE TABLE uts (t TIMESTAMP);");
  run_ok("INSERT INTO udec2 VALUES (1.50), (1234567890123.25), (NULL);");
  run_ok("INSERT INTO udec4 VALUES (1.5000), (2.0001), (NULL);");
  run_ok("INSERT INTO udate VALUES ('2024-01-01'), ('2024-02-29'), (NULL);");
  run_ok("INSERT INTO uts VALUES ('2024-01-01 00:00:00'), ('2024-01-01 12:30:00'), (NULL);");
  run_ok("CHECKPOINT;");

  compare_gpu_vs_cpu("SELECT d FROM udec2 UNION ALL SELECT d FROM udec4");
  compare_gpu_vs_cpu("SELECT d FROM udec2 UNION SELECT d FROM udec4");
  compare_gpu_vs_cpu("SELECT t FROM udate UNION ALL SELECT t FROM uts");
  compare_gpu_vs_cpu("SELECT t FROM udate UNION SELECT t FROM uts");
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION over a collated VARCHAR key falls back at plan time",
                 "[integration][gpu_execution][union]")
{
  // The binder pushes a collation over the distinct UNION's targets as it does for SELECT
  // DISTINCT, so under a non-binary default_collation the DISTINCT lowering refuses the key. Every
  // `v` is already lower case, so each nocase group holds one value and the CPU runs agree.
  auto const prior_result = con->Query("SELECT current_setting('default_collation');");
  REQUIRE(prior_result);
  REQUIRE_FALSE(prior_result->HasError());
  auto const prior = prior_result->Collection().GetValue(0, 0);
  {
    sirius::test::scoped_sirius_setting collation{*con, "default_collation", "nocase"};
    expect_plan_fallback_matches_cpu("SELECT v FROM ua UNION SELECT v FROM ub");
  }
  // A failed restore would leak nocase into every later integration case.
  REQUIRE(con->Query("SELECT current_setting('default_collation');")->Collection().GetValue(0, 0) ==
          prior);
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION over a narrowed scan",
                 "[integration][gpu_execution][union]")
{
  // As in the UNION ALL case above: the narrowed arm is restored at the union, below the DISTINCT.
  sirius::test::pinned_table_guard pin{*con, "uwide"};

  auto const before = sirius::test::get_compressed_materialization_stats(*con);
  compare_gpu_vs_cpu("SELECT k FROM uwide UNION SELECT k FROM ua");
  auto const after = sirius::test::get_compressed_materialization_stats(*con);
  REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
  REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION over identical arms",
                 "[integration][gpu_execution][union]")
{
  // DuckDB's common-subplan pass turns a subplan of two or more operators repeated across the arms
  // into a materialized CTE under the DISTINCT. Fail loudly if it stops doing so.
  std::string const query = "SELECT k FROM ua GROUP BY k UNION SELECT k FROM ua GROUP BY k";
  REQUIRE_THAT(cpu_plan_text(query), Catch::Matchers::ContainsSubstring("CTE"));
  compare_gpu_vs_cpu(query);
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION over a shared subplan whose definition is wider",
                 "[integration][gpu_execution][union]")
{
  // The CTE's definition is (v VARCHAR, k INTEGER) and its body one INTEGER column, so a
  // compressed-schema pass that read the definition would give column 0 a STRING carrier. The pin
  // makes the pass run.
  std::string const query =
    "SELECT k FROM (SELECT v, k FROM ua GROUP BY v, k) s "
    "UNION SELECT length(v)::INTEGER FROM (SELECT v, k FROM ua GROUP BY v, k) s";
  REQUIRE_THAT(cpu_plan_text(query), Catch::Matchers::ContainsSubstring("CTE"));

  sirius::test::pinned_table_guard pin{*con, "ua"};

  auto const before = sirius::test::get_compressed_materialization_stats(*con);
  compare_gpu_vs_cpu(query);
  auto const after = sirius::test::get_compressed_materialization_stats(*con);
  REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
}

TEST_CASE_METHOD(UnionFixture,
                 "gpu_execution distinct UNION over HUGEINT aggregates",
                 "[integration][gpu_execution][union]")
{
  // The DISTINCT groups on the union's HUGEINT sums.
  compare_gpu_vs_cpu("SELECT sum(k) FROM ua UNION SELECT sum(k) FROM ub");
  // Equal sums must collapse to one row.
  compare_gpu_vs_cpu("SELECT sum(k) FROM ua UNION SELECT sum(k) FROM ua");
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT sum(k) FROM ua UNION SELECT sum(k) FROM ua) t");
}

namespace {

/// One zero and one NaN in each table, with opposite signs, so every pair that DuckDB groups as one
/// value reaches the merge from different arms.
class UnionFloatFixture : public sirius::test::GpuExecutionFixture {
 public:
  UnionFloatFixture()
  {
    run_ok("CREATE TABLE fa (d DOUBLE, f REAL);");
    run_ok("CREATE TABLE fb (d DOUBLE, f REAL);");
    run_ok(
      "INSERT INTO fa VALUES "
      "(-(0.0::DOUBLE), -(0.0::REAL)), ('NaN'::DOUBLE, 'NaN'::REAL), (1.5, 1.5);");
    run_ok(
      "INSERT INTO fb VALUES "
      "(0.0, 0.0), (-('NaN'::DOUBLE), -('NaN'::REAL)), (1.5, 1.5);");
    run_ok("CHECKPOINT;");

    // A `-0.0` literal parses as DECIMAL, so prove fa stored a negative zero. Read on the CPU and
    // restore GPU execution before asserting, since the connection is shared.
    run_ok("SET gpu_execution = false;");
    auto const d_zeros = con->Query("SELECT count(*) FROM fa WHERE d = 0 AND signbit(d)");
    auto const f_zeros = con->Query("SELECT count(*) FROM fa WHERE f = 0 AND signbit(f)");
    run_ok("SET gpu_execution = true;");
    for (auto* zeros : {d_zeros.get(), f_zeros.get()}) {
      REQUIRE(zeros);
      REQUIRE_FALSE(zeros->HasError());
      REQUIRE(zeros->Collection().GetValue(0, 0).GetValue<int64_t>() == 1);
    }
  }
};

}  // namespace

TEST_CASE_METHOD(UnionFloatFixture,
                 "gpu_execution distinct UNION groups NaN and signed zero across arms",
                 "[integration][gpu_execution][union]")
{
  // Either engine may keep either member of a pair, so the rows compare canonicalized and the
  // count compares exactly.
  SECTION("DOUBLE")
  {
    compare_gpu_vs_cpu_canonical("SELECT d FROM fa UNION SELECT d FROM fb");
    compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT d FROM fa UNION SELECT d FROM fb) t");
  }

  SECTION("REAL")
  {
    compare_gpu_vs_cpu_canonical("SELECT f FROM fa UNION SELECT f FROM fb");
    compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT f FROM fa UNION SELECT f FROM fb) t");
  }
}

namespace {

class DistinctSetOperationFloatFixture : public sirius::test::GpuExecutionFixture {
 public:
  DistinctSetOperationFloatFixture()
  {
    run_ok("CREATE TABLE set_float_l (d DOUBLE, f REAL);");
    run_ok("CREATE TABLE set_float_r (d DOUBLE, f REAL);");
    run_ok("CREATE TABLE set_float_no_nan (d DOUBLE, f REAL);");
    run_ok(
      "INSERT INTO set_float_l VALUES "
      "(-(0.0::DOUBLE), -(0.0::REAL)), (0.0, 0.0), "
      "('NaN'::DOUBLE, 'NaN'::REAL), ('NaN'::DOUBLE, 'NaN'::REAL), "
      "('Infinity'::DOUBLE, 'Infinity'::REAL), ('Infinity'::DOUBLE, 'Infinity'::REAL), "
      "('-Infinity'::DOUBLE, '-Infinity'::REAL), (NULL, NULL), (NULL, NULL), "
      "(1.5, 1.5), (2.5, 2.5), (2.5, 2.5);");
    run_ok(
      "INSERT INTO set_float_r VALUES "
      "(0.0::DOUBLE, 0.0::REAL), (0.0, 0.0), "
      "(-('NaN'::DOUBLE), -('NaN'::REAL)), (-('NaN'::DOUBLE), -('NaN'::REAL)), "
      "('Infinity'::DOUBLE, 'Infinity'::REAL), ('-Infinity'::DOUBLE, '-Infinity'::REAL), "
      "(NULL, NULL), (NULL, NULL), (1.5, 1.5), (3.5, 3.5), (3.5, 3.5);");
    run_ok(
      "INSERT INTO set_float_no_nan VALUES "
      "(0.0::DOUBLE, 0.0::REAL), ('Infinity'::DOUBLE, 'Infinity'::REAL), "
      "('-Infinity'::DOUBLE, '-Infinity'::REAL), (NULL, NULL), (1.5, 1.5), (3.5, 3.5);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(DistinctSetOperationFloatFixture,
                 "gpu_execution distinct set operations match floating key equality",
                 "[integration][gpu_execution][set_operation][setop_float]")
{
  std::string const column  = GENERATE("d", "f");
  std::string const keyword = GENERATE("EXCEPT", "INTERSECT");
  INFO(column << " " << keyword);
  auto compare = [&](std::string const& left, std::string const& right) {
    auto const sql =
      "SELECT " + column + " FROM " + left + " " + keyword + " SELECT " + column + " FROM " + right;
    compare_gpu_vs_cpu_canonical(sql);
    compare_gpu_vs_cpu("SELECT count(*) FROM (" + sql + ") q");
  };
  compare("set_float_l", "set_float_r");
  compare("set_float_r", "set_float_l");
  // EXCEPT must retain one NaN here; the matching-arm case alone removes every left NaN.
  compare("set_float_l", "set_float_no_nan");
  compare("set_float_l", "set_float_r WHERE false");
}
