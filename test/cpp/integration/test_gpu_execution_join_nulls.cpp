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

// GPU-vs-CPU correctness for NULL keys in joins (issue #1095): equi-joins never
// match NULL keys (NULL != NULL), NULL-padding for LEFT/RIGHT/FULL OUTER, and
// NULL handling in SEMI/ANTI (EXISTS / NOT EXISTS) and MARK (IN) joins.
//
// Every query goes through the shared file-backed GpuExecutionFixture, which
// runs it once on the GPU (asserting a real GPU execution with no fallback) and
// once on DuckDB CPU, then compares the results (order-insensitive).

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

// Two tables with NULL keys on both sides, plus duplicate keys (10 on the left,
// 20 on the right) so match multiplicity is exercised, and non-overlapping keys
// (left 30, right 40) so each side has an unmatched non-NULL row.
class JoinNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  JoinNullFixture()
  {
    run_ok("CREATE TABLE l (id INTEGER, k INTEGER);");
    run_ok("CREATE TABLE r (id INTEGER, k INTEGER);");
    run_ok("INSERT INTO l VALUES (1, 10), (2, 20), (3, NULL), (4, 30), (5, 10);");
    run_ok("INSERT INTO r VALUES (1, 10), (2, 20), (3, 20), (4, NULL), (5, 40);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution INNER join does not match NULL keys",
                 "[integration][gpu_execution][join][nulls]")
{
  // NULL = NULL is UNKNOWN, so rows with a NULL key (l.3, r.4) never join.
  compare_gpu_vs_cpu("SELECT l.id AS lid, r.id AS rid FROM l JOIN r ON l.k = r.k");
  // Explicit: a NULL-keyed left row joins nothing.
  compare_gpu_vs_cpu("SELECT l.id FROM l JOIN r ON l.k = r.k WHERE l.k IS NULL");
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution LEFT join NULL-pads unmatched (including NULL-key) rows",
                 "[integration][gpu_execution][join][nulls]")
{
  compare_gpu_vs_cpu("SELECT l.id AS lid, r.id AS rid FROM l LEFT JOIN r ON l.k = r.k");
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution RIGHT join NULL-pads unmatched (including NULL-key) rows",
                 "[integration][gpu_execution][join][nulls]")
{
  compare_gpu_vs_cpu("SELECT l.id AS lid, r.id AS rid FROM l RIGHT JOIN r ON l.k = r.k");
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution FULL OUTER join NULL-pads both sides",
                 "[integration][gpu_execution][join][nulls]")
{
  compare_gpu_vs_cpu("SELECT l.id AS lid, r.id AS rid FROM l FULL OUTER JOIN r ON l.k = r.k");
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution SEMI / ANTI join (EXISTS / NOT EXISTS) with NULL keys",
                 "[integration][gpu_execution][join][nulls]")
{
  // EXISTS/NOT EXISTS use the join key equality (NULL != NULL), so the NULL-key
  // left row is absent from SEMI and present in ANTI.
  compare_gpu_vs_cpu("SELECT l.id FROM l WHERE EXISTS (SELECT 1 FROM r WHERE r.k = l.k)");
  compare_gpu_vs_cpu("SELECT l.id FROM l WHERE NOT EXISTS (SELECT 1 FROM r WHERE r.k = l.k)");
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution MARK join (IN) emits TRUE/FALSE/NULL",
                 "[integration][gpu_execution][join][nulls]")
{
  // IN produces a three-valued mark: TRUE when a key matches, NULL when the probe
  // is NULL or no match exists while the build side contains a NULL, else FALSE.
  //
  // The full build side contains a NULL, so an unmatched non-NULL probe (l.k=30)
  // yields NULL, never FALSE -- this exercises TRUE and NULL.
  compare_gpu_vs_cpu("SELECT l.id, l.k IN (SELECT k FROM r) AS m FROM l");
  // A NULL-free build side lets an unmatched non-NULL probe produce FALSE: l.k=30
  // is absent from {10,20,40} with no NULL present, so the mark is FALSE (while
  // l.k=NULL is still NULL and matches are TRUE) -- this exercises FALSE.
  compare_gpu_vs_cpu("SELECT l.id, l.k IN (SELECT k FROM r WHERE k IS NOT NULL) AS m FROM l");
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution mixed SEMI and ANTI joins keep matches after NULL build rows",
                 "[integration][gpu_execution][join][nulls][mixed_join]")
{
  run_ok("CREATE TABLE mixed_left (k BIGINT, v UINTEGER);");
  run_ok("INSERT INTO mixed_left VALUES (1, 134), (2, 7), (NULL, 3);");
  run_ok("CREATE TABLE mixed_right (k BIGINT, v UINTEGER);");
  run_ok("INSERT INTO mixed_right VALUES (1, NULL), (1, 0), (2, NULL);");
  run_ok("CHECKPOINT;");

  compare_gpu_vs_cpu(
    "SELECT l.* FROM mixed_left l SEMI JOIN mixed_right r "
    "ON l.k = r.k AND l.v <> r.v");
  compare_gpu_vs_cpu(
    "SELECT l.* FROM mixed_left l ANTI JOIN mixed_right r "
    "ON l.k = r.k AND l.v <> r.v");
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution mixed SEMI and ANTI nullable residuals with cast hash keys",
                 "[integration][gpu_execution][join][nulls][mixed_join]")
{
  const std::string join_type = GENERATE("SEMI", "ANTI");
  const bool right_family     = GENERATE(false, true);
  const std::string residual  = GENERATE("<>", "IS NOT DISTINCT FROM");
  run_ok("CREATE TABLE cast_big (id INTEGER, k BIGINT, v INTEGER);");
  run_ok(
    "INSERT INTO cast_big VALUES "
    "(1, 1, NULL), (2, 1, 0), (3, 2, NULL), (4, NULL, 3), (5, 3, 8);");
  run_ok("INSERT INTO cast_big SELECT 10 + i, 1, 0 FROM range(20) t(i);");
  run_ok("CREATE TABLE cast_small (id INTEGER, k SMALLINT, v INTEGER);");
  run_ok(
    "INSERT INTO cast_small VALUES "
    "(1, 1, NULL), (2, 1, 134), (3, 2, 7), (4, NULL, 3), (5, 4, 9);");
  run_ok("CHECKPOINT;");

  // The larger table stays on the physical left. Selecting the small table produces
  // RIGHT_SEMI/RIGHT_ANTI; both orientations have NULLs in the residual's build column.
  // INTEGER is unsupported by cuDF AST casts. The keys must stay on the hash path,
  // and SELECT l.* checks that build validity columns do not leak into the output.
  const std::string lhs = right_family ? "cast_small" : "cast_big";
  const std::string rhs = right_family ? "cast_big" : "cast_small";
  const auto query      = "SELECT l.* FROM " + lhs + " l " + join_type + " JOIN " + rhs +
                     " r ON CAST(l.k AS INTEGER) = CAST(r.k AS INTEGER) AND l.v " + residual +
                     " r.v";
  INFO(query);
  run_ok("SET gpu_execution = false;");
  auto plan = con->Query("EXPLAIN " + query);
  REQUIRE(plan);
  REQUIRE_FALSE(plan->HasError());
  const auto expected_type = (right_family ? "RIGHT_" : "") + join_type;
  REQUIRE(plan->ToString().find("Join Type: " + expected_type) != std::string::npos);
  compare_gpu_vs_cpu(query);
}

TEST_CASE_METHOD(JoinNullFixture,
                 "gpu_execution selective mixed SEMI and ANTI joins with one NULL residual",
                 "[integration][gpu_execution][join][nulls][mixed_join]")
{
  const std::string join_type = GENERATE("SEMI", "ANTI");
  const bool right_family     = GENERATE(false, true);
  // Only one build residual is NULL in either orientation. The equality keys narrow five
  // billion possible row pairs to 50,000 candidates; a conditional full-predicate join loses
  // that selectivity. Keep this a result regression rather than a hardware-dependent timer.
  run_ok(
    "CREATE TABLE selective_big AS SELECT i::INTEGER AS k, "
    "CASE WHEN i = 0 THEN NULL ELSE 1 END AS v FROM range(100000) t(i);");
  run_ok(
    "CREATE TABLE selective_small AS SELECT i::INTEGER AS k, "
    "CASE WHEN i = 0 THEN NULL ELSE 0 END AS v FROM range(50000) t(i);");
  run_ok("CHECKPOINT;");
  const std::string lhs = right_family ? "selective_small" : "selective_big";
  const std::string rhs = right_family ? "selective_big" : "selective_small";
  compare_gpu_vs_cpu("SELECT count(*) FROM " + lhs + " l " + join_type + " JOIN " + rhs +
                     " r ON l.k = r.k AND l.v <> r.v");
}
