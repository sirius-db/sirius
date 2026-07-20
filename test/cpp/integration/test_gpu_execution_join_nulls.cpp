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
//
// Cases tagged [!shouldfail] document confirmed GPU/CPU divergences (see
// NULL_TESTS_KNOWN_DIVERGENCES.md).

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
  compare_gpu_vs_cpu("SELECT l.id, l.k IN (SELECT k FROM r) AS m FROM l");
}
