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

// GPU-vs-CPU correctness for the SQL-integrated per-row top-k vector join: a LATERAL
// `ORDER BY dist LIMIT k` subquery (what NEAREST BY desugars to) is recognized from DuckDB's
// decorrelated DELIM_JOIN plan and executed by sirius_physical_vector_topk_join.
// compare_gpu_vs_cpu asserts the query ran on the GPU (one execution, no fallback).

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

namespace {

class VectorTopkJoinFixture : public sirius::test::GpuExecutionFixture {
 public:
  VectorTopkJoinFixture()
  {
    // Every left row has distinct L2 distances to the right rows, so the top-k set is unambiguous.
    run_ok("CREATE TABLE l (id INTEGER, v FLOAT[3]);");
    run_ok("CREATE TABLE r (id INTEGER, v FLOAT[3]);");
    run_ok("INSERT INTO l VALUES (1, [0,0,0]), (2, [10,0,0]), (3, [0,7,1]);");
    run_ok(
      "INSERT INTO r VALUES (1, [1,0,0]), (2, [0,2,0]), (3, [0.5,0,3.5]), (4, [9,0.5,0]), "
      "(5, [-4,1,1]), (6, [0,6,2]);");

    // Non-zero left vectors for cosine, with distinct similarities for the top 2.
    run_ok("CREATE TABLE lc (id INTEGER, v FLOAT[3]);");
    run_ok("INSERT INTO lc VALUES (1, [1,0,0]), (2, [0,1,1]);");

    run_ok("CREATE TABLE r_empty (id INTEGER, v FLOAT[3]);");
    run_ok("CREATE TABLE l_null (id INTEGER, v FLOAT[3]);");
    run_ok("INSERT INTO l_null VALUES (1, [0,0,0]), (2, NULL);");

    run_ok("CHECKPOINT;");
  }
};

TEST_CASE_METHOD(VectorTopkJoinFixture,
                 "gpu_execution L2 top-k LATERAL join matches CPU",
                 "[integration][gpu_execution][join][vss][vector_topk]")
{
  compare_gpu_vs_cpu(
    "SELECT l.id, rr.id FROM l, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) LIMIT 3) rr");
  // k = 1 plans without an UNNEST.
  compare_gpu_vs_cpu(
    "SELECT l.id, rr.id FROM l, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) LIMIT 1) rr");
  // k larger than the right side returns every right row.
  compare_gpu_vs_cpu(
    "SELECT l.id, rr.id FROM l, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) LIMIT 10) rr");
  // Aggregating over the join still keeps the top-k shape when a payload column is read.
  compare_gpu_vs_cpu(
    "SELECT count(*), sum(rid) FROM (SELECT rr.id AS rid FROM l, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) LIMIT 3) rr)");
}

TEST_CASE_METHOD(VectorTopkJoinFixture,
                 "gpu_execution NEAREST BY desugared form matches CPU",
                 "[integration][gpu_execution][join][vss][vector_topk]")
{
  // Exactly what `l JOIN r NEAREST 3 BY DISTANCE array_distance(l.v, r.v)` binds to; SELECT * also
  // carries both vector columns through the join.
  compare_gpu_vs_cpu(
    "SELECT * FROM l JOIN (SELECT * FROM r WHERE array_distance(l.v, r.v) IS NOT NULL "
    "ORDER BY array_distance(l.v, r.v) LIMIT 3) r ON TRUE");
  compare_gpu_vs_cpu(
    "SELECT l.id, r.id FROM l LEFT JOIN (SELECT * FROM r WHERE array_distance(l.v, r.v) IS NOT "
    "NULL ORDER BY array_distance(l.v, r.v) LIMIT 3) r ON TRUE");
}

// The distance column comes from the kernel (GPU expanded L2), which differs from DuckDB's
// array_distance in the low bits, so it is compared with a tolerance; ids compare exactly.
TEST_CASE_METHOD(VectorTopkJoinFixture,
                 "gpu_execution top-k join returns the ranking distance",
                 "[integration][gpu_execution][join][vss][vector_topk]")
{
  compare_gpu_vs_cpu_approx(
    "SELECT l.id, rr.id, rr.d FROM l, LATERAL "
    "(SELECT r.id, array_distance(l.v, r.v) AS d FROM r ORDER BY d LIMIT 3) rr",
    {2},
    1e-5);
  // A struct payload (all right columns plus the distance), selected out of order.
  compare_gpu_vs_cpu_approx(
    "SELECT rr.id, l.id, rr.d, rr.v, l.v FROM l, LATERAL "
    "(SELECT r.*, array_distance(l.v, r.v) AS d FROM r ORDER BY d LIMIT 2) rr",
    {2},
    1e-5);
}

TEST_CASE_METHOD(VectorTopkJoinFixture,
                 "gpu_execution cosine top-k join matches CPU",
                 "[integration][gpu_execution][join][vss][vector_topk]")
{
  // Similarity ranks largest first; the emitted value is the similarity itself.
  compare_gpu_vs_cpu_approx(
    "SELECT lc.id, rr.id, rr.s FROM lc, LATERAL (SELECT r.id, array_cosine_similarity(lc.v, r.v) "
    "AS s FROM r ORDER BY s DESC LIMIT 2) rr",
    {2},
    1e-5);
  compare_gpu_vs_cpu(
    "SELECT lc.id, rr.id FROM lc, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_cosine_distance(lc.v, r.v) LIMIT 2) rr");
}

TEST_CASE_METHOD(VectorTopkJoinFixture,
                 "gpu_execution LEFT top-k join and empty right side match CPU",
                 "[integration][gpu_execution][join][vss][vector_topk]")
{
  compare_gpu_vs_cpu(
    "SELECT l.id, rr.id FROM l LEFT JOIN LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) LIMIT 3) rr ON TRUE");
  // Empty right side: LEFT keeps every left row with NULLs, INNER returns nothing.
  compare_gpu_vs_cpu(
    "SELECT l.id, rr.id, rr.d FROM l LEFT JOIN LATERAL (SELECT r.id, array_distance(l.v, r.v) AS "
    "d FROM r_empty r ORDER BY d LIMIT 3) rr ON TRUE");
  compare_gpu_vs_cpu(
    "SELECT l.id, rr.id FROM l, LATERAL "
    "(SELECT r.id FROM r_empty r ORDER BY array_distance(l.v, r.v) LIMIT 3) rr");
}

TEST_CASE_METHOD(VectorTopkJoinFixture,
                 "top-k join shapes that are not taken over stay on CPU",
                 "[integration][gpu_execution][join][vss][vector_topk]")
{
  // Farthest-k (largest distance first) is not a nearest-neighbor query.
  expect_plan_fallback_matches_cpu(
    "SELECT l.id, rr.id FROM l, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) DESC LIMIT 3) rr");
  // A bare count over k = 1 prunes the arg_min away, leaving nothing that marks it as top-k.
  expect_plan_fallback_matches_cpu(
    "SELECT count(*) FROM (SELECT l.id FROM l, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) LIMIT 1) rr)");
}

TEST_CASE_METHOD(VectorTopkJoinFixture,
                 "top-k join rejects NULL vectors at runtime",
                 "[integration][gpu_execution][join][vss][vector_topk]")
{
  // The operator throws on a NULL vector; the engine's runtime fallback then answers on CPU.
  expect_gpu_fallback(
    "SELECT l.id, rr.id FROM l_null l, LATERAL "
    "(SELECT r.id FROM r ORDER BY array_distance(l.v, r.v) LIMIT 3) rr");
}

}  // namespace
