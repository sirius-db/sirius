/*
 * Copyright 2025, Sirius Contributors.
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
 * @file test_gpu_execution_vss.cpp
 * @brief End-to-end tests for brute-force vector search through gpu_execution.
 *
 * `ORDER BY <array-distance>(vec, <const>) LIMIT k` over a stored FLOAT[dim]
 * column is fused into the VSS operator on the GPU native-scan path
 * (GPU_SCAN -> VSS -> MERGE_VSS). Each query runs through transparent GPU
 * execution and through DuckDB CPU and the results are compared.
 *
 * compare_gpu_vs_cpu() also asserts exactly one GPU execution with zero
 * fallbacks, so a silent CPU fallback fails the test. The stats cannot tell
 * "ran via VSS" apart from "ran via an ordinary GPU TopN" (both are on GPU)
 * so these verify correctness on the GPU path; VSS-specific behavior (distance
 * magnitudes, the fused gather/merge) is pinned by the unit tests under
 * test/cpp/vss/. Note that `array_distance` is not supported on the generic GPU
 * TopN path, so if VSS matching regressed the query would fall back and trip the
 * zero-fallback assertion here.
 *
 * The data must be CHECKPOINTed to disk: the native scan reads on-disk blocks,
 * so rows still sitting in the WAL would be invisible to it.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

using VssFixture = sirius::test::GpuExecutionFixture;

TEST_CASE_METHOD(VssFixture,
                 "gpu_execution vss - brute-force top-k (array_distance)",
                 "[integration][gpu_execution][array][vss]")
{
  // vec=[i,i,i] as FLOAT[3]; distance to the origin is sqrt(3)*i, STRICTLY
  // increasing in i, so the k nearest are rows 0..k-1 with no distance ties:
  // GPU (cuVS L2Sqrt) and CPU (array_distance) agree on the exact row set.
  // 5000 rows (> STANDARD_VECTOR_SIZE) also forces a persistent on-disk segment.
  run_ok(
    "CREATE TABLE test_vss AS "
    "SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");  // Persist to disk for the native GPU scan.

  // Nearest-5 by L2 distance to the origin.
  compare_gpu_vs_cpu(
    "SELECT id FROM test_vss "
    "ORDER BY array_distance(vec, [0.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5;");

  // With OFFSET: exercises the merge's offset slice.
  compare_gpu_vs_cpu(
    "SELECT id FROM test_vss "
    "ORDER BY array_distance(vec, [0.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5 OFFSET 3;");

  // SELECT *: the FLOAT[3] vector column is gathered by the neighbor map and
  // round-trips back to a DuckDB ARRAY in the result (passthrough output column).
  compare_gpu_vs_cpu(
    "SELECT * FROM test_vss "
    "ORDER BY array_distance(vec, [0.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5;");
}

TEST_CASE_METHOD(VssFixture,
                 "gpu_execution vss - brute-force top-k (array_cosine_distance)",
                 "[integration][gpu_execution][array][vss]")
{
  // Distinct directions so cosine distances to [1,0,0] rank unambiguously:
  // id0 (same dir, 0) < id3 (45 deg) < {id1, id2} (orthogonal, tie): the tie
  // sits below the LIMIT-2 boundary, so the top-2 set {0,3} is well-defined.
  run_ok("CREATE TABLE test_vss_cos (id INTEGER, vec FLOAT[3]);");
  run_ok(
    "INSERT INTO test_vss_cos VALUES "
    "(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), "
    "(2, [0.0, 0.0, 1.0]), (3, [1.0, 1.0, 0.0]);");
  run_ok("CHECKPOINT;");

  compare_gpu_vs_cpu(
    "SELECT id FROM test_vss_cos "
    "ORDER BY array_cosine_distance(vec, [1.0, 0.0, 0.0]::FLOAT[3]) LIMIT 2;");
}
