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
 * @file test_gpu_execution_vss_ann.cpp
 * @brief End-to-end tests for the pinned approximate (ANN) vector-search path.
 *
 * When a table is GPU-pinned (`pin_table tier='gpu'`) and a cuVS index covers its
 * vector column (`sirius_create_ann_index`, same metric), an
 * `ORDER BY array_distance(vec, <const>) LIMIT k` auto-routes to the ANN
 * source operator (a pinned IVF-Flat search + gather) instead of the exact
 * brute-force VSS path. These tests drive the whole pipeline we depend on:
 * pin -> build index -> auto-route -> ANN execute -> result.
 *
 * Exactness trick: the operator probes `min(n_lists, 32)` inverted lists, so an
 * index built with `n_lists <= 32` probes every list, the "approximate" search
 * then considers all vectors and is exact. That lets us reuse the strict
 * GPU-vs-CPU comparator: the ANN result must equal DuckDB's exact array_distance.
 * (The linear vec=[i,i,i] data is also tie-free, so the top-k set is unambiguous.)
 *
 * compare_gpu_vs_cpu() asserts exactly one GPU execution with zero fallbacks. A
 * regression in the ANN plumbing (source-root scheduling, split_ann_source, the
 * operator's execute) surfaces here as either a hard error or a fallback, both of
 * which fail the assertion. ANN-vs-ENN routing itself and the index internals are
 * pinned by the unit tests under test/cpp/vss/.
 *
 * NOTE: ENN returns identical results with zero fallbacks, so a silent
 * route-to-ENN regression still passes. Asserting the route needs an
 * observability hook.
 *
 * Data is checkpointed to disk before pinning / index build: both read on-disk
 * blocks through the native scan path, so WAL-resident rows would be invisible.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

using AnnFixture = sirius::test::GpuExecutionFixture;

TEST_CASE_METHOD(AnnFixture,
                 "gpu_execution vss ann - pinned IVF-Flat top-k, fast-path build",
                 "[integration][gpu_execution][array][vss][ann]")
{
  // vec=[i,i,i] as FLOAT[3]; distance to the origin is sqrt(3)*i, strictly
  // increasing in i, so the k nearest are rows 0..k-1 with no ties.
  run_ok(
    "CREATE TABLE test_ann_l2 AS "
    "SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");  // Persist to disk for the native GPU scan / pin read.

  // Pin FIRST, then build the index: the index build reads the vector column
  // straight from the GPU-pinned entry.
  run_ok("SELECT * FROM pin_table(name => 'test_ann_l2', tier => 'gpu', format => 'duckdb');");
  // n_lists <= 32 => the search probes every list => exact results.
  run_ok(
    "SELECT * FROM sirius_create_ann_index('test_ann_l2', 'vec', "
    "metric => 'l2sq', n_lists => 16);");

  // Nearest-5 by L2 distance to the origin, routes to ANN.
  compare_gpu_vs_cpu(
    "SELECT id FROM test_ann_l2 "
    "ORDER BY array_distance(vec, [0.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5;");

  // With OFFSET: exercises the operator's distance-sort + offset slice.
  compare_gpu_vs_cpu(
    "SELECT id FROM test_ann_l2 "
    "ORDER BY array_distance(vec, [0.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5 OFFSET 3;");

  // SELECT *: the FLOAT[3] vector column is gathered from the pinned table by the
  // returned row indices and round-trips back to a DuckDB ARRAY.
  compare_gpu_vs_cpu(
    "SELECT * FROM test_ann_l2 "
    "ORDER BY array_distance(vec, [0.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5;");

  // A query vector inside the dataset: nearest is the exact-match row, then
  // symmetric neighbours (i=999 & i=1001 tie, etc.). An ODD limit lands on
  // complete tie-shells so the top-k SET is unambiguous between GPU and CPU.
  compare_gpu_vs_cpu(
    "SELECT id FROM test_ann_l2 "
    "ORDER BY array_distance(vec, [1000.0, 1000.0, 1000.0]::FLOAT[3]) LIMIT 7;");

  run_ok("SELECT * FROM unpin_table('test_ann_l2');");
}

TEST_CASE_METHOD(AnnFixture,
                 "gpu_execution vss ann - slow-path build (index before pin) then query",
                 "[integration][gpu_execution][array][vss][ann]")
{
  run_ok(
    "CREATE TABLE test_ann_slow AS "
    "SELECT i AS id, [i, i, i]::FLOAT[3] AS vec FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");

  // Build the index BEFORE pinning: with no GPU-pinned entry to read from, the
  // index build takes the fallback (materialize the column via the native scan).
  run_ok(
    "SELECT * FROM sirius_create_ann_index('test_ann_slow', 'vec', "
    "metric => 'l2sq', n_lists => 16);");
  // Pin AFTER: the auto-route requires the table GPU-resident at query time.
  run_ok("SELECT * FROM pin_table(name => 'test_ann_slow', tier => 'gpu', format => 'duckdb');");

  compare_gpu_vs_cpu(
    "SELECT id FROM test_ann_slow "
    "ORDER BY array_distance(vec, [0.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5;");

  run_ok("SELECT * FROM unpin_table('test_ann_slow');");
}

TEST_CASE_METHOD(AnnFixture,
                 "gpu_execution vss ann - pinned IVF-Flat cosine top-k",
                 "[integration][gpu_execution][array][vss][ann]")
{
  // vec=[1,i,0]: the cosine distance to [1,0,0] is 1 - 1/sqrt(1+i^2), strictly
  // increasing in i and well-separated for small i, so the k nearest are rows
  // 0..k-1 with no ties or float ambiguity at the LIMIT boundary. (The [i,i,i]
  // trick can't be reused for cosine since all those rows share one direction.)
  run_ok(
    "CREATE TABLE test_ann_cos AS "
    "SELECT i AS id, [1.0, i, 0.0]::FLOAT[3] AS vec FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");

  run_ok("SELECT * FROM pin_table(name => 'test_ann_cos', tier => 'gpu', format => 'duckdb');");
  // metric => 'cosine' builds a CosineExpanded index; n_lists <= 32 => exact.
  run_ok(
    "SELECT * FROM sirius_create_ann_index('test_ann_cos', 'vec', "
    "metric => 'cosine', n_lists => 16);");

  compare_gpu_vs_cpu(
    "SELECT id FROM test_ann_cos "
    "ORDER BY array_cosine_distance(vec, [1.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5;");

  // OFFSET pins the cosine sort DIRECTION: a wrong (descending) sort would slice
  // the far rows here and mismatch the CPU top-k.
  compare_gpu_vs_cpu(
    "SELECT id FROM test_ann_cos "
    "ORDER BY array_cosine_distance(vec, [1.0, 0.0, 0.0]::FLOAT[3]) LIMIT 5 OFFSET 2;");

  run_ok("SELECT * FROM unpin_table('test_ann_cos');");
}
