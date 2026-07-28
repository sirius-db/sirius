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

// GPU-vs-CPU correctness for the DuckDB-native scan reconstructing the validity
// mask of wholly-NULL columns (issue #1218). A wholly-NULL column checkpoints to
// CONSTANT all-invalid validity with no on-disk bitmap; the scan must synthesize
// the null mask, otherwise the column reads back as sentinel values with no
// NULLs. Covers projection and filters over wholly-NULL columns across data
// types (INT/BIGINT/DOUBLE/DECIMAL/DATE/VARCHAR), a partially-NULL column and
// mixed NULL/valid columns (whose valid rows must survive), and layouts spanning
// multiple data segments and multiple row groups. Scalar aggregates over
// wholly-NULL columns are covered by test_gpu_execution_aggregate_nulls.cpp.
//
// Every query goes through the shared file-backed GpuExecutionFixture, which runs
// it on the GPU (asserting a real GPU execution, no fallback) and on DuckDB CPU,
// then compares the results.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {

// `id` is fully valid; `n_*` columns are entirely NULL (one per data type so each
// decode path is exercised); `part` is partially NULL (a real per-row validity
// bitmap).
class AllNullScanFixture : public sirius::test::GpuExecutionFixture {
 public:
  AllNullScanFixture()
  {
    run_ok(
      "CREATE TABLE ans ("
      "  id     INTEGER,"
      "  n_int  INTEGER,"
      "  n_big  BIGINT,"
      "  n_dbl  DOUBLE,"
      "  n_dec  DECIMAL(10,2),"
      "  n_date DATE,"
      "  part   INTEGER);");
    run_ok(
      "INSERT INTO ans SELECT i, NULL, NULL, NULL, NULL, NULL, "
      "CASE WHEN i % 2 = 0 THEN i ELSE NULL END "
      "FROM range(1, 9) AS t(i);");
    run_ok("CHECKPOINT;");
  }
};

}  // namespace

TEST_CASE_METHOD(AllNullScanFixture,
                 "gpu_execution wholly-NULL column projection preserves NULLs",
                 "[integration][gpu_execution][scan][nulls][allnull]")
{
  compare_gpu_vs_cpu("SELECT id, n_int, n_big, n_dbl, n_dec, n_date FROM ans");
}

TEST_CASE_METHOD(AllNullScanFixture,
                 "gpu_execution wholly-NULL column IS [NOT] NULL filters",
                 "[integration][gpu_execution][scan][nulls][allnull]")
{
  // Every row is NULL: IS NULL keeps all rows, IS NOT NULL keeps none.
  compare_gpu_vs_cpu("SELECT id FROM ans WHERE n_int IS NULL");
  compare_gpu_vs_cpu("SELECT id FROM ans WHERE n_int IS NOT NULL");
  compare_gpu_vs_cpu("SELECT id FROM ans WHERE n_date IS NULL");
}

TEST_CASE_METHOD(AllNullScanFixture,
                 "gpu_execution partially-NULL column keeps its valid rows",
                 "[integration][gpu_execution][scan][nulls][allnull]")
{
  // `part` has a real per-row validity bitmap; its valid rows must survive.
  compare_gpu_vs_cpu("SELECT id, part FROM ans");
  compare_gpu_vs_cpu("SELECT COUNT(part), SUM(part), MIN(part), MAX(part) FROM ans");
  compare_gpu_vs_cpu("SELECT id FROM ans WHERE part IS NOT NULL");
}

// A column whose rows are a constant non-NULL run followed by an all-NULL run
// must not have its valid rows masked. Exact on-disk segmentation is
// storage-internal, so this asserts correctness of the shape.
namespace {
class ConstantThenNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  ConstantThenNullFixture()
  {
    run_ok("CREATE TABLE mix (id INTEGER, c INTEGER);");
    run_ok(
      "INSERT INTO mix SELECT i, CASE WHEN i < 500 THEN 7 ELSE NULL END "
      "FROM range(1000) AS t(i);");
    run_ok("CHECKPOINT;");
  }
};
}  // namespace

TEST_CASE_METHOD(ConstantThenNullFixture,
                 "gpu_execution constant-valued run then all-NULL run in one row group",
                 "[integration][gpu_execution][scan][nulls][allnull]")
{
  // If the valid run were wrongly masked, COUNT(c)/SUM(c) would drop below 500/3500.
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(c), SUM(c), MIN(c), MAX(c) FROM mix");
  compare_gpu_vs_cpu("SELECT id FROM mix WHERE c IS NOT NULL");
  compare_gpu_vs_cpu("SELECT id FROM mix WHERE c IS NULL");
}

// A large column mixing a NULL region and a valid region: the valid rows must
// survive. Sized past one fixed-width segment to span multiple data segments
// (exact segmentation is storage-internal).
namespace {
class NullThenValidLargeFixture : public sirius::test::GpuExecutionFixture {
 public:
  NullThenValidLargeFixture()
  {
    run_ok(
      "CREATE TABLE big AS "
      "SELECT i AS id, CASE WHEN i < 70000 THEN NULL ELSE i END AS c "
      "FROM range(80000) AS t(i);");
    run_ok("CHECKPOINT;");
  }
};
}  // namespace

TEST_CASE_METHOD(NullThenValidLargeFixture,
                 "gpu_execution large NULL-run then valid-run column",
                 "[integration][gpu_execution][scan][nulls][allnull]")
{
  // COUNT(c)=10000 and SUM over [70000,80000) only hold if the valid trailing
  // rows were not masked.
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(c), SUM(c), MIN(c), MAX(c) FROM big");
}

// A wholly-NULL column spanning multiple row groups, with a ragged final row
// group (the row count is deliberately not a multiple of 8).
namespace {
class WhollyNullMultiRowGroupFixture : public sirius::test::GpuExecutionFixture {
 public:
  WhollyNullMultiRowGroupFixture()
  {
    run_ok(
      "CREATE TABLE big_null AS "
      "SELECT i AS id, CAST(NULL AS INTEGER) AS n FROM range(130001) AS t(i);");
    run_ok("CHECKPOINT;");
  }
};
}  // namespace

TEST_CASE_METHOD(WhollyNullMultiRowGroupFixture,
                 "gpu_execution wholly-NULL column spanning multiple row groups",
                 "[integration][gpu_execution][scan][nulls][allnull]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(n), SUM(n) FROM big_null");
}

// VARCHAR wholly-NULL columns are reconstructed too: the scan must produce NULLs
// rather than reading back empty/garbage strings.
namespace {
class AllNullVarcharFixture : public sirius::test::GpuExecutionFixture {
 public:
  AllNullVarcharFixture()
  {
    run_ok("CREATE TABLE ans_str (id INTEGER, s VARCHAR);");
    run_ok("INSERT INTO ans_str SELECT i, NULL FROM range(1, 9) AS t(i);");
    run_ok("CHECKPOINT;");
  }
};
}  // namespace

TEST_CASE_METHOD(AllNullVarcharFixture,
                 "gpu_execution wholly-NULL VARCHAR column",
                 "[integration][gpu_execution][scan][nulls][allnull]")
{
  compare_gpu_vs_cpu("SELECT id, s FROM ans_str");
  compare_gpu_vs_cpu("SELECT COUNT(s) FROM ans_str");
  compare_gpu_vs_cpu("SELECT id FROM ans_str WHERE s IS NULL");
}
