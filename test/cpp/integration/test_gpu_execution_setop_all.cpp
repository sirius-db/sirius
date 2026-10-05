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

// GPU-vs-CPU parity for `EXCEPT ALL` and `INTERSECT ALL`, which `plan_except_intersect` lowers to
// tagged `UNION ALL`, a grouped tag sum, a copies projection and `sirius_physical_replicate`.
// compare_gpu_vs_cpu asserts one GPU execution and no fallback, so each case proves the lowering
// ran. Floating-point keys are refused at plan time and checked with
// expect_plan_fallback_matches_cpu.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <cstdint>
#include <string>

namespace {

using sirius::test::scoped_sirius_setting;

// `sa` and `sb` hold the multiplicities of every relation between m and n, NULL keys included:
// 1 is in sa 3 times and sb once, 2 once and twice, 3 twice and twice, 4 only in sa, 5 only in sb,
// NULL twice and once.
class SetOpAllFixture : public sirius::test::GpuExecutionFixture {
 public:
  SetOpAllFixture()
  {
    run_ok("CREATE TABLE sa (k INTEGER, v VARCHAR);");
    run_ok("CREATE TABLE sb (k INTEGER, v VARCHAR);");
    run_ok("CREATE TABLE sc (k INTEGER, v VARCHAR);");
    run_ok("CREATE TABLE sempty (k INTEGER, v VARCHAR);");
    run_ok(
      "INSERT INTO sa VALUES (1, 'a'), (1, 'a'), (1, 'a'), (2, 'b'), (3, 'c'), (3, 'c'), "
      "(4, 'd'), (NULL, 'n'), (NULL, 'n'), (1, NULL);");
    run_ok(
      "INSERT INTO sb VALUES (1, 'a'), (2, 'b'), (2, 'b'), (3, 'c'), (3, 'c'), (5, 'e'), "
      "(NULL, 'n'), (1, NULL), (1, NULL);");
    run_ok("INSERT INTO sc VALUES (1, 'a'), (3, 'c'), (3, 'c'), (3, 'c');");
    run_ok(
      "CREATE TABLE styped AS SELECT "
      "  (i % 4)::BIGINT AS i64, (i % 3)::SMALLINT AS i16, (i % 2)::TINYINT AS i8, "
      "  ((i % 5) * 1.25)::DECIMAL(12, 2) AS dec, (DATE '2024-01-01' + (i % 6)::INTEGER) AS d, "
      "  (TIMESTAMP '2024-01-01 00:00:00' + to_seconds(i % 7)) AS ts, (i % 2 = 0) AS b, "
      "  CASE WHEN i % 9 = 0 THEN NULL ELSE repeat('x', (i % 4)::INTEGER) END AS s "
      "FROM range(60) t(i);");
    run_ok(
      "CREATE TABLE stypedb AS SELECT * FROM styped WHERE i64 <> 1 "
      "UNION ALL SELECT * FROM styped WHERE i16 = 2;");
    run_ok(
      "CREATE TABLE sfloat AS SELECT * FROM (VALUES (0.0::DOUBLE, 0.0::REAL), "
      "(-0.0::DOUBLE, -0.0::REAL), ('nan'::DOUBLE, 'nan'::REAL), (NULL, NULL), "
      "(1.5::DOUBLE, 1.5::REAL)) "
      "t(d, r);");
    run_ok("CHECKPOINT;");
  }

  //! Compares both ALL forms of @p left against @p right.
  void compare_both(std::string const& left, std::string const& right)
  {
    compare_gpu_vs_cpu(left + " EXCEPT ALL " + right);
    compare_gpu_vs_cpu(left + " INTERSECT ALL " + right);
  }
};

}  // namespace

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL and INTERSECT ALL keep multiplicities",
                 "[integration][gpu_execution][setop_all]")
{
  compare_both("SELECT k FROM sa", "SELECT k FROM sb");
  // Arms swapped: EXCEPT ALL is not symmetric, and INTERSECT ALL must not depend on arm order.
  compare_both("SELECT k FROM sb", "SELECT k FROM sa");
  compare_both("SELECT k FROM sa", "SELECT k FROM sa");
}

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL and INTERSECT ALL over several columns",
                 "[integration][gpu_execution][setop_all]")
{
  // (1, NULL) is in sa once and sb twice; a NULL in any column is one value for grouping.
  compare_both("SELECT k, v FROM sa", "SELECT k, v FROM sb");
  compare_both("SELECT v, k FROM sb", "SELECT v, k FROM sa");
  compare_both("SELECT v FROM sa", "SELECT v FROM sb");
}

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL and INTERSECT ALL over each key type",
                 "[integration][gpu_execution][setop_all]")
{
  for (std::string const column : {"i64", "i16", "i8", "dec", "d", "ts", "b", "s"}) {
    compare_both("SELECT " + column + " FROM styped", "SELECT " + column + " FROM stypedb");
  }
  compare_both("SELECT * FROM styped", "SELECT * FROM stypedb");
}

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL and INTERSECT ALL with an empty input",
                 "[integration][gpu_execution][setop_all]")
{
  // Empty at run time: an input that yields no batches.
  compare_both("SELECT k, v FROM sa", "SELECT k, v FROM sempty");
  compare_both("SELECT k, v FROM sempty", "SELECT k, v FROM sa");
  compare_both("SELECT k, v FROM sempty", "SELECT k, v FROM sempty");
  // An input filtered to nothing at run time.
  compare_gpu_vs_cpu("SELECT k FROM sa WHERE k > 1 EXCEPT ALL SELECT k FROM sb WHERE k > 100");
  compare_gpu_vs_cpu("SELECT k FROM sa WHERE k > 100 INTERSECT ALL SELECT k FROM sb");
  // An input empty at plan time, whether or not DuckDB folds the set operation away.
  compare_gpu_vs_cpu("SELECT k FROM sa WHERE k > 1 EXCEPT ALL SELECT k FROM sb WHERE false");
  compare_gpu_vs_cpu("SELECT k FROM sa EXCEPT ALL SELECT k FROM sb WHERE false");
  compare_gpu_vs_cpu("SELECT k FROM sa INTERSECT ALL SELECT k FROM sb WHERE false");
}

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL and INTERSECT ALL inside larger queries",
                 "[integration][gpu_execution][setop_all]")
{
  // Under a join and under an aggregate, REPLICATE feeds a PARTITION as a pipeline sink.
  compare_gpu_vs_cpu(
    "SELECT t.k, c.v FROM (SELECT k FROM sa INTERSECT ALL SELECT k FROM sb) t "
    "JOIN sc c ON t.k = c.k");
  compare_gpu_vs_cpu(
    "SELECT k, count(*) FROM (SELECT k FROM sa EXCEPT ALL SELECT k FROM sb) t GROUP BY k");
  compare_gpu_vs_cpu("SELECT count(*) FROM (SELECT k, v FROM sa EXCEPT ALL SELECT k, v FROM sb) t");
  // One ALL form as an input of another.
  compare_gpu_vs_cpu("SELECT k FROM sa INTERSECT ALL SELECT k FROM sb EXCEPT ALL SELECT k FROM sc");
  compare_gpu_vs_cpu(
    "(SELECT k FROM sa EXCEPT ALL SELECT k FROM sc) INTERSECT ALL SELECT k FROM sb");
}

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL and INTERSECT ALL over many batches and partitions",
                 "[integration][gpu_execution][setop_all]")
{
  run_ok(
    "CREATE TABLE sbig_a AS SELECT CASE WHEN i % 97 = 0 THEN NULL ELSE i % 1000 END AS k, "
    "  'v' || (i % 7)::VARCHAR AS v FROM range(200000) t(i);");
  run_ok(
    "CREATE TABLE sbig_b AS SELECT CASE WHEN i % 89 = 0 THEN NULL ELSE i % 1500 END AS k, "
    "  'v' || (i % 5)::VARCHAR AS v FROM range(150000) t(i);");
  run_ok("CHECKPOINT;");
  // Small scan batches give each input many batches, and a small partition target makes the
  // aggregate exchange its partial sums across several partitions.
  scoped_sirius_setting const scan_batches{*con, "scan_task_batch_size", std::uint64_t{65536}};
  scoped_sirius_setting const partitions{*con, "hash_partition_bytes", std::uint64_t{65536}};
  compare_both("SELECT k, v FROM sbig_a", "SELECT k, v FROM sbig_b");
  compare_gpu_vs_cpu(
    "SELECT count(*), sum(k) FROM (SELECT k FROM sbig_a EXCEPT ALL SELECT k FROM sbig_b) t");
}

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL splits a hot key's copies across batches",
                 "[integration][gpu_execution][setop_all]")
{
  run_ok(
    "CREATE TABLE shot AS SELECT 7 AS k, 'hot' AS v FROM range(1000000) "
    "UNION ALL SELECT i AS k, 'cold' AS v FROM range(1000) t(i);");
  run_ok("CHECKPOINT;");
  // About 12 MB of copies against a 1 MiB cap: REPLICATE emits the hot key in many batches.
  scoped_sirius_setting const batch_bytes{*con, "concat_batch_bytes", std::uint64_t{1} << 20};
  compare_gpu_vs_cpu(
    "SELECT k, v, count(*) FROM (SELECT k, v FROM shot EXCEPT ALL SELECT k, v FROM sb) t "
    "GROUP BY k, v");
  compare_gpu_vs_cpu(
    "SELECT count(*) FROM (SELECT k, v FROM shot INTERSECT ALL "
    "SELECT k, v FROM shot WHERE k = 7 OR k < 10) t");
}

TEST_CASE_METHOD(SetOpAllFixture,
                 "gpu_execution EXCEPT ALL and INTERSECT ALL on floating-point keys fall back",
                 "[integration][gpu_execution][setop_all]")
{
  for (std::string const column : {"d", "r"}) {
    auto const left = "SELECT " + column + " FROM sfloat";
    auto const right =
      "SELECT " + column + " FROM sfloat WHERE " + column + " IS NULL OR " + column + " <> 1.5";
    expect_plan_fallback_matches_cpu(left + " EXCEPT ALL " + right);
    expect_plan_fallback_matches_cpu(left + " INTERSECT ALL " + right);
  }
}
