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

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

namespace {
class UnsignedNarrowingFixture : public sirius::test::GpuExecutionFixture {
 public:
  UnsignedNarrowingFixture()
    : optimizer_guard(*con, "in_clause,compressed_materialization,late_materialization")
  {
    run_ok("SET expression_evaluator_strategy='ast_interpret';");
    run_ok("CREATE TABLE u (id INTEGER, k UBIGINT);");
    run_ok("CREATE TABLE s (id INTEGER, k BIGINT);");
    run_ok(
      "INSERT INTO u VALUES (1,0),(2,9223372036854775807),"
      "(3,9223372036854775808),(4,9223372036854775809),"
      "(5,18446744073709551615),(6,NULL),(7,0),"
      "(8,4294967295),(9,4294967296);");
    run_ok(
      "INSERT INTO s VALUES (1,0),(2,0),(3,-1),(4,-2),"
      "(5,9223372036854775807),(6,NULL),(7,42);");
    run_ok("CHECKPOINT;");
  }

  void check_rejection(const std::string& query)
  {
    run_ok("SET enable_duckdb_fallback=true;");
    expect_plan_fallback_matches_cpu(query);
    run_ok("SET enable_duckdb_fallback=false;");
    auto result = con->Query(query);
    run_ok("SET enable_duckdb_fallback=true;");
    REQUIRE(result);
    REQUIRE(result->HasError());
    INFO(result->GetError());
    REQUIRE(result->GetError().find("Unsupported") != std::string::npos);
  }

 private:
  sirius::test::disabled_optimizers_guard optimizer_guard;
};
}  // namespace

TEST_CASE_METHOD(UnsignedNarrowingFixture,
                 "mixed unsigned joins reject lossy HUGEINT keys before execution",
                 "[integration][gpu_execution][unsigned_narrowing]")
{
  for (auto join : {"JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL OUTER JOIN"}) {
    check_rejection("SELECT u.id,s.id FROM u " + std::string(join) + " s ON u.k=s.k");
  }
  check_rejection("SELECT u.id,s.id FROM u LEFT JOIN s ON u.k IS NOT DISTINCT FROM s.k");
  check_rejection("SELECT u.id,s.id FROM u JOIN s ON u.k>s.k");
  check_rejection("SELECT u.id,s.id FROM u JOIN s ON (u.k % 3::INTEGER)=s.k");
  check_rejection("SELECT u.id FROM u WHERE EXISTS (SELECT 1 FROM s WHERE s.k=u.k)");
}

TEST_CASE_METHOD(UnsignedNarrowingFixture,
                 "mixed unsigned expressions reject lossy HUGEINT operands",
                 "[integration][gpu_execution][unsigned_narrowing]")
{
  check_rejection("SELECT id,k = -1::BIGINT,k > 9223372036854775807::BIGINT FROM u");
  check_rejection("SELECT id FROM u WHERE k > -1::BIGINT");
  check_rejection("SELECT k % 3::INTEGER,COUNT(*) FROM u GROUP BY k % 3::INTEGER");
  check_rejection("SELECT id,CASE WHEN id=1 THEN -1::BIGINT ELSE k END FROM u");
  check_rejection("SELECT id,TRY_CAST(k AS HUGEINT) FROM u");
}

TEST_CASE_METHOD(UnsignedNarrowingFixture,
                 "direct unsigned narrowing casts fall back before execution",
                 "[integration][gpu_execution][unsigned_narrowing]")
{
  for (auto target :
       {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER"}) {
    check_rejection("SELECT id,TRY_CAST(k AS " + std::string(target) + ") FROM u");
  }
  // CAST uses the same guard; these inputs also succeed on the CPU.
  check_rejection("SELECT id,CAST(k AS BIGINT) FROM u WHERE id IN (1,2,6,7)");
  check_rejection("SELECT id,CAST(k AS UINTEGER) FROM u WHERE id IN (1,6,7,8)");
}

TEST_CASE_METHOD(UnsignedNarrowingFixture,
                 "direct UINT64 operations retain exact GPU execution",
                 "[integration][gpu_execution][unsigned_narrowing]")
{
  compare_gpu_vs_cpu("SELECT id,k,k % 3::UBIGINT,k > 9223372036854775807::UBIGINT FROM u");
  compare_gpu_vs_cpu("SELECT l.id,r.id FROM u l LEFT JOIN u r ON l.k=r.k");
  compare_gpu_vs_cpu("SELECT k % 3::UBIGINT,COUNT(*) FROM u GROUP BY k % 3::UBIGINT");
  compare_gpu_vs_cpu("SELECT id,id::UINTEGER::BIGINT FROM u");
}
