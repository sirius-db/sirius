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
#include <utils/gpu_execution_fixture.hpp>

#include <filesystem>
#include <string>

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "integer to VARCHAR runs on native and Parquet scans without fallback",
                 "[integration][gpu_execution][integer_varchar_cast]")
{
  run_ok(
    "CREATE TABLE integer_strings(id INTEGER, category INTEGER, s8 TINYINT, s16 SMALLINT, "
    "s64 BIGINT, u8 UTINYINT, u16 USMALLINT, u32 UINTEGER, u64 UBIGINT)");
  run_ok(
    "INSERT INTO integer_strings VALUES "
    "(50,1,-128,-32768,-9223372036854775808,0,0,0,0),"
    "(51,9,127,32767,9223372036854775807,255,65535,4294967295,18446744073709551615),"
    "(52,-19,-1,-1,-1,1,1,1,1),(53,2,0,0,0,0,0,0,0),"
    "(54,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL),(50,NULL,1,1,1,9,9,9,9),"
    "(-2147483648,2147483647,NULL,NULL,NULL,NULL,NULL,NULL,NULL)");
  run_ok("CHECKPOINT");
  auto const parquet_path = temp_db_path + ".parquet";
  struct remove_file {
    std::string path;
    ~remove_file() { std::filesystem::remove(path); }
  } cleanup{parquet_path};
  run_ok("COPY integer_strings TO '" + parquet_path + "' (FORMAT PARQUET)");
  run_ok("SET enable_duckdb_fallback=false");

  for (std::string const source :
       {std::string{"integer_strings"}, "read_parquet('" + parquet_path + "')"}) {
    CAPTURE(source);
    compare_gpu_vs_cpu("SELECT count(*) FROM " + source +
                       " WHERE id=50 OR category::VARCHAR LIKE '%9%'");
    compare_gpu_vs_cpu("SELECT id, category::VARCHAR FROM " + source +
                       " WHERE id=50 OR category::VARCHAR LIKE '%9%'");
    compare_gpu_vs_cpu(
      "SELECT id::VARCHAR, category::VARCHAR, s8::VARCHAR, s16::VARCHAR, "
      "s64::VARCHAR, u8::VARCHAR, u16::VARCHAR, u32::VARCHAR, u64::VARCHAR FROM " +
      source);
    compare_gpu_vs_cpu(
      "SELECT TRY_CAST(category AS VARCHAR), TRY_CAST(s64 AS VARCHAR), "
      "TRY_CAST(u64 AS VARCHAR) FROM " +
      source);
  }
}
