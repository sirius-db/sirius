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

/**
 * @file test_parquet_footer_statistics.cpp
 * @brief A scan of several Parquet files gives DuckDB's optimizer the statistics of every footer.
 *
 * Without union_by_name DuckDB has no column statistics for such a scan. With them it can prove a
 * filter matches nothing, drop an IS NOT NULL on a column without NULLs and estimate the rest.
 */

#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>
#include <unistd.h>

#include <filesystem>
#include <string>

namespace {

namespace fs = std::filesystem;

class ParquetFilesFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetFilesFixture()
  {
    static int counter = 0;
    dir                = fs::temp_directory_path() /
          ("sirius_footer_stats_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
    fs::create_directories(dir);
    // x has no NULLs and covers [0, 200); y is NULL on odd rows.
    write_file("f0", 0, 100);
    write_file("f1", 100, 200);
    run_ok("CREATE VIEW v AS SELECT * FROM read_parquet('" + (dir / "*.parquet").string() + "');");
  }

  ~ParquetFilesFixture()
  {
    std::error_code ignored;
    fs::remove_all(dir, ignored);
  }

  void write_file(std::string const& name, int from, int to)
  {
    run_ok("COPY (SELECT range AS x, CASE WHEN range % 2 = 0 THEN range END AS y FROM range(" +
           std::to_string(from) + ", " + std::to_string(to) + ")) TO '" +
           (dir / (name + ".parquet")).string() + "';");
  }

  std::string cpu_plan(std::string const& query, bool statistics)
  {
    run_ok("SET gpu_execution = false;");
    run_ok(std::string("SET parquet_footer_statistics = ") + (statistics ? "true;" : "false;"));
    run_ok("SET explain_output = 'optimized_only';");
    auto result = con->Query("EXPLAIN " + query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    std::string text;
    for (auto const& row : collect_rows(*result, /*sort=*/false)) {
      for (auto const& cell : row) {
        text += cell;
      }
    }
    run_ok("RESET explain_output;");
    run_ok("RESET parquet_footer_statistics;");
    run_ok("SET gpu_execution = true;");
    return text;
  }

  fs::path dir;
};

}  // namespace

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - the optimizer sees every file's footer",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  constexpr char const* out_of_range = "SELECT count(*) FROM v WHERE x > 1000";
  CHECK(cpu_plan(out_of_range, false).find("Empty Result") == std::string::npos);
  CHECK(cpu_plan(out_of_range, true).find("Empty Result") != std::string::npos);

  constexpr char const* not_null = "SELECT count(*) FROM v WHERE x IS NOT NULL AND y IS NOT NULL";
  auto const without             = cpu_plan(not_null, false);
  auto const with                = cpu_plan(not_null, true);
  INFO(with);
  CHECK(without.find("x IS NOT NULL") != std::string::npos);
  CHECK(with.find("x IS NOT NULL") == std::string::npos);
  CHECK(with.find("y IS NOT NULL") != std::string::npos);

  compare_gpu_vs_cpu(out_of_range);
  compare_gpu_vs_cpu(not_null);
  compare_gpu_vs_cpu("SELECT count(*), sum(x), min(x), max(x) FROM v WHERE x >= 150");
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - a rewritten or added file is not described by stale "
                 "statistics",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  constexpr char const* query = "SELECT count(*) FROM v WHERE x >= 1000";
  REQUIRE(cpu_plan(query, true).find("Empty Result") != std::string::npos);

  write_file("f1", 1000, 1100);
  CHECK(cpu_plan(query, true).find("Empty Result") == std::string::npos);
  compare_gpu_vs_cpu(query);
  auto rewritten = con->Query(query);
  REQUIRE_FALSE(rewritten->HasError());
  CHECK(collect_rows(*rewritten)[0][0] == "100");

  write_file("f2", 2000, 2050);
  auto added = con->Query(query);
  REQUIRE_FALSE(added->HasError());
  CHECK(collect_rows(*added)[0][0] == "150");
}
