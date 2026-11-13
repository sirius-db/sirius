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

#include "transparent/parquet_footer_statistics.hpp"
#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <optional>
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
                 "parquet footer statistics - sirius_analyze_parquet reads them ahead of a query",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  auto analyzed =
    con->Query("SELECT files FROM sirius_analyze_parquet('" + (dir / "*.parquet").string() + "')");
  REQUIRE_FALSE(analyzed->HasError());
  CHECK(collect_rows(*analyzed)[0][0] == "2");

  auto none = con->Query("SELECT files FROM sirius_analyze_parquet('" +
                         (dir / "missing_*.parquet").string() + "')");
  CHECK(none->HasError());

  CHECK(cpu_plan("SELECT count(*) FROM v WHERE x > 1000", true).find("Empty Result") !=
        std::string::npos);
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - sirius_analyze_parquet answers every execution of a "
                 "prepared statement",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  auto prepared = con->Prepare("SELECT files FROM sirius_analyze_parquet('" +
                               (dir / "*.parquet").string() + "')");
  REQUIRE_FALSE(prepared->HasError());
  for (int execution = 0; execution < 2; ++execution) {
    auto result = prepared->Execute();
    REQUIRE_FALSE(result->HasError());
    auto const rows = collect_rows(*result);
    REQUIRE(rows.size() == 1);
    CHECK(rows[0][0] == "2");
  }
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - statistics read with other reader options are not "
                 "reused",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  // Each file holds 1.0, -1.0 and NaN, and its footer gives a maximum of 1.0 without a NaN count.
  // With can_have_nan the maximum does not bound the NaNs, so `d > 1` must keep them. DuckDB's own
  // writer omits the bounds of a column holding NaN, so the file comes from DuckDB's test data.
#ifdef SIRIUS_PROJECT_ROOT
  auto const data = fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data";
#else
  auto const data = fs::path(__FILE__).parent_path().parent_path() / "integration/data";
#endif
  auto const source = data / "parquet/arrow_nan.parquet";
  REQUIRE(fs::exists(source));
  fs::create_directories(dir / "nan");
  for (auto const* name : {"n0.parquet", "n1.parquet"}) {
    auto const copy = dir / "nan" / name;
    fs::copy_file(source, copy);
    // An old modification time lets the cache trust it, as it would for files on disk for a while.
    fs::last_write_time(copy, fs::file_time_type::clock::now() - std::chrono::hours(1));
  }
  auto const glob = (dir / "nan" / "*.parquet").string();

  auto warmed = con->Query("SELECT files FROM sirius_analyze_parquet('" + glob + "')");
  REQUIRE_FALSE(warmed->HasError());
  REQUIRE(collect_rows(*warmed)[0][0] == "2");

  auto const query =
    "SELECT count(*) FROM read_parquet('" + glob + "', can_have_nan = true) WHERE d > 1";
  INFO(cpu_plan(query, true));
  run_ok("SET gpu_execution = false;");
  auto result = con->Query(query);
  run_ok("SET gpu_execution = true;");
  REQUIRE_FALSE(result->HasError());
  CHECK(collect_rows(*result)[0][0] == "2");
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

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - a file rewritten in place within its modification "
                 "time's second is not described by stale statistics",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  // Plain, uncompressed files of equal row counts have equal sizes whatever the values.
  auto const write_plain = [&](fs::path const& path, int from, int to) {
    run_ok("COPY (SELECT range AS x FROM range(" + std::to_string(from) + ", " +
           std::to_string(to) + ")) TO '" + path.string() +
           "' (FORMAT parquet, COMPRESSION uncompressed);");
  };
  fs::create_directories(dir / "inplace");
  auto const target = dir / "inplace" / "a.parquet";
  write_plain(dir / "inplace" / "b.parquet", 0, 100);
  write_plain(target, 100, 200);
  write_plain(dir / "replacement.parquet", 1000, 1100);
  REQUIRE(fs::file_size(target) == fs::file_size(dir / "replacement.parquet"));

  auto const query = "SELECT count(*) FROM read_parquet('" +
                     (dir / "inplace" / "*.parquet").string() + "') WHERE x >= 1000";
  run_ok("SET gpu_execution = false;");
  auto before = con->Query(query);
  REQUIRE_FALSE(before->HasError());
  REQUIRE(collect_rows(*before)[0][0] == "0");

  // Same inode, size and modification time: what a clock with one-second resolution shows for a
  // rewrite within the same second.
  auto const modified = fs::last_write_time(target);
  fs::copy_file(dir / "replacement.parquet", target, fs::copy_options::overwrite_existing);
  fs::last_write_time(target, modified);
  auto after = con->Query(query);
  run_ok("SET gpu_execution = true;");
  REQUIRE_FALSE(after->HasError());
  CHECK(collect_rows(*after)[0][0] == "100");
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - statistics reused across queries follow a file that "
                 "changes",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  // Files not modified for a while are trusted by the cache, so the second plan reuses the merged
  // statistics of the first instead of reading the footers again.
  auto const age = [&](char const* name, int hours) {
    fs::last_write_time(dir / (std::string(name) + ".parquet"),
                        fs::file_time_type::clock::now() - std::chrono::hours(hours));
  };
  age("f0", 3);
  age("f1", 3);

  constexpr char const* query = "SELECT count(*) FROM v WHERE x >= 1000";
  REQUIRE(cpu_plan(query, true).find("Empty Result") != std::string::npos);
  CHECK(cpu_plan(query, true).find("Empty Result") != std::string::npos);

  // A rewritten file is described by its new footer, not by the merge made from the old one.
  write_file("f1", 1000, 1100);
  age("f1", 2);
  CHECK(cpu_plan(query, true).find("Empty Result") == std::string::npos);
  compare_gpu_vs_cpu(query);
  auto rewritten = con->Query(query);
  REQUIRE_FALSE(rewritten->HasError());
  CHECK(collect_rows(*rewritten)[0][0] == "100");
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - a glob that gains files keeps one merged entry",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  using sirius::transparent::detail::merged_statistics_cache_size;
  constexpr char const* query = "SELECT count(*) FROM v WHERE x >= 1000";
  cpu_plan(query, true);
  auto const entries = merged_statistics_cache_size();
  for (int file = 2; file < 5; ++file) {
    write_file("f" + std::to_string(file), 1000 * file, 1000 * file + 10);
    cpu_plan(query, true);
  }
  CHECK(merged_statistics_cache_size() == entries);
  auto grown = con->Query(query);
  REQUIRE_FALSE(grown->HasError());
  CHECK(collect_rows(*grown)[0][0] == "30");
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - merged entries for explicit file lists are bounded",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  using sirius::transparent::detail::merged_statistics_cache_capacity;
  using sirius::transparent::detail::merged_statistics_cache_size;
  using sirius::transparent::detail::row_count_cache_size;
  // Pairs of 17 files give 136 distinct lists, more than the cache keeps.
  constexpr int files = 17;
  for (int file = 2; file < files; ++file) {
    write_file("f" + std::to_string(file), 100 * file, 100 * file + 100);
  }
  auto const path = [&](int file) {
    return "'" + (dir / ("f" + std::to_string(file) + ".parquet")).string() + "'";
  };
  for (int first = 0; first < files; ++first) {
    for (int second = first + 1; second < files; ++second) {
      cpu_plan("SELECT count(*) FROM read_parquet([" + path(first) + ", " + path(second) +
                 "]) WHERE x >= 100000",
               true);
    }
  }
  CHECK(merged_statistics_cache_size() <= merged_statistics_cache_capacity);
  CHECK(row_count_cache_size() <= merged_statistics_cache_capacity);
}

namespace {

duckdb::LogicalGet const* first_get(duckdb::LogicalOperator const& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    return &op.Cast<duckdb::LogicalGet>();
  }
  for (auto const& child : op.children) {
    if (auto const* found = first_get(*child)) { return found; }
  }
  return nullptr;
}

}  // namespace

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - the footers' row count is the scan's key domain",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  // Two files of 100 rows. A filter on the scan does not change the unfiltered domain.
  auto const multi = con->ExtractPlan("SELECT count(*) FROM v WHERE x >= 150");
  auto const* get  = first_get(*multi);
  REQUIRE(get != nullptr);
  CHECK(sirius::transparent::parquet_scan_row_count(*get) == std::optional<std::size_t>{200});

  // A scan of one file gets no footer statistics, but its footer is read for the row count.
  auto const single      = con->ExtractPlan("SELECT x FROM read_parquet('" +
                                            (dir / "f0.parquet").string() + "') WHERE x >= 50");
  auto const* single_get = first_get(*single);
  REQUIRE(single_get != nullptr);
  CHECK(sirius::transparent::parquet_scan_row_count(*single_get) ==
        std::optional<std::size_t>{100});
}
