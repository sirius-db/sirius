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

#include "sirius_context.hpp"
#include "transparent/parquet_footer_statistics.hpp"
#include "transparent/sirius_optimizer_extension.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/host_allocation_fault.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/tpch_queries.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/multi_file/multi_file_list.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/multi_file/table_function_multi_file.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/planner.hpp>
#include <duckdb/storage/external_file_cache/external_file_cache.hpp>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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

namespace {

duckdb::MultiFileBindData const& footer_scan_bind(duckdb::LogicalOperator const& plan)
{
  auto const* get = first_get(plan);
  REQUIRE(get != nullptr);
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get->bind_data.get());
  REQUIRE(bind != nullptr);
  return *bind;
}

duckdb::LogicalGet const* footer_parquet_get(duckdb::LogicalOperator const& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto const& get  = plan.Cast<duckdb::LogicalGet>();
    auto const& name = get.function.GetName().GetIdentifierName();
    if (name == "read_parquet" || name == "parquet_scan") { return &get; }
  }
  for (auto const& child : plan.children) {
    if (auto const* get = footer_parquet_get(*child)) { return get; }
  }
  return nullptr;
}

duckdb::LogicalOperator const* footer_comparison_join(duckdb::LogicalOperator const& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return &plan; }
  for (auto const& child : plan.children) {
    if (auto const* join = footer_comparison_join(*child)) { return join; }
  }
  return nullptr;
}

constexpr char const* footer_join_query =
  "SELECT count(*)::BIGINT, sum(p.x)::BIGINT FROM footer_probe p "
  "JOIN v b ON p.x=b.x WHERE b.y IS NOT NULL";

duckdb::Value footer_fixture_statistics()
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto const text = LogicalType::VARCHAR;
  auto x          = Value::MAP(text,
                      text,
                               {Value("null_count"), Value("min"), Value("max")},
                               {Value("0"), Value("0"), Value("199")});
  auto y = Value::MAP(text, text, {Value("min"), Value("max")}, {Value("0"), Value("198")});
  return Value::MAP(text,
                    LogicalType::MAP(text, text),
                    {Value("\"x\""), Value("\"y\"")},
                    {std::move(x), std::move(y)});
}

class footer_open_observer final : public duckdb::FileSystem {
 public:
  footer_open_observer(std::string prefix, std::string watched)
    : prefix_(std::move(prefix)),
      watched_(std::move(watched)),
      local_(duckdb::FileSystem::CreateLocal())
  {
  }

  bool CanHandleFile(std::string const& path) override { return path.starts_with(prefix_); }
  bool IsManuallySet() override { return true; }
  std::string GetName() const override { return "footer-open-observer-" + prefix_; }
  duckdb::unique_ptr<duckdb::FileHandle> OpenFile(
    std::string const& path,
    duckdb::FileOpenFlags flags,
    duckdb::optional_ptr<duckdb::FileOpener> opener) override
  {
    if (path == watched_) {
      ++watched_opens;
      if (before_watched_open) { before_watched_open(); }
    }
    return local_->OpenFile(path, flags, opener);
  }
  duckdb::vector<duckdb::OpenFileInfo> Glob(std::string const& path,
                                            duckdb::FileOpener* opener) override
  {
    return local_->Glob(path, opener);
  }
  duckdb::optional<duckdb::FileMetadata> GetStatsIfExists(
    duckdb::OpenFileInfo const& path, duckdb::optional_ptr<duckdb::FileOpener> opener) override
  {
    return local_->GetStatsIfExists(path, opener);
  }
  bool FileExists(std::string const& path, duckdb::optional_ptr<duckdb::FileOpener> opener) override
  {
    return local_->FileExists(path, opener);
  }
  bool DirectoryExists(std::string const& path,
                       duckdb::optional_ptr<duckdb::FileOpener> opener) override
  {
    return local_->DirectoryExists(path, opener);
  }
  std::atomic<std::size_t> watched_opens{0};
  std::function<void()> before_watched_open;

 private:
  std::string prefix_;
  std::string watched_;
  duckdb::unique_ptr<duckdb::FileSystem> local_;
};

struct scoped_footer_observer {
  duckdb::FileSystem& filesystem;
  std::string name;
  ~scoped_footer_observer() { filesystem.UnregisterSubSystem(name); }
};

fs::path footer_engine_config(fs::path const& directory, std::string const& options)
{
  auto const path = directory / "footer-policy.yaml";
  std::ofstream out(path);
  out << "sirius:\n"
         "  space:\n"
         "    gpu:\n"
         "      - device_id: 0\n"
         "        per_stream_reservation: false\n"
         "        reservation_limit_fraction: 0.4\n"
         "        downgrade_trigger_fraction: 0.8\n"
         "        downgrade_stop_fraction: 0.6\n"
         "        memory_capacity: 256 MiB\n"
         "    host:\n"
         "      - numa_id: -1\n"
         "        reservation_limit_fraction: 0.9\n"
         "        downgrade_trigger_fraction: 0.8\n"
         "        downgrade_stop_fraction: 0.6\n"
         "        memory_capacity: 512 MiB\n"
         "        block_size: 1 MiB\n"
         "  executor:\n"
         "    scan_manager:\n"
         "      cache:\n"
         "        mode: none\n"
         "  operator_params:\n"
      << options;
  out.close();
  REQUIRE(out);
  return path;
}

}  // namespace

TEST_CASE_METHOD(ParquetFilesFixture,
                 "Footer admission preflight exposes versions, facts, growth and connections",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_probe]")
{
  auto& filesystem = duckdb::FileSystem::GetFileSystem(*con->context);
  for (auto const* name : {"f0.parquet", "f1.parquet"}) {
    auto const path      = dir / name;
    auto const cacheable = [&](int hours) {
      fs::last_write_time(path, fs::file_time_type::clock::now() + std::chrono::hours(hours));
      auto const metadata = filesystem.GetStatsIfExists(duckdb::OpenFileInfo(path.string()));
      REQUIRE(metadata);
      REQUIRE_FALSE(metadata->version_tag.empty());
      return duckdb::ExternalFileCache::IsValid(
        true, "", metadata->last_modification_time, "", metadata->last_modification_time);
    };
    CHECK_FALSE(cacheable(24));
    CHECK(cacheable(-1));
  }
  auto first = con->ExtractPlan("SELECT x FROM v");
  REQUIRE(first);
  CHECK(footer_scan_bind(*first).file_options.column_statistics == footer_fixture_statistics());
  CHECK(sirius::transparent::parquet_scan_row_count(*first_get(*first)) ==
        std::optional<std::size_t>{200});
  write_file("f2", 200, 250);
  auto grown = con->ExtractPlan("SELECT x FROM v");
  REQUIRE(grown);
  CHECK(footer_scan_bind(*grown).file_list->GetAllFiles().size() == 3);
  CHECK(sirius::transparent::parquet_scan_row_count(*first_get(*grown)) ==
        std::optional<std::size_t>{250});

  duckdb::Connection second(*con->context->db);
  run_ok("SET parquet_footer_statistics=false");
  auto second_setting = second.Query("SELECT current_setting('parquet_footer_statistics')");
  REQUIRE_FALSE(second_setting->HasError());
  CHECK(second_setting->Collection().GetValue(0, 0).GetValue<bool>());
  auto const query = "SELECT x FROM read_parquet('" + (dir / "*.parquet").string() + "')";
  auto disabled    = con->ExtractPlan(query);
  auto enabled     = second.ExtractPlan(query);
  CHECK(footer_scan_bind(*disabled).file_options.column_statistics.IsNull());
  CHECK_FALSE(footer_scan_bind(*enabled).file_options.column_statistics.IsNull());
  run_ok("RESET parquet_footer_statistics");
  run_ok("CALL pin_table('" + (dir / "*.parquet").string() + "', tier='host', name='v')");
  run_ok("CALL unpin_table('v')");
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "Footer admission preflight puts an unreadable later file inside collection",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_probe]")
{
  auto covered = con->ExtractPlan("SELECT x FROM v");
  REQUIRE(covered);
  REQUIRE_FALSE(footer_scan_bind(*covered).file_options.column_statistics.IsNull());
  auto const bad = (dir / "z_bad.parquet").string();
  {
    std::ofstream out(bad, std::ios::binary);
    out << "not a parquet file";
    REQUIRE(out.good());
  }
  auto observer    = duckdb::make_uniq<footer_open_observer>(dir.string() + "/", bad);
  auto& observed   = *observer;
  auto& filesystem = con->context->db->GetFileSystem();
  scoped_footer_observer registration{filesystem, observed.GetName()};
  filesystem.RegisterSubSystem(std::move(observer));

  run_ok("SET parquet_footer_statistics=false");
  auto bind_only = con->ExtractPlan("SELECT x FROM v");
  REQUIRE(bind_only);
  auto const files = footer_scan_bind(*bind_only).file_list->GetAllFiles();
  REQUIRE(files.size() == 3);
  REQUIRE(files.front().path != bad);
  CHECK(observed.watched_opens.load() == 0);
  run_ok("SET parquet_footer_statistics=true");
  auto collected = con->ExtractPlan("SELECT x FROM v");
  REQUIRE(collected);
  CHECK(observed.watched_opens.load() > 0);
  CHECK(footer_scan_bind(*collected).file_options.column_statistics.IsNull());
}

TEST_CASE("Footer admission preflight loads engine policy before opening two connections",
          "[integration][isolated_context][parquet_footer_statistics][footer_admission_probe]")
{
  sirius::test::scratch_dir directory("footer_policy_probe");
  auto const config = footer_engine_config(directory.path(), "    enable_dynamic_filter: false\n");
  auto db           = sirius::test::open_sirius_db(nullptr, config);
  duckdb::Connection first(*db), second(*db);
  auto first_context = first.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  auto second_context =
    second.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(first_context);
  REQUIRE(second_context);
  CHECK(first_context == second_context);
  CHECK_FALSE(first_context->get_config().get_operator_params().enable_dynamic_filter);
  CHECK_FALSE(second_context->get_config().get_operator_params().enable_dynamic_filter);
  auto changed = first.Query("SET parquet_footer_statistics=false");
  REQUIRE_FALSE(changed->HasError());
  auto unchanged = second.Query("SELECT current_setting('parquet_footer_statistics')");
  REQUIRE_FALSE(unchanged->HasError());
  CHECK(unchanged->Collection().GetValue(0, 0).GetValue<bool>());
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "Footer admission preflight executes a join with a Parquet build side",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_probe]")
{
  run_ok("CREATE TABLE footer_probe AS SELECT range AS x FROM range(2000)");
  run_ok("CHECKPOINT");
  auto plan = con->ExtractPlan(footer_join_query);
  REQUIRE(plan);
  auto const* join = footer_comparison_join(*plan);
  REQUIRE(join != nullptr);
  REQUIRE(join->children.size() == 2);
  auto const* build = footer_parquet_get(*join->children[1]);
  REQUIRE(build != nullptr);
  CHECK(sirius::transparent::parquet_scan_row_count(*build) == std::optional<std::size_t>{200});
  compare_gpu_vs_cpu(footer_join_query);
  auto result = con->Query(footer_join_query);
  REQUIRE_FALSE(result->HasError());
  CHECK(collect_rows(*result) == std::vector<std::vector<std::string>>{{"100", "9900"}});
}

namespace {

namespace footer_detail = sirius::transparent::detail;

class footer_cache_scope {
 public:
  footer_cache_scope()
    : previous_(
        footer_detail::set_cache_capacities_for_testing(footer_detail::default_cache_capacities))
  {
    footer_detail::clear_caches_for_testing();
  }
  ~footer_cache_scope()
  {
    footer_detail::clear_caches_for_testing();
    (void)footer_detail::set_cache_capacities_for_testing(previous_);
  }

 private:
  footer_detail::cache_capacities previous_;
};

void check_footer_retention(footer_detail::cache_capacities capacities)
{
  CHECK(footer_detail::statistics_cache_size() <= capacities.statistics);
  CHECK(footer_detail::merged_statistics_cache_size() <= capacities.merges);
  CHECK(footer_detail::merged_reference_count() <= capacities.merge_references);
  CHECK(footer_detail::live_statistics_records() <= capacities.statistics);
}

void check_footer_scan(duckdb::LogicalOperator const& plan,
                       std::size_t files,
                       std::optional<std::size_t> rows)
{
  auto const* get = footer_parquet_get(plan);
  REQUIRE(get != nullptr);
  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get->bind_data.get());
  REQUIRE(bind != nullptr);
  CHECK(bind->file_list->GetAllFiles().size() == files);
  CHECK(sirius::transparent::parquet_scan_row_count(*get) == rows);
  CHECK(bind->file_options.column_statistics.IsNull() == (files == 1 || !rows.has_value()));
}

class FooterAdmissionFixture : public ParquetFilesFixture {
 public:
  FooterAdmissionFixture()
  {
    run_ok("SET parquet_footer_statistics=true");
    run_ok("SET parquet_footer_statistics_max_remote_reads=-1");
    local_limit(-1);
    age_files();
  }

  void local_limit(std::int64_t limit)
  {
    run_ok("SET parquet_footer_statistics_max_local_reads=" + std::to_string(limit));
  }
  void age_files(int hours = -1)
  {
    for (auto const& entry : fs::directory_iterator(dir)) {
      if (entry.path().extension() == ".parquet") {
        fs::last_write_time(entry.path(),
                            fs::file_time_type::clock::now() + std::chrono::hours(hours));
      }
    }
  }
  void add_files(int count)
  {
    for (int index = 2; index < count; ++index) {
      write_file("f" + std::to_string(index), index * 100, (index + 1) * 100);
      age_file(index);
    }
  }
  std::string file(int index) const
  {
    return (dir / ("f" + std::to_string(index) + ".parquet")).string();
  }
  void age_file(int index)
  {
    fs::last_write_time(file(index), fs::file_time_type::clock::now() - std::chrono::hours(1));
  }
  std::string scan(std::vector<int> const& indices, bool can_have_nan = false) const
  {
    std::string paths;
    for (auto index : indices) {
      if (!paths.empty()) { paths += ", "; }
      paths += "'" + file(index) + "'";
    }
    return "SELECT x FROM read_parquet([" + paths +
           "], can_have_nan=" + (can_have_nan ? "true)" : "false)");
  }
  void warm(std::vector<int> const& indices)
  {
    std::vector<std::string> paths;
    for (auto index : indices) {
      paths.push_back(file(index));
    }
    con->BeginTransaction();
    auto const count = sirius::transparent::read_parquet_footer_statistics(*con->context, paths);
    con->Commit();
    REQUIRE(count == paths.size());
  }
  auto plan(std::string const& query = "SELECT x FROM v")
  {
    auto result = con->ExtractPlan(query);
    REQUIRE(result);
    return result;
  }
  void require_contiguous_result(int files)
  {
    auto const rows = files * 100;
    auto const sql  = "SELECT count(*)::BIGINT, sum(x)::BIGINT, min(x), max(x) FROM v";
    compare_gpu_vs_cpu(sql);
    auto result = con->Query(sql);
    REQUIRE_FALSE(result->HasError());
    std::vector<std::vector<std::string>> expected{
      {std::to_string(rows), std::to_string(rows * (rows - 1) / 2), "0", std::to_string(rows - 1)}};
    CHECK(collect_rows(*result) == expected);
  }
  footer_cache_scope caches;
};

auto footer_query_ok(duckdb::Connection& connection, std::string const& sql)
{
  auto result = connection.Query(sql);
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  return result;
}

}  // namespace

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission counts local misses at both sides of the boundary",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const files = GENERATE(2, 3, 4);
  CAPTURE(files);
  add_files(files);
  local_limit(3);
  auto const before   = footer_detail::footer_read_count();
  auto result         = plan();
  auto const admitted = files <= 3;
  CHECK(footer_detail::footer_read_count() - before == (admitted ? files : 0));
  check_footer_scan(
    *result, files, admitted ? std::optional<std::size_t>{files * 100} : std::nullopt);
  if (files == 2) {
    CHECK(footer_scan_bind(*result).file_options.column_statistics == footer_fixture_statistics());
  }
  require_contiguous_result(files);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission zero skips cold scans and minus one admits them",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const limit = GENERATE(0, -1);
  CAPTURE(limit);
  add_files(3);
  local_limit(limit);
  auto const before = footer_detail::footer_read_count();
  auto result       = plan();
  CHECK(footer_detail::footer_read_count() - before == (limit == 0 ? 0 : 3));
  check_footer_scan(*result, 3, limit == 0 ? std::nullopt : std::optional<std::size_t>{300});
  require_contiguous_result(3);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission accepts all cached inputs even above the file-count limit",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const limit = GENERATE(0, 1);
  CAPTURE(limit);
  add_files(4);
  warm({0, 1, 2, 3});
  local_limit(limit);
  auto const before = footer_detail::footer_read_count();
  auto result       = plan();
  CHECK(footer_detail::footer_read_count() == before);
  check_footer_scan(*result, 4, 400);
  require_contiguous_result(4);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission counts only misses but never uses a cached subset",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const limit = GENERATE(1, 2);
  CAPTURE(limit);
  add_files(4);
  warm({0, 1});
  local_limit(limit);
  auto const before = footer_detail::footer_read_count();
  auto result       = plan();
  CHECK(footer_detail::footer_read_count() - before == (limit == 2 ? 2 : 0));
  check_footer_scan(*result, 4, limit == 2 ? std::optional<std::size_t>{400} : std::nullopt);
  require_contiguous_result(4);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission validates versions and reader options before counting hits",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const query = scan({0, 1}, false);
  auto initial     = plan(query);
  REQUIRE(footer_scan_bind(*initial).file_options.column_statistics == footer_fixture_statistics());
  auto before   = footer_detail::footer_read_count();
  auto repeated = plan(query);
  CHECK(footer_detail::footer_read_count() == before);
  SECTION("replacement is a miss")
  {
    write_file("replacement", 1000, 1100);
    fs::rename(dir / "replacement.parquet", file(1));
    age_file(1);
    local_limit(1);
    before       = footer_detail::footer_read_count();
    auto changed = plan(query);
    CHECK(footer_detail::footer_read_count() - before == 1);
    check_footer_scan(*changed, 2, 200);
    CHECK(footer_scan_bind(*changed).file_options.column_statistics != footer_fixture_statistics());
    auto rows = con->Query("SELECT count(*) FROM v WHERE x >= 1000");
    REQUIRE_FALSE(rows->HasError());
    CHECK(collect_rows(*rows) == std::vector<std::vector<std::string>>{{"100"}});
  }
  SECTION("reader options have separate entries")
  {
    local_limit(2);
    auto other = plan(scan({0, 1}, true));
    CHECK(footer_detail::footer_read_count() - before == 2);
    CHECK(footer_scan_bind(*other).file_options.column_statistics == footer_fixture_statistics());
    before        = footer_detail::footer_read_count();
    auto original = plan(query);
    CHECK(footer_detail::footer_read_count() == before);
    check_footer_scan(*original, 2, 200);
  }
  SECTION("an untrustworthy local version is a miss on every planning")
  {
    age_files(24);
    local_limit(2);
    for (int iteration = 0; iteration < 2; ++iteration) {
      before           = footer_detail::footer_read_count();
      auto uncacheable = plan(query);
      CHECK(footer_detail::footer_read_count() - before == 2);
      check_footer_scan(*uncacheable, 2, 200);
    }
    local_limit(1);
    before       = footer_detail::footer_read_count();
    auto skipped = plan(query);
    CHECK(footer_detail::footer_read_count() == before);
    check_footer_scan(*skipped, 2, std::nullopt);
  }
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission skips facts whose only counterexamples are uncached",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  warm({0, 1});
  run_ok(
    "COPY (SELECT NULL::BIGINT AS x, NULL::BIGINT AS y "
    "UNION ALL SELECT 2000::BIGINT, 2000::BIGINT) TO '" +
    (dir / "f2.parquet").string() + "'");
  age_file(2);
  local_limit(0);
  auto const before = footer_detail::footer_read_count();
  CHECK(cpu_plan("SELECT count(*) FROM v WHERE x IS NOT NULL", true).find("x IS NOT NULL") !=
        std::string::npos);
  CHECK(cpu_plan("SELECT count(*) FROM v WHERE x > 1000", true).find("Empty Result") ==
        std::string::npos);
  CHECK(footer_detail::footer_read_count() == before);
  auto nulls   = con->Query("SELECT count(*) FROM v WHERE x IS NOT NULL");
  auto extreme = con->Query("SELECT count(*) FROM v WHERE x > 1000");
  REQUIRE_FALSE(nulls->HasError());
  REQUIRE_FALSE(extreme->HasError());
  CHECK(collect_rows(*nulls) == std::vector<std::vector<std::string>>{{"201"}});
  CHECK(collect_rows(*extreme) == std::vector<std::vector<std::string>>{{"1"}});
  compare_gpu_vs_cpu("SELECT x, y FROM v WHERE x IS NULL OR x > 1000");
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission preserves conservative caller-supplied column facts",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const limit = GENERATE(0, 2);
  local_limit(limit);
  run_ok("SET parquet_footer_statistics=false");
  auto supplied = plan();
  auto* bind    = dynamic_cast<duckdb::MultiFileBindData*>(first_get(*supplied)->bind_data.get());
  REQUIRE(bind != nullptr);
  using duckdb::LogicalType;
  using duckdb::Value;
  auto const text = LogicalType::VARCHAR;
  auto wide =
    Value::MAP(text, text, {Value("min"), Value("max")}, {Value("-1000"), Value("10000")});
  auto const caller =
    Value::MAP(text, LogicalType::MAP(text, text), {Value("\"x\""), Value("\"y\"")}, {wide, wide});
  REQUIRE(caller != footer_fixture_statistics());
  bind->file_options.column_statistics = caller;
  run_ok("SET parquet_footer_statistics=true");
  auto const before = footer_detail::footer_read_count();
  con->BeginTransaction();
  (void)sirius::transparent::add_parquet_footer_statistics(*con->context, *supplied);
  con->Commit();
  CHECK(bind->file_options.column_statistics == caller);
  if (limit == 0) { CHECK(footer_detail::footer_read_count() == before); }
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission: a later plan does not inherit an earlier row total after growth, disable or "
  "failure",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const transition = std::string(GENERATE("growth", "disabled", "failure"));
  CAPTURE(transition);
  auto covered = plan();
  check_footer_scan(*covered, 2, 200);
  if (transition == "growth") {
    add_files(4);
    local_limit(1);  // Two newly uncached inputs exceed this cap.
  } else if (transition == "disabled") {
    run_ok("SET parquet_footer_statistics=false");
  } else {
    std::ofstream bad(dir / "z_bad.parquet", std::ios::binary);
    bad << "not a parquet footer";
    REQUIRE(bad.good());
  }
  auto const before = footer_detail::footer_read_count();
  auto subsequent   = plan();
  check_footer_scan(*subsequent,
                    transition == "growth"    ? 4
                    : transition == "failure" ? 3
                                              : 2,
                    std::nullopt);
  if (transition != "failure") { CHECK(footer_detail::footer_read_count() == before); }
  if (transition == "growth") { require_contiguous_result(4); }
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission applies cache-only admission to a single-file row total",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  local_limit(0);
  auto const query = "SELECT x FROM read_parquet('" + file(0) + "')";
  auto before      = footer_detail::footer_read_count();
  auto cold        = plan(query);
  CHECK(footer_detail::footer_read_count() == before);
  check_footer_scan(*cold, 1, std::nullopt);
  warm({0});
  before   = footer_detail::footer_read_count();
  auto hit = plan(query);
  CHECK(footer_detail::footer_read_count() == before);
  check_footer_scan(*hit, 1, 100);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission can forget every distinct Parquet scan in a plan",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto joined = plan("SELECT a.x, b.x FROM read_parquet('" + file(0) +
                     "') a CROSS JOIN read_parquet('" + file(1) + "') b WHERE a.y IS NOT NULL");
  std::vector<duckdb::LogicalGet const*> scans;
  auto visit = [&](auto const& self, duckdb::LogicalOperator const& op) -> void {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
      auto const& get  = op.Cast<duckdb::LogicalGet>();
      auto const& name = get.function.GetName().GetIdentifierName();
      if (name == "read_parquet" || name == "parquet_scan") { scans.push_back(&get); }
    }
    for (auto const& child : op.children) {
      self(self, *child);
    }
  };
  visit(visit, *joined);
  REQUIRE(scans.size() == 2);
  for (auto const* get : scans) {
    REQUIRE(sirius::transparent::parquet_scan_row_count(*get) == std::optional<std::size_t>{100});
  }
  sirius::transparent::forget_parquet_scan_row_counts(*joined);
  for (auto const* get : scans) {
    CHECK_FALSE(sirius::transparent::parquet_scan_row_count(*get));
  }
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission leaves explicit analysis and pin warming uncapped",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const mode = std::string(GENERATE("analyze", "analyze_pass_off", "pin"));
  CAPTURE(mode);
  add_files(3);
  local_limit(0);
  if (mode == "analyze_pass_off") { run_ok("SET parquet_footer_statistics=false"); }
  auto const before = footer_detail::footer_read_count();
  if (mode == "pin") {
    run_ok("CALL pin_table('" + (dir / "*.parquet").string() + "', tier='host', name='v')");
    run_ok("CALL unpin_table('v')");
  } else {
    auto analyzed = con->Query("SELECT files FROM sirius_analyze_parquet('" +
                               (dir / "*.parquet").string() + "')");
    REQUIRE_FALSE(analyzed->HasError());
    CHECK(collect_rows(*analyzed) == std::vector<std::vector<std::string>>{{"3"}});
  }
  CHECK(footer_detail::footer_read_count() - before == 3);
  run_ok("SET parquet_footer_statistics=true");
  auto const warmed = footer_detail::footer_read_count();
  auto result       = plan();
  CHECK(footer_detail::footer_read_count() == warmed);
  check_footer_scan(*result, 3, 300);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission bounds all cache owners across changing scan sets",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  auto const cacheable = GENERATE(false, true);
  CAPTURE(cacheable);
  add_files(7);
  auto const capacities = footer_detail::cache_capacities{3, 2, 3};
  (void)footer_detail::set_cache_capacities_for_testing(capacities);
  if (!cacheable) { age_files(24); }
  auto const sets = std::vector<std::vector<int>>{{0, 1, 2}, {1, 2, 3}, {4, 5, 6}, {0, 2, 6}};
  for (auto const& files : sets) {
    for (bool nan : {false, true}) {
      CAPTURE(files, nan);
      auto result = plan(scan(files, nan));
      check_footer_scan(*result, files.size(), files.size() * 100);
      check_footer_retention(capacities);
    }
  }
  write_file("replacement", 1000, 1100);
  fs::rename(dir / "replacement.parquet", file(0));
  age_files(cacheable ? -1 : 24);
  auto replaced = plan(scan({0, 1, 2}));
  check_footer_scan(*replaced, 3, 300);
  check_footer_retention(capacities);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer admission uses oversized merges now and recomputes evicted facts",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]"
                 "[footer_row_count]")
{
  add_files(3);
  auto first          = plan();
  auto const expected = footer_scan_bind(*first).file_options.column_statistics;
  footer_detail::clear_caches_for_testing();
  auto const capacities = footer_detail::cache_capacities{2, 1, 2};
  (void)footer_detail::set_cache_capacities_for_testing(capacities);
  auto before    = footer_detail::footer_read_count();
  auto oversized = plan();
  CHECK(footer_detail::footer_read_count() - before == 3);
  CHECK(footer_scan_bind(*oversized).file_options.column_statistics == expected);
  check_footer_scan(*oversized, 3, 300);
  check_footer_retention(capacities);
  before     = footer_detail::footer_read_count();
  auto again = plan();
  CHECK(footer_detail::footer_read_count() > before);
  CHECK(footer_scan_bind(*again).file_options.column_statistics == expected);
  check_footer_scan(*again, 3, 300);
  check_footer_scan(*oversized, 3, 300);
  check_footer_retention(capacities);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer admission keeps an active reader valid while its cached hits are evicted",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]"
                 "[footer_row_count]")
{
  warm({0, 1});
  write_file("f2", 200, 300);
  age_file(2);
  auto observer    = duckdb::make_uniq<footer_open_observer>(dir.string() + "/", file(2));
  auto& observed   = *observer;
  auto& filesystem = con->context->db->GetFileSystem();
  scoped_footer_observer registration{filesystem, observed.GetName()};
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto released       = release.get_future().share();
  std::atomic<bool> notified{false};
  observed.before_watched_open = [&] {
    if (!notified.exchange(true)) {
      entered.set_value();
      released.wait();
    }
  };
  filesystem.RegisterSubSystem(std::move(observer));
  auto pending =
    std::async(std::launch::async, [&] { return con->ExtractPlan("SELECT x FROM v"); });
  struct release_guard {
    std::promise<void>& signal;
    bool done{false};
    void unblock()
    {
      if (!done) {
        signal.set_value();
        done = true;
      }
    }
    ~release_guard() { unblock(); }
  } unblock{release};  // Must unwind before the future waits for its worker.
  REQUIRE(entered_future.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  auto const capacities = footer_detail::cache_capacities{1, 1, 1};
  (void)footer_detail::set_cache_capacities_for_testing(capacities);
  unblock.unblock();
  REQUIRE(pending.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  auto result = pending.get();
  REQUIRE(result);
  check_footer_scan(*result, 3, 300);
  check_footer_retention(capacities);
  auto again = plan();
  CHECK(footer_scan_bind(*again).file_options.column_statistics ==
        footer_scan_bind(*result).file_options.column_statistics);
  check_footer_scan(*result, 3, 300);
  check_footer_scan(*again, 3, 300);
  check_footer_retention(capacities);
}

TEST_CASE("Footer admission SQL limits inherit engine values and isolate connections",
          "[integration][isolated_context][parquet_footer_statistics][footer_admission_feature]")
{
  footer_cache_scope caches;
  sirius::test::scratch_dir directory("footer_policy_settings");
  auto const config = footer_engine_config(directory.path(),
                                           "    parquet_footer_statistics_max_remote_reads: 7\n"
                                           "    parquet_footer_statistics_max_local_reads: 1\n");
  auto db           = sirius::test::open_sirius_db(nullptr, config);
  duckdb::Connection first(*db), second(*db);
  for (int i = 0; i < 2; ++i) {
    auto const path = directory.file("f" + std::to_string(i) + ".parquet");
    footer_query_ok(first, "COPY (SELECT range AS x FROM range(100)) TO '" + path + "'");
    fs::last_write_time(path, fs::file_time_type::clock::now() - std::chrono::hours(1));
  }
  auto const query =
    "SELECT x FROM read_parquet('" + (directory.path() / "*.parquet").string() + "')";
  for (auto const* setting : {"parquet_footer_statistics_max_remote_reads",
                              "parquet_footer_statistics_max_local_reads"}) {
    for (auto* connection : {&first, &second}) {
      auto value =
        footer_query_ok(*connection, "SELECT current_setting('" + std::string(setting) + "')");
      CHECK(value->Collection().GetValue(0, 0).IsNull());
    }
    footer_query_ok(first, "SET " + std::string(setting) + "=2");
    for (auto const* invalid : {"-2", "-9223372036854775808"}) {
      auto rejected = first.Query("SET " + std::string(setting) + "=" + invalid);
      REQUIRE(rejected);
      CHECK(rejected->HasError());
    }
    auto value = footer_query_ok(first, "SELECT current_setting('" + std::string(setting) + "')");
    CHECK(value->Collection().GetValue(0, 0).GetValue<int64_t>() == 2);
    auto other = footer_query_ok(second, "SELECT current_setting('" + std::string(setting) + "')");
    CHECK(other->Collection().GetValue(0, 0).IsNull());
    footer_query_ok(first, "RESET " + std::string(setting));
  }
  auto expect = [&](duckdb::Connection& connection, bool admitted) {
    footer_detail::clear_caches_for_testing();
    auto const before = footer_detail::footer_read_count();
    auto result       = connection.ExtractPlan(query);
    REQUIRE(result);
    check_footer_scan(*result, 2, admitted ? std::optional<std::size_t>{200} : std::nullopt);
    CHECK(footer_detail::footer_read_count() - before == (admitted ? 2 : 0));
  };
  expect(first, false);
  footer_query_ok(first, "SET parquet_footer_statistics_max_local_reads=2");
  expect(first, true);
  expect(second, false);
  footer_query_ok(first, "RESET parquet_footer_statistics_max_local_reads");
  auto reset =
    footer_query_ok(first, "SELECT current_setting('parquet_footer_statistics_max_local_reads')");
  CHECK(reset->Collection().GetValue(0, 0).IsNull());
  expect(first, false);
  footer_query_ok(first, "SET parquet_footer_statistics_max_local_reads=2");
  footer_query_ok(first, "SET parquet_footer_statistics_max_local_reads=NULL");
  expect(first, false);
}

TEST_CASE("Effective footer limits follow the engine until a connection overrides them",
          "[integration][isolated_context][parquet_footer_statistics][footer_defaults]")
{
  auto const custom = GENERATE(false, true);
  CAPTURE(custom);
  sirius::test::scratch_dir directory("effective_footer_limits");
  auto const config =
    footer_engine_config(directory.path(),
                         custom ? "    parquet_footer_statistics_max_remote_reads: 7\n"
                                  "    parquet_footer_statistics_max_local_reads: 1\n"
                                : "    enable_dynamic_filter: false\n");
  auto db = sirius::test::open_sirius_db(nullptr, config);
  duckdb::Connection first(*db), second(*db);
  auto const remote = custom ? 7 : 1024;
  auto check        = [&](duckdb::Connection& connection, int expected_remote) {
    REQUIRE(connection.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"));
    auto const limits = sirius::transparent::footer_read_limits_of(*connection.context);
    CHECK(limits.remote == expected_remote);
    CHECK(limits.local == (custom ? 1 : 16384));
  };
  check(first, remote);
  footer_query_ok(first, "SET parquet_footer_statistics_max_remote_reads=-1");
  check(first, -1);
  check(second, remote);
  footer_query_ok(first, "RESET parquet_footer_statistics_max_remote_reads");
  check(first, remote);
  footer_query_ok(first, "SET parquet_footer_statistics_max_remote_reads=0");
  check(first, 0);
  footer_query_ok(first, "SET parquet_footer_statistics_max_remote_reads=NULL");
  check(first, remote);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission preserves GPU join results with a skipped Parquet build scan",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  run_ok("CREATE TABLE footer_probe AS SELECT range AS x FROM range(2000)");
  run_ok("CHECKPOINT");
  for (auto const limit : {0, 2}) {
    CAPTURE(limit);
    local_limit(limit);
    footer_detail::clear_caches_for_testing();
    auto result      = plan(footer_join_query);
    auto const* join = footer_comparison_join(*result);
    REQUIRE(join != nullptr);
    REQUIRE(join->children.size() == 2);
    auto const* build = footer_parquet_get(*join->children[1]);
    REQUIRE(build != nullptr);
    CHECK(sirius::transparent::parquet_scan_row_count(*build) ==
          (limit == 0 ? std::nullopt : std::optional<std::size_t>{200}));
    compare_gpu_vs_cpu(footer_join_query);
    auto rows = con->Query(footer_join_query);
    REQUIRE_FALSE(rows->HasError());
    CHECK(collect_rows(*rows) == std::vector<std::vector<std::string>>{{"100", "9900"}});
  }
}

namespace {

auto footer_fix_rows(duckdb::LogicalOperator const& plan)
{
  auto const* get = footer_parquet_get(plan);
  REQUIRE(get != nullptr);
  return sirius::transparent::parquet_scan_row_count(*get);
}

duckdb::MultiFileBindData& footer_fix_bind(duckdb::LogicalOperator& plan)
{
  auto const* get = footer_parquet_get(plan);
  REQUIRE(get != nullptr);
  auto* bind = dynamic_cast<duckdb::MultiFileBindData*>(get->bind_data.get());
  REQUIRE(bind != nullptr);
  return *bind;
}

auto footer_fix_copy(duckdb::Connection& connection, duckdb::LogicalOperator const& plan)
{
  connection.BeginTransaction();
  auto copy = sirius::transparent::copy_logical_plan(plan, *connection.context);
  connection.Commit();
  REQUIRE(copy);
  CHECK(footer_scan_bind(*copy).file_list.get() != footer_scan_bind(plan).file_list.get());
  return copy;
}

std::size_t footer_fix_collect(duckdb::Connection& connection, duckdb::LogicalOperator& plan)
{
  connection.BeginTransaction();
  auto const added = sirius::transparent::add_parquet_footer_statistics(*connection.context, plan);
  connection.Commit();
  return added;
}

class footer_listing_observer final : public duckdb::FileSystem {
 public:
  explicit footer_listing_observer(unsigned count = 4) : count_(count) {}
  std::string GetName() const override { return "footer-listing-observer"; }
  bool IsManuallySet() override { return true; }
  bool CanHandleFile(std::string const& path) override
  {
    return path.starts_with("s3a://footer-guard/");
  }
  duckdb::vector<duckdb::OpenFileInfo> Glob(std::string const&, duckdb::FileOpener*) override
  {
    ++lists;
    duckdb::vector<duckdb::OpenFileInfo> files;
    for (unsigned i = 0; i < count_; ++i) {
      auto& f         = files.emplace_back("s3a://footer-guard/" + std::to_string(i) + ".parquet");
      f.extended_info = duckdb::make_shared_ptr<duckdb::ExtendedOpenFileInfo>();
      f.extended_info->options["file_size"] = duckdb::Value::BIGINT(100);
      f.extended_info->options["etag"]      = duckdb::Value("\"listed\"");
    }
    return files;
  }
  duckdb::unique_ptr<duckdb::FileHandle> OpenFile(std::string const&,
                                                  duckdb::FileOpenFlags,
                                                  duckdb::optional_ptr<duckdb::FileOpener>) override
  {
    ++requests;
    ++opens;
    throw duckdb::IOException("unexpected remote footer open");
  }
  duckdb::optional<duckdb::FileMetadata> GetStatsIfExists(
    duckdb::OpenFileInfo const&, duckdb::optional_ptr<duckdb::FileOpener>) override
  {
    ++requests;
    throw duckdb::IOException("unexpected remote footer stat");
  }
  unsigned lists = 0;
  std::atomic<unsigned> requests{0};
  std::atomic<unsigned> opens{0};

 private:
  unsigned count_;
};

class footer_reused_list final : public duckdb::SimpleMultiFileList {
 public:
  explicit footer_reused_list(duckdb::vector<duckdb::OpenFileInfo> files)
    : SimpleMultiFileList(std::move(files))
  {
  }
  static void* operator new(std::size_t);
  static void operator delete(void*) noexcept { occupied = false; }

 private:
  static inline bool occupied = false;
};

void* footer_reused_list::operator new(std::size_t)
{
  alignas(footer_reused_list) static std::byte slot[sizeof(footer_reused_list)];
  if (occupied) { throw std::bad_alloc{}; }
  occupied = true;
  return slot;
}

class footer_throwing_list final : public duckdb::SimpleMultiFileList {
 public:
  explicit footer_throwing_list(duckdb::vector<duckdb::OpenFileInfo> files)
    : SimpleMultiFileList(std::move(files))
  {
  }
  duckdb::FileExpandResult GetExpandResult() const override
  {
    ++calls;
    if (fail) { throw std::bad_alloc{}; }
    return SimpleMultiFileList::GetExpandResult();
  }
  bool fail                 = false;
  mutable std::size_t calls = 0;
};

}  // namespace

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer walk stops on local misses rather than the number of paths",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_walk]")
{
  add_files(5);
  int limit             = 1;
  std::size_t inspected = 2, reads = 0;
  bool admitted = false;
  SECTION("cold over limit") {}
  SECTION("zero stops at the first miss")
  {
    limit     = 0;
    inspected = 1;
  }
  SECTION("exactly the limit")
  {
    limit     = 5;
    inspected = reads = 5;
    admitted          = true;
  }
  SECTION("unlimited")
  {
    limit     = -1;
    inspected = reads = 5;
    admitted          = true;
  }
  SECTION("two hits before two misses")
  {
    warm({0, 1});
    inspected = 4;
  }
  SECTION("warm with zero budget")
  {
    warm({0, 1, 2, 3, 4});
    limit     = 0;
    inspected = 5;
    admitted  = true;
  }
  local_limit(limit);
  auto const visits = footer_detail::inspected_file_count();
  auto const before = footer_detail::footer_read_count();
  auto result       = plan(scan({0, 1, 2, 3, 4}));
  CHECK(footer_detail::inspected_file_count() - visits == inspected);
  CHECK(footer_detail::footer_read_count() - before == reads);
  check_footer_scan(*result, 5, admitted ? std::optional<std::size_t>{500} : std::nullopt);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer walk for explicit analysis ignores a zero planning limit",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_walk]")
{
  add_files(4);
  local_limit(0);
  auto const visits = footer_detail::inspected_file_count();
  auto const before = footer_detail::footer_read_count();
  auto result       = footer_query_ok(
    *con, "SELECT files FROM sirius_analyze_parquet('" + (dir / "*.parquet").string() + "')");
  CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == 4);
  CHECK(footer_detail::inspected_file_count() - visits == 4);
  CHECK(footer_detail::footer_read_count() - before == 4);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer walk refreshes trailing hits when cache eviction is possible",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_walk]")
{
  add_files(6);
  (void)footer_detail::set_cache_capacities_for_testing({2, 128, 128});
  local_limit(1);
  warm({0});
  warm({1});
  auto const visits = footer_detail::inspected_file_count();
  auto const before = footer_detail::footer_read_count();
  auto rejected     = plan(scan({2, 3, 0}));
  check_footer_scan(*rejected, 3, std::nullopt);
  CHECK(footer_detail::inspected_file_count() - visits == 3);
  CHECK(footer_detail::footer_read_count() == before);
  auto inserted = plan(scan({4}));
  check_footer_scan(*inserted, 1, 100);
  REQUIRE(footer_detail::footer_read_count() - before == 1);
  auto retained = plan(scan({0, 5}));
  check_footer_scan(*retained, 2, 200);
  CHECK(footer_detail::footer_read_count() - before == 2);
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "Remote footer opens distinguish the inclusive default from unlimited admission",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_defaults]")
{
  auto const count    = GENERATE(1024U, 1025U);
  auto const rollback = GENERATE(false, true);
  CAPTURE(count, rollback);
  footer_cache_scope caches;
  if (rollback) { run_ok("SET parquet_footer_statistics_max_remote_reads=-1"); }
  run_ok("SET parquet_footer_statistics=false");
  auto plan = con->ExtractPlan("SELECT x FROM v");
  REQUIRE(plan);
  auto& bind = footer_fix_bind(*plan);
  auto& fsys = con->context->db->GetFileSystem();
  auto mock  = duckdb::make_uniq<footer_listing_observer>(count);
  auto& seen = *mock;
  scoped_footer_observer registration{fsys, seen.GetName()};
  fsys.RegisterSubSystem(std::move(mock));
  auto files = fsys.Glob("s3a://footer-guard/*.parquet", nullptr);
  REQUIRE(files.size() == count);
  REQUIRE(duckdb::FileSystem::IsRemoteFile(files[0].path));
  bind.file_list = duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(std::move(files));
  run_ok("SET parquet_footer_statistics=true");
  auto const before = footer_detail::inspected_file_count();
  (void)footer_fix_collect(*con, *plan);
  CHECK(footer_detail::inspected_file_count() - before == count);
  if (rollback || count == 1024) {
    CHECK(seen.opens.load() > 0);
  } else {
    CHECK(seen.requests.load() == 0);
    check_footer_scan(*plan, count, std::nullopt);
  }
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer walk distinguishes remote exhaustion from a local stopping miss",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_walk]")
{
  auto const mixed = GENERATE(false, true);
  CAPTURE(mixed);
  run_ok("SET parquet_footer_statistics=false");
  auto result = plan();
  auto& bind  = footer_fix_bind(*result);
  auto& fsys  = con->context->db->GetFileSystem();
  auto mock   = duckdb::make_uniq<footer_listing_observer>();
  auto& seen  = *mock;
  scoped_footer_observer registration{fsys, seen.GetName()};
  fsys.RegisterSubSystem(std::move(mock));
  // DuckDB classifies s3a as remote, without SiriusHttpFS taking priority over the observer.
  auto files = fsys.Glob("s3a://footer-guard/*.parquet", nullptr);
  REQUIRE(files.size() == 4);
  REQUIRE(seen.lists == 1);
  REQUIRE(duckdb::FileSystem::IsRemoteFile(files[0].path));
  if (mixed) {
    files[0] = duckdb::OpenFileInfo(file(0));
    files[1] = duckdb::OpenFileInfo(file(1));
  }
  // Isolate admission from DuckDB's own requests while binding the schema.
  bind.file_list = duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(std::move(files));
  run_ok("SET parquet_footer_statistics=true");
  run_ok("SET parquet_footer_statistics_max_remote_reads=1");
  local_limit(1);
  auto const visits = footer_detail::inspected_file_count();
  auto const before = footer_detail::footer_read_count();
  REQUIRE(footer_fix_collect(*con, *result) == 0);
  check_footer_scan(*result, 4, std::nullopt);
  CHECK(footer_detail::inspected_file_count() - visits == (mixed ? 2U : 4U));
  CHECK(footer_detail::footer_read_count() == before);
  CHECK(seen.requests.load() == 0);
  CHECK(seen.lists == 1);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer fix isolates uncovered plans from a second connection and its copies",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]")
{
  auto const mode = std::string(GENERATE("disabled", "limited", "unreadable"));
  auto const grow = GENERATE(false, true);
  CAPTURE(mode, grow);
  auto second = std::make_unique<duckdb::Connection>(*con->context->db);
  footer_query_ok(*second, "USE " + attach_alias);
  footer_query_ok(*second, "SET parquet_footer_statistics=true");
  footer_query_ok(*second, "SET parquet_footer_statistics_max_local_reads=-1");
  auto observer = duckdb::make_uniq<footer_open_observer>(dir.string() + "/", file(1));
  auto& watched = *observer;
  auto& system  = con->context->db->GetFileSystem();
  scoped_footer_observer registration{system, watched.GetName()};
  bool refuse                 = mode == "unreadable";
  watched.before_watched_open = [&] {
    if (refuse) { throw duckdb::IOException("footer witness refused the later file"); }
  };
  system.RegisterSubSystem(std::move(observer));
  if (mode == "disabled") { run_ok("SET parquet_footer_statistics=false"); }
  if (mode == "limited") { local_limit(0); }
  footer_detail::clear_caches_for_testing();
  auto uncovered = plan();
  CHECK_FALSE(footer_fix_rows(*uncovered));
  CHECK(footer_scan_bind(*uncovered).file_options.column_statistics.IsNull());
  if (mode == "unreadable") { REQUIRE(watched.watched_opens > 0); }
  refuse = false;
  if (grow) { add_files(3); }
  auto covered = second->ExtractPlan("SELECT x FROM v");
  REQUIRE(covered);
  auto const expected = std::optional<std::size_t>{grow ? 300U : 200U};
  CHECK(footer_fix_rows(*covered) == expected);
  CHECK_FALSE(footer_fix_rows(*uncovered));
  auto uncovered_copy = footer_fix_copy(*con, *uncovered);
  auto covered_copy   = footer_fix_copy(*second, *covered);
  auto covered_again  = footer_fix_copy(*second, *covered_copy);
  CHECK_FALSE(footer_fix_rows(*uncovered_copy));
  CHECK_FALSE(footer_fix_rows(*uncovered));
  CHECK(footer_fix_rows(*covered_copy) == expected);
  CHECK(footer_fix_rows(*covered_again) == expected);
  CHECK(footer_detail::row_count_cache_size() == 1);
  covered.reset();
  covered_copy.reset();
  CHECK(footer_fix_rows(*covered_again) == expected);
  covered_again.reset();
  // ExtractPlan also leaves the post-optimizer hook's copy in this connection.
  second.reset();
  CHECK(footer_detail::row_count_cache_size() == 0);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer fix distinguishes a destroyed file list from a new list at the same address",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]")
{
  run_ok("SET parquet_footer_statistics=false");
  auto original       = plan();
  auto& binding       = footer_fix_bind(*original);
  auto files          = binding.file_list->GetAllFiles();
  binding.file_list   = duckdb::shared_ptr<duckdb::MultiFileList>(new footer_reused_list(files));
  auto const* address = binding.file_list.get();
  duckdb::weak_ptr<duckdb::MultiFileList> owner(binding.file_list);
  REQUIRE(footer_fix_collect(*con, *original) == 1);
  REQUIRE(footer_fix_rows(*original) == std::optional<std::size_t>{200});
  original.reset();
  REQUIRE(owner.expired());
  auto replacement = plan();
  footer_fix_bind(*replacement).file_list =
    duckdb::shared_ptr<duckdb::MultiFileList>(new footer_reused_list(std::move(files)));
  REQUIRE(footer_scan_bind(*replacement).file_list.get() == address);
  CHECK_FALSE(footer_fix_rows(*replacement));
  CHECK(footer_detail::row_count_cache_size() == 0);
  REQUIRE(footer_fix_collect(*con, *replacement) == 1);
  CHECK(footer_fix_rows(*replacement) == std::optional<std::size_t>{200});
  CHECK(footer_detail::row_count_cache_size() == 1);
  replacement.reset();
  CHECK(footer_detail::row_count_cache_size() == 0);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer row counts outlive merged-cache eviction with their plans and copies",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]"
                 "[footer_row_count]")
{
  (void)footer_detail::set_cache_capacities_for_testing({8, 2, 8});
  auto oldest = plan();
  auto second = plan();
  auto copy   = footer_fix_copy(*con, *oldest);
  CHECK(footer_detail::row_count_cache_size() == 2);
  oldest.reset();
  CHECK(footer_fix_rows(*copy) == std::optional<std::size_t>{200});
  CHECK(footer_detail::row_count_cache_size() == 2);
  auto third = plan();
  CHECK(footer_fix_rows(*copy) == std::optional<std::size_t>{200});
  CHECK(footer_fix_rows(*second) == std::optional<std::size_t>{200});
  CHECK(footer_fix_rows(*third) == std::optional<std::size_t>{200});
  CHECK(footer_detail::row_count_cache_size() == 3);
  auto third_copy = footer_fix_copy(*con, *third);
  (void)footer_detail::set_cache_capacities_for_testing({8, 0, 8});
  CHECK(footer_fix_rows(*second) == std::optional<std::size_t>{200});
  CHECK(footer_fix_rows(*third) == std::optional<std::size_t>{200});
  CHECK(footer_fix_rows(*third_copy) == std::optional<std::size_t>{200});
  CHECK(footer_detail::row_count_cache_size() == 3);
  auto unretained = plan();
  CHECK(footer_fix_rows(*unretained) == std::optional<std::size_t>{200});
  CHECK_FALSE(footer_scan_bind(*unretained).file_options.column_statistics.IsNull());
  CHECK(footer_detail::row_count_cache_size() == 4);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer row counts remain available to 200 live plans and their surviving copy",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]"
                 "[footer_row_count]")
{
  (void)footer_detail::set_cache_capacities_for_testing(footer_detail::default_cache_capacities);
  auto connection = std::make_unique<duckdb::Connection>(*con->context->db);
  footer_query_ok(*connection, "USE " + attach_alias);
  footer_query_ok(*connection, "SET parquet_footer_statistics=true");
  footer_query_ok(*connection, "SET parquet_footer_statistics_max_local_reads=-1");
  std::vector<duckdb::unique_ptr<duckdb::LogicalOperator>> plans;
  for (int i = 0; i < 200; ++i) {
    plans.push_back(connection->ExtractPlan("SELECT x FROM v"));
    REQUIRE(plans.back());
  }
  for (auto const& root : plans) {
    CHECK(footer_fix_rows(*root) == std::optional<std::size_t>{200});
  }
  CHECK(footer_detail::row_count_cache_size() == 200);
  auto copy = footer_fix_copy(*connection, *plans.front());
  plans.clear();
  connection.reset();
  CHECK(footer_fix_rows(*copy) == std::optional<std::size_t>{200});
  CHECK(footer_detail::row_count_cache_size() == 1);
  copy.reset();
  CHECK(footer_detail::row_count_cache_size() == 0);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer row-count sweeping bounds expired associations through planning and copying",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]"
  "[footer_row_count]")
{
  run_ok("SET parquet_footer_statistics=false");
  auto anchor = plan();
  REQUIRE(footer_fix_collect(*con, *anchor) == 1);
  auto const files = footer_scan_bind(*anchor).file_list->GetAllFiles();
  std::vector<std::weak_ptr<duckdb::SimpleMultiFileList>> owners;
  auto target = [&] {
    auto root = plan();
    auto list = std::make_shared<duckdb::SimpleMultiFileList>(files);
    // Keep the allocation address distinct after destruction, without keeping the list alive.
    owners.emplace_back(list);
    footer_fix_bind(*root).file_list = duckdb::shared_ptr<duckdb::MultiFileList>(std::move(list));
    return root;
  };
  auto copy_count = [&](duckdb::LogicalOperator const& root) {
    sirius::transparent::copy_parquet_scan_row_count(*footer_parquet_get(*anchor),
                                                     *footer_parquet_get(root));
  };
  constexpr std::size_t small_bound = 2 * 2 + 64;
  for (bool copying : {false, true}) {
    std::size_t sweeps = 0;
    for (int i = 0; i < 1000; ++i) {
      CAPTURE(copying, i);
      auto root         = target();
      auto const before = footer_detail::row_count_association_count();
      if (copying) {
        copy_count(*root);
      } else {
        REQUIRE(footer_fix_collect(*con, *root) == 1);
      }
      auto const after = footer_detail::row_count_association_count();
      if (after < before) { ++sweeps; }
      CHECK(after <= small_bound);
      CHECK(footer_fix_rows(*root) == std::optional<std::size_t>{200});
      CHECK(footer_fix_rows(*anchor) == std::optional<std::size_t>{200});
      root.reset();
      CHECK(owners.back().expired());
    }
    CHECK(sweeps > 0);
  }
  std::vector<duckdb::unique_ptr<duckdb::LogicalOperator>> burst;
  constexpr std::size_t burst_bound = 2 * 201 + 64;
  for (int i = 0; i < 200; ++i) {
    auto root = target();
    copy_count(*root);
    CHECK(footer_fix_rows(*root) == std::optional<std::size_t>{200});
    burst.push_back(std::move(root));
    CHECK(footer_detail::row_count_association_count() <= burst_bound);
  }
  burst.clear();
  CHECK(std::ranges::all_of(owners, [](auto const& owner) { return owner.expired(); }));
  auto const after_release = footer_detail::row_count_association_count();
  CAPTURE(after_release);
  CHECK(after_release > small_bound);
  CHECK(after_release <= burst_bound);
  for (int i = 0; i < 1000; ++i) {
    auto root = target();
    copy_count(*root);
    CHECK(footer_detail::row_count_association_count() <= burst_bound);
    CHECK(footer_fix_rows(*root) == std::optional<std::size_t>{200});
    CHECK(footer_fix_rows(*anchor) == std::optional<std::size_t>{200});
  }
  CHECK(footer_detail::row_count_association_count() <= small_bound);
  // This final explicit cleanup is outside the automatic-sweep observation interval.
  CHECK(footer_detail::row_count_cache_size() == 1);
  CHECK(footer_detail::row_count_association_count() == 1);
  anchor.reset();
  CHECK(footer_detail::row_count_cache_size() == 0);
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer row-count copying ignores missing sources and aliases and forgets only its owner",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]"
  "[footer_row_count]")
{
  run_ok("SET parquet_footer_statistics=false");
  auto source      = plan();
  auto target      = plan();
  auto const& from = *footer_parquet_get(*source);
  auto const& to   = *footer_parquet_get(*target);
  sirius::transparent::copy_parquet_scan_row_count(from, from);
  sirius::transparent::copy_parquet_scan_row_count(from, to);
  CHECK(footer_detail::row_count_association_count() == 0);
  CHECK_FALSE(footer_fix_rows(*target));
  REQUIRE(footer_fix_collect(*con, *source) == 1);
  auto const entries                 = footer_detail::row_count_association_count();
  footer_fix_bind(*target).file_list = footer_scan_bind(*source).file_list;
  sirius::transparent::copy_parquet_scan_row_count(from, from);
  sirius::transparent::copy_parquet_scan_row_count(from, to);
  CHECK(footer_detail::row_count_association_count() == entries);
  CHECK(footer_fix_rows(*target) == std::optional<std::size_t>{200});
  auto copied = footer_fix_copy(*con, *source);
  sirius::transparent::forget_parquet_scan_row_counts(*source);
  CHECK_FALSE(footer_fix_rows(*source));
  CHECK_FALSE(footer_fix_rows(*target));
  CHECK(footer_fix_rows(*copied) == std::optional<std::size_t>{200});
  auto unknown = footer_fix_copy(*con, *source);
  CHECK_FALSE(footer_fix_rows(*unknown));
  CHECK(footer_fix_rows(*copied) == std::optional<std::size_t>{200});
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer row counts preserve a known zero across a copy",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]"
                 "[footer_row_count]")
{
  run_ok("SET parquet_footer_statistics=false");
  auto empty = plan();
  write_file("empty", 0, 0);
  duckdb::vector<duckdb::OpenFileInfo> files;
  files.emplace_back((dir / "empty.parquet").string());
  footer_fix_bind(*empty).file_list =
    duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(std::move(files));
  REQUIRE(footer_fix_collect(*con, *empty) == 0);
  CHECK(footer_fix_rows(*empty) == std::optional<std::size_t>{0});
  auto copy = footer_fix_copy(*con, *empty);
  CHECK(footer_fix_rows(*copy) == std::optional<std::size_t>{0});
  sirius::transparent::forget_parquet_scan_row_counts(*empty);
  CHECK_FALSE(footer_fix_rows(*empty));
  CHECK(footer_fix_rows(*copy) == std::optional<std::size_t>{0});
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer fix refuses global setting changes without losing session values or NULL defaults",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]")
{
  for (auto const* name : {"parquet_footer_statistics_max_remote_reads",
                           "parquet_footer_statistics_max_local_reads"}) {
    std::string setting(name);
    CAPTURE(setting);
    auto read = [&](duckdb::Connection& connection) {
      auto result = footer_query_ok(connection, "SELECT current_setting('" + setting + "')");
      return result->Collection().GetValue(0, 0);
    };
    for (bool const explicit_value : {false, true}) {
      footer_query_ok(*con, "RESET " + setting);
      if (explicit_value) { footer_query_ok(*con, "SET " + setting + "=2"); }
      auto const before = read(*con);
      CHECK(before.IsNull() == !explicit_value);
      for (auto const& operation : {"SET GLOBAL " + setting + "=7",
                                    "SET GLOBAL " + setting + "=NULL",
                                    "RESET GLOBAL " + setting}) {
        CAPTURE(operation, explicit_value);
        auto rejected = con->Query(operation);
        REQUIRE(rejected);
        CHECK(rejected->HasError());
        auto const after = read(*con);
        CHECK(after.IsNull() == before.IsNull());
        if (!before.IsNull()) { CHECK(after.GetValue<int64_t>() == before.GetValue<int64_t>()); }
        duckdb::Connection fresh(*con->context->db);
        CHECK(read(fresh).IsNull());
      }
    }
    footer_query_ok(*con, "RESET " + setting);
    CHECK(read(*con).IsNull());
    duckdb::Connection fresh(*con->context->db);
    CHECK(read(fresh).IsNull());
  }
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer fix preserves caller facts when file-list expansion fails before the opt-out",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]")
{
  run_ok("SET parquet_footer_statistics=false");
  auto supplied = plan();
  auto& binding = footer_fix_bind(*supplied);
  auto list     = duckdb::shared_ptr<footer_throwing_list>(
    new footer_throwing_list(binding.file_list->GetAllFiles()));
  binding.file_list                      = list;
  auto const caller                      = footer_fixture_statistics();
  binding.file_options.column_statistics = caller;
  auto const reads                       = footer_detail::footer_read_count();
  REQUIRE(footer_fix_collect(*con, *supplied) == 0);
  REQUIRE(list->calls > 0);
  CHECK(binding.file_options.column_statistics == caller);
  auto const calls = list->calls;
  list->fail       = true;
  REQUIRE(footer_fix_collect(*con, *supplied) == 0);
  CHECK(list->calls == calls + 1);
  CHECK(binding.file_options.column_statistics == caller);
  CHECK_FALSE(footer_fix_rows(*supplied));
  CHECK(footer_detail::footer_read_count() == reads);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer fix rolls back a failed LRU node and can retry hit and evict the key",
                 "[.][integration][footer_host_allocation_fault][footer_fix_lru]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  run_ok("SET threads=1");
  (void)footer_detail::set_cache_capacities_for_testing({1, 2, 4});
  std::size_t observed = 0;
  {
    fault counting(fault::filter_shell_scope, 0);
    warm({0});
    observed = counting.stop();
  }
  REQUIRE(observed == 1);
  REQUIRE(footer_detail::statistics_cache_size() == 1);
  footer_detail::clear_caches_for_testing();
  auto const paths = std::vector<std::string>{file(0)};
  bool failed      = false;
  bool fired       = false;
  con->BeginTransaction();
  {
    fault injection(fault::filter_shell_scope, 1);
    try {
      (void)sirius::transparent::read_parquet_footer_statistics(*con->context, paths);
    } catch (std::exception const&) {
      failed = true;
    }
    observed = injection.stop();
    fired    = injection.fired();
  }
  con->Rollback();
  REQUIRE(fired);
  REQUIRE(observed == 1);
  REQUIRE(failed);
  CHECK(footer_detail::statistics_cache_size() == 0);
  CHECK(footer_detail::live_statistics_records() == 0);
  auto before = footer_detail::footer_read_count();
  warm({0});
  CHECK(footer_detail::footer_read_count() == before + 1);
  CHECK(footer_detail::statistics_cache_size() == 1);
  before = footer_detail::footer_read_count();
  warm({0});
  CHECK(footer_detail::footer_read_count() == before);
  warm({1});
  CHECK(footer_detail::statistics_cache_size() == 1);
  before = footer_detail::footer_read_count();
  warm({0});
  CHECK(footer_detail::footer_read_count() == before + 1);
  CHECK(footer_detail::statistics_cache_size() == 1);
  CHECK(footer_detail::live_statistics_records() == 1);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer fix does not charge a failed merged entry and keeps two later merges",
                 "[.][integration][footer_host_allocation_fault][footer_fix_merge]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  run_ok("SET threads=1");
  add_files(4);
  (void)footer_detail::set_cache_capacities_for_testing({4, 2, 4});
  std::size_t observed = 0;
  {
    fault counting(fault::filter_shell_scope, 0);
    auto warmed = plan(scan({0, 1, 2, 3}));
    observed    = counting.stop();
  }
  REQUIRE(observed == 1);
  REQUIRE(footer_detail::statistics_cache_size() == 4);
  (void)footer_detail::set_cache_capacities_for_testing({4, 0, 4});
  (void)footer_detail::set_cache_capacities_for_testing({4, 2, 4});
  REQUIRE(footer_detail::merged_statistics_cache_size() == 0);
  REQUIRE(footer_detail::merged_reference_count() == 0);
  run_ok("SET parquet_footer_statistics=false");
  auto first        = plan(scan({0, 1}));
  auto second       = plan(scan({2, 3}));
  bool fired        = false;
  std::size_t added = 0;
  {
    fault injection(fault::filter_shell_scope, 1);
    added    = footer_fix_collect(*con, *first);
    observed = injection.stop();
    fired    = injection.fired();
  }
  REQUIRE(fired);
  REQUIRE(observed == 1);
  CHECK(added == 0);
  CHECK(footer_scan_bind(*first).file_options.column_statistics.IsNull());
  CHECK_FALSE(footer_fix_rows(*first));
  CHECK(footer_detail::merged_statistics_cache_size() == 0);
  CHECK(footer_detail::merged_reference_count() == 0);
  CHECK(footer_detail::statistics_cache_size() == 4);
  auto const reads = footer_detail::footer_read_count();
  REQUIRE(footer_fix_collect(*con, *first) == 1);
  CHECK(footer_detail::merged_statistics_cache_size() == 1);
  CHECK(footer_detail::merged_reference_count() == 2);
  REQUIRE(footer_fix_collect(*con, *second) == 1);
  CHECK(footer_detail::merged_statistics_cache_size() == 2);
  CHECK(footer_detail::merged_reference_count() == 4);
  CHECK(footer_detail::footer_read_count() == reads);
  CHECK(footer_fix_rows(*first) == std::optional<std::size_t>{200});
  CHECK(footer_fix_rows(*second) == std::optional<std::size_t>{200});
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer row-count publication and copying recover from a failed association allocation",
  "[.][integration][footer_host_allocation_fault][footer_fix_row_count]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  auto const copying = GENERATE(false, true);
  CAPTURE(copying);
  run_ok("SET threads=1");
  run_ok("SET parquet_footer_statistics=false");
  auto source = plan();
  REQUIRE(footer_fix_collect(*con, *source) == 1);
  auto attempt = [&](duckdb::LogicalOperator& target) {
    if (copying) {
      sirius::transparent::copy_parquet_scan_row_count(*footer_parquet_get(*source),
                                                       *footer_parquet_get(target));
    } else {
      (void)footer_fix_collect(*con, target);
    }
  };
  auto calibration     = plan();
  std::size_t observed = 0;
  {
    fault counting(fault::filter_shell_scope, 0);
    attempt(*calibration);
    observed = counting.stop();
  }
  REQUIRE(observed == 1);
  REQUIRE(footer_fix_rows(*calibration) == std::optional<std::size_t>{200});
  calibration.reset();
  REQUIRE(footer_detail::row_count_cache_size() == 1);
  auto target        = plan();
  auto const entries = footer_detail::row_count_association_count();
  bool fired         = false;
  {
    fault injection(fault::filter_shell_scope, 1);
    attempt(*target);
    observed = injection.stop();
    fired    = injection.fired();
  }
  REQUIRE(fired);
  REQUIRE(observed == 1);
  CHECK_FALSE(footer_fix_rows(*target));
  CHECK(footer_scan_bind(*target).file_options.column_statistics.IsNull());
  CHECK(footer_fix_rows(*source) == std::optional<std::size_t>{200});
  CHECK(footer_detail::row_count_association_count() == entries);
  CHECK(footer_detail::row_count_cache_size() == 1);
  attempt(*target);
  CHECK(footer_fix_rows(*target) == std::optional<std::size_t>{200});
  CHECK(footer_fix_rows(*source) == std::optional<std::size_t>{200});
  CHECK(footer_detail::row_count_cache_size() == (copying ? 1 : 2));
  {
    fault injection(fault::filter_shell_scope, 1);
    sirius::transparent::copy_parquet_scan_row_count(*footer_parquet_get(*source),
                                                     *footer_parquet_get(*target));
    observed = injection.stop();
    fired    = injection.fired();
  }
  CHECK(observed == 0);
  CHECK_FALSE(fired);
  CHECK(footer_fix_rows(*target) == std::optional<std::size_t>{200});
}

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer fix keeps row totals in optimized TPC-H Q2 and Q20 and their Sirius copies",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]"
  "[footer_row_count]")
{
  auto const number     = GENERATE(2, 20);
  auto const statistics = GENERATE(false, true);
  auto const file_count = GENERATE(1, 3);
  CAPTURE(number, statistics, file_count);
  auto const data = fs::path(__FILE__).parent_path().parent_path() / "integration/data/parquet";
  auto setup      = [&](std::string const& sql) {
    INFO("TPC-H setup: " << sql);
    REQUIRE_NOTHROW(run_ok(sql));
  };
  setup("SET gpu_execution=false");
  setup("SET parquet_footer_statistics=false");
  auto count_query = [&](std::string const& sql) {
    auto result      = footer_query_ok(*con, sql);
    auto const count = result->Collection().GetValue(0, 0).GetValue<std::int64_t>();
    REQUIRE(count >= 0);
    return static_cast<std::size_t>(count);
  };
  std::map<std::vector<std::string>, std::size_t> manifests;
  for (auto const* table : {"part", "partsupp", "supplier", "nation", "region", "lineitem"}) {
    auto const path = (data / (std::string(table) + ".parquet")).string();
    REQUIRE(fs::exists(path));
    auto const original = "read_parquet(" + sirius::test::sql_literal(path) + ")";
    auto const count    = count_query("SELECT count(*) FROM " + original);
    REQUIRE(count >= 3);
    std::vector<std::string> paths;
    std::string pattern = path;
    if (file_count == 3) {
      auto const folder = dir / table;
      REQUIRE(fs::create_directory(folder));
      auto const temporary = "footer_split_" + std::string(table);
      setup("CREATE TEMP TABLE " + temporary +
            " AS SELECT *, row_number() OVER () AS footer_split_ordinal FROM " + original);
      for (int part = 0; part < 3; ++part) {
        auto const output = (folder / ("part" + std::to_string(part) + ".parquet")).string();
        setup("COPY (SELECT * EXCLUDE (footer_split_ordinal) FROM " + temporary +
              " WHERE (footer_split_ordinal - 1) % 3 = " + std::to_string(part) + ") TO " +
              sirius::test::sql_literal(output));
        paths.push_back(output);
      }
      setup("DROP TABLE " + temporary);
      pattern = (folder / "*.parquet").string();
    } else {
      paths.push_back(path);
    }
    setup("CREATE VIEW " + std::string(table) + " AS SELECT * FROM read_parquet(" +
          sirius::test::sql_literal(pattern) + ")");
    auto expected_schema = footer_query_ok(*con, "DESCRIBE SELECT * FROM " + original);
    auto actual_schema   = footer_query_ok(*con, "DESCRIBE SELECT * FROM " + std::string(table));
    CHECK(collect_rows(*actual_schema, false) == collect_rows(*expected_schema, false));
    CHECK(count_query("SELECT count(*) FROM ((SELECT * FROM " + std::string(table) +
                      " EXCEPT ALL SELECT * FROM " + original + ") UNION ALL (SELECT * FROM " +
                      original + " EXCEPT ALL SELECT * FROM " + std::string(table) +
                      ")) AS difference") == 0);
    std::size_t sum = 0;
    for (auto const& part : paths) {
      auto const part_rows =
        count_query("SELECT count(*) FROM read_parquet(" + sirius::test::sql_literal(part) + ")");
      REQUIRE(part_rows > 0);
      sum += part_rows;
    }
    CHECK(sum == count);
    std::ranges::sort(paths);
    manifests.emplace(std::move(paths), sum);
  }
  auto const* query        = number == 2 ? sirius::test::kTpchQ2 : sirius::test::kTpchQ20;
  auto reference           = footer_query_ok(*con, query);
  auto const expected_rows = collect_rows(*reference);
  setup("SET gpu_execution=true");
  setup(std::string("SET parquet_footer_statistics=") + (statistics ? "true" : "false"));

  // Observe the actual query's optimized plan while its planning context is active.
  struct observer_info : duckdb::OptimizerExtensionInfo {
    std::atomic<duckdb::ClientContext*> target{nullptr};
    std::function<void(duckdb::ClientContext&, duckdb::LogicalOperator const&)> inspect;
    std::function<std::string(duckdb::LogicalOperator const&)> describe;
    std::string before;
    unsigned visits = 0;
  };
  auto observed = duckdb::make_shared_ptr<observer_info>();
  struct disable_observer {
    duckdb::shared_ptr<observer_info> info;
    ~disable_observer()
    {
      info->target   = nullptr;
      info->inspect  = {};
      info->describe = {};
    }
  } reset{observed};
  observed->describe = [](duckdb::LogicalOperator const& root) {
    std::ostringstream text;
    auto visit = [&](auto const& self, duckdb::LogicalOperator const& op) -> void {
      if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto const& get  = op.Cast<duckdb::LogicalGet>();
        auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get());
        if (bind && bind->file_list) {
          auto const rows = sirius::transparent::parquet_scan_row_count(get);
          text << "scan " << get.table_index.index << " list " << bind->file_list.get() << " rows "
               << (rows ? std::to_string(*rows) : "unknown") << '\n';
          for (auto const& file : bind->file_list->GetAllFiles()) {
            text << file.path << '\n';
          }
        }
      }
      for (auto const& child : op.children) {
        self(self, *child);
      }
    };
    visit(visit, root);
    return text.str();
  };
  observed->inspect = [&](duckdb::ClientContext& context,
                          duckdb::LogicalOperator const& optimized) {
    INFO("Before optimization:\n" << observed->before);
    INFO("After optimization:\n" << observed->describe(optimized));
    INFO(optimized.ToString());
    auto inspect = [&](duckdb::LogicalOperator const& root) {
      std::size_t scans = 0;
      auto visit        = [&](auto const& self, duckdb::LogicalOperator const& op) -> void {
        if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
          auto const& get  = op.Cast<duckdb::LogicalGet>();
          auto const& name = get.function.GetName().GetIdentifierName();
          if (name == "read_parquet" || name == "parquet_scan") {
            auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(get.bind_data.get());
            REQUIRE(bind);
            auto const files = bind->file_list->GetAllFiles();
            REQUIRE(files.size() == static_cast<std::size_t>(file_count));
            std::vector<std::string> paths;
            for (auto const& file : files) {
              paths.push_back(file.path);
            }
            std::ranges::sort(paths);
            CAPTURE(paths);
            REQUIRE(manifests.contains(paths));
            auto const expected =
              statistics ? std::optional<std::size_t>{manifests.at(paths)} : std::nullopt;
            CHECK(sirius::transparent::parquet_scan_row_count(get) == expected);
            if (file_count == 3) {
              CHECK(bind->file_options.column_statistics.IsNull() == !statistics);
            }
            ++scans;
          }
        }
        for (auto const& child : op.children) {
          self(self, *child);
        }
      };
      visit(visit, root);
      REQUIRE(scans > 0);
      return scans;
    };
    auto const scans = inspect(optimized);
    duckdb::unique_ptr<duckdb::LogicalOperator> copy;
    REQUIRE_NOTHROW(copy = sirius::transparent::copy_logical_plan(optimized, context));
    REQUIRE(copy);
    CHECK(inspect(*copy) == scans);
  };
  duckdb::OptimizerExtension observer;
  observer.optimizer_info        = observed;
  observer.pre_optimize_function = [](duckdb::OptimizerExtensionInput& input,
                                      duckdb::unique_ptr<duckdb::LogicalOperator>& plan) {
    auto& info = static_cast<observer_info&>(*input.info);
    if (info.target.load() != &input.context) { return; }
    auto state = duckdb::get_sirius_connection_state(input.context);
    if (!state || state->is_internal_query_active()) { return; }
    info.before = info.describe(*plan);
  };
  observer.optimize_function = [](duckdb::OptimizerExtensionInput& input,
                                  duckdb::unique_ptr<duckdb::LogicalOperator>& plan) {
    auto& info = static_cast<observer_info&>(*input.info);
    if (info.target.load() != &input.context) { return; }
    auto state = duckdb::get_sirius_connection_state(input.context);
    if (!state || state->is_internal_query_active()) { return; }
    ++info.visits;
    info.inspect(input.context, *plan);
  };
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con->context),
                                       std::move(observer));
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  observed->target  = con->context.get();
  INFO(query);
  duckdb::unique_ptr<duckdb::QueryResult> result;
  REQUIRE_NOTHROW(result = con->Query(query));
  observed->target = nullptr;
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : "query succeeded"));
  REQUIRE_FALSE(result->HasError());
  CHECK(collect_rows(*result) == expected_rows);
  CHECK(observed->visits == 1);
  sirius::test::require_transparent_execution_delta(
    before, sirius::test::get_transparent_execution_stats(*con), 1, 0, 1);
}

namespace {

std::vector<duckdb::LogicalGet const*> footer_parquet_gets(duckdb::LogicalOperator const& plan)
{
  std::vector<duckdb::LogicalGet const*> gets;
  auto visit = [&](auto const& self, duckdb::LogicalOperator const& op) -> void {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
      auto const& get  = op.Cast<duckdb::LogicalGet>();
      auto const& name = get.function.GetName().GetIdentifierName();
      if (name == "read_parquet" || name == "parquet_scan") { gets.push_back(&get); }
    }
    for (auto const& child : op.children) {
      self(self, *child);
    }
  };
  visit(visit, plan);
  return gets;
}

}  // namespace

TEST_CASE_METHOD(
  FooterAdmissionFixture,
  "Footer admission at the default capacities plans two multi-file scans again "
  "without a footer read",
  "[integration][gpu_execution][parquet_footer_statistics][footer_admission_feature]")
{
  add_files(4);
  auto const query = "SELECT a.x, b.x FROM read_parquet(['" + file(0) + "', '" + file(1) +
                     "']) a CROSS JOIN read_parquet(['" + file(2) + "', '" + file(3) +
                     "']) b WHERE a.y IS NOT NULL";
  footer_detail::clear_caches_for_testing();
  auto before = footer_detail::footer_read_count();
  auto first  = plan(query);
  CHECK(footer_detail::footer_read_count() - before == 4);
  auto const first_scans = footer_parquet_gets(*first);
  REQUIRE(first_scans.size() == 2);
  before      = footer_detail::footer_read_count();
  auto second = plan(query);
  CHECK(footer_detail::footer_read_count() == before);
  auto const second_scans = footer_parquet_gets(*second);
  REQUIRE(second_scans.size() == 2);
  for (std::size_t i = 0; i < 2; ++i) {
    CAPTURE(i);
    auto const& bind = second_scans[i]->bind_data->Cast<duckdb::MultiFileBindData>();
    CHECK_FALSE(bind.file_options.column_statistics.IsNull());
    auto const& earlier = first_scans[i]->bind_data->Cast<duckdb::MultiFileBindData>();
    CHECK(bind.file_options.column_statistics == earlier.file_options.column_statistics);
    CHECK(sirius::transparent::parquet_scan_row_count(*second_scans[i]) ==
          std::optional<std::size_t>{200});
  }
  CHECK(footer_detail::statistics_cache_size() == 4);
  CHECK(footer_detail::merged_statistics_cache_size() == 2);
  CHECK(footer_detail::merged_reference_count() == 4);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer fix evicts the least recently used statistics, not the oldest",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]")
{
  add_files(3);
  (void)footer_detail::set_cache_capacities_for_testing({2, 128, 2});
  auto reads = [&](int index) {
    auto const before = footer_detail::footer_read_count();
    auto result       = plan("SELECT x FROM read_parquet('" + file(index) + "')");
    check_footer_scan(*result, 1, 100);
    return footer_detail::footer_read_count() - before;
  };
  warm({0});
  warm({1});
  CHECK(reads(0) == 0);
  // f1 is now the least recently used and f0 the oldest entry.
  warm({2});
  CHECK(footer_detail::statistics_cache_size() == 2);
  CHECK(reads(0) == 0);
  CHECK(reads(1) == 1);
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer fix withdraws the row total of a planned scan when it is collected again",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]")
{
  auto const second_pass = std::string(GENERATE("limited", "disabled"));
  CAPTURE(second_pass);
  auto covered = plan();
  REQUIRE(footer_fix_rows(*covered) == std::optional<std::size_t>{200});
  auto const reads = footer_detail::footer_read_count();
  if (second_pass == "limited") {
    // No cached statistics, so both files are misses over the limit.
    (void)footer_detail::set_cache_capacities_for_testing({0, 128, 0});
    local_limit(0);
    CHECK(footer_fix_collect(*con, *covered) == 0);
  } else {
    run_ok("SET gpu_execution=false");
    run_ok("SET parquet_footer_statistics=false");
    con->BeginTransaction();
    duckdb::Planner planner(*con->context);
    duckdb::Optimizer optimizer(*planner.binder, *con->context);
    duckdb::OptimizerExtensionInput input{*con->context, optimizer, nullptr};
    sirius::transparent::sirius_pre_optimizer_hook(input, covered);
    con->Commit();
    run_ok("SET gpu_execution=true");
  }
  REQUIRE(covered);
  CHECK(footer_detail::footer_read_count() == reads);
  CHECK_FALSE(footer_fix_rows(*covered));
}

TEST_CASE_METHOD(ParquetFilesFixture,
                 "parquet footer statistics - a column's range limits its distinct values",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  using sirius::transparent::parquet_scan_value_span;
  using span = std::optional<std::uint64_t>;

  // x covers [0, 200) across the two files. y holds the even numbers from 0 to 198 and NULLs: 199
  // values lie in that range, though it holds 100 of them.
  auto const multi = con->ExtractPlan("SELECT x, y FROM v WHERE x >= 150");
  auto const* get  = first_get(*multi);
  REQUIRE(get != nullptr);
  CHECK(parquet_scan_value_span(*get, 0) == span{200});
  CHECK(parquet_scan_value_span(*get, 1) == span{199});
  CHECK(parquet_scan_value_span(*get, 2) == std::nullopt);

  // One file: its own range.
  auto const single =
    con->ExtractPlan("SELECT x FROM read_parquet('" + (dir / "f1.parquet").string() + "')");
  auto const* single_get = first_get(*single);
  REQUIRE(single_get != nullptr);
  CHECK(parquet_scan_value_span(*single_get, 0) == span{100});
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "parquet footer statistics - a column's value span goes where the scan's row "
                 "count goes",
                 "[integration][gpu_execution][transparent][parquet_footer_statistics]")
{
  using span         = std::optional<std::uint64_t>;
  auto const span_of = [](duckdb::LogicalOperator const& plan) {
    auto const* get = footer_parquet_get(plan);
    REQUIRE(get != nullptr);
    return sirius::transparent::parquet_scan_value_span(*get, 0);
  };

  // A second planning reads no footer and merges nothing: it takes the spans with the merged
  // statistics it reuses.
  auto first       = plan();
  auto const reads = footer_detail::footer_read_count();
  auto second      = plan();
  CHECK(footer_detail::footer_read_count() == reads);
  CHECK(footer_detail::merged_statistics_cache_size() == 1);
  CHECK(span_of(*first) == span{200});
  CHECK(span_of(*second) == span{200});

  // A copy of the plan has them, and keeps them when the plan it was copied from forgets.
  auto copy = footer_fix_copy(*con, *first);
  CHECK(span_of(*copy) == span{200});
  sirius::transparent::forget_parquet_scan_row_counts(*first);
  CHECK_FALSE(span_of(*first));
  CHECK(span_of(*copy) == span{200});

  // A scan over the read limit has neither a row count nor a span.
  footer_detail::clear_caches_for_testing();
  local_limit(0);
  auto uncovered = plan();
  CHECK_FALSE(footer_fix_rows(*uncovered));
  CHECK_FALSE(span_of(*uncovered));
}

TEST_CASE_METHOD(FooterAdmissionFixture,
                 "Footer row count is found through the bind data a physical scan keeps",
                 "[integration][gpu_execution][parquet_footer_statistics][footer_admission_fix]"
                 "[footer_row_count]")
{
  // A physical scan keeps the function name and the bind data of the LogicalGet it was planned
  // from, not the LogicalGet, and Sirius plans from a copy of the plan.
  auto covered    = plan();
  auto copied     = footer_fix_copy(*con, *covered);
  auto const* get = footer_parquet_get(*copied);
  REQUIRE(get != nullptr);
  auto const name = get->function.GetName().GetIdentifierName();
  CHECK(sirius::transparent::parquet_scan_row_count(name, get->bind_data.get()) ==
        std::optional<std::size_t>{200});

  // Only a Parquet scan has a count, and only through the bind data of a covered plan.
  CHECK_FALSE(sirius::transparent::parquet_scan_row_count("seq_scan", get->bind_data.get()));
  CHECK_FALSE(sirius::transparent::parquet_scan_row_count(name, nullptr));
  auto const detached = get->bind_data->Copy();
  CHECK_FALSE(sirius::transparent::parquet_scan_row_count(name, detached.get()));

  // String widths are looked up the same way. These files have integer columns only, so no
  // column has one.
  CHECK(sirius::transparent::parquet_scan_string_widths(name, get->bind_data.get()).empty());
  CHECK(sirius::transparent::parquet_scan_string_widths("seq_scan", get->bind_data.get()).empty());
  CHECK(sirius::transparent::parquet_scan_string_widths(name, nullptr).empty());
}
