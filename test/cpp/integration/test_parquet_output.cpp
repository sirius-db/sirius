// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "log/logging.hpp"
#include "output/parquet_writer.hpp"
#include "sirius/exception.hpp"
#include "utils/gpu_execution_fixture.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>

namespace {
struct export_directory {
  std::filesystem::path path;
  export_directory()
  {
    std::string pattern =
      (std::filesystem::temp_directory_path() / "sirius-copy-sql-XXXXXX").string();
    if (!::mkdtemp(pattern.data())) throw std::runtime_error("mkdtemp failed");
    path = pattern;
  }
  ~export_directory() { std::filesystem::remove_all(path); }
};
std::string quote(std::filesystem::path const& path) { return "'" + path.string() + "'"; }
}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY exports ordered batch-sized Parquet parts",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  std::filesystem::create_directory(root.path / "ordered");
  run_ok("SET gpu_execution=false");
  run_ok(
    "CREATE TABLE export_input AS SELECT i::INTEGER k, (i % 7)::INTEGER v FROM range(10000) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  auto result = con->Query("COPY (SELECT * FROM export_input ORDER BY k DESC) TO " +
                           quote(root.path / "ordered") +
                           " (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION 'snappy')");
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 10000);
  run_ok("SET gpu_execution=false");
  std::vector<std::filesystem::path> parts;
  for (auto const& entry : std::filesystem::directory_iterator(root.path / "ordered"))
    parts.push_back(entry.path());
  std::sort(parts.begin(), parts.end());
  int64_t expected = 9999;
  for (auto const& file : parts) {
    auto rows = con->Query("SELECT k FROM read_parquet(" + quote(file) + ")");
    REQUIRE_FALSE(rows->HasError());
    auto metadata = con->Query("SELECT created_by FROM parquet_file_metadata(" + quote(file) + ")");
    REQUIRE_FALSE(metadata->HasError());
    REQUIRE(metadata->Collection().GetRows().GetValue(0, 0).ToString().find("cudf") !=
            std::string::npos);
    for (auto const& row : sirius::test::collect_rows(*rows, false))
      REQUIRE(std::stoll(row[0]) == expected--);
  }
  REQUIRE(expected == -1);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "COPY unsupported options use the existing fallback policy",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_input AS SELECT 42::INTEGER k");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  auto sql = "COPY (SELECT * FROM export_input) TO " + quote(root.path / "unsupported") +
             " (FORMAT PARQUET, PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 2048)";
  auto refused = con->Query(sql);
  REQUIRE(refused->HasError());
  REQUIRE_FALSE(std::filesystem::exists(root.path / "unsupported"));
  run_ok("SET enable_duckdb_fallback=true");
  run_ok(sql);
  REQUIRE(std::filesystem::is_directory(root.path / "unsupported"));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY creates a schema-only part for no rows",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_empty(k INTEGER, d DECIMAL(9,2))");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  run_ok("COPY (SELECT * FROM export_empty) TO " + quote(root.path / "empty") +
         " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
  run_ok("SET gpu_execution=false");
  auto result =
    con->Query("SELECT count(*) FROM read_parquet(" + quote(root.path / "empty/*.parquet") + ")");
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY round trips nullable scalars and decimal widths",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok(
    "CREATE TABLE export_types AS SELECT i::INTEGER k, "
    "CASE WHEN i % 3 = 0 THEN NULL ELSE (i / 10)::DECIMAL(9,2) END small_decimal, "
    "(i / 100)::DECIMAL(18,3) medium_decimal, "
    "CASE WHEN i % 5 = 0 THEN NULL ELSE 'value-' || i END text_value "
    "FROM range(100) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  run_ok(
    "COPY (SELECT k, small_decimal::DECIMAL(9,2) small_decimal, medium_decimal, text_value FROM "
    "export_types) TO " +
    quote(root.path / "types") + " (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION 'zstd')");
  run_ok("SET gpu_execution=false");
  auto result = con->Query(
    "SELECT count(*) FROM ((SELECT k, small_decimal::DECIMAL(9,2), medium_decimal, text_value FROM "
    "export_types EXCEPT ALL SELECT * FROM read_parquet(" +
    quote(root.path / "types/*.parquet") + ")) UNION ALL (SELECT * FROM read_parquet(" +
    quote(root.path / "types/*.parquet") +
    ") EXCEPT ALL SELECT k, small_decimal::DECIMAL(9,2), medium_decimal, text_value FROM "
    "export_types)) differences");
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 0);
  auto schema =
    con->Query("DESCRIBE SELECT * FROM read_parquet(" + quote(root.path / "types/*.parquet") + ")");
  REQUIRE_FALSE(schema->HasError());
  REQUIRE(schema->Collection().GetRows().GetValue(1, 1).ToString() == "DECIMAL(9,2)");
  REQUIRE(schema->Collection().GetRows().GetValue(1, 2).ToString() == "DECIMAL(18,3)");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY honors the installed cuDF codec capabilities",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_codecs AS SELECT i::INTEGER k FROM range(100) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET enable_duckdb_fallback=false");
  for (auto const* codec : {"uncompressed", "snappy", "gzip", "brotli", "zstd", "lz4_raw"}) {
    INFO("codec=" << codec);
    run_ok("SET gpu_execution=true");
    auto directory = root.path / codec;
    auto result =
      con->Query("COPY (SELECT * FROM export_codecs) TO " + quote(directory) +
                 " (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION '" + codec + "')");
    if (!sirius::output::supported_parquet_compression(sirius::output::parse_compression(codec))) {
      REQUIRE(result->HasError());
      REQUIRE_FALSE(std::filesystem::exists(directory));
      continue;
    }
    if (result->HasError()) UNSCOPED_INFO(result->GetError());
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 100);
    run_ok("SET gpu_execution=false");
    auto check = con->Query("SELECT count(*), sum(k) FROM read_parquet(" +
                            quote(directory / "*.parquet") + ")");
    REQUIRE_FALSE(check->HasError());
    REQUIRE(check->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 100);
    REQUIRE(check->Collection().GetRows().GetValue(1, 0).ToString() == "4950");
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU schema-only writer preserves nested names and decimal precision",
                 "[parquet_export_gpu][integration]")
{
  using namespace sirius;
  export_directory root;
  output::parquet_export_spec spec;
  output::parquet_field decimal;
  decimal.name              = "small_decimal";
  decimal.type              = logical_type::make_decimal(4, 2);
  decimal.decimal_precision = 4;
  output::parquet_field nested;
  nested.name = "nested";
  nested.type = logical_type::make(type_id::STRUCT);
  nested.children.push_back(decimal);
  output::parquet_field list;
  list.name    = "values";
  list.type    = logical_type::make(type_id::LIST);
  decimal.name = "element";
  list.children.push_back(decimal);
  spec.fields = {nested, list};
  output::write_empty_parquet(spec, root.path / "nested.parquet");
  run_ok("SET gpu_execution=false");
  auto schema =
    con->Query("DESCRIBE SELECT * FROM read_parquet(" + quote(root.path / "nested.parquet") + ")");
  if (schema->HasError()) UNSCOPED_INFO(schema->GetError());
  REQUIRE_FALSE(schema->HasError());
  REQUIRE(schema->Collection().GetRows().GetValue(1, 0).ToString() ==
          "STRUCT(small_decimal DECIMAL(4,2))");
  REQUIRE(schema->Collection().GetRows().GetValue(1, 1).ToString() == "DECIMAL(4,2)[]");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU writer widens the small-decimal carrier without rescaling",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  sirius::output::parquet_export_spec spec;
  sirius::output::parquet_field decimal;
  decimal.name              = "d";
  decimal.type              = sirius::logical_type::make_decimal(4, 2);
  decimal.decimal_precision = 4;
  spec.fields.push_back(decimal);
  auto column = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT16}, 3, cudf::mask_state::UNALLOCATED);
  int16_t const values[] = {123, -456, 0};
  REQUIRE(cudaMemcpy(column->mutable_view().data<int16_t>(),
                     values,
                     sizeof(values),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  sirius::output::write_parquet_part(spec,
                                     cudf::table_view({column->view()}),
                                     root.path / "decimal.parquet",
                                     cudf::get_default_stream());
  run_ok("SET gpu_execution=false");
  auto rows =
    con->Query("SELECT * FROM read_parquet(" + quote(root.path / "decimal.parquet") + ")");
  REQUIRE_FALSE(rows->HasError());
  auto actual = sirius::test::collect_rows(*rows, false);
  REQUIRE(actual == std::vector<std::vector<std::string>>{{"1.23"}, {"-4.56"}, {"0.00"}});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY rejects nonempty destinations without CPU replay",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  auto target = root.path / "existing";
  std::filesystem::create_directories(target / "nested");
  auto original = target / "nested" / "original.txt";
  std::ofstream(original) << "preserve";
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_input AS SELECT 42::INTEGER k");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=true");
  auto result = con->Query("COPY (SELECT * FROM export_input) TO " + quote(target) +
                           " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
  REQUIRE(result->HasError());
  REQUIRE(result->GetError().find("CPU replay disabled") != std::string::npos);
  REQUIRE(std::filesystem::file_size(original) == 8);
  REQUIRE(std::distance(std::filesystem::directory_iterator(target),
                        std::filesystem::directory_iterator{}) == 1);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU COPY overwrite and append options remain unsupported",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_input AS SELECT 42::INTEGER k");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  for (auto const* option : {"OVERWRITE", "OVERWRITE_OR_IGNORE", "APPEND"}) {
    INFO(option);
    auto target = root.path / option;
    std::filesystem::create_directory(target);
    auto original = target / "original.txt";
    std::ofstream(original) << "preserve";
    auto result = con->Query("COPY (SELECT * FROM export_input) TO " + quote(target) +
                             " (FORMAT PARQUET, PER_THREAD_OUTPUT true, " + option + " true)");
    REQUIRE(result->HasError());
    REQUIRE(std::filesystem::file_size(original) == 8);
    REQUIRE(std::distance(std::filesystem::directory_iterator(target),
                          std::filesystem::directory_iterator{}) == 1);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU Parquet timestamps retain CPU naive timestamp semantics",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("SET TimeZone='America/Los_Angeles'");
  run_ok(
    "CREATE TABLE export_timestamps AS SELECT "
    "TIMESTAMP '2024-03-10 01:02:03.123456' t, "
    "TIMESTAMP '2024-03-10 01:02:03.123'::TIMESTAMP_MS ms, "
    "TIMESTAMP '2024-03-10 01:02:03'::TIMESTAMP_S sec, "
    "TIMESTAMP '2024-03-10 01:02:03.123456'::TIMESTAMP_NS ns "
    "FROM range(3)");
  run_ok("CHECKPOINT");
  for (bool empty : {false, true}) {
    auto suffix = empty ? " WHERE false" : "";
    auto query  = std::string("SELECT * FROM export_timestamps") + suffix;
    auto cpu    = root.path / (empty ? "cpu_empty" : "cpu");
    auto gpu    = root.path / (empty ? "gpu_empty" : "gpu");
    run_ok("SET gpu_execution=false");
    run_ok("COPY (" + query + ") TO " + quote(cpu) + " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
    run_ok("SET gpu_execution=true");
    run_ok("SET enable_duckdb_fallback=false");
    run_ok("COPY (" + query + ") TO " + quote(gpu) + " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
    run_ok("SET gpu_execution=false");
    auto describe = [&](auto const& path) {
      auto schema =
        con->Query("DESCRIBE SELECT * FROM read_parquet(" + quote(path / "*.parquet") + ")");
      if (schema->HasError()) UNSCOPED_INFO(schema->GetError());
      REQUIRE_FALSE(schema->HasError());
      return sirius::test::collect_rows(*schema);
    };
    REQUIRE(describe(cpu) == describe(gpu));
    auto difference =
      con->Query("SELECT count(*) FROM (SELECT * FROM read_parquet(" + quote(cpu / "*.parquet") +
                 ") EXCEPT ALL SELECT * FROM read_parquet(" + quote(gpu / "*.parquet") + "))");
    REQUIRE_FALSE(difference->HasError());
    REQUIRE(difference->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 0);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU empty tuple export retains recursive CPU field names and precision",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_empty_tuple(k INTEGER)");
  run_ok("CHECKPOINT");
  auto query =
    "SELECT row(k, k::DECIMAL(9,2), row(k, TIMESTAMP '2024-01-02 03:04:05')) AS r FROM "
    "export_empty_tuple";
  run_ok(std::string("COPY (") + query + ") TO " + quote(root.path / "cpu_tuple") +
         " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  run_ok(std::string("COPY (") + query + ") TO " + quote(root.path / "gpu_tuple") +
         " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
  run_ok("SET gpu_execution=false");
  auto cpu = con->Query("DESCRIBE SELECT * FROM read_parquet(" +
                        quote(root.path / "cpu_tuple/*.parquet") + ")");
  auto gpu = con->Query("DESCRIBE SELECT * FROM read_parquet(" +
                        quote(root.path / "gpu_tuple/*.parquet") + ")");
  REQUIRE_FALSE(cpu->HasError());
  REQUIRE_FALSE(gpu->HasError());
  REQUIRE(sirius::test::collect_rows(*cpu) == sirius::test::collect_rows(*gpu));
}

namespace {
class interrupted_copy_log_gate final : public sirius::log::sink {
 public:
  void set_level(sirius::log::level) override {}
  bool should_log(sirius::log::level) const override { return true; }
  bool flush() override { return true; }
  void log(sirius::log::level,
           std::source_location const& location,
           std::string_view message) override
  {
    std::unique_lock lock(mutex);
    if (!blocked && message.find("COPY_TO_FILE") != std::string_view::npos &&
        message.find("executing on") != std::string_view::npos) {
      blocked = true;
      condition.notify_all();
      condition.wait_for(lock, std::chrono::seconds(15), [&] { return released; });
    }
    if (std::string_view(location.file_name()).ends_with("sirius_engine.cpp") &&
        message.starts_with("Error executing query:")) {
      diagnostic = true;
      condition.notify_all();
      throw std::bad_alloc();
    }
  }
  bool wait_for(bool interrupted_copy_log_gate::* flag)
  {
    std::unique_lock lock(mutex);
    return condition.wait_for(lock, std::chrono::seconds(5), [&] { return this->*flag; });
  }
  void release()
  {
    std::lock_guard lock(mutex);
    released = true;
    condition.notify_all();
  }
  std::mutex mutex;
  std::condition_variable condition;
  bool blocked    = false;
  bool diagnostic = false;
  bool released   = false;
};
struct restore_export_test_logger {
  std::shared_ptr<sirius::log::sink> previous = sirius::log::get_sink();
  ~restore_export_test_logger() { sirius::log::set_sink(previous); }
};
}  // namespace

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Cancelled COPY retains interrupt classification even when error logging throws",
                 "[parquet_export_gpu][integration]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_cancel_input AS SELECT i::INTEGER k FROM range(100) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=true");
  auto gate = std::make_shared<interrupted_copy_log_gate>();
  duckdb::unique_ptr<duckdb::QueryResult> result;
  bool blocked = false, diagnostic = false;
  {
    restore_export_test_logger restore;
    sirius::log::set_sink(gate, sirius::log::level::trace);
    std::thread query([&] {
      result =
        con->Query("COPY (SELECT * FROM export_cancel_input) TO " + quote(root.path / "cancelled") +
                   " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
    });
    blocked = gate->wait_for(&interrupted_copy_log_gate::blocked);
    if (blocked) {
      con->Interrupt();
      diagnostic = gate->wait_for(&interrupted_copy_log_gate::diagnostic);
    }
    gate->release();
    query.join();
  }
  REQUIRE(blocked);
  REQUIRE(diagnostic);
  REQUIRE(result);
  REQUIRE(result->HasError());
  REQUIRE(result->GetErrorType() == duckdb::ExceptionType::INTERRUPT);
  REQUIRE_FALSE(std::filesystem::exists(root.path / "cancelled"));
  auto next = con->Query("SELECT count(*) FROM export_cancel_input");
  REQUIRE_FALSE(next->HasError());
  REQUIRE(next->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 100);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet export preserves repeated CTE occurrences through UNION ALL",
                 "[parquet_export_gpu][integration][parquet_cte_multiplicity]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_cte_repeat(k VARCHAR, n INTEGER)");
  run_ok("INSERT INTO export_cte_repeat VALUES ('42',42),('42',42),('7',7)");
  run_ok("CHECKPOINT");
  unsigned sequence = 0;
  auto check_export = [&](std::string const& sql) {
    auto target = root.path / ("cte_" + std::to_string(sequence++));
    run_ok("SET gpu_execution=false");
    auto cpu = con->Query(sql);
    REQUIRE_FALSE(cpu->HasError());
    auto expected = sirius::test::collect_rows(*cpu);
    run_ok("SET gpu_execution=true");
    run_ok("SET enable_duckdb_fallback=false");
    auto copied = con->Query("COPY (" + sql + ") TO " + quote(target) +
                             " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
    if (copied->HasError()) UNSCOPED_INFO(copied->GetError());
    REQUIRE_FALSE(copied->HasError());
    REQUIRE(copied->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() ==
            static_cast<int64_t>(expected.size()));
    run_ok("SET gpu_execution=false");
    auto actual = con->Query("SELECT * FROM read_parquet(" + quote(target / "*.parquet") + ")");
    REQUIRE_FALSE(actual->HasError());
    REQUIRE(sirius::test::collect_rows(*actual) == expected);
  };
  for (bool compressed : {true, false}) {
    run_ok(std::string("SET enable_compressed_materialization=") + (compressed ? "true" : "false"));
    for (auto const* column : {"k", "n"}) {
      for (unsigned arms : {2U, 3U}) {
        std::string sql = std::string("WITH c AS MATERIALIZED (SELECT ") + column +
                          " FROM export_cte_repeat) SELECT * FROM c";
        for (unsigned arm = 1; arm < arms; ++arm)
          sql += " UNION ALL SELECT * FROM c";
        check_export(sql);
      }
    }
  }
  run_ok("SET sirius_test_gpu_task_retry_backoff_ms=0");
  run_ok("SET sirius_test_inject_gpu_task_oom=2");
  check_export("SELECT k FROM export_cte_repeat");
  run_ok("SET sirius_test_inject_gpu_task_oom=0");
  check_export(
    "WITH a AS MATERIALIZED (SELECT k FROM export_cte_repeat), "
    "b AS MATERIALIZED (SELECT * FROM a UNION ALL SELECT * FROM a) "
    "SELECT * FROM b UNION ALL SELECT * FROM b");
  check_export(
    "WITH c AS MATERIALIZED (SELECT k FROM export_cte_repeat) "
    "SELECT * FROM c UNION ALL SELECT * FROM c ORDER BY k DESC");
  check_export(
    "WITH c AS MATERIALIZED (SELECT k FROM export_cte_repeat WHERE false) "
    "SELECT * FROM c UNION ALL SELECT * FROM c");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU Parquet seconds preserve infinity and NULL like CPU COPY",
                 "[parquet_export_gpu][integration][parquet_timestamp_seconds]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_seconds(t TIMESTAMP_S)");
  unsigned sequence = 0;
  for (auto const* value : {"'infinity'",
                            "'-infinity'",
                            "NULL",
                            "'1970-01-01 00:00:00'",
                            "'1969-12-31 23:59:59'",
                            "'2024-03-10 01:02:03'"}) {
    INFO(value);
    auto cpu = root.path / ("cpu_" + std::to_string(sequence));
    auto gpu = root.path / ("gpu_" + std::to_string(sequence++));
    run_ok("SET gpu_execution=false");
    run_ok("DELETE FROM export_seconds");
    run_ok(std::string("INSERT INTO export_seconds VALUES (") + value + "::TIMESTAMP_S)");
    run_ok("CHECKPOINT");
    run_ok("COPY (SELECT t FROM export_seconds) TO " + quote(cpu) +
           " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
    run_ok("SET gpu_execution=true");
    run_ok("SET enable_duckdb_fallback=false");
    auto copied = con->Query("COPY (SELECT t FROM export_seconds) TO " + quote(gpu) +
                             " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
    if (copied->HasError()) UNSCOPED_INFO(copied->GetError());
    REQUIRE_FALSE(copied->HasError());
    REQUIRE(copied->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 1);
    run_ok("SET gpu_execution=false");
    auto read = [&](auto const& path) {
      auto rows = con->Query("SELECT t::VARCHAR, isinf(t), t IS NULL FROM read_parquet(" +
                             quote(path / "*.parquet") + ")");
      REQUIRE_FALSE(rows->HasError());
      return sirius::test::collect_rows(*rows);
    };
    REQUIRE(read(gpu) == read(cpu));
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU Parquet seconds normalization recurses into structs lists and arrays",
                 "[parquet_export_gpu][integration][parquet_timestamp_seconds]")
{
  using namespace sirius;
  export_directory root;
  auto seconds = [] {
    auto column = cudf::make_timestamp_column(
      cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, 4, cudf::mask_state::ALL_NULL);
    int64_t const values[]        = {std::numeric_limits<int64_t>::max(),
                                     -std::numeric_limits<int64_t>::max(),
                                     123,
                                     std::numeric_limits<int64_t>::min()};
    cudf::bitmask_type const mask = 7;
    REQUIRE(cudaMemcpy(column->mutable_view().data<int64_t>(),
                       values,
                       sizeof(values),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(
      cudaMemcpy(column->mutable_view().null_mask(), &mask, sizeof(mask), cudaMemcpyHostToDevice) ==
      cudaSuccess);
    column->set_null_count(1);
    return column;
  };
  auto list = [&] {
    auto offsets = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 5, cudf::mask_state::UNALLOCATED);
    int32_t const values[] = {0, 1, 2, 3, 4};
    REQUIRE(cudaMemcpy(offsets->mutable_view().data<int32_t>(),
                       values,
                       sizeof(values),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return cudf::make_lists_column(4, std::move(offsets), seconds(), 0, {});
  };
  output::parquet_field scalar;
  scalar.name = "t";
  scalar.type = logical_type::make(type_id::TIMESTAMP_SEC);
  auto child  = scalar;
  child.name  = "v";
  output::parquet_field structure;
  structure.name     = "st";
  structure.type     = logical_type::make(type_id::STRUCT);
  structure.children = {child};
  output::parquet_field list_field;
  list_field.name     = "li";
  list_field.type     = logical_type::make(type_id::LIST);
  list_field.children = {child};
  auto array_field    = list_field;
  array_field.name    = "ar";
  array_field.type    = logical_type::make_array(scalar.type, 1);
  output::parquet_export_spec spec;
  spec.fields = {scalar, structure, list_field, array_field};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(seconds());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(seconds());
  columns.push_back(cudf::make_structs_column(4, std::move(children), 0, {}));
  columns.push_back(list());
  columns.push_back(list());
  cudf::table table(std::move(columns));
  output::write_parquet_part(
    spec, table.view(), root.path / "nested.parquet", cudf::get_default_stream());
  run_ok("SET gpu_execution=false");
  auto rows = con->Query(
    "SELECT t::VARCHAR, isinf(t), st.v::VARCHAR, li[1]::VARCHAR, "
    "ar[1]::VARCHAR, t IS NULL FROM read_parquet(" +
    quote(root.path / "nested.parquet") + ")");
  REQUIRE_FALSE(rows->HasError());
  auto actual = sirius::test::collect_rows(*rows, false);
  REQUIRE(actual == std::vector<std::vector<std::string>>{
                      {"infinity", "true", "infinity", "infinity", "infinity", "false"},
                      {"-infinity", "true", "-infinity", "-infinity", "-infinity", "false"},
                      {"1970-01-01 00:02:03",
                       "false",
                       "1970-01-01 00:02:03",
                       "1970-01-01 00:02:03",
                       "1970-01-01 00:02:03",
                       "false"},
                      {"NULL", "NULL", "NULL", "NULL", "NULL", "true"}});
  output::write_empty_parquet(spec, root.path / "empty.parquet");
  auto empty =
    con->Query("SELECT count(*) FROM read_parquet(" + quote(root.path / "empty.parquet") + ")");
  REQUIRE_FALSE(empty->HasError());
  REQUIRE(empty->Collection().GetRows().GetValue(0, 0).GetValue<int64_t>() == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU Parquet seconds rejects finite conversion overflow before creating a file",
                 "[parquet_export_gpu][integration][parquet_timestamp_seconds]")
{
  export_directory root;
  auto column = cudf::make_timestamp_column(
    cudf::data_type{cudf::type_id::TIMESTAMP_SECONDS}, 1, cudf::mask_state::UNALLOCATED);
  sirius::output::parquet_export_spec spec;
  sirius::output::parquet_field field;
  field.name = "t";
  field.type = sirius::logical_type::make(sirius::type_id::TIMESTAMP_SEC);
  spec.fields.push_back(field);
  for (auto ticks : {std::numeric_limits<int64_t>::max() / 1000000 + 1,
                     std::numeric_limits<int64_t>::min() / 1000000 - 1}) {
    REQUIRE(cudaMemcpy(column->mutable_view().data<int64_t>(),
                       &ticks,
                       sizeof(ticks),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE_THROWS_AS(sirius::output::write_parquet_part(spec,
                                                         cudf::table_view({column->view()}),
                                                         root.path / "overflow.parquet",
                                                         cudf::get_default_stream()),
                      sirius::invalid_input_exception);
    REQUIRE_FALSE(std::filesystem::exists(root.path / "overflow.parquet"));
  }
}

// External access cannot be re-enabled with SQL. Omit [integration] so this fixture
// owns a private database for every generated scenario instead of borrowing one.
TEST_CASE("GPU COPY validates directory permissions before creating parts",
          "[parquet_export_gpu][isolated_context][parquet_authorization]")
{
  sirius::util::env_guard const enabled("SIRIUS_DISABLE", std::nullopt);
  sirius::test::GpuExecutionFixture fixture;
  auto& con   = fixture.con;
  auto run_ok = [&](std::string const& sql) { fixture.run_ok(sql); };
  REQUIRE(sirius::test::get_registered_sirius_context(*con) != nullptr);
  export_directory root;
  auto const mode     = GENERATE(0, 1, 2, 3, 4);
  auto const fallback = GENERATE(false, true);
  auto const existing = GENERATE(false, true);
  INFO("authorization mode=" << mode << " fallback=" << fallback << " existing=" << existing);
  auto gpu = root.path / "output";
  auto cpu = root.path / "cpu_output";
  if (existing) {
    std::filesystem::create_directory(gpu);
    std::filesystem::create_directory(cpu);
  }
  run_ok("SET gpu_execution=false");
  // Load Parquet before disabling external access, as a configured application would.
  run_ok("SELECT count(*) FROM duckdb_functions() WHERE function_name='parquet_schema'");
  if (mode == 0) {  // Exact destination entries do not authorize generated parts.
    run_ok("SET allowed_paths = [" + quote(gpu) + ", " + quote(cpu) + "]");
  } else if (mode == 1) {  // Wildcards in allowed_paths are literal strings.
    run_ok("SET allowed_paths = [" + quote(gpu / "*") + ", " + quote(cpu / "*") + "]");
  } else if (mode == 2) {  // Directory prefixes plus exact destination access.
    run_ok("SET allowed_paths = [" + quote(gpu) + ", " + quote(cpu) + "]");
    run_ok("SET allowed_directories = [" + quote(gpu) + ", " + quote(cpu) + "]");
  } else if (mode == 3) {  // A parent prefix covers both destination and parts.
    run_ok("SET allowed_directories = [" + quote(root.path) + "]");
  } else {  // A similarly named sibling must not match a directory prefix.
    run_ok("SET allowed_paths = [" + quote(gpu) + ", " + quote(cpu) + "]");
    run_ok("SET allowed_directories = [" + quote(root.path / "out") + "]");
  }
  run_ok(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  run_ok("SET enable_external_access=false");
  run_ok("SET gpu_execution=true");
  auto const sql = [](auto const& path) {
    return "COPY (SELECT 42::INTEGER k) TO " + quote(path) +
           " (FORMAT PARQUET, PER_THREAD_OUTPUT true)";
  };
  auto exported        = con->Query(sql(gpu));
  bool const permitted = mode == 2 || mode == 3;
  if (permitted && exported->HasError()) UNSCOPED_INFO(exported->GetError());
  REQUIRE(exported->HasError() == !permitted);
  if (!permitted) {
    // CPU planning fallback may create an authorized directory before its
    // generated-file check fails. GPU refusal itself must have no side effects.
    if (!fallback) REQUIRE(std::filesystem::exists(gpu) == existing);
    if (std::filesystem::exists(gpu)) REQUIRE(std::filesystem::is_empty(gpu));
  }
  run_ok("SET gpu_execution=false");
  auto baseline = con->Query(sql(cpu));
  REQUIRE(baseline->HasError() == !permitted);
  if (!permitted && std::filesystem::exists(cpu)) REQUIRE(std::filesystem::is_empty(cpu));
  if (permitted) {
    auto metadata =
      con->Query("SELECT created_by FROM parquet_file_metadata(" + quote(gpu / "*.parquet") + ")");
    REQUIRE_FALSE(metadata->HasError());
    REQUIRE(metadata->Collection().GetRows().GetValue(0, 0).ToString().find("cudf") !=
            std::string::npos);
    auto rows = con->Query("SELECT k FROM read_parquet(" + quote(gpu / "*.parquet") + ")");
    REQUIRE_FALSE(rows->HasError());
    REQUIRE(rows->Collection().GetRows().GetValue(0, 0).ToString() == "42");
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU grouped decimal SUM exports storage fitting its declared precision",
                 "[parquet_export_gpu][integration][parquet_decimal_storage]")
{
  export_directory root;
  run_ok("SET gpu_execution=false");
  run_ok("CREATE TABLE export_decimal_sum(k INTEGER, d DECIMAL(9,2))");
  run_ok("INSERT INTO export_decimal_sum VALUES (1,1.00),(1,2.00),(2,NULL),(3,-4.50)");
  run_ok("CHECKPOINT");
  auto expected = con->Query("SELECT k, SUM(d) AS s FROM export_decimal_sum GROUP BY k");
  REQUIRE_FALSE(expected->HasError());
  auto expected_rows = sirius::test::collect_rows(*expected);
  run_ok("SET gpu_execution=true");
  run_ok("SET enable_duckdb_fallback=false");
  run_ok("COPY (SELECT k, SUM(d) AS s FROM export_decimal_sum GROUP BY k) TO " +
         quote(root.path / "sum") + " (FORMAT PARQUET, PER_THREAD_OUTPUT true)");
  run_ok("SET gpu_execution=false");
  auto rows = con->Query("SELECT * FROM read_parquet(" + quote(root.path / "sum/*.parquet") + ")");
  REQUIRE_FALSE(rows->HasError());
  REQUIRE(sirius::test::collect_rows(*rows) == expected_rows);
  auto schema = con->Query("SELECT type, type_length, precision, scale FROM parquet_schema(" +
                           quote(root.path / "sum/*.parquet") + ") WHERE name='s'");
  REQUIRE_FALSE(schema->HasError());
  REQUIRE(sirius::test::collect_rows(*schema) ==
          std::vector<std::vector<std::string>>{{"FIXED_LEN_BYTE_ARRAY", "16", "38", "2"}});
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "GPU Parquet widens nested decimal storage and preserves scale and NULL",
                 "[parquet_export_gpu][integration][parquet_decimal_storage]")
{
  using namespace sirius;
  export_directory root;
  // Exercise both widening steps independently of aggregate carrier choices.
  auto const precision = GENERATE(18, 38);
  auto const source    = precision == 18 ? cudf::type_id::DECIMAL32 : cudf::type_id::DECIMAL64;
  auto decimal         = [&] {
    auto column =
      cudf::make_fixed_point_column(cudf::data_type{source, -2}, 3, cudf::mask_state::ALL_NULL);
    int32_t const small[] = {300, -750, 999};
    int64_t const large[] = {300, -750, 999};
    auto const* data =
      precision == 18 ? static_cast<void const*>(small) : static_cast<void const*>(large);
    auto const bytes = precision == 18 ? sizeof(small) : sizeof(large);
    REQUIRE(cudaMemcpy(column->mutable_view().head<void>(), data, bytes, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    cudf::bitmask_type const mask = 3;
    REQUIRE(
      cudaMemcpy(column->mutable_view().null_mask(), &mask, sizeof(mask), cudaMemcpyHostToDevice) ==
      cudaSuccess);
    column->set_null_count(1);
    return column;
  };
  auto list = [&] {
    auto offsets = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 4, cudf::mask_state::UNALLOCATED);
    int32_t const values[] = {0, 1, 2, 3};
    REQUIRE(cudaMemcpy(offsets->mutable_view().data<int32_t>(),
                       values,
                       sizeof(values),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    return cudf::make_lists_column(3, std::move(offsets), decimal(), 0, {});
  };
  output::parquet_field scalar;
  scalar.name              = "d";
  scalar.type              = logical_type::make_decimal(precision, 2);
  scalar.decimal_precision = precision;
  auto child               = scalar;
  child.name               = "v";
  output::parquet_field structure;
  structure.name     = "st";
  structure.type     = logical_type::make(type_id::STRUCT);
  structure.children = {child};
  output::parquet_field list_field;
  list_field.name     = "li";
  list_field.type     = logical_type::make(type_id::LIST);
  list_field.children = {child};
  auto array_field    = list_field;
  array_field.name    = "ar";
  array_field.type    = logical_type::make_array(scalar.type, 1);
  output::parquet_export_spec spec;
  spec.fields = {scalar, structure, list_field, array_field};
  std::vector<std::unique_ptr<cudf::column>> children;
  children.push_back(decimal());
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(decimal());
  columns.push_back(cudf::make_structs_column(3, std::move(children), 0, {}));
  columns.push_back(list());
  columns.push_back(list());
  cudf::table table(std::move(columns));
  output::write_parquet_part(
    spec, table.view(), root.path / "nested.parquet", cudf::get_default_stream());
  run_ok("SET gpu_execution=false");
  auto rows = con->Query("SELECT d, st.v, li[1], ar[1] FROM read_parquet(" +
                         quote(root.path / "nested.parquet") + ")");
  REQUIRE_FALSE(rows->HasError());
  REQUIRE(sirius::test::collect_rows(*rows, false) ==
          std::vector<std::vector<std::string>>{{"3.00", "3.00", "3.00", "3.00"},
                                                {"-7.50", "-7.50", "-7.50", "-7.50"},
                                                {"NULL", "NULL", "NULL", "NULL"}});
  auto schema = con->Query("SELECT type, type_length, precision, scale FROM parquet_schema(" +
                           quote(root.path / "nested.parquet") + ") WHERE precision IS NOT NULL");
  REQUIRE_FALSE(schema->HasError());
  auto fields = sirius::test::collect_rows(*schema);
  REQUIRE(fields.size() == 4);
  for (auto const& field : fields) {
    REQUIRE(field[0] == (precision == 18 ? "INT64" : "FIXED_LEN_BYTE_ARRAY"));
    if (precision == 38) REQUIRE(field[1] == "16");
    REQUIRE(field[2] == std::to_string(precision));
    REQUIRE(field[3] == "2");
  }
}
