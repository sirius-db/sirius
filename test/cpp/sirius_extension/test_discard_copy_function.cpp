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
 * @file test_discard_copy_function.cpp
 * @brief `COPY ... (FORMAT sirius_discard)` on plain DuckDB: the CPU body of the format.
 *
 * Every statement runs with `gpu_execution = false`, so it goes through DuckDB's own COPY operator
 * with the format's discarding callbacks and never reaches the GPU.
 */

#include "catch.hpp"
#include "op/discard_copy_function.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <duckdb.hpp>
#include <duckdb/parser/parsed_data/copy_info.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

namespace {

namespace fs = std::filesystem;

/// Number of entries directly inside @p dir.
std::size_t entry_count(sirius::test::scratch_dir const& dir)
{
  return static_cast<std::size_t>(
    std::distance(fs::directory_iterator(dir.path()), fs::directory_iterator()));
}

/// A DuckDB instance with the `sirius_discard` COPY format registered and GPU execution off.
struct discard_db {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  discard_db()
  {
    sirius::op::register_discard_copy_function(*db.instance);
    // The test binary loads the Sirius extension into every DuckDB instance.
    REQUIRE_FALSE(con.Query("SET gpu_execution = false")->HasError());
  }

  duckdb::unique_ptr<duckdb::QueryResult> copy(std::string const& query,
                                               std::string const& target,
                                               std::string const& options = "")
  {
    return con.Query("COPY (" + query + ") TO " + sirius::test::sql_literal(target) +
                     " (FORMAT sirius_discard" + options + ")");
  }
};

/// The single BIGINT of a successful COPY result: the row count.
std::int64_t copied_rows(duckdb::QueryResult& result)
{
  INFO(result.ToString());
  REQUIRE_FALSE(result.HasError());
  REQUIRE(result.RowCount() == 1);
  REQUIRE(result.ColumnCount() == 1);
  return result.Collection().GetRows().GetValue(0, 0).GetValue<std::int64_t>();
}

/// WARN records saying that a requested `check_order` is not checked on the CPU path.
std::size_t order_unchecked_warnings(sirius::test::scoped_recording_log_sink const& log)
{
  auto const records = log.records();
  return static_cast<std::size_t>(
    std::count_if(records.begin(), records.end(), [](auto const& record) {
      return record.level == sirius::log::level::warn &&
             record.message.rfind("sirius_discard: check_order disabled, the COPY runs on the CPU",
                                  0) == 0;
    }));
}

/// `[discard_sink]` summary lines: the GPU sink's, which carry `order_checked=1` after a check.
std::size_t discard_sink_summaries(sirius::test::scoped_recording_log_sink const& log)
{
  auto const records = log.records();
  return static_cast<std::size_t>(
    std::count_if(records.begin(), records.end(), [](auto const& record) {
      return record.message.rfind("[discard_sink] ", 0) == 0;
    }));
}

void require_binder_error(duckdb::QueryResult& result)
{
  INFO(result.ToString());
  REQUIRE(result.HasError());
  CHECK(result.GetErrorType() == duckdb::ExceptionType::BINDER);
}

duckdb::unique_ptr<duckdb::FunctionData> bind_with_options(
  duckdb::Connection& con, duckdb::CopyInfo& info, duckdb::vector<duckdb::LogicalType> const& types)
{
  auto function = sirius::op::get_discard_copy_function();
  duckdb::CopyFunctionBindInput input(info);
  duckdb::vector<duckdb::Identifier> names;
  for (std::size_t i = 0; i < types.size(); ++i) {
    names.emplace_back(duckdb::Identifier("c" + std::to_string(i)));
  }
  return function.copy_to_bind(*con.context, input, names, types);
}

}  // namespace

TEST_CASE("sirius_discard returns the row count and creates no file", "[discard_copy_function]")
{
  sirius::test::scratch_dir dir("discard_copy");
  discard_db db;
  auto const target = dir.file("x");

  SECTION("a million rows")
  {
    auto result = db.copy("SELECT range FROM range(1000000)", target);
    CHECK(copied_rows(*result) == 1000000);
  }
  SECTION("on one thread")
  {
    REQUIRE_FALSE(db.con.Query("SET threads = 1")->HasError());
    auto result = db.copy("SELECT range FROM range(1000000)", target);
    CHECK(copied_rows(*result) == 1000000);
  }
  SECTION("an ordered query with several column types")
  {
    auto result = db.copy(
      "SELECT range AS k, range::VARCHAR AS s, range / 3.0 AS d FROM range(50000) ORDER BY k DESC",
      target);
    CHECK(copied_rows(*result) == 50000);
  }
  SECTION("no rows")
  {
    auto result = db.copy("SELECT range FROM range(0)", target);
    CHECK(copied_rows(*result) == 0);
  }
  SECTION("a failing query")
  {
    auto result = db.copy(
      "SELECT CASE WHEN range = 5 THEN error('boom') ELSE range END FROM range(10)", target);
    CHECK(result->HasError());
  }

  CHECK_FALSE(fs::exists(target));
  CHECK(entry_count(dir) == 0);
}

TEST_CASE("sirius_discard leaves an existing target alone", "[discard_copy_function]")
{
  sirius::test::scratch_dir dir("discard_copy");
  discard_db db;

  SECTION("an existing file keeps its contents")
  {
    auto const target = dir.path() / "existing";
    {
      std::ofstream out(target);
      out << "keep me";
    }
    auto result = db.copy("SELECT range FROM range(3000)", target.string());
    CHECK(copied_rows(*result) == 3000);

    std::ifstream in(target);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(contents == "keep me");
    CHECK(entry_count(dir) == 1);
  }
  SECTION("an existing directory stays empty")
  {
    auto result = db.copy("SELECT range FROM range(3000)", dir.path().string());
    CHECK(copied_rows(*result) == 3000);
    CHECK(entry_count(dir) == 0);
  }
  SECTION("/dev/null")
  {
    auto result = db.copy("SELECT range FROM range(3000)", "/dev/null");
    CHECK(copied_rows(*result) == 3000);
    CHECK(fs::is_character_file("/dev/null"));
    CHECK(entry_count(dir) == 0);
  }
  SECTION("a path whose directory does not exist")
  {
    auto const target = dir.path() / "missing" / "x";
    auto result       = db.copy("SELECT range FROM range(3000)", target.string());
    CHECK(copied_rows(*result) == 3000);
    CHECK(entry_count(dir) == 0);
  }
}

TEST_CASE("sirius_discard options", "[discard_copy_function]")
{
  sirius::test::scratch_dir dir("discard_copy");
  discard_db db;
  auto const target = dir.file("x");
  auto const query  = std::string("SELECT range FROM range(100) ORDER BY range");

  SECTION("check_order is accepted")
  {
    for (auto const* options :
         {", check_order true", ", check_order false", ", check_order", ", CHECK_ORDER 1"}) {
      INFO(options);
      auto result = db.copy(query, target, options);
      CHECK(copied_rows(*result) == 100);
    }
  }
  SECTION("check_order on the CPU path is not checked, and a WARN says so")
  {
    // Out of order on purpose: the CPU body returns the count whatever the order, so the only
    // sign that nothing was verified is the warning (and the missing [discard_sink] line).
    auto const unordered = std::string("SELECT 99 - range AS k FROM range(100)");
    for (auto const* options : {", check_order true", ", check_order", ", CHECK_ORDER 1"}) {
      INFO(options);
      sirius::test::scoped_recording_log_sink log("info");
      auto result = db.copy(unordered, target, options);
      CHECK(copied_rows(*result) == 100);
      CHECK(order_unchecked_warnings(log) == 1);
      CHECK(discard_sink_summaries(log) == 0);
    }
    for (auto const* options : {"", ", check_order false"}) {
      INFO(options);
      sirius::test::scoped_recording_log_sink log("info");
      auto result = db.copy(unordered, target, options);
      CHECK(copied_rows(*result) == 100);
      CHECK(order_unchecked_warnings(log) == 0);
    }
  }
  SECTION("check_order must be a boolean")
  {
    for (auto const* options : {", check_order 'maybe'", ", check_order (true, false)"}) {
      INFO(options);
      auto result = db.copy(query, target, options);
      require_binder_error(*result);
    }
  }
  SECTION("unknown options are rejected")
  {
    auto result = db.copy(query, target, ", no_such_option 1");
    require_binder_error(*result);
    CHECK(result->GetError().find("no_such_option") != std::string::npos);
  }
  SECTION("DuckDB's file options are rejected")
  {
    for (auto const* options : {", per_thread_output true",
                                ", partition_by (range)",
                                ", use_tmp_file true",
                                ", overwrite true",
                                ", file_size_bytes '1MB'",
                                ", return_stats true",
                                ", header true"}) {
      INFO(options);
      auto result = db.copy(query, target, options);
      require_binder_error(*result);
    }
  }

  CHECK(entry_count(dir) == 0);
}

TEST_CASE("sirius_discard bind data", "[discard_copy_function]")
{
  discard_db db;
  duckdb::vector<duckdb::LogicalType> const types{duckdb::LogicalType::BIGINT,
                                                  duckdb::LogicalType::VARCHAR};

  duckdb::CopyInfo plain_info;
  auto plain_bound  = bind_with_options(db.con, plain_info, types);
  auto const& plain = plain_bound->Cast<sirius::op::discard_copy_bind_data>();
  CHECK_FALSE(plain.check_order);
  CHECK(plain.types == types);

  duckdb::CopyInfo checked_info;
  checked_info.options["check_order"] = {duckdb::Value::BOOLEAN(true)};
  auto checked_bound                  = bind_with_options(db.con, checked_info, types);
  auto const& checked                 = checked_bound->Cast<sirius::op::discard_copy_bind_data>();
  CHECK(checked.check_order);
  CHECK(checked.types == types);

  SECTION("Copy() keeps every field")
  {
    auto copy = checked.Copy();
    REQUIRE(copy != nullptr);
    CHECK(copy.get() != checked_bound.get());
    auto const& copied = copy->Cast<sirius::op::discard_copy_bind_data>();
    CHECK(copied.check_order);
    CHECK(copied.types == types);
    CHECK(checked.Equals(*copy));
    CHECK(copy->Equals(checked));
  }
  SECTION("Equals() compares check_order and the types")
  {
    CHECK(plain.Equals(*plain.Copy()));
    CHECK_FALSE(plain.Equals(checked));
    CHECK_FALSE(checked.Equals(plain));

    auto other_types = bind_with_options(db.con, plain_info, {duckdb::LogicalType::BIGINT});
    CHECK_FALSE(plain.Equals(*other_types));
  }
  SECTION("the bind callback rejects unknown options")
  {
    duckdb::CopyInfo bad_info;
    bad_info.options["compression"] = {duckdb::Value("zstd")};
    CHECK_THROWS_AS(bind_with_options(db.con, bad_info, types), duckdb::BinderException);
  }
}

TEST_CASE("sirius_discard function and registration", "[discard_copy_function]")
{
  auto function = sirius::op::get_discard_copy_function();
  CHECK(function.name == sirius::op::kDiscardCopyFormat);
  CHECK(function.copy_to_bind != nullptr);
  CHECK(function.copy_to_sink != nullptr);
  // The format is write-only.
  CHECK(function.copy_from_bind == nullptr);

  SECTION("the Sirius extension registers the format")
  {
    sirius::test::scratch_dir dir("discard_copy");
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    REQUIRE_FALSE(con.Query("SET gpu_execution = false")->HasError());
    auto result = con.Query("COPY (SELECT range FROM range(77)) TO " + dir.file_literal("x") +
                            " (FORMAT sirius_discard)");
    CHECK(copied_rows(*result) == 77);
    CHECK(entry_count(dir) == 0);
  }
  SECTION("registering again is harmless")
  {
    sirius::test::scratch_dir dir("discard_copy");
    discard_db db;
    REQUIRE_NOTHROW(sirius::op::register_discard_copy_function(*db.db.instance));
    REQUIRE_NOTHROW(sirius::op::register_discard_copy_function(*db.db.instance));

    auto result = db.copy("SELECT range FROM range(1234)", dir.file("x"));
    CHECK(copied_rows(*result) == 1234);
    CHECK(entry_count(dir) == 0);
  }
  SECTION("COPY FROM is not supported")
  {
    discard_db db;
    REQUIRE_FALSE(db.con.Query("CREATE TABLE t (a BIGINT)")->HasError());
    auto result = db.con.Query("COPY t FROM '/dev/null' (FORMAT sirius_discard)");
    CHECK(result->HasError());
  }
}
