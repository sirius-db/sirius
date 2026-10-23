/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

// Runs every distinct TPC-Havoc query on the GPU with DuckDB fallback off and compares it with
// DuckDB on the CPU, against the TPC-H SF 0.1 data the other integration tests use. Queries that
// don't pass yet are listed in test/tpch_havoc/expectations.yaml with the outcome they must have;
// everything else must run on the GPU. See test/README.md.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/tpch_havoc_queries.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <filesystem>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <vector>

namespace {

using sirius::test::tpch_havoc_expectation;
using sirius::test::tpch_havoc_outcome;
using sirius::test::tpch_havoc_query;

constexpr double kScaleFactor    = 0.1;  // integration.duckdb
constexpr double kFloatTolerance = 1e-9;

constexpr char kPlanRejected[]  = "GPU plan generation failed: ";
constexpr char kRuntimeFailed[] = "Sirius GPU execution failed: ";

// TPC-H queries whose ORDER BY gives a total order; the others return one row or have ties.
std::set<int> const kTotallyOrdered{1, 2, 4, 7, 8, 9, 12, 13, 15, 16, 20, 21, 22};

std::vector<tpch_havoc_query> const& havoc_queries()
{
  static auto const queries = [] {
    auto const dir = sirius::test::tpch_havoc_dir();
    return dir ? sirius::test::load_tpch_havoc_queries(*dir, kScaleFactor)
               : std::vector<tpch_havoc_query>{};
  }();
  return queries;
}

std::vector<tpch_havoc_expectation> const& expectations()
{
  static auto const loaded =
    sirius::test::load_tpch_havoc_expectations(sirius::test::tpch_havoc_expectations_path());
  return loaded;
}

struct observation {
  tpch_havoc_outcome outcome = tpch_havoc_outcome::gpu;
  std::string detail;  ///< The error, or the first difference from the CPU result.
};

/// The first difference between two results, or nullopt when they match. Columns are compared by
/// position and type; FLOAT and DOUBLE values within kFloatTolerance (relative) count as equal.
std::optional<std::string> first_difference(duckdb::QueryResult& gpu,
                                            duckdb::QueryResult& cpu,
                                            bool ordered)
{
  auto const& types = gpu.GetTypes();
  if (types != cpu.GetTypes()) {
    std::string spelled;
    for (auto const* result : {&gpu, &cpu}) {
      spelled += result == &gpu ? "GPU (" : "), CPU (";
      for (auto const& type : result->GetTypes()) {
        spelled += type.ToString() + " ";
      }
    }
    return "column types differ: " + spelled + ")";
  }
  if (gpu.RowCount() != cpu.RowCount()) {
    return "GPU returned " + std::to_string(gpu.RowCount()) + " rows, CPU " +
           std::to_string(cpu.RowCount());
  }
  auto const gpu_rows = sirius::test::collect_rows(gpu, !ordered);
  auto const cpu_rows = sirius::test::collect_rows(cpu, !ordered);
  for (std::size_t row = 0; row < gpu_rows.size(); ++row) {
    for (std::size_t column = 0; column < gpu_rows[row].size(); ++column) {
      auto const id = types[column].id();
      bool const is_float =
        id == duckdb::LogicalTypeId::FLOAT || id == duckdb::LogicalTypeId::DOUBLE;
      auto const tolerance = is_float ? kFloatTolerance : 0.0;
      auto const& g        = gpu_rows[row][column];
      auto const& c        = cpu_rows[row][column];
      if (!sirius::test::GpuExecutionFixture::cells_equal(g, c, tolerance)) {
        return "row " + std::to_string(row) + " column " + std::to_string(column) + ": GPU [" + g +
               "], CPU [" + c + "]";
      }
    }
  }
  return std::nullopt;
}

class tpch_havoc_fixture {
 public:
  tpch_havoc_fixture()
  {
    if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
      con_ =
        std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
    } else {
      db_  = sirius::test::open_sirius_db(nullptr, sirius::test::integration_config_path());
      con_ = std::make_unique<duckdb::Connection>(*db_);
    }
  }

  /// Runs every distinct variant of `base_query`, one DYNAMIC_SECTION per variant, so
  /// `-c Q05_V03` runs a single query.
  void run_base_query(int base_query, std::string const& scan, std::string const& evaluator)
  {
    auto const dir = sirius::test::tpch_havoc_dir();
    if (!dir) { SKIP("TPC-Havoc queries not downloaded: configured with -DSIRIUS_TPCH_HAVOC=OFF"); }
    for (auto const& query : havoc_queries()) {
      if (query.base_query != base_query) { continue; }
      DYNAMIC_SECTION(query.label)
      {
        set_up(scan, evaluator);
        run_variant(query, scan, evaluator);
      }
    }
  }

 private:
  void run_ok(std::string const& sql)
  {
    auto const result = con_->Query(sql);
    INFO(sql << ": " << (result->HasError() ? result->GetError() : "ok"));
    REQUIRE_FALSE(result->HasError());
  }

  void set_up(std::string const& scan, std::string const& evaluator)
  {
    if (scan == "native") {
      run_ok("ATTACH IF NOT EXISTS '" + sirius::test::integration_tpch_db_path().string() +
             "' AS tpch (READ_ONLY)");
      run_ok("USE tpch");
    } else {
      auto const dir =
        std::filesystem::path{SIRIUS_PROJECT_ROOT} / "test/cpp/integration/data/parquet";
      for (auto const* table :
           {"customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"}) {
        run_ok(std::string{"CREATE OR REPLACE TEMP VIEW "} + table +
               " AS SELECT * FROM read_parquet('" +
               (dir / (std::string{table} + ".parquet")).string() + "')");
      }
    }
    run_ok(sirius::test::kTpchHavocSessionSetup);
    run_ok("SET expression_evaluator_strategy = '" + evaluator + "'");
  }

  void run_variant(tpch_havoc_query const& query,
                   std::string const& scan,
                   std::string const& evaluator)
  {
    INFO(query.label << " (" << query.description << "), " << scan << " scan, " << evaluator << "\n"
                     << query.sql);
    auto const* expected =
      sirius::test::find_tpch_havoc_expectation(expectations(), query.label, scan, evaluator);
    if (expected != nullptr && expected->outcome == tpch_havoc_outcome::skip) {
      WARN(query.label << " skipped: " << expected->reason << " (#" << expected->issue << ")");
      return;
    }

    run_ok("SET gpu_execution = false");
    auto cpu = con_->Query(query.sql);
    INFO("CPU: " << (cpu->HasError() ? cpu->GetError() : "ok"));
    REQUIRE_FALSE(cpu->HasError());
    REQUIRE(cpu->RowCount() > 0);

    bool const ordered = kTotallyOrdered.count(query.base_query) != 0;
    auto const seen    = run_on_gpu(query.sql, *cpu, ordered);

    if (expected == nullptr) {
      INFO(
        query.label << " must run on the GPU and match the CPU, but got " << to_string(seen.outcome)
                    << ": " << seen.detail << "\nIf this is a known gap, add it to "
                    << sirius::test::tpch_havoc_expectations_path().string() << " with an issue.");
      CHECK(seen.outcome == tpch_havoc_outcome::gpu);
      return;
    }
    INFO("Expected " << to_string(expected->outcome) << " matching \"" << expected->reason
                     << "\" (#" << expected->issue << "), got " << to_string(seen.outcome) << ": "
                     << seen.detail);
    if (seen.outcome == tpch_havoc_outcome::gpu) {
      FAIL(query.label << " now runs on the GPU and matches the CPU. Delete it from the "
                       << to_string(expected->outcome) << " entry for #" << expected->issue
                       << " in " << sirius::test::tpch_havoc_expectations_path().string());
    }
    CHECK(seen.outcome == expected->outcome);
    if (expected->outcome != tpch_havoc_outcome::mismatch) {
      CHECK(std::regex_search(seen.detail, std::regex{expected->reason}));
    }

    // What users get by default: the fallback must still return the CPU result.
    if (expected->outcome == tpch_havoc_outcome::runtime_failed) {
      run_ok("SET gpu_execution = true");
      run_ok("SET enable_duckdb_fallback = true");
      auto const before = sirius::test::get_transparent_execution_stats(*con_);
      auto fallback     = con_->Query(query.sql);
      auto const after  = sirius::test::get_transparent_execution_stats(*con_);
      INFO("with fallback: " << (fallback->HasError() ? fallback->GetError() : "ok"));
      REQUIRE_FALSE(fallback->HasError());
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
      auto const difference = first_difference(*fallback, *cpu, ordered);
      INFO("with fallback: " << difference.value_or("matches"));
      CHECK_FALSE(difference.has_value());
    }
  }

  observation run_on_gpu(std::string const& sql, duckdb::QueryResult& cpu, bool ordered)
  {
    run_ok("SET gpu_execution = true");
    run_ok("SET enable_duckdb_fallback = false");
    auto const before = sirius::test::get_transparent_execution_stats(*con_);
    auto gpu          = con_->Query(sql);
    auto const after  = sirius::test::get_transparent_execution_stats(*con_);
    if (gpu->HasError()) {
      auto const& error = gpu->GetError();
      if (error.find(kPlanRejected) != std::string::npos) {
        return {tpch_havoc_outcome::plan_rejected, error};
      }
      if (error.find(kRuntimeFailed) != std::string::npos) {
        return {tpch_havoc_outcome::runtime_failed, error};
      }
      FAIL("GPU run failed with an error that is neither a plan rejection nor a runtime failure: "
           << error);
    }
    INFO("GPU run: executions " << before.executions << " -> " << after.executions << ", fallbacks "
                                << before.fallbacks << " -> " << after.fallbacks);
    REQUIRE(after.executions == before.executions + 1);
    REQUIRE(after.fallbacks == before.fallbacks);
    REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks);
    if (auto difference = first_difference(*gpu, cpu, ordered)) {
      return {tpch_havoc_outcome::mismatch, *difference};
    }
    return {};
  }

  std::unique_ptr<duckdb::DuckDB> db_;
  std::unique_ptr<duckdb::Connection> con_;
};

}  // namespace

// One TEST_CASE per TPC-H query keeps the CI shards balanced. PR CI runs the native scan with the
// default evaluator; the parquet and ast_jit runs are hidden until their runtime is measured.
#define TPCH_HAVOC_TEST_CASES(nn)                                                   \
  TEST_CASE_METHOD(tpch_havoc_fixture,                                              \
                   "gpu_execution - TPC-Havoc Q" #nn,                               \
                   "[integration][gpu_execution][TPC-Havoc][Q" #nn "]")             \
  {                                                                                 \
    run_base_query(std::stoi(#nn), "native", "ast_interpret");                      \
  }                                                                                 \
  TEST_CASE_METHOD(tpch_havoc_fixture,                                              \
                   "gpu_execution - TPC-Havoc Q" #nn " parquet",                    \
                   "[.][integration][gpu_execution][parquet][TPC-Havoc][Q" #nn "]") \
  {                                                                                 \
    run_base_query(std::stoi(#nn), "parquet", "ast_interpret");                     \
  }                                                                                 \
  TEST_CASE_METHOD(tpch_havoc_fixture,                                              \
                   "gpu_execution - TPC-Havoc Q" #nn " ast_jit",                    \
                   "[.][integration][gpu_execution][ast_jit][TPC-Havoc][Q" #nn "]") \
  {                                                                                 \
    run_base_query(std::stoi(#nn), "native", "ast_jit");                            \
  }

TPCH_HAVOC_TEST_CASES(01)
TPCH_HAVOC_TEST_CASES(02)
TPCH_HAVOC_TEST_CASES(03)
TPCH_HAVOC_TEST_CASES(04)
TPCH_HAVOC_TEST_CASES(05)
TPCH_HAVOC_TEST_CASES(06)
TPCH_HAVOC_TEST_CASES(07)
TPCH_HAVOC_TEST_CASES(08)
TPCH_HAVOC_TEST_CASES(09)
TPCH_HAVOC_TEST_CASES(10)
TPCH_HAVOC_TEST_CASES(11)
TPCH_HAVOC_TEST_CASES(12)
TPCH_HAVOC_TEST_CASES(13)
TPCH_HAVOC_TEST_CASES(14)
TPCH_HAVOC_TEST_CASES(15)
TPCH_HAVOC_TEST_CASES(16)
TPCH_HAVOC_TEST_CASES(17)
TPCH_HAVOC_TEST_CASES(18)
TPCH_HAVOC_TEST_CASES(19)
TPCH_HAVOC_TEST_CASES(20)
TPCH_HAVOC_TEST_CASES(21)
TPCH_HAVOC_TEST_CASES(22)
