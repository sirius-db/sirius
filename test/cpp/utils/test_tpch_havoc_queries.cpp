/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

// Checks the TPC-Havoc query set that CMake downloads (cmake/sirius-tpch-havoc.cmake): the pinned
// BenchBox commit has the expected shape, and every distinct query runs on plain DuckDB against the
// SF 0.1 TPC-H database the GPU tests use. Neither Sirius nor a GPU is involved.

#include "util/env_guard.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/tpch_havoc_queries.hpp>

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <utility>

namespace {

constexpr double kScaleFactor = 0.1;  // integration.duckdb

std::filesystem::path havoc_dir_or_skip()
{
  auto dir = sirius::test::tpch_havoc_dir();
  if (!dir) { SKIP("TPC-Havoc queries not downloaded: configured with -DSIRIUS_TPCH_HAVOC=OFF"); }
  return *dir;
}

void run_ok(duckdb::Connection& con, std::string const& sql)
{
  auto const result = con.Query(sql);
  INFO(sql << ": " << (result->HasError() ? result->GetError() : "ok"));
  REQUIRE_FALSE(result->HasError());
}

}  // namespace

TEST_CASE("tpch_havoc: the pinned query set has 184 distinct queries", "[tpch_havoc]")
{
  auto const queries = sirius::test::load_tpch_havoc_queries(havoc_dir_or_skip(), kScaleFactor);

  std::size_t labels = 0;
  std::set<std::string> distinct_labels;
  std::map<int, int> per_base_query;
  for (auto const& query : queries) {
    labels += 1 + query.duplicates.size();
    CHECK(distinct_labels.insert(query.label).second);
    ++per_base_query[query.base_query];
  }
  CHECK(labels == 220);
  CHECK(queries.size() == 184);
  // Q1-Q18 have ten distinct rewrites; Q19-Q22 are the canonical query ten times over.
  for (int base_query = 1; base_query <= 22; ++base_query) {
    INFO("Q" << base_query);
    CHECK(per_base_query[base_query] == (base_query <= 18 ? 10 : 1));
  }
}

TEST_CASE("tpch_havoc: Q11's threshold is rendered for the scale factor", "[tpch_havoc]")
{
  auto const dir = havoc_dir_or_skip();
  for (auto const& [scale_factor, fraction] :
       {std::pair{0.1, "0.0010000000"}, std::pair{10.0, "0.0000100000"}}) {
    for (auto const& query : sirius::test::load_tpch_havoc_queries(dir, scale_factor)) {
      if (query.base_query != 11) { continue; }
      INFO(query.label << " at SF " << scale_factor);
      CHECK(query.sql.find(fraction) != std::string::npos);
      CHECK(query.sql.find("{q11_fraction}") == std::string::npos);
    }
  }
}

TEST_CASE("tpch_havoc: every query runs on DuckDB at SF 0.1", "[tpch_havoc]")
{
  auto const queries = sirius::test::load_tpch_havoc_queries(havoc_dir_or_skip(), kScaleFactor);

  // The queries are under test here, not Sirius: open plain DuckDB.
  sirius::util::env_guard const disable_sirius("SIRIUS_DISABLE", "1");
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  run_ok(con,
         "ATTACH '" + sirius::test::integration_tpch_db_path().string() + "' AS tpch (READ_ONLY)");
  run_ok(con, "USE tpch");
  run_ok(con, sirius::test::kTpchHavocSessionSetup);

  // Every rewrite of a TPC-H query returns the same rows, so at least the row counts must agree.
  std::map<int, duckdb::idx_t> rows_per_base_query;
  for (auto const& query : queries) {
    auto const result = con.Query(query.sql);
    INFO(query.label << " (" << query.description
                     << "): " << (result->HasError() ? result->GetError() : "ok"));
    CHECK_FALSE(result->HasError());
    if (result->HasError()) { continue; }
    CHECK(result->RowCount() > 0);
    auto const first = rows_per_base_query.try_emplace(query.base_query, result->RowCount()).first;
    CHECK(first->second == result->RowCount());
  }
}
