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

// A deferral over a filtered scan of a COMPRESSED pin. Rows can be dropped in
// two places: inside the fused decode, and again by the conjuncts the decode
// could not carry. The pin-order rowid is only correct when both stages are
// composed.
//
// With enable_fused_scan_filter off only the residual stage drops rows; with it
// on the decode compacts first, so both settings are covered.
//
// Uses the default-on enable_late_materialization query policy.

#include <catch.hpp>
#include <late_mat/defer_directive.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/pinned_entry_census.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using CompressedFilterFixture = sirius::test::GpuExecutionFixture;

constexpr std::int64_t kCustomers = 20'000;
constexpr std::int64_t kOrders    = 60'000;
constexpr std::int64_t kLines     = 120'000;

/// Three wide strings read only by the aggregate at the far end, two joins to
/// clear the crossing floor, and a range predicate on the pinned table so the
/// scan restricts rows. The predicate is a bare range on the key, which a
/// bitpack-rooted plan can answer inside the decode.
constexpr char const* kQuery =
  "SELECT c.c_custkey, c.c_name, c.c_address, c.c_comment, count(*) AS n "
  "FROM read_parquet({C}) c "
  "JOIN read_parquet({O}) o ON c.c_custkey = o.o_custkey "
  "JOIN read_parquet({L}) l ON o.o_orderkey = l.l_orderkey "
  "WHERE c.c_custkey < 4000 {X}"
  "GROUP BY c.c_custkey, c.c_name, c.c_address, c.c_comment "
  "ORDER BY c.c_custkey";

/// A conjunct on an identity-planned string column, which no decode route can
/// answer. It leaves the range pushed down but the request no longer covering
/// the whole filter, so the decode compacts and the residual filter then runs
/// over what it kept — the two stages whose positions have to be composed.
constexpr char const* kResidualConjunct = "AND c.c_name LIKE '%7' ";

std::string query_for(fs::path const& customer,
                      fs::path const& orders,
                      fs::path const& lines,
                      bool residual)
{
  std::string sql = kQuery;
  sql.replace(sql.find("{X}"), 3, residual ? kResidualConjunct : "");
  sql.replace(sql.find("{C}"), 3, sirius::test::sql_literal(customer.string()));
  sql.replace(sql.find("{O}"), 3, sirius::test::sql_literal(orders.string()));
  sql.replace(sql.find("{L}"), 3, sirius::test::sql_literal(lines.string()));
  return sql;
}

std::vector<std::string> rows_of(duckdb::QueryResult& result)
{
  std::vector<std::string> rows;
  auto result_rows = result.Collection().GetRows();
  for (duckdb::idx_t i = 0; i < result_rows.size(); ++i) {
    std::string row;
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      row += result_rows.GetValue(c, i).ToString() + "|";
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

// Row equality alone cannot distinguish successful GPU execution from CPU replay.
auto query_on_gpu(duckdb::Connection& con, std::string const& sql)
{
  auto const before = sirius::test::get_transparent_execution_stats(con);
  auto result       = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  auto const after = sirius::test::get_transparent_execution_stats(con);
  REQUIRE(after.executions == before.executions + 1);
  REQUIRE(after.fallbacks == before.fallbacks);
  REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks);
  return result;
}

/// A bitpack block for the key, so its decode produces a range-answering mask —
/// the plan shape that lets the decode drop rows at all. The string payloads sit
/// on identity and take the full-decode route, compacted by the survivor gather.
void write_plan_file(fs::path const& plan_dir, std::string const& table)
{
  std::ofstream f(plan_dir / (table + ".txt"));
  REQUIRE(f.good());
  f << "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
       "---\n"
       "input -> identity\n"
       "---\n"
       "input -> identity\n"
       "---\n"
       "input -> identity\n";
}

}  // namespace

TEST_CASE_METHOD(CompressedFilterFixture,
                 "a deferral over a filtered compressed pin rebuilds its rowid from both filter "
                 "stages",
                 "[integration][gpu_execution][late_mat][compression][compressed_filter]")
{
  sirius::test::scratch_dir scratch{"late_mat_compressed_filter"};
  auto const& tmp     = scratch.path();
  auto const customer = tmp / "lm_comp_customer.parquet";
  auto const orders   = tmp / "orders.parquet";
  auto const lines    = tmp / "lineitem.parquet";

  run_ok("SET gpu_execution = false;");
  run_ok(
    "COPY (SELECT range AS c_custkey, "
    "'Customer#' || lpad(CAST(range AS VARCHAR), 12, '0') AS c_name, "
    "'address-' || repeat(CAST(range % 97 AS VARCHAR), 6) AS c_address, "
    "'comment for customer ' || CAST(range AS VARCHAR) AS c_comment "
    "FROM range(" +
    std::to_string(kCustomers) + ")) TO " + sirius::test::sql_literal(customer.string()) +
    " (FORMAT PARQUET);");
  run_ok("COPY (SELECT range AS o_orderkey, range % " + std::to_string(kCustomers) +
         " AS o_custkey FROM range(" + std::to_string(kOrders) + ")) TO " +
         sirius::test::sql_literal(orders.string()) + " (FORMAT PARQUET);");
  run_ok("COPY (SELECT range AS l_linekey, range % " + std::to_string(kOrders) +
         " AS l_orderkey FROM range(" + std::to_string(kLines) + ")) TO " +
         sirius::test::sql_literal(lines.string()) + " (FORMAT PARQUET);");

  // The CPU answers, taken before anything is pinned.
  std::vector<std::string> expected[2];
  for (bool const residual : {false, true}) {
    auto cpu = con->Query(query_for(customer, orders, lines, residual));
    REQUIRE(cpu);
    REQUIRE_FALSE(cpu->HasError());
    expected[residual] = rows_of(*cpu);
    REQUIRE_FALSE(expected[residual].empty());
  }
  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");

  // Plan files are matched by stem, so they must not share a directory with the Parquet input.
  auto const plan_dir = tmp / "plans";
  fs::create_directories(plan_dir);
  write_plan_file(plan_dir, "lm_comp_customer");
  run_ok("SET pin_table_compression = true;");
  run_ok("SET pin_table_compression_min_batch_size_bytes = 0;");
  run_ok("SET pin_table_compression_max_compressed_fraction = 1.5;");
  run_ok("SET pin_table_input_compression_plan_dir = " +
         sirius::test::sql_literal(plan_dir.string()) + ";");
  // c_custkey rides real as the join and group key; the pin proves it distinct, which is what
  // moves the port past the aggregate.
  run_ok("CALL pin_table(" + sirius::test::sql_literal(customer.string()) +
         ", tier='gpu', name='lm_comp_customer');");
  // Identity blocks on the payloads barely shrink the chunk, so the savings gate is what decides
  // whether it stays compressed at all.
  REQUIRE(sirius::test::census_entry(*con, "lm_comp_customer").compressed_chunks > 0);

  for (bool const fused : {false, true}) {
    CAPTURE(fused);
    run_ok(std::string("SET enable_fused_scan_filter = ") + (fused ? "true;" : "false;"));
    // Without the residual conjunct the whole filter can ride the decode, so a fused decode hands
    // on a compacted batch nothing filters again and the rowid comes from its positions alone.
    // With it, the two stages report positions in different coordinate systems and must be
    // composed. A query that deferred nothing returns the same answer, so the count is what makes
    // each comparison mean anything.
    for (bool const residual : {false, true}) {
      CAPTURE(residual);
      auto const before = sirius::late_mat::deferrals_installed();
      auto result       = query_on_gpu(*con, query_for(customer, orders, lines, residual));
      REQUIRE(rows_of(*result) == expected[residual]);
      REQUIRE(sirius::late_mat::deferrals_installed() > before);
    }
  }

  run_ok("CALL unpin_table('lm_comp_customer');");
}
