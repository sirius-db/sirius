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
#include <utility>
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
/// bitpack-rooted plan can answer inside the decode. It keeps 5% of the rows,
/// under the full-route compaction cutoff, and none of them sit at the front
/// of the chunk, so a rowid built from the residual's positions alone would
/// name rows the decode dropped.
constexpr char const* kQuery =
  "SELECT c.c_custkey, c.c_name, c.c_address, c.c_comment, count(*) AS n "
  "FROM read_parquet({C}) c "
  "JOIN read_parquet({O}) o ON c.c_custkey = o.o_custkey "
  "JOIN read_parquet({L}) l ON o.o_orderkey = l.l_orderkey "
  "WHERE c.c_custkey >= 5000 AND c.c_custkey < 6000 {X}"
  "GROUP BY c.c_custkey, c.c_name, c.c_address, c.c_comment "
  "ORDER BY c.c_custkey";

/// A conjunct on an identity-planned string column, which no decode route can
/// answer. It leaves the range pushed down but the request no longer covering
/// the whole filter, so the decode compacts and the residual filter then runs
/// over what it kept — the two stages whose positions have to be composed.
constexpr char const* kResidualConjunct = "AND c.c_name LIKE '%7' ";

/// The same shape with a second pinned table joined on a key it holds once per
/// row, so its wide name can ride along with the customer bundle.
constexpr char const* kRiderQuery =
  "SELECT c.c_custkey, c.c_name, c.c_address, c.c_comment, n.n_name, count(*) AS n "
  "FROM read_parquet({C}) c "
  "JOIN read_parquet({N}) n ON c.c_nationkey = n.n_nationkey "
  "JOIN read_parquet({O}) o ON c.c_custkey = o.o_custkey "
  "JOIN read_parquet({L}) l ON o.o_orderkey = l.l_orderkey "
  "WHERE c.c_custkey >= 5000 AND c.c_custkey < 6000 "
  "GROUP BY c.c_custkey, c.c_name, c.c_address, c.c_comment, n.n_name "
  "ORDER BY c.c_custkey";

constexpr char const* kBitpack  = "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed";
constexpr char const* kIdentity = "input -> identity";

void substitute(std::string& sql, std::string_view token, std::string const& value)
{
  sql.replace(sql.find(token), token.size(), value);
}

std::string with_paths(std::string sql, std::vector<std::pair<char const*, fs::path>> const& paths)
{
  for (auto const& [token, path] : paths) {
    substitute(sql, token, sirius::test::sql_literal(path.string()));
  }
  return sql;
}

std::string query_for(fs::path const& customer,
                      fs::path const& orders,
                      fs::path const& lines,
                      bool residual)
{
  std::string sql = kQuery;
  substitute(sql, "{X}", residual ? kResidualConjunct : "");
  return with_paths(std::move(sql), {{"{C}", customer}, {"{O}", orders}, {"{L}", lines}});
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

std::vector<std::string> cpu_rows(duckdb::Connection& con, std::string const& sql)
{
  auto cpu = con.Query(sql);
  REQUIRE(cpu);
  if (cpu->HasError()) { UNSCOPED_INFO(cpu->GetError()); }
  REQUIRE_FALSE(cpu->HasError());
  auto rows = rows_of(*cpu);
  REQUIRE_FALSE(rows.empty());
  return rows;
}

/// One plan block per column, in the file's column order.
void write_plan_file(fs::path const& plan_dir,
                     std::string const& table,
                     std::vector<char const*> const& blocks)
{
  std::ofstream f(plan_dir / (table + ".txt"));
  REQUIRE(f.good());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    if (i != 0) { f << "---\n"; }
    f << blocks[i] << "\n";
  }
}

/// Compression settings under which identity-planned string payloads still stay compressed.
/// Plan files are matched by stem, so they must not share a directory with the Parquet input.
void enable_pin_compression(sirius::test::GpuExecutionFixture& fx, fs::path const& plan_dir)
{
  fx.run_ok("SET pin_table_compression = true;");
  fx.run_ok("SET pin_table_compression_min_batch_size_bytes = 0;");
  fx.run_ok("SET pin_table_compression_max_compressed_fraction = 1.5;");
  fx.run_ok("SET pin_table_input_compression_plan_dir = " +
            sirius::test::sql_literal(plan_dir.string()) + ";");
}

void pin_compressed(sirius::test::GpuExecutionFixture& fx,
                    fs::path const& path,
                    std::string const& name)
{
  fx.run_ok("CALL pin_table(" + sirius::test::sql_literal(path.string()) + ", tier='gpu', name='" +
            name + "');");
  REQUIRE(sirius::test::census_entry(*fx.con, name).compressed_chunks > 0);
}

struct tables {
  fs::path customer;
  fs::path nation;
  fs::path orders;
  fs::path lines;
};

/// Written with Sirius off, so the CPU answers below are DuckDB's own.
void write_tables(sirius::test::GpuExecutionFixture& fx, tables const& t)
{
  fx.run_ok(
    "COPY (SELECT range AS c_custkey, range % 25 AS c_nationkey, "
    "'Customer#' || lpad(CAST(range AS VARCHAR), 12, '0') AS c_name, "
    "'address-' || repeat(CAST(range % 97 AS VARCHAR), 6) AS c_address, "
    "'comment for customer ' || CAST(range AS VARCHAR) AS c_comment "
    "FROM range(" +
    std::to_string(kCustomers) + ")) TO " + sirius::test::sql_literal(t.customer.string()) +
    " (FORMAT PARQUET);");
  fx.run_ok(
    "COPY (SELECT range AS n_nationkey, 'NATION-' || lpad(CAST(range AS VARCHAR), 40, 'x') AS "
    "n_name FROM range(25)) TO " +
    sirius::test::sql_literal(t.nation.string()) + " (FORMAT PARQUET);");
  fx.run_ok("COPY (SELECT range AS o_orderkey, range % " + std::to_string(kCustomers) +
            " AS o_custkey FROM range(" + std::to_string(kOrders) + ")) TO " +
            sirius::test::sql_literal(t.orders.string()) + " (FORMAT PARQUET);");
  fx.run_ok("COPY (SELECT range AS l_linekey, range % " + std::to_string(kOrders) +
            " AS l_orderkey FROM range(" + std::to_string(kLines) + ")) TO " +
            sirius::test::sql_literal(t.lines.string()) + " (FORMAT PARQUET);");
}

/// Restores process-wide settings a test changed, including when a REQUIRE unwinds it.
struct reset_on_exit {
  sirius::test::GpuExecutionFixture& fx;
  std::vector<std::string> names;
  ~reset_on_exit()
  {
    for (auto const& name : names) {
      fx.con->Query("RESET " + name + ";");
    }
  }
};

}  // namespace

TEST_CASE_METHOD(CompressedFilterFixture,
                 "a deferral over a filtered compressed pin rebuilds its rowid from both filter "
                 "stages",
                 "[integration][gpu_execution][late_mat][compression][compressed_filter]")
{
  sirius::test::scratch_dir scratch{"late_mat_compressed_filter"};
  auto const& tmp = scratch.path();
  tables const t{tmp / "lm_comp_customer.parquet",
                 tmp / "lm_comp_nation.parquet",
                 tmp / "orders.parquet",
                 tmp / "lineitem.parquet"};

  run_ok("SET gpu_execution = false;");
  write_tables(*this, t);
  // The CPU answers, taken before anything is pinned.
  std::vector<std::string> expected[2];
  for (bool const residual : {false, true}) {
    expected[residual] = cpu_rows(*con, query_for(t.customer, t.orders, t.lines, residual));
  }
  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");

  auto const plan_dir = tmp / "plans";
  fs::create_directories(plan_dir);
  // A bitpack block for the key, so its decode produces a range-answering mask — the plan shape
  // that lets the decode drop rows at all. The string payloads sit on identity and take the
  // full-decode route, compacted by the survivor gather.
  write_plan_file(
    plan_dir, "lm_comp_customer", {kBitpack, kBitpack, kIdentity, kIdentity, kIdentity});
  enable_pin_compression(*this, plan_dir);
  // c_custkey rides real as the join and group key; the pin proves it distinct, which is what
  // moves the port past the aggregate.
  pin_compressed(*this, t.customer, "lm_comp_customer");

  for (bool const fused : {false, true}) {
    CAPTURE(fused);
    run_ok(std::string("SET enable_fused_scan_filter = ") + (fused ? "true;" : "false;"));
    // Without the residual conjunct the whole filter can ride the decode, so a fused decode hands
    // on a compacted batch nothing filters again and the rowid comes from its positions alone.
    // With it, the two stages report positions in different coordinate systems and must be
    // composed. A query that deferred nothing, or whose decode never compacted, returns the same
    // answer, so the counts are what make each comparison mean anything.
    for (bool const residual : {false, true}) {
      CAPTURE(residual);
      auto const deferred_before  = sirius::late_mat::deferrals_installed();
      auto const survivors_before = sirius::late_mat::survivor_batches();
      auto result = query_on_gpu(*con, query_for(t.customer, t.orders, t.lines, residual));
      REQUIRE(rows_of(*result) == expected[residual]);
      REQUIRE(sirius::late_mat::deferrals_installed() > deferred_before);
      auto const survivors = sirius::late_mat::survivor_batches();
      if (!fused) {
        REQUIRE(survivors.decode_reported == survivors_before.decode_reported);
        continue;
      }
      REQUIRE(survivors.decode_reported > survivors_before.decode_reported);
      if (residual) {
        REQUIRE(survivors.composed > survivors_before.composed);
      } else {
        REQUIRE(survivors.composed == survivors_before.composed);
      }
    }
  }

  // The nation name rides along with the customer bundle, so both compressed pins are addressed
  // by one rowid each at the same port.
  write_plan_file(plan_dir, "lm_comp_nation", {kBitpack, kIdentity});
  pin_compressed(*this, t.nation, "lm_comp_nation");
  auto const rider_sql = with_paths(
    kRiderQuery, {{"{C}", t.customer}, {"{N}", t.nation}, {"{O}", t.orders}, {"{L}", t.lines}});
  run_ok("SET gpu_execution = false;");
  auto const rider_expected = cpu_rows(*con, rider_sql);
  run_ok("SET gpu_execution = true;");
  for (bool const fused : {false, true}) {
    CAPTURE(fused);
    run_ok(std::string("SET enable_fused_scan_filter = ") + (fused ? "true;" : "false;"));
    auto const before = sirius::late_mat::deferrals_installed();
    auto result       = query_on_gpu(*con, rider_sql);
    REQUIRE(rows_of(*result) == rider_expected);
    REQUIRE(sirius::late_mat::deferrals_installed() >= before + 2);
  }

  run_ok("CALL unpin_table('lm_comp_nation');");
  run_ok("CALL unpin_table('lm_comp_customer');");
}

TEST_CASE_METHOD(CompressedFilterFixture,
                 "a deferral is declined when a column's compressed chunks narrowed to different "
                 "carriers",
                 "[integration][gpu_execution][late_mat][compression][compressed_filter]")
{
  // Pinning narrows each chunk from its own range, so c_mixed is INT8 in the first row group and
  // INT32 in the rest. The port reads every chunk of a deferred column at one width, so admitting
  // this column would fail only after the scan had given its values up.
  constexpr std::int64_t kMixedCustomers = 8'192;
  sirius::test::scratch_dir scratch{"late_mat_compressed_mixed_carrier"};
  auto const& tmp     = scratch.path();
  auto const customer = tmp / "lm_mixed_customer.parquet";
  auto const orders   = tmp / "orders.parquet";
  auto const lines    = tmp / "lineitem.parquet";

  reset_on_exit const restore{*this, {"threads", "scan_task_batch_size"}};
  run_ok("SET gpu_execution = false;");
  run_ok(
    "COPY (SELECT range AS c_custkey, "
    "'Customer#' || lpad(CAST(range AS VARCHAR), 12, '0') AS c_name, "
    "'address-' || repeat(CAST(range % 97 AS VARCHAR), 6) AS c_address, "
    "'comment for customer ' || CAST(range AS VARCHAR) AS c_comment, "
    "CASE WHEN range < 2048 THEN range % 100 ELSE 40000 + range % 1000 END AS c_mixed "
    "FROM range(" +
    std::to_string(kMixedCustomers) + ")) TO " + sirius::test::sql_literal(customer.string()) +
    " (FORMAT PARQUET, ROW_GROUP_SIZE 2048);");
  run_ok("COPY (SELECT range AS o_orderkey, range % " + std::to_string(kMixedCustomers) +
         " AS o_custkey FROM range(" + std::to_string(3 * kMixedCustomers) + ")) TO " +
         sirius::test::sql_literal(orders.string()) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048);");
  run_ok("COPY (SELECT range AS l_linekey, range % " + std::to_string(3 * kMixedCustomers) +
         " AS l_orderkey FROM range(" + std::to_string(6 * kMixedCustomers) + ")) TO " +
         sirius::test::sql_literal(lines.string()) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048);");
  auto const sql = with_paths(
    "SELECT c.c_custkey, c.c_name, c.c_address, c.c_comment, c.c_mixed, count(*) AS n "
    "FROM read_parquet({C}) c "
    "JOIN read_parquet({O}) o ON c.c_custkey = o.o_custkey "
    "JOIN read_parquet({L}) l ON o.o_orderkey = l.l_orderkey "
    "WHERE c.c_custkey >= 0 "
    "GROUP BY c.c_custkey, c.c_name, c.c_address, c.c_comment, c.c_mixed "
    "ORDER BY c.c_custkey",
    {{"{C}", customer}, {"{O}", orders}, {"{L}", lines}});
  auto const expected = cpu_rows(*con, sql);

  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = false;");
  run_ok("SET threads = 1;");
  run_ok("SET scan_task_batch_size = 8192;");
  run_ok("SET enable_compressed_materialization = true;");
  auto const plan_dir = tmp / "plans";
  fs::create_directories(plan_dir);
  write_plan_file(
    plan_dir, "lm_mixed_customer", {kBitpack, kIdentity, kIdentity, kIdentity, kBitpack});
  enable_pin_compression(*this, plan_dir);
  pin_compressed(*this, customer, "lm_mixed_customer");
  REQUIRE(sirius::test::census_entry(*con, "lm_mixed_customer").compressed_chunks > 1);

  for (bool const late_mat : {false, true}) {
    CAPTURE(late_mat);
    run_ok(std::string("SET enable_late_materialization = ") + (late_mat ? "true;" : "false;"));
    auto result = query_on_gpu(*con, sql);
    REQUIRE(rows_of(*result) == expected);
  }

  run_ok("RESET enable_late_materialization;");
  run_ok("CALL unpin_table('lm_mixed_customer');");
}
