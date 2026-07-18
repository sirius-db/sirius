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

// Unit tests for the insert-delta capture and its task bodies: boundary math,
// transient vs bulk-flushed persistent segment lanes, snapshot visibility
// counting, and byte-exact staging. CPU-only, against real file-backed DuckDB
// databases. End-to-end coverage lives in test_pin_table_mvcc_insert.cpp.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <op/scan/duckdb_insert_delta.hpp>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

void exec_ok(duckdb::Connection& con, const std::string& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

struct delta_test_db {
  std::string path;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

  delta_test_db()
  {
    static int counter = 0;
    path               = "/tmp/sirius_insert_delta_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + ".db";
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
    db  = std::make_unique<duckdb::DuckDB>(path);
    con = std::make_unique<duckdb::Connection>(*db);
    con->Query("SET gpu_execution = false;");
  }

  ~delta_test_db()
  {
    con.reset();
    db.reset();
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
  }
};

/// Requires an active transaction on @p con (catalog access needs one).
duckdb::DataTable& resolve_storage(duckdb::Connection& con, const std::string& table_name)
{
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

/// build_appended_table layout: 130,000 checkpointed rows (n_cache) in row
/// groups {122880, 7120}, plus 5,000 committed small-append rows. The append
/// opens a fresh row group, so the delta is row group #2 with k_offset == 0.
constexpr std::size_t kBase  = 130000;
constexpr std::size_t kDelta = 5000;

void build_appended_table(duckdb::Connection& con)
{
  // Single-threaded CTAS keeps the row-group layout deterministic; the
  // assertions hardcode the sequential {122880, 7120} split.
  exec_ok(con, "SET threads TO 1");
  exec_ok(con,
          "CREATE TABLE t AS SELECT range::INTEGER AS k, (range*10)::BIGINT AS v, "
          "'row_'||range AS s FROM range(130000)");
  exec_ok(con, "CHECKPOINT");
  exec_ok(con,
          "INSERT INTO t SELECT (130000+range)::INTEGER, ((130000+range)*10)::BIGINT, "
          "'row_'||(130000+range) FROM range(5000)");
}

std::vector<duckdb::storage_t> const kAllCols{0, 1, 2};

std::vector<sirius::logical_type> all_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::BIGINT),
          sirius::logical_type::make(sirius::type_id::VARCHAR)};
}

}  // namespace

TEST_CASE("insert delta: capture boundary math and transient segment lanes",
          "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  build_appended_table(*env.con);

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  auto types    = all_types();
  auto plan     = capture_insert_delta_plan(storage, *env.con->context, kBase, kAllCols, types);

  REQUIRE(plan.n_cache == kBase);
  REQUIRE(plan.n_total == kBase + kDelta);
  REQUIRE(plan.delta_rows() == kDelta);
  REQUIRE(plan.buffer_manager != nullptr);
  REQUIRE(plan.partition_stats != nullptr);
  REQUIRE(plan.transaction.transaction_id != 0);

  REQUIRE(plan.row_groups.size() == 1);
  auto const& rg = plan.row_groups[0];
  REQUIRE(rg.row_group_index == 2);  // the append opened a fresh row group
  REQUIRE(rg.k_offset == 0);
  REQUIRE(rg.row_count == kDelta);
  REQUIRE(rg.row_group_start == kBase);
  REQUIRE(rg.columns.size() == 3);

  // Fixed-width columns: one transient data segment, rebased to slice start,
  // sized exactly rows * type_size.
  auto const& col_k = rg.columns[0];
  REQUIRE(col_k.data_segments.size() == 1);
  REQUIRE(col_k.data_segments[0].is_transient);
  REQUIRE(col_k.data_segments[0].segment_start == 0);
  REQUIRE(col_k.data_segments[0].segment_count == kDelta);
  REQUIRE(col_k.data_segments[0].bytes_size == kDelta * sizeof(int32_t));
  REQUIRE_FALSE(col_k.validity_segments.empty());

  auto const& col_v = rg.columns[1];
  REQUIRE(col_v.data_segments.size() == 1);
  REQUIRE(col_v.data_segments[0].bytes_size == kDelta * sizeof(int64_t));

  // Varchar: the whole used extent is staged and max_string_length is filled.
  auto const& col_s = rg.columns[2];
  REQUIRE_FALSE(col_s.data_segments.empty());
  for (auto const& s : col_s.data_segments) {
    REQUIRE(s.is_transient);
    REQUIRE(s.bytes_size > 0);
    REQUIRE(s.max_string_length.has_value());
  }

  REQUIRE(rg.transient_staging_bytes > 0);
  REQUIRE(rg.decoded_bytes_budget > 0);
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: visibility counting matches the rowid oracle",
          "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  build_appended_table(*env.con);
  // Deletes in the delta (dropped by the count) and in the base (ignored).
  exec_ok(*env.con, "DELETE FROM t WHERE k IN (5, 130000, 130001, 132048, 134999)");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  auto types    = all_types();
  auto plan     = capture_insert_delta_plan(storage, *env.con->context, kBase, kAllCols, types);
  REQUIRE(plan.row_groups.size() == 1);
  auto const& rg = plan.row_groups[0];

  std::vector<std::uint32_t> words((kDelta + 31) / 32, 0u);
  auto const visible = count_visible_delta_rows(rg, plan.transaction, words, 0);
  REQUIRE(visible == kDelta - 4);  // the base delete does not count

  for (std::size_t i = 0; i < kDelta; ++i) {
    auto const rowid   = kBase + i;
    bool const deleted = (rowid == 130000 || rowid == 130001 || rowid == 132048 || rowid == 134999);
    bool const kept    = ((words[i / 32] >> (i % 32)) & 1u) != 0;
    if (kept != !deleted) {
      INFO("delta rowid " << rowid);
      REQUIRE(kept == !deleted);
    }
  }
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: an older snapshot cannot see newer committed rows",
          "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  build_appended_table(*env.con);

  // Open the snapshot first, then commit more rows from a second connection.
  // The transaction spawns lazily, so touch the table to pin the snapshot.
  exec_ok(*env.con, "BEGIN TRANSACTION");
  exec_ok(*env.con, "SELECT count(*) FROM t");
  duckdb::Connection con2(*env.db);
  exec_ok(con2, "INSERT INTO t SELECT (135000+range)::INTEGER, 0::BIGINT, 'late' FROM range(1000)");

  auto& storage = resolve_storage(*env.con, "t");
  auto types    = all_types();
  auto plan     = capture_insert_delta_plan(storage, *env.con->context, kBase, kAllCols, types);
  // The snapshot walk still sees the newer physical rows...
  REQUIRE(plan.n_total == kBase + kDelta + 1000);

  std::size_t total_visible = 0;
  std::vector<std::uint32_t> words((plan.delta_rows() + 31) / 32, 0u);
  std::size_t bit_offset = 0;
  for (auto const& rg : plan.row_groups) {
    total_visible += count_visible_delta_rows(rg, plan.transaction, words, bit_offset);
    bit_offset += rg.row_count;
  }
  // ...but visibility excludes rows committed after this snapshot opened.
  REQUIRE(total_visible == kDelta);
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: transient staging copies are byte-exact", "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  build_appended_table(*env.con);

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  auto types    = all_types();
  auto plan     = capture_insert_delta_plan(storage, *env.con->context, kBase, kAllCols, types);
  REQUIRE(plan.row_groups.size() == 1);
  auto const& rg = plan.row_groups[0];

  std::vector<std::uint8_t> slab(rg.transient_staging_bytes, 0);
  copy_delta_row_group(rg, *plan.buffer_manager, slab.data());

  // k column: int32 values 130000..134999 at the segment's slab offset.
  auto const& kseg = rg.columns[0].data_segments[0];
  std::vector<int32_t> expected_k(kDelta);
  for (std::size_t i = 0; i < kDelta; ++i) {
    expected_k[i] = static_cast<int32_t>(kBase + i);
  }
  REQUIRE(
    std::memcmp(slab.data() + kseg.slab_offset, expected_k.data(), kDelta * sizeof(int32_t)) == 0);

  // v column: bigint values (rowid * 10).
  auto const& vseg = rg.columns[1].data_segments[0];
  std::vector<int64_t> expected_v(kDelta);
  for (std::size_t i = 0; i < kDelta; ++i) {
    expected_v[i] = static_cast<int64_t>((kBase + i) * 10);
  }
  REQUIRE(
    std::memcmp(slab.data() + vseg.slab_offset, expected_v.data(), kDelta * sizeof(int64_t)) == 0);

  // varchar: the staged dictionary header {size, end} must be self-consistent
  // with the staged extent.
  auto const& sseg        = rg.columns[2].data_segments[0];
  std::uint32_t dict_size = 0;
  std::uint32_t dict_end  = 0;
  std::memcpy(&dict_size, slab.data() + sseg.slab_offset, sizeof(dict_size));
  std::memcpy(&dict_end, slab.data() + sseg.slab_offset + 4, sizeof(dict_end));
  REQUIRE(dict_end <= sseg.bytes_size);
  REQUIRE(dict_size <= dict_end);
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: bulk-flushed appends keep persistent block references",
          "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");
  // A single commit of at least one full row group: MergeStorage flushes it as
  // persistent, checkpoint-shaped row groups.
  exec_ok(*env.con, "INSERT INTO t SELECT (10000+range)::INTEGER FROM range(130000)");
  // And a small committed append on top: transient, in its own row group.
  exec_ok(*env.con, "INSERT INTO t VALUES (140000)");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  std::vector<duckdb::storage_t> const cols{0};
  std::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto plan = capture_insert_delta_plan(storage, *env.con->context, 10000, cols, types);

  REQUIRE(plan.n_total == 140001);
  REQUIRE(plan.row_groups.size() >= 2);

  bool saw_persistent = false;
  bool saw_transient  = false;
  for (auto const& rg : plan.row_groups) {
    for (auto const& s : rg.columns[0].data_segments) {
      if (s.is_transient) {
        saw_transient = true;
      } else {
        saw_persistent = true;
        REQUIRE(s.block_id >= 0);
        REQUIRE(s.bytes_size > 0);
      }
    }
    if (!rg.columns[0].data_segments.empty() && !rg.columns[0].data_segments[0].is_transient) {
      REQUIRE(rg.transient_staging_bytes == 0);
    }
  }
  REQUIRE(saw_persistent);
  REQUIRE(saw_transient);
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: no delta means an empty plan", "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(1000)");
  exec_ok(*env.con, "CHECKPOINT");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  std::vector<duckdb::storage_t> const cols{0};
  std::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto plan = capture_insert_delta_plan(storage, *env.con->context, 1000, cols, types);
  REQUIRE(plan.empty());
  REQUIRE(plan.delta_rows() == 0);
  exec_ok(*env.con, "ROLLBACK");
}
