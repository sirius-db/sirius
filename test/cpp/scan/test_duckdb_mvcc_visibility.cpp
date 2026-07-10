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

// Gates for the duckdb MVCC visibility walk: capture must validate the
// pinned prefix against the live row-group tree and flag version state; the
// bit-packed fill must reproduce DuckDB's own row visibility exactly — the
// oracle is `SELECT rowid` on the same connection/transaction — across
// vector, row-group, AND 32-bit word boundaries, honoring the query's own
// uncommitted deletes and the pin-coverage clamp. CPU-only: real file-backed
// DuckDB databases, no GPU.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/transaction/duck_transaction.hpp>
#include <op/scan/duckdb_mvcc_visibility.hpp>
#include <scan_manager/duckdb_mvcc_metadata.hpp>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

constexpr std::size_t kRowGroupRows = 122880;  // DuckDB default row-group size

void exec_ok(duckdb::Connection& con, const std::string& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

/// File-backed database (checkpointing and persisted tombstones need real
/// storage), unique per instantiation, removed on teardown.
struct vis_test_db {
  std::string path;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

  vis_test_db()
  {
    static int counter = 0;
    path               = "/tmp/sirius_mvcc_visibility_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + ".db";
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
    open();
  }

  ~vis_test_db()
  {
    con.reset();
    db.reset();
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
  }

  void open()
  {
    db  = std::make_unique<duckdb::DuckDB>(path);
    con = std::make_unique<duckdb::Connection>(*db);
  }

  /// Close and reopen so persisted state (e.g. checkpointed tombstones) comes
  /// back UNLOADED.
  void reopen()
  {
    con.reset();
    db.reset();
    open();
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

/// Metadata over the table's CURRENT committed rows: v_base = the open
/// transaction's start_time; chunks of @p rgs_per_chunk whole row groups
/// (mirroring the coalescer's whole-row-group batches).
scan_manager::duckdb_mvcc_metadata make_metadata(duckdb::Connection& con,
                                                 duckdb::DataTable& storage,
                                                 std::size_t rgs_per_chunk)
{
  scan_manager::duckdb_mvcc_metadata metadata;
  metadata.v_base = duckdb::DuckTransaction::Get(*con.context, storage.GetAttached()).start_time;

  auto const total = static_cast<std::size_t>(storage.GetTotalRows());
  std::size_t off  = 0;
  while (off < total) {
    std::size_t chunk_rows = 0;
    for (std::size_t g = 0; g < rgs_per_chunk && off < total; ++g) {
      auto const rg_rows = std::min(kRowGroupRows, total - off);
      chunk_rows += rg_rows;
      off += rg_rows;
    }
    metadata.base_row_count_per_chunk.push_back(chunk_rows);
  }
  return metadata;
}

/// Oracle: rowids visible to @p con's current transaction, as a keep flag per
/// rowid in [0, n).
std::vector<bool> visible_rowids(duckdb::Connection& con, const std::string& table, std::size_t n)
{
  std::vector<bool> visible(n, false);
  auto result = con.Query("SELECT rowid FROM " + table);
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  while (auto chunk = result->Fetch()) {
    auto const* rowids = duckdb::FlatVector::GetData<int64_t>(chunk->data[0]);
    for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
      auto const r = rowids[i];
      if (r >= 0 && static_cast<std::size_t>(r) < n) {
        visible[static_cast<std::size_t>(r)] = true;
      }
    }
  }
  return visible;
}

bool bit_at(std::vector<std::uint32_t> const& words, std::size_t i)
{
  return ((words[i / 32] >> (i % 32)) & 1u) != 0;
}

/// Run capture + per-chunk fill and REQUIRE the produced bits equal the
/// SQL-visible oracle over the covered prefix. Returns per-chunk any-dropped.
std::vector<bool> capture_fill_and_check(duckdb::Connection& con,
                                         duckdb::DataTable& storage,
                                         scan_manager::duckdb_mvcc_metadata const& metadata,
                                         const std::string& table)
{
  auto const n_cache = metadata.n_cache();
  auto plan          = capture_mvcc_visibility_plan(storage, *con.context, metadata);
  REQUIRE(plan.mvcc_row_groups.size() == metadata.base_row_count_per_chunk.size());

  auto const oracle = visible_rowids(con, table, n_cache);

  std::vector<bool> dropped_per_chunk;
  std::size_t chunk_start = 0;
  for (std::size_t c = 0; c < plan.mvcc_row_groups.size(); ++c) {
    auto const rows = metadata.base_row_count_per_chunk[c];
    std::vector<std::uint32_t> words((rows + 31) / 32, 0xDEADBEEFu);  // poison: fill must cover all
    auto const dropped =
      fill_keep_mask_for_row_groups(plan.mvcc_row_groups[c], plan.transaction, words);
    dropped_per_chunk.push_back(dropped);
    for (std::size_t r = 0; r < rows; ++r) {
      if (bit_at(words, r) != static_cast<bool>(oracle[chunk_start + r])) {
        INFO("chunk " << c << " row " << r << " (rowid " << chunk_start + r << "): mask says "
                      << bit_at(words, r) << ", oracle says " << oracle[chunk_start + r]);
        REQUIRE(bit_at(words, r) == static_cast<bool>(oracle[chunk_start + r]));
      }
    }
    chunk_start += rows;
  }
  return dropped_per_chunk;
}

}  // namespace

TEST_CASE("mvcc visibility: clean table is all-visible with zero version state",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000)");
  exec_ok(*env.con, "CHECKPOINT");
  // A RowVersionManager created by this session's writes stays attached for
  // the process lifetime (CleanupAppend clears vectors, never the manager),
  // so same-session-written row groups read as conservatively dirty. Reopen:
  // a freshly-loaded table — the canonical ATTACH-then-pin serving shape — is
  // where the provably-clean fast path applies.
  env.reopen();

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  auto metadata = make_metadata(*env.con, storage, 1);
  REQUIRE(metadata.base_row_count_per_chunk.size() == 3);  // 3 row groups at 300k rows

  auto plan = capture_mvcc_visibility_plan(storage, *env.con->context, metadata);
  REQUIRE_FALSE(plan.any_version_state());  // L0: the job would be a complete no-op

  auto dropped = capture_fill_and_check(*env.con, storage, metadata, "t");
  for (auto d : dropped) {
    REQUIRE_FALSE(static_cast<bool>(d));
  }
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("mvcc visibility: deletes straddling vector, row-group, and word boundaries",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000)");
  exec_ok(*env.con, "CHECKPOINT");
  // Word 0/1 boundary (31,32), vector 0/1 boundary (2047,2048), row-group 0/1
  // boundary (122879,122880,122881), plus interior and tail rows.
  exec_ok(*env.con,
          "DELETE FROM t WHERE rowid IN "
          "(0, 31, 32, 63, 2047, 2048, 122879, 122880, 122881, 200000, 299999)");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");

  SECTION("one row group per chunk")
  {
    auto metadata = make_metadata(*env.con, storage, 1);
    auto plan     = capture_mvcc_visibility_plan(storage, *env.con->context, metadata);
    REQUIRE(plan.any_version_state());
    auto dropped = capture_fill_and_check(*env.con, storage, metadata, "t");
    REQUIRE(dropped == std::vector<bool>{true, true, true});
  }

  SECTION("two row groups per chunk (multi-row-group slice offsets)")
  {
    auto metadata = make_metadata(*env.con, storage, 2);
    REQUIRE(metadata.base_row_count_per_chunk.size() == 2);
    auto dropped = capture_fill_and_check(*env.con, storage, metadata, "t");
    REQUIRE(dropped == std::vector<bool>{true, true});
  }
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("mvcc visibility: fully-deleted vector and untouched sibling chunks",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000)");
  exec_ok(*env.con, "CHECKPOINT");
  env.reopen();  // start from a freshly-loaded (probe-clean) table
  exec_ok(*env.con, "DELETE FROM t WHERE rowid >= 4096 AND rowid < 6144");  // all of vector 2, RG 0

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  auto metadata = make_metadata(*env.con, storage, 1);
  auto plan     = capture_mvcc_visibility_plan(storage, *env.con->context, metadata);
  auto dropped  = capture_fill_and_check(*env.con, storage, metadata, "t");
  REQUIRE(static_cast<bool>(dropped[0]));  // the deleted vector's chunk
  // Chunks 1 and 2 are untouched: no drops, and their version-state flags are
  // clean so the job would never allocate masks for them.
  REQUIRE_FALSE(static_cast<bool>(dropped[1]));
  REQUIRE_FALSE(static_cast<bool>(dropped[2]));
  REQUIRE_FALSE(static_cast<bool>(plan.chunk_has_version_state[1]));
  REQUIRE_FALSE(static_cast<bool>(plan.chunk_has_version_state[2]));
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("mvcc visibility: own-transaction deletes are honored and rollback restores",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  exec_ok(*env.con, "DELETE FROM t WHERE rowid < 100");  // uncommitted, this transaction
  auto& storage = resolve_storage(*env.con, "t");
  auto metadata = make_metadata(*env.con, storage, 1);
  auto dropped  = capture_fill_and_check(*env.con, storage, metadata, "t");
  REQUIRE(static_cast<bool>(dropped[0]));
  exec_ok(*env.con, "ROLLBACK");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage2 = resolve_storage(*env.con, "t");
  auto metadata2 = make_metadata(*env.con, storage2, 1);
  auto dropped2  = capture_fill_and_check(*env.con, storage2, metadata2, "t");
  REQUIRE_FALSE(static_cast<bool>(dropped2[0]));  // rollback restored every row
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("mvcc visibility: pin-coverage clamp under post-metadata appends",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(150000)");
  exec_ok(*env.con, "CHECKPOINT");

  // Metadata frozen at 150,000 rows (the "pin"), then more rows land: the
  // last covered row group's live count grows past its pin-time coverage.
  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage      = resolve_storage(*env.con, "t");
  auto metadata      = make_metadata(*env.con, storage, 1);
  auto const n_cache = metadata.n_cache();
  REQUIRE(n_cache == 150000);
  exec_ok(*env.con, "COMMIT");

  exec_ok(*env.con, "INSERT INTO t SELECT range::INTEGER FROM range(50000)");
  exec_ok(*env.con, "DELETE FROM t WHERE rowid IN (140000, 149999)");  // inside the covered prefix

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage2      = resolve_storage(*env.con, "t");
  auto plan           = capture_mvcc_visibility_plan(storage2, *env.con->context, metadata);
  std::size_t covered = 0;
  for (auto const& chunk : plan.mvcc_row_groups) {
    for (auto const& slice : chunk) {
      covered += slice.row_count;
    }
  }
  REQUIRE(covered == n_cache);  // clamped: appended rows are outside the pin
  capture_fill_and_check(*env.con, storage2, metadata, "t");
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("mvcc visibility: capture validation throws on impossible states",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000)");
  exec_ok(*env.con, "CHECKPOINT");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");

  SECTION("query snapshot older than the pin snapshot")
  {
    auto metadata   = make_metadata(*env.con, storage, 1);
    metadata.v_base = metadata.v_base + 1000;  // "pin" from the future = re-pin race
    REQUIRE_THROWS_AS(capture_mvcc_visibility_plan(storage, *env.con->context, metadata),
                      std::runtime_error);
  }

  SECTION("chunk boundary off a row-group boundary")
  {
    scan_manager::duckdb_mvcc_metadata metadata;
    metadata.v_base =
      duckdb::DuckTransaction::Get(*env.con->context, storage.GetAttached()).start_time;
    metadata.base_row_count_per_chunk = {100, 299900};  // 100 is mid-row-group
    REQUIRE_THROWS_AS(capture_mvcc_visibility_plan(storage, *env.con->context, metadata),
                      std::runtime_error);
  }

  SECTION("pinned prefix beyond the live rows")
  {
    scan_manager::duckdb_mvcc_metadata metadata;
    metadata.v_base =
      duckdb::DuckTransaction::Get(*env.con->context, storage.GetAttached()).start_time;
    metadata.base_row_count_per_chunk = {400000};  // table only has 300000
    REQUIRE_THROWS_AS(capture_mvcc_visibility_plan(storage, *env.con->context, metadata),
                      std::runtime_error);
  }
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("mvcc visibility: persisted tombstones load through a reopened database",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(300000)");
  exec_ok(*env.con, "CHECKPOINT");
  exec_ok(*env.con, "DELETE FROM t WHERE rowid % 1000 = 7");  // sparse: survives the checkpoint
  exec_ok(*env.con, "CHECKPOINT");                            // tombstones persist to disk
  env.reopen();                                               // ... and come back UNLOADED

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  REQUIRE(has_any_version_state(storage, static_cast<std::size_t>(storage.GetTotalRows())));

  auto metadata = make_metadata(*env.con, storage, 1);
  auto plan     = capture_mvcc_visibility_plan(storage, *env.con->context, metadata);
  REQUIRE(plan.any_version_state());
  // The fill's GetSelVector lazily loads the persisted deletes and must mask
  // exactly the tombstoned rows.
  auto dropped = capture_fill_and_check(*env.con, storage, metadata, "t");
  REQUIRE(dropped == std::vector<bool>{true, true, true});
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("mvcc visibility: has_any_version_state flags deletes, clean tables pass",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");
  env.reopen();  // freshly-loaded table: the provably-clean baseline

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage    = resolve_storage(*env.con, "t");
  auto const total = static_cast<std::size_t>(storage.GetTotalRows());
  REQUIRE_FALSE(has_any_version_state(storage, total));
  exec_ok(*env.con, "COMMIT");

  SECTION("uncommitted delete")
  {
    exec_ok(*env.con, "BEGIN TRANSACTION");
    exec_ok(*env.con, "DELETE FROM t WHERE rowid = 5");
    auto& s = resolve_storage(*env.con, "t");
    REQUIRE(has_any_version_state(s, total));
    exec_ok(*env.con, "ROLLBACK");
  }

  SECTION("committed delete")
  {
    exec_ok(*env.con, "DELETE FROM t WHERE rowid = 5");
    exec_ok(*env.con, "BEGIN TRANSACTION");
    auto& s = resolve_storage(*env.con, "t");
    REQUIRE(has_any_version_state(s, total));
    exec_ok(*env.con, "ROLLBACK");
  }
}

TEST_CASE("mvcc visibility: any_update_chains flags updated columns only",
          "[duckdb_mvcc_visibility][scan]")
{
  vis_test_db env;
  exec_ok(*env.con,
          "CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");

  std::vector<duckdb::storage_t> const col_k{0};
  std::vector<duckdb::storage_t> const col_v{1};
  std::vector<duckdb::storage_t> const both{0, 1};

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage    = resolve_storage(*env.con, "t");
  auto const total = static_cast<std::size_t>(storage.GetTotalRows());
  REQUIRE_FALSE(any_update_chains(storage, both, total));
  exec_ok(*env.con, "COMMIT");

  exec_ok(*env.con, "UPDATE t SET v = v + 1 WHERE rowid < 10");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& s = resolve_storage(*env.con, "t");
  REQUIRE(any_update_chains(s, col_v, total));        // updated column flagged
  REQUIRE(any_update_chains(s, both, total));         // any-of semantics
  REQUIRE_FALSE(any_update_chains(s, col_k, total));  // untouched column passes
  exec_ok(*env.con, "ROLLBACK");

  exec_ok(*env.con, "CHECKPOINT");  // folds the update chain into the base data
  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& s2 = resolve_storage(*env.con, "t");
  REQUIRE_FALSE(any_update_chains(s2, both, total));
  exec_ok(*env.con, "ROLLBACK");
}
