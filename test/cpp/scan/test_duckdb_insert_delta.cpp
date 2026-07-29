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
  auto entry =
    schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, duckdb::Identifier(table_name));
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

TEST_CASE("insert delta: bulk constant inserts capture blockless CONSTANT segments",
          "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  exec_ok(*env.con, "CREATE TABLE t AS SELECT range::INTEGER AS k FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");
  // A full-row-group commit of a projected literal: every flushed segment's
  // stats are constant, so DuckDB rewrites them as blockless CONSTANT
  // segments (no backing block to read metadata from).
  exec_ok(*env.con, "INSERT INTO t SELECT 7 FROM range(130000)");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  std::vector<duckdb::storage_t> const cols{0};
  std::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto plan = capture_insert_delta_plan(storage, *env.con->context, 10000, cols, types);

  REQUIRE(plan.n_total == 140000);
  bool saw_constant = false;
  for (auto const& rg : plan.row_groups) {
    std::size_t covered = 0;
    for (auto const& s : rg.columns[0].data_segments) {
      covered += s.segment_count;
      if (s.compression != duckdb::CompressionType::COMPRESSION_CONSTANT) { continue; }
      saw_constant = true;
      REQUIRE_FALSE(s.is_transient);
      REQUIRE(s.block_id < 0);
      REQUIRE(s.bytes_size == 0);
      REQUIRE(s.additional_blocks.empty());
      // The constant value decodes from the capture-time stats snapshot.
      REQUIRE(s.segment_stats != nullptr);
    }
    REQUIRE(covered == rg.row_count);  // CONSTANT segments still tile the slice
  }
  REQUIRE(saw_constant);
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: all-NULL constant validity is captured, not skipped",
          "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  exec_ok(*env.con,
          "CREATE TABLE t AS SELECT range::INTEGER AS k, range::INTEGER AS v FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");
  // Bulk all-NULL column: the flushed data segment is CONSTANT at a sentinel
  // value and the NULL-ness lives only in the CONSTANT validity segment's
  // stats. Skipping it would serve the sentinels as valid rows.
  exec_ok(*env.con,
          "INSERT INTO t SELECT (10000+range)::INTEGER, NULL::INTEGER FROM range(130000)");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  std::vector<duckdb::storage_t> const cols{0, 1};
  std::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::INTEGER),
                                          sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto plan = capture_insert_delta_plan(storage, *env.con->context, 10000, cols, types);

  bool saw_all_null = false;
  for (auto const& rg : plan.row_groups) {
    for (auto const& s : rg.columns[1].validity_segments) {
      if (!s.all_null) { continue; }
      saw_all_null = true;
      REQUIRE(s.is_validity);
      REQUIRE(s.compression == duckdb::CompressionType::COMPRESSION_CONSTANT);
      REQUIRE_FALSE(s.is_transient);
      REQUIRE(s.block_id < 0);
      REQUIRE(s.bytes_size == 0);
    }
  }
  REQUIRE(saw_all_null);
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: appends into the boundary row group are captured",
          "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  // A PRIMARY KEY makes DuckDB append into the existing tail row group
  // instead of opening a fresh one, so the delta begins inside the last
  // cached row group (k_offset > 0). The skeleton walk starts at that
  // boundary row group and must not skip past it.
  exec_ok(*env.con, "CREATE TABLE t (k INTEGER PRIMARY KEY, v INTEGER)");
  exec_ok(*env.con, "INSERT INTO t SELECT range, range FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");
  exec_ok(*env.con, "INSERT INTO t VALUES (10000, 7), (10001, 8)");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  std::vector<duckdb::storage_t> const cols{0, 1};
  std::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::INTEGER),
                                          sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto plan = capture_insert_delta_plan(storage, *env.con->context, 10000, cols, types);

  REQUIRE(plan.n_total == 10002);
  REQUIRE(plan.row_groups.size() == 1);
  auto const& rg = plan.row_groups[0];
  REQUIRE(rg.row_group_index == 0);  // the boundary row group itself
  REQUIRE(rg.k_offset == 10000);
  REQUIRE(rg.row_count == 2);
  REQUIRE(rg.row_group_start == 10000);
  REQUIRE(rg.columns.size() == 2);
  exec_ok(*env.con, "ROLLBACK");
}

TEST_CASE("insert delta: range capture merges to the serial plan", "[duckdb_insert_delta][scan]")
{
  delta_test_db env;
  exec_ok(*env.con, "SET threads TO 1");
  exec_ok(*env.con,
          "CREATE TABLE t AS SELECT range::INTEGER AS k, 'row_'||range AS s FROM range(10000)");
  exec_ok(*env.con, "CHECKPOINT");
  // Four delta row groups: three persistent from the bulk flush plus the
  // transient tail append.
  exec_ok(*env.con,
          "INSERT INTO t SELECT (10000+range)::INTEGER, 'row_'||(10000+range) FROM range(250000)");
  exec_ok(*env.con, "INSERT INTO t VALUES (260000, 'tail')");

  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");
  std::vector<duckdb::storage_t> const cols{0, 1};
  std::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::INTEGER),
                                          sirius::logical_type::make(sirius::type_id::VARCHAR)};

  auto serial = capture_insert_delta_plan(storage, *env.con->context, 10000, cols, types);
  REQUIRE(serial.row_groups.size() >= 3);

  auto plan = prepare_insert_delta_capture(storage, *env.con->context, 10000);
  REQUIRE(plan.row_groups.size() == serial.row_groups.size());
  for (auto const& rg : plan.row_groups) {
    REQUIRE(rg.columns.empty());  // skeletons carry boundaries only
  }

  // Step-3 ranges leave a short tail range, and an oversized rg_end clamps.
  std::vector<insert_delta_row_group> merged;
  for (std::size_t begin = 0; begin < plan.row_groups.size(); begin += 3) {
    auto part = capture_insert_delta_row_group_range(plan, begin, begin + 3, cols, types);
    for (auto& rg : part) {
      merged.push_back(std::move(rg));
    }
  }
  REQUIRE(merged.size() == serial.row_groups.size());
  REQUIRE(capture_insert_delta_row_group_range(
            plan, plan.row_groups.size(), plan.row_groups.size() + 5, cols, types)
            .empty());

  auto require_same_segments = [](std::vector<insert_delta_segment> const& a,
                                  std::vector<insert_delta_segment> const& b) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      REQUIRE(a[i].segment == b[i].segment);
      REQUIRE(a[i].is_transient == b[i].is_transient);
      REQUIRE(a[i].is_validity == b[i].is_validity);
      REQUIRE(a[i].all_null == b[i].all_null);
      REQUIRE((a[i].segment_stats != nullptr) == (b[i].segment_stats != nullptr));
      REQUIRE(a[i].block_id == b[i].block_id);
      REQUIRE(a[i].additional_blocks == b[i].additional_blocks);
      REQUIRE(a[i].block_offset == b[i].block_offset);
      REQUIRE(a[i].segment_start == b[i].segment_start);
      REQUIRE(a[i].segment_count == b[i].segment_count);
      REQUIRE(a[i].compression == b[i].compression);
      REQUIRE(a[i].max_string_length == b[i].max_string_length);
      REQUIRE(a[i].bytes_size == b[i].bytes_size);
      REQUIRE(a[i].copy_src_offset == b[i].copy_src_offset);
      REQUIRE(a[i].slab_offset == b[i].slab_offset);
    }
  };
  for (std::size_t r = 0; r < merged.size(); ++r) {
    auto const& a = merged[r];
    auto const& b = serial.row_groups[r];
    REQUIRE(a.row_group == b.row_group);
    REQUIRE(a.row_group_index == b.row_group_index);
    REQUIRE(a.row_group_start == b.row_group_start);
    REQUIRE(a.k_offset == b.k_offset);
    REQUIRE(a.row_count == b.row_count);
    REQUIRE(a.has_version_state == b.has_version_state);
    REQUIRE(a.transient_staging_bytes == b.transient_staging_bytes);
    REQUIRE(a.decoded_bytes_budget == b.decoded_bytes_budget);
    REQUIRE(a.varchar_bytes_per_col == b.varchar_bytes_per_col);
    REQUIRE(a.columns.size() == b.columns.size());
    for (std::size_t c = 0; c < a.columns.size(); ++c) {
      REQUIRE(a.columns[c].column_id == b.columns[c].column_id);
      REQUIRE(a.columns[c].is_varchar == b.columns[c].is_varchar);
      require_same_segments(a.columns[c].data_segments, b.columns[c].data_segments);
      require_same_segments(a.columns[c].validity_segments, b.columns[c].validity_segments);
    }
  }
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
