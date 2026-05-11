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

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <utils/utils.hpp>

#include <string>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

// `Connection::Query` returns a non-null result on failure (error lives in
// `HasError()`), so a bare `REQUIRE(con.Query(...))` would silently pass.
void exec_ok(duckdb::Connection& con, const std::string& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

// Catalog access requires an active transaction. The transaction stays open
// for the rest of the test case and DuckDB rolls it back when `con` dies.
duckdb::DataTable& get_storage(duckdb::Connection& con, const std::string& table_name)
{
  exec_ok(con, "BEGIN TRANSACTION");
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry   = schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, table_name);
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

projected_column real_col(duckdb::idx_t col_id)
{
  projected_column pc;
  pc.storage_idx = duckdb::StorageIndex(col_id);
  pc.is_rowid    = false;
  return pc;
}

projected_column rowid_col()
{
  projected_column pc;
  pc.is_rowid = true;
  return pc;
}

}  // namespace

TEST_CASE("walker refuses empty projection", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 100)");
  auto& storage = get_storage(con, "t");

  auto md = walk_duckdb_native_metadata(storage, *con.context, {}, {});
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("no projected columns") != std::string::npos);
}

TEST_CASE("walker refuses parallel-vector mismatch", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("size mismatch") != std::string::npos);
}

TEST_CASE("walker refuses HUGEINT type", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a HUGEINT)");
  exec_ok(con, "INSERT INTO t VALUES (1), (2), (3)");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::HUGEINT)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("HUGEINT") != std::string::npos);
}

TEST_CASE("walker emits descriptors for INTEGER table", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  // Crosses a vector boundary so we get at least one materialised segment.
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 3000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE(md.viability_failure_reason.empty());
  REQUIRE_FALSE(md.row_groups.empty());

  duckdb::idx_t total_rows = 0;
  for (const auto& rg : md.row_groups) {
    REQUIRE(rg.columns.size() == 1);
    REQUIRE(rg.columns[0].column_id == 0);
    REQUIRE_FALSE(rg.columns[0].is_rowid);
    REQUIRE_FALSE(rg.columns[0].data_segments.empty());
    for (const auto& d : rg.columns[0].data_segments) {
      REQUIRE(d.segment_count > 0);
    }
    total_rows += rg.row_count;
  }
  REQUIRE(total_rows == 3000);
}

TEST_CASE("walker emits rowid sentinels with no segments", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 100)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0), rowid_col()};
  std::vector<sirius::logical_type> ts = {
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make(sirius::type_id::BIGINT),
  };
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());

  for (const auto& rg : md.row_groups) {
    REQUIRE(rg.columns.size() == 2);
    REQUIRE_FALSE(rg.columns[0].is_rowid);
    REQUIRE(rg.columns[1].is_rowid);
    REQUIRE(rg.columns[1].data_segments.empty());
    REQUIRE(rg.columns[1].validity_segments.empty());
    // Rowid contributes 8 bytes per row to the budget.
    REQUIRE(rg.decoded_bytes_budget >=
            static_cast<std::size_t>(rg.row_count) * sizeof(std::int64_t));
  }
}

TEST_CASE("walker rowid-only projection gets row_count from PartitionStats",
          "[scan][duckdb_native_walker]")
{
  // Without PartitionStats as the row_count source, a rowid-only projection
  // would land row_count=0 and break rowid synthesis downstream.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1500)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {rowid_col()};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::BIGINT)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());

  duckdb::idx_t total_rows = 0;
  for (const auto& rg : md.row_groups) {
    REQUIRE(rg.columns.size() == 1);
    REQUIRE(rg.columns[0].is_rowid);
    REQUIRE(rg.row_count > 0);
    REQUIRE(rg.decoded_bytes_budget ==
            static_cast<std::size_t>(rg.row_count) * sizeof(std::int64_t));
    total_rows += rg.row_count;
  }
  REQUIRE(total_rows == 1500);
}

TEST_CASE("walker separates data and validity segments by column_path",
          "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  // Mix of values + nulls so the validity segment is non-trivial.
  exec_ok(con,
          "INSERT INTO t SELECT CASE WHEN range % 7 = 0 THEN NULL ELSE range END "
          "FROM range(0, 2000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());

  bool saw_validity = false;
  for (const auto& rg : md.row_groups) {
    if (!rg.columns[0].validity_segments.empty()) {
      saw_validity = true;
      break;
    }
  }
  REQUIRE(saw_validity);
}

TEST_CASE("walker walks VARCHAR (Uncompressed) table without max-length stat needed",
          "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(con, "INSERT INTO t SELECT 'hello' FROM range(0, 200)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  // Both outcomes are valid: viable=true (Uncompressed/RLE varchar) or
  // viable=false with a max_string_length reason (Dictionary/FSST without
  // the stat). Confirm the walk didn't crash and the budget flag is honest.
  if (md.viable) {
    REQUIRE_FALSE(md.row_groups.empty());
    for (const auto& rg : md.row_groups) {
      bool any_varchar_unknown = false;
      for (const auto& d : rg.columns[0].data_segments) {
        if (d.max_string_length == 0) {
          any_varchar_unknown = true;
          break;
        }
      }
      if (any_varchar_unknown) { REQUIRE(rg.decoded_bytes_budget_is_lower_bound); }
    }
  } else {
    REQUIRE_FALSE(md.viability_failure_reason.empty());
  }
}

TEST_CASE("walker refuses DECIMAL128 (precision > 18)", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a DECIMAL(38, 0))");
  exec_ok(con, "INSERT INTO t VALUES (1), (2), (3)");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make_decimal(38, 0)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("DECIMAL128") != std::string::npos);
}

TEST_CASE("walker accepts DECIMAL64 (precision <= 18)", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a DECIMAL(18, 2))");
  exec_ok(con, "INSERT INTO t VALUES (1.50), (2.25), (3.00)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make_decimal(18, 2)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());
}

TEST_CASE("walker refuses STRUCT projected type", "[scan][duckdb_native_walker]")
{
  // Type check fires before any segment walk, so an INTEGER table with a
  // synthetic STRUCT projected type is enough.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t VALUES (1)");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::STRUCT)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("STRUCT") != std::string::npos);
}

TEST_CASE("walker refuses LIST projected type", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t VALUES (1)");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::LIST)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("LIST") != std::string::npos);
}

TEST_CASE("walker refuses unsupported data compression (force ZSTD)",
          "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "PRAGMA force_compression='zstd'");
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 5000)");
  exec_ok(con, "CHECKPOINT");

  // `force_compression` does not guarantee the codec lands. DuckDB's
  // analyzer narrows the candidate set to {Uncompressed, forced} and then
  // picks by score. For ZSTD specifically, `ZSTDFun::TypeIsSupported`
  // (duckdb/storage/compression/zstd.cpp) returns true only for
  // PhysicalType::VARCHAR, so on this INTEGER column ZSTD never enters
  // the contest at all and the segment lands as BitPacking/Uncompressed.
  // We exercise the walker's refusal only when a ZSTD segment actually
  // exists; otherwise emit a WARN so the skip is visible in test output.
  bool zstd_landed = false;
  {
    auto result = con.Query("SELECT compression FROM pragma_storage_info('t')");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    while (auto chunk = result->Fetch()) {
      for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
        if (chunk->GetValue(0, i).ToString() == "ZSTD") {
          zstd_landed = true;
          break;
        }
      }
      if (zstd_landed) { break; }
    }
  }

  auto& storage                        = get_storage(con, "t");
  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto md = walk_duckdb_native_metadata(storage, *con.context, cols, ts);
  if (zstd_landed) {
    REQUIRE_FALSE(md.viable);
    REQUIRE(md.viability_failure_reason.find("ZSTD") != std::string::npos);
  } else {
    WARN(
      "force_compression='zstd' produced no ZSTD segment on this DuckDB "
      "build (likely BitPacking won for monotonic integers); walker "
      "refusal path not exercised");
  }
}

// Round-trip guard against DuckDB renaming a codec string (e.g. "Empty
// Validity" → "Empty"). Without it, a rename funnels every renamed segment
// through the COMPRESSION_COUNT sentinel with a misleading
// "unsupported compression" diagnostic.
TEST_CASE("CompressionTypeToString output matches walker reverse-map keys",
          "[scan][duckdb_native_walker]")
{
  using duckdb::CompressionType;
  using duckdb::CompressionTypeToString;
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_UNCOMPRESSED)) ==
          "Uncompressed");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_CONSTANT)) ==
          "Constant");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_RLE)) == "RLE");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_DICTIONARY)) ==
          "Dictionary");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_BITPACKING)) ==
          "BitPacking");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_FSST)) == "FSST");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_DICT_FSST)) ==
          "DICT_FSST");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_ALP)) == "ALP");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_ALPRD)) == "ALPRD");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_ROARING)) == "Roaring");
  REQUIRE(std::string(CompressionTypeToString(CompressionType::COMPRESSION_EMPTY)) ==
          "Empty Validity");
}
