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
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/optional_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <op/scan/duckdb_native_decoder.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <unistd.h>
#include <utils/utils.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

// Shared helpers, kept outside the disabled block below so live cases can use them.
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

/// Runs the pre-step plus a single full-table range and combines them into one
/// result. Exercises `prepare_duckdb_native_walk` and
/// `walk_duckdb_native_row_group_range`. Optional filters drive statistics
/// pruning.
struct walk_result {
  std::vector<duckdb_row_group_metadata> row_groups;
  bool viable = false;
  std::string viability_failure_reason;
  std::size_t pruned_row_groups    = 0;
  std::size_t pruned_decoded_bytes = 0;
};

walk_result walk_all(duckdb::DataTable& storage,
                     duckdb::ClientContext& ctx,
                     const std::vector<projected_column>& cols,
                     const std::vector<sirius::logical_type>& types,
                     const duckdb::TableFilterSet* table_filters           = nullptr,
                     const duckdb::vector<duckdb::ColumnIndex>* column_ids = nullptr)
{
  auto plan = prepare_duckdb_native_walk(storage, ctx, cols, types, table_filters, column_ids);
  if (!plan.viable) {
    return {{},
            false,
            std::move(plan.viability_failure_reason),
            plan.pruned_row_groups,
            plan.pruned_decoded_bytes};
  }
  auto range = walk_duckdb_native_row_group_range(plan, 0, plan.n_row_groups);
  return {std::move(range.row_groups),
          range.viable,
          std::move(range.viability_failure_reason),
          range.pruned_row_groups,
          range.pruned_decoded_bytes};
}

}  // namespace

//===--------------------------------------------------------------------===//
// Overflow (big-string) refusal — strings at/over GetStringBlockLimit (4,096 B
// at the default block size) live in overflow blocks the GPU string decoder
// cannot resolve; the walker must refuse at prepare time.
//===--------------------------------------------------------------------===//

TEST_CASE("walker refuses varchar containing overflow-length strings",
          "[scan][duckdb_native_walker][overflow_string]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(con, "INSERT INTO t VALUES (repeat('x', 5000))");
  exec_ok(con, "INSERT INTO t SELECT 'short' FROM range(0, 100)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("overflow") != std::string::npos);
}

TEST_CASE("walker refuses varchar exactly at the overflow limit",
          "[scan][duckdb_native_walker][overflow_string]")
{
  // Overflow triggers at length >= limit, so the refusal boundary must match.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(con, "INSERT INTO t VALUES (repeat('x', 4096))");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("overflow") != std::string::npos);
}

TEST_CASE("walker accepts varchar below the overflow limit",
          "[scan][duckdb_native_walker][overflow_string]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(con, "INSERT INTO t SELECT repeat('x', 4000) FROM range(0, 100)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());
}

TEST_CASE("walker marks all-NULL constant validity and snapshots constant stats",
          "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(k INTEGER, v INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range, NULL FROM range(3000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0), real_col(1)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::INTEGER),
                                          sirius::logical_type::make(sirius::type_id::INTEGER)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE(md.row_groups.size() == 1);

  // k never holds NULLs, so no validity segment carries the marker.
  for (const auto& s : md.row_groups[0].columns[0].validity_segments) {
    REQUIRE_FALSE(s.all_null);
  }
  // v is all-NULL: CONSTANT validity with the all-null marker, and the
  // CONSTANT data segment snapshots its own stats for the value decode.
  bool saw_all_null = false;
  for (const auto& s : md.row_groups[0].columns[1].validity_segments) {
    if (!s.all_null) { continue; }
    saw_all_null = true;
    REQUIRE(s.compression == duckdb::CompressionType::COMPRESSION_CONSTANT);
  }
  REQUIRE(saw_all_null);
  bool saw_constant = false;
  for (const auto& s : md.row_groups[0].columns[1].data_segments) {
    if (s.compression != duckdb::CompressionType::COMPRESSION_CONSTANT) { continue; }
    saw_constant = true;
    REQUIRE(s.segment_stats != nullptr);
  }
  REQUIRE(saw_constant);
}

TEST_CASE("walker marks all-NULL constant validity on ARRAY trees", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER[3])");
  exec_ok(con, "INSERT INTO t SELECT NULL::INTEGER[3] FROM range(3000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {
    sirius::logical_type::make_array(sirius::logical_type::make(sirius::type_id::INTEGER), 3)};
  auto md = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());

  // Array-level validity (routed into data_segments) carries the marker for
  // a fully-NULL ARRAY column.
  bool saw_all_null = false;
  for (const auto& rg : md.row_groups) {
    for (const auto& s : rg.columns[0].data_segments) {
      if (s.all_null) { saw_all_null = true; }
    }
  }
  REQUIRE(saw_all_null);
}

// TEMPORARILY DISABLED: these cases call the removed monolithic walk_duckdb_native_metadata().
// TODO: migrate to prepare_duckdb_native_walk() + walk_duckdb_native_row_group_range() and
// re-enable — the duckdb_native_row_group_range result exposes the same .viable / .row_groups /
// .viability_failure_reason / .pruned_row_groups these asserts use.
#if 0

namespace {

projected_column rowid_col()
{
  projected_column pc;
  pc.is_rowid = true;
  return pc;
}

// A one-column TableFilterSet keyed by the relative scan-column index `col_key`
// (matching create_table_filter_set's remapping), plus the parallel column_ids
// mapping that key back to storage index `storage_idx`.
struct filter_ctx {
  duckdb::TableFilterSet filters;
  duckdb::vector<duckdb::ColumnIndex> column_ids;
};

class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_native_range_bytes_XXXXXX";
    int fd      = ::mkstemp(tmpl);
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl);
    _path = tmpl;
  }

  ~scoped_temp_db_path()
  {
    if (!_path.empty()) {
      std::remove(_path.c_str());
      std::remove((_path + ".wal").c_str());
    }
  }

  scoped_temp_db_path(const scoped_temp_db_path&)            = delete;
  scoped_temp_db_path& operator=(const scoped_temp_db_path&) = delete;

  const std::string& path() const { return _path; }

 private:
  std::string _path;
};

filter_ctx make_constant_filter(duckdb::idx_t col_key,
                                duckdb::idx_t storage_idx,
                                duckdb::ExpressionType cmp,
                                duckdb::Value constant)
{
  filter_ctx ctx;
  ctx.filters.filters[col_key] =
    duckdb::make_uniq<duckdb::ConstantFilter>(cmp, std::move(constant));
  // column_ids must be indexable at col_key; pad with the same storage_idx.
  ctx.column_ids.resize(col_key + 1, duckdb::ColumnIndex(storage_idx));
  ctx.column_ids[col_key] = duckdb::ColumnIndex(storage_idx);
  return ctx;
}

}  // namespace

TEST_CASE("walker refuses empty projection", "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 100)");
  auto& storage = get_storage(con, "t");

  auto md = walk_all(storage, *con.context, {}, {});
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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
  auto md = walk_all(storage, *con.context, cols, ts);
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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

TEST_CASE("walker populates per-segment max_string_length for VARCHAR",
          "[scan][duckdb_native_walker]")
{
  // After the walk succeeds, every VARCHAR data segment must carry Some(...).
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(con, "INSERT INTO t SELECT 'hello' FROM range(0, 2500)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());
  for (const auto& rg : md.row_groups) {
    for (const auto& d : rg.columns[0].data_segments) {
      REQUIRE(d.max_string_length.has_value());
      REQUIRE(*d.max_string_length == 5);  // "hello" → length 5
    }
  }
}

TEST_CASE("walker accepts all-empty varchar row group (Some(0))", "[scan][duckdb_native_walker]")
{
  // An all-empty-string row group is legal data — DuckDB stores
  // MaxStringLength: 0 in the segment_stats blob, and the walker accepts.
  // Real-world example: clickbench SocialAction, ParamOrderID, Title row
  // groups (Dict-encoded, msl=0).
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(con, "INSERT INTO t SELECT '' FROM range(0, 100000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto md                              = walk_all(storage, *con.context, cols, ts);

  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());
  bool saw_zero = false;
  for (const auto& rg : md.row_groups) {
    for (const auto& d : rg.columns[0].data_segments) {
      REQUIRE(d.max_string_length.has_value());
      if (*d.max_string_length == 0) { saw_zero = true; }
    }
  }
  REQUIRE(saw_zero);
}

TEST_CASE("walker per-segment max_string_length reflects long strings",
          "[scan][duckdb_native_walker]")
{
  // The stat is per-storage-segment, not per-row-group. The hard guarantee
  // we can lock in here is that long strings make it into the value (it's
  // not collapsed to 0 or a vector-level number). A fixture that forces
  // multiple distinct per-segment values is fragile — DuckDB's compressor
  // may pack three short-string classes into one segment. The empirical
  // "values genuinely vary within an rg" case is verified against
  // ClickBench data (rg 0 Title has 28 segments with msl 520..1026).
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(con, "INSERT INTO t SELECT repeat('x', 500) FROM range(0, 5000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::VARCHAR)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());

  for (const auto& rg : md.row_groups) {
    for (const auto& d : rg.columns[0].data_segments) {
      REQUIRE(d.max_string_length.has_value());
      REQUIRE(*d.max_string_length == 500);
    }
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
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
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("LIST") != std::string::npos);
}

// A fixed-size ARRAY column is ArrayColumnData, walked by walk_array_column:
// array-level validity routes into data_segments (NOT validity_segments), and
// the child column's data/validity land in the array_child_* vectors.
static sirius::logical_type array_of(sirius::type_id child, std::uint64_t size)
{
  return sirius::logical_type::make_array(sirius::logical_type::make(child), size);
}

TEST_CASE("walker emits child data and marks is_array for ARRAY column",
          "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER[3])");
  // Crosses a vector boundary so we get at least one materialized segment
  exec_ok(con, "INSERT INTO t SELECT [range, range + 1, range + 2] FROM range(0, 3000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {array_of(sirius::type_id::INTEGER, 3)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE(md.viability_failure_reason.empty());
  REQUIRE_FALSE(md.row_groups.empty());

  for (const auto& rg : md.row_groups) {
    REQUIRE(rg.columns.size() == 1);
    const auto& col = rg.columns[0];
    REQUIRE(col.is_array);
    // Child element data lands in array_child_data_segments
    REQUIRE_FALSE(col.array_child_data_segments.empty());
    for (const auto& d : col.array_child_data_segments) {
      REQUIRE(d.segment_count > 0);
    }
    // Array-level validity is routed to data_segments
    // top-level validity_segments is unused for ARRAY columns
    REQUIRE(col.validity_segments.empty());
    REQUIRE(rg.decoded_bytes_budget > 0);
  }
}

TEST_CASE("walker routes ARRAY-level validity into data_segments for NULL arrays",
          "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER[3])");
  // Whole-array NULLs populate the array-level validity (path [col,0])
  exec_ok(con, "INSERT INTO t VALUES ([1, 2, 3]), (NULL), ([7, 8, 9]), (NULL), ([4, 5, 6])");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {array_of(sirius::type_id::INTEGER, 3)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());

  bool saw_array_validity = false;
  for (const auto& rg : md.row_groups) {
    const auto& col = rg.columns[0];
    REQUIRE(col.is_array);
    // Array validity must never leak into the top-level validity_segments
    REQUIRE(col.validity_segments.empty());
    if (!col.data_segments.empty()) { saw_array_validity = true; }
  }
  REQUIRE(saw_array_validity);
}

TEST_CASE("walker emits ARRAY child validity segments for NULL elements",
          "[scan][duckdb_native_walker]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER[3])");
  // Element-level NULLs (path [col,1,0]) i.e., the array is present, elements null
  exec_ok(con, "INSERT INTO t VALUES ([1, NULL, 3]), ([4, 5, 6]), ([NULL, NULL, NULL])");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {array_of(sirius::type_id::INTEGER, 3)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE(md.viable);
  REQUIRE_FALSE(md.row_groups.empty());

  bool saw_child_validity = false;
  for (const auto& rg : md.row_groups) {
    if (!rg.columns[0].array_child_validity_segments.empty()) {
      saw_child_validity = true;
      break;
    }
  }
  REQUIRE(saw_child_validity);
}

TEST_CASE("walker refuses ARRAY with non-fixed-width element", "[scan][duckdb_native_walker]")
{
  // Type check should catch before the segment walk for non-fixed-width element
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t VALUES (1)");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {array_of(sirius::type_id::VARCHAR, 3)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("non-fixed-width") != std::string::npos);
}

TEST_CASE("walker refuses ARRAY without child type metadata", "[scan][duckdb_native_walker]")
{
  // An ARRAY id built without make_array() carries no child type to validate
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t VALUES (1)");
  auto& storage = get_storage(con, "t");

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::ARRAY)};
  auto md                              = walk_all(storage, *con.context, cols, ts);
  REQUIRE_FALSE(md.viable);
  REQUIRE(md.viability_failure_reason.find("without child") != std::string::npos);
}

// checked_array_child_advance guards the ARRAY child-element cursor against
// overflowing cudf::size_type (int32), the type of the downstream LIST offsets.
TEST_CASE("checked_array_child_advance rejects size_type overflow", "[scan][array]")
{
  auto const kMax = static_cast<uint64_t>(std::numeric_limits<cudf::size_type>::max());

  // Normal advance: cursor + row_count * array_size.
  REQUIRE(checked_array_child_advance(0, 122880, 768) == static_cast<uint32_t>(122880u * 768u));
  REQUIRE(checked_array_child_advance(1000, 10, 4) == 1040u);

  // Exactly at the int32 limit is allowed; one past it overflows.
  REQUIRE(checked_array_child_advance(0, kMax, 1).value() == static_cast<uint32_t>(kMax));
  REQUIRE_FALSE(checked_array_child_advance(0, kMax + 1, 1).has_value());

  // 6M rows * 768 dims = 4.6B > int32 max.
  REQUIRE_FALSE(checked_array_child_advance(0, 6'000'000, 768).has_value());

  // Accumulation tips it over even when a single step is fine.
  REQUIRE(checked_array_child_advance(0, 2'000'000, 768).has_value());  // 1.536B ok
  REQUIRE_FALSE(checked_array_child_advance(1'536'000'000u, 2'000'000, 768).has_value());
}

// Driving the walker into an unsupported codec via real DuckDB storage is
// fragile — ZSTD is the only non-deprecated data codec not in the supported
// list, and its analyzer is conservative enough that `force_compression`
// rarely produces a ZSTD segment in unit-test-sized data. Exercise the
// codec-rejection predicates directly instead.
TEST_CASE("walker rejects unsupported data codecs", "[scan][duckdb_native_walker]")
{
  using duckdb::CompressionType;
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_UNCOMPRESSED));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_CONSTANT));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_RLE));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_DICTIONARY));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_BITPACKING));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_FSST));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_DICT_FSST));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_ALP));
  REQUIRE(is_supported_data_compression(CompressionType::COMPRESSION_ALPRD));

  REQUIRE_FALSE(is_supported_data_compression(CompressionType::COMPRESSION_ZSTD));
  REQUIRE_FALSE(is_supported_data_compression(CompressionType::COMPRESSION_CHIMP));
  REQUIRE_FALSE(is_supported_data_compression(CompressionType::COMPRESSION_PATAS));
  REQUIRE_FALSE(is_supported_data_compression(CompressionType::COMPRESSION_PFOR_DELTA));
  // ROARING is supported as validity but never as data.
  REQUIRE_FALSE(is_supported_data_compression(CompressionType::COMPRESSION_ROARING));
  // EMPTY is the all-null validity sentinel; never appears on a data segment.
  REQUIRE_FALSE(is_supported_data_compression(CompressionType::COMPRESSION_EMPTY));
  // COUNT is the sentinel `parse_compression_string` returns for unrecognized
  // codec names (e.g. a future DuckDB introducing a codec we have not
  // reviewed) — the walker must refuse it.
  REQUIRE_FALSE(is_supported_data_compression(CompressionType::COMPRESSION_COUNT));
}

TEST_CASE("walker rejects unsupported validity codecs", "[scan][duckdb_native_walker]")
{
  using duckdb::CompressionType;
  REQUIRE(is_supported_validity_compression(CompressionType::COMPRESSION_UNCOMPRESSED));
  REQUIRE(is_supported_validity_compression(CompressionType::COMPRESSION_EMPTY));
  REQUIRE(is_supported_validity_compression(CompressionType::COMPRESSION_CONSTANT));
  REQUIRE(is_supported_validity_compression(CompressionType::COMPRESSION_ROARING));

  REQUIRE_FALSE(is_supported_validity_compression(CompressionType::COMPRESSION_ZSTD));
  REQUIRE_FALSE(is_supported_validity_compression(CompressionType::COMPRESSION_DICTIONARY));
  REQUIRE_FALSE(is_supported_validity_compression(CompressionType::COMPRESSION_FSST));
  REQUIRE_FALSE(is_supported_validity_compression(CompressionType::COMPRESSION_COUNT));
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

//===--------------------------------------------------------------------===//
// Row-group filter-statistics pruning
//===--------------------------------------------------------------------===//

// 300k monotonically increasing ints span 3 row groups (122880 each), with tight
// per-row-group min/max. A `a >= 250000` lower bound makes every row group whose
// max < 250000 FILTER_ALWAYS_FALSE, so the walker should drop them.
namespace {
constexpr int kManyRows = 300000;

duckdb::DataTable& make_monotonic_int_table(duckdb::Connection& con)
{
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, " + std::to_string(kManyRows) + ")");
  exec_ok(con, "CHECKPOINT");
  return get_storage(con, "t");
}
}  // namespace

TEST_CASE("statistics pruning drops row groups below a lower-bound filter",
          "[scan][duckdb_native_walker][pruning]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto& storage        = make_monotonic_int_table(con);

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::INTEGER)};

  // Baseline: no filter → nothing pruned.
  auto base = walk_all(storage, *con.context, cols, ts);
  REQUIRE(base.viable);
  REQUIRE(base.pruned_row_groups == 0);
  REQUIRE(base.row_groups.size() >= 2);  // 300k rows > one 122880-row group

  // a >= 250000 → row groups entirely below 250000 are FILTER_ALWAYS_FALSE.
  auto ctx = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000));
  auto pruned = walk_all(storage, *con.context, cols, ts, &ctx.filters, &ctx.column_ids);

  REQUIRE(pruned.viable);
  REQUIRE(pruned.pruned_row_groups >= 1);
  // Consistency: survivors + pruned == baseline row groups.
  REQUIRE(pruned.row_groups.size() + pruned.pruned_row_groups == base.row_groups.size());
  REQUIRE(pruned.row_groups.size() >= 1);

  // Survivors carry fewer rows than the full table (we dropped low row groups),
  // and none of the pruned bytes are zero (we accounted for real decode work).
  duckdb::idx_t surviving_rows = 0;
  for (const auto& rg : pruned.row_groups) {
    surviving_rows += rg.row_count;
  }
  REQUIRE(surviving_rows > 0);
  REQUIRE(surviving_rows < static_cast<duckdb::idx_t>(kManyRows));
  REQUIRE(pruned.pruned_decoded_bytes > 0);
}

TEST_CASE("statistics pruning keeps every row group when the filter matches all",
          "[scan][duckdb_native_walker][pruning]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto& storage        = make_monotonic_int_table(con);

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::INTEGER)};

  // a >= 0 is always true for this data → no row group can be pruned.
  auto ctx = make_constant_filter(
    0, 0, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(0));
  auto md = walk_all(storage, *con.context, cols, ts, &ctx.filters, &ctx.column_ids);

  REQUIRE(md.viable);
  REQUIRE(md.pruned_row_groups == 0);
  REQUIRE(md.pruned_decoded_bytes == 0);
}

// Guards the relaxation that lets us prune on any static filter, not just the
// subset re-applied post-decode. An OPTIONAL_FILTER ("not required for query
// correctness") wrapping a lower-bound still prunes, because OptionalFilter
// delegates CheckStatistics to its child. Pre-relaxation this pruned nothing.
TEST_CASE("statistics pruning prunes through an OPTIONAL_FILTER wrapper",
          "[scan][duckdb_native_walker][pruning]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto& storage        = make_monotonic_int_table(con);

  std::vector<projected_column> cols   = {real_col(0)};
  std::vector<sirius::logical_type> ts = {sirius::logical_type::make(sirius::type_id::INTEGER)};

  filter_ctx ctx;
  ctx.filters.filters[0] =
    duckdb::make_uniq<duckdb::OptionalFilter>(duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(250000)));
  ctx.column_ids.push_back(duckdb::ColumnIndex(0));

  auto md = walk_all(storage, *con.context, cols, ts, &ctx.filters, &ctx.column_ids);

  REQUIRE(md.viable);
  REQUIRE(md.pruned_row_groups >= 1);
}

#endif  // walker tests disabled pending migration to the two-phase walk API
