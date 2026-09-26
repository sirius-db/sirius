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

// Shared helpers for the two-phase walker. The old disabled one-phase cases
// are replaced by the versioned physical-matrix suite.
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
