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
#include <duckdb/common/column_index.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/sirius_dynamic_filter.hpp>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

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

// Keeps the throwaway DuckDB below on the CPU path; the ingestible ctor itself never
// needs the extension. Restores on scope exit so later tests can build Sirius envs.
struct sirius_disable_guard {
  sirius_disable_guard() { setenv("SIRIUS_DISABLE", "1", 1); }
  ~sirius_disable_guard() { unsetenv("SIRIUS_DISABLE"); }
};

// The remap is channel bookkeeping, so a host-only stand-in filter suffices —
// push_filter accepts any non-null sirius_dynamic_filter.
class stub_filter final : public sirius::op::sirius_dynamic_filter {
 public:
  [[nodiscard]] sirius::op::sirius_dynamic_filter_kind kind() const override
  {
    return sirius::op::sirius_dynamic_filter_kind::ZONE_MAP;
  }
};

bool push(sirius::op::sirius_dynamic_filter_set& set, std::size_t column_ids_idx)
{
  return set.push_filter(column_ids_idx, std::make_shared<stub_filter>());
}

/// Table t(a INTEGER, b BIGINT, c INTEGER) types, indexed by storage column.
duckdb::vector<sirius::logical_type> table_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::BIGINT),
          sirius::logical_type::make(sirius::type_id::INTEGER)};
}

/// Builds the bind info the way the plan generator / pipeline converter do: projected_cols and
/// projected_types follow source_ids order (projection_ids, or all of column_ids when empty),
/// and output_types covers only the first `output_arity` of them — trailing entries are
/// pure-filter columns that post_filter_and_project drops.
std::unique_ptr<duckdb_native_ingestible_table_info> make_info(
  duckdb::DataTable& storage,
  duckdb::ClientContext& ctx,
  duckdb::vector<duckdb::ColumnIndex> column_ids,
  duckdb::vector<duckdb::idx_t> projection_ids,
  std::size_t output_arity,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel)
{
  auto info            = std::make_unique<duckdb_native_ingestible_table_info>();
  info->storage        = &storage;
  info->context        = &ctx;
  info->db_path        = storage.GetAttached().GetStorageManager().GetDBPath();
  info->names          = {"a", "b", "c"};
  info->schema_name    = "main";
  info->table_name     = "t";
  info->returned_types = table_types();

  duckdb::vector<duckdb::idx_t> source_ids_fallback;
  if (projection_ids.empty()) {
    for (duckdb::idx_t i = 0; i < column_ids.size(); ++i) {
      source_ids_fallback.push_back(i);
    }
  }
  auto const& source_ids = projection_ids.empty() ? source_ids_fallback : projection_ids;

  for (std::size_t k = 0; k < source_ids.size(); ++k) {
    auto const& col_idx = column_ids[source_ids[k]];
    projected_column pc;
    pc.is_rowid = col_idx.IsRowIdColumn();
    if (!pc.is_rowid) { pc.storage_idx = duckdb::StorageIndex(col_idx.GetPrimaryIndex()); }
    info->projected_cols.push_back(pc);
    info->projected_types.push_back(info->returned_types[col_idx.GetPrimaryIndex()]);
  }
  for (std::size_t k = 0; k < output_arity; ++k) {
    info->output_types.push_back(info->projected_types[k]);
  }

  info->column_ids             = std::move(column_ids);
  info->projection_ids         = std::move(projection_ids);
  info->sirius_dynamic_filters = std::move(channel);
  return info;
}

}  // namespace

TEST_CASE("duckdb-native ingestible installs the dynamic-filter column remap",
          "[scan][duckdb_native][dynamic_filter]")
{
  sirius_disable_guard disable_sirius;

  // The ingestible ctor requires a single-file block manager, so the backing table
  // must live in a file-backed database (an in-memory one cannot serve it).
  auto tmp = fs::temp_directory_path() / ("sirius-ddbnative-remap-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  auto db_file = tmp / "remap.duckdb";

  duckdb::DuckDB db(db_file.string().c_str());
  duckdb::Connection con(db);
  exec_ok(con, "CREATE TABLE t(a INTEGER, b BIGINT, c INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range, range * 2, range * 3 FROM range(1000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  duckdb::vector<duckdb::ColumnIndex> const all_cols = {
    duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2)};

  SECTION("projection reorder, no static filters")
  {
    // column_ids = [a(0), b(1), c(2)]; output = [c, a]. Producers push in column_ids space;
    // the channel must key by output position: a(0) -> 1, b(1) -> pruned, c(2) -> 0.
    auto channel = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    auto ingestible =
      make_ingestible(make_info(storage, *con.context, all_cols, {2, 0}, 2, channel));
    REQUIRE(ingestible);

    REQUIRE(push(*channel, 2));        // c -> output position 0
    REQUIRE(push(*channel, 0));        // a -> output position 1
    REQUIRE_FALSE(push(*channel, 1));  // b produces no output column

    auto cols = channel->filtered_columns();
    std::sort(cols.begin(), cols.end());
    REQUIRE(cols == std::vector<std::size_t>{0, 1});
    REQUIRE(channel->filters_for_column(2).empty());  // nothing keyed by raw column_ids index
  }

  SECTION("pure-filter trailing column maps to the sentinel")
  {
    // All three columns are decoded, but only [a, b] are emitted — c exists solely for the
    // static filter and post_filter_and_project drops it. A dynamic filter aimed at c must be
    // rejected: mapped to c's decode position it would silently index past the emitted batch.
    auto channel        = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    auto info           = make_info(storage, *con.context, all_cols, {0, 1, 2}, 2, channel);
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->filters[2] = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_GREATERTHAN, duckdb::Value::INTEGER(10));
    auto ingestible = make_ingestible(std::move(info));
    REQUIRE(ingestible);

    REQUIRE_FALSE(push(*channel, 2));  // pure-filter column: no output position
    REQUIRE(push(*channel, 0));
    REQUIRE(push(*channel, 1));

    auto cols = channel->filtered_columns();
    std::sort(cols.begin(), cols.end());
    REQUIRE(cols == std::vector<std::size_t>{0, 1});
  }

  SECTION("empty projection_ids: decode-order identity bounded by column_ids")
  {
    // No projection pushdown: output order equals column_ids order. The remap must still be
    // installed (identity within range) so out-of-range producer references are rejected —
    // with no remap the channel would store any index unchanged.
    duckdb::vector<duckdb::ColumnIndex> two_cols = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
    auto channel    = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    auto ingestible = make_ingestible(make_info(storage, *con.context, two_cols, {}, 2, channel));
    REQUIRE(ingestible);

    REQUIRE(push(*channel, 1));
    REQUIRE_FALSE(push(*channel, 5));  // outside column_ids
    REQUIRE(channel->filtered_columns() == std::vector<std::size_t>{1});
  }

  SECTION("null channel is a no-op")
  {
    auto ingestible =
      make_ingestible(make_info(storage, *con.context, all_cols, {2, 0}, 2, nullptr));
    REQUIRE(ingestible);
  }

  fs::remove_all(tmp, ec);
}
