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

// Deferred metadata walk. CPU-only: no GPU decode path is exercised here.

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
#include <op/scan/duckdb_native_metadata.hpp>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
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

// The ingestible constructor requires a SingleFileBlockManager, hence a file-backed database.
struct file_backed_db {
  std::filesystem::path path;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

  file_backed_db()
  {
    path = std::filesystem::temp_directory_path() /
           ("sirius_deferred_walk_test_" + std::to_string(::getpid()) + "_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".duckdb");
    std::filesystem::remove(path);
    db  = std::make_unique<duckdb::DuckDB>(path.string());
    con = std::make_unique<duckdb::Connection>(*db);
  }

  ~file_backed_db()
  {
    con.reset();
    db.reset();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + ".wal", ec);
  }
};

// Catalog access requires an active transaction; it rolls back when the connection dies.
duckdb::DataTable& get_storage(duckdb::Connection& con, const std::string& table_name)
{
  if (!con.context->transaction.HasActiveTransaction()) { exec_ok(con, "BEGIN TRANSACTION"); }
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

std::unique_ptr<duckdb_native_ingestible_table_info> make_info(
  duckdb::Connection& con,
  const std::string& table_name,
  std::vector<sirius::logical_type> types,
  bool defer,
  duckdb::unique_ptr<duckdb::TableFilterSet> filters = nullptr)
{
  auto& storage             = get_storage(con, table_name);
  auto info                 = std::make_unique<duckdb_native_ingestible_table_info>();
  info->storage             = &storage;
  info->context             = con.context.get();
  info->db_path             = storage.GetAttached().GetStorageManager().GetDBPath();
  info->defer_metadata_walk = defer;
  for (std::size_t i = 0; i < types.size(); ++i) {
    info->projected_cols.push_back(real_col(i));
    info->column_ids.push_back(duckdb::ColumnIndex(i));
    info->names.push_back("c" + std::to_string(i));
  }
  info->projected_types = std::move(types);
  for (auto const& t : info->projected_types) {
    info->returned_types.push_back(t);
    info->output_types.push_back(t);
  }
  info->table_filters = std::move(filters);
  info->table_name    = table_name;
  info->catalog_name  = "memory";
  info->schema_name   = "main";
  return info;
}

}  // namespace

TEST_CASE("deferred walk: construction skips the row-group walk, ensure runs it",
          "[scan][duckdb_native_deferred_walk]")
{
  file_backed_db fx;
  exec_ok(*fx.con, "CREATE TABLE t(k INTEGER, v BIGINT)");
  exec_ok(*fx.con, "INSERT INTO t SELECT range, range * 2 FROM range(300000)");
  exec_ok(*fx.con, "CHECKPOINT");

  auto types =
    std::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER),
                                      sirius::logical_type::make(sirius::type_id::BIGINT)};

  auto eager = duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/false));
  REQUIRE_FALSE(eager.metadata_walk_pending());
  auto const& eager_plan = eager.walk_plan_for_testing();
  REQUIRE(eager_plan.viable);
  REQUIRE(eager_plan.n_row_groups > 1);  // 300k rows spans multiple row groups

  auto deferred = duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/true));
  REQUIRE(deferred.metadata_walk_pending());
  REQUIRE_FALSE(deferred.walk_plan_for_testing().viable);  // untouched plan

  deferred.ensure_metadata_prepared();
  REQUIRE_FALSE(deferred.metadata_walk_pending());
  auto const& lazy_plan = deferred.walk_plan_for_testing();
  REQUIRE(lazy_plan.viable);
  REQUIRE(lazy_plan.n_row_groups == eager_plan.n_row_groups);
  REQUIRE(lazy_plan.row_count == eager_plan.row_count);
  REQUIRE(lazy_plan.row_group_start == eager_plan.row_group_start);

  // Idempotent.
  deferred.ensure_metadata_prepared();
  REQUIRE_FALSE(deferred.metadata_walk_pending());
}

TEST_CASE("deferred walk: filter-stat pruning is preserved when the walk runs lazily",
          "[scan][duckdb_native_deferred_walk]")
{
  file_backed_db fx;
  exec_ok(*fx.con, "CREATE TABLE t(k INTEGER)");
  exec_ok(*fx.con, "INSERT INTO t SELECT range FROM range(300000)");
  exec_ok(*fx.con, "CHECKPOINT");

  auto types =
    std::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER)};

  auto make_filters = [] {
    // k < -1: provably empty per row group -> every row group stats-pruned.
    auto filters        = duckdb::make_uniq<duckdb::TableFilterSet>();
    filters->filters[0] = duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHAN, duckdb::Value::INTEGER(-1));
    return filters;
  };

  auto eager =
    duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/false, make_filters()));
  auto const& eager_plan = eager.walk_plan_for_testing();
  REQUIRE(eager_plan.viable);
  REQUIRE(eager_plan.pruned_row_groups == eager_plan.n_row_groups);

  auto deferred =
    duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/true, make_filters()));
  REQUIRE(deferred.metadata_walk_pending());
  deferred.ensure_metadata_prepared();
  auto const& lazy_plan = deferred.walk_plan_for_testing();
  REQUIRE(lazy_plan.viable);
  REQUIRE(lazy_plan.n_row_groups == eager_plan.n_row_groups);
  REQUIRE(lazy_plan.pruned_row_groups == eager_plan.pruned_row_groups);
  REQUIRE(lazy_plan.pruned_decoded_bytes == eager_plan.pruned_decoded_bytes);
}

TEST_CASE("deferred walk: unsupported projected type still refuses at construction",
          "[scan][duckdb_native_deferred_walk]")
{
  file_backed_db fx;
  exec_ok(*fx.con, "CREATE TABLE t(h HUGEINT)");
  exec_ok(*fx.con, "INSERT INTO t VALUES (1), (2), (3)");
  exec_ok(*fx.con, "CHECKPOINT");

  auto types =
    std::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::HUGEINT)};

  // Eager in both modes: an undecodable type must refuse at plan time.
  REQUIRE_THROWS_WITH(duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/true)),
                      Catch::Contains("128-bit"));
  REQUIRE_THROWS_WITH(duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/false)),
                      Catch::Contains("128-bit"));
}

TEST_CASE("deferred walk: overflow-string refusal moves from construction to ensure",
          "[scan][duckdb_native_deferred_walk]")
{
  file_backed_db fx;
  exec_ok(*fx.con, "CREATE TABLE t(s VARCHAR)");
  exec_ok(*fx.con, "INSERT INTO t VALUES (repeat('x', 5000))");
  exec_ok(*fx.con, "INSERT INTO t SELECT 'short' FROM range(0, 100)");
  exec_ok(*fx.con, "CHECKPOINT");

  auto types =
    std::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::VARCHAR)};

  REQUIRE_THROWS_WITH(duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/false)),
                      Catch::Contains("overflow"));

  // Deferred: construction succeeds and the refusal moves to the walk. The plan-time,
  // table-level overflow probe in sirius_plan_get still refuses these before this point.
  auto deferred = duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/true));
  REQUIRE(deferred.metadata_walk_pending());
  REQUIRE_THROWS_WITH(deferred.ensure_metadata_prepared(), Catch::Contains("overflow"));
  // The failed walk is not latched: still pending, and a retry throws again.
  REQUIRE(deferred.metadata_walk_pending());
  REQUIRE_THROWS_WITH(deferred.ensure_metadata_prepared(), Catch::Contains("overflow"));
}

TEST_CASE("deferred walk: split claims see the walk's row-group count",
          "[scan][duckdb_native_deferred_walk]")
{
  file_backed_db fx;
  exec_ok(*fx.con, "CREATE TABLE t(k INTEGER)");
  exec_ok(*fx.con, "INSERT INTO t SELECT range FROM range(300000)");
  exec_ok(*fx.con, "CHECKPOINT");

  auto types =
    std::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER)};

  auto eager    = duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/false));
  auto deferred = duckdb_native_gpu_ingestible(make_info(*fx.con, "t", types, /*defer=*/true));

  // Advertises work before the walk, so the provider is not skipped.
  REQUIRE_FALSE(deferred.has_processed_all_metadata());

  deferred.ensure_metadata_prepared();
  // Claims are counted, never executed, so a null ioctx resolver is fine.
  auto count_claims = [](duckdb_native_gpu_ingestible& ing) {
    std::size_t n = 0;
    while (
      auto task = ing.next_split_provider(
        [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return nullptr; })) {
      ++n;
    }
    return n;
  };
  REQUIRE(count_claims(deferred) == count_claims(eager));
  REQUIRE(deferred.has_processed_all_metadata());
}
