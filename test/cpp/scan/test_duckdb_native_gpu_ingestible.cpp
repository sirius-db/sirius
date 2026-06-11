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

// Tests for duckdb_native_gpu_ingestible: ctor validation, range claiming, and
// consumer-side coalescing of ranges into cap-sized batches (including the
// single-split tail case).

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/storage/data_table.hpp>
#include <helper/logical_type.hpp>
#include <io/io_context.hpp>
#include <io/types.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_connector.hpp>
#include <scan_manager/split_provider.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;
using namespace sirius::scan_manager;

namespace {

void exec_ok(duckdb::Connection& con, std::string const& q)
{
  auto result = con.Query(q);
  REQUIRE(result);
  if (result->HasError()) {
    INFO("query failed: " << q << "\n  error: " << result->GetError());
    REQUIRE_FALSE(result->HasError());
  }
}

duckdb::DataTable& get_storage(duckdb::Connection& con, std::string const& table_name)
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

// Minimal duckdb-native bind data. Caller fills storage / context / projected
// vectors / batch size / db_path. column_ids / projection_ids / output_types /
// table_filters stay empty: no filter and no projection-down.
std::unique_ptr<io::ingestible_table_info> make_table_info(
  duckdb::DataTable* storage,
  duckdb::ClientContext* ctx,
  std::vector<projected_column> cols,
  std::vector<sirius::logical_type> types,
  std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
  std::string db_path                = "")
{
  auto info                    = std::make_unique<duckdb_native_ingestible_table_info>();
  info->storage                = storage;
  info->context                = ctx;
  info->db_path                = std::move(db_path);
  info->projected_cols         = std::move(cols);
  info->projected_types        = std::move(types);
  info->approximate_batch_size = approximate_batch_size;
  return info;
}

// A scan_manager with the default config (it builds a real local-file uring
// backend). With an empty db_path the ctor resolves no datasource, so the
// emitted payloads carry null io handles. The decoder, the only io_ctx
// consumer, is not driven here.
std::unique_ptr<sirius_scan_manager> make_scan_manager()
{
  return std::make_unique<sirius_scan_manager>(scan_manager_config{});
}

// Scheduler that runs each enqueued thunk synchronously, so split_provider::run()
// produces every range and closes the connector before returning.
struct inline_scheduler {
  template <typename F>
  void enqueue(F&& f)
  {
    f();
  }
};

// Drive the producer directly and flatten the raw row-group ranges it emits.
// Reports the number of ranges handed out.
struct drained {
  std::vector<duckdb_row_group_metadata> row_groups;
  std::size_t range_count = 0;
};

drained drain_ranges(duckdb_native_gpu_ingestible& ingestible)
{
  drained out;
  while (ingestible.has_more_splits()) {
    auto factory = ingestible.next_split_provider();
    if (!factory) break;
    auto carriers = factory();
    if (carriers.empty()) break;
    ++out.range_count;
    for (auto& c : carriers) {
      auto* range = dynamic_cast<duckdb_native_range_input*>(c.get());
      REQUIRE(range != nullptr);
      for (auto& rg : range->row_groups) {
        out.row_groups.push_back(std::move(rg));
      }
    }
  }
  return out;
}

// Producer->connector->consumer round-trip: run the producer through a
// split_connector via the inline scheduler, then drain the consumer's
// coalescer. Returns each coalesced batch.
std::vector<std::unique_ptr<op::operator_data>> run_and_consume(
  duckdb_native_gpu_ingestible& ingestible)
{
  split_connector connector;
  split_provider provider{ingestible};
  inline_scheduler sched;
  provider.run(sched, connector);  // pushes all carriers; closes connector on return

  std::vector<std::unique_ptr<op::operator_data>> batches;
  for (;;) {
    auto batch = ingestible.consume_next_input(connector);
    if (!batch) break;
    batches.push_back(std::move(batch));
  }
  // After the consumer returns nullptr the coalescer must be drained.
  REQUIRE(ingestible.consumer_drained());
  REQUIRE(connector.is_closed());
  return batches;
}

duckdb_native_split_payload const& payload_of(op::operator_data const& batch)
{
  auto const* input = dynamic_cast<scan_operator_input const*>(&batch);
  REQUIRE(input != nullptr);
  REQUIRE(input->metadata != nullptr);
  auto const* split_info = dynamic_cast<duckdb_native_split_info const*>(&input->metadata->scan());
  REQUIRE(split_info != nullptr);
  return split_info->payload;
}

}  // namespace

//===----------------------------------------------------------------------===//
// ctor hardening
//===----------------------------------------------------------------------===//
TEST_CASE("duckdb_native_gpu_ingestible throws on null storage",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto mgr  = make_scan_manager();
  auto info = make_table_info(/*storage=*/nullptr, /*ctx=*/nullptr, {}, {});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::invalid_argument);
}

TEST_CASE("duckdb_native_gpu_ingestible throws on null context",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage, /*ctx=*/nullptr, {}, {});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::invalid_argument);
}

TEST_CASE("duckdb_native_gpu_ingestible throws on projected vectors size mismatch",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage, con.context.get(), {real_col(0)}, /*types=*/{});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::invalid_argument);
}

TEST_CASE("duckdb_native_gpu_ingestible leaves io handles null for an empty db_path",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // With no .db path the ctor resolves no datasource: the io handles stay null
  // and construction still succeeds. A backend-less scan is rejected by the
  // decoder, not the ctor; this exercises the null-handle path without driving
  // the decoder.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};

  auto batches = run_and_consume(ingestible);
  REQUIRE_FALSE(batches.empty());
  auto const& payload = payload_of(*batches[0]);
  REQUIRE(payload.io_ctx == nullptr);
  REQUIRE(payload.db_io_object == nullptr);
}

TEST_CASE("duckdb_native_gpu_ingestible throws when walker rejects the table",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // HUGEINT is unsupported: prepare_duckdb_native_walk reports non-viable and
  // the ctor surfaces the rejection as a runtime_error.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a HUGEINT)");
  exec_ok(con, "INSERT INTO t VALUES (1), (2), (3)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::HUGEINT)});
  REQUIRE_THROWS_AS((duckdb_native_gpu_ingestible{std::move(info), *mgr}), std::runtime_error);
}

TEST_CASE("duckdb_native_gpu_ingestible rejects fully stats-pruned scans",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 300000)");
  exec_ok(con, "CHECKPOINT");
  auto& storage = get_storage(con, "t");

  auto mgr          = make_scan_manager();
  auto info         = make_table_info(&storage,
                              con.context.get(),
                                      {real_col(0)},
                                      {sirius::logical_type::make(sirius::type_id::INTEGER)});
  auto* native_info = dynamic_cast<duckdb_native_ingestible_table_info*>(info.get());
  REQUIRE(native_info != nullptr);
  native_info->table_filters             = duckdb::make_uniq<duckdb::TableFilterSet>();
  native_info->table_filters->filters[0] = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(1000000));
  native_info->column_ids.push_back(duckdb::ColumnIndex(0));

  REQUIRE_THROWS_WITH((duckdb_native_gpu_ingestible{std::move(info), *mgr}),
                      Catch::Contains("fully pruned"));
}

//===----------------------------------------------------------------------===//
// producer: range claiming
//===----------------------------------------------------------------------===//
TEST_CASE("duckdb_native_gpu_ingestible emits one range for a small INTEGER table",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};
  REQUIRE(ingestible.has_more_splits());

  auto factory = ingestible.next_split_provider();
  REQUIRE(factory);
  auto carriers = factory();
  REQUIRE(carriers.size() == 1);

  auto* range = dynamic_cast<duckdb_native_range_input*>(carriers[0].get());
  REQUIRE(range != nullptr);
  REQUIRE_FALSE(range->row_groups.empty());

  // After the only range is claimed both signals report exhaustion.
  REQUIRE_FALSE(ingestible.has_more_splits());
  auto exhausted = ingestible.next_split_provider();
  REQUIRE_FALSE(exhausted);
}

TEST_CASE("duckdb_native_gpu_ingestible ranges cover all row groups exactly once",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // ~1.2M INTEGER rows span several DuckDB row groups (<=122,880 rows each), so
  // the producer hands out multiple parse ranges. Their union must cover every
  // row group exactly once.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1200000)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};
  auto result = drain_ranges(ingestible);

  REQUIRE(result.row_groups.size() >= 2);
  REQUIRE(result.range_count >= 1);

  std::sort(result.row_groups.begin(), result.row_groups.end(), [](auto const& a, auto const& b) {
    return a.row_group_index < b.row_group_index;
  });
  for (std::size_t i = 0; i < result.row_groups.size(); ++i) {
    REQUIRE(result.row_groups[i].row_group_index == i);
    REQUIRE(result.row_groups[i].row_count > 0);
  }
  for (std::size_t i = 1; i < result.row_groups.size(); ++i) {
    REQUIRE(result.row_groups[i].row_group_start > result.row_groups[i - 1].row_group_start);
  }
  REQUIRE_FALSE(ingestible.has_more_splits());
}

//===----------------------------------------------------------------------===//
// consumer: coalescing + single-split race
//===----------------------------------------------------------------------===//
TEST_CASE("duckdb_native_gpu_ingestible consumer coalesces ranges, covers all row groups once",
          "[scan][duckdb_native_gpu_ingestible]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1200000)");
  auto& storage = get_storage(con, "t");

  // A small byte cap forces the coalescer to close multiple batches across the
  // parsed ranges.
  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)},
                              /*approximate_batch_size=*/4096);
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};

  auto batches = run_and_consume(ingestible);
  REQUIRE(batches.size() >= 2);  // multiple coalesced batches

  // Every batch's payload aliases the same bind data, and the union of their
  // row groups covers every row group exactly once.
  duckdb_native_ingestible_table_info const* first_table_info = nullptr;
  std::vector<duckdb::idx_t> seen_indices;
  for (auto const& b : batches) {
    auto const& payload = payload_of(*b);
    REQUIRE(payload.table_info != nullptr);
    REQUIRE_FALSE(payload.row_groups.empty());
    if (first_table_info == nullptr) {
      first_table_info = payload.table_info;
    } else {
      REQUIRE(payload.table_info == first_table_info);
    }
    for (auto const& rg : payload.row_groups) {
      seen_indices.push_back(rg.row_group_index);
    }
  }
  std::sort(seen_indices.begin(), seen_indices.end());
  for (std::size_t i = 0; i < seen_indices.size(); ++i) {
    REQUIRE(seen_indices[i] == i);  // contiguous from 0, no gaps or duplicates
  }
  REQUIRE(seen_indices.size() >= 2);
}

TEST_CASE("duckdb_native_gpu_ingestible single-split scan serves its batch then drains",
          "[scan][duckdb_native_gpu_ingestible]")
{
  // A single-split scan (one parse range) must serve its one batch before
  // reporting drained, not drop it when the connector closes.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  auto& storage = get_storage(con, "t");

  auto mgr  = make_scan_manager();
  auto info = make_table_info(&storage,
                              con.context.get(),
                              {real_col(0)},
                              {sirius::logical_type::make(sirius::type_id::INTEGER)});
  duckdb_native_gpu_ingestible ingestible{std::move(info), *mgr};

  auto batches = run_and_consume(ingestible);
  REQUIRE(batches.size() == 1);  // one row group → one tail batch, not dropped

  auto const& payload = payload_of(*batches[0]);
  REQUIRE_FALSE(payload.row_groups.empty());
}
