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

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/data_table.hpp>
#include <io/io_context.hpp>
#include <io/types.hpp>
#include <op/scan/duckdb_native_scan_info.hpp>
#include <scan_manager/duckdb_native_split_provider.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
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

/// Storage + context + a dummy db_path (the provider requires a non-empty
/// db_path; the stub io_ctx ignores it). Caller fills in projected_cols /
/// projected_types / approximate_batch_size.
duckdb_native_scan_info make_scan_info(duckdb::DataTable& storage, duckdb::ClientContext& ctx)
{
  duckdb_native_scan_info info;
  info.storage = &storage;
  info.context = &ctx;
  info.db_path = "stub.db";
  return info;
}

/// Drain all ranges single-threaded; flatten every range's row groups. Also
/// reports the number of ranges the provider emitted.
struct drained {
  std::vector<duckdb_row_group_metadata> row_groups;
  std::size_t range_count = 0;
};

drained drain_all(duckdb_native_split_provider& provider)
{
  drained out;
  while (provider.has_more_splits()) {
    auto factory = provider.next_split_provider();
    if (!factory) break;
    auto payloads = factory();
    if (payloads.empty()) break;
    ++out.range_count;
    for (auto& p : payloads) {
      auto* batch = dynamic_cast<duckdb_native_split_provider::split_payload*>(p.get());
      REQUIRE(batch != nullptr);
      REQUIRE(batch->scan_info != nullptr);
      for (auto& rg : batch->row_groups) {
        out.row_groups.push_back(std::move(rg));
      }
    }
  }
  return out;
}

// Minimal in-process sirius_ioctx so the provider can hold non-null IO handles.
// The slicing / metadata paths under test never read bytes, so the read API
// throws if exercised. Mirrors test_datasource_factory.cpp's mock_ioctx.
class stub_io_object : public sirius::io::sirius_io_object {
 public:
  const std::string& raw_file_cache_id() const noexcept override { return _id; }
  const std::string& object_path() const noexcept override { return _id; }
  std::size_t size() const noexcept override { return 0; }

 private:
  std::string _id = "stub";
};

class stub_ioctx : public sirius::io::sirius_ioctx {
 public:
  void shutdown() override {}

  std::shared_ptr<sirius::io::sirius_io_object> create_io_object(std::string) override
  {
    return std::make_shared<stub_io_object>();
  }

  std::unique_ptr<cudf::io::datasource> make_datasource(
    std::shared_ptr<sirius::io::sirius_io_object>) override
  {
    throw std::logic_error("stub_ioctx: IO not used in provider tests");
  }

  [[nodiscard]] bool supports(std::string_view) const override { return true; }

  std::size_t host_read_io(sirius::io::sirius_io_object&,
                           std::size_t,
                           std::size_t,
                           std::uint8_t*) override
  {
    throw std::logic_error("stub_ioctx: IO not used in provider tests");
  }

  void host_read_async_io(sirius::io::sirius_io_object&,
                          std::size_t,
                          std::size_t,
                          std::uint8_t*,
                          sirius::io::io_completion_handler) override
  {
    throw std::logic_error("stub_ioctx: IO not used in provider tests");
  }

  std::size_t device_read_io(sirius::io::sirius_io_object&,
                             std::size_t,
                             std::size_t,
                             std::uint8_t*,
                             rmm::cuda_stream_view) override
  {
    throw std::logic_error("stub_ioctx: IO not used in provider tests");
  }

  void device_read_async_io(sirius::io::sirius_io_object&,
                            std::size_t,
                            std::size_t,
                            std::uint8_t*,
                            rmm::cuda_stream_view,
                            sirius::io::io_completion_handler) override
  {
    throw std::logic_error("stub_ioctx: IO not used in provider tests");
  }

  void host_read_ranges_async_io(sirius::io::sirius_io_object&,
                                 std::vector<cudf::io::text::byte_range_info> const&,
                                 std::span<cudf::host_span<std::byte>>,
                                 sirius::io::io_completion_handler) override
  {
    throw std::logic_error("stub_ioctx: IO not used in provider tests");
  }

  cudf::io::text::byte_range_info compute_physical_range(cudf::io::text::byte_range_info logical,
                                                         std::size_t) const override
  {
    return logical;
  }
};

std::shared_ptr<sirius::io::sirius_ioctx> make_mock_ioctx()
{
  return std::make_shared<stub_ioctx>();
}

}  // namespace

TEST_CASE("duckdb_native_split_provider throws on null storage",
          "[scan][duckdb_native_split_provider]")
{
  duckdb_native_scan_info info;
  // info.storage stays null.
  REQUIRE_THROWS_AS((duckdb_native_split_provider{std::move(info), make_mock_ioctx()}),
                    std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws on null context",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  duckdb_native_scan_info info;
  info.storage = &storage;
  // info.context stays null.
  REQUIRE_THROWS_AS((duckdb_native_split_provider{std::move(info), make_mock_ioctx()}),
                    std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws on projected vectors size mismatch",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto info            = make_scan_info(storage, *con.context);
  info.projected_cols  = {real_col(0)};
  info.projected_types = {};  // empty — parallel-vector violation
  REQUIRE_THROWS_AS((duckdb_native_split_provider{std::move(info), make_mock_ioctx()}),
                    std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws on null io_ctx",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto info            = make_scan_info(storage, *con.context);
  info.projected_cols  = {real_col(0)};
  info.projected_types = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  REQUIRE_THROWS_AS((duckdb_native_split_provider{std::move(info), nullptr}),
                    std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws on empty db_path",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 16)");
  auto& storage = get_storage(con, "t");

  auto info            = make_scan_info(storage, *con.context);
  info.db_path         = "";  // override the helper's dummy path
  info.projected_cols  = {real_col(0)};
  info.projected_types = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  REQUIRE_THROWS_AS((duckdb_native_split_provider{std::move(info), make_mock_ioctx()}),
                    std::invalid_argument);
}

TEST_CASE("duckdb_native_split_provider throws when walker rejects the table",
          "[scan][duckdb_native_split_provider]")
{
  // HUGEINT is not supported by the gpu decoder → walker reports non-viable
  // → split_provider surfaces the rejection as a query error.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a HUGEINT)");
  exec_ok(con, "INSERT INTO t VALUES (1), (2), (3)");
  auto& storage = get_storage(con, "t");

  auto info            = make_scan_info(storage, *con.context);
  info.projected_cols  = {real_col(0)};
  info.projected_types = {sirius::logical_type::make(sirius::type_id::HUGEINT)};
  REQUIRE_THROWS_AS((duckdb_native_split_provider{std::move(info), make_mock_ioctx()}),
                    std::runtime_error);
}

TEST_CASE("duckdb_native_split_provider emits one batch for a small INTEGER table",
          "[scan][duckdb_native_split_provider]")
{
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  auto& storage = get_storage(con, "t");

  auto info                   = make_scan_info(storage, *con.context);
  info.projected_cols         = {real_col(0)};
  info.projected_types        = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info.approximate_batch_size = 0;  // single-batch fast path

  duckdb_native_split_provider provider{std::move(info), make_mock_ioctx()};
  REQUIRE(provider.has_more_splits());

  auto factory = provider.next_split_provider();
  REQUIRE(factory);
  auto payloads = factory();
  REQUIRE(payloads.size() == 1);

  auto* batch = dynamic_cast<duckdb_native_split_provider::split_payload*>(payloads[0].get());
  REQUIRE(batch != nullptr);
  REQUIRE(!batch->row_groups.empty());
  REQUIRE(batch->scan_info != nullptr);

  // After draining, both signals must report exhaustion: has_more_splits()
  // goes false and next_split_provider() returns an empty std::function.
  REQUIRE_FALSE(provider.has_more_splits());
  auto exhausted_factory = provider.next_split_provider();
  REQUIRE_FALSE(exhausted_factory);
}

TEST_CASE("duckdb_native_split_provider ranges cover all row groups exactly once",
          "[scan][duckdb_native_split_provider]")
{
  // STANDARD_VECTOR_SIZE is 2048; a row group is up to 122,880 rows in DuckDB
  // (60 vectors). 1.2M INTEGER rows guarantees several row groups, so the
  // provider hands out multiple parse ranges. Batch *caps* now live in the scan
  // operator's batch_coalescer (see test_duckdb_native_batch_coalescer); the
  // provider's job is only to slice row groups into ranges that, taken
  // together, cover every row group exactly once.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1200000)");
  auto& storage = get_storage(con, "t");

  auto info            = make_scan_info(storage, *con.context);
  info.projected_cols  = {real_col(0)};
  info.projected_types = {sirius::logical_type::make(sirius::type_id::INTEGER)};

  duckdb_native_split_provider provider{std::move(info), make_mock_ioctx()};
  auto result = drain_all(provider);

  REQUIRE(result.row_groups.size() >= 2);  // several row groups
  REQUIRE(result.range_count >= 1);

  std::sort(result.row_groups.begin(), result.row_groups.end(), [](auto const& a, auto const& b) {
    return a.row_group_index < b.row_group_index;
  });
  // Contiguous coverage from 0 with no gaps or duplicates.
  for (std::size_t i = 0; i < result.row_groups.size(); ++i) {
    REQUIRE(result.row_groups[i].row_group_index == i);
    REQUIRE(result.row_groups[i].row_count > 0);
  }
  // Absolute row offsets strictly increase with row-group index.
  for (std::size_t i = 1; i < result.row_groups.size(); ++i) {
    REQUIRE(result.row_groups[i].row_group_start > result.row_groups[i - 1].row_group_start);
  }

  REQUIRE_FALSE(provider.has_more_splits());
}

TEST_CASE("duckdb_native_split_provider populates payload with scan_info shared_ptr",
          "[scan][duckdb_native_split_provider]")
{
  // The payload's scan_info must alias the same data even across multiple
  // batches so downstream tasks observe a consistent projection.
  auto [db_owner, con] = sirius::make_test_db_and_connection();
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 400000)");
  auto& storage = get_storage(con, "t");

  auto info                   = make_scan_info(storage, *con.context);
  info.projected_cols         = {real_col(0)};
  info.projected_types        = {sirius::logical_type::make(sirius::type_id::INTEGER)};
  info.approximate_batch_size = 64 * 1024;

  duckdb_native_split_provider provider{std::move(info), make_mock_ioctx()};

  std::shared_ptr<op::scan::duckdb_native_scan_info const> first_scan_info;
  bool any = false;
  while (provider.has_more_splits()) {
    auto factory = provider.next_split_provider();
    if (!factory) break;
    auto payloads = factory();
    if (payloads.empty()) break;
    any         = true;
    auto* batch = dynamic_cast<duckdb_native_split_provider::split_payload*>(payloads[0].get());
    REQUIRE(batch != nullptr);
    REQUIRE(batch->scan_info != nullptr);
    if (!first_scan_info) {
      first_scan_info = batch->scan_info;
    } else {
      REQUIRE(batch->scan_info.get() == first_scan_info.get());
    }
  }
  REQUIRE(any);
}
