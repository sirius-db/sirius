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

// Tests for the decoder's host-backed segment lane (insert-delta staging):
// splits whose descriptors carry host_ptr must decode from those bytes alone,
// with no file reads and no datasource, and match SQL for the same rows.

#include "test_utils.hpp"

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <op/scan/duckdb_native_decoder.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/mvcc_chunk_mask.hpp>
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

/// File-backed database; the decoder requires a SingleFileBlockManager.
struct host_backed_test_db {
  std::string path;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

  host_backed_test_db()
  {
    static int counter = 0;
    path               = "/tmp/sirius_host_backed_decode_test_" + std::to_string(::getpid()) + "_" +
           std::to_string(counter++) + ".db";
    std::remove(path.c_str());
    std::remove((path + ".wal").c_str());
    db  = std::make_unique<duckdb::DuckDB>(path);
    con = std::make_unique<duckdb::Connection>(*db);
    con->Query("SET gpu_execution = false;");
  }

  ~host_backed_test_db()
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

projected_column real_col(duckdb::idx_t col_id)
{
  projected_column pc;
  pc.storage_idx = duckdb::StorageIndex(col_id);
  pc.is_rowid    = false;
  return pc;
}

std::vector<duckdb_row_group_metadata> walk_all(duckdb::DataTable& storage,
                                                duckdb::ClientContext& ctx,
                                                const std::vector<projected_column>& cols,
                                                const std::vector<sirius::logical_type>& types)
{
  auto plan = prepare_duckdb_native_walk(storage, ctx, cols, types);
  REQUIRE(plan.viable);
  auto range = walk_duckdb_native_row_group_range(plan, 0, plan.n_row_groups);
  REQUIRE(range.viable);
  return std::move(range.row_groups);
}

/// Copies each block-backed segment's payload to a host buffer and points
/// host_ptr at it; clear_block_ids additionally erases the block reference.
void host_back_segments(duckdb::ClientContext& ctx,
                        duckdb::DataTable& storage,
                        std::vector<duckdb_row_group_metadata>& row_groups,
                        std::vector<std::shared_ptr<void>>& keepalive,
                        bool clear_block_ids)
{
  auto& buffer_manager = duckdb::BufferManager::GetBufferManager(ctx);
  auto& block_manager  = storage.GetAttached().GetStorageManager().GetBlockManager();

  auto back_one = [&](duckdb_segment_descriptor& desc) {
    if (desc.block_id < 0 || desc.bytes_size == 0) { return; }
    auto handle = block_manager.RegisterBlock(desc.block_id);
    auto pin    = buffer_manager.Pin(handle);
    auto owned  = std::make_shared<std::vector<uint8_t>>(desc.bytes_size);
    std::memcpy(owned->data(), pin.Ptr() + desc.block_offset, desc.bytes_size);
    desc.host_ptr = owned->data();
    if (clear_block_ids) {
      desc.block_id     = INVALID_BLOCK;
      desc.block_offset = 0;
    }
    keepalive.push_back(std::move(owned));
  };

  for (auto& rg : row_groups) {
    for (auto& col : rg.columns) {
      for (auto& d : col.data_segments) {
        back_one(d);
      }
      for (auto& v : col.validity_segments) {
        back_one(v);
      }
    }
  }
}

struct expected_rows {
  std::vector<int32_t> id;
  std::vector<int64_t> value;
  std::vector<double> price;
  std::vector<std::string> name;
  std::vector<int32_t> opt;
  std::vector<bool> opt_null;
};

expected_rows fetch_expected(duckdb::Connection& con)
{
  expected_rows out;
  auto result = con.Query("SELECT id, value, price, name, opt FROM t ORDER BY rowid");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  while (auto chunk = result->Fetch()) {
    for (duckdb::idx_t r = 0; r < chunk->size(); ++r) {
      out.id.push_back(chunk->GetValue(0, r).GetValue<int32_t>());
      out.value.push_back(chunk->GetValue(1, r).GetValue<int64_t>());
      out.price.push_back(chunk->GetValue(2, r).GetValue<double>());
      out.name.push_back(chunk->GetValue(3, r).GetValue<std::string>());
      auto v = chunk->GetValue(4, r);
      out.opt_null.push_back(v.IsNull());
      out.opt.push_back(v.IsNull() ? 0 : v.GetValue<int32_t>());
    }
  }
  return out;
}

template <typename T>
std::vector<T> download(void const* d_ptr, std::size_t count, rmm::cuda_stream_view stream)
{
  std::vector<T> out(count);
  cudaMemcpyAsync(out.data(), d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream.value());
  cudaStreamSynchronize(stream.value());
  return out;
}

void require_matches_expected(cudf::table const& t,
                              expected_rows const& exp,
                              rmm::cuda_stream_view stream)
{
  auto const n = exp.id.size();
  REQUIRE(t.num_columns() == 5);
  REQUIRE(static_cast<std::size_t>(t.num_rows()) == n);

  REQUIRE(download<int32_t>(t.get_column(0).view().data<int32_t>(), n, stream) == exp.id);
  REQUIRE(download<int64_t>(t.get_column(1).view().data<int64_t>(), n, stream) == exp.value);
  REQUIRE(download<double>(t.get_column(2).view().data<double>(), n, stream) == exp.price);

  // Strings: offsets + chars.
  cudf::strings_column_view name_col(t.get_column(3).view());
  auto offsets = sirius::scan_test_utils::copy_string_offsets(name_col.offsets());
  REQUIRE(offsets.size() == n + 1);
  std::vector<char> chars;
  if (offsets.back() > 0) {
    chars.resize(static_cast<std::size_t>(offsets.back()));
    cudaMemcpyAsync(chars.data(),
                    name_col.chars_begin(cudf::get_default_stream()),
                    chars.size(),
                    cudaMemcpyDeviceToHost,
                    stream.value());
    cudaStreamSynchronize(stream.value());
  }
  for (std::size_t i = 0; i < n; ++i) {
    auto const start = static_cast<std::size_t>(offsets[i]);
    auto const end   = static_cast<std::size_t>(offsets[i + 1]);
    std::string actual;
    if (end > start) { actual.assign(chars.data() + start, chars.data() + end); }
    if (actual != exp.name[i]) { FAIL("name mismatch at row " << i); }
  }

  // Nullable column: values where valid, null mask bits everywhere.
  auto const& opt_col        = t.get_column(4);
  std::size_t expected_nulls = 0;
  for (auto is_null : exp.opt_null) {
    expected_nulls += is_null ? 1 : 0;
  }
  REQUIRE(static_cast<std::size_t>(opt_col.null_count()) == expected_nulls);
  auto opt_vals = download<int32_t>(opt_col.view().data<int32_t>(), n, stream);
  std::vector<uint32_t> null_words;
  if (opt_col.view().null_mask() != nullptr) {
    null_words = download<uint32_t>(opt_col.view().null_mask(), (n + 31) / 32, stream);
  }
  for (std::size_t i = 0; i < n; ++i) {
    bool const valid = null_words.empty() ? true : ((null_words[i / 32] >> (i % 32)) & 1u) != 0u;
    if (valid == exp.opt_null[i]) { FAIL("null-mask mismatch at row " << i); }
    if (valid && opt_vals[i] != exp.opt[i]) { FAIL("opt value mismatch at row " << i); }
  }
}

/// 130,000 rows: two row groups (122,880 + 7,120), several segments per
/// column, empty strings, and NULLs every 5th row in `opt`.
void build_table(duckdb::Connection& con)
{
  // Uncompressed checkpoint keeps segment payloads byte-identical to the
  // transient shape the insert-delta job stages.
  exec_ok(con, "SET force_compression='uncompressed'");
  exec_ok(con, "CREATE TABLE t(id INTEGER, value BIGINT, price DOUBLE, name VARCHAR, opt INTEGER)");
  exec_ok(con,
          "INSERT INTO t SELECT i::INTEGER, (i*100)::BIGINT, i*1.5, "
          "CASE WHEN i%97=0 THEN '' ELSE 'item_'||i END, "
          "CASE WHEN i%5=0 THEN NULL ELSE (i%1000)::INTEGER END "
          "FROM range(130000) tbl(i)");
  exec_ok(con, "CHECKPOINT");
}

std::vector<projected_column> all_cols()
{
  return {real_col(0), real_col(1), real_col(2), real_col(3), real_col(4)};
}

std::vector<sirius::logical_type> all_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::BIGINT),
          sirius::logical_type::make(sirius::type_id::DOUBLE),
          sirius::logical_type::make(sirius::type_id::VARCHAR),
          sirius::logical_type::make(sirius::type_id::INTEGER)};
}

}  // namespace

TEST_CASE("host-backed decode matches SQL with zero file reads",
          "[scan][duckdb_native_host_backed]")
{
  host_backed_test_db env;
  build_table(*env.con);
  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");

  auto cols  = all_cols();
  auto types = all_types();
  auto md    = walk_all(storage, *env.con->context, cols, types);
  auto exp   = fetch_expected(*env.con);

  duckdb_native_ingestible_table_info info;
  info.storage         = &storage;
  info.context         = env.con->context.get();
  info.projected_cols  = cols;
  info.projected_types = types;

  auto mem_mgr    = initialize_memory_manager();
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  std::vector<std::shared_ptr<void>> keepalive;

  SECTION("delta descriptor shape: block ids cleared, null datasource")
  {
    host_back_segments(*env.con->context, storage, md, keepalive, /*clear_block_ids=*/true);
    auto table =
      decode_duckdb_native_split(md, info, /*datasource=*/nullptr, *gpu_space, stream.view());
    require_matches_expected(*table, exp, stream.view());
  }

  SECTION("host_ptr takes precedence over a still-valid block_id")
  {
    // With a null datasource any fall-through to the file lane would throw, so
    // decoding successfully proves host_ptr wins.
    host_back_segments(*env.con->context, storage, md, keepalive, /*clear_block_ids=*/false);
    auto table =
      decode_duckdb_native_split(md, info, /*datasource=*/nullptr, *gpu_space, stream.view());
    require_matches_expected(*table, exp, stream.view());
  }
}

TEST_CASE("file reads staged without a datasource throw loudly",
          "[scan][duckdb_native_host_backed]")
{
  host_backed_test_db env;
  exec_ok(*env.con, "SET force_compression='uncompressed'");
  exec_ok(*env.con,
          "CREATE TABLE t(id INTEGER, value BIGINT, price DOUBLE, name VARCHAR, opt INTEGER)");
  exec_ok(*env.con,
          "INSERT INTO t SELECT i::INTEGER, i::BIGINT, i*1.0, 'x'||i, i::INTEGER "
          "FROM range(100) tbl(i)");
  exec_ok(*env.con, "CHECKPOINT");
  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");

  auto cols  = all_cols();
  auto types = all_types();
  auto md    = walk_all(storage, *env.con->context, cols, types);  // block-backed, not host-backed

  duckdb_native_ingestible_table_info info;
  info.storage         = &storage;
  info.context         = env.con->context.get();
  info.projected_cols  = cols;
  info.projected_types = types;

  auto mem_mgr    = initialize_memory_manager();
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  bool threw_datasource = false;
  try {
    auto table =
      decode_duckdb_native_split(md, info, /*datasource=*/nullptr, *gpu_space, stream.view());
  } catch (std::exception const& e) {
    threw_datasource = std::string(e.what()).find("datasource") != std::string::npos;
  }
  REQUIRE(threw_datasource);
}

TEST_CASE("materialize_table applies the mvcc keep-mask to metadata splits",
          "[scan][duckdb_native_host_backed]")
{
  host_backed_test_db env;
  build_table(*env.con);
  exec_ok(*env.con, "BEGIN TRANSACTION");
  auto& storage = resolve_storage(*env.con, "t");

  auto cols         = all_cols();
  auto types        = all_types();
  auto md           = walk_all(storage, *env.con->context, cols, types);
  auto exp          = fetch_expected(*env.con);
  auto const n_rows = exp.id.size();

  auto mem_mgr    = initialize_memory_manager();
  auto* gpu_space = sirius::scan_test_utils::get_space(*mem_mgr, cucascade::memory::Tier::GPU);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream stream;

  std::vector<std::shared_ptr<void>> keepalive;
  host_back_segments(*env.con->context, storage, md, keepalive, /*clear_block_ids=*/true);

  auto info             = std::make_unique<duckdb_native_ingestible_table_info>();
  info->storage         = &storage;
  info->context         = env.con->context.get();
  info->projected_cols  = cols;
  info->projected_types = types;
  auto ingestible       = make_ingestible(std::move(info));

  // Keep every third row.
  auto const n_words = (n_rows + 31) / 32;
  auto words         = std::make_shared<std::vector<std::uint32_t>>(n_words, 0u);
  std::size_t kept   = 0;
  for (std::size_t i = 0; i < n_rows; ++i) {
    if (i % 3 == 0) {
      (*words)[i / 32] |= (1u << (i % 32));
      ++kept;
    }
  }

  SECTION("kept rows survive, dropped rows disappear")
  {
    auto sinfo               = std::make_unique<duckdb_native_scan_info>();
    sinfo->row_groups        = std::move(md);
    sinfo->host_backed_only  = true;
    sinfo->staging_keepalive = keepalive;
    scan_operator_input input(std::shared_ptr<scan_info>(std::move(sinfo)));
    input.gpu_memory_space = gpu_space;
    input.mvcc_keep_mask   = sirius::scan_manager::mvcc_chunk_mask{
      std::shared_ptr<std::uint32_t[]>(words, words->data()), n_rows};

    auto result = ingestible->materialize_table(input, stream.view());
    auto view   = result.table.view();
    REQUIRE(static_cast<std::size_t>(view.num_rows()) == kept);
    auto ids = download<int32_t>(view.column(0).data<int32_t>(), kept, stream.view());
    for (std::size_t k = 0; k < kept; ++k) {
      if (ids[k] != exp.id[k * 3]) { FAIL("kept-row mismatch at " << k); }
    }
  }

  SECTION("row-count mismatch throws")
  {
    auto sinfo               = std::make_unique<duckdb_native_scan_info>();
    sinfo->row_groups        = std::move(md);
    sinfo->host_backed_only  = true;
    sinfo->staging_keepalive = keepalive;
    scan_operator_input input(std::shared_ptr<scan_info>(std::move(sinfo)));
    input.gpu_memory_space = gpu_space;
    input.mvcc_keep_mask   = sirius::scan_manager::mvcc_chunk_mask{
      std::shared_ptr<std::uint32_t[]>(words, words->data()), n_rows - 1};

    REQUIRE_THROWS(ingestible->materialize_table(input, stream.view()));
  }
}
