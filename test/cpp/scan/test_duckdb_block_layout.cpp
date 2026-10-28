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

// Pins duckdb_block_payload_offset against BufferManager::Pin so the
// FILE_HEADER_SIZE * 3 constant and the per-block stride stay correct.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/single_file_block_manager.hpp>
#include <duckdb/storage/storage_info.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <op/scan/duckdb_block_layout.hpp>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_block_layout_XXXXXX";
    int fd      = ::mkstemp(tmpl);
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl);  // duckdb open expects the path to not exist
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

void exec_ok(duckdb::Connection& con, const std::string& q)
{
  auto r = con.Query(q);
  REQUIRE(r);
  if (r->HasError()) {
    INFO("query failed: " << q << "\n  error: " << r->GetError());
    REQUIRE_FALSE(r->HasError());
  }
}

duckdb::DataTable& get_storage(duckdb::Connection& con, const std::string& table_name)
{
  exec_ok(con, "BEGIN TRANSACTION");
  auto& ctx     = *con.context;
  auto& catalog = duckdb::Catalog::GetCatalog(ctx, "");
  duckdb::CatalogTransaction txn(catalog, ctx);
  auto& schema = catalog.GetSchema(txn, "main");
  auto entry =
    schema.GetEntry(txn, duckdb::CatalogType::TABLE_ENTRY, duckdb::Identifier(table_name));
  REQUIRE(entry);
  return entry->Cast<duckdb::DuckTableEntry>().GetStorage();
}

}  // namespace

TEST_CASE("duckdb_block_payload_offset uses FILE_HEADER_SIZE * 3 as BLOCK_START",
          "[scan][duckdb_block_layout]")
{
  REQUIRE(duckdb::Storage::FILE_HEADER_SIZE == 4096);
}

TEST_CASE("duckdb_block_payload_offset matches BufferManager::Pin pointer",
          "[scan][duckdb_block_layout]")
{
  scoped_temp_db_path tmp;
  duckdb::DuckDB db(tmp.path());
  duckdb::Connection con(db);
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  exec_ok(con, "CHECKPOINT");

  auto& storage = get_storage(con, "t");
  auto& sm      = storage.GetAttached().GetStorageManager();
  auto& bm      = static_cast<duckdb::SingleFileBlockManager&>(sm.GetBlockManager());
  auto& bufmgr  = duckdb::BufferManager::GetBufferManager(*con.context);

  const auto block_header_size = bm.GetBlockHeaderSize();
  const auto block_size        = bm.GetBlockSize();

  std::size_t offset0 = sirius::op::scan::duckdb_block_payload_offset(bm, 0);
  REQUIRE(offset0 ==
          static_cast<std::size_t>(duckdb::Storage::FILE_HEADER_SIZE) * 3 + block_header_size);

  auto handle = bm.RegisterBlock(0);
  auto pinned = bufmgr.Pin(handle);
  REQUIRE(pinned.Ptr() != nullptr);

  std::vector<uint8_t> from_disk(block_size);
  {
    std::ifstream f(tmp.path(), std::ios::binary);
    REQUIRE(f.good());
    f.seekg(static_cast<std::streamoff>(offset0));
    f.read(reinterpret_cast<char*>(from_disk.data()), static_cast<std::streamsize>(block_size));
    REQUIRE(static_cast<std::size_t>(f.gcount()) == block_size);
  }
  REQUIRE(std::memcmp(pinned.Ptr(), from_disk.data(), block_size) == 0);
}

TEST_CASE("duckdb_block_payload_offset advances by block_size+header per block",
          "[scan][duckdb_block_layout]")
{
  scoped_temp_db_path tmp;
  duckdb::DuckDB db(tmp.path());
  duckdb::Connection con(db);
  exec_ok(con, "CREATE TABLE t(a INTEGER)");
  exec_ok(con, "INSERT INTO t SELECT range FROM range(0, 1024)");
  exec_ok(con, "CHECKPOINT");

  auto& storage = get_storage(con, "t");
  auto& sm      = storage.GetAttached().GetStorageManager();
  auto& bm      = static_cast<duckdb::SingleFileBlockManager&>(sm.GetBlockManager());

  const std::size_t alloc =
    static_cast<std::size_t>(bm.GetBlockSize()) + static_cast<std::size_t>(bm.GetBlockHeaderSize());

  const auto base = sirius::op::scan::duckdb_block_payload_offset(bm, 0);
  REQUIRE(sirius::op::scan::duckdb_block_payload_offset(bm, 1) == base + alloc);
  REQUIRE(sirius::op::scan::duckdb_block_payload_offset(bm, 7) == base + 7 * alloc);
}
