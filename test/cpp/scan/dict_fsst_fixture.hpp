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

#pragma once

#include <duckdb.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::test::dict_fsst {

inline duckdb::unique_ptr<duckdb::MaterializedQueryResult> query(duckdb::Connection& con,
                                                                 std::string const& sql)
{
  auto result = con.Query(sql);
  if (!result || result->HasError()) {
    throw std::runtime_error(sql + ": " + (result ? result->GetError() : "no result"));
  }
  return result;
}

inline uint32_t align8(uint32_t value) { return (value + 7u) & ~7u; }

struct segment {
  uint64_t start;
  uint32_t rows;
  std::vector<uint8_t> bytes;
  uint32_t dict_size;
  uint32_t dict_count;
  uint8_t mode;
  uint8_t lengths_width;
  uint8_t indices_width;
  uint32_t symtab_size;
  bool all_null_validity = false;

  uint32_t lengths_offset() const { return align8(align8(16 + dict_size) + symtab_size); }
  uint32_t old_indices_offset() const
  {
    return align8(lengths_offset() + (dict_count * lengths_width + 7) / 8);
  }
  uint32_t indices_offset() const
  {
    return align8(lengths_offset() + ((dict_count + 31) / 32 * 32) * lengths_width / 8);
  }
};

struct shape {
  char const* name;
  std::string expression;
  uint32_t rows;
};

inline std::vector<shape> candidate_shapes()
{
  return {
    {"dictionary_shift",
     "CASE WHEN i % 101 = 0 THEN NULL WHEN i % 101 = 1 THEN '' ELSE 's' || lpad((i % "
     "101)::VARCHAR, 3, '0') END",
     10000},
    {"fsst_dictionary_shift",
     "CASE WHEN i % 101 = 0 THEN NULL WHEN i % 101 = 1 THEN '' ELSE repeat('abcdefgh', 32) || "
     "lpad((i % 101)::VARCHAR, 4, '0') END",
     10000},
    {"fsst_only", "repeat('abcdefgh', 32) || lpad(i::VARCHAR, 8, '0')", 10000},
    {"boundary",
     "CASE WHEN i % 31 = 0 THEN NULL ELSE repeat('x', 128) || lpad((i % 31)::VARCHAR, 3, '0') END",
     10000},
    {"all_null", "NULL::VARCHAR", 10000},
    {"null_empty", "CASE WHEN i % 2 = 0 THEN NULL ELSE '' END", 10000},
    {"multi_segment",
     "CASE WHEN i % 31 = 0 THEN NULL ELSE repeat('x', 128) || lpad((i % 31)::VARCHAR, 3, '0') END",
     150000},
  };
}

class fixture {
 public:
  std::string directory;
  std::string path;

  fixture()
  {
    std::string pattern =
      (std::filesystem::temp_directory_path() / "sirius_dict_fsst_XXXXXX").string();
    if (!::mkdtemp(pattern.data())) { throw std::runtime_error("mkdtemp failed"); }
    directory = pattern;
    path      = directory + "/strings.db";
  }
  fixture(fixture const&) = delete;
  ~fixture()
  {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }

  void create(shape const& input)
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    query(con, "SET SESSION gpu_execution = false");
    query(con, "ATTACH '" + path + "' AS disk (STORAGE_VERSION 'latest', BLOCK_SIZE 16384)");
    query(con, "USE disk");
    query(con, "SET threads = 1");
    query(con, "PRAGMA force_compression = 'dict_fsst'");
    query(con, "CREATE TABLE t(s VARCHAR)");
    query(con,
          "INSERT INTO t SELECT " + input.expression + " FROM range(" + std::to_string(input.rows) +
            ") r(i)");
    query(con, "CHECKPOINT");
  }

  std::vector<segment> read_segments()
  {
    duckdb::DuckDB db(path);
    duckdb::Connection con(db);
    query(con, "SET SESSION gpu_execution = false");
    auto info =
      query(con,
            "SELECT start, count, block_id, block_offset, compression, row_group_id FROM "
            "pragma_storage_info('t') WHERE segment_type = 'VARCHAR' ORDER BY row_group_id, start");
    auto validity = query(con,
                          "SELECT row_group_id, start, count, compression, stats FROM "
                          "pragma_storage_info('t') WHERE segment_type = 'VALIDITY'");
    query(con, "BEGIN");
    auto& catalog = duckdb::Catalog::GetCatalog(*con.context, "");
    auto& manager = catalog.GetAttached().GetStorageManager().GetBlockManager();
    auto& buffers = duckdb::BufferManager::GetBufferManager(*con.context);
    std::vector<segment> result;
    int64_t previous_group = -1;
    uint64_t group_start   = 0;
    uint64_t next_start    = 0;
    for (duckdb::idx_t i = 0; i < info->RowCount(); ++i) {
      auto const codec = info->GetValue(4, i).ToString();
      if (codec != "DICT_FSST") { throw std::runtime_error("fixture codec is " + codec); }
      auto const block  = info->GetValue(2, i).GetValue<int64_t>();
      auto const offset = info->GetValue(3, i).GetValue<uint32_t>();
      if (block < 0 || offset + 16 > manager.GetBlockSize()) {
        throw std::runtime_error("fixture segment is not persisted");
      }
      auto handle = manager.RegisterBlock(block);
      auto pin    = buffers.Pin(handle);
      segment seg{};
      auto group = info->GetValue(5, i).GetValue<int64_t>();
      if (group != previous_group) {
        group_start    = next_start;
        previous_group = group;
      }
      seg.start  = group_start + info->GetValue(0, i).GetValue<uint64_t>();
      seg.rows   = info->GetValue(1, i).GetValue<uint32_t>();
      next_start = seg.start + seg.rows;
      // CONSTANT all-NULL validity is staged as zero bits by the native decoder.
      // EMPTY validity instead leaves nullness to the DICT_FSST inline indices.
      auto local_start = info->GetValue(0, i).GetValue<uint64_t>();
      for (duckdb::idx_t v = 0; v < validity->RowCount(); ++v) {
        if (validity->GetValue(0, v).GetValue<int64_t>() != group) { continue; }
        auto begin = validity->GetValue(1, v).GetValue<uint64_t>();
        auto end   = begin + validity->GetValue(2, v).GetValue<uint64_t>();
        if (begin <= local_start && end >= local_start + seg.rows) {
          auto codec = validity->GetValue(3, v).ToString();
          if (codec != "Constant" && codec != "Empty Validity") {
            throw std::runtime_error("fixture validity codec is " + codec);
          }
          seg.all_null_validity =
            codec == "Constant" &&
            validity->GetValue(4, v).ToString().find("Has No Null: false") != std::string::npos;
          break;
        }
      }
      seg.bytes.assign(pin.Ptr() + offset, pin.Ptr() + manager.GetBlockSize());
      std::memcpy(&seg.dict_size, seg.bytes.data(), 4);
      std::memcpy(&seg.dict_count, seg.bytes.data() + 4, 4);
      seg.mode          = seg.bytes[8];
      seg.lengths_width = seg.bytes[9];
      seg.indices_width = seg.bytes[10];
      std::memcpy(&seg.symtab_size, seg.bytes.data() + 12, 4);
      result.push_back(std::move(seg));
    }
    query(con, "COMMIT");
    return result;
  }

  std::vector<std::optional<std::string>> cpu_rows()
  {
    duckdb::DuckDB db(path);
    duckdb::Connection con(db);
    query(con, "SET SESSION gpu_execution = false");
    auto rows = query(con, "SELECT s FROM t ORDER BY rowid");
    std::vector<std::optional<std::string>> result;
    for (duckdb::idx_t i = 0; i < rows->RowCount(); ++i) {
      auto value = rows->GetValue(0, i);
      result.push_back(value.IsNull() ? std::nullopt
                                      : std::optional<std::string>(value.ToString()));
    }
    return result;
  }
};

}  // namespace sirius::test::dict_fsst
