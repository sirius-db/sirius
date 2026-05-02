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

#include "io/types.hpp"

#include <cstddef>
#include <string>

namespace sirius::io {

/**
 * @brief @c sirius_io_object subclass identifying a DuckDB `.db` file.
 *
 * Provides the (cache_id, size) handle the prefetching cache needs to dedupe
 * inserts and bound reads. The block-id → byte-offset translation that drives
 * actual scan reads is owned by the caller (the duckdb-native scan path),
 * which already holds the `BlockManager`'s layout.
 *
 * The constructor canonicalizes the input path via @c std::filesystem::canonical
 * so two callers passing the same file under different spellings (relative vs
 * absolute, with intermediate symlinks) land on the same cache key.
 *
 * `size()` is captured at construction. A live DuckDB session may grow the
 * file as new blocks are allocated; the read-only OLAP scan path freezes the
 * size at scan-start, which is correct for that use case but means write-
 * aware callers should construct fresh objects when re-reading.
 */
class duckdb_db_io_object : public sirius_io_object {
 public:
  /**
   * @param path  Path to the .db file. Resolved to a canonical absolute path
   *              via @c std::filesystem::canonical and used as both the cache
   *              id and the stat target.
   * @throws std::invalid_argument if @p path is empty.
   * @throws std::filesystem::filesystem_error if the file is missing, an
   *         intermediate symlink is broken, or permission to resolve is denied.
   */
  explicit duckdb_db_io_object(std::string path);

  ~duckdb_db_io_object() override = default;

  duckdb_db_io_object(const duckdb_db_io_object&)            = delete;
  duckdb_db_io_object& operator=(const duckdb_db_io_object&) = delete;
  duckdb_db_io_object(duckdb_db_io_object&&)                 = delete;
  duckdb_db_io_object& operator=(duckdb_db_io_object&&)      = delete;

  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override
  {
    return _absolute_path;
  }

  [[nodiscard]] const std::string& object_path() const noexcept override { return _absolute_path; }

  [[nodiscard]] std::size_t size() const noexcept override { return _size_bytes; }

 private:
  std::string _absolute_path;
  std::size_t _size_bytes;
};

}  // namespace sirius::io
