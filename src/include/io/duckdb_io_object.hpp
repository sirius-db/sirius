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
 * Carries the (cache_id, size) handle the prefetching cache uses to dedupe
 * inserts. Block-id → byte-offset translation is done by the scan path via
 * @c duckdb_block_payload_offset; this class does not own the layout.
 *
 * The ctor canonicalizes @p path so different spellings of the same file
 * collapse to one cache key. @c size() is captured at construction — a live
 * DuckDB writer may grow the file, so write-aware callers must reconstruct.
 */
class duckdb_io_object final : public sirius_io_object {
 public:
  /**
   * @param path  Path to the .db file. Canonicalized; used as cache id + stat target.
   * @throws std::invalid_argument if @p path is empty.
   * @throws std::filesystem::filesystem_error if the file is missing or unreadable.
   */
  explicit duckdb_io_object(std::string const& path);

  ~duckdb_io_object() override = default;

  duckdb_io_object(const duckdb_io_object&)            = delete;
  duckdb_io_object& operator=(const duckdb_io_object&) = delete;
  duckdb_io_object(duckdb_io_object&&)                 = delete;
  duckdb_io_object& operator=(duckdb_io_object&&)      = delete;

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
