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

#include "io/duckdb_db_io_object.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>

namespace sirius::io {

duckdb_db_io_object::duckdb_db_io_object(std::string path) : _absolute_path(), _size_bytes(0)
{
  if (path.empty()) { throw std::invalid_argument("duckdb_db_io_object: path must be non-empty"); }
  // canonical() throws filesystem_error if the file doesn't exist or any
  // intermediate symlink is broken; that's the right behavior for a cache
  // key (won't dedupe a path we can't read).
  auto const canonical = std::filesystem::canonical(path);
  _absolute_path       = canonical.string();
  _size_bytes          = std::filesystem::file_size(canonical);
}

}  // namespace sirius::io
