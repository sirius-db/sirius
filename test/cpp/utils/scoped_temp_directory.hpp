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
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace sirius::test {

inline duckdb::unique_ptr<duckdb::MaterializedQueryResult> query(duckdb::Connection& con,
                                                                 std::string const& sql)
{
  auto result = con.Query(sql);
  if (!result || result->HasError()) {
    throw std::runtime_error(sql + ": " + (result ? result->GetError() : "no result"));
  }
  return result;
}

class scoped_temp_directory {
 public:
  std::string directory;
  std::string path;

  scoped_temp_directory()
  {
    std::string pattern = (std::filesystem::temp_directory_path() / "sirius_test_XXXXXX").string();
    if (!::mkdtemp(pattern.data())) { throw std::runtime_error("mkdtemp failed"); }
    directory = pattern;
    path      = directory + "/strings.db";
  }
  scoped_temp_directory(scoped_temp_directory const&)            = delete;
  scoped_temp_directory& operator=(scoped_temp_directory const&) = delete;
  ~scoped_temp_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
};

}  // namespace sirius::test
