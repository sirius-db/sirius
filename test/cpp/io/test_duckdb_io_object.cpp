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

#include <catch.hpp>
#include <io/duckdb_io_object.hpp>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace sirius::io;

namespace {

class scoped_tempfile {
 public:
  explicit scoped_tempfile(const std::string& contents)
  {
    // mkstemp mutates the template; std::string::data() is writable since C++17.
    std::string tmpl =
      (std::filesystem::temp_directory_path() / "sirius_io_object_XXXXXX").string();
    int fd = ::mkstemp(tmpl.data());
    REQUIRE(fd >= 0);
    ::close(fd);
    _path = tmpl;
    std::ofstream out(_path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out << contents;
    out.flush();
    REQUIRE(out.good());
  }
  ~scoped_tempfile()
  {
    if (!_path.empty()) std::remove(_path.c_str());
  }
  scoped_tempfile(const scoped_tempfile&)            = delete;
  scoped_tempfile& operator=(const scoped_tempfile&) = delete;

  const std::string& path() const { return _path; }

 private:
  std::string _path;
};

}  // namespace

TEST_CASE("duckdb_io_object reports stat-derived size", "[io][duckdb_io_object]")
{
  scoped_tempfile tf{"0123456789"};
  duckdb_io_object obj{tf.path()};
  // Cache id is the canonical form of the input path (resolves any symlinks
  // in the prefix, e.g. /tmp -> /private/tmp on some platforms).
  REQUIRE(obj.raw_file_cache_id() == std::filesystem::canonical(tf.path()).string());
  REQUIRE(obj.size() == 10);
}

TEST_CASE("duckdb_io_object canonicalizes equivalent path spellings to the same cache id",
          "[io][duckdb_io_object]")
{
  scoped_tempfile tf{"abcdef"};
  // Construct one from the path mkstemp returned and one with a redundant
  // /./ in the middle; canonical() must collapse both to the same string.
  std::string with_dot = std::filesystem::path(tf.path()).parent_path().string() + "/./" +
                         std::filesystem::path(tf.path()).filename().string();
  duckdb_io_object a{tf.path()};
  duckdb_io_object b{with_dot};
  REQUIRE(a.raw_file_cache_id() == b.raw_file_cache_id());
}

TEST_CASE("duckdb_io_object handles empty file", "[io][duckdb_io_object]")
{
  scoped_tempfile tf{""};
  duckdb_io_object obj{tf.path()};
  REQUIRE(obj.size() == 0);
}

TEST_CASE("duckdb_io_object rejects empty path", "[io][duckdb_io_object]")
{
  REQUIRE_THROWS_AS(duckdb_io_object{std::string{}}, std::invalid_argument);
}

TEST_CASE("duckdb_io_object throws when file is missing", "[io][duckdb_io_object]")
{
  REQUIRE_THROWS_AS(duckdb_io_object{"/this/path/does/not/exist/sirius_test"}, std::runtime_error);
}
