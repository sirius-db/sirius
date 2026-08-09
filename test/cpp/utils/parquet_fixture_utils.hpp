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

/**
 * @file parquet_fixture_utils.hpp
 * @brief Shared pieces for tests that generate their own parquet files.
 *
 * Several test files build a scratch directory, write parquet into it with a
 * Sirius-disabled DuckDB, and interpolate the resulting paths into SQL. Each
 * had grown its own copy of the same three details, and the details are easy to
 * get subtly wrong — see the notes on each below.
 */

#include <catch.hpp>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <system_error>

namespace sirius::test {

/// Sets SIRIUS_DISABLE=1 for a scope, restoring whatever was there before.
///
/// The harness deliberately keeps SIRIUS_DISABLE=1 so untagged tests' DuckDB
/// instances do not auto-initialize a SiriusContext (see
/// scoped_sirius_disable_clear in test_plan_printer.cpp). Plain
/// setenv/unsetenv therefore leaks a changed global into later tests, and a
/// REQUIRE that throws in between skips the unset entirely. Restoring in a
/// destructor fixes both.
struct scoped_sirius_disable {
  scoped_sirius_disable()
  {
    if (char const* value = ::getenv("SIRIUS_DISABLE")) { _saved = value; }
    ::setenv("SIRIUS_DISABLE", "1", 1);
  }
  ~scoped_sirius_disable()
  {
    if (_saved) {
      ::setenv("SIRIUS_DISABLE", _saved->c_str(), 1);
    } else {
      ::unsetenv("SIRIUS_DISABLE");
    }
  }
  scoped_sirius_disable(scoped_sirius_disable const&)            = delete;
  scoped_sirius_disable& operator=(scoped_sirius_disable const&) = delete;

 private:
  std::optional<std::string> _saved;
};

/// A single-quoted SQL literal with embedded quotes doubled.
///
/// Paths are interpolated into generated SQL, and a TMPDIR containing a quote
/// is legal — it would otherwise close the literal early and break every
/// statement built from it.
inline std::string sql_literal(std::string const& value)
{
  std::string out = "'";
  for (char const c : value) {
    if (c == '\'') { out.push_back('\''); }
    out.push_back(c);
  }
  out.push_back('\'');
  return out;
}

/// A uniquely-named scratch directory, removed on destruction.
///
/// The name mixes @p tag, the pid and a process-local counter. The directory is
/// cleared before use rather than merely created: pids are reused (pid 1 in a
/// container makes that likely), and create_directories accepts an existing
/// directory, so a previous run's files could otherwise be read in place of the
/// ones about to be written.
class scratch_dir {
 public:
  explicit scratch_dir(std::string const& tag)
  {
    static std::atomic<unsigned> counter{0};
    _path = std::filesystem::temp_directory_path() /
            ("sirius_test_" + tag + "_" + std::to_string(::getpid()) + "_" +
             std::to_string(counter.fetch_add(1)));
    std::error_code ec;
    std::filesystem::remove_all(_path, ec);
    std::filesystem::create_directories(_path);
  }

  /// Best-effort: a destructor is noexcept, so a directory that cannot be
  /// removed must not become a std::terminate.
  ~scratch_dir()
  {
    std::error_code ec;
    std::filesystem::remove_all(_path, ec);
  }

  scratch_dir(scratch_dir const&)            = delete;
  scratch_dir& operator=(scratch_dir const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const noexcept { return _path; }

  /// Absolute path of @p name inside the directory.
  [[nodiscard]] std::string file(std::string const& name) const { return (_path / name).string(); }

  /// @ref file as a quote-safe SQL literal.
  [[nodiscard]] std::string file_literal(std::string const& name) const
  {
    return sql_literal(file(name));
  }

 private:
  std::filesystem::path _path;
};

}  // namespace sirius::test
