/*
 * Copyright 2025, Sirius Contributors.
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

// Regression coverage for the public FFI Context and Fragment lifecycle (src/sirius_ffi.cpp).
// Fragment::Impl::end_lifecycle() must roll back a build() failure while the transaction is
// still open, not commit it (a7bb47e2).
//
// The public FFI surface links only DuckDB's substrait consumer (no substrait-plan-from-SQL
// helper, no raw-SQL passthrough), so no test here can construct a valid Fragment or inspect
// catalog state after a failed build(). Instead these tests use a declared column type name
// that TransformStringToLogicalType() can never resolve, which fails build() inside
// resolve_inputs() before `substrait_plan` is ever parsed — and check the one thing observable
// through the public API: that end_lifecycle() leaves the connection able to start and fail a
// second, independent Fragment cleanly.

#include "config.hpp"
#include "log/sink.hpp"
#include "sirius/ffi.hpp"

#include <catch.hpp>
#include <duckdb/common/exception/transaction_exception.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace {

// Small 2GB-GPU/4GB-host config shared with other [isolated_context] tests.
fs::path isolated_memory_config_path()
{
  std::source_location loc = std::source_location::current();
  return fs::path(loc.file_name()).parent_path().parent_path() / "scan" / "memory.yaml";
}

std::optional<std::string> saved_env(const char* name)
{
  if (auto const* value = std::getenv(name)) { return value; }
  return std::nullopt;
}

void restore_env(const char* name, std::optional<std::string> const& value)
{
  if (value) {
    setenv(name, value->c_str(), 1);
  } else {
    unsetenv(name);
  }
}

struct logging_state_guard {
  explicit logging_state_guard(fs::path root) : test_root(std::move(root)) {}

  ~logging_state_guard()
  {
    duckdb::Config::LOG_BACKEND.swap(backend);
    duckdb::Config::LOG_DIR.swap(log_dir);
    duckdb::Config::LOG_LEVEL.swap(level);
    if (sirius::log::get_sink() != sink) { sirius::log::set_sink(sink); }
    restore_env("SIRIUS_LOG_BACKEND", env_backend);
    restore_env("SIRIUS_LOG_DIR", env_log_dir);
    restore_env("SIRIUS_LOG_LEVEL", env_level);
    std::error_code error;
    fs::remove_all(test_root, error);
  }

  std::string backend{duckdb::Config::LOG_BACKEND};
  std::string log_dir{duckdb::Config::LOG_DIR};
  std::string level{duckdb::Config::LOG_LEVEL};
  std::shared_ptr<sirius::log::sink> sink{sirius::log::get_sink()};
  std::optional<std::string> env_backend{saved_env("SIRIUS_LOG_BACKEND")};
  std::optional<std::string> env_log_dir{saved_env("SIRIUS_LOG_DIR")};
  std::optional<std::string> env_level{saved_env("SIRIUS_LOG_LEVEL")};
  fs::path test_root;
};

void declare_unresolvable_column(sirius::ffi::Fragment& fragment, const std::string& type_name)
{
  fragment.declare_input_column(0, "a", type_name);
}

// Both TransactionContext::Commit() and ::Rollback() clear current_transaction before doing any
// work that can throw, so "does BeginTransaction() work afterward" cannot tell the old
// commit-on-failure bug apart from the fix. What both bugs share is that a broken/skipped
// end_lifecycle() would leave the transaction open, and the next BeginTransaction() would then
// throw TransactionException("cannot start a transaction within a transaction") — that's the
// one thing worth asserting against here.
void require_build_fails_without_transaction_exception(sirius::ffi::Fragment& fragment)
{
  bool threw_transaction_exception = false;
  bool threw_other                 = false;
  try {
    fragment.build("");
  } catch (const duckdb::TransactionException&) {
    threw_transaction_exception = true;
  } catch (...) {
    threw_other = true;
  }
  REQUIRE_FALSE(threw_transaction_exception);
  REQUIRE(threw_other);
}

}  // namespace

TEST_CASE("Fragment::build() failure during resolve_inputs() rolls back cleanly",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  auto first = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*first, "not_a_real_type_xyz");
  REQUIRE_THROWS(first->build(""));

  auto second = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*second, "also_not_a_real_type_xyz");
  require_build_fails_without_transaction_exception(*second);
}

TEST_CASE("Fragment destroyed between a failed build() and reuse also closes the lifecycle cleanly",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  // Exercises ~Fragment::Impl() -> end_lifecycle() (rather than the catch-block call in
  // build()): the failed fragment goes out of scope with no further use.
  {
    auto first = sirius::ffi::make_fragment(*context);
    declare_unresolvable_column(*first, "not_a_real_type_xyz");
    REQUIRE_THROWS(first->build(""));
  }

  auto second = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*second, "also_not_a_real_type_xyz");
  require_build_fails_without_transaction_exception(*second);
}

TEST_CASE("FFI context restores logging settings when sink construction fails",
          "[isolated_context][sirius_ffi]")
{
  auto const test_root = fs::temp_directory_path() / "sirius-ffi-log-dir-rollback-test";
  logging_state_guard restore{test_root};
  fs::remove_all(test_root);
  fs::create_directories(test_root);
  auto const blocker = test_root / "not-a-directory";
  std::ofstream(blocker) << "file";
  REQUIRE(fs::is_regular_file(blocker));
  auto const invalid_dir = blocker / "child";

  duckdb::Config::LOG_BACKEND = "noop";
  duckdb::Config::LOG_DIR     = (test_root / "previous").string();
  duckdb::Config::LOG_LEVEL   = "warn";
  setenv("SIRIUS_LOG_BACKEND", "spdlog", 1);
  setenv("SIRIUS_LOG_DIR", invalid_dir.string().c_str(), 1);
  setenv("SIRIUS_LOG_LEVEL", "debug", 1);

  REQUIRE_THROWS(sirius::ffi::make_context_from_config(isolated_memory_config_path().string()));
  REQUIRE(duckdb::Config::LOG_BACKEND == "noop");
  REQUIRE(duckdb::Config::LOG_DIR == (test_root / "previous").string());
  REQUIRE(duckdb::Config::LOG_LEVEL == "warn");
  REQUIRE(sirius::log::get_sink() == restore.sink);
}
