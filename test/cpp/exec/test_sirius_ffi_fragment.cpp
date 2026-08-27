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

// Regression coverage for Fragment::Impl::end_lifecycle() (src/sirius_ffi.cpp): a build()
// failure while the transaction is still open must roll it back, not commit it (a7bb47e2).
//
// The public FFI surface links only DuckDB's substrait consumer (no substrait-plan-from-SQL
// helper, no raw-SQL passthrough), so no test here can construct a valid Fragment or inspect
// catalog state after a failed build(). Instead these tests use a declared column type name
// that TransformStringToLogicalType() can never resolve, which fails build() inside
// resolve_inputs() before `substrait_plan` is ever parsed — and check the one thing observable
// through the public API: that end_lifecycle() leaves the connection able to start and fail a
// second, independent Fragment cleanly.

#include "sirius_ffi.hpp"

#include <catch.hpp>
#include <duckdb/common/exception/transaction_exception.hpp>

#include <cstdint>
#include <filesystem>
#include <source_location>
#include <string>

namespace fs = std::filesystem;

namespace {

// Small 2GB-GPU/4GB-host config shared with other [isolated_context] tests.
fs::path isolated_memory_config_path()
{
  std::source_location loc = std::source_location::current();
  return fs::path(loc.file_name()).parent_path().parent_path() / "scan" / "memory.yaml";
}

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
