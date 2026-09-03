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

// Regression coverage for the transaction handling in src/sirius_ffi.cpp:
//
// 1. Fragment::Impl::end_lifecycle(): a build() failure while the transaction is still open must
//    roll it back, not commit it (a7bb47e2).
// 2. lower_substrait(): Fragment::build() commits its view-creation transaction before it lowers
//    the plan, and DuckDB 1.5.5 throws "TransactionContext::ActiveTransaction called without
//    active transaction" from every catalog lookup made without one — so lowering must own a
//    transaction of its own.
//
// The public FFI surface links only DuckDB's substrait consumer (no substrait-plan-from-SQL
// helper, no raw-SQL passthrough), so a valid plan has to be built here by hand with the bundled
// protobuf, and catalog state after a failed build() cannot be inspected. The end_lifecycle()
// tests therefore use a declared column type name that TransformStringToLogicalType() can never
// resolve, which fails build() inside resolve_inputs() before `substrait_plan` is ever parsed —
// and check the one thing observable through the public API: that end_lifecycle() leaves the
// connection able to start and fail a second, independent Fragment cleanly.

#include "sirius_ffi.hpp"

#include <catch.hpp>
#include <duckdb/common/exception/transaction_exception.hpp>
#include <substrait/plan.pb.h>

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

// A plan whose only read is the engine's stream view for input stream `stream_id`, projecting a
// single nullable INTEGER column `a` — the shape a front end emits where it would otherwise emit
// a file scan. Mirrors what Fragment::build() creates: `CREATE VIEW sirius_stream_<id> AS
// SELECT * FROM sirius_stream_source(<id>)`.
std::string stream_read_plan(std::uint64_t stream_id)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");

  auto* read   = root->mutable_input()->mutable_read();
  auto* schema = read->mutable_base_schema();
  schema->add_names("a");
  auto* row_type = schema->mutable_struct_();
  row_type->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
  row_type->add_types()->mutable_i32()->set_nullability(substrait::Type::NULLABILITY_NULLABLE);
  read->mutable_named_table()->add_names(*sirius::ffi::stream_view_name(stream_id));

  return plan.SerializeAsString();
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

// Exercises the success path the two cases above never reach: a well-formed plan over a declared
// input stream. Fragment::build() commits its own transaction (type resolution + CREATE VIEW)
// before opening the query lifecycle and only then calls lower_substrait(), so the Substrait
// consumer's view lookup and the planner's binding run with no ambient transaction. Without
// lower_substrait() owning one, DuckDB 1.5.5 fails this build() with
// "TransactionContext::ActiveTransaction called without active transaction".
TEST_CASE("Fragment::build() lowers its Substrait plan without an ambient transaction",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  {
    auto first = sirius::ffi::make_fragment(*context);
    first->declare_input_column(0, "a", "INTEGER");
    REQUIRE_NOTHROW(first->build(stream_read_plan(0)));
  }

  // The transaction lower_substrait() opened must be closed again on return: the next build()
  // begins its own on the same connection and would throw "cannot start a transaction within a
  // transaction" if the first one were still open.
  auto second = sirius::ffi::make_fragment(*context);
  second->declare_input_column(1, "a", "INTEGER");
  REQUIRE_NOTHROW(second->build(stream_read_plan(1)));
}
