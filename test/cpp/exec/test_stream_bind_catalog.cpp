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

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb.hpp>
#include <exec/stream_bind_catalog.hpp>
#include <exec/stream_plan_bindings.hpp>
#include <helper/type_conversions.hpp>
#include <sirius/exception.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <vector>

using namespace sirius::exec;

namespace {

duckdb::vector<sirius::logical_type> int_schema()
{
  return sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
}

stream_input_binding make_binding()
{
  return stream_input_binding{{"a"},
                              int_schema(),
                              std::make_shared<cucascade::shared_data_repository>(),
                              {0},
                              std::nullopt,
                              nullptr};
}

}  // namespace

// ============================================================================
// CAT-1: a declared stream resolves to the schema it was declared with
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-1: a declared stream resolves its schema",
          "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;
  REQUIRE_FALSE(catalog.contains(7));

  catalog.declare(7, make_binding());

  REQUIRE(catalog.contains(7));
  const auto& binding = catalog.get(7);
  REQUIRE(binding.names.size() == 1);
  REQUIRE(binding.names[0] == "a");
  REQUIRE(binding.types.size() == 1);
  REQUIRE(binding.repository != nullptr);
  REQUIRE(binding.expected_senders == std::set<sender_id_t>{0});
  // Nothing has planned yet, so no operator has been built for this id.
  REQUIRE(binding.built == nullptr);
}

// ============================================================================
// CAT-2: an undeclared id is a defined error, not an empty schema
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-2: an undeclared id throws", "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;
  REQUIRE_THROWS_AS(catalog.get(1), sirius::invalid_input_exception);
  REQUIRE_THROWS_AS(catalog.set_built(1, nullptr), sirius::invalid_input_exception);
}

// ============================================================================
// CAT-3: a malformed declaration is rejected at declare time
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-3: a malformed declaration is rejected", "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;

  // A stream with no repository has nowhere for its senders to push.
  auto no_repo       = make_binding();
  no_repo.repository = nullptr;
  REQUIRE_THROWS_AS(catalog.declare(1, std::move(no_repo)), sirius::invalid_input_exception);

  // Names and types must agree — DuckDB's bind hands back both, and a mismatch would surface
  // much later as a column-count error inside the optimizer.
  auto mismatched = make_binding();
  mismatched.names.emplace_back("b");
  REQUIRE_THROWS_AS(catalog.declare(2, std::move(mismatched)), sirius::invalid_input_exception);

  auto empty = make_binding();
  empty.names.clear();
  empty.types.clear();
  REQUIRE_THROWS_AS(catalog.declare(3, std::move(empty)), sirius::invalid_input_exception);

  REQUIRE(catalog.declared_streams().empty());
}

// ============================================================================
// CAT-4: the plan generator's back-pointer round-trips
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-4: the built operator is recorded", "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;
  catalog.declare(4, make_binding());

  auto* fake = reinterpret_cast<sirius::op::sirius_physical_streaming_source*>(0x1234);
  catalog.set_built(4, fake);
  REQUIRE(catalog.get(4).built == fake);
}

// ============================================================================
// CAT-5: a fragment's declarations do not leak into the next one
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-5: clear drops every declaration", "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;
  catalog.declare(1, make_binding());
  catalog.declare(2, make_binding());
  REQUIRE(catalog.declared_streams() == std::vector<stream_id_t>{1, 2});

  catalog.clear();

  REQUIRE(catalog.declared_streams().empty());
  REQUIRE_FALSE(catalog.contains(1));
  // A reused connection must not serve a stale schema for a recycled id.
  REQUIRE_THROWS_AS(catalog.get(1), sirius::invalid_input_exception);
}

// ============================================================================
// CAT-5b: erase is scoped to one id, so fragments sharing a connection don't
// wipe each other's declarations
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-5b: erase drops only the named stream", "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;
  catalog.declare(1, make_binding());
  catalog.declare(2, make_binding());

  catalog.erase(1);

  // The catalog is one ClientContextState per connection. A fragment tearing down must remove
  // only the ids it declared — clear() here would take a peer fragment's schema with it.
  REQUIRE_FALSE(catalog.contains(1));
  REQUIRE(catalog.contains(2));
  REQUIRE(catalog.declared_streams() == std::vector<stream_id_t>{2});

  // Idempotent: teardown paths call it unconditionally, including after a failed build.
  catalog.erase(1);
  REQUIRE(catalog.declared_streams() == std::vector<stream_id_t>{2});
}

// ============================================================================
// CAT-6: redeclaring an id replaces it rather than keeping the old schema
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-6: a redeclared id is replaced", "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;
  catalog.declare(1, make_binding());
  catalog.set_built(1, reinterpret_cast<sirius::op::sirius_physical_streaming_source*>(0x1234));

  auto replacement  = make_binding();
  replacement.names = {"renamed"};
  catalog.declare(1, std::move(replacement));

  REQUIRE(catalog.get(1).names[0] == "renamed");
  // The stale back-pointer must not survive: it refers to an operator from the previous plan.
  REQUIRE(catalog.get(1).built == nullptr);
}

// ============================================================================
// CAT-7: a stream read binds against the declared schema, with no file behind it
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-7: sirius_stream_source binds a declared stream",
          "[stream_bind_catalog]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  register_stream_source_function(*db.instance);

  auto catalog = duckdb::make_shared_ptr<stream_bind_catalog>();
  con.context->registered_state->Insert(stream_bind_catalog::kStateKey, catalog);

  catalog->declare(
    0,
    stream_input_binding{{"l_orderkey", "l_comment"},
                         sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{
                           duckdb::LogicalType::BIGINT, duckdb::LogicalType::VARCHAR}),
                         std::make_shared<cucascade::shared_data_repository>(),
                         {0},
                         std::nullopt,
                         nullptr});

  // Binding is the whole point: DuckDB must resolve names and types for a relation that has no
  // file, no catalog entry and no rows behind it. A parquet URI could not do this.
  auto prepared = con.Prepare("SELECT * FROM sirius_stream_source(0)");
  REQUIRE_FALSE(prepared->HasError());

  REQUIRE(prepared->GetNames() == duckdb::vector<std::string>{"l_orderkey", "l_comment"});
  REQUIRE(prepared->GetTypes() == duckdb::vector<duckdb::LogicalType>{
                                    duckdb::LogicalType::BIGINT, duckdb::LogicalType::VARCHAR});

  // A projection over the stream binds too, so the fragment's plan is not limited to SELECT *.
  auto projected = con.Prepare("SELECT l_orderkey + 1 FROM sirius_stream_source(0)");
  REQUIRE_FALSE(projected->HasError());
}

// ============================================================================
// CAT-8: an undeclared stream fails at bind time, not at execution
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-8: an undeclared stream is a bind error",
          "[stream_bind_catalog]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  register_stream_source_function(*db.instance);

  auto catalog = duckdb::make_shared_ptr<stream_bind_catalog>();
  con.context->registered_state->Insert(stream_bind_catalog::kStateKey, catalog);

  // Nothing declared id 9. Failing here means a fragment referencing a stream nobody set up is
  // caught while planning, rather than surfacing later as a plan with no source.
  auto prepared = con.Prepare("SELECT * FROM sirius_stream_source(9)");
  REQUIRE(prepared->HasError());
}

// ============================================================================
// CAT-9: without a catalog on the connection the function refuses to bind
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-9: no catalog on the connection is an error",
          "[stream_bind_catalog]")
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  register_stream_source_function(*db.instance);

  auto prepared = con.Prepare("SELECT * FROM sirius_stream_source(0)");
  REQUIRE(prepared->HasError());
}

// ============================================================================
// CAT-10: the optional declared row count round-trips through the catalog
// ============================================================================

TEST_CASE("stream_bind_catalog CAT-10: estimated_rows is optional and non-throwing",
          "[stream_bind_catalog]")
{
  stream_bind_catalog catalog;

  // Undeclared id: nullopt, not a throw — this feeds the TableFunction cardinality callback,
  // which DuckDB may invoke after a fragment erased its declarations.
  REQUIRE_FALSE(catalog.estimated_rows(7).has_value());

  // Declared without a count: still nullopt (today's behavior for callers that never declare).
  catalog.declare(7, make_binding());
  REQUIRE_FALSE(catalog.estimated_rows(7).has_value());

  // Declared with a count: the value comes back verbatim.
  auto binding           = make_binding();
  binding.estimated_rows = 123456;
  catalog.declare(7, std::move(binding));
  REQUIRE(catalog.estimated_rows(7) == std::optional<std::uint64_t>{123456});

  // Redeclaration replaces the whole binding, count included.
  catalog.declare(7, make_binding());
  REQUIRE_FALSE(catalog.estimated_rows(7).has_value());

  // erase() drops it with the declaration.
  binding                = make_binding();
  binding.estimated_rows = 2;
  catalog.declare(8, std::move(binding));
  catalog.erase(8);
  REQUIRE_FALSE(catalog.estimated_rows(8).has_value());
}
