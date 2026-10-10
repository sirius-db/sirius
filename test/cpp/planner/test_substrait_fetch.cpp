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

// FetchRel through the Substrait consumer compiled into Sirius (#1963). Producers on Substrait
// 0.65+ write `count_expr` / `offset_expr`; the consumer used to read only the deprecated
// `count` / `offset`, so every LIMIT became LIMIT 0.

#include "from_substrait.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <substrait/plan.pb.h>

#include <cstdint>
#include <functional>
#include <string>

namespace {

constexpr std::int64_t kRows = 10;

/// `SELECT id FROM t` over `t(id BIGINT)`, with `configure` setting the fetch bounds.
std::string fetch_plan(std::function<void(substrait::FetchRel&)> const& configure)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("id");
  auto* fetch = root->mutable_input()->mutable_fetch();
  auto* read  = fetch->mutable_input()->mutable_read();
  read->mutable_named_table()->add_names("t");
  auto* schema = read->mutable_base_schema();
  schema->add_names("id");
  schema->mutable_struct_()->add_types()->mutable_i64()->set_nullability(
    substrait::Type::NULLABILITY_NULLABLE);
  configure(*fetch);
  std::string out;
  REQUIRE(plan.SerializeToString(&out));
  return out;
}

substrait::Expression i64_literal(std::int64_t value)
{
  substrait::Expression expr;
  expr.mutable_literal()->set_i64(value);
  return expr;
}

substrait::Expression null_literal()
{
  substrait::Expression expr;
  expr.mutable_literal()->mutable_null()->mutable_i64()->set_nullability(
    substrait::Type::NULLABILITY_NULLABLE);
  return expr;
}

struct fixture {
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  fixture()
  {
    auto created = con.Query("CREATE TABLE t AS SELECT range::BIGINT AS id FROM range(" +
                             std::to_string(kRows) + ")");
    REQUIRE_FALSE(created->HasError());
  }

  std::int64_t rows(std::function<void(substrait::FetchRel&)> const& configure)
  {
    duckdb::SubstraitToDuckDB transformer(con.context, fetch_plan(configure), /*json=*/false);
    auto result = transformer.TransformPlan()->Execute();
    REQUIRE_FALSE(result->HasError());
    std::int64_t count = 0;
    while (auto chunk = result->Fetch()) {
      count += static_cast<std::int64_t>(chunk->size());
    }
    return count;
  }

  std::string error(std::function<void(substrait::FetchRel&)> const& configure)
  {
    try {
      duckdb::SubstraitToDuckDB transformer(con.context, fetch_plan(configure), /*json=*/false);
      auto result = transformer.TransformPlan()->Execute();
      return result->HasError() ? result->GetError() : std::string();
    } catch (std::exception const& e) {
      return e.what();
    }
  }
};

}  // namespace

TEST_CASE("Substrait FetchRel: count_expr and offset_expr bound the rows", "[substrait_fetch]")
{
  fixture f;
  CHECK(f.rows([](auto& fetch) { *fetch.mutable_count_expr() = i64_literal(3); }) == 3);
  CHECK(f.rows([](auto& fetch) { *fetch.mutable_offset_expr() = i64_literal(7); }) == 3);
  CHECK(f.rows([](auto& fetch) {
          *fetch.mutable_count_expr()  = i64_literal(4);
          *fetch.mutable_offset_expr() = i64_literal(8);
        }) == 2);
  CHECK(f.rows([](auto& fetch) { *fetch.mutable_count_expr() = i64_literal(0); }) == 0);
}

TEST_CASE("Substrait FetchRel: an unset or null count returns every row", "[substrait_fetch]")
{
  fixture f;
  CHECK(f.rows([](auto&) {}) == kRows);
  CHECK(f.rows([](auto& fetch) { *fetch.mutable_count_expr() = null_literal(); }) == kRows);
  CHECK(f.rows([](auto& fetch) { *fetch.mutable_offset_expr() = null_literal(); }) == kRows);
}

TEST_CASE("Substrait FetchRel: the deprecated count and offset still work", "[substrait_fetch]")
{
  fixture f;
  CHECK(f.rows([](auto& fetch) { fetch.set_count(5); }) == 5);
  CHECK(f.rows([](auto& fetch) { fetch.set_count(-1); }) == kRows);
  CHECK(f.rows([](auto& fetch) {
          fetch.set_count(2);
          fetch.set_offset(9);
        }) == 1);
}

TEST_CASE("Substrait FetchRel: a negative or non-literal bound is refused", "[substrait_fetch]")
{
  fixture f;
  CHECK_THAT(f.error([](auto& fetch) { *fetch.mutable_count_expr() = i64_literal(-1); }),
             Catch::Matchers::ContainsSubstring("must not be negative"));
  CHECK_THAT(f.error([](auto& fetch) { *fetch.mutable_offset_expr() = i64_literal(-2); }),
             Catch::Matchers::ContainsSubstring("must not be negative"));
  CHECK_THAT(f.error([](auto& fetch) {
               fetch.mutable_count_expr()->mutable_selection()->mutable_direct_reference();
             }),
             Catch::Matchers::ContainsSubstring("must be a literal"));
}
