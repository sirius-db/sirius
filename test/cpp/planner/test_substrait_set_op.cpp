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

// SetRel through the Substrait consumer compiled into Sirius. `SET_OP_UNION_ALL` must keep
// repeated rows; the consumer used to build the set operation without `setop_all`, which
// defaults to false, so every UNION ALL ran as a UNION DISTINCT.

#include "from_substrait.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <substrait/plan.pb.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

using ids = std::vector<std::int64_t>;

void add_read(substrait::SetRel& set, std::string const& table)
{
  auto* read = set.add_inputs()->mutable_read();
  read->mutable_named_table()->add_names(table);
  auto* schema = read->mutable_base_schema();
  schema->add_names("id");
  schema->mutable_struct_()->add_types()->mutable_i64()->set_nullability(
    substrait::Type::NULLABILITY_NULLABLE);
}

/// `SELECT id FROM a UNION ALL SELECT id FROM b ...` over tables `t(id BIGINT)`.
std::string union_all_plan(std::initializer_list<std::string> tables)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("id");
  auto* set = root->mutable_input()->mutable_set();
  set->set_op(substrait::SetRel::SET_OP_UNION_ALL);
  for (auto const& table : tables) {
    add_read(*set, table);
  }
  std::string out;
  REQUIRE(plan.SerializeToString(&out));
  return out;
}

struct fixture {
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  fixture()
  {
    create("a", "(1), (2), (3)");
    create("b", "(3), (4), (5)");
    create("c", "(1), (1), (2)");
    create("d", "(4), (5), (6)");
  }

  void create(std::string const& name, std::string const& values)
  {
    auto created = con.Query("CREATE TABLE " + name + " AS SELECT id::BIGINT AS id FROM (VALUES " +
                             values + ") AS v(id)");
    REQUIRE_FALSE(created->HasError());
  }

  /// The rows of the plan, sorted.
  ids rows(std::initializer_list<std::string> tables)
  {
    duckdb::SubstraitToDuckDB transformer(con.context, union_all_plan(tables), /*json=*/false);
    auto result = transformer.TransformPlan()->Execute();
    REQUIRE_FALSE(result->HasError());
    ids out;
    while (auto chunk = result->Fetch()) {
      for (duckdb::idx_t i = 0; i < chunk->size(); ++i) {
        out.push_back(chunk->GetValue(0, i).GetValue<std::int64_t>());
      }
    }
    std::sort(out.begin(), out.end());
    return out;
  }
};

}  // namespace

TEST_CASE("Substrait SetRel: UNION_ALL keeps a row that is in both inputs", "[substrait_set_op]")
{
  fixture f;
  CHECK(f.rows({"a", "b"}) == ids{1, 2, 3, 3, 4, 5});
}

TEST_CASE("Substrait SetRel: UNION_ALL keeps a row repeated inside one input", "[substrait_set_op]")
{
  fixture f;
  CHECK(f.rows({"c", "b"}) == ids{1, 1, 2, 3, 4, 5});
}

TEST_CASE("Substrait SetRel: UNION_ALL of a table with itself doubles every row",
          "[substrait_set_op]")
{
  fixture f;
  CHECK(f.rows({"a", "a"}) == ids{1, 1, 2, 2, 3, 3});
}

TEST_CASE("Substrait SetRel: UNION_ALL of inputs that share no row", "[substrait_set_op]")
{
  fixture f;
  CHECK(f.rows({"a", "d"}) == ids{1, 2, 3, 4, 5, 6});
}
