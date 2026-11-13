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

#include "catch.hpp"
#include "op/scan/generator/generate_function.hpp"

#include <duckdb.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/storage/statistics/node_statistics.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::op::scan::generator::distribution;
using sirius::op::scan::generator::GenerateFunctionData;
using sirius::op::scan::generator::get_generate_function;
using sirius::op::scan::generator::register_generate_function;
using sirius::op::scan::generator::value_type;

// Plain DuckDB with only sirius_generate registered: no Sirius extension, no GPU runtime.
struct generate_fixture {
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  generate_fixture() { register_generate_function(*db.instance); }
};

std::string generate_sql(const std::string& columns,
                         const std::string& sizes = "num_batches := 4, rows_per_batch := 1000",
                         const std::string& seed  = "seed := 42")
{
  return "SELECT * FROM sirius_generate(" + sizes + ", " + seed + ", columns := [" + columns + "])";
}

const std::string kTwoColumns =
  "{name: 'k1', type: 'BIGINT', dist: 'zipf', p1: 1.1, p2: 0, cardinality: 1e6, "
  "null_fraction: 0.0}, "
  "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 8, p2: 16, cardinality: 0, "
  "null_fraction: 0.01}";

duckdb::LogicalGet* find_get(duckdb::LogicalOperator& node)
{
  if (node.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    return &node.Cast<duckdb::LogicalGet>();
  }
  for (auto& child : node.children) {
    if (auto* get = find_get(*child)) { return get; }
  }
  return nullptr;
}

// Binds `sql` and returns a copy of the sirius_generate bind data.
duckdb::unique_ptr<duckdb::FunctionData> bind_data_of(generate_fixture& fixture,
                                                      const std::string& sql)
{
  auto plan = fixture.con.ExtractPlan(sql);
  REQUIRE(plan);
  auto* get = find_get(*plan);
  REQUIRE(get);
  REQUIRE(get->function.GetName() == "sirius_generate");
  REQUIRE(get->bind_data);
  return get->bind_data->Copy();
}

// Runs `sql`, expecting a BinderException whose message contains every string of `needles`.
void require_bind_error(generate_fixture& fixture,
                        const std::string& sql,
                        const std::vector<std::string>& needles)
{
  INFO(sql);
  auto result = fixture.con.Query(sql);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(result->GetErrorObject().Type() == duckdb::ExceptionType::BINDER);
  for (auto const& needle : needles) {
    CHECK(result->GetError().find(needle) != std::string::npos);
  }
}

std::string column(const std::string& name, const std::string& rest)
{
  return "{name: '" + name + "', " + rest + "}";
}

}  // namespace

TEST_CASE("sirius_generate binds names and types from the columns spec", "[generate_bind]")
{
  generate_fixture fixture;
  auto plan = fixture.con.ExtractPlan(
    generate_sql(kTwoColumns + ", " + column("i", "type: 'integer', dist: 'sequential'") + ", " +
                 column("d", "type: 'FLOAT8', dist: 'normal', p1: 0, p2: 1") + ", " +
                 column("t", "type: 'text', dist: 'few_distinct', p1: 3, p2: 3")));
  REQUIRE(plan);
  auto* get = find_get(*plan);
  REQUIRE(get);
  const std::vector<std::pair<std::string, duckdb::LogicalType>> expected = {
    {"k1", duckdb::LogicalType::BIGINT},
    {"s", duckdb::LogicalType::VARCHAR},
    {"i", duckdb::LogicalType::INTEGER},
    {"d", duckdb::LogicalType::DOUBLE},
    {"t", duckdb::LogicalType::VARCHAR}};
  REQUIRE(get->names.size() == expected.size());
  REQUIRE(get->returned_types.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(get->names[i].GetIdentifierName() == expected[i].first);
    CHECK(get->returned_types[i] == expected[i].second);
  }
  // The planned query exposes the same columns, in order.
  REQUIRE(plan->types.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(plan->types[i] == expected[i].second);
  }
}

TEST_CASE("sirius_generate bind data carries the parsed spec", "[generate_bind]")
{
  generate_fixture fixture;
  auto data        = bind_data_of(fixture, generate_sql(kTwoColumns));
  auto const& spec = data->Cast<GenerateFunctionData>().spec;
  CHECK(spec.num_batches == 4);
  CHECK(spec.rows_per_batch == 1000);
  CHECK(spec.seed == 42);
  REQUIRE(spec.columns.size() == 2);
  CHECK(spec.columns[0].name == "k1");
  CHECK(spec.columns[0].type == value_type::int64);
  CHECK(spec.columns[0].dist == distribution::zipf);
  CHECK(spec.columns[0].p1 == 1.1);
  CHECK(spec.columns[0].p2 == 0);
  CHECK(spec.columns[0].cardinality == 1e6);
  CHECK(spec.columns[0].null_fraction == 0);
  CHECK(spec.columns[1].name == "s");
  CHECK(spec.columns[1].type == value_type::varchar);
  CHECK(spec.columns[1].dist == distribution::uniform);
  CHECK(spec.columns[1].p1 == 8);  // integer literal where DOUBLE is expected
  CHECK(spec.columns[1].p2 == 16);
  CHECK(spec.columns[1].null_fraction == 0.01);
}

TEST_CASE("sirius_generate defaults seed and missing struct fields to 0", "[generate_bind]")
{
  generate_fixture fixture;
  auto data        = bind_data_of(fixture,
                           "SELECT * FROM sirius_generate(num_batches := 1, rows_per_batch := 10, "
                                  "columns := [{name: 'a', type: 'INT', dist: 'sequential'}])");
  auto const& spec = data->Cast<GenerateFunctionData>().spec;
  CHECK(spec.seed == 0);
  REQUIRE(spec.columns.size() == 1);
  CHECK(spec.columns[0].type == value_type::int32);
  CHECK(spec.columns[0].dist == distribution::sequential);
  CHECK(spec.columns[0].p1 == 0);
  CHECK(spec.columns[0].p2 == 0);
  CHECK(spec.columns[0].cardinality == 0);
  CHECK(spec.columns[0].null_fraction == 0);
}

TEST_CASE("sirius_generate cannot be executed by plain DuckDB", "[generate_bind]")
{
  generate_fixture fixture;
  auto result = fixture.con.Query(generate_sql(kTwoColumns));
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("sirius_generate requires Sirius GPU execution") !=
        std::string::npos);
}

TEST_CASE("sirius_generate rejects invalid column specs naming the column", "[generate_bind]")
{
  generate_fixture fixture;
  auto bad = [&](const std::string& rest, const std::vector<std::string>& needles) {
    auto expected = needles;
    expected.emplace_back("bad_col");
    require_bind_error(fixture, generate_sql(column("bad_col", rest)), expected);
  };

  bad("type: 'DECIMAL', dist: 'uniform'", {"unsupported type", "DECIMAL"});
  bad("dist: 'uniform'", {"'type' is required"});
  bad("type: 'BIGINT'", {"'dist' is required"});
  bad("type: 'BIGINT', dist: 'gaussian'",
      {"gaussian", "uniform", "normal", "zipf", "sequential_desc", "few_distinct"});
  bad("type: 'BIGINT', dist: 'uniform', null_fraction: 1.5", {"null_fraction"});
  bad("type: 'BIGINT', dist: 'uniform', null_fraction: -0.1", {"null_fraction"});
  bad("type: 'BIGINT', dist: 'uniform', p1: 10, p2: 5", {"uniform"});
  bad("type: 'DOUBLE', dist: 'normal', p1: 0, p2: -1", {"normal"});
  bad("type: 'BIGINT', dist: 'zipf', p1: 0, cardinality: 100", {"zipf", "p1"});
  bad("type: 'BIGINT', dist: 'zipf', p1: 1.1, cardinality: 0", {"zipf", "cardinality"});
  bad("type: 'BIGINT', dist: 'few_distinct', p1: 0", {"few_distinct"});
  bad("type: 'VARCHAR', dist: 'uniform', p1: 0, p2: 4", {"VARCHAR"});
  bad("type: 'VARCHAR', dist: 'sequential', p1: 8, p2: 4", {"VARCHAR"});
  // A row's length must fit a cudf::size_type: the string kernel could not produce this column.
  bad("type: 'VARCHAR', dist: 'uniform', p1: 1, p2: 1e10", {"VARCHAR", "2147483647"});
  bad("type: 'BIGINT', dist: 'uniform', cardinality: -1", {"cardinality"});
  bad("type: 'DOUBLE', dist: 'uniform', p1: 'nan'::DOUBLE", {"p1", "finite"});
  bad("type: 'DOUBLE', dist: 'normal', p2: 'inf'::DOUBLE", {"p2", "finite"});
  bad("type: 'BIGINT', dist: 'uniform', null_fraction: 'nan'::DOUBLE", {"null_fraction", "finite"});

  require_bind_error(fixture,
                     generate_sql(column("dup", "type: 'BIGINT', dist: 'uniform'") + ", " +
                                  column("DUP", "type: 'BIGINT', dist: 'uniform'")),
                     {"DUP", "duplicate"});
}

TEST_CASE("sirius_generate rejects invalid sizes and a missing columns list", "[generate_bind]")
{
  generate_fixture fixture;
  const std::string col = column("a", "type: 'BIGINT', dist: 'uniform'");

  require_bind_error(fixture, generate_sql(col, "rows_per_batch := 10"), {"num_batches"});
  require_bind_error(fixture, generate_sql(col, "num_batches := 10"), {"rows_per_batch"});
  require_bind_error(
    fixture, generate_sql(col, "num_batches := 0, rows_per_batch := 10"), {"num_batches"});
  require_bind_error(
    fixture, generate_sql(col, "num_batches := 1, rows_per_batch := 0"), {"rows_per_batch"});
  // One batch is one cudf table: at most 2^31 - 1 rows.
  require_bind_error(fixture,
                     generate_sql(col, "num_batches := 1, rows_per_batch := 2147483648"),
                     {"rows_per_batch", "2147483647"});
  require_bind_error(fixture,
                     generate_sql(col, "num_batches := 18446744073709551615, rows_per_batch := 2"),
                     {"overflows"});
  require_bind_error(
    fixture, "SELECT * FROM sirius_generate(num_batches := 1, rows_per_batch := 1)", {"columns"});
  require_bind_error(
    fixture,
    "SELECT * FROM sirius_generate(num_batches := 1, rows_per_batch := 1, columns := [])",
    {"columns"});

  // The largest batch size is accepted.
  auto data =
    bind_data_of(fixture, generate_sql(col, "num_batches := 1, rows_per_batch := 2147483647"));
  CHECK(data->Cast<GenerateFunctionData>().spec.rows_per_batch == 2147483647ULL);
}

TEST_CASE("GenerateFunctionData Copy and Equals follow the spec", "[generate_bind]")
{
  generate_fixture fixture;
  auto first  = bind_data_of(fixture, generate_sql(kTwoColumns));
  auto second = bind_data_of(fixture, generate_sql(kTwoColumns));
  auto copy   = first->Copy();
  CHECK(copy->Equals(*first));
  CHECK(first->Equals(*copy));
  CHECK(copy->Cast<GenerateFunctionData>().spec == first->Cast<GenerateFunctionData>().spec);
  CHECK(first->Equals(*second));

  auto reseeded = bind_data_of(
    fixture, generate_sql(kTwoColumns, "num_batches := 4, rows_per_batch := 1000", "seed := 43"));
  CHECK_FALSE(first->Equals(*reseeded));
  auto resized =
    bind_data_of(fixture, generate_sql(kTwoColumns, "num_batches := 5, rows_per_batch := 1000"));
  CHECK_FALSE(first->Equals(*resized));
}

TEST_CASE("get_generate_function returns one identical definition", "[generate_bind]")
{
  auto first  = get_generate_function();
  auto second = get_generate_function();
  CHECK(first.GetName() == "sirius_generate");
  CHECK(first.GetName() == second.GetName());
  REQUIRE(first.bind != nullptr);
  REQUIRE(first.function != nullptr);
  REQUIRE(first.cardinality != nullptr);
  CHECK(first.bind == second.bind);
  CHECK(first.function == second.function);
  CHECK(first.cardinality == second.cardinality);
  CHECK(first.init_global == second.init_global);
  CHECK(first.init_local == second.init_local);
  CHECK_FALSE(first.projection_pushdown);
  CHECK_FALSE(first.filter_pushdown);
  CHECK(first.GetSignature() == second.GetSignature());

  auto options = first.GetSignature().GetTypedKwargs();
  REQUIRE(options);
  REQUIRE(second.GetSignature().GetTypedKwargs());
  CHECK(*options == *second.GetSignature().GetTypedKwargs());
  for (auto const* name : {"num_batches", "rows_per_batch", "seed"}) {
    auto option = options->Find(name);
    REQUIRE(option);
    CHECK(option->type == duckdb::LogicalType::UBIGINT);
  }
  auto columns = options->Find("columns");
  REQUIRE(columns);
  CHECK(columns->type.id() == duckdb::LogicalTypeId::LIST);

  // Registering twice is harmless, and the catalog entry uses the same callbacks.
  generate_fixture fixture;
  register_generate_function(*fixture.db.instance);
  auto plan = fixture.con.ExtractPlan(generate_sql(kTwoColumns));
  auto* get = find_get(*plan);
  REQUIRE(get);
  CHECK(get->function.bind == first.bind);
  CHECK(get->function.function == first.function);
  CHECK(get->function.cardinality == first.cardinality);
}

TEST_CASE("sirius_generate cardinality is num_batches * rows_per_batch", "[generate_bind]")
{
  generate_fixture fixture;
  auto plan =
    fixture.con.ExtractPlan(generate_sql(kTwoColumns, "num_batches := 7, rows_per_batch := 1234"));
  auto* get = find_get(*plan);
  REQUIRE(get);
  REQUIRE(get->function.cardinality);
  auto stats = get->function.cardinality(*fixture.con.context, get->bind_data.get());
  REQUIRE(stats);
  CHECK(stats->has_estimated_cardinality);
  CHECK(stats->estimated_cardinality == 7 * 1234);
  CHECK(stats->has_max_cardinality);
  CHECK(stats->max_cardinality == 7 * 1234);
  CHECK(get->EstimateCardinality(*fixture.con.context) == 7 * 1234);
}

TEST_CASE("sirius_generate accepts every type alias and dist name case-insensitively",
          "[generate_bind]")
{
  generate_fixture fixture;
  const std::vector<std::pair<std::string, duckdb::LogicalType>> aliases = {
    {"INTEGER", duckdb::LogicalType::INTEGER},
    {"int", duckdb::LogicalType::INTEGER},
    {"Int4", duckdb::LogicalType::INTEGER},
    {"bigint", duckdb::LogicalType::BIGINT},
    {"INT8", duckdb::LogicalType::BIGINT},
    {"double", duckdb::LogicalType::DOUBLE},
    {"Float8", duckdb::LogicalType::DOUBLE},
    {"varchar", duckdb::LogicalType::VARCHAR},
    {"TEXT", duckdb::LogicalType::VARCHAR},
    {"string", duckdb::LogicalType::VARCHAR}};
  std::string columns;
  for (std::size_t i = 0; i < aliases.size(); ++i) {
    if (i > 0) { columns += ", "; }
    columns += column("c" + std::to_string(i),
                      "type: '" + aliases[i].first + "', dist: 'UNIFORM', p1: 1, p2: 4");
  }
  auto plan = fixture.con.ExtractPlan(generate_sql(columns));
  REQUIRE(plan);
  auto* get = find_get(*plan);
  REQUIRE(get);
  REQUIRE(get->names.size() == aliases.size());
  REQUIRE(get->returned_types.size() == aliases.size());
  for (std::size_t i = 0; i < aliases.size(); ++i) {
    INFO(aliases[i].first);
    CHECK(get->names[i].GetIdentifierName() == "c" + std::to_string(i));
    CHECK(get->returned_types[i] == aliases[i].second);
  }

  auto data =
    bind_data_of(fixture, generate_sql(column("z", "type: 'BIGINT', dist: 'Sequential_Desc'")));
  CHECK(data->Cast<GenerateFunctionData>().spec.columns[0].dist == distribution::sequential_desc);
}

TEST_CASE("sirius_generate rejects a column without a name", "[generate_bind]")
{
  generate_fixture fixture;
  require_bind_error(
    fixture, generate_sql("{type: 'BIGINT', dist: 'uniform'}"), {"'columns' entry 0", "name"});
  require_bind_error(fixture,
                     generate_sql(column("", "type: 'BIGINT', dist: 'uniform'")),
                     {"'columns' entry 0", "name"});
}
