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

/**
 * @file test_set_operation_lowering.cpp
 * @brief Tests `plan_except_intersect` and the set-operation fork in `create_plan`.
 *
 * These tests find the set operation in an optimized logical plan and call
 * `create_plan(LogicalSetOperation&)` on it through `set_operation_planner`, past the generator's
 * dispatch switch unless a test asks for it.
 */

#include "expression/ast/node.hpp"
#include "helper/type_conversions.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_replicate.hpp"
#include "op/sirius_physical_union.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/scoped_sirius_setting.hpp"
#include "utils/scoped_temp_directory.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/operator/logical_set_operation.hpp>
#include <duckdb/planner/planner.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

namespace {

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

//! Exposes the protected `create_plan` overloads, including `create_plan(LogicalSetOperation&)`.
struct set_operation_planner : sirius::planner::sirius_physical_plan_generator {
  using sirius_physical_plan_generator::create_plan;
  using sirius_physical_plan_generator::sirius_physical_plan_generator;
};

//! Masks the optimizers Sirius masks at load and holds a transaction open while planning.
class planning_scope {
 public:
  explicit planning_scope(duckdb::Connection& con)
    : _con(con), _saved(duckdb::DBConfig::GetConfig(*con.context).options.disabled_optimizers)
  {
    auto& disabled = duckdb::DBConfig::GetConfig(*con.context).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    disabled.insert(duckdb::OptimizerType::LATE_MATERIALIZATION);
    sirius::test::query(_con, "BEGIN TRANSACTION");
  }

  ~planning_scope()
  {
    _con.Query("ROLLBACK");
    duckdb::DBConfig::GetConfig(*_con.context).options.disabled_optimizers = _saved;
  }

  planning_scope(planning_scope const&)            = delete;
  planning_scope& operator=(planning_scope const&) = delete;

 private:
  duckdb::Connection& _con;
  duckdb::set<duckdb::OptimizerType> _saved;
};

//! Sets `default_collation` for its lifetime.
class scoped_default_collation {
 public:
  scoped_default_collation(duckdb::Connection& con, std::string const& collation) : _con(con)
  {
    sirius::test::query(_con, "SET default_collation = '" + collation + "'");
  }
  ~scoped_default_collation() { _con.Query("RESET default_collation"); }

  scoped_default_collation(scoped_default_collation const&)            = delete;
  scoped_default_collation& operator=(scoped_default_collation const&) = delete;

 private:
  duckdb::Connection& _con;
};

duckdb::LogicalSetOperation* find_set_operation(duckdb::LogicalOperator& op)
{
  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_UNION:
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      return &op.Cast<duckdb::LogicalSetOperation>();
    default: break;
  }
  for (auto& child : op.children) {
    if (auto* found = find_set_operation(*child)) { return found; }
  }
  return nullptr;
}

//! Plans the first set operation in @p query's optimized logical plan; throws what the builder
//! throws. With @p through_dispatch, enters through the generator's dispatch switch instead.
duckdb::unique_ptr<sirius_physical_operator> plan_set_operation(duckdb::Connection& con,
                                                                std::string const& query,
                                                                bool through_dispatch = false)
{
  auto& context = *con.context;
  planning_scope const scope(con);

  duckdb::Parser parser(context.GetParserOptions());
  parser.ParseQuery(query);
  REQUIRE(parser.statements.size() == 1);

  duckdb::Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));
  REQUIRE(planner.plan);
  auto plan = std::move(planner.plan);

  duckdb::Optimizer optimizer(*planner.binder, context);
  plan = optimizer.Optimize(std::move(plan));
  plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  resolver.VisitOperator(*plan);

  auto* set_operation = find_set_operation(*plan);
  REQUIRE(set_operation != nullptr);
  set_operation_planner generator(context);
  if (through_dispatch) {
    return generator.create_plan(static_cast<duckdb::LogicalOperator&>(*set_operation));
  }
  return generator.create_plan(*set_operation);
}

duckdb::vector<sirius::logical_type> sirius_types(duckdb::vector<duckdb::LogicalType> const& types)
{
  return sirius::from_duckdb_vec(types);
}

//! Requires @p node to be a constant holding exactly @p expected.
template <typename T>
void require_constant(sirius::ast::node const& node, T expected)
{
  REQUIRE(node.holds<sirius::ast::constant>());
  auto const* payload = std::get_if<T>(&node.get<sirius::ast::constant>().payload);
  REQUIRE(payload != nullptr);
  CHECK(*payload == expected);
}

//! Requires @p node to be a reference to column @p column.
void require_reference_to(sirius::ast::node const& node, std::uint32_t column)
{
  REQUIRE(node.holds<sirius::ast::reference>());
  CHECK(node.get<sirius::ast::reference>().column_index == column);
}

//! The operators an ALL form lowers to, top down.
struct replicate_plan {
  sirius::op::sirius_physical_replicate& replicate;
  sirius::op::sirius_physical_projection& count_projection;
  sirius::op::sirius_physical_grouped_aggregate& aggregate;
  sirius::op::sirius_physical_union& union_op;
};

//! Requires @p plan to be `REPLICATE -> PROJECTION -> HASH_GROUP_BY -> UNION` over @p types with
//! @p tag_count tag columns, where input @p i is tagged `tags[i]`.
replicate_plan require_replicate_plan(sirius_physical_operator& plan,
                                      duckdb::vector<sirius::logical_type> const& types,
                                      std::vector<std::vector<std::int8_t>> const& tags)
{
  auto const width     = types.size();
  auto const tag_count = tags.front().size();
  auto const tinyint   = sirius::logical_type::make(sirius::type_id::TINYINT);
  auto const bigint    = sirius::logical_type::make(sirius::type_id::BIGINT);

  REQUIRE(plan.type == SiriusPhysicalOperatorType::REPLICATE);
  auto& replicate = plan.Cast<sirius::op::sirius_physical_replicate>();
  CHECK(replicate.types == types);
  CHECK(replicate.count_column() == static_cast<cudf::size_type>(width));
  CHECK(replicate.output_limits().max_rows == std::numeric_limits<cudf::size_type>::max());
  REQUIRE(replicate.children.size() == 1);

  REQUIRE(replicate.children[0]->type == SiriusPhysicalOperatorType::PROJECTION);
  auto& count_projection = replicate.children[0]->Cast<sirius::op::sirius_physical_projection>();
  auto count_types       = types;
  count_types.push_back(bigint);
  CHECK(count_projection.types == count_types);
  REQUIRE(count_projection.select_list.size() == width + 1);
  for (std::size_t i = 0; i < width; ++i) {
    REQUIRE(count_projection.select_list[i]->holds<sirius::ast::reference>());
    CHECK(count_projection.select_list[i]->get<sirius::ast::reference>().column_index == i);
  }
  REQUIRE(count_projection.children.size() == 1);

  REQUIRE(count_projection.children[0]->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
  auto& aggregate =
    count_projection.children[0]->Cast<sirius::op::sirius_physical_grouped_aggregate>();
  auto summed_types = types;
  summed_types.insert(summed_types.end(), tag_count, bigint);
  CHECK(aggregate.types == summed_types);
  std::vector<int> group_idx(width);
  std::iota(group_idx.begin(), group_idx.end(), 0);
  CHECK(aggregate.group_idx == group_idx);
  std::vector<int> sum_idx(tag_count);
  std::iota(sum_idx.begin(), sum_idx.end(), static_cast<int>(width));
  CHECK(aggregate.cudf_aggregate_idx == sum_idx);
  CHECK(aggregate.cudf_aggregates ==
        std::vector<cudf::aggregation::Kind>(tag_count, cudf::aggregation::Kind::SUM));
  REQUIRE(aggregate.children.size() == 1);

  REQUIRE(aggregate.children[0]->type == SiriusPhysicalOperatorType::UNION);
  auto& union_op    = aggregate.children[0]->Cast<sirius::op::sirius_physical_union>();
  auto tagged_types = types;
  tagged_types.insert(tagged_types.end(), tag_count, tinyint);
  CHECK(union_op.types == tagged_types);
  REQUIRE(union_op.children.size() == 2);
  for (std::size_t input = 0; input < 2; ++input) {
    auto& tagged = *union_op.children[input];
    CHECK(tagged.types == tagged_types);
    REQUIRE(tagged.type == SiriusPhysicalOperatorType::PROJECTION);
    auto const& select_list = tagged.Cast<sirius::op::sirius_physical_projection>().select_list;
    REQUIRE(select_list.size() == width + tag_count);
    for (std::size_t tag = 0; tag < tag_count; ++tag) {
      auto const& expression = *select_list[width + tag];
      require_constant(expression, tags[input][tag]);
    }
  }
  return {replicate, count_projection, aggregate, union_op};
}

class set_operation_lowering_fixture {
 public:
  set_operation_lowering_fixture()
  {
    auto const config = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" /
                        "data" / "minimal.yaml";
    db  = sirius::test::open_sirius_db(db_dir.path.c_str(), config);
    con = std::make_unique<duckdb::Connection>(*db);

    sirius::test::query(*con, "CREATE TABLE ia (k INTEGER, v VARCHAR)");
    sirius::test::query(*con, "CREATE TABLE ib (k INTEGER, v VARCHAR)");
    sirius::test::query(*con, "CREATE TABLE iwide (k BIGINT)");
    sirius::test::query(*con, "CREATE TABLE icollated (s VARCHAR COLLATE NOCASE)");
    sirius::test::query(*con, "CREATE TABLE iduration (i INTERVAL)");
    sirius::test::query(*con, "CREATE TABLE ilist (l INTEGER[])");
    sirius::test::query(*con, "INSERT INTO ia VALUES (1, 'a'), (2, 'b'), (3, 'c'), (NULL, 'n')");
    sirius::test::query(*con, "INSERT INTO ib VALUES (3, 'c'), (4, 'd'), (NULL, 'n')");
    sirius::test::query(*con, "INSERT INTO iwide VALUES (100), (200)");
    sirius::test::query(*con, "INSERT INTO icollated VALUES ('a'), ('A')");
    sirius::test::query(*con, "INSERT INTO iduration VALUES (INTERVAL 1 MONTH), (INTERVAL 30 DAY)");
    sirius::test::query(*con, "INSERT INTO ilist VALUES ([1, 2]), ([3])");
  }

  duckdb::unique_ptr<sirius_physical_operator> lower(std::string const& query)
  {
    return plan_set_operation(*con, query);
  }

  sirius::test::scoped_temp_directory db_dir;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
};

using Catch::Matchers::ContainsSubstring;

}  // namespace

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - the fork leaves UNION ALL on the union operator",
                 "[planner][set_operation][isolated_context]")
{
  auto const plan = lower("SELECT k FROM ia UNION ALL SELECT k FROM ib");
  CHECK(plan->type == SiriusPhysicalOperatorType::UNION);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form plans both inputs at the common super-type",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::BIGINT});
  auto const plan  = lower("SELECT k FROM ia INTERSECT ALL SELECT k FROM iwide");
  require_replicate_plan(*plan, types, {{1, 0}, {0, 1}});
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - INTERSECT ALL lowers to two tag sums and min copies",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR});
  auto const plan  = lower("SELECT k, v FROM ia INTERSECT ALL SELECT k, v FROM ib");
  auto const chain = require_replicate_plan(*plan, types, {{1, 0}, {0, 1}});

  // copies = CASE WHEN m < n THEN m ELSE n END over the sums at columns 2 and 3.
  auto const& copies = *chain.count_projection.select_list[2];
  REQUIRE(copies.holds<sirius::ast::case_expr>());
  auto const& case_expr = copies.get<sirius::ast::case_expr>();
  REQUIRE(case_expr.cases.size() == 1);
  REQUIRE(case_expr.cases[0].when_->holds<sirius::ast::comparison>());
  auto const& when = case_expr.cases[0].when_->get<sirius::ast::comparison>();
  CHECK(when.op == sirius::comparison_type::lt);
  require_reference_to(*when.left, 2);
  require_reference_to(*when.right, 3);
  require_reference_to(*case_expr.cases[0].then_, 2);
  require_reference_to(*case_expr.else_, 3);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form caps REPLICATE batches at concat_batch_bytes",
                 "[planner][set_operation][isolated_context]")
{
  sirius::test::scoped_sirius_setting const bytes{
    *con, "concat_batch_bytes", std::uint64_t{123456}};
  auto const plan = lower("SELECT k FROM ia INTERSECT ALL SELECT k FROM ib");
  REQUIRE(plan->type == SiriusPhysicalOperatorType::REPLICATE);
  CHECK(plan->Cast<sirius::op::sirius_physical_replicate>().output_limits().max_bytes == 123456);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - the dispatch switch refuses only the distinct forms",
                 "[planner][set_operation][isolated_context]")
{
  for (std::string const keyword : {"EXCEPT", "INTERSECT"}) {
    REQUIRE_THROWS_WITH(
      plan_set_operation(*con, "SELECT k FROM ia " + keyword + " SELECT k FROM ib", true),
      ContainsSubstring("only the ALL forms"));
    auto const plan =
      plan_set_operation(*con, "SELECT k FROM ia " + keyword + " ALL SELECT k FROM ib", true);
    CHECK(plan->type == SiriusPhysicalOperatorType::REPLICATE);
  }
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - the builder refuses the distinct forms",
                 "[planner][set_operation][isolated_context]")
{
  for (std::string const keyword : {"EXCEPT", "INTERSECT"}) {
    REQUIRE_THROWS_WITH(lower("SELECT k FROM ia " + keyword + " SELECT k FROM ib"),
                        ContainsSubstring(keyword + " (distinct) not supported"));
  }
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form on a floating-point key is refused",
                 "[planner][set_operation][isolated_context]")
{
  // DuckDB names REAL as FLOAT.
  for (std::string const type : {"FLOAT", "DOUBLE"}) {
    auto const query =
      "SELECT k::" + type + " FROM ia INTERSECT ALL SELECT k::" + type + " FROM ib";
    REQUIRE_THROWS_WITH(
      lower(query),
      ContainsSubstring("INTERSECT ALL on column 0 (" + type + "): floating-point keys"));
  }
  REQUIRE_THROWS_WITH(lower("SELECT v, k::DOUBLE FROM ia EXCEPT ALL SELECT v, k::DOUBLE FROM ib"),
                      ContainsSubstring("EXCEPT ALL on column 1 (DOUBLE): floating-point keys"));
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form keeps the shared refusals",
                 "[planner][set_operation][isolated_context]")
{
  SECTION("a collated key")
  {
    REQUIRE_THROWS_WITH(lower("SELECT s FROM icollated EXCEPT ALL SELECT s FROM icollated"),
                        ContainsSubstring("EXCEPT ALL on column 0"));
  }
  SECTION("a nested key")
  {
    REQUIRE_THROWS_WITH(lower("SELECT l FROM ilist INTERSECT ALL SELECT l FROM ilist"),
                        ContainsSubstring("nested column operation on column 'column 0'"));
  }
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form carries an aggregate's narrowed type",
                 "[planner][set_operation][isolated_context]")
{
  // DuckDB types sum(INTEGER) as HUGEINT; the Sirius aggregate plans it as BIGINT.
  auto const types = sirius_types({duckdb::LogicalType::BIGINT});
  SECTION("EXCEPT ALL")
  {
    auto const plan = lower("SELECT sum(k) FROM ia EXCEPT ALL SELECT sum(k) FROM ib");
    require_replicate_plan(*plan, types, {{1}, {-1}});
  }
  SECTION("INTERSECT ALL")
  {
    auto const plan = lower("SELECT sum(k) FROM ia INTERSECT ALL SELECT sum(k) FROM ib");
    require_replicate_plan(*plan, types, {{1, 0}, {0, 1}});
  }
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form whose inputs plan one column differently is refused",
                 "[planner][set_operation][isolated_context]")
{
  REQUIRE_THROWS_WITH(
    lower("SELECT sum(k) FROM ia INTERSECT ALL SELECT k::HUGEINT FROM ib"),
    ContainsSubstring("INTERSECT ALL inputs plan column 0 as BIGINT and HUGEINT"));
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form whose CTE input is wider than its body is refused",
                 "[planner][set_operation][isolated_context]")
{
  REQUIRE_THROWS_WITH(lower("SELECT k FROM ia "
                            "INTERSECT ALL "
                            "(WITH m AS MATERIALIZED (SELECT k, v FROM ib) "
                            " SELECT m1.k FROM m m1 JOIN m m2 ON m1.v = m2.v)"),
                      ContainsSubstring("INTERSECT ALL input 1 is planned with 2 columns, not 1"));
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form whose CTE input matches its body is accepted",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR});
  auto const plan  = lower(
    "SELECT k, v FROM ia "
     "INTERSECT ALL "
     "(WITH m AS MATERIALIZED (SELECT k, v FROM ib) SELECT k, v FROM m)");
  auto const chain = require_replicate_plan(*plan, types, {{1, 0}, {0, 1}});
  REQUIRE(chain.union_op.children[1]->children.size() == 1);
  CHECK(chain.union_op.children[1]->children[0]->type == SiriusPhysicalOperatorType::CTE);
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - an ALL form on a key DuckDB collates is refused",
                 "[planner][set_operation][isolated_context]")
{
  SECTION("a VARCHAR column with a collation")
  {
    REQUIRE_THROWS_WITH(lower("SELECT s FROM icollated INTERSECT ALL SELECT s FROM icollated"),
                        ContainsSubstring("INTERSECT ALL on column 0"));
  }
  SECTION("a plain VARCHAR column under a default collation")
  {
    scoped_default_collation const nocase(*con, "nocase");
    REQUIRE_THROWS_WITH(lower("SELECT v FROM ia INTERSECT ALL SELECT v FROM ib"),
                        ContainsSubstring("INTERSECT ALL on column 0"));
  }
  SECTION("an INTERVAL column")
  {
    REQUIRE_THROWS_WITH(lower("SELECT i FROM iduration INTERSECT ALL SELECT i FROM iduration"),
                        ContainsSubstring("INTERSECT ALL on column 0"));
  }
}

TEST_CASE_METHOD(set_operation_lowering_fixture,
                 "set_operation - EXCEPT ALL lowers to one signed tag sum and clamped copies",
                 "[planner][set_operation][isolated_context]")
{
  auto const types = sirius_types({duckdb::LogicalType::INTEGER});
  auto const plan  = lower("SELECT k FROM ia EXCEPT ALL SELECT k FROM ib");
  auto const chain = require_replicate_plan(*plan, types, {{1}, {-1}});

  // copies = CASE WHEN s > 0 THEN s ELSE 0 END over the sum at column 1.
  auto const& copies = *chain.count_projection.select_list[1];
  REQUIRE(copies.holds<sirius::ast::case_expr>());
  auto const& case_expr = copies.get<sirius::ast::case_expr>();
  REQUIRE(case_expr.cases.size() == 1);
  REQUIRE(case_expr.cases[0].when_->holds<sirius::ast::comparison>());
  auto const& when = case_expr.cases[0].when_->get<sirius::ast::comparison>();
  CHECK(when.op == sirius::comparison_type::gt);
  require_reference_to(*when.left, 1);
  require_constant(*when.right, std::int64_t{0});
  require_reference_to(*case_expr.cases[0].then_, 1);
  require_constant(*case_expr.else_, std::int64_t{0});
}
