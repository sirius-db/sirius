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
 * @file test_plan_tree_shape.cpp
 * @brief Invariants of the physical plan tree after `create_plan` runs
 *        `insert_gpu_pipeline_operators` + `set_parent_ops`: scan leaves are replaced by
 *        GPU_SCAN, joins/aggregates/sorts carry their CONCAT/PARTITION/MERGE wrap chains,
 *        DELIM JOIN internal subtrees (`join`/`distinct_root`) are rewritten and tagged, and
 *        every operator's `_parent_op` matches its position in the final tree.
 */

#include "expression/aggregate_id.hpp"
#include "expression/ast/aggregate.hpp"
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/join_condition.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_projection.hpp"
#include "plan_test_harness.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

#include <filesystem>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::test::collect;
using sirius::test::contains;
using sirius::test::find_first;
using sirius::test::generate_sirius_plan;
using sirius::test::scoped_temp_db_path;
using sirius::test::tree_to_string;

namespace {

duckdb::unique_ptr<duckdb::Expression> untranslatable_table_filter_expression()
{
  auto expression = duckdb::make_uniq<duckdb::BoundFunctionExpression>(
    duckdb::LogicalType::BOOLEAN,
    duckdb::ScalarFunction("sirius_unmapped_filter",
                           {duckdb::LogicalType::BIGINT},
                           duckdb::LogicalType::BOOLEAN,
                           nullptr),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  expression->children.push_back(
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0));
  return expression;
}

/// Every operator's `_parent_op` must equal its position in the final tree; delim joins
/// stamp their internal `join`/`distinct_root` subtrees with themselves as parent.
void require_parent_links(sirius_physical_operator* op, sirius_physical_operator* expected_parent)
{
  REQUIRE(op != nullptr);
  INFO("operator " << sirius::op::SiriusPhysicalOperatorToString(op->type));
  CHECK(op->get_parent_op() == expected_parent);
  for (auto& child : op->children) {
    require_parent_links(child.get(), op);
  }
  if (op->type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      op->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = op->Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join) { require_parent_links(delim.join.get(), op); }
    if (delim.distinct_root) { require_parent_links(delim.distinct_root.get(), op); }
  }
}

/// Assert `op` is a CONCAT -> PARTITION join-child wrap with the given build role, both
/// pointing at `join` as their downstream consumer. Returns the PARTITION's child (the
/// original wrapped subtree root).
sirius_physical_operator* require_join_child_wrap(sirius_physical_operator* op,
                                                  sirius_physical_operator* join,
                                                  bool is_build)
{
  REQUIRE(op != nullptr);
  REQUIRE(op->type == SiriusPhysicalOperatorType::CONCAT);
  auto& concat = op->Cast<sirius::op::sirius_physical_concat>();
  CHECK(concat.is_build_concat() == is_build);
  CHECK(concat.get_downstream_join() == join);

  REQUIRE(op->children.size() == 1);
  auto* partition = op->children[0].get();
  REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
  CHECK(partition->Cast<sirius::op::sirius_physical_partition>().is_build_partition() == is_build);

  REQUIRE(partition->children.size() == 1);
  return partition->children[0].get();
}

/// Assert the delim-join invariants shared by both variants: the distinct chain is
/// `MERGE_GROUP_BY -> PARTITION -> HASH_GROUP_BY` with the chain top tagged as owned by the
/// delim join and `distinct` borrowing the chain bottom, and the internal join carries the
/// standard CONCAT/PARTITION wrap on both children.
void require_delim_join_common(sirius::op::sirius_physical_delim_join& delim)
{
  REQUIRE(delim.distinct_root);
  auto* merge = delim.distinct_root.get();
  REQUIRE(merge->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
  CHECK(merge->owning_delim_join() == &delim);

  REQUIRE(merge->children.size() == 1);
  auto* partition = merge->children[0].get();
  REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
  CHECK_FALSE(partition->Cast<sirius::op::sirius_physical_partition>().is_build_partition());

  REQUIRE(partition->children.size() == 1);
  auto* hgb = partition->children[0].get();
  REQUIRE(hgb->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);

  // `distinct` always borrows the subtree bottom (the bare DISTINCT).
  REQUIRE(delim.distinct != nullptr);
  CHECK(static_cast<sirius_physical_operator*>(delim.distinct) == hgb);

  REQUIRE(delim.join);
  REQUIRE(delim.join->children.size() == 2);
  require_join_child_wrap(delim.join->children[0].get(), delim.join.get(), /*is_build=*/false);
  require_join_child_wrap(delim.join->children[1].get(), delim.join.get(), /*is_build=*/true);
}

struct plan_tree_shape_fixture {
  plan_tree_shape_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(_db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con = std::make_unique<Connection>(*db);

    // big_left is larger so the optimizer keeps small_right as the build side.
    con->Query("CREATE TABLE big_left (id INTEGER, val INTEGER)");
    con->Query(
      "INSERT INTO big_left VALUES (0,0),(1,3),(2,6),(3,9),(4,12),(5,15),(6,18),(7,21),(8,24),"
      "(9,27),(10,30),(11,33),(12,36),(13,39),(14,42),(15,45),(16,48),(17,51),(18,54),(19,57)");
    con->Query("CREATE TABLE small_right (rid INTEGER, other INTEGER)");
    con->Query("INSERT INTO small_right VALUES (0, 0), (1, 1)");
    con->Query("CREATE TABLE decimal_values (amount DECIMAL(15,2))");
    con->Query("INSERT INTO decimal_values VALUES (1.00), (2.50), (3.75)");

    // parts/items reproduce TPC-H q17's RIGHT_DELIM_JOIN: the filter on the correlated
    // table keeps the deliminator from rewriting the correlated aggregate into a plain
    // join + group-by, and the items-side fan-out (20 rows per fk) plus the cardinality
    // skew make the physical planner pick the RIGHT variant (tiny symmetric tables get
    // a LEFT_DELIM_JOIN instead).
    con->Query("CREATE TABLE parts (pk INTEGER, pname VARCHAR)");
    con->Query("INSERT INTO parts SELECT range, concat('p', range % 3) FROM range(500)");
    con->Query("CREATE TABLE items (fk INTEGER, qty INTEGER)");
    con->Query("INSERT INTO items SELECT range % 500, range * 7 % 23 FROM range(10000)");
  }

  ~plan_tree_shape_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  // Declared before db/con so the backing file outlives the database.
  scoped_temp_db_path _db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - scan leaves are replaced by GPU_SCAN",
                 "[plan_tree_shape][isolated_context]")
{
  auto plan = generate_sirius_plan(*con, "SELECT val FROM big_left WHERE id > 5");
  INFO(tree_to_string(plan.get()));

  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::TABLE_SCAN).empty());

  auto gpu_scans = collect(plan.get(), SiriusPhysicalOperatorType::GPU_SCAN);
  REQUIRE(!gpu_scans.empty());
  for (auto* scan : gpu_scans) {
    CHECK(scan->children.empty());
    CHECK(scan->declared_output_schema_is_runtime_schema());
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - a scan without a complete native carrier schema is rejected",
                 "[plan_tree_shape][isolated_context]")
{
  auto create = con->Query("CREATE TABLE mixed_schema (wide BIGINT, narrow DECIMAL(4,2))");
  REQUIRE(create);
  REQUIRE_FALSE(create->HasError());

  REQUIRE_THROWS_WITH(generate_sirius_plan(*con, "SELECT wide, narrow FROM mixed_schema"),
                      Catch::Contains("GPU scan output column 1 (DECIMAL(4,2)) has no native cuDF "
                                      "carrier"));
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - materialized sources are replaced by GPU_VALUES",
                 "[plan_tree_shape][isolated_context]")
{
  auto require_gpu_values_source = [&](const std::string& query) {
    auto plan = generate_sirius_plan(*con, query);
    INFO(tree_to_string(plan.get()));

    auto gpu_values = collect(plan.get(), SiriusPhysicalOperatorType::GPU_VALUES);
    REQUIRE(gpu_values.size() == 1);
    CHECK(gpu_values.front()->children.empty());
  };

  // VALUES -> COLUMN_DATA_SCAN holding a materialized collection.
  require_gpu_values_source("VALUES (1), (2)");
  // No-table SELECT -> DUMMY_SCAN.
  require_gpu_values_source("SELECT 40 + 2");
  // Provably-empty scan -> EMPTY_RESULT.
  require_gpu_values_source("SELECT val FROM big_left WHERE 1 = 0");
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - join children are wrapped CONCAT -> PARTITION with roles",
                 "[plan_tree_shape][isolated_context]")
{
  auto plan =
    generate_sirius_plan(*con, "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid");
  INFO(tree_to_string(plan.get()));

  auto* hj = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
  REQUIRE(hj != nullptr);
  REQUIRE(hj->children.size() == 2);

  auto* probe_subtree = require_join_child_wrap(hj->children[0].get(), hj, /*is_build=*/false);
  auto* build_subtree = require_join_child_wrap(hj->children[1].get(), hj, /*is_build=*/true);

  // Each wrapped subtree bottoms out in the table's GPU_SCAN leaf.
  CHECK(find_first(probe_subtree, SiriusPhysicalOperatorType::GPU_SCAN) != nullptr);
  CHECK(find_first(build_subtree, SiriusPhysicalOperatorType::GPU_SCAN) != nullptr);
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - aggregates wrap to MERGE (-> PARTITION) -> original",
                 "[plan_tree_shape][isolated_context]")
{
  SECTION("grouped aggregate gains a MERGE_GROUP_BY -> PARTITION fanout")
  {
    auto plan = generate_sirius_plan(*con, "SELECT val, count(*) FROM big_left GROUP BY val");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->children.size() == 1);

    auto* partition = merge->children[0].get();
    REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
    CHECK_FALSE(partition->Cast<sirius::op::sirius_physical_partition>().is_build_partition());

    REQUIRE(partition->children.size() == 1);
    CHECK(partition->children[0]->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
  }

  SECTION("COUNT(DISTINCT) records LIST locally and BIGINT after merge")
  {
    auto plan =
      generate_sirius_plan(*con, "SELECT val, count(DISTINCT id) FROM big_left GROUP BY val");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->get_types().size() == 2);
    CHECK(merge->get_types()[1].id() == sirius::type_id::BIGINT);
    REQUIRE(merge->children.size() == 1);

    auto* partition = merge->children[0].get();
    REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
    REQUIRE(partition->get_types().size() == 2);
    CHECK(partition->get_types()[1].id() == sirius::type_id::LIST);
    REQUIRE(partition->children.size() == 1);

    auto* local = partition->children[0].get();
    REQUIRE(local->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
    REQUIRE(local->get_types().size() == 2);
    CHECK(local->get_types()[1].id() == sirius::type_id::LIST);
  }

  SECTION("ungrouped aggregate gains MERGE_AGGREGATE with no PARTITION")
  {
    auto plan = generate_sirius_plan(*con, "SELECT sum(val) FROM big_left");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_AGGREGATE);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->children.size() == 1);
    CHECK(merge->children[0]->type == SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
  }

  SECTION("AVG records its two-column local accumulator schema below MERGE_AGGREGATE")
  {
    auto plan = generate_sirius_plan(*con, "SELECT avg(val) FROM big_left");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_AGGREGATE);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->get_types().size() == 1);
    REQUIRE(merge->children.size() == 1);

    auto* local = merge->children[0].get();
    REQUIRE(local->type == SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
    duckdb::vector<sirius::logical_type> const expected_local_types{
      sirius::logical_type::make(sirius::type_id::BIGINT),
      sirius::logical_type::make(sirius::type_id::BIGINT)};
    CHECK(local->get_types() == expected_local_types);
  }

  SECTION("AVG preserves its DECIMAL local sum carrier below MERGE_AGGREGATE")
  {
    auto plan = generate_sirius_plan(*con, "SELECT avg(amount) FROM decimal_values");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_AGGREGATE);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->get_types().size() == 1);
    REQUIRE(merge->children.size() == 1);

    auto* local = merge->children[0].get();
    REQUIRE(local->type == SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
    REQUIRE(local->get_types().size() == 2);
    CHECK(local->get_types()[0] == sirius::logical_type::make_decimal(15, 2));
    CHECK(local->get_types()[1].id() == sirius::type_id::BIGINT);
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - order-by and top-n wrap to their merge chains",
                 "[plan_tree_shape][isolated_context]")
{
  SECTION("order-by becomes MERGE_SORT -> SORT_PARTITION -> SORT_SAMPLE -> ORDER_BY")
  {
    auto plan = generate_sirius_plan(*con, "SELECT * FROM big_left ORDER BY val");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_SORT);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->children.size() == 1);
    auto* sort_partition = merge->children[0].get();
    REQUIRE(sort_partition->type == SiriusPhysicalOperatorType::SORT_PARTITION);
    REQUIRE(sort_partition->children.size() == 1);
    auto* sort_sample = sort_partition->children[0].get();
    REQUIRE(sort_sample->type == SiriusPhysicalOperatorType::SORT_SAMPLE);
    REQUIRE(sort_sample->children.size() == 1);
    CHECK(sort_sample->children[0]->type == SiriusPhysicalOperatorType::ORDER_BY);
  }

  SECTION("top-n becomes MERGE_TOP_N -> TOP_N")
  {
    auto plan = generate_sirius_plan(*con, "SELECT * FROM big_left ORDER BY val LIMIT 3");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_TOP_N);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->children.size() == 1);
    CHECK(merge->children[0]->type == SiriusPhysicalOperatorType::TOP_N);
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - delim join internal subtrees are rewritten and tagged",
                 "[plan_tree_shape][isolated_context]")
{
  SECTION("RIGHT_DELIM_JOIN: partition_join points at the build-side PARTITION")
  {
    // TPC-H q17 shape: correlated aggregate whose outer is a filtered join.
    auto plan = generate_sirius_plan(
      *con,
      "SELECT SUM(i.qty) FROM items i, parts p WHERE p.pk = i.fk AND p.pname = 'p1' "
      "AND i.qty < (SELECT 2 * AVG(i2.qty) FROM items i2 WHERE i2.fk = p.pk)");
    INFO(tree_to_string(plan.get()));

    auto* node = find_first(plan.get(), SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
    REQUIRE(node != nullptr);
    auto& delim = node->Cast<sirius::op::sirius_physical_right_delim_join>();
    require_delim_join_common(delim);

    // partition_join is the build-side PARTITION freshly planted by wrap_join.
    auto* build_concat = delim.join->children[1].get();
    REQUIRE(build_concat->type == SiriusPhysicalOperatorType::CONCAT);
    REQUIRE(!build_concat->children.empty());
    auto* build_partition = build_concat->children[0].get();
    REQUIRE(build_partition->type == SiriusPhysicalOperatorType::PARTITION);
    CHECK(static_cast<sirius_physical_operator*>(delim.partition_join) == build_partition);

    // The build-side placeholder DUMMY_SCAN carries no runtime data and stays un-wrapped.
    auto* dummy = find_first(delim.join.get(), SiriusPhysicalOperatorType::DUMMY_SCAN);
    REQUIRE(dummy != nullptr);
    CHECK(dummy->children.empty());
  }

  SECTION("LEFT_DELIM_JOIN: the cached chunk scan sits under the probe-side wrap")
  {
    // TPC-H q21 shape: the mixed-comparison EXISTS keeps the deliminator away and its
    // semi-join decorrelation is a join type the GPU wrap chain supports (a `<` scalar
    // correlation would decorrelate to a SINGLE join, which sirius_physical_concat
    // rejects and production falls back to CPU for).
    auto plan = generate_sirius_plan(
      *con,
      "SELECT l.id FROM big_left l "
      "WHERE EXISTS (SELECT 1 FROM small_right r WHERE r.rid = l.id AND r.other < l.val)");
    INFO(tree_to_string(plan.get()));

    auto* node = find_first(plan.get(), SiriusPhysicalOperatorType::LEFT_DELIM_JOIN);
    REQUIRE(node != nullptr);
    auto& delim = node->Cast<sirius::op::sirius_physical_left_delim_join>();
    require_delim_join_common(delim);

    // The cached chunk scan (filled at runtime by the delim join's fan-out) is buried under
    // the internal join's probe-side CONCAT/PARTITION chain.
    REQUIRE(delim.column_data_scan != nullptr);
    CHECK(contains(delim.join->children[0].get(), delim.column_data_scan));
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - set_parent_ops stamps every operator",
                 "[plan_tree_shape][isolated_context]")
{
  const std::string queries[] = {
    "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid",
    "SELECT val, count(*) FROM big_left GROUP BY val ORDER BY val",
    // RIGHT and LEFT delim joins: parent stamping must descend into the internal
    // `join`/`distinct_root` subtrees.
    "SELECT SUM(i.qty) FROM items i, parts p WHERE p.pk = i.fk AND p.pname = 'p1' "
    "AND i.qty < (SELECT 2 * AVG(i2.qty) FROM items i2 WHERE i2.fk = p.pk)",
    "SELECT l.id FROM big_left l "
    "WHERE EXISTS (SELECT 1 FROM small_right r WHERE r.rid = l.id AND r.other < l.val)",
  };

  for (const auto& query : queries) {
    DYNAMIC_SECTION("query: " << query)
    {
      auto plan = generate_sirius_plan(*con, query);
      INFO(tree_to_string(plan.get()));

      // Root has no parent; every other operator's parent is its tree position, including
      // the delim-join internal subtrees.
      require_parent_links(plan.get(), /*expected_parent=*/nullptr);

      // The sink contract PARTITION's build_pipelines relies on: its tree child terminates
      // the deeper meta-pipeline as sink.
      for (auto* partition : collect(plan.get(), SiriusPhysicalOperatorType::PARTITION)) {
        REQUIRE(partition->children.size() == 1);
        CHECK(partition->children[0]->is_sink());
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Wrap-time physical-sidecar copies (compressed materialization)
//===----------------------------------------------------------------------===//
// These cases drive `insert_gpu_pipeline_operators` over hand-built sirius trees whose leaves
// are PROJECTION operators with manually installed sidecars (a TABLE_SCAN leaf would make
// wrap_table_scan_source throw on the unknown scan function) and assert the wrap-time sidecar
// copies: join children onto CONCAT and PARTITION, HASH_GROUP_BY onto PARTITION and
// GROUPED_AGGREGATE_MERGE.

namespace {

constexpr cudf::data_type kInt8{cudf::type_id::INT8};
constexpr cudf::data_type kInt32{cudf::type_id::INT32};
constexpr cudf::data_type kInt64{cudf::type_id::INT64};

sirius::logical_type wrap_integer_type()
{
  return sirius::logical_type::make(sirius::type_id::INTEGER);
}

duckdb::vector<sirius::logical_type> wrap_integer_types(std::size_t count)
{
  duckdb::vector<sirius::logical_type> types;
  for (std::size_t i = 0; i < count; i++) {
    types.push_back(wrap_integer_type());
  }
  return types;
}

std::unique_ptr<sirius::ast::node> wrap_reference(uint32_t column_index)
{
  return std::make_unique<sirius::ast::node>(
    sirius::ast::reference{column_index, wrap_integer_type()});
}

// A childless pure-reference PROJECTION leaf over @p column_count INTEGER columns, carrying
// @p physical as its sidecar (empty for native).
duckdb::unique_ptr<sirius_physical_operator> make_projection_leaf(
  std::size_t column_count, std::vector<cudf::data_type> physical = {})
{
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  for (std::size_t i = 0; i < column_count; i++) {
    select_list.push_back(wrap_reference(static_cast<uint32_t>(i)));
  }
  auto projection = duckdb::make_uniq<sirius::op::sirius_physical_projection>(
    wrap_integer_types(column_count), std::move(select_list), /*estimated_cardinality=*/1);
  projection->set_physical_types(std::move(physical));
  return projection;
}

// An INNER hash join on column 0 of both sides with a 4-column INTEGER output.
duckdb::unique_ptr<sirius_physical_operator> make_wrap_hash_join(
  duckdb::unique_ptr<sirius_physical_operator> left,
  duckdb::unique_ptr<sirius_physical_operator> right)
{
  duckdb::LogicalDummyScan stub(0);
  stub.types = {duckdb::LogicalType::INTEGER,
                duckdb::LogicalType::INTEGER,
                duckdb::LogicalType::INTEGER,
                duckdb::LogicalType::INTEGER};
  duckdb::vector<sirius::join_condition> conditions;
  sirius::join_condition condition;
  condition.left  = wrap_reference(0);
  condition.right = wrap_reference(0);
  conditions.push_back(std::move(condition));
  return duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(stub,
                                                                  std::move(left),
                                                                  std::move(right),
                                                                  std::move(conditions),
                                                                  duckdb::JoinType::INNER,
                                                                  /*estimated_cardinality=*/1);
}

// A HASH_GROUP_BY grouping on column 0 with SUM(column 1): output [INTEGER key, BIGINT sum].
duckdb::unique_ptr<sirius::op::sirius_physical_grouped_aggregate> make_wrap_grouped_aggregate(
  duckdb::unique_ptr<sirius_physical_operator> child)
{
  duckdb::vector<sirius::logical_type> output_types;
  output_types.push_back(wrap_integer_type());
  output_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups;
  groups.push_back(wrap_reference(0));
  std::vector<std::unique_ptr<sirius::ast::node>> sum_arguments;
  sum_arguments.push_back(wrap_reference(1));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions;
  expressions.push_back(std::make_unique<sirius::ast::node>(
    sirius::ast::aggregate{sirius::aggregate_id::sum,
                           std::move(sum_arguments),
                           sirius::logical_type::make(sirius::type_id::BIGINT),
                           /*distinct=*/false}));
  auto aggregate =
    duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate>(std::move(output_types),
                                                                     std::move(expressions),
                                                                     std::move(groups),
                                                                     /*estimated_cardinality=*/1);
  aggregate->children.push_back(std::move(child));
  return aggregate;
}

// Assert `op` is the CONCAT -> PARTITION join-child wrap and both wrappers carry @p expected
// (empty = sidecar-free). Returns the original wrapped child.
sirius_physical_operator* require_wrap_sidecars(sirius_physical_operator* op,
                                                std::vector<cudf::data_type> const& expected)
{
  REQUIRE(op != nullptr);
  REQUIRE(op->type == SiriusPhysicalOperatorType::CONCAT);
  CHECK(op->get_physical_types() == expected);
  REQUIRE(op->children.size() == 1);
  auto* partition = op->children[0].get();
  REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
  CHECK(partition->get_physical_types() == expected);
  REQUIRE(partition->children.size() == 1);
  return partition->children[0].get();
}

}  // namespace

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - join-child wrap copies the physical sidecar onto CONCAT and "
                 "PARTITION",
                 "[plan_tree_shape][compressed_schema]")
{
  sirius::planner::sirius_physical_plan_generator gen(*con->context);

  SECTION("narrow children stamp both wrappers on both sides")
  {
    std::vector<cudf::data_type> const probe_sidecar{kInt32, kInt8};
    std::vector<cudf::data_type> const build_sidecar{kInt32, kInt8};
    auto plan = make_wrap_hash_join(make_projection_leaf(2, probe_sidecar),
                                    make_projection_leaf(2, build_sidecar));

    gen.insert_gpu_pipeline_operators(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::HASH_JOIN);
    auto* probe_child = require_wrap_sidecars(plan->children[0].get(), probe_sidecar);
    CHECK(probe_child->type == SiriusPhysicalOperatorType::PROJECTION);
    CHECK(probe_child->get_physical_types() == probe_sidecar);
    auto* build_child = require_wrap_sidecars(plan->children[1].get(), build_sidecar);
    CHECK(build_child->type == SiriusPhysicalOperatorType::PROJECTION);
    CHECK(build_child->get_physical_types() == build_sidecar);
  }

  SECTION("a native child leaves the wrappers sidecar-free")
  {
    auto plan = make_wrap_hash_join(make_projection_leaf(2), make_projection_leaf(2));

    gen.insert_gpu_pipeline_operators(plan);

    require_wrap_sidecars(plan->children[0].get(), {});
    require_wrap_sidecars(plan->children[1].get(), {});
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - aggregate wrap copies the group-key sidecar",
                 "[plan_tree_shape][compressed_schema]")
{
  sirius::planner::sirius_physical_plan_generator gen(*con->context);

  SECTION("a group-key sidecar lands on PARTITION and GROUPED_AGGREGATE_MERGE")
  {
    std::vector<cudf::data_type> const aggregate_sidecar{kInt8, kInt64};
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_wrap_grouped_aggregate(make_projection_leaf(2, {kInt8, kInt32}));
    plan->set_physical_types(aggregate_sidecar);

    gen.insert_gpu_pipeline_operators(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    CHECK(plan->get_physical_types() == aggregate_sidecar);
    REQUIRE(plan->children.size() == 1);
    auto* partition = plan->children[0].get();
    REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
    CHECK(partition->get_physical_types() == aggregate_sidecar);
    REQUIRE(partition->children.size() == 1);
    CHECK(partition->children[0]->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
    CHECK(partition->children[0]->get_physical_types() == aggregate_sidecar);
  }

  SECTION("a sidecar-free aggregate leaves both wrappers sidecar-free")
  {
    duckdb::unique_ptr<sirius_physical_operator> plan =
      make_wrap_grouped_aggregate(make_projection_leaf(2));

    gen.insert_gpu_pipeline_operators(plan);

    REQUIRE(plan->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    CHECK(!plan->has_physical_overrides());
    auto* partition = plan->children[0].get();
    REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
    CHECK(!partition->has_physical_overrides());
  }

  SECTION("delim-distinct partitions stay sidecar-free")
  {
    // The propagation delim case restores `distinct_root` in place, so the distinct chain never
    // carries a sidecar; the wrap must not invent one.
    auto delim = duckdb::make_uniq<sirius::op::sirius_physical_delim_join>(
      SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
      wrap_integer_types(1),
      make_projection_leaf(1),
      duckdb::vector<duckdb::const_reference<sirius_physical_operator>>{},
      /*estimated_cardinality=*/1,
      duckdb::optional_idx());
    delim->distinct_root = make_wrap_grouped_aggregate(make_projection_leaf(2));
    delim->children.push_back(make_projection_leaf(1));
    duckdb::unique_ptr<sirius_physical_operator> plan = std::move(delim);

    gen.insert_gpu_pipeline_operators(plan);

    auto& delim_ref = plan->Cast<sirius::op::sirius_physical_delim_join>();
    CHECK_FALSE(delim_ref.declared_output_schema_is_runtime_schema());
    REQUIRE(delim_ref.distinct_root);
    auto* merge = delim_ref.distinct_root.get();
    REQUIRE(merge->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    CHECK(!merge->has_physical_overrides());
    REQUIRE(merge->children.size() == 1);
    auto* partition = merge->children[0].get();
    REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
    CHECK(!partition->has_physical_overrides());
    REQUIRE(partition->children.size() == 1);
    CHECK(partition->children[0]->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - nested-loop-join wrappers stay sidecar-free",
                 "[plan_tree_shape][compressed_schema]")
{
  // NLJ is a propagation native boundary: its children arrive fully restored, so the wrap has
  // nothing to copy.
  sirius::planner::sirius_physical_plan_generator gen(*con->context);

  duckdb::LogicalDummyScan stub(0);
  stub.types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};
  duckdb::unique_ptr<sirius_physical_operator> plan =
    duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(
      stub,
      make_projection_leaf(1),
      make_projection_leaf(1),
      duckdb::vector<sirius::join_condition>{},
      duckdb::JoinType::INNER,
      /*estimated_cardinality=*/1);

  gen.insert_gpu_pipeline_operators(plan);

  REQUIRE(plan->type == SiriusPhysicalOperatorType::NESTED_LOOP_JOIN);
  require_wrap_sidecars(plan->children[0].get(), {});
  require_wrap_sidecars(plan->children[1].get(), {});
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan generation rejects an untranslatable pushed-down table filter",
                 "[plan_tree_shape][table_filter][isolated_context]")
{
  duckdb::TableFunction function;
  function.name                = "seq_scan";
  function.projection_pushdown = true;
  function.filter_pushdown     = true;

  auto get = duckdb::make_uniq<duckdb::LogicalGet>(
    0,
    std::move(function),
    nullptr,
    duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT},
    duckdb::vector<duckdb::string>{"id"});
  get->SetColumnIds({duckdb::ColumnIndex(0)});
  get->projection_ids        = {0};
  get->estimated_cardinality = 1;
  get->table_filters.filters[0] =
    duckdb::make_uniq<duckdb::ExpressionFilter>(untranslatable_table_filter_expression());

  duckdb::unique_ptr<duckdb::LogicalOperator> logical = std::move(get);
  sirius::planner::sirius_physical_plan_generator generator(*con->context);
  CHECK_THROWS_WITH(generator.create_plan(std::move(logical)),
                    Catch::Contains("Unsupported filter predicate on column 'id'"));
}
