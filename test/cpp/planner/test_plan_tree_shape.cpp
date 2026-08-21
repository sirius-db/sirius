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
// Reaching a join operator through these headers instantiates `vector<join_condition>`'s
// destructor, which needs the AST node definition.
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/join_condition.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_projection.hpp"
#include "planner/gpu_admission.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/filter/expression_filter.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

using namespace duckdb;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

namespace {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_plan_tree_shape_XXXXXX";
    int fd      = ::mkstemp(tmpl);
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl);
    _path = tmpl;
  }

  ~scoped_temp_db_path()
  {
    if (!_path.empty()) {
      std::remove(_path.c_str());
      std::remove((_path + ".wal").c_str());
    }
  }

  scoped_temp_db_path(const scoped_temp_db_path&)            = delete;
  scoped_temp_db_path& operator=(const scoped_temp_db_path&) = delete;

  const std::string& path() const { return _path; }

 private:
  std::string _path;
};

/// Generate a Sirius physical plan from a SQL query string. Throws on any failure (after
/// rolling back and restoring the optimizer settings) so a planner regression fails the
/// test instead of silently skipping it. `also_disabled` adds case-specific optimizer exclusions.
duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(
  Connection& con, const std::string& query, const std::vector<OptimizerType>& also_disabled = {})
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  // Keep STATISTICS_PROPAGATION disabled only for this shape-sensitive suite:
  // disabling it lets the deliminator retain the DELIM_JOINs asserted below.
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  disabled.insert(OptimizerType::STATISTICS_PROPAGATION);
  for (auto const optimizer : also_disabled) {
    disabled.insert(optimizer);
  }

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius_physical_operator> result;
  try {
    Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);

    if (context.config.enable_optimizer) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    ColumnBindingResolver::Verify(*plan);
    resolver.VisitOperator(*plan);

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

/// Visit every operator in the tree, including DELIM JOIN internal `join`/`distinct_root`
/// subtrees (owned outside `children[]`).
template <typename Fn>
void for_each_operator(sirius_physical_operator* root, const Fn& fn)
{
  if (!root) { return; }
  fn(root);
  for (auto& child : root->children) {
    for_each_operator(child.get(), fn);
  }
  if (root->type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      root->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    for_each_operator(delim.join.get(), fn);
    for_each_operator(delim.distinct_root.get(), fn);
  }
}

std::vector<sirius_physical_operator*> collect(sirius_physical_operator* root,
                                               SiriusPhysicalOperatorType type)
{
  std::vector<sirius_physical_operator*> out;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type == type) { out.push_back(op); }
  });
  return out;
}

sirius_physical_operator* find_first(sirius_physical_operator* root,
                                     SiriusPhysicalOperatorType type)
{
  auto all = collect(root, type);
  return all.empty() ? nullptr : all.front();
}

bool contains(sirius_physical_operator* root, const sirius_physical_operator* target)
{
  bool found = false;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op == target) { found = true; }
  });
  return found;
}

/// Render the tree (including delim-join internals) for failure diagnostics.
void tree_to_string(sirius_physical_operator* root, int depth, std::ostringstream& out)
{
  if (!root) { return; }
  out << std::string(static_cast<size_t>(depth) * 2, ' ')
      << sirius::op::SiriusPhysicalOperatorToString(root->type) << "\n";
  for (auto& child : root->children) {
    tree_to_string(child.get(), depth + 1, out);
  }
  if (root->type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      root->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    out << std::string(static_cast<size_t>(depth + 1) * 2, ' ') << "(join)\n";
    tree_to_string(delim.join.get(), depth + 2, out);
    out << std::string(static_cast<size_t>(depth + 1) * 2, ' ') << "(distinct_root)\n";
    tree_to_string(delim.distinct_root.get(), depth + 2, out);
  }
}

std::string tree_to_string(sirius_physical_operator* root)
{
  std::ostringstream out;
  tree_to_string(root, 0, out);
  return out.str();
}

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
/// stamp their internal `join`/`distinct_root` subtrees with themselves as parent. A CTE's consumer
/// child (`children[1]`) inherits the CTE's parent because that child produces the CTE's output.
void require_parent_links(sirius_physical_operator* op, sirius_physical_operator* expected_parent)
{
  REQUIRE(op != nullptr);
  INFO("operator " << sirius::op::SiriusPhysicalOperatorToString(op->type));
  CHECK(op->get_parent_op() == expected_parent);
  if (op->type == SiriusPhysicalOperatorType::CTE) {
    REQUIRE(op->children.size() == 2);
    require_parent_links(op->children[0].get(), op);
    require_parent_links(op->children[1].get(), expected_parent);
    return;
  }
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

class dynamic_filter_switch_guard {
 public:
  dynamic_filter_switch_guard(Connection& con, bool enabled)
    : _state(con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state"))
  {
    REQUIRE(_state != nullptr);
    auto& params                 = _state->get_config().get_operator_params();
    _original                    = params.enable_dynamic_filter;
    params.enable_dynamic_filter = enabled;
  }

  ~dynamic_filter_switch_guard()
  {
    _state->get_config().get_operator_params().enable_dynamic_filter = _original;
  }

  dynamic_filter_switch_guard(const dynamic_filter_switch_guard&)            = delete;
  dynamic_filter_switch_guard& operator=(const dynamic_filter_switch_guard&) = delete;

 private:
  duckdb::shared_ptr<duckdb::SiriusContext> _state;
  bool _original = true;
};

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

    // Complete a left-deep join whose outer probe key comes from the inner build side.
    con->Query("CREATE TABLE small_c (ckey INTEGER, cother INTEGER)");
    con->Query("INSERT INTO small_c VALUES (0, 0), (1, 1)");

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
                 "plan tree shape - SIP places a membership endpoint on a nested join's RHS "
                 "input",
                 "[plan_tree_shape][isolated_context]")
{
  // Preserve the left-deep shape and input sides. The outer probe key comes from the nested RHS,
  // so only SIP can place a target there.
  const std::string query =
    "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid "
    "JOIN small_c c ON r.other = c.ckey";
  const std::vector<OptimizerType> keep_shape{OptimizerType::JOIN_ORDER,
                                              OptimizerType::BUILD_SIDE_PROBE_SIDE};

  auto require_nested_join = [](sirius_physical_operator* plan) {
    auto joins = collect(plan, SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(joins.size() == 2);
    REQUIRE(joins[0]->children.size() == 2);
    REQUIRE(contains(joins[0]->children[0].get(), joins[1]));
    return joins[1];
  };

  auto require_rhs_endpoint = [&](const std::string& endpoint_query, JoinType expected_join_type) {
    auto plan = generate_sirius_plan(*con, endpoint_query, keep_shape);
    INFO(tree_to_string(plan.get()));

    auto* nested = require_nested_join(plan.get());
    auto joins   = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    auto const& targets =
      joins[0]->Cast<sirius::op::sirius_physical_hash_join>().dynamic_filter_plan().probe_targets();
    REQUIRE(targets.size() == 1);
    CHECK(targets[0].route_class == sirius::op::dynamic_filter_route_class::scan);
    CHECK(targets[0].accepts_zone_map_filters);
    CHECK(std::none_of(targets.begin(), targets.end(), [](auto const& target) {
      return target.route_class == sirius::op::dynamic_filter_route_class::direct;
    }));

    REQUIRE(nested->Cast<sirius::op::sirius_physical_hash_join>().join_type == expected_join_type);

    auto* build_subtree =
      require_join_child_wrap(nested->children[1].get(), nested, /*is_build=*/true);
    REQUIRE(build_subtree != nullptr);
    REQUIRE(build_subtree->type == SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(build_subtree->children.size() == 1);
    CHECK(build_subtree->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  };

  SECTION("filter on, unfiltered build: no endpoint wires anywhere")
  {
    // An unfiltered base-table build supplies neither filter nor opaque-build evidence, and either
    // kind would arm discovery for both routes -- with neither, no route wires.
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con, query, keep_shape);
    INFO(tree_to_string(plan.get()));

    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());

    auto* inner = require_nested_join(plan.get());
    auto* build_subtree =
      require_join_child_wrap(inner->children[1].get(), inner, /*is_build=*/true);
    REQUIRE(build_subtree != nullptr);
    CHECK(build_subtree->type == SiriusPhysicalOperatorType::GPU_SCAN);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("filter off: no endpoint anywhere; the build wrap holds the bare scan")
  {
    dynamic_filter_switch_guard switch_off(*con, /*enabled=*/false);
    auto plan = generate_sirius_plan(*con, query, keep_shape);
    INFO(tree_to_string(plan.get()));

    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());

    auto* inner = require_nested_join(plan.get());
    auto* build_subtree =
      require_join_child_wrap(inner->children[1].get(), inner, /*is_build=*/true);
    REQUIRE(build_subtree != nullptr);
    CHECK(build_subtree->type == SiriusPhysicalOperatorType::GPU_SCAN);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("filter on: equality semantics allow descent into a LEFT join's build input")
  {
    // The producing join's build filter (pushed into the small_c scan) supplies the required
    // evidence; holding back join reordering preserves the LEFT join.
    const std::string left_query =
      "SELECT * FROM big_left l LEFT JOIN small_right r ON l.id = r.rid "
      "JOIN small_c c ON r.other = c.ckey WHERE c.cother >= 0";

    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    require_rhs_endpoint(left_query, JoinType::LEFT);
  }

  SECTION("filter on: RIGHT and equality-safe OUTER joins admit build-input descent")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    require_rhs_endpoint(
      "SELECT * FROM big_left l RIGHT JOIN small_right r ON l.id = r.rid "
      "JOIN small_c c ON r.other = c.ckey WHERE c.cother >= 0",
      JoinType::RIGHT);
    require_rhs_endpoint(
      "SELECT * FROM big_left l FULL OUTER JOIN small_right r ON l.id = r.rid "
      "JOIN small_c c ON r.other = c.ckey WHERE c.cother >= 0",
      JoinType::OUTER);
  }

  SECTION("filter on: null-equal semantics block OUTER build-input descent")
  {
    // Pruning can turn an OUTER match into a NULL-padded row, which a null-equal producer could
    // accept. The build filter supplies evidence, leaving key admission as the blocking rule.
    const std::string null_equal_query =
      "SELECT * FROM big_left l FULL OUTER JOIN small_right r ON l.id = r.rid "
      "JOIN small_c c ON r.other IS NOT DISTINCT FROM c.ckey WHERE c.cother >= 0";

    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con, null_equal_query, keep_shape);
    INFO(tree_to_string(plan.get()));

    auto* nested = require_nested_join(plan.get());
    REQUIRE(nested->Cast<sirius::op::sirius_physical_hash_join>().join_type == JoinType::OUTER);
    auto* build_subtree =
      require_join_child_wrap(nested->children[1].get(), nested, /*is_build=*/true);
    REQUIRE(build_subtree != nullptr);
    CHECK(build_subtree->type == SiriusPhysicalOperatorType::GPU_SCAN);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("filter on: null-equal semantics block OUTER probe-input descent")
  {
    const std::string null_equal_query =
      "SELECT * FROM big_left l FULL OUTER JOIN small_right r ON l.id = r.rid "
      "JOIN small_c c ON l.id IS NOT DISTINCT FROM c.ckey WHERE c.cother >= 0";

    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con, null_equal_query, keep_shape);
    INFO(tree_to_string(plan.get()));

    auto* nested = require_nested_join(plan.get());
    REQUIRE(nested->Cast<sirius::op::sirius_physical_hash_join>().join_type == JoinType::OUTER);
    auto* probe_subtree =
      require_join_child_wrap(nested->children[0].get(), nested, /*is_build=*/false);
    REQUIRE(probe_subtree != nullptr);
    CHECK(probe_subtree->type == SiriusPhysicalOperatorType::GPU_SCAN);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("filter on: an aggregate-result key stops the descent at the group-by")
  {
    // The build filter arms discovery, but aggregate results cannot trace through HASH_GROUP_BY,
    // so the endpoint stays above the wrapped group-by chain and consumes merged rather than
    // partitioned batches.
    const std::string aggregate_key_query =
      "SELECT * FROM (SELECT rid, min(other) AS m FROM small_right GROUP BY rid) g "
      "JOIN small_c c ON g.m = c.ckey WHERE c.cother >= 0";

    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con, aggregate_key_query, keep_shape);
    INFO(tree_to_string(plan.get()));

    // Disabled statistics prevent selection of the separate perfect-hash aggregate path.
    REQUIRE(find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY) != nullptr);

    auto endpoints = collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(endpoints.size() == 1);
    auto* endpoint = endpoints.front();

    // The endpoint must remain above the partitioned group-by chain.
    REQUIRE(endpoint->children.size() == 1);
    CHECK(endpoint->children[0]->type != SiriusPhysicalOperatorType::PARTITION);
    CHECK(endpoint->children[0]->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("filter on, filtered build: the trace through the build block binds the scan route")
  {
    // Build-filter evidence lets the trace cross the inner build block and bind the small_right
    // scan. A scan binding takes precedence over a direct endpoint for the same key.
    const std::string filtered_build_query =
      "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid "
      "JOIN small_c c ON r.other = c.ckey WHERE c.cother >= 0";

    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con, filtered_build_query, keep_shape);
    INFO(tree_to_string(plan.get()));

    auto joins = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(joins.size() == 2);
    REQUIRE(joins[0]->children.size() == 2);
    REQUIRE(contains(joins[0]->children[0].get(), joins[1]));
    auto const& outer = joins[0]->Cast<sirius::op::sirius_physical_hash_join>();

    auto const& targets = outer.dynamic_filter_plan().probe_targets();
    REQUIRE(targets.size() == 1);
    CHECK(targets[0].route_class == sirius::op::dynamic_filter_route_class::scan);
    CHECK(targets[0].accepts_zone_map_filters);
    CHECK(std::none_of(targets.begin(), targets.end(), [](auto const& target) {
      return target.route_class == sirius::op::dynamic_filter_route_class::direct;
    }));

    auto* inner = joins[1];
    auto* build_subtree =
      require_join_child_wrap(inner->children[1].get(), inner, /*is_build=*/true);
    REQUIRE(build_subtree != nullptr);
    REQUIRE(build_subtree->type == SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(build_subtree->children.size() == 1);
    CHECK(build_subtree->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - scan-route discovery wires the probe scan's DYNAMIC_FILTER",
                 "[plan_tree_shape][isolated_context]")
{
  const std::vector<OptimizerType> keep_shape{OptimizerType::JOIN_ORDER,
                                              OptimizerType::BUILD_SIDE_PROBE_SIDE};

  SECTION("filtered build: the probe scan is wrapped in a DYNAMIC_FILTER")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con,
                                     "SELECT * FROM big_left l JOIN "
                                     "(SELECT * FROM small_right WHERE other % 2 = 0) r "
                                     "ON l.id = r.rid",
                                     keep_shape);
    INFO(tree_to_string(plan.get()));

    auto endpoints = collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(endpoints.size() == 1);
    REQUIRE(endpoints.front()->children.size() == 1);
    CHECK(endpoints.front()->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN);

    // The endpoint is on the probe side (children[0]).
    auto* hj = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(hj != nullptr);
    CHECK(contains(hj->children[0].get(), endpoints.front()));

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("unfiltered build: no DYNAMIC_FILTER anywhere")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(
      *con, "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid", keep_shape);
    INFO(tree_to_string(plan.get()));

    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
  }

  SECTION("unfiltered aggregate build: no DYNAMIC_FILTER anywhere")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con,
                                     "SELECT * FROM big_left l JOIN "
                                     "(SELECT rid, count(*) c FROM small_right GROUP BY rid) r "
                                     "ON l.id = r.rid",
                                     keep_shape);
    INFO(tree_to_string(plan.get()));

    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("filtered aggregate build: filter evidence still wires the scan route")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con,
                                     "SELECT * FROM big_left l JOIN "
                                     "(SELECT rid, count(*) c FROM small_right "
                                     " WHERE other % 2 = 0 GROUP BY rid) r "
                                     "ON l.id = r.rid",
                                     keep_shape);
    INFO(tree_to_string(plan.get()));

    auto* outer = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(outer != nullptr);
    auto const& targets =
      outer->Cast<sirius::op::sirius_physical_hash_join>().dynamic_filter_plan().probe_targets();
    REQUIRE(targets.size() == 1);
    CHECK(targets.front().route_class == sirius::op::dynamic_filter_route_class::scan);

    auto endpoints = collect(outer->children[0].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(endpoints.size() == 1);
    REQUIRE(endpoints.front()->children.size() == 1);
    CHECK(endpoints.front()->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN);
    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("unfiltered join-output build: no outer dynamic filter is wired")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con,
                                     "SELECT * FROM big_left l JOIN "
                                     "(SELECT r.rid FROM small_right r JOIN small_c c "
                                     " ON r.other = c.ckey) j "
                                     "ON l.id = j.rid",
                                     keep_shape);
    INFO(tree_to_string(plan.get()));

    auto* outer = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(outer != nullptr);
    CHECK_FALSE(
      outer->Cast<sirius::op::sirius_physical_hash_join>().dynamic_filter_plan().enabled());
    CHECK(collect(outer->children[0].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("filtered join-output build: filter evidence still wires the outer scan route")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con,
                                     "SELECT * FROM big_left l JOIN "
                                     "(SELECT r.rid FROM small_right r JOIN small_c c "
                                     " ON r.other = c.ckey WHERE c.cother % 2 = 0) j "
                                     "ON l.id = j.rid",
                                     keep_shape);
    INFO(tree_to_string(plan.get()));

    auto* outer = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(outer != nullptr);
    auto const& targets =
      outer->Cast<sirius::op::sirius_physical_hash_join>().dynamic_filter_plan().probe_targets();
    REQUIRE(targets.size() == 1);
    CHECK(targets.front().route_class == sirius::op::dynamic_filter_route_class::scan);

    auto endpoints = collect(outer->children[0].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(endpoints.size() == 1);
    REQUIRE(endpoints.front()->children.size() == 1);
    CHECK(endpoints.front()->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN);
    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION(
    "a LIMIT between the join and the scan: the endpoint stays above the LIMIT, the scan "
    "stays bare")
  {
    // Filtering below LIMIT could change its rows, so the scan trace stops and the join-edge
    // endpoint wraps it.
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con,
                                     "SELECT * FROM (SELECT * FROM big_left LIMIT 15) l JOIN "
                                     "(SELECT * FROM small_right WHERE other % 2 = 0) r "
                                     "ON l.id = r.rid",
                                     keep_shape);
    INFO(tree_to_string(plan.get()));

    auto endpoints = collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(endpoints.size() == 1);
    REQUIRE(endpoints.front()->children.size() == 1);
    CHECK(endpoints.front()->children[0]->type == SiriusPhysicalOperatorType::STREAMING_LIMIT);

    auto* scan = find_first(endpoints.front(), SiriusPhysicalOperatorType::GPU_SCAN);
    REQUIRE(scan != nullptr);
    REQUIRE(scan->get_parent_op() != nullptr);
    CHECK(scan->get_parent_op()->type != SiriusPhysicalOperatorType::DYNAMIC_FILTER);
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - an opaque build arms the scan route",
                 "[plan_tree_shape][isolated_context]")
{
  // IsFiltering cannot inspect the CTE definition through its childless CTE_SCAN build, so
  // opaque-build evidence arms discovery.
  const std::string cte_query =
    "WITH r AS MATERIALIZED (SELECT rid FROM small_right WHERE other % 2 = 0) "
    "SELECT * FROM big_left l JOIN r ON l.id = r.rid";
  const std::vector<OptimizerType> keep_shape{OptimizerType::JOIN_ORDER,
                                              OptimizerType::BUILD_SIDE_PROBE_SIDE};

  SECTION("materialized-CTE build, filter on: the scan route wraps the probe scan")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con, cte_query, keep_shape);
    INFO(tree_to_string(plan.get()));

    auto endpoints = collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(endpoints.size() == 1);
    REQUIRE(endpoints.front()->children.size() == 1);
    CHECK(endpoints.front()->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN);

    auto* hj = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(hj != nullptr);
    REQUIRE(hj->children.size() == 2);
    auto* probe_subtree = require_join_child_wrap(hj->children[0].get(), hj, /*is_build=*/false);
    CHECK(contains(probe_subtree, endpoints.front()));

    auto const& targets =
      hj->Cast<sirius::op::sirius_physical_hash_join>().dynamic_filter_plan().probe_targets();
    REQUIRE(targets.size() == 1);
    CHECK(targets[0].route_class == sirius::op::dynamic_filter_route_class::scan);
    CHECK(targets[0].accepts_zone_map_filters);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("materialized-CTE build, filter off: no DYNAMIC_FILTER anywhere")
  {
    dynamic_filter_switch_guard switch_off(*con, /*enabled=*/false);
    auto plan = generate_sirius_plan(*con, cte_query, keep_shape);
    INFO(tree_to_string(plan.get()));

    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
  }

  // This reproduces q17's delim shape: the correlated side builds from a DELIM_GET and wires only
  // through opaque-build evidence, while the flat side wires an ordinary scan-route endpoint.
  const std::string delim_query =
    "SELECT SUM(i.qty) FROM items i, parts p WHERE p.pk = i.fk AND p.pname = 'p1' "
    "AND i.qty < (SELECT 2 * AVG(i2.qty) FROM items i2 WHERE i2.fk = p.pk)";

  SECTION("delim-scan build, filter on: the scan route binds inside the delim internals")
  {
    dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
    auto plan = generate_sirius_plan(*con, delim_query);
    INFO(tree_to_string(plan.get()));

    auto* node = find_first(plan.get(), SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
    REQUIRE(node != nullptr);
    auto& delim = node->Cast<sirius::op::sirius_physical_right_delim_join>();

    auto endpoints = collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(endpoints.size() == 2);
    auto const inside =
      std::count_if(endpoints.begin(), endpoints.end(), [&](sirius_physical_operator* endpoint) {
        return contains(delim.join.get(), endpoint);
      });
    CHECK(inside == 1);
    for (auto* endpoint : endpoints) {
      REQUIRE(endpoint->children.size() == 1);
      CHECK(endpoint->children[0]->type == SiriusPhysicalOperatorType::GPU_SCAN);
    }

    std::vector<sirius::op::sirius_physical_hash_join*> armed_internal_joins;
    for (auto* join_node : collect(delim.join.get(), SiriusPhysicalOperatorType::HASH_JOIN)) {
      auto& internal_join = join_node->Cast<sirius::op::sirius_physical_hash_join>();
      if (!internal_join.dynamic_filter_plan().probe_targets().empty()) {
        armed_internal_joins.push_back(&internal_join);
      }
    }
    REQUIRE(armed_internal_joins.size() == 1);
    auto const& internal_targets =
      armed_internal_joins.front()->dynamic_filter_plan().probe_targets();
    REQUIRE(internal_targets.size() == 1);
    CHECK(internal_targets[0].route_class == sirius::op::dynamic_filter_route_class::scan);
    CHECK(internal_targets[0].accepts_zone_map_filters);

    require_parent_links(plan.get(), /*expected_parent=*/nullptr);
  }

  SECTION("delim-scan build, filter off: no endpoint anywhere")
  {
    dynamic_filter_switch_guard switch_off(*con, /*enabled=*/false);
    auto plan = generate_sirius_plan(*con, delim_query);
    INFO(tree_to_string(plan.get()));

    REQUIRE(find_first(plan.get(), SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) != nullptr);
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
  }
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
  return duckdb::make_uniq<sirius::op::sirius_physical_hash_join>(
    stub,
    std::move(left),
    std::move(right),
    std::move(conditions),
    duckdb::JoinType::INNER,
    /*left_projection_map=*/duckdb::vector<std::size_t>{},
    /*right_projection_map=*/duckdb::vector<std::size_t>{},
    /*delim_types=*/duckdb::vector<sirius::logical_type>{},
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

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - admission scan walk reaches delim join subtrees",
                 "[plan_tree_shape][gpu_admission][isolated_context]")
{
  // GPU admission sizes a query from its scans' estimated_cardinality, so a scan it cannot
  // see is a scan whose bytes are not counted — under-estimating the query and admitting it
  // onto too few GPUs. DELIM JOIN owns `join`/`distinct_root` outside `children[]`, so a walk
  // over `children` alone misses everything inside them.
  //
  // Cross-check planner::collect_gpu_scans against this suite's independently written
  // for_each_operator: the two descend the tree separately and must agree.
  auto plan = generate_sirius_plan(
    *con,
    "SELECT SUM(i.qty) FROM items i, parts p WHERE p.pk = i.fk AND p.pname = 'p1' "
    "AND i.qty < (SELECT 2 * AVG(i2.qty) FROM items i2 WHERE i2.fk = p.pk)");
  INFO(tree_to_string(plan.get()));

  REQUIRE(find_first(plan.get(), SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) != nullptr);

  auto const expected = collect(plan.get(), SiriusPhysicalOperatorType::GPU_SCAN);
  REQUIRE_FALSE(expected.empty());

  std::vector<const sirius_physical_operator*> found;
  sirius::planner::collect_gpu_scans(*plan, found);

  CHECK(found.size() == expected.size());
  for (auto const* op : expected) {
    CHECK(std::find(found.begin(), found.end(), op) != found.end());
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - admission scan walk agrees on a plain join",
                 "[plan_tree_shape][gpu_admission][isolated_context]")
{
  // Baseline: with no operator holding subtrees outside children[], both walks agree
  // trivially. Guards against a descent that double-counts.
  auto plan =
    generate_sirius_plan(*con, "SELECT b.val FROM big_left b, small_right s WHERE b.id = s.rid");
  INFO(tree_to_string(plan.get()));

  auto const expected = collect(plan.get(), SiriusPhysicalOperatorType::GPU_SCAN);
  REQUIRE(expected.size() == 2);

  std::vector<const sirius_physical_operator*> found;
  sirius::planner::collect_gpu_scans(*plan, found);
  CHECK(found.size() == expected.size());
}
