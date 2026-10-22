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

#include <duckdb/main/settings.hpp>
// Reaching a join operator through these headers instantiates `vector<join_condition>`'s
// destructor, which needs the AST node definition.
#include "expression/ast/node.hpp"
#include "expression/ast/reference.hpp"
#include "expression/join_condition.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_projection.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/gpu_admission.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "util/env_guard.hpp"
#include "utils/pipeline_conversion_test_utils.hpp"
#include "utils/scoped_sirius_setting.hpp"
#include "utils/sirius_test_env.hpp"

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
#include <exception>
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
    Parser parser(context);
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);

    if (duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(context)) {
      Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    ColumnBindingResolver resolver;
    duckdb::ColumnBindingResolver verifier(true);
    verifier.VisitOperator(*plan);
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
    duckdb::BoundScalarFunction(duckdb::ScalarFunction("sirius_unmapped_filter",
                                                       {duckdb::LogicalType::BIGINT},
                                                       duckdb::LogicalType::BOOLEAN,
                                                       nullptr)),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},
    nullptr);
  expression->GetChildrenMutable().push_back(
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
  CHECK(join->children[is_build ? 1 : 0].get() == op);
  CHECK(join->input_port_for(*op) == (is_build ? "build" : "default"));

  REQUIRE(op->children.size() == 1);
  auto* partition = op->children[0].get();
  REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
  CHECK(partition->Cast<sirius::op::sirius_physical_partition>().exchange() != nullptr);

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
  CHECK(partition->Cast<sirius::op::sirius_physical_partition>().exchange() != nullptr);

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
    db  = sirius::test::open_sirius_db(_db_path.path().c_str(), cfg);
    con = std::make_unique<Connection>(*db);
    // This suite checks the explicit DELIM_JOIN operator's wiring.
    REQUIRE_FALSE(con->Query("SET delim_join_as_cte=false")->HasError());

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

  REQUIRE_THROWS_WITH(
    generate_sirius_plan(*con, "SELECT wide, narrow FROM mixed_schema"),
    Catch::Matchers::ContainsSubstring("GPU scan output column 1 (DECIMAL(4,2)) has no native cuDF "
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
                                              OptimizerType::BUILD_SIDE_PROBE_SIDE,
                                              OptimizerType::OUTER_JOIN_SIMPLIFICATION};

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
                 "plan tree shape - native CTE deduplication preserves build-filter evidence",
                 "[plan_tree_shape][isolated_context][dynamic_filter][cte_dedup]")
{
  sirius::test::scoped_sirius_setting native_cte(*con, "delim_join_as_cte", true);
  bool const enable_filter = GENERATE(false, true);
  // A per-connection SET: plans read the connection's operator-options snapshot, so writing the
  // shared config (dynamic_filter_switch_guard) no longer turns the filter off.
  sirius::test::scoped_sirius_setting filter_switch(*con, "enable_dynamic_filter", enable_filter);
  auto plan = generate_sirius_plan(
    *con,
    "SELECT i.qty FROM items i JOIN parts p ON p.pk = i.fk WHERE p.pname = 'p1' "
    "AND i.qty < (SELECT 2 * AVG(i2.qty) FROM items i2 WHERE i2.fk = p.pk)");
  INFO(tree_to_string(plan.get()));

  REQUIRE(find_first(plan.get(), SiriusPhysicalOperatorType::CTE) != nullptr);
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::LEFT_DELIM_JOIN).empty());
  REQUIRE(collect(plan.get(), SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN).empty());

  auto skip_projections = [](sirius_physical_operator* op) {
    while (op->type == SiriusPhysicalOperatorType::PROJECTION) {
      REQUIRE(op->children.size() == 1);
      op = op->children[0].get();
    }
    return op;
  };
  std::size_t dedup_builds = 0;
  for (auto* node : collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN)) {
    auto& join = node->Cast<sirius::op::sirius_physical_hash_join>();
    REQUIRE(join.children.size() == 2);
    auto* build =
      skip_projections(require_join_child_wrap(join.children[1].get(), &join, /*is_build=*/true));
    if (build->type != SiriusPhysicalOperatorType::MERGE_GROUP_BY) { continue; }
    REQUIRE(build->children.size() == 1);
    auto* partition = build->children[0].get();
    if (partition->type != SiriusPhysicalOperatorType::PARTITION) { continue; }
    REQUIRE(partition->children.size() == 1);
    auto* aggregate = partition->children[0].get();
    REQUIRE(aggregate->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
    auto& group_by = aggregate->Cast<sirius::op::sirius_physical_grouped_aggregate>();
    if (!group_by.cudf_aggregates.empty()) { continue; }
    REQUIRE(aggregate->children.size() == 1);
    if (skip_projections(aggregate->children[0].get())->type !=
        SiriusPhysicalOperatorType::CTE_SCAN) {
      continue;
    }

    ++dedup_builds;
    auto const& targets = join.dynamic_filter_plan().probe_targets();
    if (enable_filter) {
      REQUIRE_FALSE(targets.empty());
      CHECK(std::any_of(targets.begin(), targets.end(), [](auto const& target) {
        return target.route_class == sirius::op::dynamic_filter_route_class::scan;
      }));
      CHECK(find_first(join.children[0].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER) !=
            nullptr);
    } else {
      CHECK(targets.empty());
    }
  }
  REQUIRE(dedup_builds > 0);
  require_parent_links(plan.get(), /*expected_parent=*/nullptr);
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - shared CTE filtering stays within one consumer",
                 "[plan_tree_shape][isolated_context][dynamic_filter][cte_dedup]")
{
  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  auto plan = generate_sirius_plan(
    *con,
    "WITH shared AS MATERIALIZED (SELECT id, val FROM big_left), "
    "domain AS MATERIALIZED (SELECT rid FROM small_right WHERE other = 0) "
    "SELECT s.id, s.val FROM shared s JOIN (SELECT rid FROM domain GROUP BY rid) d "
    "ON s.id = d.rid UNION ALL SELECT id, val FROM shared WHERE val > 3");
  INFO(tree_to_string(plan.get()));
  auto ctes = collect(plan.get(), SiriusPhysicalOperatorType::CTE);
  REQUIRE(ctes.size() == 2);
  for (auto* cte : ctes) {
    REQUIRE(cte->children.size() == 2);
    CHECK(collect(cte->children[0].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
    CHECK(find_first(cte->children[0].get(), SiriusPhysicalOperatorType::GPU_SCAN) != nullptr);
  }

  auto joins = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
  REQUIRE(joins.size() == 1);
  auto& join = joins.front()->Cast<sirius::op::sirius_physical_hash_join>();
  REQUIRE(join.children.size() == 2);
  auto skip_projections = [](sirius_physical_operator* op) {
    while (op->type == SiriusPhysicalOperatorType::PROJECTION) {
      REQUIRE(op->children.size() == 1);
      op = op->children[0].get();
    }
    return op;
  };

  auto* probe =
    skip_projections(require_join_child_wrap(join.children[0].get(), &join, /*is_build=*/false));
  REQUIRE(probe->type == SiriusPhysicalOperatorType::DYNAMIC_FILTER);
  REQUIRE(probe->children.size() == 1);
  REQUIRE(skip_projections(probe->children[0].get())->type == SiriusPhysicalOperatorType::CTE_SCAN);
  auto* build =
    skip_projections(require_join_child_wrap(join.children[1].get(), &join, /*is_build=*/true));
  REQUIRE(build->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
  REQUIRE(build->children.size() == 1);
  auto* aggregate = build->children[0].get();
  if (aggregate->type == SiriusPhysicalOperatorType::PARTITION) {
    REQUIRE(aggregate->children.size() == 1);
    aggregate = aggregate->children[0].get();
  }
  REQUIRE(aggregate->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
  CHECK(aggregate->Cast<sirius::op::sirius_physical_grouped_aggregate>().cudf_aggregates.empty());
  REQUIRE(aggregate->children.size() == 1);
  REQUIRE(skip_projections(aggregate->children[0].get())->type ==
          SiriusPhysicalOperatorType::CTE_SCAN);
  auto const& targets = join.dynamic_filter_plan().probe_targets();
  REQUIRE(targets.size() == 1);
  CHECK(targets[0].route_class == sirius::op::dynamic_filter_route_class::direct);
  CHECK_FALSE(targets[0].accepts_zone_map_filters);
  REQUIRE(targets[0].filter_set != nullptr);
  REQUIRE(targets[0].key_bindings.size() == 1);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).size() == 1);
  require_parent_links(plan.get(), /*expected_parent=*/nullptr);

  // Closing this consumer must close the channel advertised by its producing join.
  REQUIRE(targets[0].filter_set->accepting_filters());
  probe->Cast<sirius::op::scan::sirius_physical_dynamic_filter>().on_finalize_operator();
  CHECK_FALSE(targets[0].filter_set->accepting_filters());
}

TEST_CASE_METHOD(
  plan_tree_shape_fixture,
  "plan tree shape - null-safe right-family filters require a nonnull key and early publication",
  "[plan_tree_shape][isolated_context][dynamic_filter][right_family][join_key_nullability]")
{
  std::string const join_type    = GENERATE("SEMI", "ANTI");
  std::string const key_type     = GENERATE("INTEGER", "BIGINT");
  bool const enable_accumulation = GENERATE(false, true);
  bool const probe_not_null      = GENERATE(false, true);
  CAPTURE(join_type, key_type, enable_accumulation, probe_not_null);
  REQUIRE_FALSE(
    con->Query("CREATE TABLE right_probe(k " + key_type + (probe_not_null ? " NOT NULL" : "") + ")")
      ->HasError());
  REQUIRE_FALSE(con->Query("INSERT INTO right_probe SELECT range FROM range(20)")->HasError());
  REQUIRE_FALSE(
    con->Query("CREATE TABLE right_build(k " + key_type + ", keep INTEGER)")->HasError());
  REQUIRE_FALSE(con->Query("INSERT INTO right_build VALUES (0,0), (1,1), (NULL,0)")->HasError());

  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  sirius::test::scoped_sirius_setting accumulation(
    *con, "enable_dynamic_filter_multi_partition", enable_accumulation);
  auto plan = generate_sirius_plan(*con,
                                   "SELECT r.k FROM right_build r " + join_type +
                                     " JOIN right_probe l ON r.k IS NOT DISTINCT FROM l.k "
                                     "WHERE r.keep = 0");
  INFO(tree_to_string(plan.get()));

  auto joins = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
  REQUIRE(joins.size() == 1);
  auto& join = joins.front()->Cast<sirius::op::sirius_physical_hash_join>();
  REQUIRE(join.join_type == (join_type == "SEMI" ? JoinType::RIGHT_SEMI : JoinType::RIGHT_ANTI));
  REQUIRE(join.conditions.size() == 1);
  CHECK(
    join.conditions[0].comparison ==
    (probe_not_null ? sirius::comparison_type::equal : sirius::comparison_type::not_distinct_from));
  auto const& targets = join.dynamic_filter_plan().probe_targets();
  REQUIRE(join.children.size() == 2);
  auto* build_root = require_join_child_wrap(join.children[1].get(), &join, /*is_build=*/true);
  if (enable_accumulation && probe_not_null) {
    REQUIRE(targets.size() == 1);
    CHECK(targets[0].route_class == sirius::op::dynamic_filter_route_class::scan);
    CHECK(find_first(join.children[0].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER) !=
          nullptr);
    CHECK(find_first(join.children[1].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER) ==
          nullptr);
    REQUIRE(build_root->type == SiriusPhysicalOperatorType::BUILD_FILTER);
    REQUIRE(build_root->children.size() == 1);
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::BUILD_FILTER).size() == 1);
  } else {
    CHECK(targets.empty());
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::BUILD_FILTER).empty());
  }
  require_parent_links(plan.get(), /*expected_parent=*/nullptr);
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - early build publication stays separate from merge fusion",
                 "[plan_tree_shape][isolated_context][dynamic_filter][right_family][pipeline]")
{
  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  sirius::test::scoped_sirius_setting accumulation(
    *con, "enable_dynamic_filter_multi_partition", true);
  sirius::test::scoped_sirius_setting fusion(*con, "fuse_merge_pipelines", true);
  sirius::test::with_conversion_result(
    *con,
    "SELECT r.rid FROM (SELECT rid FROM small_right WHERE other = 0 GROUP BY rid) r "
    "SEMI JOIN big_left l ON r.rid = l.id",
    [](auto const& result) {
      sirius::pipeline::sirius_pipeline const* publication_pipeline = nullptr;
      for (auto const& pipeline : result.scheduled_pipelines) {
        auto const* source = pipeline->get_source().get();
        if (source && source->type == SiriusPhysicalOperatorType::BUILD_FILTER) {
          REQUIRE(publication_pipeline == nullptr);
          publication_pipeline = pipeline.get();
        }
      }
      REQUIRE(publication_pipeline != nullptr);
      auto const operators = publication_pipeline->get_operators();
      REQUIRE(operators.size() == 1);
      auto const* publisher = &operators[0].get();
      CHECK(publication_pipeline->get_sink().get() == publisher);
      REQUIRE(publisher->children.size() == 1);
      auto* merge =
        find_first(publisher->children[0].get(), SiriusPhysicalOperatorType::MERGE_GROUP_BY);
      REQUIRE(merge != nullptr);
      if (publisher->children[0].get() == merge) {
        CHECK_FALSE(
          merge->Cast<sirius::op::sirius_physical_grouped_aggregate_merge>().fuse_into_parent());
      }

      std::size_t input_count = 0;
      for (auto const& wiring : result.repository_wirings) {
        if (wiring.dest_pipeline.get() != publication_pipeline) { continue; }
        ++input_count;
        CHECK(wiring.barrier_type == sirius::op::MemoryBarrierType::FULL);
        REQUIRE(wiring.source_pipeline);
        CHECK(wiring.source_pipeline.get() != publication_pipeline);
        CHECK(wiring.source_pipeline->get_sink().get() == publisher->children[0].get());
      }
      REQUIRE(input_count == 1);
    });
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - early CTE build filtering preserves materialization boundaries",
                 "[plan_tree_shape][isolated_context][dynamic_filter][right_family][cte_dedup]")
{
  std::string const join_type = GENERATE("SEMI", "ANTI");
  bool const deduplicate      = GENERATE(false, true);
  CAPTURE(join_type, deduplicate);
  dynamic_filter_switch_guard switch_on(*con, /*enabled=*/true);
  sirius::test::scoped_sirius_setting accumulation(
    *con, "enable_dynamic_filter_multi_partition", true);
  auto const grouping = deduplicate ? " GROUP BY rid" : "";
  auto plan =
    generate_sirius_plan(*con,
                         "WITH shared AS MATERIALIZED (SELECT rid FROM small_right) "
                         "SELECT r.rid FROM (SELECT rid FROM shared" +
                           std::string(grouping) + ") r " + join_type +
                           " JOIN big_left l ON r.rid = l.id UNION ALL SELECT rid FROM shared");
  INFO(tree_to_string(plan.get()));
  auto ctes = collect(plan.get(), SiriusPhysicalOperatorType::CTE);
  REQUIRE(ctes.size() == 1);
  REQUIRE(ctes[0]->children.size() == 2);
  CHECK(collect(ctes[0]->children[0].get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER).empty());
  CHECK(collect(ctes[0]->children[0].get(), SiriusPhysicalOperatorType::BUILD_FILTER).empty());

  auto joins = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
  REQUIRE(joins.size() == 1);
  auto& join = joins[0]->Cast<sirius::op::sirius_physical_hash_join>();
  REQUIRE(join.join_type == (join_type == "SEMI" ? JoinType::RIGHT_SEMI : JoinType::RIGHT_ANTI));
  REQUIRE(join.children.size() == 2);
  REQUIRE(join.dynamic_filter_plan().probe_targets().size() == 1);
  CHECK(join.dynamic_filter_plan().probe_targets()[0].route_class ==
        sirius::op::dynamic_filter_route_class::scan);
  auto* stage = require_join_child_wrap(join.children[1].get(), &join, /*is_build=*/true);
  REQUIRE(stage->type == SiriusPhysicalOperatorType::BUILD_FILTER);
  REQUIRE(stage->children.size() == 1);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::BUILD_FILTER).size() == 1);

  auto skip_projections = [](sirius_physical_operator* op) {
    while (op->type == SiriusPhysicalOperatorType::PROJECTION) {
      REQUIRE(op->children.size() == 1);
      op = op->children[0].get();
    }
    return op;
  };
  auto* input = skip_projections(stage->children[0].get());
  if (deduplicate) {
    REQUIRE(input->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    REQUIRE(input->children.size() == 1);
    input = input->children[0].get();
    if (input->type == SiriusPhysicalOperatorType::PARTITION) {
      REQUIRE(input->children.size() == 1);
      input = input->children[0].get();
    }
    REQUIRE(input->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
    CHECK(input->Cast<sirius::op::sirius_physical_grouped_aggregate>().cudf_aggregates.empty());
    REQUIRE(input->children.size() == 1);
    input = skip_projections(input->children[0].get());
  }
  REQUIRE(input->type == SiriusPhysicalOperatorType::CTE_SCAN);
  CHECK(collect(plan.get(), SiriusPhysicalOperatorType::CTE_SCAN).size() == 2);
  require_parent_links(plan.get(), /*expected_parent=*/nullptr);
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
    CHECK(partition->Cast<sirius::op::sirius_physical_partition>().exchange() != nullptr);

    REQUIRE(partition->children.size() == 1);
    CHECK(partition->children[0]->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
  }

  SECTION("COUNT(DISTINCT) records LIST locally and BIGINT after merge")
  {
    auto plan = generate_sirius_plan(*con,
                                     "SELECT val, count(DISTINCT id) FROM big_left GROUP BY val",
                                     {OptimizerType::DISTINCT_AGGREGATE_REWRITE});
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

  SECTION("a wide integer SUM accepts INT64 or DECIMAL128 partials until the merge")
  {
    // The statistics cannot prove the BIGINT sum fits INT64, so the SUM is wide.
    REQUIRE_FALSE(con->Query("CREATE TABLE wide_sum (k INTEGER, x BIGINT)")->HasError());
    REQUIRE_FALSE(
      con->Query("INSERT INTO wide_sum VALUES (1, 4000000000000000000), (1, 4000000000000000000)")
        ->HasError());
    auto plan = generate_sirius_plan(*con, "SELECT k, sum(x) FROM wide_sum GROUP BY k");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    REQUIRE(merge != nullptr);
    CHECK(merge->get_alternate_physical_types().empty());
    auto* partition = merge->children[0].get();
    REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
    auto* local = partition->children[0].get();
    REQUIRE(local->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);

    auto const int64      = cudf::data_type{cudf::type_id::INT64};
    auto const decimal128 = cudf::data_type{cudf::type_id::DECIMAL128, 0};
    for (auto const* op : {local, partition}) {
      REQUIRE(op->get_types().size() == 2);
      CHECK(op->get_types()[1].id() == sirius::type_id::BIGINT);
      CHECK(op->is_alternate_physical_type(1, decimal128));
      CHECK_FALSE(op->is_alternate_physical_type(1, int64));
      CHECK_FALSE(op->is_alternate_physical_type(0, decimal128));
    }
  }

  SECTION("a decimal SUM has no alternate partial carrier")
  {
    auto plan =
      generate_sirius_plan(*con, "SELECT amount, sum(amount) FROM decimal_values GROUP BY amount");
    INFO(tree_to_string(plan.get()));

    auto* local = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_GROUP_BY);
    REQUIRE(local != nullptr);
    CHECK(local->get_alternate_physical_types().empty());
    auto* partition = find_first(plan.get(), SiriusPhysicalOperatorType::PARTITION);
    REQUIRE(partition != nullptr);
    CHECK(partition->get_alternate_physical_types().empty());
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
    // AVG(INTEGER) accumulates in 128 bits (a wide integer sum), so its local sum is
    // DECIMAL(38, 0) and its count BIGINT.
    auto plan = generate_sirius_plan(
      *con, "SELECT avg(val) FROM big_left", {OptimizerType::AGGREGATE_FUNCTION_REWRITER});
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_AGGREGATE);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->get_types().size() == 1);
    REQUIRE(merge->children.size() == 1);

    auto* local = merge->children[0].get();
    REQUIRE(local->type == SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
    duckdb::vector<sirius::logical_type> const expected_local_types{
      sirius::logical_type::make_decimal(38, 0),
      sirius::logical_type::make(sirius::type_id::BIGINT)};
    CHECK(local->get_types() == expected_local_types);
  }

  SECTION("AVG widens its DECIMAL local sum carrier to DECIMAL(38, scale) below MERGE_AGGREGATE")
  {
    auto plan = generate_sirius_plan(
      *con, "SELECT avg(amount) FROM decimal_values", {OptimizerType::AGGREGATE_FUNCTION_REWRITER});
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_AGGREGATE);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->get_types().size() == 1);
    REQUIRE(merge->children.size() == 1);

    auto* local = merge->children[0].get();
    REQUIRE(local->type == SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
    REQUIRE(local->get_types().size() == 2);
    CHECK(local->get_types()[0] == sirius::logical_type::make_decimal(38, 2));
    CHECK(local->get_types()[1].id() == sirius::type_id::BIGINT);
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - DISTINCT lowers to a zero-aggregate grouped aggregate",
                 "[plan_tree_shape][isolated_context]")
{
  // The chain every supported DISTINCT shape shares: one HASH_GROUP_BY with an empty aggregate
  // list, wrapped as insert_gpu_pipeline_operators wraps a GROUP BY. Returns the HASH_GROUP_BY so
  // each section can assert its own key layout.
  auto require_distinct_wrap_chain = [](sirius_physical_operator* plan) {
    auto* merge = find_first(plan, SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->children.size() == 1);

    auto* partition = merge->children[0].get();
    REQUIRE(partition->type == SiriusPhysicalOperatorType::PARTITION);
    CHECK(partition->Cast<sirius::op::sirius_physical_partition>().exchange() != nullptr);
    REQUIRE(partition->children.size() == 1);

    auto* hgb = partition->children[0].get();
    REQUIRE(hgb->type == SiriusPhysicalOperatorType::HASH_GROUP_BY);
    auto& aggregate = hgb->Cast<sirius::op::sirius_physical_grouped_aggregate>();
    CHECK(aggregate.aggregate_slots.empty());
    CHECK(aggregate.cudf_aggregates.empty());
    CHECK(aggregate.cudf_aggregate_idx.empty());
    CHECK_FALSE(aggregate.has_avg);
    CHECK_FALSE(aggregate.has_count_distinct);
    CHECK(aggregate.grouping_sets.empty());
    return &aggregate;
  };

  SECTION("plain DISTINCT keys the whole row in order and needs no projection")
  {
    auto plan = generate_sirius_plan(*con, "SELECT DISTINCT id, val FROM big_left");
    INFO(tree_to_string(plan.get()));

    auto* aggregate = require_distinct_wrap_chain(plan.get());
    CHECK(aggregate->group_idx == std::vector<int>{0, 1});
    CHECK(aggregate->get_output_grouping_indices() == std::vector<int>{0, 1});

    // Every distinct target is a bare reference at its own output index, so the builder emits no
    // reorder projection.
    CHECK(collect(plan.get(), SiriusPhysicalOperatorType::PROJECTION).empty());
  }

  SECTION("plain DISTINCT with ORDER BY still lowers to the GPU")
  {
    // LogicalDistinct::order_by is populated only for DISTINCT ON under an ORDER BY. This query
    // reaches the builder with order_by == nullptr and its ORDER BY becomes a separate
    // LOGICAL_ORDER above the distinct, so the DISTINCT ON + ORDER BY guard must not fire.
    auto plan = generate_sirius_plan(*con, "SELECT DISTINCT id, val FROM big_left ORDER BY id");
    INFO(tree_to_string(plan.get()));

    auto* aggregate = require_distinct_wrap_chain(plan.get());
    CHECK(aggregate->group_idx == std::vector<int>{0, 1});
    CHECK(find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_SORT) != nullptr);
  }

  SECTION("DISTINCT ON keys the targets in target order and reorders above the merge")
  {
    // The targets are (val, id) but the output columns are (id, val), so group position 0 reads
    // child column 1 and the builder's trailing push_projection restores the order.
    auto plan = generate_sirius_plan(*con, "SELECT DISTINCT ON (val, id) id, val FROM big_left");
    INFO(tree_to_string(plan.get()));

    auto* aggregate = require_distinct_wrap_chain(plan.get());
    CHECK(aggregate->group_idx == std::vector<int>{1, 0});

    auto projections = collect(plan.get(), SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(projections.size() == 1);
    REQUIRE(projections[0]->children.size() == 1);
    CHECK(projections[0]->children[0]->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);

    auto const& select_list =
      projections[0]->Cast<sirius::op::sirius_physical_projection>().select_list;
    REQUIRE(select_list.size() == 2);
    REQUIRE(select_list[0]->holds<sirius::ast::reference>());
    REQUIRE(select_list[1]->holds<sirius::ast::reference>());
    CHECK(select_list[0]->get<sirius::ast::reference>().column_index == 1);
    CHECK(select_list[1]->get<sirius::ast::reference>().column_index == 0);
  }

  SECTION(
    "DISTINCT ON with a key that is not an output column keeps the binder's prune "
    "projection")
  {
    // `val` is not in the select list, so the binder appends it to the projection under the
    // distinct and prunes it again above. Both columns are still distinct targets, so the builder
    // emits no projection of its own.
    auto plan = generate_sirius_plan(*con, "SELECT DISTINCT ON (id, val) id FROM big_left");
    INFO(tree_to_string(plan.get()));

    auto* aggregate = require_distinct_wrap_chain(plan.get());
    CHECK(aggregate->group_idx == std::vector<int>{0, 1});

    auto projections = collect(plan.get(), SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(projections.size() == 1);
    CHECK(projections[0]->get_types().size() == 1);
    REQUIRE(projections[0]->children.size() == 1);
    CHECK(projections[0]->children[0]->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
  }

  SECTION("DISTINCT over an aggregate planned narrower than declared takes the child's types")
  {
    // The DISTINCT node declares sum()'s HUGEINT; the aggregate below is planned as BIGINT.
    auto plan =
      generate_sirius_plan(*con, "SELECT DISTINCT val, sum(id) FROM big_left GROUP BY val");
    INFO(tree_to_string(plan.get()));

    duckdb::vector<sirius::logical_type> const planned{
      sirius::logical_type::make(sirius::type_id::INTEGER),
      sirius::logical_type::make(sirius::type_id::BIGINT)};
    auto* aggregate = require_distinct_wrap_chain(plan.get());
    CHECK(aggregate->get_types() == planned);
    CHECK(aggregate->group_idx == std::vector<int>{0, 1});
    CHECK(find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_GROUP_BY)->get_types() ==
          planned);
  }

  SECTION("DISTINCT ON reordering over a narrowed aggregate types the projection from the child")
  {
    auto plan = generate_sirius_plan(
      *con,
      "SELECT DISTINCT ON (s, k) k, s FROM (SELECT val AS k, sum(id) AS s FROM big_left GROUP BY "
      "val)");
    INFO(tree_to_string(plan.get()));

    auto* aggregate = require_distinct_wrap_chain(plan.get());
    CHECK(aggregate->group_idx == std::vector<int>{1, 0});

    REQUIRE(plan->type == SiriusPhysicalOperatorType::PROJECTION);
    REQUIRE(plan->children.size() == 1);
    CHECK(plan->children[0]->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY);
    CHECK(plan->get_types() == duckdb::vector<sirius::logical_type>{
                                 sirius::logical_type::make(sirius::type_id::INTEGER),
                                 sirius::logical_type::make(sirius::type_id::BIGINT)});
  }

  SECTION("DISTINCT over nested materialized CTEs checks the body's schema")
  {
    // `d` is read twice and joined on `other`, so its definition keeps both columns and is wider
    // than the one-column body the DISTINCT reads.
    auto plan = generate_sirius_plan(
      *con,
      "SELECT DISTINCT * FROM (WITH c AS MATERIALIZED (SELECT id FROM big_left), "
      "d AS MATERIALIZED (SELECT rid, other FROM small_right) "
      "SELECT d1.rid FROM d d1 JOIN d d2 ON d1.other = d2.other JOIN c ON c.id = d1.rid) s");
    INFO(tree_to_string(plan.get()));

    auto* aggregate = require_distinct_wrap_chain(plan.get());
    REQUIRE(aggregate->children.size() == 1);
    auto* outer = aggregate->children[0].get();
    REQUIRE(outer->type == SiriusPhysicalOperatorType::CTE);
    auto* inner = outer->children[1].get();
    REQUIRE(inner->type == SiriusPhysicalOperatorType::CTE);
    CHECK(inner->children[0]->get_types().size() == 2);
    CHECK(inner->get_types().size() == 1);
    CHECK(outer->get_types().size() == 1);
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - unsupported DISTINCT shapes are rejected at plan time",
                 "[plan_tree_shape][isolated_context]")
{
  // The builder's own guards throw duckdb::NotImplementedException specifically: SiriusContext
  // reports that type as an unsupported shape and logs the message, where a plain std::exception
  // would still fall back but be reported as a plan failure.
  auto require_rejected = [&](std::string const& query, std::string const& message_fragment) {
    INFO(query);
    try {
      generate_sirius_plan(*con, query);
    } catch (NotImplementedException const& e) {
      REQUIRE_THAT(std::string(e.what()), Catch::Matchers::ContainsSubstring(message_fragment));
      return;
    } catch (std::exception const& e) {
      FAIL("expected a NotImplementedException containing: " << message_fragment
                                                             << ", got: " << e.what());
    }
    FAIL("expected a NotImplementedException containing: " << message_fragment);
  };

  // The shared reject_nested_column_operation() throws std::runtime_error, so this one asserts the
  // message only.
  auto require_rejected_any = [&](std::string const& query, std::string const& message_fragment) {
    INFO(query);
    try {
      generate_sirius_plan(*con, query);
    } catch (std::exception const& e) {
      REQUIRE_THAT(std::string(e.what()), Catch::Matchers::ContainsSubstring(message_fragment));
      return;
    }
    FAIL("expected an exception containing: " << message_fragment);
  };

  SECTION("DISTINCT ON with carried columns needs a grouped FIRST")
  {
    require_rejected("SELECT DISTINCT ON (id) id, val FROM big_left",
                     "DISTINCT ON with carried (non-key) columns");
  }

  SECTION("DISTINCT ON under an ORDER BY names a specific row per group")
  {
    require_rejected("SELECT DISTINCT ON (id) id, val FROM big_left ORDER BY val",
                     "DISTINCT ON with ORDER BY");
  }

  SECTION("plain DISTINCT ordered by a column outside the select list has an uncovered output")
  {
    // `val` is added to the select list after the targets were synthesized from it, so the node
    // is two columns wide with one target. Output column 1 has no target: a different cause, and
    // a different message, from a target that is not a column reference.
    require_rejected("SELECT DISTINCT id FROM big_left ORDER BY val",
                     "output column 1 has no distinct target");
  }

  SECTION("a collated distinct target is not a plain reference to any output column")
  {
    // Binder::BindModifiers pushes the default collation over every distinct target, so a VARCHAR
    // key under a non-binary default_collation arrives as a call rather than as a bare reference:
    // the only route ordinary SQL has into the not-a-bare-reference arm. The integration suite's
    // collation case runs the same shape but can only watch the fallback counter, which both
    // uncovered-output arms move, so this is where that arm's message is pinned.
    //
    // default_collation is GLOBAL_DEFAULT-scoped, but Catch2 rebuilds the fixture -- and with it
    // the DuckDB instance -- for every leaf section, so the setting dies with this section.
    auto collate = con->Query("SET default_collation = 'nocase'");
    REQUIRE(collate);
    REQUIRE_FALSE(collate->HasError());

    require_rejected("SELECT DISTINCT pname FROM parts",
                     "no distinct target is a plain reference to output column 0");
  }

  SECTION("the not-a-bare-reference message names the offending target")
  {
    // Target 0 reads child column 1 and target 1 is the collated call, so the uncovered output
    // column 0 shares an index with a target that is a plain reference.
    auto collate = con->Query("SET default_collation = 'nocase'");
    REQUIRE(collate);
    REQUIRE_FALSE(collate->HasError());

    require_rejected("SELECT DISTINCT ON (pk, pname) 1 AS c, pk, pname FROM parts",
                     "output column 0 (falling back to CPU); target 1 is");
  }

  SECTION("a nested distinct key is rejected before the child is planned")
  {
    auto create = con->Query("CREATE TABLE nested_keys (id INTEGER, s STRUCT(x INTEGER))");
    REQUIRE(create);
    REQUIRE_FALSE(create->HasError());

    // The check runs before create_plan(*op.children[0]), so the message names the column rather
    // than the scan failing first.
    require_rejected_any("SELECT DISTINCT * FROM nested_keys", "is unsupported in DISTINCT");
  }

  SECTION("an expression DISTINCT ON target still leaves its own columns carried")
  {
    // `id + val` is appended to the select list, so the target arrives as a bare reference to that
    // appended column. Unsupported for the same reason as above: `id` and `val` are output columns
    // no target covers.
    require_rejected("SELECT DISTINCT ON (id + val) id, val FROM big_left",
                     "DISTINCT ON with carried (non-key) columns");
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
                 "plan tree shape - join input partitions get their ingress barrier at planning",
                 "[plan_tree_shape][isolated_context]")
{
  using sirius::op::MemoryBarrierType;
  auto const barriers = [](sirius_physical_operator* join) {
    auto& probe = join->children[0]->children[0]->Cast<sirius::op::sirius_physical_partition>();
    auto& build = join->children[1]->children[0]->Cast<sirius::op::sirius_physical_partition>();
    return std::make_pair(probe.input_barrier(), build.input_barrier());
  };
  const std::vector<OptimizerType> keep_shape{OptimizerType::JOIN_ORDER,
                                              OptimizerType::BUILD_SIDE_PROBE_SIDE};

  SECTION("INNER join: the probe streams and the build arrives whole")
  {
    auto plan =
      generate_sirius_plan(*con, "SELECT * FROM big_left l JOIN small_right r ON l.id = r.rid");
    INFO(tree_to_string(plan.get()));
    auto* join = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(join != nullptr);
    REQUIRE(join->Cast<sirius::op::sirius_physical_hash_join>().join_type == JoinType::INNER);
    auto const [probe, build] = barriers(join);
    CHECK(probe == MemoryBarrierType::PARTIAL);
    CHECK(build == MemoryBarrierType::FULL);
  }

  SECTION("RIGHT join: the probe drives partition sizing and arrives whole")
  {
    auto plan = generate_sirius_plan(
      *con, "SELECT * FROM big_left l RIGHT JOIN small_right r ON l.id = r.rid", keep_shape);
    INFO(tree_to_string(plan.get()));
    auto* join = find_first(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
    REQUIRE(join != nullptr);
    REQUIRE(join->Cast<sirius::op::sirius_physical_hash_join>().is_right_family());
    auto const [probe, build] = barriers(join);
    CHECK(probe == MemoryBarrierType::FULL);
    CHECK(build == MemoryBarrierType::FULL);
  }

  SECTION("RIGHT_DELIM_JOIN: the inner join is sized from its build side")
  {
    auto plan = generate_sirius_plan(
      *con,
      "SELECT SUM(i.qty) FROM items i, parts p WHERE p.pk = i.fk AND p.pname = 'p1' "
      "AND i.qty < (SELECT 2 * AVG(i2.qty) FROM items i2 WHERE i2.fk = p.pk)");
    INFO(tree_to_string(plan.get()));
    auto* node = find_first(plan.get(), SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
    REQUIRE(node != nullptr);
    auto& delim = node->Cast<sirius::op::sirius_physical_right_delim_join>();
    REQUIRE(delim.join);
    auto const [probe, build] = barriers(delim.join.get());
    CHECK(probe == MemoryBarrierType::PARTIAL);
    CHECK(build == MemoryBarrierType::FULL);
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
                 "plan tree shape - set_parent_ops links every operator",
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

TEST_CASE("set_parent_ops accepts a GPU scan without an ingestible",
          "[plan_tree_shape][set_parent_ops]")
{
  duckdb::vector<sirius::logical_type> types;
  sirius::op::scan::sirius_gpu_scan_operator scan(
    std::move(types), /*estimated_cardinality=*/0, /*ingestible=*/nullptr, /*contract_id=*/1);

  CHECK_NOTHROW(
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(scan, /*parent=*/nullptr));
  CHECK(scan.get_parent_op() == nullptr);
}

TEST_CASE("planned_aggregate_type narrows HUGEINT only", "[plan_tree_shape]")
{
  using generator = sirius::planner::sirius_physical_plan_generator;
  CHECK(generator::planned_aggregate_type(LogicalType::HUGEINT) == LogicalType::BIGINT);
  CHECK(generator::planned_aggregate_type(LogicalType::INTEGER) == LogicalType::INTEGER);
  CHECK(generator::planned_aggregate_type(LogicalType::UHUGEINT) == LogicalType::UHUGEINT);
  CHECK(generator::planned_aggregate_type(LogicalType::DECIMAL(38, 0)) ==
        LogicalType::DECIMAL(38, 0));
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
  duckdb::LogicalDummyScan stub(duckdb::TableIndex(0));
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

  duckdb::LogicalDummyScan stub(duckdb::TableIndex(0));
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
                 "plan tree shape - nested-loop-join wrap folds the side rows are decided against",
                 "[plan_tree_shape]")
{
  struct expectation {
    duckdb::JoinType join_type;
    bool fold_build;
    bool fold_probe;
  };
  for (auto const& e : {expectation{duckdb::JoinType::INNER, false, false},
                        expectation{duckdb::JoinType::LEFT, true, false},
                        expectation{duckdb::JoinType::SEMI, true, false},
                        expectation{duckdb::JoinType::ANTI, true, false},
                        expectation{duckdb::JoinType::MARK, true, false},
                        expectation{duckdb::JoinType::RIGHT, false, true},
                        expectation{duckdb::JoinType::OUTER, true, true}}) {
    DYNAMIC_SECTION(duckdb::JoinTypeToString(e.join_type))
    {
      sirius::planner::sirius_physical_plan_generator gen(*con->context);

      duckdb::LogicalDummyScan stub(duckdb::TableIndex(0));
      stub.types = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};
      duckdb::unique_ptr<sirius_physical_operator> plan =
        duckdb::make_uniq<sirius::op::sirius_physical_nested_loop_join>(
          stub,
          make_projection_leaf(1),
          make_projection_leaf(1),
          duckdb::vector<sirius::join_condition>{},
          e.join_type,
          /*estimated_cardinality=*/1);

      gen.insert_gpu_pipeline_operators(plan);

      REQUIRE(plan->type == SiriusPhysicalOperatorType::NESTED_LOOP_JOIN);
      require_join_child_wrap(plan->children[0].get(), plan.get(), /*is_build=*/false);
      require_join_child_wrap(plan->children[1].get(), plan.get(), /*is_build=*/true);
      CHECK(plan->children[0]->Cast<sirius::op::sirius_physical_concat>().concat_all() ==
            e.fold_probe);
      CHECK(plan->children[1]->Cast<sirius::op::sirius_physical_concat>().concat_all() ==
            e.fold_build);
    }
  }
}

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan generation rejects an untranslatable pushed-down table filter",
                 "[plan_tree_shape][table_filter][isolated_context]")
{
  auto begin = con->Query("BEGIN TRANSACTION");
  REQUIRE(begin);
  REQUIRE_FALSE(begin->HasError());
  auto logical = con->ExtractPlan("SELECT id FROM big_left");
  REQUIRE(logical);
  auto* scan = logical.get();
  while (scan->type != duckdb::LogicalOperatorType::LOGICAL_GET) {
    REQUIRE(scan->children.size() == 1);
    scan = scan->children.front().get();
  }
  auto& get = scan->Cast<duckdb::LogicalGet>();
  REQUIRE(get.function.GetName() == "seq_scan");
  get.table_filters.SetFilterByColumnIndex(
    duckdb::ProjectionIndex(0),
    duckdb::make_uniq<duckdb::ExpressionFilter>(untranslatable_table_filter_expression()));

  sirius::planner::sirius_physical_plan_generator generator(*con->context);
  CHECK_THROWS_WITH(
    generator.create_plan(std::move(logical)),
    Catch::Matchers::ContainsSubstring("Unsupported filter predicate on column 'id'"));
  auto rollback = con->Query("ROLLBACK");
  REQUIRE(rollback);
  REQUIRE_FALSE(rollback->HasError());
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

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - UNION arm sinks declare the union's schema, not the arm's",
                 "[plan_tree_shape][union_all][isolated_context]")
{
  // CTE results and UNION arm sinks use the consumer schema, even when materialization
  // stores a different number of columns.
  auto require_sinks_match_union = [&](const std::string& query) {
    auto plan = generate_sirius_plan(*con, query);
    INFO(tree_to_string(plan.get()));

    auto* union_op = find_first(plan.get(), SiriusPhysicalOperatorType::UNION);
    REQUIRE(union_op != nullptr);
    REQUIRE(union_op->children.size() == 2);

    auto* cte = find_first(plan.get(), SiriusPhysicalOperatorType::CTE);
    REQUIRE(cte != nullptr);
    REQUIRE(cte->children.size() == 2);
    REQUIRE(cte->children[0]->types.size() != union_op->types.size());
    CHECK(cte->types == cte->children[1]->types);
    CHECK(cte->types == union_op->types);
    CHECK_FALSE(cte->declared_output_schema_is_runtime_schema());

    for (auto& child : union_op->children) {
      REQUIRE(child->type == SiriusPhysicalOperatorType::PASSTHROUGH_SINK);
      CHECK(child->types == union_op->types);
    }
  };

  // Definition wider than the arity: the self-join on `other` keeps it from being pruned to `rid`.
  require_sinks_match_union(
    "SELECT id FROM big_left UNION ALL "
    "(WITH m AS MATERIALIZED (SELECT rid, other FROM small_right) "
    " SELECT m1.rid FROM m m1 JOIN m m2 ON m1.other = m2.other)");

  // Definition narrower: the body widens with a computed column the materialization never held.
  require_sinks_match_union(
    "SELECT id, id * 2 FROM big_left UNION ALL "
    "(WITH m AS MATERIALIZED (SELECT rid FROM small_right) SELECT rid, rid * 2 FROM m)");

  // The CTE as arm 0, so `op.types` is routed through the CTE's own body resolution.
  require_sinks_match_union(
    "(WITH m AS MATERIALIZED (SELECT rid FROM small_right) SELECT rid, rid * 2 FROM m) "
    "UNION ALL SELECT id, id * 2 FROM big_left");
}

namespace {

bool delim_join_as_cte(Connection& con)
{
  auto result = con.Query("SELECT current_setting('delim_join_as_cte')");
  REQUIRE_FALSE(result->HasError());
  return result->Collection().GetValue(0, 0).GetValue<bool>();
}

std::size_t count_delim_joins(sirius_physical_operator* plan)
{
  return collect(plan, SiriusPhysicalOperatorType::LEFT_DELIM_JOIN).size() +
         collect(plan, SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN).size();
}

}  // namespace

TEST_CASE_METHOD(plan_tree_shape_fixture,
                 "plan tree shape - Sirius defaults decorrelation to explicit delim joins",
                 "[plan_tree_shape][isolated_context][delim_join]")
{
  // The fixture pins the session value; drop it so the database default Sirius installs applies.
  REQUIRE_FALSE(con->Query("RESET delim_join_as_cte")->HasError());
  REQUIRE_FALSE(delim_join_as_cte(*con));
  // A connection opened after LOAD sees the same default without any SET.
  Connection fresh(*db);
  REQUIRE_FALSE(delim_join_as_cte(fresh));

  std::string const correlated_aggregate =
    "SELECT SUM(i.qty) FROM items i, parts p WHERE p.pk = i.fk AND p.pname = 'p1' "
    "AND i.qty < (SELECT 2 * AVG(i2.qty) FROM items i2 WHERE i2.fk = p.pk)";
  // The TPC-H Q21 pattern: correlated EXISTS and NOT EXISTS with an inequality on a second key.
  std::string const correlated_exists =
    "SELECT COUNT(*) FROM items i1, parts p WHERE p.pk = i1.fk AND p.pname = 'p1' "
    "AND EXISTS (SELECT 1 FROM items i2 WHERE i2.fk = i1.fk AND i2.qty <> i1.qty) "
    "AND NOT EXISTS (SELECT 1 FROM items i3 WHERE i3.fk = i1.fk AND i3.qty <> i1.qty "
    "AND i3.qty > 20)";

  SECTION("default: delim joins and no CTE")
  {
    for (auto const& query : {correlated_aggregate, correlated_exists}) {
      auto plan = generate_sirius_plan(*con, query);
      INFO(query << "\n" << tree_to_string(plan.get()));
      CHECK(count_delim_joins(plan.get()) > 0);
      CHECK(find_first(plan.get(), SiriusPhysicalOperatorType::CTE) == nullptr);
    }
  }

  SECTION("an explicit session SET restores DuckDB's CTE rewrite")
  {
    sirius::test::scoped_sirius_setting native_cte(*con, "delim_join_as_cte", true);
    REQUIRE(delim_join_as_cte(*con));
    // The other session keeps the database default.
    REQUIRE_FALSE(delim_join_as_cte(fresh));
    for (auto const& query : {correlated_aggregate, correlated_exists}) {
      auto plan = generate_sirius_plan(*con, query);
      INFO(query << "\n" << tree_to_string(plan.get()));
      CHECK(count_delim_joins(plan.get()) == 0);
    }
    auto plan = generate_sirius_plan(*con, correlated_aggregate);
    CHECK(find_first(plan.get(), SiriusPhysicalOperatorType::CTE) != nullptr);
  }
}

TEST_CASE("plan tree shape - an explicit global delim_join_as_cte survives loading Sirius",
          "[plan_tree_shape][isolated_context][delim_join]")
{
  scoped_temp_db_path db_path;
  auto const cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
                   "minimal.yaml";
  sirius::util::env_guard const config("SIRIUS_CONFIG_FILE", cfg.string());
  sirius::util::env_guard const enabled("SIRIUS_DISABLE", std::nullopt);
  DBConfig db_config;
  db_config.SetOptionByName("allow_unsigned_extensions", Value::BOOLEAN(true));
  db_config.SetOptionByName("delim_join_as_cte", Value::BOOLEAN(true));
  DuckDB db(db_path.path().c_str(), &db_config);
  Connection con(db);
  CHECK(delim_join_as_cte(con));
}
