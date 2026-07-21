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

#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_partition.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

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
/// test instead of silently skipping it.
duckdb::unique_ptr<sirius_physical_operator> generate_sirius_plan(Connection& con,
                                                                  const std::string& query)
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  // Keep STATISTICS_PROPAGATION disabled only for this shape-sensitive suite:
  // disabling it lets the deliminator retain the DELIM_JOINs asserted below.
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  disabled.insert(OptimizerType::STATISTICS_PROPAGATION);

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
  }
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

  SECTION("ungrouped aggregate gains MERGE_AGGREGATE with no PARTITION")
  {
    auto plan = generate_sirius_plan(*con, "SELECT sum(val) FROM big_left");
    INFO(tree_to_string(plan.get()));

    auto* merge = find_first(plan.get(), SiriusPhysicalOperatorType::MERGE_AGGREGATE);
    REQUIRE(merge != nullptr);
    REQUIRE(merge->children.size() == 1);
    CHECK(merge->children[0]->type == SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE);
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
