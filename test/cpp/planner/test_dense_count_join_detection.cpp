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

#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "utils/pipeline_conversion_test_utils.hpp"
#include "utils/scoped_sirius_setting.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/function/aggregate/distributive_functions.hpp>
#include <duckdb/function/aggregate_function.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_aggregate.hpp>
#include <duckdb/planner/operator/logical_projection.hpp>
#include <duckdb/planner/planner.hpp>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

using namespace duckdb;

namespace {

// The GPU seq_scan ingestible requires a single-file block manager, so these tests use an
// on-disk database.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_dense_count_join_XXXXXX";
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

class scoped_temp_directory {
 public:
  scoped_temp_directory()
  {
    char tmpl[] = "/tmp/sirius_dense_count_join_dso_XXXXXX";
    auto* path  = ::mkdtemp(tmpl);
    REQUIRE(path != nullptr);
    _path = path;
  }

  ~scoped_temp_directory()
  {
    std::error_code error;
    std::filesystem::remove_all(_path, error);
  }

  scoped_temp_directory(const scoped_temp_directory&)            = delete;
  scoped_temp_directory& operator=(const scoped_temp_directory&) = delete;

  const std::filesystem::path& path() const { return _path; }

 private:
  std::filesystem::path _path;
};

using logical_plan_mutator = bool (*)(duckdb::LogicalOperator&);

struct plan_generation_options {
  logical_plan_mutator mutate = nullptr;
};

void spoof_count_update(
  duckdb::Vector[], duckdb::AggregateInputData&, duckdb::idx_t, duckdb::Vector&, duckdb::idx_t)
{
}

duckdb::BoundAggregateExpression* find_first_count(duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
    auto& aggregate = op.Cast<duckdb::LogicalAggregate>();
    for (auto& expression : aggregate.expressions) {
      if (expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_AGGREGATE) {
        continue;
      }
      auto& bound = expression->Cast<duckdb::BoundAggregateExpression>();
      if (bound.function.name == "count" || bound.function.name == "count_star") { return &bound; }
    }
  }
  for (auto& child : op.children) {
    if (auto* bound = find_first_count(*child)) { return bound; }
  }
  return nullptr;
}

bool spoof_first_count_callback(duckdb::LogicalOperator& op)
{
  auto* bound = find_first_count(op);
  if (bound == nullptr) { return false; }
  bound->function.update = spoof_count_update;
  return true;
}

bool replace_first_count_with_internal_count_star(duckdb::LogicalOperator& op)
{
  auto* bound = find_first_count(op);
  if (bound == nullptr) { return false; }
  bound->function      = duckdb::CountStarFun::GetFunction();
  bound->function.name = "count_star";
  bound->function.catalog_name.clear();
  bound->function.schema_name.clear();
  bound->children.clear();
  bound->bind_info.reset();
  return true;
}

bool assign_non_system_count_provenance(duckdb::LogicalOperator& op)
{
  auto* bound = find_first_count(op);
  if (bound == nullptr) { return false; }
  bound->function.catalog_name = "user_catalog";
  bound->function.schema_name  = "main";
  return true;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> generate_sirius_plan(
  Connection& con, const std::string& query, plan_generation_options options = {})
{
  auto& context = *con.context;

  auto original_disabled = DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(OptimizerType::IN_CLAUSE);
  disabled.insert(OptimizerType::COMPRESSED_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
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
    if (options.mutate) { REQUIRE(options.mutate(*plan)); }

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

std::vector<sirius::op::sirius_physical_operator*> collect(
  sirius::op::sirius_physical_operator* root, sirius::op::SiriusPhysicalOperatorType type)
{
  std::vector<sirius::op::sirius_physical_operator*> out;
  if (!root) { return out; }
  if (root->type == type) { out.push_back(root); }
  for (auto& child : root->children) {
    auto sub = collect(child.get(), type);
    out.insert(out.end(), sub.begin(), sub.end());
  }
  return out;
}

// Join inputs whose physical root owns a producer pipeline of its own: a nested hash join, a
// grouped aggregate under a HAVING filter, and a nested dense count-join.
constexpr char const* nested_hash_join_counted_query =
  "SELECT c_id, count(o.o_id) FROM cust LEFT JOIN ("
  "  SELECT o1.o_id, o1.o_cust FROM ord o1 JOIN ord o2 ON o1.o_cust = o2.o_cust"
  ") o ON c_id = o.o_cust GROUP BY c_id";
constexpr char const* nested_hash_join_preserved_query =
  "SELECT c.c_id, count(o_id) FROM ("
  "  SELECT c1.c_id FROM cust c1 JOIN cust c2 ON c1.c_grp = c2.c_grp"
  ") c LEFT JOIN ord ON c.c_id = o_cust GROUP BY c.c_id";
constexpr char const* nested_hash_join_right_preserved_query =
  "SELECT c.c_id, count(o_id) FROM ord RIGHT JOIN ("
  "  SELECT c1.c_id FROM cust c1 JOIN cust c2 ON c1.c_grp = c2.c_grp"
  ") c ON o_cust = c.c_id GROUP BY c.c_id";
constexpr char const* having_counted_query =
  "SELECT c_id, count(t.o_cust) FROM cust LEFT JOIN ("
  "  SELECT o_cust FROM ord GROUP BY o_cust HAVING count(*) > 1"
  ") t ON c_id = t.o_cust GROUP BY c_id";
constexpr char const* nested_dense_count_join_query =
  "SELECT c_id, count(t.n) FROM cust LEFT JOIN ("
  "  SELECT o_cust AS k, count(c2.c_id) AS n FROM ord LEFT JOIN cust c2 ON o_cust = c2.c_id"
  "  GROUP BY o_cust"
  ") t ON c_id = t.k GROUP BY c_id";

// The fused operator's inputs are hash-partitioned, so its direct child is a PARTITION and the
// producer that terminates its own pipeline sits one level below that.
sirius::op::sirius_physical_operator const* dense_input_root(
  sirius::op::sirius_physical_operator const* dense, std::size_t child_index)
{
  auto const* child = dense->children[child_index].get();
  REQUIRE(child->type == sirius::op::SiriusPhysicalOperatorType::PARTITION);
  REQUIRE(child->children.size() == 1);
  return child->children[0].get();
}

sirius::pipeline::sirius_pipeline const* pipeline_containing(
  sirius::pipeline::pipeline_conversion_result const& result,
  sirius::op::SiriusPhysicalOperatorType type)
{
  for (auto const& pipeline : result.scheduled_pipelines) {
    for (auto const& op_ref : pipeline->get_operators()) {
      if (op_ref.get().type == type) { return pipeline.get(); }
    }
  }
  return nullptr;
}

// Require the single-operator DENSE_COUNT_JOIN pipeline with exactly two FULL input wirings, each
// sourced at the direct child that terminates the producer pipeline; returns the fused operator.
sirius::op::sirius_physical_operator const* require_dense_count_join_conversion(
  sirius::pipeline::pipeline_conversion_result const& result)
{
  using sirius::op::MemoryBarrierType;
  using sirius::op::sirius_physical_dense_count_join;
  using sirius::op::SiriusPhysicalOperatorType;

  auto const* dense_pipeline =
    pipeline_containing(result, SiriusPhysicalOperatorType::DENSE_COUNT_JOIN);
  REQUIRE(dense_pipeline != nullptr);
  auto const operators = dense_pipeline->get_operators();
  REQUIRE(operators.size() == 1);
  auto const* dense = &operators[0].get();
  CHECK(dense_pipeline->get_source().get() == dense);
  CHECK(dense_pipeline->get_sink().get() == dense);
  REQUIRE(dense->children.size() == 2);

  std::size_t input_count = 0;
  for (auto const& wiring : result.repository_wirings) {
    if (wiring.dest_pipeline.get() != dense_pipeline) { continue; }
    ++input_count;
    std::size_t const child_index =
      wiring.port_id == sirius_physical_dense_count_join::PRESERVED_PORT ? 0 : 1;
    CHECK((wiring.port_id == sirius_physical_dense_count_join::PRESERVED_PORT ||
           wiring.port_id == sirius_physical_dense_count_join::COUNTED_PORT));
    CHECK(wiring.barrier_type == MemoryBarrierType::FULL);
    CHECK(wiring.source_op == dense->children[child_index].get());
    REQUIRE(wiring.source_pipeline);
    CHECK(wiring.source_pipeline->get_sink().get() == dense->children[child_index].get());
  }
  REQUIRE(input_count == 2);
  return dense;
}

const sirius::op::scan::duckdb_native_ingestible_table_info& require_native_scan(
  sirius::op::sirius_physical_operator* root, std::string_view table_name, bool has_row_filter)
{
  using T    = sirius::op::SiriusPhysicalOperatorType;
  auto scans = collect(root, T::GPU_SCAN);
  REQUIRE(scans.size() == 1);

  auto const& scan = scans[0]->Cast<sirius::op::scan::sirius_gpu_scan_operator>();
  auto const* info = dynamic_cast<sirius::op::scan::duckdb_native_ingestible_table_info const*>(
    &scan.get_ingestible().table_info());
  REQUIRE(info != nullptr);
  CHECK(info->table_name == table_name);
  CHECK(scan.get_ingestible().has_row_filter() == has_row_filter);
  return *info;
}

void require_q13_counted_filter(sirius::op::sirius_physical_operator* preserved,
                                sirius::op::sirius_physical_operator* counted)
{
  auto const& preserved_scan = require_native_scan(preserved, "cust", false);
  CHECK((preserved_scan.table_filters == nullptr || preserved_scan.table_filters->filters.empty()));

  auto const& counted_scan = require_native_scan(counted, "ord", true);
  REQUIRE(counted_scan.table_filters != nullptr);
  REQUIRE(counted_scan.table_filters->filters.size() == 1);
  auto const& [column_index, filter] = *counted_scan.table_filters->filters.begin();
  REQUIRE(filter != nullptr);
  REQUIRE(column_index < counted_scan.column_ids.size());
  REQUIRE(counted_scan.column_ids[column_index].HasPrimaryIndex());
  CHECK(counted_scan.column_ids[column_index].GetPrimaryIndex() == 2);
}

struct dense_count_join_fixture {
  dense_count_join_fixture()
  {
    auto cfg = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" / "data" /
               "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);
    unsetenv("SIRIUS_DISABLE");
    db = std::make_unique<DuckDB>(db_path.path());
    setenv("SIRIUS_DISABLE", "1", 1);
    con          = std::make_unique<Connection>(*db);
    auto enabled = con->Query("SET enable_dense_count_join = true");
    REQUIRE(enabled != nullptr);
    REQUIRE_FALSE(enabled->HasError());

    con->Query("CREATE TABLE cust (c_id INTEGER, c_grp INTEGER)");
    con->Query("INSERT INTO cust SELECT range, range % 3 FROM range(20)");
    // Keep c_grp nullable so DuckDB does not rewrite count(c_grp) to count_star().
    con->Query("INSERT INTO cust VALUES (100, NULL)");
    con->Query("CREATE TABLE ord (o_id BIGINT, o_cust INTEGER, o_note VARCHAR)");
    con->Query(
      "INSERT INTO ord SELECT range, (range * 7) % 30, concat('n', range) FROM range(200)");
  }

  ~dense_count_join_fixture() { unsetenv("SIRIUS_CONFIG_FILE"); }

  bool has_dense_count_join(const std::string& query, plan_generation_options options = {})
  {
    auto plan = generate_sirius_plan(*con, query, options);
    REQUIRE(plan);
    using T          = sirius::op::SiriusPhysicalOperatorType;
    auto const fused = collect(plan.get(), T::DENSE_COUNT_JOIN);
    if (fused.empty()) { return false; }
    REQUIRE(fused.size() == 1);
    REQUIRE(collect(plan.get(), T::HASH_JOIN).empty());
    REQUIRE(collect(plan.get(), T::NESTED_LOOP_JOIN).empty());
    REQUIRE(fused[0]->children.size() == 2);
    return true;
  }

  struct fused_plan {
    duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan;
    sirius::op::sirius_physical_operator* fused;
  };

  // Plan `query` and require exactly one DENSE_COUNT_JOIN with two children.
  fused_plan require_fused(std::string const& query)
  {
    auto plan = generate_sirius_plan(*con, query);
    REQUIRE(plan);
    auto const fused =
      collect(plan.get(), sirius::op::SiriusPhysicalOperatorType::DENSE_COUNT_JOIN);
    REQUIRE(fused.size() == 1);
    REQUIRE(fused[0]->children.size() == 2);
    return {std::move(plan), fused[0]};
  }

  scoped_temp_db_path db_path;
  std::unique_ptr<DuckDB> db;
  std::unique_ptr<Connection> con;
};

}  // namespace

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join fires on COUNT(col) grouped by the preserved LEFT-join key",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  REQUIRE(has_dense_count_join(query));
  auto plan = generate_sirius_plan(*con, query);
  REQUIRE(collect(plan.get(), sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY).empty());
}

// Wraps the join under the first COUNT aggregate in a column-only projection that reverses the
// join's output order, and rewrites the aggregate's references through it: the shape the
// StarRocks translator emits (AGGREGATION_NODE over a PROJECT emit over a HASH_JOIN), which the
// planner must look through.
bool interpose_reversing_projection(duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
      op.children[0]->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto& aggregate  = op.Cast<duckdb::LogicalAggregate>();
    auto& join       = *op.children[0];
    auto const width = join.types.size();
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    duckdb::vector<duckdb::LogicalType> types;
    for (duckdb::idx_t i = 0; i < width; ++i) {
      auto const source = width - 1 - i;
      select_list.push_back(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(join.types[source], source));
      types.push_back(join.types[source]);
    }
    auto projection   = duckdb::make_uniq<duckdb::LogicalProjection>(9000, std::move(select_list));
    projection->types = std::move(types);
    projection->children.push_back(std::move(op.children[0]));
    op.children[0] = std::move(projection);
    auto remap     = [width](duckdb::Expression& expression) {
      if (expression.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
        auto& ref = expression.Cast<duckdb::BoundReferenceExpression>();
        ref.index = width - 1 - ref.index;
      }
    };
    for (auto& group : aggregate.groups) {
      remap(*group);
    }
    for (auto& expression : aggregate.expressions) {
      for (auto& child : expression->Cast<duckdb::BoundAggregateExpression>().children) {
        remap(*child);
      }
    }
    return true;
  }
  for (auto& child : op.children) {
    if (interpose_reversing_projection(*child)) { return true; }
  }
  return false;
}

// A projection that computes a column between the aggregate and the join: the aggregate would
// count a derived value, so the fusion must decline.
bool interpose_computing_projection(duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY &&
      op.children[0]->type == duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
    auto& join = *op.children[0];
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> select_list;
    duckdb::vector<duckdb::LogicalType> types;
    for (duckdb::idx_t i = 0; i < join.types.size(); ++i) {
      duckdb::unique_ptr<duckdb::Expression> column =
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(join.types[i], i);
      if (i == 0) {
        // COALESCE(col, col): same type, same value, but no longer a bare reference.
        duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> args;
        args.push_back(std::move(column));
        args.push_back(duckdb::make_uniq<duckdb::BoundReferenceExpression>(join.types[i], i));
        column = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
          duckdb::ExpressionType::OPERATOR_COALESCE, join.types[i]);
        for (auto& arg : args) {
          column->Cast<duckdb::BoundOperatorExpression>().children.push_back(std::move(arg));
        }
      }
      types.push_back(join.types[i]);
      select_list.push_back(std::move(column));
    }
    auto projection   = duckdb::make_uniq<duckdb::LogicalProjection>(9001, std::move(select_list));
    projection->types = std::move(types);
    projection->children.push_back(std::move(op.children[0]));
    op.children[0] = std::move(projection);
    return true;
  }
  for (auto& child : op.children) {
    if (interpose_computing_projection(*child)) { return true; }
  }
  return false;
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join looks through column-only projections and maps the refs",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  REQUIRE(has_dense_count_join(query, {.mutate = interpose_reversing_projection}));
  auto plan = generate_sirius_plan(*con, query, {.mutate = interpose_reversing_projection});
  using T   = sirius::op::SiriusPhysicalOperatorType;
  REQUIRE(collect(plan.get(), T::HASH_GROUP_BY).empty());
  REQUIRE(collect(plan.get(), T::PROJECTION).empty());
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines a projection that computes a column",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  REQUIRE_FALSE(has_dense_count_join(query, {.mutate = interpose_computing_projection}));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join fires on COUNT(*) and on the RIGHT-join orientation",
                 "[dense_count_join][plan]")
{
  REQUIRE(has_dense_count_join(
    "SELECT c_id, count(*) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
  REQUIRE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM ord RIGHT JOIN cust ON o_cust = c_id GROUP BY c_id"));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join fires inside the filtered two-level q13 distribution shape",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_count, count(*) AS custdist FROM ("
    "  SELECT c_id, count(o_id) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cust "
    "    AND o_note NOT LIKE '%special%requests%' GROUP BY c_id"
    ") t GROUP BY c_count";
  REQUIRE(has_dense_count_join(query));

  auto plan  = generate_sirius_plan(*con, query);
  using T    = sirius::op::SiriusPhysicalOperatorType;
  auto fused = collect(plan.get(), T::DENSE_COUNT_JOIN);
  REQUIRE(fused.size() == 1);
  REQUIRE(fused[0]->children.size() == 2);
  CHECK(collect(fused[0]->children[0].get(), T::FILTER).empty());
  CHECK(collect(fused[0]->children[1].get(), T::FILTER).empty());
  require_q13_counted_filter(fused[0]->children[0].get(), fused[0]->children[1].get());
  REQUIRE(collect(plan.get(), T::HASH_GROUP_BY).size() == 1);
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join conversion preserves its filtered counted input and FULL "
                 "barriers",
                 "[dense_count_join][pipeline]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust "
    "  AND o_note NOT LIKE '%special%requests%' GROUP BY c_id";

  REQUIRE(has_dense_count_join(query));

  sirius::test::with_conversion_result(
    *con, query, [](sirius::pipeline::pipeline_conversion_result& result) {
      using sirius::op::MemoryBarrierType;
      using sirius::op::SiriusPhysicalOperatorType;
      using sirius::op::sirius_physical_dense_count_join;
      using sirius::pipeline::repository_wiring;
      using sirius::pipeline::sirius_pipeline;

      sirius_pipeline* dense_pipeline = nullptr;
      for (auto const& pipeline : result.scheduled_pipelines) {
        for (auto const& op_ref : pipeline->get_operators()) {
          if (op_ref.get().type != SiriusPhysicalOperatorType::DENSE_COUNT_JOIN) { continue; }
          REQUIRE(dense_pipeline == nullptr);
          dense_pipeline = pipeline.get();
        }
      }

      REQUIRE(dense_pipeline != nullptr);
      auto const operators = dense_pipeline->get_operators();
      REQUIRE(operators.size() == 1);
      auto const* dense = &operators[0].get();
      CHECK(dense_pipeline->get_source().get() == dense);
      CHECK(dense_pipeline->get_sink().get() == dense);
      REQUIRE(dense->children.size() == 2);
      CHECK(collect(dense->children[0].get(), SiriusPhysicalOperatorType::FILTER).empty());
      CHECK(collect(dense->children[1].get(), SiriusPhysicalOperatorType::FILTER).empty());
      require_q13_counted_filter(dense->children[0].get(), dense->children[1].get());

      std::vector<repository_wiring const*> inputs;
      for (auto const& wiring : result.repository_wirings) {
        if (wiring.dest_pipeline.get() == dense_pipeline) { inputs.push_back(&wiring); }
      }
      REQUIRE(inputs.size() == 2);

      auto require_direct_input = [&](std::size_t child_index, std::string_view port_id) {
        repository_wiring const* match = nullptr;
        for (auto const* wiring : inputs) {
          if (wiring->port_id != port_id) { continue; }
          REQUIRE(match == nullptr);
          match = wiring;
        }
        REQUIRE(match != nullptr);
        CHECK(match->barrier_type == MemoryBarrierType::FULL);
        CHECK(match->source_op == dense->children[child_index].get());
        REQUIRE(match->source_pipeline);
        CHECK(match->source_pipeline->get_sink().get() == dense->children[child_index].get());
      };

      require_direct_input(0, sirius_physical_dense_count_join::PRESERVED_PORT);
      require_direct_input(1, sirius_physical_dense_count_join::COUNTED_PORT);
    });
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join conversion hosts nested producers behind its input ports",
                 "[dense_count_join][pipeline]")
{
  using T = sirius::op::SiriusPhysicalOperatorType;

  SECTION("nested hash join on the counted side")
  {
    sirius::test::with_conversion_result(
      *con, nested_hash_join_counted_query, [](auto const& result) {
        auto const* dense         = require_dense_count_join_conversion(result);
        auto const* join_pipeline = pipeline_containing(result, T::HASH_JOIN);
        REQUIRE(join_pipeline != nullptr);
        CHECK(join_pipeline->get_sink().get() == dense_input_root(dense, 1));
        std::vector<std::string_view> ports;
        for (auto const& wiring : result.repository_wirings) {
          if (wiring.dest_pipeline.get() == join_pipeline) { ports.push_back(wiring.port_id); }
        }
        std::sort(ports.begin(), ports.end());
        CHECK(ports == std::vector<std::string_view>{"build", "default"});
      });
  }

  SECTION("merge fusion terminates at the streaming root feeding the counted port")
  {
    auto const check_merge_boundary = [&](bool fused) {
      sirius::test::with_conversion_result(*con, having_counted_query, [&](auto const& result) {
        auto const* dense          = require_dense_count_join_conversion(result);
        auto const* merge_pipeline = pipeline_containing(result, T::MERGE_GROUP_BY);
        REQUIRE(merge_pipeline != nullptr);
        REQUIRE(merge_pipeline->get_sink());
        if (fused) {
          CHECK(merge_pipeline->get_sink().get() == dense_input_root(dense, 1));
          CHECK(merge_pipeline->get_sink()->type != T::MERGE_GROUP_BY);
        } else {
          CHECK(merge_pipeline->get_sink()->type == T::MERGE_GROUP_BY);
        }
      });
    };
    check_merge_boundary(true);
    sirius::test::scoped_sirius_setting unfused{*con, "fuse_merge_pipelines", false};
    check_merge_boundary(false);
  }
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join feeds from a PARTITION on each side",
                 "[dense_count_join][pipeline][partition]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust "
    "GROUP BY c_id";

  REQUIRE(has_dense_count_join(query));

  sirius::test::with_conversion_result(
    *con, query, [](sirius::pipeline::pipeline_conversion_result& result) {
      using sirius::op::SiriusPhysicalOperatorType;
      using sirius::op::sirius_physical_partition;

      sirius::op::sirius_physical_operator* dense = nullptr;
      for (auto const& pipeline : result.scheduled_pipelines) {
        for (auto const& op_ref : pipeline->get_operators()) {
          if (op_ref.get().type != SiriusPhysicalOperatorType::DENSE_COUNT_JOIN) { continue; }
          REQUIRE(dense == nullptr);
          dense = &op_ref.get();
        }
      }
      REQUIRE(dense != nullptr);
      REQUIRE(dense->children.size() == 2);

      // Both inputs are partitioned, so a task can take one partition of each side rather than
      // both inputs whole. No CONCAT sits between: this operator builds no hash table.
      REQUIRE(dense->children[0]->type == SiriusPhysicalOperatorType::PARTITION);
      REQUIRE(dense->children[1]->type == SiriusPhysicalOperatorType::PARTITION);
      auto& preserved_partition = dense->children[0]->Cast<sirius_physical_partition>();
      auto& counted_partition   = dense->children[1]->Cast<sirius_physical_partition>();

      // The preserved side is the build side, which makes it the sizing driver.
      CHECK(preserved_partition.is_build_partition());
      CHECK_FALSE(counted_partition.is_build_partition());

      // Sizing reads both sides' bytes, so each partition must be able to reach the other and
      // must route its count decision through the fused operator.
      CHECK(preserved_partition.get_downstream_consumer_op() == dense);
      CHECK(counted_partition.get_downstream_consumer_op() == dense);
      CHECK(preserved_partition.get_sibling_partition_op() == &counted_partition);
      CHECK(counted_partition.get_sibling_partition_op() == &preserved_partition);
    });
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines off-shape aggregates and joins",
                 "[dense_count_join][plan]")
{
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM cust JOIN ord ON c_id = o_cust GROUP BY c_id"));
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_grp, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_grp"));
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id), max(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY "
    "c_id"));
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(DISTINCT o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, sum(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
  // A nullable preserved-side COUNT has different outer-join NULL semantics.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(c_grp) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id"));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join authenticates COUNT callbacks, not its public name",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  auto plan = generate_sirius_plan(
    *con, query, plan_generation_options{.mutate = spoof_first_count_callback});
  REQUIRE(plan);
  using T = sirius::op::SiriusPhysicalOperatorType;
  CHECK(collect(plan.get(), T::DENSE_COUNT_JOIN).empty());
  CHECK(collect(plan.get(), T::HASH_JOIN).size() == 1);
  CHECK(collect(plan.get(), T::HASH_GROUP_BY).size() == 1);
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join accepts optimizer-created COUNT_STAR without provenance",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  REQUIRE(has_dense_count_join(
    query, plan_generation_options{.mutate = replace_first_count_with_internal_count_star}));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join identifies COUNT by host callbacks, not declared provenance",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";
  plan_generation_options const options{.mutate = assign_non_system_count_provenance};
  REQUIRE(has_dense_count_join(query, options));
  auto plan = generate_sirius_plan(*con, query, options);
  REQUIRE(collect(plan.get(), sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY).empty());
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join defaults to the fused plan and supports an explicit opt-out",
                 "[dense_count_join][plan]")
{
  auto const query =
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY c_id";

  auto reset = con->Query("RESET enable_dense_count_join");
  REQUIRE_FALSE(reset->HasError());
  REQUIRE(has_dense_count_join(query));

  auto off = con->Query("SET enable_dense_count_join = false");
  REQUIRE_FALSE(off->HasError());
  auto disabled_plan = generate_sirius_plan(*con, query);
  REQUIRE(disabled_plan);
  using T = sirius::op::SiriusPhysicalOperatorType;
  CHECK(collect(disabled_plan.get(), T::DENSE_COUNT_JOIN).empty());
  CHECK(collect(disabled_plan.get(), T::HASH_JOIN).size() == 1);
  CHECK(collect(disabled_plan.get(), T::HASH_GROUP_BY).size() == 1);
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines filtered aggregates, extra conditions, and "
                 "non-plain keys",
                 "[dense_count_join][plan]")
{
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FILTER (WHERE o_id > 0) FROM cust LEFT JOIN ord ON c_id = o_cust "
    "GROUP BY c_id"));
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust AND c_grp = o_cust "
    "GROUP BY c_id"));
  // A preserved-side ON residual cannot move below LEFT JOIN without dropping retained rows.
  CHECK_THROWS_WITH(
    has_dense_count_join(
      "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust AND c_grp > 0 "
      "GROUP BY c_id"),
    Catch::Contains("Any join not supported"));
  // INTEGER = BIGINT inserts a CAST, so the plain-reference gate declines.
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_cust) FROM cust LEFT JOIN ord ON c_id = o_id GROUP BY c_id"));
  CHECK_FALSE(has_dense_count_join(
    "SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id IS NOT DISTINCT FROM o_cust "
    "GROUP BY c_id"));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines intervening operators and counted-side group keys",
                 "[dense_count_join][plan]")
{
  CHECK_FALSE(
    has_dense_count_join("SELECT c_id, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust "
                         "WHERE (o_id IS NULL OR c_grp = 0) GROUP BY c_id"));
  CHECK_FALSE(has_dense_count_join(
    "SELECT o_cust, count(o_id) FROM cust LEFT JOIN ord ON c_id = o_cust GROUP BY o_cust"));
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join fires over nested joins and aggregates on either input",
                 "[dense_count_join][plan]")
{
  using T = sirius::op::SiriusPhysicalOperatorType;

  SECTION("nested hash join on the counted side")
  {
    auto const [plan, fused] = require_fused(nested_hash_join_counted_query);
    CHECK(collect(fused->children[1].get(), T::HASH_JOIN).size() == 1);
    CHECK(collect(fused->children[0].get(), T::HASH_JOIN).empty());
  }

  SECTION("nested hash join on the preserved side")
  {
    auto const [plan, fused] = require_fused(nested_hash_join_preserved_query);
    CHECK(collect(fused->children[0].get(), T::HASH_JOIN).size() == 1);
    CHECK(collect(fused->children[1].get(), T::HASH_JOIN).empty());
  }

  SECTION("GROUP BY + HAVING subquery on the counted side")
  {
    auto const [plan, fused] = require_fused(having_counted_query);
    CHECK(collect(fused->children[1].get(), T::HASH_GROUP_BY).size() == 1);
    CHECK(collect(fused->children[1].get(), T::FILTER).size() == 1);
    CHECK(collect(plan.get(), T::HASH_GROUP_BY).size() == 1);
  }

  SECTION("RIGHT orientation with the nested hash join on the preserved side")
  {
    auto const [plan, fused] = require_fused(nested_hash_join_right_preserved_query);
    CHECK(collect(fused->children[0].get(), T::HASH_JOIN).size() == 1);
    CHECK(collect(fused->children[1].get(), T::HASH_JOIN).empty());
  }

  SECTION("nested dense count-join as the counted input")
  {
    auto plan = generate_sirius_plan(*con, nested_dense_count_join_query);
    REQUIRE(plan);
    // collect() is pre-order, so the outer operator comes first.
    auto const fused = collect(plan.get(), T::DENSE_COUNT_JOIN);
    REQUIRE(fused.size() == 2);
    REQUIRE(fused[0]->children.size() == 2);
    CHECK(collect(fused[0]->children[1].get(), T::DENSE_COUNT_JOIN).size() == 1);
    CHECK(collect(fused[0]->children[0].get(), T::DENSE_COUNT_JOIN).empty());
    CHECK(collect(plan.get(), T::HASH_JOIN).empty());
  }
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines bespoke-wired input roots",
                 "[dense_count_join][plan]")
{
  using T = sirius::op::SiriusPhysicalOperatorType;

  SECTION("materialized CTE reference")
  {
    auto const query =
      "WITH t AS MATERIALIZED (SELECT o_cust, o_id FROM ord) "
      "SELECT c_id, count(t.o_id) FROM cust LEFT JOIN t ON c_id = t.o_cust GROUP BY c_id";
    CHECK_FALSE(has_dense_count_join(query));
    auto plan = generate_sirius_plan(*con, query);
    REQUIRE(plan);
    CHECK(collect(plan.get(), T::CTE).size() == 1);
  }

  // Correlated subqueries whose delim join survives the optimizer's deliminator (the correlated
  // column reaches a non-equality predicate or an aggregate argument) and roots the join input.
  auto const require_delim_root_declined = [&](std::string const& query) {
    CHECK_FALSE(has_dense_count_join(query));
    auto plan = generate_sirius_plan(*con, query);
    REQUIRE(plan);
    CHECK(collect(plan.get(), T::LEFT_DELIM_JOIN).size() +
            collect(plan.get(), T::RIGHT_DELIM_JOIN).size() ==
          1);
  };

  SECTION("semi delim join from a correlated EXISTS")
  {
    require_delim_root_declined(
      "SELECT c_id, count(t.o_id) FROM cust LEFT JOIN ("
      "  SELECT o_id, o_cust FROM ord o"
      "  WHERE EXISTS (SELECT 1 FROM ord o2 WHERE o2.o_cust = o.o_cust AND o2.o_id > o.o_id)"
      ") t ON c_id = t.o_cust GROUP BY c_id");
  }

  SECTION("single delim join from a correlated scalar subquery")
  {
    require_delim_root_declined(
      "SELECT c_id, count(t.a) FROM cust LEFT JOIN ("
      "  SELECT o_cust, (SELECT sum(o2.o_id - o.o_id) FROM ord o2 WHERE o2.o_cust = o.o_cust) AS a"
      "  FROM ord o"
      ") t ON c_id = t.o_cust GROUP BY c_id");
  }
}

TEST_CASE_METHOD(dense_count_join_fixture,
                 "dense_count_join declines a delim join at any depth of an input",
                 "[dense_count_join][plan]")
{
  using T = sirius::op::SiriusPhysicalOperatorType;

  // A correlated EXISTS compared as a value keeps a MARK delim join below the input's root. These
  // plans contain hash joins, so the fused operator's absence is checked directly.
  auto const require_declined = [&](std::string const& query) {
    auto plan = generate_sirius_plan(*con, query);
    REQUIRE(plan);
    CHECK(collect(plan.get(), T::DENSE_COUNT_JOIN).empty());
    CHECK(collect(plan.get(), T::LEFT_DELIM_JOIN).size() +
            collect(plan.get(), T::RIGHT_DELIM_JOIN).size() ==
          1);
  };

  SECTION("below a FILTER root")
  {
    require_declined(
      "SELECT c.c_id, count(o_id) FROM ord RIGHT JOIN ("
      "  SELECT c_id FROM cust c1 WHERE (EXISTS (SELECT 1 FROM cust c2"
      "    WHERE c2.c_grp = c1.c_grp AND c2.c_id > c1.c_id)) = (c_id % 2 = 0)"
      ") c ON o_cust = c.c_id GROUP BY c.c_id");
  }

  SECTION("below a FILTER root, counting every joined row")
  {
    require_declined(
      "SELECT c.c_id, count(*) FROM ord RIGHT JOIN ("
      "  SELECT c_id FROM cust c1 WHERE (EXISTS (SELECT 1 FROM cust c2"
      "    WHERE c2.c_grp = c1.c_grp AND c2.c_id > c1.c_id)) = (c_id % 2 = 0)"
      ") c ON o_cust = c.c_id GROUP BY c.c_id");
  }

  SECTION("below a FILTER root, equality-only correlation")
  {
    require_declined(
      "SELECT c.c_id, count(o_id) FROM ord RIGHT JOIN ("
      "  SELECT c_id FROM cust c1 WHERE (EXISTS (SELECT 1 FROM cust c2"
      "    WHERE c2.c_grp = c1.c_grp AND c2.c_id = c1.c_id + 1)) = (c_id % 2 = 0)"
      ") c ON o_cust = c.c_id GROUP BY c.c_id");
  }

  SECTION("below a non-identity PROJECTION")
  {
    require_declined(
      "SELECT c.k, count(o_id) FROM ord RIGHT JOIN ("
      "  SELECT c_id + 0 AS k FROM cust c1 WHERE (EXISTS (SELECT 1 FROM cust c2"
      "    WHERE c2.c_grp = c1.c_grp AND c2.c_id > c1.c_id)) = (c_id % 2 = 0)"
      ") c ON o_cust = c.k GROUP BY c.k");
  }

  SECTION("below a GROUP BY")
  {
    require_declined(
      "SELECT c.c_id, count(o_id) FROM ord RIGHT JOIN ("
      "  SELECT c_id FROM cust c1 WHERE (EXISTS (SELECT 1 FROM cust c2"
      "    WHERE c2.c_grp = c1.c_grp AND c2.c_id > c1.c_id)) = (c_id % 2 = 0)"
      "  GROUP BY c_id"
      ") c ON o_cust = c.c_id GROUP BY c.c_id");
  }
}

TEST_CASE("dense_count_join recognizes host COUNT callbacks through a dynamically loaded extension",
          "[dense_count_join][plan][dynamic_load]")
{
  scoped_temp_directory temp;
  auto const executable = std::filesystem::canonical("/proc/self/exe");
  auto const extension =
    executable.parent_path().parent_path().parent_path() / "sirius.duckdb_extension";
  auto const config = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "config" /
                      "data" / "configurator_dense_count_join.yaml";
  REQUIRE(std::filesystem::is_regular_file(extension));
  REQUIRE(std::filesystem::is_regular_file(config));

  static constexpr std::string_view script = R"PY(
import os
import sys
from pathlib import Path

extension, config, temp_root = sys.argv[1:]
root = Path(temp_root)
logs = root / "logs"
logs.mkdir()

os.environ.pop("SIRIUS_DISABLE", None)
os.environ["SIRIUS_CONFIG_FILE"] = config
os.environ["SIRIUS_LOG_DIR"] = str(logs)
os.environ["SIRIUS_LOG_BACKEND"] = "spdlog"
os.environ["SIRIUS_LOG_LEVEL"] = "info"

import duckdb

def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"

customer_path = root / "customer.parquet"
orders_path = root / "orders.parquet"
con = duckdb.connect(":memory:", config={"allow_unsigned_extensions": "true"})
con.execute(
    f"COPY (SELECT range::INTEGER AS c_custkey FROM range(9)) "
    f"TO {sql_literal(customer_path)} (FORMAT PARQUET)"
)
con.execute(
    "COPY (SELECT range::BIGINT AS o_orderkey, "
    "             (range % 8)::INTEGER AS o_custkey, "
    "             CASE WHEN range % 5 = 0 THEN 'special x requests' ELSE 'ordinary' END AS o_comment "
    "      FROM range(64)) "
    f"TO {sql_literal(orders_path)} (FORMAT PARQUET)"
)
con.execute(
    f"CREATE VIEW customer AS SELECT * FROM read_parquet([{sql_literal(customer_path)}])"
)
con.execute(
    f"CREATE VIEW orders AS SELECT * FROM read_parquet([{sql_literal(orders_path)}])"
)

con.execute(f"LOAD {sql_literal(extension)}")
con.execute("SET gpu_execution = true")
con.execute("SET enable_duckdb_fallback = false")
con.execute(
    f"CALL pin_table({sql_literal(customer_path)}, tier='host', "
    "name='customer', cols=['c_custkey'])"
).fetchall()
con.execute(
    f"CALL pin_table({sql_literal(orders_path)}, tier='host', "
    "name='orders', cols=['o_custkey','o_orderkey','o_comment'])"
).fetchall()

queries = [
    (
        "SELECT c_custkey, count(o_orderkey) FROM customer c LEFT JOIN orders o "
        "ON c.c_custkey = o.o_custkey AND o.o_comment NOT LIKE '%special%requests%' "
        "GROUP BY c_custkey ORDER BY c_custkey",
        [(0, 6), (1, 7), (2, 6), (3, 7), (4, 6), (5, 6), (6, 7), (7, 6), (8, 0)],
    ),
    (
        "SELECT c_custkey, count(*) FROM customer c LEFT JOIN orders o "
        "ON c.c_custkey = o.o_custkey AND o.o_comment NOT LIKE '%special%requests%' "
        "GROUP BY c_custkey ORDER BY c_custkey",
        [(0, 6), (1, 7), (2, 6), (3, 7), (4, 6), (5, 6), (6, 7), (7, 6), (8, 1)],
    ),
]
for query, expected in queries:
    actual = con.execute(query).fetchall()
    if actual != expected:
        raise AssertionError((query, actual, expected))
con.close()
)PY";

  auto const child_output_path = temp.path() / "python-output.txt";
  auto const child_output_fd =
    ::open(child_output_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  REQUIRE(child_output_fd >= 0);

  auto const pid = ::fork();
  if (pid != 0) { ::close(child_output_fd); }
  REQUIRE(pid >= 0);
  if (pid == 0) {
    if (::dup2(child_output_fd, STDOUT_FILENO) < 0 || ::dup2(child_output_fd, STDERR_FILENO) < 0) {
      ::_exit(126);
    }
    ::close(child_output_fd);
    ::execlp("python",
             "python",
             "-c",
             script.data(),
             extension.c_str(),
             config.c_str(),
             temp.path().c_str(),
             static_cast<char*>(nullptr));
    ::_exit(127);
  }

  int status            = 0;
  int wait_error        = 0;
  pid_t waited          = 0;
  bool timed_out        = false;
  auto const deadline   = std::chrono::steady_clock::now() + std::chrono::seconds{120};
  auto const poll_delay = std::chrono::milliseconds{20};
  while (true) {
    waited = ::waitpid(pid, &status, WNOHANG);
    if (waited == pid) { break; }
    if (waited < 0) {
      if (errno == EINTR) { continue; }
      wait_error = errno;
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      timed_out = true;
      ::kill(pid, SIGKILL);
      do {
        waited = ::waitpid(pid, &status, 0);
      } while (waited < 0 && errno == EINTR);
      break;
    }
    std::this_thread::sleep_for(poll_delay);
  }

  std::ifstream child_output_stream(child_output_path);
  std::string const child_output{std::istreambuf_iterator<char>{child_output_stream},
                                 std::istreambuf_iterator<char>{}};
  INFO("dynamic-load child output:\n" << child_output);
  REQUIRE_FALSE(timed_out);
  REQUIRE(wait_error == 0);
  REQUIRE(waited == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);

  std::string log_text;
  for (auto const& entry : std::filesystem::directory_iterator(temp.path() / "logs")) {
    if (!entry.is_regular_file()) { continue; }
    std::ifstream log_stream(entry.path());
    log_text.append(std::istreambuf_iterator<char>{log_stream}, std::istreambuf_iterator<char>{});
    log_text.push_back('\n');
  }

  static constexpr std::string_view marker = "Fusing COUNT-join into DENSE_COUNT_JOIN";
  std::vector<std::string> fusion_lines;
  std::istringstream log_lines{log_text};
  for (std::string line; std::getline(log_lines, line);) {
    if (line.find(marker) != std::string::npos) { fusion_lines.push_back(std::move(line)); }
  }

  INFO("dynamic-load Sirius logs:\n" << log_text);
  REQUIRE(fusion_lines.size() >= 2);
  bool saw_count_column = false;
  bool saw_count_star   = false;
  for (auto const& line : fusion_lines) {
    saw_count_column = saw_count_column || line.find("COUNT(col ") != std::string::npos;
    saw_count_star   = saw_count_star || line.find("COUNT(*)") != std::string::npos;
  }
  CHECK(saw_count_column);
  CHECK(saw_count_star);
}
