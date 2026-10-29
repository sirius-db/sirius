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
#include "duckdb.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/planner/planner.hpp"
#include "helper/type_conversions.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_passthrough_sink.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_window.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_window.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <algorithm>
#include <map>
#include <set>

using sirius::aggregate_id;
using sirius::planner::plan_whole_partition_window;

namespace {

struct window_fixture {
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db{nullptr};
  duckdb::Connection con{db};

  window_fixture()
  {
    auto result = con.Query(
      "CREATE TABLE w(p INTEGER, q INTEGER, v INTEGER, f DOUBLE, "
      "d DECIMAL(18,2), wide DECIMAL(38,2), s VARCHAR)");
    REQUIRE_FALSE(result->HasError());
    con.BeginTransaction();
  }
  ~window_fixture() { con.Rollback(); }

  duckdb::unique_ptr<duckdb::LogicalOperator> bind(std::string const& sql, bool optimize = false)
  {
    duckdb::Parser parser(*con.context);
    parser.ParseQuery(sql);
    duckdb::Planner planner(*con.context);
    planner.CreatePlan(std::move(parser.statements[0]));
    auto plan = std::move(planner.plan);
    if (optimize) {
      duckdb::Optimizer optimizer(*planner.binder, *con.context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);
    return plan;
  }
};

duckdb::LogicalWindow* find_window_if_present(duckdb::LogicalOperator& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_WINDOW) {
    return &plan.Cast<duckdb::LogicalWindow>();
  }
  for (auto& child : plan.children) {
    if (auto* window = find_window_if_present(*child)) { return window; }
  }
  return nullptr;
}

duckdb::LogicalWindow& find_window(duckdb::LogicalOperator& plan)
{
  auto* window = find_window_if_present(plan);
  REQUIRE(window);
  return *window;
}

}  // namespace

TEST_CASE("whole-partition window planning preserves slots and bound types",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT v, sum(v) OVER (PARTITION BY p), "
    "avg(v) OVER (PARTITION BY p) FROM w");
  auto& window = find_window(*logical);
  auto plan    = plan_whole_partition_window(window);
  REQUIRE(plan.input_width == window.children[0]->types.size());
  REQUIRE(plan.output_types == window.types);
  REQUIRE(plan.prepared_types == window.children[0]->types);
  REQUIRE(plan.hidden_expressions.empty());
  REQUIRE(plan.partition_slots.size() == 1);
  REQUIRE(plan.expressions.size() == 2);
  CHECK(plan.expressions[0].function == aggregate_id::sum);
  CHECK(plan.expressions[1].function == aggregate_id::avg);
  CHECK(plan.expressions[0].argument_slot == plan.expressions[1].argument_slot);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::INTEGER);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::HUGEINT);
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DOUBLE);
  CHECK(plan.expressions[0].output_slot == plan.input_width);
  CHECK(plan.expressions[1].output_slot == plan.input_width + 1);
  CHECK(plan.expressions[0].start == duckdb::WindowBoundary::UNBOUNDED_PRECEDING);
  CHECK(plan.expressions[0].end == duckdb::WindowBoundary::CURRENT_ROW_RANGE);
}

TEST_CASE("whole-partition window planning shares hidden expressions", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT sum(v+1) OVER (PARTITION BY p+1, q), "
    "avg(v+1) OVER (PARTITION BY q, p+1) FROM w");
  auto plan = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 2);
  REQUIRE(plan.partition_slots.size() == 2);
  CHECK(plan.partition_slots[0] == plan.input_width);
  CHECK(plan.expressions[0].argument_slot == plan.input_width + 1);
  CHECK(plan.expressions[1].argument_slot == plan.input_width + 1);
  CHECK(plan.prepared_types.size() == plan.input_width + 2);
  CHECK(plan.output_types.size() == plan.input_width + 2);

  // An expression used as both a partition key and an argument gets one column.
  logical = fixture.bind("SELECT sum(p+1) OVER (PARTITION BY p+1) FROM w");
  plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 1);
  CHECK(plan.expressions[0].argument_slot == plan.partition_slots[0]);
}

TEST_CASE("whole-partition window planning admits only whole unordered frames",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const clause =
    GENERATE("",
             "PARTITION BY p",
             "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
             "PARTITION BY p RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING");
  auto logical = fixture.bind(std::string("SELECT sum(v) OVER (") + clause + ") FROM w");
  auto plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
}

TEST_CASE("whole-partition window planning rejects unsupported bound semantics",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const expression = GENERATE(
    "sum(v) OVER (ORDER BY p)",
    "sum(v) OVER (ORDER BY p ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)",
    "sum(v) OVER (ROWS BETWEEN 1 PRECEDING AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING)",
    "sum(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE CURRENT ROW)",
    "sum(v) OVER (RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE GROUP)",
    "sum(v) OVER (RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING EXCLUDE TIES)",
    "sum(DISTINCT v) OVER ()",
    "sum(v) FILTER (WHERE v > 0) OVER ()",
    "sum(v ORDER BY p) OVER ()",
    "row_number() OVER ()",
    "rank() OVER (ORDER BY p)",
    "lag(v) OVER ()",
    "sum(v::HUGEINT) OVER ()",
    "sum(v::DECIMAL(4,0)) OVER ()",
    "avg(v::DECIMAL(4,0)) OVER ()",
    "sum(v::UBIGINT) OVER ()",
    "sum(v) OVER (PARTITION BY [p])",
    "sum(v) OVER (PARTITION BY f)",
    "sum(v) OVER (PARTITION BY s COLLATE nocase)",
    "sum(random()) OVER ()",
    "sum(atan2(f, 1)) OVER ()",
    "sum(v) OVER (), avg(v) OVER (PARTITION BY p)",
    "sum(v) OVER (PARTITION BY p, p), avg(v) OVER (PARTITION BY p, q)");
  INFO(expression);
  // The table is empty. Capability rejection must still occur before execution.
  auto logical = fixture.bind(std::string("SELECT ") + expression + " FROM w");
  REQUIRE_THROWS_AS(plan_whole_partition_window(find_window(*logical)),
                    duckdb::NotImplementedException);
}

TEST_CASE("whole-partition count and extrema preserve bound result types", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT count(*) OVER (PARTITION BY p),count(v) OVER (PARTITION BY p),"
    "min(v) OVER (PARTITION BY p),max(d) OVER (PARTITION BY p),"
    "min(s) OVER (PARTITION BY p) FROM w");
  auto plan = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 5);
  CHECK(plan.expressions[0].function == aggregate_id::count_star);
  CHECK(plan.expressions[1].function == aggregate_id::count);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::BIGINT);
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::BIGINT);
  CHECK(plan.expressions[2].result_type == duckdb::LogicalType::INTEGER);
  CHECK(plan.expressions[3].result_type == duckdb::LogicalType::DECIMAL(18, 2));
  CHECK(plan.expressions[4].result_type == duckdb::LogicalType::VARCHAR);
  CHECK(plan.prepared_types[plan.expressions[0].argument_slot] == duckdb::LogicalType::INTEGER);
}

TEST_CASE("whole-partition window planning retains decimal widening", "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind("SELECT sum(d) OVER (), avg(d) OVER (), sum(f) OVER () FROM w");
  auto plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 3);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(18, 2));
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::DECIMAL(38, 2));
  CHECK(plan.expressions[1].result_type == duckdb::LogicalType::DOUBLE);
  CHECK(plan.expressions[2].result_type == duckdb::LogicalType::DOUBLE);
}

TEST_CASE("whole-partition window planner activates only admitted windows",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind("SELECT sum(v) OVER (ORDER BY p) FROM w");
  sirius::planner::sirius_physical_plan_generator generator(*fixture.con.context);
  REQUIRE_THROWS_WITH(generator.create_plan(std::move(logical)),
                      Catch::Matchers::ContainsSubstring("window ORDER BY"));
  logical       = fixture.bind("SELECT sum(v) OVER () FROM (VALUES (1),(2)) t(v)");
  auto physical = generator.create_plan(std::move(logical));
  auto find     = [](auto&& self, sirius::op::sirius_physical_operator& op) -> bool {
    if (op.type == sirius::op::SiriusPhysicalOperatorType::WINDOW) {
      CHECK(op.types.back() == sirius::logical_type::make(sirius::type_id::HUGEINT));
      REQUIRE(op.children.size() == 1);
      CHECK(op.children[0]->type == sirius::op::SiriusPhysicalOperatorType::CONCAT);
      CHECK(op.children[0]->children[0]->type == sirius::op::SiriusPhysicalOperatorType::PARTITION);
      return true;
    }
    for (auto& child : op.children) {
      if (self(self, *child)) { return true; }
    }
    return false;
  };
  REQUIRE(find(find, *physical));
}

TEST_CASE("whole-partition activation rejects unsafe consumers and chained stages",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const query = GENERATE(
    "SELECT sum(v) OVER ()+1 FROM w",
    "SELECT s,count(*) FROM (SELECT sum(v) OVER () s FROM w) t GROUP BY s",
    "SELECT sum(s) OVER (PARTITION BY p) FROM (SELECT p,sum(d) OVER (PARTITION BY q) s FROM w) t");
  auto logical = fixture.bind(query);
  REQUIRE_THROWS_AS(sirius::planner::validate_whole_partition_window_consumers(*logical),
                    duckdb::NotImplementedException);
}

TEST_CASE("ranking window planning records the order keys and bound types",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT rank() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST),"
    "dense_rank() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST),"
    "row_number() OVER (PARTITION BY p ORDER BY v DESC NULLS LAST) FROM w");
  auto& window = find_window(*logical);
  auto plan    = plan_whole_partition_window(window);
  REQUIRE(plan.expressions.empty());
  REQUIRE(plan.hidden_expressions.empty());
  REQUIRE(plan.partition_slots.size() == 1);
  REQUIRE(plan.rankings.size() == 3);
  CHECK(plan.rankings[0].function == sirius::op::ranking_function::rank);
  CHECK(plan.rankings[1].function == sirius::op::ranking_function::dense_rank);
  CHECK(plan.rankings[2].function == sirius::op::ranking_function::row_number);
  for (std::size_t i = 0; i < plan.rankings.size(); ++i) {
    CHECK(plan.rankings[i].result_type == duckdb::LogicalType::BIGINT);
    CHECK(plan.rankings[i].output_slot == plan.input_width + i);
  }
  REQUIRE(plan.ranking_order.columns.size() == 1);
  CHECK(plan.ranking_order.direction == cudf::order::DESCENDING);
  CHECK(plan.ranking_order.nulls ==
        sirius::op::to_cudf_null_order(duckdb::OrderType::DESCENDING,
                                       duckdb::OrderByNullType::NULLS_LAST));
}

TEST_CASE("ranking window planning evaluates order expressions once and keeps several keys",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical = fixture.bind(
    "SELECT rank() OVER (PARTITION BY p ORDER BY v+1, q),"
    "dense_rank() OVER (PARTITION BY p ORDER BY v+1, q) FROM w");
  auto plan = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.hidden_expressions.size() == 1);
  REQUIRE(plan.ranking_order.columns.size() == 2);
  CHECK(static_cast<std::size_t>(plan.ranking_order.columns[0]) == plan.input_width);
  CHECK(static_cast<std::size_t>(plan.ranking_order.columns[1]) < plan.input_width);
  CHECK(plan.ranking_order.direction == cudf::order::ASCENDING);
}

TEST_CASE("ranking window planning rejects unsupported shapes", "[planner][window_plan]")
{
  window_fixture fixture;
  auto const expression =
    GENERATE("rank() OVER (ORDER BY v)",
             "row_number() OVER (PARTITION BY p)",
             "rank() OVER (PARTITION BY p ORDER BY v, q DESC)",
             "rank() OVER (PARTITION BY p ORDER BY v NULLS FIRST, q NULLS LAST)",
             "rank() OVER (PARTITION BY p ORDER BY v::DECIMAL(4,0))",
             "rank() OVER (PARTITION BY p ORDER BY s COLLATE nocase)",
             "rank() OVER (PARTITION BY p ORDER BY [v])",
             "rank() OVER (PARTITION BY f ORDER BY v)",
             "percent_rank() OVER (PARTITION BY p ORDER BY v)",
             "ntile(2) OVER (PARTITION BY p ORDER BY v)",
             "sum(v) OVER (PARTITION BY p ORDER BY v)");
  INFO(expression);
  auto logical = fixture.bind(std::string("SELECT ") + expression + " FROM w");
  REQUIRE_THROWS_AS(plan_whole_partition_window(find_window(*logical)),
                    duckdb::NotImplementedException);
}

TEST_CASE("whole-partition window planning rejects invalid mappings and null treatment",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto logical     = fixture.bind("SELECT sum(v) OVER (PARTITION BY p) FROM w");
  auto& window     = find_window(*logical);
  auto& expression = window.expressions[0]->Cast<duckdb::BoundWindowExpression>();
  SECTION("unsupported null treatment") { expression.IgnoreNullsMutable() = true; }
  SECTION("incorrect visible result type") { window.types.back() = duckdb::LogicalType::DOUBLE; }
  SECTION("incorrect input mapping") { window.types[0] = duckdb::LogicalType::VARCHAR; }
  SECTION("argument outside input schema")
  {
    expression.GetChildrenMutable()[0]->Cast<duckdb::BoundReferenceExpression>().IndexMutable() =
      window.children[0]->types.size();
  }
  REQUIRE_THROWS_AS(plan_whole_partition_window(window), duckdb::NotImplementedException);
}

TEST_CASE("whole-partition planning accepts statistics-optimized SUM", "[planner][window_plan]")
{
  window_fixture fixture;
  REQUIRE_FALSE(fixture.con.Query("INSERT INTO w(p,v) VALUES (1,10),(1,20)")->HasError());
  auto settings = fixture.con.Query("SELECT current_setting('disabled_optimizers')");
  REQUIRE_FALSE(settings->HasError());
  auto const current = settings->Collection().GetValue(0, 0).ToString();
  REQUIRE_FALSE(fixture.con
                  .Query("SET disabled_optimizers='" + current + (current.empty() ? "" : ",") +
                         "window_self_join'")
                  ->HasError());
  auto logical = fixture.bind("SELECT p,sum(v) OVER (PARTITION BY p) FROM w", true);
  auto plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
  CHECK(plan.expressions[0].result_type == duckdb::LogicalType::HUGEINT);
}

TEST_CASE("whole-partition window planning checks optimized TPC-DS shapes",
          "[planner][window_plan]")
{
  window_fixture fixture;
  auto const aggregate = GENERATE("sum", "avg");
  // The grouped SUM widens decimal sales before the window consumes them.
  auto const query =
    std::string("SELECT ") + aggregate + "(sum(d)) OVER (PARTITION BY p) FROM w GROUP BY p, q";
  auto logical = fixture.bind(query);
  auto plan    = plan_whole_partition_window(find_window(*logical));
  REQUIRE(plan.expressions.size() == 1);
  CHECK(plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(38, 2));

  auto optimized = fixture.bind(query, true);
  CHECK(optimized->types == logical->types);
  if (auto* window = find_window_if_present(*optimized)) {
    auto optimized_plan = plan_whole_partition_window(*window);
    REQUIRE(optimized_plan.expressions.size() == 1);
    CHECK(optimized_plan.expressions[0].argument_type == duckdb::LogicalType::DECIMAL(38, 2));
  } else {
    // WINDOW_SELF_JOIN can replace this entire window with aggregation + join.
    // Such a plan supplies no evidence of GPU window execution.
    SUCCEED("DuckDB eliminated the window from this optimized TPC-DS shape");
  }
}

TEST_CASE("window input builder prepares expressions and materializes complete CONCAT wiring",
          "[planner][window_plan][window_input_plan]")
{
  window_fixture fixture;
  auto const partitioned = GENERATE(false, true);
  auto logical = fixture.bind(partitioned ? "SELECT sum(v+1) OVER (PARTITION BY p+1) FROM w"
                                          : "SELECT sum(v+1) OVER () FROM w");
  auto bound   = plan_whole_partition_window(find_window(*logical));
  auto const input_types = sirius::from_duckdb_vec(find_window(*logical).children[0]->types);
  auto child    = duckdb::make_uniq<sirius::op::sirius_physical_dummy_scan>(input_types, 10);
  auto assembly = sirius::planner::build_whole_partition_window_input(std::move(child), bound, 10);
  auto* input   = &assembly->Cast<sirius::op::sirius_physical_concat>();
  REQUIRE(assembly->types == sirius::from_duckdb_vec(bound.prepared_types));
  REQUIRE(assembly->children.size() == 1);
  auto& exchange = assembly->children[0]->Cast<sirius::op::sirius_physical_partition>();
  std::vector<int> keys(bound.partition_slots.begin(), bound.partition_slots.end());
  CHECK(exchange.partition_keys() == keys);
  CHECK(exchange.input_barrier_for(*exchange.children[0]) == sirius::op::MemoryBarrierType::FULL);
  REQUIRE(exchange.children.size() == 1);
  auto& projection = exchange.children[0]->Cast<sirius::op::sirius_physical_projection>();
  CHECK(projection.select_list.size() == bound.prepared_types.size());

  auto root =
    duckdb::make_uniq<sirius::op::sirius_physical_passthrough_sink>(assembly->types, 10, "default");
  root->children.push_back(std::move(assembly));
  sirius::planner::sirius_physical_plan_generator::set_parent_ops(*root, nullptr);
  sirius::pipeline::pipeline_build_context context{nullptr};
  sirius::pipeline::sirius_pipeline_build_state state;
  auto meta = std::make_shared<sirius::pipeline::sirius_meta_pipeline>(context, state, nullptr);
  meta->build(*root);
  meta->ready();
  sirius::pipeline::sirius_pipeline_converter converter(context);
  auto converted = converter.convert(*meta);
  sirius::pipeline::assign_operator_ids(converted.scheduled_pipelines);
  cucascade::shared_data_repository_manager repositories;
  sirius::pipeline::materialize_repository_wiring(converted.repository_wirings, repositories);
  auto* port = input->get_port("default");
  REQUIRE(port);
  // CONCAT keeps its ordinary streaming port, but concat_all gates claims on completion.
  CHECK(port->type == sirius::op::MemoryBarrierType::PARTIAL);
  auto hint = input->get_next_task_hint();
  REQUIRE(hint);
  CHECK(hint->hint == sirius::op::TaskCreationHint::WAITING_FOR_INPUT_DATA);
  REQUIRE(port->src_pipeline);
  CHECK(port->src_pipeline->get_sink().get() == &exchange);
  CHECK(port->dest_pipeline->get_source().get() == input);
  CHECK(repositories.get_repository(input->get_operator_id(), "default").get() == port->repo);
}

TEST_CASE("window input builder assembles fragmented keyed and unkeyed input",
          "[planner][window_input_plan][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  window_fixture fixture;
  auto const keyed = GENERATE(false, true);
  auto logical = fixture.bind(keyed ? "SELECT sum(p) OVER (PARTITION BY p) FROM (SELECT p FROM w)"
                                    : "SELECT sum(p) OVER () FROM (SELECT p FROM w)");
  auto& window = find_window(*logical);
  auto bound   = plan_whole_partition_window(window);
  REQUIRE(bound.input_width == 1);
  auto types    = sirius::from_duckdb_vec(window.children[0]->types);
  auto first    = make_numeric_batch<int32_t>(space, {1, 2, 1}, cudf::type_id::INT32);
  auto second   = make_numeric_batch<int32_t>(space, {2, 3, 1}, cudf::type_id::INT32);
  auto target   = first->to_read_only().get_data()->get_size_in_bytes();
  auto assembly = sirius::planner::build_whole_partition_window_input(
    duckdb::make_uniq<sirius_physical_dummy_scan>(types, 6), bound, 6, target);
  auto& concat = assembly->Cast<sirius_physical_concat>();
  auto& router = concat.children[0]->Cast<sirius_physical_partition>();
  cucascade::shared_data_repository input, buckets;
  auto attach = [](sirius_physical_operator& op, cucascade::shared_data_repository& repo) {
    auto port  = std::make_unique<sirius_physical_operator::port>();
    port->repo = &repo;
    port->type = MemoryBarrierType::FULL;
    op.add_port("default", std::move(port));
  };
  attach(router, input);
  attach(concat, buckets);
  router.add_next_port_after_sink({&concat, "default"});
  input.add_data_batch(first);
  input.add_data_batch(second);
  while (auto data = router.get_next_task_input_data()) {
    data->prepare_for_processing(&space, default_stream());
    auto result = router.execute(*data, default_stream());
    router.sink(*result, default_stream());
  }
  CHECK(buckets.num_partitions() == (keyed ? 2 : 1));
  std::vector<int32_t> values;
  std::set<std::size_t> claimed;
  std::map<int32_t, std::size_t> key_buckets;
  while (auto data = concat.get_next_task_input_data()) {
    auto const bucket = *dynamic_cast<partitioned_operator_data&>(*data).get_partition_idx();
    CHECK(claimed.insert(bucket).second);
    data->prepare_for_processing(&space, default_stream());
    auto result  = concat.execute(*data, default_stream());
    auto batches = dynamic_cast<pipelineable_operator_data&>(*result).get_read_only_batches();
    REQUIRE(batches.size() == 1);
    for (auto value :
         copy_column_to_host<int32_t>(sirius::get_cudf_table_view(batches[0]).column(0))) {
      auto position = key_buckets.emplace(value, bucket).first;
      CHECK(position->second == bucket);
      values.push_back(value);
    }
  }
  std::sort(values.begin(), values.end());
  CHECK(values == std::vector<int32_t>{1, 1, 1, 2, 2, 3});
}

TEST_CASE("window adapter restores narrow inputs and removes hidden slots", "[window_adapter][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  using sirius::logical_type;
  using sirius::type_id;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  // Both columns are compressed INTEGERs. Slot 1 is a hidden argument, not visible payload.
  auto batch = make_two_column_batch<int8_t, int8_t>(
    space, {1, 2, 1, 2}, {10, 20, 30, 40}, cudf::type_id::INT8, std::nullopt, cudf::type_id::INT8);
  auto stream        = space.acquire_stream();
  auto const spilled = GENERATE(false, true);
  if (spilled) {
    auto const* host = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
    auto writable    = batch->to_mutable();
    writable.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), host, stream);
    stream.sync();
  }
  sirius_physical_window window(
    {logical_type::make(type_id::INTEGER), logical_type::make(type_id::HUGEINT)},
    {logical_type::make(type_id::INTEGER), logical_type::make(type_id::INTEGER)},
    1,
    {0},
    {{aggregate_id::sum, 1}},
    4);
  pipelineable_operator_data input({batch});
  input.prepare_for_processing(&space, stream);
  auto result = window.execute(input, stream);
  stream.sync();
  auto batches = dynamic_cast<pipelineable_operator_data&>(*result).get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto table = sirius::get_cudf_table_view(batches[0]);
  REQUIRE(table.num_columns() == 2);
  CHECK(table.column(0).type().id() == cudf::type_id::INT8);
  CHECK(table.column(1).type() == cudf::data_type{cudf::type_id::DECIMAL128, 0});
  CHECK(copy_column_to_host<__int128_t>(table.column(1)) ==
        std::vector<__int128_t>{40, 60, 40, 60});
}

TEST_CASE("window source drains assembled batches across repository partition gaps",
          "[window_adapter][gpu]")
{
  using namespace sirius::op;
  using namespace sirius::test::operator_utils;
  auto manager = sirius::test::operator_utils::initialize_memory_manager();
  auto& space  = *manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  auto outputs = types;
  outputs.push_back(sirius::from_duckdb(duckdb::LogicalType::HUGEINT));
  sirius_physical_window window(outputs, types, 1, {0}, {{aggregate_id::sum, 0}}, 3);
  cucascade::shared_data_repository repo;
  repo.set_num_partitions(5);
  auto first  = make_numeric_batch<int32_t>(space, {1, 2}, cudf::type_id::INT32);
  auto second = make_numeric_batch<int32_t>(space, {3}, cudf::type_id::INT32);
  repo.add_data_batch(first, 1);
  repo.add_data_batch(second, 4);
  auto port  = std::make_unique<sirius_physical_operator::port>();
  port->repo = &repo;
  port->type = MemoryBarrierType::FULL;
  window.add_port("default", std::move(port));
  for (auto const& expected : {first, second}) {
    auto input = window.get_next_task_input_data();
    REQUIRE(input);
    auto const& batches = dynamic_cast<pipelineable_operator_data&>(*input).get_data_batches();
    REQUIRE(batches.size() == 1);
    CHECK(batches[0] == expected);
  }
  CHECK_FALSE(window.get_next_task_input_data());
  CHECK(repo.total_size() == 0);
}
