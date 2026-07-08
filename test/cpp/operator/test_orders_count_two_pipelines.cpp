/*
 * Copyright 2025, Sirius Contributors.
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

// ---------------------------------------------------------------------------
// Course example: "One query, two pipelines" — SELECT count(*) FROM orders WHERE amount > 100
//
// This test recreates that query WITHOUT the SQL parser: it hand-builds the physical operator
// tree from internal Sirius components, reads `orders` from a real Parquet file, and drives the
// sirius_engine so the converter splits the plan into pipelines — exactly the split the course
// diagram shows.
//
//   SQL:      SELECT count(*) FROM orders WHERE amount > 100
//   Tree:     RESULT_COLLECTOR
//               └── UNGROUPED_AGGREGATE  (count_star)      <- a pipeline breaker: it is BOTH a
//                     └── FILTER (amount > 100)               sink (of the streaming pipeline) and
//                           └── TABLE_SCAN(sirius_read_parquet 'orders.parquet')  a source.
//
// After sirius_engine::initialize() the converter (src/pipeline/sirius_pipeline_converter.cpp)
// swaps the TABLE_SCAN for a GPU_SCAN and splits the ungrouped aggregate, yielding:
//
//   Pipeline A (streaming):  [GPU_SCAN -> FILTER -> UNGROUPED_AGGREGATE]   (partial count)
//        --FULL barrier-->
//   Pipeline B:              [MERGE_AGGREGATE]                             (count finalize)
//        --FULL barrier-->
//   Pipeline C:              [RESULT_COLLECTOR]
//
// Note vs. the diagram: the image draws two lanes, but structurally MERGE_AGGREGATE and
// RESULT_COLLECTOR live in *adjacent* pipelines, so the real plan is THREE pipelines. The "two
// pipelines" story is the streaming lane (A) vs. the terminal lane (B+C) across the FULL barrier.
//
// References: docs/super-sirius/physical-plan-generation.md (UNGROUPED_AGGREGATE split, FULL
// barriers) and docs/super-sirius/pipeline-execution.md (ports/barriers, execution model).
// ---------------------------------------------------------------------------

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/planner/expression/bound_aggregate_expression.hpp>
#include <duckdb/planner/expression/bound_comparison_expression.hpp>
#include <duckdb/planner/expression/bound_constant_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression/ast/node.hpp>
#include <helper/type_conversions.hpp>  // sirius::from_duckdb_vec
#include <op/sirius_physical_filter.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <op/sirius_physical_ungrouped_aggregate.hpp>
#include <sirius_engine.hpp>
#include <sirius_interface.hpp>
#include <utils/utils.hpp>  // make_test_db_and_connection, get_sirius_context

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;

using Type    = sirius::op::SiriusPhysicalOperatorType;
using Barrier = sirius::op::MemoryBarrierType;

namespace {

// Column layout of our `orders` Parquet (what the scan projects).
constexpr int kOrderKeyCol  = 0;  // o_orderkey : BIGINT
constexpr int kAmountCol    = 1;  // amount     : DOUBLE
constexpr int64_t kExpected = 3;  // rows with amount > 100: {100.01, 250.0, 500.0}

fs::path config_path() { return fs::path(__FILE__).parent_path() / "result.yaml"; }

// Write a small `orders(o_orderkey BIGINT, amount DOUBLE)` Parquet with a real mix of rows above
// and below the threshold (incl. the boundary row amount == 100.0, which strict `> 100` excludes).
// SIRIUS_DISABLE=1 keeps the extension from building a SiriusContext on this throwaway writer DB.
void generate_orders_parquet(const fs::path& path)
{
  setenv("SIRIUS_DISABLE", "1", 1);
  {
    duckdb::DuckDB gen_db(nullptr);
    duckdb::Connection gen(gen_db);
    auto r = gen.Query(
      "COPY (SELECT * FROM (VALUES "
      "  (1::BIGINT,  50.0::DOUBLE), (2::BIGINT, 100.0::DOUBLE), (3::BIGINT, 100.01::DOUBLE), "
      "  (4::BIGINT, 250.0::DOUBLE), (5::BIGINT,  99.99::DOUBLE), (6::BIGINT, 500.0::DOUBLE)) "
      "  AS t(o_orderkey, amount)) TO '" +
      path.string() + "' (FORMAT PARQUET);");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
  }
  unsetenv("SIRIUS_DISABLE");
}

// count(*) as a Sirius AST node, via a dummy DuckDB BoundAggregateExpression. The GPU aggregate
// only needs the function *name* ("count_star"); the function pointers can all be null. This
// mirrors test/cpp/operator/test_physical_ungrouped_aggregate.cpp.
std::unique_ptr<sirius::ast::node> make_count_star_node()
{
  duckdb::AggregateFunction fn("count_star",
                               {},
                               duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT),
                               0,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr);
  auto agg = duckdb::make_uniq<duckdb::BoundAggregateExpression>(
    std::move(fn),
    duckdb::vector<duckdb::unique_ptr<duckdb::Expression>>{},  // count(*) has no argument
    nullptr,
    nullptr,
    duckdb::AggregateType::NON_DISTINCT);
  return sirius::ast::from_duckdb(*agg);
}

// Find the (single) pipeline whose sink operator has the given type.
sirius::pipeline::sirius_pipeline* find_by_sink(
  const duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>>& pipelines, Type type)
{
  for (const auto& p : pipelines) {
    auto sink = p->get_sink();
    if (sink && sink->type == type) { return p.get(); }
  }
  return nullptr;
}

bool pipeline_has_operator(sirius::pipeline::sirius_pipeline& pipeline, Type type)
{
  for (auto& op : pipeline.get_operators()) {
    if (op.get().type == type) { return true; }
  }
  return false;
}

}  // namespace

TEST_CASE(
  "hand-built plan for `SELECT count(*) FROM orders WHERE amount > 100` splits into pipelines",
  "[orders_count_pipelines][shared_context]")
{
  auto tmp = fs::temp_directory_path() / ("sirius-orders-count-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  auto parquet_path = tmp / "orders.parquet";
  generate_orders_parquet(parquet_path);

  auto [db_owner, con] = sirius::make_test_db_and_connection();
  auto sirius_ctx      = sirius::get_sirius_context(con, config_path());
  REQUIRE(sirius_ctx != nullptr);

  // --- Operator 1: FROM orders  (TABLE_SCAN over the Parquet file) -------------
  // function.name == "sirius_read_parquet" + parameters == {path} makes the converter build a
  // GPU_SCAN that reads this file (bind_data is unused on that path, so it stays null). An empty
  // TableFilterSet means NO filter pushdown, so the WHERE clause stays a distinct FILTER operator
  // — matching the SCAN -> FILTER -> COUNT shape in the diagram.
  duckdb::vector<duckdb::LogicalType> orders_types;
  orders_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));  // o_orderkey
  orders_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE));  // amount

  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.push_back(duckdb::ColumnIndex(kOrderKeyCol));
  column_ids.push_back(duckdb::ColumnIndex(kAmountCol));
  duckdb::vector<duckdb::idx_t> projection_ids{kOrderKeyCol, kAmountCol};
  duckdb::vector<std::string> names{"o_orderkey", "amount"};
  duckdb::vector<duckdb::Value> parameters{duckdb::Value(parquet_path.string())};
  duckdb::virtual_column_map_t virtual_columns;
  duckdb::TableFunction table_function("sirius_read_parquet", {}, nullptr, nullptr);

  auto scan = duckdb::make_uniq<sirius::op::sirius_physical_table_scan>(
    sirius::from_duckdb_vec(orders_types),
    std::move(table_function),
    nullptr,  // bind_data — unused for the sirius_read_parquet path
    sirius::from_duckdb_vec(orders_types),
    std::move(column_ids),
    std::move(projection_ids),
    std::move(names),
    duckdb::make_uniq<duckdb::TableFilterSet>(),  // empty: no pushdown -> explicit FILTER below
    kExpected,
    duckdb::ExtraOperatorInfo(),
    std::move(parameters),
    std::move(virtual_columns));

  // --- Operator 2: WHERE amount > 100  (FILTER) --------------------------------
  auto predicate = duckdb::make_uniq<duckdb::BoundComparisonExpression>(
    duckdb::ExpressionType::COMPARE_GREATERTHAN,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(
      duckdb::LogicalType(duckdb::LogicalTypeId::DOUBLE), kAmountCol),
    duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::DOUBLE(100)));
  auto filter = duckdb::make_uniq<sirius::op::sirius_physical_filter>(
    sirius::from_duckdb_vec(orders_types), sirius::ast::from_duckdb(*predicate), kExpected);
  filter->children.push_back(std::move(scan));

  // --- Operator 3: count(*)  (UNGROUPED_AGGREGATE) -----------------------------
  duckdb::vector<duckdb::LogicalType> agg_out_types;
  agg_out_types.push_back(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT));
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
  select_list.push_back(make_count_star_node());
  auto aggregate = duckdb::make_uniq<sirius::op::sirius_physical_ungrouped_aggregate>(
    sirius::from_duckdb_vec(agg_out_types),
    std::move(select_list),
    /*estimated_cardinality=*/1,
    duckdb::TupleDataValidityType::CANNOT_HAVE_NULL_VALUES);
  aggregate->children.push_back(std::move(filter));

  // --- Wrap the tree in a materialized result collector (the plan root/sink) ---
  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->types = agg_out_types;
  prepared->names = duckdb::vector<std::string>{"count_star()"};
  // Keep `sirius_prepared` alive past the engine: the collector references the plan tree it owns.
  auto sirius_prepared =
    duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(prepared, std::move(aggregate));
  auto collector = duckdb::make_uniq_base<sirius::op::sirius_physical_operator,
                                          sirius::op::sirius_physical_materialized_collector>(
    *sirius_prepared, *con.context);

  // --- Build the pipelines: initialize() runs the converter + the aggregate split -------------
  sirius::sirius_interface iface(*con.context);
  sirius::sirius_engine engine(*con.context, iface);
  engine.initialize(std::move(collector));

  // ==== Assert the pipeline split (do this BEFORE execute(): execute() moves new_scheduled) ====
  const auto& pipelines = engine.new_scheduled;
  INFO("scheduled pipelines: " << pipelines.size());
  REQUIRE(pipelines.size() == 3);

  auto* pipe_a = find_by_sink(pipelines, Type::UNGROUPED_AGGREGATE);  // streaming lane
  auto* pipe_b = find_by_sink(pipelines, Type::MERGE_AGGREGATE);      // count finalize
  auto* pipe_c = find_by_sink(pipelines, Type::RESULT_COLLECTOR);     // terminal
  REQUIRE(pipe_a != nullptr);
  REQUIRE(pipe_b != nullptr);
  REQUIRE(pipe_c != nullptr);

  // Pipeline A is the streaming lane: GPU_SCAN source -> FILTER -> UNGROUPED_AGGREGATE sink.
  REQUIRE(pipe_a->get_source());
  REQUIRE(pipe_a->get_source()->type == Type::GPU_SCAN);
  REQUIRE(pipeline_has_operator(*pipe_a, Type::FILTER));

  // The FULL barrier: MERGE_AGGREGATE (Pipeline B's source) only starts once Pipeline A finishes
  // — every split scanned, every port drained, every task turned in (the diagram's FULL barrier).
  REQUIRE(pipe_b->get_source());
  REQUIRE(pipe_b->get_source()->type == Type::MERGE_AGGREGATE);
  auto* merge_port = pipe_b->get_source()->get_port("default");
  REQUIRE(merge_port != nullptr);
  REQUIRE(merge_port->type == Barrier::FULL);

  // Pipeline C is the terminal RESULT_COLLECTOR lane, fed by the finalized count over a FULL
  // barrier.
  REQUIRE(pipe_c->get_source());
  auto* rc_port = pipe_c->get_source()->get_port("default");
  REQUIRE(rc_port != nullptr);
  REQUIRE(rc_port->type == Barrier::FULL);

  // NOTE: This test stops at the pipeline split — the part the course diagram illustrates. It does
  // NOT call engine.execute() to read the scalar count: driving a *fully hand-built* parquet scan
  // through the task scheduler currently segfaults in task_creator::prepare_for_query, because the
  // scan is missing per-query priming that the DuckDB parser/planner path normally supplies (no
  // in-repo test drives engine.execute() from a hand-built plan). The scalar `count(*) == 3` result
  // is validated end-to-end through the SQL path instead (see the [integration] GPU-execution
  // tests, e.g. test/cpp/integration/test_pin_table_column_order.cpp).

  fs::remove_all(tmp, ec);
}
