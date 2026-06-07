/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

/**
 * @file test_pipeline_merge_fusion.cpp
 * @brief Regression tests for merge-pipeline downstream fusion in sirius_pipeline_converter.
 *
 * Verifies that pipelineable downstream operators fold into merge pipelines while
 * structural downstream sinks (ORDER BY, HASH JOIN, etc.) remain separate split targets.
 *
 * Exercises sirius_pipeline_converter only (no repository materialization) so complex
 * plans can be inspected without tripping shared data-repo registration limits.
 */

#include <catch.hpp>
#include <config.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/settings.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <op/sirius_physical_partition.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <pipeline/sirius_pipeline_converter.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <sirius_context.hpp>
#include <sirius_extension.hpp>
#include <sirius_interface.hpp>

#include <cstdlib>
#include <filesystem>
#include <source_location>
#include <string>

using namespace duckdb;

using sirius::sirius_prepared_statement_data;
using sirius::op::sirius_physical_materialized_collector;
using sirius::op::sirius_physical_operator;
using sirius::op::sirius_physical_partition;
using sirius::op::sirius_physical_result_collector;
using sirius::op::SiriusPhysicalOperatorType;
using sirius::pipeline::pipeline_conversion_result;
using sirius::pipeline::sirius_meta_pipeline;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using sirius::pipeline::sirius_pipeline_converter;
using sirius::planner::sirius_physical_plan_generator;

namespace {

void set_test_config_env()
{
  static bool env_set = false;
  if (!env_set) {
    std::source_location loc = std::source_location::current();
    auto cfg_path = std::filesystem::path(loc.file_name()).parent_path().parent_path() / "config" /
                    "data" / "minimal.yaml";
    setenv("SIRIUS_CONFIG_FILE", cfg_path.string().c_str(), 1);
    env_set = true;
  }
}

void safe_load_extension(DuckDB& db)
{
  try {
    db.LoadStaticExtension<SiriusExtension>();
  } catch (const std::exception& e) {
    std::string msg = e.what();
    if (msg.find("already exists") == std::string::npos &&
        msg.find("already loaded") == std::string::npos) {
      throw;
    }
  }
}

void safe_init_gpu_buffer(Connection& con)
{
  try {
    con.Query("CALL gpu_buffer_init('1 GB', '1 GB');");
  } catch (const std::exception& e) {
    std::string msg = e.what();
    if (msg.find("already") == std::string::npos) { throw; }
  }
}

void create_minimal_schema(Connection& con)
{
  con.Query("DROP TABLE IF EXISTS lineitem;");
  con.Query("DROP TABLE IF EXISTS orders;");

  con.Query(R"(
    CREATE TABLE orders (
      o_orderkey BIGINT NOT NULL UNIQUE PRIMARY KEY,
      o_custkey INTEGER NOT NULL,
      o_orderstatus CHAR(1) NOT NULL,
      o_totalprice DECIMAL(15,2) NOT NULL,
      o_orderdate DATE NOT NULL,
      o_orderpriority CHAR(15) NOT NULL,
      o_clerk CHAR(15) NOT NULL,
      o_shippriority INTEGER NOT NULL,
      o_comment VARCHAR(79) NOT NULL
    );
  )");

  con.Query(R"(
    CREATE TABLE lineitem (
      l_orderkey BIGINT NOT NULL,
      l_partkey BIGINT NOT NULL,
      l_suppkey BIGINT NOT NULL,
      l_linenumber INTEGER NOT NULL,
      l_quantity DECIMAL(15,2) NOT NULL,
      l_extendedprice DECIMAL(15,2) NOT NULL,
      l_discount DECIMAL(15,2) NOT NULL,
      l_tax DECIMAL(15,2) NOT NULL,
      l_returnflag CHAR(1) NOT NULL,
      l_linestatus CHAR(1) NOT NULL,
      l_shipdate DATE NOT NULL,
      l_commitdate DATE NOT NULL,
      l_receiptdate DATE NOT NULL,
      l_shipinstruct CHAR(25) NOT NULL,
      l_shipmode CHAR(10) NOT NULL,
      l_comment VARCHAR(44) NOT NULL
    );
  )");

  con.Query(R"(
    INSERT INTO orders VALUES
      (1, 1, 'O', 1000.00, '1995-01-01', '1-URGENT', 'Clerk#000000001', 0, 'comment'),
      (2, 2, 'F', 2000.00, '1996-10-15', '2-HIGH', 'Clerk#000000002', 1, 'comment'),
      (3, 3, 'F', 3000.00, '1997-06-01', '3-MEDIUM', 'Clerk#000000003', 0, 'comment');
  )");

  con.Query(R"(
    INSERT INTO lineitem VALUES
      (1, 1, 1, 1, 10.00, 1000.00, 0.05, 0.08, 'A', 'F', '1995-01-15', '1995-01-10', '1995-01-20', 'DELIVER IN PERSON', 'TRUCK', 'comment'),
      (1, 2, 2, 2, 20.00, 2000.00, 0.06, 0.07, 'N', 'O', '1996-06-01', '1996-05-01', '1996-06-15', 'NONE', 'AIR', 'comment'),
      (2, 1, 1, 1, 15.00, 1500.00, 0.04, 0.06, 'R', 'F', '1994-03-15', '1994-03-10', '1994-03-20', 'COLLECT COD', 'REG AIR', 'comment'),
      (2, 3, 2, 2, 25.00, 2500.00, 0.03, 0.05, 'A', 'F', '1993-06-01', '1993-05-15', '1993-06-10', 'TAKE BACK RETURN', 'SHIP', 'comment'),
      (3, 2, 3, 1, 30.00, 3000.00, 0.02, 0.04, 'N', 'O', '1997-07-01', '1997-06-15', '1997-07-15', 'DELIVER IN PERSON', 'TRUCK', 'comment');
  )");
}

static duckdb::shared_ptr<sirius_prepared_statement_data> g_sirius_prepared;

duckdb::unique_ptr<sirius_physical_operator> generate_gpu_plan(Connection& con,
                                                               const std::string& query)
{
  con.context->config.enable_optimizer      = true;
  con.context->config.use_replacement_scans = false;

  set<OptimizerType> disabled_optimizers;
  disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
  disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  DBConfig::GetConfig(*con.context).options.disabled_optimizers = disabled_optimizers;

  con.Query("BEGIN TRANSACTION");

  Parser parser(con.context->GetParserOptions());
  parser.ParseQuery(query);

  Planner planner(*con.context);
  auto statement_type = parser.statements[0]->type;
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan = std::move(planner.plan);

  duckdb::Optimizer optimizer(*planner.binder, *con.context);
  logical_plan = optimizer.Optimize(std::move(logical_plan));

  logical_plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver::Verify(*logical_plan);
  resolver.VisitOperator(*logical_plan);

  sirius_physical_plan_generator physical_planner(*con.context);
  auto sirius_physical_plan = physical_planner.create_plan(std::move(logical_plan));

  g_sirius_prepared =
    make_shared_ptr<sirius_prepared_statement_data>(prepared, std::move(sirius_physical_plan));

  auto gpu_collector =
    make_uniq_base<sirius_physical_result_collector, sirius_physical_materialized_collector>(
      *g_sirius_prepared, *con.context);

  con.Query("COMMIT TRANSACTION");

  return gpu_collector;
}

struct converter_test_state {
  duckdb::unique_ptr<DuckDB> db;
  duckdb::unique_ptr<Connection> con;
  duckdb::unique_ptr<sirius_physical_operator> gpu_plan;
  pipeline_conversion_result conversion;
};

converter_test_state setup_and_convert(const std::string& query)
{
  converter_test_state state;
  set_test_config_env();
  Config::MODIFIED_PIPELINE = true;
  unsetenv("SIRIUS_DISABLE");
  state.db = duckdb::make_uniq<DuckDB>(nullptr);
  safe_load_extension(*state.db);
  state.con = duckdb::make_uniq<Connection>(*state.db);
  safe_init_gpu_buffer(*state.con);
  create_minimal_schema(*state.con);

  state.gpu_plan = generate_gpu_plan(*state.con, query);
  REQUIRE(state.gpu_plan != nullptr);

  auto sirius_ctx =
    state.con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);
  const sirius::operator_params& op_params = sirius_ctx->get_config().get_operator_params();

  sirius::pipeline::pipeline_build_context build_ctx;
  build_ctx.preserve_insertion_order =
    duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(*state.con->context);
  build_ctx.num_gpus = static_cast<int>(sirius_ctx->get_config().get_hw_topology().gpus.size());

  sirius_pipeline_build_state pipeline_state;
  auto root_pipeline =
    duckdb::make_shared_ptr<sirius_meta_pipeline>(build_ctx, pipeline_state, nullptr);
  root_pipeline->build(*state.gpu_plan);
  root_pipeline->ready();

  sirius_pipeline_converter converter(build_ctx, op_params, nullptr, state.con->context.get());
  state.conversion = converter.convert(*root_pipeline);

  return state;
}

size_t count_sink_type(const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
                       SiriusPhysicalOperatorType type)
{
  size_t count = 0;
  for (const auto& pipeline : pipelines) {
    if (pipeline->get_sink()->type == type) { count++; }
  }
  return count;
}

//! True when a merge op and downstream pipelineable ops share one pipeline (fusion succeeded).
//! Does not require PROJECTION — identity projections are omitted by the planner and fusion may
//! fold straight into RESULT_COLLECTOR.
bool has_fused_merge_pipeline(const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
                              SiriusPhysicalOperatorType merge_type)
{
  for (const auto& pipeline : pipelines) {
    bool has_merge = false;
    for (auto& op : pipeline->get_operators()) {
      if (op.get().type == merge_type) { has_merge = true; }
    }
    if (has_merge && pipeline->get_sink()->type != merge_type) { return true; }
  }
  return false;
}

bool has_standalone_merge_pipeline(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
  SiriusPhysicalOperatorType merge_type)
{
  for (const auto& pipeline : pipelines) {
    if (pipeline->get_sink()->type != merge_type) { continue; }
    if (pipeline->get_operators().size() <= 1) { return true; }
  }
  return false;
}

size_t count_build_join_partitions(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines)
{
  size_t count = 0;
  for (const auto& pipeline : pipelines) {
    if (pipeline->get_sink()->type != SiriusPhysicalOperatorType::PARTITION) { continue; }
    auto& partition = pipeline->get_sink()->Cast<sirius_physical_partition>();
    if (partition.is_build_partition()) { count++; }
  }
  return count;
}

bool pipeline_has_merge_with_sink(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
  SiriusPhysicalOperatorType merge_type,
  SiriusPhysicalOperatorType sink_type)
{
  for (const auto& pipeline : pipelines) {
    bool has_merge = false;
    for (auto& op : pipeline->get_operators()) {
      if (op.get().type == merge_type) { has_merge = true; }
    }
    if (has_merge && pipeline->get_sink()->type == sink_type) { return true; }
  }
  return false;
}

size_t count_operator_type(const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines,
                           SiriusPhysicalOperatorType type)
{
  size_t count = 0;
  for (const auto& pipeline : pipelines) {
    for (auto& op : pipeline->get_operators()) {
      if (op.get().type == type) { count++; }
    }
  }
  return count;
}

}  // namespace

TEST_CASE("merge fusion folds downstream after GROUP BY", "[pipeline_converter][merge_fusion]")
{
  auto state = setup_and_convert(R"(
    SELECT SUM(l_quantity) AS total, l_returnflag
    FROM lineitem
    GROUP BY l_returnflag
  )");

  const auto& pipelines = state.conversion.scheduled_pipelines;
  REQUIRE(has_fused_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_GROUP_BY));
  REQUIRE_FALSE(
    has_standalone_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_GROUP_BY));
}

TEST_CASE("merge fusion folds downstream after TOP_N", "[pipeline_converter][merge_fusion]")
{
  auto state = setup_and_convert(R"(
    SELECT l_extendedprice, l_orderkey
    FROM lineitem
    ORDER BY l_extendedprice DESC
    LIMIT 10
  )");

  const auto& pipelines = state.conversion.scheduled_pipelines;
  REQUIRE(has_fused_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_TOP_N));
  REQUIRE_FALSE(has_standalone_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_TOP_N));
}

TEST_CASE("merge fusion stops before ORDER BY after GROUP BY", "[pipeline_converter][merge_fusion]")
{
  auto state = setup_and_convert(R"(
    SELECT SUM(l_quantity) AS total, l_returnflag
    FROM lineitem
    GROUP BY l_returnflag
    ORDER BY total DESC
  )");

  const auto& pipelines = state.conversion.scheduled_pipelines;
  // ORDER BY is a structural sink — fusion is blocked and merge stays on its own pipeline.
  REQUIRE(has_standalone_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_GROUP_BY));
  REQUIRE_FALSE(has_fused_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_GROUP_BY));
  REQUIRE_FALSE(pipeline_has_merge_with_sink(
    pipelines, SiriusPhysicalOperatorType::MERGE_GROUP_BY, SiriusPhysicalOperatorType::ORDER_BY));
  REQUIRE(count_sink_type(pipelines, SiriusPhysicalOperatorType::ORDER_BY) >= 1);
  REQUIRE(count_operator_type(pipelines, SiriusPhysicalOperatorType::MERGE_SORT) >= 1);
}

TEST_CASE("merge fusion stops before HASH JOIN after GROUP BY",
          "[pipeline_converter][merge_fusion]")
{
  auto state = setup_and_convert(R"(
    SELECT grouped.total, grouped.l_returnflag, o.o_orderkey
    FROM (
      SELECT l_returnflag, l_orderkey, SUM(l_quantity) AS total
      FROM lineitem
      GROUP BY l_returnflag, l_orderkey
    ) grouped
    JOIN orders o ON grouped.l_orderkey = o.o_orderkey
  )");

  const auto& pipelines = state.conversion.scheduled_pipelines;
  // Join follows the grouped subquery; merge must not absorb the join sink.
  REQUIRE_FALSE(pipeline_has_merge_with_sink(
    pipelines, SiriusPhysicalOperatorType::MERGE_GROUP_BY, SiriusPhysicalOperatorType::HASH_JOIN));
  REQUIRE(count_build_join_partitions(pipelines) >= 1);
  REQUIRE(count_operator_type(pipelines, SiriusPhysicalOperatorType::HASH_JOIN) >= 1);
}

TEST_CASE("merge fusion does not fold UNGROUPED_AGGREGATE downstream",
          "[pipeline_converter][merge_fusion]")
{
  auto state = setup_and_convert("SELECT avg(l_quantity) FROM lineitem");

  const auto& pipelines = state.conversion.scheduled_pipelines;
  // UNGROUPED uses the same op as partial sink and merge source — keep merge separate.
  REQUIRE(has_standalone_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_AGGREGATE));
  REQUIRE_FALSE(has_fused_merge_pipeline(pipelines, SiriusPhysicalOperatorType::MERGE_AGGREGATE));
}
