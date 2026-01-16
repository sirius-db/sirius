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

/**
 * @file test_modified_pipeline.cpp
 * @brief Tests for Config::MODIFIED_PIPELINE pipeline structure validation
 *
 * This file contains tests that verify the correctness of the modified pipeline
 * generation logic in GPUExecutor::InitializeInternal when MODIFIED_PIPELINE is enabled.
 *
 * Validation includes:
 * - Pipeline breakdown patterns (GROUP_BY → PARTITION → CONCAT → GROUP_BY, etc.)
 * - PARTITION operators added with correct port IDs (build="build", probe="default")
 * - CONCAT operators added after PARTITION for GROUP_BY/ORDER_BY/TOP_N
 * - Data repositories are non-null in each port
 * - src_pipeline and dest_pipeline are correctly connected in each port
 * - CTE handling
 * - All 22 TPC-H queries
 */

// catch2
#include <catch.hpp>

// sirius
#include <config.hpp>
#include <gpu_buffer_manager.hpp>
#include <gpu_context.hpp>
#include <gpu_executor.hpp>
#include <gpu_physical_plan_generator.hpp>
#include <gpu_pipeline.hpp>
#include <operator/gpu_physical_concat.hpp>
#include <operator/gpu_physical_cte.hpp>
#include <operator/gpu_physical_delim_join.hpp>
#include <operator/gpu_physical_partition.hpp>
#include <operator/gpu_physical_result_collector.hpp>
#include <sirius_extension.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/execution/physical_operator.hpp>
#include <duckdb/main/connection.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

// standard library
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace duckdb;

//===----------------------------------------------------------------------===//
// Test Fixture and Helper Functions
//===----------------------------------------------------------------------===//

namespace {

// Static flag to track if extension was already loaded in this process
static bool g_extension_loaded   = false;
static bool g_buffer_initialized = false;

/**
 * @brief Safely load the Sirius extension, handling the case where it's already loaded
 */
void safe_load_extension(DuckDB& db)
{
  if (!g_extension_loaded) {
    try {
      db.LoadStaticExtension<SiriusExtension>();
      g_extension_loaded = true;
    } catch (const std::exception& e) {
      // Extension might already be loaded by a previous test, that's ok
      std::string msg = e.what();
      if (msg.find("already exists") == std::string::npos &&
          msg.find("already loaded") == std::string::npos) {
        throw;  // Re-throw if it's a different error
      }
      g_extension_loaded = true;
    }
  }
}

/**
 * @brief Safely initialize the GPU buffer, handling the case where it's already initialized
 */
void safe_init_gpu_buffer(Connection& con)
{
  if (!g_buffer_initialized) {
    try {
      con.Query("CALL gpu_buffer_init('1 GB', '1 GB');");
      g_buffer_initialized = true;
    } catch (const std::exception& e) {
      // Buffer might already be initialized, that's ok
      std::string msg = e.what();
      if (msg.find("already") == std::string::npos) {
        throw;  // Re-throw if it's a different error
      }
      g_buffer_initialized = true;
    }
  }
}

/**
 * @brief Create TPC-H schema tables with minimal test data (matching tpch_load.sql)
 */
void create_tpch_schema(Connection& con)
{
  // Drop existing tables
  con.Query("DROP TABLE IF EXISTS nation;");
  con.Query("DROP TABLE IF EXISTS region;");
  con.Query("DROP TABLE IF EXISTS part;");
  con.Query("DROP TABLE IF EXISTS supplier;");
  con.Query("DROP TABLE IF EXISTS partsupp;");
  con.Query("DROP TABLE IF EXISTS orders;");
  con.Query("DROP TABLE IF EXISTS customer;");
  con.Query("DROP TABLE IF EXISTS lineitem;");

  // Create tables matching tpch_load.sql schema exactly
  con.Query(R"(
    CREATE TABLE nation (
      n_nationkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
      n_name CHAR(25) NOT NULL,
      n_regionkey INTEGER NOT NULL,
      n_comment VARCHAR(152)
    );
  )");

  con.Query(R"(
    CREATE TABLE region (
      r_regionkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
      r_name CHAR(25) NOT NULL,
      r_comment VARCHAR(152)
    );
  )");

  con.Query(R"(
    CREATE TABLE part (
      p_partkey BIGINT NOT NULL UNIQUE PRIMARY KEY,
      p_name VARCHAR(55) NOT NULL,
      p_mfgr CHAR(25) NOT NULL,
      p_brand CHAR(10) NOT NULL,
      p_type VARCHAR(25) NOT NULL,
      p_size INTEGER NOT NULL,
      p_container CHAR(10) NOT NULL,
      p_retailprice DECIMAL(15,2) NOT NULL,
      p_comment VARCHAR(23) NOT NULL
    );
  )");

  con.Query(R"(
    CREATE TABLE supplier (
      s_suppkey BIGINT NOT NULL UNIQUE PRIMARY KEY,
      s_name CHAR(25) NOT NULL,
      s_address VARCHAR(40) NOT NULL,
      s_nationkey INTEGER NOT NULL,
      s_phone CHAR(15) NOT NULL,
      s_acctbal DECIMAL(15,2) NOT NULL,
      s_comment VARCHAR(101) NOT NULL
    );
  )");

  con.Query(R"(
    CREATE TABLE partsupp (
      ps_partkey BIGINT NOT NULL,
      ps_suppkey BIGINT NOT NULL,
      ps_availqty INTEGER NOT NULL,
      ps_supplycost DECIMAL(15,2) NOT NULL,
      ps_comment VARCHAR(199) NOT NULL,
      CONSTRAINT PS_PARTSUPPKEY UNIQUE(PS_PARTKEY, PS_SUPPKEY)
    );
  )");

  con.Query(R"(
    CREATE TABLE customer (
      c_custkey INTEGER NOT NULL UNIQUE PRIMARY KEY,
      c_name VARCHAR(25) NOT NULL,
      c_address VARCHAR(40) NOT NULL,
      c_nationkey INTEGER NOT NULL,
      c_phone CHAR(15) NOT NULL,
      c_acctbal DECIMAL(15,2) NOT NULL,
      c_mktsegment CHAR(10) NOT NULL,
      c_comment VARCHAR(117) NOT NULL
    );
  )");

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

  // Insert minimal test data for all tables
  con.Query(R"(
    INSERT INTO region VALUES
      (0, 'AFRICA', 'comment'),
      (1, 'AMERICA', 'comment'),
      (2, 'ASIA', 'comment'),
      (3, 'EUROPE', 'comment'),
      (4, 'MIDDLE EAST', 'comment');
  )");

  con.Query(R"(
    INSERT INTO nation VALUES
      (0, 'ALGERIA', 0, 'comment'),
      (1, 'ARGENTINA', 1, 'comment'),
      (2, 'BRAZIL', 1, 'comment'),
      (3, 'CANADA', 1, 'comment'),
      (4, 'EGYPT', 4, 'comment'),
      (5, 'ETHIOPIA', 0, 'comment'),
      (6, 'FRANCE', 3, 'comment'),
      (7, 'GERMANY', 3, 'comment'),
      (8, 'INDIA', 2, 'comment'),
      (9, 'INDONESIA', 2, 'comment'),
      (10, 'IRAN', 4, 'comment'),
      (11, 'IRAQ', 4, 'comment'),
      (12, 'JAPAN', 2, 'comment'),
      (13, 'JORDAN', 4, 'comment'),
      (14, 'KENYA', 0, 'comment'),
      (15, 'MOROCCO', 0, 'comment'),
      (16, 'MOZAMBIQUE', 0, 'comment'),
      (17, 'PERU', 1, 'comment'),
      (18, 'CHINA', 2, 'comment'),
      (19, 'ROMANIA', 3, 'comment'),
      (20, 'SAUDI ARABIA', 4, 'comment'),
      (21, 'VIETNAM', 2, 'comment'),
      (22, 'RUSSIA', 3, 'comment'),
      (23, 'UNITED KINGDOM', 3, 'comment'),
      (24, 'UNITED STATES', 1, 'comment');
  )");

  con.Query(R"(
    INSERT INTO supplier VALUES
      (1, 'Supplier#000000001', 'addr1', 0, '000-000-0001', 1000.00, 'comment'),
      (2, 'Supplier#000000002', 'addr2', 1, '000-000-0002', 2000.00, 'comment'),
      (3, 'Supplier#000000003', 'addr3', 12, '000-000-0003', 3000.00, 'comment');
  )");

  con.Query(R"(
    INSERT INTO part VALUES
      (1, 'antique Part#1', 'Manufacturer#1', 'Brand#13', 'PROMO BRUSHED COPPER', 10, 'SM BOX', 100.00, 'comment'),
      (2, 'yellow Part#2', 'Manufacturer#2', 'Brand#41', 'MEDIUM PLATED STEEL', 20, 'JUMBO CAN', 200.00, 'comment'),
      (3, 'Part#3', 'Manufacturer#3', 'Brand#55', 'TYPE', 30, 'LG PACK', 300.00, 'comment');
  )");

  con.Query(R"(
    INSERT INTO partsupp VALUES
      (1, 1, 100, 10.00, 'comment'),
      (1, 2, 200, 20.00, 'comment'),
      (2, 1, 300, 30.00, 'comment'),
      (2, 3, 400, 40.00, 'comment'),
      (3, 2, 500, 50.00, 'comment');
  )");

  con.Query(R"(
    INSERT INTO customer VALUES
      (1, 'Customer#000000001', 'addr1', 0, '11-000-0001', 1000.00, 'BUILDING', 'comment'),
      (2, 'Customer#000000002', 'addr2', 1, '24-000-0002', 2000.00, 'HOUSEHOLD', 'comment'),
      (3, 'Customer#000000003', 'addr3', 4, '31-000-0003', 3000.00, 'MACHINERY', 'comment');
  )");

  con.Query(R"(
    INSERT INTO orders VALUES
      (1, 1, 'O', 1000.00, '1995-01-01', '1-URGENT', 'Clerk#000000001', 0, 'comment'),
      (2, 2, 'F', 2000.00, '1996-10-15', '2-HIGH', 'Clerk#000000002', 1, 'comment'),
      (3, 3, 'F', 3000.00, '1997-06-01', '3-MEDIUM', 'Clerk#000000003', 0, 'special requests comment');
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

// Static storage to keep GPUPreparedStatementData alive during test execution
// (GPUPhysicalResultCollector holds a reference to it)
static duckdb::shared_ptr<GPUPreparedStatementData> g_gpu_prepared;

/**
 * @brief Generate GPU physical plan wrapped in a result collector
 *
 * This mirrors the actual Sirius extension execution:
 * 1. A separate connection extracts the logical plan
 * 2. GPUPhysicalPlanGenerator creates the raw GPU plan
 * 3. GPUPhysicalMaterializedCollector wraps the plan (like GPUPendingStatementOrPreparedStatement
 * does)
 * 4. The result collector is passed to the executor
 */
duckdb::unique_ptr<GPUPhysicalOperator> generate_gpu_plan(Connection& con,
                                                          GPUContext& gpu_context,
                                                          const std::string& query)
{
  // Create a separate connection for extracting the logical plan (like SiriusInitPlanExtractor)
  // Connection plan_conn(*con.context->db);
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

  // Create PreparedStatementData with the query's types and names
  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);
  // prepared->plan      = make_uniq<PhysicalOperator>(
  //   PhysicalOperatorType::DUMMY_SCAN, duckdb::vector<LogicalType>{LogicalType::BOOLEAN}, 0);

  // Extract the logical plan using Sirius-style extraction (with optimizer configuration)
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan;

  logical_plan = std::move(planner.plan);

  duckdb::Optimizer optimizer(*planner.binder, *con.context);
  logical_plan = optimizer.Optimize(std::move(logical_plan));

  // After optimization, refresh types before column binding resolution
  logical_plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver::Verify(*logical_plan);
  resolver.VisitOperator(*logical_plan);

  // Create the raw GPU physical plan
  GPUPhysicalPlanGenerator physical_planner(*con.context, gpu_context);
  auto gpu_physical_plan = physical_planner.CreatePlan(std::move(logical_plan));

  // Create GPUPreparedStatementData (like GPUProcessingBind does)
  // Store in static variable to keep it alive (GPUPhysicalResultCollector holds a reference)
  g_gpu_prepared =
    make_shared_ptr<GPUPreparedStatementData>(prepared, std::move(gpu_physical_plan));

  // Create the result collector that wraps the GPU plan (like
  // GPUPendingStatementOrPreparedStatement does)
  auto gpu_collector =
    make_uniq_base<GPUPhysicalResultCollector, GPUPhysicalMaterializedCollector>(*g_gpu_prepared);

  con.Query("COMMIT TRANSACTION");

  return gpu_collector;
}

/**
 * @brief Validate that a port has a non-null repository
 */
void validate_port_repository(GPUPhysicalOperator::port* port, const std::string& context_msg)
{
  INFO("Context: " << context_msg);
  REQUIRE(port != nullptr);
  REQUIRE(port->repo != nullptr);
}

/**
 * @brief Build a map from source operator to pipelines that use it as source
 */
std::unordered_map<const GPUPhysicalOperator*, duckdb::vector<duckdb::shared_ptr<GPUPipeline>>>
build_source_to_pipelines_map(const duckdb::vector<duckdb::shared_ptr<GPUPipeline>>& pipelines)
{
  std::unordered_map<const GPUPhysicalOperator*, duckdb::vector<duckdb::shared_ptr<GPUPipeline>>>
    result;
  for (const auto& pipeline : pipelines) {
    result[pipeline->GetSource().get()].push_back(pipeline);
  }
  return result;
}

/**
 * @brief Count pipelines with PARTITION sinks (type=INVALID)
 */
size_t count_partition_sinks(const duckdb::vector<duckdb::shared_ptr<GPUPipeline>>& pipelines)
{
  size_t count = 0;
  for (const auto& pipeline : pipelines) {
    if (pipeline->GetSink()->type == PhysicalOperatorType::INVALID) { count++; }
  }
  return count;
}

/**
 * @brief Check if any pipeline contains a CONCAT operator
 */
bool has_concat_operator(const duckdb::vector<duckdb::shared_ptr<GPUPipeline>>& pipelines)
{
  for (const auto& pipeline : pipelines) {
    auto ops = pipeline->GetInnerOperators();
    for (auto& op : ops) {
      if (op.get().GetName() == "CONCAT") { return true; }
    }
  }
  return false;
}

/**
 * @brief Count pipelines with a specific sink type
 */
size_t count_sink_type(const duckdb::vector<duckdb::shared_ptr<GPUPipeline>>& pipelines,
                       PhysicalOperatorType type)
{
  size_t count = 0;
  for (const auto& pipeline : pipelines) {
    if (pipeline->GetSink()->type == type) { count++; }
  }
  return count;
}

/**
 * @brief Check if pipeline breakdown pattern matches expected:
 * Original sink (GROUP_BY, ORDER_BY, TOP_N, UNGROUPED_AGGREGATE) should be:
 * 1. Pipeline with PARTITION sink
 * 2. Dependent pipeline with CONCAT operator followed by original operation
 */
struct PipelineBreakdownInfo {
  bool has_partition_before_agg     = false;  // PARTITION sink exists
  bool has_concat_after_partition   = false;  // CONCAT in operators of dependent pipeline
  bool partition_connects_to_concat = false;  // PARTITION source connects to CONCAT
  size_t partition_count            = 0;
  size_t concat_count               = 0;
};

PipelineBreakdownInfo analyze_pipeline_breakdown(
  const duckdb::vector<duckdb::shared_ptr<GPUPipeline>>& pipelines)
{
  PipelineBreakdownInfo info;
  auto source_to_pipelines = build_source_to_pipelines_map(pipelines);

  for (const auto& pipeline : pipelines) {
    auto sink = pipeline->GetSink();

    // Count PARTITION sinks
    if (sink->type == PhysicalOperatorType::INVALID) {
      info.partition_count++;
      info.has_partition_before_agg = true;

      // Check if any dependent pipeline has CONCAT
      auto it = source_to_pipelines.find(sink.get());
      if (it != source_to_pipelines.end()) {
        for (auto& dep_pipeline : it->second) {
          auto ops = dep_pipeline->GetInnerOperators();
          for (auto& op : ops) {
            if (op.get().GetName() == "CONCAT") {
              info.has_concat_after_partition   = true;
              info.partition_connects_to_concat = true;
              info.concat_count++;
            }
          }
        }
      }
    }
  }

  return info;
}

/**
 * @brief Validate pipeline breakdown for GROUP_BY:
 * Expected: local_group → PARTITION → CONCAT → global_group
 */
void validate_groupby_breakdown(const duckdb::vector<duckdb::shared_ptr<GPUPipeline>>& pipelines,
                                const std::string& query_name)
{
  auto info = analyze_pipeline_breakdown(pipelines);

  INFO("Query: " << query_name);
  INFO("Partition count: " << info.partition_count);
  INFO("Concat count: " << info.concat_count);

  // For GROUP_BY queries, we expect:
  // - At least one PARTITION sink (replaces GROUP_BY sink)
  // - At least one CONCAT operator in dependent pipeline
  REQUIRE(info.has_partition_before_agg);
  REQUIRE(info.partition_connects_to_concat);
}

/**
 * @brief Information about HASH_JOIN pipeline breakdown
 */
struct HashJoinBreakdownInfo {
  // Build side info
  size_t build_partition_count        = 0;     // Partitions with isBuildPartition=true
  bool build_partition_has_right_port = true;  // All build partitions use "build" port

  // Probe side info
  size_t probe_partition_count          = 0;     // Partitions with isBuildPartition=false
  bool probe_partition_has_default_port = true;  // All probe partitions use "default" port

  // Join pipeline info
  bool partition_connects_to_join      = false;  // Partition source connects to HASH_JOIN operator
  size_t join_pipelines_from_partition = 0;  // Pipelines with PARTITION source containing HASH_JOIN

  // Port validation
  bool all_ports_have_valid_repos     = true;
  bool all_ports_have_valid_pipelines = true;
};

/**
 * @brief Analyze HASH_JOIN pipeline breakdown pattern
 *
 * Expected patterns:
 * - Build side: original HASH_JOIN sink replaced by PARTITION (isBuildPartition=true, port="build")
 * - Probe side: pipeline broken at HASH_JOIN operator:
 *   - Pipeline 1: ... → PARTITION (isBuildPartition=false, port="default")
 *   - Pipeline 2: PARTITION (source) → HASH_JOIN → ... → sink
 */
HashJoinBreakdownInfo analyze_hash_join_breakdown(
  const duckdb::vector<duckdb::shared_ptr<GPUPipeline>>& pipelines)
{
  HashJoinBreakdownInfo info;
  auto source_to_pipelines = build_source_to_pipelines_map(pipelines);

  for (const auto& pipeline : pipelines) {
    auto sink = pipeline->GetSink();

    // Check PARTITION sinks
    if (sink->type == PhysicalOperatorType::INVALID) {
      auto& partition = sink->Cast<GPUPhysicalPartition>();

      if (partition.isBuildPartition()) {
        // Build side partition
        info.build_partition_count++;

        // Verify port is "build"
        auto& next_ports = sink->get_next_port_after_sink();
        for (auto& [next_op, port_id] : next_ports) {
          if (std::string(port_id) != "build") { info.build_partition_has_right_port = false; }
        }
      } else {
        // Probe side partition
        info.probe_partition_count++;

        // Verify port is "default"
        auto& next_ports = sink->get_next_port_after_sink();
        for (auto& [next_op, port_id] : next_ports) {
          if (std::string(port_id) != "default") { info.probe_partition_has_default_port = false; }
        }
      }

      // Check if this partition connects to a pipeline with HASH_JOIN
      if (partition.isBuildPartition()) {
        // For build partitions: find pipeline where HASH_JOIN is the first operator
        GPUPhysicalOperator* hash_join_op = partition.getParentOp();
        for (const auto& dep_pipeline : pipelines) {
          auto inner_ops = dep_pipeline->GetInnerOperators();
          if (inner_ops.size() > 0 && &inner_ops[0].get() == hash_join_op) {
            info.partition_connects_to_join = true;
            info.join_pipelines_from_partition++;

            // Validate port on the HASH_JOIN
            auto* port = hash_join_op->get_port("build");
            if (port == nullptr || port->repo == nullptr) {
              info.all_ports_have_valid_repos = false;
            }
            if (port != nullptr) {
              if (port->src_pipeline != pipeline || port->dest_pipeline != dep_pipeline) {
                info.all_ports_have_valid_pipelines = false;
              }
            }
            break;
          }
        }
      } else {
        // For probe partitions: lookup by sink.get() (the partition itself)
        auto it = source_to_pipelines.find(sink.get());
        if (it != source_to_pipelines.end()) {
          for (auto& dep_pipeline : it->second) {
            // Check if dependent pipeline has HASH_JOIN in operators
            auto ops = dep_pipeline->GetInnerOperators();
            for (auto& op : ops) {
              if (op.get().type == PhysicalOperatorType::HASH_JOIN ||
                  op.get().type == PhysicalOperatorType::NESTED_LOOP_JOIN) {
                info.partition_connects_to_join = true;
                info.join_pipelines_from_partition++;
              }
            }

            // Validate port connections - next_op is first inner operator or sink
            auto inner_ops = dep_pipeline->GetInnerOperators();
            GPUPhysicalOperator* next_op =
              inner_ops.size() > 0 ? &inner_ops[0].get() : dep_pipeline->GetSink().get();

            auto* port = next_op->get_port("default");
            if (port == nullptr || port->repo == nullptr) {
              info.all_ports_have_valid_repos = false;
            }
            if (port != nullptr) {
              if (port->src_pipeline != pipeline || port->dest_pipeline != dep_pipeline) {
                info.all_ports_have_valid_pipelines = false;
              }
            }
          }
        }
      }
    }

    // Check for pipelines that have PARTITION source and contain HASH_JOIN
    if (pipeline->GetSource()->type == PhysicalOperatorType::INVALID) {
      auto ops = pipeline->GetInnerOperators();
      for (auto& op : ops) {
        if (op.get().type == PhysicalOperatorType::HASH_JOIN ||
            op.get().type == PhysicalOperatorType::NESTED_LOOP_JOIN) {
          // This is the continuation pipeline after partition break
          break;
        }
      }
    }
  }

  return info;
}

/**
 * @brief Validate HASH_JOIN pipeline modification
 *
 * Validates:
 * 1. Build side: HASH_JOIN sink replaced by PARTITION with "build" port
 * 2. Probe side: Pipeline broken with PARTITION ("default" port) → HASH_JOIN
 * 3. All ports have valid data repositories
 * 4. All ports have correct src_pipeline and dest_pipeline
 */
void validate_hash_join_modification(GPUExecutor& executor, const std::string& query_name = "")
{
  auto& pipelines = executor.new_scheduled;
  auto info       = analyze_hash_join_breakdown(pipelines);

  INFO("Query: " << query_name);
  INFO("Build partition count: " << info.build_partition_count);
  INFO("Probe partition count: " << info.probe_partition_count);
  INFO("Join pipelines from partition: " << info.join_pipelines_from_partition);

  // For a simple join query, we should have at least:
  // - 1 build partition (replaces HASH_JOIN sink)
  // - Build partition should use "build" port
  REQUIRE(info.build_partition_count >= 1);
  CHECK(info.build_partition_has_right_port);

  // Probe side partitions should use "default" port
  CHECK(info.probe_partition_has_default_port);

  // All ports should have valid repositories
  CHECK(info.all_ports_have_valid_repos);

  // All ports should have correct pipeline connections
  CHECK(info.all_ports_have_valid_pipelines);
}

/**
 * @brief Validate the complete modified pipeline structure
 */
void validate_modified_pipeline_structure(GPUExecutor& executor, const std::string& query_name = "")
{
  auto& new_scheduled = executor.new_scheduled;
  INFO("Query: " << query_name);
  INFO("Pipeline count: " << new_scheduled.size());
  REQUIRE(new_scheduled.size() > 0);

  // Build source -> pipelines map
  auto source_to_pipelines = build_source_to_pipelines_map(new_scheduled);

  for (size_t i = 0; i < new_scheduled.size(); i++) {
    auto& pipeline = new_scheduled[i];
    auto sink      = pipeline->GetSink();
    auto source    = pipeline->GetSource();

    REQUIRE(sink.get() != nullptr);
    REQUIRE(source.get() != nullptr);

    std::string pipeline_context = "Pipeline " + std::to_string(i) +
                                   " (source=" + source->GetName() + ", sink=" + sink->GetName() +
                                   ")";

    // Validate based on sink type
    if (sink->type == PhysicalOperatorType::INVALID) {
      // This is a PARTITION operator
      auto& partition     = sink->Cast<GPUPhysicalPartition>();
      std::string port_id = partition.isBuildPartition() ? "build" : "default";

      if (partition.isBuildPartition()) {
        // For build partitions: find pipeline where HASH_JOIN is the first operator
        GPUPhysicalOperator* hash_join_op = partition.getParentOp();
        for (const auto& dep_pipeline : new_scheduled) {
          auto inner_ops = dep_pipeline->GetInnerOperators();
          if (inner_ops.size() > 0 && &inner_ops[0].get() == hash_join_op) {
            auto* port          = hash_join_op->get_port(port_id);
            std::string context = pipeline_context + " -> PARTITION with port '" + port_id + "'";
            validate_port_repository(port, context);

            INFO("PARTITION port validation: " << context);
            REQUIRE(port->src_pipeline == pipeline);
            REQUIRE(port->dest_pipeline == dep_pipeline);
            break;
          }
        }
      } else {
        // For probe partitions: lookup by sink.get() (the partition itself)
        auto it = source_to_pipelines.find(sink.get());
        if (it != source_to_pipelines.end()) {
          for (auto& dep_pipeline : it->second) {
            // next_op is first inner operator or sink
            auto inner_ops = dep_pipeline->GetInnerOperators();
            GPUPhysicalOperator* next_op =
              inner_ops.size() > 0 ? &inner_ops[0].get() : dep_pipeline->GetSink().get();

            auto* port          = next_op->get_port(port_id);
            std::string context = pipeline_context + " -> PARTITION with port '" + port_id + "'";
            validate_port_repository(port, context);

            INFO("PARTITION port validation: " << context);
            REQUIRE(port->src_pipeline == pipeline);
            REQUIRE(port->dest_pipeline == dep_pipeline);
          }
        }
      }

      // Validate next_port_after_sink connections
      auto& next_ports = sink->get_next_port_after_sink();
      for (auto& [next_op, next_port_id] : next_ports) {
        REQUIRE(next_op != nullptr);
        REQUIRE(std::string(next_port_id) == port_id);
      }

    } else if (sink->type == PhysicalOperatorType::HASH_GROUP_BY ||
               sink->type == PhysicalOperatorType::ORDER_BY ||
               sink->type == PhysicalOperatorType::TOP_N ||
               sink->type == PhysicalOperatorType::UNGROUPED_AGGREGATE) {
      // These should have "default" port
      auto it = source_to_pipelines.find(sink.get());
      if (it != source_to_pipelines.end()) {
        for (auto& dep_pipeline : it->second) {
          // next_op is first inner operator or sink
          auto inner_ops = dep_pipeline->GetInnerOperators();
          GPUPhysicalOperator* next_op =
            inner_ops.size() > 0 ? &inner_ops[0].get() : dep_pipeline->GetSink().get();

          auto* port = next_op->get_port("default");
          std::string context =
            pipeline_context + " -> " + sink->GetName() + " with port 'default'";
          validate_port_repository(port, context);

          // Validate pipeline connections
          INFO("Aggregation sink port validation: " << context);
          REQUIRE(port->src_pipeline == pipeline);
          REQUIRE(port->dest_pipeline == dep_pipeline);
        }
      }

    } else if (sink->type == PhysicalOperatorType::RESULT_COLLECTOR) {
      // Should have "final" port
      auto* port          = sink->get_port("final");
      std::string context = pipeline_context + " -> RESULT_COLLECTOR with port 'final'";
      validate_port_repository(port, context);

      // src_pipeline should be this pipeline, dest_pipeline is nullptr
      INFO("RESULT_COLLECTOR port validation: " << context);
      REQUIRE(port->src_pipeline == pipeline);
      REQUIRE(port->dest_pipeline == nullptr);

    } else if (sink->type == PhysicalOperatorType::RIGHT_DELIM_JOIN) {
      // Validate partition_join and partition_distinct ports
      auto& delim_join         = sink->Cast<GPUPhysicalDelimJoin>();
      auto* partition_join     = delim_join.partition_join;
      auto* partition_distinct = delim_join.partition_distinct;

      REQUIRE(partition_join != nullptr);
      REQUIRE(partition_distinct != nullptr);

      // partition_join should use "build" port - find pipeline where join is first operator
      GPUPhysicalOperator* join_op = partition_join->getParentOp();
      for (const auto& dep_pipeline : new_scheduled) {
        auto inner_ops = dep_pipeline->GetInnerOperators();
        if (inner_ops.size() > 0 && &inner_ops[0].get() == join_op) {
          auto* port = join_op->get_port("build");
          std::string context =
            pipeline_context + " -> DELIM_JOIN partition_join with port 'build'";
          validate_port_repository(port, context);
          break;
        }
      }

      // partition_distinct should use "default" port
      auto it_distinct = source_to_pipelines.find(partition_distinct);
      if (it_distinct != source_to_pipelines.end()) {
        for (auto& dep_pipeline : it_distinct->second) {
          // next_op is first inner operator or sink
          auto inner_ops = dep_pipeline->GetInnerOperators();
          GPUPhysicalOperator* next_op =
            inner_ops.size() > 0 ? &inner_ops[0].get() : dep_pipeline->GetSink().get();

          auto* port = next_op->get_port("default");
          std::string context =
            pipeline_context + " -> DELIM_JOIN partition_distinct with port 'default'";
          validate_port_repository(port, context);
        }
      }

    } else if (sink->type == PhysicalOperatorType::LEFT_DELIM_JOIN) {
      // Validate partition_distinct and column_data_scan ports
      auto& delim_join         = sink->Cast<GPUPhysicalDelimJoin>();
      auto* partition_distinct = delim_join.partition_distinct;
      auto* column_data_scan   = delim_join.join->children[0].get();

      REQUIRE(partition_distinct != nullptr);
      REQUIRE(column_data_scan != nullptr);

      // partition_distinct should use "default" port
      auto it_distinct = source_to_pipelines.find(partition_distinct);
      if (it_distinct != source_to_pipelines.end()) {
        for (auto& dep_pipeline : it_distinct->second) {
          // next_op is first inner operator or sink
          auto inner_ops = dep_pipeline->GetInnerOperators();
          GPUPhysicalOperator* next_op =
            inner_ops.size() > 0 ? &inner_ops[0].get() : dep_pipeline->GetSink().get();

          auto* port = next_op->get_port("default");
          std::string context =
            pipeline_context + " -> LEFT_DELIM_JOIN partition_distinct with port 'default'";
          validate_port_repository(port, context);
        }
      }

      // column_data_scan should use "default" port
      auto it_scan = source_to_pipelines.find(column_data_scan);
      if (it_scan != source_to_pipelines.end()) {
        for (auto& dep_pipeline : it_scan->second) {
          // next_op is first inner operator or sink
          auto inner_ops = dep_pipeline->GetInnerOperators();
          GPUPhysicalOperator* next_op =
            inner_ops.size() > 0 ? &inner_ops[0].get() : dep_pipeline->GetSink().get();

          auto* port = next_op->get_port("default");
          std::string context =
            pipeline_context + " -> LEFT_DELIM_JOIN column_data_scan with port 'default'";
          validate_port_repository(port, context);
        }
      }

    } else if (sink->type == PhysicalOperatorType::CTE) {
      // CTE should have "default" port connections to CTE scans
      auto& cte_op = sink->Cast<GPUPhysicalCTE>();
      for (auto& cte_scan_ref : cte_op.cte_scans) {
        auto& cte_scan = cte_scan_ref.get();
        auto it        = source_to_pipelines.find(&cte_scan);
        if (it != source_to_pipelines.end()) {
          for (auto& dep_pipeline : it->second) {
            // next_op is first inner operator or sink
            auto inner_ops = dep_pipeline->GetInnerOperators();
            GPUPhysicalOperator* next_op =
              inner_ops.size() > 0 ? &inner_ops[0].get() : dep_pipeline->GetSink().get();

            auto* port          = next_op->get_port("default");
            std::string context = pipeline_context + " -> CTE with port 'default'";
            validate_port_repository(port, context);

            INFO("CTE port validation: " << context);
            REQUIRE(port->src_pipeline == pipeline);
            REQUIRE(port->dest_pipeline == dep_pipeline);
          }
        }
      }
    }

    // Validate TABLE_SCAN source ports
    if (source->type == PhysicalOperatorType::TABLE_SCAN) {
      // next_op is first inner operator or sink
      auto inner_ops = pipeline->GetInnerOperators();
      GPUPhysicalOperator* next_op =
        inner_ops.size() > 0 ? &inner_ops[0].get() : pipeline->GetSink().get();

      auto* port          = next_op->get_port("scan");
      std::string context = pipeline_context + " -> TABLE_SCAN source with port 'scan'";
      validate_port_repository(port, context);

      // For scan port, src_pipeline is nullptr (scan is the source)
      INFO("TABLE_SCAN port validation: " << context);
      REQUIRE(port->src_pipeline == nullptr);
      REQUIRE(port->dest_pipeline == pipeline);
    }
  }
}

/**
 * @brief Helper to run a TPC-H query test
 */
void run_tpch_query_test(Connection& con, const std::string& query, const std::string& query_name)
{
  INFO("Testing: " << query_name);

  // Generate GPU plan
  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  // Initialize executor
  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  // Validate pipeline structure
  validate_modified_pipeline_structure(executor, query_name);

  // All queries should have at least one PARTITION (for aggregation or join)
  INFO("Validating PARTITION presence for: " << query_name);
  REQUIRE(count_partition_sinks(executor.new_scheduled) >= 1);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pipeline Breakdown Pattern Tests
//===----------------------------------------------------------------------===//

TEST_CASE("Pipeline breakdown - GROUP_BY pattern", "[modified_pipeline][breakdown]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // Simple GROUP_BY query
  std::string query = R"(
    SELECT l_returnflag, SUM(l_quantity) as total
    FROM lineitem
    GROUP BY l_returnflag
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  // Validate breakdown: should have PARTITION → CONCAT → GROUP_BY
  auto info = analyze_pipeline_breakdown(executor.new_scheduled);
  REQUIRE(info.has_partition_before_agg);
  REQUIRE(info.has_concat_after_partition);
  REQUIRE(info.partition_connects_to_concat);

  validate_modified_pipeline_structure(executor, "GROUP_BY pattern");

  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("Pipeline breakdown - ORDER_BY pattern", "[modified_pipeline][breakdown]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // GROUP_BY + ORDER_BY query
  std::string query = R"(
    SELECT l_returnflag, SUM(l_quantity) as total
    FROM lineitem
    GROUP BY l_returnflag
    ORDER BY total DESC
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  // Validate breakdown
  auto info = analyze_pipeline_breakdown(executor.new_scheduled);
  REQUIRE(info.has_partition_before_agg);

  validate_modified_pipeline_structure(executor, "ORDER_BY pattern");

  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("Pipeline breakdown - TOP_N pattern", "[modified_pipeline][breakdown]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // TOP_N query (ORDER BY + LIMIT)
  std::string query = R"(
    SELECT l_orderkey, l_extendedprice
    FROM lineitem
    ORDER BY l_extendedprice DESC
    LIMIT 10
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  // TOP_N should have PARTITION
  REQUIRE(count_partition_sinks(executor.new_scheduled) >= 1);
  validate_modified_pipeline_structure(executor, "TOP_N pattern");

  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("Pipeline breakdown - UNGROUPED_AGGREGATE pattern", "[modified_pipeline][breakdown]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // Ungrouped aggregation
  std::string query = R"(
    SELECT SUM(l_extendedprice * l_discount) as revenue
    FROM lineitem
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  // UNGROUPED_AGGREGATE should have PARTITION
  REQUIRE(count_partition_sinks(executor.new_scheduled) >= 1);
  validate_modified_pipeline_structure(executor, "UNGROUPED_AGGREGATE pattern");

  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("Pipeline breakdown - HASH_JOIN pattern", "[modified_pipeline][breakdown][join]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // Join query
  std::string query = R"(
    SELECT o.o_orderkey
    FROM orders o, customer c
    WHERE o.o_custkey = c.c_custkey
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  // Validate HASH_JOIN modification pattern
  validate_hash_join_modification(executor, "HASH_JOIN pattern");

  // Also validate general pipeline structure
  validate_modified_pipeline_structure(executor, "HASH_JOIN pattern");

  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("Pipeline breakdown - HASH_JOIN build side validation",
          "[modified_pipeline][breakdown][join]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // Simple join to clearly see build side
  std::string query = R"(
    SELECT o.o_totalprice
    FROM customer c, orders o
    WHERE c.c_custkey = o.o_custkey
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  auto info = analyze_hash_join_breakdown(executor.new_scheduled);

  // Build side should be replaced by PARTITION with "build" port
  INFO("Build partition count: " << info.build_partition_count);
  REQUIRE(info.build_partition_count >= 1);
  REQUIRE(info.build_partition_has_right_port);

  // Verify the build partition connects to a pipeline where HASH_JOIN is the first operator.
  // Build partitions are NOT used as sources - instead, we find pipelines where the
  // HASH_JOIN (stored in partition.getParentOp()) is the first operator.
  bool found_build_partition_with_port = false;
  for (const auto& pipeline : executor.new_scheduled) {
    auto sink = pipeline->GetSink();
    if (sink->type == PhysicalOperatorType::INVALID) {
      auto& partition = sink->Cast<GPUPhysicalPartition>();
      if (partition.isBuildPartition()) {
        // The partition's parent_op is the HASH_JOIN
        GPUPhysicalOperator* hash_join_op = partition.getParentOp();
        REQUIRE(hash_join_op != nullptr);
        REQUIRE(hash_join_op->type == PhysicalOperatorType::HASH_JOIN);

        // Find the pipeline where HASH_JOIN is the first operator
        for (const auto& dep_pipeline : executor.new_scheduled) {
          auto inner_ops = dep_pipeline->GetInnerOperators();
          if (inner_ops.size() > 0 && &inner_ops[0].get() == hash_join_op) {
            // The port should be on the HASH_JOIN (the first operator)
            auto* port = hash_join_op->get_port("build");
            if (port != nullptr) {
              found_build_partition_with_port = true;
              INFO("Build partition should have 'build' port connecting to dep_pipeline");
              REQUIRE(port->repo != nullptr);
              REQUIRE(port->src_pipeline == pipeline);
              REQUIRE(port->dest_pipeline == dep_pipeline);
            }
            break;
          }
        }
      }
    }
  }

  INFO("Should find build partition with 'build' port connected properly");
  REQUIRE(found_build_partition_with_port);

  validate_modified_pipeline_structure(executor, "HASH_JOIN build side");

  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("Pipeline breakdown - HASH_JOIN probe side validation",
          "[modified_pipeline][breakdown][join]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // Query with multiple joins to test probe side breakdown
  std::string query = R"(
    SELECT l.l_orderkey, o.o_orderdate
    FROM lineitem l, orders o, customer c
    WHERE l.l_orderkey = o.o_orderkey
      AND o.o_custkey = c.c_custkey
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  auto info                = analyze_hash_join_breakdown(executor.new_scheduled);
  auto source_to_pipelines = build_source_to_pipelines_map(executor.new_scheduled);

  // For probe side: pipeline should be broken with PARTITION → HASH_JOIN
  // Probe partition should use "default" port
  INFO("Probe partition count: " << info.probe_partition_count);

  // When there are joins in the pipeline operators, they get broken
  // Verify probe partitions (isBuildPartition=false) connect to HASH_JOIN
  bool found_join_after_probe_partition = false;
  for (const auto& pipeline : executor.new_scheduled) {
    auto sink = pipeline->GetSink();
    if (sink->type == PhysicalOperatorType::INVALID) {
      auto& partition = sink->Cast<GPUPhysicalPartition>();
      if (!partition.isBuildPartition()) {
        // This is a probe partition - check it uses "default" port
        auto& next_ports = sink->get_next_port_after_sink();
        for (auto& [next_op, port_id] : next_ports) {
          INFO("Probe partition port: " << port_id);
          REQUIRE(std::string(port_id) == "default");
        }

        // Check if dependent pipeline has HASH_JOIN
        auto it = source_to_pipelines.find(sink.get());
        if (it != source_to_pipelines.end()) {
          for (auto& dep_pipeline : it->second) {
            auto ops = dep_pipeline->GetInnerOperators();
            for (auto& op : ops) {
              if (op.get().type == PhysicalOperatorType::HASH_JOIN) {
                found_join_after_probe_partition = true;

                // Verify the HASH_JOIN has the "default" port for probe
                auto* port = op.get().get_port("default");
                if (port != nullptr) {
                  INFO("Probe partition's dependent HASH_JOIN should have 'default' port");
                  REQUIRE(port->repo != nullptr);
                  REQUIRE(port->src_pipeline == pipeline);
                }
              }
            }
          }
        }
      }
    }
  }

  // With multiple joins, we should see probe side partitions
  if (info.probe_partition_count > 0) { CHECK(info.probe_partition_has_default_port); }

  validate_modified_pipeline_structure(executor, "HASH_JOIN probe side");

  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("Pipeline breakdown - Multi-way HASH_JOIN", "[modified_pipeline][breakdown][join]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // 4-way join to test complex join scenarios
  std::string query = R"(
    SELECT o.o_totalprice, l.l_quantity
    FROM nation n, customer c, orders o, lineitem l
    WHERE n.n_nationkey = c.c_nationkey
      AND c.c_custkey = o.o_custkey
      AND o.o_orderkey = l.l_orderkey
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  auto info = analyze_hash_join_breakdown(executor.new_scheduled);

  INFO("Multi-way join - Build partitions: " << info.build_partition_count);
  INFO("Multi-way join - Probe partitions: " << info.probe_partition_count);
  INFO("Multi-way join - Total pipelines: " << executor.new_scheduled.size());

  // With multiple joins, we should have multiple build partitions
  // Each join's build side becomes a PARTITION
  REQUIRE(info.build_partition_count >= 1);  // At least one per join
  REQUIRE(info.build_partition_has_right_port);

  // All ports should be properly connected
  REQUIRE(info.all_ports_have_valid_repos);
  REQUIRE(info.all_ports_have_valid_pipelines);

  validate_hash_join_modification(executor, "Multi-way HASH_JOIN");
  validate_modified_pipeline_structure(executor, "Multi-way HASH_JOIN");

  Config::MODIFIED_PIPELINE = false;
}

//===----------------------------------------------------------------------===//
// CTE Test
//===----------------------------------------------------------------------===//

TEST_CASE("Pipeline breakdown - CTE pattern", "[modified_pipeline][cte]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  // TPC-H Q15 uses CTE (WITH clause)
  std::string query = R"(
    WITH revenue_view AS (
      SELECT
        l_suppkey as supplier_no,
        SUM(l_extendedprice * (1 - l_discount)) as total_revenue
      FROM lineitem
      WHERE l_shipdate >= DATE '1993-05-01'
        AND l_shipdate < DATE '1993-05-01' + INTERVAL '3' MONTH
      GROUP BY l_suppkey
    )
    SELECT
      s.s_suppkey,
      s.s_name,
      r.total_revenue
    FROM supplier s, revenue_view r
    WHERE s.s_suppkey = r.supplier_no
      AND r.total_revenue = (
        SELECT MAX(total_revenue) FROM revenue_view
      )
    ORDER BY s.s_suppkey
  )";

  GPUContext gpu_context(*con.context);
  auto gpu_plan = generate_gpu_plan(con, gpu_context, query);
  REQUIRE(gpu_plan != nullptr);

  GPUExecutor executor(*con.context, gpu_context);
  executor.Initialize(std::move(gpu_plan));

  // CTE queries should have proper pipeline structure
  REQUIRE(executor.new_scheduled.size() > 0);

  // Check for CTE sink
  bool has_cte = false;
  for (const auto& pipeline : executor.new_scheduled) {
    if (pipeline->GetSink()->type == PhysicalOperatorType::CTE) {
      has_cte = true;
      break;
    }
  }
  INFO("CTE query should have CTE operator");
  REQUIRE(has_cte);

  validate_modified_pipeline_structure(executor, "CTE pattern (Q15)");

  Config::MODIFIED_PIPELINE = false;
}

//===----------------------------------------------------------------------===//
// All TPC-H Query Tests
//===----------------------------------------------------------------------===//

TEST_CASE("TPC-H Q1 - Pricing Summary Report", "[modified_pipeline][tpch][q1]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT
      l_returnflag,
      l_linestatus,
      SUM(l_quantity) as sum_qty,
      SUM(l_extendedprice) as sum_base_price,
      SUM(l_extendedprice * (1 - l_discount)) as sum_disc_price,
      SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
      AVG(l_quantity) as avg_qty,
      AVG(l_extendedprice) as avg_price,
      AVG(l_discount) as avg_disc,
      COUNT(*) as count_order
    FROM lineitem
    WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '1200' DAY
    GROUP BY l_returnflag, l_linestatus
    ORDER BY l_returnflag, l_linestatus
  )";

  run_tpch_query_test(con, query, "Q1");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q2 - Minimum Cost Supplier", "[modified_pipeline][tpch][q2]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT
      s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr,
      s.s_address, s.s_phone, s.s_comment
    FROM part p, supplier s, partsupp ps, nation n, region r
    WHERE p.p_partkey = ps.ps_partkey
      AND s.s_suppkey = ps.ps_suppkey
      AND p.p_size = 10
      AND p.p_type LIKE '%COPPER'
      AND s.s_nationkey = n.n_nationkey
      AND n.n_regionkey = r.r_regionkey
      AND r.r_name = 'EUROPE'
      AND ps.ps_supplycost = (
        SELECT MIN(ps2.ps_supplycost)
        FROM partsupp ps2, supplier s2, nation n2, region r2
        WHERE p.p_partkey = ps2.ps_partkey
          AND s2.s_suppkey = ps2.ps_suppkey
          AND s2.s_nationkey = n2.n_nationkey
          AND n2.n_regionkey = r2.r_regionkey
          AND r2.r_name = 'EUROPE'
      )
    ORDER BY s.s_acctbal DESC, n.n_name, s.s_name, p.p_partkey
    LIMIT 100
  )";

  run_tpch_query_test(con, query, "Q2");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q3 - Shipping Priority", "[modified_pipeline][tpch][q3]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT
      l.l_orderkey,
      SUM(l.l_extendedprice * (1 - l.l_discount)) as revenue,
      o.o_orderdate,
      o.o_shippriority
    FROM customer c, orders o, lineitem l
    WHERE c.c_mktsegment = 'HOUSEHOLD'
      AND c.c_custkey = o.o_custkey
      AND l.l_orderkey = o.o_orderkey
      AND o.o_orderdate < DATE '1995-03-25'
      AND l.l_shipdate > DATE '1995-03-25'
    GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority
    ORDER BY revenue DESC, o.o_orderdate
    LIMIT 10
  )";

  run_tpch_query_test(con, query, "Q3");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q4 - Order Priority Checking", "[modified_pipeline][tpch][q4]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT o.o_orderpriority, COUNT(*) as order_count
    FROM orders o
    WHERE o.o_orderdate >= DATE '1996-10-01'
      AND o.o_orderdate < DATE '1996-10-01' + INTERVAL '3' MONTH
      AND EXISTS (
        SELECT * FROM lineitem l
        WHERE l.l_orderkey = o.o_orderkey
          AND l.l_commitdate < l.l_receiptdate
      )
    GROUP BY o.o_orderpriority
    ORDER BY o.o_orderpriority
  )";

  run_tpch_query_test(con, query, "Q4");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q5 - Local Supplier Volume", "[modified_pipeline][tpch][q5]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) as revenue
    FROM orders o, lineitem l, supplier s, nation n, region r, customer c
    WHERE c.c_custkey = o.o_custkey
      AND l.l_orderkey = o.o_orderkey
      AND l.l_suppkey = s.s_suppkey
      AND c.c_nationkey = s.s_nationkey
      AND s.s_nationkey = n.n_nationkey
      AND n.n_regionkey = r.r_regionkey
      AND r.r_name = 'EUROPE'
      AND o.o_orderdate >= DATE '1997-01-01'
      AND o.o_orderdate < DATE '1997-01-01' + INTERVAL '1' YEAR
    GROUP BY n.n_name
    ORDER BY revenue DESC
  )";

  run_tpch_query_test(con, query, "Q5");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q6 - Forecasting Revenue Change", "[modified_pipeline][tpch][q6]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT SUM(l_extendedprice * l_discount) as revenue
    FROM lineitem
    WHERE l_shipdate >= DATE '1997-01-01'
      AND l_shipdate < DATE '1997-01-01' + INTERVAL '1' YEAR
      AND l_discount BETWEEN 0.03 - 0.01 AND 0.03 + 0.01
      AND l_quantity < 24
  )";

  run_tpch_query_test(con, query, "Q6");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q7 - Volume Shipping", "[modified_pipeline][tpch][q7]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT supp_nation, cust_nation, l_year, SUM(volume) as revenue
    FROM (
      SELECT
        n1.n_name as supp_nation,
        n2.n_name as cust_nation,
        EXTRACT(YEAR FROM l.l_shipdate) as l_year,
        l.l_extendedprice * (1 - l.l_discount) as volume
      FROM supplier s, lineitem l, orders o, customer c, nation n1, nation n2
      WHERE s.s_suppkey = l.l_suppkey
        AND o.o_orderkey = l.l_orderkey
        AND c.c_custkey = o.o_custkey
        AND s.s_nationkey = n1.n_nationkey
        AND c.c_nationkey = n2.n_nationkey
        AND (
          (n1.n_name = 'EGYPT' AND n2.n_name = 'UNITED STATES')
          OR (n1.n_name = 'UNITED STATES' AND n2.n_name = 'EGYPT')
        )
        AND l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
    ) as shipping
    GROUP BY supp_nation, cust_nation, l_year
    ORDER BY supp_nation, cust_nation, l_year
  )";

  run_tpch_query_test(con, query, "Q7");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q8 - National Market Share", "[modified_pipeline][tpch][q8]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT o_year,
      SUM(CASE WHEN nation = 'EGYPT' THEN volume ELSE 0 END) / SUM(volume) as mkt_share
    FROM (
      SELECT
        EXTRACT(YEAR FROM o.o_orderdate) as o_year,
        l.l_extendedprice * (1 - l.l_discount) as volume,
        n2.n_name as nation
      FROM lineitem l, part p, supplier s, orders o, customer c, nation n1, nation n2, region r
      WHERE p.p_partkey = l.l_partkey
        AND s.s_suppkey = l.l_suppkey
        AND l.l_orderkey = o.o_orderkey
        AND o.o_custkey = c.c_custkey
        AND c.c_nationkey = n1.n_nationkey
        AND n1.n_regionkey = r.r_regionkey
        AND r.r_name = 'MIDDLE EAST'
        AND s.s_nationkey = n2.n_nationkey
        AND o.o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
        AND p.p_type = 'PROMO BRUSHED COPPER'
    ) as all_nations
    GROUP BY o_year
    ORDER BY o_year
  )";

  run_tpch_query_test(con, query, "Q8");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q9 - Product Type Profit Measure", "[modified_pipeline][tpch][q9]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT nation, o_year, SUM(amount) as sum_profit
    FROM (
      SELECT
        n.n_name as nation,
        EXTRACT(YEAR FROM o.o_orderdate) as o_year,
        l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity as amount
      FROM part p, supplier s, lineitem l, partsupp ps, orders o, nation n
      WHERE s.s_suppkey = l.l_suppkey
        AND ps.ps_suppkey = l.l_suppkey
        AND ps.ps_partkey = l.l_partkey
        AND p.p_partkey = l.l_partkey
        AND o.o_orderkey = l.l_orderkey
        AND s.s_nationkey = n.n_nationkey
        AND p.p_name LIKE '%yellow%'
    ) as profit
    GROUP BY nation, o_year
    ORDER BY nation, o_year DESC
  )";

  run_tpch_query_test(con, query, "Q9");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q10 - Returned Item Reporting", "[modified_pipeline][tpch][q10]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT
      c.c_custkey, c.c_name,
      SUM(l.l_extendedprice * (1 - l.l_discount)) as revenue,
      c.c_acctbal, n.n_name, c.c_address, c.c_phone, c.c_comment
    FROM customer c, orders o, lineitem l, nation n
    WHERE c.c_custkey = o.o_custkey
      AND l.l_orderkey = o.o_orderkey
      AND o.o_orderdate >= DATE '1994-03-01'
      AND o.o_orderdate < DATE '1994-03-01' + INTERVAL '3' MONTH
      AND l.l_returnflag = 'R'
      AND c.c_nationkey = n.n_nationkey
    GROUP BY c.c_custkey, c.c_name, c.c_acctbal, c.c_phone, n.n_name, c.c_address, c.c_comment
    ORDER BY revenue DESC
    LIMIT 20
  )";

  run_tpch_query_test(con, query, "Q10");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q11 - Important Stock Identification", "[modified_pipeline][tpch][q11]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT ps.ps_partkey, SUM(ps.ps_supplycost * ps.ps_availqty) as value
    FROM partsupp ps, supplier s, nation n
    WHERE ps.ps_suppkey = s.s_suppkey
      AND s.s_nationkey = n.n_nationkey
      AND n.n_name = 'JAPAN'
    GROUP BY ps.ps_partkey
    HAVING SUM(ps.ps_supplycost * ps.ps_availqty) > (
      SELECT SUM(ps2.ps_supplycost * ps2.ps_availqty) * 0.0001
      FROM partsupp ps2, supplier s2, nation n2
      WHERE ps2.ps_suppkey = s2.s_suppkey
        AND s2.s_nationkey = n2.n_nationkey
        AND n2.n_name = 'JAPAN'
    )
    ORDER BY value DESC
  )";

  run_tpch_query_test(con, query, "Q11");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q12 - Shipping Modes and Order Priority", "[modified_pipeline][tpch][q12]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT
      l.l_shipmode,
      SUM(CASE WHEN o.o_orderpriority = '1-URGENT' OR o.o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) as high_line_count,
      SUM(CASE WHEN o.o_orderpriority <> '1-URGENT' AND o.o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) as low_line_count
    FROM orders o, lineitem l
    WHERE o.o_orderkey = l.l_orderkey
      AND l.l_shipmode IN ('TRUCK', 'REG AIR')
      AND l.l_commitdate < l.l_receiptdate
      AND l.l_shipdate < l.l_commitdate
      AND l.l_receiptdate >= DATE '1994-01-01'
      AND l.l_receiptdate < DATE '1994-01-01' + INTERVAL '1' YEAR
    GROUP BY l.l_shipmode
    ORDER BY l.l_shipmode
  )";

  run_tpch_query_test(con, query, "Q12");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q13 - Customer Distribution", "[modified_pipeline][tpch][q13]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT c_count, COUNT(*) as custdist
    FROM (
      SELECT c.c_custkey, COUNT(o.o_orderkey)
      FROM customer c LEFT OUTER JOIN orders o
        ON c.c_custkey = o.o_custkey
        AND o.o_comment NOT LIKE '%special%requests%'
      GROUP BY c.c_custkey
    ) as orders (c_custkey, c_count)
    GROUP BY c_count
    ORDER BY custdist DESC, c_count DESC
  )";

  run_tpch_query_test(con, query, "Q13");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q14 - Promotion Effect", "[modified_pipeline][tpch][q14]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT
      100.00 * SUM(CASE WHEN p.p_type LIKE 'PROMO%' THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END)
      / SUM(l.l_extendedprice * (1 - l.l_discount)) as promo_revenue
    FROM lineitem l, part p
    WHERE l.l_partkey = p.p_partkey
      AND l.l_shipdate >= DATE '1994-08-01'
      AND l.l_shipdate < DATE '1994-08-01' + INTERVAL '1' MONTH
  )";

  run_tpch_query_test(con, query, "Q14");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q15 - Top Supplier (CTE)", "[modified_pipeline][tpch][q15][cte]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    WITH revenue_view AS (
      SELECT l_suppkey as supplier_no, SUM(l_extendedprice * (1 - l_discount)) as total_revenue
      FROM lineitem
      WHERE l_shipdate >= DATE '1993-05-01'
        AND l_shipdate < DATE '1993-05-01' + INTERVAL '3' MONTH
      GROUP BY l_suppkey
    )
    SELECT s.s_suppkey, s.s_name, s.s_address, s.s_phone, r.total_revenue
    FROM supplier s, revenue_view r
    WHERE s.s_suppkey = r.supplier_no
      AND r.total_revenue = (SELECT MAX(total_revenue) FROM revenue_view)
    ORDER BY s.s_suppkey
  )";

  run_tpch_query_test(con, query, "Q15");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q16 - Parts/Supplier Relationship", "[modified_pipeline][tpch][q16]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT p.p_brand, p.p_type, p.p_size, COUNT(DISTINCT ps.ps_suppkey) as supplier_cnt
    FROM partsupp ps, part p
    WHERE p.p_partkey = ps.ps_partkey
      AND p.p_brand <> 'Brand#21'
      AND p.p_type NOT LIKE 'MEDIUM PLATED%'
      AND p.p_size IN (10, 20, 30)
      AND ps.ps_suppkey NOT IN (
        SELECT s.s_suppkey FROM supplier s WHERE s.s_comment LIKE '%Customer%Complaints%'
      )
    GROUP BY p.p_brand, p.p_type, p.p_size
    ORDER BY supplier_cnt DESC, p.p_brand, p.p_type, p.p_size
  )";

  run_tpch_query_test(con, query, "Q16");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q17 - Small-Quantity-Order Revenue", "[modified_pipeline][tpch][q17]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT SUM(l.l_extendedprice) / 7.0 as avg_yearly
    FROM lineitem l, part p
    WHERE p.p_partkey = l.l_partkey
      AND p.p_brand = 'Brand#13'
      AND p.p_container = 'JUMBO CAN'
      AND l.l_quantity < (
        SELECT 0.2 * AVG(l2.l_quantity)
        FROM lineitem l2
        WHERE l2.l_partkey = p.p_partkey
      )
  )";

  run_tpch_query_test(con, query, "Q17");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q18 - Large Volume Customer", "[modified_pipeline][tpch][q18]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice, SUM(l.l_quantity)
    FROM customer c, orders o, lineitem l
    WHERE o.o_orderkey IN (
      SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300
    )
      AND c.c_custkey = o.o_custkey
      AND o.o_orderkey = l.l_orderkey
    GROUP BY c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice
    ORDER BY o.o_totalprice DESC, o.o_orderdate
    LIMIT 100
  )";

  run_tpch_query_test(con, query, "Q18");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q19 - Discounted Revenue", "[modified_pipeline][tpch][q19]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT SUM(l.l_extendedprice * (1 - l.l_discount)) as revenue
    FROM lineitem l, part p
    WHERE (
        p.p_partkey = l.l_partkey
        AND p.p_brand = 'Brand#41'
        AND p.p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        AND l.l_quantity >= 2 AND l.l_quantity <= 12
        AND p.p_size BETWEEN 1 AND 5
        AND l.l_shipmode IN ('AIR', 'AIR REG')
        AND l.l_shipinstruct = 'DELIVER IN PERSON'
      ) OR (
        p.p_partkey = l.l_partkey
        AND p.p_brand = 'Brand#13'
        AND p.p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        AND l.l_quantity >= 14 AND l.l_quantity <= 24
        AND p.p_size BETWEEN 1 AND 10
        AND l.l_shipmode IN ('AIR', 'AIR REG')
        AND l.l_shipinstruct = 'DELIVER IN PERSON'
      ) OR (
        p.p_partkey = l.l_partkey
        AND p.p_brand = 'Brand#55'
        AND p.p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        AND l.l_quantity >= 23 AND l.l_quantity <= 33
        AND p.p_size BETWEEN 1 AND 15
        AND l.l_shipmode IN ('AIR', 'AIR REG')
        AND l.l_shipinstruct = 'DELIVER IN PERSON'
      )
  )";

  run_tpch_query_test(con, query, "Q19");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q20 - Potential Part Promotion", "[modified_pipeline][tpch][q20]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT s.s_name, s.s_address
    FROM supplier s, nation n
    WHERE s.s_suppkey IN (
      SELECT ps.ps_suppkey
      FROM partsupp ps
      WHERE ps.ps_partkey IN (
        SELECT p.p_partkey FROM part p WHERE p.p_name LIKE 'antique%'
      )
      AND ps.ps_availqty > (
        SELECT 0.5 * SUM(l.l_quantity)
        FROM lineitem l
        WHERE l.l_partkey = ps.ps_partkey
          AND l.l_suppkey = ps.ps_suppkey
          AND l.l_shipdate >= DATE '1993-01-01'
          AND l.l_shipdate < DATE '1993-01-01' + INTERVAL '1' YEAR
      )
    )
    AND s.s_nationkey = n.n_nationkey
    AND n.n_name = 'KENYA'
    ORDER BY s.s_name
  )";

  run_tpch_query_test(con, query, "Q20");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q21 - Suppliers Who Kept Orders Waiting", "[modified_pipeline][tpch][q21]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT s.s_name, COUNT(*) as numwait
    FROM supplier s, lineitem l1, orders o, nation n
    WHERE s.s_suppkey = l1.l_suppkey
      AND o.o_orderkey = l1.l_orderkey
      AND o.o_orderstatus = 'F'
      AND l1.l_receiptdate > l1.l_commitdate
      AND EXISTS (
        SELECT * FROM lineitem l2
        WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey
      )
      AND NOT EXISTS (
        SELECT * FROM lineitem l3
        WHERE l3.l_orderkey = l1.l_orderkey
          AND l3.l_suppkey <> l1.l_suppkey
          AND l3.l_receiptdate > l3.l_commitdate
      )
      AND s.s_nationkey = n.n_nationkey
      AND n.n_name = 'BRAZIL'
    GROUP BY s.s_name
    ORDER BY numwait DESC, s.s_name
    LIMIT 100
  )";

  run_tpch_query_test(con, query, "Q21");
  Config::MODIFIED_PIPELINE = false;
}

TEST_CASE("TPC-H Q22 - Global Sales Opportunity", "[modified_pipeline][tpch][q22]")
{
  DuckDB db(nullptr);
  safe_load_extension(db);
  Connection con(db);
  safe_init_gpu_buffer(con);
  Config::MODIFIED_PIPELINE = true;
  create_tpch_schema(con);

  std::string query = R"(
    SELECT cntrycode, COUNT(*) as numcust, SUM(c_acctbal) as totacctbal
    FROM (
      SELECT SUBSTRING(c_phone FROM 1 FOR 2) as cntrycode, c_acctbal
      FROM customer c
      WHERE SUBSTRING(c_phone FROM 1 FOR 2) IN ('24', '31', '11', '16', '21', '20', '34')
        AND c_acctbal > (
          SELECT AVG(c_acctbal) FROM customer
          WHERE c_acctbal > 0.00
            AND SUBSTRING(c_phone FROM 1 FOR 2) IN ('24', '31', '11', '16', '21', '20', '34')
        )
        AND NOT EXISTS (
          SELECT * FROM orders o WHERE o.o_custkey = c.c_custkey
        )
    ) as custsale
    GROUP BY cntrycode
    ORDER BY cntrycode
  )";

  run_tpch_query_test(con, query, "Q22");
  Config::MODIFIED_PIPELINE = false;
}
