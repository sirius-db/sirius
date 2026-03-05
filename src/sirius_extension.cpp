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

#include "duckdb/main/database.hpp"
#define DUCKDB_EXTENSION_MAIN

#include "config.hpp"
#include "data/sirius_converter_registry.hpp"
#include "last_gpu_buffers.hpp"
#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/duck_schema_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/relation.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/relation_statement.hpp"
#include "duckdb/planner/planner.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "from_substrait.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#include "log/logging.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "sirius_interface.hpp"

#include <cstdlib>

namespace duckdb {

const std::string PINNED_MEMORY_PARAM_KEY   = "pinned_memory_size";
bool SiriusExtension::buffer_is_initialized = false;

struct GPUTableFunctionData : public TableFunctionData {
  GPUTableFunctionData() = default;
  shared_ptr<Relation> plan;
  shared_ptr<GPUPreparedStatementData> gpu_prepared;
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<GPUContext> gpu_context;
  string query;
  bool enable_optimizer;
  bool finished   = false;
  bool plan_error = false;
  //! Original options from the connection
  ClientConfig original_config;
  set<OptimizerType> original_disabled_optimizers;

  void PrepareConnection(ClientContext& context)
  {
    // First collect original options
    original_config              = context.config;
    original_disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;

    // The user might want to disable the optimizer of the new connection
    context.config.enable_optimizer = enable_optimizer;
    // We want for sure to disable the internal compression optimizations.
    // These are DuckDB specific, no other system implements these. Also,
    // respect the user's settings if they chose to disable any specific optimizers.
    //
    // The InClauseRewriter optimization converts large `IN` clauses to a
    // "mark join" against a `ColumnDataCollection`, which may not make
    // sense in other systems and would complicate the conversion to Substrait.
    set<OptimizerType> disabled_optimizers =
      DBConfig::GetConfig(context).options.disabled_optimizers;
    disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
    disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
#ifdef DEBUG
    disabled_optimizers.insert(OptimizerType::COLUMN_LIFETIME);
#endif
    // disabled_optimizers.insert(OptimizerType::MATERIALIZED_CTE);
    // If error(varchar) gets implemented in substrait this can be removed
    // context.config.scalar_subquery_error_on_multiple_rows = false;
    DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;
  }

  // Reset configuration
  void CleanupConnection(ClientContext& context) const
  {
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled_optimizers;
    context.config                                           = original_config;
  }

  unique_ptr<LogicalOperator> ExtractPlan(ClientContext& context)
  {
    PrepareConnection(context);
    unique_ptr<LogicalOperator> plan;
    try {
      Parser parser(context.GetParserOptions());
      parser.ParseQuery(query);

      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      D_ASSERT(planner.plan);

      plan = std::move(planner.plan);

      if (context.config.enable_optimizer) {
        Optimizer optimizer(*planner.binder, context);
        plan = optimizer.Optimize(std::move(plan));
      }

      // After optimization, refresh types before column binding resolution
      // to ensure types are consistent (some optimizers may have set stale types)
      plan->ResolveOperatorTypes();

      ColumnBindingResolver resolver;
      ColumnBindingResolver::Verify(*plan);
      resolver.VisitOperator(*plan);
    } catch (...) {
      CleanupConnection(context);
      throw;
    }

    CleanupConnection(context);
    return plan;
  }
};

struct SiriusTableFunctionData : public TableFunctionData {
  SiriusTableFunctionData() = default;
  shared_ptr<::sirius::sirius_prepared_statement_data> gpu_prepared;
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<::sirius::sirius_interface> sirius_iface;
  string query;
  string substrait_blob;  // Original Substrait bytes for from_substrait fallback
  bool enable_optimizer;
  bool finished   = false;
  bool plan_error = false;
  //! Original options from the connection
  ClientConfig original_config;
  set<OptimizerType> original_disabled_optimizers;

  void PrepareConnection(ClientContext& context)
  {
    // First collect original options
    original_config              = context.config;
    original_disabled_optimizers = DBConfig::GetConfig(context).options.disabled_optimizers;

    // The user might want to disable the optimizer of the new connection
    context.config.enable_optimizer = enable_optimizer;
    // We want for sure to disable the internal compression optimizations.
    // These are DuckDB specific, no other system implements these. Also,
    // respect the user's settings if they chose to disable any specific optimizers.
    //
    // The InClauseRewriter optimization converts large `IN` clauses to a
    // "mark join" against a `ColumnDataCollection`, which may not make
    // sense in other systems and would complicate the conversion to Substrait.
    set<OptimizerType> disabled_optimizers =
      DBConfig::GetConfig(context).options.disabled_optimizers;
    disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
    disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
#ifdef DEBUG
    disabled_optimizers.insert(OptimizerType::COLUMN_LIFETIME);
#endif
    // disabled_optimizers.insert(OptimizerType::MATERIALIZED_CTE);
    // If error(varchar) gets implemented in substrait this can be removed
    // context.config.scalar_subquery_error_on_multiple_rows = false;
    DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;
  }

  // Reset configuration
  void CleanupConnection(ClientContext& context) const
  {
    DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled_optimizers;
    context.config                                           = original_config;
  }

  unique_ptr<LogicalOperator> ExtractPlan(ClientContext& context)
  {
    PrepareConnection(context);
    unique_ptr<LogicalOperator> plan;
    try {
      Parser parser(context.GetParserOptions());
      parser.ParseQuery(query);

      Planner planner(context);
      planner.CreatePlan(std::move(parser.statements[0]));
      D_ASSERT(planner.plan);

      plan = std::move(planner.plan);

      if (context.config.enable_optimizer) {
        Optimizer optimizer(*planner.binder, context);
        plan = optimizer.Optimize(std::move(plan));
      }

      // After optimization, refresh types before column binding resolution
      // to ensure types are consistent (some optimizers may have set stale types)
      plan->ResolveOperatorTypes();

      ColumnBindingResolver resolver;
      ColumnBindingResolver::Verify(*plan);
      resolver.VisitOperator(*plan);
    } catch (...) {
      CleanupConnection(context);
      throw;
    }

    CleanupConnection(context);
    return plan;
  }
};

void do_nothing_context(ClientContext*) {}

static unique_ptr<GPUPhysicalOperator> GPUGeneratePhysicalPlan(
  ClientContext& context,
  GPUContext& gpu_context,
  unique_ptr<LogicalOperator>& logical_plan,
  Connection& new_conn)
{
  GPUPhysicalPlanGenerator physical_planner = GPUPhysicalPlanGenerator(context, gpu_context);
  auto physical_plan                        = physical_planner.CreatePlan(std::move(logical_plan));
  return physical_plan;
}

// The result of the GPUProcessingBind function is a unique pointer to a FunctionData object.
// This result of this function is used as an argument to the GPUProcessingFunction function (data_p
// argument), which is called to execute the table function.
unique_ptr<FunctionData> SiriusExtension::GPUProcessingBind(ClientContext& context,
                                                            TableFunctionBindInput& input,
                                                            vector<LogicalType>& return_types,
                                                            vector<string>& names)
{
  auto result              = make_uniq<GPUTableFunctionData>();
  result->conn             = make_uniq<Connection>(*context.db);
  result->query            = input.inputs[0].ToString();
  result->enable_optimizer = true;
  result->gpu_context      = make_uniq<GPUContext>(context);
  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_processing cannot be called with a NULL parameter");
  }

  // Parse the query just to get the result type information and to create preparedstatmement data
  auto statements = result->conn->context->ParseStatements(result->query);
  Planner planner(context);
  auto statement_type = statements[0]->type;
  planner.CreatePlan(std::move(statements[0]));
  D_ASSERT(planner.plan);

  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  // generate physical plan from the logical plan
  unique_ptr<LogicalOperator> query_plan = result->ExtractPlan(context);
  SIRIUS_LOG_DEBUG("Query plan:\n{}", query_plan->ToString());
  if (buffer_is_initialized) {
    try {
      auto gpu_physical_plan =
        GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
      auto gpu_prepared    = make_shared_ptr<GPUPreparedStatementData>(std::move(prepared),
                                                                    std::move(gpu_physical_plan));
      result->gpu_prepared = gpu_prepared;
    } catch (std::exception& e) {
      ErrorData error(e);
      SIRIUS_LOG_ERROR("Error in GPUGeneratePhysicalPlan: {}", error.RawMessage());
      result->plan_error = true;
    }
  } else {
    result->gpu_prepared = nullptr;
  }

  for (auto& column : planner.names) {
    names.emplace_back(column);
  }
  for (auto& type : planner.types) {
    return_types.emplace_back(type);
  }

  return std::move(result);
}

void SiriusExtension::GPUProcessingFunction(ClientContext& context,
                                            TableFunctionInput& data_p,
                                            DataChunk& output)
{
  auto& data = (GPUTableFunctionData&)*data_p.bind_data;
  if (data.finished) { return; }

  if (!data.res) {
    auto start = std::chrono::high_resolution_clock::now();
    if (!buffer_is_initialized) {
      printf("\033[1;31m");
      printf("GPUBufferManager not initialized, please call gpu_buffer_init first\n");
      printf("\033[0m");
      printf(
        "=============================================\nError in GPUExecuteQuery, fallback to "
        "DuckDB\n=============================================\n");
      data.res = data.conn->Query(data.query);
    } else if (data.plan_error) {
      printf(
        "=============================================\nError in GPUExecuteQuery, fallback to "
        "DuckDB\n=============================================\n");
      data.res = data.conn->Query(data.query);
    } else {
      data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
      if (data.res->HasError()) {
        printf(
          "=============================================\nError in GPUExecuteQuery, fallback to "
          "DuckDB\n=============================================\n");
        data.res = data.conn->Query(data.query);
      }
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_INFO("Execute query time: {:.2f} ms", duration.count() / 1000.0);
  }

  auto result_chunk = data.res->Fetch();
  if (result_chunk == nullptr) {
    output.SetCardinality(0);
    return;
  }

  output.Reference(*result_chunk);
  return;
}

// Forward declaration — defined later in this file.
static unique_ptr<LogicalOperator> OptimizePlan(ClientContext& context,
                                                Planner& planner,
                                                Connection& new_conn);

unique_ptr<FunctionData> SiriusExtension::GPUProcessingSubstraitBind(
  ClientContext& context,
  TableFunctionBindInput& input,
  vector<LogicalType>& return_types,
  vector<string>& names)
{
  auto result              = make_uniq<GPUTableFunctionData>();
  result->conn             = make_uniq<Connection>(*context.db);
  result->query            = input.inputs[0].ToString();
  result->enable_optimizer = true;
  result->gpu_context      = make_uniq<GPUContext>(context);
  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_processing_substrait cannot be called with a NULL parameter");
  }
  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: start, input size={}", input.inputs[0].ToString().size());
  string serialized = input.inputs[0].GetValueUnsafe<string>();
  bool is_json = false;
  // Use the new connection's context to avoid deadlock with the locked caller context.
  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: transforming Substrait plan ({} bytes)", serialized.size());
  SubstraitToDuckDB transformer_s2d(result->conn->context, serialized, is_json, false);
  result->plan = transformer_s2d.TransformPlan();
  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: TransformPlan done");

  auto relation_stmt                  = make_uniq<RelationStatement>(result->plan);
  unique_ptr<SQLStatement> statements = std::move(relation_stmt);
  auto statement_type                 = statements->type;

  set<OptimizerType> disabled_optimizers =
    DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled_optimizers.insert(OptimizerType::IN_CLAUSE);
  disabled_optimizers.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  DBConfig::GetConfig(context).options.disabled_optimizers = disabled_optimizers;

  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: creating DuckDB plan");
  Planner planner(context);
  planner.CreatePlan(std::move(statements));
  D_ASSERT(planner.plan);
  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: DuckDB plan created, {} columns", planner.names.size());

  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: optimizing plan");
  auto query_plan = OptimizePlan(context, planner, *result->conn);
  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: plan optimized, generating GPU physical plan");
  try {
    auto gpu_physical_plan =
      GPUGeneratePhysicalPlan(context, *result->gpu_context, query_plan, *result->conn);
    SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: GPU physical plan generated");
    auto gpu_prepared =
      make_shared_ptr<GPUPreparedStatementData>(std::move(prepared), std::move(gpu_physical_plan));
    result->gpu_prepared = gpu_prepared;
  } catch (std::exception& e) {
    ErrorData error(e);
    SIRIUS_LOG_ERROR("Error in GPUGeneratePhysicalPlan (substrait): {}", error.RawMessage());
    result->plan_error = true;
  }

  for (auto& column : planner.names) {
    names.emplace_back(column);
  }
  for (auto& type : planner.types) {
    return_types.emplace_back(type);
  }
  SIRIUS_LOG_INFO("[gpu_processing_substrait] bind: done, {} columns, {} types", names.size(), return_types.size());

  return std::move(result);
}

void SiriusExtension::GPUProcessingSubstraitFunction(ClientContext& context,
                                                     TableFunctionInput& data_p,
                                                     DataChunk& output)
{
  auto& data = (GPUTableFunctionData&)*data_p.bind_data;
  if (data.finished) { return; }
  if (!data.res) {
    SIRIUS_LOG_INFO("[gpu_processing_substrait] exec: starting, buffer_initialized={}, plan_error={}", buffer_is_initialized, data.plan_error);
    auto start = std::chrono::high_resolution_clock::now();
    if (!buffer_is_initialized) {
      SIRIUS_LOG_ERROR("GPUBufferManager not initialized, falling back to DuckDB");
      auto con           = Connection(*context.db);
      data.plan->context = make_shared_ptr<ClientContextWrapper>(con.context);
      data.res           = data.plan->Execute();
    } else if (data.plan_error) {
      SIRIUS_LOG_ERROR("GPU plan generation failed, falling back to DuckDB");
      auto con           = Connection(*context.db);
      data.plan->context = make_shared_ptr<ClientContextWrapper>(con.context);
      data.res           = data.plan->Execute();
    } else {
      SIRIUS_LOG_INFO("[gpu_processing_substrait] exec: calling GPUExecuteQuery");
      data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
      SIRIUS_LOG_INFO("[gpu_processing_substrait] exec: GPUExecuteQuery returned, hasError={}", data.res->HasError());
      if (data.res->HasError()) {
        SIRIUS_LOG_ERROR("GPUExecuteQuery error: {}, falling back to DuckDB", data.res->GetError());
        auto con           = Connection(*context.db);
        data.plan->context = make_shared_ptr<ClientContextWrapper>(con.context);
        data.res           = data.plan->Execute();
      }
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_INFO("GPU Execute query time (substrait): {:.2f} ms", duration.count() / 1000.0);
  }

  auto result_chunk = data.res->Fetch();
  if (!result_chunk) { return; }
  output.Move(*result_chunk);
  return;
}

static unique_ptr<sirius::op::sirius_physical_operator> SiriusGeneratePhysicalPlan(
  ClientContext& context, unique_ptr<LogicalOperator>& logical_plan)
{
  sirius::planner::sirius_physical_plan_generator physical_planner =
    sirius::planner::sirius_physical_plan_generator(context);
  auto physical_plan = physical_planner.create_plan(std::move(logical_plan));
  return physical_plan;
}

// The result of the GPUExecutionBind function is a unique pointer to a FunctionData object.
// This result of this function is used as an argument to the GPUExecutionFunction function (data_p
// argument), which is called to execute the table function.
unique_ptr<FunctionData> SiriusExtension::GPUExecutionBind(ClientContext& context,
                                                           TableFunctionBindInput& input,
                                                           vector<LogicalType>& return_types,
                                                           vector<string>& names)
{
  auto result              = make_uniq<SiriusTableFunctionData>();
  result->conn             = make_uniq<Connection>(*context.db);
  result->query            = input.inputs[0].ToString();
  result->enable_optimizer = true;
  result->sirius_iface     = make_uniq<::sirius::sirius_interface>(context);

  // The outer connection may have been created before the Sirius extension was loaded,
  // so its registered_state won't have "sirius_state". Copy it from the inner connection.
  if (!context.registered_state->Get<SiriusContext>("sirius_state")) {
    auto sirius_state = result->conn->context->registered_state->Get<SiriusContext>("sirius_state");
    if (sirius_state) {
      context.registered_state->Insert("sirius_state", sirius_state);
    }
  }

  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_execution cannot be called with a NULL parameter");
  }

  // Parse the query just to get the result type information and to create preparedstatmement data
  Parser parser(context.GetParserOptions());
  parser.ParseQuery(result->query);
  Planner planner(context);
  auto statement_type = parser.statements[0]->type;
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

  // cuDF does not support HUGEINT (int128). DuckDB widens aggregates like sum(int32) to HUGEINT.
  // Downcast to BIGINT so all downstream operators and the result collector use a supported type.
  for (auto& type : planner.types) {
    if (type == LogicalType::HUGEINT) { type = LogicalType::BIGINT; }
  }

  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  // generate physical plan from the logical plan
  unique_ptr<LogicalOperator> query_plan = result->ExtractPlan(context);
  SIRIUS_LOG_DEBUG("Query plan:\n{}", query_plan->ToString());
  try {
    auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, query_plan);
    SIRIUS_LOG_DEBUG("Done generating sirius physical plan");
    auto gpu_prepared = make_shared_ptr<::sirius::sirius_prepared_statement_data>(
      std::move(prepared), std::move(sirius_physical_plan));
    result->gpu_prepared = gpu_prepared;
  } catch (std::exception& e) {
    ErrorData error(e);
    SIRIUS_LOG_ERROR("Error in SiriusGeneratePhysicalPlan: {}", error.RawMessage());
    if (Config::ENABLE_DUCKDB_FALLBACK) {
      result->plan_error = true;
    } else {
      throw std::runtime_error("Error in SiriusGeneratePhysicalPlan: " + error.RawMessage());
      return nullptr;
    }
  }

  for (auto& column : planner.names) {
    names.emplace_back(column);
  }
  for (auto& type : planner.types) {
    return_types.emplace_back(type);
  }

  return std::move(result);
}

void SiriusExtension::GPUExecutionFunction(ClientContext& context,
                                           TableFunctionInput& data_p,
                                           DataChunk& output)
{
  SIRIUS_LOG_INFO("[GPUExecutionFunction] called, bind_data={}", (void*)data_p.bind_data.get());
  auto& data = (SiriusTableFunctionData&)*data_p.bind_data;
  if (data.finished) {
    SIRIUS_LOG_INFO("[GPUExecutionFunction] already finished");
    return;
  }

  if (!data.res) {
    SIRIUS_LOG_INFO("[GPUExecutionFunction] executing: plan_error={}, gpu_prepared={}, sirius_iface={}",
                    data.plan_error, (void*)data.gpu_prepared.get(), (void*)data.sirius_iface.get());
    auto start = std::chrono::high_resolution_clock::now();
    if (data.plan_error) {
      SIRIUS_LOG_WARN("[GPUExecutionFunction] GPU plan failed, falling back to from_substrait");
      if (!data.substrait_blob.empty()) {
        data.res = data.conn->TableFunction("from_substrait", {Value::BLOB_RAW(data.substrait_blob)})
                     ->Execute();
      } else {
        SIRIUS_LOG_ERROR("[GPUExecutionFunction] No Substrait blob for fallback");
        data.res = data.conn->Query(data.query);
      }
    } else {
      SIRIUS_LOG_INFO("[GPUExecutionFunction] calling sirius_execute_query");
      try {
        data.res =
          data.sirius_iface->sirius_execute_query(context, data.query, data.gpu_prepared, {});
      } catch (const std::exception& e) {
        // Catch GPU pipeline errors (e.g. InternalException) and convert to
        // std::runtime_error so DuckDB treats it as recoverable, not FATAL.
        // This preserves the connection for CPU fallback on the Rust side.
        SIRIUS_LOG_ERROR("[GPUExecutionFunction] sirius_execute_query threw: {}", e.what());
        throw std::runtime_error(std::string("SiriusExecuteQuery error: ") + e.what());
      }
      SIRIUS_LOG_INFO("[GPUExecutionFunction] sirius_execute_query returned, hasError={}", data.res->HasError());
      if (data.res->HasError()) {
        // Propagate GPU runtime errors to the caller (Rust side) for safe fallback.
        // Do NOT fall back on the same inner connection — GPU pipeline threads may
        // still be running and accessing shared state, making reuse unsafe (SIGSEGV).
        auto error_msg = data.res->GetError();
        SIRIUS_LOG_ERROR("SiriusExecuteQuery error: {}", error_msg);
        throw std::runtime_error("SiriusExecuteQuery error: " + error_msg);
      }
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_INFO("Execute query time: {:.2f} ms", duration.count() / 1000.0);
  }

  auto result_chunk = data.res->Fetch();
  if (result_chunk == nullptr) {
    output.SetCardinality(0);
    return;
  }

  output.Reference(*result_chunk);
  return;
}

// gpu_execution_substrait: Substrait input + new Sirius execution framework.
// Combines the Substrait → DuckDB plan conversion from gpu_processing_substrait
// with the new sirius_physical_plan_generator execution from gpu_execution.
unique_ptr<FunctionData> SiriusExtension::GPUExecutionSubstraitBind(ClientContext& context,
                                                                     TableFunctionBindInput& input,
                                                                     vector<LogicalType>& return_types,
                                                                     vector<string>& names)
{
  auto result              = make_uniq<SiriusTableFunctionData>();
  result->conn             = make_uniq<Connection>(*context.db);
  result->enable_optimizer = true;
  result->sirius_iface     = make_uniq<::sirius::sirius_interface>(context);

  // The outer connection may have been created before the Sirius extension was loaded,
  // so its registered_state won't have "sirius_state". Copy it from the inner connection
  // which was opened after extension load and has the SiriusContext registered via
  // OnConnectionOpened callback.
  if (!context.registered_state->Get<SiriusContext>("sirius_state")) {
    auto sirius_state = result->conn->context->registered_state->Get<SiriusContext>("sirius_state");
    if (sirius_state) {
      context.registered_state->Insert("sirius_state", sirius_state);
      SIRIUS_LOG_INFO("[gpu_execution_substrait] registered SiriusContext on outer context");
    }
  }

  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_execution_substrait cannot be called with a NULL parameter");
  }

  // Convert Substrait → DuckDB logical plan.
  string serialized = input.inputs[0].GetValueUnsafe<string>();
  bool is_json = false;
  SubstraitToDuckDB transformer_s2d(result->conn->context, serialized, is_json, false);
  auto relation = transformer_s2d.TransformPlan();

  auto relation_stmt                  = make_uniq<RelationStatement>(relation);
  unique_ptr<SQLStatement> statements = std::move(relation_stmt);
  auto statement_type                 = statements->type;

  result->PrepareConnection(context);

  Planner planner(context);
  planner.CreatePlan(std::move(statements));
  D_ASSERT(planner.plan);

  // cuDF does not support HUGEINT. Downcast to BIGINT.
  for (auto& type : planner.types) {
    if (type == LogicalType::HUGEINT) { type = LogicalType::BIGINT; }
  }

  auto prepared       = make_shared_ptr<PreparedStatementData>(statement_type);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  // Optimize and resolve the logical plan.
  auto plan = std::move(planner.plan);
  if (context.config.enable_optimizer) {
    Optimizer optimizer(*planner.binder, context);
    plan = optimizer.Optimize(std::move(plan));
  }
  plan->ResolveOperatorTypes();
  ColumnBindingResolver resolver;
  ColumnBindingResolver::Verify(*plan);
  resolver.VisitOperator(*plan);

  result->CleanupConnection(context);

  // Generate Sirius physical plan using the new execution framework.
  SIRIUS_LOG_INFO("[gpu_execution_substrait] bind: generating Sirius physical plan");
  SIRIUS_LOG_INFO("[gpu_execution_substrait] bind: logical plan type={} children={}",
                  LogicalOperatorToString(plan->type), plan->children.size());
  // Recursively log the logical plan tree.
  {
    std::function<void(LogicalOperator*, int)> log_tree = [&](LogicalOperator* op, int depth) {
      std::string indent(depth * 2, ' ');
      SIRIUS_LOG_INFO("[gpu_execution_substrait] logical_tree: {}type={} children={}",
                      indent, LogicalOperatorToString(op->type), op->children.size());
      for (auto& child : op->children) {
        log_tree(child.get(), depth + 1);
      }
    };
    log_tree(plan.get(), 0);
  }
  try {
    auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, plan);
    SIRIUS_LOG_INFO("[gpu_execution_substrait] bind: Sirius physical plan generated OK, root type={}",
                    static_cast<int>(sirius_physical_plan->type));
    auto gpu_prepared = make_shared_ptr<::sirius::sirius_prepared_statement_data>(
      std::move(prepared), std::move(sirius_physical_plan));
    result->gpu_prepared = gpu_prepared;
  } catch (std::exception& e) {
    ErrorData error(e);
    SIRIUS_LOG_ERROR("[gpu_execution_substrait] Error in SiriusGeneratePhysicalPlan: {}", error.RawMessage());
    result->plan_error = true;
  }

  // Store the Substrait bytes for from_substrait fallback when GPU plan fails.
  result->substrait_blob = serialized;
  result->query = "-- substrait plan (no SQL fallback available)";

  for (auto& column : planner.names) {
    names.emplace_back(column);
  }
  for (auto& type : planner.types) {
    return_types.emplace_back(type);
  }

  return std::move(result);
}

static unique_ptr<LogicalOperator> OptimizePlan(ClientContext& context,
                                                Planner& planner,
                                                Connection& new_conn)
{
  unique_ptr<LogicalOperator> plan;
  plan = std::move(planner.plan);

  Optimizer optimizer(*planner.binder, context);
  plan = optimizer.Optimize(std::move(plan));
  SIRIUS_LOG_DEBUG("Query plan:\n{}", plan->ToString());

  ColumnBindingResolver resolver;
  resolver.Verify(*plan);
  resolver.VisitOperator(*plan);

  plan->ResolveOperatorTypes();

  return plan;
}

struct GPUBufferInitFunctionData : public TableFunctionData {
  GPUBufferInitFunctionData() {}
  bool finished = false;
  size_t cache_size;
  size_t processing_size;
  size_t pinned_memory_size;
};

unique_ptr<FunctionData> SiriusExtension::GPUBufferInitBind(ClientContext& context,
                                                            TableFunctionBindInput& input,
                                                            vector<LogicalType>& return_types,
                                                            vector<string>& names)
{
  auto result = make_uniq<GPUBufferInitFunctionData>();

  string gpu_cache_size      = input.inputs[0].ToString();
  string gpu_processing_size = input.inputs[1].ToString();
  string pinned_memory_size("0 GB");  // Default size of pinned memory
  if (input.named_parameters.find(PINNED_MEMORY_PARAM_KEY) != input.named_parameters.end()) {
    // If the pinned memory size is specified in the arguments then use that
    pinned_memory_size = input.named_parameters[PINNED_MEMORY_PARAM_KEY].ToString();
  }

  // parsing 2GB or 2GiB to size_t
  //  Function to parse size strings like "2GB" or "2GiB" to size_t
  auto parse_size = [](const string& size_str) -> size_t {
    size_t result     = 0;
    size_t multiplier = 1;
    string num_part;
    string unit_part;

    size_t i = 0;
    // Skip any whitespace between number and unit
    while (i < size_str.length() && isspace(size_str[i])) {
      i++;
    }

    // Find where the number ends and unit begins
    while (i < size_str.length() && (isdigit(size_str[i]) || size_str[i] == '.')) {
      num_part += size_str[i];
      i++;
    }

    // Skip any whitespace between number and unit
    while (i < size_str.length() && isspace(size_str[i])) {
      i++;
    }

    // Extract unit part
    unit_part = size_str.substr(i);

    // Convert number part to double
    double num_value = stod(num_part);

    // Determine multiplier based on unit
    if (unit_part == "B") {
      multiplier = 1;
    } else if (unit_part == "KB" || unit_part == "KiB") {
      multiplier = 1024;
    } else if (unit_part == "MB" || unit_part == "MiB") {
      multiplier = 1024 * 1024;
    } else if (unit_part == "GB" || unit_part == "GiB") {
      multiplier = 1024 * 1024 * 1024;
    } else if (unit_part == "TB" || unit_part == "TiB") {
      multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
    } else {
      throw InvalidInputException("Invalid format");
    }

    result = (size_t)(num_value * multiplier);
    return result;
  };

  // Parse the input sizes
  result->cache_size         = parse_size(gpu_cache_size);
  result->processing_size    = parse_size(gpu_processing_size);
  result->pinned_memory_size = parse_size(pinned_memory_size);

  auto type = LogicalType(LogicalTypeId::BOOLEAN);
  return_types.emplace_back(type);
  names.emplace_back("Success");
  return std::move(result);
}

void SiriusExtension::GPUBufferInitFunction(ClientContext& context,
                                            TableFunctionInput& data_p,
                                            DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<GPUBufferInitFunctionData>();
  if (data.finished) { return; }

  size_t cache_size         = data.cache_size;
  size_t processing_size    = data.processing_size;
  size_t pinned_memory_size = data.pinned_memory_size;
  if (pinned_memory_size == 0) { pinned_memory_size = std::max(cache_size, processing_size); }

  if (!buffer_is_initialized) {
    SIRIUS_LOG_DEBUG(
      "GPU Buffer Manager initialized with args: Cache Size - {}, Processing Size - {}, Pinned Mem "
      "Size - {}\n",
      cache_size,
      processing_size,
      pinned_memory_size);
    GPUBufferManager* gpuBufferManager =
      &(GPUBufferManager::GetInstance(cache_size, processing_size, pinned_memory_size));
    buffer_is_initialized = true;
  } else {
    SIRIUS_LOG_WARN("GPUBufferManager already initialized");
  }
  data.finished = true;
}

// --- gpu_allocate_buffers(sizes BIGINT[], device_id INT) table function ---
// Allocates GPU memory via GPUBufferManager for nixl destination buffers.

struct GPUAllocateBuffersData : public TableFunctionData {
  bool finished = false;
  vector<int64_t> sizes;
  int device_id = 0;
};

unique_ptr<FunctionData> SiriusExtension::GPUAllocateBuffersBind(
  ClientContext& context,
  TableFunctionBindInput& input,
  vector<LogicalType>& return_types,
  vector<string>& names)
{
  auto result = make_uniq<GPUAllocateBuffersData>();

  // First arg: LIST of sizes (BIGINT[]).
  auto& sizes_value = input.inputs[0];
  auto& list_children = ListValue::GetChildren(sizes_value);
  for (auto& child : list_children) {
    result->sizes.push_back(BigIntValue::Get(child));
  }

  // Second arg: device_id (INT).
  result->device_id = IntegerValue::Get(input.inputs[1]);

  return_types = {
    LogicalType::INTEGER,  // buffer_id
    LogicalType::BIGINT,   // addr
    LogicalType::BIGINT,   // len
  };
  names = {"buffer_id", "addr", "len"};
  return std::move(result);
}

void SiriusExtension::GPUAllocateBuffersFunction(
  ClientContext& context,
  TableFunctionInput& data_p,
  DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<GPUAllocateBuffersData>();
  if (data.finished) {
    output.SetCardinality(0);
    return;
  }

  if (!buffer_is_initialized) {
    throw InvalidInputException("GPU buffer manager not initialized. Call gpu_buffer_init first.");
  }

  auto& bm = GPUBufferManager::GetInstance();
  idx_t count = MinValue<idx_t>(data.sizes.size(), STANDARD_VECTOR_SIZE);
  output.SetCardinality(count);

  for (idx_t i = 0; i < count; i++) {
    size_t alloc_size = static_cast<size_t>(data.sizes[i]);
    // Non-caching allocation (tracked in allocation_table, can be freed).
    auto* ptr = bm.customCudaMalloc<uint8_t>(alloc_size, data.device_id, false);
    auto addr = reinterpret_cast<uintptr_t>(ptr);

    output.SetValue(0, i, Value::INTEGER(static_cast<int32_t>(i)));
    output.SetValue(1, i, Value::BIGINT(static_cast<int64_t>(addr)));
    output.SetValue(2, i, Value::BIGINT(static_cast<int64_t>(alloc_size)));
  }

  data.finished = true;
}

// --- sirius_get_last_gpu_buffers() table function ---

struct GetLastGPUBuffersData : public TableFunctionData {
  bool finished = false;
};

unique_ptr<FunctionData> SiriusExtension::GetLastGPUBuffersBind(
  ClientContext& context,
  TableFunctionBindInput& input,
  vector<LogicalType>& return_types,
  vector<string>& names)
{
  return_types = {
    LogicalType::INTEGER,  // buffer_id
    LogicalType::BIGINT,   // addr
    LogicalType::BIGINT,   // len
    LogicalType::INTEGER,  // device_id
    LogicalType::VARCHAR,  // column_name
    LogicalType::INTEGER,  // type_id
    LogicalType::BIGINT,   // num_rows
  };
  names = {"buffer_id", "addr", "len", "device_id", "column_name", "type_id", "num_rows"};
  return make_uniq<GetLastGPUBuffersData>();
}

void SiriusExtension::GetLastGPUBuffersFunction(
  ClientContext& context,
  TableFunctionInput& data_p,
  DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<GetLastGPUBuffersData>();
  if (data.finished) {
    output.SetCardinality(0);
    return;
  }

  auto buffers = LastGPUBuffers::GetInstance().Get();
  idx_t count  = MinValue<idx_t>(buffers.size(), STANDARD_VECTOR_SIZE);
  output.SetCardinality(count);

  for (idx_t i = 0; i < count; i++) {
    auto& buf = buffers[i];
    output.SetValue(0, i, Value::INTEGER(static_cast<int32_t>(i)));
    output.SetValue(1, i, Value::BIGINT(static_cast<int64_t>(buf.addr)));
    output.SetValue(2, i, Value::BIGINT(static_cast<int64_t>(buf.len)));
    output.SetValue(3, i, Value::INTEGER(buf.device_id));
    output.SetValue(4, i, Value(buf.column_name));
    output.SetValue(5, i, Value::INTEGER(buf.type_id));
    output.SetValue(6, i, Value::BIGINT(static_cast<int64_t>(buf.num_rows)));
  }

  data.finished = true;
}

// --- sirius_retain_gpu_buffers() table function ---

struct RetainGPUBuffersData : public TableFunctionData {
  bool finished = false;
};

unique_ptr<FunctionData> SiriusExtension::RetainGPUBuffersBind(
  ClientContext& context,
  TableFunctionBindInput& input,
  vector<LogicalType>& return_types,
  vector<string>& names)
{
  return_types = {LogicalType::VARCHAR};
  names = {"status"};
  return make_uniq<RetainGPUBuffersData>();
}

void SiriusExtension::RetainGPUBuffersFunction(
  ClientContext& context,
  TableFunctionInput& data_p,
  DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<RetainGPUBuffersData>();
  if (data.finished) {
    output.SetCardinality(0);
    return;
  }

  LastGPUBuffers::GetInstance().SetRetainNext(true);
  output.SetCardinality(1);
  output.SetValue(0, 0, Value("retain_next=true"));
  data.finished = true;
}

// --- sirius_release_gpu_buffers() table function ---

struct ReleaseGPUBuffersData : public TableFunctionData {
  bool finished = false;
};

unique_ptr<FunctionData> SiriusExtension::ReleaseGPUBuffersBind(
  ClientContext& context,
  TableFunctionBindInput& input,
  vector<LogicalType>& return_types,
  vector<string>& names)
{
  return_types = {LogicalType::VARCHAR, LogicalType::BIGINT};
  names = {"status", "freed_count"};
  return make_uniq<ReleaseGPUBuffersData>();
}

void SiriusExtension::ReleaseGPUBuffersFunction(
  ClientContext& context,
  TableFunctionInput& data_p,
  DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<ReleaseGPUBuffersData>();
  if (data.finished) {
    output.SetCardinality(0);
    return;
  }

  size_t freed = 0;
  if (buffer_is_initialized) {
    freed = GPUBufferManager::GetInstance().ReleaseRetainedBuffers();
  }
  LastGPUBuffers::GetInstance().ReleaseRetainedData();
  LastGPUBuffers::GetInstance().Clear();

  output.SetCardinality(1);
  output.SetValue(0, 0, Value("released"));
  output.SetValue(1, 0, Value::BIGINT(static_cast<int64_t>(freed)));
  data.finished = true;
}

void SiriusExtension::RegisterGPUFunctions(DatabaseInstance& instance)
{
  auto transaction = CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = Catalog::GetSystemCatalog(instance);
  TableFunction gpu_buffer_init("gpu_buffer_init",
                                {LogicalType::VARCHAR, LogicalType::VARCHAR},
                                GPUBufferInitFunction,
                                GPUBufferInitBind);
  gpu_buffer_init.named_parameters[PINNED_MEMORY_PARAM_KEY] = LogicalType::VARCHAR;
  CreateTableFunctionInfo gpu_buffer_init_info(gpu_buffer_init);
  catalog.CreateTableFunction(transaction, gpu_buffer_init_info);

  TableFunction gpu_processing(
    "gpu_processing", {LogicalType::VARCHAR}, GPUProcessingFunction, GPUProcessingBind);
  gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
  CreateTableFunctionInfo gpu_processing_info(gpu_processing);
  catalog.CreateTableFunction(transaction, gpu_processing_info);

  TableFunction gpu_execution("gpu_execution",
                              {LogicalType::VARCHAR},
                              GPUExecutionFunction,
                              SiriusExtension::GPUExecutionBind);
  gpu_execution.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
  CreateTableFunctionInfo gpu_execution_info(gpu_execution);
  catalog.CreateTableFunction(transaction, gpu_execution_info);

  TableFunction gpu_processing_substrait("gpu_processing_substrait",
                                         {LogicalType::BLOB},
                                         GPUProcessingSubstraitFunction,
                                         GPUProcessingSubstraitBind);
  CreateTableFunctionInfo gpu_processing_substrait_info(gpu_processing_substrait);
  catalog.CreateTableFunction(transaction, gpu_processing_substrait_info);

  TableFunction gpu_execution_substrait("gpu_execution_substrait",
                                        {LogicalType::BLOB},
                                        GPUExecutionFunction,
                                        SiriusExtension::GPUExecutionSubstraitBind);
  CreateTableFunctionInfo gpu_execution_substrait_info(gpu_execution_substrait);
  catalog.CreateTableFunction(transaction, gpu_execution_substrait_info);

  TableFunction get_last_gpu_buffers("sirius_get_last_gpu_buffers",
                                     {},
                                     GetLastGPUBuffersFunction,
                                     GetLastGPUBuffersBind);
  CreateTableFunctionInfo get_last_gpu_buffers_info(get_last_gpu_buffers);
  catalog.CreateTableFunction(transaction, get_last_gpu_buffers_info);

  TableFunction gpu_allocate_buffers("gpu_allocate_buffers",
                                     {LogicalType::LIST(LogicalType::BIGINT), LogicalType::INTEGER},
                                     GPUAllocateBuffersFunction,
                                     GPUAllocateBuffersBind);
  CreateTableFunctionInfo gpu_allocate_buffers_info(gpu_allocate_buffers);
  catalog.CreateTableFunction(transaction, gpu_allocate_buffers_info);

  TableFunction retain_gpu_buffers("sirius_retain_gpu_buffers",
                                    {},
                                    RetainGPUBuffersFunction,
                                    RetainGPUBuffersBind);
  CreateTableFunctionInfo retain_gpu_buffers_info(retain_gpu_buffers);
  catalog.CreateTableFunction(transaction, retain_gpu_buffers_info);

  TableFunction release_gpu_buffers("sirius_release_gpu_buffers",
                                     {},
                                     ReleaseGPUBuffersFunction,
                                     ReleaseGPUBuffersBind);
  CreateTableFunctionInfo release_gpu_buffers_info(release_gpu_buffers);
  catalog.CreateTableFunction(transaction, release_gpu_buffers_info);
}

static void SetUsePinMemory(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_PIN_MEM_FOR_CPU_PROCESSING = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CPU_PROCESSING to {}",
                   Config::USE_PIN_MEM_FOR_CPU_PROCESSING);
}

static void SetUseCudfExpr(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_CUDF_EXPR = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_CUDF_EXPR to {}", Config::USE_CUDF_EXPR);
}

static void SetUseCustomTopN(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_CUSTOM_TOP_N = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_CUSTOM_TOP_N to {}", Config::USE_CUSTOM_TOP_N);
}

static void SetUseOptTableScan(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_OPT_TABLE_SCAN = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_OPT_TABLE_SCAN to {}", Config::USE_OPT_TABLE_SCAN);
}

static void SetOptTableScanNumStreams(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS = IntegerValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_NUM_CUDA_STREAMS to {}",
                   Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS);
}

static void SetOptTableScanMemcpySize(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE to {}",
                   Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE);
}

static void SetPrintGPUTableMaxRows(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::PRINT_GPU_TABLE_MAX_ROWS = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config PRINT_GPU_TABLE_MAX_ROWS to {}",
                   Config::PRINT_GPU_TABLE_MAX_ROWS);
}

static void SetEnableFallbackCheck(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::ENABLE_FALLBACK_CHECK = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_FALLBACK_CHECK to {}", Config::ENABLE_FALLBACK_CHECK);
}

static void SetEnableDuckdbFallback(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::ENABLE_DUCKDB_FALLBACK = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_DUCKDB_FALLBACK to {}", Config::ENABLE_DUCKDB_FALLBACK);
}

static void SetEnableRegexJitImpl(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::ENABLE_REGEX_JIT_IMPL = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_REGEX_JIT_IMPL to {}", Config::ENABLE_REGEX_JIT_IMPL);
}

static void SetModifiedPipeline(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::MODIFIED_PIPELINE = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MODIFIED_PIPELINE to {}", Config::MODIFIED_PIPELINE);
}

static sirius::operator_params* get_operator_params(ClientContext& context)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    SIRIUS_LOG_DEBUG("SiriusContext not available; operator_params SET ignored");
    return nullptr;
  }
  return &sirius_ctx->get_config().get_operator_params();
}

static void SetDefaultScanTaskBatchSize(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->scan_task_batch_size = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config SCAN_TASK_BATCH_SIZE to {}", params->scan_task_batch_size);
}

static void SetDefaultScanTaskVarcharSize(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->default_scan_task_varchar_size = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config DEFAULT_SCAN_TASK_VARCHAR_SIZE to {}",
                   params->default_scan_task_varchar_size);
}

static void SetMaxSortPartitionBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->max_sort_partition_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MAX_SORT_PARTITION_BYTES to {}",
                   params->max_sort_partition_bytes);
}

static void SetHashPartitionBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->hash_partition_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config HASH_PARTITION_BYTES to {}", params->hash_partition_bytes);
}

static void SetConcatBatchBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->concat_batch_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config CONCAT_BATCH_BYTES to {}", params->concat_batch_bytes);
}

void SiriusExtension::InitialGPUConfigs(DBConfig& config)
{
  // Add in config option for gpu buffer manager
  config.AddExtensionOption("use_pin_memory",
                            "Whether or not the buffer manager is initialized with pinned memory",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_PIN_MEM_FOR_CPU_PROCESSING),
                            SetUsePinMemory);

  // Add in config option for expression executor
  config.AddExtensionOption("use_cudf_expr",
                            "Whether or not cudf is used to evaluate expressions",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_CUDF_EXPR),
                            SetUseCudfExpr);

  // Add in config option for top-N
  config.AddExtensionOption("use_custom_top_n",
                            "Whether or not custom kernel is used to evalaute top n",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_CUSTOM_TOP_N),
                            SetUseCustomTopN);

  // Add in config options for custom table scan
  config.AddExtensionOption("use_opt_table_scan",
                            "Whether or not the optional table scan is used",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_OPT_TABLE_SCAN),
                            SetUseOptTableScan);
  config.AddExtensionOption("opt_table_scan_num_streams",
                            "The number of cuda streams to use in the optional table scan",
                            LogicalType::INTEGER,
                            Value::INTEGER(Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS),
                            SetOptTableScanNumStreams);
  config.AddExtensionOption("opt_table_scan_memcpy_size",
                            "The memcpy size (in bytes) used by the optional table scan",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE),
                            SetOptTableScanMemcpySize);

  // Add in config options for printing gpu table
  config.AddExtensionOption("print_gpu_table_max_rows",
                            "Maximal amount of rows to render when printing gpu table",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(Config::PRINT_GPU_TABLE_MAX_ROWS),
                            SetPrintGPUTableMaxRows);

  // Add in config options for duckdb fallback checking
  config.AddExtensionOption("enable_fallback_check",
                            "Whether to enable fallback checking",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::ENABLE_FALLBACK_CHECK),
                            SetEnableFallbackCheck);

  config.AddExtensionOption(
    "enable_duckdb_fallback",
    "Whether to enable fallback to duckdb execution after an error is detected",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(Config::ENABLE_DUCKDB_FALLBACK),
    SetEnableDuckdbFallback);

  // Add in config options for special JIT implementation for regex
  config.AddExtensionOption(
    "enable_regex_jit_impl",
    "Whether to use special JIT implementation for particular regex evaluation",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(Config::ENABLE_REGEX_JIT_IMPL),
    SetEnableRegexJitImpl);

  // Add in config options for modified pipeline
  config.AddExtensionOption("modified_pipeline",
                            "Whether to use modified pipeline for GPU execution",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::MODIFIED_PIPELINE),
                            SetModifiedPipeline);

  // Add in config options for duckdb scan task
  // Default batch size
  config.AddExtensionOption("scan_task_batch_size",
                            "The default batch size for a duckdb scan task",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.scan_task_batch_size),
                            SetDefaultScanTaskBatchSize);
  // Default varchar size for estimating rows per batch
  config.AddExtensionOption(
    "default_scan_task_varchar_size",
    "The default varchar size for estimating rows per batch in a duckdb scan task",
    LogicalType::UBIGINT,
    Value::UBIGINT(sirius::operator_params{}.default_scan_task_varchar_size),
    SetDefaultScanTaskVarcharSize);

  // Add in config option for sort partition size
  config.AddExtensionOption("max_sort_partition_bytes",
                            "Maximum bytes per sort partition (0 = auto based on 33% GPU memory)",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.max_sort_partition_bytes),
                            SetMaxSortPartitionBytes);

  config.AddExtensionOption("hash_partition_bytes",
                            "Target size in bytes per hash partition",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.hash_partition_bytes),
                            SetHashPartitionBytes);

  config.AddExtensionOption("concat_batch_bytes",
                            "Target size for concat operator",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.concat_batch_bytes),
                            SetConcatBatchBytes);
}

static void LoadInternal(ExtensionLoader& loader)
{
  auto& db     = loader.GetDatabaseInstance();
  auto& config = DBConfig::GetConfig(db);
  config.extension_callbacks.push_back(make_uniq<duckdb::SiriusContextExtensionCallback>());
  sirius::converter_registry::initialize();
  SiriusExtension::InitialGPUConfigs(config);
  SiriusExtension::RegisterGPUFunctions(db);
}

void SiriusExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }

std::string SiriusExtension::Name() { return "Sirius	Extension"; }

std::string SiriusExtension::Version() const
{
#ifdef EXT_VERSION_SIRIUS
  return EXT_VERSION_SIRIUS;
#else
  return "";
#endif
}

}  // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(sirius, loader) { duckdb::LoadInternal(loader); }
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
