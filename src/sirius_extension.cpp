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
#include "data/data_batch_utils.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/open_file_info.hpp"
#include "expression_executor/expression_executor_strategy.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

// Forward-declare CUDA profiler API functions (linked via libcudart).
extern "C" int cudaProfilerStart();
extern "C" int cudaProfilerStop();
#include "data/sirius_converter_registry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/relation.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "transparent/sirius_optimizer_extension.hpp"
// #include "from_substrait.hpp"
#ifdef SIRIUS_ENABLE_LEGACY
#include "gpu_buffer_manager.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#endif
#include "duckdb/main/connection_manager.hpp"
#include "log/logging.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "pin_table.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "sirius_interface.hpp"
#include "sirius_sql_rewrite.hpp"
#include "util/segfault_backtrace.hpp"

// PinTableFunction routes parquet reads through the per-GPU sirius_ioctx
// instead of cudf's bundled file_source factory (which uses kvikio internally
// and binds to a single CUDA context). This is mandatory in multi-GPU
// configurations (enforced by sirius_config::enforce_sirius_datasource_for_multi_gpu()).
// Single-GPU users may still opt out via use_sirius_datasource=false; the
// pin pipeline always routes through sirius_ioctx when one is available.
//
// Ordering rule: include uring_reactor LAST among sirius headers — liburing.h
// transitively pulled by uring_reactor.hpp defines a BLOCK_SIZE preprocessor
// macro that collides with the BLOCK_SIZE static member in
// <blockingconcurrentqueue.h> (used by spdlog / pipeline / duckdb
// connection_manager). All consumers of blockingconcurrentqueue.h must
// precede this include.
#include "io/types.hpp"                // sirius::io::sirius_ioctx
#include "io/uring/uring_reactor.hpp"  // sirius::io::uring_io_object

#include <cstdlib>
#include <unordered_map>

namespace duckdb {

const std::string PINNED_MEMORY_PARAM_KEY = "pinned_memory_size";
#ifdef SIRIUS_ENABLE_LEGACY
bool SiriusExtension::buffer_is_initialized = false;
#endif

constexpr std::string QUERY_LABEL_PARAM_KEY = "query_label";

namespace {
unique_ptr<QueryResult> run_internal_cpu_fallback_query(ClientContext& context,
                                                        Connection& connection,
                                                        const string& query,
                                                        const string& gpu_error = "")
{
  // S3 CPU fallback is not supported. Sirius reads s3:// only on the GPU path
  // (sirius_read_parquet -> describe_parquet -> cuDF via s3_ioctx); DuckDB's CPU
  // read_parquet has no S3 filesystem, so a query that reads s3:// cannot fall
  // back to CPU. Surface a clear error (with the underlying GPU cause) instead
  // of replaying a query that would fail anyway. Local / non-s3 queries are
  // unaffected and fall through to the normal DuckDB CPU replay below.
  if (sirius::references_sirius_owned_s3_parquet(query)) {
    throw std::runtime_error(
      "S3 CPU fallback is not supported: this query reads s3:// data, GPU execution failed, and "
      "Sirius has no CPU fallback for S3 data sources. Underlying GPU error: " +
      gpu_error);
  }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return connection.Query(query); }

  duckdb::SiriusContext::InternalQueryGuard guard(*sirius_ctx);
  return connection.Query(query);
}

// Bind callback for the sirius_read_parquet table function — a thin forwarder.
// It resolves the URI to the connection's scan_manager and probes the parquet
// footer through describe_parquet (footer-only, no full-file download), then
// hands the inferred schema back to DuckDB. Bind data carries the URI and
// footer row count so the cardinality callback can expose a real estimate to
// the optimizer; the pipeline converter still reads the URI from parameters[0].
unique_ptr<FunctionData> SiriusReadParquetBind(ClientContext& context,
                                               TableFunctionBindInput& input,
                                               vector<LogicalType>& return_types,
                                               vector<string>& names)
{
  if (input.inputs.size() != 1 || input.inputs[0].IsNull()) {
    throw std::runtime_error("sirius_read_parquet expects a single non-null parquet URI");
  }
  auto const uri = input.inputs[0].GetValue<std::string>();

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw std::runtime_error("sirius_read_parquet: Sirius is not initialized on this connection");
  }

  auto bind_result = sirius_ctx->get_scan_manager().describe_parquet(uri);
  return_types     = std::move(bind_result.return_types);
  names            = std::move(bind_result.names);
  return make_uniq<SiriusReadParquetBindData>(uri, bind_result.total_num_rows);
}

// Execute callback for sirius_read_parquet. The real scan runs through the
// Sirius GPU pipeline; this table function is an internal rewrite target for
// read_parquet('s3://...') inside gpu_execution, NOT a user-facing function.
// A direct DuckDB (CPU) execution is rejected cleanly — query S3 Parquet via
// read_parquet('s3://...'), which the bind-time rewrite routes here for the GPU
// path. S3 has no CPU fallback: if GPU execution fails, an s3:// query errors
// (S3 CPU fallback is not supported) rather than replaying on the CPU.
void SiriusReadParquetFunction(ClientContext&, TableFunctionInput&, DataChunk&)
{
  throw std::runtime_error(
    "sirius_read_parquet is an internal rewrite target; query S3 Parquet with "
    "read_parquet('s3://...') inside gpu_execution()");
}

}  // namespace

unique_ptr<NodeStatistics> SiriusReadParquetCardinality(ClientContext&,
                                                        FunctionData const* bind_data_p)
{
  if (bind_data_p == nullptr) { return nullptr; }
  auto const* typed = dynamic_cast<SiriusReadParquetBindData const*>(bind_data_p);
  if (typed == nullptr) { return nullptr; }
  return make_uniq<NodeStatistics>(typed->total_num_rows, typed->total_num_rows);
}

struct SiriusTableFunctionData : public TableFunctionData {
  SiriusTableFunctionData() = default;
  shared_ptr<::sirius::sirius_prepared_statement_data> gpu_prepared;
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<::sirius::sirius_interface> sirius_iface;
  string query;
  // Pre-rewrite query used for the CPU fallback of LOCAL (non-s3) reads. The GPU
  // path runs the rewritten `query` (read_parquet('s3://…') ->
  // sirius_read_parquet('s3://…')), which throws if executed on the CPU, so a
  // fallback replays this original instead. For local / non-s3 queries the
  // rewrite is a no-op, so this equals `query` and replays normally. For s3://
  // queries there is no CPU fallback: run_internal_cpu_fallback_query detects the
  // s3:// read and raises a clear "S3 CPU fallback is not supported" error.
  string cpu_fallback_query;
  bool enable_optimizer;
  bool finished   = false;
  bool plan_error = false;
  // Real error message from a failed GPU plan generation. The CPU fallback path
  // surfaces it so the true cause (e.g. the unsupported operator) is preserved
  // instead of a generic placeholder.
  string plan_error_message;
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
    // STATISTICS_PROPAGATION is now enabled: cpu_source_task handles the
    // COLUMN_DATA_SCAN / EXPRESSION_GET / DUMMY_SCAN sources that this
    // optimizer produces (e.g. folding count(*), MIN, MAX to constants).
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

#ifdef SIRIUS_ENABLE_LEGACY
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
    // STATISTICS_PROPAGATION folds ungrouped MIN/MAX aggregates into constant
    // expressions using partition statistics, producing EXPRESSION_GET + DUMMY_SCAN.
    // The GPU pipeline cannot schedule COLUMN_DATA_SCAN sources, so disable this
    // to keep the query on the scan -> aggregate path where the GPU can execute it.
    disabled_optimizers.insert(OptimizerType::STATISTICS_PROPAGATION);
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
      data.res = run_internal_cpu_fallback_query(context, *data.conn, data.query);
    } else if (data.plan_error) {
      printf(
        "=============================================\nError in GPUExecuteQuery, fallback to "
        "DuckDB\n=============================================\n");
      data.res = run_internal_cpu_fallback_query(context, *data.conn, data.query);
    } else {
      data.res = data.gpu_context->GPUExecuteQuery(context, data.query, data.gpu_prepared, {});
      if (data.res->HasError()) {
        printf(
          "=============================================\nError in GPUExecuteQuery, fallback to "
          "DuckDB\n=============================================\n");
        data.res = run_internal_cpu_fallback_query(context, *data.conn, data.query);
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

static void RegisterLegacyGPUFunctions(CatalogTransaction& transaction, Catalog& catalog)
{
  TableFunction gpu_processing("gpu_processing",
                               {LogicalType::VARCHAR},
                               SiriusExtension::GPUProcessingFunction,
                               SiriusExtension::GPUProcessingBind);
  gpu_processing.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
  CreateTableFunctionInfo gpu_processing_info(gpu_processing);
  catalog.CreateTableFunction(transaction, gpu_processing_info);
}
#endif  // SIRIUS_ENABLE_LEGACY

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

  std::optional<std::string> query_label = std::nullopt;
  // take any query_label that was set using sirius_set_query_label SQL call.
  if (auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      sirius_ctx) {
    query_label = sirius_ctx->take_pending_query_label();
  }
  // however, give precedence to a query_label that was set inline in with
  // gpu_execution SQL call.
  if (auto it = input.named_parameters.find(QUERY_LABEL_PARAM_KEY);
      it != input.named_parameters.end() && not it->second.IsNull()) {
    query_label = it->second.ToString();
  }

  result->sirius_iface = make_uniq<::sirius::sirius_interface>(context, std::move(query_label));

  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_execution cannot be called with a NULL parameter");
  }

  // Route Sirius-owned remote parquet reads through Sirius's own bind:
  // read_parquet('s3://…') -> sirius_read_parquet('s3://…'). DuckDB core has no
  // S3 filesystem, so without this rewrite the s3:// bind fails before Sirius
  // ever runs. Local paths and non-s3 calls are left untouched.
  //
  // Capture the pre-rewrite query first: a CPU fallback of a LOCAL read must
  // replay the original read_parquet (not the rewritten sirius_read_parquet,
  // which throws off the GPU). An s3:// query has no CPU fallback — the fallback
  // path detects it and raises a clear error instead of replaying.
  result->cpu_fallback_query = result->query;
  result->query              = sirius::rewrite_sirius_owned_remote_parquet_calls(result->query);

  // Parse the query just to get the result type information and to create PreparedStatementData
  Parser parser(context.GetParserOptions());
  parser.ParseQuery(result->query);
  Planner planner(context);
  auto statement_type = parser.statements[0]->type;
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

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
      result->plan_error         = true;
      result->plan_error_message = error.RawMessage();
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
  auto& data = (SiriusTableFunctionData&)*data_p.bind_data;
  if (data.finished) { return; }

  if (!data.res) {
    auto start = std::chrono::high_resolution_clock::now();
    if (data.plan_error) {
      printf(
        "=============================================\nError in SiriusExecuteQuery, fallback to "
        "DuckDB\n=============================================\n");
      data.res = run_internal_cpu_fallback_query(
        context, *data.conn, data.cpu_fallback_query, data.plan_error_message);
    } else {
      data.res =
        data.sirius_iface->sirius_execute_query(context, data.query, data.gpu_prepared, {});
      if (data.res->HasError()) {
        if (Config::ENABLE_DUCKDB_FALLBACK) {
          SIRIUS_LOG_ERROR("SiriusExecuteQuery error: {}", data.res->GetError());
          printf(
            "=============================================\nError in SiriusExecuteQuery, fallback "
            "to DuckDB\n=============================================\n");
          data.res = run_internal_cpu_fallback_query(
            context, *data.conn, data.cpu_fallback_query, data.res->GetError());
        } else {
          throw std::runtime_error("SiriusExecuteQuery error: " + data.res->GetError());
          return;
        }
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

#ifdef SIRIUS_ENABLE_LEGACY
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
#endif  // SIRIUS_ENABLE_LEGACY

static unique_ptr<FunctionData> ProfilerBind(ClientContext& context,
                                             TableFunctionBindInput& input,
                                             vector<LogicalType>& return_types,
                                             vector<string>& names)
{
  return_types.push_back(LogicalType::BOOLEAN);
  names.push_back("ok");
  return nullptr;
}

struct PinTableFunctionData : public TableFunctionData {
  PinTableArgs args;
  bool finished = false;
};

unique_ptr<FunctionData> SiriusExtension::PinTableBind(ClientContext& context,
                                                       TableFunctionBindInput& input,
                                                       vector<LogicalType>& return_types,
                                                       vector<string>& names)
{
  auto result = make_uniq<PinTableFunctionData>();

  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw BinderException("pin_table requires a non-null path argument");
  }
  result->args.path = input.inputs[0].ToString();

  auto tier_it = input.named_parameters.find("tier");
  if (tier_it == input.named_parameters.end() || tier_it->second.IsNull()) {
    throw BinderException("pin_table requires a 'tier' named parameter");
  }
  result->args.tier = tier_it->second.ToString();
  if (result->args.tier != "gpu" && result->args.tier != "host") {
    throw NotImplementedException("pin_table tier='" + result->args.tier +
                                  "' is not supported (only 'gpu' and 'host')");
  }

  auto name_it = input.named_parameters.find("name");
  if (name_it == input.named_parameters.end() || name_it->second.IsNull()) {
    throw BinderException("pin_table requires a 'name' named parameter");
  }
  result->args.name = name_it->second.ToString();

  auto cols_it = input.named_parameters.find("cols");
  if (cols_it != input.named_parameters.end() && !cols_it->second.IsNull()) {
    vector<string> cols;
    for (auto& val : ListValue::GetChildren(cols_it->second)) {
      if (val.IsNull()) {
        throw BinderException("pin_table 'cols' list cannot contain NULL entries");
      }
      cols.push_back(val.ToString());
    }
    result->args.cols = std::move(cols);
  }

  auto n_rows_it = input.named_parameters.find("n_rows");
  if (n_rows_it != input.named_parameters.end() && !n_rows_it->second.IsNull()) {
    result->args.n_rows = n_rows_it->second.GetValue<int64_t>();
  }

  return_types.emplace_back(LogicalType::BOOLEAN);
  names.emplace_back("Success");
  return std::move(result);
}

void SiriusExtension::PinTableFunction(ClientContext& context,
                                       TableFunctionInput& data_p,
                                       DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<PinTableFunctionData>();
  if (data.finished) { return; }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("pin_table requires the Sirius context to be initialized");
  }

  // The cudf reader places parquet chunks on whichever GPU is current at
  // read_chunk() time; round-robin the file reads across all GPU memory spaces
  // so multi-file pin_table calls distribute their chunks evenly. Each file's
  // chunks all bind to the same GPU (per-file binding — see comment at
  // chunk_idx increment below). For tier='host' we additionally convert each
  // table to a host_data_representation (via the GPU↔HOST converter) so the
  // pinned data lives in pinned host memory.
  auto& memory_manager = sirius_ctx->get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw InvalidInputException("pin_table: no GPU memory space available");
  }

  // For host tier, build a target_gpu_id -> NUMA-local host memory_space map.
  // Each round-robin GPU's host conversion should pin its data on the host
  // memory_space whose NUMA node matches the GPU. Fall back to host_spaces[0]
  // when the GPU's NUMA node is unknown or no matching host space exists.
  std::unordered_map<int, cucascade::memory::memory_space*> host_space_by_gpu;
  if (data.args.tier == "host") {
    auto host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (host_spaces.empty()) {
      throw InvalidInputException("pin_table: no HOST memory space available");
    }
    auto* fallback_host = const_cast<cucascade::memory::memory_space*>(host_spaces[0]);
    auto const& topo    = sirius_ctx->get_config().get_hw_topology();
    for (auto const* gpu_space : gpu_spaces) {
      int const gpu_id = gpu_space->get_device_id();
      int numa_node    = -1;
      if (static_cast<size_t>(gpu_id) < topo.gpus.size()) {
        numa_node = topo.gpus[gpu_id].numa_node;
      }
      cucascade::memory::memory_space* picked = fallback_host;
      if (numa_node >= 0) {
        for (auto* hs : host_spaces) {
          if (hs->get_device_id() == numa_node) {
            picked = const_cast<cucascade::memory::memory_space*>(hs);
            break;
          }
        }
      }
      host_space_by_gpu[gpu_id] = picked;
    }
  }

  // Glob the user-supplied path into concrete files.
  auto& fs   = FileSystem::GetFileSystem(context);
  auto files = fs.GlobFiles(data.args.path);
  std::vector<std::string> file_paths;
  file_paths.reserve(files.size());
  for (auto& f : files) {
    file_paths.push_back(f.path);
  }

  // Only parquet files are supported. Validate by extension before reading.
  auto has_parquet_ext = [](std::string const& p) {
    constexpr std::string_view kExt = ".parquet";
    if (p.size() < kExt.size()) { return false; }
    auto tail = p.substr(p.size() - kExt.size());
    std::transform(
      tail.begin(), tail.end(), tail.begin(), [](unsigned char c) { return std::tolower(c); });
    return tail == kExt;
  };
  for (auto const& path : file_paths) {
    if (!has_parquet_ext(path)) {
      throw InvalidInputException("pin_table only supports parquet files, got non-parquet path: " +
                                  path);
    }
  }

  std::vector<std::string> cols =
    data.args.cols.has_value() ? *data.args.cols : std::vector<std::string>{};

  // Chunk read limit (target bytes per emitted batch) comes from operator_params.
  std::size_t const chunk_read_limit =
    sirius_ctx->get_config().get_operator_params().scan_task_batch_size;

  // Read each file via cudf::chunked_parquet_reader so each emitted batch stays within
  // chunk_read_limit bytes. The optional n_rows budget caps the cumulative row count
  // across files; once exhausted, stop early.
  std::vector<std::unique_ptr<cudf::table>> tables;
  // Parallel vector to tables — chunk_memory_spaces[i] is the memory_space*
  // for the i-th cudf::table in tables. Consumed by insert_pinned_entry's
  // precondition check (size==data_tables.size).
  std::vector<cucascade::memory::memory_space*> chunk_memory_spaces;
  // HOST-tier storage: one host_data_representation per chunk.
  std::vector<std::shared_ptr<cucascade::host_data_representation>> host_chunks;
  std::vector<std::string> read_column_names;  // captured from parquet metadata
  int64_t remaining_rows = data.args.n_rows.value_or(-1);
  // Per-call local counter (NOT std::atomic, NOT global). PinTableFunction is
  // single-threaded; new pin_table calls restart at chunk 0 → GPU 0 for
  // reproducibility.
  std::size_t chunk_idx = 0;

  // For tier='host' the full table may not fit in GPU memory, so each batch is downgraded
  // to a pinned host_data_representation immediately and the GPU buffers are released
  // before the next read_chunk(). The GPU↔HOST converter uses cudaMemcpyBatchAsync which
  // requires a real, non-default stream. Streams are constructed lazily inside the
  // per-file loop under the device guard so each cuda_stream binds to its target GPU's
  // context — a single pre-created stream binds to whichever device was current when
  // emplaced and would route D2H copies through the wrong device for round-robin files
  // that land on GPU 1+.
  cucascade::representation_converter_registry* registry_ptr = nullptr;
  std::unordered_map<int, rmm::cuda_stream> pin_streams_by_gpu;
  if (data.args.tier == "host") { registry_ptr = &sirius::converter_registry::get(); }

  for (auto const& path : file_paths) {
    if (data.args.n_rows.has_value() && remaining_rows <= 0) { break; }

    // Bind device BEFORE constructing chunked_parquet_reader so the cudf
    // allocator places footer + decompress + column buffers on the intended
    // GPU.
    auto* target_space =
      const_cast<cucascade::memory::memory_space*>(gpu_spaces[chunk_idx % gpu_spaces.size()]);
    int target_gpu_id = target_space->get_device_id();
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_gpu_id}};

    // Route the chunked_parquet_reader through the per-GPU sirius_ioctx
    // instead of cudf's bundled file_source factory (kvikio). The string-form
    // source_info routes reads through libkvikio's FileHandle which binds a
    // single CUDA context per file, breaking multi-GPU residency (the columns
    // end up on the wrong GPU under sanitizer races). Use of sirius_ioctx is
    // mandatory whenever more than one GPU is configured — enforced upstream
    // by sirius_config::enforce_sirius_datasource_for_multi_gpu().
    auto& scan_mgr  = sirius_ctx->get_scan_manager();
    auto datasource = scan_mgr.create_datasource(path);
    if (!datasource) {
      throw InvalidInputException("pin_table: no IO backend supports path: " + path);
    }

    auto file_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{datasource.get()}).build();
    if (!cols.empty()) { file_opts.set_column_names(cols); }
    if (data.args.n_rows.has_value()) { file_opts.set_num_rows(remaining_rows); }

    cudf::io::chunked_parquet_reader reader(chunk_read_limit, file_opts);
    int64_t file_rows_read = 0;
    while (reader.has_next()) {
      auto chunk      = reader.read_chunk();
      auto chunk_rows = static_cast<int64_t>(chunk.tbl->num_rows());
      if (chunk_rows == 0) { break; }
      if (read_column_names.empty()) {
        read_column_names.reserve(chunk.metadata.schema_info.size());
        for (auto const& col_info : chunk.metadata.schema_info) {
          read_column_names.push_back(col_info.name);
        }
      }
      file_rows_read += chunk_rows;
      if (data.args.tier == "host") {
        // Per-target-GPU stream + per-target-GPU host space: each round-robin
        // file's chunks must record their D2H copies on a stream bound to the
        // target GPU's context, and the destination host pinning should land
        // on the NUMA-local host memory_space for that GPU.
        // try_emplace default-constructs the rmm::cuda_stream when the key is
        // missing. The default ctor calls cudaStreamCreate under whatever
        // device is current — which is target_gpu_id by virtue of the
        // surrounding device_guard.
        auto [stream_it, inserted] = pin_streams_by_gpu.try_emplace(target_gpu_id);
        (void)inserted;
        rmm::cuda_stream_view target_stream = stream_it->second.view();
        auto* target_host_space             = host_space_by_gpu.at(target_gpu_id);
        cucascade::gpu_table_representation gpu_repr(
          std::move(chunk.tbl), *target_space, target_stream);
        auto host_repr = registry_ptr->convert<cucascade::host_data_representation>(
          gpu_repr, target_host_space, target_stream);
        // Sync before gpu_repr leaves scope so the async D2H copies finish before its
        // device buffers are freed.
        target_stream.synchronize();
        host_chunks.emplace_back(std::move(host_repr));
      } else {
        tables.emplace_back(std::move(chunk.tbl));
        chunk_memory_spaces.push_back(target_space);  // parallel to tables
      }
    }
    if (data.args.n_rows.has_value()) { remaining_rows -= file_rows_read; }
    // Per-call local counter, increments per file. All chunks within a single
    // chunked_parquet_reader stay on the same GPU (chunks-at-index-i
    // invariant). Cross-file alternation produces the round-robin.
    ++chunk_idx;
  }

  // A user-supplied n_rows budget caps row capture below the full file content.
  // The scan_manager must refuse cached reuse of such partial entries — see
  // pinned_entry::is_partial.
  bool const is_partial_pin = data.args.n_rows.has_value();
  auto scan_info            = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  scan_info->names          = std::move(read_column_names);
  scan_info->resolved_file_paths = std::move(file_paths);
  // todo(bobbi)
  if (data.args.tier == "host") {
    // entry.memory_space is metadata only; each host_chunk carries its own
    // per-GPU NUMA-local memory_space inside its host_data_representation.
    // Pass a representative (the first round-robin GPU's host space) so the
    // entry still has a non-null memory_space for diagnostics.
    int const first_gpu_id          = gpu_spaces[0]->get_device_id();
    auto* representative_host_space = host_space_by_gpu.at(first_gpu_id);
    sirius_ctx->get_scan_manager().insert_pinned_entry_host(
      data.args.name, std::move(scan_info), std::move(host_chunks), *representative_host_space);
  } else {
    sirius_ctx->get_scan_manager().insert_pinned_entry(
      data.args.name, std::move(scan_info), std::move(tables), std::move(chunk_memory_spaces));
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct UnpinTableFunctionData : public TableFunctionData {
  std::string name;
  bool finished = false;
};

unique_ptr<FunctionData> SiriusExtension::UnpinTableBind(ClientContext& context,
                                                         TableFunctionBindInput& input,
                                                         vector<LogicalType>& return_types,
                                                         vector<string>& names)
{
  auto result = make_uniq<UnpinTableFunctionData>();

  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw BinderException("unpin_table requires a non-null name argument");
  }
  result->name = input.inputs[0].ToString();

  return_types.emplace_back(LogicalType::BOOLEAN);
  names.emplace_back("Success");
  return std::move(result);
}

void SiriusExtension::UnpinTableFunction(ClientContext& context,
                                         TableFunctionInput& data_p,
                                         DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<UnpinTableFunctionData>();
  if (data.finished) { return; }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("unpin_table requires the Sirius context to be initialized");
  }
  sirius_ctx->get_scan_manager().remove_pinned_entry(data.name);

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct ProfilerFunctionData : public GlobalTableFunctionState {
  bool finished = false;
};

static unique_ptr<GlobalTableFunctionState> ProfilerInit(ClientContext& context,
                                                         TableFunctionInitInput& input)
{
  return make_uniq<ProfilerFunctionData>();
}

static void ProfilerStartFunction(ClientContext& context,
                                  TableFunctionInput& data_p,
                                  DataChunk& output)
{
  auto& data = data_p.global_state->Cast<ProfilerFunctionData>();
  if (data.finished) return;
  cudaProfilerStart();
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

static void ProfilerStopFunction(ClientContext& context,
                                 TableFunctionInput& data_p,
                                 DataChunk& output)
{
  auto& data = data_p.global_state->Cast<ProfilerFunctionData>();
  if (data.finished) return;
  cudaProfilerStop();
  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct SiriusSetQueryLabelData : public TableFunctionData {
  std::string label;
  bool finished = false;
};

static unique_ptr<FunctionData> SiriusSetQueryLabelBind(ClientContext& context,
                                                        TableFunctionBindInput& input,
                                                        vector<LogicalType>& return_types,
                                                        vector<string>& names)
{
  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw BinderException("sirius_set_query_label requires a non-NULL VARCHAR argument");
  }
  auto result   = make_uniq<SiriusSetQueryLabelData>();
  result->label = input.inputs[0].ToString();
  return_types.push_back(LogicalType::BOOLEAN);
  names.push_back("ok");
  return std::move(result);
}

static void SiriusSetQueryLabelFunction(ClientContext& context,
                                        TableFunctionInput& data_p,
                                        DataChunk& output)
{
  auto& data = data_p.bind_data->CastNoConst<SiriusSetQueryLabelData>();
  if (data.finished) { return; }

  if (auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      sirius_ctx) {
    sirius_ctx->set_pending_query_label(data.label);
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

void SiriusExtension::RegisterGPUFunctions(DatabaseInstance& instance)
{
  auto transaction = CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = Catalog::GetSystemCatalog(instance);

#ifdef SIRIUS_ENABLE_LEGACY
  TableFunction gpu_buffer_init("gpu_buffer_init",
                                {LogicalType::VARCHAR, LogicalType::VARCHAR},
                                GPUBufferInitFunction,
                                GPUBufferInitBind);
  gpu_buffer_init.named_parameters[PINNED_MEMORY_PARAM_KEY] = LogicalType::VARCHAR;
  CreateTableFunctionInfo gpu_buffer_init_info(gpu_buffer_init);
  catalog.CreateTableFunction(transaction, gpu_buffer_init_info);

  RegisterLegacyGPUFunctions(transaction, catalog);
#endif

  TableFunction gpu_execution("gpu_execution",
                              {LogicalType::VARCHAR},
                              GPUExecutionFunction,
                              SiriusExtension::GPUExecutionBind);
  gpu_execution.named_parameters["enable_optimizer"]    = LogicalType::BOOLEAN;
  gpu_execution.named_parameters[QUERY_LABEL_PARAM_KEY] = LogicalType::VARCHAR;
  CreateTableFunctionInfo gpu_execution_info(gpu_execution);
  catalog.CreateTableFunction(transaction, gpu_execution_info);

  // Sirius-owned S3 parquet entry point. gpu_execution rewrites
  // read_parquet('s3://...') to this table function so the bind runs through
  // Sirius's footer-only S3 path instead of DuckDB's native read_parquet.
  // Registered so the rewrite's output binds, but INTERNAL — not a public
  // surface: users query S3 Parquet with read_parquet('s3://...'), not this.
  TableFunction sirius_read_parquet("sirius_read_parquet",
                                    {LogicalType::VARCHAR},
                                    SiriusReadParquetFunction,
                                    SiriusReadParquetBind);
  sirius_read_parquet.cardinality = SiriusReadParquetCardinality;
  CreateTableFunctionInfo sirius_read_parquet_info(sirius_read_parquet);
  catalog.CreateTableFunction(transaction, sirius_read_parquet_info);

  TableFunction set_query_label("sirius_set_query_label",
                                {LogicalType::VARCHAR},
                                SiriusSetQueryLabelFunction,
                                SiriusSetQueryLabelBind);
  CreateTableFunctionInfo set_query_label_info(set_query_label);
  catalog.CreateTableFunction(transaction, set_query_label_info);

  // Profiler control functions for nsys --capture-range=cudaProfilerApi
  TableFunction profiler_start(
    "profiler_start", {}, ProfilerStartFunction, ProfilerBind, ProfilerInit);
  CreateTableFunctionInfo profiler_start_info(profiler_start);
  catalog.CreateTableFunction(transaction, profiler_start_info);

  TableFunction profiler_stop(
    "profiler_stop", {}, ProfilerStopFunction, ProfilerBind, ProfilerInit);
  CreateTableFunctionInfo profiler_stop_info(profiler_stop);
  catalog.CreateTableFunction(transaction, profiler_stop_info);

  TableFunction pin_table("pin_table", {LogicalType::VARCHAR}, PinTableFunction, PinTableBind);
  pin_table.named_parameters["tier"]   = LogicalType::VARCHAR;
  pin_table.named_parameters["name"]   = LogicalType::VARCHAR;
  pin_table.named_parameters["cols"]   = LogicalType::LIST(LogicalType::VARCHAR);
  pin_table.named_parameters["n_rows"] = LogicalType::BIGINT;
  CreateTableFunctionInfo pin_table_info(pin_table);
  catalog.CreateTableFunction(transaction, pin_table_info);

  TableFunction unpin_table(
    "unpin_table", {LogicalType::VARCHAR}, UnpinTableFunction, UnpinTableBind);
  CreateTableFunctionInfo unpin_table_info(unpin_table);
  catalog.CreateTableFunction(transaction, unpin_table_info);
}

static void SetUsePinMemory(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_PIN_MEM_FOR_CPU_PROCESSING = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CPU_PROCESSING to {}",
                   Config::USE_PIN_MEM_FOR_CPU_PROCESSING);
}

static void SetUsePinMemoryForCaching(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_PIN_MEM_FOR_CACHING = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CACHING to {}", Config::USE_PIN_MEM_FOR_CACHING);
}

static void SetUseCudfExpr(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::USE_CUDF_EXPR = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_CUDF_EXPR to {}", Config::USE_CUDF_EXPR);
}

static void SetExpressionExecutorStrategy(ClientContext& context, SetScope scope, Value& parameter)
{
  auto value = StringValue::Get(parameter);
  sirius::expression_executor_strategy parsed;
  if (!sirius::string_to_strategy(value, parsed)) {
    throw InvalidInputException(
      "Invalid expression_executor_strategy '{}'. Valid values: materialize, ast_interpret, "
      "ast_jit",
      value);
  }
  Config::EXPRESSION_EXECUTOR_STRATEGY = parsed;
  SIRIUS_LOG_DEBUG("Updated config EXPRESSION_EXECUTOR_STRATEGY to {}",
                   sirius::strategy_to_string(Config::EXPRESSION_EXECUTOR_STRATEGY));
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

static void SetCacheScanLevel(ClientContext& context, SetScope scope, Value& parameter)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    SIRIUS_LOG_DEBUG("SiriusContext not available; cache_scan_level SET ignored");
    return;
  }
  auto level_str = StringValue::Get(parameter);
  sirius::op::scan::cache_level level;
  if (!sirius::op::scan::string_to_enum(level_str, level)) {
    throw InvalidInputException(
      "Invalid cache_scan_level '{}'. Valid values: none, table_gpu, table_host, parquet",
      level_str);
  }
  auto& cfg = sirius_ctx->get_config();
  cfg.set_cache_level(level);
  SIRIUS_LOG_DEBUG("Updated config cache_scan_level to {}", level_str);
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

static void SetMaxSortPartitionMemoryFraction(ClientContext& context,
                                              SetScope scope,
                                              Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  const double fraction = parameter.GetValue<double>();
  if (fraction < 0.0 || fraction > 1.0) {
    throw InvalidInputException(
      "max_sort_partition_memory_fraction must be between 0.0 and 1.0, got {}", fraction);
  }
  params->max_sort_partition_memory_fraction = fraction;
  SIRIUS_LOG_DEBUG("Updated config MAX_SORT_PARTITION_MEMORY_FRACTION to {}",
                   params->max_sort_partition_memory_fraction);
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

static void SetSortSampleBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->sort_sample_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config SORT_SAMPLE_BYTES to {}", params->sort_sample_bytes);
}

static void SetLogLevel(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::LOG_LEVEL = StringValue::Get(parameter);
  SetGlobalLogLevel(Config::LOG_LEVEL);
  SIRIUS_LOG_DEBUG("Updated config LOG_LEVEL to {}", Config::LOG_LEVEL);
}

static void SetLogDir(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::LOG_DIR = StringValue::Get(parameter);
  InitGlobalLogger(Config::LOG_LEVEL, Config::LOG_DIR, Config::LOG_FLUSH_SECONDS);
  SIRIUS_LOG_DEBUG("Updated config LOG_DIR to {}", Config::LOG_DIR);
}

static void SetLogFlushSeconds(ClientContext& context, SetScope scope, Value& parameter)
{
  Config::LOG_FLUSH_SECONDS = IntegerValue::Get(parameter);
  SetGlobalLogFlush(Config::LOG_FLUSH_SECONDS);
  SIRIUS_LOG_DEBUG("Updated config LOG_FLUSH_SECONDS to {}", Config::LOG_FLUSH_SECONDS);
}

static void SetMaxBuildHashTableBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  params->max_build_hash_table_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MAX_BUILD_HASH_TABLE_BYTES to {}",
                   params->max_build_hash_table_bytes);
}

static void SetEnableGpuExecution(ClientContext& context, SetScope scope, Value& parameter)
{
  SIRIUS_LOG_DEBUG("Updated gpu_execution to {}", BooleanValue::Get(parameter));
}

void SiriusExtension::InitialGPUConfigs(DBConfig& config)
{
  // Add in config option for gpu buffer manager
  config.AddExtensionOption("use_pin_memory",
                            "Whether or not the buffer manager is initialized with pinned memory",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_PIN_MEM_FOR_CPU_PROCESSING),
                            SetUsePinMemory);

  config.AddExtensionOption(
    "use_pin_memory_for_caching",
    "Whether or not the cache buffer is allocated with pinned host memory instead of GPU memory",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(Config::USE_PIN_MEM_FOR_CACHING),
    SetUsePinMemoryForCaching);

  // Add in config option for expression executor
  config.AddExtensionOption("use_cudf_expr",
                            "Whether or not cudf is used to evaluate expressions",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::USE_CUDF_EXPR),
                            SetUseCudfExpr);

  config.AddExtensionOption(
    "expression_executor_strategy",
    "Strategy for the gpu_expression_executor: 'materialize', 'ast_interpret', or "
    "'ast_jit'",
    LogicalType::VARCHAR,
    Value(std::string(sirius::strategy_to_string(Config::EXPRESSION_EXECUTOR_STRATEGY))),
    SetExpressionExecutorStrategy);

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
                            "Maximum bytes per sort partition (0 = auto based on "
                            "max_sort_partition_memory_fraction of GPU memory)",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.max_sort_partition_bytes),
                            SetMaxSortPartitionBytes);
  config.AddExtensionOption(
    "max_sort_partition_memory_fraction",
    "Fraction of available GPU memory per sort partition when max_sort_partition_bytes is 0",
    LogicalType::DOUBLE,
    Value::DOUBLE(sirius::operator_params{}.max_sort_partition_memory_fraction),
    SetMaxSortPartitionMemoryFraction);

  // Logging configuration
  config.AddExtensionOption("sirius_log_level",
                            "Log level for Sirius (trace, debug, info, warn, error, critical, off)",
                            LogicalType::VARCHAR,
                            Value(Config::LOG_LEVEL),
                            SetLogLevel);
  config.AddExtensionOption("sirius_log_dir",
                            "Directory for Sirius log files",
                            LogicalType::VARCHAR,
                            Value(Config::LOG_DIR),
                            SetLogDir);
  config.AddExtensionOption("sirius_log_flush_seconds",
                            "Interval in seconds between automatic log flushes",
                            LogicalType::INTEGER,
                            Value::INTEGER(Config::LOG_FLUSH_SECONDS),
                            SetLogFlushSeconds);

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

  config.AddExtensionOption("sort_sample_bytes",
                            "Target bytes to sample before computing sort partition boundaries",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.sort_sample_bytes),
                            SetSortSampleBytes);

  config.AddExtensionOption("scan_cache_level",
                            "Scan result caching level: none, table_gpu, table_host, parquet",
                            LogicalType::VARCHAR,
                            Value("none"),
                            SetCacheScanLevel);

  config.AddExtensionOption("max_build_hash_table_bytes",
                            "Maximum size a build-side table can be where it will create a "
                            "reusable hash table for hash joins (i.e. BUILD_PROBE mode)",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(sirius::operator_params{}.max_build_hash_table_bytes),
                            SetMaxBuildHashTableBytes);

  config.AddExtensionOption(
    "gpu_execution",
    "Whether to transparently intercept SQL queries and execute them on GPU",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(true),
    SetEnableGpuExecution);
}

static void LoadInternal(ExtensionLoader& loader)
{
  sirius::util::install_segfault_backtrace_handler();

  auto& db           = loader.GetDatabaseInstance();
  auto& config       = DBConfig::GetConfig(db);
  auto callback      = make_shared_ptr<duckdb::SiriusContextExtensionCallback>();
  auto* callback_ptr = callback.get();
  config.GetCallbackManager().Register(std::move(callback));
  sirius::converter_registry::initialize();
  SiriusExtension::InitialGPUConfigs(config);
  SiriusExtension::RegisterGPUFunctions(db);

  // S3 CPU fallback is not supported: there is no DuckDB CPU FileSystem for
  // s3://. S3 parquet is read only on the GPU path (read_parquet('s3://…') is
  // rewritten to sirius_read_parquet -> describe_parquet -> cuDF via s3_ioctx).
  // A query that reads s3:// and fails on GPU surfaces a clear "S3 CPU fallback
  // is not supported" error (see run_internal_cpu_fallback_query); local reads
  // still fall back to DuckDB's CPU execution.

  // Register optimizer extension for transparent GPU execution.
  // Pre-hook disables incompatible optimizers; post-hook captures the plan.
  OptimizerExtension opt_ext;
  opt_ext.pre_optimize_function = sirius::transparent::sirius_pre_optimizer_hook;
  opt_ext.optimize_function     = sirius::transparent::sirius_optimizer_hook;
  OptimizerExtension::Register(config, std::move(opt_ext));

  // Register SiriusContext on connections that were opened before the extension
  // was loaded (e.g. when loaded via LOAD in Python or the CLI).
  for (auto& ctx : ConnectionManager::Get(db).GetConnectionList()) {
    callback_ptr->OnConnectionOpened(*ctx);
  }
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
