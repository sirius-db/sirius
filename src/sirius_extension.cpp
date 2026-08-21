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
#include "expression_evaluator/expression_evaluator_strategy.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

// Forward-declare CUDA profiler API functions (linked via libcudart).
extern "C" int cudaProfilerStart();
extern "C" int cudaProfilerStop();
#include "compression/compressed_representation.hpp"
#include "compression/compression_converters.hpp"
#include "compression/plan_register.hpp"
#include "data/sirius_converter_registry.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/extension_callback_manager.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/relation.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/column_list.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/storage/single_file_block_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "transparent/sirius_optimizer_extension.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>

#include <filesystem>
#include <fstream>
// #include "from_substrait.hpp"
#ifdef SIRIUS_ENABLE_LEGACY
#include "gpu_buffer_manager.hpp"
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#endif
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "op/result/host_table_chunk_reader.hpp"
#include "op/scan/duckdb_mvcc_visibility.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "pin_table.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "sirius_interface.hpp"
#include "sirius_sql_rewrite.hpp"
#include "util/segfault_backtrace.hpp"
#include "vss/pinned_column.hpp"
#include "vss/vector_search.hpp"

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// PinTableFunction routes parquet reads through the scan manager's sirius_ioctx
// instead of cudf's bundled file_source factory (which uses kvikio internally
// and binds to a single CUDA context). This is mandatory in multi-GPU
// configurations (enforced by sirius_config::enforce_sirius_datasource_for_multi_gpu()).
// Single-GPU users may still opt out via use_sirius_datasource=false; the
// pin pipeline always routes through sirius_ioctx when one is available.
//
// Ordering rule: include uring_reactor LAST among sirius headers — liburing.h
// transitively pulled by uring_reactor.hpp defines a BLOCK_SIZE preprocessor
// macro that collides with the BLOCK_SIZE static member in
// <blockingconcurrentqueue.h> (used by pipeline / duckdb
// connection_manager). All consumers of blockingconcurrentqueue.h must
// precede this include.
#include "io/s3/sirius_httpfs.hpp"     // sirius::io::s3::sirius_httpfs
#include "io/types.hpp"                // sirius::io::sirius_ioctx
#include "io/uring/uring_reactor.hpp"  // sirius::io::uring_io_object

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace duckdb {

const std::string PINNED_MEMORY_PARAM_KEY = "pinned_memory_size";
#ifdef SIRIUS_ENABLE_LEGACY
bool SiriusExtension::buffer_is_initialized = false;
#endif

constexpr std::string QUERY_LABEL_PARAM_KEY = "query_label";

namespace {

bool test_options_enabled() noexcept
{
  auto const* value = std::getenv("SIRIUS_ENABLE_TEST_OPTIONS");
  return value != nullptr && std::string_view{value} == "1";
}

enum class option_visibility { user, internal };

template <typename... Args>
void add_sirius_option(DBConfig& config,
                       option_visibility visibility,
                       const std::string& name,
                       std::string description,
                       Args&&... args)
{
  if (visibility == option_visibility::internal && !test_options_enabled()) { return; }
  if (visibility == option_visibility::internal) { description = "TEST ONLY: " + description; }
  config.AddExtensionOption(name, description, std::forward<Args>(args)...);
}

std::uint64_t count_narrowed_columns(
  sirius::pinned_column_storage_matrix const& column_storage) noexcept
{
  std::uint64_t count = 0;
  for (auto const& chunk : column_storage) {
    for (auto const& column : chunk) {
      count += column.narrowed ? 1 : 0;
    }
  }
  return count;
}

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

  // CpuFallbackGuard marks this replay so sirius_httpfs refuses to serve s3://
  // data reached indirectly (e.g. through a view) to the CPU plan — the
  // string-level references_sirius_owned_s3_parquet check above only catches a
  // literal read_parquet('s3://') in the query text. Both guards bind to the
  // TARGET executing connection (the replay runs on `connection`, not on the
  // context that issued the original query) and are no-ops when Sirius has no
  // per-connection state there.
  duckdb::SiriusContext::InternalQueryGuard guard(*connection.context);
  duckdb::SiriusContext::CpuFallbackGuard cpu_fallback_guard(*connection.context);
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
  // This bind reads the shared scan manager, which must not be consulted
  // after the runtime is latched unavailable.
  if (sirius_ctx->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE) {
    sirius_ctx->throw_runtime_unavailable();
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
  // Bind data carries ONLY re-executable input (the SQL template, schema and
  // label). The physical plan, interface, connection and result are
  // per-execution state (SiriusExecutionGlobalState): a plan built at bind time
  // would cache pin-registry pointers that a later unpin invalidates, and a
  // bind-held result cannot serve a prepared statement's second execution.
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
  // Schema captured at bind time; each execution rebuilds its
  // PreparedStatementData from these (parameterized execution is not
  // supported on this path, so no value_map is needed — same as the
  // transparent operator's minimal PreparedStatementData).
  vector<string> bind_names;
  vector<LogicalType> bind_types;
  std::optional<std::string> query_label;
  //! Original options from the connection
  ClientConfig original_config;

  void PrepareConnection(ClientContext& context)
  {
    // First collect original options
    original_config = context.config;
    // The user might want to disable the optimizer of the new connection.
    // (connection-local ClientConfig — safe to toggle per execution)
    context.config.enable_optimizer = enable_optimizer;
    // The old per-query DBConfig::disabled_optimizers save/modify/restore is
    // gone: that set is DB-global and read locklessly by every connection's
    // optimizer, so per-query mutation was an unprotected concurrent write.
    // The mask is published once at extension load
    // (publish_transparent_optimizer_mask); shapes previously avoided by the
    // DEBUG-only COLUMN_LIFETIME disable fall back to CPU via create_plan
    // rejection instead.
  }

  // Reset configuration
  void CleanupConnection(ClientContext& context) const { context.config = original_config; }

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
  result->query            = input.inputs[0].ToString();
  result->enable_optimizer = true;

  std::optional<std::string> query_label = std::nullopt;
  // take any query_label that was set using sirius_set_query_label SQL call
  // (connection-scoped: it lives on THIS connection's per-connection state).
  if (auto conn_state = get_sirius_connection_state(context)) {
    query_label = conn_state->take_pending_query_label();
  }
  // however, give precedence to a query_label that was set inline in with
  // gpu_execution SQL call.
  if (auto it = input.named_parameters.find(QUERY_LABEL_PARAM_KEY);
      it != input.named_parameters.end() && not it->second.IsNull()) {
    query_label = it->second.ToString();
  }
  // Stored on the bind data; each execution builds its own sirius_interface
  // from it (execution state is per-execution, not bind-held).
  result->query_label = std::move(query_label);

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

  // Parse the query just to get the result type information. The Sirius
  // physical plan is NOT built here: it is regenerated fresh inside each
  // execution's window (GPUExecutionFunction) so every execution observes the
  // pin registry and configuration current at execute time — a plan built at
  // bind would carry pin-registry pointers a later unpin invalidates.
  Parser parser(context.GetParserOptions());
  parser.ParseQuery(result->query);
  Planner planner(context);
  planner.CreatePlan(std::move(parser.statements[0]));
  D_ASSERT(planner.plan);

  result->bind_names = planner.names;
  result->bind_types = planner.types;

  for (auto& column : planner.names) {
    names.emplace_back(column);
  }
  for (auto& type : planner.types) {
    return_types.emplace_back(type);
  }

  return std::move(result);
}

// Per-execution state for gpu_execution(): a reusable prepared statement gets a
// FRESH instance for every execution (init_global runs once per execution),
// while the bind data above stays a pure re-executable template.
struct SiriusExecutionGlobalState : public GlobalTableFunctionState {
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<::sirius::sirius_interface> sirius_iface;
  bool finished = false;
  idx_t MaxThreads() const override { return 1; }
};

unique_ptr<GlobalTableFunctionState> SiriusExtension::GPUExecutionInitGlobal(
  ClientContext& context, TableFunctionInitInput& input)
{
  auto gstate  = make_uniq<SiriusExecutionGlobalState>();
  gstate->conn = make_uniq<Connection>(*context.db);
  return std::move(gstate);
}

void SiriusExtension::GPUExecutionFunction(ClientContext& context,
                                           TableFunctionInput& data_p,
                                           DataChunk& output)
{
  auto& data   = (SiriusTableFunctionData&)*data_p.bind_data;
  auto& gstate = data_p.global_state->Cast<SiriusExecutionGlobalState>();
  if (gstate.finished) { return; }

  if (!gstate.res) {
    auto start      = std::chrono::high_resolution_clock::now();
    auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    ErrorData gpu_error;
    bool gpu_failed                = false;
    bool runtime_unavailable_error = false;

    // The execution window: fresh plan extraction, Sirius physical plan
    // generation, execution and mandatory cleanup all happen inside one scope
    // on this thread; released before the first Fetch. The CPU fallback below
    // runs outside the window.
    {
      std::optional<duckdb::SiriusContext::StandaloneQueryScope> window;
      try {
        if (sirius_ctx) { window.emplace(*sirius_ctx, context, "gpu_execution"); }

        unique_ptr<LogicalOperator> query_plan;
        {
          // Suppress the optimizer hooks for this nested planning pass.
          duckdb::SiriusContext::InternalQueryGuard guard(context);
          query_plan = data.ExtractPlan(context);
        }
        SIRIUS_LOG_DEBUG("Query plan:\n{}", query_plan->ToString());
        auto sirius_physical_plan = SiriusGeneratePhysicalPlan(context, query_plan);
        SIRIUS_LOG_DEBUG("Done generating sirius physical plan");

        auto prepared     = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
        prepared->names   = data.bind_names;
        prepared->types   = data.bind_types;
        auto gpu_prepared = make_shared_ptr<::sirius::sirius_prepared_statement_data>(
          std::move(prepared), std::move(sirius_physical_plan));

        gstate.sirius_iface = make_uniq<::sirius::sirius_interface>(context, data.query_label);
        gstate.res          = gstate.sirius_iface->sirius_execute_query(
          context, data.query, gpu_prepared, {}, window->query_id());
        if (gstate.res->HasError()) {
          gpu_error = gstate.res->GetErrorObject();
          gstate.res.reset();
          gpu_failed = true;
        }
      } catch (duckdb::SiriusBeginWindowFailureException&) {
        // Typed begin-window failure: never a fallback candidate.
        throw;
      } catch (duckdb::SiriusRuntimeUnavailableException& e) {
        gpu_error                 = ErrorData(e);
        gpu_failed                = true;
        runtime_unavailable_error = true;
      } catch (std::exception& e) {
        gpu_error  = ErrorData(e);
        gpu_failed = true;
      }
      // Mandatory cleanup + release BEFORE any fallback/throw decision — no
      // exit may skip it; a non-std exception is handled by the window's
      // destructor backstop.
      if (window) {
        window->finish();
        window.reset();
      }
    }

    if (gpu_failed) {
      // Cancellation is never a fallback candidate; a pre-existing-unavailable
      // error on an S3 query keeps its stable typed message (no CPU fallback
      // exists for S3, so the S3 rewrite inside the fallback helper must not
      // replace it).
      if (gpu_error.Type() == ExceptionType::INTERRUPT) { gpu_error.Throw(); }
      if (runtime_unavailable_error &&
          sirius::references_sirius_owned_s3_parquet(data.cpu_fallback_query)) {
        gpu_error.Throw();
      }
      if (!duckdb_fallback_enabled(context)) {
        throw std::runtime_error("SiriusExecuteQuery error: " + gpu_error.RawMessage());
      }
      SIRIUS_LOG_ERROR("SiriusExecuteQuery error: {}", gpu_error.RawMessage());
      print_cpu_fallback_banner();
      gstate.res = run_internal_cpu_fallback_query(
        context, *gstate.conn, data.cpu_fallback_query, gpu_error.RawMessage());
    }
    auto end      = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    SIRIUS_LOG_INFO("Execute query time: {:.2f} ms", duration.count() / 1000.0);
  }

  auto result_chunk = gstate.res->Fetch();
  if (result_chunk == nullptr) {
    gstate.finished = true;
    output.SetCardinality(0);
    return;
  }

  output.Reference(*result_chunk);
  return;
}

[[maybe_unused]] static unique_ptr<LogicalOperator> OptimizePlan(ClientContext& context,
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

namespace {

// Resolve the kept-column logical indices for a pin: the positions of the
// user-requested @p cols within @p schema_names (preserving requested order), or
// identity over all columns when @p cols is absent/empty.
std::vector<std::size_t> resolve_pin_kept_indices(
  std::vector<std::string> const& schema_names, std::optional<std::vector<std::string>> const& cols)
{
  std::vector<std::size_t> keep;
  if (cols && !cols->empty()) {
    std::unordered_map<std::string, std::size_t> pos;
    pos.reserve(schema_names.size());
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      pos.emplace(schema_names[i], i);
    }
    for (auto const& c : *cols) {
      auto it = pos.find(c);
      if (it == pos.end()) {
        throw InvalidInputException("pin_table: column '" + c + "' not found in table schema");
      }
      keep.push_back(it->second);
    }
  } else {
    keep.resize(schema_names.size());
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      keep[i] = i;
    }
  }
  return keep;
}

// Build the table_info for a parquet pin. It drives the ingestible read and is the
// source cache_entry_info::from reads to build the pinned entry's cache descriptor:
// it carries the FULL schema in names/returned_types (build_scan_plan indexes them
// by primary index) plus projection_ids when a column subset is pinned, from which
// cache_entry_info::from derives the column_ids-aligned names the gather needs.
std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> build_parquet_pin_info(
  sirius::scan_manager::sirius_scan_manager& scan_mgr,
  std::vector<std::string> const& file_paths,
  std::optional<std::vector<std::string>> const& cols,
  std::size_t batch_size,
  vector<LogicalType>& pinned_column_types)
{
  using sirius::op::scan::parquet_ingestible_table_info;
  auto desc = scan_mgr.describe_parquet(file_paths.front());
  std::vector<std::string> schema_names(desc.names.begin(), desc.names.end());
  auto keep            = resolve_pin_kept_indices(schema_names, cols);
  bool const is_subset = cols && !cols->empty();

  auto info                 = std::make_unique<parquet_ingestible_table_info>();
  info->resolved_file_paths = file_paths;
  info->returned_types      = sirius::from_duckdb_vec(desc.return_types);  // full schema
  info->names               = desc.names;                                  // full schema
  for (auto idx : keep) {
    info->column_ids.emplace_back(duckdb::ColumnIndex(static_cast<duckdb::idx_t>(idx)));
    // Pin-time DuckDB type of each pinned column, in column_ids (batch-column)
    // order. Taken from the native DuckDB schema rather than round-tripped
    // through sirius::logical_type: the zone-map capture keys its type
    // allowlist on exact LogicalType identity (e.g. timestamp units).
    pinned_column_types.push_back(desc.return_types[idx]);
  }
  if (is_subset) {
    // Non-empty projection_ids forces scan_plan::is_projected() so the cudf reader
    // projects to exactly the pinned columns (identity into column_ids).
    for (std::size_t k = 0; k < keep.size(); ++k) {
      info->projection_ids.push_back(static_cast<duckdb::idx_t>(k));
    }
  }
  info->scan_output_arity      = keep.size();
  info->approximate_batch_size = batch_size;
  return info;
}

// A duckdb-native pin is a positional snapshot of the table's last-checkpointed
// disk image: a later checkpoint would compact tombstoned rows and flush transient
// appends, silently shifting rowids out from under the cache (and compressing the
// in-memory transient segments the query-time insert delta reads). Suppress both
// WAL auto-checkpoint triggers — the size threshold and the entry count — before
// the pin's metadata walk snapshots the on-disk row groups. The DBConfig is shared
// by every attached database, so this covers them all. Idempotent, and deliberately
// not restored on unpin (or when a later pin step fails): a restore would need pin
// refcounting to be safe against other pins taken meanwhile. Manual CHECKPOINT —
// and DETACH of the pinned database, whose close runs a shutdown checkpoint —
// while pinned are outside the supported contract.
void suppress_auto_checkpoint_for_pin(ClientContext& context)
{
  auto& config                       = DBConfig::GetConfig(context);
  config.options.checkpoint_wal_size = NumericLimits<idx_t>::Maximum();
  config.SetOptionByName("wal_autocheckpoint_entries", Value::UBIGINT(0));
}

// Resolve an attached duckdb table from the catalog and build its table_info for a
// pin. The .db must be ATTACHed. The pin captures the table's (catalog, schema,
// table) name from the resolved DuckTableEntry; that name tuple must match what a
// later query's scan derives, because it is the cache identity
// (cache_entry_info::can_serve_with_columns) — not the DataTable* pointer. A single
// info serves both the read and the cached entry.
std::unique_ptr<sirius::op::scan::duckdb_native_ingestible_table_info> build_duckdb_pin_info(
  ClientContext& context,
  std::string const& table_ref,
  std::string const& schema_override,
  std::optional<std::vector<std::string>> const& cols,
  std::size_t batch_size,
  vector<LogicalType>& pinned_column_types)
{
  using sirius::op::scan::duckdb_native_ingestible_table_info;

  // 'table_ref' is the (optionally schema/catalog-qualified) table name. Resolve it
  // through the catalog honoring the client's search path — so a bare name picks up
  // the current/USE'd database — yielding the same DataTable* a query-time scan binds.
  auto const qname          = duckdb::QualifiedName::Parse(table_ref);
  std::string const catalog = qname.catalog;  // empty => search path
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_override;
  std::string const& table  = qname.name;

  // Non-template catalog lookup + Cast (mirroring the pipeline converter). The
  // templated Catalog::GetEntry<DuckTableEntry> would ODR-use DuckTableEntry::Name
  // (a static constexpr inherited from TableCatalogEntry), emitting a duplicate
  // symbol against libduckdb_static at link time.
  auto& entry_base =
    duckdb::Catalog::GetEntry(context, duckdb::CatalogType::TABLE_ENTRY, catalog, schema, table);
  auto& entry         = entry_base.Cast<duckdb::DuckTableEntry>();
  auto& storage       = entry.GetStorage();
  auto const& columns = entry.GetColumns();
  auto schema_names   = columns.GetColumnNames();  // logical order
  auto schema_types   = columns.GetColumnTypes();  // logical order

  auto keep            = resolve_pin_kept_indices(schema_names, cols);
  auto const canonical = storage.GetAttached().GetStorageManager().GetDBPath();

  // Fixed-size ARRAY columns pin only when their child is a fixed-width scalar.
  // DuckDB stores such a child as a flat StandardColumnData the delta and decoder
  // paths can stage. A varchar or nested child (VARCHAR[N], ARRAY of ARRAY/STRUCT)
  // has no such path, and a varchar child passes the StandardColumnData cast that
  // would stage zero-byte descriptors. Decline it up front so the error names the
  // column instead of surfacing deep in metadata decoding.
  for (auto col : keep) {
    auto const type = sirius::from_duckdb(schema_types[col]);
    if (type.is_array() && !type.array_child().is_fixed_width()) {
      throw InvalidInputException(
        "pin_table: column '%s' of table '%s' is an ARRAY with a %s child, which duckdb-native "
        "pins do not support (only fixed-width scalar children); pin a column subset without it "
        "(cols=[...])",
        schema_names[col],
        table_ref,
        type.array_child().to_string());
    }
  }

  // Update chains version values in place, invisibly to the DELETE
  // keep-masks — a pin would serve stale values to every query until the
  // chains are folded away. Refuse; CHECKPOINT folds the chains into the
  // base data.
  {
    std::vector<duckdb::storage_t> pinned_storage_cols(keep.begin(), keep.end());
    if (sirius::op::scan::any_update_chains(
          storage, pinned_storage_cols, static_cast<std::size_t>(storage.GetTotalRows()))) {
      throw InvalidInputException(
        "pin_table: table '%s' has in-memory update chains on a pinned column; run CHECKPOINT "
        "before pinning",
        table_ref);
    }
    // Committed-but-uncheckpointed appends live in transient (in-memory)
    // segments the pin's disk-image walk cannot stage; without this check the
    // failure would surface as a decoder byte-size error deep in the pin.
    // Bulk-flushed appends (>= row-group-sized inserts) are checkpoint-shaped
    // and not detectable here.
    if (sirius::op::scan::any_uncheckpointed_appends(storage, pinned_storage_cols)) {
      throw InvalidInputException(
        "pin_table: table '%s' has uncheckpointed rows; run CHECKPOINT before pinning", table_ref);
    }
  }

  auto info     = std::make_unique<duckdb_native_ingestible_table_info>();
  info->storage = &storage;
  info->context = &context;
  info->db_path = canonical;
  // Qualified-name identity for the pin cache — derived from the resolved
  // DuckTableEntry so it matches the query-side derivation (the pipeline converter).
  info->catalog_name           = entry.ParentCatalog().GetName();
  info->schema_name            = entry.ParentSchema().name;
  info->table_name             = entry.name;
  info->approximate_batch_size = batch_size;
  // Full-schema names (logical order) so column_names() can derive the
  // column_ids-aligned view; the decoder itself ignores names.
  info->names.assign(schema_names.begin(), schema_names.end());
  info->returned_types = sirius::from_duckdb_vec(schema_types);
  for (auto col : keep) {
    info->column_ids.emplace_back(duckdb::ColumnIndex(static_cast<duckdb::idx_t>(col)));
    // Exact pin-time DuckDB type per pinned column (see build_parquet_pin_info).
    pinned_column_types.push_back(schema_types[col]);
    sirius::op::scan::projected_column pc;
    pc.is_rowid    = false;
    pc.storage_idx = duckdb::StorageIndex(static_cast<duckdb::idx_t>(col));
    info->projected_cols.push_back(pc);
    auto t = sirius::from_duckdb(schema_types[col]);
    info->projected_types.push_back(t);
    info->output_types.push_back(t);
  }
  return info;
}

}  // namespace

unique_ptr<FunctionData> SiriusExtension::PinTableBind(ClientContext& context,
                                                       TableFunctionBindInput& input,
                                                       vector<LogicalType>& return_types,
                                                       vector<string>& names)
{
  auto result = make_uniq<PinTableFunctionData>();

  // The positional path is optional: parquet uses it (file/glob); duckdb takes no
  // positional (its table is named by 'name' and resolved from the catalog).
  if (!input.inputs.empty() && !input.inputs[0].IsNull()) {
    result->args.path = input.inputs[0].ToString();
  }

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

  // Resolve the source format: an explicit 'format' parameter, else inferred from
  // the path extension (.parquet -> parquet, .db/.duckdb -> duckdb).
  auto to_lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
  };
  auto ends_with = [](std::string const& s, std::string_view suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  auto format_it = input.named_parameters.find("format");
  if (format_it != input.named_parameters.end() && !format_it->second.IsNull()) {
    result->args.format = to_lower(format_it->second.ToString());
    if (result->args.format != "parquet" && result->args.format != "duckdb") {
      throw BinderException("pin_table 'format' must be 'parquet' or 'duckdb', got '" +
                            result->args.format + "'");
    }
  } else if (!result->args.path.empty()) {
    auto lowered = to_lower(result->args.path);
    if (ends_with(lowered, ".parquet")) {
      result->args.format = "parquet";
    } else if (ends_with(lowered, ".db") || ends_with(lowered, ".duckdb")) {
      result->args.format = "duckdb";
    } else {
      throw BinderException("pin_table: cannot infer format from path '" + result->args.path +
                            "'; pass format => 'parquet' or 'duckdb'");
    }
  } else {
    throw BinderException("pin_table: provide a positional path (parquet) or format => 'duckdb'");
  }

  if (result->args.format == "parquet") {
    if (result->args.path.empty()) {
      throw BinderException("pin_table: format 'parquet' requires a positional path argument");
    }
  } else {
    // duckdb: 'name' is the (optionally qualified) table to pin, resolved from the
    // catalog — no path needed. 'schema' is a SQL reserved word, so the optional
    // schema override is the 'schema_name' parameter.
    auto schema_it = input.named_parameters.find("schema_name");
    if (schema_it != input.named_parameters.end() && !schema_it->second.IsNull()) {
      result->args.schema = schema_it->second.ToString();
    }
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

  auto pin_registry_guard = sirius_ctx->lock_pinned_table_registry();

  // Pin materialization mutates the shared scan-manager registry and drives
  // the shared GPU runtime, so it runs inside its own execution window.
  // finish() at the end of this body quiesces any transient per-query state
  // the materialization created; the pinned entries themselves persist across
  // windows.
  duckdb::SiriusContext::StandaloneQueryScope window(*sirius_ctx, context, "pin_table");

  // The read is driven by sirius::materialize_all_batches (pin_table.cpp), which
  // round-robins the materialized batches across all GPU memory spaces so a pin
  // distributes its chunks evenly. For tier='host' each materialized GPU table is
  // then converted to a host_data_representation (via the GPU<->HOST converter) so
  // the pinned data lives in pinned host memory.
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

  auto& scan_mgr = sirius_ctx->get_scan_manager();
  std::size_t const batch_size =
    sirius_ctx->get_config().get_operator_params().scan_task_batch_size;

  // materialize_all_batches round-robins reads across these GPUs and reports the
  // per-batch placement; insert_pinned_entry wants non-const memory_space*.
  std::vector<cucascade::memory::memory_space*> gpu_spaces_mut;
  gpu_spaces_mut.reserve(gpu_spaces.size());
  for (auto const* s : gpu_spaces) {
    gpu_spaces_mut.push_back(const_cast<cucascade::memory::memory_space*>(s));
  }

  // Build the ingestible (drives the metadata walk + decode) from one table_info.
  // duckdb-native has no standalone reader, so both formats go through their
  // gpu_ingestible — one read path.
  std::shared_ptr<sirius::op::scan::gpu_ingestible> ingestible;
  // Pin-time DuckDB types of the pinned columns, in column_ids (batch-column)
  // order — the zone-map capture keys its type allowlist on these exact types.
  vector<LogicalType> pinned_column_types;
  // The pin transaction's MVCC fence on the pinned table's own AttachedDatabase;
  // meaningful only for format == "duckdb" (see duckdb_mvcc_metadata::v_base).
  transaction_t duckdb_pin_v_base               = 0;
  std::uint64_t duckdb_pin_checkpoint_iteration = 0;

  if (data.args.format == "duckdb") {
    auto info = build_duckdb_pin_info(
      context, data.args.name, data.args.schema, data.args.cols, batch_size, pinned_column_types);
    // After the catalog resolution (so a bad table name fails without side
    // effects) but before make_ingestible snapshots the on-disk row groups.
    suppress_auto_checkpoint_for_pin(context);
    // Not the default database's counter: each AttachedDatabase has its own MVCC
    // start_time domain, and pins usually target an ATTACHed .db (the catalog
    // resolved by build_duckdb_pin_info), so read the fence off that catalog's
    // DuckTransaction.
    auto& pinned_catalog      = Catalog::GetCatalog(context, info->catalog_name);
    duckdb_pin_v_base         = DuckTransaction::Get(context, pinned_catalog).start_time;
    auto const* block_manager = dynamic_cast<SingleFileBlockManager const*>(
      &info->storage->GetAttached().GetStorageManager().GetBlockManager());
    if (block_manager == nullptr) {
      throw InvalidInputException("pin_table: DuckDB-native pins require a single-file database");
    }
    duckdb_pin_checkpoint_iteration = block_manager->GetCheckpointIteration();
    ingestible                      = sirius::op::scan::make_ingestible(std::move(info));
  } else {  // parquet
    auto& fs   = FileSystem::GetFileSystem(context);
    auto files = fs.GlobFiles(data.args.path);
    std::vector<std::string> file_paths;
    file_paths.reserve(files.size());
    for (auto& f : files) {
      file_paths.push_back(f.path);
    }
    if (file_paths.empty()) {
      throw InvalidInputException("pin_table: no parquet files matched path: " + data.args.path);
    }
    auto info =
      build_parquet_pin_info(scan_mgr, file_paths, data.args.cols, batch_size, pinned_column_types);
    ingestible = sirius::op::scan::make_ingestible(std::move(info));
  }

  auto const& pin_op_params      = sirius_ctx->get_config().get_operator_params();
  bool const capture_chunk_stats = pin_op_params.enable_pinned_zone_map_pruning;
  // Read from the connection running the CALL, so a table pins with the carriers that
  // connection asked for rather than whatever another connection set last.
  bool const compressed_pin = duckdb::compressed_materialization_enabled(context);
  if (!capture_chunk_stats && !compressed_pin) { pinned_column_types.clear(); }

  // Build the cache descriptor (table identity + column layout) from the
  // ingestible; it is stored on the pinned entry in place of the heavyweight
  // ingestible_table_info and drives later cache-hit matching + the gather.
  auto cache_info = sirius::scan_manager::cache_entry_info::from(ingestible->table_info());

  // Compression config (tier-agnostic): load the per-table plan DSL from the plan
  // directory (if configured), then resolve it into a compression_pin_config. Both
  // the host and GPU pin paths compress with this when enabled.
  const auto& comp_cfg = sirius_ctx->get_config().get_compression_config();
  if (comp_cfg.enable_pin_table_compression && comp_cfg.input_plan_dir.empty()) {
    SIRIUS_LOG_WARN(
      "[pin_table] '{}': pin_table_compression is enabled but "
      "pin_table_input_compression_plan_dir is empty; pinning uncompressed",
      data.args.name);
  }
  const bool comp_globally_enabled =
    comp_cfg.enable_pin_table_compression && !comp_cfg.input_plan_dir.empty();
  if (comp_globally_enabled) {
    namespace fs     = std::filesystem;
    const auto& name = data.args.name;
    if (!sirius::compression::plan_register::global().resolve_table_plan(name).has_value()) {
      std::error_code ec;
      for (auto const& entry : fs::directory_iterator(comp_cfg.input_plan_dir, ec)) {
        if (!entry.is_regular_file()) { continue; }
        if (entry.path().stem() == name) {
          std::ifstream f(entry.path());
          std::string dsl((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
          if (!dsl.empty()) {
            sirius::compression::plan_register::global().set_table_plan(name, std::move(dsl));
          }
          break;
        }
      }
      if (ec) {
        SIRIUS_LOG_WARN("[pin_table] cannot scan plan dir '{}': {}; skipping compression",
                        comp_cfg.input_plan_dir,
                        ec.message());
      }
    }
  }

  sirius::compression_pin_config pin_comp{};
  if (comp_globally_enabled) {
    if (auto plan_dsl =
          sirius::compression::plan_register::global().resolve_table_plan(data.args.name);
        plan_dsl.has_value()) {
      // The plan file carries one block per full-table column (schema order). A pin
      // may cache only a subset, so select the blocks for the pinned columns by their
      // full-table index (cache_info.column_ids, in pinned order) — the result lines
      // up 1:1 with the pinned table that compress_with_plan sees.
      std::vector<std::size_t> col_indices;
      col_indices.reserve(cache_info.column_ids.size());
      for (auto const& cid : cache_info.column_ids) {
        col_indices.push_back(static_cast<std::size_t>(cid.GetPrimaryIndex()));
      }
      auto selected = sirius::compression::select_plan_blocks(*plan_dsl, col_indices);
      if (selected.has_value()) {
        pin_comp.enabled                 = true;
        pin_comp.plan_dsl                = std::move(*selected);
        pin_comp.min_batch_size_bytes    = comp_cfg.min_batch_size_bytes;
        pin_comp.max_compressed_fraction = comp_cfg.max_compressed_fraction;
        pin_comp.column_names            = cache_info.column_names();
        SIRIUS_LOG_INFO("[pin_table] '{}' tier={}: compressing with plan for {} column(s)",
                        data.args.name,
                        data.args.tier,
                        pin_comp.column_names.size());
      } else {
        SIRIUS_LOG_WARN(
          "[pin_table] '{}': plan file does not cover all pinned columns; pinning uncompressed",
          data.args.name);
      }
    } else {
      SIRIUS_LOG_WARN(
        "[pin_table] '{}': pin_table_compression is enabled but no plan file was found in '{}'; "
        "pinning uncompressed",
        data.args.name,
        comp_cfg.input_plan_dir);
    }
  }

  auto attach_duckdb_mvcc_metadata = [&](std::vector<std::size_t> base_row_count_per_chunk) {
    if (data.args.format != "duckdb") { return; }
    sirius::scan_manager::duckdb_mvcc_metadata mvcc;
    mvcc.v_base                   = duckdb_pin_v_base;
    mvcc.base_row_count_per_chunk = std::move(base_row_count_per_chunk);
    mvcc.checkpoint_iteration     = duckdb_pin_checkpoint_iteration;
    scan_mgr.attach_mvcc_metadata(data.args.name, std::move(mvcc));
  };

  if (data.args.tier == "host") {
    // Stream each batch GPU->host: materialize one batch on its round-robin GPU, convert it
    // to a pinned host representation (compressed when it qualifies) on that GPU's NUMA-local
    // host space, then free the GPU table before materializing the next. Peak GPU residency
    // stays at ~one batch, so the whole table never needs to fit in GPU memory. On multi-GPU
    // the chunks land round-robin across NUMA nodes; the cached-serve path then reads each
    // chunk back on a NUMA-local GPU.
    auto host_result =
      sirius::materialize_pin_to_host(*ingestible,
                                      gpu_spaces_mut,
                                      host_space_by_gpu,
                                      *scan_mgr.io_ctx(),
                                      pinned_column_types,
                                      pin_comp,
                                      {.capture_chunk_stats               = capture_chunk_stats,
                                       .enable_compressed_materialization = compressed_pin});
    sirius_ctx->record_compressed_materialization_pin_columns_narrowed(
      count_narrowed_columns(host_result.column_storage));
    // entry.memory_space is metadata only; each host_chunk carries its own per-GPU
    // NUMA-local memory_space. Pass a representative (the first GPU's host space).
    int const first_gpu_id          = gpu_spaces_mut[0]->get_device_id();
    auto* representative_host_space = host_space_by_gpu.at(first_gpu_id);

    scan_mgr.insert_pinned_entry_host(data.args.name,
                                      std::move(cache_info),
                                      std::move(host_result.chunks),
                                      *representative_host_space,
                                      std::move(pinned_column_types),
                                      std::move(host_result.chunk_stats),
                                      std::move(host_result.column_storage));
    attach_duckdb_mvcc_metadata(std::move(host_result.base_row_count_per_chunk));
  } else if (pin_comp.enabled) {
    // GPU tier, compression enabled: narrow each materialized batch (when narrowing is
    // on), then compress it when it qualifies, keeping the compressed payload in device
    // memory; batches that do not qualify are pinned uncompressed. Both forms land in
    // one ordered chunk vector, so a table that mixes them pins without special-casing.
    auto dev_result = sirius::materialize_all_batches_compressed(
      *ingestible,
      gpu_spaces_mut,
      *scan_mgr.io_ctx(),
      pinned_column_types,
      pin_comp,
      {.capture_chunk_stats = false, .enable_compressed_materialization = compressed_pin});
    sirius_ctx->record_compressed_materialization_pin_columns_narrowed(
      count_narrowed_columns(dev_result.column_storage));

    scan_mgr.insert_pinned_entry_device(data.args.name,
                                        std::move(cache_info),
                                        std::move(dev_result.chunks),
                                        *gpu_spaces_mut[0],
                                        std::move(dev_result.column_storage));
    attach_duckdb_mvcc_metadata(std::move(dev_result.base_row_count_per_chunk));
  } else {
    // GPU tier, uncompressed: materialize every batch as a GPU-resident cudf::table
    // (with its GPU placement) and pin them in place.
    auto mat =
      sirius::materialize_all_batches(*ingestible,
                                      gpu_spaces_mut,
                                      *scan_mgr.io_ctx(),
                                      pinned_column_types,
                                      {.capture_chunk_stats               = capture_chunk_stats,
                                       .enable_compressed_materialization = compressed_pin});
    sirius_ctx->record_compressed_materialization_pin_columns_narrowed(
      count_narrowed_columns(mat.column_storage));
    auto base_row_count_per_chunk = std::move(mat.base_row_count_per_chunk);
    scan_mgr.insert_pinned_entry(data.args.name,
                                 std::move(cache_info),
                                 std::move(mat.tables),
                                 std::move(mat.chunk_memory_spaces),
                                 std::move(pinned_column_types),
                                 std::move(mat.chunk_stats),
                                 std::move(mat.column_storage));
    attach_duckdb_mvcc_metadata(std::move(base_row_count_per_chunk));
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
  window.finish();
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
  auto pin_registry_guard = sirius_ctx->lock_pinned_table_registry();
  {
    // Registry removal must be serialized against execution windows (plan
    // generation reads pinned entries); a lock-only guard suffices — unpin
    // creates no per-query runtime state to clean.
    duckdb::SiriusContext::SlotGuard slot(*sirius_ctx, context);
    sirius_ctx->get_scan_manager().remove_pinned_entry(data.name);
  }

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

  if (auto conn_state = get_sirius_connection_state(context)) {
    conn_state->set_pending_query_label(data.label);
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BOOLEAN(true));
  data.finished = true;
}

struct SiriusVectorSearchBindData : public TableFunctionData {
  sirius::vss::vector_search_request req;
  // Output column types + trailing distance, for the host_table_chunk_reader.
  duckdb::vector<sirius::logical_type> reader_types;
};

struct SiriusVectorSearchGlobalState : public GlobalTableFunctionState {
  std::unique_ptr<cucascade::host_data_representation> host_repr;
  std::unique_ptr<sirius::op::result::host_table_chunk_reader> reader;
};

// Pull the float components out of the query argument, accepting either a
// FLOAT[] ARRAY (e.g. [1,2,3]::FLOAT[3]) or a LIST of numbers.
static std::vector<float> vector_search_query_floats(const Value& query)
{
  auto const id                         = query.type().id();
  const duckdb::vector<Value>* children = nullptr;
  if (id == LogicalTypeId::ARRAY) {
    children = &ArrayValue::GetChildren(query);
  } else if (id == LogicalTypeId::LIST) {
    children = &ListValue::GetChildren(query);
  } else {
    throw BinderException(
      "sirius_knn_search: query (3rd argument) must be a FLOAT array, e.g. [..]::FLOAT[N]");
  }
  std::vector<float> out;
  out.reserve(children->size());
  for (auto const& child : *children) {
    if (child.IsNull()) {
      throw BinderException("sirius_knn_search: query vector must not contain NULLs");
    }
    out.push_back(child.GetValue<float>());
  }
  return out;
}

static unique_ptr<FunctionData> SiriusVectorSearchBind(ClientContext& context,
                                                       TableFunctionBindInput& input,
                                                       vector<LogicalType>& return_types,
                                                       vector<string>& names)
{
  auto result = make_uniq<SiriusVectorSearchBindData>();
  auto& req   = result->req;

  // Required params
  if (input.inputs.size() < 3 || input.inputs[0].IsNull() || input.inputs[1].IsNull() ||
      input.inputs[2].IsNull()) {
    throw BinderException(
      "sirius_knn_search requires three non-NULL positional arguments: table, column, query");
  }
  req.table_name  = input.inputs[0].ToString();
  req.column_name = input.inputs[1].ToString();
  req.query       = vector_search_query_floats(input.inputs[2]);

  // Optional params' default values
  req.metric                    = "l2";
  req.k                         = 10;
  std::string schema_name       = "main";
  bool output_columns_specified = false;
  for (auto& kv : input.named_parameters) {
    auto const key = StringUtil::Lower(kv.first);
    if (kv.second.IsNull()) {
      throw BinderException("sirius_knn_search: named parameter '" + kv.first + "' cannot be NULL");
    }
    if (key == "k") {
      req.k = kv.second.GetValue<int64_t>();
    } else if (key == "metric") {
      req.metric = StringUtil::Lower(kv.second.ToString());
    } else if (key == "schema_name") {
      schema_name = kv.second.ToString();
    } else if (key == "output_columns") {
      output_columns_specified = true;
      for (auto const& c : ListValue::GetChildren(kv.second)) {
        req.output_columns.push_back(c.ToString());
      }
    }
  }
  if (req.k <= 0) { throw BinderException("sirius_knn_search: k must be >= 1"); }
  if (req.metric != "l2" && req.metric != "cosine") {
    throw BinderException("sirius_knn_search: metric must be one of 'l2', 'cosine', got '" +
                          req.metric + "'");
  }
  // An explicitly-passed empty list is a user error.
  if (output_columns_specified && req.output_columns.empty()) {
    throw BinderException(
      "sirius_knn_search: output_columns cannot be empty; omit it to default to the pinned "
      "columns");
  }

  // Resolve the vector column's dimensionality and each output column's type
  // from the catalog so the return schema and the host reader agree.
  auto const qname          = QualifiedName::Parse(req.table_name);
  std::string const catalog = qname.catalog;
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_name;
  auto& entry_base =
    Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, catalog, schema, qname.name);
  auto& entry             = entry_base.Cast<DuckTableEntry>();
  req.catalog             = entry.ParentCatalog().GetName();
  req.schema              = entry.ParentSchema().name;
  req.table_name          = entry.name;  // catalog-resolved name (matches query-side derivation)
  auto const& columns     = entry.GetColumns();
  auto const schema_names = columns.GetColumnNames();
  auto const schema_types = columns.GetColumnTypes();

  // The search gathers output columns from the pin.
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("sirius_knn_search requires the Sirius context to be initialized");
  }
  const auto* pin = sirius_ctx->get_scan_manager().find_pinned_entry_for_duckdb_table(
    req.catalog, req.schema, req.table_name);
  if (pin == nullptr) {
    throw BinderException("sirius_knn_search: table '" + req.table_name +
                          "' must be pinned before it can be searched");
  }
  auto const& pinned_names = pin->cache_info.column_names();
  auto is_pinned           = [&](const std::string& col) {
    return std::find(pinned_names.begin(), pinned_names.end(), col) != pinned_names.end();
  };

  if (req.output_columns.empty()) {
    // Default to the columns that are pinned and in catalog schema order.
    for (auto const& name : schema_names) {
      if (is_pinned(name)) { req.output_columns.push_back(name); }
    }
  } else {
    // Explicitly-pass: every column must exist and be pinned.
    for (auto const& col : req.output_columns) {
      bool const in_catalog =
        std::find(schema_names.begin(), schema_names.end(), col) != schema_names.end();
      if (!in_catalog) {
        throw BinderException("sirius_knn_search: column '" + col + "' not found in table '" +
                              req.table_name + "'");
      }
      if (!is_pinned(col)) {
        throw BinderException("sirius_knn_search: output column '" + col +
                              "' is not pinned on table '" + req.table_name +
                              "'; pin it (pin_table cols => [...]) or omit output_columns");
      }
    }
  }

  auto type_of = [&](const std::string& col) -> const LogicalType& {
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      if (schema_names[i] == col) { return schema_types[i]; }
    }
    throw BinderException("sirius_knn_search: column '" + col + "' not found in table '" +
                          req.table_name + "'");
  };

  auto const& vec_type = type_of(req.column_name);
  if (vec_type.id() != LogicalTypeId::ARRAY ||
      ArrayType::GetChildType(vec_type).id() != LogicalTypeId::FLOAT) {
    throw BinderException("sirius_knn_search: column '" + req.column_name +
                          "' must be a FLOAT[N] array column");
  }
  req.dim = static_cast<int64_t>(ArrayType::GetSize(vec_type));
  if (static_cast<int64_t>(req.query.size()) != req.dim) {
    throw BinderException("sirius_knn_search: query has " + std::to_string(req.query.size()) +
                          " elements but column '" + req.column_name + "' is FLOAT[" +
                          std::to_string(req.dim) + "]");
  }

  for (auto const& col : req.output_columns) {
    return_types.push_back(type_of(col));
    names.push_back(col);
  }
  return_types.push_back(LogicalType::FLOAT);
  names.push_back("distance");

  result->reader_types = sirius::from_duckdb_vec(return_types);
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState> SiriusVectorSearchInit(ClientContext& context,
                                                                   TableFunctionInitInput& input)
{
  nvtx3::scoped_range nvtx_range{"SiriusVectorSearchInit"};
  auto& bind_data = input.bind_data->Cast<SiriusVectorSearchBindData>();

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw InvalidInputException("sirius_knn_search requires the Sirius context to be initialized");
  }

  auto state       = make_uniq<SiriusVectorSearchGlobalState>();
  state->host_repr = sirius::vss::run_vector_search(*sirius_ctx, bind_data.req);
  state->reader    = std::make_unique<sirius::op::result::host_table_chunk_reader>(
    context, *state->host_repr, bind_data.reader_types);
  return std::move(state);
}

static void SiriusVectorSearchFunction(ClientContext& context,
                                       TableFunctionInput& data_p,
                                       DataChunk& output)
{
  auto& state = data_p.global_state->Cast<SiriusVectorSearchGlobalState>();
  state.reader->get_next_chunk(output);
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
                              SiriusExtension::GPUExecutionBind,
                              SiriusExtension::GPUExecutionInitGlobal);
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
  sirius_read_parquet.cardinality         = SiriusReadParquetCardinality;
  sirius_read_parquet.projection_pushdown = true;
  sirius_read_parquet.filter_pushdown     = true;
  sirius_read_parquet.filter_prune        = true;
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

  // pin_table takes either a positional path (parquet) or no positional (duckdb,
  // where 'name' is the catalog table reference) — register both arities as a set.
  TableFunctionSet pin_table_set("pin_table");
  auto add_pin_table_overload = [&](vector<LogicalType> positional_args) {
    TableFunction pin_table(
      "pin_table", std::move(positional_args), PinTableFunction, PinTableBind);
    pin_table.named_parameters["tier"]        = LogicalType::VARCHAR;
    pin_table.named_parameters["name"]        = LogicalType::VARCHAR;
    pin_table.named_parameters["cols"]        = LogicalType::LIST(LogicalType::VARCHAR);
    pin_table.named_parameters["format"]      = LogicalType::VARCHAR;
    pin_table.named_parameters["schema_name"] = LogicalType::VARCHAR;
    pin_table_set.AddFunction(std::move(pin_table));
  };
  add_pin_table_overload({LogicalType::VARCHAR});
  add_pin_table_overload({});
  CreateTableFunctionInfo pin_table_info(pin_table_set);
  catalog.CreateTableFunction(transaction, pin_table_info);

  TableFunction unpin_table(
    "unpin_table", {LogicalType::VARCHAR}, UnpinTableFunction, UnpinTableBind);
  CreateTableFunctionInfo unpin_table_info(unpin_table);
  catalog.CreateTableFunction(transaction, unpin_table_info);

  // sirius_knn_search(table, column, query, k =>, output_columns =>, metric =>,
  // schema_name =>)
  TableFunction vector_search("sirius_knn_search",
                              {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::ANY},
                              SiriusVectorSearchFunction,
                              SiriusVectorSearchBind,
                              SiriusVectorSearchInit);
  vector_search.named_parameters["k"]              = LogicalType::BIGINT;
  vector_search.named_parameters["output_columns"] = LogicalType::LIST(LogicalType::VARCHAR);
  vector_search.named_parameters["metric"]         = LogicalType::VARCHAR;
  vector_search.named_parameters["schema_name"]    = LogicalType::VARCHAR;
  CreateTableFunctionInfo vector_search_info(vector_search);
  catalog.CreateTableFunction(transaction, vector_search_info);
}

// Process-global Config writes are refused once the Sirius runtime is
// latched unavailable (stable, session-preserving error). Connection-local
// settings (gpu_execution, enable_duckdb_fallback) and operator_params
// setters are not gated here; the latter serialize via
// lock_operator_params_slot instead.
static void throw_if_sirius_runtime_unavailable(ClientContext& context)
{
  if (auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      sirius_ctx &&
      sirius_ctx->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE) {
    sirius_ctx->throw_runtime_unavailable();
  }
}

#ifdef SIRIUS_ENABLE_LEGACY
static void SetUsePinMemory(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::USE_PIN_MEM_FOR_CPU_PROCESSING = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CPU_PROCESSING to {}",
                   Config::USE_PIN_MEM_FOR_CPU_PROCESSING);
}

static void SetUsePinMemoryForCaching(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::USE_PIN_MEM_FOR_CACHING = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_PIN_MEM_FOR_CACHING to {}", Config::USE_PIN_MEM_FOR_CACHING);
}

static void SetUseCudfExpr(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::USE_CUDF_EXPR = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_CUDF_EXPR to {}", Config::USE_CUDF_EXPR);
}
#endif

static void ApplyExpressionEvaluatorStrategy(const std::string& value)
{
  sirius::expression_evaluator_strategy parsed;
  if (!sirius::string_to_strategy(value, parsed)) {
    throw InvalidInputException(
      "Invalid expression_evaluator_strategy '%s'. Valid values: materialize, ast_interpret, "
      "ast_jit",
      value);
  }
  Config::EXPRESSION_EVALUATOR_STRATEGY = parsed;
  SIRIUS_LOG_DEBUG("Updated config EXPRESSION_EVALUATOR_STRATEGY to {}",
                   sirius::strategy_to_string(Config::EXPRESSION_EVALUATOR_STRATEGY));
}

static void SetExpressionEvaluatorStrategy(ClientContext& context, SetScope scope, Value& parameter)
{
  // Writes the process-global strategy Config — gated like every other
  // process-global setter (the deprecated alias below carries its own gate).
  throw_if_sirius_runtime_unavailable(context);
  ApplyExpressionEvaluatorStrategy(StringValue::Get(parameter));
}

// Deprecated alias for `expression_evaluator_strategy`. Kept so existing
// `SET expression_executor_strategy=...` statements keep working; remove in a future release.
static void SetExpressionExecutorStrategyDeprecated(ClientContext& context,
                                                    SetScope scope,
                                                    Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  SIRIUS_LOG_WARN(
    "The 'expression_executor_strategy' setting is deprecated; use "
    "'expression_evaluator_strategy' instead.");
  ApplyExpressionEvaluatorStrategy(StringValue::Get(parameter));
}

#ifdef SIRIUS_ENABLE_LEGACY
static void SetUseCustomTopN(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::USE_CUSTOM_TOP_N = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_CUSTOM_TOP_N to {}", Config::USE_CUSTOM_TOP_N);
}

static void SetUseOptTableScan(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::USE_OPT_TABLE_SCAN = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config USE_OPT_TABLE_SCAN to {}", Config::USE_OPT_TABLE_SCAN);
}

static void SetOptTableScanNumStreams(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS = IntegerValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_NUM_CUDA_STREAMS to {}",
                   Config::OPT_TABLE_SCAN_NUM_CUDA_STREAMS);
}

static void SetOptTableScanMemcpySize(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE to {}",
                   Config::OPT_TABLE_SCAN_CUDA_MEMCPY_SIZE);
}

static void SetPrintGPUTableMaxRows(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::PRINT_GPU_TABLE_MAX_ROWS = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config PRINT_GPU_TABLE_MAX_ROWS to {}",
                   Config::PRINT_GPU_TABLE_MAX_ROWS);
}

static void SetEnableFallbackCheck(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::ENABLE_FALLBACK_CHECK = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_FALLBACK_CHECK to {}", Config::ENABLE_FALLBACK_CHECK);
}
#endif

static void SetEnableDuckdbFallback(ClientContext& /*context*/,
                                    SetScope /*scope*/,
                                    Value& /*parameter*/)
{
  // No process-global mirror is kept.  DuckDB stores the value per-context and it
  // is read via duckdb_fallback_enabled() -> ClientContext::TryGetCurrentSetting,
  // so `SET enable_duckdb_fallback = ...` on one connection stays scoped to that
  // connection instead of leaking to every other connection (and, across the test
  // binary, to later test cases that create their own database).
}

static void SetEnableRegexJitImpl(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::ENABLE_REGEX_JIT_IMPL = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_REGEX_JIT_IMPL to {}", Config::ENABLE_REGEX_JIT_IMPL);
}

#ifdef SIRIUS_ENABLE_LEGACY
static void SetModifiedPipeline(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::MODIFIED_PIPELINE = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MODIFIED_PIPELINE to {}", Config::MODIFIED_PIPELINE);
}
#endif

static void SetFuseMergePipelines(ClientContext& /*context*/,
                                  SetScope /*scope*/,
                                  Value& /*parameter*/)
{
  // DuckDB stores this setting in the client context.
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

// operator_params are read by plan generation and the engine inside held
// execution windows, so each setter serializes its write by holding the slot
// for its single callback body. The guard is taken here in the setters,
// deliberately not inside get_operator_params(), which is also safe to call
// from code already inside a window (a helper-held lock would trip the
// same-thread reacquire check there).
static duckdb::unique_ptr<duckdb::SiriusContext::SlotGuard> lock_operator_params_slot(
  ClientContext& context)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return nullptr; }
  return duckdb::make_uniq<duckdb::SiriusContext::SlotGuard>(*sirius_ctx, context);
}

static void SetDefaultScanTaskBatchSize(ClientContext& context, SetScope scope, Value& parameter)
{
  auto const bytes = UBigIntValue::Get(parameter);
  if (bytes == 0) { throw InvalidInputException("scan_task_batch_size must be greater than zero"); }
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                    = lock_operator_params_slot(context);
  params->scan_task_batch_size = bytes;
  SIRIUS_LOG_DEBUG("Updated config SCAN_TASK_BATCH_SIZE to {}", params->scan_task_batch_size);
}

static void SetMaxSortPartitionBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                        = lock_operator_params_slot(context);
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
  auto slot             = lock_operator_params_slot(context);
  const double fraction = parameter.GetValue<double>();
  if (fraction < 0.0 || fraction > 1.0) {
    throw InvalidInputException(
      "max_sort_partition_memory_fraction must be between 0.0 and 1.0, got %f", fraction);
  }
  params->max_sort_partition_memory_fraction = fraction;
  SIRIUS_LOG_DEBUG("Updated config MAX_SORT_PARTITION_MEMORY_FRACTION to {}",
                   params->max_sort_partition_memory_fraction);
}

static void SetHashPartitionBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto const bytes = UBigIntValue::Get(parameter);
  if (bytes == 0) { throw InvalidInputException("hash_partition_bytes must be greater than zero"); }
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                    = lock_operator_params_slot(context);
  params->hash_partition_bytes = bytes;
  SIRIUS_LOG_DEBUG("Updated config HASH_PARTITION_BYTES to {}", params->hash_partition_bytes);
}

static void SetConcatBatchBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                  = lock_operator_params_slot(context);
  params->concat_batch_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config CONCAT_BATCH_BYTES to {}", params->concat_batch_bytes);
}

static void SetSortSampleBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                 = lock_operator_params_slot(context);
  params->sort_sample_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config SORT_SAMPLE_BYTES to {}", params->sort_sample_bytes);
}

static void SetLogBackend(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  auto backend = StringValue::Get(parameter);
  if (backend != "duckdb" && backend != "spdlog" && backend != "noop") {
    throw InvalidInputException("Unknown sirius_log_backend '%s' (expected: duckdb, spdlog, noop)",
                                backend);
  }
  Config::LOG_BACKEND = std::move(backend);
  install_configured_log_sink(context.db.get());
  SIRIUS_LOG_DEBUG("Updated config LOG_BACKEND to {}", Config::LOG_BACKEND);
}

static void SetLogLevel(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::LOG_LEVEL = StringValue::Get(parameter);
  // Only re-targets the current sink; no rebuild (a no-op for the duckdb backend).
  auto parsed_level = sirius::log::string_to_enum(Config::LOG_LEVEL);
  sirius::log::get_sink()->set_level(parsed_level.value_or(sirius::log::level::info));
  if (!parsed_level) {
    SIRIUS_LOG_WARN("Unknown log level '{}', defaulting to info", Config::LOG_LEVEL);
  }
  SIRIUS_LOG_DEBUG("Updated config LOG_LEVEL to {}", Config::LOG_LEVEL);
}

static void SetLogDir(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::LOG_DIR = StringValue::Get(parameter);
  // log_dir only affects the spdlog backend; rebuild it when that one is active.
  if (Config::LOG_BACKEND == "spdlog") { install_configured_log_sink(context.db.get()); }
  SIRIUS_LOG_DEBUG("Updated config LOG_DIR to {}", Config::LOG_DIR);
}

static void SetLogFlushSeconds(ClientContext& context, SetScope scope, Value& parameter)
{
  throw_if_sirius_runtime_unavailable(context);
  Config::LOG_FLUSH_SECONDS = IntegerValue::Get(parameter);
  // The flush interval is fixed at spdlog-sink construction, so rebuild it (only
  // the spdlog backend uses it).
  if (Config::LOG_BACKEND == "spdlog") { install_configured_log_sink(context.db.get()); }
  SIRIUS_LOG_DEBUG("Updated config LOG_FLUSH_SECONDS to {}", Config::LOG_FLUSH_SECONDS);
}

static void SetMaxBuildHashTableBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                          = lock_operator_params_slot(context);
  params->max_build_hash_table_bytes = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MAX_BUILD_HASH_TABLE_BYTES to {}",
                   params->max_build_hash_table_bytes);
}

static void SetMaxBroadcastJoinSize(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                       = lock_operator_params_slot(context);
  params->max_broadcast_join_size = UBigIntValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config MAX_BROADCAST_JOIN_SIZE to {}", params->max_broadcast_join_size);
}

static void SetMarkJoinBuildSwitchRatio(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot          = lock_operator_params_slot(context);
  const double ratio = parameter.GetValue<double>();
  if (!(ratio >= 0.0)) {
    throw InvalidInputException("mark_join_build_switch_ratio must be >= 0.0, got %f", ratio);
  }
  params->mark_join_build_switch_ratio = ratio;
  SIRIUS_LOG_DEBUG("Updated config MARK_JOIN_BUILD_SWITCH_RATIO to {}",
                   params->mark_join_build_switch_ratio);
}

static void SetEnableGpuExecution(ClientContext& context, SetScope scope, Value& parameter)
{
  SIRIUS_LOG_DEBUG("Updated gpu_execution to {}", BooleanValue::Get(parameter));
}

static void SetEnablePinTableCompression(ClientContext& context, SetScope scope, Value& parameter)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return; }
  sirius_ctx->get_config().get_compression_config().enable_pin_table_compression =
    BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated pin_table_compression to {}", BooleanValue::Get(parameter));
}

static void SetPinTableInputCompressionPlanDir(ClientContext& context,
                                               SetScope scope,
                                               Value& parameter)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return; }
  sirius_ctx->get_config().get_compression_config().input_plan_dir = StringValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated pin_table_input_compression_plan_dir");
}

static void SetPinTableCompressionMinBatchSizeBytes(ClientContext& context,
                                                    SetScope scope,
                                                    Value& parameter)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return; }
  sirius_ctx->get_config().get_compression_config().min_batch_size_bytes =
    static_cast<std::size_t>(UBigIntValue::Get(parameter));
  SIRIUS_LOG_DEBUG("Updated pin_table_compression_min_batch_size_bytes");
}

static void SetPinTableCompressionMaxCompressedFraction(ClientContext& context,
                                                        SetScope scope,
                                                        Value& parameter)
{
  const double fraction = DoubleValue::Get(parameter);
  if (!std::isfinite(fraction) || fraction < 0.0) {
    throw InvalidInputException(
      "pin_table_compression_max_compressed_fraction must be finite and non-negative, got %f",
      fraction);
  }
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) { return; }
  sirius_ctx->get_config().get_compression_config().max_compressed_fraction = fraction;
  SIRIUS_LOG_DEBUG("Updated pin_table_compression_max_compressed_fraction");
}

static void SetEnableRuntimeDistinctBuildProbe(ClientContext& context,
                                               SetScope scope,
                                               Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                                   = lock_operator_params_slot(context);
  params->enable_runtime_distinct_build_probe = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_RUNTIME_DISTINCT_BUILD_PROBE to {}",
                   params->enable_runtime_distinct_build_probe);
}

static void SetEnableDynamicFilter(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                     = lock_operator_params_slot(context);
  params->enable_dynamic_filter = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_DYNAMIC_FILTER to {}", params->enable_dynamic_filter);
}

static void SetEnableDynamicZoneMapFilter(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                              = lock_operator_params_slot(context);
  params->enable_dynamic_zone_map_filter = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_DYNAMIC_ZONE_MAP_FILTER to {}",
                   params->enable_dynamic_zone_map_filter);
}

static void SetDynamicFilterDomainCoverageThreshold(ClientContext& context,
                                                    SetScope scope,
                                                    Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot              = lock_operator_params_slot(context);
  const double threshold = parameter.GetValue<double>();
  if (!sirius::config::valid_domain_coverage_threshold{}(threshold)) {
    throw InvalidInputException("dynamic_filter_domain_coverage_threshold %s, got %f",
                                sirius::config::valid_domain_coverage_threshold::description(),
                                threshold);
  }
  params->dynamic_filter_domain_coverage_threshold = threshold;
  SIRIUS_LOG_DEBUG("Updated config DYNAMIC_FILTER_DOMAIN_COVERAGE_THRESHOLD to {}",
                   params->dynamic_filter_domain_coverage_threshold);
}

static void SetDynamicFilterInlistMaxL2Fraction(ClientContext& context,
                                                SetScope scope,
                                                Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot             = lock_operator_params_slot(context);
  const double fraction = parameter.GetValue<double>();
  if (!(fraction >= 0.0 && fraction <= 1.0)) {
    throw InvalidInputException(
      "dynamic_filter_inlist_max_l2_fraction must be in [0.0, 1.0], got %f", fraction);
  }
  params->dynamic_filter_inlist_max_l2_fraction = fraction;
  SIRIUS_LOG_DEBUG("Updated config DYNAMIC_FILTER_INLIST_MAX_L2_FRACTION to {}",
                   params->dynamic_filter_inlist_max_l2_fraction);
}

static void SetDynamicFilterKeepThreshold(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot              = lock_operator_params_slot(context);
  const double threshold = parameter.GetValue<double>();
  if (!(threshold >= 0.0 && threshold <= 1.0)) {
    throw InvalidInputException("dynamic_filter_keep_threshold must be in [0.0, 1.0], got %f",
                                threshold);
  }
  params->dynamic_filter_keep_threshold = threshold;
  SIRIUS_LOG_DEBUG("Updated config DYNAMIC_FILTER_KEEP_THRESHOLD to {}",
                   params->dynamic_filter_keep_threshold);
}

static void SetEnablePinnedZoneMapPruning(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                              = lock_operator_params_slot(context);
  params->enable_pinned_zone_map_pruning = BooleanValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config ENABLE_PINNED_ZONE_MAP_PRUNING to {}",
                   params->enable_pinned_zone_map_pruning);
}

static void SetAdmissionBytesPerGpu(ClientContext& context, SetScope scope, Value& parameter)
{
  auto const bytes = UBigIntValue::Get(parameter);
  auto* params     = get_operator_params(context);
  if (!params) { return; }
  auto slot                       = lock_operator_params_slot(context);
  params->admission_bytes_per_gpu = bytes;
  SIRIUS_LOG_DEBUG("Updated config ADMISSION_BYTES_PER_GPU to {}", params->admission_bytes_per_gpu);
}

static void SetAvgVariableColumnBytes(ClientContext& context, SetScope scope, Value& parameter)
{
  auto const bytes = UBigIntValue::Get(parameter);
  if (bytes == 0) {
    throw InvalidInputException("avg_variable_column_bytes must be greater than zero");
  }
  auto* params = get_operator_params(context);
  if (!params) { return; }
  auto slot                         = lock_operator_params_slot(context);
  params->avg_variable_column_bytes = bytes;
  SIRIUS_LOG_DEBUG("Updated config AVG_VARIABLE_COLUMN_BYTES to {}",
                   params->avg_variable_column_bytes);
}

static void SetEnableCompressedMaterialization(ClientContext& /*context*/,
                                               SetScope /*scope*/,
                                               Value& /*parameter*/)
{
  // Deliberately empty: DuckDB has already stored the value on the connection that ran the SET
  // by the time this hook runs. The hook exists only to copy that value somewhere else, and
  // there is nowhere else it belongs — planning and pinning read it straight back with
  // compressed_materialization_enabled(). A copy kept in shared state would answer for every
  // connection, so one connection's SET would redirect another's scans while that connection's
  // current_setting still reported the old value.
}

void SiriusExtension::InitialGPUConfigs(DBConfig& config, const sirius::sirius_config& defaults)
{
  auto const& operator_defaults    = defaults.get_operator_params();
  auto const& compression_defaults = defaults.get_compression_config();

#ifdef SIRIUS_ENABLE_LEGACY
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
#endif

  config.AddExtensionOption(
    "expression_evaluator_strategy",
    "Strategy for the expression_evaluator: 'materialize', 'ast_interpret', or "
    "'ast_jit'",
    LogicalType::VARCHAR,
    Value(std::string(sirius::strategy_to_string(Config::EXPRESSION_EVALUATOR_STRATEGY))),
    SetExpressionEvaluatorStrategy);

  // Deprecated alias for `expression_evaluator_strategy`; remove in a future release.
  config.AddExtensionOption(
    "expression_executor_strategy",
    "[DEPRECATED - use expression_evaluator_strategy] Strategy for the expression_evaluator: "
    "'materialize', 'ast_interpret', or 'ast_jit'",
    LogicalType::VARCHAR,
    Value(std::string(sirius::strategy_to_string(Config::EXPRESSION_EVALUATOR_STRATEGY))),
    SetExpressionExecutorStrategyDeprecated);

#ifdef SIRIUS_ENABLE_LEGACY
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
#endif

  add_sirius_option(
    config,
    option_visibility::user,
    "enable_duckdb_fallback",
    "Whether to enable fallback to duckdb execution after an error is detected",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(true),  // literal default: never seed from a process-global that a
                           // prior connection's SET may have mutated (that leaked the
                           // fallback policy into every freshly-created database).
    SetEnableDuckdbFallback);

  // Keep internal policy and test hooks out of the normal duckdb_settings() surface. The
  // unittest harness opts in before constructing a database. Centralizing visibility here keeps
  // option registration from growing scattered environment checks.
  add_sirius_option(config,
                    option_visibility::internal,
                    "sirius_test_inject_transparent_gpu_error",
                    "force transparent GPU execution to fail at runtime with this message",
                    LogicalType::VARCHAR,
                    Value(""));
  add_sirius_option(config,
                    option_visibility::internal,
                    "enable_pinned_zone_map_pruning",
                    "disable automatic pinned-table zone-map capture and pruning",
                    LogicalType::BOOLEAN,
                    Value::BOOLEAN(operator_defaults.enable_pinned_zone_map_pruning),
                    SetEnablePinnedZoneMapPruning);
  add_sirius_option(config,
                    option_visibility::internal,
                    "enable_dynamic_filter",
                    "disable runtime dynamic-filter discovery for eligible hash joins "
                    "(probe-side scan and join-edge targets)",
                    LogicalType::BOOLEAN,
                    Value::BOOLEAN(operator_defaults.enable_dynamic_filter),
                    SetEnableDynamicFilter);
  add_sirius_option(config,
                    option_visibility::internal,
                    "enable_dynamic_zone_map_filter",
                    "enable the clustered-keyset dynamic zone-map path",
                    LogicalType::BOOLEAN,
                    Value::BOOLEAN(operator_defaults.enable_dynamic_zone_map_filter),
                    SetEnableDynamicZoneMapFilter);
  add_sirius_option(config,
                    option_visibility::internal,
                    "scan_task_batch_size",
                    "override the internally derived scan batch target",
                    LogicalType::UBIGINT,
                    Value::UBIGINT(operator_defaults.scan_task_batch_size),
                    SetDefaultScanTaskBatchSize);
  add_sirius_option(config,
                    option_visibility::internal,
                    "fuse_merge_pipelines",
                    "toggle merge pipeline fusion",
                    LogicalType::BOOLEAN,
                    Value::BOOLEAN(true),
                    SetFuseMergePipelines);
  add_sirius_option(config,
                    option_visibility::internal,
                    "enable_runtime_distinct_build_probe",
                    "toggle the internal runtime distinct-build probe",
                    LogicalType::BOOLEAN,
                    Value::BOOLEAN(operator_defaults.enable_runtime_distinct_build_probe),
                    SetEnableRuntimeDistinctBuildProbe);
  add_sirius_option(config,
                    option_visibility::internal,
                    "concat_batch_bytes",
                    "override the internally derived CONCAT batch target",
                    LogicalType::UBIGINT,
                    Value::UBIGINT(operator_defaults.concat_batch_bytes),
                    SetConcatBatchBytes);

  // Add in config options for special JIT implementation for regex
  add_sirius_option(config,
                    option_visibility::user,
                    "enable_regex_jit_impl",
                    "Whether to use special JIT implementation for particular regex evaluation",
                    LogicalType::BOOLEAN,
                    Value::BOOLEAN(Config::ENABLE_REGEX_JIT_IMPL),
                    SetEnableRegexJitImpl);

#ifdef SIRIUS_ENABLE_LEGACY
  // Add in config options for modified pipeline
  config.AddExtensionOption("modified_pipeline",
                            "Whether to use modified pipeline for GPU execution",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(Config::MODIFIED_PIPELINE),
                            SetModifiedPipeline);
#endif

  // Add in config option for sort partition size
  config.AddExtensionOption("max_sort_partition_bytes",
                            "Maximum bytes per sort partition (0 = auto based on "
                            "max_sort_partition_memory_fraction of GPU memory)",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(operator_defaults.max_sort_partition_bytes),
                            SetMaxSortPartitionBytes);
  config.AddExtensionOption(
    "max_sort_partition_memory_fraction",
    "Fraction of available GPU memory per sort partition when max_sort_partition_bytes is 0",
    LogicalType::DOUBLE,
    Value::DOUBLE(operator_defaults.max_sort_partition_memory_fraction),
    SetMaxSortPartitionMemoryFraction);

  // Logging configuration
  config.AddExtensionOption("sirius_log_backend",
                            "Logging backend for Sirius (duckdb, spdlog, noop)",
                            LogicalType::VARCHAR,
                            Value(Config::LOG_BACKEND),
                            SetLogBackend);
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
                            Value::UBIGINT(operator_defaults.hash_partition_bytes),
                            SetHashPartitionBytes);

  config.AddExtensionOption("sort_sample_bytes",
                            "Target bytes to sample before computing sort partition boundaries",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(operator_defaults.sort_sample_bytes),
                            SetSortSampleBytes);

  config.AddExtensionOption("max_build_hash_table_bytes",
                            "Maximum size a build-side table can be where it will create a "
                            "reusable hash table for hash joins (i.e. BUILD_PROBE mode)",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(operator_defaults.max_build_hash_table_bytes),
                            SetMaxBuildHashTableBytes);

  config.AddExtensionOption("max_broadcast_join_size",
                            "Maximum build-side size in bytes for a broadcast join, where the "
                            "(small) build table is replicated to every GPU instead of "
                            "hash-partitioned across GPUs",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(operator_defaults.max_broadcast_join_size),
                            SetMaxBroadcastJoinSize);

  config.AddExtensionOption(
    "mark_join_build_switch_ratio",
    "For STANDARD-mode MARK joins, build on the left/output side via cudf::mark_join when the "
    "right (probe) side has at least this many times more rows than the left side (0 disables). "
    "Hardware-dependent — recalibrate per GPU.",
    LogicalType::DOUBLE,
    Value::DOUBLE(operator_defaults.mark_join_build_switch_ratio),
    SetMarkJoinBuildSwitchRatio);

  config.AddExtensionOption(
    "gpu_execution",
    "Whether to transparently intercept SQL queries and execute them on GPU",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(true),
    SetEnableGpuExecution);

  config.AddExtensionOption("pin_table_compression",
                            "Request Simpatico compression for pin_table chunks. Takes effect only "
                            "when pin_table_input_compression_plan_dir is non-empty and contains a "
                            "matching table plan",
                            LogicalType::BOOLEAN,
                            Value::BOOLEAN(compression_defaults.enable_pin_table_compression),
                            SetEnablePinTableCompression);

  config.AddExtensionOption(
    "pin_table_input_compression_plan_dir",
    "Directory containing per-table Simpatico plan files for pin_table compression. "
    "Files are named '<table_name>.<ext>'; their contents are the multi-column plan DSL. "
    "May be set before pin_table_compression is enabled. Tables with no matching file are pinned "
    "uncompressed. No effect on spill compression.",
    LogicalType::VARCHAR,
    Value(compression_defaults.input_plan_dir),
    SetPinTableInputCompressionPlanDir);

  config.AddExtensionOption(
    "pin_table_compression_min_batch_size_bytes",
    "Minimum uncompressed batch size in bytes below which active pin_table compression is skipped; "
    "inert until compression is enabled and a matching plan resolves",
    LogicalType::UBIGINT,
    Value::UBIGINT(compression_defaults.min_batch_size_bytes),
    SetPinTableCompressionMinBatchSizeBytes);

  config.AddExtensionOption(
    "pin_table_compression_max_compressed_fraction",
    "Discard the compressed form and pin uncompressed when the compressed size exceeds this "
    "finite, non-negative fraction of the batch's original size (values above 1 permit "
    "expansion); inert until compression is enabled and a matching plan resolves",
    LogicalType::DOUBLE,
    Value::DOUBLE(compression_defaults.max_compressed_fraction),
    SetPinTableCompressionMaxCompressedFraction);

  config.AddExtensionOption(
    "dynamic_filter_domain_coverage_threshold",
    "Skip publishing a key's dynamic filters when the hash-join build covers at least this "
    "fraction of the key's domain; values above 1.0 disable the gate",
    LogicalType::DOUBLE,
    Value::DOUBLE(operator_defaults.dynamic_filter_domain_coverage_threshold),
    SetDynamicFilterDomainCoverageThreshold);

  config.AddExtensionOption(
    "dynamic_filter_inlist_max_l2_fraction",
    "Maximum estimated cuco-set size for the exact hash IN-list dynamic filter, as a fraction of "
    "the smallest probe-GPU L2 cache, in [0, 1]; larger sets publish a Bloom filter, 0 always "
    "publishes the Bloom when supported, and 1.0 reproduces the legacy L2-fit rule",
    LogicalType::DOUBLE,
    Value::DOUBLE(operator_defaults.dynamic_filter_inlist_max_l2_fraction),
    SetDynamicFilterInlistMaxL2Fraction);

  config.AddExtensionOption(
    "dynamic_filter_keep_threshold",
    "Disable a probe scan's post-decode dynamic filtering once a measured split keeps more than "
    "this fraction of its rows (too unselective to repay the mask kernel); in [0.0, 1.0], 1.0 "
    "keeps filtering always on",
    LogicalType::DOUBLE,
    Value::DOUBLE(operator_defaults.dynamic_filter_keep_threshold),
    SetDynamicFilterKeepThreshold);

  // Default from the YAML-loaded params, so a sirius.yaml value is what connections inherit.
  config.AddExtensionOption(
    "enable_compressed_materialization",
    "Keep eligible integer and fixed-point DECIMAL columns in value-preserving narrow cuDF "
    "carriers selected from exact pin-time bounds; restore native carriers at type-sensitive "
    "boundaries (on by default)",
    LogicalType::BOOLEAN,
    Value::BOOLEAN(operator_defaults.enable_compressed_materialization),
    SetEnableCompressedMaterialization);

  config.AddExtensionOption(
    "admission_bytes_per_gpu",
    "Target projected scan-output bytes per GPU at admission; 0 disables the estimate and "
    "leaves the allocation to topology.gpus_per_query",
    LogicalType::UBIGINT,
    Value::UBIGINT(operator_defaults.admission_bytes_per_gpu),
    SetAdmissionBytesPerGpu);

  config.AddExtensionOption(
    "avg_variable_column_bytes",
    "Per-row width assumed for variable-width columns (VARCHAR, LIST, STRUCT, ARRAY) when "
    "estimating scan output at admission; must be greater than zero",
    LogicalType::UBIGINT,
    Value::UBIGINT(operator_defaults.avg_variable_column_bytes),
    SetAvgVariableColumnBytes);
}

// Publish the transparent optimizer mask once at extension load, unioned
// into any user-preset entries. DBConfig::options.disabled_optimizers is
// DB-global and read locklessly by every connection's optimizer, so the old
// per-query save/modify/restore was an unprotected concurrent write.
// Supported precondition: LOAD runs while the DatabaseInstance is quiescent
// (existing connections may exist, but no concurrent queries, optimizer runs
// or config changes). A later user SET replacing the set is likewise a
// serial, quiescent override — concurrent optimizer runs while it changes
// are unsupported; transparent execution then falls back to CPU on the
// affected plan shapes instead of re-inserting.
static void publish_transparent_optimizer_mask(DBConfig& config)
{
  auto& live = config.options.disabled_optimizers;
  // Build the UNION on a local copy and publish with a single no-throw swap:
  // an allocation failure mid-insert must not leave a partial mask behind
  // (exception-atomic publication).
  auto updated = live;
  // The InClauseRewriter converts large IN clauses into a mark join against a
  // ColumnDataCollection; COMPRESSED_MATERIALIZATION introduces DuckDB-internal
  // compressed shapes — neither is executable by the rebind path.
  updated.insert(OptimizerType::IN_CLAUSE);
  updated.insert(OptimizerType::COMPRESSED_MATERIALIZATION);
  // LATE_MATERIALIZATION rewrites `ORDER BY ... LIMIT N` over a scan into a
  // self-RIGHT_SEMI_JOIN keyed on parquet virtual columns the Sirius scan path
  // drops (src/op/scan/scan_plan.cpp), so the join's key columns would not
  // exist at runtime. See PR #732 comment 3242605041.
  updated.insert(OptimizerType::LATE_MATERIALIZATION);
  live.swap(updated);
}

static void LoadInternal(ExtensionLoader& loader)
{
  sirius::util::install_segfault_backtrace_handler();

  auto& db     = loader.GetDatabaseInstance();
  auto& config = DBConfig::GetConfig(db);

  // SIRIUS_DISABLE means: no Sirius runtime initialization and no mask
  // publication (the extension binary itself may still be loaded).
  auto callback      = make_shared_ptr<duckdb::SiriusContextExtensionCallback>();
  auto* callback_ptr = callback.get();
  config.GetCallbackManager().Register(std::move(callback));

  // The ctor already installed the db-independent backend; reinstall now that the
  // DatabaseInstance exists so the duckdb backend (which needs it) is built and an
  // unknown backend name is reported here rather than swallowed by the ctor.
  install_configured_log_sink(&db);

  sirius::converter_registry::initialize();
  // The callback constructor above already read sirius.yaml, so its params are the defaults the
  // per-connection options register with.
  SiriusExtension::InitialGPUConfigs(config, callback_ptr->get_loaded_config());
  SiriusExtension::RegisterGPUFunctions(db);

  // Register the s3:// FileSystem so DuckDB's native read_parquet('s3://') binds
  // by reading the parquet footer through Sirius's routed REST ioctx. This makes
  // the transparent form work — SET gpu_execution=true; SELECT ... FROM
  // read_parquet('s3://...') — with the captured scan run on GPU. sirius_httpfs
  // is read-only and GPU-only: it serves the bind-time footer read, never a CPU
  // data path (a query that reads s3:// and fails on GPU still surfaces a clear
  // "S3 CPU fallback is not supported" error; local reads fall back to CPU).
  db.GetFileSystem().RegisterSubSystem(make_uniq<sirius::io::s3::sirius_httpfs>());

  // Register optimizer extension for transparent GPU execution. The post-hook
  // captures the optimized plan; the pre-hook derives the single-table
  // restrictions implied by OR-ed multi-table filters so DuckDB's own pushdown,
  // join-order and build/probe-side passes can act on them. (The optimizer mask
  // an earlier pre-hook used to write per query is published once at load — see
  // publish_transparent_optimizer_mask above.)
  OptimizerExtension opt_ext;
  opt_ext.pre_optimize_function = sirius::transparent::sirius_pre_optimizer_hook;
  opt_ext.optimize_function     = sirius::transparent::sirius_optimizer_hook;
  OptimizerExtension::Register(config, std::move(opt_ext));

  // Register SiriusContext on connections that were opened before the extension
  // was loaded (e.g. when loaded via LOAD in Python or the CLI).
  for (auto& ctx : ConnectionManager::Get(db).GetConnectionList()) {
    callback_ptr->OnConnectionOpened(*ctx);
  }

  // Publish the optimizer mask last, only after every initialization step
  // above succeeded, so a failed load leaves no mask behind. SIRIUS_DISABLE
  // means no Sirius runtime initialization and no mask publication (the
  // extension binary itself may still be loaded).
  bool const sirius_disabled = [] {
    auto* val = std::getenv("SIRIUS_DISABLE");
    return val != nullptr && std::string(val) != "0";
  }();
  if (!sirius_disabled) { publish_transparent_optimizer_mask(config); }
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
