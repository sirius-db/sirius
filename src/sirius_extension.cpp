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

// Forward-declare CUDA profiler API functions (linked via libcudart).
extern "C" int cudaProfilerStart();
extern "C" int cudaProfilerStop();
#include "data/sirius_converter_registry.hpp"
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
// #include "from_substrait.hpp"
#include "gpu_buffer_manager.hpp"
#ifdef SIRIUS_ENABLE_LEGACY
#include "gpu_context.hpp"
#include "gpu_physical_plan_generator.hpp"
#endif
#include "duckdb/main/connection_manager.hpp"
#include "io/datasource_factory.hpp"
#include "io/s3/s3_ioctx.hpp"
#include "io/sirius_datasource.hpp"
#include "log/logging.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "sirius_interface.hpp"
#include "util/segfault_backtrace.hpp"

#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string_view>
#include <system_error>

namespace duckdb {

const std::string PINNED_MEMORY_PARAM_KEY   = "pinned_memory_size";
bool SiriusExtension::buffer_is_initialized = false;

namespace {

constexpr std::string_view NATIVE_READ_PARQUET_FN  = "read_parquet";
constexpr std::string_view SIRIUS_READ_PARQUET_FN  = "sirius_read_parquet";
constexpr std::string_view S3_URI_PREFIX           = "s3://";

bool is_identifier_char(char c)
{
  auto const ch = static_cast<unsigned char>(c);
  return std::isalnum(ch) || c == '_';
}

bool iequals_prefix(std::string_view value, std::string_view prefix)
{
  if (value.size() < prefix.size()) { return false; }
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    auto const lhs = static_cast<unsigned char>(value[i]);
    auto const rhs = static_cast<unsigned char>(prefix[i]);
    if (std::tolower(lhs) != std::tolower(rhs)) { return false; }
  }
  return true;
}

std::size_t skip_whitespace(std::string const& text, std::size_t pos)
{
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  return pos;
}

std::optional<std::pair<std::string, std::size_t>> parse_single_quoted_literal(
  std::string const& text, std::size_t pos)
{
  if (pos >= text.size() || text[pos] != '\'') { return std::nullopt; }

  std::string literal;
  for (std::size_t i = pos + 1; i < text.size(); ++i) {
    if (text[i] == '\'') {
      if (i + 1 < text.size() && text[i + 1] == '\'') {
        literal.push_back('\'');
        ++i;
        continue;
      }
      return std::pair{std::move(literal), i + 1};
    }
    literal.push_back(text[i]);
  }
  return std::nullopt;
}

bool should_rewrite_read_parquet_call(std::string const& query, std::size_t name_pos)
{
  auto pos = skip_whitespace(query, name_pos + NATIVE_READ_PARQUET_FN.size());
  if (pos >= query.size() || query[pos] != '(') { return false; }

  pos = skip_whitespace(query, pos + 1);
  auto literal = parse_single_quoted_literal(query, pos);
  if (!literal.has_value()) { return false; }
  return iequals_prefix(literal->first, S3_URI_PREFIX);
}

std::string rewrite_sirius_owned_remote_parquet_calls(std::string const& query)
{
  std::string rewritten;
  rewritten.reserve(query.size() + 32);

  bool in_single_quote  = false;
  bool in_double_quote  = false;
  bool in_line_comment  = false;
  bool in_block_comment = false;

  for (std::size_t i = 0; i < query.size();) {
    auto const c = query[i];

    if (in_line_comment) {
      rewritten.push_back(c);
      if (c == '\n') { in_line_comment = false; }
      ++i;
      continue;
    }
    if (in_block_comment) {
      rewritten.push_back(c);
      if (c == '*' && i + 1 < query.size() && query[i + 1] == '/') {
        rewritten.push_back('/');
        i += 2;
        in_block_comment = false;
      } else {
        ++i;
      }
      continue;
    }
    if (in_single_quote) {
      rewritten.push_back(c);
      if (c == '\'') {
        if (i + 1 < query.size() && query[i + 1] == '\'') {
          rewritten.push_back('\'');
          i += 2;
          continue;
        }
        in_single_quote = false;
      }
      ++i;
      continue;
    }
    if (in_double_quote) {
      rewritten.push_back(c);
      if (c == '"') {
        if (i + 1 < query.size() && query[i + 1] == '"') {
          rewritten.push_back('"');
          i += 2;
          continue;
        }
        in_double_quote = false;
      }
      ++i;
      continue;
    }

    if (c == '-' && i + 1 < query.size() && query[i + 1] == '-') {
      rewritten.append("--");
      i += 2;
      in_line_comment = true;
      continue;
    }
    if (c == '/' && i + 1 < query.size() && query[i + 1] == '*') {
      rewritten.append("/*");
      i += 2;
      in_block_comment = true;
      continue;
    }
    if (c == '\'') {
      rewritten.push_back(c);
      ++i;
      in_single_quote = true;
      continue;
    }
    if (c == '"') {
      rewritten.push_back(c);
      ++i;
      in_double_quote = true;
      continue;
    }

    auto const remaining = std::string_view(query).substr(i);
    if ((i == 0 || !is_identifier_char(query[i - 1])) &&
        iequals_prefix(remaining, NATIVE_READ_PARQUET_FN) &&
        (i + NATIVE_READ_PARQUET_FN.size() >= query.size() ||
         !is_identifier_char(query[i + NATIVE_READ_PARQUET_FN.size()])) &&
        should_rewrite_read_parquet_call(query, i)) {
      rewritten.append(SIRIUS_READ_PARQUET_FN);
      i += NATIVE_READ_PARQUET_FN.size();
      continue;
    }

    rewritten.push_back(c);
    ++i;
  }

  return rewritten;
}

std::string escape_sql_string_literal(std::string_view value)
{
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (char c : value) {
    escaped.push_back(c);
    if (c == '\'') { escaped.push_back('\''); }
  }
  return escaped;
}

std::filesystem::path materialize_remote_parquet_for_bind(ClientContext& context,
                                                          std::string const& uri)
{
  auto sirius_ctx = context.registered_state
                      ? context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                      : nullptr;
  if (!sirius_ctx) {
    throw InvalidInputException(
      "{} requires an initialized SiriusContext", std::string(SIRIUS_READ_PARQUET_FN));
  }

  auto const& osc = sirius_ctx->get_config().get_object_store_config();
  if (osc.endpoint.empty()) {
    throw InvalidInputException(
      "{} requires s3_endpoint to be set before gpu_execution planning",
      std::string(SIRIUS_READ_PARQUET_FN));
  }

  sirius::io::s3::s3_ioctx_config io_cfg;
  io_cfg.endpoint   = osc.endpoint;
  io_cfg.region     = osc.region.empty() ? "us-east-1" : osc.region;
  io_cfg.access_key = osc.access_key;
  io_cfg.secret_key = osc.secret_key;

  sirius::io::datasource_registry registry;
  registry.register_ioctx("s3", std::make_shared<sirius::io::s3::s3_ioctx>(std::move(io_cfg)));

  auto datasource = sirius::io::datasource_factory::create(uri, registry, sirius_ctx->get_config());
  auto const object_size = datasource->size();
  if (object_size == 0) {
    throw InvalidInputException("{} cannot bind an empty parquet object: {}",
                                std::string(SIRIUS_READ_PARQUET_FN),
                                uri);
  }

  auto remote_bytes = datasource->host_read(0, object_size);
  if (!remote_bytes || remote_bytes->size() != object_size) {
    throw InvalidInputException("{} failed to materialize remote parquet object for bind: {}",
                                std::string(SIRIUS_READ_PARQUET_FN),
                                uri);
  }

  auto const unique_id =
    static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count());
  auto tmp_path = std::filesystem::temp_directory_path() /
                  ("sirius_remote_bind_" + std::to_string(unique_id) + ".parquet");

  std::ofstream out(tmp_path, std::ios::binary);
  if (!out.good()) {
    throw InvalidInputException("{} failed to create temp parquet file {}",
                                std::string(SIRIUS_READ_PARQUET_FN),
                                tmp_path.string());
  }
  out.write(reinterpret_cast<char const*>(remote_bytes->data()),
            static_cast<std::streamsize>(object_size));
  out.close();
  if (!out.good()) {
    throw InvalidInputException("{} failed to write temp parquet file {}",
                                std::string(SIRIUS_READ_PARQUET_FN),
                                tmp_path.string());
  }
  return tmp_path;
}

void infer_local_parquet_schema(ClientContext& context,
                                std::filesystem::path const& parquet_path,
                                vector<LogicalType>& return_types,
                                vector<string>& names)
{
  Connection bind_conn(*context.db);
  auto result = bind_conn.Query("SELECT * FROM read_parquet('" +
                                escape_sql_string_literal(parquet_path.string()) +
                                "') LIMIT 0");
  if (!result) {
    throw InvalidInputException("{} failed to infer parquet schema for {}",
                                std::string(SIRIUS_READ_PARQUET_FN),
                                parquet_path.string());
  }
  if (result->HasError()) {
    throw InvalidInputException("{} failed to infer parquet schema for {}: {}",
                                std::string(SIRIUS_READ_PARQUET_FN),
                                parquet_path.string(),
                                result->GetError());
  }

  names        = result->names;
  return_types = result->types;
}

unique_ptr<FunctionData> SiriusReadParquetBind(ClientContext& context,
                                               TableFunctionBindInput& input,
                                               vector<LogicalType>& return_types,
                                               vector<string>& names)
{
  if (input.inputs.size() != 1 || input.inputs[0].IsNull()) {
    throw InvalidInputException("{} expects a single non-null parquet URI",
                                std::string(SIRIUS_READ_PARQUET_FN));
  }

  auto const uri = input.inputs[0].GetValue<std::string>();
  if (!iequals_prefix(uri, S3_URI_PREFIX)) {
    throw InvalidInputException("{} currently only supports s3:// URIs inside gpu_execution",
                                std::string(SIRIUS_READ_PARQUET_FN));
  }

  struct scoped_path {
    std::filesystem::path path;
    ~scoped_path()
    {
      std::error_code ec;
      if (!path.empty()) { std::filesystem::remove(path, ec); }
    }
  } tmp_file{materialize_remote_parquet_for_bind(context, uri)};

  infer_local_parquet_schema(context, tmp_file.path, return_types, names);
  return nullptr;
}

void SiriusReadParquetFunction(ClientContext&,
                               TableFunctionInput&,
                               DataChunk&)
{
  throw InvalidInputException(
    "{} is an internal Sirius table function and must execute through gpu_execution",
    std::string(SIRIUS_READ_PARQUET_FN));
}

}  // namespace

struct SiriusTableFunctionData : public TableFunctionData {
  SiriusTableFunctionData() = default;
  shared_ptr<::sirius::sirius_prepared_statement_data> gpu_prepared;
  unique_ptr<QueryResult> res;
  unique_ptr<Connection> conn;
  unique_ptr<::sirius::sirius_interface> sirius_iface;
  string query;
  string planned_query;
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
      parser.ParseQuery(planned_query.empty() ? query : planned_query);

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
  result->planned_query    = rewrite_sirius_owned_remote_parquet_calls(result->query);
  result->enable_optimizer = true;
  result->sirius_iface     = make_uniq<::sirius::sirius_interface>(context);
  if (input.inputs[0].IsNull()) {
    throw BinderException("gpu_execution cannot be called with a NULL parameter");
  }

  // Parse the query just to get the result type information and to create preparedstatmement data
  Parser parser(context.GetParserOptions());
  parser.ParseQuery(result->planned_query);
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
  auto& data = (SiriusTableFunctionData&)*data_p.bind_data;
  if (data.finished) { return; }

  if (!data.res) {
    auto start = std::chrono::high_resolution_clock::now();
    if (data.plan_error) {
      printf(
        "=============================================\nError in SiriusExecuteQuery, fallback to "
        "DuckDB\n=============================================\n");
      data.res = data.conn->Query(data.query);
    } else {
      data.res =
        data.sirius_iface->sirius_execute_query(context, data.query, data.gpu_prepared, {});
      if (data.res->HasError()) {
        if (Config::ENABLE_DUCKDB_FALLBACK) {
          SIRIUS_LOG_ERROR("SiriusExecuteQuery error: {}", data.res->GetError());
          printf(
            "=============================================\nError in SiriusExecuteQuery, fallback "
            "to DuckDB\n=============================================\n");
          data.res = data.conn->Query(data.query);
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

static unique_ptr<FunctionData> ProfilerBind(ClientContext& context,
                                             TableFunctionBindInput& input,
                                             vector<LogicalType>& return_types,
                                             vector<string>& names)
{
  return_types.push_back(LogicalType::BOOLEAN);
  names.push_back("ok");
  return nullptr;
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

#ifdef SIRIUS_ENABLE_LEGACY
  RegisterLegacyGPUFunctions(transaction, catalog);
#endif

  TableFunction gpu_execution("gpu_execution",
                              {LogicalType::VARCHAR},
                              GPUExecutionFunction,
                              SiriusExtension::GPUExecutionBind);
  gpu_execution.named_parameters["enable_optimizer"] = LogicalType::BOOLEAN;
  CreateTableFunctionInfo gpu_execution_info(gpu_execution);
  catalog.CreateTableFunction(transaction, gpu_execution_info);

  TableFunction sirius_read_parquet("sirius_read_parquet",
                                    {LogicalType::VARCHAR},
                                    SiriusReadParquetFunction,
                                    SiriusReadParquetBind);
  CreateTableFunctionInfo sirius_read_parquet_info(sirius_read_parquet);
  catalog.CreateTableFunction(transaction, sirius_read_parquet_info);

  // Profiler control functions for nsys --capture-range=cudaProfilerApi
  TableFunction profiler_start(
    "profiler_start", {}, ProfilerStartFunction, ProfilerBind, ProfilerInit);
  CreateTableFunctionInfo profiler_start_info(profiler_start);
  catalog.CreateTableFunction(transaction, profiler_start_info);

  TableFunction profiler_stop(
    "profiler_stop", {}, ProfilerStopFunction, ProfilerBind, ProfilerInit);
  CreateTableFunctionInfo profiler_stop_info(profiler_stop);
  catalog.CreateTableFunction(transaction, profiler_stop_info);
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

static sirius::io::object_store_config* get_object_store_config(ClientContext& context)
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    SIRIUS_LOG_DEBUG("SiriusContext not available; object_store_config SET ignored");
    return nullptr;
  }
  return &sirius_ctx->get_config().get_object_store_config();
}

static void SetS3Transport(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* cfg = get_object_store_config(context);
  if (!cfg) { return; }
  auto value = StringValue::Get(parameter);
  sirius::io::object_store_config::transport t;
  if (!sirius::io::string_to_enum(std::string_view{value}, t)) {
    throw InvalidInputException(
      "Invalid s3_transport '{}'. Valid values: auto, http, rdma", value);
  }
  cfg->s3_transport = t;
  SIRIUS_LOG_DEBUG("Updated config s3_transport to {}", value);
}

static void SetS3Endpoint(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* cfg = get_object_store_config(context);
  if (!cfg) { return; }
  cfg->endpoint = StringValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config s3_endpoint to {}", cfg->endpoint);
}

static void SetS3Region(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* cfg = get_object_store_config(context);
  if (!cfg) { return; }
  cfg->region = StringValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config s3_region to {}", cfg->region);
}

static void SetS3AccessKey(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* cfg = get_object_store_config(context);
  if (!cfg) { return; }
  cfg->access_key = StringValue::Get(parameter);
  // Don't log the credential itself.
  SIRIUS_LOG_DEBUG("Updated config s3_access_key (len={})", cfg->access_key.size());
}

static void SetS3SecretKey(ClientContext& context, SetScope scope, Value& parameter)
{
  auto* cfg = get_object_store_config(context);
  if (!cfg) { return; }
  cfg->secret_key = StringValue::Get(parameter);
  SIRIUS_LOG_DEBUG("Updated config s3_secret_key (len={})", cfg->secret_key.size());
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

  // Object-store configuration. Values are consumed by the S3 / RDMA S3
  // backends landing in later PRs; in PR7 they are just stored on the
  // per-connection sirius_config.
  config.AddExtensionOption("s3_transport",
                            "Transport for S3 datasource: 'auto', 'http', or 'rdma'",
                            LogicalType::VARCHAR,
                            Value("auto"),
                            SetS3Transport);
  config.AddExtensionOption("s3_endpoint",
                            "Endpoint URL for S3-compatible object store (empty = AWS default)",
                            LogicalType::VARCHAR,
                            Value(""),
                            SetS3Endpoint);
  config.AddExtensionOption("s3_region",
                            "Region for S3-compatible object store",
                            LogicalType::VARCHAR,
                            Value(""),
                            SetS3Region);
  config.AddExtensionOption("s3_access_key",
                            "Access key ID for S3-compatible object store",
                            LogicalType::VARCHAR,
                            Value(""),
                            SetS3AccessKey);
  config.AddExtensionOption("s3_secret_key",
                            "Secret access key for S3-compatible object store",
                            LogicalType::VARCHAR,
                            Value(""),
                            SetS3SecretKey);
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
