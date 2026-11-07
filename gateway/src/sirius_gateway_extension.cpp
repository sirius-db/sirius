#define DUCKDB_EXTENSION_MAIN

#include "sirius_gateway_extension.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/main/config.hpp"
#include "gateway_state.hpp"
#include "statement_router.hpp"

#include <unistd.h>

#include <filesystem>

namespace duckdb {

namespace {

using sirius_gateway::gateway_state;

struct gateway_info : TableFunctionInfo {
  explicit gateway_info(std::shared_ptr<gateway_state> state_p) : state(std::move(state_p)) {}
  std::shared_ptr<gateway_state> state;
};

gateway_state& gateway_of(TableFunctionBindInput& input)
{
  return *input.info->Cast<gateway_info>().state;
}

/// A result computed at bind time, emitted by the scan.
struct rows_data : TableFunctionData {
  vector<vector<Value>> rows;
  unique_ptr<FunctionData> Copy() const override { return make_uniq<rows_data>(*this); }
};

struct run_data : TableFunctionData {
  std::shared_ptr<sirius_gateway::rank_result> result;
  unique_ptr<FunctionData> Copy() const override { return make_uniq<run_data>(*this); }
};

struct scan_state : GlobalTableFunctionState {
  idx_t offset = 0;
};

unique_ptr<GlobalTableFunctionState> init_scan(ClientContext&, TableFunctionInitInput&)
{
  return make_uniq<scan_state>();
}

void scan_rows(ClientContext&, TableFunctionInput& input, DataChunk& output)
{
  auto& rows  = input.bind_data->Cast<rows_data>().rows;
  auto& state = input.global_state->Cast<scan_state>();
  idx_t count = 0;
  for (; state.offset < rows.size() && count < STANDARD_VECTOR_SIZE; ++state.offset, ++count) {
    for (idx_t c = 0; c < rows[state.offset].size(); ++c) {
      output.SetValue(c, count, rows[state.offset][c]);
    }
  }
  output.SetCardinality(count);
}

unique_ptr<FunctionData> status_rows(gateway_state& gateway,
                                     vector<LogicalType>& types,
                                     vector<Identifier>& names)
{
  names              = {"rank",
                        "name",
                        "registered",
                        "pid",
                        "gpu",
                        "host",
                        "uri",
                        "state",
                        "cluster",
                        "error",
                        "last_query_id",
                        "running_query_id",
                        "queued"};
  types              = {LogicalType::UINTEGER,
                        LogicalType::VARCHAR,
                        LogicalType::BOOLEAN,
                        LogicalType::BIGINT,
                        LogicalType::VARCHAR,
                        LogicalType::VARCHAR,
                        LogicalType::VARCHAR,
                        LogicalType::VARCHAR,
                        LogicalType::VARCHAR,
                        LogicalType::VARCHAR,
                        LogicalType::UBIGINT,
                        LogicalType::UBIGINT,
                        LogicalType::UBIGINT};
  auto data          = make_uniq<rows_data>();
  auto const last    = Value::UBIGINT(gateway.last_query_id());
  auto const running = Value::UBIGINT(gateway.running_query_id());
  auto const queued  = Value::UBIGINT(gateway.waiting());
  for (auto const& r : gateway.status()) {
    data->rows.push_back({Value::UINTEGER(r.index),
                          Value(r.name),
                          Value::BOOLEAN(r.registered),
                          Value::BIGINT(r.pid),
                          Value(r.gpu),
                          Value(r.host),
                          Value(r.uri),
                          Value(r.state),
                          Value(sirius_gateway::to_string(r.cluster)),
                          r.error.empty() ? Value(LogicalType::VARCHAR) : Value(r.error),
                          last,
                          running,
                          queued});
  }
  return std::move(data);
}

std::string string_setting(ClientContext& context, char const* name)
{
  Value value;
  return context.TryGetCurrentSetting(name, value) && !value.IsNull() ? value.ToString()
                                                                      : std::string();
}

/// The start options that come from `sirius_gateway_*` settings.
void read_settings(ClientContext& context, sirius_gateway::start_options& options)
{
  options.duckdb_binary    = string_setting(context, "sirius_gateway_duckdb_binary");
  options.sirius_extension = string_setting(context, "sirius_gateway_sirius_extension");
  options.rank_init        = string_setting(context, "sirius_gateway_rank_init");
  options.rank_launcher    = string_setting(context, "sirius_gateway_rank_launcher");
  options.log_dir          = string_setting(context, "sirius_gateway_log_dir");
  options.boot_id_sql      = string_setting(context, "sirius_gateway_rank_boot_id_sql");
  options.exit_sql         = string_setting(context, "sirius_gateway_rank_exit_sql");
  if (options.log_dir.empty()) {
    options.log_dir =
      (std::filesystem::temp_directory_path() / ("sirius_gateway_" + std::to_string(getpid())))
        .string();
  }
}

unique_ptr<FunctionData> start_bind(ClientContext& context,
                                    TableFunctionBindInput& input,
                                    vector<LogicalType>& types,
                                    vector<Identifier>& names)
{
  sirius_gateway::start_options options;
  options.ranks = input.inputs[0].GetValue<uint32_t>();
  auto list     = [](Value const& value) {
    std::vector<std::string> items;
    if (value.IsNull() || value.ToString().empty()) { return items; }
    for (auto item : StringUtil::Split(value.ToString(), ',')) {
      StringUtil::Trim(item);
      items.push_back(item);
    }
    return items;
  };
  for (auto const& [name, value] : input.named_parameters) {
    auto const key = StringUtil::Lower(name.GetIdentifierName());
    if (key == "gpus") {
      options.gpus = list(value);
    } else if (key == "hosts") {
      options.hosts = list(value);
    } else if (key == "exchange_base_port") {
      options.exchange_base_port = value.GetValue<uint32_t>();
    } else if (key == "quack_base_port") {
      options.quack_base_port = value.GetValue<uint32_t>();
    }
  }
  read_settings(context, options);
  auto& gateway = gateway_of(input);
  gateway.start(context, std::move(options));
  return status_rows(gateway, types, names);
}

unique_ptr<FunctionData> stop_bind(ClientContext& context,
                                   TableFunctionBindInput& input,
                                   vector<LogicalType>& types,
                                   vector<Identifier>& names)
{
  gateway_of(input).stop(context);
  return status_rows(gateway_of(input), types, names);
}

/// sirius_gateway_add_rank('host:port', exchange_port := P, token := T[, fingerprint := F,
/// name := N, exchange_host := H]): the rank's Quack address, and where its peers reach its
/// exchange (default: the Quack host).
unique_ptr<FunctionData> add_rank_bind(ClientContext& context,
                                       TableFunctionBindInput& input,
                                       vector<LogicalType>& types,
                                       vector<Identifier>& names)
{
  sirius_gateway::rank_spec spec;
  auto address = input.inputs[0].ToString();
  if (StringUtil::StartsWith(address, "quack:")) { address = address.substr(6); }
  auto const colon = address.rfind(':');
  if (colon == std::string::npos) {
    throw InvalidInputException("sirius_gateway_add_rank: '%s' is not host:port", address);
  }
  spec.quack_host = address.substr(0, colon);
  spec.quack_port = std::stoul(address.substr(colon + 1));
  for (auto const& [name, value] : input.named_parameters) {
    auto const key = StringUtil::Lower(name.GetIdentifierName());
    if (key == "exchange_port") {
      spec.exchange_port = value.GetValue<uint32_t>();
    } else if (key == "token") {
      spec.token = value.ToString();
    } else if (key == "fingerprint") {
      spec.fingerprint = value.ToString();
    } else if (key == "name") {
      spec.name = value.ToString();
    } else if (key == "exchange_host") {
      spec.exchange_host = value.ToString();
    }
  }
  if (spec.exchange_port == 0 || spec.token.empty()) {
    throw InvalidInputException("sirius_gateway_add_rank needs exchange_port := and token :=");
  }
  if (spec.name.empty()) { spec.name = address; }
  if (spec.exchange_host.empty()) { spec.exchange_host = spec.quack_host; }
  gateway_of(input).add_rank(context, std::move(spec));
  return status_rows(gateway_of(input), types, names);
}

unique_ptr<FunctionData> drop_rank_bind(ClientContext& context,
                                        TableFunctionBindInput& input,
                                        vector<LogicalType>& types,
                                        vector<Identifier>& names)
{
  gateway_of(input).drop_rank(context, input.inputs[0].ToString());
  return status_rows(gateway_of(input), types, names);
}

/// Test hook: sirius_gateway_test_serve(port, token, init_sql) starts a CPU DuckDB that runs
/// init_sql and serves Quack, for tests to add as a rank. Console only.
unique_ptr<FunctionData> test_serve_bind(ClientContext& context,
                                         TableFunctionBindInput& input,
                                         vector<LogicalType>& types,
                                         vector<Identifier>& names)
{
  sirius_gateway::start_options options;
  read_settings(context, options);
  options.sirius_extension.clear();
  gateway_of(input).test_serve(options,
                               input.inputs[0].GetValue<uint32_t>(),
                               input.inputs[1].ToString(),
                               input.inputs[2].ToString());
  return status_rows(gateway_of(input), types, names);
}

unique_ptr<FunctionData> restart_bind(ClientContext& context,
                                      TableFunctionBindInput& input,
                                      vector<LogicalType>& types,
                                      vector<Identifier>& names)
{
  gateway_of(input).restart(context);
  return status_rows(gateway_of(input), types, names);
}

unique_ptr<FunctionData> status_bind(ClientContext&,
                                     TableFunctionBindInput& input,
                                     vector<LogicalType>& types,
                                     vector<Identifier>& names)
{
  return status_rows(gateway_of(input), types, names);
}

/// Test hook: sends a signal to one rank's process, to stand in for a crash (SIGKILL) or a rank
/// that stops answering (SIGSTOP). Console only, like every sirius_gateway_* function.
unique_ptr<FunctionData> test_signal_bind(ClientContext&,
                                          TableFunctionBindInput& input,
                                          vector<LogicalType>& types,
                                          vector<Identifier>& names)
{
  gateway_of(input).signal_rank(input.inputs[0].GetValue<uint32_t>(),
                                input.inputs[1].GetValue<int32_t>());
  return status_rows(gateway_of(input), types, names);
}

unique_ptr<FunctionData> log_bind(ClientContext&,
                                  TableFunctionBindInput& input,
                                  vector<LogicalType>& types,
                                  vector<Identifier>& names)
{
  names     = {"entry", "sql"};
  types     = {LogicalType::UBIGINT, LogicalType::VARCHAR};
  auto data = make_uniq<rows_data>();
  auto log  = gateway_of(input).ddl_log();
  for (idx_t i = 0; i < log.size(); ++i) {
    data->rows.push_back({Value::UBIGINT(i + 1), Value(log[i])});
  }
  return std::move(data);
}

/// DuckDB needs the result's schema at bind time, and only rank 0's result has it, so the bind
/// runs the statement on every rank.
unique_ptr<FunctionData> run_bind(ClientContext& context,
                                  TableFunctionBindInput& input,
                                  vector<LogicalType>& types,
                                  vector<Identifier>& names)
{
  auto data    = make_uniq<run_data>();
  data->result = gateway_of(input).run(context, input.inputs[0].ToString());
  types        = data->result->types;
  for (auto const& name : data->result->names) {
    names.emplace_back(name);
  }
  if (types.empty()) {
    types = {LogicalType::BOOLEAN};
    names = {"Success"};
  }
  return std::move(data);
}

void scan_run(ClientContext&, TableFunctionInput& input, DataChunk& output)
{
  auto& chunks = input.bind_data->Cast<run_data>().result->chunks;
  auto& state  = input.global_state->Cast<scan_state>();
  if (state.offset < chunks.size()) {
    chunks[state.offset++]->Copy(output);
  } else {
    output.SetCardinality(0);
  }
}

void route_function(DataChunk& args, ExpressionState&, Vector& result)
{
  BinaryExecutor::Execute<string_t, string_t, string_t>(
    args.data[0], args.data[1], result, args.size(), [&](string_t, string_t sql) {
      return StringVector::AddString(result, sirius_gateway::route(sql.GetString()));
    });
}

void LoadInternal(ExtensionLoader& loader)
{
  auto& db     = loader.GetDatabaseInstance();
  auto gateway = make_shared_ptr<gateway_info>(std::make_shared<gateway_state>(db));

  auto add = [&](char const* name,
                 vector<LogicalType> args,
                 table_function_t scan,
                 table_function_bind_t bind) {
    TableFunction function(name, std::move(args), scan, bind, init_scan);
    function.function_info = gateway;
    return function;
  };
  auto start = add("sirius_gateway_start", {}, scan_rows, start_bind);
  start.GetSignature().AddParameter("ranks", LogicalType::INTEGER);
  start.GetSignature().AddKeywordOnly("gpus", LogicalType::VARCHAR, Value(""));
  start.GetSignature().AddKeywordOnly("hosts", LogicalType::VARCHAR, Value(""));
  start.GetSignature().AddKeywordOnly(
    "exchange_base_port", LogicalType::INTEGER, Value::INTEGER(27000));
  start.GetSignature().AddKeywordOnly(
    "quack_base_port", LogicalType::INTEGER, Value::INTEGER(9500));
  loader.RegisterFunction(start);
  auto add_rank = add("sirius_gateway_add_rank", {LogicalType::VARCHAR}, scan_rows, add_rank_bind);
  add_rank.GetSignature().AddKeywordOnly("exchange_port", LogicalType::INTEGER, Value::INTEGER(0));
  add_rank.GetSignature().AddKeywordOnly("token", LogicalType::VARCHAR, Value(""));
  add_rank.GetSignature().AddKeywordOnly("fingerprint", LogicalType::VARCHAR, Value(""));
  add_rank.GetSignature().AddKeywordOnly("name", LogicalType::VARCHAR, Value(""));
  add_rank.GetSignature().AddKeywordOnly("exchange_host", LogicalType::VARCHAR, Value(""));
  loader.RegisterFunction(add_rank);
  loader.RegisterFunction(
    add("sirius_gateway_drop_rank", {LogicalType::VARCHAR}, scan_rows, drop_rank_bind));
  loader.RegisterFunction(add("sirius_gateway_test_serve",
                              {LogicalType::INTEGER, LogicalType::VARCHAR, LogicalType::VARCHAR},
                              scan_rows,
                              test_serve_bind));
  loader.RegisterFunction(add("sirius_gateway_stop", {}, scan_rows, stop_bind));
  loader.RegisterFunction(add("sirius_gateway_restart", {}, scan_rows, restart_bind));
  loader.RegisterFunction(add("sirius_gateway_status", {}, scan_rows, status_bind));
  loader.RegisterFunction(add("sirius_gateway_log", {}, scan_rows, log_bind));
  loader.RegisterFunction(add("sirius_gateway_test_signal",
                              {LogicalType::INTEGER, LogicalType::INTEGER},
                              scan_rows,
                              test_signal_bind));
  loader.RegisterFunction(add("sirius_gateway_run", {LogicalType::VARCHAR}, scan_run, run_bind));

  // The Quack callback that routes each client statement:
  // SET GLOBAL quack_authorization_function = 'sirius_gateway_route'.
  ScalarFunction route("sirius_gateway_route",
                       {LogicalType::VARCHAR, LogicalType::VARCHAR},
                       LogicalType::VARCHAR,
                       route_function);
  route.SetVolatile();
  route.SetFallible();
  loader.RegisterFunction(route);

  auto& config = DBConfig::GetConfig(db);
  config.AddExtensionOption("sirius_gateway_duckdb_binary",
                            "DuckDB binary that runs each rank",
                            LogicalType::VARCHAR,
                            Value(""));
  config.AddExtensionOption("sirius_gateway_sirius_extension",
                            "Sirius extension each rank loads; empty runs CPU ranks without Sirius",
                            LogicalType::VARCHAR,
                            Value(""));
  config.AddExtensionOption("sirius_gateway_rank_init",
                            "SQL each rank's session runs after the exchange settings",
                            LogicalType::VARCHAR,
                            Value(""));
  config.AddExtensionOption(
    "sirius_gateway_rank_launcher",
    "Program that starts a rank on another host: <host> <gpu> <quack port> <duckdb> <init file>",
    LogicalType::VARCHAR,
    Value(""));
  config.AddExtensionOption("sirius_gateway_log_dir",
                            "Directory for rank init files and logs; empty uses a temporary one",
                            LogicalType::VARCHAR,
                            Value(""));
  // A rank waiting on a peer that never ran the query has no bound of its own, so 0 can hold
  // the gateway's one turn forever.
  config.AddExtensionOption(
    "sirius_gateway_statement_timeout",
    "Seconds one statement may run before it is cancelled on every rank; 0 is none. Raise it "
    "for long queries",
    LogicalType::UBIGINT,
    Value::UBIGINT(3600));
  config.AddExtensionOption("sirius_gateway_cancel_grace",
                            "Seconds a cancelled statement may keep running on a rank before the "
                            "gateway kills the rank and restarts every rank",
                            LogicalType::UBIGINT,
                            Value::UBIGINT(30));
  config.AddExtensionOption("sirius_gateway_rank_boot_id_sql",
                            "Query that names a registered rank's process; it changes when the "
                            "rank restarts",
                            LogicalType::VARCHAR,
                            Value("SELECT boot_id FROM sirius_rank_info()"));
  config.AddExtensionOption("sirius_gateway_rank_exit_sql",
                            "Statement that ends a registered rank's process, for its launcher to "
                            "start it again",
                            LogicalType::VARCHAR,
                            Value("CALL sirius_rank_exit()"));
  config.AddExtensionOption("sirius_gateway_query_id_setting",
                            "Setting that carries the query number to each rank; empty sends none",
                            LogicalType::VARCHAR,
                            Value("sirius_distributed_query_id"));
}

}  // namespace

void SiriusGatewayExtension::Load(ExtensionLoader& loader) { LoadInternal(loader); }
std::string SiriusGatewayExtension::Name() { return "sirius_gateway"; }
std::string SiriusGatewayExtension::Version() const { return ""; }

}  // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(sirius_gateway, loader) { duckdb::LoadInternal(loader); }
}
