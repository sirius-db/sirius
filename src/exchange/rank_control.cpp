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

#include "exchange/rank_control.hpp"

#include "log/logging.hpp"
#include "sirius_context.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/parser/parsed_data/create_table_function_info.hpp>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <thread>

namespace sirius::exchange {
namespace {

using namespace duckdb;

/// Set once this process's exchange connects: its peers are then fixed for its life.
std::atomic<bool> exchange_joined{false};

std::string const& boot_id()
{
  static std::string const id = [] {
    std::random_device device;
    std::string hex;
    for (int i = 0; i < 4; ++i) {
      char word[9];
      std::snprintf(word, sizeof(word), "%08x", static_cast<unsigned>(device()));
      hex += word;
    }
    return hex;
  }();
  return id;
}

struct one_row : TableFunctionData {
  bool done = false;
};

unique_ptr<FunctionData> info_bind(ClientContext&,
                                   TableFunctionBindInput&,
                                   vector<LogicalType>& types,
                                   vector<Identifier>& names)
{
  types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::BOOLEAN};
  names = {"pid", "boot_id", "joined"};
  return make_uniq<one_row>();
}

void info_scan(ClientContext&, TableFunctionInput& input, DataChunk& output)
{
  auto& data = input.bind_data->CastNoConst<one_row>();
  if (data.done) { return; }
  output.data[0].Append(Value::BIGINT(getpid()));
  output.data[1].Append(Value(boot_id()));
  output.data[2].Append(Value::BOOLEAN(exchange_joined.load()));
  output.CheckCardinality(1);
  data.done = true;
}

unique_ptr<FunctionData> exit_bind(ClientContext&,
                                   TableFunctionBindInput&,
                                   vector<LogicalType>& types,
                                   vector<Identifier>& names)
{
  // The function exists in every Sirius process. Only one that joined an exchange has a reason
  // to restart, so no client can end a process that runs alone.
  if (!exchange_joined.load()) {
    throw InvalidInputException(
      "sirius_rank_exit: this process has not joined an exchange, so it does not restart");
  }
  types = {LogicalType::BIGINT, LogicalType::BOOLEAN};
  names = {"pid", "exiting"};
  return make_uniq<one_row>();
}

/// Ends the process once the reply has left. shutdown_exchange() frees the exchange's listener
/// port and shared-memory mirrors; static destructors are skipped, since other threads still run.
/// A watchdog ends the process anyway if the shutdown hangs.
void exit_soon(shared_ptr<SiriusContext> context)
{
  std::this_thread::sleep_for(std::chrono::milliseconds{300});
  std::thread([] {
    std::this_thread::sleep_for(std::chrono::seconds{30});
    _exit(1);
  }).detach();
  SIRIUS_LOG_INFO("sirius_rank_exit: shutting down for a restart");
  if (context && context->is_initialized()) {
    try {
      context->shutdown_exchange();
    } catch (...) {  // exiting anyway
    }
  }
  try {
    sirius::log::get_sink()->flush();
  } catch (...) {
  }
  _exit(0);
}

void exit_scan(ClientContext& context, TableFunctionInput& input, DataChunk& output)
{
  auto& data = input.bind_data->CastNoConst<one_row>();
  if (data.done) { return; }
  static std::atomic<bool> exiting{false};
  if (!exiting.exchange(true)) {
    std::thread(exit_soon, context.registered_state->Get<SiriusContext>("sirius_state")).detach();
  }
  output.data[0].Append(Value::BIGINT(getpid()));
  output.data[1].Append(Value::BOOLEAN(true));
  output.CheckCardinality(1);
  data.done = true;
}

}  // namespace

void note_exchange_joined() noexcept { exchange_joined = true; }

void register_rank_control_functions(DatabaseInstance& instance)
{
  auto transaction = CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = Catalog::GetSystemCatalog(instance);
  TableFunction info("sirius_rank_info", {}, info_scan, info_bind);
  CreateTableFunctionInfo info_info(info);
  catalog.CreateTableFunction(transaction, info_info);
  TableFunction exit("sirius_rank_exit", {}, exit_scan, exit_bind);
  CreateTableFunctionInfo exit_info(exit);
  catalog.CreateTableFunction(transaction, exit_info);
}

}  // namespace sirius::exchange
