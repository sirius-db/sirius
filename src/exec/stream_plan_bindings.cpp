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

#include "exec/stream_plan_bindings.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/main/client_context.hpp"
#include "helper/type_conversions.hpp"
#include "sirius/exception.hpp"

#include <string>

namespace sirius::exec {

namespace {

duckdb::unique_ptr<duckdb::FunctionData> stream_source_bind(
  duckdb::ClientContext& context,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<std::string>& names)
{
  if (input.inputs.size() != 1 || input.inputs[0].IsNull()) {
    throw sirius::invalid_input_exception(
      "sirius_stream_source expects a single non-null stream id");
  }
  // SQL hands this over as a signed INT64. Reject a negative before the cast, or it wraps to a
  // huge unsigned id and the failure surfaces as a confusing "no input stream declared with id
  // 18446744073709551615".
  auto const signed_id = input.inputs[0].GetValue<std::int64_t>();
  if (signed_id < 0) {
    throw sirius::invalid_input_exception("sirius_stream_source: stream id must not be negative (" +
                                          std::to_string(signed_id) + ")");
  }
  auto const stream_id = static_cast<stream_id_t>(signed_id);

  // Undeclared id = bind error (not a silent empty scan).
  auto const& binding = catalog_for(context)->get(stream_id);

  names        = duckdb::vector<std::string>(binding.names.begin(), binding.names.end());
  return_types = sirius::to_duckdb_vec(binding.types);
  return duckdb::make_uniq<stream_source_bind_data>(stream_id);
}

/// Never runs: plan generator replaces this scan with STREAMING_SOURCE.
void stream_source_function(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&)
{
  throw sirius::invalid_input_exception(
    "sirius_stream_source cannot be executed by DuckDB: it is an internal marker for a Sirius "
    "streaming input and is only valid inside a GPU-executed fragment plan");
}

}  // namespace

void register_stream_source_function(duckdb::DatabaseInstance& instance)
{
  auto transaction = duckdb::CatalogTransaction::GetSystemTransaction(instance);
  auto& catalog    = duckdb::Catalog::GetSystemCatalog(instance);

  duckdb::TableFunction stream_source(kStreamSourceFunctionName,
                                      {duckdb::LogicalType::BIGINT},
                                      stream_source_function,
                                      stream_source_bind);

  duckdb::CreateTableFunctionInfo info(stream_source);
  // Idempotent: extension callback and explicit callers may both register.
  info.on_conflict = duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
  catalog.CreateTableFunction(transaction, info);
}

}  // namespace sirius::exec
