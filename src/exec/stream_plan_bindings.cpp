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

#include <stdexcept>

namespace sirius::exec {

namespace {

/// Resolve the catalog registered on this connection, or explain why the fragment is not set up.
duckdb::shared_ptr<stream_bind_catalog> catalog_for(duckdb::ClientContext& context)
{
  auto catalog = context.registered_state->Get<stream_bind_catalog>(stream_bind_catalog::kStateKey);
  if (!catalog) {
    throw std::runtime_error(
      "sirius_stream_source: no stream catalog on this connection — the fragment must declare its "
      "input streams before the plan is bound");
  }
  return catalog;
}

duckdb::unique_ptr<duckdb::FunctionData> stream_source_bind(
  duckdb::ClientContext& context,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& return_types,
  duckdb::vector<std::string>& names)
{
  if (input.inputs.size() != 1 || input.inputs[0].IsNull()) {
    throw std::runtime_error("sirius_stream_source expects a single non-null stream id");
  }
  auto const stream_id = static_cast<stream_id_t>(input.inputs[0].GetValue<std::int64_t>());

  // An unregistered id is a defined bind-time error, not a silent empty scan: it means the
  // fragment's plan references a stream nobody declared, which would otherwise surface much
  // later as a plan with no source.
  auto const& binding = catalog_for(context)->get(stream_id);

  names        = duckdb::vector<std::string>(binding.names.begin(), binding.names.end());
  return_types = sirius::to_duckdb_vec(binding.types);
  return duckdb::make_uniq<stream_source_bind_data>(stream_id);
}

/// Never runs. The Sirius plan generator replaces this scan with a STREAMING_SOURCE, and this
/// path has no CPU fallback — a stream's batches only exist on the GPU side of the fragment.
void stream_source_function(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&)
{
  throw std::runtime_error(
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
  // Projection pushdown is deliberately off: a streaming source hands over whole batches in the
  // tier they already sit in, so there is no per-column read to prune.
  duckdb::CreateTableFunctionInfo info(stream_source);
  catalog.CreateTableFunction(transaction, info);
}

}  // namespace sirius::exec
