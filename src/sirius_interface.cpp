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

#include "sirius_interface.hpp"

#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/query_result.hpp"
#include "duckdb/main/settings.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_parquet_sink.hpp"
#include "op/sirius_physical_streaming_source.hpp"
#include "sirius_context.hpp"
#include "transparent/read_view_registry.hpp"

#include <absl/cleanup/cleanup.h>
#include <duckdb/common/types/column/column_data_collection.hpp>
#include <duckdb/common/types/data_chunk.hpp>

#include <optional>
#include <stdexcept>

namespace sirius {
namespace {
void collect_read_views(op::sirius_physical_operator const& node,
                        std::shared_ptr<transparent::read_view_registry>& result)
{
  std::shared_ptr<transparent::read_view_registry> candidate;
  if (node.type == op::SiriusPhysicalOperatorType::GPU_SCAN) {
    candidate = node.Cast<op::scan::sirius_gpu_scan_operator>().read_views();
  } else if (node.type == op::SiriusPhysicalOperatorType::STREAMING_SOURCE) {
    candidate = node.Cast<op::sirius_physical_streaming_source>().read_views();
  }
  if (candidate) {
    if (result && result != candidate) {
      throw std::runtime_error("finalized GPU plan contains multiple read-view registries");
    }
    result = std::move(candidate);
  }
  for (auto const& child : node.get_children()) {
    collect_read_views(child.get(), result);
  }
}
}  // namespace

sirius_prepared_statement_data::sirius_prepared_statement_data(
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared_p,
  duckdb::unique_ptr<op::sirius_physical_operator> sirius_physical_plan_p)
  : sirius_physical_plan(std::move(sirius_physical_plan_p)), prepared(std::move(prepared_p))
{
  if (sirius_physical_plan) { collect_read_views(*sirius_physical_plan, read_views); }
}

sirius_interface::sirius_interface(duckdb::ClientContext& client_context,
                                   std::optional<std::string> query_label,
                                   std::optional<std::string> session_label)
  : client_context(client_context),
    query_label(std::move(query_label)),
    session_label(std::move(session_label))
{
}

void sirius_interface::sirius_process_error(duckdb::ErrorData& error,
                                            const duckdb::string& query) const
{
  error.FinalizeError();
  if (duckdb::Settings::Get<duckdb::ErrorsAsJSONSetting>(client_context)) {
    error.ConvertErrorToJSON();
  } else {
    error.AddErrorLocation(query);
  }
}

void sirius_interface::cleanup_internal() { sirius_active_query.reset(); }

duckdb::unique_ptr<duckdb::QueryResult> sirius_interface::sirius_execute_query(
  duckdb::ClientContext& context,
  const duckdb::string& query,
  duckdb::shared_ptr<sirius_prepared_statement_data> statement,
  sirius::query_id_t query_id)
{
  D_ASSERT(!sirius_active_query);
  try {
    absl::Cleanup cleanup                = [this] { cleanup_internal(); };
    sirius_active_query                  = duckdb::make_uniq<sirius_active_query_context>();
    sirius_active_query->query           = query;
    sirius_active_query->sirius_prepared = std::move(statement);
    auto& prepared                       = *sirius_active_query->sirius_prepared;
    duckdb::identifier_map_t<duckdb::BoundParameterData> empty_parameters;
    prepared.prepared->Bind(context, empty_parameters);

    sirius_active_query->engine =
      duckdb::make_uniq<sirius_engine>(context, query_id, query_label, session_label);
    auto& engine = *sirius_active_query->engine;
    if (prepared.sirius_physical_plan->type == op::SiriusPhysicalOperatorType::COPY_TO_FILE) {
      auto* sink = &prepared.sirius_physical_plan->Cast<op::sirius_physical_parquet_sink>();
      engine.initialize(std::move(prepared.sirius_physical_plan));
      try {
        engine.execute();  // Returns only after all asynchronous plan users retire.
        if (context.IsInterrupted()) throw duckdb::InterruptException();
        auto runtime = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
        auto gpu_spaces =
          runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
        if (gpu_spaces.empty()) throw internal_exception("Parquet export has no configured GPU");
        duckdb::unique_ptr<duckdb::QueryResult> completion;
        sink->finish(
          gpu_spaces.front()->get_device_id(), [&](output::parquet_export_result const& result) {
            auto collection =
              duckdb::make_uniq<duckdb::ColumnDataCollection>(context, prepared.prepared->types);
            duckdb::ColumnDataAppendState append;
            collection->InitializeAppend(append);
            duckdb::DataChunk chunk;
            chunk.Initialize(duckdb::Allocator::Get(context), prepared.prepared->types);
            chunk.data[0].Append(duckdb::Value::BIGINT(static_cast<int64_t>(result.rows)));
            chunk.CheckCardinality(1);
            collection->Append(append, chunk);
            completion = duckdb::make_uniq<duckdb::QueryResult>(prepared.prepared->statement_type,
                                                                prepared.prepared->properties,
                                                                prepared.prepared->names,
                                                                std::move(collection),
                                                                context.GetClientProperties());
          });
        return completion;
      } catch (...) {
        // A failed retirement retains the plan in the lifecycle registry. Leave
        // cleanup to its eventual destruction rather than racing live writers.
        if (engine.plan_users_quiescent()) sink->abort();
        throw;
      }
    }
    auto collector =
      duckdb::make_uniq<op::sirius_physical_materialized_collector>(prepared, context);
    D_ASSERT(collector->result_column_types == prepared.prepared->types);
    engine.initialize(std::move(collector));
    engine.execute();
    // Detach the materialized result before releasing the plan and executor.
    return engine.get_result();
  } catch (duckdb::SiriusBeginWindowFailureException&) {
    // Preserve the dynamic type: a partially mutated runtime must not fall back.
    throw;
  } catch (duckdb::SiriusRuntimeUnavailableException&) {
    // The caller routes this typed error to the transaction-preserving fallback.
    throw;
  } catch (std::exception& e) {
    if (auto runtime = context.registered_state->Get<duckdb::SiriusContext>("sirius_state"))
      runtime->get_query_lifecycle_registry().report_failure(std::current_exception(), query_id);
    duckdb::ErrorData error(e);
    sirius_process_error(error, query);
    return duckdb::make_uniq<duckdb::QueryResult>(std::move(error));
  }
}

}  // namespace sirius
