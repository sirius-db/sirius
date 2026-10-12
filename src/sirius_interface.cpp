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
#include "log/logging.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_streaming_source.hpp"
#include "sirius_context.hpp"
#include "transparent/read_view_registry.hpp"

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
  duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
  const duckdb::QueryParameters& parameters,
  sirius::query_id_t query_id)
{
  D_ASSERT(!sirius_active_query);
  try {
    sirius_active_query                  = duckdb::make_uniq<sirius_active_query_context>();
    sirius_active_query->query           = query;
    sirius_active_query->sirius_prepared = std::move(statement_p);
    auto& prepared                       = *sirius_active_query->sirius_prepared;
    duckdb::identifier_map_t<duckdb::BoundParameterData> empty_parameters;
    prepared.prepared->Bind(
      context, parameters.statement_args ? *parameters.statement_args : empty_parameters);

    sirius_active_query->engine =
      duckdb::make_uniq<sirius_engine>(context, query_id, query_label, session_label);
    auto& engine = *sirius_active_query->engine;
    auto collector =
      duckdb::make_uniq<op::sirius_physical_materialized_collector>(prepared, context);
    D_ASSERT(collector->result_column_types == prepared.prepared->types);
    engine.initialize(std::move(collector));
    engine.execute();
    auto result = engine.get_result();
    // Detach the materialized result before releasing the plan and executor.
    cleanup_internal();
    return result;
  } catch (duckdb::SiriusBeginWindowFailureException&) {
    // Preserve the dynamic type: a partially mutated runtime must not fall back.
    cleanup_internal();
    throw;
  } catch (duckdb::SiriusRuntimeUnavailableException&) {
    // The caller routes this typed error to the transaction-preserving fallback.
    cleanup_internal();
    throw;
  } catch (std::exception& e) {
    cleanup_internal();
    duckdb::ErrorData error(e);
    sirius_process_error(error, query);
    return duckdb::make_uniq<duckdb::QueryResult>(std::move(error));
  } catch (...) {
    cleanup_internal();
    throw;
  }
}

}  // namespace sirius
