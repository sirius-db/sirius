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

#pragma once

#include "data/data_repository_manager_registry.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/query_parameters.hpp"
#include "sirius_engine.hpp"

#include <optional>

namespace sirius {
namespace transparent {
class read_view_registry;
}

class sirius_prepared_statement_data {
 public:
  sirius_prepared_statement_data(
    duckdb::shared_ptr<duckdb::PreparedStatementData> _prepared,
    duckdb::unique_ptr<op::sirius_physical_operator> _sirius_physical_plan);
  //! The sirius physical plan
  duckdb::unique_ptr<op::sirius_physical_operator> sirius_physical_plan;
  //! The prepared statement data
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared;
  //! Query-local contracts collected from the finalized GPU plan.
  std::shared_ptr<transparent::read_view_registry> read_views;
};

struct sirius_active_query_context {
 public:
  //! The query that is currently being executed
  duckdb::string query;
  //! Prepared statement data
  duckdb::shared_ptr<sirius_prepared_statement_data> sirius_prepared;
  //! The query executor
  duckdb::unique_ptr<sirius_engine> engine;
};

class sirius_interface {
 public:
  sirius_interface(duckdb::ClientContext& client_context,
                   std::optional<std::string> query_label   = std::nullopt,
                   std::optional<std::string> session_label = std::nullopt);
  //! The client context
  duckdb::ClientContext& client_context;
  //! Optional label for this query's telemetry instance name
  std::optional<std::string> query_label;
  //! Optional sticky per-connection label selecting the telemetry query group
  std::optional<std::string> session_label;
  //! The currently active query context
  duckdb::unique_ptr<sirius_active_query_context> sirius_active_query;
  //! Execute the query
  duckdb::unique_ptr<duckdb::QueryResult> sirius_execute_query(
    duckdb::ClientContext& context,
    const duckdb::string& query,
    duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
    const duckdb::QueryParameters& parameters,
    sirius::query_id_t query_id);

 private:
  void cleanup_internal();
  void sirius_process_error(duckdb::ErrorData& error, const duckdb::string& query) const;
};

}  // namespace sirius
