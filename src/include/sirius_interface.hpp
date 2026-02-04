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

#include "duckdb/main/client_context.hpp"
#include "sirius_engine.hpp"

namespace sirius {

class sirius_prepared_statement_data {
 public:
  sirius_prepared_statement_data(
    duckdb::shared_ptr<duckdb::PreparedStatementData> _prepared,
    duckdb::unique_ptr<op::sirius_physical_operator> _sirius_physical_plan)
    : prepared(_prepared), sirius_physical_plan(std::move(_sirius_physical_plan))
  {
  }
  //! The sirius physical plan
  duckdb::unique_ptr<op::sirius_physical_operator> sirius_physical_plan;
  //! The prepared statement data
  duckdb::shared_ptr<duckdb::PreparedStatementData> prepared;
};

struct sirius_active_query_context {
 public:
  //! The query that is currently being executed
  duckdb::string query;
  //! Prepared statement data
  duckdb::shared_ptr<sirius_prepared_statement_data> sirius_prepared;
  //! The query executor
  duckdb::unique_ptr<sirius_engine> engine;
  //! The progress bar
  duckdb::unique_ptr<duckdb::ProgressBar> progress_bar;

 public:
  //! Set the open result
  void set_open_result(duckdb::BaseQueryResult& result) { open_result = &result; }
  //! Check if the result is open
  bool is_open_result(duckdb::BaseQueryResult& result) { return open_result == &result; }
  //! Check if the result has open result
  bool has_open_result() const { return open_result != nullptr; }

 private:
  //! The currently open result
  duckdb::BaseQueryResult* open_result = nullptr;
};

class sirius_interface {
 public:
  sirius_interface(duckdb::ClientContext& client_context);
  //! The client context
  duckdb::ClientContext& client_context;
  //! The currently active query context
  duckdb::unique_ptr<sirius_active_query_context> sirius_active_query;
  //! The current query progress
  duckdb::QueryProgress query_progress;
  //! Check if the pending query result is executable
  void check_executable_internal(duckdb::PendingQueryResult& pending);
  //! Fetch the result from the pending query result
  duckdb::unique_ptr<duckdb::QueryResult> fetch_result_internal(
    duckdb::PendingQueryResult& pending);
  //! Cleanup the query result
  void cleanup_internal(duckdb::BaseQueryResult* result, bool invalidate_transaction);
  //! Begin the query
  void begin_query_internal(const duckdb::string& query);
  //! End the query
  duckdb::ErrorData end_query_internal(bool success, bool invalidate_transaction);
  //! Process the error
  void sirius_process_error(duckdb::ErrorData& error, const duckdb::string& query) const;
  //! Create the error result
  template <class T>
  duckdb::unique_ptr<T> sirius_error_result(duckdb::ErrorData error,
                                            const duckdb::string& query = duckdb::string());
  //! Create the pending statement internal
  duckdb::unique_ptr<duckdb::PendingQueryResult> sirius_pending_statement_internal(
    duckdb::ClientContext& context,
    duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
    const duckdb::PendingQueryParameters& parameters);
  //! Create the pending statement or prepared statement
  duckdb::unique_ptr<duckdb::PendingQueryResult> sirius_pending_statement_or_prepared_statement(
    duckdb::ClientContext& context,
    const duckdb::string& query,
    duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
    const duckdb::PendingQueryParameters& parameters);
  //! Execute the query
  duckdb::unique_ptr<duckdb::QueryResult> sirius_execute_query(
    duckdb::ClientContext& context,
    const duckdb::string& query,
    duckdb::shared_ptr<sirius_prepared_statement_data>& statement_p,
    const duckdb::PendingQueryParameters& parameters);
  //! Execute the pending query result
  duckdb::unique_ptr<duckdb::QueryResult> sirius_execute_pending_query_result(
    duckdb::PendingQueryResult& pending);
  //! Get the sirius engine
  sirius_engine& get_sirius_engine();
};

}  // namespace sirius
