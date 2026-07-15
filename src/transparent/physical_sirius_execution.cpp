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

#include "transparent/physical_sirius_execution.hpp"

#include "log/logging.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"
#include "sirius_interface.hpp"
#include "transparent/sirius_optimizer_extension.hpp"

#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/query_result.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>

namespace sirius::transparent {

// ---------------------------------------------------------------------------
// Global source state — owns the sirius_interface and the materialized result.
// ---------------------------------------------------------------------------
struct SiriusGlobalSourceState : public duckdb::GlobalSourceState {
  duckdb::unique_ptr<sirius::sirius_interface> iface;
  duckdb::unique_ptr<duckdb::QueryResult> result;
  duckdb::unique_ptr<duckdb::DataChunk> current_chunk;
  duckdb::SiriusContext* sirius_context = nullptr;
  bool finished                         = false;

  duckdb::idx_t MaxThreads() override { return 1; }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
PhysicalSiriusExecution::PhysicalSiriusExecution(
  duckdb::PhysicalPlan& physical_plan,
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan,
  std::string query_sql,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<std::string> names,
  duckdb::idx_t estimated_cardinality)
  : duckdb::PhysicalOperator(
      physical_plan, PhysicalSiriusExecution::TYPE, std::move(types), estimated_cardinality),
    logical_plan_(std::move(logical_plan)),
    query_sql_(std::move(query_sql)),
    result_names_(std::move(names))
{
}

// ---------------------------------------------------------------------------
// Source interface
// ---------------------------------------------------------------------------
duckdb::unique_ptr<duckdb::GlobalSourceState> PhysicalSiriusExecution::GetGlobalSourceState(
  duckdb::ClientContext& context) const
{
  auto state      = duckdb::make_uniq<SiriusGlobalSourceState>();
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  auto query_label =
    sirius_ctx ? sirius_ctx->take_pending_query_label() : std::optional<std::string>{};
  state->iface = duckdb::make_uniq<sirius::sirius_interface>(context, std::move(query_label));
  state->sirius_context = sirius_ctx.get();
  return std::move(state);
}

duckdb::unique_ptr<duckdb::LocalSourceState> PhysicalSiriusExecution::GetLocalSourceState(
  duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const
{
  return duckdb::make_uniq<duckdb::LocalSourceState>();
}

duckdb::SourceResultType PhysicalSiriusExecution::GetDataInternal(
  duckdb::ExecutionContext& context,
  duckdb::DataChunk& chunk,
  duckdb::OperatorSourceInput& input) const
{
  auto& state = input.global_state.Cast<SiriusGlobalSourceState>();
  if (state.finished) { return duckdb::SourceResultType::FINISHED; }

  // Lazy execution: run the GPU query on first GetData call.
  if (!state.result) {
    SIRIUS_LOG_INFO("Transparent GPU execution: executing query");
    if (state.sirius_context) { state.sirius_context->record_transparent_execution(); }
    if (!logical_plan_ && query_sql_.empty()) {
      throw duckdb::InternalException(
        "Transparent GPU execution is missing the logical plan template");
    }

    // Build a minimal PreparedStatementData with the output schema.
    auto prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
      duckdb::StatementType::SELECT_STATEMENT);
    prepared->types = types;
    prepared->names = result_names_;

    // Rebuild a fresh Sirius physical plan for this execution. DuckDB may reuse
    // the same prepared physical operator across multiple EXECUTE calls.
    //
    // Prefer LogicalOperator::Copy when the plan supports it (cheap deep clone
    // via serialization). When the plan contains a non-serializable LogicalGet,
    // fall back to re-parsing + re-binding the unbound SQL statement, which
    // exercises the same bind path the very first run did.
    duckdb::unique_ptr<duckdb::LogicalOperator> fresh_plan;
    if (logical_plan_) {
      try {
        // Use the dynamic-filter-aware copy so LogicalComparisonJoin::filter_pushdown and
        // LogicalGet::dynamic_filters survive the serialize/deserialize round-trip — they are
        // not in DuckDB's serialization schema and would otherwise be null on the fresh plan,
        // making downstream Sirius wiring silently miss runtime-computed dynamic filters.
        //
        // DuckDB reuses the same prepared physical operator across EXECUTE calls
        // (comment above). Concurrent executions race on the mutable logical_plan_
        // unique_ptr — a reset during another thread's copy_logical_plan is
        // use-after-free. Lock to serialize the read+reset.
        static std::mutex logical_plan_mutex;
        std::lock_guard<std::mutex> lp_lock(logical_plan_mutex);
        if (logical_plan_) {
          fresh_plan = sirius::transparent::copy_logical_plan(*logical_plan_, context.client);
        }
      } catch (duckdb::NotImplementedException&) {
        // Drop logical_plan_ — we know it can't be copied, so future executes
        // will skip straight to the replan path.
        static std::mutex logical_plan_mutex;
        std::lock_guard<std::mutex> lp_lock(logical_plan_mutex);
        logical_plan_.reset();
      }
    }
    if (!fresh_plan) {
      auto sirius_ctx = context.client.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      duckdb::unique_ptr<duckdb::SiriusContext::InternalQueryGuard> guard;
      if (sirius_ctx) {
        guard = duckdb::make_uniq<duckdb::SiriusContext::InternalQueryGuard>(*sirius_ctx);
      }
      duckdb::Parser parser(context.client.GetParserOptions());
      parser.ParseQuery(query_sql_);
      if (parser.statements.size() != 1) {
        throw duckdb::InternalException(
          "Transparent GPU execution: replan expected exactly one statement");
      }
      duckdb::Planner duckdb_planner(context.client);
      duckdb_planner.CreatePlan(std::move(parser.statements[0]));
      duckdb::Optimizer optimizer(*duckdb_planner.binder, context.client);
      fresh_plan = optimizer.Optimize(std::move(duckdb_planner.plan));
    }
    sirius::planner::sirius_physical_plan_generator planner(context.client);
    auto sirius_plan = planner.create_plan(std::move(fresh_plan));

    auto gpu_prepared = duckdb::make_shared_ptr<sirius::sirius_prepared_statement_data>(
      std::move(prepared), std::move(sirius_plan));

    // Execute via the standard sirius_interface path.
    duckdb::PendingQueryParameters parameters;
    state.result = state.iface->sirius_execute_query(
      context.client, "transparent_execution", gpu_prepared, parameters);

    if (state.result->HasError()) {
      auto error_msg = state.result->GetError();
      SIRIUS_LOG_ERROR("Transparent GPU execution error: {}", error_msg);
      throw duckdb::InternalException("Sirius GPU execution failed: " + error_msg);
    }

    SIRIUS_LOG_INFO("Transparent GPU execution: query completed");
  }

  // Fetch the next chunk from the materialized result.
  state.current_chunk = state.result->Fetch();
  if (!state.current_chunk || state.current_chunk->size() == 0) {
    state.current_chunk.reset();
    state.finished = true;
    return duckdb::SourceResultType::FINISHED;
  }

  chunk.Reference(*state.current_chunk);
  return duckdb::SourceResultType::HAVE_MORE_OUTPUT;
}

}  // namespace sirius::transparent
