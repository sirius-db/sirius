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
#include "sirius_sql_rewrite.hpp"
#include "transparent/sirius_optimizer_extension.hpp"

#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/execution/executor.hpp>
#include <duckdb/execution/operator/helper/physical_result_collector.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/pending_query_result.hpp>
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
  // Private executor for the CPU fallback plan (see run_cpu_fallback_plan). Kept
  // alive here so the materialized result's backing pipelines outlive GetData.
  duckdb::unique_ptr<duckdb::Executor> cpu_executor;

  duckdb::idx_t MaxThreads() override { return 1; }
};

// Run a stored DuckDB CPU plan on a private Executor. Reusing the outer query's
// ClientContext (not a fresh Connection) keeps it on the same transaction and MVCC
// snapshot, including this transaction's uncommitted writes. Returns a materialized
// result; throws if the CPU plan itself fails.
duckdb::unique_ptr<duckdb::QueryResult> run_cpu_fallback_plan(
  duckdb::ClientContext& client,
  duckdb::PreparedStatementData& cpu_prepared,
  duckdb::unique_ptr<duckdb::Executor>& out_executor)
{
  // Force a materialized (non-streaming) result so the whole plan runs before any
  // row is streamed out of the operator.
  cpu_prepared.output_type = duckdb::QueryResultOutputType::FORCE_MATERIALIZED;
  cpu_prepared.memory_type = duckdb::QueryResultMemoryType::IN_MEMORY;

  auto collector = duckdb::PhysicalResultCollector::GetResultCollector(client, cpu_prepared);
  D_ASSERT(collector->type == duckdb::PhysicalOperatorType::RESULT_COLLECTOR);

  // Suppress profiling for the nested run: it shares the context's single
  // QueryProfiler with the outer query, so letting it re-initialize would corrupt
  // the outer profile. No-op when profiling is already off. Restored on every path.
  auto& client_config              = duckdb::ClientConfig::GetConfig(client);
  const bool saved_enable_profiler = client_config.enable_profiler;
  client_config.enable_profiler    = false;

  out_executor   = duckdb::make_uniq<duckdb::Executor>(client);
  auto& executor = *out_executor;
  try {
    executor.Initialize(std::move(collector));
    duckdb::PendingExecutionResult exec_result;
    while (!duckdb::PendingQueryResult::IsResultReady(exec_result = executor.ExecuteTask())) {
      if (exec_result == duckdb::PendingExecutionResult::BLOCKED) { executor.WaitForTask(); }
    }
    if (executor.HasError()) { executor.ThrowException(); }
    auto result                   = executor.GetResult();
    client_config.enable_profiler = saved_enable_profiler;
    return result;
  } catch (...) {
    // Drain any still-registered tasks before the executor is destroyed
    // (~Executor asserts executor_tasks == 0), then restore profiling.
    executor.CancelTasks();
    client_config.enable_profiler = saved_enable_profiler;
    throw;
  }
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
PhysicalSiriusExecution::PhysicalSiriusExecution(
  duckdb::PhysicalPlan& physical_plan,
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan,
  std::string query_sql,
  duckdb::vector<duckdb::LogicalType> types,
  duckdb::vector<std::string> names,
  duckdb::shared_ptr<duckdb::PreparedStatementData> cpu_fallback_prepared,
  bool cpu_plan_reads_s3,
  duckdb::idx_t estimated_cardinality)
  : duckdb::PhysicalOperator(
      physical_plan, PhysicalSiriusExecution::TYPE, std::move(types), estimated_cardinality),
    logical_plan_(std::move(logical_plan)),
    query_sql_(std::move(query_sql)),
    result_names_(std::move(names)),
    cpu_fallback_prepared_(std::move(cpu_fallback_prepared)),
    cpu_plan_reads_s3_(cpu_plan_reads_s3)
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

    // Attempt GPU execution. Any failure — a thrown exception or an error-carrying
    // result — is captured in gpu_error and routed to the CPU fallback below. The
    // result is fully materialized before the first Fetch, so falling back here
    // cannot duplicate rows.
    duckdb::ErrorData gpu_error;
    bool gpu_failed = false;
    try {
      if (!logical_plan_ && query_sql_.empty()) {
        throw duckdb::ExecutorException(
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
          fresh_plan = sirius::transparent::copy_logical_plan(*logical_plan_, context.client);
        } catch (duckdb::NotImplementedException&) {
          // Drop logical_plan_ — we know it can't be copied, so future executes
          // will skip straight to the replan path.
          logical_plan_.reset();
        }
      }
      if (!fresh_plan) {
        auto sirius_ctx =
          context.client.registered_state->Get<duckdb::SiriusContext>("sirius_state");
        duckdb::unique_ptr<duckdb::SiriusContext::InternalQueryGuard> guard;
        if (sirius_ctx) {
          guard = duckdb::make_uniq<duckdb::SiriusContext::InternalQueryGuard>(*sirius_ctx);
        }
        duckdb::Parser parser(context.client.GetParserOptions());
        parser.ParseQuery(query_sql_);
        if (parser.statements.size() != 1) {
          throw duckdb::ExecutorException(
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

      // TEST-ONLY fault injection: plan generation has already succeeded, so failing
      // here exercises the runtime CPU fallback path (not the plan-time path).
      {
        duckdb::Value inject;
        if (context.client.TryGetCurrentSetting("sirius_test_inject_transparent_gpu_error",
                                                inject) &&
            !inject.IsNull() && !inject.ToString().empty()) {
          throw duckdb::ExecutorException("injected transparent GPU failure: " + inject.ToString());
        }
      }

      // Execute via the standard sirius_interface path.
      duckdb::PendingQueryParameters parameters;
      state.result = state.iface->sirius_execute_query(
        context.client, "transparent_execution", gpu_prepared, parameters);

      if (state.result->HasError()) {
        gpu_error = state.result->GetErrorObject();
        state.result.reset();
        gpu_failed = true;
      }
    } catch (std::exception& e) {
      gpu_error  = duckdb::ErrorData(e);
      gpu_failed = true;
    }

    if (gpu_failed) {
      const std::string gpu_msg = gpu_error.RawMessage();
      SIRIUS_LOG_ERROR("Transparent GPU execution error: {}", gpu_msg);

      // A user interrupt is never a fallback candidate — propagate it as-is.
      if (gpu_error.Type() == duckdb::ExceptionType::INTERRUPT) { gpu_error.Throw(); }

      // S3 is GPU-only: DuckDB's CPU read_parquet cannot serve Sirius-owned s3://,
      // so surface a clear error instead of a fallback that would fail anyway.
      if (cpu_plan_reads_s3_ || sirius::references_sirius_owned_s3_parquet(query_sql_)) {
        throw duckdb::ExecutorException(
          "S3 CPU fallback is not supported: this query reads s3:// data, GPU execution failed, "
          "and Sirius has no CPU fallback for S3 data sources. Underlying GPU error: " +
          gpu_msg);
      }

      // Fallback disabled, or no CPU plan stashed: surface the GPU error. Sanitize
      // INTERNAL/FATAL types (which would invalidate the whole database/session)
      // down to a plain ExecutorException; preserve other types.
      if (!cpu_fallback_prepared_ || !duckdb::duckdb_fallback_enabled(context.client)) {
        if (gpu_error.Type() == duckdb::ExceptionType::INTERNAL ||
            gpu_error.Type() == duckdb::ExceptionType::FATAL) {
          throw duckdb::ExecutorException("Sirius GPU execution failed: " + gpu_msg);
        }
        gpu_error.Throw("Sirius GPU execution failed: ");
      }

      // Fall back: run the stored CPU plan on a private executor bound to the same
      // ClientContext (same transaction / MVCC snapshot).
      duckdb::print_cpu_fallback_banner();
      SIRIUS_LOG_WARN(
        "Transparent execution: GPU execution failed at runtime; falling back to DuckDB CPU "
        "within the same transaction. GPU error: {}",
        gpu_msg);
      if (state.sirius_context) { state.sirius_context->record_transparent_runtime_fallback(); }

      // CpuFallbackGuard marks the replay so sirius_httpfs refuses to serve s3://
      // reached indirectly (e.g. through a view) to the CPU plan.
      std::optional<duckdb::SiriusContext::CpuFallbackGuard> fallback_guard;
      if (state.sirius_context) { fallback_guard.emplace(*state.sirius_context); }
      state.result =
        run_cpu_fallback_plan(context.client, *cpu_fallback_prepared_, state.cpu_executor);
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
