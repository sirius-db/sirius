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

#include "sirius_engine.hpp"

#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "io/sirius_datasource.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "pipeline/sirius_plan_printer.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius/exception.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "sirius_interface.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_repository_manager.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>

namespace sirius {

namespace {

std::shared_ptr<const telemetry::telemetry_context> get_telemetry_context_from_client_context(
  duckdb::ClientContext& context)
{
  if (not context.registered_state) {
    throw invalid_input_exception("Sirius context is not registered.");
  }

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (not sirius_ctx or not sirius_ctx->is_initialized()) {
    throw invalid_input_exception("Sirius context is not initialized.");
  }

  return sirius_ctx->get_telemetry_context();
}

}  // namespace

sirius_engine::sirius_engine(duckdb::ClientContext& context, sirius_interface& sirius_iface)
  : context(context),
    sirius_iface(sirius_iface),
    telemetry_context_(get_telemetry_context_from_client_context(this->context)),
    query_handle_(
      quent::query::create(telemetry_context_->context(),
                           quent::query::Init{
                             .instance_name  = sirius_iface.query_label.value_or("unnamed_query"),
                             .query_group_id = telemetry_context_->query_group_id(),
                           }))
{
  // The query group is session-scoped and owned by telemetry_context; every query in this
  // context is reported under it (see telemetry_context::query_group_id).
}

sirius_engine::~sirius_engine() { query_handle_->exit(); }

void sirius_engine::reset()
{
  sirius_physical_plan = nullptr;
  sirius_owned_plan.reset();
  sirius_root_pipelines.clear();
  root_pipeline_idx = 0;
  total_pipelines   = 0;
  sirius_pipelines.clear();
  new_scheduled.clear();
}

void sirius_engine::cancel_tasks()
{
  sirius_pipelines.clear();
  sirius_root_pipelines.clear();
}

bool sirius_engine::has_result_collector()
{
  return sirius_physical_plan->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR;
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_engine::get_result()
{
  D_ASSERT(has_result_collector());
  if (!sirius_physical_plan) { throw invalid_input_exception("sirius_physical_plan is NULL"); }

  auto& result_collector =
    sirius_physical_plan.get()->Cast<op::sirius_physical_materialized_collector>();
  duckdb::unique_ptr<duckdb::QueryResult> res = result_collector.get_result();
  return res;
}

void sirius_engine::initialize(duckdb::unique_ptr<op::sirius_physical_operator> plan)
{
  SIRIUS_LOG_DEBUG("Initializing sirius_engine");
  query_handle_->planning();
  reset();
  sirius_owned_plan = std::move(plan);
  initialize_internal(*sirius_owned_plan);
}

void sirius_engine::execute()
{
  nvtx3::scoped_range nvtx_range{"sirius::query"};
  query_handle_->executing();

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    throw invalid_input_exception("Sirius context is not initialized.");
  }

  // Create the query with the pipelines
  sirius_ctx->create_query(std::move(new_scheduled),
                           telemetry::query_telemetry_info{
                             .query_id  = query_handle_->uuid(),
                             .worker_id = telemetry_context_->worker_id(),
                           });
  auto future = sirius_ctx->get_task_scheduler().start_query();
  try {
    future.get();
    sirius_ctx->get_task_scheduler().wait_for_completion();
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("Error executing query: {}", e.what());
    // Drain all in-flight GPU tasks before returning.  QueryEnd() will call
    // clear_all_repositories() immediately after execute() throws; without
    // this drain, tasks still running in the thread pool hold raw pointers to
    // those repositories and cause a use-after-free / heap corruption.
    sirius_ctx->get_task_scheduler().drain_after_error();
    throw;
  } catch (...) {
    SIRIUS_LOG_ERROR("Unknown error executing query");
    sirius_ctx->get_task_scheduler().drain_after_error();
    throw;
  }

  // All tasks completed — operators and pipelines are still alive here.
  // Warn about any intermediate operators that were never finalized.
  if (auto query = sirius_ctx->get_query()) {
    for (const auto& pipeline : query->get_pipelines()) {
      for (const auto& op_ref : pipeline->get_operators()) {
        const auto& op = op_ref.get();
        if (!op.finalized.load()) {
          SIRIUS_LOG_WARN("[execute] operator '{}' (id={}) was not finalized",
                          op.get_name(),
                          op.get_operator_id());
        }
      }
    }
  }
}

void sirius_engine::initialize_internal(op::sirius_physical_operator& plan)
{
  auto sirius_ctx_ptr = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx_ptr) {
    throw invalid_input_exception(
      "Sirius context is not initialized. Check that SIRIUS_DISABLE is not set "
      "and review extension loading logs for errors.");
  }
  const sirius::operator_params& op_params = sirius_ctx_ptr->get_config().get_operator_params();

  sirius_physical_plan = &plan;

  // Create plan-time build context (decoupled from engine)
  const pipeline::pipeline_build_context build_ctx{
    sirius_ctx_ptr->get_telemetry_context(),
    duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(context),
    static_cast<int>(sirius_ctx_ptr->get_config().get_hw_topology().gpus.size())};

  // The RESULT_COLLECTOR wrap is added after the plan generator's own `set_parent_ops` ran;
  // re-walk so the wrapped child's `_parent_op` points at RESULT_COLLECTOR (the tree-parent
  // wiring needs it to route the final sink pipeline's output there). Flag-gated: the legacy
  // path relies on ctor-set `_parent_op` values this re-walk would overwrite.
  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    sirius::planner::sirius_physical_plan_generator::set_parent_ops(*sirius_physical_plan,
                                                                    /*parent=*/nullptr);
  }

  // Build meta-pipeline tree from operator plan
  pipeline::sirius_pipeline_build_state state;
  auto root_pipeline =
    duckdb::make_shared_ptr<pipeline::sirius_meta_pipeline>(build_ctx, state, nullptr);
  root_pipeline->build(*sirius_physical_plan);
  root_pipeline->ready();
  root_pipeline->get_pipelines(sirius_root_pipelines, false);
  root_pipeline_idx = 0;

  // Convert meta-pipelines into execution-ready pipelines
  pipeline::sirius_pipeline_converter converter(build_ctx, op_params);
  auto result = converter.convert(*root_pipeline);

  // Materialize plan-time wiring descriptors into runtime repositories and ports.
  pipeline::materialize_repository_wiring(result.repository_wirings,
                                          sirius_ctx_ptr->get_data_repository_manager());

  new_scheduled   = std::move(result.scheduled_pipelines);
  total_pipelines = result.meta_pipeline_count;

  // NOTE: dead code preserved for operator ID numbering stability
  auto invalid_op = make_uniq<op::sirius_physical_operator>(
    op::SiriusPhysicalOperatorType::INVALID, duckdb::vector<sirius::logical_type>{}, 0);

  // Collect all pipelines for progress tracking
  root_pipeline->get_pipelines(sirius_pipelines, true);
  SIRIUS_LOG_DEBUG("total_pipelines = {}", sirius_pipelines.size());

  // Auto-log the enriched query plan
  pipeline::sirius_plan_printer plan_printer(new_scheduled);
  SIRIUS_LOG_INFO("Query Plan:\n{}", plan_printer.render());
}

}  // namespace sirius
