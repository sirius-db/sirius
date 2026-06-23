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

#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "io/sirius_datasource.hpp"
#include "log/logging.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_iceberg_scan.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "pipeline/sirius_plan_printer.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius/exception.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"
#include "sirius_interface.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_repository_manager.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius {

namespace {

const telemetry::telemetry_context& get_telemetry_context_from_client_context(
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
    query_group_uuid_(uuid::now_v7()),
    query_group_observer_(quent::query_group::create_observer(telemetry_context_.context())),
    query_handle_(
      quent::query::create(telemetry_context_.context(),
                           quent::query::Init{
                             .instance_name  = sirius_iface.query_label.value_or("unnamed_query"),
                             .query_group_id = query_group_uuid_,
                           }))
{
  // Declare the query group under this engine
  query_group_observer_->declaration(query_group_uuid_,
                                     quent::query_group::Declaration{
                                       .instance_name = "default_group",
                                       .engine_id     = telemetry_context_.engine_id(),
                                     });
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
  new_pipeline_breakers.clear();
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
  // Pre-fetch and fully materialize iceberg delete data before initialize_internal()
  // assigns operator IDs to pipeline-breaker operators (PARTITION, CONCAT, etc.).
  // All DuckDB connections are opened under InternalQueryGuard so that
  // QueryBegin/QueryEnd side-effects on the shared SiriusContext are suppressed.
  prefetch_iceberg_delete_data(*sirius_owned_plan);
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
                             .worker_id = telemetry_context_.worker_id(),
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

duckdb::unique_ptr<op::sirius_physical_operator> sirius_engine::construct_sirius_specific_operator(
  op::sirius_physical_operator* op)
{
  if (op->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& scan_physical_op = op->Cast<op::sirius_physical_table_scan>();
    if (scan_physical_op.function.name == "parquet_scan" ||
        scan_physical_op.function.name == "read_parquet") {
      // Gather the configured GPU device ids so the parquet-scan op can build
      // one translated filter tree per GPU (scalars end up on the right device
      // for each task's dispatch target). Falls back to the current device if
      // SiriusContext is not yet registered.
      std::vector<int> gpu_device_ids;
      auto ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      if (ctx != nullptr) {
        for (auto const* space :
             ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
          gpu_device_ids.push_back(space->get_device_id());
        }
      }
      return duckdb::make_uniq<op::sirius_physical_parquet_scan>(&scan_physical_op,
                                                                 std::move(gpu_device_ids));
    } else if (scan_physical_op.function.name == "iceberg_scan") {
      return construct_iceberg_scan_operator(scan_physical_op);
    } else if (scan_physical_op.function.name == "seq_scan") {
      return duckdb::make_uniq<op::sirius_physical_duckdb_scan>(&scan_physical_op);
    } else {
      throw std::runtime_error("Unsupported scan function: " + scan_physical_op.function.name);
    }
  } else if (op->type == op::SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    auto& group_by_physical_op = op->Cast<op::sirius_physical_grouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_grouped_aggregate_merge>(&group_by_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::ORDER_BY) {
    auto& order_by_physical_op = op->Cast<op::sirius_physical_order>();
    return duckdb::make_uniq<op::sirius_physical_merge_sort>(&order_by_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::TOP_N) {
    auto& topn_physical_op = op->Cast<op::sirius_physical_top_n>();
    return duckdb::make_uniq<op::sirius_physical_top_n_merge>(&topn_physical_op);
  } else if (op->type == op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE) {
    auto& ungrouped_agg_physical_op = op->Cast<op::sirius_physical_ungrouped_aggregate>();
    return duckdb::make_uniq<op::sirius_physical_ungrouped_aggregate_merge>(
      &ungrouped_agg_physical_op);
  } else {
    throw internal_exception("Unsupported operator type" +
                             SiriusPhysicalOperatorToString(op->type) +
                             " for constructing sirius specific operator.");
  }
}

/// Resolve the Iceberg table path from the scan operator.
/// For file-based scans: parameters[0] contains the path.
/// For REST catalog scans: parameters is empty; derive path from bind_data's
/// first data file (strip /data/filename.parquet to get the table root).
static std::string resolve_iceberg_table_path(op::sirius_physical_table_scan& scan_op)
{
  if (!scan_op.parameters.empty()) { return scan_op.parameters[0].ToString(); }

  // REST catalog: derive from bind_data file list.
  if (scan_op.bind_data) {
    auto& bind_data = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
    if (bind_data.file_list && !bind_data.file_list->IsEmpty()) {
      auto files = bind_data.file_list->GetAllFiles();
      if (!files.empty()) {
        auto const& first_path = files[0].path;
        // Strip "/data/<filename>" to get table root.
        auto data_pos = first_path.rfind("/data/");
        if (data_pos != std::string::npos) { return first_path.substr(0, data_pos); }
      }
    }
  }
  return {};
}

duckdb::unique_ptr<op::sirius_physical_operator> sirius_engine::construct_iceberg_scan_operator(
  op::sirius_physical_table_scan& scan_op)
{
  auto iceberg_scan = duckdb::make_uniq<op::sirius_physical_iceberg_scan>(&scan_op);

  auto table_path = resolve_iceberg_table_path(scan_op);
  if (!table_path.empty()) {
    auto it = iceberg_delete_data_cache_.find(table_path);
    if (it != iceberg_delete_data_cache_.end()) { iceberg_scan->delete_data = it->second; }
  }

  return iceberg_scan;
}

void sirius_engine::prefetch_iceberg_delete_data(op::sirius_physical_operator& plan)
{
  // Walk the plan tree and fully materialize delete data for every iceberg scan.
  // This runs in initialize() BEFORE initialize_internal() so that operator IDs
  // for pipeline-breaker nodes (PARTITION, CONCAT, …) haven't been assigned yet.
  // All DuckDB connections (for positional-delete reads, snapshot queries) are
  // opened under a single InternalQueryGuard to prevent transaction side-effects.

  if (plan.type != op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    if (plan.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
      auto& collector = plan.Cast<op::sirius_physical_result_collector>();
      prefetch_iceberg_delete_data(collector.plan);
    } else {
      for (auto& child : plan.children) {
        prefetch_iceberg_delete_data(*child);
      }
    }
    return;
  }

  auto& scan_op = plan.Cast<op::sirius_physical_table_scan>();
  if (scan_op.function.name != "iceberg_scan") { return; }

  std::string const table_path = resolve_iceberg_table_path(scan_op);
  if (table_path.empty()) { return; }
  if (iceberg_delete_data_cache_.count(table_path)) { return; }  // already fetched

  // Iceberg delete-data reading is disabled in this build: the metadata reader
  // (src/op/scan/iceberg_metadata_reader.cpp) is stale against the new io
  // interface and is excluded from the build. Treat every iceberg table as
  // having no delete files (V1 semantics) so positional/equality deletes are
  // simply not applied rather than producing an undefined-symbol link error.
  SIRIUS_LOG_WARN(
    "[sirius_engine] iceberg delete-data reading is disabled; treating '{}' as having no deletes.",
    table_path);
  iceberg_delete_data_cache_.emplace(table_path, std::make_shared<op::scan::IcebergDeleteData>());
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
  pipeline::pipeline_build_context build_ctx;
  build_ctx.preserve_insertion_order =
    duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(context);
  build_ctx.num_gpus = static_cast<int>(sirius_ctx_ptr->get_config().get_hw_topology().gpus.size());

  // Build meta-pipeline tree from operator plan
  pipeline::sirius_pipeline_build_state state;
  auto root_pipeline =
    duckdb::make_shared_ptr<pipeline::sirius_meta_pipeline>(build_ctx, state, nullptr);
  root_pipeline->build(*sirius_physical_plan);
  root_pipeline->ready();
  root_pipeline->get_pipelines(sirius_root_pipelines, false);
  root_pipeline_idx = 0;

  // Convert meta-pipelines into execution-ready pipelines
  pipeline::sirius_pipeline_converter converter(
    build_ctx, op_params, &iceberg_delete_data_cache_, &context);
  auto result = converter.convert(*root_pipeline);

  // Materialize plan-time wiring descriptors into runtime repositories and ports.
  pipeline::materialize_repository_wiring(result.repository_wirings,
                                          sirius_ctx_ptr->get_data_repository_manager());

  new_scheduled         = std::move(result.scheduled_pipelines);
  new_pipeline_breakers = std::move(result.inserted_operators);
  total_pipelines       = result.meta_pipeline_count;

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
