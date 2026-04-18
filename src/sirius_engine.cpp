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
#include "duckdb/parallel/thread_context.hpp"
#include "io/datasource_factory.hpp"
#include "io/uring/uring_ioctx.hpp"
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
#include "pipeline/sirius_pipeline_converter.hpp"
#include "sirius/exception.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <nvtx3/nvtx3.hpp>

#include <cucascade/data/data_repository_manager.hpp>

#include <stdexcept>

namespace sirius {

sirius_engine::sirius_engine(duckdb::ClientContext& context, sirius_interface& sirius_iface)
  : context(context),
    sirius_iface(sirius_iface),
    datasource_registry_(std::make_shared<io::datasource_registry>())
{
  // Register the default local-file backend (io_uring + O_DIRECT).
  // Reactor tuning lives in PR13 (sirius_config::uring_config).
  datasource_registry_->register_ioctx("file", std::make_shared<io::uring_ioctx>());
}

sirius_engine::~sirius_engine()
{
  // Explicit shutdown before dropping the registry so any in-flight
  // rings/reactors get a chance to quiesce while the ioctx is still alive.
  if (datasource_registry_) {
    for (auto const& scheme : datasource_registry_->schemes()) {
      if (auto ioctx = datasource_registry_->lookup(scheme)) { ioctx->shutdown(); }
    }
    datasource_registry_->clear();
  }
}

io::datasource_registry& sirius_engine::datasource_registry() noexcept
{
  return *datasource_registry_;
}

sirius_config const& sirius_engine::config() const
{
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    throw std::runtime_error(
      "sirius_engine::config(): SiriusContext is not registered on this ClientContext. "
      "This accessor may only be used during query execution.");
  }
  return sirius_ctx->get_config();
}

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

void sirius_engine::insert_repository(
  std::string_view port_id,
  duckdb::shared_ptr<pipeline::sirius_pipeline> input_pipeline,
  duckdb::shared_ptr<pipeline::sirius_pipeline> dependent_pipeline,
  op::MemoryBarrierType barrier_type)
{
  auto next_op            = dependent_pipeline->get_operators().size() == 0
                              ? dependent_pipeline->get_sink().get()
                              : &dependent_pipeline->get_operators()[0].get();
  size_t op_id            = next_op->operator_id;
  auto& data_repo_manager = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                              ->get_data_repository_manager();
  data_repo_manager.add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<op::sirius_physical_operator::port>(
                      barrier_type,
                      data_repo_manager.get_repository(op_id, port_id).get(),
                      input_pipeline,
                      dependent_pipeline));
  input_pipeline->get_sink()->add_next_port_after_sink({next_op, port_id});

  if (next_op->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto partition_op = next_op->Cast<op::sirius_physical_right_delim_join>().partition_join;
    partition_op->add_port(port_id,
                           std::make_unique<op::sirius_physical_operator::port>(
                             op::MemoryBarrierType::FULL,
                             data_repo_manager.get_repository(op_id, port_id).get(),
                             input_pipeline,
                             dependent_pipeline));
  } else if (next_op->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
    throw std::runtime_error("Left delim join should never be a source");
  }
}

void sirius_engine::insert_repository(
  std::string_view port_id,
  op::sirius_physical_operator* cur_op,
  duckdb::shared_ptr<pipeline::sirius_pipeline> input_pipeline,
  duckdb::shared_ptr<pipeline::sirius_pipeline> dependent_pipeline,
  op::MemoryBarrierType barrier_type)
{
  auto& data_repo_manager = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")
                              ->get_data_repository_manager();
  auto next_op = dependent_pipeline->get_operators().size() == 0
                   ? dependent_pipeline->get_sink().get()
                   : &dependent_pipeline->get_operators()[0].get();
  size_t op_id = next_op->operator_id;
  data_repo_manager.add_new_repository(
    op_id, port_id, std::make_unique<::cucascade::shared_data_repository>());
  next_op->add_port(port_id,
                    std::make_unique<op::sirius_physical_operator::port>(
                      barrier_type,
                      data_repo_manager.get_repository(op_id, port_id).get(),
                      input_pipeline,
                      dependent_pipeline));
  cur_op->add_next_port_after_sink({next_op, port_id});

  if (next_op->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto partition_op = next_op->Cast<op::sirius_physical_right_delim_join>().partition_join;
    partition_op->add_port(port_id,
                           std::make_unique<op::sirius_physical_operator::port>(
                             op::MemoryBarrierType::FULL,
                             data_repo_manager.get_repository(op_id, port_id).get(),
                             input_pipeline,
                             dependent_pipeline));
  } else if (next_op->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
    throw std::runtime_error("Left delim join should never be a source");
  }
}

void sirius_engine::cancel_tasks()
{
  sirius_pipelines.clear();
  sirius_root_pipelines.clear();
}

duckdb::shared_ptr<pipeline::sirius_pipeline> sirius_engine::create_child_pipeline(
  pipeline::sirius_pipeline& current, op::sirius_physical_operator& op)
{
  D_ASSERT(!current.operators.empty());
  D_ASSERT(op.is_source());
  // found another operator that is a source, schedule a child pipeline
  // 'op' is the source, and the sink is the same
  auto child_pipeline    = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(*this);
  child_pipeline->sink   = current.get_sink();
  child_pipeline->source = &op;

  // the child pipeline has the same operators up until 'op'
  for (auto current_op : current.get_operators()) {
    if (&current_op.get() == &op) { break; }
    child_pipeline->operators.push_back(current_op);
  }

  return child_pipeline;
}

bool sirius_engine::has_result_collector()
{
  return sirius_physical_plan->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR;
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_engine::get_result()
{
  D_ASSERT(has_result_collector());
  if (!sirius_physical_plan) throw invalid_input_exception("sirius_physical_plan is NULL");
  if (sirius_physical_plan.get() == NULL)
    throw invalid_input_exception("sirius_physical_plan is NULL");
  auto& result_collector =
    sirius_physical_plan.get()->Cast<op::sirius_physical_materialized_collector>();
  duckdb::unique_ptr<duckdb::QueryResult> res = result_collector.get_result();
  return res;
}

void sirius_engine::initialize(duckdb::unique_ptr<op::sirius_physical_operator> plan)
{
  SIRIUS_LOG_DEBUG("Initializing sirius_engine");
  reset();
  sirius_owned_plan = std::move(plan);
  // Pre-fetch iceberg delete-file metadata before initialize_internal() assigns
  // operator IDs to pipeline-breaker operators (PARTITION, CONCAT, etc.).
  // The DuckDB metadata connection is opened under InternalQueryGuard so that
  // QueryBegin/QueryEnd side-effects on the shared SiriusContext are suppressed.
  prefetch_iceberg_metadata(*sirius_owned_plan);
  initialize_internal(*sirius_owned_plan);
}

void sirius_engine::execute()
{
  nvtx3::scoped_range nvtx_range{"sirius::query"};

  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (sirius_ctx == nullptr) {
    throw invalid_input_exception("Sirius context is not initialized.");
  }

  // Create the query with the pipelines
  sirius_ctx->create_query(std::move(new_scheduled));
  auto future = sirius_ctx->get_pipeline_executor().start_query();
  try {
    future.get();
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("Error executing query: {}", e.what());
    // Drain all in-flight GPU tasks before returning.  QueryEnd() will call
    // clear_all_repositories() immediately after execute() throws; without
    // this drain, tasks still running in the thread pool hold raw pointers to
    // those repositories and cause a use-after-free / heap corruption.
    sirius_ctx->get_pipeline_executor().drain_after_error();
    throw;
  } catch (...) {
    SIRIUS_LOG_ERROR("Unknown error executing query");
    sirius_ctx->get_pipeline_executor().drain_after_error();
    throw;
  }
}

duckdb::unique_ptr<op::sirius_physical_operator> sirius_engine::construct_sirius_specific_operator(
  op::sirius_physical_operator* op)
{
  if (op->type == op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    auto& scan_physical_op = op->Cast<op::sirius_physical_table_scan>();
    if (scan_physical_op.function.name == "parquet_scan" ||
        scan_physical_op.function.name == "read_parquet") {
      return duckdb::make_uniq<op::sirius_physical_parquet_scan>(&scan_physical_op);
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

duckdb::unique_ptr<op::sirius_physical_operator> sirius_engine::construct_iceberg_scan_operator(
  op::sirius_physical_table_scan& scan_op)
{
  auto iceberg_scan = duckdb::make_uniq<op::sirius_physical_iceberg_scan>(&scan_op);

  if (!scan_op.parameters.empty()) {
    std::string const table_path = scan_op.parameters[0].ToString();
    auto it                      = iceberg_metadata_cache_.find(table_path);
    if (it != iceberg_metadata_cache_.end()) {
      iceberg_scan->positional_delete_files = it->second.positional_delete_files;
      iceberg_scan->equality_delete_files   = it->second.equality_delete_files;
    }
  }

  return iceberg_scan;
}

void sirius_engine::prefetch_iceberg_metadata(op::sirius_physical_operator& plan)
{
  // Walk the plan tree and fetch delete-file metadata for every iceberg scan.
  // This runs in initialize() BEFORE initialize_internal() so that operator IDs
  // for pipeline-breaker nodes (PARTITION, CONCAT, …) haven't been assigned yet.

  if (plan.type != op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    if (plan.type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
      auto& collector = plan.Cast<op::sirius_physical_result_collector>();
      prefetch_iceberg_metadata(collector.plan);
    } else {
      for (auto& child : plan.children) {
        prefetch_iceberg_metadata(*child);
      }
    }
    return;
  }

  auto& scan_op = plan.Cast<op::sirius_physical_table_scan>();
  if (scan_op.function.name != "iceberg_scan" || scan_op.parameters.empty()) { return; }

  std::string const table_path = scan_op.parameters[0].ToString();
  if (iceberg_metadata_cache_.count(table_path)) { return; }  // already fetched

  // Opening a secondary connection triggers QueryBegin/QueryEnd on the shared
  // SiriusContext.  InternalQueryGuard suppresses those side-effects.
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    SIRIUS_LOG_WARN("[sirius_engine] SiriusContext not available; treating iceberg '{}' as V1.",
                    table_path);
    iceberg_metadata_cache_.emplace(table_path, op::scan::IcebergDeleteFiles{});
    return;
  }

  duckdb::SiriusContext::InternalQueryGuard guard(*sirius_ctx);
  auto files = op::scan::read_iceberg_delete_metadata(context, table_path);
  iceberg_metadata_cache_.emplace(table_path, std::move(files));
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

  // Build meta-pipeline tree from operator plan
  pipeline::sirius_pipeline_build_state state;
  auto root_pipeline =
    duckdb::make_shared_ptr<pipeline::sirius_meta_pipeline>(*this, state, nullptr);
  root_pipeline->build(*sirius_physical_plan);
  root_pipeline->ready();
  root_pipeline->get_pipelines(sirius_root_pipelines, false);
  root_pipeline_idx = 0;

  // Convert meta-pipelines into execution-ready pipelines
  pipeline::sirius_pipeline_converter converter(*this, op_params);
  auto result = converter.convert(*root_pipeline);

  new_scheduled         = std::move(result.scheduled_pipelines);
  new_pipeline_breakers = std::move(result.pipeline_breakers);
  total_pipelines       = result.meta_pipeline_count;

  // NOTE: dead code preserved for operator ID numbering stability
  auto invalid_op = make_uniq<op::sirius_physical_operator>(
    op::SiriusPhysicalOperatorType::INVALID, duckdb::vector<duckdb::LogicalType>{}, 0);

  // Collect all pipelines for progress tracking
  root_pipeline->get_pipelines(sirius_pipelines, true);
  SIRIUS_LOG_DEBUG("total_pipelines = {}", sirius_pipelines.size());
}

}  // namespace sirius
