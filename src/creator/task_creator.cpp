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

#include "creator/task_creator.hpp"

#include "log/logging.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "planner/query.hpp"
#include "sirius_context.hpp"

#include <duckdb/execution/execution_context.hpp>
#include <duckdb/parallel/thread_context.hpp>

#include <iterator>
#include <optional>
#include <queue>

namespace sirius::creator {

//------------------------------------------------------------------------------
// task_creator
//------------------------------------------------------------------------------

task_creator::task_creator(exec::thread_pool_config config,
                           sirius::memory::sirius_memory_reservation_manager& mem_res_mgr)
  : _running(false), _config(config), _mem_res_mgr(mem_res_mgr), _kiosk(config.num_threads)
{
}

task_creator::~task_creator() { stop(); }

void task_creator::set_client_context(::duckdb::ClientContext& client_context)
{
  _client_context = std::addressof(client_context);
  _thread_context = std::make_unique<duckdb::ThreadContext>(client_context);
  _execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_context, *_thread_context, nullptr);
}

void task_creator::set_pipeline_executor(sirius::pipeline::pipeline_executor& pipeline_executor)
{
  _pipeline_executor = &pipeline_executor;
}

void task_creator::prepare_for_query(const sirius::planner::query& query)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);

  // Clear the global state maps for the new query
  _scan_operator_global_state_map.clear();
  _parquet_scan_operator_global_state_map.clear();
  _gpu_operator_global_state_map.clear();

  // Iterate through all pipelines in the query and create global states
  const auto& pipelines = query.get_pipelines();
  for (const auto& pipeline : pipelines) {
    // Get the sink operator of the pipeline
    auto source_operator = pipeline->get_source();
    if (source_operator == nullptr) { continue; }

    size_t operator_id = source_operator->get_operator_id();

    if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      _scan_operator_global_state_map.emplace(
        operator_id,
        std::make_shared<op::scan::duckdb_scan_task_global_state>(
          pipeline,
          *_pipeline_executor,
          *_client_context,
          &source_operator->Cast<op::sirius_physical_duckdb_scan>()));
    } else if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::PARQUET_SCAN) {
      const auto& op_params =
        _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state")
          ->get_config()
          .get_operator_params();
      _parquet_scan_operator_global_state_map.emplace(
        operator_id,
        std::make_shared<op::scan::parquet_scan_task_global_state>(
          pipeline,
          &source_operator->Cast<op::sirius_physical_parquet_scan>(),
          op_params.scan_task_batch_size));
    } else {
      _gpu_operator_global_state_map.emplace(
        operator_id, std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline));
    }
  }
}

void task_creator::drain_pending_tasks()
{
  // Drain any queued task creation requests that haven't been picked up yet
  _task_creation_queue.drain();
  // Wait for any in-flight task creation lambdas to finish
  _kiosk.wait_all();
}

void task_creator::reset()
{
  // Clear the scan operator global state map for the new query
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _scan_operator_global_state_map.clear();
  _parquet_scan_operator_global_state_map.clear();
  _gpu_operator_global_state_map.clear();
  _thread_context.reset();
  _execution_context.reset();
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node)
{
  if (node == nullptr) { return nullptr; }

  if (node->type == ::sirius::op::SiriusPhysicalOperatorType::PARQUET_SCAN) {
    size_t operator_id             = node->get_operator_id();
    auto parquet_task_global_state = _parquet_scan_operator_global_state_map.at(operator_id);
    if (parquet_task_global_state->has_more_partitions()) {
      return node;
    } else {
      return nullptr;
    }
  }
  auto hint = node->get_next_task_hint();

  if (hint.has_value() && hint.value().hint == op::TaskCreationHint::READY) {
    if (hint.value().producer == nullptr) {
      throw std::runtime_error(
        "During get_operator_for_next_task Producer is nullptr for operator " + node->get_name());
    }
    // WSM TODO: how do we handle other ports that are not default?
    return hint.value().producer;
  } else if (hint.has_value() &&
             hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    auto* producer = hint.value().producer;
    // DuckDB scan tasks create their own continuations internally, so the
    // task creator should never schedule additional scans from downstream.
    // (Parquet scans are fine — they use partition indices that self-limit.)
    if (producer != nullptr && producer->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      return nullptr;
    }
    return get_operator_for_next_task(producer);
  }
  return nullptr;
}

void task_creator::stop()
{
  _task_creation_queue.interrupt();
  stop_thread_pool();
}

void task_creator::start_thread_pool()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _thread_pool = std::make_unique<exec::thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
  _manager_thread = std::thread(&task_creator::manager_loop, this);
}

void task_creator::stop_thread_pool()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _kiosk.stop();
  _task_creation_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _kiosk.wait_all();
  if (_thread_pool) { _thread_pool->stop(); }
}

void task_creator::schedule(op::sirius_physical_operator* node)
{
  auto request  = std::make_unique<task_creation_request>();
  request->node = node;
  _task_creation_queue.push(std::move(request));
}

void task_creator::manager_loop()
{
  while (_running.load()) {
    auto ticket = _kiosk.acquire();  // block till a thread is available
    if (!ticket.is_valid()) {
      SIRIUS_LOG_INFO("Task Creator: Kiosk interrupted, stopping manager loop");
      break;
    }

    auto request = _task_creation_queue.pop();
    if (!request) {
      SIRIUS_LOG_INFO("Task Creator: task queue interrupted, stopping manager loop");
      break;
    }

    auto node = request->node;
    if (node == nullptr) { continue; }

    node = get_operator_for_next_task(node);

    if (node == nullptr) { continue; }

    // Schedule the task creation work on the thread pool
    _thread_pool->schedule([this, node, ticket = std::move(ticket)]() mutable {
      try {
        // Get what we need to create the task
        auto pipeline = node->get_pipeline();
        std::vector<cucascade::shared_data_repository*> destination_data_repositories;
        // special handling for delim joins
        if (pipeline->get_sink()->type ==
            ::sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto& delim_join    = pipeline->get_sink()->Cast<op::sirius_physical_right_delim_join>();
          auto partition_join = delim_join.partition_join;
          auto distinct_op    = delim_join.distinct.get();
          for (auto& [next_op, port_id] : partition_join->get_next_port_after_sink()) {
            destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
          }
          for (auto& [next_op, port_id] : distinct_op->get_next_port_after_sink()) {
            destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
          }
        } else if (pipeline->get_sink()->type ==
                   ::sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto& delim_join      = pipeline->get_sink()->Cast<op::sirius_physical_left_delim_join>();
          auto distinct_op      = delim_join.distinct.get();
          auto column_data_scan = delim_join.column_data_scan;
          for (auto& [next_op, port_id] : column_data_scan->get_next_port_after_sink()) {
            destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
          }
          for (auto& [next_op, port_id] : distinct_op->get_next_port_after_sink()) {
            destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
          }
        } else {
          auto next_port_after_sink = pipeline->get_sink()->get_next_port_after_sink();
          for (auto& [next_op, port_id] : next_port_after_sink) {
            destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
          }
        }
        // scheduling scan task
        if (node->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
          // Check to see if you need to create a new global s for this scan operator
          size_t operator_id          = node->get_operator_id();
          auto scan_task_global_state = _scan_operator_global_state_map.at(operator_id);

          const auto& op_params =
            _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state")
              ->get_config()
              .get_operator_params();
          auto scan_task_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
            *scan_task_global_state,
            *_execution_context,
            op_params.scan_task_batch_size,
            op_params.default_scan_task_varchar_size);
          if (destination_data_repositories.empty()) {
            throw std::runtime_error(
              "No destination data repositories provided for scan task creation.");
          }
          auto scan_task = std::make_unique<op::scan::duckdb_scan_task>(
            get_next_task_id(),
            destination_data_repositories[0],  // WSM amin TODO: is this correct? there probably
                                               // needs to be multiple possible destination data
                                               // repositories
            std::move(scan_task_local_state),
            scan_task_global_state);
          pipeline->mark_task_created();  // WSM TODO: this needs to be done atomically
                                          // with the task creation
          _pipeline_executor->schedule(std::move(scan_task));
        } else if (node->type == ::sirius::op::SiriusPhysicalOperatorType::PARQUET_SCAN) {
          size_t operator_id             = node->get_operator_id();
          auto parquet_task_global_state = _parquet_scan_operator_global_state_map.at(operator_id);
          auto* parquet_scan             = &node->Cast<op::sirius_physical_parquet_scan>();
          pipeline->mark_task_created();
          auto const partition_idx = parquet_task_global_state->get_next_rg_partition_idx();
          if (!partition_idx.has_value()) {
            pipeline->mark_task_completed();
            if (pipeline->is_pipeline_finished()) {
              auto output_consumers = pipeline->get_output_consumers();
              for (auto& output_consumer : output_consumers) {
                schedule(output_consumer);
              }
            }
            return;
          }
          if (!parquet_task_global_state->has_more_partitions()) {
            parquet_scan->has_more_partitions = false;
          }

          auto parquet_task_local_state = std::make_unique<op::scan::parquet_scan_task_local_state>(
            *parquet_task_global_state, *partition_idx);

          if (destination_data_repositories.empty()) {
            throw std::runtime_error(
              "No destination data repositories provided for parquet scan task creation.");
          }
          auto parquet_task =
            std::make_unique<op::scan::parquet_scan_task>(get_next_task_id(),
                                                          destination_data_repositories[0],
                                                          std::move(parquet_task_local_state),
                                                          parquet_task_global_state);
          _pipeline_executor->schedule(std::move(parquet_task));
          // scheduling pipeline task
        } else {
          // need to exhaust input batches until all ports are empty
          while (!node->all_ports_empty()) {
            // Mark task created BEFORE popping data from ports to prevent a race
            // condition where update_pipeline_status() sees empty ports and matching
            // task counters, prematurely marking the pipeline as finished.
            pipeline->mark_task_created();

            auto input_data = node->get_next_task_input_data();
            if (!input_data || input_data->get_data_batches().empty()) {
              // No data was available (e.g., another thread already consumed it).
              // Balance the counter. mark_task_completed() calls update_pipeline_status()
              // which is correct: if all ports are truly empty and all real tasks have
              // completed, the pipeline should finish.
              pipeline->mark_task_completed();
              if (pipeline->is_pipeline_finished()) {
                auto output_consumers = pipeline->get_output_consumers();
                for (auto& output_consumer : output_consumers) {
                  this->schedule(output_consumer);
                }
              }
              break;
            }

            // Check to see if you need to create a new global state for this operator
            size_t operator_id                  = node->get_operator_id();
            auto gpu_pipeline_task_global_state = _gpu_operator_global_state_map.at(operator_id);

            auto local_state =
              std::make_unique<pipeline::gpu_pipeline_task_local_state>(std::move(input_data));
            auto task =
              std::make_unique<pipeline::gpu_pipeline_task>(get_next_task_id(),
                                                            destination_data_repositories,
                                                            std::move(local_state),
                                                            gpu_pipeline_task_global_state);
            _pipeline_executor->schedule(std::move(task));
          }
        }
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("Task Creator: Exception during task creation: {}", e.what());
        stop();
      }
    });
  }
}

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius::creator
