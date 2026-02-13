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
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_executor.hpp"

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
  _gpu_operator_global_state_map.clear();
  _thread_context.reset();
  _execution_context.reset();
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node)
{
  if (node == nullptr) { return nullptr; }
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
    return get_operator_for_next_task(hint.value().producer);
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

    // Schedule the task creation work on the thread pool
    _thread_pool->schedule(
      [this, request = std::move(request), ticket = std::move(ticket)]() mutable {
        try {
          auto node = request->node;
          if (node == nullptr) { return; }

          node = get_operator_for_next_task(node);
          if (node == nullptr) { return; }

          // Get what we need to create the task
          auto pipeline = node->get_pipeline();
          std::vector<cucascade::shared_data_repository*> destination_data_repositories;
          auto next_port_after_sink = pipeline->get_sink()->get_next_port_after_sink();
          for (auto& [next_op, port_id] : next_port_after_sink) {
            destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
          }

          // scheduling scan task
          if (node->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
            // Check to see if you need to create a new global state for this scan operator
            size_t operator_id = node->get_operator_id();
            {
              std::lock_guard<std::mutex> lock(_global_state_mutex);
              auto it = _scan_operator_global_state_map.find(operator_id);
              if (it == _scan_operator_global_state_map.end()) {
                // If not found, create new global state and store it in the map
                auto scan_task_global_state =
                  std::make_shared<op::scan::duckdb_scan_task_global_state>(
                    pipeline,
                    *_pipeline_executor,
                    *_client_context,
                    &node->Cast<op::sirius_physical_duckdb_scan>());
                _scan_operator_global_state_map[operator_id] = scan_task_global_state;
              }
            }

            auto scan_task_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
              *_scan_operator_global_state_map[operator_id], *_execution_context);
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
              _scan_operator_global_state_map[operator_id]);
            pipeline->mark_task_created();  // WSM TODO: this needs to be done atomically
                                            // with the task creation
            _pipeline_executor->schedule(std::move(scan_task));
            // scheduling pipeline task
          } else {
            // need to exhaust input batches until all ports are empty
            while (!node->all_ports_empty()) {
              auto input_data = node->get_next_task_input_data();
              if (!input_data) { break; }
              pipeline->mark_task_created();  // WSM TODO: this needs to be done atomically with the
                                              // task creation

              // Check to see if you need to create a new global state for this operator
              size_t operator_id = node->get_operator_id();
              {
                std::lock_guard<std::mutex> lock(_global_state_mutex);
                auto it = _gpu_operator_global_state_map.find(operator_id);
                if (it == _gpu_operator_global_state_map.end()) {
                  // If not found, create new global state and store it in the map
                  auto gpu_pipeline_task_global_state =
                    std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline);
                  _gpu_operator_global_state_map[operator_id] = gpu_pipeline_task_global_state;
                }
              }

              auto local_state =
                std::make_unique<pipeline::gpu_pipeline_task_local_state>(std::move(input_data));
              auto task = std::make_unique<pipeline::gpu_pipeline_task>(
                get_next_task_id(),
                destination_data_repositories,
                std::move(local_state),
                _gpu_operator_global_state_map[operator_id]);
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
