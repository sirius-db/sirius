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

#include "op/scan/duckdb_scan_task.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/pipeline_executor.hpp"

#include <duckdb/parallel/thread_context.hpp>

#include <iterator>
#include <queue>

namespace sirius::creator {

//------------------------------------------------------------------------------
// task_creator
//------------------------------------------------------------------------------

task_creator::task_creator(size_t num_threads,
                           sirius::memory::sirius_memory_reservation_manager& mem_res_mgr)
  : _num_threads(num_threads), _running(false), _mem_res_mgr(mem_res_mgr)
{
  
}

task_creator::~task_creator() { stop_thread_pool(); }

void task_creator::set_client_context(::duckdb::ClientContext& client_context)
{
  _client_context = std::addressof(client_context);
}

void task_creator::set_pipeline_hashmap(sirius_pipeline_hashmap& sirius_pipeline_map)
{
  _sirius_pipeline_map = &sirius_pipeline_map;
  for (const auto& i : _sirius_pipeline_map->_vec) {
    if (i->get_source()->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      _priority_scans.push(i);
    }
  }
}

void task_creator::set_pipeline_executor(sirius::pipeline::pipeline_executor& pipeline_executor)
{
  _pipeline_executor = &pipeline_executor;
}

void task_creator::reset()
{
  _priority_scans = std::queue<duckdb::shared_ptr<pipeline::sirius_pipeline>>{};
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(op::sirius_physical_operator* node) {
  auto hint = node->get_next_task_hint();
  
  if (hint.has_value() && hint.value().hint == op::TaskCreationHint::READY) {
    // WSM TODO: how do we handle other ports that are not default?
    return hint.value().producer;
  } else if (hint.has_value() && hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    return get_operator_for_next_task(hint.value().producer);
  } else {
    if (!_priority_scans.empty()) {
      duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline = _priority_scans.front();
      auto* scan_node = pipeline->get_source().get();
      // TODO: amin or WSM. Need to implement get next task hint for scan node. 
      // It should return ready if there are more scans tasks to be created.
      auto scan_hint = scan_node->get_next_task_hint();
      if (scan_hint.has_value() && scan_hint.value().hint == op::TaskCreationHint::READY) {
        return scan_node;
      } else {
        // WSM TODO: this probably needs a lock guard or task creator needs to be single threaded.
        _priority_scans.pop();
        return nullptr;
      }
    }
  }
  return nullptr;
}

void task_creator::start()
{
  // start_thread_pool();
  // while (!_priority_scans.empty()) {
  //   duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline = _priority_scans.front();
  //   auto* scan_node                                        = pipeline->get_source().get();
  //   schedule(std::make_unique<task_creation_info>(scan_node, pipeline));
  //   _priority_scans.pop();
  // }
}

void task_creator::stop() { 
  _task_creation_queue.interrupt();
  stop_thread_pool(); 
}

void task_creator::start_thread_pool()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _threads.reserve(_num_threads);
  for (size_t i = 0; i < _num_threads; ++i) {
    _threads.emplace_back(&task_creator::worker_function, this, i);
  }
}

void task_creator::stop_thread_pool()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  on_stop();
  for (auto& thread : _threads) {
    if (thread.joinable()) { thread.join(); }
  }
  _threads.clear();
}

void task_creator::schedule(op::sirius_physical_operator* node)
{
  _task_creation_queue.push(std::make_unique<task_creation_request>(node));
}

void task_creator::worker_function(int worker_id)
{
  while (_running.load()) {
    // WSM TODO: is this queue blocking? 
    auto request = _task_creation_queue.pop();
    auto node = request->node;

    // WSM TODO: is this correct?
    if (node == nullptr) {
      // Task queue is closed.
      break;
    }


    node = get_operator_for_next_task(node);
    if (node == nullptr) {
      continue;
    }
    try {
      // Get what we need to create the task
      auto pipeline = node->get_pipeline();
      std::vector<cucascade::shared_data_repository*> destination_data_repositories;
      auto next_port_after_sink = pipeline->get_sink()->get_next_port_after_sink();
      for (auto& [next_op, port_id] : next_port_after_sink) {
        destination_data_repositories.push_back(next_op->get_port(port_id)->repo);
      }

      // scheduling scan task
      if (node->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
        auto scan_task_global_state = std::make_shared<op::scan::duckdb_scan_task_global_state>(
          pipeline,
          *_pipeline_executor,
          *_client_context,
          &node->Cast<op::sirius_physical_duckdb_scan>());
        duckdb::ThreadContext thread_ctx(*_client_context);
        duckdb::ExecutionContext exec_ctx(*_client_context, thread_ctx, nullptr);
        auto scan_task_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
          *scan_task_global_state, exec_ctx);
        if (destination_data_repositories.empty()) {
          throw std::runtime_error(
            "No destination data repositories provided for scan task creation.");
        }
        auto scan_task =
          std::make_unique<op::scan::duckdb_scan_task>(get_next_task_id(),
                                                       destination_data_repositories[0],  // WSM TODO: is this correct? there probably needs to be multiple possible destination data repositories
                                                       std::move(scan_task_local_state),
                                                       std::move(scan_task_global_state));
        
                                                       // WSM todo we should be scheduling directly pipeline_executor, which in turn will schedule with the scan executor
                                                       _pipeline_executor->schedule(std::move(scan_task));
        // scheduling pipeline task
      } else {
        // need to exhaust input batches until all ports are empty
        while (!node->all_ports_empty()) {
          auto input_batch = node->get_input_batch(); // WSM TODO: rename this to get_next_task_input_batch
          auto global_state =
            std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline);
          auto local_state =
            std::make_unique<pipeline::gpu_pipeline_task_local_state>(input_batch);
          auto task =
            std::make_unique<pipeline::gpu_pipeline_task>(get_next_task_id(),
                                                          destination_data_repositories,
                                                          std::move(local_state),
                                                          std::move(global_state));
          _pipeline_executor->schedule(std::move(task));
        }
      }

    } catch (const std::exception& e) {
      stop();
    }
  }
}

void task_creator::on_stop() { _task_creation_queue.interrupt(); }

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius::creator
