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
#include "op/sirius_physical_table_scan.hpp"
#include "pipeline/gpu_pipeline_task.hpp"

#include <duckdb/parallel/thread_context.hpp>

#include <queue>

namespace sirius::creator {

//------------------------------------------------------------------------------
// task_creation_queue
//------------------------------------------------------------------------------

task_creation_queue::task_creation_queue(size_t num_threads) : _num_threads(num_threads) {}

void task_creation_queue::open() { _is_open.store(true, std::memory_order_release); }

void task_creation_queue::close()
{
  _is_open.store(false, std::memory_order_release);
  // Wake up all threads blocked in wait_dequeue by pushing nullptr sentinels
  for (size_t i = 0; i < _num_threads; ++i) {
    _queue.enqueue(nullptr);
  }
}

void task_creation_queue::push(std::unique_ptr<task_creation_info> info)
{
  _queue.enqueue(std::move(info));
}

std::unique_ptr<task_creation_info> task_creation_queue::pull()
{
  std::unique_ptr<task_creation_info> info;
  while (true) {
    if (_queue.try_dequeue(info)) { return info; }

    // If the queue is closed and empty, return nullptr to indicate no more tasks.
    if (!_is_open.load(std::memory_order_acquire)) { return nullptr; }

    // Otherwise, wait for a task to become available.
    _queue.wait_dequeue(info);
    if (info) { return info; }
  }
}

//------------------------------------------------------------------------------
// task_creator
//------------------------------------------------------------------------------

task_creator::task_creator(size_t num_threads,
                           pipeline::pipeline_executor& pipeline_executor,
                           op::scan::duckdb_scan_executor& duckdb_scan_executor,
                           sirius::memory::sirius_memory_reservation_manager& mem_res_mgr)
  : _num_threads(num_threads),
    _running(false),
    _pipeline_executor(pipeline_executor),
    _duckdb_scan_executor(duckdb_scan_executor),
    _mem_res_mgr(mem_res_mgr)
{
  _task_creation_queue = std::make_unique<task_creation_queue>(num_threads);
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
    if (i->get_source()->type == ::duckdb::PhysicalOperatorType::TABLE_SCAN) {
      _priority_scans.push(i);
    }
  }
}

void task_creator::reset()
{
  _priority_scans = std::queue<duckdb::shared_ptr<pipeline::sirius_pipeline>>{};
}

void task_creator::process_next_task(op::sirius_physical_operator* node)
{
  auto hint = node->get_next_task_hint();
  // printf("Node %p provided hint of type %zu\n", node, hint.index());
  if (std::holds_alternative<op::sirius_physical_operator*>(hint)) {
    // printf("Processing next task for hint node %p\n", node);
    auto* hint_node = std::get<op::sirius_physical_operator*>(hint);
    auto pipeline   = hint_node->get_port("default")->dest_pipeline;
    schedule(std::make_unique<task_creation_info>(hint_node, pipeline));
  } else if (std::holds_alternative<duckdb::shared_ptr<pipeline::sirius_pipeline>>(hint)) {
    auto pipeline = std::get<duckdb::shared_ptr<pipeline::sirius_pipeline>>(hint);
    process_next_task(&pipeline->get_inner_operators()[0].get());
  } else {
    if (!_priority_scans.empty()) {
      duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline = _priority_scans.front();
      auto* scan_node                                        = pipeline->get_source().get();
      schedule(std::make_unique<task_creation_info>(scan_node, pipeline));
      _priority_scans.pop();
    }
  }
}

void task_creator::start()
{
  start_thread_pool();
  while (!_priority_scans.empty()) {
    duckdb::shared_ptr<pipeline::sirius_pipeline> pipeline = _priority_scans.front();
    auto* scan_node                                        = pipeline->get_source().get();
    schedule(std::make_unique<task_creation_info>(scan_node, pipeline));
    _priority_scans.pop();
  }
}

void task_creator::stop() { stop_thread_pool(); }

void task_creator::start_thread_pool()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();
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

void task_creator::schedule(std::unique_ptr<task_creation_info> info)
{
  _task_creation_queue->push(std::move(info));
}

void task_creator::worker_function(int worker_id)
{
  while (_running.load()) {
    std::unique_ptr<task_creation_info> info = _task_creation_queue->pull();
    if (info == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      // scheduling scan task
      if (info->_node->type == ::duckdb::PhysicalOperatorType::TABLE_SCAN) {
        info->_pipeline->get_source()->set_creator(this);
        auto scan_task_global_state = std::make_shared<op::scan::duckdb_scan_task_global_state>(
          info->_pipeline,
          _duckdb_scan_executor,
          *_client_context,
          &info->_node->Cast<op::sirius_physical_table_scan>());
        duckdb::ThreadContext thread_ctx(*_client_context);
        duckdb::ExecutionContext exec_ctx(*_client_context, thread_ctx, nullptr);
        auto scan_task_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
          *scan_task_global_state, exec_ctx);
        if (info->destination_data_repositories.empty()) {
          throw std::runtime_error(
            "No destination data repositories provided for scan task creation.");
        }
        auto scan_task =
          std::make_unique<op::scan::duckdb_scan_task>(get_next_task_id(),
                                                       info->destination_data_repositories[0],
                                                       std::move(scan_task_local_state),
                                                       std::move(scan_task_global_state));
        _duckdb_scan_executor.schedule(std::move(scan_task));
        // scheduling pipeline task
      } else {
        duckdb::reference<sirius::op::sirius_physical_operator> node =
          info->_pipeline->get_inner_operators()[0];
        info->_pipeline->get_sink()->set_creator(this);
        // need to exhaust input batches until all ports are empty
        while (!node.get().all_ports_empty()) {
          auto input_batch = node.get().get_input_batch();
          auto global_state =
            std::make_shared<pipeline::gpu_pipeline_task_global_state>(info->_pipeline);
          auto local_state =
            std::make_unique<pipeline::gpu_pipeline_task_local_state>(input_batch, nullptr);
          auto task =
            std::make_unique<pipeline::gpu_pipeline_task>(get_next_task_id(),
                                                          info->destination_data_repositories,
                                                          std::move(local_state),
                                                          std::move(global_state));
          _pipeline_executor.schedule(std::move(task));
        }
      }

    } catch (const std::exception& e) {
      stop();
    }
  }
}

void task_creator::on_start() { _task_creation_queue->open(); }

void task_creator::on_stop() { _task_creation_queue->close(); }

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius::creator
