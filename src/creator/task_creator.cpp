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

namespace sirius {

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

task_creator::task_creator(std::unique_ptr<task_creation_queue> task_creation_queue,
                           size_t num_threads,
                           gpu_pipeline_hashmap& gpu_pipeline_map)
  : _task_creation_queue(std::move(task_creation_queue)),
    _num_threads(num_threads),
    _running(false),
    _gpu_pipeline_map(gpu_pipeline_map)
{
  for (int i = 0; i < gpu_pipeline_map._vec.size(); ++i) {
    if (gpu_pipeline_map._vec[i]->GetSource()->type == ::duckdb::PhysicalOperatorType::TABLE_SCAN) {
      priority_scans.push(gpu_pipeline_map._vec[i]);
    }
  }
}

task_creator::~task_creator() { stop_thread_pool(); }

void task_creator::process_next_task(::duckdb::GPUPhysicalOperator* node)
{
  auto hint = node->get_next_task_hint();
  if (std::holds_alternative<::duckdb::GPUPhysicalOperator*>(hint)) {
    auto* hint_node = std::get<::duckdb::GPUPhysicalOperator*>(hint);
    auto pipeline   = _gpu_pipeline_map._map[hint_node];
    schedule(std::make_unique<task_creation_info>(hint_node, pipeline));
  } else if (std::holds_alternative<::duckdb::shared_ptr<::duckdb::GPUPipeline>>(hint)) {
    auto pipeline = std::get<::duckdb::shared_ptr<::duckdb::GPUPipeline>>(hint);
    process_next_task(&pipeline->GetOperators()[0].get());
  } else {
    if (!priority_scans.empty()) {
      ::duckdb::shared_ptr<::duckdb::GPUPipeline> pipeline = priority_scans.front();
      auto* scan_node                                      = pipeline->GetSource().get();
      schedule(std::make_unique<task_creation_info>(scan_node, pipeline));
      priority_scans.pop();
    }
  }
}

void task_creator::start()
{
  while (!priority_scans.empty()) {
    ::duckdb::shared_ptr<::duckdb::GPUPipeline> pipeline = priority_scans.front();
    auto* scan_node                                      = pipeline->GetSource().get();
    schedule(std::make_unique<task_creation_info>(scan_node, pipeline));
    priority_scans.pop();
  }
}

void task_creator::start_thread_pool()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();
  _threads.reserve(_num_threads);
  for (size_t i = 0; i < _num_threads; ++i) {
    _threads.push_back(std::make_unique<std::thread>(&task_creator::worker_function, this, i));
  }
}

void task_creator::stop_thread_pool()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  on_stop();
  for (auto& thread : _threads) {
    if (thread->joinable()) { thread->join(); }
  }
  _threads.clear();
}

void task_creator::schedule(std::unique_ptr<task_creation_info> info)
{
  _task_creation_queue->push(std::move(info));
}

void task_creator::worker_function(int worker_id)
{
  while (true) {
    if (!_running.load()) {
      // Executor is stopped.
      break;
    }
    std::unique_ptr<task_creation_info> info = _task_creation_queue->pull();
    if (info == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      ::duckdb::reference<::duckdb::GPUPhysicalOperator> node = info->pipeline->GetOperators()[0];
      auto input_batch                                        = node.get().get_input_batch();
      // TODO: Create task from input_batch, node, and pipeline
    } catch (const std::exception& e) {
      schedule(std::move(info));
    }
  }
}

void task_creator::on_start() { _task_creation_queue->open(); }

void task_creator::on_stop() { _task_creation_queue->close(); }

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius
