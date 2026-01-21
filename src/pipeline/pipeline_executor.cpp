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

#include "pipeline/pipeline_executor.hpp"

#include "config.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "pipeline/pipeline_queue.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>

namespace sirius {
namespace pipeline {

pipeline_executor::pipeline_executor(const parallel::task_executor_config& gpu_task_executor_config,
                                     sirius::memory::sirius_memory_reservation_manager& mem_mgr)
  : sirius::parallel::itask_executor(std::make_unique<pipeline_queue>(1),
                                     {.num_threads = 1, .retry_on_error = false})
{
  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto num_gpus   = gpu_spaces.size();
  // Initialize GPU pipeline executors for each available GPU
  _gpu_executors.reserve(num_gpus);
  for (auto* space : gpu_spaces) {
    _gpu_executors.push_back(
      std::make_unique<gpu_pipeline_executor>(gpu_task_executor_config, space, this));
  }
  _task_request_queue = std::make_unique<task_request_queue>(1);
}

void pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue->push(std::move(task));
}

void pipeline_executor::on_start()
{
  _task_queue->open();
  _task_request_queue->open();
}

void pipeline_executor::on_stop()
{
  _task_queue->close();
  _task_request_queue->close();
}

void pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();
  _threads.reserve(_config.num_threads);
  for (int i = 0; i < _config.num_threads; ++i) {
    _threads.push_back(std::make_unique<sirius::parallel::task_executor_thread>(
      std::make_unique<std::thread>(&pipeline_executor::worker_loop, this, i)));
  }
  // Start all GPU executors
  for (auto& gpu_exec : _gpu_executors) {
    gpu_exec->start();
  }
}

void pipeline_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  // Stop all GPU executors
  for (auto& gpu_exec : _gpu_executors) {
    gpu_exec->stop();
  }
  on_stop();
  for (auto& thread : _threads) {
    if (thread->_internal_thread->joinable()) { thread->_internal_thread->join(); }
  }
  _threads.clear();
}

void pipeline_executor::worker_loop(int worker_id)
{
  while (true) {
    if (!_running.load()) {
      // Executor is stopped.
      break;
    }
    auto request = _task_request_queue->pull();
    if (request == nullptr) {
      // Task request queue is closed.
      break;
    }
    auto task = _task_queue->pull();
    if (task == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      // TODO
      // Make reservation (prioritize GPU with the same memory space as the input)
      // If approved, dispatch to the corresponding GPU executor
      // If no reservation, use some policy to pick the best GPU executor
      // Dispatch to the selected GPU executor based on the request
      int gpu_id = request ? request->device_id : 0;
      dispatch_to_gpu_executor(std::move(task), gpu_id);  // For now we just dispatch to GPU 0
    } catch (const std::exception& e) {
      on_task_error(worker_id, std::move(task), e);
    }
  }
}

void pipeline_executor::submit_task_request(std::unique_ptr<task_request> request)
{
  _task_request_queue->push(std::move(request));
}

void pipeline_executor::dispatch_to_gpu_executor(std::unique_ptr<sirius::parallel::itask> task,
                                                 int gpu_id)
{
  if (gpu_id < 0 || gpu_id >= static_cast<int>(_gpu_executors.size())) {
    throw std::runtime_error("Invalid GPU ID: " + std::to_string(gpu_id));
  }
  _gpu_executors[gpu_id]->schedule(std::move(task));
}

}  // namespace pipeline
}  // namespace sirius
