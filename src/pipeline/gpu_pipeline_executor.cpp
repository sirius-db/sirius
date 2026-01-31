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

#include "pipeline/gpu_pipeline_executor.hpp"

#include "pipeline/gpu_pipeline_queue.hpp"
#include "pipeline/pipeline_executor.hpp"

namespace sirius {
namespace pipeline {

gpu_pipeline_executor::gpu_pipeline_executor(exec::thread_pool_config config,
                                             cucascade::memory::memory_space* mem_space,
                                             pipeline_executor* pipeline_exec)
  : _config(config), _memory_space(mem_space), _pipeline_exec(pipeline_exec)
{
}

void gpu_pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue.push(std::move(task));
}

void gpu_pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _manager_thread = std::thread(&gpu_pipeline_executor::manager_loop, this);
}

void gpu_pipeline_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _kiosk.wait_all();
}

void gpu_pipeline_executor::submit_task_request()
{
  _pipeline_exec->submit_task_request(
    std::make_unique<task_request>(_memory_space->get_device_id(), false));
}

void gpu_pipeline_executor::manager_loop()
{
  while (_running.load()) {
    auto ticket = _kiosk.acquire();
    if (!ticket.is_valid()) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: Kiosk interrupted, stopping manager loop");
      break;
    }
    auto pipeline_task = _task_queue.pop();
    if (!pipeline_task) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: task queue interrupted, stopping manager loop");
      break;
    }
    auto* gpu_task   = cast_to_gpu_pipeline_task(pipeline_task.get());
    auto bytes_needs = gpu_task->get_estimated_reservation_size();
    auto reservation = _memory_space->make_reservation(bytes_needs);
    if (!reservation) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to acquire memory reservation for task {}",
                       gpu_task->get_task_id());
      break;
    }
    if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_itask_local_state*>(
          gpu_task->local_state())) {
      local_state->set_reservation(std::move(reservation));
    } else {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast local state for task {}",
                       gpu_task->get_task_id());
      break;
    }
    _thread_pool->schedule(
      [task = std::move(pipeline_task), ticket = std::move(ticket)]() mutable { task->execute(); });
  }
}

gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(sirius::parallel::itask* task)
{
  // Safely cast to gpu_pipeline_task
  return dynamic_cast<gpu_pipeline_task*>(task);
}

}  // namespace pipeline
}  // namespace sirius
