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

#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_queue.hpp"
#include "pipeline/task_request.hpp"

namespace sirius {
namespace pipeline {

gpu_pipeline_executor::gpu_pipeline_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_space* mem_space,
  exec::publisher<std::unique_ptr<task_request>> task_request_publisher)
  : _config(config),
    _task_request_publisher(std::move(task_request_publisher)),
    _memory_space(mem_space),
    _task_creator(nullptr)
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

void gpu_pipeline_executor::manager_loop()
{
  while (_running.load()) {
    auto ticket = _kiosk.acquire();  // block till a thread is available
    if (!ticket.is_valid()) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: Kiosk interrupted, stopping manager loop");
      break;
    }
    if (!_task_request_publisher.send(
          std::make_unique<pipeline::task_request>(_memory_space->get_device_id(), false))) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: Failed to send task request, channel is closed");
      break;
    }
    auto pipeline_task = _task_queue.pop();  // block till a task is available
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
    auto output_consumers = gpu_task->get_output_consumers();
    _thread_pool->schedule([this,
                            task      = std::move(pipeline_task),
                            ticket    = std::move(ticket),
                            consumers = std::move(output_consumers)]() mutable {
      task->execute();
      task.reset();
      if (_schedule_callback) {
        for (auto* consumer : consumers) {
          _schedule_callback(consumer);
        }
      }
    });
  }
}

gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(sirius::parallel::itask* task)
{
  // Safely cast to gpu_pipeline_task
  return dynamic_cast<gpu_pipeline_task*>(task);
}

void gpu_pipeline_executor::set_schedule_callback(
  std::function<void(sirius::op::sirius_physical_operator*)> schedule_fn)
{
  _schedule_callback = std::move(schedule_fn);
}

void gpu_pipeline_executor::set_task_creator(creator::task_creator* creator)
{
  _task_creator = creator;
}

}  // namespace pipeline
}  // namespace sirius
