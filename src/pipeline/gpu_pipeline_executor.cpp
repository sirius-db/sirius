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

#include "creator/task_creator.hpp"
#include "cucascade/memory/stream_pool.hpp"
#include "cuda_runtime_api.h"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_queue.hpp"
#include "pipeline/task_request.hpp"

#include <rmm/cuda_device.hpp>

#include <util/stream_check_wrapper.hpp>

namespace sirius {
namespace pipeline {

gpu_pipeline_executor::gpu_pipeline_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_space* mem_space,
  exec::publisher<std::unique_ptr<task_request>> task_request_publisher)
  : _config(config),
    _stream_pool(rmm::cuda_device_id{mem_space->get_device_id()}, _config.num_threads),
    _task_request_publisher(std::move(task_request_publisher)),
    _memory_space(mem_space)
{
}

gpu_pipeline_executor::~gpu_pipeline_executor() { stop(); }

void gpu_pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue.push(std::move(task));
}

void gpu_pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _thread_pool =
    std::make_unique<exec::thread_pool>(_config.num_threads,
                                        _config.thread_name_prefix,
                                        _config.cpu_affinity_list,
                                        [device_id = _memory_space->get_device_id()]() noexcept {
                                          cudaSetDevice(device_id);
                                          sirius::util::enable_log_on_default_stream();
                                        });
  _manager_thread = std::thread(&gpu_pipeline_executor::manager_loop, this);
}

void gpu_pipeline_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _kiosk.stop();
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _kiosk.wait_all();
  if (_thread_pool) { _thread_pool->stop(); }
}

void gpu_pipeline_executor::manager_loop()
{
  rmm::cuda_set_device_raii set_device_guard(rmm::cuda_device_id{_memory_space->get_device_id()});
  sirius::util::enable_log_on_default_stream();
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
    auto* pipeline        = gpu_task->get_pipeline();
    auto exc_stream       = _stream_pool.acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
    _thread_pool->schedule([this,
                            task       = std::move(pipeline_task),
                            ticket     = std::move(ticket),
                            exc_stream = std::move(exc_stream),
                            consumers  = std::move(output_consumers),
                            pipeline]() mutable {
      try {
        task->execute(exc_stream);
      } catch (...) {
        if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
        return;
      }
      task.reset();

      // Check if query is complete BEFORE scheduling downstream tasks.
      // mark_completed() signals the future that engine.execute() is waiting on,
      // which may destroy the engine and its operators. We must not schedule
      // tasks that reference those operators after signaling completion.
      bool query_complete = false;
      if (_completion_handler && pipeline) {
        auto sink = pipeline->get_sink();
        if (sink && sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
          query_complete = pipeline->is_pipeline_finished();
        }
      }

      if (!query_complete && _task_creator) {
        for (auto* consumer : consumers) {
          _task_creator->schedule(consumer);
        }
      }

      if (query_complete && _completion_handler) { _completion_handler->mark_completed(); }
    });
  }
}

gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(sirius::parallel::itask* task)
{
  // Safely cast to gpu_pipeline_task
  return dynamic_cast<gpu_pipeline_task*>(task);
}

void gpu_pipeline_executor::set_task_creator(sirius::creator::task_creator* task_creator)
{
  _task_creator = task_creator;
}

void gpu_pipeline_executor::drain_leftover_tasks() { _task_queue.drain(); }

void gpu_pipeline_executor::set_completion_handler(completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

}  // namespace pipeline
}  // namespace sirius
