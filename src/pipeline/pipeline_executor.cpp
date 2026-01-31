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
#include "creator/task_creator.hpp"
#include "exec/config.hpp"
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "pipeline/pipeline_queue.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius {
namespace pipeline {

pipeline_executor::pipeline_executor(const parallel::task_executor_config& gpu_task_executor_config,
                                     sirius::memory::sirius_memory_reservation_manager& mem_mgr,
                                     const cucascade::memory::system_topology_info* sys_topology)
{
  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  // Initialize GPU pipeline executors for each available GPU
  for (auto* space : gpu_spaces) {
    auto config   = gpu_task_executor_config;
    int device_id = space->get_device_id();
    if (sys_topology) {
      auto it = std::find_if(sys_topology->gpus.begin(),
                             sys_topology->gpus.end(),
                             [device_id](const cucascade::memory::gpu_topology_info& dev) {
                               return dev.id == device_id;
                             });

      if (it != sys_topology->gpus.end()) { config.cpu_affinity_list = it->cpu_cores; }
    }
    _gpu_executors.emplace(
      device_id,
      std::make_unique<gpu_pipeline_executor>(
        exec::thread_pool_config{}, const_cast<cucascade::memory::memory_space*>(space), this));
  }
}

void pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  if (task->is<sirius::op::scan::duckdb_scan_task>()) {
    _scan_queue.push(std::move(task));
  } else {
    _task_queue.push(std::move(task));
  }
}

void pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->start();
  }
}

void pipeline_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _task_queue.interrupt();
  _scan_queue.interrupt();
  _task_request_queue.interrupt();
  // Stop all GPU executors
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->stop();
  }
}

void pipeline_executor::submit_task_request(std::unique_ptr<task_request> request)
{
  _task_request_queue.push(std::move(request));
}

void pipeline_executor::dispatch_to_gpu_executor(std::unique_ptr<sirius::parallel::itask> task,
                                                 int gpu_id)
{
  auto it = _gpu_executors.find(gpu_id);
  if (it == _gpu_executors.end()) {
    throw std::runtime_error("Invalid GPU ID: " + std::to_string(gpu_id));
  }
  it->second->schedule(std::move(task));
}

void pipeline_executor::set_task_creator(sirius::creator::task_creator& task_creator)
{
  _task_creator = &task_creator;
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_task_creator(task_creator);
  }
}

void pipeline_executor::set_scan_executor(sirius::op::scan::duckdb_scan_executor& scan_executor)
{
  _scan_executor = &scan_executor;
}

void pipeline_executor::management_eventloop()
{
  while (_running.load()) {
    auto request = _task_request_queue.pop();
    if (request == nullptr) {
      SIRIUS_LOG_INFO("Task request queue closed, exiting management event loop.");
      break;
    }
    if (request->is_scan) {
      // Pop from scan queue and dispatch to scan executor
      auto task = _scan_queue.pop();
      if (task == nullptr) {
        SIRIUS_LOG_INFO("Scan queue closed, exiting management event loop.");
        break;
      }
      if (_scan_executor) {
        _scan_executor->schedule(std::move(task));
      } else {
        SIRIUS_LOG_ERROR("Scan executor not set, cannot dispatch scan task.");
      }
    } else {
      // Pop from task queue and dispatch to GPU executor
      auto task = _task_queue.pop();
      if (task == nullptr) {
        SIRIUS_LOG_INFO("Task queue closed, exiting management event loop.");
        break;
      }
      dispatch_to_gpu_executor(std::move(task), request->device_id);
    }
  }
}

}  // namespace pipeline
}  // namespace sirius
