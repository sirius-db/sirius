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
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/pipeline_queue.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius {
namespace pipeline {

pipeline_executor::pipeline_executor(const exec::thread_pool_config& gpu_executor_config,
                                     const exec::thread_pool_config& scan_executor_config,
                                     sirius::memory::sirius_memory_reservation_manager& mem_mgr,
                                     const cucascade::memory::system_topology_info* sys_topology)
{
  // Create the scan executor with memory manager for host allocations
  // Pass a publisher so it can submit task requests without depending on pipeline_executor
  _scan_executor = std::make_unique<sirius::op::scan::duckdb_scan_executor>(
    scan_executor_config, &mem_mgr, _task_request_channel.make_publisher());

  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  // Initialize GPU pipeline executors for each available GPU
  for (auto* space : gpu_spaces) {
    auto config   = gpu_executor_config;
    int device_id = space->get_device_id();
    if (sys_topology) {
      auto it = std::find_if(sys_topology->gpus.begin(),
                             sys_topology->gpus.end(),
                             [device_id](const cucascade::memory::gpu_topology_info& dev) {
                               return dev.id == device_id;
                             });

      if (it != sys_topology->gpus.end()) { config.cpu_affinity_list = it->cpu_cores; }
    }
    // Pass a publisher so gpu_pipeline_executor can submit task requests
    _gpu_executors.emplace(
      device_id,
      std::make_unique<gpu_pipeline_executor>(config,
                                              const_cast<cucascade::memory::memory_space*>(space),
                                              _task_request_channel.make_publisher()));
  }
}

pipeline_executor::~pipeline_executor() = default;

void pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  if (task->is<sirius::op::scan::duckdb_scan_task>()) {
    _scan_executor->schedule(std::move(task));
  } else {
    _task_queue.push(std::move(task));
  }
}

void pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _scan_executor->start();
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->start();
  }
}

void pipeline_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _task_queue.interrupt();
  _task_request_channel.close();
  _scan_executor->stop();
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->stop();
  }
}

void pipeline_executor::set_task_creator(sirius::creator::task_creator& task_creator)
{
  _task_creator = &task_creator;

  // Create a callback that captures task_creator and schedules operators
  auto schedule_callback = [&task_creator](sirius::op::sirius_physical_operator* op) {
    task_creator.schedule(op);
  };

  _scan_executor->set_schedule_callback(schedule_callback);
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_schedule_callback(schedule_callback);
  }
}

[[nodiscard]] sirius::op::scan::duckdb_scan_executor&
pipeline_executor::get_scan_executor() noexcept
{
  return *_scan_executor;
}

[[nodiscard]] const sirius::op::scan::duckdb_scan_executor& pipeline_executor::get_scan_executor()
  const noexcept
{
  return *_scan_executor;
}

void pipeline_executor::set_scan_caching_enabled(bool enabled)
{
  _scan_executor->set_scan_caching_enabled(enabled);
}

void pipeline_executor::set_priority_scans(const std::vector<op::sirius_physical_operator*>& scans)
{
  _scan_executor->prepare_cache_for_scan_operators(scans);

  std::lock_guard<std::mutex> lock(_priority_scans_mutex);
  while (!_priority_scans.empty()) {
    _priority_scans.pop();
  }
  for (auto* scan : scans) {
    _priority_scans.push(scan);
  }
}

void pipeline_executor::management_eventloop()
{
  while (_running.load()) {
    auto request = _task_request_channel.get();
    if (request == nullptr) {
      SIRIUS_LOG_INFO("Task request channel closed, exiting management event loop.");
      break;
    }
    if (!request->is_scan) {
      auto task = _task_queue.pop();
      if (task == nullptr) {
        SIRIUS_LOG_INFO("Task queue closed, exiting management event loop.");
        break;
      }
      _gpu_executors.at(request->device_id)->schedule(std::move(task));
    } else {
      // if scan executor doesn't have more tasks to run, it sends a request
      // pipeline executor will schedule more tasks from priority scans if available
      std::lock_guard<std::mutex> lock(_priority_scans_mutex);
      while (!_priority_scans.empty()) {
        auto* scan = _priority_scans.front();
        if (scan->can_create_more_tasks()) {
          _task_creator->schedule(scan);
        } else {
          _priority_scans.pop();
        }
      }
    }
  }
}

// WSM TODO: delete this?
void pipeline_executor::submit_task_request(std::unique_ptr<task_request> request)
{
  _task_request_queue->push(std::move(request));
}

// WSM TODO: delete this?
void pipeline_executor::dispatch_to_gpu_executor(std::unique_ptr<sirius::parallel::itask> task,
                                                 int gpu_id)
{
  if (gpu_id < 0 || gpu_id >= static_cast<int>(_gpu_executors.size())) {
    throw std::runtime_error("Invalid GPU ID: " + std::to_string(gpu_id));
  }
  _gpu_executors[gpu_id]->schedule(std::move(task));
}

void pipeline_executor::set_task_creator(creator::task_creator* creator)
{
  for (auto& gpu_exec : _gpu_executors) {
    gpu_exec->set_task_creator(creator);
  }
}

}  // namespace pipeline
}  // namespace sirius
