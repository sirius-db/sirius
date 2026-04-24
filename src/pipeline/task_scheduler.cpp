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

#include "pipeline/task_scheduler.hpp"

#include "creator/task_creator.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/config.hpp"
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/cpu_source_task.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius {
namespace pipeline {

task_scheduler::task_scheduler(
  const exec::thread_pool_config& gpu_executor_config,
  const exec::thread_pool_config& scan_executor_config,
  sirius::memory::sirius_memory_reservation_manager& mem_mgr,
  const cucascade::memory::system_topology_info* sys_topology,
  const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>* downgrade_executors)
{
  // Create the scan executor with memory manager for host allocations
  // Pass a publisher so it can submit task requests without depending on task_scheduler
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
    // Find matching downgrade executor for this GPU space
    sirius::parallel::downgrade_executor* dg_exec = nullptr;
    if (downgrade_executors) {
      for (auto& de : *downgrade_executors) {
        if (de->get_space_id() == space->get_id()) {
          dg_exec = de.get();
          break;
        }
      }
    }

    // Pass a publisher so gpu_pipeline_executor can submit task requests
    _gpu_executors.emplace(
      device_id,
      std::make_unique<gpu_pipeline_executor>(config,
                                              const_cast<cucascade::memory::memory_space*>(space),
                                              _task_request_channel.make_publisher(),
                                              dg_exec));
  }
}

task_scheduler::~task_scheduler() { stop(); }

void task_scheduler::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  if (task->is<sirius::op::scan::duckdb_scan_task>()) {
    _scan_executor->schedule(std::move(task));
  } else if (task->is<sirius::op::scan::parquet_scan_task>()) {
    _scan_executor->schedule(std::move(task));
  } else if (task->is<sirius::op::scan::cpu_source_task>()) {
    _scan_executor->schedule(std::move(task));
  } else {
    _task_queue.push(std::move(task));
  }
}

void task_scheduler::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _scan_executor->start();
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->start();
  }
  _management_thread = std::thread(&task_scheduler::management_eventloop, this);
}

void task_scheduler::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _task_queue.interrupt();
  _task_request_channel.close();
  _scan_executor->stop();
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->stop();
  }
  if (_management_thread.joinable()) { _management_thread.join(); }
}

void task_scheduler::set_task_creator(sirius::creator::task_creator& task_creator)
{
  _task_creator = &task_creator;

  _scan_executor->set_task_creator(_task_creator);
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_task_creator(_task_creator);
  }
}

[[nodiscard]] sirius::op::scan::duckdb_scan_executor& task_scheduler::get_scan_executor() noexcept
{
  return *_scan_executor;
}

[[nodiscard]] const sirius::op::scan::duckdb_scan_executor& task_scheduler::get_scan_executor()
  const noexcept
{
  return *_scan_executor;
}

void task_scheduler::set_scan_caching_config(sirius::op::scan::cache_level level)
{
  _scan_executor->set_scan_caching_enabled(level);
}

void task_scheduler::prepare_for_query(duckdb::shared_ptr<planner::query> query)
{
  // Drain leftover tasks from previous query
  _scan_executor->drain_leftover_tasks();
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_leftover_tasks();
  }

  auto scans = query->get_scan_operators();
  _scan_executor->prepare_cache_for_scan_operators(scans);

  std::lock_guard<std::mutex> lock(_priority_scans_mutex);
  while (!_priority_scans.empty()) {
    _priority_scans.pop();
  }
  for (auto* scan : scans) {
    _priority_scans.push(scan);
  }
}

std::future<void> task_scheduler::start_query()
{
  // Create a new completion handler for this query
  _completion_handler      = std::make_unique<completion_handler>();
  std::future<void> future = _completion_handler->get_awaitable();

  // Set completion handler on all executors
  _scan_executor->set_completion_handler(_completion_handler.get());
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_completion_handler(_completion_handler.get());
  }

  constexpr int k_initial_scans = 2;
  std::lock_guard<std::mutex> lock(_priority_scans_mutex);
  for (int i = 0; i < k_initial_scans && !_priority_scans.empty(); ++i) {
    auto* scan_op = _priority_scans.front();
    _task_creator->schedule(scan_op);
    _priority_scans.pop();
  }

  return future;
}

void task_scheduler::terminate_query(std::exception_ptr error)
{
  _completion_handler->report_error(error);
  stop();
}

void task_scheduler::drain_after_error()
{
  SIRIUS_LOG_INFO("task_scheduler: draining after error");
  // Drain the task creator first so no thread is inside get_next_task_input_data()/
  // pop_data_batch() when QueryEnd() clears repositories (avoids use-after-free).
  if (_task_creator) { _task_creator->stop_thread_pool(); }
  // Drain the top-level task queue so management_eventloop doesn't dispatch
  // stale tasks from the failed query.
  _task_queue.drain();

  // Stop the scan executor's manager loop, wait for in-flight scan tasks to
  // finish, then restart the manager for the next query.  We must use
  // drain_and_wait() (not just drain + wait_all) because the scan manager
  // thread holds a kiosk ticket while blocked on pop(); without interrupting
  // the queue and stopping the kiosk first, wait_all() deadlocks.
  _scan_executor->drain_and_wait();

  // Interrupt each GPU executor's manager loop, wait for in-flight thread-pool
  // tasks to finish, then restart the manager for the next query.
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_and_wait();
  }
  if (_task_creator) { _task_creator->start_thread_pool(); }
  SIRIUS_LOG_INFO("task_scheduler: DONE draining after error");
}

void task_scheduler::management_eventloop()
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
    }
  }
}

}  // namespace pipeline
}  // namespace sirius
