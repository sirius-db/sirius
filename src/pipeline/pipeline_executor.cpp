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

#include "creator/task_creator.hpp"
#include "exec/config.hpp"
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"

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

pipeline_executor::~pipeline_executor() { stop(); }

void pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  if (task->is<sirius::op::scan::duckdb_scan_task>()) {
    _scan_executor->schedule(std::move(task));
  } else if (task->is<sirius::op::scan::parquet_scan_task>()) {
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
  _management_thread = std::thread(&pipeline_executor::management_eventloop, this);
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
  if (_management_thread.joinable()) { _management_thread.join(); }
}

void pipeline_executor::set_task_creator(sirius::creator::task_creator& task_creator)
{
  _task_creator = &task_creator;

  _scan_executor->set_task_creator(_task_creator);
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_task_creator(_task_creator);
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

void pipeline_executor::set_scan_caching_config(sirius::op::scan::cache_level level)
{
  _scan_executor->set_scan_caching_enabled(level);
}

void pipeline_executor::prepare_for_query(duckdb::shared_ptr<planner::query> query)
{
  _current_query = query;

  // Drain leftover tasks from previous query
  _scan_executor->drain_leftover_tasks();
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_leftover_tasks();
  }

  auto scans = query->get_scan_operators();
  SIRIUS_LOG_INFO("[prepare_for_query] found {} scan operators", scans.size());
  for (size_t i = 0; i < scans.size(); ++i) {
    SIRIUS_LOG_INFO("[prepare_for_query] scan[{}]: type={}", i, static_cast<int>(scans[i]->type));
  }
  _scan_executor->prepare_cache_for_scan_operators(scans);

  std::lock_guard<std::mutex> lock(_priority_scans_mutex);
  while (!_priority_scans.empty()) {
    _priority_scans.pop();
  }
  for (auto* scan : scans) {
    _priority_scans.push(scan);
  }
}

std::future<void> pipeline_executor::start_query()
{
  // Create a new completion handler for this query
  _completion_handler      = std::make_unique<completion_handler>();
  std::future<void> future = _completion_handler->get_awaitable();

  // Set completion handler on all executors
  _scan_executor->set_completion_handler(_completion_handler.get());
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_completion_handler(_completion_handler.get());
  }

  // Set on_finished callbacks on result-collector pipelines so that 0-row scans
  // (which produce no GPU tasks) can signal the completion handler directly.
  auto* ch = _completion_handler.get();
  if (_current_query) {
  for (auto& pipeline : _current_query->get_pipelines()) {
    auto sink = pipeline->get_sink();
    if (sink && sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
      pipeline->on_finished = [ch]() {
        if (ch && !ch->is_completed()) {
          SIRIUS_LOG_INFO("pipeline on_finished: signaling completion (0-row scan)");
          ch->mark_completed();
        }
      };
    }
  }
  }

  // Schedule initial scan operators from the priority queue.
  // Each scan operator gets ONE task_creator::schedule() call — the scan
  // continuation mechanism (via next_task_hint) creates additional tasks.
  {
    std::lock_guard<std::mutex> lock(_priority_scans_mutex);
    while (!_priority_scans.empty()) {
      auto* scan_op = _priority_scans.front();
      SIRIUS_LOG_INFO("[start_query] scheduling scan type={} (remaining={})",
                      static_cast<int>(scan_op->type), _priority_scans.size());
      _task_creator->schedule(scan_op);
      _priority_scans.pop();
    }
  }

  return future;
}

void pipeline_executor::terminate_query(std::exception_ptr error)
{
  _completion_handler->report_error(error);
  stop();
}

void pipeline_executor::drain_after_error()
{
  SIRIUS_LOG_INFO("pipeline_executor: draining after error");
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
  SIRIUS_LOG_INFO("pipeline_executor: DONE draining after error");
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
      // Scan re-scheduling disabled — continuation handled by task creator's
      // next_task_hint mechanism. Re-enabling causes scan interleaving regression.
    }
  }
}

void pipeline_executor::schedule_next_scan_tasks()
{
  std::lock_guard<std::mutex> lock(_priority_scans_mutex);
  SIRIUS_LOG_INFO("[schedule_next_scan_tasks] priority_scans queue size={}", _priority_scans.size());
  if (_priority_scans.empty()) {
    return;
  }
  auto num_threads = _scan_executor->get_num_threads();
  while (!_priority_scans.empty()) {
    auto* scan_op = _priority_scans.front();
    SIRIUS_LOG_INFO("[schedule_next_scan_tasks] scheduling scan type={} on {} threads (remaining={})",
                    static_cast<int>(scan_op->type), num_threads, _priority_scans.size());
    for (auto i = 0; i != num_threads; ++i) {
      _task_creator->schedule(scan_op);
    }
    _priority_scans.pop();
  }
}

}  // namespace pipeline
}  // namespace sirius
