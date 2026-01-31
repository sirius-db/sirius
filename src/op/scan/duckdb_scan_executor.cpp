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

#include "op/scan/duckdb_scan_executor.hpp"

#include "log/logging.hpp"
#include "pipeline/pipeline_executor.hpp"
#include "pipeline/task_request.hpp"

namespace sirius::op::scan {

duckdb_scan_executor::duckdb_scan_executor(exec::thread_pool_config config) : _config(config), _kiosk(config.num_threads) {}

void duckdb_scan_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue.push(std::move(task));
}

void duckdb_scan_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _thread_pool    = std::make_unique<exec::thread_pool>(_config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
  _manager_thread = std::thread(&duckdb_scan_executor::manager_loop, this);
}

void duckdb_scan_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _kiosk.stop();
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _kiosk.wait_all();
  if (_thread_pool) { _thread_pool->stop(); }
}

void duckdb_scan_executor::wait_all() { _kiosk.wait_all(); }

void duckdb_scan_executor::set_pipeline_executor(sirius::pipeline::pipeline_executor& pipeline_exec)
{
  _pipeline_exec = &pipeline_exec;
}

void duckdb_scan_executor::submit_scan_request()
{
  if (_pipeline_exec) {
    // Device ID 0 for scan tasks (CPU-based), is_scan = true
    _pipeline_exec->submit_task_request(
      std::make_unique<sirius::pipeline::task_request>(0, true));
  }
}

void duckdb_scan_executor::manager_loop()
{
  while (_running.load()) {
    auto ticket = _kiosk.acquire();  // block till a thread is available
    if (!ticket.is_valid()) {
      SIRIUS_LOG_INFO("DuckDB Scan Executor: Kiosk interrupted, stopping manager loop");
      break;
    }
    auto scan_task = _task_queue.pop();  // block till a task is available
    if (!scan_task) {
      SIRIUS_LOG_INFO("DuckDB Scan Executor: task queue interrupted, stopping manager loop");
      break;
    }
    _thread_pool->schedule(
      [task = std::move(scan_task), ticket = std::move(ticket)]() mutable { task->execute(); });
  }
}

}  // namespace sirius::op::scan
