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
#include "op/scan/duckdb_scan_task.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline_itask_local_state.hpp"

#include <cucascade/memory/common.hpp>

namespace sirius::op::scan {

duckdb_scan_executor::duckdb_scan_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_reservation_manager* mem_mgr,
  exec::publisher<std::unique_ptr<sirius::pipeline::task_request>> task_request_publisher)
  : _config(config),
    _kiosk(config.num_threads),
    _task_request_publisher(std::move(task_request_publisher)),
    _mem_mgr(mem_mgr)
{
}

void duckdb_scan_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue.push(std::move(task));
}

void duckdb_scan_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _thread_pool = std::make_unique<exec::thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
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

void duckdb_scan_executor::set_schedule_callback(
  std::function<void(sirius::op::sirius_physical_operator*)> schedule_fn)
{
  _schedule_callback = std::move(schedule_fn);
}

void duckdb_scan_executor::submit_scan_request()
{
  // Device ID 0 for scan tasks (CPU-based), is_scan = true
  [[maybe_unused]] auto result =
    _task_request_publisher.send(std::make_unique<sirius::pipeline::task_request>(0, true));
}

void duckdb_scan_executor::manager_loop()
{
  while (_running.load()) {
    auto ticket = _kiosk.acquire();  // block till a thread is available
    if (!ticket.is_valid()) {
      SIRIUS_LOG_INFO("DuckDB Scan Executor: Kiosk interrupted, stopping manager loop");
      break;
    }
    auto task = _task_queue.try_pop();
    if (!task && !_running) {
      SIRIUS_LOG_INFO("DuckDB Scan Executor: task queue interrupted, stopping manager loop");
      break;
    }
    submit_scan_request();  // tell pipeline executor to submit a scan task request
    task = _task_queue.pop();
    if (!task) {
      SIRIUS_LOG_INFO("DuckDB Scan Executor: task queue interrupted, stopping manager loop");
      break;
    }

    std::vector<sirius_physical_operator*> output_consumers;
    // Make host memory reservation and set it on the local state
    if (auto* scan_task = dynamic_cast<sirius::op::scan::duckdb_scan_task*>(task.get())) {
      auto bytes_needed = scan_task->get_estimated_reservation_size();
      auto reservation  = _mem_mgr->request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST}, bytes_needed);
      if (!reservation) {
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to acquire host memory reservation");
        break;
      }
      if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_itask_local_state*>(
            scan_task->local_state())) {
        local_state->set_reservation(std::move(reservation));
      } else {
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to cast local state for task");
        break;
      }
      output_consumers = scan_task->get_output_consumers();
    }

    _thread_pool->schedule([this,
                            t         = std::move(task),
                            ticket    = std::move(ticket),
                            consumers = std::move(output_consumers)]() mutable {
      t->execute();
      t.reset();
      if (_schedule_callback) {
        for (auto* consumer : consumers) {
          _schedule_callback(consumer);
        }
      }
    });
  }
}

}  // namespace sirius::op::scan
