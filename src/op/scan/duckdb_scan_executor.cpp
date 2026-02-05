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

#include "creator/task_creator.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/completion_handler.hpp"
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

duckdb_scan_executor::~duckdb_scan_executor() { stop(); }

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
  if (_thread_pool) { _thread_pool->stop(); }
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _kiosk.wait_all();
}

void duckdb_scan_executor::wait_all() { _kiosk.wait_all(); }

void duckdb_scan_executor::set_task_creator(sirius::creator::task_creator* task_creator)
{
  _task_creator = task_creator;
}

void duckdb_scan_executor::drain_leftover_tasks() { _task_queue.drain(); }

void duckdb_scan_executor::set_completion_handler(
  sirius::pipeline::completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

void duckdb_scan_executor::cache_scan_results_for_query(const std::string& query)
{
  if (!_caching_enabled) { return; }
  std::hash<std::string> hash_fn;
  auto new_query_hash = hash_fn(query);
  if (new_query_hash == _query_hash) {
    SIRIUS_LOG_INFO("Scan results for query already cached, preloading: {}", query);
    return;
  }
  SIRIUS_LOG_INFO("Caching scan results for query: {}", query);
  _query_hash = new_query_hash;
  _cache.clear();
}

void duckdb_scan_executor::set_scan_caching_enabled(bool enabled)
{
  _caching_enabled = enabled;
  SIRIUS_LOG_INFO("Scan caching {}", enabled ? "enabled" : "disabled");
}

void duckdb_scan_executor::prepare_cache_for_scan_operators(
  const std::vector<sirius::op::sirius_physical_operator*>& scan_operators)
{
  if (!_caching_enabled) { return; }

  std::lock_guard<std::mutex> lock(_cache_mutex);
  _preload_mode = !_cache.empty();

  if (!_preload_mode) {
    for (auto* op : scan_operators) {
      auto operator_id    = op->get_pipeline()->get_pipeline_id();
      _cache[operator_id] = std::make_unique<cache_entry>();  // Create empty entry
    }
  } else {
    // In PRELOAD mode: verify all operator IDs are present in the cache
    for (auto* op : scan_operators) {
      auto operator_id = op->get_pipeline()->get_pipeline_id();
      if (_cache.find(operator_id) == _cache.end()) {
        SIRIUS_LOG_ERROR("Cache entry not found for operator {} in PRELOAD mode", operator_id);
      }
    }
  }
}

void duckdb_scan_executor::submit_scan_request()
{
  // Device ID 0 for scan tasks (CPU-based), is_scan = true
  [[maybe_unused]] auto result =
    _task_request_publisher.send(std::make_unique<sirius::pipeline::task_request>(0, true));
}

std::vector<std::shared_ptr<cucascade::data_batch>> duckdb_scan_executor::get_scan_output(
  op::scan::duckdb_scan_task* task)
{
  if (!_caching_enabled) {
    return task->compute_task();
  } else {
    auto pipe_id = task->get_pipeline_id();
    std::lock_guard<std::mutex> lock(_cache_mutex);
    // todo (amin) : we need to clone the batches to avoid modifying the original batches
    auto& entry = _cache.at(pipe_id);
    if (!entry) { throw std::runtime_error("Scan results for query not cached"); }
    if (_preload_mode) {
      if (entry->batch_index >= entry->batches.size()) {
        throw std::runtime_error("Scan results for query not cached");
      }
      return entry->batches[entry->batch_index++];
    } else {
      auto batches = task->compute_task();
      entry->batches.push_back(batches);
      return batches;
    }
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
    auto task = _task_queue.try_pop();
    if (!task) {
      if (!_running) {
        SIRIUS_LOG_INFO("DuckDB Scan Executor: task queue interrupted, stopping manager loop");
        break;
      } else {
        submit_scan_request();  // tell pipeline executor to submit a scan task request
        task = _task_queue.pop();
        if (!task) {
          SIRIUS_LOG_INFO("DuckDB Scan Executor: task queue interrupted, stopping manager loop");
          break;
        }
      }
    }

    // Make host memory reservation and set it on the local state
    auto* scan_task = dynamic_cast<sirius::op::scan::duckdb_scan_task*>(task.get());
    // todo (amin): fix this later, and make the reservation in the executor.
    if (scan_task and false) {
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
    }

    _thread_pool->schedule([this,
                            ticket    = std::move(ticket),
                            t         = std::move(task),
                            scan_task = std::move(scan_task)]() mutable {
      try {
        auto consumers = scan_task->get_output_consumers();
        auto batches   = get_scan_output(scan_task);
        scan_task->publish_output(std::move(batches));
        t.reset();
        if (_task_creator) {
          for (auto* consumer : consumers) {
            _task_creator->schedule(consumer);
          }
        }
      } catch (...) {
        /// this is fatal error
        if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
      }
    });
  }
}

}  // namespace sirius::op::scan
