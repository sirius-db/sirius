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
#include "cucascade/data/cpu_data_representation.hpp"
#include "cucascade/data/data_batch.hpp"
#include "cucascade/data/gpu_data_representation.hpp"
#include "data/cached_data_representation.hpp"
#include "data/data_batch_utils.hpp"
#include "data/host_parquet_representation.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/common.hpp>

#include <mutex>

namespace sirius::op::scan {

duckdb_scan_executor::duckdb_scan_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_reservation_manager* mem_mgr,
  exec::publisher<std::unique_ptr<sirius::pipeline::task_request>> task_request_publisher)
  : _config(config),
    _kiosk(_config.num_threads),
    _task_request_publisher(std::move(task_request_publisher)),
    _mem_mgr(mem_mgr)
{
  auto gpu_spaces   = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  _gpu_memory_space = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  _stream_pool      = std::make_unique<cucascade::memory::exclusive_stream_pool>(
    rmm::cuda_device_id(_gpu_memory_space->get_device_id()), _config.num_threads);
}

duckdb_scan_executor::~duckdb_scan_executor()
{
  {
    std::lock_guard lock(_cache_mutex);
    _cache.clear();
  }
  stop();
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

void duckdb_scan_executor::drain_and_wait()
{
  // Stop the kiosk so the manager_loop's acquire() returns an invalid ticket
  // (the manager may be blocked there when all thread-pool slots are full).
  _kiosk.stop();

  // Interrupt pop() so the manager_loop sees a nullptr and breaks out.
  _task_queue.interrupt();

  // Join the manager thread so we know it has exited.
  if (_manager_thread.joinable()) { _manager_thread.join(); }

  // Wait for all in-flight thread-pool tasks to finish.
  _kiosk.wait_all();

  // Clear any remaining tasks from the queue.
  _task_queue.drain();

  // Re-enable the kiosk and queue so the executor is ready for the next query.
  _kiosk.resume();
  _task_queue.reactivate();
  _manager_thread = std::thread(&duckdb_scan_executor::manager_loop, this);
}

void duckdb_scan_executor::set_completion_handler(
  sirius::pipeline::completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

bool duckdb_scan_executor::cache_scan_results_for_query(const std::string& query)
{
  if (_cache_level == cache_level::NONE) { return false; }
  // Only track queries that go through the Sirius GPU execution path.
  // Other SQL statements (SET, INSERT, etc.) don't produce scan tasks
  // and should not invalidate the cache.
  if (query.find("gpu_execution") == std::string::npos &&
      query.find("gpu_processing") == std::string::npos) {
    return false;
  }
  std::hash<std::string> hash_fn;
  auto new_query_hash = hash_fn(query);
  if (new_query_hash == _query_hash) {
    SIRIUS_LOG_INFO("Scan results for query already cached, preloading: {}", query);
    return true;
  }
  SIRIUS_LOG_INFO("Caching scan results for query: {}", query);
  _query_hash = new_query_hash;
  _cache.clear();
  return false;
}

void duckdb_scan_executor::set_scan_caching_enabled(cache_level level)
{
  if (level == _cache_level) { return; }
  {
    std::lock_guard lock(_cache_mutex);
    _cache.clear();  // Clear cache when changing caching config
  }
  _cache_level = level;
  std::string level_str;
  enum_to_string(level, level_str);
  SIRIUS_LOG_INFO("Scan caching level set to {}", level_str);
}

void duckdb_scan_executor::prepare_cache_for_scan_operators(
  const std::vector<sirius::op::sirius_physical_operator*>& scan_operators)
{
  if (_cache_level == cache_level::NONE) { return; }

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
      auto iter        = _cache.find(operator_id);
      if (iter == _cache.end()) {
        SIRIUS_LOG_ERROR("Cache entry not found for operator {} in PRELOAD mode", operator_id);
        throw std::runtime_error("Cache entry not found for operator " +
                                 std::to_string(operator_id) + " in PRELOAD mode");
      }
      iter->second->batch_index = 0;  // Reset batch index for PRELOAD mode
    }
  }
}

void duckdb_scan_executor::submit_scan_request()
{
  // Device ID 0 for scan tasks (CPU-based), is_scan = true
  [[maybe_unused]] auto result =
    _task_request_publisher.send(std::make_unique<sirius::pipeline::task_request>(0, true));
}

std::vector<std::shared_ptr<cucascade::data_batch>> duckdb_scan_executor::clone_cached_batches(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  scan_type type,
  rmm::cuda_stream_view stream)
{
  if (_cache_level == cache_level::TABLE_GPU) { return batches; }

  std::vector<std::shared_ptr<cucascade::data_batch>> cloned;
  cloned.reserve(batches.size());

  if (type == scan_type::DUCKDB) {
    for (auto& batch : batches) {
      cloned.push_back(batch->clone(get_next_batch_id(), stream));
    }
  } else {
    // Parquet cache entries use host representations that support shallow_clone
    for (auto& batch : batches) {
      auto* idata_rep = batch->get_data();
      if (auto* host_data = dynamic_cast<cached_host_data_representation*>(idata_rep)) {
        cloned.push_back(
          std::make_shared<cucascade::data_batch>(get_next_batch_id(), host_data->shallow_clone()));
      } else if (auto* parquet_rep = dynamic_cast<cached_host_parquet_representation*>(idata_rep)) {
        cloned.push_back(std::make_shared<cucascade::data_batch>(get_next_batch_id(),
                                                                 parquet_rep->shallow_clone()));
      } else {
        throw std::runtime_error("Unexpected data representation in parquet cache entry");
      }
    }
  }
  return cloned;
}

void duckdb_scan_executor::mark_scan_pipeline_finished(op::sirius_physical_operator* scan_op)
{
  auto pipeline = scan_op->get_pipeline();
  if (scan_op->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
    scan_op->Cast<op::sirius_physical_duckdb_scan>().exhausted.store(true);
  } else if (scan_op->type == op::SiriusPhysicalOperatorType::PARQUET_SCAN) {
    scan_op->Cast<op::sirius_physical_parquet_scan>().has_more_partitions.store(false);
  }
  pipeline->update_pipeline_status();
}

std::unique_ptr<op::operator_data> duckdb_scan_executor::get_scan_output(
  pipeline::sirius_pipeline_itask* task, rmm::cuda_stream_view stream)
{
  auto type =
    dynamic_cast<duckdb_scan_task*>(task) != nullptr ? scan_type::DUCKDB : scan_type::PARQUET;

  if (_cache_level == cache_level::NONE) {
    return task->compute_task(stream);
  } else {
    auto pipe_id = task->get_pipeline_id();
    std::lock_guard<std::mutex> lock(_cache_mutex);
    auto& entry = _cache.at(pipe_id);
    if (!entry) { throw std::runtime_error("Scan results for query not cached"); }
    if (_preload_mode) {
      // Preload is normally handled by preload_into_pipelines() before any tasks
      // are created.  This fallback handles the edge case where a task slips through.
      if (entry->batch_index >= entry->batches.size()) {
        return std::make_unique<op::operator_data>(
          std::vector<std::shared_ptr<cucascade::data_batch>>{});
      }
      auto cloned = clone_cached_batches(entry->batches[entry->batch_index++], type, stream);
      return std::make_unique<op::operator_data>(std::move(cloned));
    } else {
      auto scan_output = task->compute_task(stream);
      entry->batches.push_back(clone_cached_batches(scan_output->get_data_batches(), type, stream));
      return scan_output;
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

    auto* scan_task = dynamic_cast<pipeline::sirius_pipeline_itask*>(task.get());
    if (scan_task && scan_task->is<parquet_scan_task>()) {
      auto* parquet_task = dynamic_cast<parquet_scan_task*>(scan_task);
      if (_cache_level != cache_level::NONE) {
        bool wrap_batch_data     = _cache_level != cache_level::TABLE_GPU;
        bool cache_decoded_table = _cache_level == cache_level::TABLE_HOST;
        parquet_task->set_materialized_columns(
          wrap_batch_data, cache_decoded_table, _gpu_memory_space);
      }
      auto bytes_needed = scan_task->get_estimated_reservation_size();
      auto reservation  = _mem_mgr->request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST}, bytes_needed);
      if (!reservation) {
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to acquire host memory reservation");
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to acquire host memory reservation");
        break;
      }
      if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(
            scan_task->local_state())) {
        local_state->set_reservation(std::move(reservation));
      } else {
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to cast local state for task");
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to cast local state for task");
        break;
      }
    }

    auto exc_stream = _stream_pool->acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
    _thread_pool->schedule([this,
                            ticket    = std::move(ticket),
                            stream    = std::move(exc_stream),
                            t         = std::move(task),
                            scan_task = std::move(scan_task)]() mutable {
      try {
        auto consumers = scan_task->get_output_consumers();
        {
          auto output_data = get_scan_output(scan_task, stream);
          stream->synchronize();
          scan_task->publish_output(*output_data, stream);
        }

        t.reset();
        if (_task_creator && !(_completion_handler && _completion_handler->is_completed())) {
          for (auto* consumer : consumers) {
            _task_creator->schedule(consumer);
          }
        }
      } catch (...) {
        if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
      }
    });
  }
}

std::vector<op::sirius_physical_operator*> duckdb_scan_executor::preload_into_pipelines(
  const std::vector<op::sirius_physical_operator*>& scan_operators)
{
  if (!_preload_mode) { return {}; }

  std::vector<op::sirius_physical_operator*> all_consumers;
  std::lock_guard<std::mutex> lock(_cache_mutex);

  auto exc_stream = _stream_pool->acquire_stream(
    cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);

  for (auto* scan_op : scan_operators) {
    // Determine scan type — skip unknown operator types
    scan_type type;
    if (scan_op->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      type = scan_type::DUCKDB;
    } else if (scan_op->type == op::SiriusPhysicalOperatorType::PARQUET_SCAN) {
      type = scan_type::PARQUET;
    } else {
      continue;
    }

    auto pipeline    = scan_op->get_pipeline();
    auto pipeline_id = pipeline->get_pipeline_id();
    auto it          = _cache.find(pipeline_id);
    if (it == _cache.end() || !it->second) {
      SIRIUS_LOG_WARN("Preload: no cache entry for scan pipeline {}", pipeline_id);
      continue;
    }

    auto& entry = it->second;

    // Find destination data repositories (same logic as task_creator)
    auto sink = pipeline->get_sink();
    if (!sink) { continue; }
    auto next_ports = sink->get_next_port_after_sink();
    if (next_ports.empty()) { continue; }

    // Collect all cached batches, cloned or shared based on cache level + scan type
    std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
    while (entry->batch_index < entry->batches.size()) {
      auto cloned = clone_cached_batches(entry->batches[entry->batch_index++], type, exc_stream);
      all_batches.insert(all_batches.end(),
                         std::make_move_iterator(cloned.begin()),
                         std::make_move_iterator(cloned.end()));
    }

    // Push batches to all destination repos
    for (auto& batch : all_batches) {
      for (auto& [next_op, port_id] : next_ports) {
        auto* port = next_op->get_port(port_id);
        if (port && port->repo) { port->repo->add_data_batch(batch); }
      }
    }

    mark_scan_pipeline_finished(scan_op);

    SIRIUS_LOG_INFO("Preload: injected {} batches for {} scan pipeline {}",
                    all_batches.size(),
                    type == scan_type::DUCKDB ? "DuckDB" : "parquet",
                    pipeline_id);

    auto consumers = pipeline->get_output_consumers();
    all_consumers.insert(all_consumers.end(), consumers.begin(), consumers.end());
  }

  if (_cache_level != cache_level::TABLE_GPU) { exc_stream->synchronize(); }

  return all_consumers;
}

}  // namespace sirius::op::scan
