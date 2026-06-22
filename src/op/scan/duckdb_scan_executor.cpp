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
#include "op/scan/cpu_source_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/common.hpp>

#include <algorithm>
#include <cctype>
#include <limits>
#include <mutex>

namespace sirius::op::scan {

namespace {

bool starts_with_query_keyword(const std::string& query, const std::string_view keyword)
{
  auto pos = query.find_first_not_of(" \t\r\n");
  if (pos == std::string::npos) { return false; }
  if (query.size() - pos < keyword.size()) { return false; }
  for (std::size_t i = 0; i < keyword.size(); ++i) {
    auto ch = static_cast<unsigned char>(query[pos + i]);
    if (std::tolower(ch) != keyword[i]) { return false; }
  }
  return true;
}

bool is_cacheable_query_text(const std::string& query)
{
  return query.find("gpu_execution") != std::string::npos ||
         query.find("gpu_processing") != std::string::npos ||
         starts_with_query_keyword(query, "select") || starts_with_query_keyword(query, "with");
}

}  // namespace

duckdb_scan_executor::duckdb_scan_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_reservation_manager* mem_mgr,
  exec::publisher<std::unique_ptr<sirius::pipeline::task_request>> task_request_publisher)
  : sirius::parallel::itask_executor(config),
    _task_request_publisher(std::move(task_request_publisher)),
    _mem_mgr(mem_mgr)
{
  auto gpu_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  for (auto* space : gpu_spaces) {
    _gpu_memory_spaces.push_back(const_cast<cucascade::memory::memory_space*>(space));
  }
  // Keep backward compat: _gpu_memory_space points to first GPU for stream pool init
  _gpu_memory_space = _gpu_memory_spaces.empty() ? nullptr : _gpu_memory_spaces[0];

  // FIX-01 (v1.2): construct one exclusive_stream_pool per GPU, keyed by
  // device_id. Each pool's streams are created under that device's
  // rmm::cuda_set_device_raii so acquired streams are correctly bound to the
  // target GPU when select_target_gpu() picks a non-zero device. Mirrors
  // gpu_pipeline_executor's per-executor pool shape lifted to the multi-GPU
  // scan case — duckdb_scan_executor is shared across all GPUs, so it needs a
  // map rather than a single pool. Without this binding, a wrong-device stream
  // surfaces as a cudaErrorInvalidValue inside cuda_memcpy.
  for (auto* space : _gpu_memory_spaces) {
    auto const dev_id = space->get_device_id();
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{dev_id}};
    _gpu_stream_pools.emplace(dev_id,
                              std::make_unique<cucascade::memory::exclusive_stream_pool>(
                                rmm::cuda_device_id{dev_id}, config.num_threads));
  }
}

duckdb_scan_executor::~duckdb_scan_executor()
{
  {
    std::lock_guard lock(_cache_mutex);
    _cache.clear();
  }
  stop();
}

void duckdb_scan_executor::set_task_creator(sirius::creator::task_creator* task_creator)
{
  _task_creator = task_creator;
}

void duckdb_scan_executor::set_completion_handler(
  sirius::pipeline::completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

bool duckdb_scan_executor::cache_scan_results_for_query(const std::string& query)
{
  if (_cache_level == cache_level::NONE) { return false; }
  // Only track statements that can drive Sirius scan tasks.
  // Transparent execution now runs plain SELECT/WITH SQL, while helper
  // statements like SET and CREATE VIEW should not invalidate the cache.
  if (!is_cacheable_query_text(query)) { return false; }
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
  // Reset affinity map alongside _scan_round_robin at query start. Without
  // this paired reset, a new query's counter values would collide with stale
  // entries from a prior query.
  _scan_round_robin.store(0, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lock(_batch_affinity_mutex);
    _batch_gpu_affinity.clear();
  }

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

int duckdb_scan_executor::select_target_gpu()
{
  if (_gpu_memory_spaces.size() <= 1) {
    return _gpu_memory_spaces.empty() ? 0 : _gpu_memory_spaces[0]->get_device_id();
  }

  // Proportional distribution based on available GPU memory.
  // GPU with 2x free memory gets 2x batches.
  size_t total_available = 0;
  for (auto* space : _gpu_memory_spaces) {
    total_available += space->get_available_memory();
  }

  if (total_available == 0) {
    // All GPUs full, round-robin fallback
    auto fallback_counter = _scan_round_robin.fetch_add(1);
    auto idx              = fallback_counter % _gpu_memory_spaces.size();
    auto device_id        = _gpu_memory_spaces[idx]->get_device_id();
    // Record affinity in the fallback branch too — a stale entry from this
    // path would otherwise cause a disjointedness assert to pass by accident
    // (counter not in map → not counted in intersection).
    {
      std::lock_guard<std::mutex> lock(_batch_affinity_mutex);
      _batch_gpu_affinity[fallback_counter] = device_id;
    }
    return device_id;
  }

  // Weighted selection: use round-robin counter as deterministic distribution
  // seed. Stride the counter by ~min(avail)/num_gpus so the cumulative walk
  // actually rotates between GPUs on consecutive calls — a raw `counter %
  // total_available` stays inside the first GPU's range for billions of calls
  // when GPUs have comparable free memory. Proportional weighting is preserved:
  // a GPU with 2x the memory of the smallest still wins 2x the slots per cycle.
  size_t min_available = std::numeric_limits<size_t>::max();
  for (auto* space : _gpu_memory_spaces) {
    min_available = std::min(min_available, space->get_available_memory());
  }
  size_t stride     = std::max<size_t>(1, min_available / _gpu_memory_spaces.size());
  auto counter      = _scan_round_robin.fetch_add(1);
  size_t target     = (counter * stride) % total_available;
  size_t cumulative = 0;
  for (auto* space : _gpu_memory_spaces) {
    cumulative += space->get_available_memory();
    if (target < cumulative) {
      SIRIUS_LOG_DEBUG("Scan executor: distributing scan batch to GPU {} (available: {} bytes)",
                       space->get_device_id(),
                       space->get_available_memory());
      // Record batch→GPU affinity and emit the audit log atomically. The
      // counter doubles as batch_id=K in the log, so tests can cross-reference
      // the log with this map. Log prefix "[mgpu-audit] scan_batch assigned
      // to GPU N" is load-bearing — verification greps depend on it.
      {
        std::lock_guard<std::mutex> lock(_batch_affinity_mutex);
        _batch_gpu_affinity[counter] = space->get_device_id();
      }
      SIRIUS_LOG_INFO(
        "[mgpu-audit] scan_batch assigned to GPU {} batch_id={} (available: {} bytes)",
        space->get_device_id(),
        counter,
        space->get_available_memory());
      return space->get_device_id();
    }
  }
  return _gpu_memory_spaces.back()->get_device_id();
}

std::unique_ptr<op::operator_data> duckdb_scan_executor::get_scan_output(
  pipeline::sirius_pipeline_itask* task, rmm::cuda_stream_view stream)
{
  bool is_duckdb_scan  = dynamic_cast<duckdb_scan_task*>(task) != nullptr;
  bool is_parquet_scan = !is_duckdb_scan and dynamic_cast<parquet_scan_task*>(task) != nullptr;
  bool is_cpu_source =
    !is_duckdb_scan && !is_parquet_scan && dynamic_cast<cpu_source_task*>(task) != nullptr;

  // CPU source tasks always compute directly (small data, no caching)
  if (is_cpu_source) { return task->compute_task(stream); }

  auto clone_batches = [&](const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
                           rmm::cuda_stream_view stream) {
    if (_cache_level == cache_level::TABLE_GPU) { return batches; }
    std::vector<std::shared_ptr<cucascade::data_batch>> cloned_batches;
    cloned_batches.reserve(batches.size());
    if (is_duckdb_scan) {
      for (auto& batch : batches) {
        // clone() lives on the accessors (not on data_batch) — take a scoped
        // read-only accessor (concurrent shared locks are permitted) and
        // clone through it.
        auto ro = batch->get_read_only();
        cloned_batches.push_back(ro->clone(get_next_batch_id(), stream));
      }
    } else if (is_parquet_scan) {
      for (auto& batch : batches) {
        // Scoped read-only accessor on the source batch for the dynamic_cast
        // probe + shallow_clone() call. The shallow_clone produces a NEW
        // idata_representation owned by a brand-new data_batch — its lifetime
        // is independent of the source accessor, so dropping `ro` at
        // end-of-iteration is safe.
        auto ro         = batch->get_read_only();
        auto* idata_rep = ro->get_data();
        if (auto* host_data = dynamic_cast<cached_host_data_representation*>(idata_rep);
            host_data) {
          cloned_batches.push_back(
            cucascade::data_batch::make(get_next_batch_id(), host_data->shallow_clone()));
        } else if (auto* parquet_rep = dynamic_cast<cached_host_parquet_representation*>(idata_rep);
                   parquet_rep) {
          cloned_batches.push_back(
            cucascade::data_batch::make(get_next_batch_id(), parquet_rep->shallow_clone()));
        } else {
          throw std::runtime_error("Invalid data representation type");
        }
      }
    } else {
      throw std::runtime_error("Invalid task type");
    }
    return cloned_batches;
  };

  if (_cache_level == cache_level::NONE) {
    return task->compute_task(stream);
  } else {
    auto pipe_id = task->get_pipeline_id();
    std::lock_guard<std::mutex> lock(_cache_mutex);
    auto& entry = _cache.at(pipe_id);
    if (!entry) { throw std::runtime_error("Scan results for query not cached"); }
    if (_preload_mode) {
      if (entry->batch_index >= entry->batches.size()) {
        throw std::runtime_error("Scan results for query not cached");
      }
      auto batches = entry->batches[entry->batch_index++];
      return std::make_unique<op::pipelineable_operator_data>(
        clone_batches(std::move(batches), stream));
    } else {
      auto scan_output               = task->compute_task(stream);
      auto& pipelineable_scan_output = dynamic_cast<op::pipelineable_operator_data&>(*scan_output);
      entry->batches.push_back(clone_batches(pipelineable_scan_output.get_data_batches(), stream));
      return scan_output;
    }
  }
}

void duckdb_scan_executor::manager_loop()
{
  while (_running.load()) {
    auto slot = _bounded_pool->reserve();  // block till a thread is available
    if (!slot) {
      SIRIUS_LOG_INFO("DuckDB Scan Executor: pool interrupted, stopping manager loop");
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

    // FIX-01 (v1.2): select the target GPU BEFORE dispatch so both the parquet
    // reservation path AND the stream-acquire path below see the same device.
    // Hoisted from the inner parquet-scan-task block so the dispatch lambda can
    // capture target_gpu_id into its rmm::cuda_set_device_raii guard. For
    // non-parquet scan tasks (duckdb_scan_task, cpu_source_task) this still
    // produces a well-defined target device — cpu_source_task ignores the
    // stream anyway, and duckdb_scan_task runs on the host side until handoff.
    int target_gpu_id = select_target_gpu();

    if (scan_task && scan_task->is<parquet_scan_task>()) {
      auto* parquet_task = dynamic_cast<parquet_scan_task*>(scan_task);

      // Plumb target_gpu_id into the task's LOCAL state so
      // parquet_scan_task::compute_task reads the real device id rather than
      // the nullopt sentinel that routed _datasource to backends.begin()
      // (GPU 0's io_backend) regardless of which GPU the dispatch guard
      // pinned. Local state (per-task) avoids a shared-global race.
      //
      // Must happen BEFORE _bounded_pool->dispatch(...) below: once
      // compute_task runs, _datasource is cached on the task and the
      // preferred lookup is skipped on subsequent calls.
      if (auto* parquet_local_state =
            dynamic_cast<parquet_scan_task_local_state*>(scan_task->local_state())) {
        parquet_local_state->set_preferred_device_id(target_gpu_id);
      } else {
        SIRIUS_LOG_ERROR(
          "duckdb_scan_executor: parquet_scan_task local_state downcast failed "
          "(preferred-device plumbing skipped; compute_task will fall back to "
          "global get_preferred_device_id())");
      }

      if (_cache_level != cache_level::NONE) {
        bool wrap_batch_data     = _cache_level != cache_level::TABLE_GPU;
        bool cache_decoded_table = _cache_level == cache_level::TABLE_HOST;

        // Find the GPU memory space for the target GPU
        cucascade::memory::memory_space* target_gpu_space = _gpu_memory_space;
        for (auto* space : _gpu_memory_spaces) {
          if (space->get_device_id() == target_gpu_id) {
            target_gpu_space = space;
            break;
          }
        }
        parquet_task->set_materialized_columns(
          wrap_batch_data, cache_decoded_table, target_gpu_space);
      }
      auto reservation_info = scan_task->get_estimated_reservation_size_info();
      auto reservation      = _mem_mgr->request_reservation(
        cucascade::memory::any_memory_space_in_tier_with_preference{
          cucascade::memory::Tier::HOST, static_cast<size_t>(target_gpu_id)},
        reservation_info.reservation_size);
      if (!reservation) {
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to acquire host memory reservation");
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to acquire host memory reservation");
        break;
      }
      if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(
            scan_task->local_state())) {
        local_state->set_reservation(std::move(reservation), reservation_info);
      } else {
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to cast local state for task");
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to cast local state for task");
        break;
      }
    } else if (scan_task && scan_task->is<duckdb_scan_task>()) {
      auto reservation_info = scan_task->get_estimated_reservation_size_info();
      if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(
            scan_task->local_state())) {
        local_state->set_reservation(nullptr, reservation_info);
      } else {
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to cast local state for duckdb_scan_task");
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to cast local state for duckdb_scan_task");
        break;
      }
    } else if (scan_task && scan_task->is<cpu_source_task>()) {
      auto reservation_info = scan_task->get_estimated_reservation_size_info();
      auto reservation      = _mem_mgr->request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST},
        reservation_info.reservation_size);
      if (!reservation) {
        SIRIUS_LOG_ERROR(
          "DuckDB Scan Executor: Failed to acquire host memory reservation for CPU source");
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to acquire host memory reservation for CPU source");
        break;
      }
      if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(
            scan_task->local_state())) {
        local_state->set_reservation(std::move(reservation), reservation_info);
      } else {
        _completion_handler->report_error(
          "DuckDB Scan Executor: Failed to cast local state for cpu_source_task");
        SIRIUS_LOG_ERROR("DuckDB Scan Executor: Failed to cast local state for cpu_source_task");
        break;
      }
    }

    // FIX-01 (v1.2): acquire exec_stream from the target GPU's pool (not from
    // the former single GPU-0-bound pool). This ensures the scan task's H2D
    // copies + cudf::io::read_parquet allocations run on a stream bound to
    // target_gpu_id — eliminates the cudaErrorInvalidValue reported at
    // cuda_memcpy.cu from v1.1 E2E verification.
    auto pool_iter = _gpu_stream_pools.find(target_gpu_id);
    if (pool_iter == _gpu_stream_pools.end()) {
      SIRIUS_LOG_ERROR(
        "duckdb_scan_executor: no stream pool for GPU {} (FIX-01 invariant violated)",
        target_gpu_id);
      if (_completion_handler) {
        _completion_handler->report_error(
          "duckdb_scan_executor: missing stream pool for target GPU " +
          std::to_string(target_gpu_id));
      }
      break;
    }
    // acquire_stream() may lazily create a new rmm::cuda_stream on GROW
    // policy; wrap the call in a device guard so the new stream binds to
    // target_gpu_id.
    cucascade::memory::borrowed_stream exc_stream = [&] {
      rmm::cuda_set_device_raii acquire_guard{rmm::cuda_device_id{target_gpu_id}};
      return pool_iter->second->acquire_stream(
        cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
    }();

    _bounded_pool->dispatch(
      std::move(slot),
      [this,
       stream        = std::move(exc_stream),
       t             = std::move(task),
       scan_task     = std::move(scan_task),
       target_gpu_id = target_gpu_id]() mutable {
        // _bounded_pool workers are GPU-agnostic; cudaSetDevice is
        // thread-local. Pin this worker to target_gpu_id BEFORE any cudf/RMM
        // call so the (stream, current_device) pair stays consistent through
        // the entire scan task lifetime.
        rmm::cuda_set_device_raii dispatch_guard{rmm::cuda_device_id{target_gpu_id}};
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
        } catch (std::exception const& e) {
          SIRIUS_LOG_ERROR("DuckDB Scan Executor: Exception during scan task execution: {}",
                           e.what());
          if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
        } catch (...) {
          SIRIUS_LOG_ERROR("DuckDB Scan Executor: Unknown exception during scan task execution");
          if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
        }
      });
  }
}

}  // namespace sirius::op::scan
