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

#include "creator/task_creator.hpp"

#include "log/logging.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_scheduler.hpp"
#include "planner/query.hpp"
#include "sirius_context.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/parallel/thread_context.hpp>

#include <algorithm>
#include <optional>
#include <unordered_map>

namespace sirius::creator {

//------------------------------------------------------------------------------
// task_creator
//------------------------------------------------------------------------------

task_creator::task_creator(exec::thread_pool_config config,
                           sirius::memory::sirius_memory_reservation_manager& mem_res_mgr,
                           const cucascade::memory::system_topology_info* sys_topology)
  : _running(false), _config(config), _mem_res_mgr(mem_res_mgr), _sys_topology(sys_topology)
{
  // Normalize numa_node=-1 (Linux convention for non-NUMA / single-NUMA
  // hosts) to 0 so it matches the host memory space, which is built with
  // numa_id=0 on those hosts. Without normalization, host-sourced tasks on
  // single-NUMA boxes fall through to the default GPU.
  //
  // Record every GPU under its NUMA key (not just the first) so the
  // round-robin walk spreads work across all GPUs sharing a NUMA node.
  if (_sys_topology) {
    for (size_t i = 0; i < _sys_topology->gpus.size(); ++i) {
      auto raw_numa       = _sys_topology->gpus[i].numa_node;
      int normalized_numa = (raw_numa < 0) ? 0 : raw_numa;
      _numa_to_gpu[normalized_numa].push_back(static_cast<int>(_sys_topology->gpus[i].id));
    }
  }
}

task_creator::~task_creator() { stop(); }

void task_creator::set_client_context(::duckdb::ClientContext& client_context)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _client_context = std::addressof(client_context);
  _thread_context = std::make_unique<duckdb::ThreadContext>(client_context);
  _execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_context, *_thread_context, nullptr);
}

void task_creator::set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _task_scheduler = &task_scheduler;
}

void task_creator::prepare_for_query(const sirius::planner::query& query)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);

  _gpu_operator_global_state_map.clear();

  const auto& pipelines = query.get_pipelines();
  for (const auto& pipeline : pipelines) {
    pipeline->set_task_creator(this);
    auto source_operator = pipeline->get_source();
    if (source_operator == nullptr) {
      SIRIUS_LOG_WARN("Pipeline has no source operator; skipping task creation for this pipeline.");
      continue;
    }
    size_t operator_id = source_operator->get_operator_id();
    auto gs            = std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline);
    _gpu_operator_global_state_map.emplace(operator_id, std::move(gs));
  }
}

void task_creator::drain_pending_tasks()
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  // Drain any queued task creation requests that haven't been picked up yet
  _task_creation_queue.interrupt();
  _task_creation_queue.drain();
  // Wait for any in-flight task creation lambdas to finish. When called from
  // task_scheduler::drain_after_error(), stop_thread_pool() has already joined
  // the worker pool and reset _bounded_pool to null — in that case the pool's
  // own destructor already drained in-flight work, so this wait_all is
  // redundant. Dereferencing the null pointer here throws std::system_error
  // (EPERM) from the pthread_mutex call on garbage memory, surfacing to the
  // caller as "Operation not permitted" and breaking otherwise-successful
  // multi-file SF1000 scans.
  if (_bounded_pool) { _bounded_pool->wait_all(); }
  _task_creation_queue.reactivate();
}

void task_creator::reset(bool /*keep_parquet_metadata*/)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _gpu_operator_global_state_map.clear();
  _thread_context.reset();
  _execution_context.reset();
}

next_task_target task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node, std::optional<std::size_t> downstream_request)
{
  if (node == nullptr) { return {}; }

  auto hint = node->get_next_task_hint(downstream_request);

  if (hint.has_value() && hint.value().hint == op::TaskCreationHint::READY) {
    if (hint.value().producer == nullptr) {
      throw std::runtime_error(
        "During get_operator_for_next_task Producer is nullptr for operator " + node->get_name());
    }
    return {hint.value().producer, hint.value().upto_n_task_requested};
  } else if (hint.has_value() &&
             hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    auto* producer = hint.value().producer;
    // A request of zero tasks means "don't create anything for me right now"
    // (e.g. hash_join while the table is being built). Don't recurse upstream.
    if (hint.value().upto_n_task_requested == 0) { return {}; }
    // Forward the current operator's request upward. Each upstream operator's combine rule
    // decides whether to promote its local default based on this value and its own relation.
    return get_operator_for_next_task(producer, hint.value().upto_n_task_requested);
  }
  return {};
}

void task_creator::stop()
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _task_creation_queue.interrupt();
  do_stop_thread_pool();
}

void task_creator::start_thread_pool()
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  // Re-arm the request queue. stop_thread_pool() calls
  // _task_creation_queue.interrupt() so the manager's pop() unblocks; without
  // a paired reactivate() here, subsequent schedule() pushes silently no-op
  // and the next query's manager_loop sees an empty/inactive queue forever.
  _task_creation_queue.reactivate();
  _bounded_pool = std::make_unique<exec::bounded_thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
  _manager_thread = std::thread(&task_creator::manager_loop, this);
}

void task_creator::do_stop_thread_pool()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _bounded_pool->interrupt();
  _task_creation_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();
  _bounded_pool->stop();
  _bounded_pool.reset();
}

void task_creator::stop_thread_pool()
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  do_stop_thread_pool();
}

void task_creator::schedule(op::sirius_physical_operator* node)
{
  auto request  = std::make_unique<task_creation_request>();
  request->node = node;
  _task_creation_queue.push(std::move(request));
}

void task_creator::manager_loop()
{
  while (_running.load()) {
    auto slot = _bounded_pool->reserve();  // block till a thread is available
    if (!slot) {
      SIRIUS_LOG_INFO("Task Creator: pool interrupted, stopping manager loop");
      break;
    }
    auto request = _task_creation_queue.pop();
    if (!request) {
      // This is likely because the task creator was interrupted and the queue was drained
      continue;
    }

    auto node = request->node;
    if (node == nullptr) { continue; }

    auto target = get_operator_for_next_task(node);
    node        = target.node;

    if (node == nullptr) { continue; }

    std::size_t upto_n_task_requested = target.upto_n_task_requested;

    // Dispatch the task creation work to the pool
    _bounded_pool->dispatch(std::move(slot), [this, node, upto_n_task_requested]() mutable {
      try {
        // Get what we need to create the task
        auto pipeline = node->get_pipeline();
        std::vector<cucascade::shared_data_repository*> destination_data_repositories;

        for (const auto& port_info : pipeline->get_next_ports_after_sink()) {
          destination_data_repositories.push_back(
            port_info.next_operator->get_port(port_info.next_operator_port_name)->repo);
        }

        // Create tasks until either all ports are empty or we hit the cap requested by the
        // operator hint. upto_n_task_requested is the maximum number of tasks to create in this
        // dispatch; task_creation_hint::ALL_TASKS means drain.
        std::size_t tasks_created = 0;
        while (tasks_created < upto_n_task_requested && !node->all_ports_empty()) {
          auto task_lock  = pipeline->get_task_creation_lock();
          auto input_data = node->get_next_task_input_data();
          auto* pipelineable_input =
            dynamic_cast<op::pipelineable_operator_data*>(input_data.get());
          if (!input_data ||
              (pipelineable_input && pipelineable_input->get_data_batches().empty())) {
            // no data to create task for
            break;
          }

          size_t operator_id                  = node->get_operator_id();
          auto gpu_pipeline_task_global_state = _gpu_operator_global_state_map.at(operator_id);
          auto local_state =
            std::make_unique<pipeline::gpu_pipeline_task_local_state>(std::move(input_data));

          // pipelineable_input remains valid here: the cast happened before
          // the move into local_state, and unique_ptr move transfers
          // ownership without relocating the object.
          {
            std::optional<int> preferred_device_id;
            // Operating-data preference (highest priority): the scan manager
            // round-robins fresh-read scan splits across the available GPUs and
            // stamps the chosen device onto the split's operating data. Honor it
            // first so each split's task lands on its assigned GPU; the locality
            // heuristics below only run when no upstream preference was set.
            if (local_state->_input_data) {
              preferred_device_id = local_state->_input_data->get_preferred_device_id();
            }
            // Partition affinity: if the input is tagged with a partition
            // index, pin the task to partition_idx % num_gpus.
            // Partition-based operators (hash_join, grouped_aggregate_merge,
            // …) use cuco hash tables under the hood, and cuco tables must
            // live on a single device — a stream bound to GPU A touching a
            // counter built under GPU B trips cudaErrorInvalidValue at
            // counter_storage.cuh. Routing on partition_idx keeps every
            // task of a given partition on one GPU while still spreading
            // partitions across GPUs.
            if (auto* partitioned =
                  dynamic_cast<op::partitioned_operator_data*>(pipelineable_input);
                !preferred_device_id.has_value() && partitioned && _sys_topology &&
                !_sys_topology->gpus.empty()) {
              auto n_gpus         = _sys_topology->gpus.size();
              auto idx            = partitioned->get_partition_idx() % n_gpus;
              preferred_device_id = static_cast<int>(_sys_topology->gpus[idx].id);
            }
            if (!preferred_device_id.has_value() && pipelineable_input &&
                !pipelineable_input->get_data_batches().empty()) {
              std::unordered_map<int, size_t> gpu_bytes;
              std::unordered_map<int, size_t> host_bytes;
              for (const auto& batch : pipelineable_input->get_data_batches()) {
                if (!batch) { continue; }
                auto ro     = batch->to_read_only();
                auto* space = ro.get_memory_space();
                if (!space || !ro.get_data()) { continue; }
                auto size = ro.get_data()->get_size_in_bytes();
                if (space->get_tier() == cucascade::memory::Tier::GPU) {
                  gpu_bytes[space->get_device_id()] += size;
                } else if (space->get_tier() == cucascade::memory::Tier::HOST) {
                  // Normalize numa_id=-1 (non-NUMA / single-NUMA hosts, per
                  // the Linux /sys/bus/pci/devices/*/numa_node convention)
                  // to 0 so the NUMA-affinity lookup matches the normalized
                  // `_numa_to_gpu` map key. Without this, host_bytes[-1]
                  // never hits `_numa_to_gpu[0]`, preferred_device_id stays
                  // nullopt, and every host-sourced pipeline task falls
                  // back to `_gpu_executors.begin()->first`.
                  int host_key = space->get_device_id();
                  if (host_key < 0) host_key = 0;
                  host_bytes[host_key] += size;
                }
              }
              if (!gpu_bytes.empty()) {
                // Data-locality: route to GPU with most data by bytes
                preferred_device_id =
                  std::max_element(gpu_bytes.begin(),
                                   gpu_bytes.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; })
                    ->first;
              } else if (!host_bytes.empty() && !_numa_to_gpu.empty()) {
                // NUMA-affinity: no GPU data, route to a GPU on the same
                // NUMA as the host data. When that NUMA hosts multiple
                // GPUs, pick round-robin across them — pinning every
                // host-sourced pipeline task to a single GPU defeats
                // multi-GPU speedup.
                auto top_host =
                  std::max_element(host_bytes.begin(),
                                   host_bytes.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; })
                    ->first;
                auto it = _numa_to_gpu.find(top_host);
                if (it != _numa_to_gpu.end() && !it->second.empty()) {
                  auto idx            = _numa_to_gpu_rr.fetch_add(1) % it->second.size();
                  preferred_device_id = it->second[idx];
                }
              }
              SIRIUS_LOG_DEBUG(
                "Task Creator: locality score gpu_sources={} host_sources={} preferred_device={}",
                gpu_bytes.size(),
                host_bytes.size(),
                preferred_device_id.value_or(-1));
            }
            // Cached-scan locality: scan_operator_with_pinned_table_input is
            // NOT a pipelineable_operator_data (see
            // sirius_gpu_scan_operator_data.hpp), so the data-locality block
            // above skipped it wholesale. Without this branch, every
            // pinned-table scan task gets dispatched round-robin by the
            // scheduler and triggers a peer DMA or host staging when the
            // consumer GPU differs from the chunk's home GPU. The pinned
            // chunk's GPU residency is preserved on the batch
            // (cached_parquet_gpu_ingestible pins each chunk_memory_space
            // into the gpu_table_representation), so we just read it here.
            if (!preferred_device_id.has_value()) {
              if (auto* cached =
                    dynamic_cast<op::scan::scan_operator_input*>(local_state->_input_data.get())) {
                if (cached->is_resident()) {
                  auto ro     = cached->get_cached_batch()->to_read_only();
                  auto* space = ro.get_memory_space();
                  if (space) {
                    if (space->get_tier() == cucascade::memory::Tier::GPU) {
                      preferred_device_id = space->get_device_id();
                    } else if (space->get_tier() == cucascade::memory::Tier::HOST &&
                               !_numa_to_gpu.empty()) {
                      // tier='host' pinned chunks carry a NUMA-local host
                      // memory_space; map back through _numa_to_gpu to pick
                      // a GPU on the same NUMA. Normalize numa_id=-1 to 0
                      // to match the convention used by the pipelineable
                      // locality block above.
                      int host_key = space->get_device_id();
                      if (host_key < 0) host_key = 0;
                      auto it = _numa_to_gpu.find(host_key);
                      if (it != _numa_to_gpu.end() && !it->second.empty()) {
                        auto idx            = _numa_to_gpu_rr.fetch_add(1) % it->second.size();
                        preferred_device_id = it->second[idx];
                      }
                    }
                    SIRIUS_LOG_DEBUG(
                      "Task Creator: cached-scan locality tier={} device_id={} "
                      "preferred_device={}",
                      static_cast<int>(space->get_tier()),
                      space->get_device_id(),
                      preferred_device_id.value_or(-1));
                  }
                }
              }
            }
            if (preferred_device_id.has_value()) {
              local_state->set_preferred_device_id(preferred_device_id.value());
            }
          }

          auto task_id = get_next_task_id();
          auto task    = std::make_unique<pipeline::gpu_pipeline_task>(task_id,
                                                                    destination_data_repositories,
                                                                    std::move(local_state),
                                                                    gpu_pipeline_task_global_state);
          task_lock.unlock();
          _task_scheduler->schedule(std::move(task));
          ++tasks_created;
        }
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("Task Creator: Exception during task creation: {}", e.what());
        _task_scheduler->terminate_query(std::current_exception());
        stop();
      }
    });
  }
}

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius::creator
