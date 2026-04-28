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
#include "op/scan/cpu_source_task.hpp"
#include "op/scan/duckdb_scan_executor.hpp"
#include "op/scan/duckdb_scan_task.hpp"
#include "op/scan/iceberg_scan_task.hpp"
#include "op/scan/parquet_scan_task.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_iceberg_scan.hpp"
#include "op/sirius_physical_parquet_scan.hpp"
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
  // Build NUMA node -> GPU device mapping for HOST data locality routing.
  // numa_node=-1 is the Linux convention for a non-NUMA / single-NUMA host
  // (see /sys/bus/pci/devices/*/numa_node). On those hosts the single host
  // memory space is constructed with numa_id=0, so normalize -1 to 0 here so
  // host_bytes lookups against this map actually find an entry. Without this
  // normalization SCHED-02 never fires on single-NUMA multi-GPU boxes and
  // every host-sourced pipeline task falls through to the default GPU.
  //
  // Record ALL GPUs on each NUMA (not just the first) — SCHED-02 can then
  // round-robin across them, which matters when two or more GPUs share one
  // NUMA node: single-socket boxes, the audit test host, and any GPU whose
  // topology entry reports -1.
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
  _client_context = std::addressof(client_context);
  _thread_context = std::make_unique<duckdb::ThreadContext>(client_context);
  _execution_context =
    std::make_unique<duckdb::ExecutionContext>(client_context, *_thread_context, nullptr);
}

void task_creator::set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler)
{
  _task_scheduler = &task_scheduler;
}

void task_creator::prepare_for_query(const sirius::planner::query& query)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);

  _scan_operator_global_state_map.clear();
  _gpu_operator_global_state_map.clear();

  const auto& pipelines = query.get_pipelines();
  for (const auto& pipeline : pipelines) {
    // Give each pipeline a pointer to this task_creator so that when a pipeline
    // finishes (including via downstream notification), it can schedule output consumers.
    pipeline->set_task_creator(this);

    auto source_operator = pipeline->get_source();
    if (source_operator == nullptr) { continue; }

    size_t operator_id = source_operator->get_operator_id();

    if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      _scan_operator_global_state_map.emplace(
        operator_id,
        std::make_shared<op::scan::duckdb_scan_task_global_state>(
          pipeline,
          *_task_scheduler,
          *_client_context,
          &source_operator->Cast<op::sirius_physical_duckdb_scan>()));
    } else if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::PARQUET_SCAN) {
      auto it = _parquet_scan_operator_global_state_map.find(operator_id);
      if (it != _parquet_scan_operator_global_state_map.end()) {
        it->second->rebind(pipeline, &source_operator->Cast<op::sirius_physical_parquet_scan>());
      } else {
        // Approach C (Phase 5 Plan 04): seed parquet_scan_task_global_state with
        // the per-GPU cucascade io backend map from SiriusContext. The map is
        // captured by copy into the global_state; scan tasks look up the backend
        // for their preferred_device_id in compute_task().
        auto* sirius_ctx =
          _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
        const auto& op_params = sirius_ctx->get_config().get_operator_params();
        auto gpu_io_backends  = sirius_ctx->get_gpu_io_backends();
        _parquet_scan_operator_global_state_map.emplace(
          operator_id,
          std::make_shared<op::scan::parquet_scan_task_global_state>(
            pipeline,
            &source_operator->Cast<op::sirius_physical_parquet_scan>(),
            op_params.scan_task_batch_size,
            std::move(gpu_io_backends)));
      }
    } else if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::CPU_SOURCE) {
      _cpu_source_operator_global_state_map.emplace(
        operator_id,
        std::make_shared<op::scan::cpu_source_task_global_state>(
          pipeline, &source_operator->Cast<op::sirius_physical_cpu_source>()));
    } else if (source_operator->type == ::sirius::op::SiriusPhysicalOperatorType::ICEBERG_SCAN) {
      SIRIUS_LOG_INFO("[task_creator::prepare_for_query] ICEBERG_SCAN operator_id={}", operator_id);
      auto it = _parquet_scan_operator_global_state_map.find(operator_id);
      if (it != _parquet_scan_operator_global_state_map.end()) {
        SIRIUS_LOG_INFO("[task_creator::prepare_for_query] rebind existing state for id={}",
                        operator_id);
        it->second->rebind(pipeline, &source_operator->Cast<op::sirius_physical_iceberg_scan>());
      } else {
        SIRIUS_LOG_INFO("[task_creator::prepare_for_query] creating NEW state for id={}",
                        operator_id);
        // Approach C (Phase 5 Plan 04/05): seed iceberg_scan_task_global_state
        // with the per-GPU cucascade io backend map from SiriusContext. The map
        // is forwarded to the base parquet_scan_task_global_state so that both
        // the data-file footer pre-reads AND build_delete_pipeline() (delete-file
        // reads via Approach A helpers — Plan 05-05) can resolve backends via
        // get_gpu_io_backends().
        auto* sirius_ctx =
          _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
        const auto& op_params = sirius_ctx->get_config().get_operator_params();
        auto gpu_io_backends  = sirius_ctx->get_gpu_io_backends();
        _parquet_scan_operator_global_state_map.emplace(
          operator_id,
          std::make_shared<op::scan::iceberg_scan_task_global_state>(
            pipeline,
            &source_operator->Cast<op::sirius_physical_iceberg_scan>(),
            op_params.scan_task_batch_size,
            std::move(gpu_io_backends)));
      }
    } else {
      auto gs = std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline);
      _gpu_operator_global_state_map.emplace(operator_id, std::move(gs));
    }
  }
  _num_scans_in_plan =
    _scan_operator_global_state_map.size() + _parquet_scan_operator_global_state_map.size();
}

void task_creator::drain_pending_tasks()
{
  // Drain any queued task creation requests that haven't been picked up yet
  _task_creation_queue.drain();
  // Wait for any in-flight task creation lambdas to finish
  _bounded_pool->wait_all();
}

void task_creator::reset(bool keep_parquet_metadata)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _scan_operator_global_state_map.clear();
  if (!keep_parquet_metadata) { _parquet_scan_operator_global_state_map.clear(); }
  _gpu_operator_global_state_map.clear();
  _cpu_source_operator_global_state_map.clear();
  _thread_context.reset();
  _execution_context.reset();
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node)
{
  if (node == nullptr) { return nullptr; }

  if (node->type == ::sirius::op::SiriusPhysicalOperatorType::ICEBERG_SCAN) {
    size_t operator_id             = node->get_operator_id();
    auto parquet_task_global_state = _parquet_scan_operator_global_state_map.at(operator_id);
    if (parquet_task_global_state->has_more_partitions()) {
      return node;
    } else {
      return nullptr;
    }
  }
  auto hint = node->get_next_task_hint();

  if (hint.has_value() && hint.value().hint == op::TaskCreationHint::READY) {
    if (hint.value().producer == nullptr) {
      throw std::runtime_error(
        "During get_operator_for_next_task Producer is nullptr for operator " + node->get_name());
    }
    // WSM TODO: how do we handle other ports that are not default?
    return hint.value().producer;
  } else if (hint.has_value() &&
             hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    auto* producer = hint.value().producer;
    // DuckDB scan tasks create their own continuations internally, so the
    // task creator should never schedule additional scans from downstream.
    // (Parquet scans are fine — they use partition indices that self-limit.)
    if (producer != nullptr && producer->type == op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
      auto& global_state = _scan_operator_global_state_map.at(producer->get_operator_id());
      if (global_state->is_source_drained() || !global_state->can_create_more_tasks()) {
        return nullptr;
      }
    }
    return get_operator_for_next_task(producer);
  }
  return nullptr;
}

void task_creator::stop()
{
  _task_creation_queue.interrupt();
  stop_thread_pool();
}

void task_creator::start_thread_pool()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _bounded_pool = std::make_unique<exec::bounded_thread_pool>(
    _config.num_threads, _config.thread_name_prefix, _config.cpu_affinity_list);
  _manager_thread = std::thread(&task_creator::manager_loop, this);
}

void task_creator::stop_thread_pool()
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
      SIRIUS_LOG_INFO("Task Creator: task queue interrupted, stopping manager loop");
      break;
    }

    auto node = request->node;
    if (node == nullptr) { continue; }

    node = get_operator_for_next_task(node);

    if (node == nullptr) { continue; }

    // Dispatch the task creation work to the pool
    _bounded_pool->dispatch(std::move(slot), [this, node]() mutable {
      try {
        // Get what we need to create the task
        auto pipeline = node->get_pipeline();
        std::vector<cucascade::shared_data_repository*> destination_data_repositories;
        // special handling for delim joins
        if (pipeline->get_sink()->type ==
            ::sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
          auto& delim_join    = pipeline->get_sink()->Cast<op::sirius_physical_right_delim_join>();
          auto partition_join = delim_join.partition_join;
          auto distinct_op    = delim_join.distinct.get();
          for (auto& next_port : partition_join->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
          for (auto& next_port : distinct_op->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
        } else if (pipeline->get_sink()->type ==
                   ::sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
          auto& delim_join      = pipeline->get_sink()->Cast<op::sirius_physical_left_delim_join>();
          auto distinct_op      = delim_join.distinct.get();
          auto column_data_scan = delim_join.column_data_scan;
          for (auto& next_port : column_data_scan->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
          for (auto& next_port : distinct_op->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
        } else {
          for (auto& next_port : pipeline->get_sink()->get_next_port_after_sink()) {
            destination_data_repositories.push_back(
              next_port.next_operator->get_port(next_port.next_operator_port_name)->repo);
          }
        }
        // scheduling scan task
        if (node->type == ::sirius::op::SiriusPhysicalOperatorType::DUCKDB_SCAN) {
          // Check to see if you need to create a new global s for this scan operator
          size_t operator_id          = node->get_operator_id();
          auto scan_task_global_state = _scan_operator_global_state_map.at(operator_id);

          const auto& op_params =
            _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state")
              ->get_config()
              .get_operator_params();
          auto scan_task_local_state = std::make_unique<op::scan::duckdb_scan_task_local_state>(
            *scan_task_global_state,
            *_execution_context,
            op_params.scan_task_batch_size,
            op_params.default_scan_task_varchar_size);
          if (destination_data_repositories.empty()) {
            throw std::runtime_error(
              "No destination data repositories provided for scan task creation.");
          }
          auto scan_task = std::make_unique<op::scan::duckdb_scan_task>(
            get_next_task_id(),
            destination_data_repositories[0],  // WSM amin TODO: is this correct? there probably
                                               // needs to be multiple possible destination data
                                               // repositories
            std::move(scan_task_local_state),
            scan_task_global_state);
          pipeline->mark_task_created();  // WSM TODO: this needs to be done atomically
                                          // with the task creation
          _task_scheduler->schedule(std::move(scan_task));
        } else if (node->type == ::sirius::op::SiriusPhysicalOperatorType::ICEBERG_SCAN) {
          size_t operator_id             = node->get_operator_id();
          auto parquet_task_global_state = _parquet_scan_operator_global_state_map.at(operator_id);
          // ICEBERG_SCAN inherits from PARQUET_SCAN; Cast<> is type-checked by enum so use
          // static_cast for iceberg nodes.
          auto* parquet_scan = (node->type == op::SiriusPhysicalOperatorType::ICEBERG_SCAN)
                                 ? static_cast<op::sirius_physical_parquet_scan*>(node)
                                 : &node->Cast<op::sirius_physical_parquet_scan>();
          while (true) {
            pipeline->mark_task_created();
            auto partition = parquet_task_global_state->claim_next_rg_partition();
            if (!partition.has_value()) {
              pipeline->mark_task_completed();
              if (pipeline->is_pipeline_finished()) {
                auto output_consumers = pipeline->get_output_consumers();
                for (auto& output_consumer : output_consumers) {
                  schedule(output_consumer);
                }
              }
              return;
            }
            if (!parquet_task_global_state->has_more_partitions()) {
              parquet_scan->has_more_partitions = false;
            }

            auto parquet_task_local_state =
              std::make_unique<op::scan::parquet_scan_task_local_state>(*parquet_task_global_state,
                                                                        *partition);

            if (destination_data_repositories.empty()) {
              throw std::runtime_error(
                "No destination data repositories provided for parquet scan task creation.");
            }
            auto parquet_task =
              std::make_unique<op::scan::parquet_scan_task>(get_next_task_id(),
                                                            destination_data_repositories[0],
                                                            std::move(parquet_task_local_state),
                                                            parquet_task_global_state);
            _task_scheduler->schedule(std::move(parquet_task));

            // If there is only a single scan in the plan, continue creating scan tasks to create
            // I/O parallelism. Otherwise, let the plan drive the creation of more tasks.
            if (_num_scans_in_plan >= 2) { break; }
          }
        } else if (node->type == ::sirius::op::SiriusPhysicalOperatorType::CPU_SOURCE) {
          SIRIUS_LOG_DEBUG("Task Creator: creating cpu_source_task for operator {}",
                           node->get_name());
          size_t operator_id     = node->get_operator_id();
          auto cpu_source_global = _cpu_source_operator_global_state_map.at(operator_id);

          pipeline->mark_task_created();

          if (destination_data_repositories.empty()) {
            throw std::runtime_error(
              "No destination data repositories provided for cpu source task creation.");
          }

          auto local_state = std::make_unique<op::scan::cpu_source_task_local_state>();
          auto task        = std::make_unique<op::scan::cpu_source_task>(get_next_task_id(),
                                                                  destination_data_repositories[0],
                                                                  std::move(local_state),
                                                                  cpu_source_global,
                                                                  *_client_context);
          SIRIUS_LOG_DEBUG("Task Creator: scheduling cpu_source_task, dest_repos={}",
                           destination_data_repositories.size());
          _task_scheduler->schedule(std::move(task));
        } else {
          // need to exhaust input batches until all ports are empty
          while (!node->all_ports_empty()) {
            // Mark task created BEFORE popping data from ports to prevent a race
            // condition where update_pipeline_status() sees empty ports and matching
            // task counters, prematurely marking the pipeline as finished.
            pipeline->mark_task_created();

            auto input_data = node->get_next_task_input_data();
            auto* pipelineable_input =
              dynamic_cast<op::pipelineable_operator_data*>(input_data.get());
            if (!input_data ||
                (pipelineable_input && pipelineable_input->get_data_batches().empty())) {
              // No data was available (e.g., another thread already consumed it).
              // Balance the counter. mark_task_completed() calls update_pipeline_status()
              // which is correct: if all ports are truly empty and all real tasks have
              // completed, the pipeline should finish.
              pipeline->mark_task_completed();
              if (pipeline->is_pipeline_finished()) {
                auto output_consumers = pipeline->get_output_consumers();
                for (auto& output_consumer : output_consumers) {
                  this->schedule(output_consumer);
                }
              }
              break;
            }

            // Check to see if you need to create a new global state for this operator
            size_t operator_id                  = node->get_operator_id();
            auto gpu_pipeline_task_global_state = _gpu_operator_global_state_map.at(operator_id);

            auto local_state =
              std::make_unique<pipeline::gpu_pipeline_task_local_state>(std::move(input_data));

            // Compute preferred GPU based on data locality (SCHED-01, SCHED-02)
            // pipelineable_input was cast from input_data before the move into local_state;
            // the moved-from unique_ptr transfers ownership but leaves the underlying object
            // at the same address, so the raw pointer remains valid here.
            {
              std::optional<int> preferred_device_id;
              // SCHED-00: if the input is tagged with a partition index, pin the
              // task to partition_idx % num_gpus. Partition-based operators
              // (hash_join, grouped_aggregate_merge, …) use cuco hash tables
              // under the hood, and cuco tables must live on a single device —
              // a stream bound to GPU A touching a counter built under GPU B
              // trips cudaErrorInvalidValue at counter_storage.cuh. Routing on
              // partition_idx keeps every task of a given partition on one GPU
              // while still spreading partitions across GPUs.
              if (auto* partitioned =
                    dynamic_cast<op::partitioned_operator_data*>(pipelineable_input);
                  partitioned && _sys_topology && !_sys_topology->gpus.empty()) {
                auto n_gpus         = _sys_topology->gpus.size();
                auto idx            = partitioned->get_partition_idx() % n_gpus;
                preferred_device_id = static_cast<int>(_sys_topology->gpus[idx].id);
              }
              if (!preferred_device_id.has_value() && pipelineable_input &&
                  !pipelineable_input->get_data_batches().empty()) {
                std::unordered_map<int, size_t> gpu_bytes;
                std::unordered_map<int, size_t> host_bytes;
                for (const auto& batch : pipelineable_input->get_data_batches()) {
                  auto* space = batch->get_memory_space();
                  if (!space || !batch->get_data()) { continue; }
                  auto size = batch->get_data()->get_size_in_bytes();
                  if (space->get_tier() == cucascade::memory::Tier::GPU) {
                    gpu_bytes[space->get_device_id()] += size;
                  } else if (space->get_tier() == cucascade::memory::Tier::HOST) {
                    // Normalize numa_id=-1 (non-NUMA / single-NUMA hosts, per
                    // the Linux /sys/bus/pci/devices/*/numa_node convention) to
                    // 0 so the SCHED-02 lookup matches the normalized
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
                  // SCHED-01: Route to GPU with most data by bytes
                  preferred_device_id = std::max_element(gpu_bytes.begin(),
                                                         gpu_bytes.end(),
                                                         [](const auto& a, const auto& b) {
                                                           return a.second < b.second;
                                                         })
                                          ->first;
                } else if (!host_bytes.empty() && !_numa_to_gpu.empty()) {
                  // SCHED-02: No GPU data, route to a GPU on the same NUMA as
                  // the host data. When that NUMA hosts multiple GPUs, pick
                  // round-robin across them — pinning every host-sourced
                  // pipeline task to a single GPU defeats multi-GPU speedup.
                  auto top_host = std::max_element(host_bytes.begin(),
                                                   host_bytes.end(),
                                                   [](const auto& a, const auto& b) {
                                                     return a.second < b.second;
                                                   })
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
              if (preferred_device_id.has_value()) {
                local_state->set_preferred_device_id(preferred_device_id.value());
              }
            }

            auto task_id = get_next_task_id();
            auto task =
              std::make_unique<pipeline::gpu_pipeline_task>(task_id,
                                                            destination_data_repositories,
                                                            std::move(local_state),
                                                            gpu_pipeline_task_global_state);
            _task_scheduler->schedule(std::move(task));
          }
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
