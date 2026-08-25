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
#include "memory/topology_index.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/task_scheduler.hpp"
#include "planner/query.hpp"
#include "planner/query_index.hpp"
#include "sirius_context.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/parallel/thread_context.hpp>

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace sirius::creator {

//------------------------------------------------------------------------------
// task_creator
//------------------------------------------------------------------------------

task_creator::task_creator(task_creator_config config,
                           sirius::memory::sirius_memory_reservation_manager& mem_res_mgr,
                           std::shared_ptr<const sirius::memory::topology_index> topology_index)
  : _running(false),
    _config(std::move(config)),
    _mem_res_mgr(mem_res_mgr),
    _topology_index(std::move(topology_index))
{
  // NUMA-aware GPU routing (HOST-data locality via gpus_of(numa)) is served by
  // the shared topology_index; no ad-hoc device<->NUMA maps are built here.
  // numa_node -1 ("unknown", per the Linux /sys/bus/pci/devices/*/numa_node
  // convention on non-NUMA / single-NUMA hosts) is the index's grouping key and
  // is queried verbatim at routing time.
  //
  // Materialize the active executor set sorted+deduped (topology_index preserves
  // manager order, not sorted order) so partition affinity below stays inverse
  // to sirius_physical_partition's device->slot mapping.
  if (_topology_index) {
    auto ids        = _topology_index->gpu_ids();
    _active_gpu_ids = std::vector<int>(ids.begin(), ids.end());
    std::sort(_active_gpu_ids.begin(), _active_gpu_ids.end());
    _active_gpu_ids.erase(std::unique(_active_gpu_ids.begin(), _active_gpu_ids.end()),
                          _active_gpu_ids.end());
  }
}

task_creator::~task_creator() { stop(); }

void task_creator::set_active_gpu_ids(std::vector<int> ids, std::size_t full_count)
{
  _active_gpu_ids = std::move(ids);
  _full_gpu_count = full_count;
}

const std::vector<int>& task_creator::get_active_gpu_ids() const noexcept
{
  return _active_gpu_ids;
}

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

void task_creator::notify_pipeline_closure(std::size_t pipeline_id,
                                           std::size_t source_operator_id) noexcept
{
  _query_stage_manager->notify_pipeline_closed(_query_id, pipeline_id, source_operator_id);
}

void task_creator::prepare_for_query(const sirius::planner::query& query)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);

  // Stashed for the stage hooks: task creation happens deep in the manager loop
  // where the query is no longer in scope, and recovering it by unpacking the
  // priority's high bits would tie the hooks to that encoding.
  _query_id = query.query_id();

  _gpu_operator_global_state_map.clear();

  const auto& pipelines = query.get_pipelines();

  auto* sirius_ctx =
    _client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context =
    sirius_ctx->get_telemetry_context();

  auto pipeline_priorities = compute_pipeline_priorities(query);

  for (const auto& pipeline : pipelines) {
    pipeline->set_task_creator(this);
    auto source_operator = pipeline->get_source();
    if (source_operator == nullptr) {
      SIRIUS_LOG_WARN("Pipeline has no source operator; skipping task creation for this pipeline.");
      continue;
    }
    size_t operator_id = source_operator->get_operator_id();
    auto gs =
      std::make_shared<pipeline::gpu_pipeline_task_global_state>(pipeline, telemetry_context);
    if (auto it = pipeline_priorities.find(pipeline.get()); it != pipeline_priorities.end()) {
      gs->set_priority(it->second);
    }
    _gpu_operator_global_state_map.emplace(operator_id, std::move(gs));
  }

  std::lock_guard<std::mutex> lookahead_lock(_lookahead_mutex);
  _lookahead_queue.clear();
  auto scan_operators      = query.get_scan_operators();
  _index_of_next_lookahead = 0;
  if (!scan_operators.empty()) {
    auto begin = scan_operators.begin() + 1;
    for (auto it = begin; it != scan_operators.end(); ++it) {
      _lookahead_queue.push_back(*it);
    }
  }
}

std::unordered_map<const pipeline::sirius_pipeline*, exec::queue_priority>
task_creator::compute_pipeline_priorities(const sirius::planner::query& query) const
{
  // Partition the pipeline DAG into branches (linear chains between branch points) and give each
  // pipeline a scheduling priority. LOWER priority values are dispatched first by the pipeline-
  // level priority queue, so a pipeline's priority ascends with its execution order. The final
  // priorities are compacted to a dense, contiguous 0..N-1 range (N = number of pipelines) so the
  // assignment is easy to read off against the plan. The rules (see query_index for the branch
  // definition):
  //   - Branches are ordered by plan order; an earlier branch is ALWAYS strictly lower (runs
  //     first) than a later one (guaranteed by a per-branch stride larger than any branch length).
  //   - Within a branch, source ranks the head (closest to the scan) lowest; sink reverses it.
  //   - A pipeline shared by several branches (a join/merge endpoint) takes the MIN priority of
  //     the branches that reach it, so it runs as soon as its earliest-needed branch wants it.
  std::unordered_map<const pipeline::sirius_pipeline*, exec::queue_priority> priorities;

  auto options  = planner::build_index_options{.branch_order = planner::barrier_order{}};
  auto index    = planner::query_index::build_index(query, options);
  auto branches = index->get_branches();
  if (branches.empty()) { return priorities; }

  const bool sink_first = _config.priority == priority_order::sink;

  // Stride larger than any branch length keeps cross-branch ordering strictly dominant over the
  // within-branch offset.
  std::size_t max_branch_len = 0;
  for (const auto& chain : branches) {
    max_branch_len = std::max(max_branch_len, chain.size());
  }
  const exec::queue_priority stride = static_cast<exec::queue_priority>(max_branch_len) + 1;

  const std::size_t num_branches = branches.size();
  for (std::size_t b = 0; b < num_branches; ++b) {
    const auto& chain               = branches[b];
    const auto len                  = chain.size();
    const exec::queue_priority base = static_cast<exec::queue_priority>(b) * stride;
    for (std::size_t pos = 0; pos < len; ++pos) {
      // source: head (pos 0) gets the smallest offset so it runs first; sink reverses
      // within-branch.
      const exec::queue_priority within   = sink_first
                                              ? static_cast<exec::queue_priority>(len - 1 - pos)
                                              : static_cast<exec::queue_priority>(pos);
      const exec::queue_priority priority = base + within;
      auto [it, inserted]                 = priorities.try_emplace(chain[pos], priority);
      if (!inserted) { it->second = std::min(it->second, priority); }
    }
  }

  // The strided values above are correct in relative order but sparse (short branches leave gaps).
  // Compact them to a dense 0..N-1 range by ranking the assigned priorities: each branch occupies a
  // disjoint value range and a shared endpoint's min comes from a single branch's range, so every
  // pipeline's raw priority is distinct and the rank is a clean bijection preserving execution
  // order.
  std::vector<exec::queue_priority> sorted;
  sorted.reserve(priorities.size());
  for (const auto& [pipeline, priority] : priorities) {
    sorted.push_back(priority);
  }
  std::sort(sorted.begin(), sorted.end());
  // Inject the query id into the high 32 bits so tasks are ordered by query first, then by the
  // within-query pipeline rank in the low 32 bits. The priority queue picks the LOWEST value first,
  // so an earlier query (smaller id) always runs before a later one, and within a query the dense
  // 0..N-1 rank preserves pipeline execution order. See query_priority_bits() for the masking
  // contract that keeps the packed value non-negative.
  const exec::queue_priority query_bits = sirius::query_priority_bits(query.query_id());
  for (auto& [pipeline, priority] : priorities) {
    const auto rank = std::lower_bound(sorted.begin(), sorted.end(), priority) - sorted.begin();
    priority        = query_bits | static_cast<exec::queue_priority>(rank);
  }
  return priorities;
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
  // Clear lookahead state so any schedule_lookahead() call from the
  // management_eventloop that races with QueryEnd (after query_.reset() destroys
  // operators but before task_creator_->reset() clears the queue) finds an
  // empty queue and exits cleanly instead of dereferencing dangling pointers.
  {
    std::lock_guard<std::mutex> lookahead_lock(_lookahead_mutex);
    _lookahead_queue.clear();
    _index_of_next_lookahead = 0;
  }
  _task_creation_queue.reactivate();
}

void task_creator::reset()
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _gpu_operator_global_state_map.clear();
  _thread_context.reset();
  _execution_context.reset();
  {
    std::lock_guard<std::mutex> lookahead_lock(_lookahead_mutex);
    _lookahead_queue.clear();
    _index_of_next_lookahead = 0;
  }
}

next_task_result task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node,
  std::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& visited_pipelines)
{
  // The only case with nobody to blame: there was no operator to ask.
  if (node == nullptr) { return {nullptr, next_task_state::depleted}; }
  if (auto pipeline = node->get_pipeline()) { visited_pipelines.push_back(std::move(pipeline)); }

  auto hint = node->get_next_task_hint();
  if (!hint.has_value()) { return {node, next_task_state::depleted}; }

  if (hint.value().hint == op::TaskCreationHint::READY) {
    if (hint.value().producer == nullptr) {
      throw std::runtime_error(
        "During get_operator_for_next_task Producer is nullptr for operator " + node->get_name());
    }
    // WSM TODO: how do we handle other ports that are not default?
    return {hint.value().producer, next_task_state::ready};
  }

  if (hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    auto const deeper = get_operator_for_next_task(hint.value().producer, visited_pipelines);
    // A null producer means the chain ran out below us, so this node is the last
    // one that was actually asked -- keep it rather than propagating the null,
    // or the failure loses its subject on the way back up.
    if (deeper.op == nullptr) { return {node, next_task_state::depleted}; }
    return deeper;
  }

  return {node, next_task_state::depleted};
}

void task_creator::stop()
{
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
  _bounded_pool =
    std::make_unique<exec::bounded_thread_pool>(_config.thread_pool.num_threads,
                                                _config.thread_pool.thread_name_prefix,
                                                _config.thread_pool.cpu_affinity_list);
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

void task_creator::schedule_lookahead(std::optional<int> device_id_hint)
{
  if (_config.strategy != request_type::lookahead) { return; }
  std::lock_guard lock(_lookahead_mutex);
  for (; _index_of_next_lookahead < _lookahead_queue.size(); ++_index_of_next_lookahead) {
    auto* node = _lookahead_queue[_index_of_next_lookahead];
    if (node == nullptr) { continue; }
    auto hint = node->get_next_task_hint();
    if (!hint.has_value()) {
      if (!node->get_pipeline()->is_pipeline_finished()) { return; }
      continue;
    }
    if (hint.value().hint == op::TaskCreationHint::READY) {
      SIRIUS_LOG_TRACE("Task Creator: scheduling lookahead for operator {} (id {})",
                       node->get_name(),
                       node->get_operator_id());
      auto request  = std::make_unique<task_creation_request>();
      request->node = node;
      request->type = request_type::lookahead;
      _task_creation_queue.push(std::move(request));
      ++_index_of_next_lookahead;
      return;
    }
  }
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

    auto node         = request->node;
    auto request_kind = request->type;
    if (node == nullptr) { continue; }

    std::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> visited_pipelines;
    auto* requested_node = node;
    auto const next      = get_operator_for_next_task(node, visited_pipelines);

    if (next.state != next_task_state::ready) {
      // Same re-evaluation the creation path does on exit: get_next_task_hint()
      // can have drained ports (hash join's discard sweep) in ANY pipeline the
      // hint walk visited, making it finishable. A visited upstream pipeline
      // whose tasks all completed earlier gets no later mark_task_completed(),
      // so this is its only chance to be marked finished.
      std::sort(visited_pipelines.begin(), visited_pipelines.end());
      visited_pipelines.erase(std::unique(visited_pipelines.begin(), visited_pipelines.end()),
                              visited_pipelines.end());
      for (auto& visited : visited_pipelines) {
        visited->update_pipeline_status(false);
      }

      // Report both ends of the walk: the operator the request named, and the
      // one that actually could not produce.  They differ whenever the walk
      // descended through operators waiting on input, and it is the latter that
      // says where the pipeline is actually stuck.
      //
      // Observation only, so an unnumbered operator is reported as the sentinel
      // rather than throwing out of the manager loop.
      auto const id_of = [](op::sirius_physical_operator const* op) {
        return op != nullptr && op->has_operator_id()
                 ? op->get_operator_id()
                 : op::sirius_physical_operator::invalid_operator_id;
      };
      _query_stage_manager->notify_failed_to_create_task(
        _query_id, id_of(requested_node), id_of(next.op != nullptr ? next.op : requested_node));
      continue;
    }
    node = next.op;

    // Dispatch the task creation work to the pool
    _bounded_pool->dispatch(std::move(slot), [this, node, request_kind]() mutable {
      try {
        // Get what we need to create the task
        auto pipeline = node->get_pipeline();
        std::vector<cucascade::shared_data_repository*> destination_data_repositories;

        for (const auto& port_info : pipeline->get_next_ports_after_sink()) {
          destination_data_repositories.push_back(
            port_info.next_operator->get_port(port_info.next_operator_port_name)->repo);
        }

        while (!node->all_ports_empty()) {
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
                !preferred_device_id.has_value() && partitioned && !_active_gpu_ids.empty()) {
              // Index the active executor set so every task of a partition lands
              // on the same real GPU (required for cuco tables); the physical
              // topology would yield phantom pins when num_gpus < physical count.
              auto idx            = partitioned->get_partition_idx() % _active_gpu_ids.size();
              preferred_device_id = _active_gpu_ids[idx];
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
                  // Key by the host space's NUMA node verbatim; topology_index
                  // groups GPUs under that same key (including the -1 "unknown"
                  // sentinel for non-NUMA / single-NUMA hosts), so gpus_of()
                  // resolves without any normalization.
                  host_bytes[space->get_device_id()] += size;
                }
              }
              if (!gpu_bytes.empty()) {
                // Data-locality: route to GPU with most data by bytes
                preferred_device_id =
                  std::max_element(gpu_bytes.begin(),
                                   gpu_bytes.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; })
                    ->first;
              } else if (!host_bytes.empty() && _topology_index) {
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
                auto gpus = _topology_index->gpus_of(top_host);
                if (!gpus.empty()) {
                  auto idx            = _numa_affinity_rr.fetch_add(1) % gpus.size();
                  preferred_device_id = gpus[idx];
                }
              }
              SIRIUS_LOG_DEBUG(
                "Task Creator: locality score gpu_sources={} host_sources={} preferred_device={}",
                gpu_bytes.size(),
                host_bytes.size(),
                preferred_device_id.value_or(-1));
            }
            // Cached-scan locality: a resident scan_operator_input is
            // NOT a pipelineable_operator_data (see
            // sirius_gpu_scan_operator_data.hpp), so the data-locality block
            // above skipped it wholesale. Without this branch, every
            // pinned-table scan task gets dispatched round-robin by the
            // scheduler and triggers a peer DMA or host staging when the
            // consumer GPU differs from the chunk's home GPU. The pinned
            // chunk's GPU residency is preserved on the batch
            // (the cached provider pins each chunk_memory_space
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
                               _topology_index) {
                      // tier='host' pinned chunks carry a NUMA-local host
                      // memory_space; map back through the topology index to
                      // pick a GPU on the same NUMA. The host space's device id
                      // is the NUMA key verbatim (-1 = "unknown"), matching the
                      // pipelineable locality block above.
                      auto gpus = _topology_index->gpus_of(space->get_device_id());
                      if (!gpus.empty()) {
                        auto idx            = _numa_affinity_rr.fetch_add(1) % gpus.size();
                        preferred_device_id = gpus[idx];
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
            // Confine the task to the admitted subset. Every preference above except the
            // partition pin comes from where data lives rather than from the subset, and the
            // scheduler treats a preference as binding — so an excluded id would be honoured.
            // Clamping a residency-derived one costs the locality it encoded, but honouring
            // it would put the query on a GPU it was not admitted to. An unpreferred task
            // escapes too: the scheduler gives those to whichever executor asks first. Pin
            // those as well, but only on a real subset, since a pin costs the scheduler's
            // freedom to place them wherever frees up first.
            if (!_active_gpu_ids.empty()) {
              bool const names_excluded_device =
                preferred_device_id.has_value() &&
                std::find(_active_gpu_ids.begin(), _active_gpu_ids.end(), *preferred_device_id) ==
                  _active_gpu_ids.end();
              bool const unpinned_on_a_subset =
                !preferred_device_id.has_value() && _active_gpu_ids.size() < _full_gpu_count;
              if (names_excluded_device || unpinned_on_a_subset) {
                auto const idx      = _admission_rr.fetch_add(1) % _active_gpu_ids.size();
                preferred_device_id = _active_gpu_ids[idx];
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
          _query_stage_manager->notify_task_created(
            _query_id, operator_id, node->type, gpu_pipeline_task_global_state->get_priority());
          _task_scheduler->schedule(std::move(task));

          if (request_kind == request_type::lookahead) { break; }
        }
        // Unconditional re-evaluation at every creation exit: with the
        // source-exhaustion finish guard, "last task completed at T1,
        // connector closed at T2>T1" has no later mark_task_completed() to
        // re-check the pipeline — this call, observing the now-exhausted
        // source, is the paired re-evaluation. Without it, normal fast-GPU
        // queries would hang.
        pipeline->update_pipeline_status(false);
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
