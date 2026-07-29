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
    _task_creation_queue([](const task_creation_request& request) -> exec::index_keys {
      // The request carries its own keys: they are resolved at schedule() time, where the
      // node's pipeline (and therefore its query and priority) is unambiguously alive. The
      // query key is what makes drain(query_index{...}) able to drop one query's pending
      // requests and leave every other query's in place.
      return exec::index_keys{
        request.priority,
        request.node != nullptr ? request.node->type : op::SiriusPhysicalOperatorType::INVALID,
        static_cast<exec::query_key>(sirius::value_of(request.query_id)),
        request.device_id};
    }),
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

void task_creator::query_task_state::enter_in_flight()
{
  std::lock_guard<std::mutex> lock(in_flight_mutex);
  ++in_flight;
}

void task_creator::query_task_state::leave_in_flight()
{
  {
    std::lock_guard<std::mutex> lock(in_flight_mutex);
    --in_flight;
    if (in_flight != 0) { return; }
  }
  in_flight_cv.notify_all();
}

void task_creator::query_task_state::wait_for_in_flight()
{
  std::unique_lock<std::mutex> lock(in_flight_mutex);
  in_flight_cv.wait(lock, [this] { return in_flight == 0; });
}

std::shared_ptr<task_creator::query_task_state> task_creator::get_query_state(
  sirius::query_id_t query_id) const
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  auto it = _query_states.find(query_id);
  return it == _query_states.end() ? nullptr : it->second;
}

void task_creator::set_client_context(sirius::query_id_t query_id,
                                      ::duckdb::ClientContext& client_context)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  // The window begins before the plan exists, so this is where the query's entry is created;
  // prepare_for_query then fills in the pipeline global states.
  auto& state = _query_states[query_id];
  if (!state) { state = std::make_shared<query_task_state>(); }
  state->client_context = std::addressof(client_context);
}

void task_creator::set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _task_scheduler = &task_scheduler;
}

void task_creator::prepare_for_query(const sirius::planner::query& query)
{
  const auto query_id = query.query_id();

  auto state = get_query_state(query_id);
  if (!state) {
    throw sirius::internal_exception(
      "task_creator::prepare_for_query: no state registered for query {}; "
      "set_client_context must run first (execution-window begin)",
      query_id);
  }

  const auto& pipelines = query.get_pipelines();

  auto* sirius_ctx =
    state->client_context->registered_state->Get<duckdb::SiriusContext>("sirius_state").get();
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context =
    sirius_ctx->get_telemetry_context();

  auto pipeline_priorities = compute_pipeline_priorities(query);

  // Filled once, here, and never mutated afterwards: task-creation workers read it without a
  // lock, holding only their shared_ptr to this state. Note the map is NOT cleared — other
  // queries own their own entries, and this one is brand new.
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
      // Mirror it onto the pipeline so schedule() can key a request without locking.
      pipeline->set_priority(it->second);
    }
    state->global_states.emplace(operator_id, std::move(gs));
  }

  std::lock_guard<std::mutex> lookahead_lock(state->lookahead_mutex);
  state->lookahead_queue.clear();
  auto scan_operators            = query.get_scan_operators();
  state->index_of_next_lookahead = 0;
  if (!scan_operators.empty()) {
    auto begin = scan_operators.begin() + 1;
    for (auto it = begin; it != scan_operators.end(); ++it) {
      state->lookahead_queue.push_back(*it);
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

  auto options  = planner::build_index_options{.branch_order = planner::build_probe{}};
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

void task_creator::drain_pending_tasks(sirius::query_id_t query_id)
{
  // Drop only THIS query's queued requests. No interrupt()/reactivate(): the queue stays open
  // the whole time, so other queries' producers and consumers are never stalled.
  _task_creation_queue.drain(
    exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});

  auto state = get_query_state(query_id);
  if (!state) { return; }

  // Wait out this query's in-flight creation lambdas — the per-query stand-in for
  // _bounded_pool->wait_all(), which would also wait on every other query's work. Workers
  // decrement on exit (including by exception), so this cannot hang on a throwing lambda.
  state->wait_for_in_flight();

  // Clear lookahead state so any schedule_lookahead() racing with query teardown finds an
  // empty queue and exits cleanly instead of dereferencing operators that are about to die.
  std::lock_guard<std::mutex> lookahead_lock(state->lookahead_mutex);
  state->lookahead_queue.clear();
  state->index_of_next_lookahead = 0;
}

void task_creator::reset(sirius::query_id_t query_id)
{
  // Requests hold raw operator pointers into this query's plan, and in-flight lambdas
  // dereference them, so both must be gone before the caller destroys planner::query.
  drain_pending_tasks(query_id);

  std::shared_ptr<query_task_state> state;
  {
    std::lock_guard<std::mutex> lock(_global_state_mutex);
    auto it = _query_states.find(query_id);
    if (it == _query_states.end()) { return; }
    state = std::move(it->second);
    _query_states.erase(it);
  }
  // `state` is released here, outside the lock. Any worker that resolved it before the erase
  // still holds its own shared_ptr and finishes safely against a map that no longer lists it.
}

void task_creator::reset_all()
{
  std::vector<sirius::query_id_t> query_ids;
  {
    std::lock_guard<std::mutex> lock(_global_state_mutex);
    query_ids.reserve(_query_states.size());
    for (const auto& [query_id, state] : _query_states) {
      query_ids.push_back(query_id);
    }
  }
  for (const auto query_id : query_ids) {
    reset(query_id);
  }
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node)
{
  if (node == nullptr) { return nullptr; }

  auto hint = node->get_next_task_hint();
  if (!hint.has_value()) { return nullptr; }

  if (hint.value().hint == op::TaskCreationHint::READY) {
    if (hint.value().producer == nullptr) {
      throw std::runtime_error(
        "During get_operator_for_next_task Producer is nullptr for operator " + node->get_name());
    }
    // WSM TODO: how do we handle other ports that are not default?
    return hint.value().producer;
  } else if (hint.value().hint == op::TaskCreationHint::WAITING_FOR_INPUT_DATA) {
    return get_operator_for_next_task(hint.value().producer);
  }
  return nullptr;
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

namespace {

//! The query and scheduling priority of the pipeline `node` belongs to.
//! An unplaced operator (no pipeline yet) yields query 0 / priority 0; such a request is
//! droppable and will simply find no state when the worker resolves it.
std::pair<sirius::query_id_t, exec::queue_priority> request_keys_for(
  const op::sirius_physical_operator* node)
{
  if (node == nullptr) { return {sirius::make_query_id(0), 0}; }
  const auto pipe = node->get_pipeline();
  if (!pipe) { return {sirius::make_query_id(0), 0}; }
  return {pipe->get_query_id(), pipe->get_priority()};
}

}  // namespace

void task_creator::schedule(op::sirius_physical_operator* node)
{
  const auto [query_id, priority] = request_keys_for(node);
  auto request                    = std::make_unique<task_creation_request>();
  request->node                   = node;
  request->query_id               = query_id;
  request->priority               = priority;
  _task_creation_queue.push(std::move(request));
}

void task_creator::schedule(op::sirius_physical_operator* node, sirius::query_id_t query_id)
{
  const auto [_, priority] = request_keys_for(node);
  auto request             = std::make_unique<task_creation_request>();
  request->node            = node;
  request->query_id        = query_id;
  request->priority        = priority;
  _task_creation_queue.push(std::move(request));
}

void task_creator::schedule_lookahead(sirius::query_id_t query_id,
                                      std::optional<int> device_id_hint)
{
  if (_config.strategy != request_type::lookahead) { return; }
  auto state = get_query_state(query_id);
  if (!state) { return; }

  std::lock_guard lock(state->lookahead_mutex);
  for (; state->index_of_next_lookahead < state->lookahead_queue.size();
       ++state->index_of_next_lookahead) {
    auto* node = state->lookahead_queue[state->index_of_next_lookahead];
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
      const auto [_, priority] = request_keys_for(node);
      auto request             = std::make_unique<task_creation_request>();
      request->node            = node;
      request->type            = request_type::lookahead;
      request->query_id        = query_id;
      request->priority        = priority;
      request->device_id       = device_id_hint.value_or(exec::no_preferred_device);
      _task_creation_queue.push(std::move(request));
      ++state->index_of_next_lookahead;
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

    auto node           = request->node;
    auto request_kind   = request->type;
    const auto query_id = request->query_id;
    if (node == nullptr) { continue; }

    // Resolve the query's state ONCE, here, and hand the shared_ptr to the worker. The worker
    // then reads global_states (immutable after prepare_for_query) with no lock and no risk of
    // the map being rehashed or erased underneath it: its own reference keeps the state alive
    // even if reset(query_id) removes the registry entry mid-flight.
    auto query_state = get_query_state(query_id);
    if (!query_state) {
      // The query was reset while this request sat in the queue (finished, or failed and was
      // cleaned up). Dropping it is the correct outcome — `node` may already be dangling.
      continue;
    }

    node = get_operator_for_next_task(node);

    if (node == nullptr) { continue; }

    // Counted before dispatch so drain_pending_tasks(query_id) cannot observe zero in-flight
    // while this task creation is still queued to run.
    query_state->enter_in_flight();
    // Dispatch the task creation work to the pool
    _bounded_pool->dispatch(
      std::move(slot), [this, node, request_kind, query_state = std::move(query_state)]() mutable {
        // Released on every exit path, including the catch below, so a throwing creation can
        // never strand drain_pending_tasks() waiting forever.
        struct in_flight_guard {
          const std::shared_ptr<query_task_state>& state;
          ~in_flight_guard() { state->leave_in_flight(); }
        } guard{query_state};
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

            size_t operator_id = node->get_operator_id();
            // Read from this query's own map. Operator ids restart at 0 per query, so the id is
            // only meaningful within the entry; a globally-keyed map would hand back another
            // query's state here.
            auto gs_it = query_state->global_states.find(operator_id);
            if (gs_it == query_state->global_states.end()) { break; }
            auto gpu_pipeline_task_global_state = gs_it->second;
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
                  preferred_device_id = std::max_element(gpu_bytes.begin(),
                                                         gpu_bytes.end(),
                                                         [](const auto& a, const auto& b) {
                                                           return a.second < b.second;
                                                         })
                                          ->first;
                } else if (!host_bytes.empty() && _topology_index) {
                  // NUMA-affinity: no GPU data, route to a GPU on the same
                  // NUMA as the host data. When that NUMA hosts multiple
                  // GPUs, pick round-robin across them — pinning every
                  // host-sourced pipeline task to a single GPU defeats
                  // multi-GPU speedup.
                  auto top_host = std::max_element(host_bytes.begin(),
                                                   host_bytes.end(),
                                                   [](const auto& a, const auto& b) {
                                                     return a.second < b.second;
                                                   })
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
                if (auto* cached = dynamic_cast<op::scan::scan_operator_input*>(
                      local_state->_input_data.get())) {
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
            task_lock.unlock();
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
