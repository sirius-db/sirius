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
#include "sirius/exception.hpp"
#include "sirius_context.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/parallel/thread_context.hpp>

#include <algorithm>
#include <cassert>
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
      // The request carries ALL of its own keys, resolved at schedule() time where the node's
      // pipeline (and therefore its query, priority and type) is unambiguously alive. Nothing
      // here dereferences `node`: this runs inside the queue mutex on every push, and reading a
      // freed operator there would corrupt the index of a queue every query shares. The query key
      // is what makes drain(query_index{...}) able to drop one query's pending requests and leave
      // every other query's in place.
      // No device key: creation requests are only ever pop()'ed in priority
      // order, never popped per-device, so a real device key would only
      // maintain a secondary index nothing reads.
      return exec::index_keys{request.priority,
                              request.operator_type,
                              static_cast<exec::query_key>(sirius::value_of(request.query_id)),
                              exec::no_preferred_device};
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
    auto ids         = _topology_index->gpu_ids();
    _default_gpu_ids = std::vector<int>(ids.begin(), ids.end());
    std::sort(_default_gpu_ids.begin(), _default_gpu_ids.end());
    _default_gpu_ids.erase(std::unique(_default_gpu_ids.begin(), _default_gpu_ids.end()),
                           _default_gpu_ids.end());
  }
}

task_creator::~task_creator() { stop(); }

void task_creator::set_active_gpu_ids(sirius::query_id_t query_id,
                                      std::vector<int> ids,
                                      std::size_t full_count)
{
  // get-or-create rather than get-or-throw: in production set_client_context
  // (execution-window begin) has already registered the entry, but unit tests
  // drive sirius_engine::initialize_internal directly with no window, and
  // admission is a valid first touch — the entry is completed by whichever
  // registration call runs next on the same id.
  auto state            = get_or_create_query_task_global_state(query_id);
  state->active_gpu_ids = std::move(ids);
  state->full_gpu_count = full_count;
}

std::vector<int> task_creator::get_active_gpu_ids(sirius::query_id_t query_id) const
{
  if (auto state = get_query_task_global_state(query_id)) { return state->active_gpu_ids; }
  return _default_gpu_ids;
}

std::shared_ptr<task_creator::query_task_global_state> task_creator::get_query_task_global_state(
  sirius::query_id_t query_id) const
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  auto it = _query_task_global_states.find(query_id);
  return it == _query_task_global_states.end() ? nullptr : it->second;
}

std::shared_ptr<task_creator::query_task_global_state>
task_creator::get_or_create_query_task_global_state(sirius::query_id_t query_id)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  auto& state = _query_task_global_states[query_id];
  if (!state) {
    state = std::make_shared<query_task_global_state>();
    // Seed the admission default: a query that never narrows (no
    // set_active_gpu_ids call) runs on every executor, exactly as before
    // per-query admission existed.
    state->active_gpu_ids = _default_gpu_ids;
    state->full_gpu_count = _default_gpu_ids.size();
  }
  return state;
}

void task_creator::set_client_context(sirius::query_id_t query_id,
                                      ::duckdb::ClientContext& client_context)
{
  // The window begins before the plan exists, so this is where the query's entry is normally
  // created; prepare_for_query then fills in the pipeline global states.
  auto state            = get_or_create_query_task_global_state(query_id);
  state->client_context = std::addressof(client_context);
}

void task_creator::set_task_scheduler(sirius::pipeline::task_scheduler& task_scheduler)
{
  std::lock_guard<std::mutex> lock(_global_state_mutex);
  _task_scheduler = &task_scheduler;
}

void task_creator::prepare_for_query(const sirius::planner::query& query,
                                     std::shared_ptr<pipeline::completion_handler> handler)
{
  const auto query_id = query.query_id();

  auto state = get_query_task_global_state(query_id);
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

  state->completion_handler = handler;

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
    gs->set_completion_handler(handler);
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
  // Inject the query id into the high 32 bits so each query occupies its own contiguous band of
  // priorities: within a query the dense 0..N-1 rank preserves pipeline execution order, and the
  // queues' per-query indexes key off the banding. Cross-query dispatch order is NOT this value
  // order — the queues' fair pops rotate round-robin across query bands so an earlier query
  // cannot starve a later one (F1). See query_priority_bits() for the masking contract that
  // keeps the packed value non-negative.
  const exec::queue_priority query_bits = sirius::query_priority_bits(query.query_id());
  for (auto& [pipeline, priority] : priorities) {
    const auto rank = std::lower_bound(sorted.begin(), sorted.end(), priority) - sorted.begin();
    priority        = query_bits | static_cast<exec::queue_priority>(rank);
  }
  return priorities;
}

void task_creator::drain_pending_tasks(sirius::query_id_t query_id)
{
  auto state = get_query_task_global_state(query_id);

  // Neutralize lookahead FIRST, before the request drain below. schedule_lookahead() runs on
  // the task scheduler's management thread — a producer this drain does not otherwise own — and
  // it dereferences this query's operators and pushes a creation request under only
  // lookahead_mutex. Taking that mutex here, ahead of the drain, orders the race both ways: a
  // walk already in progress blocks this clear, so its push lands BEFORE the drain and is
  // dropped by it (and the plan is still alive during the walk — the caller destroys it only
  // after this function returns); a walk starting after the clear finds the queue empty and is
  // a no-op. With the old order (clear LAST) a racing lookahead could push AFTER the drain had
  // passed: the stale request survived with a raw operator pointer into a plan about to die,
  // and nothing waited for the worker that would eventually dereference it (register D3).
  if (state) {
    std::lock_guard<std::mutex> lookahead_lock(state->lookahead_mutex);
    state->lookahead_queue.clear();
    state->index_of_next_lookahead = 0;
  }

  // Drop only THIS query's queued requests. No interrupt()/reactivate(): the queue stays open
  // the whole time, so other queries' producers and consumers are never stalled.
  _task_creation_queue.drain(
    exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});

  if (!state) { return; }

  // Wait out this query's in-flight creation work. The pool tracks it per query via the slot
  // attached in manager_loop, so this waits on THIS query only — never on a co-tenant's lambda,
  // and never on the manager thread's own idle reservation, which carries no query. Slots are
  // released by RAII on every exit path including an exception, so a throwing creation lambda
  // cannot strand this wait.
  if (_bounded_pool) { _bounded_pool->drain_and_wait(query_id); }
}

void task_creator::reset(sirius::query_id_t query_id)
{
  // Requests hold raw operator pointers into this query's plan, and in-flight lambdas
  // dereference them, so both must be gone before the caller destroys planner::query.
  drain_pending_tasks(query_id);

  std::shared_ptr<query_task_global_state> state;
  {
    std::lock_guard<std::mutex> lock(_global_state_mutex);
    auto it = _query_task_global_states.find(query_id);
    if (it == _query_task_global_states.end()) { return; }
    state = std::move(it->second);
    _query_task_global_states.erase(it);
  }
  // `state` is released here, outside the lock. Any worker that resolved it before the erase
  // still holds its own shared_ptr and finishes safely against a map that no longer lists it.
}

void task_creator::reset_all()
{
  std::vector<sirius::query_id_t> query_ids;
  {
    std::lock_guard<std::mutex> lock(_global_state_mutex);
    query_ids.reserve(_query_task_global_states.size());
    for (const auto& [query_id, state] : _query_task_global_states) {
      query_ids.push_back(query_id);
    }
  }
  for (const auto query_id : query_ids) {
    reset(query_id);
  }
}

op::sirius_physical_operator* task_creator::get_operator_for_next_task(
  op::sirius_physical_operator* node,
  std::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& visited_pipelines)
{
  if (node == nullptr) { return nullptr; }
  if (auto pipeline = node->get_pipeline()) { visited_pipelines.push_back(std::move(pipeline)); }

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
    return get_operator_for_next_task(hint.value().producer, visited_pipelines);
  }
  return nullptr;
}

namespace {
//! Set for the duration of a creation-worker lambda, so stop() can assert it is not being called
//! from inside its own pool (see task_creator::stop).
thread_local bool t_in_creation_worker = false;
}  // namespace

bool task_creator::is_pool_worker_thread() { return t_in_creation_worker; }

void task_creator::stop()
{
  // Calling this from a creation worker is a guaranteed self-deadlock: do_stop_thread_pool()
  // calls _bounded_pool->wait_all(), which blocks until active_ == 0, but the calling worker IS
  // an active slot. If it somehow got past, _bounded_pool.reset() would join the calling thread
  // with itself inside a noexcept function. The rogue call sites that did this are gone; assert
  // so a new one is caught in a debug build rather than hanging CI.
  assert(!is_pool_worker_thread() &&
         "task_creator::stop() must not be called from one of its own pool workers");

  // Takes the same mutex as start/stop_thread_pool. Without it, a stop() racing a
  // start_thread_pool() could reassign _bounded_pool and _manager_thread while the other thread
  // was inside do_stop_thread_pool() joining them.
  std::lock_guard<std::mutex> lock(_pool_lifecycle_mutex);
  _task_creation_queue.interrupt();
  do_stop_thread_pool();
}

void task_creator::start_thread_pool()
{
  std::lock_guard<std::mutex> lock(_pool_lifecycle_mutex);
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
  std::lock_guard<std::mutex> lock(_pool_lifecycle_mutex);
  do_stop_thread_pool();
}

namespace {

//! The query and scheduling priority of the pipeline `node` belongs to.
std::pair<sirius::query_id_t, exec::queue_priority> request_keys_for(
  const op::sirius_physical_operator* node)
{
  if (node == nullptr) {
    throw sirius::internal_exception("task_creator::schedule: null operator");
  }
  const auto pipe = node->get_pipeline();
  if (!pipe) {
    throw sirius::internal_exception(
      "task_creator::schedule: operator {} (id {}) has no pipeline; it was never placed by "
      "planner::query::build_indices",
      node->get_name(),
      // get_operator_id() itself throws on an unnumbered operator; report the sentinel instead so
      // the pipeline-less diagnosis is not masked by an id-assignment one.
      node->has_operator_id() ? std::to_string(node->get_operator_id())
                              : std::string{"unassigned"});
  }
  return {pipe->get_query_id(), pipe->get_priority()};
}

}  // namespace

void task_creator::schedule(op::sirius_physical_operator* node)
{
  const auto [query_id, priority] = request_keys_for(node);
  // Most calls here come from a completion callback (notify_downstream_pipelines, or the GPU
  // executor scheduling a finished task's consumers). If the query is tearing down, `node` points
  // into a plan that is about to be destroyed and a drain has very likely already passed this
  // queue — so refuse rather than enqueue. Previously this was achieved by interrupting the
  // queue, which refused EVERY query's pushes at once.
  if (!accepts_work(query_id)) { return; }
  auto request           = std::make_unique<task_creation_request>();
  request->node          = node;
  request->query_id      = query_id;
  request->priority      = priority;
  request->operator_type = node->type;
  report_if_dropped(_task_creation_queue.push(std::move(request)), query_id);
}

void task_creator::report_if_dropped(bool pushed, sirius::query_id_t query_id) const
{
  if (pushed) { return; }
  // multi_index_priority_queue::push returns false for exactly one reason: the queue is
  // interrupted, i.e. shutting down. For a query the gate still reports as accepting work that is
  // a genuine loss -- the request is destroyed, its pipeline never gets its task, and the query
  // waits on a completion that can never arrive. Silence here is what turned every dropped-work
  // bug in this subsystem into an unexplained hang.
  //
  // For a quiescing/closed query the drop is the documented teardown contract, not a bug.
  if (accepts_work(query_id)) {
    SIRIUS_LOG_ERROR(
      "task_creator: creation request for query {} was DROPPED by an interrupted queue while the "
      "query was still accepting work; that query will not receive the task it was waiting for",
      query_id);
  } else {
    SIRIUS_LOG_DEBUG("task_creator: dropped a creation request for tearing-down query {}",
                     query_id);
  }
}

bool task_creator::accepts_work(sirius::query_id_t query_id) const noexcept
{
  return _query_lifecycle == nullptr || _query_lifecycle->accepts_work(query_id);
}

void task_creator::report_fatal_error(const std::shared_ptr<pipeline::completion_handler>& handler,
                                      std::exception_ptr error)
{
  // Report and return: no stop(). This runs on a creation pool worker (and, via
  // notify_downstream_pipelines, on a GPU executor worker), where stopping the creator would
  // either join the calling thread with itself or tear down task creation for every other
  // in-flight query. Unwinding is the failing query's own business — its future.get() catch in
  // sirius_engine::execute runs drain_after_error(query_id).
  if (_task_scheduler != nullptr) { _task_scheduler->terminate_query(handler, std::move(error)); }
}

void task_creator::report_fatal_error(sirius::query_id_t query_id, std::exception_ptr error)
{
  // A query whose state was already dropped has no handler left to report to; the error has
  // nowhere to go, which is correct — that query is already being torn down.
  auto state = get_query_task_global_state(query_id);
  report_fatal_error(state ? state->completion_handler : nullptr, std::move(error));
}

void task_creator::schedule_lookahead()
{
  if (_config.strategy != request_type::lookahead) { return; }

  // Rotate across the ACCEPTING queries — the F1 round-robin idea applied to warm-up. This
  // used to hard-code the oldest entry, so with two live queries only the first ever received
  // lookahead and every newer query started cold (register D3). The scan starts after the last
  // query a lookahead was scheduled for (wrapping) and, within one call, tries each live query
  // once until one can actually be warmed — so a query with nothing warmable right now (or one
  // that is quiescing/closed per the lifecycle registry, routinely the oldest entry mid-cleanup)
  // cannot pin the rotation and starve the others.
  std::vector<std::pair<sirius::query_id_t, std::shared_ptr<query_task_global_state>>> rotation;
  {
    std::lock_guard<std::mutex> lock(_global_state_mutex);
    if (_query_task_global_states.empty()) { return; }
    rotation.reserve(_query_task_global_states.size());
    auto start = _has_last_lookahead_query
                   ? _query_task_global_states.upper_bound(_last_lookahead_query)
                   : _query_task_global_states.begin();
    if (start == _query_task_global_states.end()) { start = _query_task_global_states.begin(); }
    for (auto it = start; it != _query_task_global_states.end(); ++it) {
      rotation.emplace_back(it->first, it->second);
    }
    for (auto it = _query_task_global_states.begin(); it != start; ++it) {
      rotation.emplace_back(it->first, it->second);
    }
  }

  for (auto& [query_id, state] : rotation) {
    if (!state || !accepts_work(query_id)) { continue; }
    if (try_schedule_lookahead_for(query_id, *state)) {
      std::lock_guard<std::mutex> lock(_global_state_mutex);
      _last_lookahead_query     = query_id;
      _has_last_lookahead_query = true;
      return;
    }
  }
}

bool task_creator::try_schedule_lookahead_for(sirius::query_id_t query_id,
                                              query_task_global_state& state)
{
  // Teardown safety of the operator derefs below: see the declaration comment — the
  // lookahead_mutex plus drain_pending_tasks() clearing the lookahead queue under it BEFORE its
  // request drain make a lookahead racing this query's teardown a no-op, never a UAF.
  std::lock_guard lock(state.lookahead_mutex);
  for (; state.index_of_next_lookahead < state.lookahead_queue.size();
       ++state.index_of_next_lookahead) {
    auto* node = state.lookahead_queue[state.index_of_next_lookahead];
    if (node == nullptr) { continue; }
    auto hint = node->get_next_task_hint();
    if (!hint.has_value()) {
      // The next scan is not warmable yet and its pipeline is still live: this query has no
      // lookahead to offer right now — the caller rotates on to the next query.
      if (!node->get_pipeline()->is_pipeline_finished()) { return false; }
      continue;
    }
    if (hint.value().hint == op::TaskCreationHint::READY) {
      SIRIUS_LOG_TRACE("Task Creator: scheduling lookahead for operator {} (id {}) of query {}",
                       node->get_name(),
                       node->get_operator_id(),
                       query_id);
      const auto [_, priority] = request_keys_for(node);
      auto request             = std::make_unique<task_creation_request>();
      request->node            = node;
      request->type            = request_type::lookahead;
      request->query_id        = query_id;
      request->priority        = priority;
      request->operator_type   = node->type;
      report_if_dropped(_task_creation_queue.push(std::move(request)), query_id);
      ++state.index_of_next_lookahead;
      return true;
    }
  }
  return false;
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
    auto query_state = get_query_task_global_state(query_id);
    if (!query_state) {
      // The query was reset while this request sat in the queue (finished, or failed and was
      // cleaned up). Dropping it is the correct outcome — `node` may already be dangling.
      continue;
    }

    // Attribute the slot to this query BEFORE touching `node` again.
    //
    // get_operator_for_next_task() dereferences the operator (recursively, via
    // get_next_task_hint()) on THIS thread. Attaching only just before dispatch would leave that
    // dereference outside the counted region, so drain_pending_tasks(query_id) could return while
    // the manager was still walking the query's operators — and the caller's next act is to let
    // the plan be destroyed.
    //
    // The slot's own RAII covers every exit: an early `continue` below destroys it and decrements
    // the query's count; the dispatch path moves it into the worker, which releases it when the
    // creation lambda returns. Entered once, left once, with no separate bookkeeping to keep in
    // sync. (Attribution cannot happen at reserve() time: the manager reserves before it knows
    // which query it will serve, and an idle manager must not be counted against anyone.)
    slot.attach(query_id);

    std::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> visited_pipelines;
    node = get_operator_for_next_task(node, visited_pipelines);

    if (node == nullptr) {
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
      continue;
    }

    // Dispatch the task creation work to the pool
    _bounded_pool->dispatch(
      std::move(slot), [this, node, request_kind, query_state = std::move(query_state)]() mutable {
        // Scoped marker so task_creator::stop() can assert it is never called from here.
        struct worker_marker {
          worker_marker() { t_in_creation_worker = true; }
          ~worker_marker() { t_in_creation_worker = false; }
        } marker;
        try {
          // Get what we need to create the task
          auto pipeline = node->get_pipeline();
          // Shared ownership (step 6/B4): the task carries these across queue hops and
          // blocking windows, so it co-owns them via the port's owning handle rather than
          // borrowing the raw alias. Null for dependency-only ports and raw-stubbed test
          // ports, exactly as the raw pointer could be.
          std::vector<std::shared_ptr<cucascade::shared_data_repository>>
            destination_data_repositories;

          for (const auto& port_info : pipeline->get_next_ports_after_sink()) {
            destination_data_repositories.push_back(
              port_info.next_operator->get_port(port_info.next_operator_port_name)->repo_owner);
          }

          while (!node->all_ports_empty()) {
            auto task_lock  = pipeline->get_task_creation_lock();
            auto input_data = node->get_next_task_input_data();
            auto* pipelineable_input =
              dynamic_cast<op::pipelineable_operator_data*>(input_data.get());
            // Late-mat (SIRIUS_EXP_LATE_MAT): hand the consuming operator's
            // port-materialization directive to the task input so
            // prepare_for_processing can materialize deferred batches. Only ever
            // non-null when the defer policy installed it (gate on).
            if (pipelineable_input && node->late_mat_port_directive) {
              pipelineable_input->late_mat_directive = node->late_mat_port_directive;
            }
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
                  !preferred_device_id.has_value() && partitioned &&
                  !query_state->active_gpu_ids.empty()) {
                // Index the active executor set so every task of a partition lands
                // on the same real GPU (required for cuco tables); the physical
                // topology would yield phantom pins when num_gpus < physical count.
                auto idx = partitioned->get_partition_idx() % query_state->active_gpu_ids.size();
                preferred_device_id = query_state->active_gpu_ids[idx];
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
              // Confine the task to the admitted subset. Every preference above except the
              // partition pin comes from where data lives rather than from the subset, and the
              // scheduler treats a preference as binding — so an excluded id would be honoured.
              // Clamping a residency-derived one costs the locality it encoded, but honouring
              // it would put the query on a GPU it was not admitted to. An unpreferred task
              // escapes too: the scheduler gives those to whichever executor asks first. Pin
              // those as well, but only on a real subset, since a pin costs the scheduler's
              // freedom to place them wherever frees up first.
              if (auto const& admitted = query_state->active_gpu_ids; !admitted.empty()) {
                bool const names_excluded_device =
                  preferred_device_id.has_value() &&
                  std::find(admitted.begin(), admitted.end(), *preferred_device_id) ==
                    admitted.end();
                bool const unpinned_on_a_subset =
                  !preferred_device_id.has_value() && admitted.size() < query_state->full_gpu_count;
                if (names_excluded_device || unpinned_on_a_subset) {
                  auto const idx      = _admission_rr.fetch_add(1) % admitted.size();
                  preferred_device_id = admitted[idx];
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
          report_fatal_error(query_state->completion_handler, std::current_exception());
        }
      });
  }
}

uint64_t task_creator::get_next_task_id() { return _task_id.fetch_add(1); }

}  // namespace sirius::creator
