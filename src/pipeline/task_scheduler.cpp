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

#include "pipeline/task_scheduler.hpp"

#include "creator/task_creator.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/config.hpp"
#include "exec/multi_index_priority_queue.hpp"
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/sirius_pipeline_itask.hpp"
#include "planner/query.hpp"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius {
namespace pipeline {

namespace {
// Covers normal stragglers, including OOM retry backoff.
constexpr auto k_ledger_drain_timeout = std::chrono::seconds(60);
}  // namespace

task_scheduler::task_scheduler(
  const exec::thread_pool_config& gpu_executor_config,
  sirius::memory::sirius_memory_reservation_manager& mem_mgr,
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
  const cucascade::memory::system_topology_info* sys_topology,
  const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>* downgrade_executors)
  : _task_queue([](const sirius::parallel::itask& task) -> exec::index_keys {
      // Derive the multi-index keys from the task. The queue orders by priority
      // (lower value = dispatched first) and additionally indexes by operator type,
      // query id, and preferred device. Non-pipeline tasks fall back to the maximum
      // priority so they sort last, with sentinel index keys.
      if (const auto* gpu_task = dynamic_cast<const pipeline::gpu_pipeline_task*>(&task)) {
        const exec::queue_priority priority = gpu_task->get_priority();
        // The scheduling priority packs query_id in its high 32 bits and the
        // within-query pipeline rank in the low 32 (see task_creator), so the query
        // id is recoverable here without extra plumbing.
        const exec::query_key query_id =
          static_cast<exec::query_key>(static_cast<std::uint64_t>(priority) >> 32);
        exec::operator_key operator_type = op::SiriusPhysicalOperatorType::INVALID;
        if (const auto* pipe = gpu_task->get_pipeline()) {
          if (auto source = pipe->get_source()) { operator_type = source->type; }
        }
        const auto pref = gpu_task->get_preferred_device_id();
        return exec::index_keys{priority,
                                operator_type,
                                query_id,
                                pref.has_value() ? pref.value() : exec::no_preferred_device};
      }
      return exec::index_keys{std::numeric_limits<exec::queue_priority>::max(),
                              op::SiriusPhysicalOperatorType::INVALID,
                              0,
                              exec::no_preferred_device};
    }),
    _telemetry_context(std::move(telemetry_context))
{
  _downgrade_executors  = downgrade_executors;
  _task_queue_telemetry = std::make_unique<telemetry::TaskQueueHandleWrapper>(
    *_telemetry_context, "task-scheduler-gpu-queue", _telemetry_context->shared_group_id());

  // Self-publisher: schedule() uses this to wake management_eventloop when a
  // new task is pushed, so the loop can re-run the matcher against any device
  // that is already in _ready_devices.
  _self_publisher.emplace(_task_request_channel.make_publisher());

  auto gpu_spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  // Initialize GPU pipeline executors for each available GPU
  for (auto* space : gpu_spaces) {
    auto config   = gpu_executor_config;
    int device_id = space->get_device_id();
    if (sys_topology) {
      auto it = std::find_if(sys_topology->gpus.begin(),
                             sys_topology->gpus.end(),
                             [device_id](const cucascade::memory::gpu_topology_info& dev) {
                               return dev.id == device_id;
                             });

      if (it != sys_topology->gpus.end()) { config.cpu_affinity_list = it->cpu_cores; }
    }
    // Find matching downgrade executor for this GPU space
    sirius::parallel::downgrade_executor* dg_exec = nullptr;
    if (downgrade_executors) {
      for (auto& de : *downgrade_executors) {
        if (de->get_space_id() == space->get_id()) {
          dg_exec = de.get();
          break;
        }
      }
    }

    // Pass a publisher so gpu_pipeline_executor can submit task requests
    _gpu_executors.emplace(
      device_id,
      std::make_unique<gpu_pipeline_executor>(config,
                                              const_cast<cucascade::memory::memory_space*>(space),
                                              _task_request_channel.make_publisher(),
                                              dg_exec,
                                              _telemetry_context));
  }
}

task_scheduler::~task_scheduler() { stop(); }

void task_scheduler::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
    pipeline_task->telemetry_handle().queued({
      .queue_resource_id      = _task_queue_telemetry->handle->uuid(),
      .queue_capacity_entries = 1,
    });
  }
  _task_queue.push(std::move(task));
  if (_self_publisher) {
    auto wake                 = std::make_unique<task_request>();
    wake->kind                = task_request_kind::task_available;
    [[maybe_unused]] auto _ok = _self_publisher->send(std::move(wake));
  }
}

void task_scheduler::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->start();
  }
  _management_thread = std::thread(&task_scheduler::management_eventloop, this);
}

void task_scheduler::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  // Pull-signal model: the management event loop blocks on
  // _task_request_channel.get(), so closing the channel is what wakes it.
  // _task_queue.interrupt() is still useful to reject any concurrent
  // schedule() calls but no longer needed to wake the loop.
  _task_queue.interrupt();
  _task_request_channel.close();
  // Join the management thread first so it can finish processing any drained
  // events without dispatching to executors that are about to be stopped.
  if (_management_thread.joinable()) { _management_thread.join(); }
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->stop();
  }
}

void task_scheduler::set_task_creator(sirius::creator::task_creator& task_creator)
{
  _task_creator = &task_creator;

  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_task_creator(_task_creator);
  }
}

void task_scheduler::prepare_for_query(duckdb::shared_ptr<planner::query> query)
{
  // Drain leftover tasks from previous query
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_leftover_tasks();
  }

  std::lock_guard lock(_query_mutex);
  _query = std::move(query);

  _completion_handler = std::make_shared<completion_handler>();

  // Executors keep raw references owned by the scheduler.
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_completion_handler(_completion_handler.get());
  }

  // The creator may still be parked from the previous query.
  if (_task_creator) {
    _task_creator->set_completion_handler(_completion_handler);
    _task_creator->start_thread_pool();
  }

  // Pipelines hold the handler weakly so they cannot extend the query lifetime.
  if (_query) {
    for (auto& pipeline : _query->get_pipelines()) {
      if (pipeline) { pipeline->set_completion_handler(_completion_handler); }
    }
  }
}

std::future<void> task_scheduler::start_query()
{
  std::scoped_lock lock(_query_mutex);
  const auto& scans = _query->get_scan_operators();

  // A query with no schedulable scan can never complete. Plan generation should have
  // rejected it, so fail loudly instead of dereferencing an empty vector.
  if (scans.empty()) {
    throw std::runtime_error("task_scheduler: query has no schedulable scan sources");
  }

  _task_creator->schedule(scans.front());

  return _completion_handler->get_awaitable();
}

void task_scheduler::terminate_query(std::exception_ptr error)
{
  std::shared_ptr<completion_handler> completion;
  {
    std::scoped_lock lock(_query_mutex);
    completion = _completion_handler;
  }
  // Teardown belongs to drain_after_error(); stop() is terminal and cannot be restarted.
  if (completion) { (void)completion->report_error(std::move(error)); }
}

void task_scheduler::interrupt_query_scan_sources()
{
  duckdb::shared_ptr<planner::query> query;
  {
    std::lock_guard<std::mutex> query_lock(_query_mutex);
    query = _query;
  }
  if (!query) { return; }
  for (auto* scan : query->get_scan_operators()) {
    if (scan) { scan->interrupt_source(); }
  }
}

void task_scheduler::drain_after_error()
{
  SIRIUS_LOG_INFO("task_scheduler: draining after error");
  // Reject new producer work before joining and draining existing work.
  if (_completion_handler) { _completion_handler->close_work(); }

  // Wake creator workers parked in a scan source's blocking split wait before joining them:
  // their producers may be dead, and the closes that would wake them only run in query cleanup
  // after this returns.
  interrupt_query_scan_sources();

  // Stop the creator before draining executors: completion callbacks may otherwise enqueue
  // requests containing plan-owned operator pointers. Keep it parked until the next query.
  if (_task_creator) { _task_creator->stop_thread_pool(); }

  // Downgrade workers can return borrowed tasks, so join them before draining queues.
  if (_downgrade_executors) {
    for (auto& downgrade_exec : *_downgrade_executors) {
      if (downgrade_exec) { downgrade_exec->stop(); }
    }
  }

  // Drain the top-level task queue so management_eventloop doesn't dispatch
  // stale tasks from the failed query.
  _task_queue.drain();

  // Flush any task currently between the scheduler queue and an executor queue.
  {
    std::lock_guard<std::mutex> dispatch_lock(_dispatch_mutex);
  }

  // Interrupt each GPU executor's manager loop, wait for in-flight thread-pool
  // tasks to finish, then restart the manager for the next query.
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_and_wait();
  }

  // No executor can now create requests. Discard any queued requests and their data references
  // while leaving the creator parked.
  if (_task_creator) { _task_creator->drain_pending_tasks(/*reactivate=*/false); }

  // Executor restarts emit readiness signals, so clear anything dispatched between drains.
  _task_queue.drain();

  // QueryEnd destroys the plan after this returns, so wait indefinitely rather than risk
  // teardown while a producer callback still holds a slot.
  if (_completion_handler) {
    std::size_t waited_s = 0;
    while (!_completion_handler->wait_quiescent(k_ledger_drain_timeout)) {
      waited_s += std::chrono::duration_cast<std::chrono::seconds>(k_ledger_drain_timeout).count();
      SIRIUS_LOG_ERROR(
        "task_scheduler::drain_after_error: {} unit(s) of work still outstanding after {}s; "
        "teardown fail-stopped until they drain (only a mid-flight producer callback can hold "
        "one here, so this indicates a wedged thread)",
        _completion_handler->outstanding_work(),
        waited_s);
    }
  }

  // Restart downgrade workers only after every borrowed task has been returned and drained.
  if (_downgrade_executors) {
    for (auto& downgrade_exec : *_downgrade_executors) {
      if (downgrade_exec) { downgrade_exec->start(); }
    }
  }
  SIRIUS_LOG_INFO("task_scheduler: DONE draining after error");
}

void task_scheduler::wait_for_completion()
{
  // Keep execution active while work that outlived the completion signal drains normally.
  if (!_completion_handler) { return; }

  // An early-LIMIT finish leaves scan connectors open, and a creator worker can be parked in
  // one's split wait, holding its request slot. Interrupt sources first: the ledger cannot
  // drain past a parked worker, and the creator join below would never return. On a normal
  // finish every connector is already closed, so this is a no-op.
  interrupt_query_scan_sources();

  const auto deadline  = std::chrono::steady_clock::now() + k_ledger_drain_timeout;
  const auto time_left = [&deadline]() {
    return std::max(std::chrono::duration_cast<std::chrono::milliseconds>(
                      deadline - std::chrono::steady_clock::now()),
                    std::chrono::milliseconds(0));
  };

  bool quiescent = _completion_handler->wait_quiescent(time_left());

  // Park the creator so lookahead cannot create work after the first zero.
  if (_task_creator) { _task_creator->stop_thread_pool(); }

  // Drain requests that raced the first zero and creator shutdown.
  while (quiescent && _completion_handler->outstanding_work() != 0) {
    if (_task_creator) { _task_creator->drain_pending_tasks(/*reactivate=*/false); }
    quiescent = _completion_handler->wait_quiescent(time_left());
  }

  // Closing rejects new slots; the final wait covers acquisitions that won the race.
  _completion_handler->close_work();
  if (quiescent) { quiescent = _completion_handler->wait_quiescent(time_left()); }

  // Surface an error that lost the race with the completion signal.
  if (auto late_error = _completion_handler->take_late_error()) {
    std::rethrow_exception(late_error);
  }
  if (!quiescent) {
    const std::size_t remaining = _completion_handler->outstanding_work();
    SIRIUS_LOG_ERROR(
      "task_scheduler::wait_for_completion: {} unit(s) of work still outstanding {}s after the "
      "query signalled completion; work was still in flight when the query was marked complete",
      remaining,
      std::chrono::duration_cast<std::chrono::seconds>(k_ledger_drain_timeout).count());
    throw premature_completion_error(
      "task_scheduler: work still outstanding at query completion (" + std::to_string(remaining) +
      " unit(s) after " +
      std::to_string(
        std::chrono::duration_cast<std::chrono::seconds>(k_ledger_drain_timeout).count()) +
      "s) — the query was marked complete while work was still in flight");
  }
}

std::exception_ptr task_scheduler::take_preserved_error() noexcept
{
  return _completion_handler ? _completion_handler->take_late_error() : nullptr;
}

void task_scheduler::management_eventloop()
{
  telemetry::TaskManagerLoopThreadHandleWrapper manager_thread_telemetry{
    *_telemetry_context, "task-scheduler-thread", _telemetry_context->shared_group_id()};

  // Each pass tops up the two things the matcher needs — known ready devices
  // and a non-empty queue — sleeping only for whichever is missing. The queue
  // sleep (not the channel) is what hears the downgrade executor returning
  // extracted tasks: that return is a direct _task_queue.push() with no
  // task_available event, and blocking solely on the channel deadlocked once
  // every executor had parked (#1467). Both sleeps are interrupted by stop().
  while (_running.load()) {
    // Devices: block only when none is parked (every executor is then busy
    // and will post device_ready), then drain the pending burst.
    auto evt = _task_request_channel.try_get();
    if (!evt && _ready_devices.empty()) {
      evt = _task_request_channel.get();
      if (evt == nullptr) {
        SIRIUS_LOG_INFO("Task request channel closed, exiting management event loop.");
        break;
      }
    }
    while (evt) {
      if (evt->kind == task_request_kind::device_ready) {
        // A restarted manager may re-announce while its old credit is still pending.
        // Deduplication preserves one reserved worker per dispatch credit.
        if (std::find(_ready_devices.begin(), _ready_devices.end(), evt->device_id) ==
            _ready_devices.end()) {
          _ready_devices.emplace_back(evt->device_id);
        }
      }
      evt = _task_request_channel.try_get();
    }

    // Work: let the creator pre-create for a waiting device (lookahead
    // strategy only), then sleep until something is pushed.
    if (_task_queue.empty()) {
      if (_task_creator && !_ready_devices.empty()) {
        _task_creator->schedule_lookahead(*_ready_devices.begin());
      }
      if (!_task_queue.wait()) {
        SIRIUS_LOG_INFO("Task queue interrupted, exiting management event loop.");
        break;
      }
    }

    // Matcher: for each ready device, try to find a dispatchable task.
    // A task is dispatchable to device X if:
    //   (a) its preferred_device_id == X (exact match), OR
    //   (b) it has no preferred_device_id (any device will do).
    // Tasks with a preference for a DIFFERENT device must wait — they may
    // reference GPU-resident data (cache=table_gpu, partitioned batches) that
    // is only valid on the preferred device. Step (ii) will introduce an
    // explicit strict-vs-prefer bit; for now, all preferences are treated as
    // binding to guarantee correctness.
    // Every step in the pass is non-blocking; drain_after_error() cycles this lock to flush
    // an in-flight pass before it drains the executor queues.
    std::lock_guard<std::mutex> dispatch_lock(_dispatch_mutex);
    for (auto it = _ready_devices.begin(); it != _ready_devices.end();) {
      const int device_id = *it;
      std::unique_ptr<sirius::parallel::itask> task;

      // Exact preference match: the device index returns the highest-priority
      // (lowest value) task preferring exactly this device.
      task = _task_queue.try_pop_from(exec::gpu_index{device_id}).value_or(nullptr);
      if (!task) {
        // Pick a task with no preference; any ready device may claim it.
        task =
          _task_queue.try_pop_from(exec::gpu_index{exec::no_preferred_device}).value_or(nullptr);
      }
      if (!task) {
        // No dispatchable task for this device. Leave device in _ready_devices
        // and move on — it will match when an appropriate task arrives.
        ++it;
        continue;
      }
      uint64_t task_id = 0;
      if (auto* gpu_task = dynamic_cast<pipeline::gpu_pipeline_task*>(task.get())) {
        task_id = gpu_task->get_task_id();
      }

      if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
        pipeline_task->telemetry_handle().routing({
          .instance_name              = "",
          .preferred_device_id        = device_id,
          .manager_thread_resource_id = manager_thread_telemetry.handle->uuid(),
        });
      }

      // // Log prefix "[mgpu-audit] pipeline_task dispatched to GPU N" is
      // // load-bearing — verification greps depend on it.
      // SIRIUS_LOG_INFO(
      //   "[mgpu-audit] pipeline_task dispatched to GPU {} task_id={}", device_id, task_id);
      _gpu_executors.at(device_id)->schedule(std::move(task));
      it = _ready_devices.erase(it);
    }
  }
}

}  // namespace pipeline
}  // namespace sirius
