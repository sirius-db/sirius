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
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius {
namespace pipeline {

task_scheduler::task_scheduler(
  const exec::thread_pool_config& gpu_executor_config,
  sirius::memory::sirius_memory_reservation_manager& mem_mgr,
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
  const cucascade::memory::system_topology_info* sys_topology,
  const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>* downgrade_executors)
  // Shared with every gpu_pipeline_executor's queue so both agree on which query a task
  // belongs to; see pipeline::index_keys_for.
  : _task_queue(&index_keys_for), _telemetry_context(std::move(telemetry_context))
{
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
  // Refuse work for a query that is tearing down. A task creation worker can land here after
  // that query's queue drain already ran, and the task would then sit in the shared queue holding
  // raw repository pointers into a manager about to be erased.
  if (_query_lifecycle != nullptr && task &&
      !_query_lifecycle->accepts_work(sirius::make_query_id(index_keys_for(*task).query_id))) {
    return;
  }
  if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
    pipeline_task->telemetry_handle().queued({
      .queue_resource_id      = _task_queue_telemetry->handle->uuid(),
      .queue_capacity_entries = 1,
    });
  }
  // Read before the move: reporting a drop needs the query, and `task` is gone after push().
  const auto pushed_query =
    task ? sirius::make_query_id(index_keys_for(*task).query_id) : sirius::make_query_id(0);
  if (!_task_queue.push(std::move(task))) {
    // push returns false only when the queue is interrupted. If the gate still reports this query
    // as accepting work, the task is destroyed and its query waits forever on a completion that
    // cannot arrive -- the silent-drop failure mode behind several "query just hangs" reports.
    if (_query_lifecycle == nullptr || _query_lifecycle->accepts_work(pushed_query)) {
      SIRIUS_LOG_ERROR(
        "task_scheduler: task for query {} was DROPPED by an interrupted queue while the query was "
        "still accepting work; that query will not complete",
        pushed_query);
    } else {
      SIRIUS_LOG_DEBUG("task_scheduler: dropped a task for tearing-down query {}", pushed_query);
    }
  }
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

void task_scheduler::set_query_lifecycle_registry(sirius::exec::query_lifecycle_registry* registry)
{
  _query_lifecycle = registry;

  // Propagated so each device queue refuses a dying query's tasks too — notably the OOM
  // reschedule, which re-enters itask_executor::schedule from a worker thread.
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_query_lifecycle_registry(registry);
  }
}

void task_scheduler::start_query(const planner::query& query)
{
  const auto& scans = query.get_scan_operators();

  // A query with no schedulable scan can never complete. Plan generation should have
  // rejected it, so fail loudly instead of dereferencing an empty vector.
  if (scans.empty()) {
    throw std::runtime_error("task_scheduler: query has no schedulable scan sources");
  }

  // The caller already holds the future from its own completion handler.
  _task_creator->schedule(scans.front());
}

void task_scheduler::terminate_query(const std::shared_ptr<completion_handler>& handler,
                                     std::exception_ptr error)
{
  // Report to THIS query's handler and nothing else.
  if (handler) { handler->report_error(std::move(error)); }
}

void task_scheduler::drain_after_error(sirius::query_id_t query_id)
{
  SIRIUS_LOG_INFO("task_scheduler: draining after error for query {}", query_id);
  // Teardown ordering is load-bearing. The executor drains below run in-flight tasks to
  // completion, and a completing task schedules its downstream consumers via
  // task_creator::schedule(). Each such request holds a raw sirius_physical_operator* owned by
  // the engine, which is destroyed the moment execute() returns. If those requests are processed
  // after the plan dies, manager_loop dereferences a freed operator in
  // get_operator_for_next_task() — a use-after-free seen under multi-partition sort with many
  // in-flight pipeline tasks.
  //
  // This used to be achieved by stopping the shared task_creator and keeping its queue
  // interrupted across the drains, so that late schedule() pushes returned false. That worked but
  // was process-wide: it halted task creation for EVERY in-flight query and dropped their queued
  // requests too. The lifecycle gate does the same job scoped to one query — from here on,
  // schedule() for this query is refused while every other query keeps producing.
  if (_query_lifecycle != nullptr) { _query_lifecycle->quiesce(query_id); }

  // Drop this query's queued work so management_eventloop cannot dispatch a stale task from it.
  _task_queue.drain(exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});

  // Let in-flight tasks finish, then drop whatever this query still has staged per device.
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->wait_and_drain_query(query_id);
  }

  // Discard the creation requests this query accumulated — they hold raw operator pointers into a
  // plan that QueryEnd is about to destroy. Other queries keep their pending requests.
  if (_task_creator) { _task_creator->drain_pending_tasks(query_id); }

  // A task completing during the drains above can have handed one more task to the scheduler
  // queue before the gate refused its successor, so sweep this query once more.
  _task_queue.drain(exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});

  SIRIUS_LOG_INFO("task_scheduler: DONE draining after error for query {}", query_id);
}

void task_scheduler::wait_for_completion(sirius::query_id_t query_id)
{
  // Once the query has signaled completion, NOTHING should still be queued. Rather
  // than drain (which would hide the bug), validate that every queue is empty and
  // throw if not — a non-empty queue means tasks were still being scheduled when we
  // declared the query done.
  //
  // Halt this query's producers so the checks are not racing its own task creation. This used to
  // be _task_creator->stop_thread_pool(), which tore down the SHARED creation pool and interrupted
  // the shared queue — halting every other in-flight query, and dropping their queued requests,
  // on every successful completion. The gate does it for one query.
  if (_query_lifecycle != nullptr) { _query_lifecycle->quiesce(query_id); }
  const exec::query_index this_query{static_cast<exec::query_key>(sirius::value_of(query_id))};
  try {
    // Only THIS query's tasks must be gone. The old check used the whole-queue size(), so query A
    // completing normally threw "task queue not empty" because query B had work queued.
    if (const std::size_t remaining = _task_queue.size(this_query); remaining != 0) {
      SIRIUS_LOG_ERROR(
        "task_scheduler::wait_for_completion: pipeline _task_queue NOT empty for query {} at "
        "completion — {} task(s) still queued; work was still scheduled when the query was "
        "marked complete",
        query_id,
        remaining);
      throw std::runtime_error(
        "task_scheduler: pipeline task queue not empty at query completion (" +
        std::to_string(remaining) + " task(s) remaining)");
    }

    // Each executor must finish its in-flight tasks and then have none of this query's queued.
    for (auto& [device_id, gpu_exec] : _gpu_executors) {
      gpu_exec->wait_and_validate_empty(query_id);
    }
  } catch (...) {
    if (_task_creator) { _task_creator->drain_pending_tasks(query_id); }
    throw;
  }
  if (_task_creator) { _task_creator->drain_pending_tasks(query_id); }
}

void task_scheduler::drain_query_tasks(sirius::query_id_t query_id)
{
  // Pending work only, and only this query's: the scheduler's own queue first, then each GPU
  // executor's staging queue. In-flight tasks are untouched — quiescing those is
  // wait_for_completion / drain_after_error's job.
  _task_queue.drain(exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_query_tasks(query_id);
  }
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
        _ready_devices.emplace_back(evt->device_id);
      }
      evt = _task_request_channel.try_get();
    }

    // _ready_devices can be empty here: this loop also wakes on task_available events, which carry
    // no device, and a per-query drain can empty _task_queue between the push and this check.
    // Dereferencing begin() on an empty vector is UB — it used to yield a garbage device id.
    if (_task_queue.empty() && !_ready_devices.empty()) {
      // No query id: the task_creator picks the oldest live query itself, since this loop has
      // none to inherit.
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
    for (auto it = _ready_devices.begin(); it != _ready_devices.end();) {
      const int device_id = *it;
      std::unique_ptr<sirius::parallel::itask> task;

      // Exact preference match: the device index returns the highest-priority
      // (lowest value) task preferring exactly this device.
      task = _task_queue.try_pop_from(exec::gpu_index{device_id}).value_or(nullptr);
      if (!task) {
        // Pick a task with no preference (any device will do). Which GPU gets it is decided by
        // whichever executor signalled ready first, not by any counter.
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
