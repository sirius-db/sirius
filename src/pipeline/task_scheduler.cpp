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
#include "log/logging.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "pipeline/gpu_pipeline_executor.hpp"
#include "pipeline/sirius_pipeline_itask.hpp"
#include "planner/query.hpp"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
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
  exec::queue_ordering task_queue_ordering,
  const cucascade::memory::system_topology_info* sys_topology,
  const std::vector<std::unique_ptr<sirius::parallel::downgrade_executor>>* downgrade_executors)
  : _task_queue(task_queue_ordering), _telemetry_context(std::move(telemetry_context))
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
  if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
    pipeline_task->telemetry_handle().queued({
      .queue_resource_id      = _task_queue_telemetry->handle->uuid(),
      .queue_capacity_entries = 1,
    });
  }
  [[maybe_unused]] auto _ = _task_queue.push(std::move(task));
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

  _completion_handler = std::make_unique<completion_handler>();

  // Set completion handler on all executors
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->set_completion_handler(_completion_handler.get());
  }

  // Reset the round-robin counter so the walk is reproducible across
  // iterations of the same query (cache=table_gpu warm path keys cache
  // entries by device_id; without this reset the second iteration's source
  // tasks would assign to a different GPU and miss the cache entries).
  _no_pref_rr_counter.store(0, std::memory_order_relaxed);
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
  _completion_handler->report_error(std::move(error));
  stop();
}

void task_scheduler::drain_after_error()
{
  SIRIUS_LOG_INFO("task_scheduler: draining after error");
  // Teardown ordering is load-bearing. The scan/gpu executor drains below run
  // in-flight tasks to completion, and a completing task schedules its
  // downstream consumers via task_creator::schedule() (gpu_pipeline_executor and
  // duckdb_scan_executor both do this). Each such request holds a raw
  // sirius_physical_operator* owned by the engine, which is destroyed the moment
  // execute() returns. If the task_creator is live (or restarted) while those
  // requests are still in flight, its manager_loop dereferences a freed operator
  // in get_operator_for_next_task() — a use-after-free that crashes intermittently
  // under multi-partition sort with many in-flight pipeline tasks.
  //
  // So: stop the task_creator FIRST and keep its queue interrupted across the
  // executor drains. With the queue interrupted, schedule() pushes from
  // completion callbacks return false and the requests (and their dangling
  // operator pointers) are dropped instead of processed. Only AFTER every
  // executor has quiesced do we drain the creation queue and restart the
  // creator for the next query.
  if (_task_creator) { _task_creator->stop_thread_pool(); }

  // Drain the top-level task queue so management_eventloop doesn't dispatch
  // stale tasks from the failed query.
  _task_queue.drain();

  // Interrupt each GPU executor's manager loop, wait for in-flight thread-pool
  // tasks to finish, then restart the manager for the next query.
  for (auto& [device_id, gpu_exec] : _gpu_executors) {
    gpu_exec->drain_and_wait();
  }

  // Now that no executor can generate further task_creation_requests, discard
  // any that accumulated (and release the shared_ptr<data_batch> references they
  // hold, which would otherwise survive clear_all_repositories at QueryEnd and
  // show up as inter-iteration "memory leak" warnings). drain_pending_tasks()
  // reactivates the queue when done.
  if (_task_creator) { _task_creator->drain_pending_tasks(); }

  // Belt-and-suspenders: the executor restarts above emit device_ready signals,
  // and the management loop may have dispatched a leftover task into an executor
  // queue between the two drains. Clear the top-level queue once more so the
  // next query starts from empty.
  _task_queue.drain();

  if (_task_creator) { _task_creator->start_thread_pool(); }
  SIRIUS_LOG_INFO("task_scheduler: DONE draining after error");
}

void task_scheduler::wait_for_completion()
{
  // Once the query has signaled completion, NOTHING should still be queued. Rather
  // than drain (which would hide the bug), validate that every queue is empty and
  // throw if not — a non-empty queue means tasks were still being scheduled when we
  // declared the query done.
  //
  // Halt the producer (task_creator) first so the checks are not racing new task
  // creation. The task_creator is always restarted afterwards (even on throw) so the
  // next query can run.
  if (_task_creator) { _task_creator->stop_thread_pool(); }
  try {
    // The task_scheduler's pipeline task queue must be empty.
    if (const std::size_t remaining = _task_queue.size(); remaining != 0) {
      SIRIUS_LOG_ERROR(
        "task_scheduler::wait_for_completion: pipeline _task_queue NOT empty at query "
        "completion — {} task(s) still queued; work was still scheduled when the query was "
        "marked complete",
        remaining);
      throw std::runtime_error(
        "task_scheduler: pipeline task queue not empty at query completion (" +
        std::to_string(remaining) + " task(s) remaining)");
    }

    // Each executor must finish its in-flight tasks and then have an empty queue.
    for (auto& [device_id, gpu_exec] : _gpu_executors) {
      gpu_exec->wait_and_validate_empty();
    }
  } catch (...) {
    if (_task_creator) {
      _task_creator->drain_pending_tasks();
      _task_creator->start_thread_pool();
    }
    throw;
  }
  if (_task_creator) {
    _task_creator->drain_pending_tasks();
    _task_creator->start_thread_pool();
  }
}

void task_scheduler::management_eventloop()
{
  telemetry::TaskManagerLoopThreadHandleWrapper manager_thread_telemetry{
    *_telemetry_context, "task-scheduler-thread", _telemetry_context->shared_group_id()};

  // Pull-signal scheduler. The loop blocks on _task_request_channel for two
  // event kinds:
  //   - device_ready  : a gpu_pipeline_executor has reserved a worker thread
  //                     and is ready to accept a task.
  //   - task_available: schedule() pushed a new task into _task_queue and is
  //                     asking us to re-run the matcher in case a device was
  //                     already waiting.
  // Tasks remain in _task_queue (downgrade-visible) until we have a ready
  // device to match them against — this is the property the push model in
  // PR #732 lost and that this loop restores.
  while (_running.load()) {
    // Block for the next event.
    auto evt = _task_request_channel.get();
    if (evt == nullptr) {
      SIRIUS_LOG_INFO("Task request channel closed, exiting management event loop.");
      break;
    }
    if (evt->kind == task_request_kind::device_ready && !evt->is_scan) {
      _ready_devices.emplace_back(evt->device_id);
    }
    // Drain any further events that are already queued, so a single matcher
    // pass handles a burst of ready signals plus task pushes together.
    while (auto more = _task_request_channel.try_get()) {
      if (more->kind == task_request_kind::device_ready && !more->is_scan) {
        _ready_devices.emplace_back(more->device_id);
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

      // Exact preference match.
      task = _task_queue.pop_if(
        [device_id](const sirius::parallel::itask& t) -> bool {
          const auto* gpu_task = dynamic_cast<const pipeline::gpu_pipeline_task*>(&t);
          if (!gpu_task) { return false; }
          auto pref = gpu_task->get_preferred_device_id();
          return pref.has_value() && pref.value() == device_id;
        },
        /*front_to_back=*/true);
      if (!task) {
        // Fallback: take the first task with NO preference, or whose preferred
        // device does not exist in _gpu_executors (a stale preference from a
        // different env / config is meaningless — treat as unpreferred).
        task = _task_queue.pop_if(
          [this](const sirius::parallel::itask& t) -> bool {
            const auto* gpu_task = dynamic_cast<const pipeline::gpu_pipeline_task*>(&t);
            if (!gpu_task) { return true; }
            auto pref = gpu_task->get_preferred_device_id();
            if (!pref.has_value()) { return true; }
            return _gpu_executors.count(pref.value()) == 0;
          },
          /*front_to_back=*/true);
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

      // Log prefix "[mgpu-audit] pipeline_task dispatched to GPU N" is
      // load-bearing — verification greps depend on it.
      SIRIUS_LOG_INFO(
        "[mgpu-audit] pipeline_task dispatched to GPU {} task_id={}", device_id, task_id);
      _gpu_executors.at(device_id)->schedule(std::move(task));
      it = _ready_devices.erase(it);
    }
  }
}

}  // namespace pipeline
}  // namespace sirius
