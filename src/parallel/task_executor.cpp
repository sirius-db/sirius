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

#include "parallel/task_executor.hpp"

#include "log/logging.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline_itask.hpp"
#include "telemetry/telemetry_context.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius {
namespace parallel {

itask_executor::itask_executor(
  exec::thread_pool_config config,
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
  std::optional<int> device_id)
  : _config(std::move(config)),
    // Shared with the task_scheduler's queue so both derive a task's query the same way; see
    // pipeline::index_keys_for.
    _task_queue(&pipeline::index_keys_for),
    _telemetry_context(std::move(telemetry_context)),
    _task_queue_telemetry(std::make_unique<telemetry::TaskQueueHandleWrapper>(
      *_telemetry_context,
      _config.thread_name_prefix + "-task-queue",
      device_id.has_value() ? _telemetry_context->gpu_device_group_id(*device_id)
                            : _telemetry_context->engine_id()))
{
}

itask_executor::~itask_executor() { stop(); }

void itask_executor::schedule(std::unique_ptr<itask> task)
{
  if (task) {
    // The OOM reschedule path re-enters here from a pool worker after a 50 ms backoff, so a
    // drain for this query may already have passed. Refuse rather than re-arm work behind it.
    if (_query_lifecycle != nullptr && !_query_lifecycle->accepts_work(sirius::make_query_id(
                                         pipeline::index_keys_for(*task).query_id))) {
      return;
    }
    if (auto* pipeline_task = dynamic_cast<pipeline::sirius_pipeline_itask*>(task.get())) {
      pipeline_task->telemetry_handle().queued({
        .queue_resource_id      = _task_queue_telemetry->handle->uuid(),
        .queue_capacity_entries = 1,
      });
    }
  }
  if (!_task_queue.push(std::move(task))) {
    SIRIUS_LOG_WARN("Task queue interrupted, dropping task");
  }
}

void itask_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _bounded_pool = std::make_unique<exec::bounded_thread_pool>(_config.num_threads,
                                                              _config.thread_name_prefix,
                                                              _config.cpu_affinity_list,
                                                              get_per_thread_init());
  _task_queue.reactivate();
  _manager_thread = std::thread([this] { manager_loop(); });
  on_start();
}

void itask_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _bounded_pool->interrupt();
  _task_queue.interrupt();
  on_stop();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();
  _bounded_pool->stop();
  _bounded_pool.reset();
  on_stopped();
}

void itask_executor::wait_all()
{
  if (_bounded_pool) { _bounded_pool->wait_all(); }
}

void itask_executor::drain_leftover_tasks() { _task_queue.drain(); }

void itask_executor::drain_query_tasks(sirius::query_id_t query_id)
{
  _task_queue.drain(exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});
}

void itask_executor::drain_and_wait()
{
  // Guard: if the executor has never been started (or has been stopped),
  // _bounded_pool is nullptr. drain_after_error may legitimately be called
  // before any work has been dispatched, in which case there is nothing to
  // drain. Without this guard the next line dereferences nullptr and crashes
  // inside pthread_mutex_lock on the (offset-zero) std::mutex member —
  // observed during task_scheduler::drain_after_error after an early
  // sirius_engine::execute failure.
  if (!_bounded_pool) {
    SIRIUS_LOG_INFO("itask_executor::drain_and_wait: skipped — pool not initialized");
    return;
  }

  // Interrupt the pool so the manager's reserve() unblocks with an invalid slot.
  _bounded_pool->interrupt();

  // Interrupt pop() so the manager loop sees a nullptr and breaks out.
  _task_queue.interrupt();

  // Join the manager thread so we know it has exited.
  if (_manager_thread.joinable()) { _manager_thread.join(); }

  // Wait for all in-flight thread-pool tasks to finish.
  _bounded_pool->wait_all();

  // Clear any remaining tasks from the queue.
  _task_queue.drain();

  // Re-enable the pool and queue so the executor is ready for the next query.
  _bounded_pool->resume();
  _task_queue.reactivate();
  _manager_thread = std::thread([this] { manager_loop(); });
}

void itask_executor::wait_and_validate_empty()
{
  // Same quiescing as drain_and_wait(): interrupt the pool + queue so the manager
  // exits, join it, and wait for all in-flight thread-pool tasks to finish.
  _bounded_pool->interrupt();
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();

  // Instead of draining, VALIDATE the queue is empty. A non-empty queue here means
  // tasks were still scheduled on this executor when the query was declared complete.
  const std::size_t remaining = _task_queue.size();

  // Re-enable the pool/queue and restart the manager so the executor is left in a
  // usable state for the next query, whether or not validation passes.
  _bounded_pool->resume();
  _task_queue.reactivate();
  _manager_thread = std::thread([this] { manager_loop(); });

  if (remaining != 0) {
    SIRIUS_LOG_ERROR(
      "itask_executor::wait_and_validate_empty: task queue NOT empty at query completion — "
      "{} task(s) still queued; tasks were still being scheduled when the query was marked "
      "complete",
      remaining);
    throw std::runtime_error(
      "task_executor: task queue not empty at query completion (" + std::to_string(remaining) +
      " task(s) remaining) — premature completion while work was still scheduled");
  }
}

}  // namespace parallel
}  // namespace sirius
