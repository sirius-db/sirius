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
#include "pipeline/task_submission.hpp"
#include "telemetry/telemetry_context.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius {
namespace parallel {

itask_executor::itask_executor(
  sirius::exec::query_lifecycle_registry& lifecycle,
  exec::thread_pool_config config,
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context,
  std::optional<int> device_id)
  : _query_lifecycle(lifecycle),
    _config(std::move(config)),
    // Shared with the task_scheduler's queue so both derive a task's query the same way; see
    // pipeline::index_keys_for.
    _task_queue(&pipeline::index_keys_for),
    _telemetry_context(std::move(telemetry_context)),
    _task_queue_telemetry(_telemetry_context->context().task_queue_observer()->handle())

{
  _task_queue_telemetry.created({
    .worker_id = _telemetry_context->worker_id(),
    .gpu_device_id =
      device_id.has_value()
        ? std::optional(_telemetry_context->gpu_device_telemetry_handles(*device_id).device.id())
        : std::nullopt,
    .label = _config.thread_name_prefix + "-task-queue",
  });
}

itask_executor::~itask_executor() { stop(); }

bool itask_executor::schedule(std::unique_ptr<itask> input)
{
  // Declared before task so an exception destroys unsubmitted work before settling its publisher.
  exec::query_lifecycle_registry::submission_guard submission;
  auto task = std::move(input);
  try {
    if (task) {
      // The OOM reschedule path re-enters here from a pool worker after a 50 ms backoff, so a
      // drain for this query may already have passed. Refuse rather than re-arm work behind it.

      submission = pipeline::begin_submission(
        _query_lifecycle, *task, "task_executor: query lifecycle registration is missing");
      if (!submission) { return false; }

      if (auto* pipeline_task = dynamic_cast<pipeline::sirius_pipeline_itask*>(task.get())) {
        pipeline_task->telemetry_fsm().queued(
          {.queue = {.target = _task_queue_telemetry.id(), .data = {.entries = 1}}});
      }
    }
    if (!_task_queue.try_push(task)) {
      if (auto* gpu_task = dynamic_cast<pipeline::gpu_pipeline_task*>(task.get())) {
        if (auto handler = gpu_task->get_completion_handler()) {
          handler->report_error("task executor queue closed before dispatch");
        }
      }
      SIRIUS_LOG_WARN("Task queue interrupted, dropping task");
      return false;
    }
    return true;
  } catch (...) {
    // try_push preserves ownership on failure. Report before task destruction can complete
    // its pipeline; the caller's catch runs after this local owner has already unwound.
    if (auto* gpu_task = dynamic_cast<pipeline::gpu_pipeline_task*>(task.get())) {
      if (auto completion = gpu_task->get_completion_handler()) {
        completion->report_error(std::current_exception());
      }
    }
    throw;
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

  quiesce_manager();
  _task_queue.drain();
  resume_manager();
}

void itask_executor::quiesce_manager()
{
  // Releasing the manager thread's pool slot is a PRECONDITION for wait_all(), not an
  // optimization. manager_loop() calls reserve() and then blocks in _task_queue.pop(), so an idle
  // manager permanently holds an active slot; wait_all() waits for active_ == 0 and would never
  // return. interrupt() makes reserve() hand back an invalid slot and pop() return nullptr, which
  // is what lets the loop exit and the join succeed.
  _bounded_pool->interrupt();
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _bounded_pool->wait_all();
}

void itask_executor::resume_manager()
{
  _bounded_pool->resume();
  _task_queue.reactivate();
  _manager_thread = std::thread([this] { manager_loop(); });
}

void itask_executor::validate_query_empty(sirius::query_id_t query_id)
{
  // Only THIS query's tasks must be gone. The original validated the WHOLE queue, so a query
  // completing normally threw because a co-tenant had work legitimately queued.
  const std::size_t remaining =
    _task_queue.size(exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});

  if (remaining != 0) {
    SIRIUS_LOG_ERROR(
      "itask_executor::validate_query_empty: task queue NOT empty for query {} at completion — "
      "{} task(s) still queued; tasks were still being scheduled when the query was marked "
      "complete",
      query_id,
      remaining);
    throw std::runtime_error(
      "task_executor: task queue not empty at query completion (" + std::to_string(remaining) +
      " task(s) remaining) — premature completion while work was still scheduled");
  }
}

void itask_executor::discard_query_tasks(sirius::query_id_t query_id)
{
  // Leave accepted staging tasks for the manager: each consumed a device_ready signal.
  // Removing one here could strand the manager in pop() with no scheduler readiness left.
  // The GPU manager discards cancelled tasks before preparation and advertises readiness again.
  // The owner waits for leases after disposal, covering staging and manager-local tasks too.
  if (_bounded_pool) { _bounded_pool->discard_queued(query_id); }
}

}  // namespace parallel
}  // namespace sirius
