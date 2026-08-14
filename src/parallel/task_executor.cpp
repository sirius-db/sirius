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
  auto bounced = _task_queue.push_or_bounce(std::move(task));
  while (bounced) {
    if (!_running.load()) {
      // Real shutdown: dropping is the teardown contract, and it stays loud.
      SIRIUS_LOG_WARN("Task queue interrupted at shutdown, dropping task");
      return;
    }
    // Transient quiesce bracket (another query's error-path drain joining the
    // manager): the queue reactivates as soon as the join lands, so wait it
    // out instead of dropping — a dropped successor silently hangs this
    // task's query. The bracket cannot deadlock on this retry: quiesce
    // reactivates the queue BEFORE wait_all(), so a retrying pool worker
    // always gets through.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    bounced = _task_queue.push_or_bounce(std::move(bounced));
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
  {
    std::lock_guard<std::mutex> lifecycle_lock(_manager_lifecycle_mutex);
    _manager_thread = std::thread([this] { manager_loop(); });
  }
  on_start();
}

void itask_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  // A drain bracket may be mid-quiesce; its interrupts are already in place, so waiting here
  // cannot stall it — but joining/moving _manager_thread underneath it would be UB.
  std::lock_guard<std::mutex> lifecycle_lock(_manager_lifecycle_mutex);
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
  std::lock_guard<std::mutex> lifecycle_lock(_manager_lifecycle_mutex);

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
  // Reactivate the queue the moment the join lands: the interrupt exists only
  // to pop the (single) manager out of its blocking pop(). Staying interrupted
  // until resume_manager() bounced every CO-TENANT push in the window — and a
  // bounced successor is a pipeline that never finishes, i.e. that query hangs
  // (observed: a repeatedly-failing query starving 3 healthy ones). Reopening
  // here also lets a pool worker parked in schedule()'s bounce-retry proceed,
  // which wait_all() below depends on (that worker holds an active slot).
  // The failing query's own pushes stay refused by the lifecycle gate, and its
  // queued tasks are swept by the drain_query_tasks() that follows.
  _task_queue.reactivate();
  _bounded_pool->wait_all();
}

void itask_executor::resume_manager()
{
  _bounded_pool->resume();
  _task_queue.reactivate();
  _manager_thread = std::thread([this] { manager_loop(); });
}

void itask_executor::wait_and_validate_empty(sirius::query_id_t query_id)
{
  // Never started (or already stopped): nothing dispatched, nothing to validate. drain_and_wait()
  // has always had this guard; the whole-executor variant this replaced did not, and
  // null-dereferenced when a query failed before any work reached this executor.
  if (!_bounded_pool) { return; }

  // Per-query wait, no quiesce bracket. This is the success path: the query's completion handler
  // has already fired, so nothing of this query remains to be popped and the pop-to-attach window
  // that the error path must worry about cannot occur here. Untagged slots (a manager parked in
  // pop()) are ignored by drain_and_wait, which is exactly what lets this return without stopping
  // the executor -- and therefore without interrupting the shared queue and dropping a co-tenant's
  // in-transit task, which the previous bracket did on EVERY successful completion.
  // wait_for_query, NOT drain_and_wait: this is the success path, so the query's remaining work
  // must be allowed to RUN. Dropping it here would silently discard tasks the query legitimately
  // scheduled and then report success.
  _bounded_pool->wait_for_query(query_id);

  // Only THIS query's tasks must be gone. The original validated the WHOLE queue, so a query
  // completing normally threw because a co-tenant had work legitimately queued.
  const std::size_t remaining =
    _task_queue.size(exec::query_index{static_cast<exec::query_key>(sirius::value_of(query_id))});

  if (remaining != 0) {
    SIRIUS_LOG_ERROR(
      "itask_executor::wait_and_validate_empty: task queue NOT empty for query {} at completion — "
      "{} task(s) still queued; tasks were still being scheduled when the query was marked "
      "complete",
      query_id,
      remaining);
    throw std::runtime_error(
      "task_executor: task queue not empty at query completion (" + std::to_string(remaining) +
      " task(s) remaining) — premature completion while work was still scheduled");
  }
}

void itask_executor::wait_and_drain_query(sirius::query_id_t query_id)
{
  // Error-path counterpart of wait_and_validate_empty: let in-flight work finish so no thread is
  // still touching the failing query's plan, then drop that query's queued tasks.
  //
  // This one KEEPS the quiesce bracket, deliberately. Unlike the success path, a failing query can
  // still have tasks sitting in the queue that the manager is free to pop at any moment, and there
  // is a window between pop() and slot::attach() where the task belongs to neither the queue nor
  // the per-query slot count. Joining the manager is what closes it: after that, no task can be
  // in-hand. The caller's next act is to let the plan be destroyed, so "almost certainly quiesced"
  // is not good enough here.
  //
  // What did change: the drain in the middle is per-query. The original cleared the ENTIRE queue,
  // destroying every co-tenant's queued work and leaving those queries waiting on completions that
  // could never arrive. The residual cost is that co-tenant pushes are refused across the bracket
  // — on the error path only, not on every successful completion as before.
  //
  // The bracket must run to completion before another failing query's bracket may begin: an
  // interleaved second quiesce no-ops on the already-joined manager, and its resume_manager()
  // then assigns onto the joinable thread the first resume just created (std::terminate).
  std::lock_guard<std::mutex> lifecycle_lock(_manager_lifecycle_mutex);
  if (!_bounded_pool) {
    drain_query_tasks(query_id);
    return;
  }
  quiesce_manager();
  drain_query_tasks(query_id);
  resume_manager();
}

}  // namespace parallel
}  // namespace sirius
