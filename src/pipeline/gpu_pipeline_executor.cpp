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

#include "pipeline/gpu_pipeline_executor.hpp"

#include "creator/task_creator.hpp"
#include "cucascade/memory/stream_pool.hpp"
#include "cuda/device_health.hpp"
#include "cuda_runtime_api.h"
#include "downgrade/downgrade_executor.hpp"
#include "event/event.hpp"
#include "exec/leased_callable.hpp"
#include "log/logging.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/task_request.hpp"
#include "telemetry-bridge/gen/quent.hpp"
#include "telemetry/telemetry_context.hpp"

#include <rmm/cuda_device.hpp>

#include <util/stream_check_wrapper.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
namespace sirius {
namespace pipeline {

namespace {

/**
 * One stalled reserve phase of one task: from the first time the executor announces that the
 * task is waiting for memory until the task holds its final reservation or gives up.
 *
 * Construction enters a waiter on the memory_wait_board (when one is installed) BEFORE the stall
 * is announced and before the task blocks, so a reader of the board sees the stall without
 * waiting for event delivery. One scope covers the whole phase: the blocking wait, a partial
 * grant, the downgrade request and the final reservation attempt, so the hold is continuous
 * across the hand-off from waiting to downgrading.
 *
 * Destruction releases the waiter and then publishes exactly one memory_wait_ended_for_task,
 * under the same masked query id as the stall events it closes. It runs on every exit, early
 * returns and exceptions included; the event is only a wake-up, so losing it costs latency, never
 * a stuck hold.
 */
class memory_wait_scope {
 public:
  memory_wait_scope(event::query_event_publisher& publisher,
                    memory::memory_wait_board* board,
                    query_id_t masked_query_id,
                    std::size_t operator_id,
                    int gpu)
    : _publisher(publisher), _query_id(masked_query_id), _operator_id(operator_id), _gpu(gpu)
  {
    if (board != nullptr) { _waiter.emplace(board->enter_wait(gpu)); }
  }

  memory_wait_scope(memory_wait_scope const&)            = delete;
  memory_wait_scope& operator=(memory_wait_scope const&) = delete;
  memory_wait_scope(memory_wait_scope&&)                 = delete;
  memory_wait_scope& operator=(memory_wait_scope&&)      = delete;

  /// Bytes the phase ended up holding; stays 0 when the phase gives up.
  void set_granted(std::size_t bytes) noexcept { _granted = bytes; }

  ~memory_wait_scope()
  {
    _waiter.reset();
    _publisher.publish_memory_wait_ended_for_task(_query_id, _operator_id, _gpu, _granted);
  }

 private:
  event::query_event_publisher& _publisher;
  std::optional<memory::memory_wait_board::waiter> _waiter;
  query_id_t _query_id;
  std::size_t _operator_id;
  int _gpu;
  std::size_t _granted{0};
};

/// Operator id of @p pipeline's source operator, or the invalid id when there is no pipeline.
std::size_t source_operator_id(sirius_pipeline const* pipeline) noexcept
{
  return pipeline != nullptr ? pipeline->get_source_operator().first
                             : op::sirius_physical_operator::invalid_operator_id;
}

}  // namespace

gpu_pipeline_executor::gpu_pipeline_executor(
  sirius::exec::query_lifecycle_registry& lifecycle,
  exec::thread_pool_config config,
  cucascade::memory::memory_space* mem_space,
  exec::publisher<std::unique_ptr<task_request>> task_request_publisher,
  sirius::parallel::downgrade_executor* downgrade_executor,
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context)
  : sirius::parallel::itask_executor(
      lifecycle, config, std::move(telemetry_context), mem_space->get_device_id()),
    _stream_pool(rmm::cuda_device_id{mem_space->get_device_id()}, config.num_threads),
    _task_request_publisher(std::move(task_request_publisher)),
    _memory_space(mem_space),
    _downgrade_executor(downgrade_executor)
{
}

gpu_pipeline_executor::~gpu_pipeline_executor() { stop(); }

sirius::exec::invocable<void() noexcept> gpu_pipeline_executor::get_per_thread_init()
{
  int device_id          = _memory_space->get_device_id();
  auto thread_id_counter = std::make_shared<std::atomic<uint32_t>>(0);

  return [device_id,
          telemetry_context = _telemetry_context,
          thread_prefix     = _config.thread_name_prefix,
          thread_id_counter]() mutable noexcept {
    const int32_t thread_id = thread_id_counter->fetch_add(1, std::memory_order_relaxed);

    try {
      auto& thread_handle = telemetry::executor_thread_telemetry_state.handle;
      thread_handle.reset();
      thread_handle = telemetry_context->context().executor_thread_observer()->handle();
      thread_handle->spawned({
        .label = std::format("{}-gpu{}-exec-{}", thread_prefix, device_id, thread_id),
        .group_id =
          telemetry_context->gpu_device_telemetry_handles(device_id).executor_threads.id(),
      });
    } catch (std::exception const& e) {
      SIRIUS_LOG_ERROR("get_per_thread_init telemetry failed: {}", e.what());
    }

    // Per-thread init runs on a worker thread just spawned by the
    // bounded_pool. cudaSetDevice pins this thread to the executor's GPU
    // context; silent failure would cause every downstream CUDA call on this
    // thread to land on GPU 0 regardless of device_id. We cannot use
    // CUCASCADE_CUDA_TRY here because the lambda is noexcept — inline the
    // check instead.
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      SIRIUS_LOG_ERROR("gpu_pipeline_executor per-thread init: cudaSetDevice({}) failed: {}",
                       device_id,
                       cudaGetErrorString(err));
    }
    sirius::util::enable_log_on_default_stream();
  };
}

void gpu_pipeline_executor::manager_loop()
{
  quent::Handle<quent::TaskManagerLoopThread> manager_thread_handle =
    _telemetry_context->context().task_manager_loop_thread_observer()->handle();
  manager_thread_handle.spawned({
    .label    = std::format("gpu-{}-exec-manager", _memory_space->get_device_id()),
    .group_id = _telemetry_context->gpu_device_telemetry_handles(_memory_space->get_device_id())
                  .manager_threads.id(),
  });

  rmm::cuda_set_device_raii set_device_guard(rmm::cuda_device_id{_memory_space->get_device_id()});
  sirius::util::enable_log_on_default_stream();
  while (_running.load()) {
    auto slot = _bounded_pool->reserve();  // block till a thread is available
    if (!slot) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: pool interrupted, stopping publisher loop");
      break;
    }
    // Pull-signal backpressure: now that we hold a reserved thread slot, tell the
    // task_scheduler this device is ready for work. The scheduler will only move
    // a task out of its downgrade-visible queue into our _task_queue when a
    // ready signal has been received from us — preventing tasks from piling
    // up here where the downgrade executor can't see them.
    auto ready       = std::make_unique<task_request>();
    ready->kind      = task_request_kind::device_ready;
    ready->device_id = _memory_space->get_device_id();

    std::unique_ptr<parallel::itask> pipeline_task = nullptr;
    {
      if (!_task_request_publisher.send(std::move(ready))) {
        SIRIUS_LOG_INFO(
          "GPU Pipeline Executor: task_request channel closed, stopping publisher loop");
        break;
      }
      pipeline_task = _task_queue.pop();  // block till a task is available
    }
    if (!pipeline_task) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: task queue interrupted, stopping publisher loop");
      break;
    }
    // Everything past this point is per-task and must not be able to stop the manager thread:
    // it serves every in-flight query on this device.
    process_task(std::move(pipeline_task), std::move(slot), manager_thread_handle);
  }
}

void gpu_pipeline_executor::process_task(
  std::unique_ptr<parallel::itask> pipeline_task,
  exec::bounded_thread_pool::slot slot,
  quent::Handle<quent::TaskManagerLoopThread>& manager_thread_handle) noexcept
{
  // Resolved as soon as the task is known so every failure path below — including an unexpected
  // throw — fails THIS task's query rather than the engine.
  std::shared_ptr<completion_handler> iteration_completion;
  std::optional<query_id_t> iteration_query;
  try {
    auto* gpu_task = cast_to_gpu_pipeline_task(pipeline_task.get());
    if (!gpu_task) {
      // Only gpu_pipeline_tasks are ever scheduled onto a GPU executor, so this is a
      // programming error rather than a query failure. There is no pipeline here and therefore
      // no query whose completion handler could own the error; reporting it to some arbitrary
      // in-flight query would fail the wrong one. Drop it loudly instead: this used to throw,
      // which escaped manager_loop (a std::thread entry function) and aborted the process.
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast pipeline task to gpu_pipeline_task");
      return;
    }
    iteration_completion = gpu_task->get_completion_handler();
    // Cancellation leaves accepted staging tasks for this manager to consume. Drop them before
    // memory preparation (which can block), then resume the normal readiness cycle. The task
    // lease protects its resources even if cancellation races this advisory check.
    auto const* pipe    = gpu_task->get_pipeline();
    auto const query_id = pipe ? pipe->get_query_id() : make_query_id(0);
    iteration_query     = query_id;
    if (!_query_lifecycle.accepts_work(query_id)) { return; }
    // Pass this executor's memory space so cross-space inputs (host/disk tiers and GPU data on
    // another device, which prepare clones into this space) are counted in the reservation.
    auto reservation_info = gpu_task->get_estimated_reservation_size_info(_memory_space);
    auto bytes_needs      = reservation_info.reservation_size;
    gpu_task->telemetry_fsm().reserving({
      .requested_bytes      = reservation_info.reservation_size,
      .input_basis          = reservation_info.input_basis,
      .peak_estimate        = reservation_info.peak_memory_estimate,
      .bytes_to_materialize = reservation_info.bytes_to_materialize_input,
      .manager_thread       = {.target = manager_thread_handle.id(), .data = {}},
    });
    // Clamp the reservation request to what this memory space can actually
    // grant (its reservation limit). The history-based estimate can balloon far
    // past capacity — a small input that once drove a near-cap peak yields a
    // huge peak/estimate ratio that extrapolates to multi-GiB estimates on a
    // GiB-budget GPU. make_reservation() then returns only a partial reservation
    // and the downgrade predicate below (which must reserve the *full*
    // bytes_needs) can never succeed, so the task livelocks through the
    // OOM-reschedule loop until the retry cap trips and the query fails. Capping
    // at get_max_memory() keeps both the reservation and the downgrade target
    // achievable; any per-batch overflow during execution is still handled by
    // the OOM-reschedule + tiering path. (Telemetry above intentionally reports
    // the pre-clamp estimate so the estimator can still be analyzed.)
    if (auto const space_max = _memory_space->get_max_memory();
        space_max > 0 && bytes_needs > space_max) {
      SIRIUS_LOG_DEBUG(
        "GPU Pipeline Executor: clamping reservation request {} -> {} bytes (space max) for "
        "pipeline {} task {}",
        bytes_needs,
        space_max,
        gpu_task->get_pipeline_id(),
        gpu_task->get_task_id());
      bytes_needs = space_max;
    }
    SIRIUS_LOG_TRACE(
      "[GPU:{}] GPU Pipeline Executor: Acquiring memory reservation for pipeline {} of {} bytes "
      "for task {}. Memory available: {}, total reserved: {}, max: {}",
      _memory_space->get_device_id(),
      gpu_task->get_pipeline_id(),
      bytes_needs,
      gpu_task->get_task_id(),
      _memory_space->get_available_memory(),
      _memory_space->get_total_reserved_memory(),
      _memory_space->get_max_memory());
    // Try without blocking first, purely so the blocking case can be reported.
    // make_reservation() parks until the memory frees up, and a listener told
    // only once it returns learns nothing it can act on -- by then the stall it
    // would have exploited is over.  A reservation that succeeds outright is the
    // common case and raises nothing.
    //
    // Declared before the reservation so an early return releases a partial grant before the
    // stall's board hold, never after.
    std::optional<memory_wait_scope> stall;
    auto enter_stall = [&](query_id_t masked_query_id, std::size_t operator_id, int gpu) {
      if (!stall) {
        stall.emplace(
          *_query_event_publisher, _memory_wait_board.get(), masked_query_id, operator_id, gpu);
      }
    };
    auto reservation = _memory_space->make_reservation_or_null(bytes_needs);
    exec::query_lifecycle_registry::memory_wait_guard memory_wait_activity;
    if (!reservation || reservation->size() < bytes_needs) {
      memory_wait_activity =
        _query_lifecycle.begin_memory_wait(pipe ? pipe->get_query_id() : make_query_id(0));
    }
    if (!reservation) {
      auto const* stalled_pipe       = gpu_task->get_pipeline();
      auto const stalled_operator_id = source_operator_id(stalled_pipe);
      auto const stalled_query_id    = make_query_id(
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(gpu_task->get_priority()) >> 32));
      auto const stalled_gpu = _memory_space != nullptr ? _memory_space->get_device_id() : -1;
      enter_stall(stalled_query_id, stalled_operator_id, stalled_gpu);
      _query_event_publisher->publish_wait_for_memory_for_task(
        stalled_query_id, stalled_operator_id, stalled_gpu, bytes_needs);
      reservation = _memory_space->make_reservation(bytes_needs);
    }
    if (!reservation) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to acquire memory reservation for task {}",
                       gpu_task->get_task_id());
      if (auto handler = gpu_task->get_completion_handler()) {
        handler->report_error(
          "GPU Pipeline Executor: Failed to acquire memory reservation for task " +
          std::to_string(gpu_task->get_task_id()));
      }
      return;
    } else if (reservation->size() < bytes_needs && _downgrade_executor) {
      size_t shortfall    = bytes_needs - reservation->size();
      size_t partial_size = reservation->size();

      gpu_task->telemetry_fsm().downgrading({
        .shortfall_bytes = shortfall,
        .partial_bytes   = partial_size,
        .manager_thread  = {manager_thread_handle.id()},
      });

      SIRIUS_LOG_DEBUG(
        "GPU Pipeline Executor: requested reservation size {} but only got {} bytes, reservation "
        "shortfall {} bytes for pipeline {} "
        "task {}, requesting predicate-based downgrade",
        bytes_needs,
        partial_size,
        shortfall,
        gpu_task->get_pipeline_id(),
        gpu_task->get_task_id());

      // Reported before the (blocking) downgrade rather than after: the value of
      // knowing is that the GPU is about to stall, and a listener told only once
      // it has finished learns nothing it can act on.
      auto const* downgrade_pipe     = gpu_task->get_pipeline();
      auto const stalled_operator_id = source_operator_id(downgrade_pipe);
      auto const stalled_query_id    = make_query_id(
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(gpu_task->get_priority()) >> 32));
      auto const stalled_gpu = _memory_space != nullptr ? _memory_space->get_device_id() : -1;
      // Usually already engaged by the blocking wait above, whose partial grant led here; the
      // same scope then carries on through the downgrade.
      enter_stall(stalled_query_id, stalled_operator_id, stalled_gpu);
      _query_event_publisher->publish_memory_downgrade_for_task(
        stalled_query_id, stalled_operator_id, stalled_gpu, shortfall);

      reservation.reset();  // release partial reservation before downgrade

      std::unique_ptr<cucascade::memory::reservation> new_reservation;
      auto* mem_space = _memory_space;
      size_t freed    = 0;
      std::mutex reservation_mutex;
      try {
        freed =
          _downgrade_executor
            ->request_downgrade([mem_space, bytes_needs, &new_reservation, &reservation_mutex]() {
              std::lock_guard<std::mutex> lock(reservation_mutex);
              if (new_reservation) { return true; }
              auto res = mem_space->make_reservation_or_null(bytes_needs);
              if (res && res->size() >= bytes_needs) { new_reservation = std::move(res); }
              return new_reservation != nullptr;
            })
            .get();
      } catch (const std::exception& e) {
        if (_query_lifecycle.report_failure(std::current_exception(),
                                            pipe ? pipe->get_query_id() : make_query_id(0))) {
          throw;
        }
        // The downgrade request failed or was cancelled. This task cannot get its reservation, so
        // fail its query
        SIRIUS_LOG_INFO(
          "GPU Pipeline Executor: downgrade request failed or cancelled for task {}: {}",
          gpu_task->get_task_id(),
          e.what());
        if (iteration_completion) {
          iteration_completion->report_error(
            "GPU Pipeline Executor: downgrade request failed or cancelled for task " +
            std::to_string(gpu_task->get_task_id()) + ": " + e.what());
        }
        return;
      }

      if (new_reservation) {
        reservation = std::move(new_reservation);
      } else {
        // Predicate never succeeded — try one final reservation attempt
        reservation = _memory_space->make_reservation(bytes_needs);
      }

      // One INFO line per executor-driven (reserve-path) downgrade. The shortfall
      // detail above is DEBUG, and the downgrade executor's own per-request line is
      // DEBUG too, so without this a normal run cannot tell whether its spilling came
      // from here or from the periodic monitor. Tagged to pair with the
      // "[downgrade][monitor]" lines in downgrade_executor::monitor_loop.
      SIRIUS_LOG_INFO(
        "[downgrade][reserve] pipeline {} task {}: needed {} bytes (shortfall {}), downgrade freed "
        "{} bytes, reservation {}",
        gpu_task->get_pipeline_id(),
        gpu_task->get_task_id(),
        bytes_needs,
        shortfall,
        freed,
        reservation ? (reservation->size() >= bytes_needs ? "satisfied" : "partial") : "failed");

      if (!reservation) {
        SIRIUS_LOG_ERROR(
          "GPU Pipeline Executor: Failed to acquire memory reservation after "
          "downgrade for task {} (freed {} bytes)",
          gpu_task->get_task_id(),
          freed);
        if (auto handler = gpu_task->get_completion_handler()) {
          handler->report_error(
            "GPU Pipeline Executor: Failed to acquire memory reservation "
            "after downgrade for task " +
            std::to_string(gpu_task->get_task_id()));
        }
        return;
      }
      if (reservation->size() < bytes_needs) {
        SIRIUS_LOG_WARN(
          "GPU Pipeline Executor: after downgrade ({} bytes freed), reservation "
          "still partial ({}/{} bytes) for pipeline {} task {} -- proceeding "
          "with partial reservation",
          freed,
          reservation->size(),
          bytes_needs,
          gpu_task->get_pipeline_id(),
          gpu_task->get_task_id());
      }
    } else if (reservation->size() < bytes_needs) {
      // No downgrade executor available -- warn and proceed (this should never happen)
      SIRIUS_LOG_WARN(
        "GPU Pipeline Executor: Acquired memory reservation does not match "
        "requested size for pipeline {} of {} bytes needed for task "
        "{}. Reservation size: {}. WARNING: Downgrade executor is not available",
        gpu_task->get_pipeline_id(),
        bytes_needs,
        gpu_task->get_task_id(),
        reservation->size());
    }
    memory_wait_activity.reset();
    // The reserve phase is over and the task holds its reservation.
    if (stall) {
      stall->set_granted(reservation ? reservation->size() : 0);
      stall.reset();
    }
    if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(
          gpu_task->local_state())) {
      local_state->set_reservation(std::move(reservation), reservation_info);
    } else {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast local state for task {}",
                       gpu_task->get_task_id());
      if (auto handler = gpu_task->get_completion_handler()) {
        handler->report_error("GPU Pipeline Executor: Failed to cast local state for task " +
                              std::to_string(gpu_task->get_task_id()));
      }
      return;
    }
    auto output_consumers = gpu_task->get_output_consumers();
    auto* pipeline        = gpu_task->get_pipeline();
    auto exc_stream       = _stream_pool.acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
    // Resolved once at the top of this function: every report below belongs to THIS task's query,
    // so a failure or completion can never land on another in-flight query's promise. The
    // shared_ptr also keeps the handler alive past the owning sirius_engine, which is destroyed
    // before query cleanup drains the queues. Copied (not moved) into the lambda so the catch
    // blocks below still have it if dispatch itself throws.
    auto completion = iteration_completion;
    auto work       = pipeline_task->take_work_lease();
    auto const gpu  = _memory_space != nullptr ? _memory_space->get_device_id() : -1;
    exec::dispatch_with_work(
      *_bounded_pool,
      std::move(slot),
      query_id,
      std::move(work),
      [this, iteration_completion, query_id](std::exception_ptr error) noexcept {
        _query_lifecycle.report_failure(error, query_id);
        if (iteration_completion) { iteration_completion->report_error(error); }
      },
      [this,
       query_id,
       task       = std::move(pipeline_task),
       exc_stream = std::move(exc_stream),
       consumers  = std::move(output_consumers),
       completion = std::move(completion),
       pipeline,
       gpu]() mutable {
        // The enclosing leased callable retains all captures through this error boundary.
        try {
          try {
            if (completion) completion->record_task_started();
            task->execute(::cuda::stream_ref{exc_stream.get()});
            _tasks_executed.fetch_add(1, std::memory_order_relaxed);
          } catch (task_reschedule_exception& ex) {
            // Only THIS query's error state suppresses the reschedule. Previously one query's
            // failure silently stopped every other query's tasks from rescheduling.
            if (completion && completion->has_error()) { return; }
            bool const oom_reschedule = dynamic_cast<oom_reschedule_exception*>(&ex) != nullptr;
            auto* gpu_task            = cast_to_gpu_pipeline_task(task.get());
            if (!gpu_task) {
              SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast task for reschedule");
              if (completion) {
                completion->report_error(
                  "GPU Pipeline Executor: Failed to cast task for reschedule");
              }
              return;
            }

            // Sync the stream to ensure all memory is released before the reschedule.
            auto const status = cudaStreamSynchronize(exc_stream.get().get());
            check_cuda_health(status);
            if (status != cudaSuccess)
              throw std::runtime_error(std::string("CUDA retry synchronization failed: ") +
                                       cudaGetErrorName(status));

            // Determine retry count and original task ID for this rescheduled attempt.
            auto* cur_local = dynamic_cast<gpu_pipeline_task_local_state*>(gpu_task->local_state());
            uint32_t next_retry_count = 1;
            uint64_t orig_task_id     = gpu_task->get_task_id();
            if (cur_local && cur_local->original_task_id.has_value()) {
              next_retry_count = cur_local->retry_count + 1;
              orig_task_id     = cur_local->original_task_id.value();
            }

            // Bumped from 10 to 100 as part of follow-up #17. SF100 Q11 with
            // cache=table_gpu + num_gpus=2 exhausted the old 10-retry budget
            // against cross-GPU BUILD_PROBE batch-lock contention: the batch
            // was held in `processing` on one GPU while the probe task on the
            // other GPU needed it. Each convert-release cycle is O(100ms) at
            // SF100 scale, so 10 retries × 5ms backoff (50 ms total) was far
            // too short. With 100 retries × 50 ms backoff (~5 s) the probe
            // tasks get enough patience to clear the contention window while
            // still bailing out on truly wedged queries.
            auto const MAX_RETRIES = completion && completion->injections
                                       ? completion->injections->gpu_task_retry_limit
                                       : 100;
            if (next_retry_count > MAX_RETRIES) {
              SIRIUS_LOG_ERROR(
                "GPU Pipeline Executor: task {} (original task {}) exceeded {} retries at "
                "operator index {} — terminating query: {}",
                gpu_task->get_task_id(),
                orig_task_id,
                MAX_RETRIES,
                ex.get_resume_operator_index(),
                ex.what());
              if (completion) {
                completion->report_error(
                  std::make_exception_ptr(
                    std::runtime_error("GPU pipeline task exceeded maximum retry limit (" +
                                       std::to_string(MAX_RETRIES) + ") for original task " +
                                       std::to_string(orig_task_id) + ": " + ex.what())),
                  oom_reschedule ? transparent::late_failure_cause::oom_exhausted
                                 : transparent::late_failure_cause::retry_exhausted);
              }
              return;
            }

            SIRIUS_LOG_WARN(
              "GPU Pipeline Executor: reschedule (retry {}/{}) for task {} (original task {}), "
              "resuming from operator index {}: {}",
              next_retry_count,
              MAX_RETRIES,
              gpu_task->get_task_id(),
              orig_task_id,
              ex.get_resume_operator_index(),
              ex.what());

            auto intermediate_data = ex.release_intermediate_data();
            if (auto pipelineable_data =
                  dynamic_cast<op::pipelineable_operator_data*>(intermediate_data.get())) {
              // We want to release the read-only lock on the data so that when its added back to
              // the task queue it could be downgraded if needed.
              pipelineable_data->remove_read_only_lock();
            }

            // Build the rescheduled task via virtual factory (preserves derived type).
            auto new_local_state = std::make_unique<gpu_pipeline_task_local_state>(
              std::move(intermediate_data), ex.get_resume_operator_index());
            new_local_state->retry_count      = next_retry_count;
            new_local_state->original_task_id = orig_task_id;
            if (cur_local) { new_local_state->inherit_retry_reservation_floor(*cur_local); }

            // Preserve the per-task device pin across reschedule. Dropping it lets
            // an OOM'd partition task scatter to the wrong GPU and touch a cuco
            // table built on another device (cudaErrorInvalidValue). Only the
            // local_state pin needs copying; a global-state pin already survives.
            if (cur_local && cur_local->get_preferred_device_id().has_value()) {
              new_local_state->set_preferred_device_id(
                cur_local->get_preferred_device_id().value());
            }

            // An OOM retry counts as pending on this device until the retried task is destroyed.
            // A later retry of the same task carries its token over rather than adding a second
            // one, so one retry chain counts once; a new token is taken only for an OOM on a task
            // that holds none, or holds one for another device.
            if (_memory_wait_board) {
              auto holdoff = cur_local != nullptr
                               ? std::exchange(cur_local->oom_holdoff, std::nullopt)
                               : std::nullopt;
              if (oom_reschedule && (!holdoff || holdoff->device() != gpu)) {
                holdoff = _memory_wait_board->hold_oom(gpu);
              }
              new_local_state->oom_holdoff = std::move(holdoff);
            }

            auto new_task_id =
              _task_creator ? _task_creator->get_next_task_id() : gpu_task->get_task_id();
            auto new_task =
              gpu_task->create_rescheduled_task(new_task_id, std::move(new_local_state));

            // Backoff before rescheduling to allow other tasks to complete and
            // free memory (true OOM case) or release a contended batch
            // (cross-GPU processing contention, follow-up #17). 50 ms gives
            // typical SF100 probe tasks time to finish their current work
            // without putting the rescheduled task into a tight busy-spin.
            std::this_thread::sleep_for(
              std::chrono::milliseconds(completion && completion->injections
                                          ? completion->injections->gpu_task_retry_backoff_ms
                                          : 50));

            // Schedule the rescheduled task. It goes back through manager_loop()
            // to acquire a fresh reservation before execution.
            if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
              pipeline_task->set_telemetry_finalized();
              pipeline_task->telemetry_fsm().finalizing({
                .success = false,
              });
            }
            if (!_task_creator) { throw std::logic_error("GPU retry requires a task creator"); }
            bool const report_reschedule =
              _query_event_publisher->has_subscribers(event::event_type::task_completed);
            auto const rescheduled_operator_id =
              report_reschedule ? source_operator_id(pipeline)
                                : op::sirius_physical_operator::invalid_operator_id;
            _task_creator->reschedule(std::move(new_task));
            // Not a memory-freed signal: this attempt's task, and so its reservation, is still
            // captured here and is released only when this callable is destroyed.
            if (report_reschedule) {
              _query_event_publisher->publish_task_completed(
                query_id,
                rescheduled_operator_id,
                gpu,
                oom_reschedule ? event::task_outcome::oom_rescheduled
                               : event::task_outcome::rescheduled);
            }
            return;
          }
          if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
            pipeline_task->set_telemetry_finalized();
            pipeline_task->telemetry_fsm().finalizing({
              .success = true,
            });
          }
          bool const report_completion =
            _query_event_publisher->has_subscribers(event::event_type::task_completed);
          auto const completed_operator_id = report_completion
                                               ? source_operator_id(pipeline)
                                               : op::sirius_physical_operator::invalid_operator_id;
          // The enclosing leased callable protects the epilogue after task destruction.
          task.reset();
          // The task's reservation, its inputs and any OOM hold-off it carried are released now.
          if (report_completion) {
            _query_event_publisher->publish_task_completed(
              query_id, completed_operator_id, gpu, event::task_outcome::succeeded);
          }

          // Check if query is complete BEFORE scheduling downstream tasks.
          // mark_completed() signals the future that engine.execute() is waiting on,
          // which may destroy the engine and its operators. We must not schedule
          // tasks that reference those operators after signaling completion.
          bool query_complete = false;
          if (completion && pipeline && pipeline->is_query_terminal()) {
            query_complete = pipeline->is_pipeline_finished();
          }

          if (!query_complete && _task_creator) {
            // Schedule consumers explicitly here to drive the scheduler's
            // round-robin rotation per-batch. notify_downstream_pipelines() in
            // the task destructor only fires once the pipeline drains —
            // mid-pipeline batches need to start rotating before that point so
            // they reach all GPUs.
            for (auto* consumer : consumers) {
              if (consumer) { _task_creator->schedule(consumer); }
            }
          }

          if (query_complete && completion) {
            // Only the query owner retires work; this worker still holds its query lease.
            completion->mark_completed();
          }
        } catch (...) {
          auto error = std::current_exception();
          _query_lifecycle.report_failure(error, query_id);
          if (completion)
            completion->report_error(error, transparent::late_failure_cause::gpu_error);
          log::log_noexcept([&] {
            try {
              std::rethrow_exception(error);
            } catch (std::exception const& e) {
              SIRIUS_LOG_ERROR("GPU task execution, retry or completion failed: {}", e.what());
            } catch (...) {
              SIRIUS_LOG_ERROR("GPU task execution, retry or completion failed: unknown error");
            }
          });
        }
      });
  } catch (const std::exception& e) {
    _query_lifecycle.report_failure(std::current_exception(), iteration_query);
    if (iteration_completion) { iteration_completion->report_error(std::current_exception()); }
    log::log_noexcept([&] {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Exception while preparing task for dispatch: {}",
                       e.what());
    });
  } catch (...) {
    _query_lifecycle.report_failure(std::current_exception(), iteration_query);
    if (iteration_completion) { iteration_completion->report_error(std::current_exception()); }
    log::log_noexcept([] {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: unknown error while preparing task for dispatch");
    });
  }
}

gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(sirius::parallel::itask* task)
{
  // Safely cast to gpu_pipeline_task
  return dynamic_cast<gpu_pipeline_task*>(task);
}

void gpu_pipeline_executor::set_task_creator(sirius::creator::task_creator* task_creator)
{
  _task_creator = task_creator;
}

bool gpu_pipeline_executor::is_task_queue_empty() const noexcept { return _task_queue.empty(); }

executor_metrics gpu_pipeline_executor::get_metrics() const noexcept
{
  return {_tasks_executed.load(std::memory_order_relaxed)};
}

}  // namespace pipeline
}  // namespace sirius
