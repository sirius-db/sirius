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
#include "cuda_runtime_api.h"
#include "downgrade/downgrade_executor.hpp"
#include "log/logging.hpp"
#include "memory/pinned_reservation_guard.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/task_request.hpp"
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
#include <string>
#include <utility>
namespace sirius {
namespace pipeline {

namespace {

/**
 * Blocking reservation acquisition with the unsatisfiable-reservation
 * livelock guard.
 *
 * cucascade's memory_space::make_reservation waits on the space's notification
 * channel with no timeout and no logging. gpu-tier pinned tables permanently
 * occupy the space's allocated-bytes budget and are invisible to the downgrade
 * executor, so a demand larger than (reservation limit − pinned bytes) can
 * NEVER be granted in full — the wait livelocks forever (observed: 113 min,
 * zero log output, TPC-H q5 with all tables pinned into a shrunken pool).
 *
 * Happy path is byte-identical to make_reservation(): the same non-blocking
 * first attempt, nothing else. Only when that attempt fails (a would-block
 * situation that debug-audited healthy runs never reach) do we:
 *   1. fail fast when the demand is provably unsatisfiable (conservative
 *      check: demand > limit − unevictable pinned bytes; transient pressure
 *      can never trigger it) — returns nullptr with *fail_fast_reason set;
 *   2. otherwise block exactly as before, wrapped in a reservation_wait_scope
 *      so an INFO line reports the outstanding wait every ~10 s.
 *
 * nullptr with an EMPTY *fail_fast_reason preserves the pre-existing meaning:
 * the space is shutting down.
 */
std::unique_ptr<cucascade::memory::reservation> acquire_reservation_blocking_with_guard(
  cucascade::memory::memory_space* space,
  std::size_t bytes_needs,
  std::size_t pipeline_id,
  uint64_t task_id,
  std::string* fail_fast_reason)
{
  if (auto reservation = space->make_reservation_or_null(bytes_needs)) { return reservation; }

  const std::size_t limit  = space->get_max_memory();
  const std::size_t pinned = sirius::memory::unevictable_pinned_bytes(space);
  if (pinned > 0 && sirius::memory::reservation_is_unsatisfiable(bytes_needs, limit, pinned)) {
    *fail_fast_reason = std::format(
      "GPU Pipeline Executor: unsatisfiable memory reservation for pipeline {} task {} on GPU "
      "{}: demand {} bytes > reservation limit {} bytes - unevictable gpu-tier pinned {} bytes "
      "(max satisfiable {} bytes; currently available {} bytes). Pinned tables cannot be "
      "evicted by the downgrade executor, so this wait would never complete. Unpin tables, pin "
      "to tier='host', or raise the GPU pool limit.",
      pipeline_id,
      task_id,
      space->get_device_id(),
      bytes_needs,
      limit,
      pinned,
      sirius::memory::max_satisfiable_reservation(limit, pinned),
      space->get_available_memory());
    return nullptr;
  }

  sirius::memory::reservation_wait_scope wait_scope(space, bytes_needs, pipeline_id, task_id);
  return space->make_reservation(bytes_needs);
}

}  // namespace

gpu_pipeline_executor::gpu_pipeline_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_space* mem_space,
  exec::publisher<std::unique_ptr<task_request>> task_request_publisher,
  sirius::parallel::downgrade_executor* downgrade_executor,
  std::shared_ptr<const telemetry::telemetry_context> telemetry_context)
  : sirius::parallel::itask_executor(
      config, std::move(telemetry_context), mem_space->get_device_id()),
    _stream_pool(rmm::cuda_device_id{mem_space->get_device_id()}, config.num_threads),
    _task_request_publisher(std::move(task_request_publisher)),
    _memory_space(mem_space),
    _downgrade_executor(downgrade_executor)
{
}

gpu_pipeline_executor::~gpu_pipeline_executor() { stop(); }

absl::AnyInvocable<void() noexcept> gpu_pipeline_executor::get_per_thread_init()
{
  int device_id          = _memory_space->get_device_id();
  auto thread_id_counter = std::make_shared<std::atomic<uint32_t>>(0);

  return [device_id,
          telemetry_context = _telemetry_context,
          thread_prefix     = _config.thread_name_prefix,
          thread_id_counter]() mutable noexcept {
    const int32_t thread_id = thread_id_counter->fetch_add(1, std::memory_order_relaxed);
    telemetry::thread_local_executor_thread_telemtry_init(
      *telemetry_context,
      std::format("{}-gpu{}-exec-{}", thread_prefix, device_id, thread_id),
      telemetry_context->executor_thread_group_id(device_id));

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
  telemetry::TaskManagerLoopThreadHandleWrapper manager_thread_telemetry{
    *_telemetry_context,
    std::format("gpu-{}-exec-manager", _memory_space->get_device_id()),
    _telemetry_context->manager_thread_group_id(_memory_space->get_device_id())};

  rmm::cuda_set_device_raii set_device_guard(rmm::cuda_device_id{_memory_space->get_device_id()});
  sirius::util::enable_log_on_default_stream();
  while (_running.load()) {
    auto slot = _bounded_pool->reserve();  // block till a thread is available
    if (!slot) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: pool interrupted, stopping manager loop");
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
          "GPU Pipeline Executor: task_request channel closed, stopping manager loop");
        break;
      }
      pipeline_task = _task_queue.pop();  // block till a task is available
    }
    if (!pipeline_task) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: task queue interrupted, stopping manager loop");
      break;
    }
    // Everything past this point is per-task and must not be able to stop the manager thread:
    // it serves every in-flight query on this device.
    process_task(std::move(pipeline_task), std::move(slot), manager_thread_telemetry);
  }
}

void gpu_pipeline_executor::process_task(
  std::unique_ptr<parallel::itask> pipeline_task,
  exec::bounded_thread_pool::slot slot,
  telemetry::TaskManagerLoopThreadHandleWrapper& manager_thread_telemetry) noexcept
{
  // Resolved as soon as the task is known so every failure path below — including an unexpected
  // throw — fails THIS task's query rather than the engine.
  std::shared_ptr<completion_handler> iteration_completion;
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
    // Resolve the task's query once it is known: it attributes the reserved slot (so
    // drain_and_wait(query_id) covers this execution) and any downgrade request issued below
    // (so the query's own cleanup — and only its own — can cancel it). Not done at reserve()
    // time: the manager parks in pop() holding the slot before any task exists, and counting
    // that against a query would make its drain wait for work that may never arrive.
    auto task_query_id = sirius::make_query_id(0);
    if (auto const* pipe = gpu_task->get_pipeline()) {
      task_query_id = pipe->get_query_id();
      slot.attach(task_query_id);
    }
    // Pass this executor's memory space so cross-space inputs (host/disk tiers and GPU data on
    // another device, which prepare clones into this space) are counted in the reservation.
    auto reservation_info = gpu_task->get_estimated_reservation_size_info(_memory_space);
    auto bytes_needs      = reservation_info.reservation_size;
    gpu_task->telemetry_handle().reserving({
      .instance_name              = "",
      .requested_bytes            = reservation_info.reservation_size,
      .input_basis                = reservation_info.input_basis,
      .peak_estimate              = reservation_info.peak_memory_estimate,
      .bytes_to_materialize       = reservation_info.bytes_to_materialize_input,
      .manager_thread_resource_id = manager_thread_telemetry.handle->uuid(),
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
    std::string fail_fast_reason;
    auto reservation = acquire_reservation_blocking_with_guard(_memory_space,
                                                               bytes_needs,
                                                               gpu_task->get_pipeline_id(),
                                                               gpu_task->get_task_id(),
                                                               &fail_fast_reason);
    if (!reservation && !fail_fast_reason.empty()) {
      // Provably-unsatisfiable demand: fail the QUERY, not the
      // executor — drop the task (slot RAII returns it to the pool) and keep
      // serving subsequent work, mirroring the retry-cap terminate-query path.
      SIRIUS_LOG_ERROR("{}", fail_fast_reason);
      if (iteration_completion && !iteration_completion->has_error()) {
        iteration_completion->report_error(fail_fast_reason);
      }
      return;
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

      gpu_task->telemetry_handle().downgrading({
        .instance_name              = "",
        .shortfall_bytes            = shortfall,
        .partial_bytes              = partial_size,
        .manager_thread_resource_id = manager_thread_telemetry.handle->uuid(),
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

      reservation.reset();  // release partial reservation before downgrade

      std::unique_ptr<cucascade::memory::reservation> new_reservation;
      auto* mem_space = _memory_space;
      size_t freed    = 0;
      std::mutex reservation_mutex;
      try {
        freed =
          _downgrade_executor
            ->request_downgrade(task_query_id,
                                [mem_space, bytes_needs, &new_reservation, &reservation_mutex]() {
                                  std::lock_guard<std::mutex> lock(reservation_mutex);
                                  if (new_reservation) { return true; }
                                  auto res = mem_space->make_reservation_or_null(bytes_needs);
                                  if (res && res->size() >= bytes_needs) {
                                    new_reservation = std::move(res);
                                  }
                                  return new_reservation != nullptr;
                                })
            .get();
      } catch (const std::exception& e) {
        // The downgrade executor cancelled this request (its queue was drained). This task cannot
        // get its reservation, so fail its query
        SIRIUS_LOG_INFO("GPU Pipeline Executor: downgrade request cancelled for task {}: {}",
                        gpu_task->get_task_id(),
                        e.what());
        if (iteration_completion) {
          iteration_completion->report_error(
            "GPU Pipeline Executor: downgrade request cancelled for task " +
            std::to_string(gpu_task->get_task_id()) + ": " + e.what());
        }
        return;
      }

      if (new_reservation) {
        reservation = std::move(new_reservation);
      } else {
        // Predicate never succeeded — try one final reservation attempt. This
        // call blocks indefinitely too, so it takes the same guard as the
        // first acquisition (fail-fast on unsatisfiable demand + periodic
        // wait visibility).
        reservation = acquire_reservation_blocking_with_guard(_memory_space,
                                                              bytes_needs,
                                                              gpu_task->get_pipeline_id(),
                                                              gpu_task->get_task_id(),
                                                              &fail_fast_reason);
        if (!reservation && !fail_fast_reason.empty()) {
          SIRIUS_LOG_ERROR("{}", fail_fast_reason);
          if (iteration_completion && !iteration_completion->has_error()) {
            iteration_completion->report_error(fail_fast_reason);
          }
          return;
        }
      }

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
    _bounded_pool->dispatch(
      std::move(slot),
      [this,
       task       = std::move(pipeline_task),
       exc_stream = std::move(exc_stream),
       consumers  = std::move(output_consumers),
       completion = std::move(completion),
       pipeline]() mutable {
        try {
          task->execute(exc_stream);
          _tasks_executed.fetch_add(1, std::memory_order_relaxed);
        } catch (task_reschedule_exception& ex) {
          // Only THIS query's error state suppresses the reschedule. Previously one query's
          // failure silently stopped every other query's tasks from rescheduling.
          if (completion && completion->has_error()) { return; }
          auto* gpu_task = cast_to_gpu_pipeline_task(task.get());
          if (!gpu_task) {
            SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast task for reschedule");
            if (completion) {
              completion->report_error("GPU Pipeline Executor: Failed to cast task for reschedule");
            }
            return;
          }

          // Sync the stream to ensure all memory is released before the reschedule.
          exc_stream->synchronize();

          // Determine retry count and original task ID for this rescheduled attempt.
          auto* cur_local = dynamic_cast<gpu_pipeline_task_local_state*>(gpu_task->local_state());
          uint32_t next_retry_count = 1;
          uint64_t orig_task_id     = gpu_task->get_task_id();
          if (cur_local && cur_local->original_task_id.has_value()) {
            next_retry_count = cur_local->retry_count + 1;
            orig_task_id     = cur_local->original_task_id.value();
          }

          // The retry cap is per-query state stamped on the pipeline's build context
          // from the admission-time operator_params snapshot (register E1) — one
          // coherent read here, never the live config struct mid-execution. Configured
          // via operator_params.gpu_reservation_max_retries (YAML) / SET
          // gpu_reservation_max_retries; see exec::default_gpu_reservation_max_retries
          // for the default's provenance (follow-up #17: 100 retries x 50 ms backoff
          // rides out cross-GPU BUILD_PROBE batch-lock contention at SF100 while still
          // bailing out on truly wedged queries). Tasks without a pipeline (tests) use
          // the process default.
          const uint32_t max_retries = pipeline ? pipeline->reservation_max_retries()
                                                : exec::default_gpu_reservation_max_retries;
          if (next_retry_count > max_retries) {
            SIRIUS_LOG_ERROR(
              "GPU Pipeline Executor: task {} (original task {}) exceeded {} retries at "
              "operator index {} — terminating query: {}",
              gpu_task->get_task_id(),
              orig_task_id,
              max_retries,
              ex.get_resume_operator_index(),
              ex.what());
            if (completion) {
              completion->report_error(std::make_exception_ptr(std::runtime_error(
                "GPU pipeline task exceeded maximum retry limit (" + std::to_string(max_retries) +
                ") for original task " + std::to_string(orig_task_id) + ": " + ex.what())));
            }
            return;
          }

          SIRIUS_LOG_WARN(
            "GPU Pipeline Executor: reschedule (retry {}/{}) for task {} (original task {}), "
            "resuming from operator index {}: {}",
            next_retry_count,
            max_retries,
            gpu_task->get_task_id(),
            orig_task_id,
            ex.get_resume_operator_index(),
            ex.what());

          auto intermediate_data = ex.release_intermediate_data();
          if (auto pipelineable_data =
                dynamic_cast<op::pipelineable_operator_data*>(intermediate_data.get())) {
            // We want to release the read-only lock on the data so that when its added back to the
            // task queue it could be downgraded if needed.
            pipelineable_data->remove_read_only_lock();
          }

          // Build the rescheduled task via virtual factory (preserves derived type).
          auto new_local_state = std::make_unique<gpu_pipeline_task_local_state>(
            std::move(intermediate_data), ex.get_resume_operator_index());
          new_local_state->retry_count      = next_retry_count;
          new_local_state->original_task_id = orig_task_id;

          // Preserve the per-task device pin across reschedule. Dropping it lets
          // an OOM'd partition task scatter to the wrong GPU and touch a cuco
          // table built on another device (cudaErrorInvalidValue). Only the
          // local_state pin needs copying; a global-state pin already survives.
          if (cur_local && cur_local->get_preferred_device_id().has_value()) {
            new_local_state->set_preferred_device_id(cur_local->get_preferred_device_id().value());
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
          std::this_thread::sleep_for(std::chrono::milliseconds(50));

          // Schedule the rescheduled task. It goes back through manager_loop()
          // to acquire a fresh reservation before execution.
          if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
            pipeline_task->telemetry_handle().finalizing({
              .instance_name = "",
              .success       = false,
            });
            pipeline_task->telemetry_handle().exit();
            pipeline_task->set_telemetry_finalized();
          }
          this->schedule(std::move(new_task));
          return;
        } catch (const std::exception& e) {
          SIRIUS_LOG_ERROR("GPU Pipeline Executor: Exception during task execution: {}", e.what());
          if (completion) { completion->report_error(std::current_exception()); }
          return;
        } catch (...) {
          SIRIUS_LOG_ERROR("GPU Pipeline Executor: unknown error during task execution");
          if (completion) { completion->report_error(std::current_exception()); }
          return;
        }
        if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
          pipeline_task->telemetry_handle().finalizing({
            .instance_name = "",
            .success       = true,
          });
          pipeline_task->telemetry_handle().exit();
          pipeline_task->set_telemetry_finalized();
        }
        task.reset();

        // Check if query is complete BEFORE scheduling downstream tasks.
        // mark_completed() signals the future that engine.execute() is waiting on,
        // which may destroy the engine and its operators. We must not schedule
        // tasks that reference those operators after signaling completion.
        bool query_complete = false;
        if (completion && pipeline) {
          auto sink = pipeline->get_sink();
          if (sink && sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
            query_complete = pipeline->is_pipeline_finished();
          }
        }

        if (!query_complete && _task_creator) {
          // Schedule consumers explicitly here to drive the scheduler's
          // round-robin rotation per-batch. notify_downstream_pipelines() in
          // the task destructor only fires once the pipeline drains —
          // mid-pipeline batches need to start rotating before that point so
          // they reach all GPUs.
          // schedule() throws on a consumer with no pipeline; this runs outside the execute()
          // try/catch above, so report it to this task's query rather than letting it escape the
          // pool callable.
          try {
            for (auto* consumer : consumers) {
              if (consumer) { _task_creator->schedule(consumer); }
            }
          } catch (const std::exception& e) {
            SIRIUS_LOG_ERROR("GPU Pipeline Executor: failed to schedule downstream consumers: {}",
                             e.what());
            if (completion) { completion->report_error(std::current_exception()); }
            return;
          }
        }

        if (query_complete && completion) {
          // Scoped to the finishing query: its pending creation requests point at operators
          // that mark_completed() may let the engine destroy. Any other query's requests are
          // left alone.
          _task_creator->drain_pending_tasks(pipeline->get_query_id());
          completion->mark_completed();
        }
      });
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("GPU Pipeline Executor: Exception while preparing task for dispatch: {}",
                     e.what());
    try {
      if (iteration_completion) { iteration_completion->report_error(std::current_exception()); }
    } catch (...) {  // reporting must never take down the manager thread
    }
  } catch (...) {
    SIRIUS_LOG_ERROR("GPU Pipeline Executor: unknown error while preparing task for dispatch");
    try {
      if (iteration_completion) { iteration_completion->report_error(std::current_exception()); }
    } catch (...) {
    }
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
