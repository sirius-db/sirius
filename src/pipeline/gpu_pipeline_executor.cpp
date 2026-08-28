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
    auto* gpu_task = cast_to_gpu_pipeline_task(pipeline_task.get());
    if (!gpu_task) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast pipeline task to gpu_pipeline_task");
      if (_completion_handler) {
        _completion_handler->report_error(
          "GPU Pipeline Executor: Failed to cast pipeline task to gpu_pipeline_task");
      }
      break;
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
    auto reservation = _memory_space->make_reservation(bytes_needs);
    if (!reservation) {
      SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to acquire memory reservation for task {}",
                       gpu_task->get_task_id());
      if (_completion_handler) {
        _completion_handler->report_error(
          "GPU Pipeline Executor: Failed to acquire memory reservation for task " +
          std::to_string(gpu_task->get_task_id()));
      }
      break;
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
            ->request_downgrade([mem_space, bytes_needs, &new_reservation, &reservation_mutex]() {
              std::lock_guard<std::mutex> lock(reservation_mutex);
              if (new_reservation) { return true; }
              auto res = mem_space->make_reservation_or_null(bytes_needs);
              if (res && res->size() >= bytes_needs) { new_reservation = std::move(res); }
              return new_reservation != nullptr;
            })
            .get();
      } catch (const std::exception& e) {
        SIRIUS_LOG_INFO("GPU Pipeline Executor: downgrade request cancelled for task {}: {}",
                        gpu_task->get_task_id(),
                        e.what());
        break;
      }

      if (new_reservation) {
        reservation = std::move(new_reservation);
      } else {
        // Predicate never succeeded — try one final reservation attempt
        reservation = _memory_space->make_reservation(bytes_needs);
      }

      if (!reservation) {
        SIRIUS_LOG_ERROR(
          "GPU Pipeline Executor: Failed to acquire memory reservation after "
          "downgrade for task {} (freed {} bytes)",
          gpu_task->get_task_id(),
          freed);
        if (_completion_handler) {
          _completion_handler->report_error(
            "GPU Pipeline Executor: Failed to acquire memory reservation "
            "after downgrade for task " +
            std::to_string(gpu_task->get_task_id()));
        }
        break;
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
      if (_completion_handler) {
        _completion_handler->report_error(
          "GPU Pipeline Executor: Failed to cast local state for task " +
          std::to_string(gpu_task->get_task_id()));
      }
      break;
    }
    auto output_consumers = gpu_task->get_output_consumers();
    auto* pipeline        = gpu_task->get_pipeline();
    auto exc_stream       = _stream_pool.acquire_stream(
      cucascade::memory::exclusive_stream_pool::stream_acquire_policy::GROW);
    _bounded_pool->dispatch(
      std::move(slot),
      [this,
       task       = std::move(pipeline_task),
       exc_stream = std::move(exc_stream),
       consumers  = std::move(output_consumers),
       pipeline]() mutable {
        try {
          task->execute(exc_stream);
          _tasks_executed.fetch_add(1, std::memory_order_relaxed);
        } catch (task_reschedule_exception& ex) {
          if (_completion_handler && _completion_handler->has_error()) {
            // If the completion handler is already in an error state, then we can just return and
            // not try to reschedule
            return;
          }
          auto* gpu_task = cast_to_gpu_pipeline_task(task.get());
          if (!gpu_task) {
            SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast task for reschedule");
            if (_completion_handler) {
              _completion_handler->report_error(
                "GPU Pipeline Executor: Failed to cast task for reschedule");
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

          // Bumped from 10 to 100 as part of follow-up #17. SF100 Q11 with
          // cache=table_gpu + num_gpus=2 exhausted the old 10-retry budget
          // against cross-GPU BUILD_PROBE batch-lock contention: the batch
          // was held in `processing` on one GPU while the probe task on the
          // other GPU needed it. Each convert-release cycle is O(100ms) at
          // SF100 scale, so 10 retries × 5ms backoff (50 ms total) was far
          // too short. With 100 retries × 50 ms backoff (~5 s) the probe
          // tasks get enough patience to clear the contention window while
          // still bailing out on truly wedged queries.
          static constexpr uint32_t MAX_RETRIES = 100;
          if (next_retry_count > MAX_RETRIES) {
            SIRIUS_LOG_ERROR(
              "GPU Pipeline Executor: task {} (original task {}) exceeded {} retries at "
              "operator index {} — terminating query: {}",
              gpu_task->get_task_id(),
              orig_task_id,
              MAX_RETRIES,
              ex.get_resume_operator_index(),
              ex.what());
            if (_completion_handler) {
              _completion_handler->report_error(std::make_exception_ptr(std::runtime_error(
                "GPU pipeline task exceeded maximum retry limit (" + std::to_string(MAX_RETRIES) +
                ") for original task " + std::to_string(orig_task_id) + ": " + ex.what())));
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
            // We want to release the read-only lock on the data so that when its added back to the
            // task queue it could be downgraded if needed.
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
          if (_task_creator) { _task_creator->stop(); }
          if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
          return;
        } catch (...) {
          SIRIUS_LOG_ERROR("GPU Pipeline Executor: unknown error during task execution");
          if (_task_creator) { _task_creator->stop(); }
          if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
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
        if (_completion_handler && pipeline && pipeline->is_query_terminal()) {
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

        if (query_complete && _completion_handler) {
          _task_creator->drain_pending_tasks();
          _completion_handler->mark_completed();
        }
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

bool gpu_pipeline_executor::is_task_queue_empty() const noexcept { return _task_queue.is_empty(); }

executor_metrics gpu_pipeline_executor::get_metrics() const noexcept
{
  return {_tasks_executed.load(std::memory_order_relaxed)};
}

void gpu_pipeline_executor::set_completion_handler(completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

}  // namespace pipeline
}  // namespace sirius
