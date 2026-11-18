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
#include "exec/leased_callable.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/task_request.hpp"
#include "telemetry-bridge/gen/quent.hpp"
#include "telemetry/telemetry_context.hpp"

#include <rmm/cuda_device.hpp>

#include <util/stream_check_wrapper.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
namespace sirius {
namespace pipeline {

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

sirius::exec::invocable<void()> gpu_pipeline_executor::get_per_thread_init()
{
  int device_id          = _memory_space->get_device_id();
  auto thread_id_counter = std::make_shared<std::atomic<uint32_t>>(0);

  return [device_id,
          telemetry_context = _telemetry_context,
          thread_prefix     = _config.thread_name_prefix,
          thread_id_counter]() mutable {
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
    // thread to land on GPU 0 regardless of device_id.
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      throw std::runtime_error(
        std::format("gpu_pipeline_executor per-thread init: cudaSetDevice({}) failed: {}",
                    device_id,
                    cudaGetErrorString(err)));
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
    // A task that has waited too long, or was overtaken too often, is served before anything
    // new is dispatched (see reservation_wait_policy).
    wait_for_starving_waiters();
    // Try without blocking first. A reservation that succeeds outright is the common case; one
    // that does not is waited for on the task's worker thread, so this loop keeps dispatching
    // the tasks that do fit (reservation_wait_policy::wait_on_worker), or right here.
    auto const policy = reservation_policy();
    reservation_wait const wait{bytes_needs, query_id, manager_thread_handle.id()};
    auto reservation    = _memory_space->make_reservation_or_null(bytes_needs);
    bool const deferred = !reservation && policy.wait_on_worker;
    if (reservation) {
      note_bypass();
    } else if (!deferred) {
      reservation = acquire_reservation(*gpu_task, wait, /*on_worker=*/false);
      if (!reservation) { return; }
    }
    if (reservation && !attach_reservation(*gpu_task, std::move(reservation), reservation_info)) {
      return;
    }
    if (deferred) { _reservation_waits.fetch_add(1, std::memory_order_relaxed); }
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
       deferred,
       wait,
       reservation_info]() mutable {
        // The enclosing leased callable retains all captures through this error boundary.
        try {
          if (deferred) {
            // The manager could not reserve at once: wait here, on this task's own slot.
            auto* waiting_task = cast_to_gpu_pipeline_task(task.get());
            auto reservation   = acquire_reservation(*waiting_task, wait, /*on_worker=*/true);
            if (!reservation ||
                !attach_reservation(*waiting_task, std::move(reservation), reservation_info)) {
              return;
            }
          }
          try {
            if (completion) completion->record_task_started();
            task->execute(::cuda::stream_ref{exc_stream.get()});
            _tasks_executed.fetch_add(1, std::memory_order_relaxed);
          } catch (task_reschedule_exception& ex) {
            // Only THIS query's error state suppresses the reschedule. Previously one query's
            // failure silently stopped every other query's tasks from rescheduling.
            if (completion && completion->has_error()) { return; }
            auto* gpu_task = cast_to_gpu_pipeline_task(task.get());
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
            auto const is_oom      = dynamic_cast<oom_reschedule_exception*>(&ex) != nullptr;
            // What the downgrade executor could not do, if spilling itself is failing: without
            // it the user only sees the OOM that follows from it.
            auto const spill_note = [this]() -> std::string {
              if (!_downgrade_executor) { return {}; }
              auto failure = _downgrade_executor->spill_failure();
              return failure ? "; spilling GPU data is failing: " + *failure : std::string{};
            };
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
                completion->report_error(std::make_exception_ptr(std::runtime_error(
                                           "GPU pipeline task exceeded maximum retry limit (" +
                                           std::to_string(MAX_RETRIES) + ") for original task " +
                                           std::to_string(orig_task_id) + ": " + ex.what() +
                                           (is_oom ? spill_note() : std::string{}))),
                                         is_oom ? transparent::late_failure_cause::oom_exhausted
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
              // the task queue it could be downgraded if needed. This comes before
              // shrink_for_retry(): the input can only spill or drop batches it no longer pins.
              pipelineable_data->remove_read_only_lock();
            }
            // On OOM, let the input shrink its own working set; a retry that shrank keeps the
            // reservation floor it had before this OOM instead of the raised one. The retry
            // count still increments, so the retry cap bounds a run of shrinks.
            bool const shrank =
              is_oom && intermediate_data && intermediate_data->shrink_for_retry();
            // An OOM while the task's reservation already covered the whole memory space, with
            // an input that cannot shrink: the retry would get the same memory and fail the
            // same way, up to the retry limit. The first such OOM is retried once (the space
            // may have been shared with allocations made outside reservations); the second
            // fails the query.
            if (is_oom && !shrank && next_retry_count >= 2 && cur_local &&
                reservation_policy().fail_fast_on_full_space_oom) {
              auto const space_max = _memory_space->get_max_memory();
              auto const held      = cur_local->get_reservation_bytes();
              if (space_max > 0 && held >= space_max) {
                auto const& info       = cur_local->get_reservation_size_info();
                auto const input_bytes = info ? info->input_basis : std::size_t{0};
                auto message           = std::format(
                  "GPU task cannot fit in GPU memory: {} for original task {}, attempt {}, while "
                            "the task held a reservation for the whole GPU memory limit ({} of {} bytes on "
                            "GPU {}); its input is {} bytes. A retry cannot succeed. Reduce the input of "
                            "one task (for a sort: smaller partitions, or sort_merge_mode = 'windowed') or "
                            "raise sirius.memory.gpu.usage_limit_bytes{}",
                  ex.what(),
                  orig_task_id,
                  next_retry_count,
                  held,
                  space_max,
                  _memory_space->get_device_id(),
                  input_bytes,
                  spill_note());
                SIRIUS_LOG_ERROR("GPU Pipeline Executor: {}", message);
                if (completion) {
                  completion->report_error(
                    std::make_exception_ptr(std::runtime_error(std::move(message))),
                    transparent::late_failure_cause::oom_exhausted);
                }
                return;
              }
            }
            if (shrank) {
              SIRIUS_LOG_WARN(
                "GPU Pipeline Executor: input of task {} (original task {}) shrank for retry {}; "
                "keeping the reservation floor from before this OOM",
                gpu_task->get_task_id(),
                orig_task_id,
                next_retry_count);
            }

            // Build the rescheduled task via virtual factory (preserves derived type).
            auto new_local_state = std::make_unique<gpu_pipeline_task_local_state>(
              std::move(intermediate_data), ex.get_resume_operator_index());
            new_local_state->retry_count      = next_retry_count;
            new_local_state->original_task_id = orig_task_id;
            if (cur_local) {
              if (shrank) {
                new_local_state->inherit_retry_reservation_floor_before_last_oom(*cur_local);
              } else {
                new_local_state->inherit_retry_reservation_floor(*cur_local);
              }
            }

            // Preserve the per-task device pin across reschedule. Dropping it lets
            // an OOM'd partition task scatter to the wrong GPU and touch a cuco
            // table built on another device (cudaErrorInvalidValue). Only the
            // local_state pin needs copying; a global-state pin already survives.
            if (cur_local && cur_local->get_preferred_device_id().has_value()) {
              new_local_state->set_preferred_device_id(
                cur_local->get_preferred_device_id().value());
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
            _task_creator->reschedule(std::move(new_task));
            return;
          }
          if (auto* pipeline_task = dynamic_cast<sirius_pipeline_itask*>(task.get())) {
            pipeline_task->set_telemetry_finalized();
            pipeline_task->telemetry_fsm().finalizing({
              .success = true,
            });
          }
          // The enclosing leased callable protects the epilogue after task destruction.
          task.reset();

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

namespace {
std::mutex& reservation_policy_mutex()
{
  static std::mutex mutex;
  return mutex;
}
exec::reservation_wait_policy& reservation_policy_storage()
{
  static exec::reservation_wait_policy policy;
  return policy;
}
}  // namespace

void gpu_pipeline_executor::set_reservation_policy(const exec::reservation_wait_policy& policy)
{
  std::lock_guard lock(reservation_policy_mutex());
  reservation_policy_storage() = policy;
}

exec::reservation_wait_policy gpu_pipeline_executor::reservation_policy()
{
  std::lock_guard lock(reservation_policy_mutex());
  return reservation_policy_storage();
}

/// Registers the calling worker as waiting for a reservation for as long as it lives.
class gpu_pipeline_executor::waiter_registration {
 public:
  explicit waiter_registration(gpu_pipeline_executor& executor) : _executor(executor)
  {
    std::lock_guard lock(_executor._waiters_mutex);
    _it = _executor._waiters.insert(_executor._waiters.end(),
                                    waiter_state{std::chrono::steady_clock::now(), 0});
  }
  ~waiter_registration()
  {
    {
      std::lock_guard lock(_executor._waiters_mutex);
      _executor._waiters.erase(_it);
    }
    _executor._waiters_cv.notify_all();
  }
  waiter_registration(const waiter_registration&)            = delete;
  waiter_registration& operator=(const waiter_registration&) = delete;

 private:
  gpu_pipeline_executor& _executor;
  std::list<waiter_state>::iterator _it;
};

void gpu_pipeline_executor::note_bypass()
{
  std::lock_guard lock(_waiters_mutex);
  if (_waiters.empty()) { return; }
  for (auto& waiter : _waiters) {
    ++waiter.bypassed;
  }
  _bypass_dispatches.fetch_add(1, std::memory_order_relaxed);
}

void gpu_pipeline_executor::wait_for_starving_waiters()
{
  using namespace std::chrono_literals;
  std::unique_lock lock(_waiters_mutex);
  if (_waiters.empty()) { return; }
  auto const policy = reservation_policy();
  while (_running.load()) {
    auto const now      = std::chrono::steady_clock::now();
    bool const starving = std::ranges::any_of(_waiters, [&](waiter_state const& waiter) {
      return waiter.bypassed >= policy.max_bypass_tasks ||
             now - waiter.since >= policy.max_bypass_time;
    });
    if (!starving) { return; }
    _waiters_cv.wait_for(lock, 5ms);
  }
}

bool gpu_pipeline_executor::attach_reservation(
  gpu_pipeline_task& task,
  std::unique_ptr<cucascade::memory::reservation> reservation,
  const reservation_size_info& reservation_info)
{
  if (auto* local_state =
        dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(task.local_state())) {
    local_state->set_reservation(std::move(reservation), reservation_info);
    return true;
  }
  SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast local state for task {}",
                   task.get_task_id());
  if (auto handler = task.get_completion_handler()) {
    handler->report_error("GPU Pipeline Executor: Failed to cast local state for task " +
                          std::to_string(task.get_task_id()));
  }
  return false;
}

std::unique_ptr<cucascade::memory::reservation> gpu_pipeline_executor::acquire_reservation(
  gpu_pipeline_task& task, const reservation_wait& wait, bool on_worker)
{
  using namespace std::chrono_literals;
  auto* const gpu_task   = &task;
  auto const bytes_needs = wait.bytes_needs;
  auto const query_id    = wait.query_id;
  auto const policy      = reservation_policy();
  auto const cancelled   = [&] {
    return !_running.load() || !_query_lifecycle.accepts_work(query_id);
  };

  auto memory_wait_activity      = _query_lifecycle.begin_memory_wait(query_id);
  auto const* pipe               = gpu_task->get_pipeline();
  auto const stalled_operator_id = pipe != nullptr
                                     ? pipe->get_source_operator().first
                                     : op::sirius_physical_operator::invalid_operator_id;
  auto const event_query         = make_query_id(
    static_cast<std::uint32_t>(static_cast<std::uint64_t>(gpu_task->get_priority()) >> 32));
  auto const device_id = _memory_space != nullptr ? _memory_space->get_device_id() : -1;
  _query_event_publisher->publish_wait_for_memory_for_task(
    event_query, stalled_operator_id, device_id, bytes_needs);

  // On a worker: poll instead of memory_space::make_reservation(). Its wait wakes one waiter per
  // release, and with several waiting tasks the one woken may not be the one that now fits.
  // "No reservation is active" is what that call treats as idle: nothing will be released, so
  // a partial reservation is all there is.
  auto const wait_or_partial = [&]() -> std::unique_ptr<cucascade::memory::reservation> {
    for (;;) {
      if (auto res = _memory_space->make_reservation_or_null(bytes_needs)) { return res; }
      if (cancelled()) { return nullptr; }
      if (_memory_space->get_active_reservation_count() == 0) {
        return _memory_space->make_reservation_upto(bytes_needs);
      }
      std::this_thread::sleep_for(1ms);
    }
  };

  std::optional<waiter_registration> waiting;
  std::unique_ptr<cucascade::memory::reservation> reservation;
  if (on_worker) {
    waiting.emplace(*this);
    // Let the tasks that are running release memory first, briefly: evicting is the slower way.
    auto const deadline = std::chrono::steady_clock::now() + policy.patience;
    for (;;) {
      reservation = _memory_space->make_reservation_or_null(bytes_needs);
      if (reservation) { return reservation; }
      if (cancelled()) { return nullptr; }
      if (_memory_space->get_active_reservation_count() == 0 ||
          std::chrono::steady_clock::now() >= deadline) {
        break;
      }
      std::this_thread::sleep_for(1ms);
    }
    if (!_downgrade_executor) {
      reservation = wait_or_partial();
      if (!reservation && cancelled()) { return nullptr; }
    }
  } else {
    reservation = _memory_space->make_reservation(bytes_needs);
  }
  if (!reservation && (!on_worker || !_downgrade_executor)) {
    SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to acquire memory reservation for task {}",
                     gpu_task->get_task_id());
    if (auto handler = gpu_task->get_completion_handler()) {
      handler->report_error(
        "GPU Pipeline Executor: Failed to acquire memory reservation for task " +
        std::to_string(gpu_task->get_task_id()));
    }
    return nullptr;
  }
  if (reservation && reservation->size() >= bytes_needs) { return reservation; }
  if (!_downgrade_executor) {
    // No downgrade executor available -- warn and proceed (this should never happen)
    SIRIUS_LOG_WARN(
      "GPU Pipeline Executor: Acquired memory reservation does not match "
      "requested size for pipeline {} of {} bytes needed for task "
      "{}. Reservation size: {}. WARNING: Downgrade executor is not available",
      gpu_task->get_pipeline_id(),
      bytes_needs,
      gpu_task->get_task_id(),
      reservation->size());
    return reservation;
  }

  // What can be had now: the partial reservation, or on a worker (which holds none) what is free.
  size_t const partial_size =
    reservation ? reservation->size()
                : std::min<size_t>(bytes_needs, _memory_space->get_available_memory());
  size_t const shortfall = bytes_needs - partial_size;

  gpu_task->telemetry_fsm().downgrading({
    .shortfall_bytes = shortfall,
    .partial_bytes   = partial_size,
    .manager_thread  = {wait.manager_thread},
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
  _query_event_publisher->publish_memory_downgrade_for_task(
    event_query, stalled_operator_id, device_id, shortfall);

  reservation.reset();  // release partial reservation before downgrade

  std::unique_ptr<cucascade::memory::reservation> new_reservation;
  auto* mem_space = _memory_space;
  size_t freed    = 0;
  std::mutex reservation_mutex;
  try {
    freed = _downgrade_executor
              ->request_downgrade([mem_space, bytes_needs, &new_reservation, &reservation_mutex]() {
                std::lock_guard<std::mutex> lock(reservation_mutex);
                if (new_reservation) { return true; }
                auto res = mem_space->make_reservation_or_null(bytes_needs);
                if (res && res->size() >= bytes_needs) { new_reservation = std::move(res); }
                return new_reservation != nullptr;
              })
              .get();
  } catch (const std::exception& e) {
    if (_query_lifecycle.report_failure(std::current_exception(), query_id)) { throw; }
    // The downgrade request failed or was cancelled. This task cannot get its reservation, so
    // fail its query
    SIRIUS_LOG_INFO("GPU Pipeline Executor: downgrade request failed or cancelled for task {}: {}",
                    gpu_task->get_task_id(),
                    e.what());
    if (auto handler = gpu_task->get_completion_handler()) {
      handler->report_error(
        "GPU Pipeline Executor: downgrade request failed or cancelled for task " +
        std::to_string(gpu_task->get_task_id()) + ": " + e.what());
    }
    return nullptr;
  }

  if (new_reservation) {
    reservation = std::move(new_reservation);
  } else if (on_worker) {
    // Predicate never succeeded: wait for the running tasks, then take what there is.
    reservation = wait_or_partial();
    if (!reservation && cancelled()) { return nullptr; }
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
    return nullptr;
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
  return reservation;
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
  return {_tasks_executed.load(std::memory_order_relaxed),
          _reservation_waits.load(std::memory_order_relaxed),
          _bypass_dispatches.load(std::memory_order_relaxed)};
}

}  // namespace pipeline
}  // namespace sirius
