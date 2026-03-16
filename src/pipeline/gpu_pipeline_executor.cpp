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
#include "log/logging.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/task_request.hpp"

#include <rmm/cuda_device.hpp>

#include <util/stream_check_wrapper.hpp>

namespace sirius {
namespace pipeline {

gpu_pipeline_executor::gpu_pipeline_executor(
  exec::thread_pool_config config,
  cucascade::memory::memory_space* mem_space,
  exec::publisher<std::unique_ptr<task_request>> task_request_publisher)
  : _config(config),
    _stream_pool(rmm::cuda_device_id{mem_space->get_device_id()}, _config.num_threads),
    _task_request_publisher(std::move(task_request_publisher)),
    _memory_space(mem_space),
    _kiosk(_config.num_threads)
{
}

gpu_pipeline_executor::~gpu_pipeline_executor() { stop(); }

void gpu_pipeline_executor::schedule(std::unique_ptr<sirius::parallel::itask> task)
{
  _task_queue.push(std::move(task));
}

void gpu_pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  _thread_pool =
    std::make_unique<exec::thread_pool>(_config.num_threads,
                                        _config.thread_name_prefix,
                                        _config.cpu_affinity_list,
                                        [device_id = _memory_space->get_device_id()]() noexcept {
                                          cudaSetDevice(device_id);
                                          sirius::util::enable_log_on_default_stream();
                                        });
  _manager_thread = std::thread(&gpu_pipeline_executor::manager_loop, this);
}

void gpu_pipeline_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  _kiosk.stop();
  _task_queue.interrupt();
  if (_manager_thread.joinable()) { _manager_thread.join(); }
  _kiosk.wait_all();
  if (_thread_pool) { _thread_pool->stop(); }
}

void gpu_pipeline_executor::manager_loop()
{
  rmm::cuda_set_device_raii set_device_guard(rmm::cuda_device_id{_memory_space->get_device_id()});
  sirius::util::enable_log_on_default_stream();
  while (_running.load()) {
    auto ticket = _kiosk.acquire();  // block till a thread is available
    if (!ticket.is_valid()) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: Kiosk interrupted, stopping manager loop");
      break;
    }
    if (!_task_request_publisher.send(
          std::make_unique<pipeline::task_request>(_memory_space->get_device_id(), false))) {
      SIRIUS_LOG_INFO("GPU Pipeline Executor: Failed to send task request, channel is closed");
      break;
    }
    auto pipeline_task = _task_queue.pop();  // block till a task is available
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
    auto bytes_needs = gpu_task->get_estimated_reservation_size();
    SIRIUS_LOG_TRACE(
      "GPU Pipeline Executor: Acquiring memory reservation of {} bytes for task {}. Memory "
      "available: {}, total reserved: {}, max: {}",
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
    }
    if (auto* local_state = dynamic_cast<sirius::pipeline::sirius_pipeline_task_local_state*>(
          gpu_task->local_state())) {
      local_state->set_reservation(std::move(reservation));
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
    _thread_pool->schedule([this,
                            task       = std::move(pipeline_task),
                            ticket     = std::move(ticket),
                            exc_stream = std::move(exc_stream),
                            consumers  = std::move(output_consumers),
                            pipeline]() mutable {
      try {
        task->execute(exc_stream);
      } catch (oom_reschedule_exception& oom) {
        if (_completion_handler && _completion_handler->has_error()) {
          // If the completion handler is already in an error state, then we can just return and not
          // try to reschedule
          return;
        }
        auto* gpu_task = cast_to_gpu_pipeline_task(task.get());
        if (!gpu_task) {
          SIRIUS_LOG_ERROR("GPU Pipeline Executor: Failed to cast task for OOM reschedule");
          if (_completion_handler) {
            _completion_handler->report_error(
              "GPU Pipeline Executor: Failed to cast task for OOM reschedule");
          }
          return;
        }

        // Determine retry count and original task ID for this rescheduled attempt.
        auto* cur_local = dynamic_cast<gpu_pipeline_task_local_state*>(gpu_task->local_state());
        uint32_t next_retry_count = 1;
        uint64_t orig_task_id     = gpu_task->get_task_id();
        if (cur_local && cur_local->retry_count > 0) {
          next_retry_count = cur_local->retry_count + 1;
          orig_task_id     = cur_local->original_task_id;
        }

        static constexpr uint32_t MAX_OOM_RETRIES = 10;
        if (next_retry_count > MAX_OOM_RETRIES) {
          SIRIUS_LOG_ERROR(
            "GPU Pipeline Executor: task {} (original task {}) exceeded {} OOM retries at "
            "operator index {} — terminating query",
            gpu_task->get_task_id(),
            orig_task_id,
            MAX_OOM_RETRIES,
            oom.get_resume_operator_index());
          if (_completion_handler) {
            _completion_handler->report_error(std::make_exception_ptr(
              std::runtime_error("GPU pipeline task exceeded maximum OOM retry limit (" +
                                 std::to_string(MAX_OOM_RETRIES) + ") for original task " +
                                 std::to_string(orig_task_id))));
          }
          return;
        }

        SIRIUS_LOG_WARN(
          "GPU Pipeline Executor: OOM reschedule (retry {}/{}) for task {} (original task {}), "
          "resuming from operator index {}",
          next_retry_count,
          MAX_OOM_RETRIES,
          gpu_task->get_task_id(),
          orig_task_id,
          oom.get_resume_operator_index());

        // Mark old task as rescheduled so its destructor skips mark_task_completed().
        // The rescheduled task will handle completion instead.
        gpu_task->mark_as_rescheduled();

        // Prepare intermediate data batches for re-processing.
        // Operator outputs are in idle state; transition to task_created so
        // lock_or_prepare_batch() can lock them for the rescheduled task.
        auto intermediate_data = oom.release_intermediate_data();
        if (intermediate_data) {
          for (auto& batch : intermediate_data->get_data_batches()) {
            if (batch) { batch->try_to_create_task(); }
          }
        }

        // Build the rescheduled task via virtual factory (preserves derived type).
        auto new_local_state = std::make_unique<gpu_pipeline_task_local_state>(
          std::move(intermediate_data), oom.get_resume_operator_index());
        new_local_state->retry_count      = next_retry_count;
        new_local_state->original_task_id = orig_task_id;

        auto new_task_id =
          _task_creator ? _task_creator->get_next_task_id() : gpu_task->get_task_id();
        auto new_task = gpu_task->create_rescheduled_task(new_task_id, std::move(new_local_state));

        // Brief backoff before rescheduling to allow other tasks to complete
        // and free memory. Without this, a rescheduled task can spin in a tight
        // OOM → reschedule → OOM loop consuming CPU without making progress.
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        // Schedule the rescheduled task. It goes back through manager_loop()
        // to acquire a fresh reservation before execution.
        this->schedule(std::move(new_task));
        return;
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("GPU Pipeline Executor: Exception during task execution: {}", e.what());
        if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
        return;
      } catch (...) {
        SIRIUS_LOG_ERROR("GPU Pipeline Executor: unknown error during task execution");
        if (_completion_handler) { _completion_handler->report_error(std::current_exception()); }
        return;
      }
      task.reset();

      // Check if query is complete BEFORE scheduling downstream tasks.
      // mark_completed() signals the future that engine.execute() is waiting on,
      // which may destroy the engine and its operators. We must not schedule
      // tasks that reference those operators after signaling completion.
      bool query_complete = false;
      if (_completion_handler && pipeline) {
        auto sink = pipeline->get_sink();
        if (sink && sink->type == op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
          query_complete = pipeline->is_pipeline_finished();
        }
      }

      if (!query_complete && _task_creator) {
        bool pipeline_done = pipeline && pipeline->is_pipeline_finished();
        for (auto* consumer : consumers) {
          if (consumer) { _task_creator->schedule(consumer); }
        }
      }

      if (query_complete && _completion_handler) { _completion_handler->mark_completed(); }
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

void gpu_pipeline_executor::drain_leftover_tasks() { _task_queue.drain(); }

void gpu_pipeline_executor::drain_and_wait()
{
  // Stop the kiosk so acquire() unblocks immediately with an invalid ticket.
  // This is necessary when the manager loop is blocked at acquire() (i.e. all
  // thread-pool slots are full) — interrupting the task queue alone won't help.
  _kiosk.stop();

  // Interrupt pop() so the manager_loop sees a nullptr return and breaks out
  // of its while loop (within at most the 10 ms poll interval in pop()).
  _task_queue.interrupt();
  _task_queue.drain();

  // Join the manager thread so we know it has exited (it will have seen either
  // an invalid ticket from the stopped kiosk or a nullptr from the interrupted queue).
  if (_manager_thread.joinable()) { _manager_thread.join(); }

  // Wait for all in-flight thread-pool tasks to finish (each holds a kiosk
  // ticket that is released when the lambda completes).
  _kiosk.wait_all();

  // Re-enable the kiosk so the restarted manager can acquire tickets again.
  _kiosk.resume();

  // Reset the queue and restart the manager so the executor is ready for
  // the next query.
  _task_queue.reset();
  _manager_thread = std::thread(&gpu_pipeline_executor::manager_loop, this);
}

bool gpu_pipeline_executor::is_task_queue_empty() const noexcept { return _task_queue.is_empty(); }

void gpu_pipeline_executor::set_completion_handler(completion_handler* handler) noexcept
{
  _completion_handler = handler;
}

}  // namespace pipeline
}  // namespace sirius
