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

#include "pipeline/gpu_pipeline_queue.hpp"
#include "pipeline/pipeline_executor.hpp"

namespace sirius {
namespace parallel {

void local_task_buffer::produce(sirius::unique_ptr<itask> task)
{
  {
    std::lock_guard<std::mutex> lock(_mtx);
    _queue.push(std::move(task));
  }
  _cv.notify_one();  // wake consumer
}

sirius::unique_ptr<itask> local_task_buffer::consume()
{
  std::unique_lock<std::mutex> lock(_mtx);
  _cv.wait(lock, [&] { return (!_queue.empty()) || !_is_open.load(std::memory_order_acquire); });
  auto task = std::move(_queue.front());
  _queue.pop();
  return std::move(task);
}

void local_task_buffer::open() { _is_open.store(true, std::memory_order_release); }

void local_task_buffer::close()
{
  _is_open.store(false, std::memory_order_release);
  _cv.notify_all();
}

gpu_pipeline_executor::gpu_pipeline_executor(task_executor_config config,
                                             const memory::memory_space* mem_space,
                                             pipeline_executor* pipeline_exec)
  : itask_executor(sirius::make_unique<gpu_pipeline_queue>(config.num_threads), config),
    _local_task_buffer(sirius::make_unique<local_task_buffer>()),
    _memory_space_view(mem_space),
    _pipeline_exec(pipeline_exec)
{
}

void gpu_pipeline_executor::schedule(sirius::unique_ptr<itask> task)
{
  _task_queue->push(std::move(task));
}

void gpu_pipeline_executor::on_start()
{
  _task_queue->open();
  _local_task_buffer->open();
}

void gpu_pipeline_executor::on_stop()
{
  _task_queue->close();
  _local_task_buffer->close();
}

void gpu_pipeline_executor::start()
{
  bool expected = false;
  if (!_running.compare_exchange_strong(expected, true)) { return; }
  on_start();
  _threads.reserve(_config.num_threads);
  for (int i = 0; i < _config.num_threads; ++i) {
    _threads.push_back(sirius::make_unique<task_executor_thread>(
      sirius::make_unique<sirius::thread>(&gpu_pipeline_executor::worker_loop, this, i)));
  }
  _gpu_pipeline_executor_manager_thread =
    sirius::make_unique<sirius::thread>(&gpu_pipeline_executor::manager_loop, this);
}

void gpu_pipeline_executor::stop()
{
  bool expected = true;
  if (!_running.compare_exchange_strong(expected, false)) { return; }
  on_stop();
  for (auto& thread : _threads) {
    if (thread->_internal_thread->joinable()) { thread->_internal_thread->join(); }
  }
  if (_gpu_pipeline_executor_manager_thread->joinable()) {
    _gpu_pipeline_executor_manager_thread->join();
  }
  _gpu_pipeline_executor_manager_thread.reset();
  _threads.clear();
}

void gpu_pipeline_executor::worker_loop(int worker_id)
{
  while (true) {
    if (!_running.load()) {
      // Executor is stopped.
      break;
    }
    auto task = _task_queue->pull();
    if (task == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      // TODO:
      // if reservation hasn't been made, request reservation (blocking)
      // set stream reservation
      task->execute();
      // reset memory resource
    } catch (const std::exception& e) {
      on_task_error(worker_id, std::move(task), e);
    }
  }
}

void gpu_pipeline_executor::submit_task_request(sirius::unique_ptr<task_request> request)
{
  _pipeline_exec->submit_task_request(std::move(request));
}

void gpu_pipeline_executor::manager_loop()
{
  while (true) {
    if (!_running.load()) {
      // Executor is stopped.
      break;
    }
    auto task = _local_task_buffer->consume();
    if (task == nullptr) {
      // Task queue is closed.
      break;
    }
    try {
      schedule(std::move(task));
    } catch (const std::exception& e) {
      throw e;
    }
  }
}

gpu_pipeline_task* gpu_pipeline_executor::cast_to_gpu_pipeline_task(itask* task)
{
  // Safely cast to gpu_pipeline_task
  return dynamic_cast<gpu_pipeline_task*>(task);
}

}  // namespace parallel
}  // namespace sirius
