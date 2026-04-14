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

namespace sirius {
namespace parallel {

void itask_executor::schedule(std::unique_ptr<itask> task)
{
  static_cast<void>(_task_queue.push(std::move(task)));
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

}  // namespace parallel
}  // namespace sirius
