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

#include "pipeline/gpu_pipeline_queue.hpp"
#include "config.hpp"

namespace sirius {
namespace parallel {

void gpu_pipeline_queue::open() {
    _is_open.store(true, std::memory_order_release);
}

void gpu_pipeline_queue::close() {
    _is_open.store(false, std::memory_order_release);
    // Wake up all threads blocked in wait_dequeue by pushing nullptr sentinels
    for (size_t i = 0; i < _num_threads; ++i) {
        _task_queue.enqueue(nullptr);
    }
}

void gpu_pipeline_queue::push(sirius::unique_ptr<itask> task) {
    _task_queue.enqueue(std::move(task));
}

sirius::unique_ptr<itask> gpu_pipeline_queue::pull() {
    sirius::unique_ptr<itask> task;
    while (true) {
        if (_task_queue.try_dequeue(task)) { return task; }

        // If the queue is closed and empty, return nullptr to indicate no more tasks.
        if (!_is_open.load(std::memory_order_acquire)) { return nullptr; }

        // Otherwise, wait for a task to become available.
        _task_queue.wait_dequeue(task);
        if (task) { return task; }
    }
}

} // namespace parallel
} // namespace sirius

