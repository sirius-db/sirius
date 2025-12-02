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

#include "pipeline/task_request.hpp"
#include "config.hpp"

namespace sirius {
namespace parallel {

void task_request_queue::open() {
   _is_open.store(true, std::memory_order_release);
}

void task_request_queue::close() {
    _is_open.store(false, std::memory_order_release);
    // Wake up all threads blocked in wait_dequeue by pushing nullptr sentinels
    for (size_t i = 0; i < _num_threads; ++i) {
        _request_queue.enqueue(nullptr);
    }
}

void task_request_queue::push(unique_ptr<task_request> request) {
    _request_queue.enqueue(std::move(request));
}

unique_ptr<task_request> task_request_queue::pull() {
    unique_ptr<task_request> request;
    while (true) {
        if (_request_queue.try_dequeue(request)) { return request; }

        // If the queue is closed and empty, return nullptr to indicate no more tasks.
        if (!_is_open.load(std::memory_order_acquire)) { return nullptr; }

        // Otherwise, wait for a task to become available.
        _request_queue.wait_dequeue(request);
        if (request) { return request; }
    }
}

} // namespace parallel
} // namespace sirius

