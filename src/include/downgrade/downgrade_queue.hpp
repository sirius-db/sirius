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

#pragma once
#include "parallel/task_executor.hpp"
#include "task_completion.hpp"
#include "helper/helper.hpp"
#include "data/data_repository.hpp"
#include "config.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief A thread-safe task queue specifically for managing downgrade_task instances.
 * 
 * This class provides a thread-safe queue implementation for scheduling and retrieving
 * downgrade tasks. It uses mutexes and semaphores to ensure safe concurrent access.
 * Currently, it uses a simple FIFO queue (sirius::queue), but future versions may
 * implement more sophisticated scheduling algorithms such as priority scheduling,
 * task stealing, or work balancing.
 */
class downgrade_task_queue : public itask_queue {
public:
    /**
     * @brief Constructs a new downgrade_task_queue object
     * 
     * Initializes an empty queue in closed state. Call open() to begin accepting tasks.
     */
    downgrade_task_queue() = default;

    /**
     * @brief Opens the task queue to start accepting and returning tasks
     * 
     * After calling this method, the queue will accept new tasks via push() and
     * return tasks via pull(). This method is thread-safe.
     */
    void open() override {
        sirius::lock_guard<sirius::mutex> lock(_mutex);
        _is_open = true;
    }

    /**
     * @brief Closes the task queue, stopping task acceptance and retrieval
     * 
     * After calling this method, the queue will no longer accept new tasks or
     * return tasks. Any threads waiting on pull() will be signaled to wake up.
     * This method is thread-safe.
     */
    void close() override {
        // Wake up all waiting threads by releasing the semaphore enough times
        for (int i = 0; i < Config::NUM_DOWNGRADE_EXECUTOR_THREADS; ++i) {
            _sem.release();
        }
        sirius::lock_guard<sirius::mutex> lock(_mutex);
        _is_open = false;
    }

    /**
     * @brief Pushes a new task to be scheduled (base interface implementation)
     * 
     * @param task The task to be scheduled (must be a downgrade_task)
     * @throws std::runtime_error If the queue is not currently open for accepting tasks
     */
    void push(sirius::unique_ptr<itask> task) override {
        // Convert itask to downgrade_task - since we know it's a downgrade_task
        auto downgrade_task = sirius::unique_ptr<downgrade_task>(static_cast<downgrade_task*>(task.release()));
        push(std::move(downgrade_task));
    }

    /**
     * @brief Downgrade-specific push overload for type safety and convenience
     * 
     * @param downgrade_task The downgrade task to be scheduled
     * @throws std::runtime_error If the queue is not currently open for accepting tasks
     */
    void push(sirius::unique_ptr<downgrade_task> downgrade_task) {
        enqueue_task(std::move(downgrade_task));
    }
    
    /**
     * @brief Pulls a task to execute (base interface implementation)
     * 
     * This is a blocking call that will wait until a task becomes available or the
     * queue is closed. Returns nullptr if the queue is closed and no tasks are available.
     * 
     * @return A unique pointer to the task to execute, or nullptr if none available
     * @throws std::runtime_error If the queue is closed and not returning tasks
     */
    sirius::unique_ptr<itask> pull() override {
        // Delegate to downgrade-specific version and return as base type
        auto downgrade_task = pull_downgrade_task();
        return std::move(downgrade_task);
    }

    /**
     * @brief Downgrade-specific pull method for type safety and convenience  
     * 
     * This is a blocking call that will wait until a task becomes available or the
     * queue is closed. Returns nullptr if the queue is closed and no tasks are available.
     * 
     * @return A unique pointer to the downgrade task to execute, or nullptr if none available
     * @throws std::runtime_error If the queue is closed and not returning tasks
     */
    sirius::unique_ptr<downgrade_task> pull_downgrade_task() {
        return dequeue_task();
    }

    /**
     * @brief Enqueues a downgrade task into the queue
     * 
     * This is the internal implementation for adding tasks to the queue.
     * It's thread-safe and will only accept tasks if the queue is open.
     * 
     * @param downgrade_task The downgrade task to enqueue
     */
    void enqueue_task(sirius::unique_ptr<downgrade_task> downgrade_task) {
        sirius::lock_guard<sirius::mutex> lock(_mutex);
        if (downgrade_task && _is_open) {
            _task_queue.push(std::move(downgrade_task));
        }
        _sem.release(); // signal that one item is available
    }

    /**
     * @brief Dequeues a downgrade task from the queue
     * 
     * This is the internal implementation for retrieving tasks from the queue.
     * It's thread-safe and will block until a task is available or the queue is closed.
     * 
     * @return A unique pointer to the dequeued downgrade task, or nullptr if queue is empty and closed
     */
    sirius::unique_ptr<downgrade_task> dequeue_task() {
        _sem.acquire(); // wait until there's something
        sirius::lock_guard<sirius::mutex> lock(_mutex);
        // If the queue is closed and empty, return nullptr to signal shutdown
        if (!_is_open && _task_queue.empty()) {
            return nullptr;
        }
        
        // If there's a task available, return it
        if (!_task_queue.empty()) {
            auto task = std::move(_task_queue.front());
            _task_queue.pop();
            return task;
        }
        
        // Queue is empty but might be open (spurious semaphore release from close())
        return nullptr;
    }

    /**
     * @brief Checks if the task queue is empty
     * 
     * @return true if the queue contains no tasks, false otherwise
     */
    bool is_empty() const {
        sirius::lock_guard<sirius::mutex> lock(_mutex);
        return _task_queue.empty();
    }
    
private:
    sirius::queue<sirius::unique_ptr<downgrade_task>> _task_queue;  ///< FIFO queue storing the downgrade tasks
    bool _is_open = false;                                         ///< Whether the queue is open for operations
    mutable sirius::mutex _mutex;                                  ///< Mutex for thread-safe access (mutable for const methods)
    std::counting_semaphore<> _sem{0};                             ///< Semaphore for blocking/signaling (starts with 0 permits)
};

} // namespace parallel
} // namespace sirius