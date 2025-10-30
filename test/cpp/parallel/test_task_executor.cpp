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

#include "catch.hpp"
#include "parallel/task_executor.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

using namespace sirius::parallel;
using namespace std::chrono_literals;

/**
 * Dummy task for tests.
 */
struct dummy_task_global_state : public itask_global_state {
  std::atomic<int> counter{0};
};

struct dummy_task_local_state : public itask_local_state {
  explicit dummy_task_local_state(int id) : _id(id) {}
  int _id;
};

class dummy_task : public itask {
public:
  dummy_task(std::unique_ptr<dummy_task_local_state> local_state, std::shared_ptr<dummy_task_global_state> global_state)
    : itask(std::move(local_state), std::move(global_state)) {}

  void execute() override {
    auto* g = static_cast<dummy_task_global_state*>(_global_state.get());
    auto* l = static_cast<dummy_task_local_state*>(_local_state.get());
    // Simulate work
    std::this_thread::sleep_for(10ms);
    g->counter.fetch_add(1 + l->_id);
  }
};

/**
 * Dummy task queue for tests.
 */
class dummy_task_queue : public itask_queue {
public:
  ~dummy_task_queue() override = default;

  void open() override {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _open = true;
    }
    _cv.notify_all();
  }

  void close() override {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _open = false;
    }
    _cv.notify_all();
  }

  void push(std::unique_ptr<itask> task) override {
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (!_open) {
        return;
      }
      _tasks.push(std::move(task));
    }
    _cv.notify_one();
  }

  std::unique_ptr<itask> pull() override {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [&]() {
      return !_tasks.empty() || !_open;
    });
    if (_tasks.empty()) {
      return nullptr;
    }
    auto task = std::move(_tasks.front());
    _tasks.pop();
    return task;
  }

private:
  std::queue<std::unique_ptr<itask>> _tasks;
  std::mutex _mutex;
  std::condition_variable _cv;
  bool _open = false;
};

/**
 * Dummy task executor for tests.
 */
class dummy_task_executor : public itask_executor {
public:
  using itask_executor::itask_executor;
};

TEST_CASE("Executor can start and stop gracefully", "[task_executor]") {
  auto queue = std::make_unique<dummy_task_queue>();
  task_executor_config config{4, false};
  dummy_task_executor executor(std::move(queue), config);

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("Executor executes scheduled tasks", "[task_executor]") {
  auto queue = std::make_unique<dummy_task_queue>();
  auto g = std::make_shared<dummy_task_global_state>();
  task_executor_config config{4, false};
  dummy_task_executor executor(std::move(queue), config);
  REQUIRE_NOTHROW(executor.start());

  // Schedule some tasks
  int num_tasks = 20;
  for (int i = 0; i < num_tasks; ++i) {
    executor.schedule(std::make_unique<dummy_task>(std::make_unique<dummy_task_local_state>(i), g));
  }

  // Wait for tasks to complete by polling the counter
  int expected_counter = num_tasks * (num_tasks + 1) / 2;
  auto start_time = std::chrono::steady_clock::now();
  auto timeout = std::chrono::seconds(5);
  while (g->counter.load() < expected_counter) {
    std::this_thread::sleep_for(50ms);
    if (std::chrono::steady_clock::now() - start_time > timeout) {
      FAIL("Test timed out waiting for tasks to complete");
    }
  }
  REQUIRE(g->counter.load() == expected_counter);

  REQUIRE_NOTHROW(executor.stop());
}

