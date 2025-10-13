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
struct DummyTaskGlobalState : public ITaskGlobalState {
  std::atomic<int> counter{0};
};

struct DummyTaskLocalState : public ITaskLocalState {
  explicit DummyTaskLocalState(int id) : id_(id) {}
  int id_;
};

class DummyTask : public ITask {
public:
  DummyTask(std::unique_ptr<DummyTaskLocalState> local_state, std::shared_ptr<DummyTaskGlobalState> global_state)
    : ITask(std::move(local_state), std::move(global_state)) {}

  void Execute() override {
    auto* g = static_cast<DummyTaskGlobalState*>(global_state_.get());
    auto* l = static_cast<DummyTaskLocalState*>(local_state_.get());
    // Simulate work
    std::this_thread::sleep_for(10ms);
    g->counter.fetch_add(1 + l->id_);
  }
};

/**
 * Dummy task queue for tests.
 */
class DummyTaskQueue : public ITaskQueue {
public:
  ~DummyTaskQueue() override = default;

  void Open() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      open_ = true;
    }
    cv_.notify_all();
  }

  void Close() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      open_ = false;
    }
    cv_.notify_all();
  }

  void Push(std::unique_ptr<ITask> task) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!open_) {
        return;
      }
      tasks_.push(std::move(task));
    }
    cv_.notify_one();
  }

  std::unique_ptr<ITask> Pull() override {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&]() {
      return !tasks_.empty() || !open_;
    });
    if (tasks_.empty()) {
      return nullptr;
    }
    auto task = std::move(tasks_.front());
    tasks_.pop();
    return task;
  }

private:
  std::queue<std::unique_ptr<ITask>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool open_ = false;
};

/**
 * Dummy task executor for tests.
 */
class DummyTaskExecutor : public ITaskExecutor {
public:
  using ITaskExecutor::ITaskExecutor;
};

TEST_CASE("Executor can start and stop gracefully", "[task_executor]") {
  auto queue = std::make_unique<DummyTaskQueue>();
  TaskExecutorConfig config{4, false};
  DummyTaskExecutor executor(std::move(queue), config);

  REQUIRE_NOTHROW(executor.Start());
  REQUIRE_NOTHROW(executor.Stop());
}

TEST_CASE("Executor executes scheduled tasks", "[task_executor]") {
  auto queue = std::make_unique<DummyTaskQueue>();
  auto g = std::make_shared<DummyTaskGlobalState>();
  TaskExecutorConfig config{4, false};
  DummyTaskExecutor executor(std::move(queue), config);
  REQUIRE_NOTHROW(executor.Start());

  // Schedule some tasks
  int num_tasks = 20;
  for (int i = 0; i < num_tasks; ++i) {
    executor.Schedule(std::make_unique<DummyTask>(std::make_unique<DummyTaskLocalState>(i), g));
  }

  // Wait and check the result
  executor.Wait();
  int expected_counter = num_tasks * (num_tasks + 1) / 2;
  REQUIRE(g->counter.load() == expected_counter);

  REQUIRE_NOTHROW(executor.Stop());
}

