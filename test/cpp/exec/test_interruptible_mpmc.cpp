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
#include "exec/interruptible_mpmc.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

// =============================================================================
// Test payload types
// =============================================================================

struct test_payload {
  int id;
  std::string data;

  test_payload(int i, std::string d) : id(i), data(std::move(d)) {}
};

// =============================================================================
// Basic functionality tests with unique_ptr
// =============================================================================

TEST_CASE("interruptible_mpmc basic push and pop with unique_ptr", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  REQUIRE(queue.is_open());
  REQUIRE(queue.push(std::make_unique<int>(42)));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(*result == 42);
}

TEST_CASE("interruptible_mpmc push and pop with unique_ptr custom type", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<test_payload>> queue;

  REQUIRE(queue.push(std::make_unique<test_payload>(1, "hello")));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(result->id == 1);
  REQUIRE(result->data == "hello");
}

TEST_CASE("interruptible_mpmc try_pop returns nullptr on empty queue", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  auto result = queue.try_pop();
  REQUIRE(result == nullptr);
}

TEST_CASE("interruptible_mpmc multiple items FIFO order", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  for (int i = 0; i < 10; ++i) {
    REQUIRE(queue.push(std::make_unique<int>(i)));
  }

  for (int i = 0; i < 10; ++i) {
    auto result = queue.try_pop();
    REQUIRE(result != nullptr);
    REQUIRE(*result == i);
  }
}

// =============================================================================
// Basic functionality tests with shared_ptr
// =============================================================================

TEST_CASE("interruptible_mpmc basic push and pop with shared_ptr", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::shared_ptr<int>> queue;

  REQUIRE(queue.is_open());
  REQUIRE(queue.push(std::make_shared<int>(42)));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(*result == 42);
}

TEST_CASE("interruptible_mpmc push and pop with shared_ptr custom type", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::shared_ptr<test_payload>> queue;

  REQUIRE(queue.push(std::make_shared<test_payload>(1, "world")));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(result->id == 1);
  REQUIRE(result->data == "world");
}

TEST_CASE("interruptible_mpmc shared_ptr maintains reference count", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::shared_ptr<int>> queue;

  auto ptr = std::make_shared<int>(123);
  REQUIRE(ptr.use_count() == 1);

  REQUIRE(queue.push(ptr));
  REQUIRE(ptr.use_count() == 2);

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(*result == 123);
  REQUIRE(ptr.use_count() == 2);  // Both ptr and result point to same object
}

// =============================================================================
// Emplace tests
// =============================================================================

TEST_CASE("interruptible_mpmc emplace constructs in-place", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<test_payload>> queue;

  REQUIRE(queue.emplace(42, "emplaced"));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(result->id == 42);
  REQUIRE(result->data == "emplaced");
}

TEST_CASE("interruptible_mpmc emplace multiple items", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<test_payload>> queue;

  for (int i = 0; i < 5; ++i) {
    REQUIRE(queue.emplace(i, "item_" + std::to_string(i)));
  }

  for (int i = 0; i < 5; ++i) {
    auto result = queue.try_pop();
    REQUIRE(result != nullptr);
    REQUIRE(result->id == i);
    REQUIRE(result->data == "item_" + std::to_string(i));
  }
}

TEST_CASE("interruptible_mpmc emplace fails after interrupt", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.emplace(42));
}

// =============================================================================
// Interruption tests
// =============================================================================

TEST_CASE("interruptible_mpmc interrupt closes queue", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  REQUIRE(queue.is_open());
  queue.interrupt();
  REQUIRE_FALSE(queue.is_open());
}

TEST_CASE("interruptible_mpmc push fails after interrupt", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.push(std::make_unique<int>(42)));
}

TEST_CASE("interruptible_mpmc blocking pop returns nullptr after interrupt", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;
  std::atomic<bool> pop_returned{false};
  std::unique_ptr<int> pop_result;

  std::thread consumer([&]() {
    pop_result   = queue.pop();
    pop_returned = true;
  });

  // Give the consumer time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE_FALSE(pop_returned.load());

  // Interrupt should unblock the consumer
  queue.interrupt();

  // Wait for consumer to return
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (!pop_returned.load()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      consumer.detach();
      FAIL("Timeout waiting for pop to return after interrupt");
    }
  }

  consumer.join();
  REQUIRE(pop_result == nullptr);
}

TEST_CASE("interruptible_mpmc interrupt wakes up multiple blocked consumers",
          "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;
  constexpr int num_consumers = 4;
  std::atomic<int> consumers_returned{0};

  std::vector<std::thread> consumers;
  for (int i = 0; i < num_consumers; ++i) {
    consumers.emplace_back([&]() {
      auto result = queue.pop();
      REQUIRE(result == nullptr);
      consumers_returned.fetch_add(1);
    });
  }

  // Give consumers time to block
  std::this_thread::sleep_for(50ms);
  REQUIRE(consumers_returned.load() == 0);

  // Interrupt should wake all consumers
  queue.interrupt();

  // Wait for all consumers to return
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 2s;
  while (consumers_returned.load() < num_consumers) {
    std::this_thread::sleep_for(20ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      for (auto& c : consumers) {
        if (c.joinable()) c.detach();
      }
      FAIL("Timeout waiting for all consumers to return");
    }
  }

  for (auto& c : consumers) {
    c.join();
  }
  REQUIRE(consumers_returned.load() == num_consumers);
}

TEST_CASE("interruptible_mpmc reset after interrupt re-enables queue", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.is_open());
  REQUIRE_FALSE(queue.push(std::make_unique<int>(1)));

  queue.reset();
  REQUIRE(queue.is_open());
  REQUIRE(queue.push(std::make_unique<int>(42)));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(*result == 42);
}

// =============================================================================
// Multi-threaded tests
// =============================================================================

TEST_CASE("interruptible_mpmc concurrent producers and consumers", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;
  constexpr int num_producers      = 4;
  constexpr int num_consumers      = 4;
  constexpr int items_per_producer = 100;
  constexpr int total_items        = num_producers * items_per_producer;

  std::atomic<int> produced_count{0};
  std::atomic<int> consumed_count{0};
  std::atomic<bool> stop_consumers{false};

  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  // Start consumers
  for (int i = 0; i < num_consumers; ++i) {
    consumers.emplace_back([&]() {
      while (!stop_consumers.load()) {
        auto result = queue.try_pop();
        if (result != nullptr) {
          consumed_count.fetch_add(1);
        } else {
          std::this_thread::yield();
        }
      }
      // Drain remaining items
      while (auto result = queue.try_pop()) {
        consumed_count.fetch_add(1);
      }
    });
  }

  // Start producers
  for (int i = 0; i < num_producers; ++i) {
    producers.emplace_back([&, producer_id = i]() {
      for (int j = 0; j < items_per_producer; ++j) {
        int value = producer_id * items_per_producer + j;
        while (!queue.push(std::make_unique<int>(value))) {
          if (!queue.is_open()) return;
          std::this_thread::yield();
        }
        produced_count.fetch_add(1);
      }
    });
  }

  // Wait for all producers to finish
  for (auto& p : producers) {
    p.join();
  }

  // Wait for all items to be consumed
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      stop_consumers = true;
      for (auto& c : consumers) {
        if (c.joinable()) c.detach();
      }
      FAIL("Timeout waiting for consumption");
    }
  }

  // Stop consumers
  stop_consumers = true;
  for (auto& c : consumers) {
    c.join();
  }

  REQUIRE(produced_count.load() == total_items);
  REQUIRE(consumed_count.load() == total_items);
}

TEST_CASE("interruptible_mpmc concurrent emplace and pop", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<test_payload>> queue;
  constexpr int num_producers      = 4;
  constexpr int items_per_producer = 50;
  constexpr int total_items        = num_producers * items_per_producer;

  std::atomic<int> produced_count{0};
  std::atomic<int> consumed_count{0};
  std::atomic<bool> stop_consumer{false};

  std::thread consumer([&]() {
    while (!stop_consumer.load() || consumed_count.load() < total_items) {
      auto result = queue.try_pop();
      if (result != nullptr) {
        consumed_count.fetch_add(1);
      } else {
        std::this_thread::yield();
      }
    }
  });

  std::vector<std::thread> producers;
  for (int i = 0; i < num_producers; ++i) {
    producers.emplace_back([&, producer_id = i]() {
      for (int j = 0; j < items_per_producer; ++j) {
        int value = producer_id * items_per_producer + j;
        while (!queue.emplace(value, "data_" + std::to_string(value))) {
          if (!queue.is_open()) return;
          std::this_thread::yield();
        }
        produced_count.fetch_add(1);
      }
    });
  }

  for (auto& p : producers) {
    p.join();
  }

  // Wait for consumption
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      stop_consumer = true;
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout");
    }
  }

  stop_consumer = true;
  consumer.join();

  REQUIRE(produced_count.load() == total_items);
  REQUIRE(consumed_count.load() == total_items);
}

TEST_CASE("interruptible_mpmc blocking pop receives pushed items", "[interruptible_mpmc]")
{
  interruptible_mpmc<std::unique_ptr<int>> queue;
  std::atomic<int> received_value{0};
  std::atomic<bool> received{false};

  std::thread consumer([&]() {
    auto result = queue.pop();
    if (result != nullptr) {
      received_value = *result;
      received       = true;
    }
  });

  // Give consumer time to start blocking
  std::this_thread::sleep_for(50ms);

  // Push an item
  REQUIRE(queue.push(std::make_unique<int>(999)));

  // Wait for consumer to receive
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 1s;
  while (!received.load()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      queue.interrupt();
      consumer.detach();
      FAIL("Timeout waiting for consumer to receive item");
    }
  }

  consumer.join();
  REQUIRE(received_value.load() == 999);
}
