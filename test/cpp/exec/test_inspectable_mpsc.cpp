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
#include "exec/inspectable_mpsc.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

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
// Basic functionality tests
// =============================================================================

TEST_CASE("inspectable_mpsc basic push and pop", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  REQUIRE(queue.push(std::make_unique<int>(42)));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(*result == 42);
}

TEST_CASE("inspectable_mpsc push and pop with custom type", "[inspectable_mpsc]")
{
  inspectable_mpsc<test_payload> queue;

  REQUIRE(queue.push(std::make_unique<test_payload>(1, "hello")));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(result->id == 1);
  REQUIRE(result->data == "hello");
}

TEST_CASE("inspectable_mpsc try_pop returns nullptr on empty queue", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  auto result = queue.try_pop();
  REQUIRE(result == nullptr);
}

TEST_CASE("inspectable_mpsc multiple items FIFO order", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

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
// Emplace tests
// =============================================================================

TEST_CASE("inspectable_mpsc emplace constructs in-place", "[inspectable_mpsc]")
{
  inspectable_mpsc<test_payload> queue;

  REQUIRE(queue.emplace(42, "emplaced"));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(result->id == 42);
  REQUIRE(result->data == "emplaced");
}

TEST_CASE("inspectable_mpsc emplace multiple items", "[inspectable_mpsc]")
{
  inspectable_mpsc<test_payload> queue;

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

TEST_CASE("inspectable_mpsc emplace fails after interrupt", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.emplace(42));
}

// =============================================================================
// Interruption tests
// =============================================================================

TEST_CASE("inspectable_mpsc interrupt closes queue", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  REQUIRE(queue.is_open());
  queue.interrupt();
  REQUIRE_FALSE(queue.is_open());
}

TEST_CASE("inspectable_mpsc push fails after interrupt", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.push(std::make_unique<int>(42)));
}

TEST_CASE("inspectable_mpsc blocking pop returns nullptr after interrupt", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
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

// =============================================================================
// Reactivation tests
// =============================================================================

TEST_CASE("inspectable_mpsc reactivate restores operation", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  queue.interrupt();
  REQUIRE_FALSE(queue.is_open());
  REQUIRE_FALSE(queue.push(std::make_unique<int>(1)));

  queue.reactivate();
  REQUIRE(queue.is_open());
  REQUIRE(queue.push(std::make_unique<int>(42)));

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(*result == 42);
}

// =============================================================================
// Drain tests
// =============================================================================

TEST_CASE("inspectable_mpsc drain removes all items", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  for (int i = 0; i < 5; ++i) {
    REQUIRE(queue.push(std::make_unique<int>(i)));
  }

  REQUIRE(queue.size() == 5);
  queue.drain();
  REQUIRE(queue.is_empty());
  REQUIRE(queue.size() == 0);
}

// =============================================================================
// State query tests
// =============================================================================

TEST_CASE("inspectable_mpsc is_empty and size track state", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  REQUIRE(queue.is_empty());
  REQUIRE(queue.size() == 0);

  for (int i = 0; i < 3; ++i) {
    REQUIRE(queue.push(std::make_unique<int>(i)));
  }

  REQUIRE_FALSE(queue.is_empty());
  REQUIRE(queue.size() == 3);

  auto result = queue.try_pop();
  REQUIRE(result != nullptr);
  REQUIRE(queue.size() == 2);
}

// =============================================================================
// Blocking pop with push from another thread
// =============================================================================

TEST_CASE("inspectable_mpsc blocking pop receives pushed item", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
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
