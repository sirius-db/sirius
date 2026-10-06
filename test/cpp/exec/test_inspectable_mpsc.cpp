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
#include <functional>
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

TEST_CASE("inspectable_mpsc default ordering is FIFO", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  REQUIRE(queue.get_ordering() == queue_ordering::FIFO);
}

TEST_CASE("inspectable_mpsc explicit LIFO ordering pops newest first", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue(queue_ordering::LIFO);
  REQUIRE(queue.get_ordering() == queue_ordering::LIFO);

  for (int i = 0; i < 5; ++i) {
    REQUIRE(queue.push(std::make_unique<int>(i)));
  }

  for (int i = 4; i >= 0; --i) {
    auto result = queue.try_pop();
    REQUIRE(result != nullptr);
    REQUIRE(*result == i);
  }
  REQUIRE(queue.try_pop() == nullptr);
}

TEST_CASE("inspectable_mpsc LIFO blocking pop also pops newest first", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue(queue_ordering::LIFO);
  REQUIRE(queue.push(std::make_unique<int>(1)));
  REQUIRE(queue.push(std::make_unique<int>(2)));
  REQUIRE(queue.push(std::make_unique<int>(3)));

  auto a = queue.pop();
  REQUIRE(a != nullptr);
  REQUIRE(*a == 3);
  auto b = queue.pop();
  REQUIRE(b != nullptr);
  REQUIRE(*b == 2);
  auto c = queue.pop();
  REQUIRE(c != nullptr);
  REQUIRE(*c == 1);
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

// =============================================================================
// Multi-threaded MPSC stress tests (SAFE-01)
// =============================================================================

TEST_CASE("inspectable_mpsc concurrent producers single consumer", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  constexpr int num_producers      = 4;
  constexpr int items_per_producer = 100;
  constexpr int total_items        = num_producers * items_per_producer;

  std::atomic<int> produced_count{0};
  std::atomic<int> consumed_count{0};

  // Start consumer thread (try_pop loop)
  std::thread consumer([&]() {
    while (consumed_count.load() < total_items) {
      auto result = queue.try_pop();
      if (result != nullptr) {
        consumed_count.fetch_add(1);
      } else {
        std::this_thread::yield();
      }
    }
  });

  // Start 4 producer threads
  std::vector<std::thread> producers;
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

  // Join all producers first
  for (auto& p : producers) {
    p.join();
  }

  // Wait for consumer with 5s timeout
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout waiting for consumption");
    }
  }

  consumer.join();

  REQUIRE(produced_count.load() == total_items);
  REQUIRE(consumed_count.load() == total_items);
}

TEST_CASE("inspectable_mpsc concurrent emplace single consumer", "[inspectable_mpsc]")
{
  inspectable_mpsc<test_payload> queue;
  constexpr int num_producers      = 4;
  constexpr int items_per_producer = 50;
  constexpr int total_items        = num_producers * items_per_producer;

  std::atomic<int> produced_count{0};
  std::atomic<int> consumed_count{0};

  // Single consumer with try_pop
  std::thread consumer([&]() {
    while (consumed_count.load() < total_items) {
      auto result = queue.try_pop();
      if (result != nullptr) {
        consumed_count.fetch_add(1);
      } else {
        std::this_thread::yield();
      }
    }
  });

  // 4 producers using emplace
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

  // Wait for consumption with 5s timeout
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout waiting for consumption");
    }
  }

  consumer.join();

  REQUIRE(produced_count.load() == total_items);
  REQUIRE(consumed_count.load() == total_items);
}

TEST_CASE("inspectable_mpsc blocking pop with concurrent producers", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  constexpr int num_producers      = 4;
  constexpr int items_per_producer = 25;
  constexpr int total_items        = num_producers * items_per_producer;
  std::atomic<int> consumed_count{0};

  // Consumer thread using blocking pop()
  std::thread consumer([&]() {
    while (consumed_count.load() < total_items) {
      auto result = queue.pop();
      if (result != nullptr) {
        consumed_count.fetch_add(1);
      } else {
        // Interrupted -- exit
        break;
      }
    }
  });

  // 4 producers with yield between pushes to create interleaving
  std::vector<std::thread> producers;
  for (int i = 0; i < num_producers; ++i) {
    producers.emplace_back([&, producer_id = i]() {
      for (int j = 0; j < items_per_producer; ++j) {
        int value = producer_id * items_per_producer + j;
        while (!queue.push(std::make_unique<int>(value))) {
          if (!queue.is_open()) return;
          std::this_thread::yield();
        }
        std::this_thread::yield();
      }
    });
  }

  // Join producers
  for (auto& p : producers) {
    p.join();
  }

  // Wait for consumer with 5s timeout
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 5s;
  while (consumed_count.load() < total_items) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      queue.interrupt();
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout waiting for consumption");
    }
  }

  // Interrupt to unblock consumer if it re-entered pop() after consuming all items
  queue.interrupt();
  consumer.join();

  REQUIRE(consumed_count.load() == total_items);
}

TEST_CASE("inspectable_mpsc interrupt unblocks concurrent pop", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  std::atomic<bool> consumer_started{false};
  std::atomic<bool> consumer_returned{false};
  std::unique_ptr<int> pop_result;

  // Consumer thread: block in pop()
  std::thread consumer([&]() {
    consumer_started  = true;
    pop_result        = queue.pop();
    consumer_returned = true;
  });

  // Wait until consumer has started (spin with yield, 100ms timeout)
  auto start   = std::chrono::steady_clock::now();
  auto timeout = 100ms;
  while (!consumer_started.load()) {
    std::this_thread::sleep_for(1ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      queue.interrupt();
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout waiting for consumer to start");
    }
  }

  // Give consumer a moment to actually enter the blocking wait
  std::this_thread::sleep_for(50ms);

  // Verify consumer is still blocked
  REQUIRE_FALSE(consumer_returned.load());

  // Interrupt should unblock the consumer
  queue.interrupt();

  // Wait for consumer to return with 1s timeout
  start   = std::chrono::steady_clock::now();
  timeout = 1s;
  while (!consumer_returned.load()) {
    std::this_thread::sleep_for(10ms);
    if (std::chrono::steady_clock::now() - start > timeout) {
      if (consumer.joinable()) consumer.detach();
      FAIL("Timeout waiting for consumer to return after interrupt");
    }
  }

  consumer.join();
  REQUIRE(pop_result == nullptr);
}

// =============================================================================
// Predicate inspection: pop_if tests
// =============================================================================

TEST_CASE("inspectable_mpsc pop_if front_to_back finds and removes match", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  for (int v : {10, 20, 30, 40, 50}) {
    REQUIRE(queue.push(std::make_unique<int>(v)));
  }

  auto result = queue.pop_if([](const int& v) { return v == 30; }, true);
  REQUIRE(result != nullptr);
  REQUIRE(*result == 30);
  REQUIRE(queue.size() == 4);

  // Verify remaining order: 10, 20, 40, 50
  std::vector<int> remaining;
  while (auto item = queue.try_pop()) {
    remaining.push_back(*item);
  }
  REQUIRE(remaining == std::vector<int>{10, 20, 40, 50});
}

TEST_CASE("inspectable_mpsc pop_if back_to_front finds and removes match", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  for (int v : {10, 20, 30, 40, 50}) {
    REQUIRE(queue.push(std::make_unique<int>(v)));
  }

  auto result = queue.pop_if([](const int& v) { return v == 30; }, false);
  REQUIRE(result != nullptr);
  REQUIRE(*result == 30);

  // Verify remaining order: 10, 20, 40, 50
  std::vector<int> remaining;
  while (auto item = queue.try_pop()) {
    remaining.push_back(*item);
  }
  REQUIRE(remaining == std::vector<int>{10, 20, 40, 50});
}

TEST_CASE("inspectable_mpsc pop_if front_to_back returns first of duplicates", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  for (int v : {10, 20, 30, 20, 50}) {
    REQUIRE(queue.push(std::make_unique<int>(v)));
  }

  auto result = queue.pop_if([](const int& v) { return v == 20; }, true);
  REQUIRE(result != nullptr);
  REQUIRE(*result == 20);

  // The second 20 remains; order: 10, 30, 20, 50
  std::vector<int> remaining;
  while (auto item = queue.try_pop()) {
    remaining.push_back(*item);
  }
  REQUIRE(remaining == std::vector<int>{10, 30, 20, 50});
}

TEST_CASE("inspectable_mpsc pop_if back_to_front returns last of duplicates", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  for (int v : {10, 20, 30, 20, 50}) {
    REQUIRE(queue.push(std::make_unique<int>(v)));
  }

  auto result = queue.pop_if([](const int& v) { return v == 20; }, false);
  REQUIRE(result != nullptr);
  REQUIRE(*result == 20);

  // The first 20 remains; order: 10, 20, 30, 50
  std::vector<int> remaining;
  while (auto item = queue.try_pop()) {
    remaining.push_back(*item);
  }
  REQUIRE(remaining == std::vector<int>{10, 20, 30, 50});
}

TEST_CASE("inspectable_mpsc pop_if no match returns nullptr", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  for (int v : {10, 20, 30}) {
    REQUIRE(queue.push(std::make_unique<int>(v)));
  }

  auto result = queue.pop_if([](const int& v) { return v == 99; }, true);
  REQUIRE(result == nullptr);
  REQUIRE(queue.size() == 3);
}

TEST_CASE("inspectable_mpsc pop_if empty queue returns nullptr", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;

  auto result = queue.pop_if([](const int& /*v*/) { return true; }, true);
  REQUIRE(result == nullptr);
}

// =============================================================================
// Predicate inspection: mutable_pop_if tests
// =============================================================================

TEST_CASE("inspectable_mpsc mutable_pop_if removes matching element", "[inspectable_mpsc]")
{
  inspectable_mpsc<test_payload> queue;
  for (int i = 1; i <= 5; ++i) {
    REQUIRE(queue.emplace(i, "item_" + std::to_string(i)));
  }

  auto result = queue.mutable_pop_if([](test_payload& p) { return p.id == 3; }, true);
  REQUIRE(result != nullptr);
  REQUIRE(result->id == 3);
  REQUIRE(queue.size() == 4);
}

TEST_CASE("inspectable_mpsc mutable_pop_if respects search direction", "[inspectable_mpsc]")
{
  inspectable_mpsc<test_payload> queue;
  for (int id : {1, 2, 3, 2, 5}) {
    REQUIRE(queue.emplace(id, "item_" + std::to_string(id)));
  }

  // back_to_front should remove the last element with id==2
  auto result = queue.mutable_pop_if([](test_payload& p) { return p.id == 2; }, false);
  REQUIRE(result != nullptr);
  REQUIRE(result->id == 2);

  // Remaining ids: [1, 2, 3, 5] (the first 2 stays)
  std::vector<int> remaining;
  while (auto item = queue.try_pop()) {
    remaining.push_back(item->id);
  }
  REQUIRE(remaining == std::vector<int>{1, 2, 3, 5});
}

// =============================================================================
// Predicate inspection: order preservation
// =============================================================================

TEST_CASE("inspectable_mpsc pop_if preserves remaining element order", "[inspectable_mpsc]")
{
  inspectable_mpsc<int> queue;
  for (int v : {1, 2, 3, 4, 5}) {
    REQUIRE(queue.push(std::make_unique<int>(v)));
  }

  // Remove 3, then 1, then 5
  auto r1 = queue.pop_if([](const int& v) { return v == 3; }, true);
  REQUIRE(r1 != nullptr);
  REQUIRE(*r1 == 3);

  auto r2 = queue.pop_if([](const int& v) { return v == 1; }, true);
  REQUIRE(r2 != nullptr);
  REQUIRE(*r2 == 1);

  auto r3 = queue.pop_if([](const int& v) { return v == 5; }, true);
  REQUIRE(r3 != nullptr);
  REQUIRE(*r3 == 5);

  // Remaining: [2, 4]
  std::vector<int> remaining;
  while (auto item = queue.try_pop()) {
    remaining.push_back(*item);
  }
  REQUIRE(remaining == std::vector<int>{2, 4});
}
