/*
 * Copyright 2026, Sirius Contributors.
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
#include "exec/semi_future.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::exec;
using namespace std::chrono_literals;

namespace {

template <class counter_t>
std::uint64_t counter_value(counter_t const& counter)
{
  if constexpr (requires { counter.load(std::memory_order_relaxed); }) {
    return counter.load(std::memory_order_relaxed);
  } else {
    return static_cast<std::uint64_t>(counter);
  }
}

std::uint64_t raw_futex_wake_count()
{
  return counter_value(sirius::exec::detail::raw_futex_wake_count());
}

void wait_until_count(std::atomic<int>& counter, int expected)
{
  auto const deadline = std::chrono::steady_clock::now() + 1s;
  while (counter.load(std::memory_order_acquire) < expected &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
  REQUIRE(counter.load(std::memory_order_acquire) == expected);
}

void rethrow_if_set(std::exception_ptr ep)
{
  if (ep) { std::rethrow_exception(ep); }
}

}  // namespace

TEST_CASE("semi_future untimed get wakes and does not issue raw futex wake", "[exec][semi_future]")
{
  auto const before = raw_futex_wake_count();

  promise<int> p;
  auto sf = p.get_semi_future();
  std::atomic<int> entered{0};
  int value = 0;
  std::exception_ptr error;

  std::thread waiter([&] {
    entered.fetch_add(1, std::memory_order_release);
    try {
      value = std::move(sf).get();
    } catch (...) {
      error = std::current_exception();
    }
  });

  wait_until_count(entered, 1);
  std::this_thread::sleep_for(10ms);
  p.set_value(42);
  waiter.join();

  rethrow_if_set(error);
  CHECK(value == 42);
  CHECK(raw_futex_wake_count() - before == 0);
}

TEST_CASE("semi_future timed get wakes promptly before its deadline", "[exec][semi_future]")
{
  constexpr int kWaiters = 8;

  std::vector<std::unique_ptr<promise<int>>> promises;
  promises.reserve(kWaiters);
  std::vector<int> values(static_cast<std::size_t>(kWaiters), 0);
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(kWaiters));
  std::vector<std::thread> waiters;
  waiters.reserve(kWaiters);
  std::atomic<int> entered{0};

  for (int i = 0; i < kWaiters; ++i) {
    auto p  = std::make_unique<promise<int>>();
    auto sf = p->get_semi_future();
    waiters.emplace_back([sf = std::move(sf), &entered, &values, &errors, i]() mutable {
      entered.fetch_add(1, std::memory_order_release);
      try {
        values[static_cast<std::size_t>(i)] = std::move(sf).get(500ms);
      } catch (...) {
        errors[static_cast<std::size_t>(i)] = std::current_exception();
      }
    });
    promises.push_back(std::move(p));
  }

  wait_until_count(entered, kWaiters);
  std::this_thread::sleep_for(20ms);
  auto const fulfill_start = std::chrono::steady_clock::now();
  for (int i = 0; i < kWaiters; ++i) {
    promises[static_cast<std::size_t>(i)]->set_value(100 + i);
  }
  for (auto& t : waiters) {
    t.join();
  }
  auto const elapsed = std::chrono::steady_clock::now() - fulfill_start;

  CHECK(elapsed < 250ms);
  for (int i = 0; i < kWaiters; ++i) {
    rethrow_if_set(errors[static_cast<std::size_t>(i)]);
    CHECK(values[static_cast<std::size_t>(i)] == 100 + i);
  }
}

TEST_CASE("semi_future timed get times out when the promise is not fulfilled",
          "[exec][semi_future]")
{
  promise<int> p;
  auto sf = p.get_semi_future();

  auto const start = std::chrono::steady_clock::now();
  CHECK_THROWS_AS(std::move(sf).get(40ms), std::runtime_error);
  auto const elapsed = std::chrono::steady_clock::now() - start;

  CHECK(elapsed >= 20ms);
  CHECK(elapsed < 500ms);
}

TEST_CASE("semi_future timed and untimed waiters on one core both wake", "[exec][semi_future]")
{
  auto core = std::make_shared<sirius::exec::detail::core<int>>();
  semi_future<int> untimed{std::make_unique<sirius::exec::detail::leaf_state<int>>(core)};
  semi_future<int> timed{std::make_unique<sirius::exec::detail::leaf_state<int>>(core)};

  auto const before = raw_futex_wake_count();
  std::atomic<int> entered{0};
  std::atomic<bool> untimed_woke{false};
  int timed_value = 0;
  std::exception_ptr untimed_error;
  std::exception_ptr timed_error;

  std::thread untimed_thread([&] {
    entered.fetch_add(1, std::memory_order_release);
    try {
      untimed.wait();
      untimed_woke.store(true, std::memory_order_release);
    } catch (...) {
      untimed_error = std::current_exception();
    }
  });
  std::thread timed_thread([&] {
    entered.fetch_add(1, std::memory_order_release);
    try {
      timed_value = std::move(timed).get(500ms);
    } catch (...) {
      timed_error = std::current_exception();
    }
  });

  wait_until_count(entered, 2);
  std::this_thread::sleep_for(20ms);
  core->set_try(try_t<int>(77));

  untimed_thread.join();
  timed_thread.join();

  rethrow_if_set(untimed_error);
  rethrow_if_set(timed_error);
  CHECK(untimed_woke.load(std::memory_order_acquire));
  CHECK(timed_value == 77);
  CHECK(raw_futex_wake_count() > before);
}

TEST_CASE("semi_future raw futex wake is gated to parked timed waiters", "[exec][semi_future]")
{
  SECTION("no waiter")
  {
    auto const before = raw_futex_wake_count();
    promise<int> p;
    auto sf = p.get_semi_future();

    p.set_value(7);

    CHECK(raw_futex_wake_count() - before == 0);
    CHECK(std::move(sf).get() == 7);
  }

  SECTION("callback waiter")
  {
    auto const before = raw_futex_wake_count();
    promise<int> p;
    auto sf = p.get_semi_future();
    std::atomic<int> observed{0};

    std::move(sf).install_callback([&observed](try_t<int>&& t) {
      observed.store(std::move(t).get(), std::memory_order_release);
    });
    p.set_value(8);

    CHECK(observed.load(std::memory_order_acquire) == 8);
    CHECK(raw_futex_wake_count() - before == 0);
  }

  SECTION("parked timed waiter")
  {
    auto const before = raw_futex_wake_count();
    promise<int> p;
    auto sf = p.get_semi_future();
    std::atomic<int> entered{0};
    int value = 0;
    std::exception_ptr error;

    std::thread waiter([sf = std::move(sf), &entered, &value, &error]() mutable {
      entered.fetch_add(1, std::memory_order_release);
      try {
        value = std::move(sf).get(500ms);
      } catch (...) {
        error = std::current_exception();
      }
    });

    wait_until_count(entered, 1);
    std::this_thread::sleep_for(20ms);
    p.set_value(9);
    waiter.join();

    rethrow_if_set(error);
    CHECK(value == 9);
    CHECK(raw_futex_wake_count() > before);
  }
}
