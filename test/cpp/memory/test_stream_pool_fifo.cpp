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

/**
 * @file test_stream_pool_fifo.cpp
 * @brief exclusive_stream_pool BLOCK checkout is FIFO and starvation-free (register issue F9).
 *
 * All BLOCK-policy waiters share one condition variable. With race-wins wake-ups a released
 * stream went to whichever waiter won the race, and a caller that released and immediately
 * re-acquired always beat a parked waiter (it needs no CV round trip at all). These cases pin
 * the ticket handoff: streams go to the longest-waiting caller, re-acquirers queue at the tail,
 * and GROW callers never take a pooled stream a parked waiter is owed.
 */

#include "catch.hpp"

#include <rmm/cuda_device.hpp>

#include <cucascade/memory/stream_pool.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace {

using cucascade::memory::borrowed_stream;
using cucascade::memory::exclusive_stream_pool;
using steady_clock = std::chrono::steady_clock;

constexpr auto kBlock = exclusive_stream_pool::stream_acquire_policy::BLOCK;
constexpr auto kGrow  = exclusive_stream_pool::stream_acquire_policy::GROW;

struct stream_pool_fixture {
  bool valid = false;
  std::unique_ptr<exclusive_stream_pool> pool;

  explicit stream_pool_fixture(std::size_t pool_size)
  {
    try {
      pool  = std::make_unique<exclusive_stream_pool>(rmm::cuda_device_id{0}, pool_size);
      valid = true;
    } catch (const std::exception& e) {
      WARN("Skipping stream-pool test (no usable GPU): " << e.what());
    }
  }

  [[nodiscard]] static bool wait_until(const std::function<bool()>& done,
                                       std::chrono::seconds deadline)
  {
    const auto give_up = steady_clock::now() + deadline;
    while (!done()) {
      if (steady_clock::now() > give_up) { return false; }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return true;
  }
};

}  // namespace

TEST_CASE("F9: BLOCK checkout hands streams to waiters in arrival order",
          "[memory][stream_pool][concurrency]")
{
  stream_pool_fixture f(/*pool_size=*/1);
  if (!f.valid) { return; }

  auto held = std::make_optional<borrowed_stream>(f.pool->acquire_stream(kBlock));

  constexpr int kWaiters = 4;
  std::mutex order_mutex;
  std::vector<int> service_order;
  std::vector<std::thread> waiters;
  waiters.reserve(kWaiters);

  // Arrival order enforced by staggered starts (100 ms apart). Each waiter records its grant
  // and holds the stream briefly, so the single stream cascades through the queue one waiter
  // at a time.
  for (int i = 0; i < kWaiters; ++i) {
    waiters.emplace_back([&, i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(100 * i));
      auto s = f.pool->acquire_stream(kBlock);
      {
        std::lock_guard<std::mutex> lock(order_mutex);
        service_order.push_back(i);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    });
  }

  // Everybody parked while the stream is held.
  std::this_thread::sleep_for(std::chrono::milliseconds(100 * kWaiters + 200));
  const bool none_served_early = [&] {
    std::lock_guard<std::mutex> lock(order_mutex);
    return service_order.empty();
  }();

  held.reset();
  for (auto& t : waiters) {
    t.join();
  }

  REQUIRE(none_served_early);
  std::lock_guard<std::mutex> lock(order_mutex);
  REQUIRE(service_order.size() == kWaiters);
  for (int i = 0; i < kWaiters; ++i) {
    INFO("service_order[" << i << "]=" << service_order[i]);
    REQUIRE(service_order[i] == i);
  }
}

TEST_CASE("F9: a release-and-reacquire caller queues behind a parked waiter",
          "[memory][stream_pool][concurrency]")
{
  stream_pool_fixture f(/*pool_size=*/1);
  if (!f.valid) { return; }

  std::atomic<int> order_counter{0};
  std::atomic<int> waiter_order{-1};
  std::atomic<int> reacquirer_order{-1};

  auto held = std::make_optional<borrowed_stream>(f.pool->acquire_stream(kBlock));

  std::thread waiter([&] {
    auto s = f.pool->acquire_stream(kBlock);
    waiter_order.store(order_counter.fetch_add(1), std::memory_order_release);
    // Hold long enough that the re-acquirer is provably parked behind us before we release.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  // Let the waiter park, then release and IMMEDIATELY re-acquire from this thread. With
  // race-wins semantics the re-acquire wins every time (no CV round trip); the ticket queue
  // sends it to the tail instead.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const bool waiter_parked = waiter_order.load(std::memory_order_acquire) == -1;
  held.reset();
  auto reacquired = std::make_optional<borrowed_stream>(f.pool->acquire_stream(kBlock));
  reacquirer_order.store(order_counter.fetch_add(1), std::memory_order_release);

  // If the re-acquirer barged (the failure mode), the waiter is parked on the stream we hold;
  // release it before joining so a failing run reports cleanly instead of hanging.
  const bool waiter_done = stream_pool_fixture::wait_until(
    [&] { return waiter_order.load(std::memory_order_acquire) != -1; }, std::chrono::seconds(10));
  if (!waiter_done) { reacquired.reset(); }
  waiter.join();
  REQUIRE(waiter_parked);
  REQUIRE(waiter_done);
  REQUIRE(waiter_order.load(std::memory_order_acquire) == 0);
  REQUIRE(reacquirer_order.load(std::memory_order_acquire) == 1);
}

TEST_CASE("F9: GROW does not take a pooled stream a parked waiter is owed",
          "[memory][stream_pool][concurrency]")
{
  stream_pool_fixture f(/*pool_size=*/1);
  if (!f.valid) { return; }

  auto held = std::make_optional<borrowed_stream>(f.pool->acquire_stream(kBlock));

  std::atomic<bool> waiter_served{false};
  std::thread waiter([&] {
    auto s = f.pool->acquire_stream(kBlock);
    waiter_served.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const bool waiter_parked = !waiter_served.load(std::memory_order_acquire);

  // Release the held stream, then take the GROW path while the waiter is being woken. GROW
  // must not race the waiter for the pooled stream: it mints a fresh one, and the waiter is
  // served regardless.
  held.reset();
  auto grown = std::make_optional<borrowed_stream>(f.pool->acquire_stream(kGrow));

  // If GROW stole the pooled stream (the failure mode), the waiter is parked on the stream we
  // hold; release it before joining so a failing run reports cleanly instead of hanging.
  const bool served = stream_pool_fixture::wait_until(
    [&] { return waiter_served.load(std::memory_order_acquire); }, std::chrono::seconds(10));
  if (!served) { grown.reset(); }
  waiter.join();
  REQUIRE(waiter_parked);
  REQUIRE(served);
}
