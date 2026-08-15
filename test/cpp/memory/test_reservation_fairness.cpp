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
 * @file test_reservation_fairness.cpp
 * @brief Blocking reservations are granted FIFO per memory space (register issue F9).
 *
 * cucascade's memory_space::make_reservation parks on the space's notification channel. With
 * race-wins wake-ups, a released reservation went to whichever waiter won the race — and a heavy
 * caller that releases and immediately re-requests never has to win a race at all (no CV round
 * trip), so it could perpetually beat a light caller's single parked wait. These cases pin the
 * FIFO discipline: the longest-waiting caller is served first, and a fresh blocking caller
 * queues behind parked waiters instead of barging through the non-blocking fast path.
 *
 * Assertions run only after every spawned thread has been unblocked and joined, so a failing
 * discipline reports cleanly instead of terminating on a joinable thread.
 */

#include "catch.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using steady_clock = std::chrono::steady_clock;

constexpr std::size_t kMiB = 1024 * 1024;

//! One small GPU memory space; big enough for a handful of 16-24 MiB grants, small enough that a
//! test-held reservation provably forces callers to wait.
struct fairness_fixture {
  bool valid = false;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* mem_space = nullptr;

  fairness_fixture()
  {
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(256 * kMiB)
        .set_reservation_fraction_per_gpu(0.75)
        .set_per_numa_region_capacity(256 * kMiB)
        .use_gpu_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      manager =
        std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    } catch (const std::exception& e) {
      WARN("Skipping reservation-fairness test (no usable GPU): " << e.what());
      return;
    }
    mem_space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    if (!mem_space) {
      WARN("Skipping reservation-fairness test: no GPU memory space available.");
      return;
    }
    valid = true;
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

TEST_CASE("F9: a heavy caller's release-and-re-request loop cannot starve a parked waiter",
          "[memory][reservation_fairness][concurrency]")
{
  fairness_fixture f;
  if (!f.valid) { return; }

  // Leave 32 MiB of head room: the heavy caller's 24 MiB requests fit one at a time, so while it
  // holds one, nobody else's 24 MiB request can.
  const std::size_t space_max = f.mem_space->get_max_memory();
  const std::size_t grant     = 24 * kMiB;
  REQUIRE(space_max > 4 * grant);
  auto hold = f.mem_space->make_reservation_or_null(space_max - 32 * kMiB);
  REQUIRE(hold);

  std::atomic<bool> stop_heavy{false};
  std::atomic<int> heavy_grants{0};

  // The heavy caller: grab, hold briefly, release, immediately re-request. With race-wins
  // semantics its fresh request claims the just-freed memory before a parked waiter's retry
  // (it needs no CV round trip), so the waiter can lose every round.
  std::thread heavy([&] {
    while (!stop_heavy.load(std::memory_order_acquire)) {
      auto r = f.mem_space->make_reservation(grant);
      if (!r) { return; }  // space shut down
      heavy_grants.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      r.reset();
    }
  });

  // Let the heavy loop establish itself, then park ONE light wait.
  const bool heavy_running = fairness_fixture::wait_until(
    [&] { return heavy_grants.load(std::memory_order_relaxed) >= 3; }, std::chrono::seconds(10));
  const int grants_before_light = heavy_grants.load(std::memory_order_relaxed);

  std::atomic<bool> light_served{false};
  std::atomic<int> grants_when_served{0};
  std::thread light([&] {
    auto r = f.mem_space->make_reservation(grant);
    grants_when_served.store(heavy_grants.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
    light_served.store(r != nullptr, std::memory_order_release);
    // Hold briefly so the grant is unambiguous in the accounting.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  });

  const bool served =
    heavy_running &&
    fairness_fixture::wait_until([&] { return light_served.load(std::memory_order_acquire); },
                                 std::chrono::seconds(20));

  // Unblock and join everything BEFORE asserting: a starved light waiter needs the hold's
  // memory to ever return.
  stop_heavy.store(true, std::memory_order_release);
  if (!served) { hold.reset(); }
  light.join();
  heavy.join();

  REQUIRE(heavy_running);
  REQUIRE(served);
  // Served within a BOUNDED number of heavy grants: FIFO puts the light waiter at the head, and
  // the heavy caller's next request queues behind it.
  const int grants_to_serve_light =
    grants_when_served.load(std::memory_order_relaxed) - grants_before_light;
  INFO("heavy grants before light registered="
       << grants_before_light << " grants until light served=" << grants_to_serve_light);
  REQUIRE(grants_to_serve_light <= 3);
}

TEST_CASE("F9: parked reservation waiters are served in arrival order",
          "[memory][reservation_fairness][concurrency]")
{
  fairness_fixture f;
  if (!f.valid) { return; }

  const std::size_t space_max = f.mem_space->get_max_memory();
  const std::size_t grant     = 16 * kMiB;
  REQUIRE(space_max > 6 * grant);

  constexpr int kWaiters = 4;

  // Occupy the space down to 4 MiB of head room, split so that releasing one chunk frees room
  // for EXACTLY ONE 16 MiB grant (16 + 4 head room). Stepping the releases one at a time makes
  // the expected service order fully deterministic: after each release only the head-of-queue
  // waiter can be served, and the next chunk is not released until that grant is recorded.
  auto base = f.mem_space->make_reservation_or_null(space_max - 4 * kMiB - (kWaiters - 1) * grant);
  REQUIRE(base);
  std::vector<std::unique_ptr<cucascade::memory::reservation>> chunks;
  for (int i = 0; i < kWaiters - 1; ++i) {
    auto chunk = f.mem_space->make_reservation_or_null(grant);
    REQUIRE(chunk);
    chunks.push_back(std::move(chunk));
  }

  std::mutex order_mutex;
  std::vector<int> service_order;
  std::vector<std::unique_ptr<cucascade::memory::reservation>> grants(kWaiters);
  std::vector<std::thread> waiters;
  waiters.reserve(kWaiters);

  // Registration order is enforced by staggered starts (100 ms apart — far beyond scheduling
  // jitter). Each waiter KEEPS its reservation, so a later waiter can only be served by a chunk
  // released below, never by an earlier waiter's memory.
  for (int i = 0; i < kWaiters; ++i) {
    waiters.emplace_back([&, i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(100 * i));
      auto r = f.mem_space->make_reservation(grant);
      std::lock_guard<std::mutex> lock(order_mutex);
      service_order.push_back(i);
      grants[i] = std::move(r);
    });
  }

  const auto served_count = [&] {
    std::lock_guard<std::mutex> lock(order_mutex);
    return service_order.size();
  };

  // All parked (nothing can be served while every hold lives).
  std::this_thread::sleep_for(std::chrono::milliseconds(100 * kWaiters + 200));
  const bool none_served_early = served_count() == 0;

  // Release room for one grant at a time; each step must serve the longest-waiting caller.
  bool steps_completed = none_served_early;
  for (int step = 0; steps_completed && step < kWaiters; ++step) {
    if (step < kWaiters - 1) {
      chunks[step].reset();
    } else {
      base.reset();  // last step: the remaining waiter gets the big chunk
    }
    steps_completed = fairness_fixture::wait_until(
      [&] { return served_count() == static_cast<std::size_t>(step + 1); },
      std::chrono::seconds(20));
  }

  // Unblock and join everything BEFORE asserting.
  for (auto& chunk : chunks) {
    chunk.reset();
  }
  base.reset();
  for (auto& t : waiters) {
    t.join();
  }

  REQUIRE(none_served_early);
  REQUIRE(steps_completed);
  std::lock_guard<std::mutex> lock(order_mutex);
  REQUIRE(service_order.size() == kWaiters);
  for (int i = 0; i < kWaiters; ++i) {
    INFO("service_order[" << i << "]=" << service_order[i]);
    REQUIRE(service_order[i] == i);
    REQUIRE(grants[i] != nullptr);
  }
}

TEST_CASE("F9: shutdown wakes every parked reservation waiter",
          "[memory][reservation_fairness][concurrency]")
{
  fairness_fixture f;
  if (!f.valid) { return; }

  const std::size_t space_max = f.mem_space->get_max_memory();
  auto hold                   = f.mem_space->make_reservation_or_null(space_max - 4 * kMiB);
  REQUIRE(hold);

  std::atomic<int> woken{0};
  std::atomic<int> null_results{0};
  std::vector<std::thread> waiters;
  for (int i = 0; i < 3; ++i) {
    waiters.emplace_back([&] {
      auto r = f.mem_space->make_reservation(16 * kMiB);
      if (r == nullptr) { null_results.fetch_add(1, std::memory_order_relaxed); }
      woken.fetch_add(1, std::memory_order_relaxed);
    });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  const bool none_woken_early = woken.load(std::memory_order_relaxed) == 0;

  // Not just one waiter: every parked wait must observe the shutdown and return null.
  f.mem_space->shutdown();
  const bool all_woken = fairness_fixture::wait_until(
    [&] { return woken.load(std::memory_order_relaxed) == 3; }, std::chrono::seconds(10));

  // If the shutdown failed to wake someone, hand out memory so join() cannot hang.
  if (!all_woken) { hold.reset(); }
  for (auto& t : waiters) {
    t.join();
  }

  REQUIRE(none_woken_early);
  REQUIRE(all_woken);
  REQUIRE(null_results.load(std::memory_order_relaxed) == 3);
}
