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
#include "memory/numa_affinity.hpp"
#include "scan_manager/host_pin_placement.hpp"

#include <algorithm>
#include <array>
#include <thread>
#include <utility>
#include <vector>

using sirius::scan_manager::host_pin_placement;

TEST_CASE("host pins interleave resolved host spaces and continue across tables",
          "[host_pin_placement]")
{
  std::vector<std::pair<int, int>> const mapping{{0, 0}, {1, 0}, {2, 1}, {3, 1}};
  auto const order = host_pin_placement::interleaved_devices(mapping);
  REQUIRE(order == std::vector<int>{0, 2, 1, 3});
  host_pin_placement placement;
  // First table has two chunks. A second call reconstructs the order but shares
  // the manager-owned cursor: neither its first chunk nor a later re-pin resets it.
  REQUIRE(placement.next_gpu(order) == 0);
  REQUIRE(placement.next_gpu(order) == 2);
  auto const next_order = host_pin_placement::interleaved_devices(mapping);
  REQUIRE(placement.next_gpu(next_order) == 1);
  REQUIRE(placement.next_gpu(next_order) == 3);
  REQUIRE(placement.next_gpu(next_order) == 0);
}

TEST_CASE("host pin order handles sparse ids and ignores enumeration order", "[host_pin_placement]")
{
  std::vector<std::pair<int, int>> mapping{{19, 9}, {11, 3}, {7, 9}, {4, 3}};
  auto const expected = std::vector<int>{4, 7, 11, 19};
  REQUIRE(host_pin_placement::interleaved_devices(mapping) == expected);
  std::reverse(mapping.begin(), mapping.end());
  REQUIRE(host_pin_placement::interleaved_devices(mapping) == expected);
}

TEST_CASE("host pin order supports uneven groups and fallback host spaces", "[host_pin_placement]")
{
  SECTION("uneven groups visit every GPU once per cycle")
  {
    std::vector<std::pair<int, int>> const mapping{{0, 0}, {1, 0}, {2, 0}, {7, 8}};
    REQUIRE(host_pin_placement::interleaved_devices(mapping) == std::vector<int>{0, 7, 1, 2});
  }
  SECTION("fallback GPUs share the resolved destination, including unknown NUMA")
  {
    std::vector<std::pair<int, int>> const mapping{{3, -1}, {0, -1}, {2, -1}, {1, -1}};
    REQUIRE(host_pin_placement::interleaved_devices(mapping) == std::vector<int>{0, 1, 2, 3});
  }
  SECTION("a single GPU remains usable across pins")
  {
    std::vector<std::pair<int, int>> const mapping{{7, 42}};
    auto const order = host_pin_placement::interleaved_devices(mapping);
    host_pin_placement placement;
    REQUIRE(placement.next_gpu(order) == 7);
    REQUIRE(placement.next_gpu(order) == 7);
  }
}

TEST_CASE("host pin placement is context local and rejects empty orders", "[host_pin_placement]")
{
  host_pin_placement first, second;
  std::vector<int> const order{0, 2, 1, 3};
  REQUIRE(host_pin_placement::interleaved_devices({}).empty());
  REQUIRE_THROWS_AS(first.next_gpu({}), std::invalid_argument);
  REQUIRE(first.next_gpu(order) == 0);
  REQUIRE(first.next_gpu(order) == 2);
  REQUIRE(second.next_gpu(order) == 0);
}

TEST_CASE("alternating host spaces rotate across all local GPUs",
          "[host_pin_placement][numa_affinity]")
{
  sirius::memory::numa_affinity affinity;
  std::array<int, 2> const host0{0, 1}, host1{2, 3};
  std::vector<int> actual;
  for (int i = 0; i < 4; ++i) {
    actual.push_back(affinity.next_gpu(0, host0).value());
    actual.push_back(affinity.next_gpu(1, host1).value());
  }
  REQUIRE(actual == std::vector<int>{0, 2, 1, 3, 0, 2, 1, 3});
  REQUIRE_FALSE(affinity.next_gpu(9, {}).has_value());
  REQUIRE(affinity.next_gpu(0, host0) == 0);
  REQUIRE(affinity.next_gpu(1, host1) == 2);
}

TEST_CASE("NUMA affinity supports uneven and unknown domains independently",
          "[host_pin_placement][numa_affinity]")
{
  sirius::memory::numa_affinity affinity;
  std::array<int, 1> const unknown{7};
  std::array<int, 3> const known{2, 5, 9};
  for (int gpu : known) {
    REQUIRE(affinity.next_gpu(-1, unknown) == 7);
    REQUIRE(affinity.next_gpu(4, known) == gpu);
  }
  REQUIRE(affinity.next_gpu(4, known) == 2);
}

TEST_CASE("concurrent placement retains balanced host and GPU counters",
          "[host_pin_placement][numa_affinity]")
{
  host_pin_placement pins;
  sirius::memory::numa_affinity affinity;
  std::array<int, 4> const order{0, 2, 1, 3};
  std::array<int, 2> const host0{0, 1}, host1{2, 3};
  std::array<std::array<int, 4>, 4> pin_counts{}, affinity_counts{};
  std::vector<std::thread> workers;
  for (std::size_t worker = 0; worker < pin_counts.size(); ++worker) {
    workers.emplace_back([&, worker] {
      for (int i = 0; i < 1000; ++i) {
        ++pin_counts[worker][pins.next_gpu(order)];
        ++affinity_counts[worker][affinity.next_gpu(0, host0).value()];
        ++affinity_counts[worker][affinity.next_gpu(1, host1).value()];
      }
    });
  }
  for (auto& worker : workers)
    worker.join();
  for (std::size_t gpu = 0; gpu < order.size(); ++gpu) {
    int pinned = 0, scheduled = 0;
    for (std::size_t worker = 0; worker < pin_counts.size(); ++worker) {
      pinned += pin_counts[worker][gpu];
      scheduled += affinity_counts[worker][gpu];
    }
    REQUIRE(pinned == 1000);
    REQUIRE(scheduled == 2000);
  }
}
