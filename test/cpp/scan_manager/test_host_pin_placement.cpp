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

TEST_CASE("host data is routed only to the query's admitted GPUs",
          "[host_pin_placement][numa_affinity]")
{
  using sirius::memory::admitted_local_gpus;
  std::array<int, 4> const local{0, 1, 2, 3};
  std::array<int, 3> const admitted{0, 1, 2};

  // No subset: every local GPU, in the topology's order.
  REQUIRE(admitted_local_gpus(local, {}) == std::vector<int>{0, 1, 2, 3});
  REQUIRE(admitted_local_gpus(local, local) == std::vector<int>{0, 1, 2, 3});
  REQUIRE(admitted_local_gpus({}, {}).empty());

  // A subset keeps the local order and drops the excluded GPU.
  REQUIRE(admitted_local_gpus(local, admitted) == std::vector<int>{0, 1, 2});
  std::array<int, 4> const reordered{3, 2, 1, 0};
  REQUIRE(admitted_local_gpus(reordered, admitted) == std::vector<int>{2, 1, 0});

  // No admitted GPU on the data's NUMA node, or no GPU on it at all: locality names no GPU, and
  // the walk over an empty list leaves the task without a preference.
  std::array<int, 2> const other_node{4, 5};
  REQUIRE(admitted_local_gpus(other_node, admitted).empty());
  REQUIRE(admitted_local_gpus({}, admitted).empty());
  sirius::memory::numa_affinity no_local;
  REQUIRE_FALSE(no_local.next_gpu(0, admitted_local_gpus(other_node, admitted)).has_value());

  // Three chunks of one scan on three admitted GPUs land on three different GPUs, wherever the
  // walk happens to start. Walking all four GPUs would name GPU 3 for one chunk in every four.
  sirius::memory::numa_affinity affinity;
  for (int start = 0; start < 4; ++start) {
    auto const gpus = admitted_local_gpus(local, admitted);
    std::vector<int> chunks;
    for (int i = 0; i < 3; ++i) {
      chunks.push_back(affinity.next_gpu(0, gpus).value());
    }
    std::sort(chunks.begin(), chunks.end());
    REQUIRE(chunks == std::vector<int>{0, 1, 2});
    (void)affinity.next_gpu(0, gpus);  // shift the starting point for the next round
  }
}

TEST_CASE("queries admitted to different subsets rotate independently on one NUMA node",
          "[host_pin_placement][numa_affinity]")
{
  // Two queries alternate on node 0, one admitted to {0, 1} and one to {0, 1, 2}. On a counter
  // shared by both lists the first query would only see even values and get GPU 0 every time.
  sirius::memory::numa_affinity affinity;
  std::array<int, 2> const two{0, 1};
  std::array<int, 3> const three{0, 1, 2};
  std::vector<int> first, second;
  for (int i = 0; i < 6; ++i) {
    first.push_back(affinity.next_gpu(0, two).value());
    second.push_back(affinity.next_gpu(0, three).value());
  }
  REQUIRE(first == std::vector<int>{0, 1, 0, 1, 0, 1});
  REQUIRE(second == std::vector<int>{0, 1, 2, 0, 1, 2});

  // The same list on another node is a separate walk.
  REQUIRE(affinity.next_gpu(1, two) == 0);
  REQUIRE(affinity.next_gpu(0, two) == 0);
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
