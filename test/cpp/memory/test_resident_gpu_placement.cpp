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

// test
#include <catch.hpp>

// sirius
#include <memory/resident_gpu_placement.hpp>

// standard library
#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using sirius::memory::resident_gpu_placement;

namespace {

using bytes_list = std::vector<std::pair<int, std::size_t>>;

}  // namespace

TEST_CASE("data on an admitted GPU stays on that GPU", "[resident_gpu_placement]")
{
  resident_gpu_placement placement;
  std::array<int, 3> const admitted{0, 1, 2};

  // No admitted set recorded: every GPU counts as admitted.
  REQUIRE(placement.gpu_for_home(3, {}) == 3);
  for (int const home : admitted) {
    REQUIRE(placement.gpu_for_home(home, admitted) == home);
  }

  // None of those calls advanced the walk over the admitted set.
  REQUIRE(placement.gpu_for_home(3, admitted) == 0);
}

TEST_CASE("chunks on an excluded GPU spread evenly over the admitted GPUs",
          "[resident_gpu_placement]")
{
  // Chunks round-robin on four GPUs, three of them admitted: GPU 3's chunks have to move.
  resident_gpu_placement placement;
  std::array<int, 3> const admitted{0, 1, 2};

  // The walk carries over from one query to the next.
  for (int query = 0; query < 4; ++query) {
    std::map<int, int> tasks;
    for (int chunk = 0; chunk < 24; ++chunk) {
      ++tasks[placement.gpu_for_home(chunk % 4, admitted)];
    }
    INFO("query=" << query);
    REQUIRE(tasks.size() == 3u);
    // 6 of its own plus a third of GPU 3's 6.
    REQUIRE(tasks[0] == 8);
    REQUIRE(tasks[1] == 8);
    REQUIRE(tasks[2] == 8);
    (void)placement.gpu_for_home(3, admitted);  // shift the next query's starting point
  }

  // An uneven count leaves the admitted GPUs at most one task apart.
  std::map<int, int> moved;
  for (int chunk = 0; chunk < 5; ++chunk) {
    ++moved[placement.gpu_for_home(3, admitted)];
  }
  auto const [fewest, most] = std::minmax_element(
    moved.begin(), moved.end(), [](auto const& a, auto const& b) { return a.second < b.second; });
  REQUIRE(moved.size() == 3u);
  REQUIRE(most->second - fewest->second <= 1);
}

TEST_CASE("the most-bytes pick counts only the admitted GPUs", "[resident_gpu_placement]")
{
  resident_gpu_placement placement;
  std::array<int, 3> const admitted{0, 1, 2};

  // GPU 3 holds the most, but the query does not run there.
  bytes_list const mixed{{3, 1000}, {1, 10}, {2, 40}};
  REQUIRE(placement.gpu_for_bytes(mixed, admitted) == 2);

  // Without a subset the overall largest holder is picked.
  REQUIRE(placement.gpu_for_bytes(mixed, {}) == 3);
  std::array<int, 4> const all{0, 1, 2, 3};
  REQUIRE(placement.gpu_for_bytes(mixed, all) == 3);

  // A tie goes to the first of the tied GPUs in range order.
  bytes_list const tied{{2, 40}, {1, 40}, {3, 40}};
  REQUIRE(placement.gpu_for_bytes(tied, admitted) == 2);
  REQUIRE(placement.gpu_for_bytes(tied, {}) == 2);

  // The task creator's own container.
  std::unordered_map<int, std::size_t> const map{{3, 1000}, {1, 10}};
  REQUIRE(placement.gpu_for_bytes(map, admitted) == 1);
  REQUIRE(placement.gpu_for_bytes(map, {}) == 3);

  // No data gives no GPU, with or without a subset.
  REQUIRE_FALSE(placement.gpu_for_bytes(bytes_list{}, admitted).has_value());
  REQUIRE_FALSE(placement.gpu_for_bytes(bytes_list{}, {}).has_value());

  // None of those picks advanced the walk.
  REQUIRE(placement.gpu_for_home(3, admitted) == 0);
}

TEST_CASE("data held only by excluded GPUs walks the admitted GPUs", "[resident_gpu_placement]")
{
  resident_gpu_placement placement;
  std::array<int, 2> const admitted{0, 1};
  bytes_list const excluded_only{{2, 100}, {3, 50}};

  // By-bytes picks and single-home picks share one walk.
  REQUIRE(placement.gpu_for_bytes(excluded_only, admitted) == 0);
  REQUIRE(placement.gpu_for_home(2, admitted) == 1);
  REQUIRE(placement.gpu_for_bytes(excluded_only, admitted) == 0);
  REQUIRE(placement.gpu_for_home(3, admitted) == 1);
}

TEST_CASE("queries admitted to different sets walk independently", "[resident_gpu_placement]")
{
  // On a counter shared by both sets, the first query would get GPU 0 every time.
  resident_gpu_placement placement;
  std::array<int, 2> const two{0, 1};
  std::array<int, 3> const three{0, 1, 2};
  std::vector<int> first, second;
  for (int i = 0; i < 6; ++i) {
    first.push_back(placement.gpu_for_home(3, two));
    second.push_back(placement.gpu_for_home(3, three));
  }
  REQUIRE(first == std::vector<int>{0, 1, 0, 1, 0, 1});
  REQUIRE(second == std::vector<int>{0, 1, 2, 0, 1, 2});
}

TEST_CASE("concurrent placement of excluded-GPU chunks stays balanced", "[resident_gpu_placement]")
{
  resident_gpu_placement placement;
  std::array<int, 3> const admitted{0, 1, 2};
  constexpr int num_threads      = 6;
  constexpr int picks_per_thread = 300;

  std::array<std::array<int, 3>, num_threads> counts{};
  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&, t] {
      for (int i = 0; i < picks_per_thread; ++i) {
        ++counts[t][placement.gpu_for_home(3, admitted)];
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (int gpu = 0; gpu < 3; ++gpu) {
    int total = 0;
    for (auto const& per_thread : counts) {
      total += per_thread[gpu];
    }
    REQUIRE(total == num_threads * picks_per_thread / 3);
  }
}
