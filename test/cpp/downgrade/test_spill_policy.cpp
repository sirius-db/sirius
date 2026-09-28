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
#include "downgrade/spill_policy.hpp"

#include <optional>
#include <vector>

using sirius::parallel::select_next_spill_candidate;

namespace {
constexpr std::size_t GB = 1ull << 30;
}

TEST_CASE("select_next_spill_candidate without target follows policy order", "[spill_policy]")
{
  std::vector<std::size_t> sizes{4 * GB, 2 * GB, 1 * GB};
  std::vector<bool> used{false, false, false};

  auto pick = select_next_spill_candidate(sizes, used, std::nullopt);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 0);

  used[0] = true;
  pick    = select_next_spill_candidate(sizes, used, std::nullopt);
  REQUIRE(*pick == 1);
}

TEST_CASE("select_next_spill_candidate best-fits the final pick", "[spill_policy]")
{
  // Policy order would spill 5 GB for a 512 MB deficit; best-fit takes the 512 MB batch.
  std::vector<std::size_t> sizes{5 * GB, 4 * GB, GB / 2, 2 * GB};
  std::vector<bool> used{false, false, false, false};

  auto pick = select_next_spill_candidate(sizes, used, GB / 2);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 2);
}

TEST_CASE("select_next_spill_candidate breaks best-fit ties in policy order", "[spill_policy]")
{
  std::vector<std::size_t> sizes{2 * GB, 2 * GB, 2 * GB};
  std::vector<bool> used{false, false, false};

  auto pick = select_next_spill_candidate(sizes, used, GB);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 0);
}

TEST_CASE("select_next_spill_candidate falls back to policy order when nothing covers",
          "[spill_policy]")
{
  // Deficit larger than any single candidate: fall back to policy order.
  std::vector<std::size_t> sizes{2 * GB, 3 * GB, 1 * GB};
  std::vector<bool> used{false, false, false};

  auto pick = select_next_spill_candidate(sizes, used, 10 * GB);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 0);
}

TEST_CASE("select_next_spill_candidate skips dispatched candidates", "[spill_policy]")
{
  std::vector<std::size_t> sizes{4 * GB, 2 * GB, 1 * GB};
  std::vector<bool> used{true, false, true};

  auto pick = select_next_spill_candidate(sizes, used, std::nullopt);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 1);

  pick = select_next_spill_candidate(sizes, used, 3 * GB);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 1);
}

TEST_CASE("select_next_spill_candidate returns nullopt when exhausted", "[spill_policy]")
{
  std::vector<std::size_t> sizes{GB, GB};
  std::vector<bool> used{true, true};

  REQUIRE_FALSE(select_next_spill_candidate(sizes, used, std::nullopt).has_value());
  REQUIRE_FALSE(select_next_spill_candidate(sizes, used, GB).has_value());
  REQUIRE_FALSE(
    select_next_spill_candidate(std::vector<std::size_t>{}, std::vector<bool>{}, std::nullopt)
      .has_value());
}

TEST_CASE("select_next_spill_candidate models the q9 cliff repair", "[spill_policy]")
{
  // Partitions of ~4.5-4.9 GB with a marginal ~600 MB overflow: policy order would spill
  // 4.9 GB (8x the need).
  std::vector<std::size_t> sizes{4900ull << 20, 4600ull << 20, 4500ull << 20, 700ull << 20};
  std::vector<bool> used{false, false, false, false};

  auto pick = select_next_spill_candidate(sizes, used, 600ull << 20);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 3);

  // With no right-sized batch, overshoot is bounded by the smallest covering one.
  std::vector<std::size_t> only_big{4900ull << 20, 4600ull << 20, 4500ull << 20};
  std::vector<bool> unused{false, false, false};
  pick = select_next_spill_candidate(only_big, unused, 600ull << 20);
  REQUIRE(pick.has_value());
  REQUIRE(*pick == 2);  // smallest covering, not the 4.9 GB policy pick
}
