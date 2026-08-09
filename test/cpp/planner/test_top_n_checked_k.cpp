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

#include "planner/top_n_checked_k.hpp"

#include <catch.hpp>

#include <limits>

using sirius::planner::checked_top_n_k;

namespace {
constexpr auto k_size_max = std::numeric_limits<std::size_t>::max();
constexpr auto k_row_max = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
}  // namespace

TEST_CASE("checked_top_n_k admits plain limit and limit-plus-offset shapes",
          "[dynamic_filter][top_n]")
{
  REQUIRE(checked_top_n_k(1, 0) == 1);
  REQUIRE(checked_top_n_k(7, 5) == 12);
  REQUIRE(checked_top_n_k(k_row_max, 0) == k_row_max);
  REQUIRE(checked_top_n_k(1, k_row_max - 1) == k_row_max);
}

TEST_CASE("checked_top_n_k rejects zero limits, wrapping sums, and K beyond the cuDF row space",
          "[dynamic_filter][top_n]")
{
  REQUIRE_FALSE(checked_top_n_k(0, 0).has_value());
  REQUIRE_FALSE(checked_top_n_k(0, 5).has_value());
  // One past the cudf::size_type row bound, from either operand.
  REQUIRE_FALSE(checked_top_n_k(k_row_max, 1).has_value());
  REQUIRE_FALSE(checked_top_n_k(1, k_row_max).has_value());
  // std::size_t wrap-around must reject, never truncate.
  REQUIRE_FALSE(checked_top_n_k(k_size_max, 1).has_value());
  REQUIRE_FALSE(checked_top_n_k(1, k_size_max).has_value());
  REQUIRE_FALSE(checked_top_n_k(k_size_max, k_size_max).has_value());
}
