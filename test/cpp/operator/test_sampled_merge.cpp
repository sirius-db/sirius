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
#include "op/order/sampled_merge.hpp"

namespace sm = sirius::op::sampled_merge;

TEST_CASE("sampled merge arithmetic is checked over 64-bit coordinates", "[sampled_merge][cpu]")
{
  auto max = std::numeric_limits<uint64_t>::max();
  for (auto k : {uint64_t{1}, uint64_t{128}, max / 4, max}) {
    for (auto r : {uint64_t{1}, uint64_t{1024}, max - 1, max}) {
      auto extreme = sm::choose_parameters(k, r);
      CHECK(extreme.stride >= 1);
      CHECK(extreme.sample_step >= 1);
      CHECK(extreme.uncertainty <= r / 4);
      CHECK(extreme.max_tile_rows <= r);
    }
  }
  CHECK(sm::sample_count(8, 3) == 2);
  CHECK(sm::sample_row(1, 3) == 5);
  CHECK(sm::sample_row(max, 1) == max);
  CHECK(sm::conceptual_cut(max - 1, {max, max - 1}, 7, 11) == 11);
  CHECK(sm::conceptual_cut(max, {max - 1, max - 1}, 7, 11) == 7);
  CHECK(sm::conceptual_cut(max, {max, max - 1}, 7, 11) == max);
  CHECK_THROWS_AS(sm::choose_parameters(0, 1), std::invalid_argument);
  CHECK_THROWS_AS(sm::choose_parameters(1, 0), std::invalid_argument);
  CHECK_THROWS_AS(sm::sample_count(1, 0), std::invalid_argument);
  CHECK_THROWS_AS(sm::sample_row(1, 0), std::invalid_argument);
  CHECK_THROWS_AS(sm::sample_row(max, 2), std::overflow_error);
  CHECK_THROWS_AS(sm::conceptual_cut(0, {0, max}, 0, 0), std::overflow_error);
  CHECK_THROWS_AS(sm::checked_add(max, 1), std::overflow_error);
  CHECK_THROWS_AS(sm::checked_mul(max, 2), std::overflow_error);
}

TEST_CASE("coalescing makes one byte-targeted pass with native row caps", "[sampled_merge][cpu]")
{
  std::vector<sm::contribution_size> inputs{{3, 30}, {2, 20}, {4, 80}, {1, 100}, {2, 10}, {2, 10}};
  auto groups      = sm::plan_coalescing(inputs, 100, 100);
  std::size_t next = 0;
  for (auto group : groups) {
    REQUIRE(group.begin == next);
    REQUIRE(group.end > group.begin);
    REQUIRE(group.end <= inputs.size());
    uint64_t rows = 0, bytes = 0;
    for (auto i = group.begin; i < group.end; ++i) {
      rows += inputs[i].rows;
      bytes += inputs[i].bytes;
    }
    CHECK(group.rows == rows);
    CHECK(group.bytes == bytes);
    CHECK((bytes <= 100 || group.end == group.begin + 1));
    next = group.end;
  }
  CHECK(next == inputs.size());
  CHECK(sm::plan_coalescing({}, 1, 1).empty());
  auto capped = sm::plan_coalescing(inputs, 1000, 5);
  for (auto group : capped) {
    CHECK(group.rows <= 5);
  }
  CHECK_THROWS_AS(sm::plan_coalescing(inputs, 0, 5), std::invalid_argument);
  CHECK_THROWS_AS(sm::plan_coalescing(inputs, 1, 0), std::invalid_argument);
  CHECK_THROWS_AS(sm::plan_coalescing(inputs, 1, 2), std::invalid_argument);
  std::vector<sm::contribution_size> empty_input{{0, 10}};
  CHECK_THROWS_AS(sm::plan_coalescing(empty_input, 1, 2), std::invalid_argument);
  auto max = std::numeric_limits<uint64_t>::max();
  std::vector<sm::contribution_size> huge{{1, max - 1}, {1, 2}, {max, max}};
  auto extreme = sm::plan_coalescing(huge, max, max);
  REQUIRE(extreme.size() == 3);
  CHECK(extreme[0].bytes == max - 1);
  CHECK(extreme[2].rows == max);
}
