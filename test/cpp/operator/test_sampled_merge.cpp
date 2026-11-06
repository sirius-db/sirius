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
#include "utils/sampled_merge_reference.hpp"

#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <tuple>

namespace sm = sirius::op::sampled_merge;

namespace {
void verify(bool condition)
{
  if (!condition) { throw std::runtime_error("Sampled merge reference invariant failed"); }
}

template <typename T, typename Less = std::less<T>>
void verify_plan(std::vector<std::vector<T>> const& inputs, uint64_t limit, Less less = {})
{
  auto plan = sirius::test::reference_sampled_cuts(inputs, limit, less);
  std::vector<sm::coordinate> exact;
  uint64_t k = 0;
  for (uint64_t i = 0; i < inputs.size(); ++i) {
    k += !inputs[i].empty();
    for (uint64_t j = 0; j < inputs[i].size(); ++j) {
      exact.push_back({i, j});
    }
    verify(plan.cuts[i].front() == 0 && plan.cuts[i].back() == inputs[i].size());
    verify(std::is_sorted(plan.cuts[i].begin(), plan.cuts[i].end()));
  }
  auto conceptual_less = [&](sm::coordinate a, sm::coordinate b) {
    auto const& x = inputs[a.input][a.row];
    auto const& y = inputs[b.input][b.row];
    if (less(x, y)) { return true; }
    if (less(y, x)) { return false; }
    return a < b;
  };
  std::sort(exact.begin(), exact.end(), conceptual_less);
  std::vector<sm::coordinate> concatenated;
  for (std::size_t t = 0; t <= plan.splitters.size(); ++t) {
    std::vector<sm::coordinate> tile;
    for (uint64_t i = 0; i < inputs.size(); ++i) {
      for (auto j = plan.cuts[i][t]; j < plan.cuts[i][t + 1]; ++j) {
        tile.push_back({i, j});
      }
    }
    verify(tile.size() <= limit);
    std::sort(tile.begin(), tile.end(), conceptual_less);
    concatenated.insert(concatenated.end(), tile.begin(), tile.end());
    if (t < plan.splitters.size()) {
      auto rank = std::upper_bound(exact.begin(), exact.end(), plan.splitters[t], conceptual_less) -
                  exact.begin();
      // An independent global oracle: every contribution's cut must describe
      // exactly this prefix, not merely satisfy an aggregate cardinality bound.
      verify(concatenated.size() == static_cast<std::size_t>(rank));
      std::vector<uint64_t> counts(inputs.size());
      for (auto it = exact.begin(); it != exact.begin() + rank; ++it) {
        ++counts[it->input];
      }
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        verify(counts[i] == plan.cuts[i][t + 1]);
      }
      auto p = sm::choose_parameters(k, limit);
      auto q = (t + 1) * p.sample_step;
      verify(p.stride * q <= static_cast<uint64_t>(rank));
      verify(static_cast<uint64_t>(rank) <= p.stride * q + p.uncertainty);
    }
  }
  verify(concatenated == exact);  // order, conservation and exact row identities
}
}  // namespace

TEST_CASE("sampled merge arithmetic is checked over 64-bit coordinates", "[sampled_merge][cpu]")
{
  auto max = std::numeric_limits<uint64_t>::max();
  auto p   = sm::choose_parameters(128, 1'048'576);
  CHECK(p.stride == 2049);
  CHECK(p.sample_step == 383);
  CHECK(p.uncertainty == 262144);
  CHECK(p.max_tile_rows == 1046911);
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
  auto groups = sm::plan_coalescing(inputs, 100, 100);
  REQUIRE(groups.size() == 4);
  CHECK(groups[0].begin == 0);
  CHECK(groups[0].end == 2);
  CHECK(groups[0].bytes == 50);
  CHECK(groups[0].rows == 5);
  CHECK(groups[1].begin == 2);
  CHECK(groups[1].end == 3);
  CHECK(groups[2].begin == 3);
  CHECK(groups[2].end == 4);
  CHECK(groups[3].begin == 4);
  CHECK(groups[3].end == 6);
  CHECK(groups[3].bytes == 20);
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

TEST_CASE("sampled cuts exhaust small sorted binary inputs", "[sampled_merge][cpu]")
{
  std::vector<std::vector<int>> runs;
  for (int n = 0; n <= 4; ++n) {
    for (int zeros = 0; zeros <= n; ++zeros) {
      std::vector<int> run(n, 1);
      std::fill_n(run.begin(), zeros, 0);
      runs.push_back(std::move(run));
    }
  }
  uint64_t cases = 0;
  for (auto const& a : runs) {
    for (auto const& b : runs) {
      for (auto const& c : runs) {
        for (uint64_t limit = 1; limit <= 9; ++limit) {
          REQUIRE_NOTHROW(verify_plan(std::vector{a, b, c}, limit));
          ++cases;
        }
      }
    }
  }
  CHECK(cases == 30375);
}

TEST_CASE("sampled cuts conserve randomized unequal contributions", "[sampled_merge][cpu]")
{
  std::mt19937_64 rng(0x71'1ed);
  for (uint64_t trial = 0; trial < 100000; ++trial) {
    INFO("trial=" << trial);
    std::vector<std::vector<int>> inputs(rng() % 12);
    for (auto& input : inputs) {
      input.resize(rng() % 36);
      for (auto& key : input) {
        key = rng() % (trial % 2 ? 3 : 1000);
      }
      std::sort(input.begin(), input.end());
    }
    REQUIRE_NOTHROW(verify_plan(inputs, 1 + rng() % 128));
  }
}

TEST_CASE("sampled cut endpoints cover tails and giant peer groups", "[sampled_merge][cpu]")
{
  REQUIRE_NOTHROW(verify_plan<int>({}, 1));
  REQUIRE_NOTHROW(verify_plan<int>({{}, {}}, 1));
  REQUIRE_NOTHROW(verify_plan<int>({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}, 4));
  REQUIRE_NOTHROW(verify_plan<int>({std::vector<int>(10000, 1), {1}, {}, {1, 1}}, 101));
  auto no_samples = sirius::test::reference_sampled_cuts<int>({{1}, {2}}, 2);
  CHECK(no_samples.samples == 0);
  CHECK(no_samples.splitters.empty());
  CHECK(no_samples.cuts == std::vector<std::vector<uint64_t>>{{0, 1}, {0, 1}});
  // s=1, g=R: the final splitter is exactly the last sample, creating an empty tail.
  auto last = sirius::test::reference_sampled_cuts<int>({{1, 1}, {1, 1}}, 2);
  CHECK(last.splitters.back() == sm::coordinate{1, 1});
  CHECK(last.cuts[0][last.cuts[0].size() - 2] == 2);
  CHECK(last.cuts[1][last.cuts[1].size() - 2] == 2);
  CHECK_THROWS_AS(sirius::test::reference_sampled_cuts<int>({{2, 1}}, 1), std::invalid_argument);
}

TEST_CASE("conceptual cuts use comparator equivalence for nulls NaNs and compound keys",
          "[sampled_merge][cpu]")
{
  using key = std::tuple<std::optional<double>, std::string>;
  // SQL-style ASC NULLS LAST, NaNs after finite values, second key descending.
  auto less = [](key const& a, key const& b) {
    auto const& [x, xs] = a;
    auto const& [y, ys] = b;
    if (x.has_value() != y.has_value()) { return x.has_value(); }
    if (x && y) {
      if (std::isnan(*x) != std::isnan(*y)) { return !std::isnan(*x); }
      if (*x < *y) { return true; }
      if (*y < *x) { return false; }
    }
    return xs > ys;
  };
  auto nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<std::vector<key>> inputs{
    {{nan, "a"}, {std::nullopt, "b"}, {2, "x"}, {-0.0, "a"}, {0.0, "a"}},
    {{nan, "a"}, {nan, "z"}, {std::nullopt, "b"}, {-1, "a"}, {2, "y"}},
    {{std::nullopt, "a"}, {nan, "a"}, {2, "y"}}};
  for (auto& input : inputs) {
    std::sort(input.begin(), input.end(), less);
  }
  for (uint64_t limit = 1; limit <= 20; ++limit) {
    REQUIRE_NOTHROW(verify_plan(inputs, limit, less));
  }
}
