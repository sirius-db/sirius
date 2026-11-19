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
 * Host-only tests of Static Selection (`op/dynamic_filter/static_filter_selection.hpp`): the
 * Negligible-Share Rule, Filtered Decode's preference, and their combination for a filter that is
 * not published yet. The Consumer's use of the same functions is covered by
 * `test_dynamic_filter_merge.cpp`.
 */

#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "op/dynamic_filter/static_filter_selection.hpp"

#include <catch.hpp>

#include <algorithm>
#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

namespace {

namespace op = sirius::op;
using op::membership_filter_kind;
using op::static_filter_facts;
using estimates = std::vector<std::optional<double>>;

constexpr static_filter_facts small_in_list(std::uint64_t keys, std::optional<double> keep = {})
{
  return {.kind = membership_filter_kind::small_in_list, .keys = keys, .estimated_keep = keep};
}

constexpr static_filter_facts hash_in_list(std::uint64_t keys, std::optional<double> keep = {})
{
  return {.kind = membership_filter_kind::hash_in_list, .keys = keys, .estimated_keep = keep};
}

constexpr static_filter_facts bloom(std::optional<double> keep = {})
{
  return {.kind = membership_filter_kind::bloom, .estimated_keep = keep};
}

/**
 * @brief A filter of a given kind that is none of the concrete Membership Filter classes.
 */
class stub_filter final : public op::sirius_dynamic_filter {
 public:
  explicit stub_filter(op::sirius_dynamic_filter_kind kind) : _kind{kind} {}
  [[nodiscard]] op::sirius_dynamic_filter_kind kind() const override { return _kind; }

 private:
  op::sirius_dynamic_filter_kind _kind;
};

// The rule is usable in constant expressions.
static_assert(op::negligible_after_stronger(0.5, std::array<std::optional<double>, 1>{0.1}));
static_assert(!op::negligible_after_stronger(0.5, std::array<std::optional<double>, 1>{0.9}));
static_assert(op::decode_preference(small_in_list(12), bloom()) < 0);

}  // namespace

TEST_CASE("the removed share counts only strictly stronger estimates",
          "[dynamic_filter][static_selection]")
{
  SECTION("a filter without an estimate has no removed share")
  {
    CHECK_FALSE(op::removed_share_after_stronger(std::nullopt, estimates{0.01}));
    CHECK_FALSE(op::negligible_after_stronger(std::nullopt, estimates{0.01}));
  }
  SECTION("no other filter, or none stronger, leaves the share empty")
  {
    CHECK_FALSE(op::removed_share_after_stronger(0.5, estimates{}));
    CHECK_FALSE(op::removed_share_after_stronger(0.5, estimates{std::nullopt, 0.5, 0.9}));
    CHECK_FALSE(op::negligible_after_stronger(0.999, estimates{0.999}));
  }
  SECTION("the filter's own estimate among the others changes nothing")
  {
    auto const alone    = op::removed_share_after_stronger(0.5, estimates{0.1});
    auto const included = op::removed_share_after_stronger(0.5, estimates{0.5, 0.1});
    REQUIRE(alone);
    REQUIRE(included);
    CHECK(*alone == *included);
  }
  SECTION("every stronger estimate multiplies in")
  {
    auto const removed = op::removed_share_after_stronger(0.5, estimates{0.5, 0.25, 0.75, 0.5});
    REQUIRE(removed);
    CHECK(*removed == 0.25 * 0.5);
  }
}

TEST_CASE("the Negligible-Share Rule skips a filter that removes under a tenth of the batch",
          "[dynamic_filter][static_selection]")
{
  SECTION("a Bloom filter behind a selective IN-list is negligible")
  {
    // TPC-H q8: an IN-list keeping 0.67% precedes a Bloom filter keeping 6.1%.
    CHECK(op::negligible_after_stronger(0.061, estimates{0.0067, 0.061}));
    // TPC-H q21: an IN-list keeping 4% precedes a Bloom filter keeping 48.7%.
    CHECK(op::negligible_after_stronger(0.487, estimates{0.04, 0.487}));
  }
  SECTION("the threshold itself is not negligible")
  {
    auto const removed = op::removed_share_after_stronger(0.5, estimates{0.2});
    REQUIRE(removed);
    REQUIRE(*removed == op::negligible_removed_share);
    CHECK_FALSE(op::negligible_after_stronger(0.5, estimates{0.2}));
    CHECK(op::negligible_after_stronger(0.5, estimates{0.19}));
  }
  SECTION("a strong filter behind a weak one stays")
  {
    CHECK_FALSE(op::negligible_after_stronger(0.05, estimates{0.05, 0.9}));
    CHECK_FALSE(op::negligible_after_stronger(0.3, estimates{0.25}));
  }
}

TEST_CASE("decode preference orders by kind and then by key count",
          "[dynamic_filter][static_selection]")
{
  CHECK(op::decode_preference(small_in_list(12), hash_in_list(1)) < 0);
  CHECK(op::decode_preference(hash_in_list(1'000'000), bloom()) < 0);
  CHECK(op::decode_preference(bloom(), static_filter_facts{}) < 0);
  CHECK(op::decode_preference(hash_in_list(10), hash_in_list(11)) < 0);
  CHECK(op::decode_preference(hash_in_list(11), hash_in_list(10)) > 0);
  CHECK(op::decode_preference(bloom(), bloom()) == 0);

  SECTION("the estimate does not take part, although equality compares it")
  {
    CHECK(op::decode_preference(hash_in_list(10, 0.9), hash_in_list(10, 0.1)) == 0);
    CHECK(hash_in_list(10, 0.9) != hash_in_list(10, 0.1));
    CHECK(op::decode_preference(bloom(0.01), hash_in_list(10, 0.99)) > 0);
  }
}

TEST_CASE("facts of a filter that is no concrete Membership Filter",
          "[dynamic_filter][static_selection]")
{
  stub_filter zone_map{op::sirius_dynamic_filter_kind::ZONE_MAP};
  CHECK(op::facts_of(zone_map) == static_filter_facts{});

  stub_filter approximate{op::sirius_dynamic_filter_kind::BLOOM};
  approximate.set_estimated_selectivity(0.25);
  CHECK(op::facts_of(approximate) == bloom(0.25));
}

TEST_CASE(
  "a filter is statically selected unless the decode passes it over and its share is "
  "negligible",
  "[dynamic_filter][static_selection]")
{
  std::vector const strong_in_list{hash_in_list(1'000'000, 0.0067)};
  // Behind a filter keeping half of a batch, a filter keeping 60% of it still removes a fifth.
  std::vector const mild_in_list{hash_in_list(1'000'000, 0.5)};

  SECTION("a negligible Bloom filter behind an IN-list is not selected at cap 1")
  {
    CHECK_FALSE(op::statically_selected(bloom(0.061), strong_in_list, 1));
  }
  SECTION("a second decode slot carries it")
  {
    CHECK(op::statically_selected(bloom(0.061), strong_in_list, 2));
  }
  SECTION("without Filtered Decode slots only the Negligible-Share Rule decides")
  {
    CHECK_FALSE(op::statically_selected(bloom(0.061), strong_in_list, 0));
    CHECK(op::statically_selected(bloom(0.6), mild_in_list, 0));
    CHECK(op::statically_selected(bloom(0.061), {}, 0));
  }
  SECTION("an empty Channel selects everything")
  {
    CHECK(op::statically_selected(bloom(0.999), {}, 1));
    CHECK(op::statically_selected(static_filter_facts{}, {}, 1));
  }
  SECTION("a non-negligible filter is selected wherever it sorts")
  {
    CHECK(op::statically_selected(bloom(0.6), mild_in_list, 1));
  }
  SECTION("an IN-list arriving after a stronger Bloom filter takes the decode slot")
  {
    // TPC-H q20: the Bloom filter is stronger by estimate, but the decode prefers the IN-list.
    std::vector const strong_bloom{bloom(0.001)};
    CHECK(op::statically_selected(hash_in_list(500'000, 0.05), strong_bloom, 1));
  }
  SECTION("a tie in decode preference leaves the filter selected")
  {
    // The Consumer breaks the tie by its snapshot order, which may favor either filter.
    std::vector const equal{hash_in_list(100, 0.01)};
    CHECK(op::statically_selected(hash_in_list(100, 0.05), equal, 1));
    CHECK_FALSE(op::statically_selected(hash_in_list(101, 0.05), equal, 1));
    CHECK(op::statically_selected(bloom(0.05), std::vector{bloom(0.001)}, 1));
    CHECK_FALSE(op::statically_selected(bloom(0.05), std::vector{bloom(0.001)}, 0));
  }
  SECTION("a published filter without an estimate takes a slot but shadows nothing")
  {
    std::vector const unknown{hash_in_list(100)};
    CHECK(op::statically_selected(bloom(0.05), unknown, 1));
  }
  SECTION("a candidate without an estimate is always selected")
  {
    CHECK(op::statically_selected(bloom(), strong_in_list, 1));
  }
}

TEST_CASE("a filter that is not statically selected is one the Negligible-Share Rule skips",
          "[dynamic_filter][static_selection]")
{
  std::mt19937 generator{20261119};
  std::uniform_int_distribution<int> kind{1, 3};
  std::uniform_int_distribution<std::uint64_t> keys{1, 64};
  std::uniform_real_distribution<double> keep{0.0, 1.0};
  std::bernoulli_distribution known{0.8};
  auto const draw = [&] {
    static_filter_facts facts{.kind = static_cast<membership_filter_kind>(kind(generator))};
    if (facts.kind != membership_filter_kind::bloom) { facts.keys = keys(generator); }
    if (known(generator)) { facts.estimated_keep = keep(generator); }
    return facts;
  };
  std::size_t rejected = 0;
  for (int round = 0; round < 2000; ++round) {
    auto const candidate = draw();
    std::vector<static_filter_facts> published(round % 4);
    std::ranges::generate(published, draw);
    auto const cap = static_cast<std::size_t>(round % 3);
    if (op::statically_selected(candidate, published, cap)) { continue; }
    ++rejected;
    auto const all = published | std::views::transform(&static_filter_facts::estimated_keep);
    REQUIRE(op::negligible_after_stronger(candidate.estimated_keep, all));
    auto const preferred = std::ranges::count_if(published, [&](static_filter_facts const& other) {
      return op::decode_preference(other, candidate) < 0;
    });
    REQUIRE(std::cmp_greater_equal(preferred, cap));
    // More decode slots than published filters always carry the candidate.
    REQUIRE(op::statically_selected(candidate, published, published.size() + 1));
  }
  REQUIRE(rejected > 0);
}
