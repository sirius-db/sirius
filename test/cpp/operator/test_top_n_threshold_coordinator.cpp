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

#include "op/dynamic_filter/dynamic_filter_stats.hpp"
#include "op/dynamic_filter/top_n_threshold_coordinator.hpp"

#include <catch.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <random>
#include <thread>
#include <vector>

using sirius::op::dynamic_filter_stats;
using sirius::op::exact_host_key_tuple;
using sirius::op::exact_host_scalar;
using sirius::op::threshold_offer_result;
using sirius::op::top_n_key_semantics;
using sirius::op::top_n_threshold_coordinator;
using sirius::op::top_n_threshold_witness;

namespace {

constexpr auto k_int32 = cudf::data_type{cudf::type_id::INT32};

std::vector<top_n_key_semantics> single_asc_key()
{
  return {{.storage_type = k_int32,
           .order        = cudf::order::ASCENDING,
           .null_order   = cudf::null_order::AFTER}};
}

top_n_threshold_witness witness(std::vector<std::optional<std::int32_t>> const& values)
{
  std::vector<std::optional<exact_host_scalar>> components;
  components.reserve(values.size());
  for (auto const& v : values) {
    components.push_back(v ? std::optional{exact_host_scalar{*v, k_int32}} : std::nullopt);
  }
  return top_n_threshold_witness{exact_host_key_tuple{std::move(components)}, {}};
}

std::int32_t head_value(exact_host_key_tuple const& tuple)
{
  return std::get<std::int32_t>(tuple.component(0)->value());
}

}  // namespace

TEST_CASE("top_n_threshold_coordinator starts empty and frozen semantics are readable",
          "[dynamic_filter][top_n]")
{
  top_n_threshold_coordinator coordinator{12, single_asc_key(), true};
  REQUIRE_FALSE(coordinator.tightest_boundary().has_value());
  REQUIRE(coordinator.boundary_update_count() == 0);
  REQUIRE(coordinator.k() == 12);
  REQUIRE(coordinator.keys().size() == 1);
  REQUIRE(coordinator.lex_admitted());
}

TEST_CASE("top_n_threshold_coordinator::offer tightens monotonically and counts",
          "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  top_n_threshold_coordinator coordinator{3, single_asc_key(), true, &stats};

  // Stage 1 has no channel targets, so an accepted offer reports NO_ACCEPTING_TARGET.
  REQUIRE(coordinator.offer(witness({10})) == threshold_offer_result::NO_ACCEPTING_TARGET);
  REQUIRE(coordinator.boundary_update_count() == 1);
  REQUIRE(head_value(*coordinator.tightest_boundary()) == 10);

  SECTION("a tighter boundary replaces the shared one")
  {
    REQUIRE(coordinator.offer(witness({4})) == threshold_offer_result::NO_ACCEPTING_TARGET);
    REQUIRE(coordinator.boundary_update_count() == 2);
    REQUIRE(head_value(*coordinator.tightest_boundary()) == 4);
    REQUIRE(stats.top_n_offers.load() == 2);
    REQUIRE(stats.top_n_offers_not_tighter.load() == 0);
  }
  SECTION("a looser or tying boundary is rejected without an update")
  {
    REQUIRE(coordinator.offer(witness({10})) == threshold_offer_result::NOT_TIGHTER);
    REQUIRE(coordinator.offer(witness({11})) == threshold_offer_result::NOT_TIGHTER);
    REQUIRE(coordinator.boundary_update_count() == 1);
    REQUIRE(head_value(*coordinator.tightest_boundary()) == 10);
    REQUIRE(stats.top_n_offers.load() == 3);
    REQUIRE(stats.top_n_offers_not_tighter.load() == 2);
  }
  SECTION("a null head component is unsupported and changes nothing")
  {
    REQUIRE(coordinator.offer(witness({std::nullopt})) ==
            threshold_offer_result::UNSUPPORTED_BOUNDARY);
    REQUIRE(coordinator.boundary_update_count() == 1);
    REQUIRE(head_value(*coordinator.tightest_boundary()) == 10);
    REQUIRE(stats.top_n_offers_unsupported.load() == 1);
  }
}

TEST_CASE("top_n_threshold_coordinator honors multi-key tightness with null tails",
          "[dynamic_filter][top_n]")
{
  std::vector<top_n_key_semantics> keys{{.storage_type = k_int32,
                                         .order        = cudf::order::ASCENDING,
                                         .null_order   = cudf::null_order::AFTER},
                                        {.storage_type = k_int32,
                                         .order        = cudf::order::ASCENDING,
                                         .null_order   = cudf::null_order::AFTER}};
  top_n_threshold_coordinator coordinator{5, std::move(keys), true};

  REQUIRE(coordinator.offer(witness({7, std::nullopt})) ==
          threshold_offer_result::NO_ACCEPTING_TARGET);
  // Equal head, engaged tail: with nulls-last the value orders before the null -- tighter.
  REQUIRE(coordinator.offer(witness({7, 3})) == threshold_offer_result::NO_ACCEPTING_TARGET);
  // Equal tuple ties are rejected strictly.
  REQUIRE(coordinator.offer(witness({7, 3})) == threshold_offer_result::NOT_TIGHTER);
  REQUIRE(coordinator.boundary_update_count() == 2);
}

TEST_CASE("top_n_threshold_coordinator rejects offers after finish and cancel",
          "[dynamic_filter][top_n]")
{
  top_n_threshold_coordinator coordinator{3, single_asc_key(), true};
  REQUIRE(coordinator.offer(witness({10})) == threshold_offer_result::NO_ACCEPTING_TARGET);

  SECTION("finish is idempotent and keeps the boundary readable")
  {
    coordinator.finish();
    coordinator.finish();
    REQUIRE(coordinator.offer(witness({1})) == threshold_offer_result::REJECTED_STATE);
    REQUIRE(head_value(*coordinator.tightest_boundary()) == 10);
  }
  SECTION("cancel rejects offers and finish after cancel stays safe")
  {
    coordinator.cancel();
    REQUIRE(coordinator.offer(witness({1})) == threshold_offer_result::REJECTED_STATE);
    coordinator.finish();
    REQUIRE(coordinator.offer(witness({1})) == threshold_offer_result::REJECTED_STATE);
  }
}

TEST_CASE("top_n_threshold_coordinator boundary never loosens under random offers",
          "[dynamic_filter][top_n]")
{
  auto const keys = single_asc_key();
  top_n_threshold_coordinator coordinator{3, keys, true};
  std::mt19937 rng{20260805};
  std::uniform_int_distribution<std::int32_t> dist{-1000, 1000};

  std::optional<exact_host_key_tuple> previous;
  for (int i = 0; i < 500; ++i) {
    (void)coordinator.offer(witness({dist(rng)}));
    auto current = coordinator.tightest_boundary();
    REQUIRE(current.has_value());
    if (previous) {
      // Monotone strengthening: the boundary only ever moves earlier in the sort order.
      REQUIRE(current->lex_compare(*previous, keys) <= 0);
    }
    previous = std::move(current);
  }
}

TEST_CASE("top_n_threshold_coordinator concurrent offers settle on the tightest boundary",
          "[dynamic_filter][top_n]")
{
  dynamic_filter_stats stats;
  auto const keys = single_asc_key();
  top_n_threshold_coordinator coordinator{3, keys, true, &stats};

  constexpr int k_threads           = 8;
  constexpr int k_offers_per_thread = 200;
  std::vector<std::vector<std::int32_t>> per_thread_values(k_threads);
  for (int t = 0; t < k_threads; ++t) {
    std::mt19937 rng{static_cast<unsigned>(1000 + t)};
    std::uniform_int_distribution<std::int32_t> dist{0, 100000};
    for (int i = 0; i < k_offers_per_thread; ++i) {
      per_thread_values[t].push_back(dist(rng));
    }
  }

  std::vector<std::thread> threads;
  threads.reserve(k_threads);
  for (int t = 0; t < k_threads; ++t) {
    threads.emplace_back([&coordinator, &per_thread_values, t] {
      for (auto const v : per_thread_values[t]) {
        (void)coordinator.offer(witness({v}));
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  std::int32_t global_min = per_thread_values[0][0];
  for (auto const& values : per_thread_values) {
    global_min = std::min(global_min, *std::min_element(values.begin(), values.end()));
  }
  // Every offer was compared under the mutex, so the final boundary is the tightest offered.
  REQUIRE(head_value(*coordinator.tightest_boundary()) == global_min);
  REQUIRE(stats.top_n_offers.load() == k_threads * k_offers_per_thread);
  REQUIRE(coordinator.boundary_update_count() >= 1);
  REQUIRE(coordinator.boundary_update_count() + stats.top_n_offers_not_tighter.load() ==
          static_cast<std::uint64_t>(k_threads * k_offers_per_thread));
}

TEST_CASE("top_n_threshold_coordinator finish races offers without deadlock or torn state",
          "[dynamic_filter][top_n]")
{
  auto const keys = single_asc_key();
  top_n_threshold_coordinator coordinator{3, keys, true};

  constexpr int k_threads = 4;
  std::vector<std::thread> threads;
  threads.reserve(k_threads);
  for (int t = 0; t < k_threads; ++t) {
    threads.emplace_back([&coordinator, t] {
      for (std::int32_t v = 0; v < 200; ++v) {
        (void)coordinator.offer(witness({t * 1000 + v}));
      }
    });
  }
  coordinator.finish();
  for (auto& thread : threads) {
    thread.join();
  }

  REQUIRE(coordinator.offer(witness({-1})) == threshold_offer_result::REJECTED_STATE);
  // The boundary, if any offer won the race, is one accepted value -- never a torn tuple.
  auto const boundary = coordinator.tightest_boundary();
  if (boundary) {
    auto const v = head_value(*boundary);
    REQUIRE(v >= 0);
    REQUIRE(v < 4000);
  }
}
