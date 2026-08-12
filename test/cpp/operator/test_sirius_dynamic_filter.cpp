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

#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <catch.hpp>

#include <algorithm>
#include <memory>
#include <thread>
#include <vector>

using sirius::op::sirius_ast_lowerable;
using sirius::op::sirius_dynamic_filter;
using sirius::op::sirius_dynamic_filter_kind;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_dynamic_zone_map_filter;
using sirius::op::zone_map_entry;

namespace {

std::unique_ptr<cudf::scalar> make_int32_scalar(int32_t v)
{
  return std::make_unique<cudf::numeric_scalar<int32_t>>(
    v, true, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

zone_map_entry make_zone(int32_t lo, int32_t hi)
{
  return zone_map_entry{make_int32_scalar(lo), make_int32_scalar(hi)};
}

std::unique_ptr<sirius_dynamic_zone_map_filter> make_single_zone_filter(int32_t lo, int32_t hi)
{
  std::vector<zone_map_entry> zones;
  zones.push_back(make_zone(lo, hi));
  return std::make_unique<sirius_dynamic_zone_map_filter>(std::move(zones));
}

}  // namespace

TEST_CASE("sirius_dynamic_filter_set is empty after construction", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  REQUIRE(set.empty());
  REQUIRE(set.filtered_columns().empty());
  REQUIRE(set.filters_for_column(0).empty());
}

TEST_CASE("sirius_dynamic_filter_set::push_filter ignores null filters", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  set.push_filter(0, nullptr);
  REQUIRE(set.empty());
}

TEST_CASE("sirius_dynamic_filter_set retains and exposes pushed filters", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  set.push_filter(3, make_single_zone_filter(0, 100));
  set.push_filter(3, make_single_zone_filter(50, 150));
  set.push_filter(7, make_single_zone_filter(-1, 1));

  REQUIRE_FALSE(set.empty());

  auto cols = set.filtered_columns();
  std::sort(cols.begin(), cols.end());
  REQUIRE(cols == std::vector<std::size_t>{3, 7});

  REQUIRE(set.filters_for_column(3).size() == 2);
  REQUIRE(set.filters_for_column(7).size() == 1);
  REQUIRE(set.filters_for_column(99).empty());

  REQUIRE(set.filters_for_column(3)[0]->kind() == sirius_dynamic_filter_kind::ZONE_MAP);
}

TEST_CASE("filters_for_column snapshot survives the set's destruction", "[dynamic_filter]")
{
  std::vector<std::shared_ptr<sirius_dynamic_filter const>> snapshot;
  {
    sirius_dynamic_filter_set set;
    set.push_filter(0, make_single_zone_filter(0, 100));
    snapshot = set.filters_for_column(0);
  }
  REQUIRE(snapshot.size() == 1);
  REQUIRE(snapshot[0]->kind() == sirius_dynamic_filter_kind::ZONE_MAP);
}

TEST_CASE("the same filter can be co-owned by multiple channels (fan-out)", "[dynamic_filter]")
{
  std::shared_ptr<sirius_dynamic_filter const> shared_filter = make_single_zone_filter(0, 100);

  sirius_dynamic_filter_set set_a;
  sirius_dynamic_filter_set set_b;
  set_a.push_filter(0, shared_filter);
  set_b.push_filter(5, shared_filter);

  auto const a_snapshot = set_a.filters_for_column(0);
  auto const b_snapshot = set_b.filters_for_column(5);

  REQUIRE(a_snapshot.size() == 1);
  REQUIRE(b_snapshot.size() == 1);
  REQUIRE(a_snapshot[0].get() == b_snapshot[0].get());
}

TEST_CASE("sirius_dynamic_filter_set::push_filter is thread-safe", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;

  constexpr int kThreads             = 8;
  constexpr int kPushesPerThread     = 32;
  constexpr std::size_t kColumnCount = 4;

  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&set, t]() {
      for (int i = 0; i < kPushesPerThread; ++i) {
        auto col = static_cast<std::size_t>((t + i) % kColumnCount);
        set.push_filter(col, make_single_zone_filter(t * 1000 + i, t * 1000 + i + 1));
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  std::size_t total = 0;
  for (std::size_t col = 0; col < kColumnCount; ++col) {
    total += set.filters_for_column(col).size();
  }
  REQUIRE(total == static_cast<std::size_t>(kThreads) * kPushesPerThread);
}

//===----------------------------------------------------------------------===//
// Filter availability
//===----------------------------------------------------------------------===//

TEST_CASE("has_filters reflects pushed filters and ignores null", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  REQUIRE_FALSE(set.has_filters());
  set.push_filter(0, nullptr);
  REQUIRE_FALSE(set.has_filters());
  set.push_filter(0, make_single_zone_filter(0, 100));
  REQUIRE(set.has_filters());
}

TEST_CASE("sirius_dynamic_filter_set rejects pushes after consumer close", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  REQUIRE(set.accepting_filters());

  set.close_for_new_filters();

  REQUIRE_FALSE(set.accepting_filters());
  REQUIRE_FALSE(set.push_filter(0, make_single_zone_filter(0, 100)));
  REQUIRE(set.empty());
  REQUIRE_FALSE(set.has_filters());
}

TEST_CASE("sirius_dynamic_filter_set tracks wired producers", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  REQUIRE_FALSE(set.has_producers());

  set.register_producer();

  REQUIRE(set.has_producers());
}

TEST_CASE("ignore_columns drops pushes for the marked output columns", "[dynamic_filter]")
{
  // Hive-partition values are path-derived, so their output positions reject filter pushes.
  sirius_dynamic_filter_set set;
  set.ignore_columns({0});

  REQUIRE_FALSE(set.push_filter(0, make_single_zone_filter(0, 100)));  // ignored
  REQUIRE(set.push_filter(1, make_single_zone_filter(0, 100)));        // kept
  REQUIRE(set.filtered_columns() == std::vector<std::size_t>{1});
}

TEST_CASE("sirius_dynamic_zone_map_filter rejects empty zones", "[dynamic_filter]")
{
  REQUIRE_THROWS_AS(sirius_dynamic_zone_map_filter(std::vector<zone_map_entry>{}),
                    std::invalid_argument);
}

TEST_CASE("sirius_dynamic_zone_map_filter rejects null bounds", "[dynamic_filter]")
{
  std::vector<zone_map_entry> zones;
  zones.push_back(zone_map_entry{nullptr, make_int32_scalar(10)});
  REQUIRE_THROWS_AS(sirius_dynamic_zone_map_filter(std::move(zones)), std::invalid_argument);

  std::vector<zone_map_entry> zones2;
  zones2.push_back(zone_map_entry{make_int32_scalar(0), nullptr});
  REQUIRE_THROWS_AS(sirius_dynamic_zone_map_filter(std::move(zones2)), std::invalid_argument);
}

TEST_CASE("sirius_dynamic_zone_map_filter::to_ast emits a single bounded conjunction for N=1",
          "[dynamic_filter]")
{
  auto filter = make_single_zone_filter(0, 100);

  cudf::ast::tree tree;
  auto const& col_ref = tree.emplace<cudf::ast::column_reference>(0);
  auto const& root    = filter->to_ast(tree, col_ref);

  // For N=1 we expect: 2 literals + 2 comparisons + 1 AND = 5 new nodes,
  // plus the column_reference already pushed = 6 total.
  REQUIRE(tree.size() == 6);
  // Root must be the AND node, which is the last node emplaced.
  REQUIRE(&root == &tree.back());
}

TEST_CASE("sirius_dynamic_zone_map_filter::to_ast OR-conjoins multiple zones", "[dynamic_filter]")
{
  std::vector<zone_map_entry> zones;
  zones.push_back(make_zone(0, 10));
  zones.push_back(make_zone(20, 30));
  zones.push_back(make_zone(40, 50));
  auto filter = std::make_unique<sirius_dynamic_zone_map_filter>(std::move(zones));

  cudf::ast::tree tree;
  auto const& col_ref = tree.emplace<cudf::ast::column_reference>(0);
  auto const& root    = filter->to_ast(tree, col_ref);

  // Each zone contributes 2 literals + 2 ops + 1 AND = 5 nodes.
  // OR-conjoining N zones contributes (N - 1) OR ops on top.
  // Plus the column_reference already pushed.
  // Expected: 1 + N * 5 + (N - 1) = 1 + 15 + 2 = 18.
  REQUIRE(tree.size() == 18);
  REQUIRE(&root == &tree.back());
  REQUIRE(filter->num_zones() == 3);
}

TEST_CASE("sirius_ast_lowerable::to_standalone_ast wraps to_ast in a fresh tree",
          "[dynamic_filter]")
{
  auto filter = make_single_zone_filter(0, 100);

  auto tree = filter->to_standalone_ast([](cudf::ast::tree& t) -> cudf::ast::expression const& {
    return t.emplace<cudf::ast::column_reference>(0);
  });

  // Same node count as the in-place to_ast for N=1: 1 col_ref + 2 lit + 2 op + 1 AND = 6.
  REQUIRE(tree.size() == 6);
}
