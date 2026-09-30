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
#include <atomic>
#include <barrier>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using sirius::op::column_ref_resolver_fn;
using sirius::op::dynamic_filter_snapshot;
using sirius::op::merge_ast_dynamic_filters_into_tree;
using sirius::op::sirius_ast_lowerable;
using sirius::op::sirius_dynamic_filter;
using sirius::op::sirius_dynamic_filter_kind;
using sirius::op::sirius_dynamic_filter_set;
using sirius::op::sirius_dynamic_zone_map_filter;
using sirius::op::zone_map_entry;

namespace {

std::vector<std::shared_ptr<sirius_dynamic_filter const>> filters_on_column(
  dynamic_filter_snapshot const& snapshot, std::size_t col_idx)
{
  std::vector<std::shared_ptr<sirius_dynamic_filter const>> out;
  for (auto const& entry : snapshot.entries()) {
    if (entry.column_index == col_idx) { out.push_back(entry.filter); }
  }
  return out;
}

std::vector<std::size_t> filtered_columns(dynamic_filter_snapshot const& snapshot)
{
  std::vector<std::size_t> cols;
  for (auto const& entry : snapshot.entries()) {
    if (std::find(cols.begin(), cols.end(), entry.column_index) == cols.end()) {
      cols.push_back(entry.column_index);
    }
  }
  return cols;
}

std::unique_ptr<cudf::scalar> make_int32_scalar(int32_t v)
{
  return std::make_unique<cudf::numeric_scalar<int32_t>>(
    v, true, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

std::unique_ptr<cudf::scalar> make_int64_scalar(int64_t v)
{
  return std::make_unique<cudf::numeric_scalar<int64_t>>(
    v, true, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

std::unique_ptr<cudf::scalar> make_float64_scalar(double v)
{
  return std::make_unique<cudf::numeric_scalar<double>>(
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

// Stand-in for a runtime-only filter (e.g., bloom): inherits the base but NOT the AST mixin.
// Lets us verify that the consumer-side merge skips filters lacking the AST capability.
class stub_runtime_only_filter final : public sirius_dynamic_filter {
 public:
  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::ZONE_MAP;
  }
};

static_assert(!std::is_copy_constructible_v<sirius_dynamic_filter_set::producer>);
static_assert(std::is_nothrow_move_constructible_v<sirius_dynamic_filter_set::producer>);
static_assert(std::is_nothrow_move_assignable_v<sirius_dynamic_filter_set::producer>);

}  // namespace

TEST_CASE("sirius_dynamic_filter_set is empty after construction", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto const snapshot = set.snapshot();
  REQUIRE_FALSE(set.has_filters());
  REQUIRE(snapshot.empty());
  REQUIRE(filtered_columns(snapshot).empty());
  REQUIRE(filters_on_column(snapshot, 0).empty());
}

TEST_CASE("dynamic filter registration freezes before terminal observations",
          "[dynamic_filter][channel_lifecycle]")
{
  sirius_dynamic_filter_set set;
  auto const before = set.snapshot();
  REQUIRE(before.empty());
  REQUIRE_FALSE(before.terminal());
  set.freeze_registration();
  set.freeze_registration();
  auto const after = set.snapshot();
  REQUIRE(after.empty());
  REQUIRE(after.terminal());
  REQUIRE_FALSE(before.terminal());
  REQUIRE_THROWS_AS(set.register_producer({0}), std::logic_error);
  REQUIRE_FALSE(set.has_producers());
}

TEST_CASE("dynamic filter snapshots distinguish pending and terminal with and without filters",
          "[dynamic_filter][channel_lifecycle]")
{
  sirius_dynamic_filter_set set;
  auto producer = set.register_producer({0});
  set.freeze_registration();
  auto const pending_empty = set.snapshot();
  REQUIRE(pending_empty.empty());
  REQUIRE_FALSE(pending_empty.terminal());

  SECTION("skipped input terminates without a filter")
  {
    producer.finish(sirius_dynamic_filter_set::completion::SKIPPED);
    REQUIRE(set.snapshot().empty());
    REQUIRE(set.snapshot().terminal());
  }

  SECTION("published input retains an immutable completed prefix")
  {
    auto filter = std::make_shared<stub_runtime_only_filter>();
    REQUIRE(producer.push_filter(0, filter));
    auto const pending_filter = set.snapshot();
    REQUIRE_FALSE(pending_filter.empty());
    REQUIRE_FALSE(pending_filter.terminal());
    REQUIRE(pending_filter.entries().front().filter == filter);
    producer.finish(sirius_dynamic_filter_set::completion::PUBLISHED);
    auto const terminal_filter = set.snapshot();
    REQUIRE(terminal_filter.terminal());
    REQUIRE(terminal_filter.generation() == 1);
    REQUIRE_FALSE(pending_filter.terminal());
    REQUIRE_FALSE(producer.push_filter(0, filter));
  }
}

TEST_CASE("each dynamic filter producer completes exactly once for every outcome",
          "[dynamic_filter][channel_lifecycle]")
{
  using completion = sirius_dynamic_filter_set::completion;
  for (auto outcome :
       {completion::PUBLISHED, completion::SKIPPED, completion::FAILED, completion::CANCELLED}) {
    sirius_dynamic_filter_set set;
    auto remaining = set.register_producer({1});
    {
      auto first = set.register_producer({0});
      set.freeze_registration();
      first.finish(outcome);
      first.finish(completion::CANCELLED);
      REQUIRE_FALSE(first.push_filter(0, std::make_shared<stub_runtime_only_filter>()));
      REQUIRE_FALSE(set.snapshot().terminal());
    }
    REQUIRE_FALSE(set.snapshot().terminal());
    remaining.finish(outcome);
    REQUIRE(set.snapshot().terminal());
  }
}

TEST_CASE("dropping a producer during construction unwinding resolves its registration",
          "[dynamic_filter][channel_lifecycle]")
{
  sirius_dynamic_filter_set set;
  REQUIRE_THROWS_AS(
    [&] {
      auto producer = set.register_producer({0});
      set.freeze_registration();
      throw std::runtime_error("injected publisher construction failure");
    }(),
    std::runtime_error);
  REQUIRE(set.snapshot().terminal());
  REQUIRE(set.snapshot().empty());
}

TEST_CASE("moving producer rights transfers completion and resolves overwritten rights",
          "[dynamic_filter][channel_lifecycle]")
{
  sirius_dynamic_filter_set first;
  sirius_dynamic_filter_set second;
  auto original    = first.register_producer({0});
  auto overwritten = second.register_producer({0});
  first.freeze_registration();
  second.freeze_registration();

  auto moved = std::move(original);
  original.finish();
  REQUIRE_FALSE(original.push_filter(0, std::make_shared<stub_runtime_only_filter>()));
  REQUIRE_FALSE(first.snapshot().terminal());

  overwritten = std::move(moved);
  REQUIRE(second.snapshot().terminal());
  moved.finish();
  REQUIRE_FALSE(first.snapshot().terminal());
  REQUIRE(overwritten.push_filter(0, std::make_shared<stub_runtime_only_filter>()));
  overwritten.finish(sirius_dynamic_filter_set::completion::PUBLISHED);
  REQUIRE(first.snapshot().terminal());
  REQUIRE(first.snapshot().generation() == 1);
}

TEST_CASE("consumer closure rejects pushes without pretending producer work has retired",
          "[dynamic_filter][channel_lifecycle]")
{
  sirius_dynamic_filter_set set;
  auto producer = set.register_producer({0});
  set.freeze_registration();
  REQUIRE(producer.push_filter(0, std::make_shared<stub_runtime_only_filter>()));
  set.close_for_new_filters();
  REQUIRE_FALSE(producer.push_filter(0, std::make_shared<stub_runtime_only_filter>()));
  auto const closed = set.snapshot();
  REQUIRE_FALSE(closed.terminal());
  REQUIRE(closed.generation() == 1);
  producer.finish(sirius_dynamic_filter_set::completion::CANCELLED);
  REQUIRE(set.snapshot().terminal());
  REQUIRE(set.snapshot().generation() == closed.generation());
}

TEST_CASE("producer finish and push expose one coherent terminal snapshot",
          "[dynamic_filter][channel_lifecycle][concurrent]")
{
  auto filter = std::make_shared<stub_runtime_only_filter>();
  for (int repetition = 0; repetition < 64; ++repetition) {
    sirius_dynamic_filter_set set;
    auto producer = set.register_producer({0});
    set.freeze_registration();
    std::barrier start{3};
    bool accepted = false;
    sirius::op::dynamic_filter_snapshot terminal;
    std::jthread push([&] {
      start.arrive_and_wait();
      accepted = producer.push_filter(0, filter);
    });
    std::jthread finish([&] {
      start.arrive_and_wait();
      producer.finish(sirius_dynamic_filter_set::completion::PUBLISHED);
      terminal = set.snapshot();
    });
    start.arrive_and_wait();
    push.join();
    finish.join();

    auto const final = set.snapshot();
    REQUIRE(terminal.terminal());
    REQUIRE(final.terminal());
    REQUIRE(terminal.generation() == (accepted ? 1 : 0));
    REQUIRE(final.generation() == terminal.generation());
    REQUIRE_FALSE(producer.push_filter(0, filter));
  }
}

TEST_CASE("snapshot and producer owners outlive the channel wrapper independently",
          "[dynamic_filter][channel_lifecycle]")
{
  std::optional<sirius_dynamic_filter_set::producer> producer;
  sirius::op::dynamic_filter_snapshot captured;
  std::weak_ptr<sirius_dynamic_filter const> filter_lifetime;
  {
    sirius_dynamic_filter_set set;
    producer.emplace(set.register_producer({2}));
    set.freeze_registration();
    auto filter     = std::make_shared<stub_runtime_only_filter>();
    filter_lifetime = filter;
    REQUIRE(producer->push_filter(2, std::move(filter)));
    captured = set.snapshot();
    REQUIRE(producer->push_filter(2, std::make_shared<stub_runtime_only_filter>()));
    REQUIRE(captured.generation() == 1);
    REQUIRE(set.snapshot().generation() == 2);
  }
  REQUIRE(producer->push_filter(2, std::make_shared<stub_runtime_only_filter>()));
  producer->finish(sirius_dynamic_filter_set::completion::PUBLISHED);
  producer.reset();
  REQUIRE_FALSE(filter_lifetime.expired());
  REQUIRE(captured.entries().front().column_index == 2);
  REQUIRE(captured.entries().front().filter->kind() == sirius_dynamic_filter_kind::ZONE_MAP);
  captured = {};
  REQUIRE(filter_lifetime.expired());
}

TEST_CASE("sirius_dynamic_filter_set::push_filter ignores null filters", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0});
  REQUIRE_FALSE(set_producer.push_filter(0, nullptr));
  REQUIRE_FALSE(set.has_filters());
}

TEST_CASE("sirius_dynamic_filter_set retains and exposes pushed filters", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({3, 7});
  REQUIRE(set_producer.push_filter(3, make_single_zone_filter(0, 100)));
  REQUIRE(set_producer.push_filter(3, make_single_zone_filter(50, 150)));
  REQUIRE(set_producer.push_filter(7, make_single_zone_filter(-1, 1)));

  auto const snapshot = set.snapshot();
  REQUIRE_FALSE(snapshot.empty());

  auto cols = filtered_columns(snapshot);
  std::sort(cols.begin(), cols.end());
  REQUIRE(cols == std::vector<std::size_t>{3, 7});

  REQUIRE(filters_on_column(snapshot, 3).size() == 2);
  REQUIRE(filters_on_column(snapshot, 7).size() == 1);
  REQUIRE(filters_on_column(snapshot, 99).empty());

  REQUIRE(filters_on_column(snapshot, 3)[0]->kind() == sirius_dynamic_filter_kind::ZONE_MAP);
}

TEST_CASE("dynamic filter snapshot survives the set's destruction", "[dynamic_filter]")
{
  dynamic_filter_snapshot snapshot;
  {
    sirius_dynamic_filter_set set;
    auto set_producer = set.register_producer({0});
    REQUIRE(set_producer.push_filter(0, make_single_zone_filter(0, 100)));
    snapshot = set.snapshot();
  }
  REQUIRE(snapshot.entries().size() == 1);
  REQUIRE(snapshot.entries()[0].filter->kind() == sirius_dynamic_filter_kind::ZONE_MAP);
}

TEST_CASE("the same filter can be co-owned by multiple channels (fan-out)", "[dynamic_filter]")
{
  std::shared_ptr<sirius_dynamic_filter const> shared_filter = make_single_zone_filter(0, 100);

  sirius_dynamic_filter_set set_a;
  auto set_a_producer = set_a.register_producer({0});
  sirius_dynamic_filter_set set_b;
  auto set_b_producer = set_b.register_producer({5});
  REQUIRE(set_a_producer.push_filter(0, shared_filter));
  REQUIRE(set_b_producer.push_filter(5, shared_filter));

  auto const a_snapshot = set_a.snapshot();
  auto const b_snapshot = set_b.snapshot();

  REQUIRE(filters_on_column(a_snapshot, 0).size() == 1);
  REQUIRE(filters_on_column(b_snapshot, 5).size() == 1);
  REQUIRE(filters_on_column(a_snapshot, 0)[0].get() == filters_on_column(b_snapshot, 5)[0].get());
}

TEST_CASE("sirius_dynamic_filter_set::push_filter is thread-safe", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto producer = set.register_producer({0, 1, 2, 3});

  constexpr int kThreads             = 8;
  constexpr int kPushesPerThread     = 32;
  constexpr std::size_t kColumnCount = 4;

  std::atomic<int> rejected{0};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&producer, &rejected, t]() {
      for (int i = 0; i < kPushesPerThread; ++i) {
        auto col = static_cast<std::size_t>((t + i) % kColumnCount);
        if (!producer.push_filter(col, make_single_zone_filter(t * 1000 + i, t * 1000 + i + 1))) {
          rejected.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  REQUIRE(rejected.load(std::memory_order_relaxed) == 0);

  REQUIRE(set.snapshot().generation() == static_cast<std::size_t>(kThreads) * kPushesPerThread);
}

//===----------------------------------------------------------------------===//
// Filter availability
//===----------------------------------------------------------------------===//

TEST_CASE("has_filters reflects pushed filters and ignores null", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0});
  REQUIRE_FALSE(set.has_filters());
  REQUIRE_FALSE(set_producer.push_filter(0, nullptr));
  REQUIRE_FALSE(set.has_filters());
  REQUIRE(set_producer.push_filter(0, make_single_zone_filter(0, 100)));
  REQUIRE(set.has_filters());
}

TEST_CASE("sirius_dynamic_filter_set rejects pushes after consumer close", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0});
  REQUIRE(set.accepting_filters());

  set.close_for_new_filters();

  REQUIRE_FALSE(set.accepting_filters());
  REQUIRE_FALSE(set_producer.push_filter(0, make_single_zone_filter(0, 100)));
  REQUIRE_FALSE(set.has_filters());
}

TEST_CASE("sirius_dynamic_filter_set tracks wired producers", "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  REQUIRE_FALSE(set.has_producers());
  REQUIRE_FALSE(set.has_unscoped_producer());

  SECTION("a scoped producer declares its planned target columns")
  {
    auto producer = set.register_producer({2, 0});

    REQUIRE(set.has_producers());
    REQUIRE_FALSE(set.has_unscoped_producer());
    REQUIRE(set.planned_target_columns() == std::vector<std::size_t>{0, 2});
  }

  SECTION("multiple scoped producers union their targets")
  {
    auto first  = set.register_producer({1});
    auto second = set.register_producer({3, 1});

    REQUIRE(set.has_producers());
    REQUIRE_FALSE(set.has_unscoped_producer());
    REQUIRE(set.planned_target_columns() == std::vector<std::size_t>{1, 3});
  }

  SECTION("an empty target list registers an unscoped producer")
  {
    auto producer = set.register_producer({});

    REQUIRE(set.has_producers());
    REQUIRE(set.has_unscoped_producer());
  }
}

TEST_CASE("ignore_columns drops pushes for the marked output columns", "[dynamic_filter]")
{
  // Hive-partition values are path-derived, so their output positions reject filter pushes.
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0, 1});
  set.ignore_columns({0});

  REQUIRE_FALSE(set_producer.push_filter(0, make_single_zone_filter(0, 100)));
  REQUIRE(set_producer.push_filter(1, make_single_zone_filter(0, 100)));
  REQUIRE(filtered_columns(set.snapshot()) == std::vector<std::size_t>{1});
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

TEST_CASE("sirius_dynamic_zone_map_filter::supports allowlists lowerable non-float types",
          "[dynamic_filter]")
{
  REQUIRE_FALSE(sirius_dynamic_zone_map_filter::supports(cudf::data_type{cudf::type_id::FLOAT32}));
  REQUIRE_FALSE(sirius_dynamic_zone_map_filter::supports(cudf::data_type{cudf::type_id::FLOAT64}));

  for (auto const id : {cudf::type_id::INT8,
                        cudf::type_id::INT16,
                        cudf::type_id::INT32,
                        cudf::type_id::INT64,
                        cudf::type_id::UINT8,
                        cudf::type_id::UINT16,
                        cudf::type_id::UINT32,
                        cudf::type_id::UINT64,
                        cudf::type_id::BOOL8,
                        cudf::type_id::TIMESTAMP_DAYS,
                        cudf::type_id::TIMESTAMP_SECONDS,
                        cudf::type_id::TIMESTAMP_MILLISECONDS,
                        cudf::type_id::TIMESTAMP_MICROSECONDS,
                        cudf::type_id::TIMESTAMP_NANOSECONDS,
                        cudf::type_id::DECIMAL32,
                        cudf::type_id::DECIMAL64,
                        cudf::type_id::DECIMAL128,
                        cudf::type_id::STRING}) {
    REQUIRE(sirius_dynamic_zone_map_filter::supports(cudf::data_type{id}));
  }

  for (auto const id : {cudf::type_id::EMPTY,
                        cudf::type_id::DURATION_SECONDS,
                        cudf::type_id::LIST,
                        cudf::type_id::STRUCT,
                        cudf::type_id::DICTIONARY32}) {
    REQUIRE_FALSE(sirius_dynamic_zone_map_filter::supports(cudf::data_type{id}));
  }
}

TEST_CASE("sirius_dynamic_zone_map_filter rejects unsupported and mismatched bound types",
          "[dynamic_filter]")
{
  std::vector<zone_map_entry> float64_zones;
  float64_zones.push_back(zone_map_entry{make_float64_scalar(0.0), make_float64_scalar(10.0)});
  REQUIRE_THROWS_AS(sirius_dynamic_zone_map_filter(std::move(float64_zones)),
                    std::invalid_argument);

  std::vector<zone_map_entry> mismatched_zones;
  mismatched_zones.push_back(zone_map_entry{make_int32_scalar(0), make_int64_scalar(10)});
  REQUIRE_THROWS_AS(sirius_dynamic_zone_map_filter(std::move(mismatched_zones)),
                    std::invalid_argument);

  std::vector<zone_map_entry> cross_zone_mismatch;
  cross_zone_mismatch.push_back(zone_map_entry{make_int32_scalar(0), make_int32_scalar(10)});
  cross_zone_mismatch.push_back(zone_map_entry{make_int64_scalar(20), make_int64_scalar(30)});
  REQUIRE_THROWS_AS(sirius_dynamic_zone_map_filter(std::move(cross_zone_mismatch)),
                    std::invalid_argument);
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

TEST_CASE("merge_ast_dynamic_filters_into_tree returns existing_root unchanged for empty set",
          "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto const snapshot = set.snapshot();
  cudf::ast::tree tree;
  auto const& base = tree.emplace<cudf::ast::column_reference>(99);

  auto const& root = merge_ast_dynamic_filters_into_tree(
    tree, base, snapshot, [&tree](std::size_t) -> cudf::ast::expression const& {
      return tree.emplace<cudf::ast::column_reference>(0);
    });

  REQUIRE(&root == &base);
  REQUIRE(tree.size() == 1);
}

TEST_CASE(
  "merge_ast_dynamic_filters_into_tree AND-conjoins per-column fragments with existing root",
  "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0, 1});
  REQUIRE(set_producer.push_filter(0, make_single_zone_filter(0, 100)));
  REQUIRE(set_producer.push_filter(1, make_single_zone_filter(-5, 5)));

  auto const snapshot = set.snapshot();
  cudf::ast::tree tree;
  auto const& base = tree.emplace<cudf::ast::column_reference>(99);

  std::vector<std::size_t> resolved_cols;
  auto const& root = merge_ast_dynamic_filters_into_tree(
    tree,
    base,
    snapshot,
    [&tree, &resolved_cols](std::size_t col_idx) -> cudf::ast::expression const& {
      resolved_cols.push_back(col_idx);
      return tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(col_idx));
    });

  REQUIRE(&tree.back() == &root);

  // Resolver should have been called once per column with AST-capable filters.
  std::sort(resolved_cols.begin(), resolved_cols.end());
  REQUIRE(resolved_cols == std::vector<std::size_t>{0, 1});

  // 1 base col_ref + 2 cols × (1 col_ref + 2 lit + 2 op + 1 AND) + 1 cross-col AND
  //   + 1 final AND with base = 1 + 12 + 1 + 1 = 15.
  REQUIRE(tree.size() == 15);
}

TEST_CASE("merge_ast_dynamic_filters_into_tree AND-conjoins multiple filters per column",
          "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0});
  REQUIRE(set_producer.push_filter(0, make_single_zone_filter(0, 100)));
  REQUIRE(set_producer.push_filter(0, make_single_zone_filter(10, 200)));

  auto const snapshot = set.snapshot();
  cudf::ast::tree tree;
  auto const& base = tree.emplace<cudf::ast::column_reference>(99);

  auto const& root = merge_ast_dynamic_filters_into_tree(
    tree, base, snapshot, [&tree](std::size_t col_idx) -> cudf::ast::expression const& {
      return tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(col_idx));
    });

  REQUIRE(&tree.back() == &root);
  // 1 base + (1 col_ref + 2*(2 lit + 2 op + 1 AND) + 1 AND combining filters)
  //   + 1 final AND with base = 1 + 12 + 1 = 14.
  REQUIRE(tree.size() == 14);
}

TEST_CASE("merge_ast_dynamic_filters_into_tree skips filters lacking the AST capability",
          "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0});
  REQUIRE(set_producer.push_filter(0, std::make_unique<stub_runtime_only_filter>()));

  auto const snapshot = set.snapshot();
  cudf::ast::tree tree;
  auto const& base = tree.emplace<cudf::ast::column_reference>(99);

  std::size_t resolver_calls = 0;
  auto const& root           = merge_ast_dynamic_filters_into_tree(
    tree,
    base,
    snapshot,
    [&tree, &resolver_calls](std::size_t col_idx) -> cudf::ast::expression const& {
      ++resolver_calls;
      return tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(col_idx));
    });

  REQUIRE(&root == &base);
  REQUIRE(resolver_calls == 0);  // resolver is lazy; no AST-capable filter, no resolve
  REQUIRE(tree.size() == 1);     // only the base remains
}

TEST_CASE(
  "merge_ast_dynamic_filters_into_tree mixes AST-capable and non-capable filters per column",
  "[dynamic_filter]")
{
  sirius_dynamic_filter_set set;
  auto set_producer = set.register_producer({0});
  REQUIRE(set_producer.push_filter(0, std::make_unique<stub_runtime_only_filter>()));
  REQUIRE(set_producer.push_filter(0, make_single_zone_filter(0, 100)));

  auto const snapshot = set.snapshot();
  cudf::ast::tree tree;
  auto const& base = tree.emplace<cudf::ast::column_reference>(99);

  auto const& root = merge_ast_dynamic_filters_into_tree(
    tree, base, snapshot, [&tree](std::size_t col_idx) -> cudf::ast::expression const& {
      return tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(col_idx));
    });

  REQUIRE(&tree.back() == &root);
  // 1 base + 1 col_ref + 2 lit + 2 op + 1 AND (zone_map) + 1 final AND with base = 8.
  REQUIRE(tree.size() == 8);
}
