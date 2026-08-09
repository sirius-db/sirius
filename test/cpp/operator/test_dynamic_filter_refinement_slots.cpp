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
 * @file test_dynamic_filter_refinement_slots.cpp
 * @brief Model and property tests for `sirius_dynamic_filter_set` refinement slots, coherent
 *        snapshots, and the channel generation -- host-only, no GPU state. Worker threads record
 *        violations and the main thread asserts after joining (Catch2 v2 assertions are not
 *        thread-safe).
 */

#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <catch.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using sirius::op::dynamic_filter_refinement_publisher;
using sirius::op::refinement_publish_result;
using sirius::op::sirius_dynamic_filter;
using sirius::op::sirius_dynamic_filter_kind;
using sirius::op::sirius_dynamic_filter_set;

namespace {

/// Host-only stand-in filter carrying the revision that published it, so a snapshot's generation
/// can be checked against the filter pointer it is bound to.
struct tagged_filter final : sirius_dynamic_filter {
  explicit tagged_filter(std::uint64_t tag) : tag(tag) {}
  [[nodiscard]] sirius_dynamic_filter_kind kind() const override
  {
    return sirius_dynamic_filter_kind::ZONE_MAP;
  }
  std::uint64_t tag;
};

std::shared_ptr<sirius_dynamic_filter const> make_tagged(std::uint64_t tag)
{
  return std::make_shared<tagged_filter const>(tag);
}

std::uint64_t tag_of(std::shared_ptr<sirius_dynamic_filter const> const& filter)
{
  return static_cast<tagged_filter const&>(*filter).tag;
}

}  // namespace

static_assert(!std::is_copy_constructible_v<dynamic_filter_refinement_publisher>);
static_assert(!std::is_copy_assignable_v<dynamic_filter_refinement_publisher>);
static_assert(std::is_move_constructible_v<dynamic_filter_refinement_publisher>);

TEST_CASE("refinement slot: first publish populates and counts once, replacement bumps only the "
          "generation",
          "[dynamic_filter]")
{
  auto set       = std::make_shared<sirius_dynamic_filter_set>();
  auto publisher = set->register_refinement_slot(3);
  REQUIRE(set->has_producers());
  REQUIRE_FALSE(set->has_filters());
  REQUIRE(set->filter_count() == 0);
  REQUIRE(set->generation() == 0);

  REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::ACCEPTED);
  REQUIRE(set->has_filters());
  REQUIRE(set->filter_count() == 1);
  REQUIRE(set->generation() == 1);

  auto const first = set->snapshot();
  REQUIRE(first.generation == 1);
  REQUIRE(first.logical_filter_count == 1);
  REQUIRE(first.columns.size() == 1);
  REQUIRE(first.columns[0].column == 3);
  REQUIRE(first.columns[0].filters.size() == 1);
  REQUIRE(tag_of(first.columns[0].filters[0]) == 1);

  // Replacement: generation moves, the logical filter count does not.
  REQUIRE(publisher.publish(2, make_tagged(2)) == refinement_publish_result::ACCEPTED);
  REQUIRE(set->filter_count() == 1);
  REQUIRE(set->generation() == 2);
  auto const second = set->snapshot();
  REQUIRE(second.columns[0].filters.size() == 1);
  REQUIRE(tag_of(second.columns[0].filters[0]) == 2);

  // The earlier snapshot still owns the superseded filter.
  REQUIRE(tag_of(first.columns[0].filters[0]) == 1);

  // Publishing through a moved-to handle targets the same slot.
  auto moved = std::move(publisher);
  REQUIRE(moved.publish(3, make_tagged(3)) == refinement_publish_result::ACCEPTED);
  REQUIRE(set->generation() == 3);
}

TEST_CASE("refinement slot rejections make no visible change", "[dynamic_filter]")
{
  auto set = std::make_shared<sirius_dynamic_filter_set>();

  SECTION("revision sequencing: a fresh slot holds revision 0 and equal or lower is STALE")
  {
    auto publisher = set->register_refinement_slot(0);
    REQUIRE(publisher.publish(0, make_tagged(0)) == refinement_publish_result::STALE);
    REQUIRE(publisher.publish(2, make_tagged(2)) == refinement_publish_result::ACCEPTED);
    REQUIRE(publisher.publish(2, make_tagged(9)) == refinement_publish_result::STALE);
    REQUIRE(publisher.publish(1, make_tagged(9)) == refinement_publish_result::STALE);
    REQUIRE(set->generation() == 1);
    REQUIRE(tag_of(set->snapshot().columns[0].filters[0]) == 2);
  }
  SECTION("closed channel")
  {
    auto publisher = set->register_refinement_slot(0);
    set->close_for_new_filters();
    REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::CLOSED);
    REQUIRE_FALSE(set->has_filters());
    REQUIRE(set->generation() == 0);
  }
  SECTION("ignored primary ordinal")
  {
    auto publisher = set->register_refinement_slot(4);
    set->ignore_columns({4});
    REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::IGNORED);
    REQUIRE_FALSE(set->has_filters());
  }
  SECTION("ignored referenced ordinal")
  {
    auto publisher = set->register_refinement_slot(4, {5, 6});
    set->ignore_columns({6});
    REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::IGNORED);
    REQUIRE_FALSE(set->has_filters());
  }
  SECTION("null filter")
  {
    auto publisher = set->register_refinement_slot(4);
    REQUIRE(publisher.publish(1, nullptr) == refinement_publish_result::IGNORED);
    REQUIRE_FALSE(set->has_filters());
    REQUIRE(set->generation() == 0);
  }
}

TEST_CASE("appends bump count and generation, slots merge after appends per column",
          "[dynamic_filter]")
{
  auto set = std::make_shared<sirius_dynamic_filter_set>();
  REQUIRE(set->push_filter(2, make_tagged(100)));
  REQUIRE(set->push_filter(2, make_tagged(101)));
  REQUIRE(set->filter_count() == 2);
  REQUIRE(set->generation() == 2);

  auto slot_two  = set->register_refinement_slot(2);
  auto slot_nine = set->register_refinement_slot(9);
  REQUIRE(slot_two.publish(1, make_tagged(200)) == refinement_publish_result::ACCEPTED);
  REQUIRE(set->filter_count() == 3);
  REQUIRE(set->generation() == 3);

  auto const snap = set->snapshot();
  REQUIRE(snap.logical_filter_count == 3);
  // Column 2 carries the appends first (insertion order), then the slot value; the unpopulated
  // slot on column 9 contributes nothing.
  REQUIRE(snap.columns.size() == 1);
  REQUIRE(snap.columns[0].column == 2);
  REQUIRE(snap.columns[0].filters.size() == 3);
  REQUIRE(tag_of(snap.columns[0].filters[0]) == 100);
  REQUIRE(tag_of(snap.columns[0].filters[1]) == 101);
  REQUIRE(tag_of(snap.columns[0].filters[2]) == 200);

  // An accepted append after the slot value keeps bumping both signals.
  REQUIRE(set->push_filter(9, make_tagged(102)));
  REQUIRE(set->filter_count() == 4);
  REQUIRE(set->generation() == 4);
}

TEST_CASE("snapshots survive replacement, close, and the channel's destruction",
          "[dynamic_filter]")
{
  sirius::op::dynamic_filter_snapshot snapshot;
  {
    auto set       = std::make_shared<sirius_dynamic_filter_set>();
    auto publisher = set->register_refinement_slot(0);
    REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::ACCEPTED);
    snapshot = set->snapshot();
    REQUIRE(publisher.publish(2, make_tagged(2)) == refinement_publish_result::ACCEPTED);
    set->close_for_new_filters();
    // set and publisher leave scope: the publisher's co-ownership dies with it.
  }
  REQUIRE(snapshot.generation == 1);
  REQUIRE(snapshot.columns.size() == 1);
  REQUIRE(tag_of(snapshot.columns[0].filters[0]) == 1);
}

TEST_CASE("a publisher outliving every consumer reference stays safe", "[dynamic_filter]")
{
  auto set       = std::make_shared<sirius_dynamic_filter_set>();
  auto publisher = set->register_refinement_slot(0);

  SECTION("channel released but open: the handle's co-ownership keeps publishes accepted")
  {
    set.reset();
    REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::ACCEPTED);
  }
  SECTION("closed before release: a late publish is rejected, never touching freed state")
  {
    set->close_for_new_filters();
    set.reset();
    REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::CLOSED);
  }
}

TEST_CASE("snapshot coherence holds under concurrent replacement", "[dynamic_filter]")
{
  auto set       = std::make_shared<sirius_dynamic_filter_set>();
  auto publisher = set->register_refinement_slot(0);

  // Sequentially accepted revisions 1..N make generation == revision == filter tag, so any
  // snapshot pairing a filter pointer with a different generation is a coherence violation.
  constexpr std::uint64_t k_revisions = 2000;
  std::atomic<bool> done{false};
  std::atomic<int> violations{0};

  std::vector<std::thread> readers;
  for (int t = 0; t < 3; ++t) {
    readers.emplace_back([&] {
      std::uint64_t last_seen = 0;
      while (!done.load(std::memory_order_acquire)) {
        auto const snap = set->snapshot();
        bool ok         = snap.generation >= last_seen;  // per-thread monotonic
        if (snap.generation == 0) {
          ok = ok && snap.columns.empty() && snap.logical_filter_count == 0;
        } else {
          ok = ok && snap.logical_filter_count == 1 && snap.columns.size() == 1 &&
               snap.columns[0].filters.size() == 1 &&
               tag_of(snap.columns[0].filters[0]) == snap.generation;
        }
        if (!ok) { violations.fetch_add(1, std::memory_order_relaxed); }
        last_seen = snap.generation;
      }
    });
  }
  for (std::uint64_t revision = 1; revision <= k_revisions; ++revision) {
    if (publisher.publish(revision, make_tagged(revision)) !=
        refinement_publish_result::ACCEPTED) {
      violations.fetch_add(1, std::memory_order_relaxed);
    }
  }
  done.store(true, std::memory_order_release);
  for (auto& reader : readers) {
    reader.join();
  }

  REQUIRE(violations.load() == 0);
  REQUIRE(set->generation() == k_revisions);
  REQUIRE(set->filter_count() == 1);
}

TEST_CASE("concurrent publish, snapshot, and close settle without deadlock or torn state",
          "[dynamic_filter]")
{
  auto set       = std::make_shared<sirius_dynamic_filter_set>();
  auto publisher = set->register_refinement_slot(0);
  REQUIRE(publisher.publish(1, make_tagged(1)) == refinement_publish_result::ACCEPTED);
  auto const pre_close = set->snapshot();

  std::atomic<bool> stop{false};
  std::atomic<std::uint64_t> accepted{1};  // revision 1 above
  std::atomic<int> violations{0};
  std::thread writer([&] {
    std::uint64_t revision = 2;
    while (!stop.load(std::memory_order_acquire)) {
      auto const result = publisher.publish(revision, make_tagged(revision));
      if (result == refinement_publish_result::ACCEPTED) {
        accepted.fetch_add(1, std::memory_order_release);
        ++revision;
      } else if (result != refinement_publish_result::CLOSED) {
        violations.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });
  std::thread reader([&] {
    while (!stop.load(std::memory_order_acquire)) {
      auto const snap = set->snapshot();
      if (snap.columns.size() != 1 || snap.columns[0].filters.size() != 1 ||
          tag_of(snap.columns[0].filters[0]) != snap.generation) {
        violations.fetch_add(1, std::memory_order_relaxed);
      }
    }
  });
  // Close only once publishes are demonstrably in flight, so the close races active traffic
  // rather than an idle channel.
  while (accepted.load(std::memory_order_acquire) < 100) {
    std::this_thread::yield();
  }
  set->close_for_new_filters();
  stop.store(true, std::memory_order_release);
  writer.join();
  reader.join();

  REQUIRE(violations.load() == 0);
  REQUIRE(publisher.publish(1000000, make_tagged(0)) == refinement_publish_result::CLOSED);
  // The pre-close snapshot is untouched by everything that followed it.
  REQUIRE(pre_close.generation == 1);
  REQUIRE(tag_of(pre_close.columns[0].filters[0]) == 1);
}
