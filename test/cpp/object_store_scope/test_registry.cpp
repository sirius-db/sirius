// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/registry.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace sirius::test::object_store_test;

struct reentry_observation {
  core::registry* active;
  unsigned releases{0};
  bool observed{false};
  bool reentered{false};
  std::exception_ptr error;
};

// Fatal assertions must not leave a callback referring to dead stack locals or re-entering
// a registry that is being destroyed during test cleanup.
struct stop_reentry {
  std::shared_ptr<reentry_observation> observation;
  ~stop_reentry() { observation->active = nullptr; }
};

}  // namespace

TEST_CASE("Object-store registry allows last-owner deleters to reenter every write outcome",
          "[object_store_scope]")
{
  for (auto const action : {"replace", "drop", "refuse"}) {
    INFO(action);
    core::registry registry;
    auto const value        = settings_for("s3://bucket/target/");
    auto const nested       = settings_for("s3://bucket/nested/");
    auto const nested_owner = std::make_shared<int>(8);
    auto observation        = std::make_shared<reentry_observation>();
    observation->active     = &registry;
    stop_reentry guard{observation};
    auto owner = std::shared_ptr<int>(
      new int(4), [observation, nested, nested_owner, action](int* pointer) noexcept {
        ++observation->releases;
        if (auto* active = observation->active) {
          try {
            auto hit = active->lookup("s3://bucket/target/object");
            observation->observed =
              std::string_view(action) == "replace" ? bool(hit.hit) : !hit.hit;
            auto inserted = active->publish(nested, nested_owner, {});
            auto removed  = active->drop(nested->uri_prefix);
            observation->reentered =
              inserted.published && removed.published && !inserted.refused && !removed.refused;
          } catch (...) {
            observation->error = std::current_exception();
          }
        }
        delete pointer;
      });
    if (std::string_view(action) == "refuse") {
      auto invalid = valid_settings(value->uri_prefix);
      invalid.endpoint.clear();
      auto result = registry.publish(
        std::make_shared<const core::settings>(std::move(invalid)), std::move(owner), {});
      REQUIRE(result.refused);
      CHECK_FALSE(result.published);
    } else {
      auto result = registry.publish(value, std::move(owner), {});
      REQUIRE(result.published);
      result = {};
      if (std::string_view(action) == "replace") {
        result = registry.publish(value, std::make_shared<int>(5), {});
      } else {
        result = registry.drop(value->uri_prefix);
      }
      REQUIRE(result.published);
      // This call's set, not the one left by the deleter's nested publish and drop.
      CHECK(result.published != registry.capture());
      CHECK(result.published->size() == (std::string_view(action) == "replace" ? 1U : 0U));
    }
    CHECK_FALSE(observation->error);
    CHECK(observation->releases == 1);
    CHECK(observation->observed);
    CHECK(observation->reentered);
  }
}

TEST_CASE("Object-store registry concurrent writers compose and readers see complete sets",
          "[object_store_scope]")
{
  constexpr std::size_t writers = 8;
  core::registry registry;
  std::array<std::shared_ptr<const core::settings>, writers> values;
  std::array<std::shared_ptr<int>, writers> owners;
  for (std::size_t i = 0; i < writers; ++i) {
    values[i] = settings_for("s3://concurrent-bucket/prefix-" + std::to_string(i) + "/");
    owners[i] = std::make_shared<int>(static_cast<int>(i));
  }
  std::barrier start(static_cast<std::ptrdiff_t>(writers + 2));
  std::atomic<std::size_t> finished{0};
  std::array<std::exception_ptr, writers> errors;
  std::array<bool, writers> writes_ok;
  writes_ok.fill(true);
  bool snapshots_ok        = true;
  std::size_t observations = 0;
  std::exception_ptr reader_error;
  std::vector<std::thread> threads;
  for (std::size_t i = 0; i < writers; ++i) {
    threads.emplace_back([&, i] {
      start.arrive_and_wait();
      try {
        for (unsigned n = 0; n < 24; ++n) {
          auto put     = registry.publish(values[i], owners[i], {});
          writes_ok[i] = writes_ok[i] && put.published && !put.refused;
          if (n % 3 == 0) {
            auto drop    = registry.drop(values[i]->uri_prefix);
            writes_ok[i] = writes_ok[i] && drop.published && !drop.refused;
          }
        }
        auto final   = registry.publish(values[i], owners[i], {});
        writes_ok[i] = writes_ok[i] && final.published && !final.refused;
      } catch (...) {
        errors[i] = std::current_exception();
      }
      finished.fetch_add(1);
    });
  }
  std::thread reader([&] {
    start.arrive_and_wait();
    try {
      do {
        auto const set = registry.capture();
        if (!set) {
          snapshots_ok = false;
          continue;
        }
        snapshots_ok = snapshots_ok && set->size() <= writers;
        std::array<bool, writers> seen{};
        for (auto const& entry : *set) {
          if (!entry || !entry->value) {
            snapshots_ok = false;
            continue;
          }
          auto const found = std::find(values.begin(), values.end(), entry->value);
          if (found == values.end()) {
            snapshots_ok = false;
            continue;
          }
          auto const i = static_cast<std::size_t>(found - values.begin());
          snapshots_ok = snapshots_ok && !seen[i] && entry->owner.get() == owners[i].get() &&
                         same_owner(entry->owner, owners[i]);
          seen[i] = true;
        }
        ++observations;
      } while (finished.load() != writers);
    } catch (...) {
      reader_error = std::current_exception();
    }
  });
  start.arrive_and_wait();
  for (auto& thread : threads) {
    thread.join();
  }
  reader.join();
  CHECK_FALSE(reader_error);
  CHECK(snapshots_ok);
  CHECK(observations > 0);
  REQUIRE(registry.capture()->size() == writers);
  for (std::size_t i = 0; i < writers; ++i) {
    CHECK_FALSE(errors[i]);
    CHECK(writes_ok[i]);
    auto const hit = registry.lookup(values[i]->uri_prefix + "object");
    REQUIRE(hit.hit);
    CHECK(hit.hit->value == values[i]);
    CHECK(hit.hit->owner.get() == owners[i].get());
  }
}
