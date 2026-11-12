// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "allocation_fault.hpp"
#include "object_store/registry.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace sirius::test::object_store_test;
namespace fault = sirius::test::allocation_fault;

void expect_published(core::registry::outcome const& result, core::registry const& registry)
{
  CHECK_FALSE(result.refused);
  REQUIRE(result.published);
  CHECK(result.published == registry.capture());
}

struct fault_result {
  fault::observation allocation{};
  core::registry::outcome outcome;
  bool threw{false};
  std::exception_ptr unexpected;
};

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

template <class Operation>
fault_result invoke_with_fault(std::size_t ordinal, Operation&& operation)
{
  fault_result result;
  fault::arm(ordinal);
  try {
    result.outcome = operation();
  } catch (std::bad_alloc const&) {
    result.threw = true;
  } catch (...) {
    fault::disarm();
    result.unexpected = std::current_exception();
  }
  result.allocation = fault::disarm();
  return result;
}

enum class operation { insert_empty, insert, replace, drop, drop_last, absent_drop };

char const* name_of(operation op)
{
  switch (op) {
    case operation::insert_empty: return "insert-empty";
    case operation::insert: return "insert";
    case operation::replace: return "replace";
    case operation::drop: return "drop";
    case operation::drop_last: return "drop-last";
    case operation::absent_drop: return "absent-drop";
  }
  return "unknown";
}

struct allocation_fixture {
  explicit allocation_fixture(operation operation) : op(operation)
  {
    if (op != operation::insert_empty) {
      auto count = op == operation::drop_last ? 1U : 8U;
      for (unsigned i = 0; i < count; ++i) {
        auto value     = settings_for("s3://allocation-bucket/prefix-" + std::to_string(i) + "/");
        auto owner     = std::make_shared<int>(static_cast<int>(i));
        auto published = registry.publish(value, owner, {});
        REQUIRE(published.published);
        owners.push_back(std::move(owner));
      }
    }
    auto const replacing = op == operation::replace;
    candidate            = settings_for(replacing ? "s3://allocation-bucket/prefix-0/"
                                       : "s3://allocation-bucket/new-prefix/");
    candidate_owner      = std::make_shared<int>(99);
    key                  = op == operation::absent_drop ? "s3://allocation-bucket/absent-prefix/"
                                                        : "s3://allocation-bucket/prefix-0/";
    before               = registry.capture();
    for (auto const& owner : owners) {
      owner_counts.push_back(owner.use_count());
    }
    candidate_count = candidate_owner.use_count();
  }

  core::registry::outcome perform()
  {
    if (op == operation::insert_empty || op == operation::insert || op == operation::replace) {
      return registry.publish(candidate, candidate_owner, {});
    }
    return registry.drop(key);
  }

  void check_unchanged() const
  {
    auto const now = registry.capture();
    REQUIRE(now == before);
    REQUIRE(now->size() == before->size());
    for (std::size_t i = 0; i < before->size(); ++i) {
      CHECK((*now)[i] == (*before)[i]);
    }
    for (std::size_t i = 0; i < owners.size(); ++i) {
      CHECK(owners[i].use_count() == owner_counts[i]);
    }
    CHECK(candidate_owner.use_count() == candidate_count);
  }

  void check_success(core::registry::outcome const& result) const
  {
    CHECK_FALSE(result.refused);
    if (op == operation::absent_drop) {
      CHECK_FALSE(result.published);
      check_unchanged();
      return;
    }
    expect_published(result, registry);
    auto const& now = *result.published;
    CHECK(result.published != before);
    auto const publishing =
      op == operation::insert_empty || op == operation::insert || op == operation::replace;
    auto const size =
      publishing ? before->size() + (op == operation::replace ? 0U : 1U) : before->size() - 1;
    REQUIRE(now.size() == size);
    std::size_t found_candidate = 0;
    for (std::size_t i = 0; i < now.size(); ++i) {
      auto const& entry = now[i];
      REQUIRE(entry);
      REQUIRE(entry->value);
      if (i > 0) { CHECK(now[i - 1]->value->uri_prefix.size() >= entry->value->uri_prefix.size()); }
      if (publishing && entry->value->uri_prefix == candidate->uri_prefix) {
        ++found_candidate;
        CHECK(entry->value == candidate);
        CHECK(entry->owner.get() == candidate_owner.get());
        CHECK(same_owner(entry->owner, candidate_owner));
      } else {
        auto old = std::find(before->begin(), before->end(), entry);
        CHECK(old != before->end());
        if (!publishing) { CHECK(entry->value->uri_prefix != key); }
      }
      for (std::size_t j = 0; j < i; ++j) {
        CHECK(now[j]->value->uri_prefix != entry->value->uri_prefix);
      }
    }
    CHECK(found_candidate == (publishing ? 1U : 0U));
  }

  operation op;
  core::registry registry;
  std::vector<std::shared_ptr<int>> owners;
  std::vector<long> owner_counts;
  std::shared_ptr<const core::settings> candidate;
  std::shared_ptr<int> candidate_owner;
  long candidate_count{};
  std::string key;
  std::shared_ptr<const core::scope_set> before;
};

}  // namespace

TEST_CASE("Object-store registry publishes immutable versions and exact outcomes",
          "[object_store_core]")
{
  core::registry registry;
  auto const empty = registry.capture();
  REQUIRE(empty);
  CHECK(empty->empty());
  auto absent = registry.drop("");
  CHECK_FALSE(absent.refused);
  CHECK_FALSE(absent.published);
  CHECK(registry.capture() == empty);

  auto a       = settings_for("s3://bucket/a/");
  auto b       = settings_for("s3://bucket/b/");
  auto owner_a = std::make_shared<int>(1);
  auto owner_b = std::make_shared<int>(2);
  auto first   = registry.publish(a, owner_a, {});
  expect_published(first, registry);
  auto const original_a = first.published->front();
  auto second           = registry.publish(b, owner_b, {});
  expect_published(second, registry);
  REQUIRE(second.published->size() == 2);
  CHECK((*second.published)[0] == original_a);
  CHECK((*second.published)[1]->value == b);
  CHECK(first.published->size() == 1);
  CHECK(empty->empty());

  auto replacement_owner = std::make_shared<int>(3);
  auto third             = registry.publish(a, replacement_owner, {});
  expect_published(third, registry);
  REQUIRE(third.published->size() == 2);
  CHECK((*third.published)[0] == (*second.published)[1]);
  CHECK((*third.published)[1] != original_a);
  CHECK((*third.published)[1]->owner.get() == replacement_owner.get());
  CHECK(original_a->owner.get() == owner_a.get());
  CHECK(registry.lookup("S3://bucket/a/object").hit == (*third.published)[1]);

  auto const before  = registry.capture();
  auto const a_count = replacement_owner.use_count();
  auto invalid       = valid_settings("s3://bucket/a/");
  invalid.region.clear();
  auto refused = registry.publish(std::make_shared<const core::settings>(invalid), owner_a, {});
  REQUIRE(refused.refused);
  CHECK(refused.refused->field == "region");
  CHECK(refused.refused->why == core::reason::required);
  CHECK_FALSE(refused.published);
  CHECK(registry.capture() == before);
  CHECK(replacement_owner.use_count() == a_count);
  auto bad_drop = registry.drop("not-a-scope");
  REQUIRE(bad_drop.refused);
  CHECK(bad_drop.refused->field == "uri_prefix");
  CHECK(bad_drop.refused->why == core::reason::prefix_shape);
  CHECK_FALSE(bad_drop.published);
  CHECK(registry.capture() == before);
  auto missing = registry.drop("s3://bucket/missing/");
  CHECK_FALSE(missing.refused);
  CHECK_FALSE(missing.published);
  CHECK(registry.capture() == before);

  auto fallback = registry.publish(settings_for(""), owner_a, {});
  expect_published(fallback, registry);
  CHECK(fallback.published->back()->value->uri_prefix.empty());
  CHECK(registry.lookup("s3://elsewhere/object").hit == fallback.published->back());
  auto removed_default = registry.drop("");
  expect_published(removed_default, registry);
  CHECK(removed_default.published->size() == 2);
  auto removed_a = registry.drop(a->uri_prefix);
  expect_published(removed_a, registry);
  REQUIRE(removed_a.published->size() == 1);
  CHECK(removed_a.published->front() == (*second.published)[1]);
  auto last = registry.drop(b->uri_prefix);
  expect_published(last, registry);
  CHECK(last.published->empty());
  CHECK(last.published != empty);
  CHECK(before->size() == 2);
}

TEST_CASE("Object-store registry retains opaque alias owners beyond its lifetime",
          "[object_store_core]")
{
  struct publisher {
    int padding{11};
    int identity{29};
  };
  int destroyed = 0;
  auto original = std::shared_ptr<publisher>(new publisher, [&](publisher* pointer) {
    ++destroyed;
    delete pointer;
  });
  std::weak_ptr<publisher> weak = original;
  std::shared_ptr<void> alias(original, &original->identity);
  auto const identity = alias.get();
  core::registry::outcome retained_outcome;
  std::shared_ptr<const core::scope_set> captured;
  core::resolution hit;
  {
    core::registry registry;
    retained_outcome = registry.publish(settings_for("s3://bucket/"), alias, {});
    captured         = registry.capture();
    hit              = registry.lookup("s3://bucket/object");
    REQUIRE(hit.hit);
    CHECK(hit.hit->owner.get() == identity);
    CHECK(same_owner(hit.hit->owner, original));
    alias.reset();
    original.reset();
    CHECK_FALSE(weak.expired());
  }
  CHECK(destroyed == 0);
  REQUIRE(captured->size() == 1);
  CHECK(captured->front()->owner.get() == identity);
  captured.reset();
  retained_outcome = {};
  CHECK_FALSE(weak.expired());
  hit = {};
  CHECK(weak.expired());
  CHECK(destroyed == 1);
}

TEST_CASE("Object-store registry releases removed owners after retained outcomes and captures",
          "[object_store_core]")
{
  int destroyed = 0;
  core::registry registry;
  auto owner     = std::shared_ptr<int>(new int(7), [&](int* pointer) {
    ++destroyed;
    delete pointer;
  });
  auto published = registry.publish(settings_for("s3://bucket/"), owner, {});
  auto capture   = registry.capture();
  auto lookup    = registry.lookup("s3://bucket/object");
  owner.reset();
  auto removed = registry.drop("s3://bucket/");
  REQUIRE(removed.published);
  CHECK(removed.published->empty());
  CHECK(destroyed == 0);
  published = {};
  capture.reset();
  CHECK(destroyed == 0);
  lookup = {};
  CHECK(destroyed == 1);
  removed = {};
  CHECK(destroyed == 1);
}

TEST_CASE("Object-store registry allows last-owner deleters to reenter every write outcome",
          "[object_store_core]")
{
  for (auto const action : {"replace", "drop", "refuse"}) {
    INFO(action);
    operation_deadline deadline(action);
    core::registry registry;
    auto const value        = settings_for("s3://bucket/target/");
    auto const nested       = settings_for("s3://bucket/nested/");
    auto const nested_owner = std::make_shared<int>(8);
    auto observation        = std::make_shared<reentry_observation>();
    observation->active     = &registry;
    stop_reentry guard{observation};
    auto owner = std::shared_ptr<int>(
      new int(4), [observation, nested, nested_owner, action](int* pointer) noexcept {
        fault::disarm();
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
    }
    CHECK_FALSE(observation->error);
    CHECK(observation->releases == 1);
    CHECK(observation->observed);
    CHECK(observation->reentered);
  }
}

TEST_CASE("Object-store registry concurrent writers compose and readers see complete sets",
          "[object_store_core]")
{
  operation_deadline deadline("concurrent core publications");
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

TEST_CASE("Object-store registry survives each reached allocation failure", "[object_store_core]")
{
  if (!fault::supported()) {
    SKIP("ThreadSanitizer owns the allocation functions; allocation injection is disabled");
  }
  operation_deadline deadline("allocation failure sweep");
  for (auto const op : {operation::insert_empty,
                        operation::insert,
                        operation::replace,
                        operation::drop,
                        operation::drop_last,
                        operation::absent_drop}) {
    INFO(name_of(op));
    allocation_fixture baseline(op);
    auto counted = invoke_with_fault(0, [&] { return baseline.perform(); });
    REQUIRE_FALSE(counted.unexpected);
    REQUIRE_FALSE(counted.threw);
    REQUIRE_FALSE(counted.allocation.hit);
    baseline.check_success(counted.outcome);
    auto const requests    = counted.allocation.requests;
    std::size_t propagated = 0;
    std::size_t recovered  = 0;
    for (std::size_t ordinal = 1; ordinal <= requests + 1; ++ordinal) {
      INFO("allocation " << ordinal << " of " << requests);
      std::vector<std::weak_ptr<int>> weak_owners;
      {
        allocation_fixture fixture(op);
        for (auto const& owner : fixture.owners) {
          weak_owners.emplace_back(owner);
        }
        weak_owners.emplace_back(fixture.candidate_owner);
        auto result = invoke_with_fault(ordinal, [&] { return fixture.perform(); });
        REQUIRE_FALSE(result.unexpected);
        CHECK(result.allocation.hit == (ordinal <= requests));
        if (result.allocation.hit) { CHECK(result.allocation.requests == ordinal); }
        if (result.threw) {
          CHECK(result.allocation.hit);
          CHECK_FALSE(result.outcome.refused);
          CHECK_FALSE(result.outcome.published);
          fixture.check_unchanged();
          ++propagated;
        } else {
          fixture.check_success(result.outcome);
          if (result.allocation.hit) { ++recovered; }
        }
      }
      for (auto const& weak : weak_owners) {
        CHECK(weak.expired());
      }
    }
    CHECK(propagated + recovered == requests);
    if (op != operation::absent_drop) { CHECK(propagated > 0); }
    std::printf("allocation_coverage scenario=%s requests=%zu propagated=%zu recovered=%zu\n",
                name_of(op),
                requests,
                propagated,
                recovered);
  }
}

TEST_CASE("Object-store allocation faults disarm before nothrow recovery", "[object_store_core]")
{
  if (!fault::supported()) {
    SKIP("ThreadSanitizer owns the allocation functions; allocation injection is disabled");
  }
  for (unsigned form = 0; form < 4; ++form) {
    auto allocate = [form]() -> void* {
      switch (form) {
        case 0: return ::operator new(64, std::nothrow);
        case 1: return ::operator new[](64, std::nothrow);
        case 2: return ::operator new(64, std::align_val_t{64}, std::nothrow);
        default: return ::operator new[](64, std::align_val_t{64}, std::nothrow);
      }
    };
    fault::arm(1);
    auto* failed        = allocate();
    auto* recovered     = allocate();
    auto const observed = fault::disarm();
    auto release        = [form](void* pointer) {
      switch (form) {
        case 0: ::operator delete(pointer); break;
        case 1: ::operator delete[](pointer); break;
        case 2: ::operator delete(pointer, std::align_val_t{64}); break;
        default: ::operator delete[](pointer, std::align_val_t{64}); break;
      }
    };
    auto const first_was_null     = failed == nullptr;
    auto const recovery_succeeded = recovered != nullptr;
    release(failed);
    release(recovered);
    CHECK(first_was_null);
    CHECK(recovery_succeeded);
    CHECK(observed.hit);
    CHECK(observed.requests == 1);
  }
}

TEST_CASE("Object-store allocation failures release transferred owners outside registry locks",
          "[object_store_core]")
{
  if (!fault::supported()) {
    SKIP("ThreadSanitizer owns the allocation functions; allocation injection is disabled");
  }
  operation_deadline deadline("allocation failure with reentrant candidate cleanup");
  allocation_fixture baseline(operation::insert);
  auto counted = invoke_with_fault(0, [&] { return baseline.perform(); });
  REQUIRE_FALSE(counted.threw);
  REQUIRE_FALSE(counted.unexpected);
  auto const requests    = counted.allocation.requests;
  std::size_t propagated = 0;
  std::size_t recovered  = 0;
  for (std::size_t ordinal = 1; ordinal <= requests; ++ordinal) {
    INFO("reentrant failure " << ordinal << " of " << requests);
    allocation_fixture fixture(operation::insert);
    auto const nested_value = settings_for("s3://allocation-bucket/cleanup/");
    auto const nested_owner = std::make_shared<int>(8);
    auto observation        = std::make_shared<reentry_observation>();
    observation->active     = &fixture.registry;
    stop_reentry guard{observation};
    auto transferred = std::shared_ptr<int>(
      new int(99),
      [observation, before = fixture.before, nested_value, nested_owner](int* pointer) noexcept {
        fault::disarm();
        ++observation->releases;
        if (auto* active = observation->active) {
          try {
            observation->observed = active->capture() == before;
            auto captured         = active->lookup("s3://allocation-bucket/prefix-0/object");
            auto put              = active->publish(nested_value, nested_owner, {});
            auto drop             = active->drop(nested_value->uri_prefix);
            observation->reentered =
              captured.hit && put.published && drop.published && !put.refused && !drop.refused;
          } catch (...) {
            observation->error = std::current_exception();
          }
        }
        delete pointer;
      });
    auto const raw_owner = transferred.get();
    auto result          = invoke_with_fault(ordinal, [&] {
      return fixture.registry.publish(fixture.candidate, std::move(transferred), {});
    });
    REQUIRE_FALSE(result.unexpected);
    REQUIRE(result.allocation.hit);
    CHECK(result.allocation.requests == ordinal);
    CHECK_FALSE(transferred);
    if (result.threw) {
      CHECK(observation->releases == 1);
      CHECK(observation->observed);
      auto const after_cleanup = fixture.registry.capture();
      CHECK(*after_cleanup == *fixture.before);
      ++propagated;
    } else {
      REQUIRE(result.outcome.published);
      CHECK_FALSE(result.outcome.refused);
      auto hit = fixture.registry.lookup(fixture.candidate->uri_prefix + "object");
      REQUIRE(hit.hit);
      CHECK(hit.hit->owner.get() == raw_owner);
      CHECK(observation->releases == 0);
      hit            = {};
      result.outcome = {};
      auto drop      = fixture.registry.drop(fixture.candidate->uri_prefix);
      REQUIRE(drop.published);
      CHECK(observation->releases == 1);
      ++recovered;
    }
    CHECK_FALSE(observation->error);
    CHECK(observation->reentered);
    CHECK(observation->releases == 1);
  }
  CHECK(propagated > 0);
  CHECK(propagated + recovered == requests);
  std::printf(
    "allocation_coverage scenario=reentrant-insert requests=%zu propagated=%zu recovered=%zu\n",
    requests,
    propagated,
    recovered);
}
