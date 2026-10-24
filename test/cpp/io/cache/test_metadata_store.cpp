/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "io/cache/metadata_store.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <thread>
#include <vector>

using namespace sirius::io;
namespace {
struct value : io_object_metadata {
  explicit value(size_t bytes) : bytes(bytes) {}
  size_t bytes;
};

class fake_object final : public io_object {
 public:
  fake_object(std::string path, std::string key, object_identity identity)
    : path_(std::move(path)), key_(std::move(key)), identity_(std::move(identity))
  {
  }
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return key_; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return path_; }
  [[nodiscard]] size_t size() const noexcept override { return 1; }
  [[nodiscard]] object_identity identity() const override { return identity_; }

 private:
  std::string path_;
  std::string key_;
  object_identity identity_;
};

object_identity tag(std::string version = "v1")
{
  return {identity_kind::object_tag, 32, std::move(version)};
}

bool wait_for_joined(cache::metadata_store const& store, uint64_t count)
{
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (store.statistics().joined < count && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  return store.statistics().joined >= count;
}
}  // namespace

TEST_CASE("Metadata concurrent misses share success or failure and permit retry",
          "[metadata_store]")
{
  for (bool fail : {false, true}) {
    cache::metadata_store store;
    fake_object object("file", "file:v1", tag());
    std::promise<void> started, release;
    auto gate     = release.get_future().share();
    auto payload  = std::make_shared<value>(32);
    auto producer = std::async(std::launch::async, [&] {
      return store.get_or_load(object, [&]() -> std::shared_ptr<io_object_metadata> {
        started.set_value();
        gate.wait();
        if (fail) throw std::runtime_error("parse failed");
        return payload;
      });
    });
    started.get_future().wait();
    std::atomic<int> extra_loads{0};
    auto waiter = std::async(std::launch::async, [&] {
      return store.get_or_load(object, [&] {
        ++extra_loads;
        return payload;
      });
    });
    bool joined = wait_for_joined(store, 1);
    release.set_value();
    CHECK(joined);
    if (fail) {
      CHECK_THROWS_WITH(producer.get(), "parse failed");
      CHECK_THROWS_WITH(waiter.get(), "parse failed");
      CHECK_FALSE(store.get_metadata(object));
      CHECK(store.get_or_load(object, [&] { return payload; }) == payload);
      CHECK(store.statistics().failures == 1);
      CHECK(store.statistics().builds == 2);
    } else {
      CHECK(producer.get() == payload);
      CHECK(waiter.get() == payload);
      CHECK(store.statistics().builds == 1);
    }
    CHECK(extra_loads == 0);
    CHECK(store.describe_metadata_store().find("in_flight=0") != std::string::npos);
  }
}

TEST_CASE("Cancelling a metadata waiter leaves the shared producer alive", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object object("file", "file:v1", tag());
  std::promise<void> started, release;
  auto gate     = release.get_future().share();
  auto payload  = std::make_shared<value>(32);
  auto producer = std::async(std::launch::async, [&] {
    return store.get_or_load(object, [&] {
      started.set_value();
      gate.wait();
      return payload;
    });
  });
  started.get_future().wait();
  std::stop_source stop;
  auto waiter = std::async(std::launch::async, [&] {
    return store.get_or_load(object, [&] { return payload; }, nullptr, stop.get_token());
  });
  bool joined = wait_for_joined(store, 1);
  stop.request_stop();
  auto status = waiter.wait_for(std::chrono::seconds(5));
  release.set_value();
  CHECK(joined);
  CHECK(status == std::future_status::ready);
  CHECK_THROWS_AS(waiter.get(), std::system_error);
  CHECK(producer.get() == payload);
  CHECK(store.get_metadata(object) == payload);
  CHECK(store.statistics().cancelled_waiters == 1);
}

TEST_CASE("Observing a newer identity fences an older in-flight publication", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object old("file", "file:old", tag("old"));
  fake_object newer("file", "file:new", tag("new"));
  auto payload = std::make_shared<value>(32);
  auto result  = store.get_or_load(old, [&] {
    store.open_observed(newer);
    return payload;
  });
  CHECK(result == payload);
  CHECK_FALSE(store.get_metadata(old));
  CHECK_FALSE(store.has_path("file"));
  CHECK(store.statistics().stale == 1);
  auto fresh = std::make_shared<value>(64);
  store.register_metadata(newer, fresh);
  CHECK(store.get_metadata(newer) == fresh);
  CHECK(result == payload);
}

TEST_CASE("Metadata flights do not join a different identity", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object old("file", "file:old", tag("old"));
  fake_object newer("file", "file:new", tag("new"));
  auto payload = std::make_shared<value>(32);
  auto fresh   = std::make_shared<value>(64);
  std::promise<void> started, release;
  auto gate     = release.get_future().share();
  auto producer = std::async(std::launch::async, [&] {
    return store.get_or_load(old, [&] {
      started.set_value();
      gate.wait();
      return payload;
    });
  });
  started.get_future().wait();
  auto independent = std::async(std::launch::async, [&] {
    auto other_identity = store.get_or_load(newer, [&] { return fresh; });
    return other_identity == fresh;
  });
  auto status      = independent.wait_for(std::chrono::seconds(5));
  release.set_value();
  CHECK(status == std::future_status::ready);
  CHECK(independent.get());
  CHECK(producer.get() == payload);
  CHECK(store.get_metadata(newer) == fresh);
  CHECK(store.statistics().joined == 0);
  CHECK(store.statistics().builds == 2);
}

TEST_CASE("Metadata profiles are independent cache identities", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object object("file", "file:v1", tag());
  auto first  = std::make_shared<value>(32);
  auto second = std::make_shared<value>(64);
  CHECK(store.get_or_load(object, [&] { return first; }, nullptr, {}, "profile-a") == first);
  CHECK(store.get_or_load(object, [&] { return second; }, nullptr, {}, "profile-b") == second);
  CHECK(store.get_metadata(object, "profile-a") == first);
  CHECK(store.get_metadata(object, "profile-b") == second);
  CHECK(store.statistics().builds == 2);
}

TEST_CASE("Targeted metadata invalidation preserves other profiles", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object object("file", "file:v1", tag());
  auto canonical = std::make_shared<value>(32);
  auto profile   = std::make_shared<value>(64);
  store.register_metadata(object, canonical, "");
  store.register_metadata(object, profile, "iceberg_flat_v1");

  CHECK(store.invalidate("file", "iceberg_flat_v1"));
  CHECK(store.get_metadata(object, "") == canonical);
  CHECK_FALSE(store.get_metadata(object, "iceberg_flat_v1"));
  CHECK_FALSE(store.invalidate("file", "iceberg_flat_v1"));
}

TEST_CASE("Same-key opens preserve warm metadata and advance the observation fence",
          "[metadata_store]")
{
  cache::metadata_store store;
  fake_object first("file", "file:a", tag("a"));
  fake_object delayed("file", "file:b", tag("b"));
  fake_object latest("file", "file:a", tag("a"));
  auto payload = std::make_shared<value>(32);
  store.register_metadata(first, payload);
  store.open_observed(latest);
  store.open_observed(delayed);
  store.register_metadata(delayed, std::make_shared<value>(64));
  CHECK(store.get_metadata(latest) == payload);
  CHECK(store.statistics().replacements == 0);
  CHECK(store.statistics().stale == 1);
}

TEST_CASE("Metadata store isolates identity and ioctx", "[metadata_store]")
{
  cache::metadata_store store, other;
  auto v = std::make_shared<value>(100);
  fake_object first("file", "file:v1", tag());
  store.register_metadata(first, v);
  CHECK(store.get_metadata(first) == v);
  CHECK_FALSE(other.get_metadata(first));
  CHECK(store.get_metadata(first) == v);

  auto next = std::make_shared<value>(200);
  fake_object replacement("file", "file:v2", tag("v2"));
  store.register_metadata(replacement, next);
  CHECK(store.statistics().entries == 1);
  store.register_metadata(first, v);
  CHECK(store.get_metadata(replacement) == next);
  CHECK(store.statistics().bypasses == 1);
  store.register_metadata(replacement, next);
  CHECK(store.statistics().entries == 1);
  CHECK(store.statistics().bypasses == 1);
  CHECK(v->bytes == 100);
}

TEST_CASE("Identity replacement invalidates entries for a path", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object first("file", "file:v1", tag("v1"));
  store.register_metadata(first, std::make_shared<value>(100));
  CHECK(store.statistics().entries == 1);

  fake_object replacement("file", "file:v2", tag("v2"));
  store.open_observed(replacement);
  CHECK_FALSE(store.get_metadata(first));
  CHECK_FALSE(store.has_path("file"));
  CHECK(store.statistics().entries == 0);
}

TEST_CASE("Identity replacement retires every metadata profile for a path", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object first("file", "file:v1", tag("v1"));
  auto canonical = std::make_shared<value>(100);
  auto iceberg   = std::make_shared<value>(200);
  auto extra     = std::make_shared<value>(300);
  store.register_metadata(first, canonical, "");
  store.register_metadata(first, iceberg, "iceberg_flat_v1");
  store.register_metadata(first, extra, "other");
  CHECK(store.statistics().entries == 3);

  fake_object replacement("file", "file:v2", tag("v2"));
  store.open_observed(replacement);

  CHECK_FALSE(store.get_metadata(first, ""));
  CHECK_FALSE(store.get_metadata(first, "iceberg_flat_v1"));
  CHECK_FALSE(store.get_metadata(first, "other"));
  CHECK(store.statistics().entries == 0);
}

TEST_CASE("Metadata store supports concurrent publication", "[metadata_store]")
{
  cache::metadata_store store;
  std::atomic<bool> all_hits{true};
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back([&, i] {
      for (int j = 0; j < 100; ++j) {
        auto object = fake_object("file-" + std::to_string(i) + "-" + std::to_string(j),
                                  "key-" + std::to_string(i) + "-" + std::to_string(j),
                                  tag());
        store.register_metadata(object, std::make_shared<value>(512));
        auto held = store.get_metadata(object);
        if (!held) all_hits.store(false, std::memory_order_relaxed);
      }
    });
  for (auto& thread : threads)
    thread.join();
  CHECK(all_hits.load(std::memory_order_relaxed));
  CHECK(store.statistics().entries == 400);
}

TEST_CASE("Open-local entries do not advertise reusable path hints", "[metadata_store]")
{
  cache::metadata_store store;
  fake_object object("file", "file:open", {identity_kind::opened_handle, 42, "1"});
  auto v = std::make_shared<value>(256);
  store.register_metadata(object, v);
  CHECK_FALSE(store.has_path("file"));
  CHECK(store.get_metadata(object) == v);
  fake_object reopened("file", "file:open2", {identity_kind::opened_handle, 42, "2"});
  CHECK_FALSE(store.get_metadata(reopened));
}

TEST_CASE("Local metadata reuse requires the observed identity", "[metadata_store]")
{
  object_identity original{identity_kind::local_stat, 256, "3"};
  original.device   = 1;
  original.inode    = 2;
  original.ctime_ns = 3;
  fake_object object("file", "file:local", original);
  cache::metadata_store store;
  auto stable = std::make_shared<value>(256);
  store.register_metadata(object, stable);
  CHECK(store.get_metadata(object) == stable);

  auto replacement  = original;
  replacement.inode = 4;
  fake_object replaced("file", "file:local", replacement);
  CHECK_FALSE(store.get_metadata(replaced));
  auto fresh = std::make_shared<value>(512);
  store.register_metadata(replaced, fresh);
  CHECK(store.get_metadata(replaced) == fresh);
}

TEST_CASE("Local stat identity reuses metadata across opens", "[metadata_store]")
{
  object_identity local{identity_kind::local_stat, 256, "same-ctime"};
  local.device   = 1;
  local.inode    = 2;
  local.ctime_ns = 3;
  fake_object first("file", "file:local", local);
  fake_object reopened("file", "file:local", local);
  cache::metadata_store store;
  auto old_value = std::make_shared<value>(256);
  store.register_metadata(first, old_value);
  CHECK(store.get_metadata(reopened) == old_value);
  bool loaded    = false;
  auto new_value = store.get_or_load(reopened, [&] {
    loaded = true;
    return std::make_shared<value>(512);
  });
  CHECK_FALSE(loaded);
  CHECK(new_value == old_value);
  CHECK(store.get_metadata(reopened) == old_value);
}

TEST_CASE("Metadata replacement releases payloads outside the store lock", "[metadata_store]")
{
  cache::metadata_store store;
  bool released = false;
  struct reentrant_value : value {
    std::function<void()> on_release;
    explicit reentrant_value(std::function<void()> callback)
      : value(128), on_release(std::move(callback))
    {
    }
    ~reentrant_value() override { on_release(); }
  };
  fake_object first("file", "file:v1", tag("v1"));
  store.register_metadata(
    first, std::make_shared<reentrant_value>([&] { released = store.statistics().entries == 1; }));
  fake_object second("file", "file:v2", tag("v2"));
  store.register_metadata(second, std::make_shared<value>(128));
  CHECK(released);
}

TEST_CASE("Metadata path and profile delimiters cannot alias another tuple",
          "[metadata_store][r4_review]")
{
  auto const reverse = GENERATE(false, true);
  cache::metadata_store store;
  fake_object first("p", "same-key", tag());
  fake_object second("p\nprofile=q", "same-key", tag());
  auto a = std::make_shared<value>(32);
  auto b = std::make_shared<value>(64);
  CHECK(store.get_or_load(first, [&] { return a; }, nullptr, {}, "q") == a);
  CHECK(store.get_or_load(second, [&] { return b; }) == b);
  CHECK(store.get_metadata(first, "q") == a);
  CHECK(store.get_metadata(second) == b);
  CHECK(store.statistics().entries == 2);
  CHECK(store.statistics().builds == 2);
  CHECK(store.has_path(second.object_path()));
  // has_path is a hint for the canonical profile only.
  CHECK_FALSE(store.has_path(first.object_path()));
  if (reverse) {
    CHECK(store.invalidate(second.object_path()));
    CHECK_FALSE(store.get_metadata(second));
    CHECK(store.get_metadata(first, "q") == a);
  } else {
    CHECK(store.invalidate(first.object_path(), "q"));
    CHECK_FALSE(store.get_metadata(first, "q"));
    CHECK(store.get_metadata(second) == b);
  }
  CHECK(store.statistics().entries == 1);
  CHECK_FALSE(
    store.invalidate(reverse ? second.object_path() : first.object_path(), reverse ? "" : "q"));
  CHECK(store.statistics().entries == 1);
}

TEST_CASE("Metadata identity retirement follows the raw path across encoded profiles",
          "[metadata_store][r4_review]")
{
  cache::metadata_store store;
  fake_object first("p", "same-key", tag());
  fake_object second("p\nprofile=q", "same-key", tag());
  auto a = std::make_shared<value>(32);
  auto b = std::make_shared<value>(64);
  for (auto const& profile :
       std::vector<std::string>{"", "q", "\nprofile=q", "1:p:q", std::string("a\0b", 3)})
    store.register_metadata(first, a, profile);
  store.register_metadata(second, b);
  CHECK(store.statistics().entries == 6);
  fake_object newer(first.object_path(), "new", tag("v2"));
  store.open_observed(newer);
  CHECK(store.statistics().entries == 1);
  CHECK(store.statistics().replacements == 5);
  CHECK_FALSE(store.has_path(first.object_path()));
  CHECK(store.has_path(second.object_path()));
  CHECK(store.get_metadata(second) == b);
  store.register_metadata(newer, a, "q");
  CHECK(store.get_metadata(newer, "q") == a);
  CHECK(store.get_metadata(second) == b);
}

TEST_CASE("Metadata colliding path profile spellings keep independent flights",
          "[metadata_store][r4_review]")
{
  cache::metadata_store store;
  fake_object first("p", "same-key", tag());
  fake_object second("p\nprofile=q", "same-key", tag());
  auto a = std::make_shared<value>(32);
  auto b = std::make_shared<value>(64);
  std::promise<void> started, release;
  auto gate     = release.get_future().share();
  auto producer = std::async(std::launch::async, [&] {
    return store.get_or_load(
      first,
      [&] {
        started.set_value();
        gate.wait();
        return a;
      },
      nullptr,
      {},
      "q");
  });
  started.get_future().wait();
  auto independent =
    std::async(std::launch::async, [&] { return store.get_or_load(second, [&] { return b; }); });
  auto status = independent.wait_for(std::chrono::seconds(2));
  release.set_value();
  CHECK(status == std::future_status::ready);
  CHECK(independent.get() == b);
  CHECK(producer.get() == a);
  CHECK(store.statistics().joined == 0);
  CHECK(store.statistics().builds == 2);
  CHECK(store.get_metadata(first, "q") == a);
  CHECK(store.get_metadata(second) == b);
}

TEST_CASE("Metadata tuple encoding accepts empty paths and embedded delimiters",
          "[metadata_store][r4_review]")
{
  cache::metadata_store store;
  std::vector<std::pair<std::string, std::string>> tuples{{"", ""},
                                                          {"", "1:p"},
                                                          {"1:", "p"},
                                                          {"p:", "q"},
                                                          {"p", ":q"},
                                                          {"p\nprofile=", "q"},
                                                          {"p", "\nprofile=q"},
                                                          {std::string("p\0q", 3), ""},
                                                          {"p", std::string("\0q", 2)}};
  std::vector<std::unique_ptr<fake_object>> objects;
  std::vector<std::shared_ptr<value>> payloads;
  for (auto const& [path, profile] : tuples) {
    objects.push_back(std::make_unique<fake_object>(path, "same-key", tag()));
    payloads.push_back(std::make_shared<value>(payloads.size()));
    store.register_metadata(*objects.back(), payloads.back(), profile);
  }
  CHECK(store.statistics().entries == tuples.size());
  for (size_t i = 0; i < tuples.size(); ++i) {
    CHECK(store.get_metadata(*objects[i], tuples[i].second) == payloads[i]);
    CHECK(store.invalidate(tuples[i].first, tuples[i].second));
    CHECK(store.statistics().entries == tuples.size() - i - 1);
  }
}
