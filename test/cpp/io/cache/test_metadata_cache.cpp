/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "io/cache/metadata_cache.hpp"
#include "io/cache/metadata_store.hpp"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <thread>
#include <vector>

using namespace sirius::io;
using cache_type = cache::metadata_cache;
namespace {
struct value : io_object_metadata {
  explicit value(size_t bytes) : bytes(bytes) {}
  size_t bytes;
};
class fake_object final : public io_object {
 public:
  fake_object(std::string path, std::string key) : path_(std::move(path)), key_(std::move(key)) {}
  [[nodiscard]] const std::string& raw_file_cache_id() const noexcept override { return key_; }
  [[nodiscard]] const std::string& object_path() const noexcept override { return path_; }
  [[nodiscard]] size_t size() const noexcept override { return 1; }

 private:
  std::string path_;
  std::string key_;
};
object_identity tag(std::string version = "v1")
{
  return {identity_kind::object_tag, 32, std::move(version)};
}
cache_type::options policy() { return {}; }
}  // namespace
TEST_CASE("Metadata cache isolates identity profile and scope", "[metadata_cache]")
{
  cache_type c(policy());
  auto v = std::make_shared<value>(100);
  c.put(1, "file", "parser", tag(), 1, v);
  CHECK(c.get(1, "file", "parser", tag()) == v);
  CHECK_FALSE(c.get(2, "file", "parser", tag()));
  CHECK_FALSE(c.get(1, "file", "other", tag()));
  CHECK_FALSE(c.get(1, "file", "parser", tag("v2")));
  auto next = std::make_shared<value>(200);
  c.put(1, "file", "parser", tag("v2"), 2, next);
  CHECK(c.statistics().entries == 1);
  c.put(1, "file", "parser", tag(), 1, v);
  CHECK(c.get(1, "file", "parser", tag("v2")) == next);
  CHECK(c.statistics().bypasses == 1);
  c.put(1, "file", "parser", tag("v2"), 4, next);
  c.put(1, "file", "parser", tag("late"), 3, v);
  CHECK(c.get(1, "file", "parser", tag("v2")) == next);
  CHECK(c.statistics().entries == 1);
  CHECK(c.statistics().bypasses == 2);
  c.erase_scope(1);
  CHECK(c.statistics().entries == 0);
  CHECK(v->bytes == 100);
}
TEST_CASE("Metadata store preserves raw-key compatibility across generations", "[metadata_cache]")
{
  cache::metadata_store store;
  auto first       = std::make_shared<fake_object>("file", "file:v1");
  auto first_value = std::make_shared<value>(100);
  store.register_metadata(*first, first_value);
  CHECK(store.get_metadata("file:v1") == first_value);

  auto second       = std::make_shared<fake_object>("file", "file:v2");
  auto second_value = std::make_shared<value>(200);
  store.register_metadata(*second, second_value);
  CHECK_FALSE(store.get_metadata("file:v1"));
  CHECK(store.get_metadata("file:v2") == second_value);
}
TEST_CASE("Metadata cache supports concurrent publication", "[metadata_cache]")
{
  auto p = policy();
  cache_type c(p);
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i)
    threads.emplace_back([&, i] {
      for (int j = 0; j < 100; ++j) {
        c.put(i, std::to_string(j), "p", tag(), j, std::make_shared<value>(512));
        auto held = c.get(i, std::to_string(j), "p", tag());
      }
      c.erase_scope(i);
    });
  for (auto& thread : threads)
    thread.join();
  CHECK(c.statistics().entries == 0);
}

TEST_CASE("Open-local entries never suppress the next open footer probe", "[metadata_cache]")
{
  cache_type c(policy());
  object_identity identity{identity_kind::opened_handle, 42, "1"};
  auto v = std::make_shared<value>(256);
  c.put(1, "file", "p", identity, 1, v);
  CHECK_FALSE(c.contains(1, "file"));
  CHECK(c.get(1, "file", "p", identity) == v);
  identity.version = "2";
  CHECK_FALSE(c.get(1, "file", "p", identity));
  c.erase_scope(1);
  CHECK(c.statistics().entries == 0);
}

TEST_CASE("Local metadata reuse requires the observed identity", "[metadata_cache]")
{
  cache_type c(policy());
  object_identity original{identity_kind::local_stat, 256, "3"};
  original.device   = 1;
  original.inode    = 2;
  original.ctime_ns = 3;
  auto stable       = std::make_shared<value>(256);
  c.put(1, "file", "p", original, 1, stable);
  CHECK(c.get(1, "file", "p", original) == stable);

  auto replacement  = original;
  replacement.inode = 4;
  CHECK_FALSE(c.get(1, "file", "p", replacement));
  auto fresh = std::make_shared<value>(512);
  c.put(1, "file", "p", replacement, 3, fresh);
  CHECK(c.get(1, "file", "p", replacement) == fresh);
}

TEST_CASE("Metadata eviction releases payloads outside the manager lock", "[metadata_cache]")
{
  cache_type c(policy());
  bool released = false;
  struct reentrant_value : value {
    std::function<void()> on_release;
    explicit reentrant_value(std::function<void()> callback)
      : value(128), on_release(std::move(callback))
    {
    }
    ~reentrant_value() override { on_release(); }
  };
  c.put(1, "file", "p", tag(), 1, std::make_shared<reentrant_value>([&] {
          released = c.statistics().entries == 0;
        }));
  c.erase_scope(1);
  CHECK(released);
}
