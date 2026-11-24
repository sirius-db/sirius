// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"

#include <chrono>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;
using namespace std::chrono_literals;

namespace {

struct sample_pool : block_base<sample_pool> {
  config_option<int> num_threads =
    option<int>("num_threads").validate([](int v) { return v > 0; }).fallback(4).build();
  config_option<std::vector<int>> cpu_affinity = option<std::vector<int>>("cpu_affinity").build();

  auto fields()
  {
    return field_list(num_threads,
                      cpu_affinity,
                      removed("thread_name_prefix", "removed; thread names are internal"));
  }
};

struct sample_downgrade : block_base<sample_downgrade> {
  config_option<double> trigger = option<double>("trigger_fraction").fallback(0.8).build();
  config_option<double> stop    = option<double>("stop_fraction").fallback(0.6).build();
  config_option<std::chrono::milliseconds> period =
    option<std::chrono::milliseconds>("monitor_period").fallback(100ms).build();

  auto fields() { return field_list(trigger, stop, period); }

  void validate(std::string_view path) const
  {
    if (stop.get() >= trigger.get()) {
      throw config_error("'" + std::string(path) +
                         "': stop_fraction must be below trigger_fraction");
    }
  }
};

struct sample_gpu_space : block_base<sample_gpu_space> {
  config_option<int> device_id  = option<int>("device_id").mark_mandatory().build();
  config_option<bytes> capacity = option<bytes>("memory_capacity").build();

  auto fields() { return field_list(device_id, capacity); }
};

struct sample_executor : block_base<sample_executor> {
  sample_pool pipeline;
  sample_downgrade downgrade;
  std::vector<sample_gpu_space> gpus;

  auto fields()
  {
    return field_list(sub("pipeline", pipeline), sub("downgrade", downgrade), list("gpu", gpus));
  }
};

node scalars(std::initializer_list<std::pair<std::string, std::string>> entries)
{
  node block = node::mapping();
  for (auto const& [key, text] : entries) {
    block.set(key, node::scalar(text));
  }
  return block;
}

std::string read_error(config_block& block, const node& n)
{
  try {
    block.read(n, "sirius.executor");
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("an absent block takes every fallback", "[config][config_block]")
{
  sample_executor executor;
  executor.read(node{}, "sirius.executor");
  REQUIRE_FALSE(executor.user_defined());
  REQUIRE_FALSE(executor.pipeline.user_defined());
  REQUIRE(executor.pipeline.num_threads.is_fallback());
  REQUIRE(executor.pipeline.num_threads.get() == 4);
  REQUIRE(executor.pipeline.cpu_affinity.is_unset());
  REQUIRE(executor.downgrade.period.get() == 100ms);
  REQUIRE(executor.gpus.empty());
}

TEST_CASE("nested blocks read their own mapping", "[config][config_block]")
{
  node root = node::mapping();
  root.set("pipeline", scalars({{"num_threads", "8"}}));
  root.set("downgrade", node::mapping());  // present but empty
  sample_executor executor;
  executor.read(root, "sirius.executor");
  REQUIRE(executor.pipeline.user_defined());
  REQUIRE(executor.pipeline.num_threads.user_defined());
  REQUIRE(executor.pipeline.num_threads.get() == 8);
  REQUIRE(executor.downgrade.user_defined());
  REQUIRE(executor.downgrade.trigger.is_fallback());
}

TEST_CASE("errors name the dotted path", "[config][config_block]")
{
  sample_executor executor;
  node root = node::mapping();
  root.set("pipeline", scalars({{"num_threads", "0"}}));
  REQUIRE(read_error(executor, root) ==
          "'sirius.executor.pipeline.num_threads': value out of range: '0'");

  root = node::mapping();
  root.set("pipeline", scalars({{"num_thread", "2"}}));
  REQUIRE(read_error(executor, root) ==
          "unknown config key: 'num_thread' in sirius.executor.pipeline");

  root = node::mapping();
  root.set("pipeline", node::scalar("fast"));
  REQUIRE(read_error(executor, root) ==
          "'sirius.executor.pipeline': expected a mapping, got a scalar");

  REQUIRE(read_error(executor, scalars({{"scan", "1"}})) ==
          "unknown config key: 'scan' in sirius.executor");
}

TEST_CASE("removed keys fail even with a null value, before other errors", "[config][config_block]")
{
  sample_executor executor;
  node pipeline = scalars({{"num_threads", "oops"}});
  pipeline.set("thread_name_prefix", node{});
  node root = node::mapping();
  root.set("pipeline", pipeline);
  REQUIRE(read_error(executor, root) ==
          "'sirius.executor.pipeline.thread_name_prefix': removed; thread names are internal");
}

TEST_CASE("validate() checks rules across fields", "[config][config_block]")
{
  sample_executor executor;
  node root = node::mapping();
  root.set("downgrade", scalars({{"trigger_fraction", "0.5"}}));
  REQUIRE(read_error(executor, root) ==
          "'sirius.executor.downgrade': stop_fraction must be below trigger_fraction");
}

TEST_CASE("lists read one block per element", "[config][config_block]")
{
  sample_executor executor;
  node root = node::mapping();
  root.set("gpu",
           node::sequence({scalars({{"device_id", "0"}, {"memory_capacity", "8 GiB"}}),
                           scalars({{"device_id", "1"}})}));
  executor.read(root, "sirius.executor");
  REQUIRE(executor.gpus.size() == 2);
  REQUIRE(executor.gpus[0].capacity.get() == 8_GiB);
  REQUIRE(executor.gpus[1].device_id.get() == 1);
  REQUIRE(executor.gpus[1].capacity.is_unset());

  root.set("gpu", node::sequence({scalars({{"device_id", "0"}}), scalars({})}));
  REQUIRE(read_error(executor, root) == "'sirius.executor.gpu[1].device_id': required but missing");

  root.set("gpu", scalars({{"device_id", "0"}}));
  REQUIRE(read_error(executor, root) ==
          "'sirius.executor.gpu': expected a sequence, got a mapping");
}

TEST_CASE("write emits what was read and reads back to it", "[config][config_block]")
{
  node pipeline = scalars({{"num_threads", "8"}});
  pipeline.set("cpu_affinity", node::sequence({node::scalar("1"), node::scalar("3")}));
  node root = node::mapping();
  root.set("pipeline", pipeline);
  root.set("downgrade", node::mapping());
  root.set("gpu", node::sequence({scalars({{"device_id", "2"}, {"memory_capacity", "1536 MiB"}})}));

  sample_executor executor;
  executor.read(root, "sirius.executor");
  // File values only: the empty downgrade block stays (it was present), fallbacks do not.
  REQUIRE(executor.write(false) == root);

  sample_executor reread;
  reread.read(executor.write(true), "sirius.executor");
  REQUIRE(reread.pipeline.cpu_affinity.get() == std::vector<int>{1, 3});
  REQUIRE(reread.downgrade.trigger.user_defined());  // fallbacks were written out
  REQUIRE(reread.downgrade.trigger.get() == 0.8);
  REQUIRE(reread.gpus.at(0).capacity.get() == bytes(1536_MiB));

  // An absent block writes nothing.
  sample_executor empty;
  empty.read(node{}, "sirius.executor");
  REQUIRE(empty.write(false) == node::mapping());
}

TEST_CASE("blocks are read and written through the type-erased interface", "[config][config_block]")
{
  sample_executor executor;
  config_block& erased = executor;
  erased.read(scalars({}), "sirius.executor");
  REQUIRE(erased.user_defined());
  REQUIRE(erased.write(true).find("pipeline")->find("num_threads")->text() == "4");
}

// ---- failure guarantees, lists and declaration checks -------------------------

namespace {

struct sample_tls : block_base<sample_tls> {
  config_option<std::string> cert = option<std::string>("cert").mark_mandatory().build();
  auto fields() { return field_list(cert); }
};

struct sample_memory : block_base<sample_memory> {
  config_option<bytes, double> capacity =
    option<bytes>("capacity_bytes").or_type<double>("capacity_fraction").fallback(0.9).build();
  sample_tls tls;
  std::vector<sample_gpu_space> gpus;
  auto fields() { return field_list(capacity, sub("tls", tls), list("gpu", gpus)); }
  void validate(std::string_view path) const
  {
    // Must cope with unset options: validate also runs for an absent block.
    if (auto const* fraction = capacity.get_if<double>(); fraction != nullptr && *fraction == 0) {
      throw config_error("'" + std::string(path) + "': capacity must not be zero");
    }
  }
};

struct duplicate_option_keys : block_base<duplicate_option_keys> {
  config_option<int> a = option<int>("k").build();
  config_option<int> b = option<int>("k").build();
  auto fields() { return field_list(a, b); }
};

struct option_and_sub_share_a_key : block_base<option_and_sub_share_a_key> {
  config_option<int> a = option<int>("k").build();
  sample_pool pool;
  auto fields() { return field_list(a, sub("k", pool)); }
};

struct option_and_removed_share_a_key : block_base<option_and_removed_share_a_key> {
  config_option<int> a = option<int>("k").build();
  auto fields() { return field_list(a, removed("k", "gone")); }
};

struct empty_sub_key : block_base<empty_sub_key> {
  sample_pool pool;
  auto fields() { return field_list(sub("", pool)); }
};

struct holds_a_bad_list : block_base<holds_a_bad_list> {
  std::vector<duplicate_option_keys> items;
  auto fields() { return field_list(list("items", items)); }
};

std::string read_error_at(config_block& block, const node& n, std::string_view path)
{
  try {
    block.read(n, path);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

static_assert(!std::is_copy_assignable_v<config_block>);
static_assert(!std::is_copy_constructible_v<config_block>);
static_assert(std::is_copy_constructible_v<sample_executor>);
static_assert(std::is_nothrow_move_constructible_v<sample_executor>);

TEST_CASE("a failed read leaves the block unchanged", "[config][config_block]")
{
  node root = node::mapping();
  root.set("pipeline", scalars({{"num_threads", "8"}}));
  root.set("gpu", node::sequence({scalars({{"device_id", "5"}})}));
  sample_executor executor;
  executor.read(root, "sirius.executor");

  node bad = node::mapping();
  bad.set("pipeline", scalars({{"num_threads", "16"}}));
  bad.set("gpu", node::sequence());
  bad.set("downgrade", scalars({{"trigger_fraction", "x"}}));  // fails after pipeline was read
  REQUIRE_THROWS_AS(executor.read(bad, "sirius.executor"), config_error);
  REQUIRE(executor.pipeline.num_threads.get() == 8);
  REQUIRE(executor.gpus.size() == 1);
  REQUIRE(executor.user_defined());

  // An absent block whose read fails on an unknown key stays absent.
  sample_executor absent;
  REQUIRE_THROWS_AS(absent.read(scalars({{"zzz", "1"}}), "sirius.executor"), config_error);
  REQUIRE_FALSE(absent.user_defined());
  REQUIRE(absent.pipeline.num_threads.is_unset());
}

TEST_CASE("a later read replaces everything", "[config][config_block]")
{
  node pipeline = scalars({{"num_threads", "8"}});
  pipeline.set("cpu_affinity", node::sequence({node::scalar("1")}));
  node root = node::mapping();
  root.set("pipeline", pipeline);
  root.set("gpu", node::sequence({scalars({{"device_id", "5"}})}));
  sample_executor executor;
  executor.read(root, "sirius.executor");

  executor.read(node{}, "sirius.executor");
  REQUIRE(executor.pipeline.num_threads.is_fallback());
  REQUIRE(executor.pipeline.cpu_affinity.is_unset());
  REQUIRE(executor.gpus.empty());
  REQUIRE_FALSE(executor.user_defined());
}

TEST_CASE("mandatory options are required only when their block is present",
          "[config][config_block]")
{
  sample_memory memory;
  memory.read(node::mapping(), "sirius.memory");
  REQUIRE(memory.tls.cert.is_unset());
  REQUIRE_FALSE(memory.tls.user_defined());

  node root = node::mapping();
  root.set("tls", node{});  // a null block is absent
  memory.read(root, "sirius.memory");
  REQUIRE_FALSE(memory.tls.user_defined());
  REQUIRE(memory.write(false) == node::mapping());

  root.set("tls", node::mapping());  // present: cert is now required
  REQUIRE(read_error_at(memory, root, "sirius.memory") ==
          "'sirius.memory.tls.cert': required but missing");
}

TEST_CASE("alternatives inside a block", "[config][config_block]")
{
  sample_memory memory;
  memory.read(scalars({{"capacity_bytes", "1 GiB"}}), "sirius.memory");
  REQUIRE(memory.capacity.get<bytes>() == 1_GiB);
  REQUIRE(read_error_at(memory,
                        scalars({{"capacity_bytes", "1 GiB"}, {"capacity_fraction", "0.5"}}),
                        "sirius.memory") ==
          "'sirius.memory': 'capacity_bytes' and 'capacity_fraction' are mutually exclusive");
  REQUIRE(read_error_at(memory, scalars({{"capacity_fraction", "0"}}), "sirius.memory") ==
          "'sirius.memory': capacity must not be zero");
}

TEST_CASE("messages at the root path", "[config][config_block]")
{
  sample_executor executor;
  REQUIRE(read_error_at(executor, node::scalar("x"), "") == "expected a mapping, got a scalar");
  REQUIRE(read_error_at(executor, scalars({{"zzz", "1"}}), "") == "unknown config key: 'zzz'");
}

TEST_CASE("list shapes", "[config][config_block]")
{
  sample_executor executor;
  node root = node::mapping();
  root.set("gpu", node::sequence());
  executor.read(root, "sirius.executor");
  REQUIRE(executor.gpus.empty());
  REQUIRE(executor.write(false) == node::mapping());  // an empty list writes like an absent one

  root.set("gpu", node{});
  executor.read(root, "sirius.executor");
  REQUIRE(executor.gpus.empty());

  root.set("gpu", node::sequence({node::scalar("x")}));
  REQUIRE(read_error_at(executor, root, "sirius.executor") ==
          "'sirius.executor.gpu[0]': expected a mapping, got a scalar");

  root.set("gpu", node::sequence({scalars({{"device_id", "0"}, {"bogus", "1"}})}));
  REQUIRE(read_error_at(executor, root, "sirius.executor") ==
          "unknown config key: 'bogus' in sirius.executor.gpu[0]");

  // A null element is an element with every fallback; a mandatory option is then not required.
  root.set("gpu", node::sequence({node{}}));
  executor.read(root, "sirius.executor");
  REQUIRE(executor.gpus.size() == 1);
  REQUIRE(executor.gpus[0].device_id.is_unset());
}

TEST_CASE("declaration bugs are logic errors", "[config][config_block]")
{
  REQUIRE_THROWS_WITH(duplicate_option_keys{}.check_declaration(),
                      "config block: key 'k' is claimed by two fields");
  REQUIRE_THROWS_WITH(option_and_sub_share_a_key{}.check_declaration(),
                      "config block: key 'k' is claimed by two fields");
  REQUIRE_THROWS_WITH(option_and_removed_share_a_key{}.check_declaration(),
                      "config block: key 'k' is claimed by two fields");
  REQUIRE_THROWS_WITH(empty_sub_key{}.check_declaration(), "config block: empty key");
  REQUIRE_THROWS_AS(holds_a_bad_list{}.check_declaration(), std::logic_error);
  duplicate_option_keys bad;
  REQUIRE_THROWS_AS(bad.read(node{}, ""), std::logic_error);

  sample_executor{}.check_declaration();
  sample_memory{}.check_declaration();
}

TEST_CASE("blocks copy deeply and write through const", "[config][config_block]")
{
  node root = node::mapping();
  root.set("pipeline", scalars({{"num_threads", "8"}}));
  root.set("gpu", node::sequence({scalars({{"device_id", "5"}})}));
  sample_executor original;
  original.read(root, "sirius.executor");

  sample_executor copy = original;
  REQUIRE(copy.user_defined());
  copy.pipeline.num_threads.assign(2);
  copy.gpus.clear();
  REQUIRE(original.pipeline.num_threads.get() == 8);
  REQUIRE(original.gpus.size() == 1);

  std::vector<sample_executor> many{original, original};
  auto moved = std::move(many);
  REQUIRE(moved[1].gpus.at(0).device_id.get() == 5);

  const sample_executor& constant = original;
  REQUIRE(constant.write(false) == root);
}

TEST_CASE("write order is declaration order; values survive any input order",
          "[config][config_block]")
{
  node root = node::mapping();
  root.set("gpu", node::sequence({scalars({{"memory_capacity", "1 GiB"}, {"device_id", "3"}})}));
  root.set("pipeline", scalars({{"num_threads", "8"}}));
  sample_executor executor;
  executor.read(root, "sirius.executor");
  REQUIRE(executor.write(false) != root);

  sample_executor reread;
  reread.read(executor.write(false), "sirius.executor");
  REQUIRE(reread.gpus.at(0).device_id.get() == 3);
  REQUIRE(reread.gpus.at(0).capacity.get() == 1_GiB);
  REQUIRE(reread.pipeline.num_threads.get() == 8);
}
