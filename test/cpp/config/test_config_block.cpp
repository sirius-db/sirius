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

struct pool_options : block_base<pool_options> {
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

struct downgrade_options : block_base<downgrade_options> {
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

struct gpu_space_options : block_base<gpu_space_options> {
  config_option<int> device_id  = option<int>("device_id").mark_mandatory().build();
  config_option<bytes> capacity = option<bytes>("memory_capacity").build();

  auto fields() { return field_list(device_id, capacity); }
};

struct executor_options : block_base<executor_options> {
  pool_options pipeline;
  downgrade_options downgrade;
  std::vector<gpu_space_options> gpus;

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
    block.read(n, "sirius.executor", environment::prod);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("an absent block takes every fallback", "[config][config_block]")
{
  executor_options executor;
  executor.read(node{}, "sirius.executor", environment::prod);
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
  executor_options executor;
  executor.read(root, "sirius.executor", environment::prod);
  REQUIRE(executor.pipeline.user_defined());
  REQUIRE(executor.pipeline.num_threads.user_defined());
  REQUIRE(executor.pipeline.num_threads.get() == 8);
  REQUIRE(executor.downgrade.user_defined());
  REQUIRE(executor.downgrade.trigger.is_fallback());
}

TEST_CASE("errors name the dotted path", "[config][config_block]")
{
  executor_options executor;
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
  executor_options executor;
  node pipeline = scalars({{"num_threads", "oops"}});
  pipeline.set("thread_name_prefix", node{});
  node root = node::mapping();
  root.set("pipeline", pipeline);
  REQUIRE(read_error(executor, root) ==
          "'sirius.executor.pipeline.thread_name_prefix': removed; thread names are internal");
}

TEST_CASE("validate() checks rules across fields", "[config][config_block]")
{
  executor_options executor;
  node root = node::mapping();
  root.set("downgrade", scalars({{"trigger_fraction", "0.5"}}));
  REQUIRE(read_error(executor, root) ==
          "'sirius.executor.downgrade': stop_fraction must be below trigger_fraction");
}

TEST_CASE("lists read one block per element", "[config][config_block]")
{
  executor_options executor;
  node root = node::mapping();
  root.set("gpu",
           node::sequence({scalars({{"device_id", "0"}, {"memory_capacity", "8 GiB"}}),
                           scalars({{"device_id", "1"}})}));
  executor.read(root, "sirius.executor", environment::prod);
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

  executor_options executor;
  executor.read(root, "sirius.executor", environment::prod);
  // File values only: the empty downgrade block stays (it was present), fallbacks do not.
  REQUIRE(executor.write(false) == root);

  executor_options reread;
  reread.read(executor.write(true), "sirius.executor", environment::prod);
  REQUIRE(reread.pipeline.cpu_affinity.get() == std::vector<int>{1, 3});
  REQUIRE(reread.downgrade.trigger.user_defined());  // fallbacks were written out
  REQUIRE(reread.downgrade.trigger.get() == 0.8);
  REQUIRE(reread.gpus.at(0).capacity.get() == bytes(1536_MiB));

  // An absent block writes nothing.
  executor_options empty;
  empty.read(node{}, "sirius.executor", environment::prod);
  REQUIRE(empty.write(false) == node::mapping());
}

TEST_CASE("blocks are read and written through the type-erased interface", "[config][config_block]")
{
  executor_options executor;
  config_block& erased = executor;
  erased.read(scalars({}), "sirius.executor", environment::prod);
  REQUIRE(erased.user_defined());
  REQUIRE(erased.write(true).find("pipeline")->find("num_threads")->text() == "4");
}
