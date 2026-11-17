// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/executor.hpp"
#include "config/yaml.hpp"

#include <chrono>
#include <string>
#include <string_view>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;
using namespace std::chrono_literals;

namespace {

struct executor_root : block_base<executor_root> {
  executor_options executor;
  auto fields() { return field_list(sub("executor", executor)); }
};

executor_root read(std::string_view text)
{
  executor_root root;
  read_yaml(root, text, "sirius", environment::prod);
  return root;
}

std::string read_error(std::string_view text)
{
  executor_root root;
  try {
    read_yaml(root, text, "sirius", environment::prod);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("executor blocks declare their keys once", "[config][config_executor]")
{
  executor_root{}.check_declaration();
}

TEST_CASE("absent executor blocks leave thread counts to the machine", "[config][config_executor]")
{
  auto const root      = read("sirius: {}\n");
  auto const& executor = root.executor;
  REQUIRE(executor.task_creator.num_threads.is_unset());
  REQUIRE(executor.pipeline.num_threads.is_unset());
  REQUIRE(executor.downgrade.num_threads.is_unset());
  REQUIRE(executor.task_creator.cpu_affinity.get().empty());
  REQUIRE(executor.downgrade.cpu_affinity.get().empty());
  REQUIRE(executor.downgrade.monitor_period.get() == 10ms);
  REQUIRE(executor.downgrade.copy_chunk_bytes.get() == 1_GiB);
}

TEST_CASE("executor blocks read thread counts, affinities and downgrade settings",
          "[config][config_executor]")
{
  auto const root      = read(R"(sirius:
  executor:
    task_creator:
      num_threads: 2
      cpu_affinity: [0, 1]
    pipeline:
      num_threads: 6
    downgrade:
      num_threads: 3
      cpu_affinity: [4]
      monitor_period: 1s
      copy_chunk_bytes: 256 MiB
)");
  auto const& executor = root.executor;
  REQUIRE(executor.task_creator.num_threads.get() == 2);
  REQUIRE(executor.task_creator.cpu_affinity.get() == std::vector<int>{0, 1});
  REQUIRE(executor.pipeline.num_threads.user_defined());
  REQUIRE(executor.pipeline.num_threads.get() == 6);
  REQUIRE(executor.downgrade.num_threads.get() == 3);
  REQUIRE(executor.downgrade.monitor_period.get() == 1000ms);
  REQUIRE(executor.downgrade.copy_chunk_bytes.get() == 256_MiB);

  // Bare numbers keep their unit; 0 turns the monitor off.
  REQUIRE(read("sirius:\n  executor:\n    downgrade:\n      monitor_period: 0\n")
            .executor.downgrade.monitor_period.get() == 0ms);
}

TEST_CASE("executor blocks reject bad values and retired keys", "[config][config_executor]")
{
  for (auto const* block : {"task_creator", "pipeline", "downgrade"}) {
    INFO(block);
    REQUIRE(
      read_error(std::string("sirius:\n  executor:\n    ") + block + ":\n      num_threads: 0\n") ==
      std::string("'sirius.executor.") + block +
        ".num_threads': value out of range: '0' (must be greater than 0)");
    REQUIRE(read_error(std::string("sirius:\n  executor:\n    ") + block +
                       ":\n      thread_name_prefix: x\n") ==
            std::string("unknown config key: 'thread_name_prefix' in sirius.executor.") + block);
  }
  REQUIRE(read_error("sirius:\n  executor:\n    pipeline:\n      cpu_affinity: [0]\n") ==
          "unknown config key: 'cpu_affinity' in sirius.executor.pipeline");
  REQUIRE(read_error("sirius:\n  executor:\n    downgrade:\n      monitor_period: -1ms\n") ==
          "'sirius.executor.downgrade.monitor_period': value out of range: '-1 ms' (must not be "
          "negative)");
  REQUIRE(read_error("sirius:\n  executor:\n    task_creator:\n      strategy: ~\n") ==
          "'sirius.executor.task_creator.strategy': removed; task creation policy is internal and "
          "currently demand-driven; remove this key");
  REQUIRE_THAT(
    read_error("sirius:\n  executor:\n    task_creator:\n      priority_order: [a]\n"),
    Catch::Matchers::StartsWith("'sirius.executor.task_creator.priority_order': removed"));
}

TEST_CASE("executor blocks write what they read", "[config][config_executor]")
{
  constexpr std::string_view text = R"(sirius:
  executor:
    task_creator:
      cpu_affinity: [2, 3]
    downgrade:
      monitor_period: 2 min
)";
  auto const root                 = read(text);
  REQUIRE(parse_yaml(write_yaml(root, "sirius")) == parse_yaml(text));
  auto const all = read(write_yaml(root, "sirius", true));
  REQUIRE(all.executor.downgrade.copy_chunk_bytes.user_defined());
  REQUIRE(all.executor.pipeline.num_threads.is_unset());  // no fallback to write
}
