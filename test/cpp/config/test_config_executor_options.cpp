// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/executor.hpp"
#include "config/yaml.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <source_location>
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
  read_yaml(root, text, "sirius");
  return root;
}

std::string read_error(std::string_view text)
{
  executor_root root;
  try {
    read_yaml(root, text, "sirius");
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
    REQUIRE_THAT(
      read_error(std::string("sirius:\n  executor:\n    ") + block + ":\n      num_threads: 0\n"),
      Catch::Matchers::StartsWith(std::string("'sirius.executor.") + block +
                                  ".num_threads': value out of range: '0' (must be greater than "
                                  "zero); "));
    REQUIRE(read_error(std::string("sirius:\n  executor:\n    ") + block +
                       ":\n      thread_name_prefix: x\n") ==
            std::string("unknown config key: 'thread_name_prefix' in sirius.executor.") + block);
  }
  REQUIRE(read_error("sirius:\n  executor:\n    pipeline:\n      cpu_affinity: [0]\n") ==
          "unknown config key: 'cpu_affinity' in sirius.executor.pipeline");
  REQUIRE(read_error("sirius:\n  executor:\n    downgrade:\n      monitor_period: -1ms\n") ==
          "'sirius.executor.downgrade.monitor_period': value out of range: '-1 ms' (must not be "
          "negative)"
          "; period of the memory pressure monitor");
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

TEST_CASE("a null or empty pool leaves num_threads unset", "[config][config_executor]")
{
  for (auto const* body : {"task_creator: {}\n",
                           "task_creator: ~\n",
                           "task_creator:\n      num_threads: ~\n",
                           "task_creator:\n      num_threads:\n"}) {
    INFO(body);
    auto const root = read(std::string("sirius:\n  executor:\n    ") + body);
    REQUIRE(root.executor.task_creator.num_threads.is_unset());
    REQUIRE_FALSE(root.executor.task_creator.num_threads.user_defined());
  }
}

TEST_CASE("thread counts and affinities are checked", "[config][config_executor]")
{
  auto error = [](std::string_view body) {
    return read_error(std::string("sirius:\n  executor:\n    downgrade:\n      ") +
                      std::string(body) + "\n");
  };
  REQUIRE(error("num_threads: -1") ==
          "'sirius.executor.downgrade.num_threads': value out of range: '-1' (must be greater than "
          "zero); downgrade (spill) executor worker threads");
  REQUIRE(error("num_threads: abc") ==
          "'sirius.executor.downgrade.num_threads': invalid integer value: 'abc'"
          "; downgrade (spill) executor worker threads");
  REQUIRE(error("num_threads: 1.5") ==
          "'sirius.executor.downgrade.num_threads': invalid integer value: '1.5'"
          "; downgrade (spill) executor worker threads");
  REQUIRE(error("num_threads: 2147483648") ==
          "'sirius.executor.downgrade.num_threads': integer out of range: '2147483648'"
          "; downgrade (spill) executor worker threads");
  REQUIRE(
    error("num_threads: 010") ==
    "'sirius.executor.downgrade.num_threads': invalid integer value: '010' (leading zeros are "
    "not allowed)"
    "; downgrade (spill) executor worker threads");
  REQUIRE(error("cpu_affinity: 3") ==
          "'sirius.executor.downgrade.cpu_affinity': expected a sequence, got a scalar"
          "; CPU cores downgrade executor threads are pinned to");
  REQUIRE(error("cpu_affinity: [a]") ==
          "'sirius.executor.downgrade.cpu_affinity': element 0: invalid integer value: 'a'"
          "; CPU cores downgrade executor threads are pinned to");
  REQUIRE(error("cpu_affinity: [~]") ==
          "'sirius.executor.downgrade.cpu_affinity': element 0: expected a scalar, got null"
          "; CPU cores downgrade executor threads are pinned to");

  // Ids are not checked, as today.
  auto const root = read(
    "sirius:\n  executor:\n    downgrade:\n      cpu_affinity: [-1, 3, 3]\n"
    "    task_creator:\n      cpu_affinity: []\n");
  REQUIRE(root.executor.downgrade.cpu_affinity.get() == std::vector<int>{-1, 3, 3});
  REQUIRE(root.executor.task_creator.cpu_affinity.user_defined());
}

TEST_CASE("monitor_period and copy_chunk_bytes forms", "[config][config_executor]")
{
  auto value = [](std::string_view body) {
    return read(std::string("sirius:\n  executor:\n    downgrade:\n      ") + std::string(body) +
                "\n")
      .executor.downgrade;
  };
  auto error = [](std::string_view body) {
    return read_error(std::string("sirius:\n  executor:\n    downgrade:\n      ") +
                      std::string(body) + "\n");
  };
  REQUIRE(value("monitor_period: 10.0").monitor_period.get() == 10ms);
  REQUIRE(value("monitor_period: 0s").monitor_period.get() == 0ms);
  REQUIRE_THAT(error("monitor_period: 0.5ms"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
  REQUIRE_THAT(error("monitor_period: 1.5"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
  REQUIRE_THAT(error("monitor_period: -1"),
               Catch::Matchers::ContainsSubstring("must not be negative"));
  REQUIRE(error("monitor_period: abc") ==
          "'sirius.executor.downgrade.monitor_period': invalid time value: 'abc'; period of the "
          "memory pressure monitor");

  REQUIRE(value("copy_chunk_bytes: 0").copy_chunk_bytes.get() == 0_B);  // the builtin converter
  REQUIRE(value("copy_chunk_bytes: 4194304").copy_chunk_bytes.get() == 4_MiB);
  REQUIRE(error("copy_chunk_bytes: -1") ==
          "'sirius.executor.downgrade.copy_chunk_bytes': byte value must be non-negative"
          "; copy granularity of GPU-to-host spills");
  REQUIRE(error("copy_chunk_bytes: 1 XB") ==
          "'sirius.executor.downgrade.copy_chunk_bytes': unknown byte suffix: 'XB'"
          "; copy granularity of GPU-to-host spills");
  REQUIRE_THAT(error("copy_chunk_bytes: 1.5"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
}

TEST_CASE("removed task_creator keys come first and read as before", "[config][config_executor]")
{
  REQUIRE(read_error("sirius:\n  executor:\n    task_creator:\n      priority_order: ~\n") ==
          "'sirius.executor.task_creator.priority_order': removed; scheduling priority is internal "
          "and currently source-first; remove this key");
  REQUIRE_THAT(read_error("sirius:\n  executor:\n    task_creator:\n      num_threads: 0\n"
                          "      strategy: x\n"),
               Catch::Matchers::StartsWith("'sirius.executor.task_creator.strategy': removed"));
}

TEST_CASE("executor blocks must be mappings", "[config][config_executor]")
{
  REQUIRE(read_error("sirius:\n  executor:\n    task_creator: 5\n") ==
          "'sirius.executor.task_creator': expected a mapping, got a scalar");
  REQUIRE(read_error("sirius:\n  executor: [1]\n") ==
          "'sirius.executor': expected a mapping, got a sequence");
}

TEST_CASE("executor fixtures read through the new blocks", "[config][config_executor]")
{
  auto fixture = [](std::string const& name) {
    auto const dir =
      std::filesystem::path(std::source_location::current().file_name()).parent_path() / "data";
    std::ifstream in(dir / name);
    REQUIRE(in);
    return std::string(std::istreambuf_iterator<char>(in), {});
  };
  struct expectation {
    char const* file;
    char const* error;
  };
  for (auto const& [file, error] : std::initializer_list<expectation>{
         {"invalid_downgrade_thread_name_prefix.yaml",
          "unknown config key: 'thread_name_prefix' in sirius.executor.downgrade"},
         {"invalid_pipeline_cpu_affinity.yaml",
          "unknown config key: 'cpu_affinity' in sirius.executor.pipeline"},
         {"invalid_pipeline_thread_name_prefix.yaml",
          "unknown config key: 'thread_name_prefix' in sirius.executor.pipeline"},
         {"invalid_task_creator_thread_name_prefix.yaml",
          "unknown config key: 'thread_name_prefix' in sirius.executor.task_creator"},
         {"invalid_task_creator_strategy.yaml",
          "'sirius.executor.task_creator.strategy': removed; task creation policy is internal and "
          "currently demand-driven; remove this key"},
         {"invalid_task_creator_priority_order.yaml",
          "'sirius.executor.task_creator.priority_order': removed; scheduling priority is internal "
          "and currently source-first; remove this key"}}) {
    INFO(file);
    REQUIRE(read_error(fixture(file)) == error);
  }
  REQUIRE(read(fixture("valid_task_creator_strategy_omitted.yaml"))
            .executor.task_creator.num_threads.get() == 1);
}
