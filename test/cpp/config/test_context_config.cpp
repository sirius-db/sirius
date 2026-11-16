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

#include "catch.hpp"
#include "exec/thread_defaults.hpp"
#include "sirius_config.hpp"

#include <sirius/context/config_builder.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>  // IWYU pragma: keep

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>
#include <type_traits>

namespace {

using sirius::ContextConfig;
using sirius::ContextConfigBuilder;
using sirius::ErrorCode;

static_assert(!std::is_default_constructible_v<ContextConfig>);
static_assert(std::is_nothrow_copy_constructible_v<ContextConfig>);

struct scoped_yaml {
  explicit scoped_yaml(const std::string& text)
  {
    static std::atomic<unsigned> ordinal{0};
    path =
      std::filesystem::temp_directory_path() / ("sirius_public_config_" + std::to_string(getpid()) +
                                                "_" + std::to_string(ordinal++) + ".yaml");
    write(text);
  }
  void write(const std::string& text) const
  {
    std::ofstream out(path);
    out << text;
    REQUIRE(out.good());
  }
  ~scoped_yaml()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  std::filesystem::path path;
};

cucascade::memory::gpu_memory_space_config gpu_space(const sirius::sirius_config& config)
{
  const auto& spaces = config.get_memory_space_configs();
  auto gpu           = std::find_if(spaces.begin(), spaces.end(), [](const auto& space) {
    return std::holds_alternative<cucascade::memory::gpu_memory_space_config>(space);
  });
  REQUIRE(gpu != spaces.end());
  return std::get<cucascade::memory::gpu_memory_space_config>(*gpu);
}

}  // namespace

TEST_CASE("hardware thread defaults round and clamp each pool", "[config][thread_defaults]")
{
  struct example {
    unsigned int cpu;
    std::size_t gpus;
    sirius::exec::hardware_thread_defaults expected;
  };
  for (auto const& row : {
         example{0, 0, {4, 4, 1, 18, 2, 2}},
         example{20, 1, {7, 4, 2, 18, 3, 2}},
         example{31, 1, {8, 4, 3, 18, 5, 2}},
         example{32, 1, {8, 4, 4, 18, 5, 4}},
         example{39, 2, {7, 4, 2, 18, 6, 4}},
         example{64, 8, {4, 4, 1, 18, 10, 4}},
         example{96, 4, {8, 4, 3, 18, 16, 4}},
         example{192, 8, {8, 4, 3, 18, 32, 4}},
         example{std::numeric_limits<unsigned int>::max(), 1, {8, 4, 4, 18, 32, 4}},
         example{32, std::numeric_limits<std::size_t>::max(), {4, 4, 1, 18, 5, 4}},
       }) {
    CAPTURE(row.cpu, row.gpus);
    auto const actual = sirius::exec::derive_thread_defaults(row.cpu, row.gpus);
    CHECK(actual.pipeline == row.expected.pipeline);
    CHECK(actual.task_creator == row.expected.task_creator);
    CHECK(actual.downgrade == row.expected.downgrade);
    CHECK(actual.scan_manager == row.expected.scan_manager);
    CHECK(actual.rest_reactors == row.expected.rest_reactors);
    CHECK(actual.uring_reactors == row.expected.uring_reactors);
  }
}

TEST_CASE("resolved hardware thread defaults preserve individual YAML overrides",
          "[context_config][config][thread_defaults]")
{
  // Low-level memory spaces let resolution exercise multiple GPU counts without
  // requiring those devices or allocating any GPU memory.
  for (int const gpus : {1, 2, 8}) {
    for (auto const* override_key : {"",
                                     "pipeline",
                                     "task_creator",
                                     "downgrade",
                                     "scan_manager",
                                     "uring_n_reactors",
                                     "rest_n_reactors",
                                     "null"}) {
      CAPTURE(gpus, override_key);
      auto root = YAML::Load(R"(sirius:
  topology: {gpus_per_query: 1}
  executor:
    scan_manager: {reserve_rest_reactor_cpus: false}
)");
      for (int gpu = 0; gpu < gpus; ++gpu) {
        YAML::Node space;
        space["device_id"]       = gpu * 2;
        space["memory_capacity"] = "512MiB";
        root["sirius"]["space"]["gpu"].push_back(space);
      }
      auto executor = root["sirius"]["executor"];
      std::string const key(override_key);
      if (key == "pipeline" || key == "task_creator" || key == "downgrade" ||
          key == "scan_manager") {
        executor[key]["num_threads"] = 13;
      } else if (key == "uring_n_reactors" || key == "rest_n_reactors") {
        executor["scan_manager"][key] = key == "rest_n_reactors" ? 0 : 7;
      } else if (key == "null") {
        executor["pipeline"]["num_threads"]          = YAML::Null;
        executor["task_creator"]                     = YAML::Load("{}");
        executor["downgrade"]["num_threads"]         = YAML::Null;
        executor["scan_manager"]["num_threads"]      = YAML::Null;
        executor["scan_manager"]["uring_n_reactors"] = YAML::Null;
        executor["scan_manager"]["rest_n_reactors"]  = YAML::Null;
      }
      auto const expected =
        sirius::exec::derive_thread_defaults(std::thread::hardware_concurrency(), gpus);
      auto const parsed = sirius::parsed_sirius_config::from_node(root);
      for (int repeat = 0; repeat < 2; ++repeat) {
        auto const config   = parsed.resolve({});
        auto const pipeline = key == "pipeline" ? 13 : expected.pipeline;
        CHECK(config.get_gpu_pipeline_executor_config().num_threads == pipeline);
        CHECK(config.get_task_creator_config().thread_pool.num_threads ==
              (key == "task_creator" ? 13 : expected.task_creator));
        CHECK(config.get_downgrade_executor_config().thread_pool.num_threads ==
              (key == "downgrade" ? 13 : expected.downgrade));
        auto const& scan = config.get_scan_manager_config();
        CHECK(scan.thread_pool.num_threads == (key == "scan_manager" ? 13 : expected.scan_manager));
        CHECK(scan.uring_n_reactors == (key == "uring_n_reactors" ? 7 : expected.uring_reactors));
        CHECK(scan.rest_n_reactors == (key == "rest_n_reactors" ? 0 : expected.rest_reactors));
        CHECK(scan.pipeline_width == pipeline);
        CHECK(scan.rest.n_max_concurrent_scans == std::max(8, 2 * pipeline));
      }
    }
  }
}

TEST_CASE("REST fan-out validation uses the resolved hardware reactor count",
          "[context_config][config][thread_defaults]")
{
  auto root          = YAML::Load(R"(sirius:
  space:
    gpu: [{device_id: 0, memory_capacity: 512MiB}]
  executor:
    scan_manager: {reserve_rest_reactor_cpus: false}
)");
  auto const threads = sirius::exec::derive_thread_defaults(std::thread::hardware_concurrency(), 1);
  auto rest          = root["sirius"]["executor"]["scan_manager"]["rest"];
  rest["fan_out"]    = threads.rest_reactors;
  auto const valid   = sirius::parsed_sirius_config::from_node(root).resolve({});
  CHECK(valid.get_scan_manager_config().rest.fan_out == threads.rest_reactors);

  rest["fan_out"] = threads.rest_reactors + 1;
  // Without an explicit reactor count, validation waits for hardware resolution.
  auto const invalid = sirius::parsed_sirius_config::from_node(root);
  CHECK_THROWS_WITH(invalid.resolve({}),
                    Catch::Matchers::ContainsSubstring("must be at most rest_n_reactors"));
}

TEST_CASE("public configuration builds from defaults", "[context_config][config]")
{
  auto result = ContextConfigBuilder{}.build();
  REQUIRE(result.has_value());
}

TEST_CASE("public configuration builds after its YAML file changes or is deleted",
          "[context_config][config]")
{
  const std::string text = R"(sirius:
  topology: {gpus_per_query: 1}
  memory:
    gpu: {usage_limit_bytes: 8Gi, reservation_limit_fraction: 0.7}
  executor:
    pipeline: {num_threads: 3}
  operator_params: {hash_partition_bytes: 32Mi}
)";
  scoped_yaml file(text);
  auto builder = ContextConfigBuilder::from_yaml(file.path);
  REQUIRE(builder.has_value());
  file.write("invalid: yaml");
  auto config = builder->build();
  REQUIRE(config.has_value());
  std::filesystem::remove(file.path);
  REQUIRE(builder->build().has_value());
}

TEST_CASE("public configuration returns errors for invalid inputs", "[context_config][config]")
{
  SECTION("file I/O")
  {
    scoped_yaml file("");
    std::filesystem::remove(file.path);
    auto result = ContextConfigBuilder::from_yaml(file.path);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::configuration_io);
    CHECK(result.error().message.find(file.path.string()) != std::string::npos);
    auto directory = ContextConfigBuilder::from_yaml(std::filesystem::temp_directory_path());
    REQUIRE_FALSE(directory.has_value());
    CHECK(directory.error().code == ErrorCode::configuration_io);
  }
  SECTION("YAML syntax")
  {
    scoped_yaml file("sirius: [");
    auto result = ContextConfigBuilder::from_yaml(file.path);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::malformed_yaml);
  }
  SECTION("YAML semantics")
  {
    scoped_yaml file("sirius: {unknown: 1}");
    auto result = ContextConfigBuilder::from_yaml(file.path);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().code == ErrorCode::invalid_configuration);
  }
  SECTION("malformed time values")
  {
    for (const auto& value : {std::string("-ms"), std::string(400, '9') + "ms"}) {
      scoped_yaml file("sirius: {executor: {scan_manager: {rest: {upkeep_interval_ms: '" + value +
                       "'}}}}");
      auto result = ContextConfigBuilder::from_yaml(file.path);
      REQUIRE_FALSE(result.has_value());
      CHECK(result.error().code == ErrorCode::invalid_configuration);
      CHECK(result.error().message.find(file.path.string()) != std::string::npos);
      CHECK(result.error().message.find("upkeep_interval_ms") != std::string::npos);
    }
  }
}

TEST_CASE("public configuration defers GPU availability checks", "[context_config][config]")
{
  scoped_yaml file("sirius:\n  topology: {num_gpus: 2147483647}\n");
  auto result = ContextConfigBuilder::from_yaml(file.path);
  REQUIRE(result.has_value());
  REQUIRE(result->build().has_value());
}

TEST_CASE("public configuration defers GPU capacity checks", "[context_config][config]")
{
  scoped_yaml file("sirius:\n  memory:\n    gpu: {usage_limit_bytes: 18446744073709551615}\n");
  auto yaml = ContextConfigBuilder::from_yaml(file.path);
  REQUIRE(yaml.has_value());
  REQUIRE(yaml->build().has_value());
}

TEST_CASE("configuration resolver applies YAML GPU limits", "[context_config][config]")
{
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                             /*with_runtime_attributes=*/true));
  const auto& topology = discovery.get_topology();

  SECTION("byte limit preserves explicit settings")
  {
    constexpr std::uint64_t capacity = 256ULL << 20;
    auto parsed = sirius::parsed_sirius_config::from_node(YAML::Load(R"(sirius:
  topology: {num_gpus: 1}
  memory:
    gpu: {usage_limit_bytes: 256Mi, reservation_limit_fraction: 0.7}
  operator_params: {hash_partition_bytes: 32Mi}
)"));
    auto config = parsed.resolve(topology);

    const auto gpu = gpu_space(config);
    CHECK(gpu.memory_capacity == capacity);
    CHECK(gpu.reservation_limit_fraction == 0.7);
    CHECK(config.get_operator_params().scan_task_batch_size == capacity / 40);
    CHECK(config.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu() ==
          4 * (capacity / 40));
    CHECK(config.get_operator_params().hash_partition_bytes == (32ULL << 20));
  }
  SECTION("oversized byte limits fail resolution with source context")
  {
    auto parsed = sirius::parsed_sirius_config::from_node(
      YAML::Load("sirius: {topology: {num_gpus: 1}, memory: {gpu: {usage_limit_bytes: "
                 "18446744073709551615}}}"),
      "capacity.yaml");
    REQUIRE_THROWS_WITH(parsed.resolve(topology),
                        Catch::Matchers::ContainsSubstring("capacity.yaml") &&
                          Catch::Matchers::ContainsSubstring("usage_limit_bytes exceeds"));
  }
  SECTION("fraction limits use physical capacity")
  {
    auto physical =
      sirius::parsed_sirius_config::from_node(
        YAML::Load("sirius: {topology: {num_gpus: 1}, memory: {gpu: {usage_limit_fraction: 1.0}}}"))
        .resolve(topology);
    const auto total = gpu_space(physical).memory_capacity;
    REQUIRE(total > 0);

    for (const auto fraction : {0.0, 0.5, 1.0}) {
      CAPTURE(fraction);
      auto root = YAML::Load("sirius: {topology: {num_gpus: 1}}");
      root["sirius"]["memory"]["gpu"]["usage_limit_fraction"] = fraction;
      auto config = sirius::parsed_sirius_config::from_node(root).resolve(topology);
      CHECK(gpu_space(config).memory_capacity == static_cast<std::size_t>(total * fraction));
    }
  }
}

TEST_CASE("dynamic-filter Bloom budget always follows resolved scan size",
          "[context_config][config][dynamic_filter]")
{
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                             /*with_runtime_attributes=*/true));
  auto root            = YAML::Load("sirius: {topology: {num_gpus: 1}}");
  auto params          = root["sirius"]["operator_params"];
  auto const max_bytes = std::numeric_limits<std::uint64_t>::max();
  for (auto const scan_bytes : {std::uint64_t{1}, std::uint64_t{8} << 20, max_bytes}) {
    CAPTURE(scan_bytes);
    params["scan_task_batch_size"] = scan_bytes;
    auto const parsed              = sirius::parsed_sirius_config::from_node(root);
    auto const resolved            = parsed.resolve(discovery.get_topology());
    auto const expected            = scan_bytes > max_bytes / 4 ? max_bytes : 4 * scan_bytes;
    CHECK(resolved.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu() == expected);

    // Resolution is repeatable and does not mutate the parsed settings.
    CHECK(parsed.resolve(discovery.get_topology())
            .get_operator_params()
            .max_dynamic_filter_bloom_bytes_per_gpu() == expected);
  }
  params["max_dynamic_filter_bloom_bytes_per_gpu"] = "17MiB";
  CHECK_THROWS_WITH(sirius::parsed_sirius_config::from_node(root),
                    Catch::Matchers::ContainsSubstring("always 4x scan_task_batch_size"));
}
