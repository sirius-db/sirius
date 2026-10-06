/*
 * Copyright 2025, Sirius Contributors.
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
#include "sirius_config.hpp"

#include <sirius/context/config_builder.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <limits>
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

TEST_CASE("public configuration preserves defaults", "[context_config][config]")
{
  auto result = ContextConfigBuilder{}.build();
  REQUIRE(result.has_value());
}

TEST_CASE("public configuration preserves YAML and captures the file", "[context_config][config]")
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
  builder->gpu_usage_limit_bytes(4ULL << 30);
  auto overridden = builder->build();
  REQUIRE(overridden.has_value());
}

TEST_CASE("public GPU overrides replace each other and builder copies are independent",
          "[context_config][config]")
{
  ContextConfigBuilder builder;
  builder.gpu_usage_limit_fraction(std::numeric_limits<double>::quiet_NaN())
    .gpu_usage_limit_bytes(8ULL << 30);
  REQUIRE(builder.build().has_value());

  auto copy = builder;
  builder.gpu_usage_limit_fraction(-0.1);
  auto invalid = builder.build();
  REQUIRE_FALSE(invalid.has_value());
  CHECK(invalid.error().code == ErrorCode::invalid_configuration);
  REQUIRE(copy.build().has_value());

  builder.gpu_usage_limit_fraction(0.5);
  REQUIRE(builder.build().has_value());
  copy = builder;
  builder.gpu_usage_limit_fraction(-0.1);
  REQUIRE_FALSE(builder.build().has_value());
  REQUIRE(copy.build().has_value());
}

TEST_CASE("public byte override replaces a YAML fraction", "[context_config][config]")
{
  scoped_yaml file("sirius:\n  memory:\n    gpu: {usage_limit_fraction: 0.5}\n");
  auto builder = ContextConfigBuilder::from_yaml(file.path);
  REQUIRE(builder.has_value());
  auto config = builder->gpu_usage_limit_bytes(8ULL << 30).build();
  REQUIRE(config.has_value());
  auto invalid = builder->gpu_usage_limit_fraction(-0.1).build();
  REQUIRE_FALSE(invalid.has_value());
  CHECK(invalid.error().code == ErrorCode::invalid_configuration);
  CHECK(invalid.error().message.find(file.path.string()) != std::string::npos);
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
  SECTION("fraction validation and recovery")
  {
    ContextConfigBuilder builder;
    for (double value : {-0.1,
                         1.1,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
      auto result = builder.gpu_usage_limit_fraction(value).build();
      REQUIRE_FALSE(result.has_value());
      CHECK(result.error().code == ErrorCode::invalid_configuration);
    }
    REQUIRE(builder.gpu_usage_limit_fraction(0.5).build().has_value());
  }
}

TEST_CASE("public GPU overrides conflict with explicit memory spaces", "[context_config][config]")
{
  const std::string text = R"(sirius:
  space:
    gpu: [{device_id: 0, memory_capacity: 8Gi}]
    host: [{numa_id: 0, memory_capacity: 1Gi}]
    disk: [{disk_id: 0, memory_capacity: 1Gi, mount_path: /tmp}]
)";
  scoped_yaml file(text);
  auto builder = ContextConfigBuilder::from_yaml(file.path);
  REQUIRE(builder.has_value());
  auto config = builder->build();
  REQUIRE(config.has_value());
  auto result = builder->gpu_usage_limit_bytes(4ULL << 30).build();
  REQUIRE_FALSE(result.has_value());
  CHECK(result.error().code == ErrorCode::invalid_configuration);
}

TEST_CASE("public GPU fraction endpoints preserve YAML semantics", "[context_config][config]")
{
  for (double fraction : {0.0, 1.0}) {
    auto result = ContextConfigBuilder{}.gpu_usage_limit_fraction(fraction).build();
    REQUIRE(result.has_value());
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
  auto result =
    ContextConfigBuilder{}.gpu_usage_limit_bytes(std::numeric_limits<std::uint64_t>::max()).build();
  REQUIRE(result.has_value());
  scoped_yaml file("sirius:\n  memory:\n    gpu: {usage_limit_bytes: 18446744073709551615}\n");
  auto yaml = ContextConfigBuilder::from_yaml(file.path);
  REQUIRE(yaml.has_value());
  REQUIRE(yaml->build().has_value());
}

TEST_CASE("configuration resolver applies programmatic GPU overrides", "[context_config][config]")
{
  SECTION("byte override replaces YAML fraction and preserves explicit settings")
  {
    sirius::sirius_config config;
    constexpr std::uint64_t capacity = 256ULL << 20;
    config.load_from_node(YAML::Load(R"(sirius:
  topology: {num_gpus: 1}
  memory:
    gpu: {usage_limit_fraction: 0.5, reservation_limit_fraction: 0.7}
  operator_params: {hash_partition_bytes: 32Mi}
)"),
                          sirius::gpu_usage_limit{capacity});

    const auto gpu = gpu_space(config);
    CHECK(gpu.memory_capacity == capacity);
    CHECK(gpu.reservation_limit_fraction == 0.7);
    CHECK(config.get_operator_params().scan_task_batch_size == capacity / 40);
    CHECK(config.get_operator_params().hash_partition_bytes == (32ULL << 20));
  }
  SECTION("fraction override replaces YAML bytes")
  {
    sirius::sirius_config physical;
    physical.load_from_node(
      YAML::Load("sirius: {topology: {num_gpus: 1}, memory: {gpu: {usage_limit_fraction: 1.0}}}"));
    const auto total = gpu_space(physical).memory_capacity;
    REQUIRE(total > 0);

    for (const auto fraction : {0.0, 0.5, 1.0}) {
      CAPTURE(fraction);
      sirius::sirius_config config;
      config.load_from_node(
        YAML::Load("sirius: {topology: {num_gpus: 1}, memory: {gpu: {usage_limit_bytes: 256Mi}}}"),
        sirius::gpu_usage_limit{fraction});
      CHECK(gpu_space(config).memory_capacity == static_cast<std::size_t>(total * fraction));
    }
  }
}
