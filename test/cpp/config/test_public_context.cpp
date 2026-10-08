// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "catch.hpp"
#include "utils/log_test_utils.hpp"

#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/pinned_memory.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <sirius/context/config_builder.hpp>
#include <sirius/context/context.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {

struct config_file {
  explicit config_file(const YAML::Node& root)
  {
    static std::atomic<unsigned> next{0};
    path = std::filesystem::temp_directory_path() /
           ("sirius_context_" + std::to_string(getpid()) + "_" + std::to_string(next++) + ".yaml");
    std::ofstream out(path);
    out << root;
    REQUIRE(out.good());
  }
  ~config_file()
  {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  std::filesystem::path path;
};

sirius::ContextConfig from_yaml(const YAML::Node& root)
{
  config_file file(root);
  auto builder = sirius::ContextConfigBuilder::from_yaml(file.path);
  REQUIRE(builder.has_value());
  auto config = builder->build();
  REQUIRE(config.has_value());
  return *config;
}

void check_default_allocators()
{
  // A successful allocation after teardown catches dangling global resources.
  auto pinned = cudf::get_pinned_memory_resource();
  void* ptr   = pinned.allocate_sync(256, alignof(std::max_align_t));
  REQUIRE(ptr != nullptr);
  pinned.deallocate_sync(ptr, 256, alignof(std::max_align_t));
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count) == cudaSuccess);
  for (int device = 0; device < device_count; ++device) {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device}};
    rmm::device_buffer buffer(
      256, rmm::cuda_stream_default, cudf::get_current_device_resource_ref());
    REQUIRE(buffer.data() != nullptr);
    rmm::cuda_stream_default.synchronize();
  }
}

}  // namespace

TEST_CASE("public context creates and destroys defaults", "[public_context][isolated_context]")
{
  auto config = sirius::ContextConfigBuilder{}.build();
  REQUIRE(config.has_value());
  auto const threshold = cudf::get_allocate_host_as_pinned_threshold();
  {
    auto context = sirius::Context::create(*config);
    INFO((context ? "initialized" : context.error().message));
    REQUIRE(context.has_value());
    REQUIRE(*context != nullptr);
  }
  CHECK(cudf::get_allocate_host_as_pinned_threshold() == threshold);
  check_default_allocators();
}

TEST_CASE("public context can be recreated", "[public_context][isolated_context]")
{
  auto config = sirius::ContextConfigBuilder{}.build();
  REQUIRE(config.has_value());
  for (int iteration = 0; iteration < 2; ++iteration) {
    auto context = sirius::Context::create(*config);
    INFO((context ? "initialized" : context.error().message));
    REQUIRE(context.has_value());
  }
  check_default_allocators();
}

TEST_CASE("public context returns hardware resolution errors and permits retry",
          "[public_context][isolated_context]")
{
  YAML::Node root;
  root["sirius"]["memory"]["gpu"]["usage_limit_bytes"] = std::numeric_limits<std::uint64_t>::max();
  auto failed                                          = sirius::Context::create(from_yaml(root));
  REQUIRE_FALSE(failed.has_value());
  CHECK(failed.error().code == sirius::ErrorCode::context_initialization);
  CHECK(failed.error().message.find("capacity") != std::string::npos);

  auto config = sirius::ContextConfigBuilder{}.build();
  REQUIRE(config.has_value());
  auto retry = sirius::Context::create(*config);
  INFO((retry ? "initialized" : retry.error().message));
  REQUIRE(retry.has_value());
}

TEST_CASE("public context rolls back late initialization failures before retry",
          "[public_context][isolated_context]")
{
  auto root = YAML::LoadFile(
    (std::filesystem::path(__FILE__).parent_path() / "data" / "init_failure_pinned_rollback.yaml")
      .string());
  // Exercise global batch telemetry as well as the pinned/device resources.
  root["sirius"]["telemetry"]["enable_quent"]        = true;
  root["sirius"]["telemetry"]["enable_batch_events"] = true;
  root["sirius"]["telemetry"]["enable_nvtx"]         = false;
  sirius::test::scoped_recording_log_sink logs;
  auto const threshold = cudf::get_allocate_host_as_pinned_threshold();
  auto failed          = sirius::Context::create(from_yaml(root));
  REQUIRE_FALSE(failed.has_value());
  CHECK(failed.error().code == sirius::ErrorCode::context_initialization);
  CHECK(cudf::get_allocate_host_as_pinned_threshold() == threshold);
  check_default_allocators();

  // One reactor fits the same host pool; all other settings remain the same.
  root["sirius"]["executor"]["scan_manager"]["uring_n_reactors"] = 1;
  {
    auto retry = sirius::Context::create(from_yaml(root));
    INFO((retry ? "initialized" : retry.error().message));
    REQUIRE(retry.has_value());
  }
  unsigned installations = 0;
  for (auto const& record : logs.sink().records()) {
    CHECK(record.message.find("already installed") == std::string::npos);
    if (record.message.find("Batch telemetry installed") != std::string::npos) { ++installations; }
  }
  CHECK(installations == 2);
  check_default_allocators();
}
