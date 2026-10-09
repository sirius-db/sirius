// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/error_internal.hpp"
#include "catch.hpp"

#include <sirius/c/context/config_builder.h>
#include <sirius/c/version.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using builder_owner =
  std::unique_ptr<sirius_context_config_builder, decltype(&sirius_context_config_builder_release)>;
using config_owner =
  std::unique_ptr<sirius_context_config, decltype(&sirius_context_config_release)>;
using error_owner = std::unique_ptr<sirius_error, decltype(&sirius_error_destroy)>;

struct yaml_file {
  explicit yaml_file(const std::string& text)
  {
    static std::atomic<unsigned> ordinal{0};
    path =
      std::filesystem::temp_directory_path() / ("sirius_c_config_" + std::to_string(getpid()) +
                                                "_" + std::to_string(ordinal++) + "_\xff.yaml");
    write(text);
  }
  void write(const std::string& text) const
  {
    std::ofstream file(path);
    file << text;
    REQUIRE(file.good());
  }
  ~yaml_file()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  std::filesystem::path path;
};
}  // namespace

TEST_CASE("C configuration owns retained snapshots", "[c_api_config][config]")
{
  CHECK(sirius_abi_version() == SIRIUS_ABI_VERSION);
  sirius_context_config_builder* raw = nullptr;
  REQUIRE(sirius_context_config_builder_create(&raw, nullptr) == SIRIUS_SUCCESS);
  builder_owner builder(raw, sirius_context_config_builder_release);
  REQUIRE(builder);
  sirius_context_config_builder_retain(raw);
  builder_owner copy(raw, sirius_context_config_builder_release);
  builder.reset();
  sirius_context_config* snapshot = nullptr;
  REQUIRE(sirius_context_config_builder_build(copy.get(), &snapshot, nullptr) == SIRIUS_SUCCESS);
  config_owner config(snapshot, sirius_context_config_release);
  REQUIRE(config);
  sirius_context_config_retain(snapshot);
  config_owner retained(snapshot, sirius_context_config_release);
  copy.reset();
  config.reset();
  REQUIRE(retained);
  sirius_context_config_retain(nullptr);
  sirius_context_config_release(nullptr);
  sirius_context_config_builder_retain(nullptr);
  sirius_context_config_builder_release(nullptr);
  sirius_error_destroy(nullptr);
  CHECK(std::string(sirius_error_message(nullptr)).empty());
  CHECK(sirius_error_message_size(nullptr) == 0);
}

TEST_CASE("C configuration builds after its YAML file changes or disappears",
          "[c_api_config][config]")
{
  yaml_file file("sirius: {executor: {pipeline: {num_threads: 3}}}");
  const auto path                    = file.path.string();
  sirius_context_config_builder* raw = nullptr;
  REQUIRE(sirius_context_config_builder_from_yaml(path.data(), path.size(), &raw, nullptr) ==
          SIRIUS_SUCCESS);
  builder_owner builder(raw, sirius_context_config_builder_release);
  file.write("sirius: [");
  for (int i = 0; i < 2; ++i) {
    sirius_context_config* snapshot = nullptr;
    REQUIRE(sirius_context_config_builder_build(builder.get(), &snapshot, nullptr) ==
            SIRIUS_SUCCESS);
    config_owner config(snapshot, sirius_context_config_release);
    REQUIRE(config);
    std::filesystem::remove(file.path);
  }
}

TEST_CASE("C configuration returns categorized diagnostics", "[c_api_config][config]")
{
  yaml_file file("");
  sirius_status expected = SIRIUS_SUCCESS;
  SECTION("missing file")
  {
    std::filesystem::remove(file.path);
    expected = SIRIUS_CONFIGURATION_IO;
  }
  SECTION("directory")
  {
    std::filesystem::remove(file.path);
    std::filesystem::create_directory(file.path);
    expected = SIRIUS_CONFIGURATION_IO;
  }
  SECTION("syntax")
  {
    file.write("sirius: [");
    expected = SIRIUS_MALFORMED_YAML;
  }
  SECTION("settings")
  {
    file.write("sirius: {unknown: 1}");
    expected = SIRIUS_INVALID_CONFIGURATION;
  }
  SECTION("malformed time")
  {
    file.write("sirius: {executor: {scan_manager: {rest: {upkeep_interval_ms: '-ms'}}}}");
    expected = SIRIUS_INVALID_CONFIGURATION;
  }
  const auto path          = file.path.string();
  auto* raw                = reinterpret_cast<sirius_context_config_builder*>(std::uintptr_t{1});
  sirius_error* diagnostic = nullptr;
  auto status =
    sirius_context_config_builder_from_yaml(path.data(), path.size(), &raw, &diagnostic);
  error_owner error(diagnostic, sirius_error_destroy);
  CHECK(status == expected);
  CHECK(raw == nullptr);
  REQUIRE(error);
  const std::string message(sirius_error_message(error.get()),
                            sirius_error_message_size(error.get()));
  CHECK(message.find(path) != std::string::npos);
}

TEST_CASE("C configuration rejects invalid arguments and clears outputs", "[c_api_config][config]")
{
  CHECK(sirius_context_config_builder_create(nullptr, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_context_config_builder_from_yaml("", 0, nullptr, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  auto* builder = reinterpret_cast<sirius_context_config_builder*>(std::uintptr_t{1});
  CHECK(sirius_context_config_builder_from_yaml(nullptr, 0, &builder, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(builder == nullptr);
  builder = reinterpret_cast<sirius_context_config_builder*>(std::uintptr_t{1});
  CHECK(sirius_context_config_builder_from_yaml("a\0b", 3, &builder, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(builder == nullptr);
  auto* config = reinterpret_cast<sirius_context_config*>(std::uintptr_t{1});
  CHECK(sirius_context_config_builder_build(nullptr, &config, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(config == nullptr);
  CHECK(sirius_context_config_builder_build(nullptr, nullptr, nullptr) == SIRIUS_INVALID_ARGUMENT);
  auto* error = reinterpret_cast<sirius_error*>(std::uintptr_t{1});
  REQUIRE(sirius_context_config_builder_create(&builder, &error) == SIRIUS_SUCCESS);
  builder_owner owner(builder, sirius_context_config_builder_release);
  CHECK(error == nullptr);
  CHECK(sirius_context_config_builder_build(builder, nullptr, nullptr) == SIRIUS_INVALID_ARGUMENT);
}

TEST_CASE("C configuration supports concurrent builds from immutable settings",
          "[c_api_config][config]")
{
  sirius_context_config_builder* raw = nullptr;
  REQUIRE(sirius_context_config_builder_create(&raw, nullptr) == SIRIUS_SUCCESS);
  builder_owner builder(raw, sirius_context_config_builder_release);
  std::vector<std::future<bool>> workers;
  for (int i = 0; i < 4; ++i) {
    workers.push_back(std::async(std::launch::async, [raw] {
      sirius_context_config_builder_retain(raw);
      builder_owner owned(raw, sirius_context_config_builder_release);
      for (int j = 0; j < 32; ++j) {
        sirius_context_config* snapshot = nullptr;
        auto status = sirius_context_config_builder_build(owned.get(), &snapshot, nullptr);
        config_owner config(snapshot, sirius_context_config_release);
        if (status != SIRIUS_SUCCESS || !config) return false;
      }
      return true;
    }));
  }
  for (auto& worker : workers)
    CHECK(worker.get());
}

TEST_CASE("C exception boundary preserves statuses without allocating diagnostics",
          "[c_api_config][config]")
{
  auto* error = reinterpret_cast<sirius_error*>(std::uintptr_t{1});
  CHECK(sirius::c_api::invoke(&error, []() -> sirius_status { throw std::bad_alloc{}; }) ==
        SIRIUS_ALLOCATION_FAILURE);
  CHECK(error == nullptr);
  error = reinterpret_cast<sirius_error*>(std::uintptr_t{1});
  CHECK(sirius::c_api::with_diagnostic(SIRIUS_CONFIGURATION_IO, &error, []() -> sirius_error* {
          throw std::bad_alloc{};
        }) == SIRIUS_CONFIGURATION_IO);
  CHECK(error == nullptr);
  CHECK(sirius::c_api::invoke(nullptr, []() -> sirius_status {
          throw std::runtime_error("failure");
        }) == SIRIUS_INTERNAL_ERROR);
  CHECK(sirius::c_api::invoke(&error, []() -> sirius_status {
          throw std::runtime_error("failure");
        }) == SIRIUS_INTERNAL_ERROR);
  error_owner diagnostic(error, sirius_error_destroy);
  REQUIRE(diagnostic);
  CHECK(std::string(sirius_error_message(diagnostic.get())) == "failure");
  CHECK(sirius::c_api::invoke(nullptr, []() -> sirius_status { throw 1; }) ==
        SIRIUS_INTERNAL_ERROR);
}
