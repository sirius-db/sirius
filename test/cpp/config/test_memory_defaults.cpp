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

// Defaults and validation of the memory tiers: the host tier's default size (one third of a
// GPU's CPU NUMA node per GPU on it), the spill directory check, and the executor switches.

#include "catch.hpp"
#include "sirius_config.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

cucascade::memory::system_topology_info discover()
{
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                             /*with_runtime_attributes=*/true));
  return discovery.get_topology();
}

std::vector<cucascade::memory::host_memory_space_config> host_spaces(
  const sirius::sirius_config& config)
{
  std::vector<cucascade::memory::host_memory_space_config> hosts;
  for (auto const& space : config.get_memory_space_configs()) {
    if (auto const* host = std::get_if<cucascade::memory::host_memory_space_config>(&space)) {
      hosts.push_back(*host);
    }
  }
  return hosts;
}

sirius::sirius_config resolve(const std::string& yaml,
                              const cucascade::memory::system_topology_info& topology)
{
  return sirius::parsed_sirius_config::from_node(YAML::Load(yaml)).resolve(topology);
}

std::size_t third_of(std::size_t node_bytes)
{
  return static_cast<std::size_t>(static_cast<double>(node_bytes) * (1.0 / 3.0));
}

}  // namespace

TEST_CASE("the default host capacity is one third of the GPU's NUMA node per GPU",
          "[memory_defaults][config]")
{
  auto topology = discover();
  REQUIRE(topology.num_gpus >= 1);
  auto const node     = topology.gpus[0].numa_node;
  auto const capacity = topology.get_numa_memory_capacity(node);
  if (!capacity) { SKIP("the memory capacity of GPU 0's NUMA node is unknown"); }

  SECTION("one GPU on the node")
  {
    auto const hosts = host_spaces(resolve("sirius: {topology: {gpu_ids: [0]}}", topology));
    REQUIRE(hosts.size() == 1);
    CHECK(hosts[0].numa_id == node);
    CHECK(hosts[0].memory_capacity == third_of(*capacity));
  }
  SECTION("two GPUs on the node take two thirds, and a third GPU adds nothing")
  {
    if (topology.num_gpus < 2) { SKIP("needs two GPUs"); }
    // Put every GPU on GPU 0's node, whatever the machine's layout is.
    for (auto& gpu : topology.gpus) {
      gpu.numa_node = node;
    }
    auto const two = host_spaces(resolve("sirius: {topology: {gpu_ids: [0, 1]}}", topology));
    REQUIRE(two.size() == 1);
    CHECK(two[0].memory_capacity == 2 * third_of(*capacity));
    CHECK(two[0].reservation_limit_fraction == 1.0);
    if (topology.num_gpus >= 3) {
      auto const three = host_spaces(resolve("sirius: {topology: {gpu_ids: [0, 1, 2]}}", topology));
      REQUIRE(three.size() == 1);
      CHECK(three[0].memory_capacity == 2 * third_of(*capacity));
    }
    // A reservation limit given in bytes stays that many bytes.
    auto const limited = host_spaces(resolve(
      "sirius: {topology: {gpu_ids: [0, 1]}, memory: {host: {reservation_limit_bytes: 1Gi}}}",
      topology));
    REQUIRE(limited.size() == 1);
    CHECK(limited[0].reservation_limit() == Approx(double(1ULL << 30)).epsilon(1e-6));
  }
  SECTION("an explicit capacity is used as given")
  {
    auto const bytes = host_spaces(resolve(
      "sirius: {topology: {gpu_ids: [0]}, memory: {host: {capacity_bytes: 2Gi}}}", topology));
    REQUIRE(bytes.size() == 1);
    CHECK(bytes[0].memory_capacity == (2ULL << 30));
    auto const fraction = host_spaces(resolve(
      "sirius: {topology: {gpu_ids: [0]}, memory: {host: {capacity_fraction: 0.5}}}", topology));
    REQUIRE(fraction.size() == 1);
    CHECK(fraction[0].memory_capacity ==
          static_cast<std::size_t>(static_cast<double>(*capacity) * 0.5));
  }
  SECTION("prepin_fraction raises the pools pinned at start")
  {
    auto const plain =
      host_spaces(resolve("sirius: {topology: {gpu_ids: [0]}, memory: {host: {capacity_bytes: "
                          "4Gi, initial_number_pools: 4}}}",
                          topology));
    REQUIRE(plain.size() == 1);
    CHECK(plain[0].initial_number_pools == 4);
    auto const pinned =
      host_spaces(resolve("sirius: {topology: {gpu_ids: [0]}, memory: {host: {capacity_bytes: "
                          "4Gi, initial_number_pools: 4, prepin_fraction: 0.5}}}",
                          topology));
    REQUIRE(pinned.size() == 1);
    // 2 GiB in pools of 128 blocks of 1 MiB.
    CHECK(pinned[0].initial_number_pools == 16);
    // It never lowers the count derived from the NUMA node's memory.
    auto const derived = host_spaces(resolve(
      "sirius: {topology: {gpu_ids: [0]}, memory: {host: {capacity_bytes: 4Gi}}}", topology));
    auto const derived_pinned =
      host_spaces(resolve("sirius: {topology: {gpu_ids: [0]}, memory: {host: {capacity_bytes: "
                          "4Gi, prepin_fraction: 0.01}}}",
                          topology));
    REQUIRE(derived.size() == 1);
    REQUIRE(derived_pinned.size() == 1);
    CHECK(derived_pinned[0].initial_number_pools == derived[0].initial_number_pools);
    REQUIRE_THROWS(sirius::parsed_sirius_config::from_node(
      YAML::Load("sirius: {memory: {host: {prepin_fraction: 1.5}}}")));
  }
}

TEST_CASE("a spill directory that cannot take files is rejected", "[memory_defaults][config]")
{
  auto const root = std::filesystem::temp_directory_path() /
                    ("sirius_spill_dir_check_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  CHECK_NOTHROW(sirius::validate_spill_directory(""));
  CHECK_NOTHROW(sirius::validate_spill_directory(root.string()));
  CHECK(std::filesystem::is_empty(root));  // the probe file is gone

  // A missing directory is created.
  auto const missing = root / "missing" / "nested";
  CHECK_NOTHROW(sirius::validate_spill_directory(missing.string()));
  CHECK(std::filesystem::is_directory(missing));

  auto const file = root / "a_file";
  std::ofstream(file) << "x";
  REQUIRE_THROWS_WITH(sirius::validate_spill_directory(file.string()),
                      Catch::Matchers::ContainsSubstring("downgrade_root_dirs") &&
                        Catch::Matchers::ContainsSubstring(file.string()) &&
                        Catch::Matchers::ContainsSubstring("is not a directory"));
  // One that cannot be created is an error that names it.
  auto const below_file = (file / "sub").string();
  REQUIRE_THROWS_WITH(sirius::validate_spill_directory(below_file),
                      Catch::Matchers::ContainsSubstring(below_file) &&
                        Catch::Matchers::ContainsSubstring("cannot be"));

  if (::geteuid() != 0) {
    auto const read_only = root / "read_only";
    std::filesystem::create_directories(read_only);
    std::filesystem::permissions(
      read_only, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    REQUIRE_THROWS_WITH(sirius::validate_spill_directory(read_only.string()),
                        Catch::Matchers::ContainsSubstring("is not writable"));
    REQUIRE_THROWS_WITH(sirius::validate_spill_directory((read_only / "sub").string()),
                        Catch::Matchers::ContainsSubstring("cannot be created"));
    std::filesystem::permissions(read_only, std::filesystem::perms::owner_all);
  }
  std::filesystem::remove_all(root);
}

TEST_CASE("executor memory-wait switches are read from YAML", "[memory_defaults][config]")
{
  auto const topology = discover();
  auto const defaults = resolve("sirius: {topology: {gpu_ids: [0]}}", topology);
  CHECK_FALSE(defaults.get_reservation_wait_policy().wait_on_worker);
  CHECK(defaults.get_reservation_wait_policy().fail_fast_on_full_space_oom);
  CHECK_FALSE(defaults.get_downgrade_executor_config().prioritize_reserve_requests);

  auto const changed = resolve(R"(sirius:
  topology: {gpu_ids: [0]}
  executor:
    pipeline: {num_threads: 2, wait_for_memory_on_worker: true, fail_fast_on_full_gpu_oom: false}
    downgrade: {prioritize_reserve_requests: true}
)",
                               topology);
  CHECK(changed.get_gpu_pipeline_executor_config().num_threads == 2);
  CHECK(changed.get_reservation_wait_policy().wait_on_worker);
  CHECK_FALSE(changed.get_reservation_wait_policy().fail_fast_on_full_space_oom);
  CHECK(changed.get_downgrade_executor_config().prioritize_reserve_requests);

  REQUIRE_THROWS(sirius::parsed_sirius_config::from_node(
    YAML::Load("sirius: {executor: {pipeline: {no_such_key: 1}}}")));
}

TEST_CASE("the GPU downgrade band defaults to 0.95 / 0.9 and a lone trigger keeps its stop below",
          "[memory_defaults][config]")
{
  auto const topology = discover();
  auto gpu_of         = [&](const std::string& yaml) {
    for (auto const& space : resolve(yaml, topology).get_memory_space_configs()) {
      if (auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&space)) {
        return *gpu;
      }
    }
    throw std::runtime_error("no GPU space");
  };
  auto const defaults = gpu_of("sirius: {topology: {gpu_ids: [0]}}");
  CHECK(defaults.downgrade_trigger_fraction == Approx(0.95));
  CHECK(defaults.downgrade_stop_fraction == Approx(0.9));

  // Only the trigger, at the former default: the stop follows it instead of failing validation.
  auto const lone =
    gpu_of("sirius: {topology: {gpu_ids: [0]}, memory: {gpu: {downgrade_trigger_fraction: 0.8}}}");
  CHECK(lone.downgrade_trigger_fraction == Approx(0.8));
  CHECK(lone.downgrade_stop_fraction == Approx(0.8 * 0.9 / 0.95));

  auto const pair = gpu_of(
    "sirius: {topology: {gpu_ids: [0]}, memory: {gpu: "
    "{downgrade_trigger_fraction: 0.8, downgrade_stop_fraction: 0.6}}}");
  CHECK(pair.downgrade_stop_fraction == Approx(0.6));
  // A stop at or above the default trigger still needs its own trigger.
  REQUIRE_THROWS(sirius::parsed_sirius_config::from_node(
    YAML::Load("sirius: {memory: {gpu: {downgrade_stop_fraction: 0.97}}}")));
}
