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
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

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
  // REST reactors are clamp(T/6, 8, 16) without kTLS and clamp(T/6, 4, 16) with it, unless the
  // pool is sized from fewer than 16 NIC IRQ CPUs.
  struct example {
    unsigned int cpu;
    std::size_t gpus;
    sirius::exec::hardware_thread_defaults expected;
    std::size_t rest_with_ktls;
  };
  for (auto const& row : {
         example{0, 0, {4, 4, 1, 18, 8, 2}, 4},
         example{20, 1, {7, 4, 2, 18, 8, 2}, 4},
         example{31, 1, {8, 4, 3, 18, 8, 2}, 5},
         example{32, 1, {8, 4, 4, 18, 8, 4}, 5},
         example{39, 2, {7, 4, 2, 18, 8, 4}, 6},
         example{64, 8, {4, 4, 1, 18, 10, 4}, 10},
         example{96, 4, {8, 4, 3, 18, 16, 4}, 16},
         example{192, 8, {8, 4, 3, 18, 16, 4}, 16},
         example{std::numeric_limits<unsigned int>::max(), 1, {8, 4, 4, 18, 16, 4}, 16},
         example{32, std::numeric_limits<std::size_t>::max(), {4, 4, 1, 18, 8, 4}, 5},
       }) {
    CAPTURE(row.cpu, row.gpus);
    auto const check_non_rest = [&](sirius::exec::hardware_thread_defaults const& actual) {
      CHECK(actual.pipeline == row.expected.pipeline);
      CHECK(actual.task_creator == row.expected.task_creator);
      CHECK(actual.downgrade == row.expected.downgrade);
      CHECK(actual.scan_manager == row.expected.scan_manager);
      CHECK(actual.uring_reactors == row.expected.uring_reactors);
    };
    auto const actual = sirius::exec::derive_thread_defaults(row.cpu, row.gpus);
    check_non_rest(actual);
    CHECK(actual.rest_reactors == row.expected.rest_reactors);

    // Kernel TLS lowers the floor to 4 and changes nothing else.
    auto const with_ktls = sirius::exec::derive_thread_defaults(row.cpu, row.gpus, true);
    check_non_rest(with_ktls);
    CHECK(with_ktls.rest_reactors == row.rest_with_ktls);

    // Fewer than 16 NIC IRQ CPUs: one reactor per IRQ CPU, still at least the floor.
    for (auto const [irq_cpus, expected] :
         {std::pair<std::size_t, std::size_t>{1, 4}, {4, 4}, {6, 6}, {15, 15}}) {
      CAPTURE(irq_cpus);
      auto const from_nic = sirius::exec::derive_thread_defaults(row.cpu, row.gpus, true, irq_cpus);
      check_non_rest(from_nic);
      CHECK(from_nic.rest_reactors == expected);
    }
    // 16 or more IRQ CPUs: the CPU-count default.
    for (std::size_t const irq_cpus : {16, 17, 64}) {
      CAPTURE(irq_cpus);
      CHECK(sirius::exec::derive_thread_defaults(row.cpu, row.gpus, true, irq_cpus).rest_reactors ==
            row.rest_with_ktls);
    }
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

namespace {

// A NIC whose queue IRQs are each delivered to one CPU: steered iff those CPUs are a strict
// subset of the node's.
cucascade::memory::nic_topology make_nic(std::vector<int> irq_cpus, std::vector<int> node_cpus)
{
  cucascade::memory::nic_topology nic;
  nic.interface = "eth0";
  nic.node_cpus = std::move(node_cpus);
  nic.irq_cpus  = std::move(irq_cpus);
  int queue     = 0;
  for (int const cpu : nic.irq_cpus) {
    nic.queue_irqs.push_back(
      {.irq = 100 + queue, .queue = queue, .cpus = {cpu}, .requested_cpus = {cpu}});
    ++queue;
  }
  return nic;
}

}  // namespace

TEST_CASE("REST reactors are sized from the NIC only with kTLS and steered IRQs",
          "[context_config][config][thread_defaults]")
{
  cucascade::memory::kernel_tls_info loaded;
  loaded.loaded = true;
  cucascade::memory::kernel_tls_info const not_loaded{};
  auto const steered = make_nic({2, 3, 4, 5}, {0, 1, 2, 3, 4, 5, 6, 7});

  CHECK(sirius::rest_reactors_sized_from_nic(loaded, steered));
  // No kTLS, or no NIC information, or IRQs that are not steered: the CPU-count default.
  CHECK_FALSE(sirius::rest_reactors_sized_from_nic(not_loaded, steered));
  CHECK_FALSE(sirius::rest_reactors_sized_from_nic(loaded, std::nullopt));
  // IRQ CPUs that cover the whole node leave no subset of cores to align with.
  CHECK_FALSE(sirius::rest_reactors_sized_from_nic(loaded, make_nic({0, 1, 2, 3}, {0, 1, 2, 3})));
  // No queue IRQs at all (a virtual interface).
  CHECK_FALSE(sirius::rest_reactors_sized_from_nic(loaded, make_nic({}, {0, 1, 2, 3})));
  // An IRQ spread over several CPUs is not steered.
  auto spread                         = make_nic({2, 3}, {0, 1, 2, 3, 4, 5, 6, 7});
  spread.queue_irqs[0].cpus           = {2, 3};
  spread.queue_irqs[0].requested_cpus = {2, 3};
  CHECK_FALSE(sirius::rest_reactors_sized_from_nic(loaded, spread));
}

TEST_CASE("rest.min_mtu defaults to the NIC's MTU only for a jumbo-frame NIC",
          "[context_config][config][thread_defaults]")
{
  auto const with_mtu = [](int mtu) {
    auto nic = make_nic({2, 3}, {0, 1, 2, 3, 4, 5, 6, 7});
    nic.mtu  = mtu;
    return nic;
  };
  CHECK(sirius::default_rest_min_mtu(with_mtu(9000)) == 9000);
  // The NIC's own value, not a fixed jumbo size.
  CHECK(sirius::default_rest_min_mtu(with_mtu(9001)) == 9001);
  CHECK(sirius::default_rest_min_mtu(with_mtu(8500)) == 8500);
  CHECK(sirius::default_rest_min_mtu(with_mtu(1501)) == 1501);
  // Standard Ethernet and unknown MTUs, and no NIC information: no check.
  CHECK(sirius::default_rest_min_mtu(with_mtu(1500)) == 0);
  CHECK(sirius::default_rest_min_mtu(with_mtu(1280)) == 0);
  CHECK(sirius::default_rest_min_mtu(with_mtu(0)) == 0);
  CHECK(sirius::default_rest_min_mtu(std::nullopt) == 0);
}

TEST_CASE("resolved REST defaults follow kTLS, NIC IRQ steering and the NIC's MTU",
          "[context_config][config][thread_defaults]")
{
  struct named_keys {
    std::optional<int> rest_n_reactors;
    std::optional<int> min_mtu;
    std::string interface;
    std::optional<int> max_connections;
    std::optional<int> queue_depth;
  };
  auto const make_root = [](named_keys const& named) {
    auto root         = YAML::Load(R"(sirius:
  space:
    gpu: [{device_id: 0, memory_capacity: 512MiB}]
  executor:
    scan_manager: {reserve_rest_reactor_cpus: false}
)");
    auto scan_manager = root["sirius"]["executor"]["scan_manager"];
    if (named.rest_n_reactors) { scan_manager["rest_n_reactors"] = *named.rest_n_reactors; }
    if (named.min_mtu) { scan_manager["rest"]["min_mtu"] = *named.min_mtu; }
    if (!named.interface.empty()) { scan_manager["rest"]["interface"] = named.interface; }
    if (named.max_connections) { scan_manager["rest"]["max_connections"] = *named.max_connections; }
    if (named.queue_depth) { scan_manager["rest"]["queue_depth"] = *named.queue_depth; }
    return root;
  };
  cucascade::memory::system_topology_info with_ktls{};
  with_ktls.kernel_tls.loaded = true;
  cucascade::memory::system_topology_info const without_ktls{};

  int probes = 0;
  std::string probed_interface;
  std::optional<cucascade::memory::nic_topology> nic;
  sirius::egress_nic_probe const probe = [&](std::string_view interface) {
    ++probes;
    probed_interface = interface;
    return nic;
  };
  auto const resolve = [&](named_keys const& named, auto const& topology) {
    return sirius::parsed_sirius_config::from_node(make_root(named))
      .resolve(topology, probe)
      .get_scan_manager_config();
  };
  auto const cpu_range = [](int first, int count) {
    std::vector<int> cpus(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      cpus[static_cast<std::size_t>(i)] = first + i;
    }
    return cpus;
  };
  auto const hw = std::thread::hardware_concurrency();
  // The CPU-count defaults: clamp(T/6, 8, 16) without kTLS, clamp(T/6, 4, 16) with it.
  auto const fallback = sirius::exec::derive_thread_defaults(hw, 1).rest_reactors;
  REQUIRE(fallback >= 8);
  REQUIRE(fallback <= 16);
  auto const fallback_ktls = sirius::exec::derive_thread_defaults(hw, 1, true).rest_reactors;
  REQUIRE(fallback_ktls >= 4);
  REQUIRE(fallback_ktls <= 16);
  // Six queue IRQs steered to CPUs 2-7 of a 16-CPU node.
  auto const steered = make_nic(cpu_range(2, 6), cpu_range(0, 16));

  SECTION("kTLS loaded and IRQs steered to fewer than 16 CPUs: one reactor per IRQ CPU")
  {
    nic = steered;
    CHECK(resolve({}, with_ktls).rest_n_reactors == 6);
    // One lookup serves both the reactor count and rest.min_mtu.
    CHECK(probes == 1);
    CHECK(probed_interface.empty());
  }
  SECTION("a NIC-sized pool keeps the kTLS floor of 4 reactors")
  {
    nic = make_nic({2, 3}, cpu_range(0, 16));
    CHECK(resolve({}, with_ktls).rest_n_reactors == 4);
  }
  SECTION("15 IRQ CPUs map to 15 reactors; 16 or more fall back to the CPU-count default")
  {
    nic = make_nic(cpu_range(0, 15), cpu_range(0, 64));
    CHECK(resolve({}, with_ktls).rest_n_reactors == 15);
    nic = make_nic(cpu_range(0, 16), cpu_range(0, 64));
    CHECK(resolve({}, with_ktls).rest_n_reactors == fallback_ktls);
    nic = make_nic(cpu_range(0, 32), cpu_range(0, 64));
    CHECK(resolve({}, with_ktls).rest_n_reactors == fallback_ktls);
  }
  SECTION("the configured interface is the one probed")
  {
    nic = steered;
    CHECK(resolve({.interface = "enp1s0"}, with_ktls).rest_n_reactors == 6);
    CHECK(probed_interface == "enp1s0");
  }
  SECTION("kTLS loaded but IRQs not steered, or no NIC: clamp(T/6, 4, 16)")
  {
    nic = make_nic({0, 1, 2, 3}, {0, 1, 2, 3});
    CHECK(resolve({}, with_ktls).rest_n_reactors == fallback_ktls);
    nic.reset();
    CHECK(resolve({}, with_ktls).rest_n_reactors == fallback_ktls);
  }
  SECTION("no kTLS: clamp(T/6, 8, 16), whatever the NIC")
  {
    nic = steered;
    CHECK(resolve({}, without_ktls).rest_n_reactors == fallback);
    // Nothing needs the NIC once rest.min_mtu is named too.
    CHECK(probes == 1);
    probes = 0;
    CHECK(resolve({.min_mtu = 0}, without_ktls).rest_n_reactors == fallback);
    CHECK(probes == 0);
  }
  SECTION("rest.max_connections is 128 with kTLS and 64 without")
  {
    nic = steered;
    CHECK(resolve({}, with_ktls).rest.max_connections == 128);
    CHECK(resolve({}, without_ktls).rest.max_connections == 64);
  }
  SECTION("a named rest.max_connections wins")
  {
    for (int const named : {7, 64, 128, 200}) {
      CAPTURE(named);
      auto const expected = static_cast<std::size_t>(named);
      CHECK(resolve({.max_connections = named}, with_ktls).rest.max_connections == expected);
      CHECK(resolve({.max_connections = named}, without_ktls).rest.max_connections == expected);
    }
  }
  SECTION("the derived rest.max_connections fits a named rest.queue_depth")
  {
    // 128 connections need queue_depth >= 256; a smaller named depth caps the default at half.
    CHECK(resolve({.queue_depth = 200}, with_ktls).rest.max_connections == 100);
    CHECK(resolve({.queue_depth = 200}, without_ktls).rest.max_connections == 64);
    CHECK(resolve({.queue_depth = 256}, with_ktls).rest.max_connections == 128);
  }
  SECTION("a named reactor count wins")
  {
    nic = steered;
    for (int const named : {0, 5}) {
      CAPTURE(named);
      auto const expected = static_cast<std::size_t>(named);
      CHECK(resolve({.rest_n_reactors = named, .min_mtu = 0}, with_ktls).rest_n_reactors ==
            expected);
      CHECK(resolve({.rest_n_reactors = named, .min_mtu = 0}, without_ktls).rest_n_reactors ==
            expected);
    }
    CHECK(probes == 0);
  }
  SECTION("rest.min_mtu is the NIC's MTU for a jumbo-frame NIC, whatever kTLS says")
  {
    nic      = steered;
    nic->mtu = 9000;
    CHECK(resolve({}, with_ktls).rest.min_mtu == 9000);
    CHECK(resolve({}, without_ktls).rest.min_mtu == 9000);
  }
  SECTION("rest.min_mtu is 0 for a standard-MTU NIC or without NIC information")
  {
    nic      = steered;
    nic->mtu = 1500;
    CHECK(resolve({}, with_ktls).rest.min_mtu == 0);
    nic.reset();
    CHECK(resolve({}, with_ktls).rest.min_mtu == 0);
  }
  SECTION("a named rest.min_mtu wins, an explicit 0 included, and needs no NIC")
  {
    nic      = steered;
    nic->mtu = 9000;
    CHECK(resolve({.min_mtu = 0}, without_ktls).rest.min_mtu == 0);
    CHECK(resolve({.min_mtu = 8000}, without_ktls).rest.min_mtu == 8000);
    CHECK(probes == 0);
  }
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
    // The hash-table cap, 8x the 32 MiB partition, is above four scan batches.
    CHECK(config.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu() ==
          8 * (32ULL << 20));
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

TEST_CASE("dynamic-filter Bloom budget follows resolved scan size and hash-table cap",
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
    auto const scan_term           = scan_bytes > max_bytes / 4 ? max_bytes : 4 * scan_bytes;
    auto const expected =
      std::max(scan_term, resolved.get_operator_params().max_build_hash_table_bytes);
    CHECK(resolved.get_operator_params().max_dynamic_filter_bloom_bytes_per_gpu() == expected);

    // Resolution is repeatable and does not mutate the parsed settings.
    CHECK(parsed.resolve(discovery.get_topology())
            .get_operator_params()
            .max_dynamic_filter_bloom_bytes_per_gpu() == expected);
  }
  params["max_dynamic_filter_bloom_bytes_per_gpu"] = "17MiB";
  CHECK_THROWS_WITH(sirius::parsed_sirius_config::from_node(root),
                    Catch::Matchers::ContainsSubstring(
                      "derived from scan_task_batch_size and max_build_hash_table_bytes"));
}
