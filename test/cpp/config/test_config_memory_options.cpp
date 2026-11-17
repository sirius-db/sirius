// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/memory.hpp"
#include "config/options/space.hpp"
#include "config/options/topology.hpp"
#include "config/yaml.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;

namespace {

struct memory_root : block_base<memory_root> {
  topology_options topology;
  memory_options memory;
  space_options space;
  auto fields()
  {
    return field_list(sub("topology", topology), sub("memory", memory), sub("space", space));
  }
};

memory_root read(std::string_view text)
{
  memory_root root;
  read_yaml(root, text, "sirius", environment::prod);
  return root;
}

std::string read_error(std::string_view text)
{
  memory_root root;
  try {
    read_yaml(root, text, "sirius", environment::prod);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("memory blocks declare their keys once", "[config][config_memory]")
{
  memory_root{}.check_declaration();
}

TEST_CASE("absent memory blocks take today's defaults", "[config][config_memory]")
{
  auto const root = read("sirius: {}\n");
  REQUIRE(root.topology.gpus.get<std::size_t>() == 0);
  REQUIRE(root.topology.gpus_per_query.get() == 0);

  auto const& gpu = root.memory.gpu;
  REQUIRE(gpu.usage_limit.get<double>() == 0.95);
  REQUIRE(gpu.reservation_limit.get<double>() == 1.0);
  REQUIRE(gpu.downgrade_trigger_fraction.get() == 0.8);
  REQUIRE(gpu.downgrade_stop_fraction.get() == 0.6);

  auto const& host = root.memory.host;
  REQUIRE(host.capacity.get<double>() == 0.9);
  REQUIRE(host.reservation_limit.get<double>() == 1.0);
  REQUIRE(host.downgrade_trigger_fraction.get() == 0.9);
  REQUIRE(host.downgrade_stop_fraction.get() == 0.8);
  REQUIRE(host.block_size.get() == 1_MiB);
  REQUIRE(host.pool_size.get() == 128);
  REQUIRE(host.initial_number_pools.is_unset());  // derived from NUMA memory at resolve

  auto const& disk = root.memory.disk;
  REQUIRE(disk.disk_id.get() == 0);
  REQUIRE(disk.capacity_bytes.get() == 1_TiB);
  REQUIRE(disk.downgrade_root_dirs.get().empty());

  REQUIRE_FALSE(root.memory.configured());
  REQUIRE_FALSE(root.space.configured());
}

TEST_CASE("topology selects GPUs by count or by id", "[config][config_memory]")
{
  REQUIRE(read("sirius:\n  topology:\n    num_gpus: 2\n").topology.gpus.get<std::size_t>() == 2);
  REQUIRE(
    read("sirius:\n  topology:\n    gpu_ids: [1, 0]\n").topology.gpus.get<std::vector<int>>() ==
    std::vector<int>{1, 0});
  REQUIRE(read_error("sirius:\n  topology:\n    num_gpus: 1\n    gpu_ids: [0]\n") ==
          "'sirius.topology': 'num_gpus' and 'gpu_ids' are mutually exclusive");
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: []\n") ==
          "'sirius.topology.gpu_ids': must contain at least one device id");
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: [0, -1]\n") ==
          "'sirius.topology.gpu_ids': must contain only non-negative device ids");
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: [2, 0, 2]\n") ==
          "'sirius.topology.gpu_ids': must not contain duplicate device ids");
  REQUIRE(read_error("sirius:\n  topology:\n    num_gpus: -1\n") ==
          "'sirius.topology.num_gpus': must be a nonnegative integer, got '-1'");
  REQUIRE(read_error("sirius:\n  topology:\n    gpus_per_query: -1\n") ==
          "'sirius.topology.gpus_per_query': value out of range: '-1' (must not be negative)");
}

TEST_CASE("GPU memory takes bytes or a fraction", "[config][config_memory]")
{
  auto const root = read(
    "sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: 8 GiB\n"
    "      reservation_limit_fraction: 0.5\n");
  REQUIRE(root.memory.gpu.usage_limit.get<bytes>() == 8_GiB);
  REQUIRE(root.memory.gpu.usage_limit.user_defined());
  REQUIRE(root.memory.gpu.reservation_limit.get<double>() == 0.5);
  REQUIRE(root.memory.configured());

  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: 1 GiB\n"
                     "      usage_limit_fraction: 0.5\n") ==
          "'sirius.memory.gpu': 'usage_limit_bytes' and 'usage_limit_fraction' are mutually "
          "exclusive");
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      usage_limit_fraction: 1.5\n") ==
          "'sirius.memory.gpu.usage_limit_fraction': value out of range: '1.5' (must be between "
          "0.0 and 1.0)");
  // Zero usage is allowed (it is special-cased); a zero reservation share is not.
  REQUIRE(read("sirius:\n  memory:\n    gpu:\n      usage_limit_fraction: 0\n")
            .memory.gpu.usage_limit.get<double>() == 0.0);
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      reservation_limit_fraction: 0\n") ==
          "'sirius.memory.gpu.reservation_limit_fraction': value out of range: '0.0' (must be "
          "greater than 0.0 and at most 1.0)");
}

TEST_CASE("downgrade fractions keep 0 < stop < trigger", "[config][config_memory]")
{
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      downgrade_stop_fraction: 0.9\n") ==
          "'sirius.memory.gpu': downgrade_stop_fraction must be less than "
          "downgrade_trigger_fraction");
  REQUIRE(read_error("sirius:\n  memory:\n    host:\n      downgrade_stop_fraction: 0\n") ==
          "'sirius.memory.host': downgrade_stop_fraction must be greater than zero");
  REQUIRE(read_error("sirius:\n  space:\n    gpu:\n      - downgrade_trigger_fraction: 0.5\n") ==
          "'sirius.space.gpu[0]': downgrade_stop_fraction must be less than "
          "downgrade_trigger_fraction");
}

TEST_CASE("host memory rejects values cuCascade cannot use", "[config][config_memory]")
{
  for (auto const* key : {"capacity_bytes: 0",
                          "capacity_fraction: 0",
                          "block_size: 0",
                          "pool_size: 0",
                          "reservation_limit_fraction: 0"}) {
    INFO(key);
    REQUIRE_THAT(read_error(std::string("sirius:\n  memory:\n    host:\n      ") + key + "\n"),
                 Catch::Matchers::ContainsSubstring("value out of range"));
  }
  REQUIRE(read_error("sirius:\n  memory:\n    host:\n      pool_size: -1\n") ==
          "'sirius.memory.host.pool_size': must be a nonnegative integer, got '-1'");

  auto const root = read(
    "sirius:\n  memory:\n    host:\n      capacity_bytes: 16 GiB\n"
    "      block_size: 2 MiB\n      initial_number_pools: 3\n");
  REQUIRE(root.memory.host.capacity.get<bytes>() == 16_GiB);
  REQUIRE(root.memory.host.block_size.get() == 2_MiB);
  REQUIRE(root.memory.host.initial_number_pools.get() == 3);
}

TEST_CASE("an empty tier block counts as configured", "[config][config_memory]")
{
  auto const root = read("sirius:\n  memory:\n    disk: {}\n");
  REQUIRE(root.memory.configured());
  REQUIRE(root.memory.disk.user_defined());
  REQUIRE_FALSE(root.memory.gpu.user_defined());
}

TEST_CASE("space lists describe memory spaces one by one", "[config][config_memory]")
{
  auto const root = read(R"(sirius:
  space:
    gpu:
      - device_id: 0
        memory_capacity: 20 GiB
      - device_id: 1
        per_stream_reservation: true
    host:
      - numa_id: 0
        memory_capacity: 64 GiB
    disk:
      - disk_id: 0
        mount_path: /scratch
        memory_capacity: 1 TiB
)");
  REQUIRE(root.space.configured());
  REQUIRE(root.space.gpu.size() == 2);
  REQUIRE(root.space.gpu[0].memory_capacity.get() == 20_GiB);
  REQUIRE_FALSE(root.space.gpu[0].per_stream_reservation.get());  // Sirius's default
  REQUIRE(root.space.gpu[1].per_stream_reservation.get());
  REQUIRE(root.space.gpu[1].memory_capacity.get() == 0_B);
  REQUIRE(root.space.gpu[1].reservation_limit_fraction.get() == 0.9);  // cuCascade's defaults
  REQUIRE(root.space.host.at(0).initial_number_pools.get() ==
          cucascade::memory::default_initial_number_pools);
  REQUIRE(root.space.disk.at(0).mount_path.get() == "/scratch");

  REQUIRE(read_error("sirius:\n  space:\n    gpu: {device_id: 0}\n") ==
          "'sirius.space.gpu': expected a sequence, got a mapping");
}

TEST_CASE("memory blocks write what they read", "[config][config_memory]")
{
  constexpr std::string_view text = R"(sirius:
  topology:
    gpu_ids: [0, 2]
    gpus_per_query: 1
  memory:
    gpu:
      usage_limit_bytes: 1536 MiB
    host:
      capacity_fraction: 0.5
      pool_size: 64
    disk:
      downgrade_root_dirs: /spill
)";
  auto const root                 = read(text);
  REQUIRE(parse_yaml(write_yaml(root, "sirius")) == parse_yaml(text));

  auto const all = read(write_yaml(root, "sirius", true));
  REQUIRE(all.memory.gpu.downgrade_trigger_fraction.user_defined());
  REQUIRE(all.memory.host.block_size.get() == 1_MiB);
  REQUIRE(all.topology.gpus.get<std::vector<int>>() == std::vector<int>{0, 2});
}
