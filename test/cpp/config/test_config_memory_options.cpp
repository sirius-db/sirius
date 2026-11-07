// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/memory.hpp"
#include "config/options/space.hpp"
#include "config/options/topology.hpp"
#include "config/yaml.hpp"

#include <cucascade/memory/config.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
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
  read_yaml(root, text, "sirius");
  return root;
}

std::string read_error(std::string_view text)
{
  memory_root root;
  try {
    read_yaml(root, text, "sirius");
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
          "'sirius.topology': 'num_gpus' and 'gpu_ids' are mutually exclusive; GPUs Sirius "
          "uses, by count or by device id");
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: []\n") ==
          "'sirius.topology.gpu_ids': must contain at least one device id"
          "; GPUs Sirius uses, by count or by device id");
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: [0, -1]\n") ==
          "'sirius.topology.gpu_ids': must contain only non-negative device ids"
          "; GPUs Sirius uses, by count or by device id");
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: [2, 0, 2]\n") ==
          "'sirius.topology.gpu_ids': must not contain duplicate device ids"
          "; GPUs Sirius uses, by count or by device id");
  REQUIRE(read_error("sirius:\n  topology:\n    num_gpus: -1\n") ==
          "'sirius.topology.num_gpus': must be a nonnegative integer, got '-1'"
          "; GPUs Sirius uses, by count or by device id");
  REQUIRE(read_error("sirius:\n  topology:\n    gpus_per_query: -1\n") ==
          "'sirius.topology.gpus_per_query': value out of range: '-1' (must not be negative)"
          "; GPUs allocated to each query at admission");
}

TEST_CASE("GPU memory takes bytes or a fraction", "[config][config_memory]")
{
  auto const root = read(
    "sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: 8 GiB\n"
    "      reservation_limit_fraction: 0.5\n");
  REQUIRE(root.memory.gpu.usage_limit.get<ubytes>() == 8_GiB);
  REQUIRE(root.memory.gpu.usage_limit.user_defined());
  REQUIRE(root.memory.gpu.reservation_limit.get<double>() == 0.5);
  REQUIRE(root.memory.configured());

  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: 1 GiB\n"
                     "      usage_limit_fraction: 0.5\n") ==
          "'sirius.memory.gpu': 'usage_limit_bytes' and 'usage_limit_fraction' are mutually "
          "exclusive; GPU memory capacity Sirius may use on each GPU");
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      usage_limit_fraction: 1.5\n") ==
          "'sirius.memory.gpu.usage_limit_fraction': value out of range: '1.5' (must be between "
          "0.0 and 1.0)"
          "; GPU memory capacity Sirius may use on each GPU");
  // Zero usage is allowed (it is special-cased); a zero reservation share is not.
  REQUIRE(read("sirius:\n  memory:\n    gpu:\n      usage_limit_fraction: 0\n")
            .memory.gpu.usage_limit.get<double>() == 0.0);
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      reservation_limit_fraction: 0\n") ==
          "'sirius.memory.gpu.reservation_limit_fraction': value out of range: '0.0' (must be "
          "greater than 0.0 and at most 1.0)"
          "; part of the GPU capacity that reservations may take");
}

TEST_CASE("downgrade fractions keep 0 < stop < trigger", "[config][config_memory]")
{
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      downgrade_stop_fraction: 0.9\n") ==
          "'sirius.memory.gpu.downgrade_stop_fraction': must be less than "
          "downgrade_trigger_fraction");
  REQUIRE(read_error("sirius:\n  memory:\n    host:\n      downgrade_stop_fraction: 0\n") ==
          "'sirius.memory.host.downgrade_stop_fraction': must be greater than zero");
  REQUIRE(read_error("sirius:\n  space:\n    gpu:\n      - downgrade_trigger_fraction: 0.5\n") ==
          "'sirius.space.gpu[0].downgrade_stop_fraction': must be less than "
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
          "'sirius.memory.host.pool_size': must be a nonnegative integer, got '-1'; allocation "
          "blocks per pinned host pool");

  auto const root = read(
    "sirius:\n  memory:\n    host:\n      capacity_bytes: 16 GiB\n"
    "      block_size: 2 MiB\n      initial_number_pools: 3\n");
  REQUIRE(root.memory.host.capacity.get<ubytes>() == 16_GiB);
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

TEST_CASE("memory blocks keep upstream's unknown keys and pairs", "[config][config_memory]")
{
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      track_per_stream_reservation: true\n") ==
          "unknown config key: 'track_per_stream_reservation' in sirius.memory.gpu");
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      per_stream_reservation: true\n") ==
          "unknown config key: 'per_stream_reservation' in sirius.memory.gpu");
  for (auto const* pair : {"gpu:\n      reservation_limit_bytes: 1 GiB\n      "
                           "reservation_limit_fraction: 0.5",
                           "host:\n      reservation_limit_bytes: 1 GiB\n      "
                           "reservation_limit_fraction: 0.5",
                           "host:\n      capacity_bytes: 1 GiB\n      capacity_fraction: 0.5"}) {
    INFO(pair);
    REQUIRE_THAT(read_error(std::string("sirius:\n  memory:\n    ") + pair + "\n"),
                 Catch::Matchers::ContainsSubstring("are mutually exclusive"));
  }
  // A null alternative is absent.
  auto const root = read(
    "sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: ~\n      usage_limit_fraction: 0.5\n");
  REQUIRE(root.memory.gpu.usage_limit.get<double>() == 0.5);
}

TEST_CASE("which memory blocks count as configured", "[config][config_memory]")
{
  REQUIRE_FALSE(read("sirius:\n  memory:\n    gpu: ~\n").memory.configured());
  auto const null_leaf = read("sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: ~\n");
  REQUIRE(null_leaf.memory.configured());
  REQUIRE_FALSE(null_leaf.memory.gpu.usage_limit.user_defined());  // not an explicit capacity
  REQUIRE_FALSE(read("sirius:\n  space:\n    gpu: []\n").space.configured());

  REQUIRE_FALSE(read("sirius: {}\n").space.gpu_capacities_configured());
  REQUIRE(read("sirius:\n  space:\n    gpu:\n      - memory_capacity: 1 GiB\n")
            .space.gpu_capacities_configured());
  REQUIRE_FALSE(read("sirius:\n  space:\n    gpu:\n      - memory_capacity: 1 GiB\n"
                     "      - memory_capacity: 0\n")
                  .space.gpu_capacities_configured());
}

TEST_CASE("space entries default to cuCascade's values", "[config][config_memory]")
{
  auto const root = read("sirius:\n  space:\n    gpu: [{}]\n    host: [{}]\n    disk: [{}]\n");
  auto const& gpu = root.space.gpu.at(0);
  REQUIRE(gpu.device_id.get() == -1);
  REQUIRE(gpu.downgrade_trigger_fraction.get() == 0.75);
  REQUIRE(gpu.downgrade_stop_fraction.get() == 0.65);
  REQUIRE(gpu.memory_capacity.get() == 0_B);
  auto const& host = root.space.host.at(0);
  REQUIRE(host.numa_id.get() == -1);
  REQUIRE(host.block_size.get() == 1_MiB);
  REQUIRE(host.pool_size.get() == cucascade::memory::default_pool_size);
  auto const& disk = root.space.disk.at(0);
  REQUIRE(disk.disk_id.get() == -1);
  REQUIRE(disk.mount_path.get().empty());
  REQUIRE(disk.memory_capacity.get() == 0_B);
}

TEST_CASE("the space path keeps today's ranges", "[config][config_memory]")
{
  // The zero checks agreed for sirius.memory do not apply to sirius.space entries.
  auto const root = read(R"(sirius:
  space:
    gpu:
      - reservation_limit_fraction: 0
    host:
      - reservation_limit_fraction: 0
        block_size: 0
        pool_size: 0
)");
  REQUIRE(root.space.gpu.at(0).reservation_limit_fraction.get() == 0.0);
  REQUIRE(root.space.host.at(0).block_size.get() == 0_B);
  REQUIRE(root.space.host.at(0).pool_size.get() == 0);
}

TEST_CASE("fraction boundaries", "[config][config_memory]")
{
  REQUIRE(read("sirius:\n  memory:\n    gpu:\n      usage_limit_fraction: 1.0\n")
            .memory.gpu.usage_limit.get<double>() == 1.0);
  REQUIRE(read("sirius:\n  memory:\n    host:\n      capacity_fraction: 1\n")
            .memory.host.capacity.get<double>() == 1.0);
  for (auto const* value : {"1.0000001", ".nan", ".inf", "-0.1"}) {
    INFO(value);
    REQUIRE_THAT(
      read_error(std::string("sirius:\n  memory:\n    gpu:\n      usage_limit_fraction: ") + value +
                 "\n"),
      Catch::Matchers::ContainsSubstring("value out of range"));
  }
  REQUIRE(read_error("sirius:\n  memory:\n    host:\n      capacity_fraction: 1.5\n") ==
          "'sirius.memory.host.capacity_fraction': value out of range: '1.5' (must be greater than "
          "0.0 and at most 1.0); pinned host memory capacity per NUMA node");
  REQUIRE(read_error("sirius:\n  memory:\n    gpu:\n      downgrade_trigger_fraction: 0.6\n") ==
          "'sirius.memory.gpu.downgrade_stop_fraction': must be less than "
          "downgrade_trigger_fraction");  // stop == trigger
  // A null member takes its default before the pair rule runs.
  REQUIRE(read("sirius:\n  memory:\n    gpu:\n      downgrade_trigger_fraction: ~\n"
               "      downgrade_stop_fraction: 0.5\n")
            .memory.gpu.downgrade_trigger_fraction.get() == 0.8);
}

TEST_CASE("topology shapes", "[config][config_memory]")
{
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: 0\n") ==
          "'sirius.topology.gpu_ids': expected a sequence, got a scalar; GPUs Sirius uses, by "
          "count or by device id");
  REQUIRE(read_error("sirius:\n  topology:\n    gpu_ids: [~]\n") ==
          "'sirius.topology.gpu_ids': element 0: expected a scalar, got null"
          "; GPUs Sirius uses, by count or by device id");
  auto const root = read("sirius:\n  topology:\n    num_gpus: all\n    gpu_ids: ~\n");
  REQUIRE(root.topology.gpus.user_defined());
  REQUIRE(root.topology.gpus.get<std::size_t>() == 0);
}

TEST_CASE("byte values reach UINT64_MAX, as the engine's fields do", "[config][config_memory]")
{
  auto const root =
    read("sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: 18446744073709551615\n");
  REQUIRE(root.memory.gpu.usage_limit.get<ubytes>().count() ==
          std::numeric_limits<std::uint64_t>::max());
  REQUIRE_THAT(
    read_error("sirius:\n  memory:\n    host:\n      block_size: 18446744073709551616\n"),
    Catch::Matchers::ContainsSubstring("byte value out of range"));
}

TEST_CASE("space lists write what they read", "[config][config_memory]")
{
  constexpr std::string_view text = R"(sirius:
  space:
    gpu:
      - device_id: 1
        per_stream_reservation: true
    disk:
      - mount_path: /scratch
        memory_capacity: 2 TiB
)";
  auto const root                 = read(text);
  REQUIRE(parse_yaml(write_yaml(root, "sirius")) == parse_yaml(text));
  auto const all = read(write_yaml(root, "sirius", true));
  REQUIRE(all.space.gpu.at(0).downgrade_stop_fraction.user_defined());
  REQUIRE(all.space.disk.at(0).memory_capacity.get() == 2_TiB);
}

TEST_CASE("validators reject NaN and check their bounds", "[config][config_memory]")
{
  double const nan = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(fraction{}(nan));
  REQUIRE_FALSE(positive_fraction{}(nan));
  REQUIRE_FALSE(positive<double>{}(nan));
  REQUIRE_FALSE(non_negative<double>{}(nan));
  REQUIRE(non_negative<double>{}(0.0));
  REQUIRE_FALSE(positive_fraction{}(0.0));
  REQUIRE(positive<bytes>{}(1_B));
  REQUIRE_FALSE(positive<bytes>{}(0_B));
}
