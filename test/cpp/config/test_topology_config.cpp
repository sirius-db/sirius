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

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

namespace {

/// Write @p text to @p path and hand back an RAII remover, so a failing REQUIRE inside a
/// test still cleans up the temp file.
struct scoped_yaml {
  explicit scoped_yaml(std::string const& name, std::string const& text)
    : path(std::filesystem::temp_directory_path() / name)
  {
    std::ofstream out(path);
    out << text;
    REQUIRE(out);
  }
  ~scoped_yaml()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  scoped_yaml(scoped_yaml const&)            = delete;
  scoped_yaml& operator=(scoped_yaml const&) = delete;

  std::filesystem::path path;
};

}  // namespace

TEST_CASE("Host initial pools resolve from NUMA RAM unless explicitly configured",
          "[topology_config][config][host_pool]")
{
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                             /*with_runtime_attributes=*/true));
  auto topology = discovery.get_topology();
  REQUIRE_FALSE(topology.gpus.empty());
  topology.gpus.resize(1);
  topology.num_gpus       = 1;
  auto const numa_id      = topology.gpus.front().numa_node;
  topology.numa_nodes     = {{.id = numa_id, .memory_capacity = 64ULL << 30, .has_cpus = true}};
  topology.num_numa_nodes = 1;

  std::string host_settings     = "capacity_bytes: 48Gi";
  std::size_t expected_pools    = 51;
  std::size_t expected_capacity = 48ULL << 30;
  std::string expected_error;
  SECTION("default uses discovered NUMA RAM") {}
  SECTION("larger NUMA capacity changes the initial count")
  {
    topology.numa_nodes.front().memory_capacity = 128ULL << 30;
    expected_pools                              = 102;
  }
  SECTION("custom pool dimensions change the initial count")
  {
    host_settings += ", block_size: 2Mi, pool_size: 256";
    expected_pools = 12;
  }
  SECTION("block alignment is applied before counting pools")
  {
    topology.numa_nodes.front().memory_capacity = 1600;
    host_settings     = "capacity_bytes: 160, block_size: 1, pool_size: 1";
    expected_capacity = 160;
    expected_pools    = 160 / alignof(std::max_align_t);
  }
  SECTION("block alignment overflow is rejected")
  {
    host_settings += ", block_size: " + std::to_string(std::numeric_limits<std::size_t>::max());
    expected_error = "invalid block_size or pool_size";
  }
  SECTION("pool byte size overflow is rejected")
  {
    // Keep the count within YAML's signed range so resolution checks the byte-size overflow.
    constexpr std::size_t block_size = 1ULL << 20;
    constexpr auto pool_size         = std::numeric_limits<std::size_t>::max() / block_size + 1;
    host_settings +=
      ", block_size: " + std::to_string(block_size) + ", pool_size: " + std::to_string(pool_size);
    expected_error = "pool size in bytes overflows";
  }
  SECTION("default host budget uses ninety percent of NUMA RAM")
  {
    host_settings.clear();
    expected_capacity = static_cast<std::size_t>(static_cast<double>(64ULL << 30) * 0.9);
  }
  SECTION("smaller host budget caps initial allocation")
  {
    host_settings     = "capacity_bytes: 2Gi";
    expected_pools    = 16;
    expected_capacity = 2ULL << 30;
  }
  SECTION("explicit count overrides automatic sizing")
  {
    host_settings += ", initial_number_pools: 3";
    expected_pools = 3;
  }
  SECTION("explicit positive count does not require discovered RAM")
  {
    host_settings += ", initial_number_pools: 3";
    topology.numa_nodes.front().memory_capacity = 0;
    expected_pools                              = 3;
  }
  SECTION("automatic count can round down to zero")
  {
    host_settings     = "capacity_bytes: 64Mi";
    expected_pools    = 0;
    expected_capacity = 64ULL << 20;
  }
  SECTION("explicit zero disables preallocation without requiring discovered RAM")
  {
    host_settings += ", initial_number_pools: 0";
    topology.numa_nodes.front().memory_capacity = 0;
    expected_pools                              = 0;
  }
  SECTION("automatic sizing rejects unknown RAM")
  {
    topology.numa_nodes.front().memory_capacity = 0;
    expected_error                              = "set initial_number_pools explicitly";
  }
  scoped_yaml yaml("sirius_initial_host_pools.yaml",
                   "sirius:\n"
                   "  topology: { num_gpus: 1 }\n"
                   "  memory:\n"
                   "    gpu: { usage_limit_bytes: 1Gi }\n"
                   "    host: { " +
                     host_settings + " }\n");
  auto parsed = sirius::parsed_sirius_config::from_file(yaml.path);
  if (!expected_error.empty()) {
    REQUIRE_THROWS_WITH(parsed.resolve(topology),
                        Catch::Matchers::ContainsSubstring(expected_error));
    return;
  }
  auto resolved          = parsed.resolve(topology);
  std::size_t host_count = 0;
  for (auto const& space : resolved.get_memory_space_configs()) {
    if (auto const* host = std::get_if<cucascade::memory::host_memory_space_config>(&space)) {
      CHECK(host->initial_number_pools == expected_pools);
      CHECK(host->memory_capacity == expected_capacity);
      ++host_count;
    }
  }
  CHECK(host_count == 1);
}

TEST_CASE("parsed hardware configuration resolves independently after copying",
          "[topology_config][config]")
{
  scoped_yaml yaml("sirius_deferred_topology.yaml",
                   "sirius:\n"
                   "  topology: { num_gpus: 1 }\n"
                   "  memory:\n"
                   "    gpu: { usage_limit_bytes: 1Gi }\n"
                   "    host: { capacity_bytes: 2Gi }\n"
                   "  operator_params: { concat_batch_bytes: 123456789 }\n"
                   "  telemetry: { output_directory: deferred_telemetry }\n");
  auto parsed = sirius::parsed_sirius_config::from_file(yaml.path);
  CHECK(parsed.get_telemetry_config().output_directory == "deferred_telemetry");
  CHECK(parsed.get_operator_params().concat_batch_bytes == 123456789);

  auto copy = parsed;
  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                             /*with_runtime_attributes=*/true));
  auto resolved = copy.resolve(discovery.get_topology());
  REQUIRE_FALSE(resolved.get_memory_space_configs().empty());
  CHECK(resolved.get_operator_params().scan_task_batch_size == (1ULL << 30) / 40);
  CHECK(resolved.get_operator_params().concat_batch_bytes == 123456789);
  CHECK(parsed.get_operator_params().concat_batch_bytes == 123456789);

  // A new resolution leaves existing runtime overrides and the specification unchanged.
  resolved.get_operator_params().concat_batch_bytes = 987654321;
  auto another                                      = parsed.resolve(discovery.get_topology());
  CHECK(resolved.get_operator_params().concat_batch_bytes == 987654321);
  CHECK(another.get_operator_params().concat_batch_bytes == 123456789);
}

TEST_CASE("sirius_config rejects the removed telemetry.exporter key", "[topology_config][config]")
{
  scoped_yaml yaml("sirius_telemetry_exporter.yaml",
                   "sirius:\n"
                   "  telemetry: { exporter: ndjson }\n");
  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path));
}

TEST_CASE("sirius_config parses topology.gpus_per_query", "[topology_config][config]")
{
  scoped_yaml yaml("sirius_gpus_per_query.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    gpus_per_query: 2\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.gpus_per_query() == 2);
}

TEST_CASE("sirius_config defaults gpus_per_query to 0 when absent", "[topology_config][config]")
{
  // 0 means "admit every GPU", so an absent key must not narrow anything.
  scoped_yaml yaml("sirius_gpus_per_query_absent.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    num_gpus: 1\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.gpus_per_query() == 0);
}

TEST_CASE("sirius_config accepts an explicit gpus_per_query of 0", "[topology_config][config]")
{
  scoped_yaml yaml("sirius_gpus_per_query_zero.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    gpus_per_query: 0\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.gpus_per_query() == 0);
}

TEST_CASE("sirius_config rejects a negative gpus_per_query", "[topology_config][config]")
{
  // Admission treats any non-positive value as "no cap", so a negative would silently read
  // as "use every GPU" — the opposite of what someone writing -1 could mean. Reject at load.
  scoped_yaml yaml("sirius_gpus_per_query_negative.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    gpus_per_query: -1\n");

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path));
}

TEST_CASE("sirius_config rejects a zero avg_variable_column_bytes", "[topology_config][config]")
{
  // Zero would make variable-width columns contribute nothing to the per-row width, so a
  // mixed schema is under-estimated and the query admitted onto too few GPUs. Unlike
  // admission_bytes_per_gpu, where 0 is the documented off switch, there is no reading of
  // zero here that means anything.
  scoped_yaml yaml("sirius_avg_var_zero.yaml",
                   "sirius:\n"
                   "  operator_params:\n"
                   "    avg_variable_column_bytes: 0\n");

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path));
}

TEST_CASE("sirius_config accepts a zero admission_bytes_per_gpu", "[topology_config][config]")
{
  // 0 is the off switch: it disables the estimate and leaves sizing to gpus_per_query.
  scoped_yaml yaml("sirius_admission_bytes_zero.yaml",
                   "sirius:\n"
                   "  operator_params:\n"
                   "    admission_bytes_per_gpu: 0\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.get_operator_params().admission_bytes_per_gpu == 0);
}

TEST_CASE("sirius_config accepts num_gpus: all and refuses 0", "[topology_config][config]")
{
  scoped_yaml all("sirius_num_gpus_all.yaml", "sirius:\n  topology: { num_gpus: all }\n");
  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(all.path));

  // 0 fails, and the error names the word to use.
  scoped_yaml zero("sirius_num_gpus_zero.yaml", "sirius:\n  topology: { num_gpus: 0 }\n");
  sirius::sirius_config zero_cfg;
  CHECK_THROWS_WITH(zero_cfg.load_from_file(zero.path),
                    Catch::Matchers::ContainsSubstring("num_gpus: all"));

  scoped_yaml word("sirius_num_gpus_word.yaml", "sirius:\n  topology: { num_gpus: some }\n");
  sirius::sirius_config word_cfg;
  CHECK_THROWS(word_cfg.load_from_file(word.path));
}

TEST_CASE("sirius_config skips the distributed section", "[topology_config][config]")
{
  // The launch scripts read sirius.distributed; Sirius itself must not refuse the file.
  scoped_yaml yaml("sirius_distributed_section.yaml",
                   "sirius:\n"
                   "  topology: { num_gpus: all }\n"
                   "  distributed:\n"
                   "    cluster: default\n"
                   "    node: { gpus: [0, 1], quack_port: 9500 }\n");
  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
}
