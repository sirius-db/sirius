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
#include "sirius_context.hpp"

#include <cuda_runtime_api.h>

#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <cstdlib>  // for setenv/putenv
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <source_location>
#include <string>

namespace fs = std::filesystem;

struct finally {
  std::function<void()> func;
  ~finally()
  {
    if (func) { func(); }
  }
};

TEST_CASE("Sirius configuration loading from file with configurator",
          "[sirius][context][isolated_context]")
{
  finally cleanup_env{[]() {
    unsetenv("SIRIUS_CONFIG_FILE");
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  std::source_location loc = std::source_location::current();
  fs::path cfg             = fs::path(loc.file_name()).parent_path() / "data" / "configurator.yaml";

  unsetenv("SIRIUS_DISABLE");
  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // get client context from con
  auto& client_ctx = *con.context;
  // get registered sirius context
  auto sirius_ctx = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");

  REQUIRE(sirius_ctx != nullptr);

  auto& manager = sirius_ctx->get_memory_manager();
  REQUIRE(manager.get_all_memory_spaces().size() == 3);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).size() == 1);
}

TEST_CASE("Sirius configuration loading from file with spaces",
          "[sirius][context][isolated_context]")
{
  finally cleanup_env{[]() {
    unsetenv("SIRIUS_CONFIG_FILE");
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  std::source_location loc = std::source_location::current();
  fs::path cfg             = fs::path(loc.file_name()).parent_path() / "data" / "spaces.yaml";

  unsetenv("SIRIUS_DISABLE");
  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");

  REQUIRE(sirius_ctx != nullptr);

  auto& manager = sirius_ctx->get_memory_manager();
  REQUIRE(manager.get_all_memory_spaces().size() == 4);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).size() == 2);
  REQUIRE(manager.get_all_memory_spaces().size() == 4);
}

// ============================================================================
// Multi-GPU Foundation Validation Tests
// ============================================================================
//
// Device Guard Audit — GPU thread entry points and their cudaSetDevice usage:
//
// 1. gpu_pipeline_executor::get_per_thread_init()
//    File: src/pipeline/gpu_pipeline_executor.cpp
//    Guard: cudaSetDevice(device_id) — called per worker thread before any GPU work.
//    Also: manager_loop() uses rmm::cuda_set_device_raii for the manager thread.
//    Status: VERIFIED — device_id obtained from memory_space->get_device_id().
//
// 2. downgrade_executor::get_per_thread_init()
//    File: src/downgrade/downgrade_executor.cpp
//    Guard: cudaSetDevice(device_id) — called per worker thread.
//    Note: Returns nullptr if _memory_space is null (host-only downgrade).
//    Status: VERIFIED — device_id obtained from memory_space->get_device_id().
//
// 3. sirius_memory_reservation_manager constructor
//    File: src/memory/sirius_memory_reservation_manager.cpp (via parent class)
//    Guard: rmm::cuda_set_device_raii — used during GPU memory resource creation
//           inside cucascade::memory::memory_reservation_manager constructor.
//    Status: VERIFIED — each GPU space is created with the correct device context.
//
// 4. duckdb_scan_executor
//    File: src/op/scan/duckdb_scan_executor.cpp
//    Role: Host-side DuckDB scanning. GPU upload goes through converters which
//          are dispatched in the pipeline executor's GPU context.
//    Status: N/A — no direct GPU operations; data upload handled by converters.
//
// 5. Legacy CUDA wrappers (src/cuda/cudf/*.cu)
//    Role: Only called from gpu_processing (legacy) path, never from gpu_execution.
//    Status: N/A for Super Sirius (new path).
//
// Summary: All GPU thread entry points in the Super Sirius (gpu_execution) path
// correctly set the CUDA device before performing GPU operations. The device_id
// is derived from the memory_space associated with each executor, ensuring
// multi-GPU correctness when multiple executors target different devices.
// ============================================================================

TEST_CASE("topology_discovery populates GPU info", "[multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  REQUIRE(device_count >= 1);

  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover());

  auto const& topology = discovery.get_topology();
  REQUIRE(topology.num_gpus >= 1);
  REQUIRE(topology.gpus.size() == topology.num_gpus);

  for (unsigned int i = 0; i < topology.num_gpus; ++i) {
    auto const& gpu = topology.gpus[i];
    REQUIRE(gpu.id == i);
    REQUIRE(gpu.numa_node >= -1);
    REQUIRE_FALSE(gpu.name.empty());
  }
}

TEST_CASE("reservation_manager_configurator builds N GPU spaces", "[multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  REQUIRE(device_count >= 1);

  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover());
  auto const& topology = discovery.get_topology();

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(topology.num_gpus).use_host_per_numa();
  auto configs = builder.build(topology);

  // Count GPU and HOST tier configs (memory_space_config is a variant post-bump f47de0b)
  size_t gpu_count  = 0;
  size_t host_count = 0;
  for (auto const& cfg : configs) {
    if (std::holds_alternative<cucascade::memory::gpu_memory_space_config>(cfg)) { ++gpu_count; }
    if (std::holds_alternative<cucascade::memory::host_memory_space_config>(cfg)) { ++host_count; }
  }

  REQUIRE(gpu_count == topology.num_gpus);
  REQUIRE(host_count >= 1);
}

TEST_CASE("memory_manager creates independent spaces per GPU", "[multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  REQUIRE(device_count >= 1);

  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 512ull << 20;  // 512 MB for test
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;  // 1 GB

  builder.set_number_of_gpus(static_cast<size_t>(device_count))
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();

  auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == static_cast<size_t>(device_count));

  // Verify each GPU space has a unique device_id
  std::set<int> device_ids;
  for (auto const* space : gpu_spaces) {
    REQUIRE(space != nullptr);
    device_ids.insert(space->get_device_id());
  }
  REQUIRE(device_ids.size() == static_cast<size_t>(device_count));

  auto host_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(host_spaces.size() >= 1);

  manager->shutdown();
  sirius::converter_registry::shutdown();
}

TEST_CASE("converter_registry has gpu_to_gpu converter (MEM-03)", "[multi_gpu_foundation]")
{
  sirius::converter_registry::reset_for_testing();
  sirius::converter_registry::initialize();

  auto& registry = sirius::converter_registry::get();

  // Validate that GPU-to-GPU converter is registered (prerequisite for MEM-03 cross-GPU transfer)
  bool has_gpu_to_gpu =
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::gpu_table_representation>();
  REQUIRE(has_gpu_to_gpu);

  sirius::converter_registry::shutdown();
}

TEST_CASE("multi_gpu_config_two_gpus", "[.][multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs");
    return;
  }

  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 512ull << 20;
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;

  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();

  auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);

  REQUIRE(gpu_spaces[0]->get_device_id() == 0);
  REQUIRE(gpu_spaces[1]->get_device_id() == 1);
  REQUIRE(gpu_spaces[0]->get_device_id() != gpu_spaces[1]->get_device_id());

  manager->shutdown();
  sirius::converter_registry::shutdown();
}
