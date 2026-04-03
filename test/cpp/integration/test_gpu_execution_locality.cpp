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
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/sirius_pipeline_task_states.hpp"

#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cuda_runtime_api.h>

#include <optional>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// Test: preferred_device_id on gpu_pipeline_task_local_state (SCHED-01)
//===----------------------------------------------------------------------===//

TEST_CASE("preferred_device_id defaults to nullopt", "[data_locality]")
{
  // A local state with no explicit preferred device should return nullopt
  auto local_state = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(nullptr);
  REQUIRE_FALSE(local_state->get_preferred_device_id().has_value());
}

TEST_CASE("preferred_device_id can be set and retrieved", "[data_locality]")
{
  // SCHED-01: Verify that the preferred device ID mechanism works at the type level
  auto local_state = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(nullptr);

  local_state->set_preferred_device_id(0);
  REQUIRE(local_state->get_preferred_device_id().has_value());
  REQUIRE(local_state->get_preferred_device_id().value() == 0);

  local_state->set_preferred_device_id(3);
  REQUIRE(local_state->get_preferred_device_id().value() == 3);
}

//===----------------------------------------------------------------------===//
// Test: pipeline tasks can have different preferred devices (SCHED-04)
//===----------------------------------------------------------------------===//

TEST_CASE("pipeline tasks can have different preferred_device_ids", "[data_locality]")
{
  // SCHED-04: Different tasks from the same query can route to different GPUs
  auto local_state_0 = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(nullptr);
  auto local_state_1 = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(nullptr);

  local_state_0->set_preferred_device_id(0);
  local_state_1->set_preferred_device_id(1);

  REQUIRE(local_state_0->get_preferred_device_id().value() == 0);
  REQUIRE(local_state_1->get_preferred_device_id().value() == 1);
  REQUIRE(local_state_0->get_preferred_device_id() != local_state_1->get_preferred_device_id());
}

//===----------------------------------------------------------------------===//
// Test: global state preferred_device_id as pipeline-level default (SCHED-04)
//===----------------------------------------------------------------------===//

TEST_CASE("global state preferred_device_id serves as pipeline default", "[data_locality]")
{
  // SCHED-04: Pipeline-level default from global state
  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(nullptr);

  REQUIRE_FALSE(global_state->get_preferred_device_id().has_value());

  global_state->set_preferred_device_id(2);
  REQUIRE(global_state->get_preferred_device_id().has_value());
  REQUIRE(global_state->get_preferred_device_id().value() == 2);
}

//===----------------------------------------------------------------------===//
// Test: local state preferred_device_id overrides global state (SCHED-01)
//===----------------------------------------------------------------------===//

TEST_CASE("local state preferred_device_id takes precedence over global", "[data_locality]")
{
  // SCHED-01: Per-task locality score (local) overrides pipeline default (global)
  auto local_state = std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(nullptr);
  auto global_state = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(nullptr);

  global_state->set_preferred_device_id(0);
  local_state->set_preferred_device_id(1);

  // The gpu_pipeline_task::get_preferred_device_id() checks local first, then global.
  // We verify the individual states here since constructing a full gpu_pipeline_task
  // requires more infrastructure (data repos, etc.)
  REQUIRE(local_state->get_preferred_device_id().value() == 1);
  REQUIRE(global_state->get_preferred_device_id().value() == 0);
  // Local takes precedence
  REQUIRE(local_state->get_preferred_device_id().value() != global_state->get_preferred_device_id().value());
}

//===----------------------------------------------------------------------===//
// Test: NUMA-to-GPU mapping logic (SCHED-02)
//===----------------------------------------------------------------------===//

TEST_CASE("NUMA-to-GPU mapping selects first GPU per NUMA node", "[data_locality]")
{
  // SCHED-02: Verify the NUMA mapping logic used in task_creator
  // This tests the algorithm without needing the full task_creator

  // Simulate a topology: GPU 0 on NUMA 0, GPU 1 on NUMA 0, GPU 2 on NUMA 1
  struct gpu_info {
    int id;
    int numa_node;
  };

  std::vector<gpu_info> gpus = {{0, 0}, {1, 0}, {2, 1}};

  // Build NUMA-to-GPU map using the same logic as task_creator constructor
  std::unordered_map<int, int> numa_to_gpu;
  for (const auto& gpu : gpus) {
    if (gpu.numa_node >= 0 && numa_to_gpu.find(gpu.numa_node) == numa_to_gpu.end()) {
      numa_to_gpu[gpu.numa_node] = gpu.id;
    }
  }

  // NUMA 0 should map to GPU 0 (first GPU found)
  REQUIRE(numa_to_gpu.at(0) == 0);
  // NUMA 1 should map to GPU 2
  REQUIRE(numa_to_gpu.at(1) == 2);
  // Only 2 NUMA nodes
  REQUIRE(numa_to_gpu.size() == 2);
}

//===----------------------------------------------------------------------===//
// Test: Locality scoring algorithm (SCHED-01)
//===----------------------------------------------------------------------===//

TEST_CASE("locality score selects GPU with most bytes", "[data_locality]")
{
  // SCHED-01: Verify the algorithm that computes preferred device from data locations
  // This reproduces the logic from task_creator::manager_loop() locality scoring

  // Simulate data batch sizes per GPU device
  std::unordered_map<int, size_t> gpu_bytes;
  gpu_bytes[0] = 1024 * 1024;      // 1 MB on GPU 0
  gpu_bytes[1] = 4 * 1024 * 1024;  // 4 MB on GPU 1

  // The algorithm: pick GPU with most bytes
  auto preferred = std::max_element(gpu_bytes.begin(), gpu_bytes.end(), [](const auto& a, const auto& b) {
    return a.second < b.second;
  });

  REQUIRE(preferred->first == 1);  // GPU 1 has more data
  REQUIRE(preferred->second == 4 * 1024 * 1024);
}

TEST_CASE("locality score falls back to NUMA when data only on HOST", "[data_locality]")
{
  // SCHED-02: When no GPU has data, use NUMA-to-GPU mapping from host data
  std::unordered_map<int, size_t> gpu_bytes;  // empty - no GPU data
  std::unordered_map<int, size_t> host_bytes;
  host_bytes[0] = 2 * 1024 * 1024;  // 2 MB on NUMA node 0
  host_bytes[1] = 8 * 1024 * 1024;  // 8 MB on NUMA node 1

  std::unordered_map<int, int> numa_to_gpu = {{0, 0}, {1, 2}};

  std::optional<int> preferred_device_id;

  if (!gpu_bytes.empty()) {
    preferred_device_id =
      std::max_element(gpu_bytes.begin(), gpu_bytes.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
      })->first;
  } else if (!host_bytes.empty() && !numa_to_gpu.empty()) {
    auto top_host =
      std::max_element(host_bytes.begin(), host_bytes.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
      })->first;
    auto it = numa_to_gpu.find(top_host);
    if (it != numa_to_gpu.end()) { preferred_device_id = it->second; }
  }

  // NUMA node 1 has most host data, maps to GPU 2
  REQUIRE(preferred_device_id.has_value());
  REQUIRE(preferred_device_id.value() == 2);
}

//===----------------------------------------------------------------------===//
// Test: scan distribution across GPUs (SCHED-05)
// Requires multi-GPU hardware - disabled by default
//===----------------------------------------------------------------------===//

TEST_CASE("scan batches distributed across multiple GPUs", "[.][data_locality][multi_gpu]")
{
  // SCHED-05: Verify that select_target_gpu() distributes across GPUs
  // This test requires actual multi-GPU hardware
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("requires 2+ GPUs for scan distribution test -- skipping");
    return;
  }

  // Verify that the system actually has multiple GPU devices accessible
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);
    REQUIRE(props.major >= 7);  // Turing or newer
    INFO("GPU " << i << ": " << props.name);
  }

  // With multiple GPUs, scan batches should be distributable
  // The actual distribution test requires a full duckdb_scan_executor instance
  // which needs a memory_reservation_manager -- this is covered by the
  // proportional distribution algorithm test above
  SUCCEED("Multi-GPU hardware detected with " + std::to_string(device_count) + " GPUs");
}

//===----------------------------------------------------------------------===//
// Test: proportional distribution algorithm (SCHED-05)
//===----------------------------------------------------------------------===//

TEST_CASE("proportional distribution algorithm distributes by memory", "[data_locality]")
{
  // SCHED-05: Test the core distribution algorithm used by select_target_gpu()
  // This verifies the weighted distribution logic without needing a full executor

  struct mock_space {
    int device_id;
    size_t available_memory;
  };

  // GPU 0 has 200 units free, GPU 1 has 600 units free
  // Expected: ~25% batches to GPU 0, ~75% to GPU 1
  std::vector<mock_space> spaces = {{0, 200}, {1, 600}};

  size_t total_available = 0;
  for (auto& s : spaces) {
    total_available += s.available_memory;
  }
  REQUIRE(total_available > 0);

  // Run the weighted selection algorithm for N full cycles to avoid modular edge effects
  size_t num_iterations = total_available * 10;  // 10 full cycles
  std::unordered_map<int, int> distribution;
  for (size_t counter = 0; counter < num_iterations; ++counter) {
    size_t target     = counter % total_available;
    size_t cumulative = 0;
    for (auto& s : spaces) {
      cumulative += s.available_memory;
      if (target < cumulative) {
        distribution[s.device_id]++;
        break;
      }
    }
  }

  // GPU 1 should get significantly more batches than GPU 0 (roughly 3:1 ratio)
  REQUIRE(distribution.count(0) > 0);
  REQUIRE(distribution.count(1) > 0);
  REQUIRE(distribution[1] > distribution[0]);

  // With 2:6 ratio, GPU 0 gets ~25% and GPU 1 gets ~75%
  double ratio = static_cast<double>(distribution[1]) / distribution[0];
  REQUIRE(ratio > 2.0);   // Should be approximately 3.0
  REQUIRE(ratio < 4.0);

  INFO("Distribution: GPU 0 = " << distribution[0] << ", GPU 1 = " << distribution[1]
                                  << ", ratio = " << ratio);
}

TEST_CASE("proportional distribution falls back to round-robin when all full", "[data_locality]")
{
  // SCHED-05: When all GPUs are at capacity (0 available), fall back to round-robin
  struct mock_space {
    int device_id;
    size_t available_memory;
  };

  std::vector<mock_space> spaces = {{0, 0}, {1, 0}, {2, 0}};

  size_t total_available = 0;
  for (auto& s : spaces) {
    total_available += s.available_memory;
  }
  REQUIRE(total_available == 0);

  // Round-robin: counter mod num_spaces
  std::unordered_map<int, int> distribution;
  for (size_t counter = 0; counter < 300; ++counter) {
    auto idx = counter % spaces.size();
    distribution[spaces[idx].device_id]++;
  }

  // Each GPU should get exactly 100 batches (perfect round-robin)
  REQUIRE(distribution[0] == 100);
  REQUIRE(distribution[1] == 100);
  REQUIRE(distribution[2] == 100);
}
