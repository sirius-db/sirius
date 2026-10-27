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
#include "memory/topology_index.hpp"
#include "preload/hw_caps.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <unistd.h>

#include <filesystem>
#include <string>

using cucascade::memory::system_topology_info;
using sirius::memory::topology_index;
using namespace sirius::preload;

// C2C is detected directly from the CUDA driver attributes of a device (the pinned cuCascade has
// no gpu_runtime_attributes::c2c). The pure decision is tested here; gpu_has_c2c itself delegates
// to it after reading the two attributes.
TEST_CASE("c2c_from_attributes needs native host atomics and host page tables",
          "[preload_policy][hw_caps]")
{
  CHECK_FALSE(c2c_from_attributes(0, 0));
  CHECK_FALSE(c2c_from_attributes(1, 0));
  CHECK_FALSE(c2c_from_attributes(0, 1));
  CHECK(c2c_from_attributes(1, 1));
}

TEST_CASE("gpu_has_c2c rejects out-of-range device ordinals", "[preload_policy][hw_caps]")
{
  system_topology_info topo;
  topo.num_gpus = 0;
  topology_index const index{topo, {}};

  CHECK_FALSE(gpu_has_c2c(index, -1));
  CHECK_FALSE(gpu_has_c2c(index, 1 << 20));  // far past any real device count
}

TEST_CASE("disk_uses_gds reports no GDS on the pinned cuCascade", "[preload_policy][hw_caps]")
{
  auto const dir = std::filesystem::temp_directory_path() /
                   ("sirius_preload_hw_caps_" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  {
    cucascade::memory::disk_memory_space_config cfg;
    cfg.disk_id         = 0;
    cfg.memory_capacity = std::size_t{1} << 30;
    cfg.mount_paths     = dir.string();
    cucascade::memory::memory_space const disk{cfg};
    CHECK_FALSE(disk_uses_gds(disk));
  }
  std::filesystem::remove_all(dir);
}
