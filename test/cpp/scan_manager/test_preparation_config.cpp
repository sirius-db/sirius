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
#include "scan_manager/config.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_config.hpp"

#include <cucascade/memory/memory_reservation_manager.hpp>

TEST_CASE("Preparation options have finite defaults and preserve post-resolution overrides",
          "[scan_preparation][preparation_config]")
{
  sirius::scan_manager::scan_manager_config config;
  config.thread_pool.num_threads = 3;
  auto defaults                  = config.preparation.resolve(3);
  CHECK(defaults.max_inflight_jobs == 3);
  CHECK(defaults.max_active_units >= defaults.max_inflight_jobs);
  CHECK(defaults.max_pending_results >= defaults.max_inflight_jobs);
  CHECK(defaults.max_control_work > 0);
  CHECK(defaults.drain_quantum > 0);
  CHECK(defaults.interrupt_check_interval == std::chrono::milliseconds(10));
  CHECK_FALSE(defaults.underfilled_batch_residence.has_value());  // benchmark precedes activation
  config.preparation.interrupt_check_interval    = std::chrono::milliseconds(3);
  config.preparation.max_inflight_jobs           = 2;
  config.preparation.max_active_units            = 7;
  config.preparation.max_pending_results         = 9;
  config.preparation.max_control_work            = 4;
  config.preparation.drain_quantum               = 5;
  config.preparation.underfilled_batch_residence = std::chrono::milliseconds(11);
  // Resolve defaults against deterministic capacities before applying internal overrides.
  // apply_defaults() replaces all settings and discovers the machine's actual topology.
  cucascade::memory::system_topology_info topology{};
  topology.num_gpus       = 1;
  topology.num_numa_nodes = 1;
  topology.gpus           = {{.id = 0, .numa_node = 0}};
  topology.numa_nodes     = {{.id = 0, .memory_capacity = 1ULL << 30, .has_cpus = true}};
  auto engine             = sirius::parsed_sirius_config{}.resolve(topology);
  engine.set_scan_manager_config(config);
  auto const& resolved = engine.get_scan_manager_config();
  REQUIRE(resolved.thread_pool.num_threads == 3);
  auto snapshot = resolved.preparation.resolve(resolved.thread_pool.num_threads);
  CHECK(snapshot.max_inflight_jobs == 2);
  CHECK(snapshot.max_active_units == 7);
  CHECK(snapshot.max_pending_results == 9);
  CHECK(snapshot.max_control_work == 4);
  CHECK(snapshot.drain_quantum == 5);
  CHECK(snapshot.interrupt_check_interval == std::chrono::milliseconds(3));
  CHECK(snapshot.underfilled_batch_residence == std::chrono::milliseconds(11));
  config.preparation.max_inflight_jobs = 1;
  CHECK(snapshot.max_inflight_jobs == 2);  // independent attempt value
}
TEST_CASE("Invalid preparation options fail before constructing a scan manager",
          "[scan_preparation][preparation_config]")
{
  sirius::scan_manager::scan_manager_config config;
  SECTION("zero jobs") { config.preparation.max_inflight_jobs = 0; }
  SECTION("zero units") { config.preparation.max_active_units = 0; }
  SECTION("zero results") { config.preparation.max_pending_results = 0; }
  SECTION("zero control") { config.preparation.max_control_work = 0; }
  SECTION("zero quantum") { config.preparation.drain_quantum = 0; }
  SECTION("zero duration")
  {
    config.preparation.underfilled_batch_residence = std::chrono::milliseconds(0);
  }
  SECTION("negative duration")
  {
    config.preparation.underfilled_batch_residence = std::chrono::milliseconds(-1);
  }
  SECTION("zero interrupt interval")
  {
    config.preparation.interrupt_check_interval = std::chrono::milliseconds(0);
  }
  SECTION("negative interrupt interval")
  {
    config.preparation.interrupt_check_interval = std::chrono::milliseconds(-1);
  }
  SECTION("no workers") { config.thread_pool.num_threads = 0; }
  CHECK_THROWS_AS(sirius::scan_manager::validate_scan_manager_config(config),
                  std::invalid_argument);
}

TEST_CASE("Manager construction validates preparation before creating worker or IO state",
          "[scan_preparation][preparation_config]")
{
  cucascade::memory::host_memory_space_config host_config;
  host_config.numa_id              = 0;
  host_config.memory_capacity      = 16 * 1024;
  host_config.block_size           = 1024;
  host_config.pool_size            = 16;
  host_config.initial_number_pools = 1;
  cucascade::memory::memory_reservation_manager memory({host_config});
  sirius::scan_manager::scan_manager_config config;
  config.preparation.max_inflight_jobs = 0;
  CHECK_THROWS_WITH(sirius::scan_manager::sirius_scan_manager(config, memory, nullptr),
                    "preparation limits and configured residence must be positive");
}
