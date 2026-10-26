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

TEST_CASE("Preparation options bound metadata jobs and preserve post-resolution overrides",
          "[scan_preparation][preparation_config]")
{
  sirius::scan_manager::scan_manager_config config;
  config.thread_pool.num_threads = 3;
  auto defaults                  = config.preparation.resolve(3);
  CHECK(defaults.max_inflight_metadata_jobs == 4096);
  CHECK(defaults.max_inflight_dv_jobs == 0);
  config.preparation.max_inflight_metadata_jobs = 0;
  config.preparation.max_inflight_dv_jobs       = 0;
  CHECK(config.preparation.resolve(3).max_inflight_dv_jobs == 0);
  CHECK(config.preparation.resolve(3).max_inflight_metadata_jobs == 0);
  CHECK(defaults.lookahead_batches == 2);
  CHECK(defaults.max_control_work > 0);
  CHECK(defaults.drain_quantum > 0);
  CHECK(defaults.interrupt_check_interval == std::chrono::milliseconds(10));
  CHECK(defaults.partial_batch);
  config.preparation.interrupt_check_interval   = std::chrono::milliseconds(3);
  config.preparation.max_inflight_metadata_jobs = 2;
  config.preparation.max_inflight_dv_jobs       = 7;
  config.preparation.lookahead_batches          = 7;
  config.preparation.max_control_work           = 4;
  config.preparation.drain_quantum              = 5;
  config.preparation.partial_batch              = sirius::scan_manager::partial_batch_options{
    50, std::chrono::milliseconds(11), std::chrono::milliseconds(11)};
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
  CHECK(snapshot.max_inflight_metadata_jobs == 2);
  CHECK(snapshot.max_inflight_dv_jobs == 7);
  CHECK(snapshot.lookahead_batches == 7);
  CHECK(snapshot.max_control_work == 4);
  CHECK(snapshot.drain_quantum == 5);
  CHECK(snapshot.interrupt_check_interval == std::chrono::milliseconds(3));
  REQUIRE(snapshot.partial_batch);
  CHECK(snapshot.partial_batch->max_residence == std::chrono::milliseconds(11));
  config.preparation.max_inflight_metadata_jobs = 1;
  CHECK(snapshot.max_inflight_metadata_jobs == 2);  // independent attempt value
}
TEST_CASE("Invalid preparation options fail before constructing a scan manager",
          "[scan_preparation][preparation_config]")
{
  sirius::scan_manager::scan_manager_config config;
  SECTION("zero lookahead") { config.preparation.lookahead_batches = 0; }
  SECTION("zero control") { config.preparation.max_control_work = 0; }
  SECTION("zero quantum") { config.preparation.drain_quantum = 0; }
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
  config.preparation.lookahead_batches = 0;
  sirius::exec::query_lifecycle_registry lifecycle;
  CHECK_THROWS_WITH(sirius::scan_manager::sirius_scan_manager(lifecycle, config, memory, nullptr),
                    "preparation limits must be positive");
}

TEST_CASE("Size-based partial batching defaults to bounded residence with validated overrides",
          "[scan_preparation][preparation_config][partial_batch]")
{
  using namespace sirius::scan_manager;
  using namespace std::chrono_literals;
  preparation_config config;
  auto defaults = config.resolve(4);
  REQUIRE(defaults.partial_batch);
  CHECK(defaults.partial_batch->min_fill_percent == 50);
  CHECK(defaults.partial_batch->min_residence == 50ms);
  CHECK(defaults.partial_batch->max_residence == 2s);
  config.partial_batch = partial_batch_options{25, 20ms, 500ms};
  auto changed         = config.resolve(4);
  CHECK(changed.partial_batch->min_fill_percent == 25);
  CHECK(changed.partial_batch->min_residence == 20ms);
  CHECK(changed.partial_batch->max_residence == 500ms);
  CHECK(defaults.partial_batch->min_fill_percent == 50);
  config.partial_batch = partial_batch_options{};
  SECTION("zero fill") { config.partial_batch->min_fill_percent = 0; }
  SECTION("fill above target") { config.partial_batch->min_fill_percent = 101; }
  SECTION("negative minimum time") { config.partial_batch->min_residence = -1ms; }
  SECTION("zero maximum time") { config.partial_batch->max_residence = 0ms; }
  SECTION("zero minimum time") { config.partial_batch->min_residence = 0ms; }
  SECTION("maximum precedes minimum") { config.partial_batch->max_residence = 49ms; }
  CHECK_THROWS_AS(config.resolve(4), std::invalid_argument);
}
