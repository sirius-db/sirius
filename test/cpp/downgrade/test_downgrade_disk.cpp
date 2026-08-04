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

// sirius
#include "data/convertible_data_batch.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"

// data utilities
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <utils/utils.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf / rmm
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::parallel;

namespace {

/// Helper: get the tier of a data_batch using a temporary read-only lock.
inline cucascade::memory::Tier get_batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

cucascade::memory::memory_space* get_gpu_space(
  sirius::memory::sirius_memory_reservation_manager& mgr)
{
  auto* space = mgr.get_memory_space(cucascade::memory::Tier::GPU, 0);
  if (space) return space;
  auto spaces = mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  if (!spaces.empty()) return const_cast<cucascade::memory::memory_space*>(spaces.front());
  return nullptr;
}

std::shared_ptr<cucascade::data_batch> make_gpu_batch(cucascade::memory::memory_space& gpu_space,
                                                      size_t num_rows = 1000)
{
  auto stream = cudf::get_default_stream();
  auto mr     = gpu_space.get_default_allocator();

  std::vector<cudf::data_type> col_types                 = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {std::make_pair(0, 100000)};

  auto table = sirius::create_cudf_table_with_random_data(num_rows, col_types, ranges, stream, mr);

  return sirius::make_data_batch(
    std::move(table), gpu_space, stream, sirius::telemetry::batch_telemetry_info{});
}

// Pre-exhaust all HOST capacity so that subsequent make_reservation_or_null calls
// return null (rather than throwing). Returns held reservations — caller must keep
// them alive for the duration of the test.
//
// Background: fixed_size_host_memory_resource::reserve() throws when
// bytes > _memory_limit, but returns nullptr when all slots are in use.
// We drain HOST by claiming 1MB (the minimum block size) chunks until null.
std::vector<std::unique_ptr<cucascade::memory::reservation>> exhaust_host_capacity(
  sirius::memory::sirius_memory_reservation_manager& mem_mgr)
{
  std::vector<std::unique_ptr<cucascade::memory::reservation>> held;

  auto* host_space = mem_mgr.get_memory_space(cucascade::memory::Tier::HOST, 0);
  if (!host_space) {
    auto spaces = mem_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (!spaces.empty()) host_space = const_cast<cucascade::memory::memory_space*>(spaces.front());
  }
  if (!host_space) return held;

  constexpr size_t kBlockSize = 1ull << 20;  // 1 MB (minimum reservation block)
  while (true) {
    auto res = host_space->make_reservation_or_null(kBlockSize);
    if (!res) break;
    held.push_back(std::move(res));
  }
  return held;
}

const auto GPU_SPACE_ID = cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0);

// These tests exercise a single query's repositories; the executor sweeps the registry,
// so each test registers its manager under one fixed query id.
const sirius::query_id_t kTestQueryId = sirius::make_query_id(1);

// Build a manager with a small HOST tier and NO disk, with a low GPU downgrade trigger so that
// modest GPU pressure is enough to keep should_downgrade_memory() true. Used to reproduce the
// "monitor spins because nothing can be downgraded" scenario.
std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_no_disk_pressure_manager()
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 256ull << 20;  // 256 MB
  const size_t host_capacity = 2ull << 20;    // 2 MB — exhausted quickly

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(0.9)
    .set_downgrade_fractions_per_gpu(0.1, 0.05)  // trigger GPU downgrade under modest pressure
    .set_per_numa_region_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_numa_region(0.75)
    .set_downgrade_fractions_per_numa_region(0.1, 0.05);  // also let a HOST monitor reach pressure
  // No set_disk_mounting_point — DISK tier is absent.

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();
  return manager;
}

downgrade_executor make_monitoring_executor(
  sirius::data::data_repository_manager_registry& repo_registry,
  cucascade::memory::memory_space* gpu_space,
  sirius::memory::sirius_memory_reservation_manager& mgr)
{
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period = std::chrono::milliseconds{10}};
  return downgrade_executor(config, repo_registry, GPU_SPACE_ID, gpu_space, mgr);
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Downgrade task falls back to DISK when HOST is full", "[downgrade_disk]")
{
  // Set HOST capacity small (2MB, limit = 1.5MB after 0.75 fraction).
  // Pre-exhaust HOST so make_reservation_or_null returns null gracefully.
  // DISK (4GB) must then be chosen by any_memory_space_in_tiers{HOST, DISK}.
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 2ull << 30;
  const size_t host_capacity = 2ull << 20;  // 2 MB — small enough to exhaust quickly
  const double limit_ratio   = 0.75;

  auto tmp_dir = std::filesystem::temp_directory_path() / "sirius_test_disk";
  std::filesystem::create_directories(tmp_dir);

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_numa_region(limit_ratio)
    .set_disk_mounting_point(0, 4ull << 30, tmp_dir.string());

  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  // Pre-exhaust HOST so make_reservation_or_null returns null for HOST
  auto held_host = exhaust_host_capacity(*mem_mgr);
  REQUIRE_FALSE(held_host.empty());

  auto batch = make_gpu_batch(*gpu_space);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);

  rmm::cuda_stream stream;
  std::vector<const cucascade::memory::memory_space*> target_spaces;
  auto host_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  for (auto* hs : host_spaces) {
    target_spaces.push_back(hs);
  }
  auto disk_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
  for (auto* ds : disk_spaces) {
    target_spaces.push_back(ds);
  }
  sirius::convertible_data_batch batch_converter(batch);
  auto converted = batch_converter.convert(target_spaces, stream, *mem_mgr, false);
  REQUIRE(converted.has_value());

  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::DISK);
}

TEST_CASE("Downgrade task uses HOST when HOST has capacity", "[downgrade_disk]")
{
  // HOST (4GB) and DISK (4GB) both available — HOST is listed first in tier preference,
  // so any_memory_space_in_tiers{HOST, DISK} must pick HOST.
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity = 2ull << 30;
  const double limit_ratio  = 0.75;

  auto tmp_dir = std::filesystem::temp_directory_path() / "sirius_test_disk";
  std::filesystem::create_directories(tmp_dir);

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(4ull << 30)
    .use_host_per_gpu()
    .set_reservation_fraction_per_numa_region(limit_ratio)
    .set_disk_mounting_point(0, 4ull << 30, tmp_dir.string());

  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  auto batch = make_gpu_batch(*gpu_space);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);

  rmm::cuda_stream stream;
  std::vector<const cucascade::memory::memory_space*> target_spaces;
  auto host_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  for (auto* hs : host_spaces) {
    target_spaces.push_back(hs);
  }
  auto disk_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
  for (auto* ds : disk_spaces) {
    target_spaces.push_back(ds);
  }
  sirius::convertible_data_batch batch_converter(batch);
  auto converted = batch_converter.convert(target_spaces, stream, *mem_mgr, true);
  REQUIRE(converted.has_value());

  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
}

TEST_CASE("Downgrade task returns false when HOST full and no DISK tier", "[downgrade_disk]")
{
  // HOST exhausted, no DISK tier configured.
  // The non-blocking reservation probe must return false (skip) rather than
  // blocking forever or throwing, so the scheduler can retry later.
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 2ull << 30;
  const size_t host_capacity = 2ull << 20;  // 2 MB — small enough to exhaust quickly
  const double limit_ratio   = 0.75;

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_numa_region(limit_ratio);
  // No set_disk_mounting_point — DISK tier is absent

  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  // Pre-exhaust HOST so make_reservation_or_null returns null
  auto held_host = exhaust_host_capacity(*mem_mgr);
  REQUIRE_FALSE(held_host.empty());

  auto batch = make_gpu_batch(*gpu_space);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);

  rmm::cuda_stream stream;
  std::vector<const cucascade::memory::memory_space*> target_spaces;
  auto host_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  for (auto* hs : host_spaces) {
    target_spaces.push_back(hs);
  }
  auto disk_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK);
  for (auto* ds : disk_spaces) {
    target_spaces.push_back(ds);
  }
  sirius::convertible_data_batch batch_converter(batch);
  auto converted = batch_converter.convert(target_spaces, stream, *mem_mgr, false);
  REQUIRE_FALSE(converted.has_value());
  // Batch must remain on GPU
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);
}

// Regression: before the fix, the monitor thread fired a downgrade request every
// monitor_period (10ms) whenever GPU memory was under pressure -- even when nothing could be
// downgraded (HOST full, no DISK configured) -- spinning forever (~100 fruitless full repo+queue
// scans and warnings per second). The stateless viability gate must back off instead.
TEST_CASE("monitor backs off when no downgrade target is viable (no disk)", "[downgrade_disk]")
{
  auto mem_mgr    = make_no_disk_pressure_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  // Exhaust HOST so no downgrade can be accepted there; with no DISK there is no viable target.
  auto held_host = exhaust_host_capacity(*mem_mgr);
  REQUIRE_FALSE(held_host.empty());

  // Create GPU pressure (reservation-only bookkeeping) so should_downgrade_memory() stays true.
  auto gpu_hold = gpu_space->make_reservation_or_null(128ull << 20);
  REQUIRE(gpu_hold);
  REQUIRE(gpu_space->should_downgrade_memory());

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto executor  = make_monitoring_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  using namespace std::chrono_literals;
  // Sample two windows. With the gate, the monitor never issues a request (target check happens
  // before firing), so the count stays flat across both windows.
  std::this_thread::sleep_for(300ms);
  size_t n1 = executor.monitor_requests_issued_for_testing();
  std::this_thread::sleep_for(300ms);
  size_t n2 = executor.monitor_requests_issued_for_testing();

  executor.stop();

  REQUIRE(n2 == n1);  // backed off: no continued firing
  REQUIRE(n1 <= 1);   // at most a single startup-race request before the gate engaged
}

// The gate is stateless: it re-checks viability every cycle, so the monitor must resume firing the
// instant a target becomes available again -- here, when HOST capacity is released.
TEST_CASE("monitor resumes after a downgrade target frees up (no disk)", "[downgrade_disk]")
{
  auto mem_mgr    = make_no_disk_pressure_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  auto held_host = exhaust_host_capacity(*mem_mgr);
  REQUIRE_FALSE(held_host.empty());

  auto gpu_hold = gpu_space->make_reservation_or_null(128ull << 20);
  REQUIRE(gpu_hold);
  REQUIRE(gpu_space->should_downgrade_memory());

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto executor  = make_monitoring_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  using namespace std::chrono_literals;
  // Confirm it is backed off (count stable while HOST is full).
  std::this_thread::sleep_for(200ms);
  size_t stalled = executor.monitor_requests_issued_for_testing();
  std::this_thread::sleep_for(200ms);
  REQUIRE(executor.monitor_requests_issued_for_testing() == stalled);

  // Release HOST capacity -> a downgrade target is viable again. GPU pressure (gpu_hold) is still
  // held, so should_downgrade_memory() remains true and the monitor should start firing again.
  held_host.clear();

  size_t before = executor.monitor_requests_issued_for_testing();
  std::this_thread::sleep_for(300ms);
  size_t after = executor.monitor_requests_issued_for_testing();

  executor.stop();

  REQUIRE(after > before);  // resumed firing once a target became viable
}

// The viability gate must reflect HOST being full of *real stored downgraded data*, not just held
// reservations: a downgrade conversion makes a reservation on HOST, so the gate probes with that
// exact operation. This fills HOST by converting real batches (the production path) and verifies
// the GPU monitor backs off -- a reservation-count-only check would wrongly see headroom here.
TEST_CASE("monitor backs off when HOST is full of stored downgraded data (no disk)",
          "[downgrade_disk]")
{
  auto mem_mgr    = make_no_disk_pressure_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  rmm::cuda_stream stream;
  std::vector<const cucascade::memory::memory_space*> host_targets;
  for (auto* hs : mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST)) {
    host_targets.push_back(hs);
  }

  // Convert GPU batches to HOST until HOST can no longer accept one. Keep them alive so HOST stays
  // full via stored allocations (the conversion's transient reservation is gone afterwards).
  std::vector<std::shared_ptr<cucascade::data_batch>> on_host;
  for (int i = 0; i < 64; ++i) {
    auto batch = make_gpu_batch(*gpu_space, 100000);
    sirius::convertible_data_batch conv(batch);
    auto res = conv.convert(host_targets, stream, *mem_mgr, false);
    if (!res.has_value()) { break; }  // HOST full
    REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
    on_host.push_back(std::move(batch));
  }
  REQUIRE_FALSE(on_host.empty());
  // Confirm HOST is genuinely full: one more conversion must fail.
  {
    auto extra = make_gpu_batch(*gpu_space, 100000);
    sirius::convertible_data_batch conv(extra);
    REQUIRE_FALSE(conv.convert(host_targets, stream, *mem_mgr, false).has_value());
  }

  auto gpu_hold = gpu_space->make_reservation_or_null(128ull << 20);
  REQUIRE(gpu_hold);
  REQUIRE(gpu_space->should_downgrade_memory());

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto executor  = make_monitoring_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(300ms);
  size_t n1 = executor.monitor_requests_issued_for_testing();
  std::this_thread::sleep_for(300ms);
  size_t n2 = executor.monitor_requests_issued_for_testing();

  executor.stop();

  REQUIRE(n2 == n1);
  REQUIRE(n1 <= 1);
}

// A downgrade executor bound to a HOST source has no target when no disk is configured (HOST cannot
// downgrade to itself; processing_loop only adds HOST targets for GPU sources). Even under HOST
// memory pressure it must back off rather than fire monitor requests every cycle.
TEST_CASE("HOST-source monitor backs off when no disk is configured", "[downgrade_disk]")
{
  auto mem_mgr = make_no_disk_pressure_manager();

  auto* host_space = mem_mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  if (!host_space) {
    auto spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (!spaces.empty()) {
      host_space = const_cast<cucascade::memory::memory_space*>(spaces.front());
    }
  }
  REQUIRE(host_space != nullptr);

  // Create HOST pressure so should_downgrade_memory() on the HOST space is true.
  auto host_hold = host_space->make_reservation_or_null(1ull << 20);
  REQUIRE(host_hold);
  REQUIRE(host_space->should_downgrade_memory());

  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade-host"},
    .monitor_period = std::chrono::milliseconds{10}};
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  downgrade_executor executor(config, repo_registry, host_space->get_id(), host_space, *mem_mgr);
  executor.start();

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(300ms);
  size_t n1 = executor.monitor_requests_issued_for_testing();
  std::this_thread::sleep_for(300ms);
  size_t n2 = executor.monitor_requests_issued_for_testing();

  executor.stop();

  REQUIRE(n2 == n1);  // HOST source with no disk has no target -> never fires
  REQUIRE(n1 == 0);
}
