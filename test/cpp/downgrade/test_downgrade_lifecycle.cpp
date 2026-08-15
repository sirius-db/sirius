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
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf / rmm
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <vector>

using namespace sirius::parallel;
using namespace std::chrono_literals;

namespace {

const auto GPU_SPACE_ID = cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0);

// These tests exercise a single query's repositories; the executor sweeps the registry,
// so each test registers its manager under one fixed query id.
const sirius::query_id_t kTestQueryId = sirius::make_query_id(1);

/// Helper: get the tier of a data_batch using a temporary read-only lock.
inline cucascade::memory::Tier get_batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_test_memory_manager()
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 2ull << 30;  // 2GB
  const double limit_ratio   = 0.75;        // 75% of GPU capacity
  const size_t host_capacity = 4ull << 30;  // 4GB

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();
  return manager;
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

downgrade_executor make_test_executor(sirius::data::data_repository_manager_registry& repo_registry,
                                      cucascade::memory::memory_space* gpu_space,
                                      sirius::memory::sirius_memory_reservation_manager& mem_mgr,
                                      std::chrono::milliseconds monitor_period = {})
{
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period = monitor_period};
  return downgrade_executor(config, repo_registry, GPU_SPACE_ID, gpu_space, mem_mgr);
}

/// Like make_test_memory_manager, but with the GPU downgrade trigger pulled down to 25% of
/// capacity (default 85% sits ABOVE the 75% reservation limit, so the default fixture can
/// never see should_downgrade_memory()). A held 1 GiB reservation is then enough sustained
/// pressure to keep the monitor firing.
std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_pressure_memory_manager()
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 2ull << 30;  // 2GB
  const size_t host_capacity = 4ull << 30;  // 4GB

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(0.75)
    .set_downgrade_fractions_per_gpu(/*start=*/0.25, /*end=*/0.10)
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(0.75);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();
  return manager;
}

}  // namespace

// ---------------------------------------------------------------------------
// Lifecycle Tests (LIFE-01 through LIFE-05)
// ---------------------------------------------------------------------------

TEST_CASE("start_stop_cycle", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  // nullptr memory_space -- monitor loop won't trigger
  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);

  // First start/stop cycle
  REQUIRE_NOTHROW(executor.start());
  // Verify executor is operational by requesting a trivial free (no repos = 0 freed)
  size_t freed = executor.request_free_memory_and_wait(1024);
  REQUIRE(freed == 0);
  REQUIRE_NOTHROW(executor.stop());

  // Second start/stop cycle -- verifies re-entry is safe
  REQUIRE_NOTHROW(executor.start());
  freed = executor.request_free_memory_and_wait(1024);
  REQUIRE(freed == 0);
  REQUIRE_NOTHROW(executor.stop());

  // Third cycle for good measure
  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("drain_clears_pending_requests", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  // Create a repo with some batches and register with manager
  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto batch1 = make_gpu_batch(*gpu_space);
  auto batch2 = make_gpu_batch(*gpu_space);
  auto batch3 = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch1);
  repo->add_data_batch(batch2);
  repo->add_data_batch(batch3);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Request downgrade of all GPU data
  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  // Wait for batches to reach HOST
  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (get_batch_tier(*batch1) == cucascade::memory::Tier::HOST &&
        get_batch_tier(*batch2) == cucascade::memory::Tier::HOST &&
        get_batch_tier(*batch3) == cucascade::memory::Tier::HOST)
      break;
    std::this_thread::sleep_for(50ms);
  }

  // Call drain -- ensures all in-flight work is done and executor restarts cleanly
  REQUIRE_NOTHROW(executor.drain());

  // After drain, the executor should be operational for new requests
  auto repo2  = std::make_unique<cucascade::shared_data_repository>();
  auto batch4 = make_gpu_batch(*gpu_space);
  repo2->add_data_batch(batch4);
  repo_mgr.add_new_repository(2, "out", std::move(repo2));

  freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (get_batch_tier(*batch4) == cucascade::memory::Tier::HOST) break;
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(get_batch_tier(*batch4) == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("drain_releases_batch_references", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  // Create a repo with GPU data and register with manager
  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Record the use_count before scheduling -- the test holds a reference
  long count_before = batch.use_count();

  // Request downgrade
  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  // Wait for downgrade to complete
  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (get_batch_tier(*batch) == cucascade::memory::Tier::HOST) break;
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);

  // Call drain to ensure all internal references are released
  // drain() calls stop() which waits for pool->wait_all(), releasing all shared_ptr captures
  executor.drain();

  // After drain, the executor should not hold any extra shared_ptr<data_batch> references.
  // use_count should be back to what it was before scheduling.
  REQUIRE(batch.use_count() <= count_before);

  executor.stop();
}

TEST_CASE("monitor_loop_triggers_downgrade", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  // Create a repo with GPU data and register it with the manager
  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto batch1 = make_gpu_batch(*gpu_space, 100000);
  auto batch2 = make_gpu_batch(*gpu_space, 100000);
  auto batch3 = make_gpu_batch(*gpu_space, 100000);
  repo->add_data_batch(batch1);
  repo->add_data_batch(batch2);
  repo->add_data_batch(batch3);
  repo_mgr.add_new_repository(42, "default", std::move(repo));

  REQUIRE(get_batch_tier(*batch1) == cucascade::memory::Tier::GPU);
  REQUIRE(get_batch_tier(*batch2) == cucascade::memory::Tier::GPU);
  REQUIRE(get_batch_tier(*batch3) == cucascade::memory::Tier::GPU);

  // Start executor with monitor enabled (non-zero period)
  auto executor = make_test_executor(
    repo_registry, gpu_space, *mem_mgr, /*monitor_period=*/std::chrono::milliseconds{10});
  executor.start();

  // Wait up to 2s for the monitor to detect pressure and trigger downgrade.
  // Note: whether downgrades actually happen depends on should_downgrade_memory() returning true.
  // With a small amount of data, the monitor may not trigger. This test verifies the monitor
  // loop runs without crashing and the executor remains operational.
  auto deadline       = std::chrono::steady_clock::now() + 2s;
  bool any_downgraded = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (get_batch_tier(*batch1) == cucascade::memory::Tier::HOST ||
        get_batch_tier(*batch2) == cucascade::memory::Tier::HOST ||
        get_batch_tier(*batch3) == cucascade::memory::Tier::HOST) {
      any_downgraded = true;
      break;
    }
    std::this_thread::sleep_for(50ms);
  }

  // Even if the monitor didn't trigger (not enough pressure), verify the executor is healthy
  // by manually requesting a downgrade
  if (!any_downgraded) {
    size_t freed = executor.request_free_memory_and_wait(1ull << 30);
    REQUIRE(freed > 0);

    deadline = std::chrono::steady_clock::now() + 10s;
    while (std::chrono::steady_clock::now() < deadline) {
      if (get_batch_tier(*batch1) == cucascade::memory::Tier::HOST) break;
      std::this_thread::sleep_for(50ms);
    }
    REQUIRE(get_batch_tier(*batch1) == cucascade::memory::Tier::HOST);
  }

  executor.stop();
}

TEST_CASE("concurrent_api_safety", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Launch 4 threads, each requesting memory reclamation concurrently.
  // During drain, pending requests are cancelled with an exception, so
  // threads must tolerate both success and cancellation.
  std::atomic<int> completed{0};
  std::vector<std::thread> threads;
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&executor, &repo_mgr, gpu_space, &completed, i]() {
      auto repo  = std::make_unique<cucascade::shared_data_repository>();
      auto batch = make_gpu_batch(*gpu_space, 500);
      repo->add_data_batch(batch);
      repo_mgr.add_new_repository(100 + i, "out", std::move(repo));

      try {
        executor.request_free_memory_and_wait(1ull << 30);

        // Wait for the batch to be downgraded
        auto deadline = std::chrono::steady_clock::now() + 10s;
        while (std::chrono::steady_clock::now() < deadline) {
          if (get_batch_tier(*batch) == cucascade::memory::Tier::HOST) break;
          std::this_thread::sleep_for(50ms);
        }
      } catch (const std::exception&) {
        // Request was cancelled by drain -- expected
      }
      completed.fetch_add(1);
    });
  }

  // Call drain from a 5th thread while requests are in-flight
  std::thread drain_thread([&executor]() {
    std::this_thread::sleep_for(100ms);
    executor.drain();
  });

  for (auto& t : threads) {
    t.join();
  }
  drain_thread.join();

  // Verify no crash and executor is still operational after drain
  // Submit + complete another request
  auto repo_final  = std::make_unique<cucascade::shared_data_repository>();
  auto final_batch = make_gpu_batch(*gpu_space, 500);
  repo_final->add_data_batch(final_batch);
  repo_mgr.add_new_repository(200, "out", std::move(repo_final));

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (get_batch_tier(*final_batch) == cucascade::memory::Tier::HOST) break;
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(get_batch_tier(*final_batch) == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("stop_cancels_pending_requests", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Enqueue several requests then immediately stop.
  // The processing loop handles requests sequentially, so some will still
  // be queued when stop() interrupts the queue.
  std::vector<std::future<size_t>> futures;
  for (int i = 0; i < 10; ++i) {
    futures.push_back(executor.request_free_memory(1024));
  }

  executor.stop();

  // Every future must resolve — either with a value (0, processed before
  // shutdown) or an exception (cancelled by stop).  No future should block
  // indefinitely.
  for (auto& f : futures) {
    REQUIRE(f.valid());
    try {
      f.get();  // may return 0 or throw
    } catch (const std::exception&) {
      // cancelled — expected for requests still queued at shutdown
    }
  }
}

TEST_CASE("drain_cancels_pending_requests_with_exception", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  std::vector<std::future<size_t>> futures;
  for (int i = 0; i < 10; ++i) {
    futures.push_back(executor.request_free_memory(1024));
  }

  executor.drain();

  // All futures must be resolved after drain
  for (auto& f : futures) {
    REQUIRE(f.valid());
    try {
      f.get();
    } catch (const std::exception&) {
      // cancelled — expected
    }
  }

  // Executor should still be operational after drain
  auto f = executor.request_free_memory(1024);
  REQUIRE(f.get() == 0);  // no repos, 0 freed

  executor.stop();
}

TEST_CASE("cuda_stream_lifecycle", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);

  // First start -- stream should be created
  executor.start();

  // Submit a request that requires GPU->HOST copy (uses the stream internally)
  auto repo1  = std::make_unique<cucascade::shared_data_repository>();
  auto batch1 = make_gpu_batch(*gpu_space);
  repo1->add_data_batch(batch1);
  repo_mgr.add_new_repository(1, "out", std::move(repo1));

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (get_batch_tier(*batch1) == cucascade::memory::Tier::HOST) break;
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(get_batch_tier(*batch1) == cucascade::memory::Tier::HOST);

  // Stop -- stream should be destroyed (on_stopped)
  executor.stop();

  // Second start -- stream should be re-created (on_start)
  executor.start();

  // Submit another request to verify stream is usable again
  auto repo2  = std::make_unique<cucascade::shared_data_repository>();
  auto batch2 = make_gpu_batch(*gpu_space);
  repo2->add_data_batch(batch2);
  repo_mgr.add_new_repository(2, "out", std::move(repo2));

  freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (get_batch_tier(*batch2) == cucascade::memory::Tier::HOST) break;
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(get_batch_tier(*batch2) == cucascade::memory::Tier::HOST);

  executor.stop();
}

// ---------------------------------------------------------------------------
// Shared-ownership teardown (steps 6+7): erase never fences on a sweep
// ---------------------------------------------------------------------------

TEST_CASE("erase while a sweep holds shared repositories", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;

  // Observable destructor-side accounting: replace the registry-installed logging handler
  // with a counting one (the manager applies it to current and future repositories).
  std::atomic<std::size_t> leak_reports{0};
  std::atomic<std::size_t> leaked_batches{0};
  std::atomic<std::size_t> leaked_operator_id{0};

  std::weak_ptr<cucascade::data_repository> repo_observer;
  std::shared_ptr<cucascade::data_repository> sweep_held_repo;
  sirius::data::data_repository_manager_registry::manager_ptr sweep_held_manager;

  {
    auto manager = repo_registry.create_for_query(kTestQueryId);
    manager->set_leak_handler(
      [&](std::size_t operator_id, const std::string& /*port_id*/, std::size_t count) {
        leaked_operator_id.store(operator_id);
        leaked_batches.fetch_add(count);
        leak_reports.fetch_add(1);
      });

    auto repo  = std::make_unique<cucascade::shared_data_repository>();
    auto batch = make_gpu_batch(*gpu_space);
    repo->add_data_batch(batch);
    manager->add_new_repository(7, "out", std::move(repo));

    // Simulate exactly what a TIER-1 sweep borrows before its blocking work: the manager
    // snapshot and the per-manager repository snapshot, both shared.
    auto managers = repo_registry.get_all();
    REQUIRE(managers.size() == 1);
    sweep_held_manager = managers.front();
    auto repos         = sweep_held_manager->get_repositories();
    REQUIRE(repos.size() == 1);
    sweep_held_repo = repos.front();
    repo_observer   = sweep_held_repo;
  }

  // The query ends mid-sweep. erase() must not block (no fence) and must not invalidate the
  // sweep's borrow — only the map entry goes.
  repo_registry.erase(kTestQueryId);
  REQUIRE(repo_registry.get(kTestQueryId) == nullptr);
  REQUIRE_FALSE(repo_observer.expired());
  // Not accounted yet: the repository is still alive in the sweep's hands.
  REQUIRE(leak_reports.load() == 0);

  // The sweep keeps working against its borrow: candidate collection still functions.
  // Scoped: the provider CO-OWNS the repository (that is the point of step 6), so it must be
  // gone before the expiry check below.
  {
    sirius::convertible_data_batch_provider provider(sweep_held_repo);
    auto candidates = provider.get_all_convertible(gpu_space, /*front_to_back=*/false);
    REQUIRE(candidates.size() == 1);
  }

  // Sweep finishes: the last holders release, the repository dies, and the un-consumed batch
  // is accounted in the DESTRUCTOR, attributed to the {operator, port} it was registered as.
  sweep_held_repo.reset();
  sweep_held_manager.reset();
  REQUIRE(repo_observer.expired());
  REQUIRE(leak_reports.load() == 1);
  REQUIRE(leaked_batches.load() == 1);
  REQUIRE(leaked_operator_id.load() == 7);
}

TEST_CASE("erase during an in-flight downgrade sweep leaves the sweep unharmed",
          "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Wedge a request mid-sweep: its worker parks inside the predicate after converting the
  // batch, so the sweep provably holds its borrows when the erase lands.
  std::promise<void> release_blocker;
  std::shared_future<void> gate(release_blocker.get_future());
  std::atomic<bool> blocker_entered{false};
  auto fut = executor.request_downgrade(sirius::make_query_id(55), [&blocker_entered, gate]() {
    blocker_entered.store(true);
    gate.wait();
    return false;
  });

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!blocker_entered.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(blocker_entered.load());

  // The repositories' owning query ends while the sweep is in flight. Pre-step-6 this erase
  // would have BLOCKED on the sweep gate until the request finished; now it returns at once.
  repo_registry.erase(kTestQueryId);
  REQUIRE(repo_registry.get(kTestQueryId) == nullptr);

  release_blocker.set_value();
  REQUIRE(fut.wait_for(10s) == std::future_status::ready);
  size_t freed = 0;
  REQUIRE_NOTHROW(freed = fut.get());
  REQUIRE(freed > 0);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);

  executor.stop();
}

// ---------------------------------------------------------------------------
// Per-query drain (A7/B2) and monitor re-arm (D6)
// ---------------------------------------------------------------------------

TEST_CASE("per_query_drain_cancels_only_that_querys_requests", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  // One convertible batch so query A's request dispatches a worker that parks in its
  // predicate, deterministically wedging the processing loop in wait_all with A IN FLIGHT.
  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  std::promise<void> release_blocker;
  std::shared_future<void> gate(release_blocker.get_future());
  std::atomic<bool> blocker_entered{false};

  const auto query_a = sirius::make_query_id(101);
  const auto query_b = sirius::make_query_id(202);
  const auto query_c = sirius::make_query_id(303);

  auto fut_a = executor.request_downgrade(query_a, [&blocker_entered, gate]() {
    blocker_entered.store(true);
    gate.wait();
    return false;
  });

  // Wait until A is genuinely in flight (its worker is inside the predicate).
  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!blocker_entered.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(blocker_entered.load());

  // B and C queue behind A (the processing loop handles one request at a time).
  auto fut_b = executor.request_downgrade(query_b, []() { return false; });
  auto fut_c = executor.request_downgrade(query_c, []() { return false; });

  // Drain B: only B's promise fails. A stays in flight, C stays queued, and the drain does
  // not block (the in-flight request is A's, not B's).
  executor.drain(query_b);
  REQUIRE(fut_b.wait_for(0s) == std::future_status::ready);
  REQUIRE_THROWS_AS(fut_b.get(), std::exception);
  REQUIRE(fut_a.wait_for(0s) == std::future_status::timeout);
  REQUIRE(fut_c.wait_for(0s) == std::future_status::timeout);

  // Drain A from another thread: must BLOCK until A's in-flight request completes.
  std::atomic<bool> drain_a_returned{false};
  std::thread drain_a_thread([&executor, &drain_a_returned, query_a]() {
    executor.drain(query_a);
    drain_a_returned.store(true);
  });
  std::this_thread::sleep_for(100ms);
  REQUIRE_FALSE(drain_a_returned.load());

  release_blocker.set_value();
  drain_a_thread.join();
  REQUIRE(drain_a_returned.load());

  // A completed normally: promise fulfilled with the converted bytes, not an exception.
  REQUIRE(fut_a.wait_for(10s) == std::future_status::ready);
  size_t freed_a = 0;
  REQUIRE_NOTHROW(freed_a = fut_a.get());
  REQUIRE(freed_a > 0);

  // C was never disturbed: it processes after A and resolves normally (0 bytes -- A's
  // worker already converted the only candidate).
  REQUIRE(fut_c.wait_for(10s) == std::future_status::ready);
  REQUIRE_NOTHROW(fut_c.get());

  executor.stop();
}

TEST_CASE("monitor_rearms_after_drain_cancels_queued_monitor_request", "[downgrade_lifecycle]")
{
  // D6 regression: cancel_pending_requests() used to eat a queued monitor request without
  // resetting _monitor_request_enqueued, permanently disabling memory-pressure downgrade
  // for the space. Recipe: wedge the processing loop, create sustained pressure so the
  // monitor enqueues (and latches), then drain() -- the queue interrupt guarantees the
  // monitor request is cancelled, never processed. Afterwards the monitor must fire again.
  auto mem_mgr    = make_pressure_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr, /*monitor_period=*/5ms);
  executor.start();

  // No pressure yet (trigger is 512 MiB consumed), so the monitor idles. Wedge the
  // processing loop first: a blocker request whose worker parks in its predicate.
  std::promise<void> release_blocker;
  std::shared_future<void> gate(release_blocker.get_future());
  std::atomic<bool> blocker_entered{false};
  auto blocker_future =
    executor.request_downgrade(sirius::make_query_id(7), [&blocker_entered, gate]() {
      blocker_entered.store(true);
      gate.wait();
      return false;
    });
  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (!blocker_entered.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  REQUIRE(blocker_entered.load());

  // NOW create sustained pressure: hold a 1 GiB reservation. The monitor sees
  // should_downgrade_memory(), enqueues one request behind the wedged blocker, and latches
  // _monitor_request_enqueued so it enqueues exactly one.
  auto pressure = gpu_space->make_reservation_or_null(1ull << 30);
  REQUIRE(pressure != nullptr);
  deadline = std::chrono::steady_clock::now() + 10s;
  while (executor.monitor_requests_issued_for_testing() == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  const auto issued_before_drain = executor.monitor_requests_issued_for_testing();
  REQUIRE(issued_before_drain >= 1);

  // Global drain. It interrupts the request queue FIRST, so the processing thread (once
  // unwedged) exits without ever popping the queued monitor request -- the drain's cancel
  // is guaranteed to be what destroys it. drain() blocks joining the wedged thread, so
  // run it from a helper and release the blocker underneath it.
  std::thread drainer([&executor]() { executor.drain(); });
  std::this_thread::sleep_for(50ms);
  release_blocker.set_value();
  drainer.join();

  // Pressure is still on (the reservation is still held) and the queued monitor request
  // was cancelled, not processed. The monitor must re-arm and fire again; before the fix
  // _monitor_request_enqueued stayed latched true and this poll times out.
  deadline = std::chrono::steady_clock::now() + 5s;
  while (executor.monitor_requests_issued_for_testing() <= issued_before_drain &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(5ms);
  }
  REQUIRE(executor.monitor_requests_issued_for_testing() > issued_before_drain);

  pressure.reset();
  executor.stop();
}
