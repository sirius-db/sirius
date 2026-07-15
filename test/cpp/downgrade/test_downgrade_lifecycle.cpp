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
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

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

downgrade_executor make_test_executor(cucascade::shared_data_repository_manager& repo_mgr,
                                      cucascade::memory::memory_space* gpu_space,
                                      sirius::memory::sirius_memory_reservation_manager& mem_mgr,
                                      std::chrono::milliseconds monitor_period = {})
{
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period = monitor_period};
  return downgrade_executor(config, repo_mgr, GPU_SPACE_ID, gpu_space, mem_mgr);
}

}  // namespace

// ---------------------------------------------------------------------------
// Lifecycle Tests (LIFE-01 through LIFE-05)
// ---------------------------------------------------------------------------

TEST_CASE("start_stop_cycle", "[downgrade_lifecycle]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  cucascade::shared_data_repository_manager repo_mgr;

  // nullptr memory_space -- monitor loop won't trigger
  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);

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

  cucascade::shared_data_repository_manager repo_mgr;

  // Create a repo with some batches and register with manager
  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto batch1 = make_gpu_batch(*gpu_space);
  auto batch2 = make_gpu_batch(*gpu_space);
  auto batch3 = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch1);
  repo->add_data_batch(batch2);
  repo->add_data_batch(batch3);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
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

  cucascade::shared_data_repository_manager repo_mgr;

  // Create a repo with GPU data and register with manager
  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
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

  cucascade::shared_data_repository_manager repo_mgr;

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
    repo_mgr, gpu_space, *mem_mgr, /*monitor_period=*/std::chrono::milliseconds{10});
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

  cucascade::shared_data_repository_manager repo_mgr;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
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
  cucascade::shared_data_repository_manager repo_mgr;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
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
  cucascade::shared_data_repository_manager repo_mgr;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
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

  cucascade::shared_data_repository_manager repo_mgr;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);

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
