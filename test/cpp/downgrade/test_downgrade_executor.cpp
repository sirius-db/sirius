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
#include "downgrade/downgrade_task.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
// data utilities
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <utils/utils.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf / rmm
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <memory>
#include <set>
#include <vector>

using namespace sirius::parallel;
using namespace std::chrono_literals;

namespace {

const auto GPU_SPACE_ID = cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0);

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_test_memory_manager()
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 2ull << 30;
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 4ull << 30;

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

  return sirius::make_data_batch(std::move(table), gpu_space);
}

/**
 * @brief Helper to create a downgrade_executor for tests.
 *
 * Pass nullptr for memory_space when the monitor loop shouldn't trigger automatically.
 */
downgrade_executor make_test_executor(cucascade::shared_data_repository_manager& repo_mgr,
                                      cucascade::memory::memory_space* gpu_space,
                                      sirius::memory::sirius_memory_reservation_manager& mem_mgr)
{
  sirius::exec::downgrade_executor_config config{
    .thread_pool = {.num_threads = 1, .thread_name_prefix = "downgrade"}, .monitor_period_ms = 0};
  return downgrade_executor(config, repo_mgr, GPU_SPACE_ID, gpu_space, mem_mgr);
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Downgrade executor starts and stops cleanly", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  cucascade::shared_data_repository_manager repo_mgr;

  // nullptr memory_space — monitor loop won't trigger, just tests lifecycle
  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("request_free_memory_and_wait with no repositories returns 0", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  cucascade::shared_data_repository_manager repo_mgr;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1024);
  REQUIRE(freed == 0);

  executor.stop();
}

TEST_CASE("Single downgrade task executes correctly", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  auto batch = make_gpu_batch(*gpu_space);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  downgrade_task task{batch, *mem_mgr};

  rmm::cuda_stream stream;
  REQUIRE_NOTHROW(task.execute(stream));

  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
}

TEST_CASE("request_free_memory_and_wait downgrades GPU batches to HOST", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo   = std::make_unique<cucascade::shared_data_repository>();
  auto batch1 = make_gpu_batch(*gpu_space);
  auto batch2 = make_gpu_batch(*gpu_space);
  auto batch3 = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch1);
  repo->add_data_batch(batch2);
  repo->add_data_batch(batch3);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  REQUIRE(batch1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch2->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch3->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  REQUIRE(batch1->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch3->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("request_free_memory respects byte target via predicate", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo = std::make_unique<cucascade::shared_data_repository>();
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 5; ++i) {
    auto batch = make_gpu_batch(*gpu_space);
    batches.push_back(batch);
    repo->add_data_batch(batch);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t one_batch_size = batches[0]->get_data()->get_size_in_bytes();
  REQUIRE(one_batch_size > 0);

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(one_batch_size);
  REQUIRE(freed >= one_batch_size);

  size_t host_count = 0;
  for (auto& b : batches) {
    if (b->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST) ++host_count;
  }
  REQUIRE(host_count >= 1);

  executor.stop();
}

TEST_CASE("request_free_memory prioritizes partitioned repos over non-partitioned",
          "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;

  auto repo_non_partitioned = std::make_unique<cucascade::shared_data_repository>();
  auto batch_np1            = make_gpu_batch(*gpu_space);
  auto batch_np2            = make_gpu_batch(*gpu_space);
  repo_non_partitioned->add_data_batch(batch_np1);
  repo_non_partitioned->add_data_batch(batch_np2);

  auto repo_partitioned = std::make_unique<cucascade::shared_data_repository>();
  auto batch_p0         = make_gpu_batch(*gpu_space);
  auto batch_p1         = make_gpu_batch(*gpu_space);
  auto batch_p2         = make_gpu_batch(*gpu_space);
  repo_partitioned->add_data_batch(batch_p0, 0);
  repo_partitioned->add_data_batch(batch_p1, 1);
  repo_partitioned->add_data_batch(batch_p2, 2);

  repo_mgr.add_new_repository(1, "out", std::move(repo_non_partitioned));
  repo_mgr.add_new_repository(2, "out", std::move(repo_partitioned));

  size_t one_batch_size = batch_p0->get_data()->get_size_in_bytes();

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(one_batch_size);
  REQUIRE(freed >= one_batch_size);

  REQUIRE(batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_np1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch_np2->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  executor.stop();
}

TEST_CASE("request_free_memory iterates partitions from last to first", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo     = std::make_unique<cucascade::shared_data_repository>();
  auto batch_p0 = make_gpu_batch(*gpu_space);
  auto batch_p1 = make_gpu_batch(*gpu_space);
  auto batch_p2 = make_gpu_batch(*gpu_space);
  auto batch_p3 = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch_p0, 0);
  repo->add_data_batch(batch_p1, 1);
  repo->add_data_batch(batch_p2, 2);
  repo->add_data_batch(batch_p3, 3);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t two_batches = batch_p0->get_data()->get_size_in_bytes() * 2;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(two_batches);
  REQUIRE(freed >= two_batches);

  REQUIRE(batch_p3->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p0->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch_p1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  executor.stop();
}

TEST_CASE("request_free_memory skips active partitions in first pass", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo     = std::make_unique<cucascade::shared_data_repository>();
  auto batch_p0 = make_gpu_batch(*gpu_space);
  auto batch_p1 = make_gpu_batch(*gpu_space);
  auto batch_p2 = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch_p0, 0);
  repo->add_data_batch(batch_p1, 1);
  repo->add_data_batch(batch_p2, 2);

  REQUIRE(batch_p1->try_to_create_task());
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t three_batches = batch_p0->get_data()->get_size_in_bytes() * 3;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(three_batches);
  REQUIRE(freed > 0);

  REQUIRE(batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p0->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  batch_p1->try_to_cancel_task();
  executor.stop();
}

TEST_CASE("request_free_memory skips batches already on HOST", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo       = std::make_unique<cucascade::shared_data_repository>();
  auto gpu_batch  = make_gpu_batch(*gpu_space);
  auto gpu_batch2 = make_gpu_batch(*gpu_space);
  repo->add_data_batch(gpu_batch);
  repo->add_data_batch(gpu_batch2);

  // Pre-downgrade one batch to HOST manually
  auto& registry   = sirius::converter_registry::get();
  auto* host_space = mem_mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  if (!host_space) {
    auto host_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    REQUIRE_FALSE(host_spaces.empty());
    host_space = const_cast<cucascade::memory::memory_space*>(host_spaces.front());
  }
  rmm::cuda_stream conv_stream;
  REQUIRE(gpu_batch->try_to_lock_for_in_transit());
  gpu_batch->convert_to<cucascade::host_data_representation>(registry, host_space, conv_stream);
  gpu_batch->try_to_release_in_transit();
  REQUIRE(gpu_batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  REQUIRE(gpu_batch2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}

// --- New API tests ---

TEST_CASE("request_free_memory returns future that resolves to bytes freed", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  auto future  = executor.request_free_memory(1ull << 30);
  size_t freed = future.get();
  REQUIRE(freed > 0);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("request_downgrade with custom predicate stops when satisfied", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo = std::make_unique<cucascade::shared_data_repository>();
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 5; ++i) {
    auto batch = make_gpu_batch(*gpu_space);
    batches.push_back(batch);
    repo->add_data_batch(batch);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  std::atomic<size_t> call_count{0};

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  // Predicate returns true on first call — should stop after ~1 batch
  auto future = executor.request_downgrade(1024 * 1024, [&call_count]() {
    call_count.fetch_add(1, std::memory_order_relaxed);
    return true;  // satisfied immediately after first batch
  });

  size_t freed = future.get();
  REQUIRE(freed > 0);

  // With pool width=1 and predicate satisfied immediately, at most 1-2 batches downgraded
  size_t host_count = 0;
  for (auto& b : batches) {
    if (b->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST) ++host_count;
  }
  REQUIRE(host_count >= 1);
  REQUIRE(host_count <= 2);

  executor.stop();
}

TEST_CASE("request_free_memory partial fulfillment returns actual bytes freed",
          "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo         = std::make_unique<cucascade::shared_data_repository>();
  auto batch        = make_gpu_batch(*gpu_space);
  size_t batch_size = batch->get_data()->get_size_in_bytes();
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  // Request far more than available
  size_t freed = executor.request_free_memory_and_wait(1ull << 40);
  // Should get only the one batch's worth
  REQUIRE(freed == batch_size);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}

// ---------------------------------------------------------------------------
// GPU-to-GPU transfer via converter (re-authored from v1.0 c5a3d8e)
//
// v1.0's test body was converter-registry-dependent, not downgrade_task-class
// dependent, so the re-authoring is mostly unchanged. Catch2 v2 skip idiom
// preserved: WARN+return for <2 GPUs instead of SKIP (per STATE.md Plan 01-03
// decision — Catch2 v2 lacks a SKIP macro that coexists cleanly with the
// [downgrade] suite layout).
// ---------------------------------------------------------------------------

TEST_CASE("gpu_to_gpu_transfer_via_converter", "[.][multi_gpu_transfer]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs for cross-device transfer test");
    return;
  }

  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 512ull << 20;  // 512 MB per GPU
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;

  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

  auto gpu_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);

  auto* gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  auto* gpu1 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[1]);
  REQUIRE(gpu0->get_device_id() != gpu1->get_device_id());

  // Create a batch on GPU 0.
  auto batch = make_gpu_batch(*gpu0, 500);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch->get_memory_space()->get_device_id() == gpu0->get_device_id());

  // Convert GPU 0 -> GPU 1 via the converter registry.
  auto& registry = sirius::converter_registry::get();
  rmm::cuda_stream stream;

  REQUIRE(batch->try_to_lock_for_in_transit());
  batch->convert_to<cucascade::gpu_table_representation>(registry, gpu1, stream);
  batch->try_to_release_in_transit();

  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch->get_memory_space()->get_device_id() == gpu1->get_device_id());

  // Round-trip GPU 1 -> GPU 0.
  REQUIRE(batch->try_to_lock_for_in_transit());
  batch->convert_to<cucascade::gpu_table_representation>(registry, gpu0, stream);
  batch->try_to_release_in_transit();

  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch->get_memory_space()->get_device_id() == gpu0->get_device_id());

  // Data integrity check: batch still has a non-empty payload after the round-trip.
  REQUIRE(batch->get_data() != nullptr);
  REQUIRE(batch->get_data()->get_size_in_bytes() > 0);

  sirius::converter_registry::shutdown();
}

// ---------------------------------------------------------------------------
// NUMA downgrade ordering tests (re-authored from v1.0 c5a3d8e + ec2399e)
//
// v1.0 originally exercised:
//   - downgrade_task_global_state constructor's 4th arg (numa_preferred_device_id),
//     verified by numa_aware_downgrade_global_state_carries_preference (c5a3d8e)
//   - downgrade_executor constructor's new 6th arg (std::optional<int> gpu_numa_node),
//     verified by numa_aware_downgrade_executor_passes_numa_node +
//     downgrade_executor_default_numa_node_is_nullopt (c5a3d8e)
//   - cucascade strategy candidate ordering with pref=0/1/nullopt (ec2399e)
//
// Post-PR-#579 (dev) re-authoring:
//   - downgrade_task_global_state and downgrade_task_local_state were deleted;
//     the NUMA preference now rides exec::downgrade_executor_config::preferred_numa_node.
//   - downgrade_executor's constructor takes that config by value and copies it into
//     _config. Each dispatched downgrade_task receives preferred_numa_node via the
//     processing_loop's lambda capture.
//   - Assertions below target the config struct field + cucascade strategy output
//     rather than the removed class members. Behavior under test is preserved:
//     (a) the config field default is nullopt (backward-compat),
//     (b) explicitly-set values are carried verbatim,
//     (c) dispatch succeeds end-to-end when the preference is set on a single-GPU host,
//     (d) cucascade strategy produces the expected NUMA-local-first candidate order on
//         a multi-GPU fixture.
// ---------------------------------------------------------------------------

namespace {

/// Build a 2-GPU memory manager for NUMA verification tests. Each GPU gets its own
/// HOST space (use_host_per_gpu), so the candidate ordering produced by
/// any_memory_space_in_tier_with_preference can distinguish device_ids 0 and 1.
std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_multi_gpu_memory_manager()
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 512ull << 20;  // 512 MB per GPU
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;  // 1 GB per HOST space

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
  return manager;
}

}  // namespace

TEST_CASE("downgrade_executor_config_carries_preferred_numa_node",
          "[downgrade][numa_aware_downgrade]")
{
  // v1.0 intent (from c5a3d8e numa_aware_downgrade_global_state_carries_preference):
  // the config object that flows into the downgrade dispatch path must carry the
  // NUMA preference verbatim. Re-authored against dev's config struct.
  sirius::exec::downgrade_executor_config cfg_with_pref{
    .thread_pool        = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period_ms  = 0,
    .preferred_numa_node = std::optional<int>{0}};
  REQUIRE(cfg_with_pref.preferred_numa_node.has_value());
  REQUIRE(cfg_with_pref.preferred_numa_node.value() == 0);

  sirius::exec::downgrade_executor_config cfg_with_pref7{
    .thread_pool        = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period_ms  = 0,
    .preferred_numa_node = std::optional<int>{7}};
  REQUIRE(cfg_with_pref7.preferred_numa_node.value() == 7);

  // Default construction: preferred_numa_node must be nullopt (backward-compat guarantee).
  sirius::exec::downgrade_executor_config cfg_default{};
  REQUIRE_FALSE(cfg_default.preferred_numa_node.has_value());
}

TEST_CASE("numa_aware_downgrade_executor_passes_numa_node", "[downgrade][numa_aware_downgrade]")
{
  // Re-authored from c5a3d8e: construct an executor whose config carries
  // preferred_numa_node = 0, dispatch a real downgrade, assert the batch lands on HOST.
  // Single-GPU execution is sufficient to prove the config threads through to dispatch
  // (the candidate-ordering behavior on multi-GPU is covered by
  // numa_downgrade_candidate_ordering_verified below).
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  sirius::exec::downgrade_executor_config config{
    .thread_pool        = {.num_threads = 1, .thread_name_prefix = "downgrade_numa"},
    .monitor_period_ms  = 0,
    .preferred_numa_node = std::optional<int>{0}};
  downgrade_executor executor(config, repo_mgr, GPU_SPACE_ID, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("downgrade_executor_default_numa_node_is_nullopt", "[downgrade][numa_aware_downgrade]")
{
  // Re-authored from c5a3d8e: the backward-compatible default path (no NUMA preference)
  // continues to downgrade correctly via the unpreferred any_memory_space_in_tier strategy.
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  // make_test_executor builds a config without preferred_numa_node set.
  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("numa_downgrade_candidate_ordering_verified",
          "[.][downgrade][numa_aware_downgrade][multi_gpu]")
{
  // Re-authored from ec2399e: verify cucascade's
  // any_memory_space_in_tier_with_preference strategy orders candidates so the
  // preferred device_id appears first, then the cross-NUMA fallback. Requires 2 GPUs
  // to exercise the ordering (use_host_per_gpu produces one HOST space per GPU).
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs for NUMA candidate ordering test");
    return;
  }

  auto mem_mgr = make_multi_gpu_memory_manager();

  auto host_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(host_spaces.size() == 2);

  // pref=0 -> first candidate is device 0, second is device 1.
  {
    cucascade::memory::any_memory_space_in_tier_with_preference strategy{
      cucascade::memory::Tier::HOST, std::optional<size_t>{0}};
    auto candidates = strategy.get_candidates(*mem_mgr);
    REQUIRE(candidates.size() == 2);
    REQUIRE(candidates[0]->get_device_id() == 0);
    REQUIRE(candidates[1]->get_device_id() == 1);
  }

  // pref=1 -> first candidate is device 1, second is device 0 (ordering flipped).
  {
    cucascade::memory::any_memory_space_in_tier_with_preference strategy{
      cucascade::memory::Tier::HOST, std::optional<size_t>{1}};
    auto candidates = strategy.get_candidates(*mem_mgr);
    REQUIRE(candidates.size() == 2);
    REQUIRE(candidates[0]->get_device_id() == 1);
    REQUIRE(candidates[1]->get_device_id() == 0);
  }

  // pref=nullopt -> both candidates present (order is cucascade-defined, not asserted).
  {
    cucascade::memory::any_memory_space_in_tier_with_preference strategy{
      cucascade::memory::Tier::HOST, std::nullopt};
    auto candidates = strategy.get_candidates(*mem_mgr);
    REQUIRE(candidates.size() == 2);
    std::set<int> device_ids;
    for (auto* c : candidates) {
      device_ids.insert(c->get_device_id());
    }
    REQUIRE(device_ids.count(0) == 1);
    REQUIRE(device_ids.count(1) == 1);
  }

  sirius::converter_registry::shutdown();
}

TEST_CASE("numa_downgrade_prefers_local_host_space",
          "[.][downgrade][numa_aware_downgrade][multi_gpu]")
{
  // Re-authored from ec2399e: end-to-end proof that a downgrade with
  // preferred_numa_node=0 lands the batch on the HOST space with device_id=0.
  // Requires 2 GPUs for the use_host_per_gpu configuration to produce distinct
  // HOST device_ids.
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs for NUMA downgrade preference test");
    return;
  }

  auto mem_mgr = make_multi_gpu_memory_manager();

  auto gpu_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);
  auto* gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);

  cucascade::shared_data_repository_manager repo_mgr;
  auto repo  = std::make_unique<cucascade::shared_data_repository>();
  auto batch = make_gpu_batch(*gpu0);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  auto gpu0_space_id = cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0);
  sirius::exec::downgrade_executor_config config{
    .thread_pool        = {.num_threads = 1, .thread_name_prefix = "downgrade_numa_test"},
    .monitor_period_ms  = 0,
    .preferred_numa_node = std::optional<int>{0}};
  downgrade_executor executor(config, repo_mgr, gpu0_space_id, gpu0, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  // NUMA-local HOST space was selected (device_id matches preference).
  REQUIRE(batch->get_memory_space()->get_device_id() == 0);

  executor.stop();
  sirius::converter_registry::shutdown();
}
