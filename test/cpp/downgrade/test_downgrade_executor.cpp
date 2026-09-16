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
#include "data/data_repository_manager_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/partition/gpu_partition_impl.hpp"
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
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf / rmm
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace sirius::parallel;
using namespace std::chrono_literals;

namespace {

/// Helper: get the memory tier of a data_batch via to_read_only()
cucascade::memory::Tier get_batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

/// Helper: get byte size of a data_batch via to_read_only()
size_t get_batch_size(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_data()->get_size_in_bytes();
}

const auto GPU_SPACE_ID = cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0);

// These tests exercise a single query's repositories; the executor sweeps the registry,
// so each test registers its manager under one fixed query id.
const sirius::query_id_t kTestQueryId = sirius::make_query_id(1);

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
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();
  return manager;
}

// `memory_space` compares its memory limit minus available bytes with its downgrade thresholds. For
// this fixed-capacity allocator, available bytes equal capacity minus allocated bytes, so the
// equivalent allocated-byte boundary is capacity minus memory limit plus threshold.
constexpr size_t kPressureGpuCapacity = 256ull << 20;
constexpr size_t kPressureGpuLimit    = kPressureGpuCapacity * 9 / 10;
constexpr size_t kPressureStartBytes  = kPressureGpuCapacity / 10;
constexpr size_t kPressureStopBytes   = kPressureGpuCapacity / 20;
constexpr size_t kPartitionRows       = 1ull << 20;
constexpr int kPartitionCount         = 4;
constexpr size_t kPressureTriggerAllocated =
  kPressureGpuCapacity - kPressureGpuLimit + kPressureStartBytes;
constexpr size_t kPressureStoppedAllocated =
  kPressureGpuCapacity - kPressureGpuLimit + kPressureStopBytes;

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager>
make_partition_pressure_memory_manager()
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(kPressureGpuCapacity)
    .set_reservation_fraction_per_gpu(0.9)
    .set_downgrade_fractions_per_gpu(0.1, 0.05)
    .set_per_numa_region_capacity(512ull << 20)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(0.9);

  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
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

/**
 * @brief Helper to create a downgrade_executor for tests.
 *
 * Pass nullptr for memory_space when the monitor loop shouldn't trigger automatically.
 */
downgrade_executor make_test_executor(sirius::data::data_repository_manager_registry& repo_registry,
                                      cucascade::memory::memory_space* gpu_space,
                                      sirius::memory::sirius_memory_reservation_manager& mem_mgr)
{
  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period = std::chrono::milliseconds{0}};
  return downgrade_executor(config, repo_registry, GPU_SPACE_ID, gpu_space, mem_mgr);
}

// Captures a fixed-only partition family after its input is destroyed. `logical_bytes` is the sum
// charged to sibling batches; allocator snapshots show when shared column allocations are
// reclaimed.
struct fixed_partition_family {
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  size_t allocated_before{};
  size_t allocated_with_family{};
  size_t logical_bytes{};
};

fixed_partition_family make_fixed_partition_family(
  cucascade::memory::memory_space& gpu_space,
  cucascade::memory::reservation_aware_resource_adaptor& allocator)
{
  auto stream = cudf::get_default_stream();
  // Synchronize around allocation snapshots so pending writes and input deallocation do not
  // contaminate the family-level physical-memory measurement.
  stream.synchronize();
  auto const allocated_before = allocator.get_total_allocated_bytes();

  std::vector<cudf::data_type> column_types              = {cudf::data_type{cudf::type_id::INT64},
                                                            cudf::data_type{cudf::type_id::INT64}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {std::make_pair(0, 1000000),
                                                            std::make_pair(0, 1000000)};
  auto input_table = sirius::create_cudf_table_with_random_data(
    kPartitionRows, column_types, ranges, stream, gpu_space.get_default_allocator());
  auto input = sirius::make_data_batch(
    std::move(input_table), gpu_space, stream, sirius::telemetry::batch_telemetry_info{});

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  {
    auto ro = input->to_read_only();
    batches =
      sirius::op::gpu_partition_impl::hash_partition(ro, {0}, kPartitionCount, stream, gpu_space);
  }
  stream.synchronize();
  input.reset();
  stream.synchronize();

  REQUIRE(batches.size() == static_cast<size_t>(kPartitionCount));
  size_t logical_bytes = 0;
  for (auto const& batch : batches) {
    auto ro         = batch->to_read_only();
    auto const view = sirius::get_cudf_table_view(ro);
    REQUIRE(view.num_rows() > 0);
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.column(0).offset() == 0);
    REQUIRE(view.column(1).offset() == 0);
    logical_bytes += ro.get_data()->get_size_in_bytes();
  }
  REQUIRE(logical_bytes == kPartitionRows * 2 * sizeof(int64_t));

  auto const allocated_with_family = allocator.get_total_allocated_bytes();
  REQUIRE(allocated_with_family >= allocated_before + logical_bytes);
  return {std::move(batches), allocated_before, allocated_with_family, logical_bytes};
}

size_t count_batches_in_tier(const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
                             cucascade::memory::Tier tier)
{
  size_t count = 0;
  for (auto const& batch : batches) {
    if (get_batch_tier(*batch) == tier) { ++count; }
  }
  return count;
}

// Treat a batch locked by an in-flight conversion as not yet converted so monitor polling never
// blocks the test thread.
bool all_batches_in_tier_nonblocking(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches, cucascade::memory::Tier tier)
{
  for (auto const& batch : batches) {
    auto ro = batch->try_to_read_only();
    if (!ro || ro->get_memory_space()->get_tier() != tier) { return false; }
  }
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Downgrade executor starts and stops cleanly", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  // nullptr memory_space — monitor loop won't trigger, just tests lifecycle
  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("request_free_memory_and_wait with no repositories returns 0", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1024);
  REQUIRE(freed == 0);

  executor.stop();
}

TEST_CASE("request_free_memory_and_wait downgrades GPU batches to HOST", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  auto batch1    = make_gpu_batch(*gpu_space);
  auto batch2    = make_gpu_batch(*gpu_space);
  auto batch3    = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch1);
  repo->add_data_batch(batch2);
  repo->add_data_batch(batch3);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  REQUIRE(get_batch_tier(*batch1) == cucascade::memory::Tier::GPU);
  REQUIRE(get_batch_tier(*batch2) == cucascade::memory::Tier::GPU);
  REQUIRE(get_batch_tier(*batch3) == cucascade::memory::Tier::GPU);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  REQUIRE(get_batch_tier(*batch1) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*batch2) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*batch3) == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("request_free_memory respects byte target via predicate", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 5; ++i) {
    auto batch = make_gpu_batch(*gpu_space);
    batches.push_back(batch);
    repo->add_data_batch(batch);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t one_batch_size = get_batch_size(*batches[0]);
  REQUIRE(one_batch_size > 0);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(one_batch_size);
  REQUIRE(freed >= one_batch_size);

  size_t host_count = 0;
  for (auto& b : batches) {
    if (get_batch_tier(*b) == cucascade::memory::Tier::HOST) ++host_count;
  }
  REQUIRE(host_count >= 1);

  executor.stop();
}

// NOTE: The old scored_repo sort prioritized partitioned repos over non-partitioned.
// The new lazy tiered iteration processes repos in for_each_repository order, which
// follows insertion order. This test verifies the lazy iteration works correctly
// across multiple repos without asserting a specific priority ordering.
TEST_CASE("request_free_memory downgrades across multiple repos", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);

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

  size_t one_batch_size = get_batch_size(*batch_p0);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Request enough to downgrade at least one batch
  size_t freed = executor.request_free_memory_and_wait(one_batch_size);
  REQUIRE(freed >= one_batch_size);

  // At least one batch should have been downgraded
  size_t host_count = 0;
  for (auto* b : {&batch_np1, &batch_np2, &batch_p0, &batch_p1, &batch_p2}) {
    if (get_batch_tier(**b) == cucascade::memory::Tier::HOST) ++host_count;
  }
  REQUIRE(host_count >= 1);

  executor.stop();
}

TEST_CASE("request_free_memory iterates partitions from last to first", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  auto batch_p0  = make_gpu_batch(*gpu_space);
  auto batch_p1  = make_gpu_batch(*gpu_space);
  auto batch_p2  = make_gpu_batch(*gpu_space);
  auto batch_p3  = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch_p0, 0);
  repo->add_data_batch(batch_p1, 1);
  repo->add_data_batch(batch_p2, 2);
  repo->add_data_batch(batch_p3, 3);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t two_batches = get_batch_size(*batch_p0) * 2;

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(two_batches);
  REQUIRE(freed >= two_batches);

  REQUIRE(get_batch_tier(*batch_p3) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*batch_p2) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*batch_p0) == cucascade::memory::Tier::GPU);
  REQUIRE(get_batch_tier(*batch_p1) == cucascade::memory::Tier::GPU);

  executor.stop();
}

TEST_CASE("request_free_memory skips active partitions in first pass", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  auto batch_p0  = make_gpu_batch(*gpu_space);
  auto batch_p1  = make_gpu_batch(*gpu_space);
  auto batch_p2  = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch_p0, 0);
  repo->add_data_batch(batch_p1, 1);
  repo->add_data_batch(batch_p2, 2);

  // Lock batch_p1 in read_only state so downgrade executor skips it (state != idle)
  auto batch_p1_lock = batch_p1->to_read_only();
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t three_batches = get_batch_size(*batch_p0) * 3;

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(three_batches);
  REQUIRE(freed > 0);

  REQUIRE(get_batch_tier(*batch_p2) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*batch_p0) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*batch_p1) == cucascade::memory::Tier::GPU);

  // Release the read lock by moving it to a temporary that goes out of scope
  {
    auto discard = std::move(batch_p1_lock);
  }
  executor.stop();
}

TEST_CASE("request_free_memory skips batches already on HOST", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr  = *repo_registry.create_for_query(kTestQueryId);
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
  {
    // Acquire exclusive lock and convert to host representation
    auto mut = gpu_batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, host_space, conv_stream);
    // mut goes out of scope → releases exclusive lock, batch returns to idle
  }
  REQUIRE(get_batch_tier(*gpu_batch) == cucascade::memory::Tier::HOST);

  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  REQUIRE(get_batch_tier(*gpu_batch2) == cucascade::memory::Tier::HOST);

  executor.stop();
}

// --- New API tests ---

TEST_CASE("request_free_memory returns future that resolves to bytes freed", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  auto batch     = make_gpu_batch(*gpu_space);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  auto future  = executor.request_free_memory(1ull << 30);
  size_t freed = future.get();
  REQUIRE(freed > 0);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("request_downgrade with custom predicate stops when satisfied", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 5; ++i) {
    auto batch = make_gpu_batch(*gpu_space);
    batches.push_back(batch);
    repo->add_data_batch(batch);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  std::atomic<size_t> call_count{0};

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Predicate returns true on first call — should stop after ~1 batch
  auto future = executor.request_downgrade([&call_count]() {
    call_count.fetch_add(1, std::memory_order_relaxed);
    return true;  // satisfied immediately after first batch
  });

  size_t freed = future.get();
  REQUIRE(freed > 0);

  // With pool width=1 and predicate satisfied immediately, at most 1-2 batches downgraded
  size_t host_count = 0;
  for (auto& b : batches) {
    if (get_batch_tier(*b) == cucascade::memory::Tier::HOST) ++host_count;
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

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr    = *repo_registry.create_for_query(kTestQueryId);
  auto repo         = std::make_unique<cucascade::shared_data_repository>();
  auto batch        = make_gpu_batch(*gpu_space);
  size_t batch_size = get_batch_size(*batch);
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Request far more than available
  size_t freed = executor.request_free_memory_and_wait(1ull << 40);
  // Should get only the one batch's worth
  REQUIRE(freed == batch_size);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("shared fixed hash partitions release physical GPU bytes only with the last sibling",
          "[downgrade_executor][hash_partition][memory_pressure]")
{
  auto mem_mgr    = make_partition_pressure_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);
  auto* allocator =
    gpu_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(allocator != nullptr);

  auto family = make_fixed_partition_family(*gpu_space, *allocator);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  for (size_t partition = 0; partition < family.batches.size(); ++partition) {
    repo->add_data_batch(family.batches[partition], partition);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  size_t previous_allocated = family.allocated_with_family;
  size_t logical_freed      = 0;
  for (size_t expected_host_count = 1; expected_host_count <= family.batches.size();
       ++expected_host_count) {
    auto const freed = executor.request_free_memory_and_wait(1);
    logical_freed += freed;
    REQUIRE(freed > 0);
    REQUIRE(count_batches_in_tier(family.batches, cucascade::memory::Tier::HOST) ==
            expected_host_count);

    auto const allocated = allocator->get_total_allocated_bytes();
    REQUIRE(previous_allocated >= allocated);
    auto const physically_reclaimed = previous_allocated - allocated;
    if (expected_host_count < family.batches.size()) {
      // Logical bytes were reported as freed, but a remaining sibling still owns the two
      // combined fixed-width allocations.
      REQUIRE(physically_reclaimed < freed);
    } else {
      // Destroying the final GPU representation drops the shared owner and reclaims both
      // complete combined columns, not merely the last partition's logical slice.
      REQUIRE(physically_reclaimed >= family.logical_bytes);
    }
    previous_allocated = allocated;
  }

  executor.stop();
  REQUIRE(logical_freed == family.logical_bytes);
  REQUIRE(previous_allocated + family.logical_bytes <= family.allocated_with_family);
}

TEST_CASE("task reservation predicate retries until a shared partition family is reclaimed",
          "[downgrade_executor][hash_partition][memory_pressure]")
{
  auto mem_mgr    = make_partition_pressure_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);
  auto* allocator =
    gpu_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(allocator != nullptr);

  auto family = make_fixed_partition_family(*gpu_space, *allocator);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  for (size_t partition = 0; partition < family.batches.size(); ++partition) {
    repo->add_data_batch(family.batches[partition], partition);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  // Request current reservation headroom plus half the family's payload. Partial sibling downgrades
  // cannot satisfy it because shared columns remain allocated until the final sibling leaves GPU.
  auto const currently_allocated = allocator->get_total_allocated_bytes();
  REQUIRE(currently_allocated < gpu_space->get_max_memory());
  auto const reservation_bytes =
    gpu_space->get_max_memory() - currently_allocated + family.logical_bytes / 2;
  REQUIRE(gpu_space->make_reservation_or_null(reservation_bytes) == nullptr);

  std::mutex reservation_mutex;
  std::unique_ptr<cucascade::memory::reservation> acquired_reservation;
  std::vector<size_t> allocated_at_attempt;

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();
  auto future = executor.request_downgrade([&]() {
    std::lock_guard<std::mutex> lock(reservation_mutex);
    allocated_at_attempt.push_back(allocator->get_total_allocated_bytes());
    if (!acquired_reservation) {
      auto reservation = gpu_space->make_reservation_or_null(reservation_bytes);
      if (reservation && reservation->size() >= reservation_bytes) {
        acquired_reservation = std::move(reservation);
      }
    }
    return acquired_reservation != nullptr;
  });

  auto const logical_freed = future.get();
  executor.stop();

  REQUIRE(acquired_reservation != nullptr);
  REQUIRE(allocated_at_attempt.size() == family.batches.size());
  REQUIRE(allocated_at_attempt.size() > 1);
  for (size_t attempt = 0; attempt + 1 < allocated_at_attempt.size(); ++attempt) {
    REQUIRE(family.allocated_with_family >= allocated_at_attempt[attempt]);
    REQUIRE(family.allocated_with_family - allocated_at_attempt[attempt] <
            family.logical_bytes / family.batches.size());
  }
  REQUIRE(allocated_at_attempt.back() + family.logical_bytes <= family.allocated_with_family);
  REQUIRE(logical_freed == family.logical_bytes);
  REQUIRE(count_batches_in_tier(family.batches, cucascade::memory::Tier::HOST) ==
          family.batches.size());

  acquired_reservation.reset();
  REQUIRE(allocator->get_total_allocated_bytes() + family.logical_bytes <=
          family.allocated_with_family);
}

TEST_CASE("partition-family pressure monitor completes one bounded sweep and becomes quiescent",
          "[downgrade_executor][hash_partition][memory_pressure]")
{
  auto mem_mgr    = make_partition_pressure_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);
  auto* allocator =
    gpu_space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(allocator != nullptr);

  auto family = make_fixed_partition_family(*gpu_space, *allocator);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  for (size_t partition = 0; partition < family.batches.size(); ++partition) {
    repo->add_data_batch(family.batches[partition], partition);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  // Hold enough reservation-only pressure to cross the trigger while the family is resident, while
  // keeping baseline plus held pressure below the stop boundary after the family is reclaimed.
  constexpr size_t kPressureMargin = 1ull << 20;
  auto const target_allocated      = kPressureTriggerAllocated + kPressureMargin;
  REQUIRE(family.allocated_with_family < target_allocated);
  auto const pressure_bytes = target_allocated - family.allocated_with_family;
  REQUIRE(family.allocated_before + pressure_bytes <= kPressureStoppedAllocated);
  auto pressure_reservation = gpu_space->make_reservation_or_null(pressure_bytes);
  REQUIRE(pressure_reservation != nullptr);
  REQUIRE(gpu_space->should_downgrade_memory());

  sirius::exec::downgrade_executor_config config{
    .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade-family"},
    .monitor_period = std::chrono::milliseconds{10}};
  downgrade_executor executor(config, repo_registry, GPU_SPACE_ID, gpu_space, *mem_mgr);
  executor.start();

  auto const deadline = std::chrono::steady_clock::now() + 5s;
  bool completed      = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (all_batches_in_tier_nonblocking(family.batches, cucascade::memory::Tier::HOST)) {
      completed = true;
      break;
    }
    std::this_thread::sleep_for(10ms);
  }

  std::this_thread::sleep_for(100ms);
  auto const requests_before = executor.monitor_requests_issued_for_testing();
  std::this_thread::sleep_for(100ms);
  auto const requests_after = executor.monitor_requests_issued_for_testing();
  executor.stop();

  REQUIRE(completed);
  REQUIRE(gpu_space->should_stop_downgrading_memory());
  REQUIRE(requests_before >= 1);
  REQUIRE(requests_before <= 2);
  REQUIRE(requests_after == requests_before);
}
