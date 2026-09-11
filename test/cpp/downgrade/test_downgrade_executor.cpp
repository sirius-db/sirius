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
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <chrono>
#include <memory>
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

/// Zero-reclaimable view batch over shared column storage, the scan's pinned view-forward shape.
std::shared_ptr<cucascade::data_batch> make_zero_reclaimable_alias_batch(
  cucascade::memory::memory_space& gpu_space, size_t num_rows = 1000)
{
  auto stream = cudf::get_default_stream();
  auto owned  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                         static_cast<cudf::size_type>(num_rows),
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         gpu_space.get_default_allocator());
  std::shared_ptr<cudf::column> column(std::move(owned));
  std::vector<cudf::column_view> views{column->view()};
  auto const alloc_size = column->alloc_size();
  return sirius::make_data_batch_from_view(
    cudf::table_view{views},
    std::vector<std::shared_ptr<cudf::column>>{std::move(column)},
    alloc_size,
    gpu_space,
    stream,
    sirius::telemetry::batch_telemetry_info{},
    /*reclaimable_size=*/std::size_t{0});
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

TEST_CASE("request_free_memory credits only reclaimable bytes and leaves aliases on GPU",
          "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr     = *repo_registry.create_for_query(kTestQueryId);
  auto repo          = std::make_unique<cucascade::shared_data_repository>();
  auto owning        = make_gpu_batch(*gpu_space);
  auto alias         = make_zero_reclaimable_alias_batch(*gpu_space);
  size_t owning_size = get_batch_size(*owning);
  repo->add_data_batch(owning);
  repo->add_data_batch(alias);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Request more than the repo holds: only the owning batch's bytes are creditable.
  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed == owning_size);
  REQUIRE(get_batch_tier(*owning) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*alias) == cucascade::memory::Tier::GPU);

  executor.stop();
}
