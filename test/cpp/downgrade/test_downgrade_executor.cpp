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
#include "memory/multiple_blocks_allocation_accessor.hpp"
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
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
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

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> make_test_memory_manager(
  size_t gpu_capacity = 2ull << 30)
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
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

std::unique_ptr<cudf::table> make_int32_table(cucascade::memory::memory_space& gpu_space,
                                              cudf::size_type rows,
                                              int byte_pattern,
                                              ::cuda::stream_ref stream)
{
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          rows,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          gpu_space.get_default_allocator());
  REQUIRE(cudaMemsetAsync(column->mutable_view().data<int32_t>(),
                          byte_pattern,
                          rows * sizeof(int32_t),
                          stream.get()) == cudaSuccess);
  stream.sync();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::vector<int32_t> read_host_int32(cucascade::data_batch& batch, cudf::size_type rows)
{
  auto ro    = batch.to_read_only();
  auto* host = dynamic_cast<cucascade::host_data_representation const*>(ro.get_data());
  REQUIRE(host != nullptr);
  auto const& table = host->get_host_table();
  REQUIRE(table != nullptr);
  REQUIRE(table->columns.size() == 1);
  auto const& column = table->columns.front();
  REQUIRE(column.type_id == static_cast<int32_t>(cudf::type_id::INT32));
  REQUIRE(column.num_rows == rows);
  REQUIRE(column.null_count == 0);
  REQUIRE(column.has_data);
  REQUIRE(column.data_size == rows * sizeof(int32_t));
  sirius::memory::multiple_blocks_allocation_accessor<int32_t> accessor;
  accessor.initialize(column.data_offset, table->allocation);
  std::vector<int32_t> values(rows);
  for (cudf::size_type row = 0; row < rows; ++row) {
    values[row] = accessor.get(row, table->allocation);
  }
  return values;
}

// Declare after the data owners and executor so assertion failures unblock GPU work before
// teardown.
struct producer_gate {
  ::cuda::stream_ref stream;
  std::atomic<bool> released{false};

  ~producer_gate()
  {
    released.store(true, std::memory_order_release);
    static_cast<void>(cudaStreamSynchronize(stream.get()));
  }

  static void CUDART_CB wait(void* user_data)
  {
    auto& gate = *static_cast<producer_gate*>(user_data);
    while (!gate.released.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(1ms);
    }
  }
};

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

TEST_CASE("request_free_memory preserves pending producer writes",
          "[downgrade_executor][producer_ordering]")
{
  constexpr cudf::size_type rows = 1024;
  constexpr int32_t final_value  = 0x22222222;
  auto mem_mgr                   = make_test_memory_manager(64ull << 20);
  auto* gpu_space                = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  auto const use_default_stream = GENERATE(false, true);
  ::cuda::stream_ref writer_stream =
    use_default_stream ? ::cuda::stream_ref{cudaStream_t{nullptr}} : ::cuda::stream_ref{producer};
  CAPTURE(use_default_stream);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  auto table     = make_int32_table(*gpu_space, rows, 0x11, writer_stream);
  auto* data     = table->mutable_view().column(0).data<int32_t>();
  std::shared_ptr<cudf::table> table_owner;
  std::shared_ptr<cucascade::data_batch> batch;
  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  producer_gate gate{writer_stream};

  REQUIRE(cudaLaunchHostFunc(writer_stream.get(), producer_gate::wait, &gate) == cudaSuccess);
  REQUIRE(cudaMemsetAsync(data, 0x22, rows * sizeof(int32_t), writer_stream.get()) == cudaSuccess);
  SECTION("unique_ptr table")
  {
    batch = sirius::make_data_batch(
      std::move(table), *gpu_space, writer_stream, sirius::telemetry::batch_telemetry_info{});
  }
  SECTION("table rvalue")
  {
    batch = sirius::make_data_batch(
      std::move(*table), *gpu_space, writer_stream, sirius::telemetry::batch_telemetry_info{});
  }
  SECTION("owned table view")
  {
    table_owner = std::move(table);
    batch       = sirius::make_data_batch_from_view(table_owner->view(),
                                              table_owner,
                                              table_owner->alloc_size(),
                                              *gpu_space,
                                              writer_stream,
                                              sirius::telemetry::batch_telemetry_info{});
  }
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_writer_event() != nullptr);
    REQUIRE(cudaEventQuery(ro.get_writer_event()) == cudaErrorNotReady);
  }

  executor.start();
  auto freed = executor.request_free_memory(1ull << 30);
  REQUIRE(freed.wait_for(100ms) == std::future_status::timeout);
  gate.released.store(true, std::memory_order_release);
  REQUIRE(freed.get() > 0);
  writer_stream.sync();
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
  auto values = read_host_int32(*batch, rows);
  REQUIRE(std::count(values.begin(), values.end(), final_value) == rows);
  executor.stop();
}

TEST_CASE("GPU downgrade rejects a missing writer event until the producer records one",
          "[downgrade_executor][producer_ordering]")
{
  constexpr cudf::size_type rows   = 1024;
  constexpr int32_t expected_value = 0x33333333;
  auto mem_mgr                     = make_test_memory_manager(64ull << 20);
  auto* gpu_space                  = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);
  rmm::cuda_stream producer{rmm::cuda_stream::flags::non_blocking};
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr      = *repo_registry.create_for_query(kTestQueryId);
  auto repo           = std::make_unique<cucascade::shared_data_repository>();
  auto table          = make_int32_table(*gpu_space, rows, 0x33, producer);
  auto representation = std::make_unique<cucascade::gpu_table_representation>(
    std::move(table), *gpu_space, ::cuda::stream_ref{cudaStream_t{nullptr}});
  REQUIRE(representation->get_writer_event() == nullptr);
  auto batch = cucascade::data_batch::make(sirius::get_next_batch_id(), std::move(representation));
  auto* host_space = mem_mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(host_space != nullptr);
  sirius::convertible_data_batch wrapper(batch);
  REQUIRE_THROWS_WITH(wrapper.convert({host_space}, producer, *mem_mgr, true),
                      "GPU batch must have a writer event before conversion");
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::GPU);
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
  {
    auto mut = batch->to_mutable();
    mut.get_data()->record_writer_event(producer);
    REQUIRE(mut.get_data()->get_writer_event() != nullptr);
  }
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();
  REQUIRE(executor.request_free_memory_and_wait(1ull << 30) > 0);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
  auto values = read_host_int32(*batch, rows);
  REQUIRE(std::count(values.begin(), values.end(), expected_value) == rows);
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

  // Already-true predicate: the pre-dispatch check must satisfy the request without spilling
  // anything.
  auto future = executor.request_downgrade([&call_count]() {
    call_count.fetch_add(1, std::memory_order_relaxed);
    return true;  // satisfied before any batch is dispatched
  });

  size_t freed = future.get();
  REQUIRE(freed == 0);
  REQUIRE(call_count.load() >= 1);

  size_t host_count = 0;
  for (auto& b : batches) {
    if (get_batch_tier(*b) == cucascade::memory::Tier::HOST) ++host_count;
  }
  REQUIRE(host_count == 0);

  executor.stop();
}

TEST_CASE("request_downgrade stops once the predicate becomes satisfied", "[downgrade_executor]")
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

  // Predicate becomes true from the second evaluation on. Pool width is 1, so the pre-dispatch
  // check stops the loop after exactly one batch.
  auto future = executor.request_downgrade(
    [&call_count]() { return call_count.fetch_add(1, std::memory_order_relaxed) >= 1; });

  size_t freed = future.get();
  REQUIRE(freed > 0);

  size_t host_count = 0;
  for (auto& b : batches) {
    if (get_batch_tier(*b) == cucascade::memory::Tier::HOST) ++host_count;
  }
  REQUIRE(host_count == 1);

  executor.stop();
}

TEST_CASE("request_free_memory does not overshoot its byte target", "[downgrade_executor]")
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

  // Target of 1.5 batches: planned-bytes gating stops dispatch at 2 batches instead of
  // continuing until a completed conversion flips the predicate.
  size_t freed = executor.request_free_memory_and_wait(one_batch_size + one_batch_size / 2);
  REQUIRE(freed == 2 * one_batch_size);

  size_t host_count = 0;
  for (auto& b : batches) {
    if (get_batch_tier(*b) == cucascade::memory::Tier::HOST) ++host_count;
  }
  REQUIRE(host_count == 2);

  executor.stop();
}

TEST_CASE("request_free_memory best-fits a small deficit instead of a whole large batch",
          "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr = *repo_registry.create_for_query(kTestQueryId);
  auto repo      = std::make_unique<cucascade::shared_data_repository>();
  // Policy order is last partition first, so the LARGE batches are the policy picks and the
  // small one sits last in iteration order.
  auto batch_small  = make_gpu_batch(*gpu_space, /*num_rows=*/1000);  // partition 0
  auto batch_large1 = make_gpu_batch(*gpu_space, /*num_rows=*/8000);  // partition 1
  auto batch_large2 = make_gpu_batch(*gpu_space, /*num_rows=*/8000);  // partition 2
  repo->add_data_batch(batch_small, 0);
  repo->add_data_batch(batch_large1, 1);
  repo->add_data_batch(batch_large2, 2);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t small_size = get_batch_size(*batch_small);
  REQUIRE(small_size > 0);
  REQUIRE(get_batch_size(*batch_large2) > small_size);

  auto executor = make_test_executor(repo_registry, gpu_space, *mem_mgr);
  executor.start();

  // Deficit the small batch covers: best-fit spills it, where policy order alone would have
  // spilled batch_large2.
  size_t freed = executor.request_free_memory_and_wait(small_size);
  REQUIRE(freed == small_size);

  REQUIRE(get_batch_tier(*batch_small) == cucascade::memory::Tier::HOST);
  REQUIRE(get_batch_tier(*batch_large1) == cucascade::memory::Tier::GPU);
  REQUIRE(get_batch_tier(*batch_large2) == cucascade::memory::Tier::GPU);

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
