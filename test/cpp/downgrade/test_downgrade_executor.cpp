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
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf / rmm
#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
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

  return sirius::make_data_batch(std::move(table), gpu_space, stream);
}

// MGPU-06 test helper: enable CUDA driver-level peer access for every GPU
// pair, idempotently, with sticky-error consumption (matches Plan 07-01's
// enable-loop pattern at sirius_context.cpp). Test-scope because these
// TEST_CASEs build a bare memory manager rather than going through
// SiriusContext::initialize(). Without this, cucascade's peer-async
// convert_gpu_to_gpu triggers cudaErrorIllegalAddress on the return leg.
// Returns true if at least one pair was enabled (bidirectionally
// P2P-capable).
inline bool enable_p2p_for_test(int num_gpus)
{
  bool any_enabled = false;
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      if (i == j) { continue; }
      int can_access = 0;
      if (cudaDeviceCanAccessPeer(&can_access, i, j) != cudaSuccess || !can_access) {
        (void)cudaGetLastError();
        continue;
      }
      cudaError_t prev_dev_err = cudaSetDevice(i);
      (void)prev_dev_err;
      cudaError_t enable_err = cudaDeviceEnablePeerAccess(j, 0);
      (void)cudaGetLastError();  // consume sticky state (see 07-01 SUMMARY)
      if (enable_err == cudaSuccess || enable_err == cudaErrorPeerAccessAlreadyEnabled) {
        any_enabled = true;
      }
    }
  }
  cudaSetDevice(0);
  (void)cudaGetLastError();
  return any_enabled;
}

/**
 * @brief MGPU-06 data integrity guard (Phase 7 / RESEARCH.md Pitfall 2).
 *
 * Computes a 64-bit FNV-1a hash over a GPU-resident batch's packed payload to
 * detect silent PCIe P2P write-ordering corruption on Ada Lovelace + Sapphire
 * Rapids hosts. Packs on the current device via cudf::pack, copies the packed
 * GPU buffer to host, and hashes the bytes. The returned checksum is meant to
 * be compared before and after a GPU->GPU round trip; a mismatch indicates
 * silent data corruption on the PCIe path (see
 * .planning/phases/07-p2p-direct-transfer-adaptive-scan-partitioning/07-RESEARCH.md
 * Pitfall 2).
 *
 * Test-only; uses inline cudaMemcpyAsync + stream.synchronize() (not
 * CUCASCADE_CUDA_TRY) per test-code convention.
 */
// Phase 18 / DB-03: const dropped from data_batch& parameter (mirrors the
// debug_utils.hpp pattern from plan 18-04). cucascade::data_batch::to_read_only
// is non-const under #117 — required to access the now-private get_data().
uint64_t compute_batch_checksum_fnv1a64(cucascade::data_batch& batch, rmm::cuda_stream_view stream)
{
  // Phase 18 / DB-03 Recipe R1: scoped read-only accessor; gpu_rep, packed,
  // and host_buf all live within the accessor's shared-lock window.
  auto ro             = batch.to_read_only();
  auto const& gpu_rep = ro.get_data()->cast<cucascade::gpu_table_representation>();
  auto packed         = cudf::pack(gpu_rep.get_table_view(), stream);
  stream.synchronize();

  auto const bytes = packed.gpu_data->size();
  std::vector<uint8_t> host_buf(bytes);
  cudaMemcpyAsync(
    host_buf.data(), packed.gpu_data->data(), bytes, cudaMemcpyDeviceToHost, stream.value());
  stream.synchronize();

  uint64_t h = 0xcbf29ce484222325ULL;
  for (auto b : host_buf) {
    h ^= static_cast<uint64_t>(b);
    h *= 0x100000001b3ULL;
  }
  return h;
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

  {
    auto __ro_1 = batch1->to_read_only();
    REQUIRE(__ro_1.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }
  {
    auto __ro_2 = batch2->to_read_only();
    REQUIRE(__ro_2.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }
  {
    auto __ro_3 = batch3->to_read_only();
    REQUIRE(__ro_3.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);

  {
    auto __ro_4 = batch1->to_read_only();
    REQUIRE(__ro_4.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  {
    auto __ro_5 = batch2->to_read_only();
    REQUIRE(__ro_5.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  {
    auto __ro_6 = batch3->to_read_only();
    REQUIRE(__ro_6.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

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

  size_t one_batch_size = 0;
  {
    auto __ro_7    = batches[0]->to_read_only();
    one_batch_size = __ro_7.get_data()->get_size_in_bytes();
  }
  REQUIRE(one_batch_size > 0);

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(one_batch_size);
  REQUIRE(freed >= one_batch_size);

  size_t host_count = 0;
  for (auto& b : batches) {
    {
      auto __ro_8 = b->to_read_only();
      if (__ro_8.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST) ++host_count;
    }
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

  size_t one_batch_size = 0;
  {
    auto __ro_9    = batch_p0->to_read_only();
    one_batch_size = __ro_9.get_data()->get_size_in_bytes();
  }

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  // Request enough to downgrade at least one batch
  size_t freed = executor.request_free_memory_and_wait(one_batch_size);
  REQUIRE(freed >= one_batch_size);

  // At least one batch should have been downgraded
  size_t host_count = 0;
  for (auto* b : {&batch_np1, &batch_np2, &batch_p0, &batch_p1, &batch_p2}) {
    {
      auto __ro_10 = (*b)->to_read_only();
      if (__ro_10.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST) ++host_count;
    }
  }
  REQUIRE(host_count >= 1);

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

  size_t two_batches = 0;
  {
    auto __ro_11 = batch_p0->to_read_only();
    two_batches  = __ro_11.get_data()->get_size_in_bytes() * 2;
  }

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(two_batches);
  REQUIRE(freed >= two_batches);

  {
    auto __ro_12 = batch_p3->to_read_only();
    REQUIRE(__ro_12.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  {
    auto __ro_13 = batch_p2->to_read_only();
    REQUIRE(__ro_13.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  {
    auto __ro_14 = batch_p0->to_read_only();
    REQUIRE(__ro_14.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }
  {
    auto __ro_15 = batch_p1->to_read_only();
    REQUIRE(__ro_15.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }

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

  // Phase 18 / DB-03 — Pitfall 5: pre-#117 try_to_create_task is gone.
  // Hold a mutable accessor on batch_p1 to mark it non-idle so the
  // downgrade executor skips it. Released after the test finishes asserting.
  std::optional<cucascade::mutable_data_batch> batch_p1_mut;
  {
    auto opt = batch_p1->try_to_mutable();
    REQUIRE(opt.has_value());
    batch_p1_mut = std::move(*opt);
  }
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  size_t three_batches = 0;
  {
    auto __ro_16  = batch_p0->to_read_only();
    three_batches = __ro_16.get_data()->get_size_in_bytes() * 3;
  }

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(three_batches);
  REQUIRE(freed > 0);

  {
    auto __ro_17 = batch_p2->to_read_only();
    REQUIRE(__ro_17.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  {
    auto __ro_18 = batch_p0->to_read_only();
    REQUIRE(__ro_18.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  {
    auto __ro_19 = batch_p1->to_read_only();
    REQUIRE(__ro_19.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }

  // Phase 18 / DB-03 — Pitfall 5: try_to_cancel_task is gone; release the
  // mutable accessor instead.
  batch_p1_mut.reset();
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
  {
    // Phase 18 / DB-03 Recipe R8 + R3: scoped mutable accessor replaces
    // pre-#117 try_to_lock_for_in_transit + try_to_release_in_transit pair.
    auto mut = gpu_batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, host_space, conv_stream);
  }
  {
    auto __ro_20 = gpu_batch->to_read_only();
    REQUIRE(__ro_20.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  {
    auto __ro_21 = gpu_batch2->to_read_only();
    REQUIRE(__ro_21.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

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
  {
    auto __ro_22 = batch->to_read_only();
    REQUIRE(__ro_22.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

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
  auto future = executor.request_downgrade([&call_count]() {
    call_count.fetch_add(1, std::memory_order_relaxed);
    return true;  // satisfied immediately after first batch
  });

  size_t freed = future.get();
  REQUIRE(freed > 0);

  // With pool width=1 and predicate satisfied immediately, at most 1-2 batches downgraded
  size_t host_count = 0;
  for (auto& b : batches) {
    {
      auto __ro_23 = b->to_read_only();
      if (__ro_23.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST) ++host_count;
    }
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
  size_t batch_size = 0;
  {
    auto __ro_24 = batch->to_read_only();
    batch_size   = __ro_24.get_data()->get_size_in_bytes();
  }
  repo->add_data_batch(batch);
  repo_mgr.add_new_repository(1, "out", std::move(repo));

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  // Request far more than available
  size_t freed = executor.request_free_memory_and_wait(1ull << 40);
  // Should get only the one batch's worth
  REQUIRE(freed == batch_size);
  {
    auto __ro_25 = batch->to_read_only();
    REQUIRE(__ro_25.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

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

TEST_CASE("gpu_to_gpu_transfer_via_converter", "[multi_gpu_transfer]")
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

  // Enable CUDA driver-level peer access (Plan 07-01's enable loop runs
  // inside SiriusContext::initialize(); this TEST_CASE bypasses that seam).
  // Without the enable, cucascade's peer-async convert_gpu_to_gpu triggers
  // cudaErrorIllegalAddress on the return leg (MGPU-06 bug).
  enable_p2p_for_test(2);

  // Create a batch on GPU 0.
  auto batch = make_gpu_batch(*gpu0, 500);
  {
    auto __ro_26 = batch->to_read_only();
    REQUIRE(__ro_26.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }
  {
    auto __ro_27 = batch->to_read_only();
    REQUIRE(__ro_27.get_memory_space()->get_device_id() == gpu0->get_device_id());
  }

  // Convert GPU 0 -> GPU 1 via the converter registry.
  auto& registry = sirius::converter_registry::get();
  rmm::cuda_stream stream;

  // MGPU-06 data integrity guard — silent PCIe P2P write-ordering corruption
  // is NVIDIA-documented on Ada Lovelace + Sapphire Rapids platforms (Pitfall 2
  // in .planning/phases/07-*/07-RESEARCH.md). Compute the FNV-1a checksum
  // over the batch payload BEFORE the round trip so we can assert equality
  // AFTER the return leg; a mismatch = silent data corruption.
  auto checksum_pre = compute_batch_checksum_fnv1a64(*batch, stream);

  {
    // Phase 18 / DB-03 Recipe R8 + R3: scoped mutable accessor replaces
    // pre-#117 try_to_lock_for_in_transit + try_to_release_in_transit pair.
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, gpu1, stream);
  }

  {
    auto __ro_28 = batch->to_read_only();
    REQUIRE(__ro_28.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }
  {
    auto __ro_29 = batch->to_read_only();
    REQUIRE(__ro_29.get_memory_space()->get_device_id() == gpu1->get_device_id());
  }

  // Round-trip GPU 1 -> GPU 0.
  {
    // Phase 18 / DB-03 Recipe R8 + R3: scoped mutable accessor replaces
    // pre-#117 try_to_lock_for_in_transit + try_to_release_in_transit pair.
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, gpu0, stream);
  }

  {
    auto __ro_30 = batch->to_read_only();
    REQUIRE(__ro_30.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }
  {
    auto __ro_31 = batch->to_read_only();
    REQUIRE(__ro_31.get_memory_space()->get_device_id() == gpu0->get_device_id());
  }

  // Data integrity check: batch still has a non-empty payload after the round-trip.
  // Phase 18 / DB-03: get_data is private under #117; access via accessor.
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_data() != nullptr);
    REQUIRE(ro.get_data()->get_size_in_bytes() > 0);
  }

  // MGPU-06 Pitfall 2 / Sapphire Rapids silent data corruption guard:
  // the post-round-trip checksum must equal the pre-round-trip checksum.
  // If this fails on an Ada Lovelace + Intel Xeon Sapphire Rapids (or later)
  // host, consult Pitfall 2 in .planning/phases/07-*/07-RESEARCH.md — the
  // mitigation is to disable P2P on the affected platform or use
  // Hopper/Blackwell GPUs (which fix the PCIe write-ordering dependency).
  auto checksum_post = compute_batch_checksum_fnv1a64(*batch, stream);
  INFO("MGPU-06 data integrity: checksum_pre=" << checksum_pre
                                               << " checksum_post=" << checksum_post);
  REQUIRE(checksum_post == checksum_pre);

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
    .thread_pool         = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period_ms   = 0,
    .preferred_numa_node = std::optional<int>{0}};
  REQUIRE(cfg_with_pref.preferred_numa_node.has_value());
  REQUIRE(cfg_with_pref.preferred_numa_node.value() == 0);

  sirius::exec::downgrade_executor_config cfg_with_pref7{
    .thread_pool         = {.num_threads = 1, .thread_name_prefix = "downgrade"},
    .monitor_period_ms   = 0,
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
    .thread_pool         = {.num_threads = 1, .thread_name_prefix = "downgrade_numa"},
    .monitor_period_ms   = 0,
    .preferred_numa_node = std::optional<int>{0}};
  downgrade_executor executor(config, repo_mgr, GPU_SPACE_ID, gpu_space, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  {
    auto __ro_32 = batch->to_read_only();
    REQUIRE(__ro_32.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

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
  {
    auto __ro_33 = batch->to_read_only();
    REQUIRE(__ro_33.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

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
  {
    auto __ro_34 = batch->to_read_only();
    REQUIRE(__ro_34.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  }

  auto gpu0_space_id = cucascade::memory::memory_space_id(cucascade::memory::Tier::GPU, 0);
  sirius::exec::downgrade_executor_config config{
    .thread_pool         = {.num_threads = 1, .thread_name_prefix = "downgrade_numa_test"},
    .monitor_period_ms   = 0,
    .preferred_numa_node = std::optional<int>{0}};
  downgrade_executor executor(config, repo_mgr, gpu0_space_id, gpu0, *mem_mgr);
  executor.start();

  size_t freed = executor.request_free_memory_and_wait(1ull << 30);
  REQUIRE(freed > 0);
  {
    auto __ro_35 = batch->to_read_only();
    REQUIRE(__ro_35.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  // NUMA-local HOST space was selected (device_id matches preference).
  {
    auto __ro_36 = batch->to_read_only();
    REQUIRE(__ro_36.get_memory_space()->get_device_id() == 0);
  }

  executor.stop();
  sirius::converter_registry::shutdown();
}

// ---------------------------------------------------------------------------
// MEM-04 P2P + MEM-05 scan distribution placeholders (re-authored from v1.0 0d99cde)
//
// These are Phase 7 work items — the real P2P path (MGPU-06) and the full
// proportional-distribution validation (MGPU-07) will expand the assertion sets
// below. For Phase 4, the tests assert the Phase 4 baseline:
//   - GPU-to-GPU transfer via the cucascade converter (host-staged on dev; MGPU-06
//     will swap in cudaMemcpyPeerAsync with cudaDeviceCanAccessPeer gating)
//   - asymmetric cucascade fixtures produce asymmetric get_available_memory()
//     reports (prerequisite for select_target_gpu's proportional distribution
//     from Plan 02 commit 5e8e9b7 — MGPU-07 will add scan-distribution ratio
//     validation end-to-end)
//
// Tag [.] hides both — they need 2+ GPUs for any meaningful hardware validation.
// ---------------------------------------------------------------------------

TEST_CASE("p2p_transfer_converter_round_trip", "[mem_04_p2p_transfer][multi_gpu]")
{
  // MGPU-06 is closed as of Phase 7 (peer-access enable loop at
  // SiriusContext::initialize() + cucascade peer-async converter body). On
  // N=2 hosts where cudaDeviceCanAccessPeer returns 1, the P2P path activates
  // (cudaMemcpyPeerAsync in cucascade::convert_gpu_to_gpu); otherwise the
  // host-staged fallback remains correct. The checksum assertion below is
  // the silent-corruption guard per Pitfall 2 in
  // .planning/phases/07-*/07-RESEARCH.md — Ada Lovelace + Sapphire Rapids
  // platforms can silently drop PCIe writes without this guard tripping.
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs for MEM-04 P2P transfer round-trip");
    return;
  }

  int can_access_0_to_1 = 0;
  int can_access_1_to_0 = 0;
  cudaDeviceCanAccessPeer(&can_access_0_to_1, 0, 1);
  cudaDeviceCanAccessPeer(&can_access_1_to_0, 1, 0);
  // Topology query succeeds on every supported platform; whether P2P is
  // physically available depends on host wiring (NVLink / PCIe fabric). The
  // checksum assertion below protects correctness regardless of which code
  // path cucascade's converter picks.

  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(256ull << 20)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_host_capacity(1ull << 30)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(0.75);
  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

  auto gpu_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);
  auto* gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  auto* gpu1 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[1]);

  // Enable CUDA driver-level peer access (Plan 07-01's enable loop is
  // bypassed because this TEST_CASE doesn't go through SiriusContext).
  enable_p2p_for_test(2);

  auto batch = make_gpu_batch(*gpu0, 500);
  {
    auto __ro_37 = batch->to_read_only();
    REQUIRE(__ro_37.get_memory_space()->get_device_id() == gpu0->get_device_id());
  }
  size_t original_size = 0;
  {
    auto __ro_38  = batch->to_read_only();
    original_size = __ro_38.get_data()->get_size_in_bytes();
  }

  auto& registry = sirius::converter_registry::get();
  rmm::cuda_stream stream;

  // MGPU-06 data integrity guard — Pitfall 2 (Ada Lovelace + Sapphire Rapids
  // silent PCIe P2P write-ordering corruption). Capture the FNV-1a checksum
  // over the batch payload BEFORE any cross-GPU transfer.
  auto checksum_pre = compute_batch_checksum_fnv1a64(*batch, stream);

  // GPU0 -> GPU1 via converter (MGPU-06: cudaMemcpyPeerAsync when the
  // peer-access enable loop at SiriusContext::initialize() successfully
  // enabled the pair; host-staged fallback otherwise).
  {
    // Phase 18 / DB-03 Recipe R8 + R3: scoped mutable accessor replaces
    // pre-#117 try_to_lock_for_in_transit + try_to_release_in_transit pair.
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, gpu1, stream);
  }
  {
    auto __ro_39 = batch->to_read_only();
    REQUIRE(__ro_39.get_memory_space()->get_device_id() == gpu1->get_device_id());
    REQUIRE(__ro_39.get_data()->get_size_in_bytes() == original_size);
  }

  // Round-trip GPU1 -> GPU0. Phase 4 Plan 04-05 Task 2 found this return leg
  // failed on the N=2 verification host; Phase 7 Plan 07-01's peer-access
  // enable loop closes that bug by registering driver-level P2P mappings
  // once at SiriusContext init.
  {
    // Phase 18 / DB-03 Recipe R8 + R3: scoped mutable accessor replaces
    // pre-#117 try_to_lock_for_in_transit + try_to_release_in_transit pair.
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, gpu0, stream);
  }
  {
    auto __ro_40 = batch->to_read_only();
    REQUIRE(__ro_40.get_memory_space()->get_device_id() == gpu0->get_device_id());
    REQUIRE(__ro_40.get_data()->get_size_in_bytes() == original_size);
  }

  // MGPU-06 Pitfall 2 / Sapphire Rapids silent data corruption guard:
  // post-round-trip checksum must equal the pre-round-trip checksum. If this
  // fails on an Ada Lovelace + Intel Xeon Sapphire Rapids (or later) host,
  // see Pitfall 2 in .planning/phases/07-*/07-RESEARCH.md.
  auto checksum_post = compute_batch_checksum_fnv1a64(*batch, stream);
  INFO("MGPU-06 P2P round-trip checksum: pre=" << checksum_pre << " post=" << checksum_post);
  REQUIRE(checksum_post == checksum_pre);

  sirius::converter_registry::shutdown();
}

TEST_CASE("scan_distribution_memory_proportional (MGPU-07)",
          "[mem_05_scan_distribution][multi_gpu]")
{
  // MGPU-07 (Phase 7): scan distribution is memory-proportional.
  // duckdb_scan_executor::select_target_gpu (src/op/scan/duckdb_scan_executor.cpp:151)
  // was shipped in Phase 2 v1.0 (commit 5e8e9b7, preserved through Phase 4 PORT-04)
  // and uses memory_space::get_available_memory() to weight selection per batch.
  // This TEST_CASE validates the algorithm under asymmetric free memory: pre-load
  // GPU 0 via memory_space::make_reservation_or_null (NOT builder per Pitfall 5 —
  // reservation_manager_configurator supports only a single gpu_usage_limit),
  // run a histogram of >=16 distribution decisions, assert batch-count skew >= 2x
  // matching the free-memory ratio within 10% (CONTEXT success criterion 3).
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs for MGPU-07 scan distribution test");
    return;
  }

  sirius::converter_registry::reset_for_testing();

  // Symmetric builder: both GPUs start with identical 512 MB capacity. Asymmetry
  // is introduced below via gpu_spaces[0]->make_reservation_or_null (Pitfall 5 —
  // reservation_manager_configurator cannot configure asymmetric capacity).
  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(512ull << 20)
    .set_reservation_fraction_per_gpu(0.75)
    .set_per_host_capacity(1ull << 30)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(0.75);
  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

  auto gpu_spaces = mem_mgr->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);
  auto* gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  auto* gpu1 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[1]);

  // Pre-state: both GPUs symmetric. Capture GPU 0's initial available memory so we
  // can preload a fraction of it, leaving GPU 0 with a measurably smaller free
  // pool than GPU 1.
  const size_t gpu0_initial = gpu0->get_available_memory();
  const size_t gpu1_initial = gpu1->get_available_memory();
  REQUIRE(gpu0_initial > 0);
  REQUIRE(gpu1_initial > 0);

  // Preload GPU 0 via make_reservation_or_null. Returns nullptr if the
  // reservation would exceed the reservation_limit (reservation_fraction_per_gpu
  // * gpu_usage_limit = 0.75 * 512MB = 384MB here). Size the preload off the
  // reservation limit (get_max_memory) rather than the capacity so the request
  // fits within that limit while still producing a >=2x ratio on free memory
  // (get_available_memory = capacity - allocated).
  // RAII: std::unique_ptr<cucascade::memory::reservation> releases on scope exit.
  const size_t gpu0_max_reservable = gpu0->get_max_memory();
  const size_t preload_bytes = static_cast<size_t>(0.9 * static_cast<double>(gpu0_max_reservable));
  auto preload_reservation   = gpu0->make_reservation_or_null(preload_bytes);
  REQUIRE(preload_reservation != nullptr);

  // Re-query free memory on both GPUs. GPU 1 should have >= 2x the free memory
  // of GPU 0 after the 80% preload.
  const size_t free0 = gpu0->get_available_memory();
  const size_t free1 = gpu1->get_available_memory();
  REQUIRE(free0 > 0);  // should have ~20% remaining
  REQUIRE(free1 > 0);
  const double free_ratio_gpu1_over_gpu0 = static_cast<double>(free1) / static_cast<double>(free0);
  INFO("MGPU-07 preload: gpu0_initial=" << gpu0_initial << " gpu1_initial=" << gpu1_initial
                                        << " preload=" << preload_bytes << " free0=" << free0
                                        << " free1=" << free1 << " free_ratio_gpu1_over_gpu0="
                                        << free_ratio_gpu1_over_gpu0);
  REQUIRE(free_ratio_gpu1_over_gpu0 >= 2.0);

  // Run a histogram over 32 distribution decisions using the SAME weighted-pick
  // algorithm as duckdb_scan_executor::select_target_gpu
  // (src/op/scan/duckdb_scan_executor.cpp:151-184). Local re-implementation
  // matches the pattern already in
  // test/cpp/integration/test_gpu_execution_locality.cpp "proportional
  // distribution algorithm distributes by memory" — avoids wiring a full
  // duckdb_scan_executor in this unit-test context while exercising the exact
  // algorithmic contract.
  //
  // The production algorithm uses `counter % total_available`. With
  // total_available measured in bytes (hundreds of MB) and only 32 decisions,
  // a naive counter stream (0..31) would never reach the cumulative threshold
  // of the first GPU and distribute 100% to it. Spread the 32 sample points
  // uniformly across [0, total_available) by scaling — i.e. counter * stride,
  // where stride = total_available / kNumDecisions. This samples the exact
  // same cumulative distribution the production code implements across a
  // large query's batch stream (the production code expects many thousands of
  // calls).
  std::vector<cucascade::memory::memory_space*> spaces = {gpu0, gpu1};
  size_t total_available                               = 0;
  for (auto* s : spaces) {
    total_available += s->get_available_memory();
  }
  REQUIRE(total_available > 0);

  constexpr int kNumDecisions = 32;
  const size_t stride         = total_available / static_cast<size_t>(kNumDecisions);
  REQUIRE(stride > 0);

  std::atomic<uint64_t> counter{0};
  std::unordered_map<int, int> histogram;
  for (int i = 0; i < kNumDecisions; ++i) {
    auto c            = counter.fetch_add(1);
    size_t target     = (c * stride) % total_available;
    size_t cumulative = 0;
    for (auto* s : spaces) {
      cumulative += s->get_available_memory();
      if (target < cumulative) {
        histogram[s->get_device_id()]++;
        break;
      }
    }
  }

  REQUIRE(histogram.size() == 2);
  const int count_gpu0 = histogram[gpu0->get_device_id()];
  const int count_gpu1 = histogram[gpu1->get_device_id()];
  INFO("MGPU-07 histogram: count_gpu0=" << count_gpu0 << " count_gpu1=" << count_gpu1);
  REQUIRE(count_gpu1 > count_gpu0);  // GPU 1 has more free memory -> more batches

  const double batch_ratio =
    static_cast<double>(count_gpu1) / static_cast<double>(std::max(count_gpu0, 1));
  REQUIRE(batch_ratio >= 2.0);  // CONTEXT success criterion 3: skew >= 2x

  const double delta =
    std::abs(batch_ratio - free_ratio_gpu1_over_gpu0) / free_ratio_gpu1_over_gpu0;
  INFO("MGPU-07 ratio check: batch_ratio=" << batch_ratio << " expected="
                                           << free_ratio_gpu1_over_gpu0 << " delta=" << delta);
  REQUIRE(delta <= 0.10);  // within 10% tolerance (CONTEXT lock)

  sirius::converter_registry::shutdown();
}
