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
#include "task_completion.hpp"

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
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf / rmm
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <chrono>
#include <memory>
#include <thread>
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
  task_executor_config config{1, false};
  return downgrade_executor(config, repo_mgr, GPU_SPACE_ID, gpu_space, mem_mgr);
}

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Downgrade executor starts and stops cleanly", "[downgrade_executor]")
{
  auto mem_mgr = make_test_memory_manager();
  cucascade::shared_data_repository_manager repo_mgr;

  // nullptr memory_space — monitor loop won't trigger, just tests lifecycle
  auto executor = make_test_executor(repo_mgr, nullptr, *mem_mgr);

  REQUIRE_NOTHROW(executor.start());
  REQUIRE_NOTHROW(executor.stop());
}

TEST_CASE("run_downgrade_pass with empty repositories returns 0", "[downgrade_executor]")
{
  auto mem_mgr = make_test_memory_manager();
  cucascade::shared_data_repository_manager repo_mgr;

  auto executor = make_test_executor(repo_mgr, nullptr, *mem_mgr);
  executor.start();

  std::vector<downgrade_repository_info> repos;
  size_t scheduled = executor.run_downgrade_pass(repos, 1024);
  REQUIRE(scheduled == 0);

  executor.stop();
}

TEST_CASE("Single downgrade task executes correctly", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;
  sirius::task_completion_message_queue msg_queue;

  auto batch = make_gpu_batch(*gpu_space);
  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  auto global_state = std::make_shared<downgrade_task_global_state>(*mem_mgr, repo_mgr, msg_queue);
  auto local_state  = std::make_unique<downgrade_task_local_state>(0, 0, batch);
  downgrade_task task(std::move(local_state), global_state);

  rmm::cuda_stream stream;
  REQUIRE_NOTHROW(task.execute(stream));

  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
}

TEST_CASE("run_downgrade_pass downgrades GPU batches from a single non-partitioned repo",
          "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;

  cucascade::shared_data_repository repo;
  auto batch1 = make_gpu_batch(*gpu_space);
  auto batch2 = make_gpu_batch(*gpu_space);
  auto batch3 = make_gpu_batch(*gpu_space);
  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);
  repo.add_data_batch(batch3);

  REQUIRE(batch1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch2->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch3->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  std::vector<downgrade_repository_info> repos = {{&repo}};
  size_t scheduled                             = executor.run_downgrade_pass(repos, 1ull << 30);
  REQUIRE(scheduled == 3);

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    bool all_on_host = true;
    if (batch1->get_memory_space()->get_tier() != cucascade::memory::Tier::HOST)
      all_on_host = false;
    if (batch2->get_memory_space()->get_tier() != cucascade::memory::Tier::HOST)
      all_on_host = false;
    if (batch3->get_memory_space()->get_tier() != cucascade::memory::Tier::HOST)
      all_on_host = false;
    if (all_on_host) break;
    std::this_thread::sleep_for(50ms);
  }

  REQUIRE(batch1->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch3->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}

TEST_CASE("run_downgrade_pass respects amount_to_downgrade limit", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;

  cucascade::shared_data_repository repo;
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (int i = 0; i < 5; ++i) {
    auto batch = make_gpu_batch(*gpu_space);
    batches.push_back(batch);
    repo.add_data_batch(batch);
  }

  size_t one_batch_size = batches[0]->get_data()->get_size_in_bytes();
  REQUIRE(one_batch_size > 0);

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  std::vector<downgrade_repository_info> repos = {{&repo}};
  size_t scheduled                             = executor.run_downgrade_pass(repos, one_batch_size);
  REQUIRE(scheduled >= 1);
  REQUIRE(scheduled < 5);

  executor.stop();
}

TEST_CASE("run_downgrade_pass prioritizes partitioned repos over non-partitioned",
          "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;

  cucascade::shared_data_repository repo_non_partitioned;
  auto batch_np1 = make_gpu_batch(*gpu_space);
  auto batch_np2 = make_gpu_batch(*gpu_space);
  repo_non_partitioned.add_data_batch(batch_np1);
  repo_non_partitioned.add_data_batch(batch_np2);

  cucascade::shared_data_repository repo_partitioned;
  auto batch_p0 = make_gpu_batch(*gpu_space);
  auto batch_p1 = make_gpu_batch(*gpu_space);
  auto batch_p2 = make_gpu_batch(*gpu_space);
  repo_partitioned.add_data_batch(batch_p0, 0);
  repo_partitioned.add_data_batch(batch_p1, 1);
  repo_partitioned.add_data_batch(batch_p2, 2);

  size_t one_batch_size = batch_p0->get_data()->get_size_in_bytes();

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  std::vector<downgrade_repository_info> repos = {{&repo_non_partitioned}, {&repo_partitioned}};
  size_t scheduled                             = executor.run_downgrade_pass(repos, one_batch_size);
  REQUIRE(scheduled >= 1);

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST) break;
    std::this_thread::sleep_for(50ms);
  }

  REQUIRE(batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_np1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch_np2->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  executor.stop();
}

TEST_CASE("run_downgrade_pass iterates partitions from last to first", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;

  cucascade::shared_data_repository repo;
  auto batch_p0 = make_gpu_batch(*gpu_space);
  auto batch_p1 = make_gpu_batch(*gpu_space);
  auto batch_p2 = make_gpu_batch(*gpu_space);
  auto batch_p3 = make_gpu_batch(*gpu_space);
  repo.add_data_batch(batch_p0, 0);
  repo.add_data_batch(batch_p1, 1);
  repo.add_data_batch(batch_p2, 2);
  repo.add_data_batch(batch_p3, 3);

  size_t two_batches = batch_p0->get_data()->get_size_in_bytes() * 2;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  std::vector<downgrade_repository_info> repos = {{&repo}};
  size_t scheduled                             = executor.run_downgrade_pass(repos, two_batches);
  REQUIRE(scheduled == 2);

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (batch_p3->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST &&
        batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST)
      break;
    std::this_thread::sleep_for(50ms);
  }

  REQUIRE(batch_p3->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p0->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(batch_p1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  executor.stop();
}

TEST_CASE("run_downgrade_pass skips active partitions in first pass", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;

  cucascade::shared_data_repository repo;
  auto batch_p0 = make_gpu_batch(*gpu_space);
  auto batch_p1 = make_gpu_batch(*gpu_space);
  auto batch_p2 = make_gpu_batch(*gpu_space);
  repo.add_data_batch(batch_p0, 0);
  repo.add_data_batch(batch_p1, 1);
  repo.add_data_batch(batch_p2, 2);

  REQUIRE(batch_p1->try_to_create_task());

  size_t three_batches = batch_p0->get_data()->get_size_in_bytes() * 3;

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  std::vector<downgrade_repository_info> repos = {{&repo}};
  size_t scheduled                             = executor.run_downgrade_pass(repos, three_batches);
  REQUIRE(scheduled >= 2);

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST &&
        batch_p0->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST)
      break;
    std::this_thread::sleep_for(50ms);
  }

  REQUIRE(batch_p2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p0->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  REQUIRE(batch_p1->get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  batch_p1->try_to_cancel_task();
  executor.stop();
}

TEST_CASE("run_downgrade_pass skips batches already on HOST", "[downgrade_executor]")
{
  auto mem_mgr    = make_test_memory_manager();
  auto* gpu_space = get_gpu_space(*mem_mgr);
  REQUIRE(gpu_space != nullptr);

  cucascade::shared_data_repository_manager repo_mgr;

  cucascade::shared_data_repository repo;
  auto gpu_batch  = make_gpu_batch(*gpu_space);
  auto gpu_batch2 = make_gpu_batch(*gpu_space);
  repo.add_data_batch(gpu_batch);
  repo.add_data_batch(gpu_batch2);

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

  auto executor = make_test_executor(repo_mgr, gpu_space, *mem_mgr);
  executor.start();

  std::vector<downgrade_repository_info> repos = {{&repo}};
  size_t scheduled                             = executor.run_downgrade_pass(repos, 1ull << 30);
  REQUIRE(scheduled == 1);

  auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (gpu_batch2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST) break;
    std::this_thread::sleep_for(50ms);
  }
  REQUIRE(gpu_batch2->get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);

  executor.stop();
}
