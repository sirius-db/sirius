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
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

// cudf / rmm
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <filesystem>
#include <memory>
#include <vector>

using namespace sirius::parallel;

namespace {

std::unique_ptr<sirius::memory::sirius_memory_reservation_manager>
make_test_memory_manager_with_disk(size_t host_capacity, size_t disk_capacity)
{
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity = 2ull << 30;
  const double limit_ratio  = 0.75;

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio);

  // Add disk tier — use a temp directory
  if (disk_capacity > 0) {
    auto tmp_dir = std::filesystem::temp_directory_path() / "sirius_test_disk";
    std::filesystem::create_directories(tmp_dir);
    builder.set_disk_mounting_point(0, disk_capacity, tmp_dir.string());
  }

  auto space_configs = builder.build();
  auto manager       = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(
    std::move(space_configs));

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

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("Downgrade task falls back to DISK when HOST is full", "[downgrade_disk]")
{
  // HOST capacity = 0 forces fallback to DISK
  auto mem_mgr = make_test_memory_manager_with_disk(0, 4ull << 30);
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

  REQUIRE(batch->get_memory_space()->get_tier() == cucascade::memory::Tier::DISK);
}

TEST_CASE("Downgrade task uses HOST when HOST has capacity", "[downgrade_disk]")
{
  // HOST capacity = 4GB, DISK capacity = 4GB; HOST should be preferred
  auto mem_mgr = make_test_memory_manager_with_disk(4ull << 30, 4ull << 30);
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

TEST_CASE("Downgrade task throws when both HOST and DISK reservation fail", "[downgrade_disk]")
{
  // HOST capacity = 0, DISK capacity = 0 — both tiers exhausted
  auto mem_mgr = make_test_memory_manager_with_disk(0, 0);
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
  REQUIRE_THROWS(task.execute(stream));
}
