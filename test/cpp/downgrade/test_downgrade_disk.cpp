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
    .set_per_host_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio)
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
    .set_per_host_capacity(4ull << 30)
    .use_host_per_gpu()
    .set_reservation_fraction_per_host(limit_ratio)
    .set_disk_mounting_point(0, 4ull << 30, tmp_dir.string());

  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

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
  // Register HOST with 0-byte reservation limit so that any reserve() call throws.
  // No DISK tier. The exception from fixed_size_host_memory_resource::reserve must
  // propagate out of downgrade_task::execute().
  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity = 2ull << 30;
  const double limit_ratio  = 0.75;

  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_host_capacity(2ull << 20)
    .use_host_per_gpu()
    .set_reservation_limit_per_host(0);  // limit = 0: any request > 0 bytes throws

  auto space_configs = builder.build();
  auto mem_mgr =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));
  sirius::converter_registry::initialize();

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
