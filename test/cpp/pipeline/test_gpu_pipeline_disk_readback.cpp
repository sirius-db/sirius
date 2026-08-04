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
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "utils/utils.hpp"

#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

cucascade::memory::memory_space* get_space(sirius::memory::sirius_memory_reservation_manager& mgr,
                                           cucascade::memory::Tier tier)
{
  auto* space = mgr.get_memory_space(tier, 0);
  if (space) return space;
  auto spaces = mgr.get_memory_spaces_for_tier(tier);
  if (!spaces.empty()) return const_cast<cucascade::memory::memory_space*>(spaces.front());
  return nullptr;
}

/**
 * @brief Build a memory manager with GPU + HOST + DISK tiers for disk round-trip tests.
 *
 * Uses a temp directory for the DISK mounting point so tests are self-contained.
 * Returns the manager and the tmp_dir path (caller must keep alive for test duration).
 */
std::pair<std::unique_ptr<sirius::memory::sirius_memory_reservation_manager>, std::filesystem::path>
make_test_memory_manager_with_disk()
{
  sirius::converter_registry::reset_for_testing();

  auto tmp_dir = std::filesystem::temp_directory_path() / "sirius_test_disk_readback";
  std::filesystem::create_directories(tmp_dir);

  const size_t gpu_capacity  = 2ull << 30;   // 2 GB
  const size_t host_capacity = 4ull << 30;   // 4 GB
  const size_t disk_capacity = 16ull << 30;  // 16 GB
  const double limit_ratio   = 0.75;

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_host_per_gpu()
    .set_reservation_fraction_per_numa_region(limit_ratio)
    .set_disk_mounting_point(0, disk_capacity, tmp_dir.string());

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();
  return {std::move(manager), tmp_dir};
}

/**
 * @brief Create a GPU-resident data batch with a single INT32 column.
 */
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

}  // namespace

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("DISK->GPU round-trip conversion via converter registry", "[gpu_pipeline_disk]")
{
  // The Tier::GPU arm of lock_or_prepare_batch calls
  //   batch->convert_to<gpu_table_representation>(registry, gpu_space, stream)
  // The registry dispatches on typeid(*source). When source is disk_data_representation,
  // the DISK->GPU converter registered in Phase 1 (register_builtin_converters with pipeline
  // backend) is selected automatically. This test verifies the round-trip works end-to-end.
  auto [mgr, tmp_dir] = make_test_memory_manager_with_disk();

  auto* gpu_space  = get_space(*mgr, cucascade::memory::Tier::GPU);
  auto* disk_space = get_space(*mgr, cucascade::memory::Tier::DISK);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(disk_space != nullptr);

  auto& registry = sirius::converter_registry::get();
  rmm::cuda_stream stream;

  auto batch = make_gpu_batch(*gpu_space);
  REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  // --- GPU -> DISK ---
  {
    auto mut = batch->to_mutable();
    REQUIRE_NOTHROW(
      mut.convert_to<cucascade::disk_data_representation>(registry, disk_space, stream));
  }

  REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::DISK);

  // --- DISK -> GPU ---
  // This mirrors exactly what lock_or_prepare_batch Tier::GPU arm does when the batch is
  // disk-resident and the target memory space is GPU.
  {
    auto mut = batch->to_mutable();
    REQUIRE_NOTHROW(
      mut.convert_to<cucascade::gpu_table_representation>(registry, gpu_space, stream));
  }

  REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
}

TEST_CASE("DISK->GPU conversion preserves data correctness", "[gpu_pipeline_disk]")
{
  // After a GPU->DISK->GPU round-trip the batch must still have the expected column count,
  // column type, row count, and actual cell values.
  auto [mgr, tmp_dir] = make_test_memory_manager_with_disk();

  auto* gpu_space  = get_space(*mgr, cucascade::memory::Tier::GPU);
  auto* disk_space = get_space(*mgr, cucascade::memory::Tier::DISK);
  REQUIRE(gpu_space != nullptr);
  REQUIRE(disk_space != nullptr);

  auto& registry = sirius::converter_registry::get();
  rmm::cuda_stream stream;

  const size_t num_rows = 500;
  auto batch            = make_gpu_batch(*gpu_space, num_rows);
  REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  // Snapshot column values before round-trip
  auto table_before = sirius::get_cudf_table_view(*batch);
  std::vector<int32_t> values_before(static_cast<size_t>(table_before.num_rows()));
  cudaMemcpy(values_before.data(),
             table_before.column(0).data<int32_t>(),
             sizeof(int32_t) * values_before.size(),
             cudaMemcpyDeviceToHost);

  const size_t size_before = [&batch]() {
    auto ro = batch->to_read_only();
    return ro.get_data()->get_size_in_bytes();
  }();
  REQUIRE(size_before > 0);

  // GPU -> DISK
  {
    auto mut = batch->to_mutable();
    REQUIRE_NOTHROW(
      mut.convert_to<cucascade::disk_data_representation>(registry, disk_space, stream));
  }
  REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::DISK);

  // DISK -> GPU
  {
    auto mut = batch->to_mutable();
    REQUIRE_NOTHROW(
      mut.convert_to<cucascade::gpu_table_representation>(registry, gpu_space, stream));
  }
  REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);

  // Verify schema and shape
  auto table_after = sirius::get_cudf_table_view(*batch);
  REQUIRE(table_after.num_columns() == 1);
  REQUIRE(table_after.column(0).type() == cudf::data_type{cudf::type_id::INT32});
  REQUIRE(static_cast<size_t>(table_after.num_rows()) == num_rows);

  // Verify actual cell values match
  std::vector<int32_t> values_after(static_cast<size_t>(table_after.num_rows()));
  cudaMemcpy(values_after.data(),
             table_after.column(0).data<int32_t>(),
             sizeof(int32_t) * values_after.size(),
             cudaMemcpyDeviceToHost);
  REQUIRE(values_before == values_after);

  // Size should be unchanged (same data, same GPU format)
  const size_t size_after = [&batch]() {
    auto ro = batch->to_read_only();
    return ro.get_data()->get_size_in_bytes();
  }();
  REQUIRE(size_after == size_before);
}
