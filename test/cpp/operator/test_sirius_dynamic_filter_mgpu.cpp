/*
 * Copyright 2026, Sirius Contributors.
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

// Focused regressions for device-local dynamic-filter replicas. Each case builds the producer-side
// filter on logical GPU 0, materializes device-local replicas, then consumes them from remote probe
// GPUs. Before replica support, the corresponding cross-device dereference is the
// cudaErrorIllegalAddress reported by TPC-H Q2.

#include "op/dynamic_filter/dynamic_filter_replica_transfer.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"
#include "operator_test_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

constexpr int kBuildDevice = 0;
constexpr int kProbeDevice = 1;
constexpr std::array<int, 2> kReplicaDevices{kBuildDevice, kProbeDevice};

bool require_two_gpus()
{
  int count      = 0;
  auto const err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count < 2) {
    WARN("dynamic-filter replica test requires at least two visible GPUs; skipping");
    return false;
  }
  return true;
}

bool require_three_gpus()
{
  int count      = 0;
  auto const err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count < 3) {
    WARN("dynamic-filter overlap test requires at least three visible GPUs; skipping");
    return false;
  }
  return true;
}

template <typename T>
std::unique_ptr<cudf::column> make_values(std::vector<T> const& values,
                                          cudf::data_type type,
                                          rmm::cuda_stream_view stream)
{
  auto col       = cudf::make_numeric_column(type,
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       cudf::get_current_device_resource_ref());
  auto const err = cudaMemcpyAsync(col->mutable_view().data<T>(),
                                   values.data(),
                                   values.size() * sizeof(T),
                                   cudaMemcpyHostToDevice,
                                   stream.value());
  REQUIRE(err == cudaSuccess);
  // Callers commonly pass a temporary initializer vector. Complete the pageable-host transfer
  // before that vector is destroyed; these focused tests do not benchmark ingestion.
  stream.synchronize();
  return col;
}

std::vector<std::uint8_t> mask_to_host(cudf::column_view const& mask, rmm::cuda_stream_view stream)
{
  REQUIRE(mask.type().id() == cudf::type_id::BOOL8);
  std::vector<std::uint8_t> host(static_cast<std::size_t>(mask.size()));
  auto const err = cudaMemcpyAsync(host.data(),
                                   mask.data<bool>(),
                                   host.size() * sizeof(bool),
                                   cudaMemcpyDeviceToHost,
                                   stream.value());
  REQUIRE(err == cudaSuccess);
  stream.synchronize();
  return host;
}

template <typename Filter>
void replicate_to_both_devices(
  Filter& filter, std::span<sirius::op::dynamic_filter_replica_space const> replica_spaces)
{
  filter.replicate_to_devices(replica_spaces);
  REQUIRE(filter.is_available_on_device(kBuildDevice));
  REQUIRE(filter.is_available_on_device(kProbeDevice));
}

template <typename MemoryManager>
std::vector<sirius::op::dynamic_filter_replica_space> get_replica_spaces(
  MemoryManager& memory_manager, std::size_t expected_device_count = kReplicaDevices.size())
{
  auto const gpu_spaces  = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  auto const host_spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(gpu_spaces.size() == expected_device_count);
  REQUIRE_FALSE(host_spaces.empty());

  std::vector<sirius::op::dynamic_filter_replica_space> result;
  result.reserve(gpu_spaces.size());
  for (auto const* gpu_space_view : gpu_spaces) {
    auto* gpu_space = memory_manager.get_memory_space(cucascade::memory::Tier::GPU,
                                                      gpu_space_view->get_device_id());
    REQUIRE(gpu_space != nullptr);
    auto const local_host =
      std::find_if(host_spaces.begin(), host_spaces.end(), [gpu_space](auto const* host_space) {
        return host_space->get_device_id() == gpu_space->get_device_id();
      });
    auto const* host_space = local_host == host_spaces.end() ? host_spaces.front() : *local_host;
    result.emplace_back(*gpu_space, *host_space);
  }
  std::sort(result.begin(), result.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.get_gpu_space().get_device_id() < rhs.get_gpu_space().get_device_id();
  });
  for (std::size_t i = 0; i < result.size(); ++i) {
    REQUIRE(result[i].get_gpu_space().get_device_id() == static_cast<int>(i));
  }
  return result;
}

}  // namespace

TEST_CASE("IN-list replica built on GPU 0 computes an exact mask on GPU 1",
          "[dynamic_filter][mgpu][replica][in_list]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  auto& target_space  = replica_spaces.back().get_gpu_space();
  auto* target_mr =
    target_space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(target_mr != nullptr);
  auto const target_reserved_before  = target_space.get_total_reserved_memory();
  auto const target_active_before    = target_space.get_active_reservation_count();
  auto const target_allocated_before = target_mr->get_total_allocated_bytes();
  std::unique_ptr<sirius::op::sirius_dynamic_in_list_filter> filter;
  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const& build_space = replica_spaces.front().get_gpu_space();
    auto const stream       = build_space.acquire_stream();
    auto keys = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
    filter    = std::make_unique<sirius::op::sirius_dynamic_in_list_filter>(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();
    keys.reset();  // replication must not depend on the constructor's borrowed column_view

    REQUIRE(filter->replica_count() == 1);
    REQUIRE(filter->is_available_on_device(kBuildDevice));
    REQUIRE_FALSE(filter->is_available_on_device(kProbeDevice));

    // Exercise the low-level availability contract independently of production scheduling: before
    // explicit replication, a GPU 1 caller skips the optional filter and never touches GPU 0's set.
    {
      rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
      auto const probe_stream = cudf::get_default_stream();
      auto early_probe =
        make_values<std::int64_t>({2, 7}, cudf::data_type{cudf::type_id::INT64}, probe_stream);
      auto early_mask = filter->compute_mask(
        early_probe->view(), kProbeDevice, probe_stream, cudf::get_current_device_resource_ref());
      REQUIRE(early_mask == nullptr);
    }

    replicate_to_both_devices(*filter, replica_spaces);
    REQUIRE(filter->replica_count() == kReplicaDevices.size());
    REQUIRE(target_space.get_total_reserved_memory() == target_reserved_before);
    REQUIRE(target_space.get_active_reservation_count() == target_active_before);
    REQUIRE(target_mr->get_total_allocated_bytes() > target_allocated_before);
  }

  {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
    auto const stream = cudf::get_default_stream();
    auto probe =
      make_values<std::int64_t>({1, 2, 3, 4, 6, 9}, cudf::data_type{cudf::type_id::INT64}, stream);
    auto mask = filter->compute_mask(
      probe->view(), kProbeDevice, stream, cudf::get_current_device_resource_ref());
    REQUIRE(mask != nullptr);
    REQUIRE(mask_to_host(mask->view(), stream) == std::vector<std::uint8_t>{0, 1, 0, 1, 1, 0});
  }

  rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
  filter.reset();
  REQUIRE(target_mr->get_total_allocated_bytes() == target_allocated_before);
}

TEST_CASE("dynamic-filter replicas require destination reservation admission",
          "[dynamic_filter][mgpu][replica][reservation]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  auto& target_space  = replica_spaces.back().get_gpu_space();
  auto* target_mr =
    target_space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(target_mr != nullptr);

  auto const allocated_before = target_mr->get_total_allocated_bytes();
  REQUIRE(allocated_before < target_space.get_max_memory());
  auto reservation_pressure =
    target_space.make_reservation_or_null(target_space.get_max_memory() - allocated_before);
  REQUIRE(reservation_pressure != nullptr);
  REQUIRE(target_space.get_available_memory() > (1U << 20));
  REQUIRE(target_space.make_reservation_or_null(rmm::CUDA_ALLOCATION_ALIGNMENT) == nullptr);

  auto require_omitted = [&](auto& filter) {
    auto const target_allocated = target_mr->get_total_allocated_bytes();
    filter.replicate_to_devices(replica_spaces);
    REQUIRE(filter.is_available_on_device(kBuildDevice));
    REQUIRE_FALSE(filter.is_available_on_device(kProbeDevice));
    REQUIRE(target_mr->get_total_allocated_bytes() == target_allocated);
  };

  rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
  auto& build_space = replica_spaces.front().get_gpu_space();
  auto const stream = build_space.acquire_stream();

  SECTION("IN-list")
  {
    auto keys = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
    sirius::op::sirius_dynamic_in_list_filter filter(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();
    require_omitted(filter);
  }

  SECTION("Small IN-list")
  {
    auto keys = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
    sirius::op::sirius_dynamic_small_in_list_filter filter(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();
    require_omitted(filter);
  }

  SECTION("Bloom")
  {
    auto keys = cudf::sequence(1024,
                               cudf::numeric_scalar<std::int64_t>(0, true, stream),
                               cudf::numeric_scalar<std::int64_t>(1, true, stream),
                               stream,
                               build_space.get_default_allocator());
    sirius::op::sirius_dynamic_bloom_filter filter(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();
    require_omitted(filter);
  }
}

TEST_CASE("IN-list peer copies fan out to three GPUs before publication",
          "[dynamic_filter][mgpu][replica][peer_overlap]")
{
  if (!require_three_gpus()) { return; }

  constexpr std::size_t device_count = 3;
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(device_count);
  auto replica_spaces = get_replica_spaces(*memory_manager, device_count);
  if (!cucascade::memory::probe_peer_dma_works(0, 1) ||
      !cucascade::memory::probe_peer_dma_works(0, 2)) {
    WARN(
      "dynamic-filter overlap test requires direct peer DMA from GPU 0 to GPUs 1 and 2; "
      "skipping");
    return;
  }
  std::unique_ptr<sirius::op::sirius_dynamic_in_list_filter> filter;

  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const& build_space = replica_spaces.front().get_gpu_space();
    auto const stream       = build_space.acquire_stream();
    auto keys = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
    filter    = std::make_unique<sirius::op::sirius_dynamic_in_list_filter>(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();

    filter->replicate_to_devices(replica_spaces);
    REQUIRE(filter->replica_count() == device_count);
  }

  for (int device_id = 1; device_id < static_cast<int>(device_count); ++device_id) {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{device_id}};
    auto const& probe_space = replica_spaces[static_cast<std::size_t>(device_id)].get_gpu_space();
    auto const stream       = probe_space.acquire_stream();
    REQUIRE(filter->is_available_on_device(device_id));
    auto probe =
      make_values<std::int64_t>({1, 2, 4, 9}, cudf::data_type{cudf::type_id::INT64}, stream);
    auto mask =
      filter->compute_mask(probe->view(), device_id, stream, probe_space.get_default_allocator());
    REQUIRE(mask != nullptr);
    REQUIRE(mask_to_host(mask->view(), stream) == std::vector<std::uint8_t>{0, 1, 1, 0});
  }

  rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
  filter.reset();
}

TEST_CASE("dynamic-filter replica transfer borrows fixed blocks from a Sirius HOST space",
          "[dynamic_filter][mgpu][replica][transfer]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager      = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces      = get_replica_spaces(*memory_manager);
  auto const& source_space = replica_spaces.front().get_gpu_space();
  auto const& target       = replica_spaces.back();
  auto const& target_space = target.get_gpu_space();
  auto* const host_resource =
    target.get_host_staging_space()
      .get_memory_resource_as<cucascade::memory::fixed_size_host_memory_resource>();
  REQUIRE(host_resource != nullptr);

  // Exercise a real batch over three noncontiguous fixed blocks. Make each block's contents
  // distinct so a duplicated or reordered chunk cannot satisfy the exact-payload assertion.
  auto const block_size = host_resource->get_block_size();
  auto const bytes      = 2 * block_size + 4096;
  std::vector<std::byte> expected(bytes);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    auto const block_index  = i / block_size;
    auto const block_offset = i % block_size;
    expected[i] = static_cast<std::byte>((block_offset * 131U + block_index * 17U) & 0xffU);
  }

  std::unique_ptr<rmm::device_buffer> source;
  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const stream = source_space.acquire_stream();
    source =
      std::make_unique<rmm::device_buffer>(bytes, stream, source_space.get_default_allocator());
    REQUIRE(cudaMemcpyAsync(
              source->data(), expected.data(), bytes, cudaMemcpyHostToDevice, stream.value()) ==
            cudaSuccess);
    stream.synchronize();
  }

  {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
    auto const stream = target_space.acquire_stream();
    rmm::device_buffer destination(bytes, stream, target_space.get_default_allocator());
    auto const allocated_before = host_resource->get_total_allocated_bytes();
    auto const route            = sirius::op::detail::enqueue_replica_copy(
      destination.data(),
      rmm::cuda_device_id{kProbeDevice},
      source->data(),
      source_space,
      bytes,
      stream,
      target.get_host_staging_space(),
      sirius::op::detail::replica_transfer_policy::force_host_staging);

    REQUIRE(route == sirius::op::detail::replica_transfer_route::host_staging);
    REQUIRE(host_resource->get_total_allocated_bytes() == allocated_before);

    int current_device = -1;
    REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);
    REQUIRE(current_device == kProbeDevice);

    std::vector<std::byte> actual(bytes);
    auto const err = cudaMemcpyAsync(
      actual.data(), destination.data(), bytes, cudaMemcpyDeviceToHost, stream.value());
    REQUIRE(err == cudaSuccess);
    stream.synchronize();
    REQUIRE(actual == expected);
  }

  rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
  source.reset();
}

TEST_CASE("Bloom replica built on GPU 0 has no false negatives on GPU 1",
          "[dynamic_filter][mgpu][replica][bloom]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  std::unique_ptr<sirius::op::sirius_dynamic_bloom_filter> filter;
  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const& build_space = replica_spaces.front().get_gpu_space();
    auto const stream       = build_space.acquire_stream();
    auto keys               = cudf::sequence(1024,
                               cudf::numeric_scalar<std::int64_t>(0, true, stream),
                               cudf::numeric_scalar<std::int64_t>(1, true, stream),
                               stream,
                               build_space.get_default_allocator());
    filter                  = std::make_unique<sirius::op::sirius_dynamic_bloom_filter>(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();
    keys.reset();  // replication must use filter-owned source material

    REQUIRE(filter->replica_count() == 1);
    REQUIRE(filter->is_available_on_device(kBuildDevice));
    REQUIRE_FALSE(filter->is_available_on_device(kProbeDevice));
    replicate_to_both_devices(*filter, replica_spaces);
    REQUIRE(filter->replica_count() == kReplicaDevices.size());
  }

  {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
    auto const stream = cudf::get_default_stream();
    // Positions 0, 2, and 4 are build keys. Misses may be Bloom false positives, so only the
    // no-false-negative contract is asserted for them.
    auto probe = make_values<std::int64_t>(
      {0, 2048, 511, 4096, 1023}, cudf::data_type{cudf::type_id::INT64}, stream);
    auto mask = filter->compute_mask(
      probe->view(), kProbeDevice, stream, cudf::get_current_device_resource_ref());
    REQUIRE(mask != nullptr);
    auto const host_mask = mask_to_host(mask->view(), stream);
    REQUIRE(host_mask.size() == 5);
    REQUIRE(host_mask[0] != 0);
    REQUIRE(host_mask[2] != 0);
    REQUIRE(host_mask[4] != 0);
  }

  rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
  filter.reset();
}

TEST_CASE("zone-map replica built on GPU 0 lowers and evaluates its AST on GPU 1",
          "[dynamic_filter][mgpu][replica][zone_map]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  std::unique_ptr<sirius::op::sirius_dynamic_zone_map_filter> filter;
  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const& build_space = replica_spaces.front().get_gpu_space();
    auto const stream       = build_space.acquire_stream();
    std::vector<sirius::op::zone_map_entry> zones;
    zones.push_back({std::make_unique<cudf::numeric_scalar<std::int64_t>>(
                       3, true, stream, build_space.get_default_allocator()),
                     std::make_unique<cudf::numeric_scalar<std::int64_t>>(
                       6, true, stream, build_space.get_default_allocator())});
    filter = std::make_unique<sirius::op::sirius_dynamic_zone_map_filter>(std::move(zones),
                                                                          /*inclusive_min=*/true,
                                                                          /*inclusive_max=*/true);
    stream.synchronize();

    REQUIRE(filter->is_available_on_device(kBuildDevice));
    REQUIRE_FALSE(filter->is_available_on_device(kProbeDevice));
    replicate_to_both_devices(*filter, replica_spaces);
  }

  {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
    auto const stream = cudf::get_default_stream();
    auto probe        = cudf::sequence(10,
                                cudf::numeric_scalar<std::int64_t>(0, true, stream),
                                cudf::numeric_scalar<std::int64_t>(1, true, stream),
                                stream,
                                cudf::get_current_device_resource_ref());
    std::vector<cudf::column_view> columns{probe->view()};
    cudf::table_view input{columns};

    cudf::ast::tree tree;
    auto const& col_ref = tree.emplace<cudf::ast::column_reference>(0);
    auto const& root    = filter->to_ast(tree, col_ref, kProbeDevice);
    auto mask = cudf::compute_column(input, root, stream, cudf::get_current_device_resource_ref());

    REQUIRE(mask_to_host(mask->view(), stream) ==
            std::vector<std::uint8_t>{0, 0, 0, 1, 1, 1, 1, 0, 0, 0});
  }

  rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
  filter.reset();
}

TEST_CASE("small IN-list replica built on GPU 0 computes an exact mask on GPU 1",
          "[dynamic_filter][mgpu][replica][small_in_list]")
{
  if (!require_two_gpus()) { return; }

  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager(2);
  auto replica_spaces = get_replica_spaces(*memory_manager);
  auto& target_space  = replica_spaces.back().get_gpu_space();
  auto* target_mr =
    target_space.get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(target_mr != nullptr);
  auto const target_reserved_before  = target_space.get_total_reserved_memory();
  auto const target_active_before    = target_space.get_active_reservation_count();
  auto const target_allocated_before = target_mr->get_total_allocated_bytes();
  std::unique_ptr<sirius::op::sirius_dynamic_small_in_list_filter> filter;
  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const& build_space = replica_spaces.front().get_gpu_space();
    auto const stream       = build_space.acquire_stream();
    auto keys = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
    filter    = std::make_unique<sirius::op::sirius_dynamic_small_in_list_filter>(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();
    keys.reset();  // replication must use the filter-owned needle snapshot

    REQUIRE(filter->replica_count() == 1);
    REQUIRE(filter->is_available_on_device(kBuildDevice));
    REQUIRE_FALSE(filter->is_available_on_device(kProbeDevice));

    // A remote caller must skip the optional filter until its device-local needle snapshot is
    // published; it must never dereference the source GPU's buffer.
    {
      rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
      auto const probe_stream = cudf::get_default_stream();
      auto early_probe =
        make_values<std::int64_t>({2, 7}, cudf::data_type{cudf::type_id::INT64}, probe_stream);
      auto early_mask = filter->compute_mask(
        early_probe->view(), kProbeDevice, probe_stream, cudf::get_current_device_resource_ref());
      REQUIRE(early_mask == nullptr);
    }

    replicate_to_both_devices(*filter, replica_spaces);
    REQUIRE(filter->replica_count() == kReplicaDevices.size());
    REQUIRE(target_space.get_total_reserved_memory() == target_reserved_before);
    REQUIRE(target_space.get_active_reservation_count() == target_active_before);
    REQUIRE(target_mr->get_total_allocated_bytes() > target_allocated_before);
  }

  {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
    auto const stream = cudf::get_default_stream();
    auto probe =
      make_values<std::int64_t>({1, 2, 3, 4, 6, 9}, cudf::data_type{cudf::type_id::INT64}, stream);
    auto mask = filter->compute_mask(
      probe->view(), kProbeDevice, stream, cudf::get_current_device_resource_ref());
    REQUIRE(mask != nullptr);
    REQUIRE(mask_to_host(mask->view(), stream) == std::vector<std::uint8_t>{0, 1, 0, 1, 1, 0});
  }

  rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
  filter.reset();
  REQUIRE(target_space.get_total_reserved_memory() == target_reserved_before);
  REQUIRE(target_space.get_active_reservation_count() == target_active_before);
  REQUIRE(target_mr->get_total_allocated_bytes() == target_allocated_before);
}
