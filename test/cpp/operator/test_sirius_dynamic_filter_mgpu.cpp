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
// filter on logical GPU 0, materializes its replica set for logical GPUs {0, 1}, then consumes the
// filter from a probe column resident on logical GPU 1. Before replica support, the corresponding
// cross-device dereference is the cudaErrorIllegalAddress reported by TPC-H Q2.

#include "op/dynamic_filter_replica_transfer.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "operator_test_utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>

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
  MemoryManager const& memory_manager)
{
  auto const spaces = memory_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(spaces.size() == kReplicaDevices.size());

  std::vector<sirius::op::dynamic_filter_replica_space> result;
  result.reserve(spaces.size());
  for (auto const* space : spaces) {
    result.emplace_back(*space);
  }
  std::sort(result.begin(), result.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.get().get_device_id() < rhs.get().get_device_id();
  });
  for (std::size_t i = 0; i < result.size(); ++i) {
    REQUIRE(result[i].get().get_device_id() == kReplicaDevices[i]);
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
  std::unique_ptr<sirius::op::sirius_dynamic_in_list_filter> filter;
  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const& build_space = replica_spaces.front().get();
    auto const stream       = build_space.acquire_stream();
    auto keys = make_values<std::int64_t>({2, 4, 6}, cudf::data_type{cudf::type_id::INT64}, stream);
    filter    = std::make_unique<sirius::op::sirius_dynamic_in_list_filter>(
      keys->view(), stream, build_space.get_default_allocator());
    stream.synchronize();
    keys.reset();  // replication must not depend on the constructor's borrowed column_view

    REQUIRE(filter->replica_count() == 1);
    REQUIRE(filter->is_available_on_device(kBuildDevice));
    REQUIRE_FALSE(filter->is_available_on_device(kProbeDevice));

    // Replica publication is opportunistic: a consumer scheduled on GPU 1 before its local copy
    // is ready must skip the optional filter, never touch GPU 0's set or wait for it.
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
}

TEST_CASE("dynamic-filter replica transfer supports forced portable-host staging",
          "[dynamic_filter][mgpu][replica][transfer]")
{
  if (!require_two_gpus()) { return; }

  std::unique_ptr<cudf::column> source;
  {
    rmm::cuda_set_device_raii const build_device{rmm::cuda_device_id{kBuildDevice}};
    auto const stream = cudf::get_default_stream();
    source =
      make_values<std::int64_t>({11, 22, 33, 44}, cudf::data_type{cudf::type_id::INT64}, stream);
  }

  {
    rmm::cuda_set_device_raii const probe_device{rmm::cuda_device_id{kProbeDevice}};
    auto const stream = cudf::get_default_stream();
    auto destination  = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                 source->size(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 cudf::get_current_device_resource_ref());
    auto transfer     = sirius::op::detail::enqueue_replica_transfer(
      destination->mutable_view().data<std::int64_t>(),
      rmm::cuda_device_id{kProbeDevice},
      source->view().data<std::int64_t>(),
      rmm::cuda_device_id{kBuildDevice},
      static_cast<std::size_t>(source->size()) * sizeof(std::int64_t),
      stream,
      sirius::op::detail::replica_transfer_policy::force_portable_host);

    REQUIRE(transfer.route() == sirius::op::detail::replica_transfer_route::portable_host);
    transfer.wait();
    REQUIRE(transfer.complete());

    std::vector<std::int64_t> actual(static_cast<std::size_t>(destination->size()));
    auto const err = cudaMemcpyAsync(actual.data(),
                                     destination->view().data<std::int64_t>(),
                                     actual.size() * sizeof(std::int64_t),
                                     cudaMemcpyDeviceToHost,
                                     stream.value());
    REQUIRE(err == cudaSuccess);
    stream.synchronize();
    REQUIRE(actual == std::vector<std::int64_t>{11, 22, 33, 44});
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
    auto const& build_space = replica_spaces.front().get();
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
    auto const& build_space = replica_spaces.front().get();
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
