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

// Shared helpers for dynamic-filter replica tests in sirius_unittest and sirius_cuda_fault_test.

#pragma once

#include "op/dynamic_filter/dynamic_filter_replica_space.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <catch.hpp>
#include <cucascade/memory/common.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace sirius::test::replica {

inline constexpr std::size_t default_replica_device_count = 2;

/**
 * @brief A non-null column of @p values, complete on @p stream before return.
 */
template <typename T>
std::unique_ptr<cudf::column> make_values(std::vector<T> const& values,
                                          cudf::data_type type,
                                          ::cuda::stream_ref stream)
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
                                   stream.get());
  REQUIRE(err == cudaSuccess);
  // Callers commonly pass a temporary initializer vector. Complete the pageable-host transfer
  // before that vector is destroyed; these focused tests do not benchmark ingestion.
  stream.sync();
  return col;
}

/**
 * @brief Copies a BOOL8 @p mask to the host.
 */
inline std::vector<std::uint8_t> mask_to_host(cudf::column_view const& mask,
                                              ::cuda::stream_ref stream)
{
  REQUIRE(mask.type().id() == cudf::type_id::BOOL8);
  std::vector<std::uint8_t> host(static_cast<std::size_t>(mask.size()));
  auto const err = cudaMemcpyAsync(host.data(),
                                   mask.data<bool>(),
                                   host.size() * sizeof(bool),
                                   cudaMemcpyDeviceToHost,
                                   stream.get());
  REQUIRE(err == cudaSuccess);
  stream.sync();
  return host;
}

/**
 * @brief One replica space per GPU of @p memory_manager, ordered by device ID, each staged through
 * its local host space when one exists.
 */
template <typename MemoryManager>
std::vector<sirius::op::dynamic_filter_replica_space> get_replica_spaces(
  MemoryManager& memory_manager, std::size_t expected_device_count = default_replica_device_count)
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

}  // namespace sirius::test::replica
