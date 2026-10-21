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

#pragma once

/**
 * @file dynamic_filter_mask_test_utils.hpp
 * @brief Helpers shared by the membership mask and cached checkpoint tests in
 * `test_dynamic_filter_merge.cpp` and the hidden benchmarks in `bench_fused_membership.cpp`: device
 * upload and readback of plain values, the three real membership filters, a
 * `try_fused_membership_mask` call that returns its round count, a lone filter's production mask,
 * and the cached checkpoint layout and prefix counts.
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda/stream>
#include <cuda_runtime.h>

#include <op/dynamic_filter/dynamic_filter_mask_ops.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/gpu_ingestible_types.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace sirius::test::dynamic_filter_mask {

/// Copies @p values into a new device column of @p type without a null mask.
template <class T>
std::unique_ptr<cudf::column> make_device_column(std::vector<T> const& values,
                                                 cudf::data_type type,
                                                 ::cuda::stream_ref stream)
{
  auto column = cudf::make_fixed_width_column(
    type, static_cast<cudf::size_type>(values.size()), cudf::mask_state::UNALLOCATED, stream);
  if (!values.empty()) {
    RMM_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<T>(),
                                 values.data(),
                                 values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice,
                                 stream.get()));
    stream.sync();
  }
  return column;
}

template <class T>
std::vector<T> copy_to_host(T const* data, std::size_t count, ::cuda::stream_ref stream)
{
  std::vector<T> result(count);
  if (count != 0) {
    RMM_CUDA_TRY(cudaMemcpyAsync(
      result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost, stream.get()));
  }
  stream.sync();
  return result;
}

/// Builds a hash-set IN-list (@p kind 0), Bloom (1) or small IN-list (2) filter over @p keys.
inline std::unique_ptr<sirius::op::sirius_mask_applicable> make_membership_filter(
  int kind, cudf::column_view keys, ::cuda::stream_ref stream)
{
  auto const mr = cudf::get_current_device_resource_ref();
  switch (kind) {
    case 0: return std::make_unique<sirius::op::sirius_dynamic_in_list_filter>(keys, stream, mr);
    case 1: return std::make_unique<sirius::op::sirius_dynamic_bloom_filter>(keys, stream, mr);
    default:
      return std::make_unique<sirius::op::sirius_dynamic_small_in_list_filter>(keys, stream, mr);
  }
}

/// Runs `try_fused_membership_mask` on the current device and returns its submitted round count.
inline std::size_t fuse_membership(std::span<sirius::op::fused_membership_input const> steps,
                                   cudf::mutable_column_view running,
                                   std::optional<cudf::column_view> residual,
                                   cudf::device_span<cudf::size_type> counts,
                                   ::cuda::stream_ref stream)
{
  std::size_t rounds = 0;
  sirius::op::try_fused_membership_mask(
    steps, rmm::get_current_cuda_device().value(), running, residual, counts, rounds, stream);
  return rounds;
}

/// The BOOL8 values `filter.compute_mask` produces for @p probe on the current device.
inline std::vector<std::uint8_t> lone_mask(sirius::op::sirius_mask_applicable const& filter,
                                           cudf::column_view probe,
                                           ::cuda::stream_ref stream)
{
  auto const mask = filter.compute_mask(
    probe, rmm::get_current_cuda_device().value(), stream, cudf::get_current_device_resource_ref());
  return copy_to_host(
    mask->view().data<std::uint8_t>(), static_cast<std::size_t>(mask->size()), stream);
}

/// A cached layout projecting column 0, with column 1 as a BOOLEAN residual when @p residual.
inline sirius::op::scan::cached_view_layout cached_layout(bool residual = false)
{
  sirius::op::scan::cached_view_layout layout;
  layout.output_columns    = {0};
  layout.projected_columns = {0};
  if (residual) {
    layout.residual = std::make_unique<sirius::ast::node>(
      sirius::ast::reference{1, sirius::logical_type::make(sirius::type_id::BOOLEAN)});
  }
  return layout;
}

inline std::vector<cudf::size_type> counts_of(sirius::op::scan::application_result const& result)
{
  return {result.prefix_counts().begin(), result.prefix_counts().end()};
}

}  // namespace sirius::test::dynamic_filter_mask
