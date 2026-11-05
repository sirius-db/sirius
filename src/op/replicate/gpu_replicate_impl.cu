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

#include "op/replicate/gpu_replicate_impl.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace sirius::op::gpu_replicate_impl {

namespace {

/// @brief Whole bytes of one copy of a row; at least one, so every copy advances the byte prefix.
__device__ __forceinline__ std::int64_t bytes_of(std::int32_t bits)
{
  return ::cuda::std::max(std::int64_t{1}, ::cuda::ceil_div(static_cast<std::int64_t>(bits), 8));
}

/// @brief Bytes of all copies of row `i`.
struct copies_bytes {
  std::int64_t const* counts;
  std::int32_t const* bits;
  __device__ __forceinline__ std::int64_t operator()(cudf::size_type i) const
  {
    return counts[i] * bytes_of(bits[i]);
  }
};

/// @brief First output row whose first byte is at or past `k * max_bytes`, for `k >= 1`.
struct byte_cut {
  std::int64_t const* row_prefix;
  std::int64_t const* byte_prefix;
  std::int32_t const* bits;
  cudf::size_type rows;
  std::int64_t max_bytes;
  __device__ __forceinline__ std::int64_t operator()(std::int64_t k) const
  {
    auto const target = k * max_bytes;
    // The callers keep `target` below the total bytes, so some row's copies span it.
    auto const row = static_cast<cudf::size_type>(
      thrust::upper_bound(thrust::seq, byte_prefix, byte_prefix + rows, target) - byte_prefix);
    auto const rows_before  = row == 0 ? std::int64_t{0} : row_prefix[row - 1];
    auto const bytes_before = row == 0 ? std::int64_t{0} : byte_prefix[row - 1];
    auto const row_bytes    = bytes_of(bits[row]);
    return rows_before + ::cuda::ceil_div(target - bytes_before, row_bytes);
  }
};

/// @brief Computes the input row range `[input_begin, input_end)` holding output rows
/// `[output_begin, output_end)`.
struct input_rows {
  std::int64_t const* row_prefix;
  cudf::size_type rows;
  __device__ __forceinline__ thrust::pair<cudf::size_type, cudf::size_type> operator()(
    thrust::pair<std::int64_t, std::int64_t> output_range) const
  {
    auto const* end  = row_prefix + rows;
    auto const first = thrust::upper_bound(thrust::seq, row_prefix, end, output_range.first);
    auto const last  = thrust::lower_bound(thrust::seq, row_prefix, end, output_range.second);
    return {static_cast<cudf::size_type>(first - row_prefix),
            static_cast<cudf::size_type>(last - row_prefix + 1)};
  }
};

/// @brief Emits a boundary only when its preceding output copy crosses a global byte target.
template <typename Index>
struct output_boundary_cut {
  std::int64_t const* row_prefix;
  std::int64_t const* byte_prefix;
  std::int32_t const* bits;
  cudf::size_type rows;
  std::int64_t max_bytes;

  __device__ __forceinline__ Index operator()(std::int64_t boundary) const
  {
    auto const previous = boundary - 1;
    auto const row      = static_cast<cudf::size_type>(
      thrust::upper_bound(thrust::seq, row_prefix, row_prefix + rows, previous) - row_prefix);
    auto const rows_before  = row == 0 ? std::int64_t{0} : row_prefix[row - 1];
    auto const bytes_before = row == 0 ? std::int64_t{0} : byte_prefix[row - 1];
    auto const width        = bytes_of(bits[row]);
    auto const start        = bytes_before + (previous - rows_before) * width;
    bool const crosses      = width >= max_bytes || width >= max_bytes - start % max_bytes;
    return crosses ? static_cast<Index>(boundary) : Index{0};
  }
};

template <typename T>
std::vector<T> to_host(rmm::device_uvector<T> const& values, ::cuda::stream_ref stream)
{
  std::vector<T> host(values.size());
  if (!host.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      host.data(), values.data(), values.size() * sizeof(T), cudaMemcpyDeviceToHost, stream.get()));
  }
  stream.sync();
  return host;
}

/// @brief Queues a copy of the last of the non-empty device array @p values into @p host.
void copy_last_async(std::span<std::int64_t const> values,
                     std::int64_t& host,
                     ::cuda::stream_ref stream)
{
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &host, values.data() + values.size() - 1, sizeof(host), cudaMemcpyDeviceToHost, stream.get()));
}

//! The caller selects an index width that can represent every interior output
//! position.
template <typename Index>
std::vector<std::int64_t> output_boundary_cuts(std::int64_t const* row_prefix,
                                               std::int64_t const* byte_prefix,
                                               std::int32_t const* bits,
                                               cudf::size_type rows,
                                               std::int64_t total_rows,
                                               std::int64_t max_bytes,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  if (total_rows == 1) { return {}; }
  rmm::device_uvector<Index> device_cuts(total_rows - 1, stream, mr);
  thrust::transform_n(rmm::exec_policy_nosync(stream, mr),
                      thrust::counting_iterator<std::int64_t>(1),
                      total_rows - 1,
                      device_cuts.begin(),
                      output_boundary_cut<Index>{row_prefix, byte_prefix, bits, rows, max_bytes});
  auto selected = to_host(device_cuts, stream);
  std::erase(selected, Index{0});
  if constexpr (std::is_same_v<Index, std::int64_t>) {
    return selected;
  } else {
    std::vector<std::int64_t> result(selected.size());
    std::copy(selected.begin(), selected.end(), result.begin());
    return result;
  }
}

}  // namespace

plan plan_slices(cudf::table_view const& data,
                 cudf::column_view const& counts,
                 limits const& caps,
                 ::cuda::stream_ref stream,
                 rmm::device_async_resource_ref mr)
{
  if (!cudf::is_integral_not_bool(counts.type())) {
    throw sirius::internal_exception("REPLICATE: the count column is not an integer column");
  }
  if (counts.size() != data.num_rows()) {
    throw sirius::internal_exception(
      "REPLICATE: {} counts for {} rows", counts.size(), data.num_rows());
  }
  if (caps.max_rows <= 0 || caps.max_bytes == 0) {
    throw sirius::internal_exception("REPLICATE: output caps must be positive");
  }
  if (counts.null_count() > 0) {
    throw sirius::internal_exception("REPLICATE: the count column has {} nulls",
                                     counts.null_count());
  }

  auto const rows = data.num_rows();
  auto row_prefix = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, rows, cudf::mask_state::UNALLOCATED, stream, mr);
  if (rows == 0) { return {std::move(row_prefix), {}}; }

  std::unique_ptr<cudf::column> widened;
  if (counts.type().id() != cudf::type_id::INT64) {
    // EXCEPT/INTERSECT ALL carry INT64 count columns from the planner; this is for generality.
    widened = cudf::cast(counts, cudf::data_type{cudf::type_id::INT64}, stream, mr);
  }
  auto const* counts64 =
    widened ? widened->view().data<std::int64_t>() : counts.data<std::int64_t>();
  auto const policy = rmm::exec_policy_nosync(stream, mr);
  if (thrust::any_of(
        policy, counts64, counts64 + rows, [] __device__(std::int64_t c) { return c < 0; })) {
    throw sirius::internal_exception("REPLICATE: the count column has a negative value");
  }

  auto* const row_prefix_data = row_prefix->mutable_view().data<std::int64_t>();
  thrust::inclusive_scan(policy, counts64, counts64 + rows, row_prefix_data);
  // Every row costs at least one byte, so a one-byte target is a one-row limit.
  auto const limit_bytes =
    caps.max_bytes != 1 && caps.max_bytes != std::numeric_limits<std::size_t>::max();
  auto const bit_counts = limit_bytes ? cudf::row_bit_count(data, stream, mr) : nullptr;
  auto const* bits      = bit_counts ? bit_counts->view().data<std::int32_t>() : nullptr;
  rmm::device_uvector<std::int64_t> byte_prefix(limit_bytes ? rows : 0, stream, mr);
  if (limit_bytes) {
    auto const copies_bytes_it = thrust::make_transform_iterator(
      thrust::counting_iterator<cudf::size_type>(0), copies_bytes{counts64, bits});
    thrust::inclusive_scan(policy, copies_bytes_it, copies_bytes_it + rows, byte_prefix.begin());
  }

  std::int64_t total_rows  = 0;
  std::int64_t total_bytes = 0;
  copy_last_async({row_prefix_data, static_cast<std::size_t>(rows)}, total_rows, stream);
  if (limit_bytes) {
    copy_last_async({byte_prefix.data(), byte_prefix.size()}, total_bytes, stream);
  }
  stream.sync();
  if (total_rows == 0) { return {std::move(row_prefix), {}}; }

  // Cuts are output rows where a new slice starts: each multiple of max_rows, and the first row
  // starting at or past each multiple of max_bytes. Between two cuts both caps hold.
  // The byte total fits in INT64, so a larger cap acts as INT64_MAX.
  auto const max_bytes = static_cast<std::int64_t>(
    std::min<std::size_t>(caps.max_bytes, std::numeric_limits<std::int64_t>::max()));
  auto const row_limit = caps.max_bytes == 1 ? cudf::size_type{1} : caps.max_rows;
  std::vector<std::int64_t> cuts;
  rmm::device_uvector<std::int64_t> device_byte_cuts(0, stream, mr);
  // Enumerate the smaller space: byte thresholds or interior output-row positions.
  if (limit_bytes && (total_bytes - 1) / max_bytes > total_rows - 1) {
    if (total_rows - 1 <= std::numeric_limits<std::int32_t>::max()) {
      cuts = output_boundary_cuts<std::int32_t>(
        row_prefix_data, byte_prefix.data(), bits, rows, total_rows, max_bytes, stream, mr);
    } else {
      cuts = output_boundary_cuts<std::int64_t>(
        row_prefix_data, byte_prefix.data(), bits, rows, total_rows, max_bytes, stream, mr);
    }
    std::vector<std::int64_t> row_cuts{0};
    for (std::int64_t cut = row_limit; cut < total_rows; cut += row_limit) {
      row_cuts.push_back(cut);
    }
    row_cuts.push_back(total_rows);
    std::vector<std::int64_t> merged;
    merged.reserve(cuts.size() + row_cuts.size());
    std::merge(
      cuts.begin(), cuts.end(), row_cuts.begin(), row_cuts.end(), std::back_inserter(merged));
    auto const duplicates = std::ranges::unique(merged);
    merged.erase(duplicates.begin(), duplicates.end());
    cuts = std::move(merged);
  } else {
    auto const byte_cut_count = limit_bytes ? (total_bytes - 1) / max_bytes : 0;
    device_byte_cuts.resize(byte_cut_count, stream);
    thrust::transform_n(policy,
                        thrust::counting_iterator<std::int64_t>(1),
                        byte_cut_count,
                        device_byte_cuts.begin(),
                        byte_cut{row_prefix_data, byte_prefix.data(), bits, rows, max_bytes});
    cuts = to_host(device_byte_cuts, stream);
    for (std::int64_t cut = row_limit; cut < total_rows; cut += row_limit) {
      cuts.push_back(cut);
    }
    cuts.push_back(0);
    cuts.push_back(total_rows);
    std::ranges::sort(cuts);
    auto const duplicates = std::ranges::unique(cuts);
    cuts.erase(duplicates.begin(), duplicates.end());
  }

  if (rows == 1) {
    std::vector<slice> slices;
    slices.reserve(cuts.size() - 1);
    for (std::size_t s = 1; s < cuts.size(); ++s) {
      slices.push_back({cuts[s - 1], cuts[s], 0, 1});
    }
    return {std::move(row_prefix), std::move(slices)};
  }

  auto const slice_count = cuts.size() - 1;
  std::vector<thrust::pair<std::int64_t, std::int64_t>> output_ranges(slice_count);
  for (std::size_t s = 0; s < slice_count; ++s) {
    output_ranges[s] = {cuts[s], cuts[s + 1]};
  }
  rmm::device_uvector<thrust::pair<std::int64_t, std::int64_t>> device_output_ranges(
    slice_count, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(device_output_ranges.data(),
                                output_ranges.data(),
                                slice_count * sizeof(output_ranges[0]),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  rmm::device_uvector<thrust::pair<cudf::size_type, cudf::size_type>> device_row_ranges(
    slice_count, stream, mr);
  thrust::transform(policy,
                    device_output_ranges.begin(),
                    device_output_ranges.end(),
                    device_row_ranges.begin(),
                    input_rows{row_prefix_data, rows});
  auto const row_ranges = to_host(device_row_ranges, stream);

  std::vector<slice> slices;
  slices.reserve(slice_count);
  for (std::size_t s = 0; s < slice_count; ++s) {
    slices.push_back({.output_begin = output_ranges[s].first,
                      .output_end   = output_ranges[s].second,
                      .input_begin  = row_ranges[s].first,
                      .input_end    = row_ranges[s].second});
  }
  return {std::move(row_prefix), std::move(slices)};
}

std::unique_ptr<cudf::table> materialize(cudf::table_view const& data,
                                         plan const& expansion,
                                         slice const& part,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto const rows        = static_cast<cudf::size_type>(part.output_end - part.output_begin);
  auto const* row_prefix = expansion.row_prefix->view().data<std::int64_t>();
  rmm::device_uvector<cudf::size_type> gather_map(rows, stream, mr);
  // Search the slice's input rows so the map indexes rows_view, including runs split across slices.
  thrust::upper_bound(rmm::exec_policy_nosync(stream, mr),
                      row_prefix + part.input_begin,
                      row_prefix + part.input_end,
                      thrust::counting_iterator<std::int64_t>(part.output_begin),
                      thrust::counting_iterator<std::int64_t>(part.output_end),
                      gather_map.begin());
  cudf::column_view const map_view{
    cudf::data_type{cudf::type_id::INT32}, rows, gather_map.data(), nullptr, 0};
  auto const rows_view = cudf::slice(data, {part.input_begin, part.input_end}, stream).front();
  return cudf::gather(rows_view,
                      map_view,
                      cudf::out_of_bounds_policy::DONT_CHECK,
                      cudf::negative_index_policy::NOT_ALLOWED,
                      stream,
                      mr);
}

}  // namespace sirius::op::gpu_replicate_impl
