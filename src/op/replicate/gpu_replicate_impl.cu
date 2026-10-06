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
#include <cudf/filling.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/algorithm>
#include <cuda_runtime.h>
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
#include <vector>

namespace sirius::op::gpu_replicate_impl {

namespace {

//! Whole bytes of one copy of a row; at least one, so every copy advances the byte prefix.
__device__ std::int64_t bytes_of(std::int32_t bits)
{
  return ::cuda::std::max(std::int32_t{1}, (bits + 7) / 8);
}

//! Bytes of all copies of row `i`.
struct copies_bytes {
  std::int64_t const* counts;
  std::int32_t const* bits;
  __device__ std::int64_t operator()(cudf::size_type i) const
  {
    return counts[i] * bytes_of(bits[i]);
  }
};

//! First output row whose first byte is at or past `k * max_bytes`, for `k >= 1`.
struct byte_cut {
  std::int64_t const* row_prefix;
  std::int64_t const* byte_prefix;
  std::int32_t const* bits;
  cudf::size_type rows;
  std::int64_t max_bytes;
  __device__ std::int64_t operator()(std::int64_t k) const
  {
    auto const target = k * max_bytes;
    // The callers keep `target` below the total bytes, so some row's copies span it.
    auto const row = static_cast<cudf::size_type>(
      thrust::upper_bound(thrust::seq, byte_prefix, byte_prefix + rows, target) - byte_prefix);
    auto const rows_before  = row == 0 ? std::int64_t{0} : row_prefix[row - 1];
    auto const bytes_before = row == 0 ? std::int64_t{0} : byte_prefix[row - 1];
    auto const row_bytes    = bytes_of(bits[row]);
    return rows_before + (target - bytes_before + row_bytes - 1) / row_bytes;
  }
};

//! Input row range `[first_row, end_row)` holding output rows `[lo, hi)`.
struct input_rows {
  std::int64_t const* row_prefix;
  cudf::size_type rows;
  __device__ thrust::pair<cudf::size_type, cudf::size_type> operator()(
    thrust::pair<std::int64_t, std::int64_t> output_range) const
  {
    auto const* end  = row_prefix + rows;
    auto const first = thrust::upper_bound(thrust::seq, row_prefix, end, output_range.first);
    auto const last  = thrust::lower_bound(thrust::seq, row_prefix, end, output_range.second);
    return {static_cast<cudf::size_type>(first - row_prefix),
            static_cast<cudf::size_type>(last - row_prefix + 1)};
  }
};

//! Copies of input row `first_row + i` that fall in output rows `[lo, hi)`.
struct clipped_count {
  std::int64_t const* row_prefix;
  cudf::size_type first_row;
  std::int64_t lo;
  std::int64_t hi;
  __device__ cudf::size_type operator()(cudf::size_type i) const
  {
    auto const row   = first_row + i;
    auto const start = row == 0 ? std::int64_t{0} : row_prefix[row - 1];
    return static_cast<cudf::size_type>(::cuda::std::min(row_prefix[row], hi) -
                                        ::cuda::std::max(start, lo));
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

//! Queues a copy of the last of @p count values at @p values into @p host.
void copy_last_async(std::int64_t const* values,
                     cudf::size_type count,
                     std::int64_t& host,
                     ::cuda::stream_ref stream)
{
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(&host, values + count - 1, sizeof(host), cudaMemcpyDeviceToHost, stream.get()));
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
  // A CASE result can carry an all-valid mask, so the null count decides, not the mask.
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
  auto const bit_counts = cudf::row_bit_count(data, stream, mr);
  auto const* bits      = bit_counts->view().data<std::int32_t>();
  rmm::device_uvector<std::int64_t> byte_prefix(rows, stream, mr);
  auto const copies_bytes_it = thrust::make_transform_iterator(
    thrust::counting_iterator<cudf::size_type>(0), copies_bytes{counts64, bits});
  thrust::inclusive_scan(policy, copies_bytes_it, copies_bytes_it + rows, byte_prefix.begin());

  std::int64_t total_rows  = 0;
  std::int64_t total_bytes = 0;
  copy_last_async(row_prefix_data, rows, total_rows, stream);
  copy_last_async(byte_prefix.data(), rows, total_bytes, stream);
  stream.sync();
  if (total_rows == 0) { return {std::move(row_prefix), {}}; }

  // Cuts are output rows where a new slice starts: each multiple of max_rows, and the first row
  // starting at or past each multiple of max_bytes. Between two cuts both caps hold.
  auto const max_bytes      = static_cast<std::int64_t>(caps.max_bytes);
  auto const byte_cut_count = (total_bytes - 1) / max_bytes;
  rmm::device_uvector<std::int64_t> device_byte_cuts(byte_cut_count, stream, mr);
  thrust::transform(policy,
                    thrust::counting_iterator<std::int64_t>(1),
                    thrust::counting_iterator<std::int64_t>(byte_cut_count + 1),
                    device_byte_cuts.begin(),
                    byte_cut{row_prefix_data, byte_prefix.data(), bits, rows, max_bytes});
  auto cuts = to_host(device_byte_cuts, stream);
  for (std::int64_t cut = caps.max_rows; cut < total_rows; cut += caps.max_rows) {
    cuts.push_back(cut);
  }
  cuts.push_back(0);
  cuts.push_back(total_rows);
  std::ranges::sort(cuts);
  auto const duplicates = std::ranges::unique(cuts);
  cuts.erase(duplicates.begin(), duplicates.end());

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
    slices.push_back(
      {output_ranges[s].first, output_ranges[s].second, row_ranges[s].first, row_ranges[s].second});
  }
  return {std::move(row_prefix), std::move(slices)};
}

std::unique_ptr<cudf::table> materialize(cudf::table_view const& data,
                                         plan const& expansion,
                                         slice const& part,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto const rows = part.end_row - part.first_row;
  rmm::device_uvector<cudf::size_type> counts(rows, stream, mr);
  thrust::transform(
    rmm::exec_policy_nosync(stream, mr),
    thrust::counting_iterator<cudf::size_type>(0),
    thrust::counting_iterator<cudf::size_type>(rows),
    counts.begin(),
    clipped_count{
      expansion.row_prefix->view().data<std::int64_t>(), part.first_row, part.lo, part.hi});
  cudf::column_view const count_view{
    cudf::data_type{cudf::type_id::INT32}, rows, counts.data(), nullptr, 0};
  auto const rows_view = cudf::slice(data, {part.first_row, part.end_row}, stream).front();
  return cudf::repeat(rows_view, count_view, stream, mr);
}

}  // namespace sirius::op::gpu_replicate_impl
